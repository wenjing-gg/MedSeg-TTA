from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from medseg_tta.models.nnunet import PlainConvUNet
from medseg_tta.models.nnunet2d import PlainConvUNet2D
from medseg_tta.models.unet2d import UNet2d
from medseg_tta.methods.prior_estimation.adami.three_d.legacy.unet3d import UNet3d
from medseg_tta.methods.output_level_regularization.tent.common.legacy.dataset2D import MedicalImageDataset2D
from medseg_tta.methods.prior_estimation.vptta.three_d.legacy.vptta3d_core import build_test_loader as build_3d_loader


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def robust_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        return torch.load(path, map_location=map_location, weights_only=False)


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in obj:
                return obj[key]
    return obj


def clean_state_dict(state_dict):
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def build_2d_loader(args: argparse.Namespace) -> DataLoader:
    image_dir = args.image_dir or os.path.join(args.target_dir, "image")
    mask_dir = args.mask_dir or os.path.join(args.target_dir, "mask")
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Expected image/ and mask/ under {args.target_dir}")
    dataset = MedicalImageDataset2D(
        image_dir=image_dir,
        mask_dir=mask_dir,
        phase="test",
        image_size=(args.image_size, args.image_size),
        normalize=True,
    )
    return DataLoader(dataset, batch_size=args.batch_test, shuffle=False, num_workers=args.num_workers, pin_memory=True)


def build_model(args: argparse.Namespace, dimension: str, modality: str) -> nn.Module:
    if dimension == "two_d":
        if args.model_type == "nnunet2d":
            return PlainConvUNet2D(
                input_channels=args.input_channels,
                n_stages=5,
                features_per_stage=(32, 64, 128, 256, 512),
                kernel_sizes=3,
                strides=(1, 2, 2, 2, 2),
                n_conv_per_stage=2,
                num_classes=args.num_classes,
                n_conv_per_stage_decoder=2,
                deep_supervision=False,
            )
        return UNet2d(in_channels=args.input_channels, n_classes=args.num_classes)

    if args.model_type == "nnunet":
        return PlainConvUNet(
            args.input_channels,
            6,
            tuple(args.nnunet_features),
            nn.Conv3d,
            3,
            (1, 2, 2, 2, 2, 2),
            (2, 2, 2, 2, 2, 2),
            args.num_classes,
            (2, 2, 2, 2, 2),
            False,
            nn.BatchNorm3d,
            None,
            None,
            None,
            nn.ReLU,
            deep_supervision=False,
        )
    return UNet3d(in_chns=args.input_channels, n_classes=args.num_classes)


def load_checkpoint(model: nn.Module, args: argparse.Namespace, device: torch.device) -> None:
    ckpt_path = args.checkpoint
    if ckpt_path == "default":
        raise ValueError("Please pass --checkpoint or --model_path for DeTTA adaptation.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing pretrained checkpoint: {ckpt_path}")
    state = clean_state_dict(extract_state_dict(robust_torch_load(ckpt_path, map_location=device)))
    missing, unexpected = model.load_state_dict(state, strict=not args.non_strict)
    if args.non_strict:
        print(f"Checkpoint loaded with strict=False: missing={len(missing)}, unexpected={len(unexpected)}")


def collect_bn_affine_params(model: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    params = []
    names = []
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types):
            module.train()
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in {"weight", "bias"}:
                    param.requires_grad_(True)
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
    if not params:
        raise RuntimeError("No BatchNorm affine parameters found for DeTTA adaptation.")
    return params, names


def configure_for_detta(model: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    model.train()
    model.requires_grad_(False)
    return collect_bn_affine_params(model)


def unpack_model_output(output):
    if isinstance(output, (tuple, list)):
        seg = output[0]
        denoise = output[1] if len(output) > 1 else None
        return seg, denoise
    return output, None


def output_to_prob(output: torch.Tensor) -> torch.Tensor:
    if output.shape[1] == 1:
        if output.detach().min() >= 0 and output.detach().max() <= 1:
            return output
        return torch.sigmoid(output)
    channel_sum = output.detach().sum(dim=1)
    looks_like_prob = (
        output.detach().min() >= -1e-4
        and output.detach().max() <= 1.0 + 1e-4
        and torch.allclose(channel_sum.mean(), torch.ones_like(channel_sum.mean()), atol=1e-2)
    )
    return output if looks_like_prob else torch.softmax(output, dim=1)


def entropy_loss(output: torch.Tensor) -> torch.Tensor:
    prob = output_to_prob(output)
    if prob.shape[1] == 1:
        return -(prob * torch.log(prob + 1e-6) + (1 - prob) * torch.log(1 - prob + 1e-6)).mean()
    return -(prob * torch.log(prob + 1e-6)).sum(dim=1).mean()


def prediction_from_output(output: torch.Tensor) -> torch.Tensor:
    prob = output_to_prob(output)
    if prob.shape[1] == 1:
        return (prob > 0.5).long()
    return torch.argmax(prob, dim=1, keepdim=True)


def corrupt_input(x: torch.Tensor, noise_std: float, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    noisy = x + torch.randn_like(x) * noise_std if noise_std > 0 else x.clone()
    if mask_ratio > 0:
        keep_mask = (torch.rand_like(x) > mask_ratio).float()
        noisy = noisy * keep_mask
    else:
        keep_mask = torch.ones_like(x)
    return noisy.clamp(float(x.min().detach()), float(x.max().detach())), keep_mask


def detta_loss(model: nn.Module, x: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    noisy, keep_mask = corrupt_input(x, args.noise_std, args.mask_ratio)
    clean_seg, _ = unpack_model_output(model(x))
    noisy_seg, denoised = unpack_model_output(model(noisy))

    losses = []
    if denoised is not None and args.loss_mode in {"denoise", "hybrid"}:
        losses.append(F.l1_loss(denoised * (1 - keep_mask), x * (1 - keep_mask)))
    if denoised is None or args.loss_mode in {"consistency", "hybrid"}:
        clean_prob = output_to_prob(clean_seg).detach()
        noisy_prob = output_to_prob(noisy_seg)
        losses.append(F.mse_loss(noisy_prob, clean_prob))
    if args.entropy_weight > 0:
        losses.append(args.entropy_weight * entropy_loss(noisy_seg))
    if not losses:
        raise RuntimeError("No DeTTA loss terms were enabled.")
    return sum(losses), noisy_seg


def dice_scores(output: torch.Tensor, mask: torch.Tensor, num_classes: int) -> list[float]:
    pred = prediction_from_output(output)
    if mask.ndim == pred.ndim - 1:
        mask = mask.unsqueeze(1)
    scores = []
    label_count = 2 if num_classes <= 1 else num_classes
    for label in range(1, label_count):
        pred_label = pred == label
        mask_label = mask == label
        denom = pred_label.sum() + mask_label.sum()
        if denom > 0:
            scores.append((2 * (pred_label & mask_label).sum() / denom).item())
    return scores or [0.0]


def batch_to_tensors(batch, dimension: str, device: torch.device):
    if dimension == "two_d":
        images, masks, names = batch
        return images.to(device).float(), masks.to(device).long(), list(names)
    images = batch["image"].to(device).float()
    masks = batch["mask"].to(device).long()
    names = batch["name"] if isinstance(batch["name"], list) else [batch["name"]]
    return images, masks, [str(name) for name in names]


def build_loader(args: argparse.Namespace, dimension: str, modality: str) -> DataLoader:
    if dimension == "two_d":
        return build_2d_loader(args)
    image_size = tuple(int(item) for item in args.image_size_3d)
    return build_3d_loader(args, modality, image_size)


def save_report(output_dir: Path, rows: list[dict[str, str]], losses: list[float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["case", "dice"])
            writer.writeheader()
            writer.writerows(rows)
    summary_lines = []
    if rows:
        values = [float(row["dice"]) for row in rows]
        summary_lines.append(f"mean_dice={float(np.mean(values)):.6f}")
        summary_lines.append(f"std_dice={float(np.std(values)):.6f}")
    if losses:
        summary_lines.append(f"mean_tta_loss={float(np.mean(losses)):.6f}")
    (output_dir / "metrics.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def run_detta(args: argparse.Namespace, dimension: str, modality: str) -> int:
    setup_seed(args.seed)
    if args.batch_size is not None:
        args.batch_test = args.batch_size
    else:
        args.batch_size = args.batch_test
    if isinstance(args.positive_labels, str):
        args.positive_labels = [int(item) for item in args.positive_labels.split(",") if item.strip()]
    if not args.positive_labels:
        args.positive_labels = [1]

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    loader = build_loader(args, dimension, modality)
    model = build_model(args, dimension, modality).to(device)
    load_checkpoint(model, args, device)
    params, names = configure_for_detta(model)
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.99))
    print(f"DeTTA trainable BN affine params: {len(params)}")
    if args.debug:
        print("First params:", ", ".join(names[:10]))

    output_dir = Path(args.output_dir or Path(args.target_dir or args.target_root or ".") / "detta_results")
    metric_rows: list[dict[str, str]] = []
    loss_values: list[float] = []

    for batch in tqdm(loader, desc=f"DeTTA-{dimension}-{modality.upper()}"):
        images, masks, names = batch_to_tensors(batch, dimension, device)
        model.train()
        logits = None
        for _ in range(args.adapt_steps):
            loss, logits = detta_loss(model, images, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_values.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            if logits is None:
                logits, _ = unpack_model_output(model(images))
            batch_scores = [
                float(np.mean(dice_scores(logits[index:index + 1], masks[index:index + 1], args.num_classes)))
                for index in range(logits.shape[0])
            ]
        for name, score in zip(names, batch_scores):
            metric_rows.append({"case": str(name), "dice": f"{score:.6f}"})

    if args.save_adapted:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "detta_adapted_state_dict.pth")
    save_report(output_dir, metric_rows, loss_values)
    if metric_rows:
        print(f"Mean Dice: {np.mean([float(row['dice']) for row in metric_rows]):.6f}")
    print(f"Results saved to: {output_dir}")
    return 0


def build_parser(description: str, dimension: str, modality: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    default_channels = 4 if dimension == "three_d" and modality == "mri" else 1
    default_classes = 4 if dimension == "three_d" and modality == "mri" else 2
    parser.add_argument("--checkpoint", "--model_path", dest="checkpoint", type=str, default="default")
    parser.add_argument("--target_dir", type=str, default=None)
    parser.add_argument("--target_root", type=str, default=None)
    parser.add_argument("--source_root", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default="CT")
    parser.add_argument("--base_dir", type=str, default="/home/yuwenjing/data/tta_dataset")
    parser.add_argument("--subfolder", type=str, default=None)
    parser.add_argument("--img", type=str, default="all" if modality == "mri" else "ct")
    parser.add_argument("--brats_layout", action=argparse.BooleanOptionalAction, default=dimension == "three_d" and modality == "mri")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--model_type", type=str, default="unet3d" if dimension == "three_d" else "unet2d")
    parser.add_argument("--input_channels", type=int, default=default_channels)
    parser.add_argument("--num_classes", type=int, default=default_classes)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_size_3d", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--intensity_range", type=float, nargs=2, default=(-200.0, 400.0) if modality == "ct" else None)
    parser.add_argument("--batch_test", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--adapt_steps", "--step", dest="adapt_steps", type=int, default=1)
    parser.add_argument("--loss_mode", choices=["denoise", "consistency", "hybrid"], default="hybrid")
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--positive_labels", type=str, default="1")
    parser.add_argument("--nnunet_features", type=int, nargs=6, default=[32, 64, 125, 256, 320, 320])
    parser.add_argument("--non_strict", action="store_true")
    parser.add_argument("--save_adapted", action="store_true")
    parser.add_argument("--seed", type=int, default=1445)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None, dimension: str = "three_d", modality: str = "ct") -> int:
    parser = build_parser(f"DeTTA {dimension} {modality.upper()} test-time adaptation", dimension, modality)
    args = parser.parse_args(argv)
    if args.gpu is not None:
        args.device = f"cuda:{args.gpu}"
    return run_detta(args, dimension, modality)
