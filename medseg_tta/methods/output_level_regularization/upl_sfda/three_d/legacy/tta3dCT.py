from __future__ import annotations

import argparse
import copy
import csv
import datetime
import glob
import os
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from medpy.metric.binary import hd95 as hd95_medpy
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from unet3d import UNet3d
from utils import parse_config, set_random


def _safe_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def dice_score(pred, target, eps=1e-5):
    inter = float(np.logical_and(pred, target).sum())
    total = float(pred.sum() + target.sum())
    if total == 0:
        return 1.0
    return (2.0 * inter + eps) / (total + eps)


def iou_score(pred, target, eps=1e-5):
    inter = float(np.logical_and(pred, target).sum())
    union = float(np.logical_or(pred, target).sum())
    if union == 0:
        return 1.0
    return (inter + eps) / (union + eps)


def pa_score(pred, target, eps=1e-5):
    correct = float((pred == target).sum())
    total = float(target.size)
    return (correct + eps) / (total + eps)


def rve_score(pred, target):
    pred_vol = float(pred.sum())
    target_vol = float(target.sum())
    if target_vol == 0:
        return 0.0 if pred_vol == 0 else 1.0
    return abs(pred_vol - target_vol) / target_vol


def sensitivity_score(pred, target, eps=1e-5):
    tp = float(np.logical_and(pred, target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    return (tp + eps) / (tp + fn + eps)


def ppv_score(pred, target, eps=1e-5):
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    return (tp + eps) / (tp + fp + eps)


def hd95_score(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return 373.1287
    try:
        return float(hd95_medpy(pred, target))
    except Exception:
        return 373.1287


def merge_logits_to_binary(logits: torch.Tensor, bg_channel: int = 0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    bg = probs[:, bg_channel : bg_channel + 1]
    fg = (probs.sum(dim=1, keepdim=True) - bg).clamp(min=0.0, max=1.0)
    return torch.cat([bg, fg], dim=1)


def binary_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, bg_channel: int = 0):
    probs = merge_logits_to_binary(logits, bg_channel=bg_channel)
    pred = torch.argmax(probs, dim=1).detach().cpu().numpy().astype(np.uint8) > 0
    target = labels.detach().cpu().numpy().astype(np.uint8) > 0
    pred = np.squeeze(pred)
    target = np.squeeze(target)
    return {
        "dice": dice_score(pred, target),
        "hd95": hd95_score(pred, target),
        "iou": iou_score(pred, target),
        "pa": pa_score(pred, target),
        "rve": rve_score(pred, target),
        "sensitivity": sensitivity_score(pred, target),
        "ppv": ppv_score(pred, target),
    }


class CTDataset3D(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size=(128, 128, 128),
        spacing=(1.0, 1.0, 1.0),
        intensity_range=(-200, 400),
        positive_labels=None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.spacing = spacing
        self.intensity_range = intensity_range
        self.positive_labels = positive_labels or [1]
        self.supported_extensions = [".nii.gz", ".nii", ".mha", ".mhd"]
        self.data_dicts = self._collect_data_pairs()
        if not self.data_dicts:
            raise ValueError(f"No valid CT pairs found in {image_dir} and {mask_dir}")
        self.transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=self.spacing, mode=("bilinear", "nearest")),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=self.intensity_range[0],
                    a_max=self.intensity_range[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Resized(keys=["image", "label"], spatial_size=self.image_size, mode=("trilinear", "nearest")),
                NormalizeIntensityd(keys=["image"], nonzero=True),
                ToTensord(keys=["image", "label"]),
            ]
        )

    def _collect_data_pairs(self):
        image_files = []
        for ext in self.supported_extensions:
            image_files.extend(glob.glob(os.path.join(self.image_dir, f"*{ext}")))
        image_files.sort()
        pairs = []
        for img_path in image_files:
            base_name = self._get_base_name(os.path.basename(img_path))
            mask_path = self._find_mask_path(base_name)
            if mask_path is not None:
                pairs.append({"image": img_path, "label": mask_path, "image_name": os.path.basename(img_path)})
        return pairs

    def _get_base_name(self, filename: str):
        for ext in self.supported_extensions:
            if filename.endswith(ext):
                return filename[: -len(ext)]
        return Path(filename).stem

    def _find_mask_path(self, base_name: str) -> Optional[str]:
        patterns = [
            base_name,
            f"{base_name}_seg",
            f"{base_name}_segmentation",
            f"{base_name}_mask",
            f"{base_name}_label",
            f"{base_name}_gt",
            f"{base_name}-mask",
            f"{base_name}-liver_mask",
        ]
        if base_name.endswith("-image"):
            patterns.insert(0, base_name[:-6] + "-liver_mask")
        for pattern in patterns:
            for ext in self.supported_extensions:
                candidate = os.path.join(self.mask_dir, f"{pattern}{ext}")
                if os.path.exists(candidate):
                    return candidate
        return None

    def _binarize_label(self, label_tensor: torch.Tensor) -> torch.Tensor:
        if label_tensor.ndim == 4 and label_tensor.shape[0] == 1:
            label_tensor = label_tensor.squeeze(0)
        mask = torch.zeros_like(label_tensor, dtype=torch.bool)
        for label_id in self.positive_labels:
            mask |= label_tensor == label_id
        return mask.long().unsqueeze(0)

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx: int):
        item = self.transforms(self.data_dicts[idx].copy())
        image = item["image"].float()
        label = self._binarize_label(item["label"].long())
        return image, label, self.data_dicts[idx]["image_name"]


def resolve_ct_dirs(target_dir: Optional[str], image_dir: Optional[str], mask_dir: Optional[str]):
    if image_dir and mask_dir:
        return image_dir, mask_dir
    if target_dir:
        return os.path.join(target_dir, "image"), os.path.join(target_dir, "mask")
    raise ValueError("Please provide either --target_dir or both --image_dir and --mask_dir")


def get_ct_loader(args):
    image_dir, mask_dir = resolve_ct_dirs(args.target_dir, args.image_dir, args.mask_dir)
    dataset = CTDataset3D(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=(args.image_size, args.image_size, args.image_size),
        spacing=tuple(args.spacing),
        intensity_range=tuple(args.intensity_range),
        positive_labels=args.positive_labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, image_dir


def build_config(args):
    config = parse_config(args.config)
    config.setdefault("train", {})
    config.setdefault("network", {})
    config["train"]["dataset"] = "ct"
    config["train"]["gpu"] = args.gpu
    config["train"]["lr"] = args.lr
    config["train"]["num_classes"] = args.model_num_classes
    config["train"]["pl_threshold_mms"] = args.pl_threshold
    config["network"]["in_chns"] = args.input_channels
    config["network"]["n_classes_mms"] = args.model_num_classes
    return config


def resolve_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def evaluate_case(model, image, label, bg_channel):
    model.eval()
    with torch.no_grad():
        logits = model(image)
    return binary_metrics_from_logits(logits, label.squeeze(1), bg_channel=bg_channel)


def adapt_case(model, image, adapt_steps):
    history = []
    for _ in range(adapt_steps):
        model.train()
        model.save_nii(image)
        loss, entropy = model.train_target(image)
        history.append({"loss": _safe_float(loss), "entropy": _safe_float(entropy)})
    return history


def save_report(result_dir, rows, summary, args):
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(result_dir, f"upl_sfda_ct_{timestamp}.csv")
    txt_path = os.path.join(result_dir, f"upl_sfda_ct_{timestamp}.txt")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["file"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("UPL-SFDA CT Test-Time Adaptation Report\n")
        handle.write(f"time: {timestamp}\n")
        handle.write(f"target_dir: {args.target_dir or args.image_dir}\n")
        handle.write(f"model_path: {args.model_path}\n")
        handle.write(f"adapt_steps: {args.adapt_steps}\n")
        handle.write(f"pl_threshold: {args.pl_threshold}\n\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value:.6f}\n")

    return csv_path, txt_path


def main():
    parser = argparse.ArgumentParser(description="UPL-SFDA CT test-time adaptation for 3D segmentation")
    parser.add_argument("--config", type=str, default="./config/train3d.cfg")
    parser.add_argument("--target_dir", type=str, default=None, help="Directory containing image/ and mask/")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--result_dir", type=str, default="./results/upl_sfda_ct")
    parser.add_argument("--adapt_steps", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--input_channels", type=int, default=1)
    parser.add_argument("--model_num_classes", type=int, default=2)
    parser.add_argument("--bg_channel", type=int, default=0)
    parser.add_argument("--positive_labels", type=int, nargs="+", default=[1])
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--intensity_range", type=float, nargs=2, default=[-200.0, 400.0])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--pl_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_random(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config = build_config(args)
    test_loader, resolved_image_dir = get_ct_loader(args)

    source_model = UNet3d(config).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    source_model.load_state_dict(resolve_checkpoint_state(checkpoint), strict=False)
    source_state = copy.deepcopy(source_model.state_dict())

    rows = []
    summary_store = {key: [] for key in ["dice", "hd95", "iou", "pa", "rve", "sensitivity", "ppv", "adapt_loss", "adapt_entropy"]}

    for image, label, filename in tqdm(test_loader, desc="UPL-SFDA CT"):
        image = image.to(device)
        label = label.to(device)

        case_model = UNet3d(config).to(device)
        case_model.load_state_dict(source_state, strict=False)

        adapt_history = adapt_case(case_model, image, args.adapt_steps)
        metrics = evaluate_case(case_model, image, label, args.bg_channel)

        mean_loss = float(np.mean([item["loss"] for item in adapt_history])) if adapt_history else 0.0
        mean_entropy = float(np.mean([item["entropy"] for item in adapt_history])) if adapt_history else 0.0

        row = {
            "file": filename[0] if isinstance(filename, (list, tuple)) else filename,
            **{key: round(value, 6) for key, value in metrics.items()},
            "adapt_loss": round(mean_loss, 6),
            "adapt_entropy": round(mean_entropy, 6),
        }
        rows.append(row)

        for key, value in metrics.items():
            summary_store[key].append(value)
        summary_store["adapt_loss"].append(mean_loss)
        summary_store["adapt_entropy"].append(mean_entropy)

    summary = {key: float(np.mean(values)) if values else 0.0 for key, values in summary_store.items()}
    csv_path, txt_path = save_report(args.result_dir, rows, summary, args)

    print("UPL-SFDA CT adaptation finished")
    print(f"target images: {resolved_image_dir}")
    print(f"csv: {csv_path}")
    print(f"report: {txt_path}")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
