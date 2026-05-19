import os
import sys
import cv2
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from diffusers import DDPMScheduler

from dataset.utils_brats_all import get_data_loader
from diffusion.controlnet.models.UNet2DModel import UNet2DModel
from diffusion.controlnet.models.controlnet import ControlNetModel
from diffusion.controlnet.models.pipeline_controlnet import DDPMControlNetPipeline
from utils import set_seed

sys.path.append('.')


@dataclass
class Config:
    output_dir: str
    unet_ckpt_dir: str
    controlnet_ckpt_dir: str
    source_root: str
    target_root: str
    train_path: str
    test_path: str
    img: str
    mode: str
    batch_size: int
    num_workers: int
    r_steps: np.ndarray
    run_num: int
    num_inference_steps: int
    seed: int
    device: int


def get_args():
    script_dir = Path(__file__).resolve().parent
    artifact_dir = script_dir.parent / "artifacts"
    parser = argparse.ArgumentParser(description="RSA translation stage")
    parser.add_argument("--output-dir", type=str, default=str(artifact_dir / "translated"))
    parser.add_argument("--unet-ckpt-dir", type=str, default=str(artifact_dir / "checkpoints" / "vs_ddpm"))
    parser.add_argument("--controlnet-ckpt-dir", type=str, default=str(artifact_dir / "checkpoints" / "vs_controlnet"))
    parser.add_argument("--source-root", type=str, default=str(artifact_dir / "source"))
    parser.add_argument("--target-root", type=str, default=str(artifact_dir / "target"))
    parser.add_argument("--train-path", type=str, default="train")
    parser.add_argument("--test-path", type=str, default="test")
    parser.add_argument("--img", type=str, default="t2f")
    parser.add_argument("--mode", type=str, default="target_to_target")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--r0", type=float, default=30.0)
    parser.add_argument("--r1", type=float, default=80.0)
    parser.add_argument("--n-steps", type=int, default=2)
    parser.add_argument("--run-num", type=int, default=3)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def convert_3d_to_2d(tensor_3d):
    batch_size, channels, depth, height, width = tensor_3d.shape
    return tensor_3d.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, channels, height, width)


def generate_condition_batch(image_2d: torch.Tensor, threshold: float) -> torch.Tensor:
    images = image_2d.detach().cpu().numpy()
    conditions = []
    for image in images:
        image = np.squeeze(image)
        image = (image * 255).astype(np.uint8)
        image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0)
        edge = cv2.Canny(image, int(threshold), int(threshold))
        condition = torch.from_numpy(edge).unsqueeze(0).float() / 255.0
        conditions.append(condition)
    return torch.stack(conditions, dim=0)


if __name__ == "__main__":
    args = get_args()
    config = Config(
        output_dir=args.output_dir,
        unet_ckpt_dir=args.unet_ckpt_dir,
        controlnet_ckpt_dir=args.controlnet_ckpt_dir,
        source_root=args.source_root,
        target_root=args.target_root,
        train_path=args.train_path,
        test_path=args.test_path,
        img=args.img,
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        r_steps=np.linspace(float(args.r0), float(args.r1), int(args.n_steps)),
        run_num=args.run_num,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        device=args.device,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{config.device}")
    set_seed(config.seed)

    noise_scheduler = DDPMScheduler.from_pretrained(config.unet_ckpt_dir, subfolder='scheduler')
    unet = UNet2DModel.from_pretrained(config.unet_ckpt_dir, subfolder='unet')
    controlnet = ControlNetModel.from_pretrained(config.controlnet_ckpt_dir, subfolder='controlnet')
    unet.to(device)
    controlnet.to(device)
    pipeline = DDPMControlNetPipeline(
        contronet=controlnet,
        unet=unet,
        scheduler=noise_scheduler,
        use_bar=False,
    )

    train_loader, _ = get_data_loader(
        source_root=config.source_root,
        target_root=config.target_root,
        train_path=config.train_path,
        test_path=config.test_path,
        batch_train=config.batch_size,
        batch_test=config.batch_size,
        nw=config.num_workers,
        img=config.img,
        mode=config.mode,
    )

    for sampled_batch in train_loader:
        image_batch = sampled_batch["image"].to(device)
        image = convert_3d_to_2d(image_batch)
        case_samples = []
        for threshold in config.r_steps:
            conditions = generate_condition_batch(image, threshold).to(device)
            for run_idx in range(config.run_num):
                samples = pipeline(
                    conditions,
                    num_inference_steps=config.num_inference_steps,
                    generator=torch.manual_seed(config.seed + run_idx),
                    output_type='numpy',
                )[0]
                samples = np.squeeze(samples, axis=-1)
                case_samples.append(samples)
        samples = np.stack(case_samples, axis=0)
        img_id = sampled_batch.get("name", [str(sampled_batch["idx"][0].item())])[0]
        sio.savemat(
            f"{config.output_dir}/{img_id}.mat",
            {
                "samples": samples,
                "thresholds": np.asarray(config.r_steps, dtype=np.float32),
            },
        )
