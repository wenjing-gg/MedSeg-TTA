from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from medseg_tta.methods.prior_estimation.adami.three_d.legacy.unet3d import UNet3d
from medseg_tta.models.nnunet import PlainConvUNet


SUPPORTED_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".mhd", ".npy", ".npz")
BRATS_MODALITIES = ("t1c", "t1n", "t2w", "t2f")


def resolve_dirs(
    target_dir: Optional[str],
    image_dir: Optional[str],
    mask_dir: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if image_dir and mask_dir:
        return image_dir, mask_dir
    if target_dir:
        return image_dir or os.path.join(target_dir, "image"), mask_dir or os.path.join(target_dir, "mask")
    return None, None


def _auto_select_subfolder(dataset_path: str, dataset_type: str) -> str:
    if not os.path.exists(dataset_path):
        return "CT_" if dataset_type == "CT" else ""
    subfolders = [
        item for item in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, item))
    ]
    if not subfolders:
        return ""
    underscore_folders = sorted(item for item in subfolders if item.endswith("_"))
    return underscore_folders[0] if underscore_folders else sorted(subfolders)[0]


def get_dataset_paths(
    dataset_type: str,
    base_dir: str = "/home/yuwenjing/data/tta_dataset",
    subfolder: Optional[str] = None,
) -> tuple[str, str]:
    dataset_mapping = {"CT": "TTA-3DCT"}
    if dataset_type not in dataset_mapping:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    dataset_path = os.path.join(base_dir, dataset_mapping[dataset_type])
    if subfolder is None:
        subfolder = _auto_select_subfolder(dataset_path, dataset_type)
    if subfolder:
        return os.path.join(dataset_path, subfolder, "image"), os.path.join(dataset_path, subfolder, "mask")
    return os.path.join(dataset_path, "image"), os.path.join(dataset_path, "mask")


def match_channels(image: torch.Tensor, input_channels: int) -> torch.Tensor:
    if image.shape[0] == input_channels:
        return image
    if image.shape[0] == 1 and input_channels > 1:
        return image.repeat(input_channels, 1, 1, 1)
    if image.shape[0] > input_channels:
        return image[:input_channels]
    repeats = int(np.ceil(input_channels / image.shape[0]))
    return image.repeat(repeats, 1, 1, 1)[:input_channels]


def binarize_label_tensor(label_tensor: torch.Tensor, positive_ids: list[int]) -> torch.Tensor:
    if label_tensor.ndim == 4 and label_tensor.shape[0] == 1:
        label_tensor = label_tensor.squeeze(0)
    mask = torch.zeros_like(label_tensor, dtype=torch.bool)
    for positive_id in positive_ids:
        mask |= label_tensor == positive_id
    return mask.long().unsqueeze(0)


class Prompt3D(nn.Module):
    """Low-frequency multiplicative prompt for 3D Fourier amplitudes."""

    def __init__(self, channels: int, image_size: tuple[int, int, int], prompt_alpha: float):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.prompt_size = tuple(max(1, int(size * prompt_alpha)) for size in image_size)
        self.padding = tuple((size - prompt) // 2 for size, prompt in zip(image_size, self.prompt_size))
        init_prompt = torch.ones((1, channels, *self.prompt_size), dtype=torch.float32)
        self.data_prompt = nn.Parameter(init_prompt, requires_grad=True)

    def update(self, init_data: torch.Tensor) -> None:
        with torch.no_grad():
            self.data_prompt.copy_(init_data.to(self.data_prompt.device, dtype=self.data_prompt.dtype))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, depth, height, width = x.shape
        pd, ph, pw = self.prompt_size
        zd, yh, xw = self.padding

        fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        amp = torch.fft.fftshift(torch.abs(fft), dim=(-3, -2, -1))
        phase = torch.angle(fft)

        prompt = F.pad(
            self.data_prompt,
            [xw, width - xw - pw, yh, height - yh - ph, zd, depth - zd - pd],
            mode="constant",
            value=1.0,
        ).contiguous()
        prompted_amp = torch.fft.ifftshift(amp * prompt, dim=(-3, -2, -1))
        real = torch.cos(phase) * prompted_amp
        imag = torch.sin(phase) * prompted_amp
        prompted_fft = torch.complex(real=real, imag=imag)
        prompted_x = torch.fft.ifftn(prompted_fft, dim=(-3, -2, -1)).real
        low_freq = amp[:, :, zd:zd + pd, yh:yh + ph, xw:xw + pw]
        return prompted_x, low_freq


class PromptMemory3D:
    def __init__(self, size: int, dimension: int):
        self.size = size
        self.dimension = dimension
        self.memory: OrderedDict[bytes, np.ndarray] = OrderedDict()

    def push(self, keys: np.ndarray, prompts: np.ndarray) -> None:
        keys = keys.reshape(len(keys), self.dimension).astype(np.float32)
        prompts = prompts.astype(np.float32)
        if len(prompts) == 1 and len(keys) > 1:
            prompts = np.repeat(prompts, len(keys), axis=0)
        prompts = prompts.reshape(len(keys), *prompts.shape[1:])
        for key, prompt in zip(keys, prompts):
            if len(self.memory) >= self.size:
                self.memory.popitem(last=False)
            self.memory[key.tobytes()] = prompt

    def get_neighbours(self, keys: np.ndarray, k: int) -> torch.Tensor:
        if not self.memory:
            raise RuntimeError("Prompt memory is empty.")
        keys = keys.reshape(len(keys), self.dimension).astype(np.float32)
        all_keys = np.stack(
            [np.frombuffer(key, dtype=np.float32) for key in self.memory.keys()],
            axis=0,
        )
        prompts = list(self.memory.values())
        k = min(k, len(prompts))
        samples = []
        for key in keys:
            denom = np.linalg.norm(all_keys, axis=1) * max(np.linalg.norm(key), 1e-8)
            similarity = np.dot(all_keys, key.T) / np.maximum(denom, 1e-8)
            top_k = np.argpartition(similarity, -k)[-k:]
            weights = similarity[top_k] / 0.2
            weights = np.exp(weights - np.max(weights))
            weights = weights / max(np.sum(weights), 1e-8)
            blended = sum(prompts[index] * weight for index, weight in zip(top_k, weights))
            samples.append(torch.from_numpy(blended).float())
        return torch.stack(samples)


class VolumeFolderDataset(Dataset):
    def __init__(
        self,
        target_dir: str,
        image_dir: str | None,
        mask_dir: str | None,
        image_size: tuple[int, int, int],
        modality: str,
        intensity_range: tuple[float, float] | None,
        input_channels: int,
    ):
        self.image_dir = Path(image_dir or Path(target_dir) / "image")
        self.mask_dir = Path(mask_dir or Path(target_dir) / "mask") if (mask_dir or target_dir) else None
        self.image_size = image_size
        self.modality = modality.lower()
        self.intensity_range = intensity_range
        self.input_channels = input_channels
        self.items = self._collect_items()
        if not self.items:
            raise FileNotFoundError(f"No supported 3D volumes found in {self.image_dir}")

    def _collect_items(self) -> list[tuple[Path, Path | None]]:
        images = []
        for ext in SUPPORTED_EXTENSIONS:
            images.extend(Path(path) for path in glob.glob(str(self.image_dir / f"*{ext}")))
        images = sorted(images)
        return [(image, self._find_mask(image)) for image in images]

    def _find_mask(self, image: Path) -> Path | None:
        if self.mask_dir is None or not self.mask_dir.exists():
            return None
        stem = image.name
        for ext in SUPPORTED_EXTENSIONS:
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        candidates = [stem, f"{stem}_mask", f"{stem}_seg", f"{stem}_label", f"{stem}_gt", stem.replace("image", "mask")]
        for candidate in candidates:
            for ext in SUPPORTED_EXTENSIONS:
                path = self.mask_dir / f"{candidate}{ext}"
                if path.exists():
                    return path
        return None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | bool]:
        image_path, mask_path = self.items[index]
        image = self._load_volume(image_path).astype(np.float32)
        image = self._normalize(image)
        image_tensor = torch.from_numpy(self._image_to_channels(image)).float().unsqueeze(0)
        image_tensor = F.interpolate(image_tensor, size=self.image_size, mode="trilinear", align_corners=False)
        image_tensor = image_tensor.squeeze(0)

        has_mask = mask_path is not None
        if has_mask:
            mask = self._load_volume(mask_path).astype(np.float32)
            mask = self._mask_to_volume(mask)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = F.interpolate(mask_tensor, size=self.image_size, mode="nearest").squeeze(0).long()
        else:
            mask_tensor = torch.zeros((1, *self.image_size), dtype=torch.long)
        return {"image": image_tensor, "mask": mask_tensor, "name": image_path.name, "has_mask": has_mask}

    def _load_volume(self, path: Path) -> np.ndarray:
        suffix = "".join(path.suffixes[-2:]) if path.name.endswith(".nii.gz") else path.suffix
        if suffix in {".npy", ".npz"}:
            data = np.load(path)
            if isinstance(data, np.lib.npyio.NpzFile):
                key = data.files[0]
                return np.asarray(data[key])
            return np.asarray(data)
        if suffix in {".mha", ".mhd"}:
            import SimpleITK as sitk

            return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        import nibabel as nib

        return np.asarray(nib.load(str(path)).get_fdata())

    def _image_to_channels(self, image: np.ndarray) -> np.ndarray:
        image = np.squeeze(image)
        if image.ndim == 3:
            channels = image[None]
        elif image.ndim == 4:
            if image.shape[0] <= 8:
                channels = image
            elif image.shape[-1] <= 8:
                channels = np.moveaxis(image, -1, 0)
            else:
                raise ValueError(
                    f"Cannot infer channel axis for volume with shape {image.shape}. "
                    "Expected D,H,W, C,D,H,W, or D,H,W,C."
                )
        else:
            raise ValueError(f"Expected a 3D or 4D image volume, got shape {image.shape}")

        channels = channels.astype(np.float32)
        if channels.shape[0] == self.input_channels:
            return channels
        if channels.shape[0] == 1 and self.input_channels > 1:
            return np.repeat(channels, self.input_channels, axis=0)
        if channels.shape[0] > self.input_channels:
            return channels[: self.input_channels]
        repeats = int(np.ceil(self.input_channels / channels.shape[0]))
        return np.tile(channels, (repeats, 1, 1, 1))[: self.input_channels]

    def _mask_to_volume(self, mask: np.ndarray) -> np.ndarray:
        mask = np.squeeze(mask)
        if mask.ndim == 4:
            if mask.shape[0] <= 8:
                mask = mask[0]
            elif mask.shape[-1] <= 8:
                mask = mask[..., 0]
        if mask.ndim != 3:
            raise ValueError(f"Expected a 3D mask volume, got shape {mask.shape}")
        return mask

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if self.modality == "ct" and self.intensity_range is not None:
            low, high = self.intensity_range
            image = np.clip(image, low, high)
            return (image - low) / max(high - low, 1e-6)
        mean = float(image.mean())
        std = float(image.std())
        if std > 1e-6:
            return (image - mean) / std
        return image - mean


class CTDataset3D(Dataset):
    """SaTTCA/TENT-style 3D CT reader for image/ and mask/ folders."""

    supported_extensions = (".nii.gz", ".nii", ".mha", ".mhd")

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: tuple[int, int, int],
        spacing: tuple[float, float, float],
        intensity_range: tuple[float, float],
        input_channels: int,
        positive_labels: list[int],
        normalize: bool = True,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.spacing = spacing
        self.intensity_range = intensity_range
        self.input_channels = input_channels
        self.positive_labels = positive_labels
        self.normalize = normalize
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"Mask directory does not exist: {mask_dir}")
        self.data_dicts = self._collect_data_pairs()
        if not self.data_dicts:
            raise ValueError(f"No valid CT image-mask pairs found under:\n  {image_dir}\n  {mask_dir}")
        self.transforms = self._get_test_transforms()

    def _collect_data_pairs(self) -> list[dict[str, str]]:
        image_files: list[str] = []
        for ext in self.supported_extensions:
            image_files.extend(glob.glob(os.path.join(self.image_dir, f"*{ext}")))
        pairs = []
        for image_path in sorted(image_files):
            image_name = os.path.basename(image_path)
            base_name = self._base_name(image_name)
            mask_path = self._find_mask_path(base_name)
            if mask_path:
                pairs.append({"image": image_path, "label": mask_path, "image_name": image_name})
            else:
                print(f"[Warning] Skip CT image without matched mask: {image_name}")
        return pairs

    def _base_name(self, filename: str) -> str:
        for ext in self.supported_extensions:
            if filename.endswith(ext):
                return filename[: -len(ext)]
        return os.path.splitext(filename)[0]

    def _find_mask_path(self, base_name: str) -> Optional[str]:
        if base_name.endswith("-image"):
            patterns = [base_name[:-6] + "-liver_mask"]
        else:
            patterns = [
                base_name,
                f"{base_name}_seg",
                f"{base_name}_segmentation",
                f"{base_name}_mask",
                f"{base_name}_label",
                f"{base_name}_gt",
                f"{base_name}-liver_mask",
                f"{base_name}-mask",
            ]
        for pattern in patterns:
            for ext in self.supported_extensions:
                mask_path = os.path.join(self.mask_dir, f"{pattern}{ext}")
                if os.path.exists(mask_path):
                    return mask_path
        return None

    def _get_test_transforms(self):
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

        transforms_list = [
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
        ]
        if self.normalize:
            transforms_list.append(NormalizeIntensityd(keys=["image"], nonzero=True))
        transforms_list.append(ToTensord(keys=["image", "label"]))
        return Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.data_dicts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | bool]:
        item = self.transforms(self.data_dicts[index].copy())
        image = item["image"].float()
        label = item["label"].long()
        return {
            "image": match_channels(image, self.input_channels),
            "mask": binarize_label_tensor(label, self.positive_labels),
            "name": self.data_dicts[index]["image_name"],
            "has_mask": True,
        }


class BraTSMRIDataset(Dataset):
    """SaTTCA/TENT-style BraTS reader for case folders and modality suffixes."""

    def __init__(
        self,
        target_root: str,
        img: str,
        image_size: tuple[int, int, int],
        input_channels: int,
    ):
        self.target_root = target_root
        self.img = img
        self.image_size = image_size
        self.input_channels = input_channels
        if not os.path.exists(target_root):
            raise FileNotFoundError(f"Target MRI root does not exist: {target_root}")
        self.case_names = [
            item for item in sorted(os.listdir(target_root))
            if not item.startswith(".") and os.path.isdir(os.path.join(target_root, item))
        ]
        if not self.case_names:
            raise ValueError(f"No BraTS-style case folders found under {target_root}")
        self.transforms = self._get_infer_transform()

    def _get_infer_transform(self):
        import monai.transforms as transforms

        base = []
        if self.img == "all":
            base.extend([
                transforms.ConcatItemsd(keys=list(BRATS_MODALITIES), name="image", dim=0),
                transforms.DeleteItemsd(keys=list(BRATS_MODALITIES)),
            ])
        base.extend([
            transforms.SpatialPadD(keys=["image", "label"], spatial_size=(218, 218, 218), method="symmetric", mode="constant"),
            transforms.Resized(keys=["label"], spatial_size=self.image_size, mode="nearest"),
            transforms.Resized(keys=["image"], spatial_size=self.image_size, mode="trilinear"),
            transforms.NormalizeIntensityd(keys=["image"], nonzero=True, dtype=np.float32),
            transforms.EnsureTyped(keys=["image", "label"]),
        ])
        return transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.case_names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | bool]:
        name = self.case_names[index]
        base_path = os.path.join(self.target_root, name, name)
        data: dict[str, np.ndarray] = {}
        if self.img == "all":
            for modality in BRATS_MODALITIES:
                data[modality] = self._load_nii(f"{base_path}-{modality}.nii.gz")[None].astype(np.float32)
        else:
            data["image"] = self._load_nii(f"{base_path}-{self.img}.nii.gz")[None].astype(np.float32)
        label = self._load_nii(f"{base_path}-seg.nii.gz")[None].astype(np.float32)
        label[label == 4] = 0
        data["label"] = label
        item = self.transforms(data)
        return {
            "image": match_channels(item["image"].float(), self.input_channels),
            "mask": item["label"].long(),
            "name": name,
            "has_mask": True,
        }

    def _load_nii(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        import nibabel as nib

        proxy = nib.load(path)
        data = proxy.get_fdata()
        proxy.uncache()
        return np.asarray(data)


def build_test_loader(
    args: argparse.Namespace,
    modality: str,
    image_size: tuple[int, int, int],
) -> DataLoader:
    if modality == "ct":
        image_dir, mask_dir = resolve_dirs(args.target_dir, args.image_dir, args.mask_dir)
        if image_dir is None or mask_dir is None:
            image_dir, mask_dir = get_dataset_paths(args.dataset_type, args.base_dir, args.subfolder)
        dataset = CTDataset3D(
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            spacing=tuple(args.spacing),
            intensity_range=tuple(args.intensity_range),
            input_channels=args.input_channels,
            positive_labels=args.positive_labels,
        )
    elif args.brats_layout:
        target_root = args.target_root or args.target_dir
        if target_root is None:
            raise ValueError("MRI BraTS layout requires --target_root or --target_dir")
        dataset = BraTSMRIDataset(
            target_root=target_root,
            img=args.img,
            image_size=image_size,
            input_channels=args.input_channels,
        )
    else:
        dataset = VolumeFolderDataset(
            target_dir=args.target_dir,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            image_size=image_size,
            modality=modality,
            intensity_range=tuple(args.intensity_range) if args.intensity_range else None,
            input_channels=args.input_channels,
        )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def output_root(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    root = args.target_root or args.target_dir or os.getcwd()
    return Path(root) / "vptta3d_results"


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


def dice_score(output: torch.Tensor, mask: torch.Tensor, num_classes: int) -> float:
    with torch.no_grad():
        pred = prediction_from_output(output)
        scores = []
        label_count = 2 if num_classes <= 1 else num_classes
        for label in range(1, label_count):
            pred_label = pred == label
            mask_label = mask == label
            denom = pred_label.sum() + mask_label.sum()
            if denom > 0:
                scores.append((2 * (pred_label & mask_label).sum() / denom).item())
        return float(np.mean(scores)) if scores else 0.0


def dice_scores_per_case(output: torch.Tensor, mask: torch.Tensor, num_classes: int) -> list[float]:
    return [
        dice_score(output[index:index + 1], mask[index:index + 1], num_classes)
        for index in range(output.shape[0])
    ]


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model_type == "nnunet":
        model = PlainConvUNet(
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
    else:
        model = UNet3d(in_chns=args.input_channels, n_classes=args.num_classes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint
    cleaned = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned, strict=not args.non_strict)
    return model


def build_parser(description: str, modality: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    default_channels = 4 if modality == "mri" else 1
    default_classes = 4 if modality == "mri" else 2
    parser.add_argument("--checkpoint", type=str, required=True, help="shared initialized source checkpoint")
    parser.add_argument("--target_dir", type=str, default=None, help="target root; for CT it contains image/ and mask/")
    parser.add_argument("--target_root", type=str, default=None, help="BraTS-style target root used by SaTTCA/TENT MRI loaders")
    parser.add_argument("--source_root", type=str, default=None, help="kept for SaTTCA/TENT CLI compatibility")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default="CT")
    parser.add_argument("--base_dir", type=str, default="/home/yuwenjing/data/tta_dataset")
    parser.add_argument("--subfolder", type=str, default=None)
    parser.add_argument("--img", type=str, default="all" if modality == "mri" else "ct", choices=["all", "t1c", "t1n", "t2w", "t2f", "ct"])
    parser.add_argument("--brats_layout", action=argparse.BooleanOptionalAction, default=modality == "mri")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_type", choices=["unet3d", "nnunet"], default="unet3d")
    parser.add_argument("--input_channels", type=int, default=default_channels)
    parser.add_argument("--num_classes", type=int, default=default_classes)
    parser.add_argument("--image_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_test", type=int, default=None, help="SaTTCA/TENT-compatible alias for --batch_size")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--adapt_steps", type=int, default=1)
    parser.add_argument("--memory_size", type=int, default=40)
    parser.add_argument("--neighbor", type=int, default=16)
    parser.add_argument("--prompt_alpha", type=float, default=0.05)
    parser.add_argument("--intensity_range", type=float, nargs=2, default=(-200.0, 400.0) if modality == "ct" else None)
    parser.add_argument("--positive_labels", type=str, default="1")
    parser.add_argument("--nnunet_features", type=int, nargs=6, default=[32, 64, 125, 256, 320, 320])
    parser.add_argument("--non_strict", action="store_true", help="allow partial checkpoint loading")
    parser.add_argument("--save_predictions", action="store_true", help="write adapted hard predictions as compressed npz files")
    parser.add_argument("--save_prompt", action="store_true", help="write the final learned prompt tensor")
    return parser


def run_vptta3d(args: argparse.Namespace, modality: str) -> int:
    if args.batch_test is not None:
        args.batch_size = args.batch_test
    if isinstance(args.positive_labels, str):
        args.positive_labels = [int(item) for item in args.positive_labels.split(",") if item.strip()]
    if not args.positive_labels:
        args.positive_labels = [1]
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    image_size = tuple(int(item) for item in args.image_size)
    loader = build_test_loader(args, modality, image_size)

    model = build_model(args).to(device)
    model.requires_grad_(False)
    prompt = Prompt3D(args.input_channels, image_size, args.prompt_alpha).to(device)
    optimizer = torch.optim.Adam(prompt.parameters(), lr=args.lr)
    memory = PromptMemory3D(size=args.memory_size, dimension=prompt.data_prompt.numel())
    dice_values = []
    metric_rows = []

    output_dir = output_root(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc=f"VPTTA-3D-{modality.upper()}"):
        image = batch["image"].to(device=device, dtype=torch.float32)
        mask = batch["mask"].to(device=device)
        has_masks = torch.as_tensor(batch["has_mask"]).bool().view(-1).tolist()

        model.eval()
        prompt.train()
        if len(memory.memory) >= args.neighbor:
            with torch.no_grad():
                _, low_freq = prompt(image)
            init_prompt = memory.get_neighbours(low_freq.detach().cpu().numpy(), args.neighbor).to(device)
        else:
            init_prompt = torch.ones_like(prompt.data_prompt.detach())
        prompt.update(init_prompt)

        for _ in range(args.adapt_steps):
            prompted_image, _ = prompt(image)
            logits = model(prompted_image)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = entropy_loss(logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        prompt.eval()
        with torch.no_grad():
            prompted_image, low_freq = prompt(image)
            logits = model(prompted_image)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
        memory.push(low_freq.detach().cpu().numpy(), prompt.data_prompt.detach().cpu().numpy())
        pred = prediction_from_output(logits.detach()).cpu().numpy().astype(np.uint8)
        if args.save_predictions:
            names = batch["name"] if isinstance(batch["name"], list) else [batch["name"]]
            for item, name in zip(pred, names):
                np.savez_compressed(output_dir / f"{Path(name).stem}_pred.npz", pred=item.squeeze(0))
        if any(has_masks):
            names = batch["name"] if isinstance(batch["name"], list) else [batch["name"]]
            scores = dice_scores_per_case(logits.detach(), mask, args.num_classes)
            for name, score, case_has_mask in zip(names, scores, has_masks):
                if not case_has_mask:
                    continue
                dice_values.append(score)
                metric_rows.append({"case": str(name), "dice": f"{score:.6f}"})

    if dice_values:
        mean_dice = float(np.mean(dice_values))
        print(f"Mean Dice: {mean_dice:.6f}")
        (output_dir / "metrics.txt").write_text(f"mean_dice={mean_dice:.6f}\n", encoding="utf-8")
        with (output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["case", "dice"])
            writer.writeheader()
            writer.writerows(metric_rows)
    else:
        print("No masks found; adaptation completed without metric calculation.")
    if args.save_prompt:
        torch.save(prompt.data_prompt.detach().cpu(), output_dir / f"vptta3d_{modality}_prompt.pt")
    return 0


def main(argv: Iterable[str] | None = None, modality: str = "mri") -> int:
    parser = build_parser(f"VPTTA 3D {modality.upper()} adaptation", modality)
    args = parser.parse_args(argv)
    return run_vptta3d(args, modality)
