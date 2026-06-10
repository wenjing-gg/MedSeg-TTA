import os
import glob
import argparse
import datetime
import traceback
from typing import Tuple, Optional, Dict, List

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# MONAI
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, Resized, NormalizeIntensityd, ToTensord
)

# ==== 你自己的依赖（按需替换为正确的导入路径）====
#from nnunet import PlainConvUNet
from unet3d import UNet3d
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
# ===============================================


# ---------------- 路径相关辅助 ---------------- #
def get_dataset_type_from_path(data_path: str) -> str:
    data_path = data_path.replace('\\', '/').lower()
    if 'tta-3dct' in data_path or 'tta-ct' in data_path or 'ct' in data_path:
        return 'CT'
    return 'CT'  # 默认返回CT


def get_dataset_paths(dataset_type: str,
                      base_dir: str = "/home/yuwenjing/data/tta_dataset",
                      subfolder: str = None) -> Tuple[str, str]:
    """
    根据数据集类型返回 (image_dir, mask_dir)
    仅在未提供 target_dir / image_dir / mask_dir 时才使用。
    """
    dataset_mapping = {'CT': 'TTA-3DCT'}

    if dataset_type not in dataset_mapping:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    dataset_folder = dataset_mapping[dataset_type]
    dataset_path = os.path.join(base_dir, dataset_folder)

    if subfolder is None:
        subfolder = _auto_select_subfolder(dataset_path, dataset_type)

    if subfolder:
        image_dir = os.path.join(dataset_path, subfolder, 'image')
        mask_dir = os.path.join(dataset_path, subfolder, 'mask')
    else:
        image_dir = os.path.join(dataset_path, 'image')
        mask_dir = os.path.join(dataset_path, 'mask')

    return image_dir, mask_dir


def _auto_select_subfolder(dataset_path: str, dataset_type: str) -> str:
    """
    自动选择子文件夹（优先带下划线）
    """
    if not os.path.exists(dataset_path):
        return 'CT_' if dataset_type == 'CT' else ''

    try:
        subfolders = [f for f in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, f))]
    except PermissionError:
        return ''

    if not subfolders:
        return ''

    underscore_folders = sorted([f for f in subfolders if f.endswith('_')])
    return underscore_folders[0] if underscore_folders else sorted(subfolders)[0]


def resolve_dirs(target_dir: Optional[str],
                 image_dir: Optional[str],
                 mask_dir: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    解析优先级：
    1) 若显式提供 image_dir/mask_dir，则使用它们；
    2) 否则若提供 target_dir，则使用 target_dir/image 与 target_dir/mask；
    3) 否则返回 (None, None)，由上层决定是否走 dataset_type + base_dir 的映射。
    """
    if image_dir and mask_dir:
        return image_dir, mask_dir
    if target_dir:
        return (image_dir or os.path.join(target_dir, "image"),
                mask_dir or os.path.join(target_dir, "mask"))
    return None, None


# ---------------- 核心数据集（仅测试/验证预处理） ---------------- #
class CTDataset3D(Dataset):
    """
    3D CT图像分割数据集，支持 .nii/.nii.gz/.mha/.mhd
    仅用于推理/验证；phase 默认 'test'，且不包含任何训练增强。
    """

    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 phase: str = 'test',
                 image_size: Tuple[int, int, int] = (128, 128, 128),
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 intensity_range: Tuple[float, float] = (-200, 400),
                 normalize: bool = True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.phase = phase  # 默认 'test'
        self.image_size = image_size
        self.spacing = spacing
        self.intensity_range = intensity_range
        self.normalize = normalize

        self.supported_extensions = ['.nii.gz', '.nii', '.mha', '.mhd']

        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"Mask directory does not exist: {mask_dir}")

        self.data_dicts = self._collect_data_pairs()
        if len(self.data_dicts) == 0:
            raise ValueError(f"No valid image-mask pairs found under:\n  {image_dir}\n  {mask_dir}")

        print(f"Found {len(self.data_dicts)} valid CT image-mask pairs for {self.phase} phase")
        self.transforms = self._get_test_transforms()  # 仅测试/验证预处理

    def _collect_data_pairs(self) -> List[Dict[str, str]]:
        data_dicts = []
        image_files = []
        for ext in self.supported_extensions:
            image_files.extend(glob.glob(os.path.join(self.image_dir, f"*{ext}")))
        image_files.sort()

        for img_path in image_files:
            img_name = os.path.basename(img_path)
            base_name = self._get_base_name(img_name)
            mask_path = self._find_mask_path(base_name)

            if mask_path and self._is_valid_file(img_path) and self._is_valid_file(mask_path):
                data_dicts.append({'image': img_path, 'label': mask_path, 'image_name': img_name})
            else:
                print(f"[Warning] Skip invalid pair: {img_name}")
        return data_dicts

    def _get_base_name(self, filename: str) -> str:
        for ext in self.supported_extensions:
            if filename.endswith(ext):
                return filename[:-len(ext)]
        return os.path.splitext(filename)[0]

    def _find_mask_path(self, base_name: str) -> Optional[str]:
        # 兼容 LiTS 等命名
        if base_name.endswith('-image'):
            liver_base = base_name[:-6] + '-liver_mask'
            patterns = [liver_base]
        else:
            patterns = [
                base_name,
                f"{base_name}_seg",
                f"{base_name}_segmentation",
                f"{base_name}_mask",
                f"{base_name}_label",
                f"{base_name}_gt",
                f"{base_name}-liver_mask",
                f"{base_name}-mask"
            ]
        for pattern in patterns:
            for ext in self.supported_extensions:
                mask_path = os.path.join(self.mask_dir, f"{pattern}{ext}")
                if os.path.exists(mask_path):
                    return mask_path
        return None

    def _is_valid_file(self, file_path: str) -> bool:
        try:
            if file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
                _ = nib.load(file_path).get_fdata()
            return True
        except Exception:
            return False

    def _get_test_transforms(self):
        transforms_list = [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            Orientationd(keys=['image', 'label'], axcodes="RAS"),
            Spacingd(keys=['image', 'label'], pixdim=self.spacing, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=['image'],
                a_min=self.intensity_range[0], a_max=self.intensity_range[1],
                b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=("trilinear", "nearest")),
        ]
        if self.normalize:
            transforms_list.append(NormalizeIntensityd(keys=['image'], nonzero=True))
        transforms_list.append(ToTensord(keys=['image', 'label']))
        return Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.data_dicts)

    def __getitem__(self, idx: int):
        data_dict = self.data_dicts[idx].copy()
        try:
            data_dict = self.transforms(data_dict)
        except Exception as e:
            print(f"Error applying transforms to {data_dict.get('image_name', '<unknown>')}: {e}")
            raise e

        image = data_dict['image']  # Tensor, shape: (1, D, H, W)
        label = data_dict['label']  # Tensor, shape: (1, D, H, W)
        filename = data_dict['image_name']

        image = image.float() if isinstance(image, torch.Tensor) else torch.tensor(image, dtype=torch.float32)
        label = label.long() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)

        # 保留 label 的通道维度 (1, D, H, W)，与 labels.squeeze(1) 对齐
        return image, label, filename


# ---------------- 仅测试 DataLoader ---------------- #
def get_ct_test_loader(image_dir: str = None,
                       mask_dir: str = None,
                       dataset_type: str = None,
                       subfolder: str = None,
                       base_dir: str = "/home/yuwenjing/data/tta_dataset",
                       target_dir: str = None,
                       batch_size: int = 2,
                       num_workers: int = 4,
                       image_size: Tuple[int, int, int] = (128, 128, 128),
                       spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                       intensity_range: Tuple[float, float] = (-200, 400)) -> Tuple[DataLoader, str]:
    """
    构建仅用于测试/验证的 DataLoader。

    路径优先级：
    1) 若给了 image_dir/mask_dir -> 直接用；
    2) 否则若给了 target_dir -> 使用 target_dir/image & target_dir/mask；
    3) 否则若给了 dataset_type -> 使用 base_dir + 映射；
    4) 否则报错。
    """
    # 先根据 target_dir / image_dir / mask_dir 解析
    image_dir, mask_dir = resolve_dirs(target_dir, image_dir, mask_dir)

    if image_dir is None or mask_dir is None:
        if dataset_type is not None:
            image_dir, mask_dir = get_dataset_paths(dataset_type, base_dir, subfolder)
        else:
            raise ValueError("Please provide either (image_dir & mask_dir) or target_dir, "
                             "or set dataset_type to use base_dir mapping.")

    test_dataset = CTDataset3D(
        image_dir=image_dir, mask_dir=mask_dir, phase='test',
        image_size=image_size, spacing=spacing,
        intensity_range=intensity_range, normalize=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return test_loader, get_dataset_type_from_path(image_dir)


# ---------------- 测试主流程 ---------------- #
def safe_value(val):
    return val.item() if isinstance(val, torch.Tensor) else val


def test_on_target(args, device):
    print(f"\n{'='*40}")
    print(f"🧪 开始在目标数据集上测试 (img: {args.img.upper()}, model: {args.model_type})")
    print(f"{'='*40}\n")

    try:
        # 选择模型
        if args.model_type == 'nnunet':
            model = PlainConvUNet(
                4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3,
                (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None,
                nn.ReLU, deep_supervision=True
            ).to(device)
            print("已选择 nnUNet 模型架构")
            default_model_path = "/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth"
        else:
            model = UNet3d().to(device)
            print("已选择 UNet3d 模型架构")
            default_model_path = "/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth"

        best_model_path = args.checkpoint if args.checkpoint != "default" else default_model_path
        print(f"加载模型权重: {best_model_path}")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"未找到预训练权重: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # 解析目录
        image_dir, mask_dir = resolve_dirs(args.target_dir, args.image_dir, args.mask_dir)
        if not image_dir or not mask_dir:
            raise ValueError("无法解析 image_dir / mask_dir，请确认 --target_dir 或显式传入 --image_dir/--mask_dir")

        print(f"目标数据目录: {args.target_dir}")
        print(f"图像目录: {image_dir}")
        print(f"掩码目录: {mask_dir}")

        # 只构建测试 DataLoader
        target_test_loader, _ = get_ct_test_loader(
            image_dir=image_dir,
            mask_dir=mask_dir,
            target_dir=None,
            dataset_type=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=(args.image_size, args.image_size, args.image_size),
            spacing=args.spacing,
            intensity_range=args.intensity_range
        )

        # 前向一次，获取类别数
        with torch.no_grad():
            first_imgs, first_labels, *_ = next(iter(target_test_loader))
            first_outputs = model(first_imgs.to(device))
            dice_tmp = cal_dice(first_outputs, first_labels.to(device).squeeze(1))
            num_classes = len(dice_tmp)

        def make_empty_lists(n):
            return [[] for _ in range(n)]

        all_dice_values         = make_empty_lists(num_classes)
        all_hd95_values         = make_empty_lists(num_classes)
        all_IoU_values          = make_empty_lists(num_classes)
        all_pa_values           = make_empty_lists(num_classes)
        all_RVE_values          = make_empty_lists(num_classes)
        all_sensitivity_values  = make_empty_lists(num_classes)
        all_ppv_values          = make_empty_lists(num_classes)

        # 推理
        with torch.no_grad():
            for imgs, labels, *_ in tqdm(target_test_loader, desc='推理进度'):
                imgs = imgs.to(device)
                labels = labels.to(device)  # (B, 1, D, H, W)

                outputs = model(imgs)

                dice_values        = cal_dice(outputs, labels.squeeze(1))
                hd95_values        = cal_hd95(outputs, labels.squeeze(1))
                IoU_values         = IoU(outputs, labels.squeeze(1))
                pa_values          = PA(outputs, labels.squeeze(1), num_classes)
                RVE_values         = cal_RVE(outputs, labels.squeeze(1))
                sensitivity_values = cal_sensitivity(outputs, labels.squeeze(1))
                ppv_values         = cal_ppv(outputs, labels.squeeze(1))

                for i in range(num_classes):
                    all_dice_values[i].append(safe_value(dice_values[i]))
                    all_hd95_values[i].append(safe_value(hd95_values[i]))
                    all_IoU_values[i].append(safe_value(IoU_values[i]))
                    all_pa_values[i].append(safe_value(pa_values[i]))
                    all_RVE_values[i].append(safe_value(RVE_values[i]))
                    all_sensitivity_values[i].append(safe_value(sensitivity_values[i]))
                    all_ppv_values[i].append(safe_value(ppv_values[i]))

        # 汇总
        def mean_std(ls):
            return [np.mean(v) for v in ls], [np.std(v) for v in ls]

        mean_dice,         std_dice         = mean_std(all_dice_values)
        mean_hd95,         std_hd95         = mean_std(all_hd95_values)
        mean_IoU,          std_IoU          = mean_std(all_IoU_values)
        mean_pa,           std_pa           = mean_std(all_pa_values)
        mean_RVE,          std_RVE          = mean_std(all_RVE_values)
        mean_sensitivity,  std_sensitivity  = mean_std(all_sensitivity_values)
        mean_ppv,          std_ppv          = mean_std(all_ppv_values)

        # 报告
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "nnUNet" if args.model_type == "nnunet" else "UNet3D"

        lines = [
            "="*40,
            f"测试时间: {timestamp}",
            "测试配置:",
            f"- 图像模态: {args.img}",
            f"- 模型类型: {model_name}",
            f"- 模型路径: {best_model_path}",
            f"- 测试数据: {args.target_dir}",
            f"- 图像目录: {image_dir}",
            f"- 掩码目录: {mask_dir}",
            "",
            "性能指标(均值 ± 标准差):"
        ]
        for i in range(num_classes):
            lines.append(f"\n[Class {i}]")
            lines.append(f"Dice:        {mean_dice[i]:.4f} ± {std_dice[i]:.4f}")
            lines.append(f"HD95(mm):    {mean_hd95[i]:.2f} ± {std_hd95[i]:.2f}")
            lines.append(f"IoU:         {mean_IoU[i]:.4f} ± {std_IoU[i]:.4f}")
            lines.append(f"PA:          {mean_pa[i]:.4f} ± {std_pa[i]:.4f}")
            lines.append(f"RVE:         {mean_RVE[i]:.4f} ± {std_RVE[i]:.4f}")
            lines.append(f"Sensitivity: {mean_sensitivity[i]:.4f} ± {std_sensitivity[i]:.4f}")
            lines.append(f"PPV:         {mean_ppv[i]:.4f} ± {std_ppv[i]:.4f}")

        lines.append("="*40)
        report = "\n".join(lines)
        result_file = os.path.join(args.output_dir, f"test_{args.img}_{args.model_type}_{timestamp}.txt")
        with open(result_file, 'w') as f:
            f.write(report)

        print(report)
        print(f"✅ 结果已保存到: {result_file}")
        return True

    except Exception as e:
        error_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        error_msg = (
            f"\n🔥 测试失败\n"
            f"图像模态: {args.img}\n"
            f"模型类型: {args.model_type}\n"
            f"错误信息: {str(e)}\n"
            f"追踪信息:\n{traceback.format_exc()}"
        )
        print(error_msg)
        os.makedirs(args.output_dir, exist_ok=True)
        error_log = os.path.join(args.output_dir, "test_errors.log")
        with open(error_log, 'a') as f:
            f.write(f"[{error_timestamp}] {error_msg}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test on target dataset (test-only)")
    # 关键参数
    parser.add_argument('--checkpoint',   type=str,
                        default="/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth",
                        help='预训练权重路径，或 "default" 使用默认')
    parser.add_argument('--target_dir',   type=str,
                        default="/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB",
                        help='数据集根目录（包含 image/ 与 mask/ 子目录）')
    parser.add_argument('--batch_size',   type=int, default=2)
    parser.add_argument('--num_workers',  type=int, default=2)
    parser.add_argument('--image_size',   type=int, default=256)
    parser.add_argument('--output_dir',   type=str, default='./target_test_results')

    # 其它可选
    parser.add_argument('--model_type',   type=str, default='unet3d', choices=['unet3d', 'nnunet'])
    parser.add_argument('--img',          type=str, default='ct', help='数据模态标识，仅用于打印')
    parser.add_argument('--spacing',      type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument('--intensity_range', type=float, nargs=2, default=(-200, 400))

    # 显式传入 image_dir/mask_dir（可选；若不传，将自动用 target_dir 下的 image/mask）
    parser.add_argument('--image_dir',    type=str, default=None)
    parser.add_argument('--mask_dir',     type=str, default=None)

    # dataset_type 仅在完全未提供 target_dir 和 image_dir/mask_dir 时才生效（一般用不到）
    parser.add_argument('--dataset_type', type=str, default=None, help='数据集类型（例如 CT）')

    args = parser.parse_args()
    device = torch.device("cuda:0")
    test_on_target(args, device)


if __name__ == "__main__":
    main()
