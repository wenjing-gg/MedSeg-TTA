import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import argparse
import datetime
import traceback
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import copy
import math
# from urils_DLTTA.loss import DiceLoss
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from einops import rearrange, reduce
from utils_uda import FDA_source_to_target
import torch.nn.functional as F

# MONAI
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, Resized, NormalizeIntensityd, ToTensord
)
import models.moment_tta.loss3d as moment_tta_losses
# ==== 你自己的依赖（按需替换为正确的导入路径）====
from medseg_tta.models.nnunet import PlainConvUNet
from unet3d import UNet3d
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
# ===============================================

# ==== TENT ====
# import tent
import torch.optim as optim
# ==============




TENT = ['Weighted_self_entropy_loss',
         {'weights':[1, 10], 'idc':[0, 1], 'act':'sigmoid'}]

TENT_Prostate = ['Weighted_self_entropy_loss',
         {'weights':[1], 'idc':[0], 'act':'sigmoid'}]

RN_w_CR = ['RN_w_CR_loss',
         {'idc':[0, 1], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'3d'}]

RN_w_CR_Prostate = ['RN_w_CR_loss',
         {'idc':[0], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'3d'}]


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "TTA-3DCT"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "uda_mima"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "outputs" / "UDA-MIMA_3D"

LSIZE_RIGA = ['KL_class_ratio_entropy_loss',
         {'weights':[1, 5], 'idc':[0, 1],
         'class_ratio_prior':[0.0708947674981479, 0.01705511685075431], 'act':'sigmoid'}]

LSIZE_Prostate = ['KL_class_ratio_entropy_loss',
         {'weights':[1], 'idc':[0],
         'class_ratio_prior':[0.034565616183810766,0.032456], 'act':'sigmoid_onelabel'}]

LSIZECentroid = ['Constrain_prior_w_self_entropy_loss',
                 {'idc': [0, 1],
                  'weights_se':[0.8, 0.2],'lamb_se':1,
                  'class_ratio_prior':[0.0708947674981479, 0.01705511685075431],
                  'lamb_moment':0.0001, 'temp':1.01,'margin':0,
                  'mom_est':[[254.9747, 255.98499], [253.97656, 255.04263]],
                  'moment_fn':'soft_centroid', 'lamb_consprior':1,
                  'power': 1, 'act':'sigmoid'}]


LSIZEDistCentroid = ['Constrain_prior_w_self_entropy_loss',
                     {'idc':[0, 1],
                      'weights_se':[0.8, 0.2],'lamb_se':1,
                      'class_ratio_prior':[0.0708947674981479, 0.01705511685075431],
                     'lamb_moment':0.0001, 'temp':1.01,'margin':0,
                     'mom_est':[[39.157185, 37.18185 ],[17.430944,18.484102]],
                     'moment_fn':'soft_dist_centroid', 'lamb_consprior':1,
                     'power': 1, 'act':'sigmoid'}]




# ---------------- 路径相关辅助 ---------------- #
def get_dataset_type_from_path(data_path: str) -> str:
    data_path = data_path.replace('\\', '/').lower()
    if 'tta-3dct' in data_path or 'tta-ct' in data_path or 'ct' in data_path:
        return 'CT'
    return 'CT'  # 默认返回CT


def get_dataset_paths(dataset_type: str,
                      base_dir: str = "/home/yuwenjing/data/tta_dataset",
                      subfolder: str = None) -> Tuple[str, str]:
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
    if image_dir and mask_dir:
        return image_dir, mask_dir
    if target_dir:
        return (image_dir or os.path.join(target_dir, "image"),
                mask_dir or os.path.join(target_dir, "mask"))
    return None, None


# ---------------- 标签二值化工具 ---------------- #
def binarize_label_tensor(label_tensor: torch.Tensor, positive_ids: List[int]) -> torch.Tensor:
    """
    将标签张量二值化：label ∈ positive_ids -> 1, 否则 -> 0
    输入可为 (1, D, H, W) 或 (D, H, W)；返回统一为 (1, D, H, W)
    """
    if label_tensor.ndim == 4 and label_tensor.shape[0] == 1:
        lt = label_tensor.squeeze(0)
    else:
        lt = label_tensor
    mask = torch.zeros_like(lt, dtype=torch.bool)
    for pid in positive_ids:
        mask |= (lt == pid)
    bin_label = mask.long()  # 0/1
    return bin_label.unsqueeze(0)


# ---------------- 核心数据集（仅测试/验证预处理） ---------------- #
class CTDataset3D(Dataset):
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 phase: str = 'test',
                 image_size: Tuple[int, int, int] = (128, 128, 128),
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 intensity_range: Tuple[float, float] = (-200, 400),
                 normalize: bool = True,
                 positive_labels: Optional[List[int]] = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.phase = phase
        self.image_size = image_size
        self.spacing = spacing
        self.intensity_range = intensity_range
        self.normalize = normalize
        self.positive_labels = positive_labels or [1]
        self.supported_extensions = ['.nii.gz', '.nii', '.mha', '.mhd']

        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"Mask directory does not exist: {mask_dir}")

        self.data_dicts = self._collect_data_pairs()
        if len(self.data_dicts) == 0:
            raise ValueError(f"No valid image-mask pairs found under:\n  {image_dir}\n  {mask_dir}")

        print(f"Found {len(self.data_dicts)} valid CT image-mask pairs for {self.phase} phase")
        self.transforms = self._get_test_transforms()

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

        image = data_dict['image']  # (1, D, H, W)
        label = data_dict['label']  # (1, D, H, W)
        filename = data_dict['image_name']

        image = image.float() if isinstance(image, torch.Tensor) else torch.tensor(image, dtype=torch.float32)
        label = label.long() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)

        # 二值化标签：只保留肿瘤（前景）=1
        label = binarize_label_tensor(label, self.positive_labels)

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
                       intensity_range: Tuple[float, float] = (-200, 400),
                       positive_labels: Optional[List[int]] = None) -> Tuple[DataLoader, str]:
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
        intensity_range=intensity_range, normalize=True,
        positive_labels=positive_labels
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return test_loader, get_dataset_type_from_path(image_dir)


# ---------------- 工具 ---------------- #
def safe_value(val):
    return val.item() if isinstance(val, torch.Tensor) else val


# [EPISODIC]
def snapshot_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

def reset_optimizer_state(optimizer: optim.Optimizer):
    optimizer.state.clear()


# BN/Dropout 稳定性增强
def freeze_bn_running_stats(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.track_running_stats = False  # 使用当前 batch 统计，不再更新 running_mean/var

def disable_dropout(m):
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        m.p = 0.0
        m.forward = lambda x: x


# 将多通道输出合并为二类（背景 + 前景=肿瘤）
def merge_logits_to_binary(logits: torch.Tensor, bg_channel: int = 0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    p_bg = probs[:, bg_channel:bg_channel+1]
    p_tumor = (probs.sum(dim=1, keepdim=True) - p_bg).clamp(min=0.0, max=1.0)
    bin_probs = torch.cat([p_bg, p_tumor], dim=1)
    return bin_probs  # (B, 2, D, H, W)


def aggregate_mean_std_1d(values: List[float]):
    if len(values) == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def format_delta(after: float, before: float, higher_is_better: bool = True, decimals: int = 4) -> str:
    diff = after - before
    judge = diff if higher_is_better else -diff
    arrow = "↑" if judge > 0 else ("↓" if judge < 0 else "→")
    return f"{diff:+.{decimals}f} {arrow}"


# ======= 「真更新了吗」两项检查所需的辅助函数 =======
def _param_name_map(model: nn.Module) -> Dict[int, str]:
    """将参数内存地址 -> 名称 的映射做成字典，便于打印收集到的参数名。"""
    return {id(p): n for n, p in model.named_parameters()}

def list_adapt_params(model: nn.Module, params_iter):
    """检查一：打印 TENT 收集到的可学习参数数量与名称（前若干个）。"""
    params = list(params_iter)
    name_map = _param_name_map(model)
    names = [name_map.get(id(p), "<unnamed>") for p in params]
    print(f"[update-check] collected params: {len(params)}")
    if len(params) == 0:
        print("⚠️ [update-check] No params collected for adaptation! "
              "Likely your norms have no affine params (e.g., InstanceNorm affine=False).")
    else:
        preview = names[:20]
        for i, nm in enumerate(preview):
            print(f"   - {i:02d}: {nm}")
        if len(names) > len(preview):
            print(f"   ... (+{len(names)-len(preview)} more)")
    return params  # 返回列表供后续使用

def clone_params(params: List[torch.nn.Parameter]):
    """适应前快照：克隆当前参数值。"""
    return [p.detach().clone() for p in params]

def l2_param_delta(params: List[torch.nn.Parameter], snaps: List[torch.Tensor]) -> float:
    """适应后与快照的 L2 改变量。"""
    s = 0.0
    for p, q in zip(params, snaps):
        s += torch.sum((p.detach() - q)**2).item()
    return math.sqrt(s)


# ---------------- 测试 + TTA 前后对比（仅肿瘤） ---------------- #
def test_on_target(args, device,model):
    print(f"\n{'='*40}")
    print(f"🧪 开始在目标数据集上测试 (img: {args.img.upper()}, model: {args.model_type})")
    print(f"{'='*40}\n")

    # 结果保存目录
    result_dir = args.tent_results_dir
    os.makedirs(result_dir, exist_ok=True)
    weights_dir = os.path.join(result_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    try:

        # 解析目录
        image_dir, mask_dir = resolve_dirs(args.target_dir, args.image_dir, args.mask_dir)
        if not image_dir or not mask_dir:
            raise ValueError("无法解析 image_dir / mask_dir，请确认 --target_dir 或显式传入 --image_dir/--mask_dir")
        print(f"目标数据目录: {args.target_dir}")
        print(f"图像目录: {image_dir}")
        print(f"掩码目录: {mask_dir}")

        # DataLoader（只含肿瘤标签）
        target_test_loader, _ = get_ct_test_loader(
            image_dir=image_dir,
            mask_dir=mask_dir,
            target_dir=None,
            dataset_type=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=(args.image_size, args.image_size, args.image_size),
            spacing=args.spacing,
            intensity_range=args.intensity_range,
            positive_labels=args.positive_labels
        )

        # ----- Step 1: TTA 前 baseline（仅肿瘤） -----
        baseline_model = model
        baseline_model.eval()
        tumor_idx = 1  # 合并到二类后，channel 1 对应肿瘤

        before_vals = {
            'dice': [], 'hd95': [], 'iou': [], 'pa': [], 'rve': [], 'sen': [], 'ppv': []
        }

        with torch.no_grad():
            for imgs, labels, *_ in tqdm(target_test_loader, desc='Baseline 推理（肿瘤）'):
                imgs = imgs.to(device)
                labels = labels.to(device)  # (B,1,D,H,W), 0/1

                logits = baseline_model(imgs)
                bin_outputs = merge_logits_to_binary(logits, bg_channel=args.bg_channel)

                dice_values = cal_dice(bin_outputs, labels.squeeze(1))
                hd95_values = cal_hd95(bin_outputs, labels.squeeze(1))
                iou_values  = IoU(bin_outputs, labels.squeeze(1))
                pa_values   = PA(bin_outputs, labels.squeeze(1), 2)
                rve_values  = cal_RVE(bin_outputs, labels.squeeze(1))
                sen_values  = cal_sensitivity(bin_outputs, labels.squeeze(1))
                ppv_values  = cal_ppv(bin_outputs, labels.squeeze(1))

                before_vals['dice'].append(safe_value(dice_values[tumor_idx]))
                before_vals['hd95'].append(safe_value(hd95_values[tumor_idx]))
                before_vals['iou'].append(safe_value(iou_values[tumor_idx]))
                before_vals['pa'].append(safe_value(pa_values[tumor_idx]))
                before_vals['rve'].append(safe_value(rve_values[tumor_idx]))
                before_vals['sen'].append(safe_value(sen_values[tumor_idx]))
                before_vals['ppv'].append(safe_value(ppv_values[tumor_idx]))

        before_mean = {k: aggregate_mean_std_1d(v)[0] for k, v in before_vals.items()}
        before_std  = {k: aggregate_mean_std_1d(v)[1] for k, v in before_vals.items()}

        # # ----- Step 2: TTA 后（TENT，自适应），同样只统计肿瘤 -----
        # model = tent.configure_model(model)  # 冻结除了 BN affine 的其余参数
        # if args.freeze_bn_stats:
        #     model.apply(freeze_bn_running_stats)
        # if args.disable_dropout:
        #     model.apply(disable_dropout)

        # # 确保可学习参数在设备上
        # for _, p in model.named_parameters():
        #     if p.requires_grad and p.device != device:
        #         p.data = p.data.to(device)

        # # === 检查一：收集到的参数？ ===
        # raw_params_iter, _ = tent.collect_params(model)
        # adapt_params = list_adapt_params(model, raw_params_iter)
        # optimizer = optim.Adam(adapt_params, lr=args.lr, weight_decay=0.0)  # 关键：weight_decay=0
        # print(f"[update-check] optimizer lr = {optimizer.param_groups[0]['lr']:.3e}")
        # tented_model = tent.Tent(model, optimizer)

        # src_state = snapshot_state_dict(tented_model.model) if args.episodic else None

        # after_vals = {
        #     'dice': [], 'hd95': [], 'iou': [], 'pa': [], 'rve': [], 'sen': [], 'ppv': []
        # }

        # # tented_model.train()
        # for bidx, (imgs, labels, *_) in enumerate(tqdm(target_test_loader, desc='TENT 推理+适应（肿瘤）')):
        #     imgs = imgs.to(device)
        #     labels = labels.to(device)

        #     # episodic：每个病例适应前恢复源模型
        #     if args.episodic and src_state is not None:
        #         tented_model.model.load_state_dict(src_state, strict=True)
        #         reset_optimizer_state(optimizer)

        #     # 每批进入适应前，做一次参数快照（供检查二使用）
        #     adapt_params_now = optimizer.param_groups[0]['params']
        #     snaps = clone_params(adapt_params_now)

        #     # 多步适应：前 (adapt_steps-1) 次只做适应，不取输出
        #     if args.adapt_steps > 1:
        #         for _ in range(args.adapt_steps - 1):
        #             _ = tented_model(imgs)  # 仅用于更新 BN γ/β

        #     # 最后一步拿 logits 用于评估
        #     logits = tented_model(imgs)

        #     # === 检查二：适应后参数是否真的变化（L2 delta） ===
        #     delta = l2_param_delta(adapt_params_now, snaps)
        #     # 前3个 batch、或每10个 batch、或 delta==0 时打印
        #     if bidx < 3 or (bidx % 10 == 0) or delta == 0.0:
        #         print(f"[update-check] batch {bidx:03d} param L2 delta = {delta:.6e}")

        #     bin_outputs = merge_logits_to_binary(logits, bg_channel=args.bg_channel)

        #     with torch.no_grad():
        #         dice_values = cal_dice(bin_outputs.detach(), labels.squeeze(1))
        #         hd95_values = cal_hd95(bin_outputs.detach(), labels.squeeze(1))
        #         iou_values  = IoU(bin_outputs.detach(), labels.squeeze(1))
        #         pa_values   = PA(bin_outputs.detach(), labels.squeeze(1), 2)
        #         rve_values  = cal_RVE(bin_outputs.detach(), labels.squeeze(1))
        #         sen_values  = cal_sensitivity(bin_outputs.detach(), labels.squeeze(1))
        #         ppv_values  = cal_ppv(bin_outputs.detach(), labels.squeeze(1))

        #         after_vals['dice'].append(safe_value(dice_values[tumor_idx]))
        #         after_vals['hd95'].append(safe_value(hd95_values[tumor_idx]))
        #         after_vals['iou'].append(safe_value(iou_values[tumor_idx]))
        #         after_vals['pa'].append(safe_value(pa_values[tumor_idx]))
        #         after_vals['rve'].append(safe_value(rve_values[tumor_idx]))
        #         after_vals['sen'].append(safe_value(sen_values[tumor_idx]))
        #         after_vals['ppv'].append(safe_value(ppv_values[tumor_idx]))

        # after_mean = {k: aggregate_mean_std_1d(v)[0] for k, v in after_vals.items()}
        # after_std  = {k: aggregate_mean_std_1d(v)[1] for k, v in after_vals.items()}

        # ----- 报告（仅肿瘤） -----
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "nnUNet" if args.model_type == "nnunet" else "UNet3D"

        # adapted_model_path = None
        # if args.tent_save:
        #     adapted_model_path = os.path.join(
        #         weights_dir,
        #         f"{model_name}_{args.img}_tta_adapted_{timestamp}.pth"
        #     )
            # torch.save(tented_model.model.state_dict(), adapted_model_path)
            # print(f"✅ 已保存测试适应后的模型权重: {adapted_model_path}")

        def line_pair(metric_name_cn, key, higher_is_better=True, nd=4, nd_std=4):
            b_mean, b_std = before_mean[key], before_std[key]
            # a_mean, a_std = after_mean[key], after_std[key]
            # delta = format_delta( b_mean, higher_is_better=higher_is_better, decimals=nd)
            return (f"{metric_name_cn:<12} "
                    f"UDA3D: {b_mean:.{nd}f} ± {b_std:.{nd_std}f} | "
                    # f"After: {a_mean:.{nd}f} ± {a_std:.{nd_std}f} | "
                    # f"Δ: {delta}")
            )

        lines = [
            "="*40,
            f"测试时间: {timestamp}",
            "测试配置:",
            f"- 图像模态: {args.img}",
            f"- 模型类型: {model_name}",
            # f"- 模型路径: {best_model_path}",
            f"- 测试数据: {args.target_dir}",
            f"- 图像目录: {image_dir}",
            f"- 掩码目录: {mask_dir}",
            # f"- TENT 学习率: {args.lr}",
            # f"- adapt_steps: {args.adapt_steps}",
            # f"- Episodic 模式: {args.episodic}",
            # f"- 冻结BN统计: {args.freeze_bn_stats}",
            # f"- 关闭Dropout: {args.disable_dropout}",
            # f"- 背景通道(bg_channel): {args.bg_channel}",
            # f"- 正类标签(positive_labels): {args.positive_labels}",
            # f"- 是否保存适应后权重: {args.tent_save}",
            # f"- 适应后权重路径: {adapted_model_path if adapted_model_path else 'N/A'}",
            "",
            "== 指标对比（仅肿瘤）：Before TTA  vs  After TTA  ==",
        ]
        lines.append(line_pair("Dice", 'dice', higher_is_better=True))
        lines.append(line_pair("HD95(mm)", 'hd95', higher_is_better=False, nd=2, nd_std=2))
        lines.append(line_pair("IoU", 'iou', higher_is_better=True))
        lines.append(line_pair("PA", 'pa', higher_is_better=True))
        lines.append(line_pair("RVE", 'rve', higher_is_better=False))
        lines.append(line_pair("Sensitivity", 'sen', higher_is_better=True))
        lines.append(line_pair("PPV", 'ppv', higher_is_better=True))

        lines.append("="*40)
        report = "\n".join(lines)

        result_file = os.path.join(result_dir, f"{model_name}_{args.img}_{timestamp}.txt")
        with open("UDA3D.txt", 'a') as f:
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
        os.makedirs(result_dir, exist_ok=True)
        error_log = os.path.join(result_dir, "test_errors.log")
        with open(error_log, 'a') as f:
            f.write(f"[{error_timestamp}] {error_msg}\n")
        return False


class PixelDiscriminator_(nn.Module):
    def __init__(self, input_nc, ndf=128, num_classes=7):
        super(PixelDiscriminator_, self).__init__()

        self.D = nn.Sequential(
            nn.Conv3d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf, ndf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls1 = nn.Conv3d(ndf // 2, num_classes, kernel_size=1, stride=1)
        self.cls2 = nn.Conv3d(ndf // 2, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.D(x)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        out = torch.cat((src_out, tgt_out), dim=1)
        return out
    

class PosNeg(nn.Module):
    def __init__(self, input_nc, ndf=64, num_classes=7):
        super(PosNeg, self).__init__()
        self.proto_projection = nn.Sequential(
            nn.Conv3d(input_nc, ndf, kernel_size=1),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True))
        self.proto_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten())
        self.proto_D = nn.Sequential(
            nn.Conv3d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls = nn.Conv3d(ndf * 2, num_classes, kernel_size=1, stride=1)

    def forward(self, fea, label):
        # 调试输出
        # print(f"Input fea shape: {fea.shape}")
        # print(f"Input label shape: {label.shape}")
        
        # 调整标签尺寸以匹配特征图
        if label.shape[2:] != fea.shape[2:]:  # 比较空间维度 (D, H, W)
            label = F.interpolate(
                label.float(), 
                size=fea.shape[2:],  # 使用特征图的空间维度
                mode='trilinear',    
                align_corners=False
            )
            # print(f"Resized label shape: {label.shape}")
        
        mask_pos = label.cuda()
        mask_neg = 1. - mask_pos
        
        # 应用掩码前确保特征图和掩码维度匹配
        out_pos = self.cls(self.proto_D(fea.cuda())) * mask_pos
        out_neg = self.cls(self.proto_D(fea.cuda())) * mask_neg
        
        return out_pos, out_neg
    

# 深度信息最大化损失（用于互信息计算）
class DeepInfoMaxLoss(nn.Module):
    def __init__(self, type="fc"):
        super(DeepInfoMaxLoss, self).__init__()
        self.type = type

    def forward(self, x, y, z):
        if self.type == "fc":
            return -torch.mean(torch.log(torch.sigmoid(torch.sum(x * y, dim=1)) + 1e-6)) - \
                   torch.mean(torch.log(1 - torch.sigmoid(torch.sum(x * z, dim=1)) + 1e-6))
        else:  # conv
            return -torch.mean(torch.log(torch.sigmoid(torch.sum(x * y, dim=1)) + 1e-6)) - \
                   torch.mean(torch.log(1 - torch.sigmoid(torch.sum(x * z, dim=1)) + 1e-6))

def to_one(label, num_classes=7):
    # print(label.shape)
    label = rearrange(label, 'b 1 h w d -> b 1 h w d')
    label = torch.where(label != 0, 1, 0)
    return label.float()

# 熵置信度掩码，用于生成伪标签
def entropy_confidence_mask(logits, th=0.1):
    prob = torch.softmax(logits, dim=1)
    entropy = torch.sum(-prob * torch.log(prob + 1e-10), dim=1).detach()
    mask = entropy.ge(th)
    return mask

def train_on_target(args, device, epoch):
    print("\n" + "=" * 40)
    print(f"🧪 开始在目标域上训练数据集: {os.path.basename(args.target_dir)}")
    print("=" * 40 + "\n")

    # ---------------- Paths --------------------------------------------------
    result_dir = os.path.join("UDA_3D", "tta3d_results")
    weights_dir = os.path.join(result_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # ---------------- Model --------------------------------------------------
    if 1:
        # 选择模型
        if args.model_type == 'nnunet':
            model = PlainConvUNet(
                4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3,
                (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None,
                nn.ReLU, deep_supervision=True
            ).to(device)
            print("已选择 nnUNet 模型架构")
            default_model_path = str(DEFAULT_CHECKPOINT_ROOT / "nnunet_best_CT.pth")
        else:
            model = UNet3d().to(device)
            print("已选择 UNet3d 模型架构")
            default_model_path = str(DEFAULT_CHECKPOINT_ROOT / "unet3d_best_CT.pth")

        # 选择权重
        best_model_path = args.checkpoint if args.checkpoint != "default" else default_model_path
        print(f"加载模型权重: {best_model_path}")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"未找到预训练权重: {best_model_path}")

        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=0.000001)
    # segmentation_loss = DiceLoss().to(device)
    # domain_classifier_loss = nn.BCELoss()
    # m = nn.Sigmoid()
    # loss_name, loss_params = TENT_Prostate
    # loss_class = getattr(moment_tta_losses, loss_name)
    # loss_fn = loss_class(**loss_params)
    MI = PosNeg(256).cuda()
    loss_MI = DeepInfoMaxLoss(type="conv")
    # ---------------- Data ---------------------------------------------------
    if 1:
        # 解析目录
        image_dir, mask_dir = resolve_dirs(args.target_dir, args.image_dir, args.mask_dir)
        if not image_dir or not mask_dir:
            raise ValueError("无法解析 image_dir / mask_dir，请确认 --target_dir 或显式传入 --image_dir/--mask_dir")
        print(f"目标数据目录: {args.target_dir}")
        print(f"图像目录: {image_dir}")
        print(f"掩码目录: {mask_dir}")

        # DataLoader（只含肿瘤标签）
        target_test_loader, _ = get_ct_test_loader(
            image_dir=image_dir,
            mask_dir=mask_dir,
            target_dir=None,
            dataset_type=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=(args.image_size, args.image_size, args.image_size),
            spacing=args.spacing,
            intensity_range=args.intensity_range,
            positive_labels=args.positive_labels
        )
        source_image_dir, source_mask_dir = resolve_dirs(args.source_dir, args.image_dir, args.mask_dir)
        source_loader, _ = get_ct_test_loader(
            image_dir=source_image_dir,
            mask_dir=source_mask_dir,
            target_dir=None,
            dataset_type=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=(args.image_size, args.image_size, args.image_size),
            spacing=args.spacing,
            intensity_range=args.intensity_range,
            positive_labels=args.positive_labels
        )

    # ---------------- Eval loop ---------------------------------------------
    for i in range(epoch):
        train_bar = tqdm(target_test_loader, desc="训练进度")
        len_dataloader = len(target_test_loader)
        source_iter = enumerate(source_loader)
        for step, batch in enumerate(train_bar):
            imgs, labels, _ = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            _, inputs = source_iter.__next__()  # inputs 是一个 batch，结构如 (src_imgs, src_labels, ...)
            src_img, src_label, *_ = inputs      # 直接解包 batch，得到张量 src_img、src_label
            src_img = src_img.to(device)         # 将张量移动到设备
            src_label = src_label.to(device)
            
            p = float(step + i * len_dataloader) / epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            pred_seg,tgt_fea = model(x = imgs,feat = True)

            mask = entropy_confidence_mask(pred_seg, 0.1)
            tgt_pseudo_label = pred_seg.max(1).indices
            tgt_pseudo_label[torch.where(mask)] = 255


            tgt_img_aug = FDA_source_to_target(imgs, src_img)
            img_aug_min = reduce(tgt_img_aug, 'b c h w d -> b c 1 1 1', 'min')
            img_aug_max = reduce(tgt_img_aug, 'b c h w d -> b c 1 1 1', 'max')
            tgt_img_aug = (tgt_img_aug - img_aug_min) / (img_aug_max - img_aug_min)

            tgt_logits_aug, tgt_fea_aug = model(x = tgt_img_aug, feat = True)
            tgt_pseudo_aug = tgt_logits_aug.max(1).indices
            tgt_pseudo_aug = tgt_pseudo_aug.unsqueeze(1)
            tgt_pseudo_aug_one = to_one(tgt_pseudo_aug, 2)

            tgt_pos, tgt_neg = MI(tgt_fea_aug, tgt_pseudo_aug_one)
            src_pos, src_neg = MI(tgt_fea, tgt_pseudo_aug_one)
            loss_mu = 0.5 * loss_MI(src_pos, src_neg, tgt_pos) + 0.5 * loss_MI(src_neg, src_pos, tgt_neg)

            total_loss = loss_mu
            # total_loss = loss_fn(bin_outputs)
            total_loss.backward()
            optimizer.step()

            train_bar.set_description(f"Training: Loss: {total_loss.item():.4f}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = "UDA_3D"

        adapted_path = os.path.join(weights_dir, f"{model_tag}_tta3d_{args.dataset}_adapted_{timestamp}.pth")
        torch.save(model.state_dict(), adapted_path)
        print(f"✅ 已保存适应后的模型权重: {adapted_path}")

        test_on_target(args, device, model)

    return True


def main():
    parser = argparse.ArgumentParser(description="Test on target dataset with TENT (tumor-only reporting)")

    # 必选/常用参数
    parser.add_argument('--checkpoint',   type=str,
                        default="default",
                        help='预训练权重路径，或 "default" 使用模型默认路径')
    parser.add_argument('--dataset',   type=str,
                        default="CT",
                        help='Name of dataset')
    parser.add_argument('--target_dir',   type=str,
                        default=str(DEFAULT_DATA_ROOT / "3D-IRCADB"),
                        help='数据集根目录（包含 image/ 与 mask/ 子目录）')
    parser.add_argument('--source_dir',   type=str,
                        default=str(DEFAULT_DATA_ROOT / "LiTS_"),
                        help='数据集根目录（包含 image/ 与 mask/ 子目录）')
    parser.add_argument('--batch_size',   type=int, default=1)  # 按病例适应更稳
    parser.add_argument('--num_workers',  type=int, default=2)
    parser.add_argument('--image_size',   type=int, default=128)
    parser.add_argument('--output_dir',   type=str, default='./target_test_results')  # 兼容旧参数（未使用）

    # 模型 / TENT
    parser.add_argument('--model_type',   type=str, default='unet3d', choices=['unet3d', 'nnunet'])
    parser.add_argument('--lr',           type=float, default=1e-4, help='TENT 学习率（稳健推荐：1e-5）')
    parser.add_argument('--gpu',          type=int, default=0, help='GPU 编号')
    parser.add_argument('--tent_save',    action='store_true', help='保存适应后的模型权重')
    parser.add_argument('--tent_results_dir', type=str,
                        default=str(DEFAULT_RESULTS_ROOT),
                        help='TENT 结果与适应后权重保存目录')
    parser.add_argument('--adapt_steps',  type=int, default=2,
                        help='每个batch的TENT适应步数（建议2-4之间）')

    # [EPISODIC]（默认开启）
    parser.add_argument('--episodic',     default=False,
                        help='开启 episodic：每个batch/病例前重置至源模型并清空优化器状态')

    # 稳定性选项（默认开启）
    parser.add_argument('--freeze_bn_stats', action='store_true',
                        help='冻结 BN running stats，仅更新 affine（γ/β）')
    parser.add_argument('--disable_dropout', action='store_true',
                        help='关闭 Dropout')

    # 任务/评估相关
    parser.add_argument('--img',          type=str, default='ct', help='数据模态标识，仅用于打印')
    parser.add_argument('--spacing',      type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument('--intensity_range', type=float, nargs=2, default=(-200, 400))
    parser.add_argument('--bg_channel',   type=int, default=0,
                        help='模型输出中背景通道索引（用于将输出合并为二类）')
    parser.add_argument('--positive_labels', type=str, default='1',
                        help="把哪些标签 id 视为肿瘤，逗号分隔。如 '1' 或 '2' 或 '1,2'。若原始标签即 0/1，保持默认即可。")

    # 显式传入 image_dir/mask_dir（可选）
    parser.add_argument('--image_dir',    type=str, default=None)
    parser.add_argument('--mask_dir',     type=str, default=None)

    # dataset_type 仅在完全未提供 target_dir 和 image_dir/mask_dir 时才生效
    parser.add_argument('--dataset_type', type=str, default=None, help='数据集类型（例如 CT）')

    args = parser.parse_args()

    # 默认把稳健开关打开（除非用户手动关）
    if not args.episodic:
        args.episodic = True
    if not args.freeze_bn_stats:
        args.freeze_bn_stats = True
    if not args.disable_dropout:
        args.disable_dropout = True

    # 解析 positive_labels
    try:
        args.positive_labels = [int(x) for x in args.positive_labels.split(',') if x.strip() != '']
    except Exception:
        raise ValueError("positive_labels 参数格式错误，应为逗号分隔的整数，如 '1' 或 '2' 或 '1,2'")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    print(f"📌 正类标签(positive_labels): {args.positive_labels} | 背景通道(bg_channel): {args.bg_channel}")
    print(f"🔧 默认设置：episodic={args.episodic}, freeze_bn_stats={args.freeze_bn_stats}, "
          f"disable_dropout={args.disable_dropout}, lr={args.lr}, adapt_steps={args.adapt_steps}, "
          f"batch_size={args.batch_size}")
    for epoch in range(10):
        train_on_target(args, device, epoch)
    # test_on_target(args, device)


if __name__ == "__main__":
    main()
