import os
import glob
import argparse
import datetime
import traceback
from typing import Tuple, Optional, Dict, List
import copy
import math
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, Resized, NormalizeIntensityd, ToTensord
from nnunet import PlainConvUNet
from unet3d import UNet3d
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
import tent
import torch.optim as optim

def get_dataset_type_from_path(data_path: str) -> str:
    data_path = data_path.replace('\\', '/').lower()
    if 'tta-3dct' in data_path or 'tta-ct' in data_path or 'ct' in data_path:
        return 'CT'
    return 'CT'

def get_dataset_paths(dataset_type: str, base_dir: str='/home/yuwenjing/data/tta_dataset', subfolder: str=None) -> Tuple[str, str]:
    dataset_mapping = {'CT': 'TTA-3DCT'}
    if dataset_type not in dataset_mapping:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')
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
    return (image_dir, mask_dir)

def _auto_select_subfolder(dataset_path: str, dataset_type: str) -> str:
    if not os.path.exists(dataset_path):
        return 'CT_' if dataset_type == 'CT' else ''
    try:
        subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    except PermissionError:
        return ''
    if not subfolders:
        return ''
    underscore_folders = sorted([f for f in subfolders if f.endswith('_')])
    return underscore_folders[0] if underscore_folders else sorted(subfolders)[0]

def resolve_dirs(target_dir: Optional[str], image_dir: Optional[str], mask_dir: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if image_dir and mask_dir:
        return (image_dir, mask_dir)
    if target_dir:
        return (image_dir or os.path.join(target_dir, 'image'), mask_dir or os.path.join(target_dir, 'mask'))
    return (None, None)

def binarize_label_tensor(label_tensor: torch.Tensor, positive_ids: List[int]) -> torch.Tensor:
    if label_tensor.ndim == 4 and label_tensor.shape[0] == 1:
        lt = label_tensor.squeeze(0)
    else:
        lt = label_tensor
    mask = torch.zeros_like(lt, dtype=torch.bool)
    for pid in positive_ids:
        mask |= lt == pid
    bin_label = mask.long()
    return bin_label.unsqueeze(0)

class CTDataset3D(Dataset):

    def __init__(self, image_dir: str, mask_dir: str, phase: str='test', image_size: Tuple[int, int, int]=(128, 128, 128), spacing: Tuple[float, float, float]=(1.0, 1.0, 1.0), intensity_range: Tuple[float, float]=(-200, 400), normalize: bool=True, positive_labels: Optional[List[int]]=None):
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
            raise ValueError(f'Image directory does not exist: {image_dir}')
        if not os.path.exists(mask_dir):
            raise ValueError(f'Mask directory does not exist: {mask_dir}')
        self.data_dicts = self._collect_data_pairs()
        if len(self.data_dicts) == 0:
            raise ValueError(f'No valid image-mask pairs found under:\n  {image_dir}\n  {mask_dir}')
        print(f'Found {len(self.data_dicts)} valid CT image-mask pairs for {self.phase} phase')
        self.transforms = self._get_test_transforms()

    def _collect_data_pairs(self) -> List[Dict[str, str]]:
        data_dicts = []
        image_files = []
        for ext in self.supported_extensions:
            image_files.extend(glob.glob(os.path.join(self.image_dir, f'*{ext}')))
        image_files.sort()
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            base_name = self._get_base_name(img_name)
            mask_path = self._find_mask_path(base_name)
            if mask_path and self._is_valid_file(img_path) and self._is_valid_file(mask_path):
                data_dicts.append({'image': img_path, 'label': mask_path, 'image_name': img_name})
            else:
                print(f'[Warning] Skip invalid pair: {img_name}')
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
            patterns = [base_name, f'{base_name}_seg', f'{base_name}_segmentation', f'{base_name}_mask', f'{base_name}_label', f'{base_name}_gt', f'{base_name}-liver_mask', f'{base_name}-mask']
        for pattern in patterns:
            for ext in self.supported_extensions:
                mask_path = os.path.join(self.mask_dir, f'{pattern}{ext}')
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
        transforms_list = [LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']), Orientationd(keys=['image', 'label'], axcodes='RAS'), Spacingd(keys=['image', 'label'], pixdim=self.spacing, mode=('bilinear', 'nearest')), ScaleIntensityRanged(keys=['image'], a_min=self.intensity_range[0], a_max=self.intensity_range[1], b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=['image', 'label'], source_key='image'), Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=('trilinear', 'nearest'))]
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
            print(f'Error applying transforms to {data_dict.get('image_name', '<unknown>')}: {e}')
            raise e
        image = data_dict['image']
        label = data_dict['label']
        filename = data_dict['image_name']
        image = image.float() if isinstance(image, torch.Tensor) else torch.tensor(image, dtype=torch.float32)
        label = label.long() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)
        label = binarize_label_tensor(label, self.positive_labels)
        return (image, label, filename)

def get_ct_test_loader(image_dir: str=None, mask_dir: str=None, dataset_type: str=None, subfolder: str=None, base_dir: str='/home/yuwenjing/data/tta_dataset', target_dir: str=None, batch_size: int=2, num_workers: int=4, image_size: Tuple[int, int, int]=(128, 128, 128), spacing: Tuple[float, float, float]=(1.0, 1.0, 1.0), intensity_range: Tuple[float, float]=(-200, 400), positive_labels: Optional[List[int]]=None) -> Tuple[DataLoader, str]:
    image_dir, mask_dir = resolve_dirs(target_dir, image_dir, mask_dir)
    if image_dir is None or mask_dir is None:
        if dataset_type is not None:
            image_dir, mask_dir = get_dataset_paths(dataset_type, base_dir, subfolder)
        else:
            raise ValueError('Please provide either (image_dir & mask_dir) or target_dir, or set dataset_type to use base_dir mapping.')
    test_dataset = CTDataset3D(image_dir=image_dir, mask_dir=mask_dir, phase='test', image_size=image_size, spacing=spacing, intensity_range=intensity_range, normalize=True, positive_labels=positive_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return (test_loader, get_dataset_type_from_path(image_dir))

def safe_value(val):
    return val.item() if isinstance(val, torch.Tensor) else val

def snapshot_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

def reset_optimizer_state(optimizer: optim.Optimizer):
    optimizer.state.clear()

def freeze_bn_running_stats(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.track_running_stats = False

def disable_dropout(m):
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        m.p = 0.0
        m.forward = lambda x: x

def merge_logits_to_binary(logits: torch.Tensor, bg_channel: int=0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    p_bg = probs[:, bg_channel:bg_channel + 1]
    p_tumor = (probs.sum(dim=1, keepdim=True) - p_bg).clamp(min=0.0, max=1.0)
    bin_probs = torch.cat([p_bg, p_tumor], dim=1)
    return bin_probs

def aggregate_mean_std_1d(values: List[float]):
    if len(values) == 0:
        return (0.0, 0.0)
    return (float(np.mean(values)), float(np.std(values)))

def format_delta(after: float, before: float, higher_is_better: bool=True, decimals: int=4) -> str:
    diff = after - before
    judge = diff if higher_is_better else -diff
    arrow = '↑' if judge > 0 else '↓' if judge < 0 else '→'
    return f'{diff:+.{decimals}f} {arrow}'

def _param_name_map(model: nn.Module) -> Dict[int, str]:
    return {id(p): n for n, p in model.named_parameters()}

def list_adapt_params(model: nn.Module, params_iter):
    params = list(params_iter)
    name_map = _param_name_map(model)
    names = [name_map.get(id(p), '<unnamed>') for p in params]
    print(f'[update-check] collected params: {len(params)}')
    if len(params) == 0:
        print('⚠️ [update-check] No params collected for adaptation! Likely your norms have no affine params (e.g., InstanceNorm affine=False).')
    else:
        preview = names[:20]
        for i, nm in enumerate(preview):
            print(f'   - {i:02d}: {nm}')
        if len(names) > len(preview):
            print(f'   ... (+{len(names) - len(preview)} more)')
    return params

def clone_params(params: List[torch.nn.Parameter]):
    return [p.detach().clone() for p in params]

def l2_param_delta(params: List[torch.nn.Parameter], snaps: List[torch.Tensor]) -> float:
    s = 0.0
    for p, q in zip(params, snaps):
        s += torch.sum((p.detach() - q) ** 2).item()
    return math.sqrt(s)

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 开始在目标数据集上测试 (img: {args.img.upper()}, model: {args.model_type})')
    print(f'{'=' * 40}\n')
    result_dir = args.tent_results_dir
    os.makedirs(result_dir, exist_ok=True)
    weights_dir = os.path.join(result_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    try:
        if args.model_type == 'nnunet':
            model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
            print('已选择 nnUNet 模型架构')
            default_model_path = '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth'
        else:
            model = UNet3d(in_chns=1, n_classes=2).to(device)
            print('已选择 UNet3d 模型架构')
            default_model_path = '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth'
        best_model_path = args.checkpoint if args.checkpoint != 'default' else default_model_path
        print(f'加载模型权重: {best_model_path}')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f'未找到预训练权重: {best_model_path}')
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        image_dir, mask_dir = resolve_dirs(args.target_dir, args.image_dir, args.mask_dir)
        if not image_dir or not mask_dir:
            raise ValueError('无法解析 image_dir / mask_dir，请确认 --target_dir 或显式传入 --image_dir/--mask_dir')
        print(f'目标数据目录: {args.target_dir}')
        print(f'图像目录: {image_dir}')
        print(f'掩码目录: {mask_dir}')
        target_test_loader, _ = get_ct_test_loader(image_dir=image_dir, mask_dir=mask_dir, target_dir=None, dataset_type=None, batch_size=args.batch_size, num_workers=args.num_workers, image_size=(args.image_size, args.image_size, args.image_size), spacing=args.spacing, intensity_range=args.intensity_range, positive_labels=args.positive_labels)
        baseline_model = copy.deepcopy(model).to(device).eval()
        tumor_idx = 1
        before_vals = {'dice': [], 'hd95': [], 'iou': [], 'pa': [], 'rve': [], 'sen': [], 'ppv': []}
        with torch.no_grad():
            for imgs, labels, *_ in tqdm(target_test_loader, desc='Baseline 推理（肿瘤）'):
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = baseline_model(imgs)
                bin_outputs = merge_logits_to_binary(logits, bg_channel=args.bg_channel)
                dice_values = cal_dice(bin_outputs, labels.squeeze(1))
                hd95_values = cal_hd95(bin_outputs, labels.squeeze(1))
                iou_values = IoU(bin_outputs, labels.squeeze(1))
                pa_values = PA(bin_outputs, labels.squeeze(1), 2)
                rve_values = cal_RVE(bin_outputs, labels.squeeze(1))
                sen_values = cal_sensitivity(bin_outputs, labels.squeeze(1))
                ppv_values = cal_ppv(bin_outputs, labels.squeeze(1))
                before_vals['dice'].append(safe_value(dice_values[tumor_idx]))
                before_vals['hd95'].append(safe_value(hd95_values[tumor_idx]))
                before_vals['iou'].append(safe_value(iou_values[tumor_idx]))
                before_vals['pa'].append(safe_value(pa_values[tumor_idx]))
                before_vals['rve'].append(safe_value(rve_values[tumor_idx]))
                before_vals['sen'].append(safe_value(sen_values[tumor_idx]))
                before_vals['ppv'].append(safe_value(ppv_values[tumor_idx]))
        before_mean = {k: aggregate_mean_std_1d(v)[0] for k, v in before_vals.items()}
        before_std = {k: aggregate_mean_std_1d(v)[1] for k, v in before_vals.items()}
        model = tent.configure_model(model)
        if args.freeze_bn_stats:
            model.apply(freeze_bn_running_stats)
        if args.disable_dropout:
            model.apply(disable_dropout)
        for _, p in model.named_parameters():
            if p.requires_grad and p.device != device:
                p.data = p.data.to(device)
        raw_params_iter, _ = tent.collect_params(model)
        adapt_params = list_adapt_params(model, raw_params_iter)
        optimizer = optim.Adam(adapt_params, lr=args.lr, weight_decay=0.0)
        print(f'[update-check] optimizer lr = {optimizer.param_groups[0]['lr']:.3e}')
        tented_model = tent.Tent(model, optimizer)
        src_state = snapshot_state_dict(tented_model.model) if args.episodic else None
        after_vals = {'dice': [], 'hd95': [], 'iou': [], 'pa': [], 'rve': [], 'sen': [], 'ppv': []}
        tented_model.train()
        for bidx, (imgs, labels, *_) in enumerate(tqdm(target_test_loader, desc='TENT 推理+适应（肿瘤）')):
            imgs = imgs.to(device)
            labels = labels.to(device)
            if args.episodic and src_state is not None:
                tented_model.model.load_state_dict(src_state, strict=True)
                reset_optimizer_state(optimizer)
            adapt_params_now = optimizer.param_groups[0]['params']
            snaps = clone_params(adapt_params_now)
            if args.adapt_steps > 1:
                for _ in range(args.adapt_steps - 1):
                    _ = tented_model(imgs)
            logits = tented_model(imgs)
            delta = l2_param_delta(adapt_params_now, snaps)
            if bidx < 3 or bidx % 10 == 0 or delta == 0.0:
                print(f'[update-check] batch {bidx:03d} param L2 delta = {delta:.6e}')
            bin_outputs = merge_logits_to_binary(logits, bg_channel=args.bg_channel)
            with torch.no_grad():
                dice_values = cal_dice(bin_outputs.detach(), labels.squeeze(1))
                hd95_values = cal_hd95(bin_outputs.detach(), labels.squeeze(1))
                iou_values = IoU(bin_outputs.detach(), labels.squeeze(1))
                pa_values = PA(bin_outputs.detach(), labels.squeeze(1), 2)
                rve_values = cal_RVE(bin_outputs.detach(), labels.squeeze(1))
                sen_values = cal_sensitivity(bin_outputs.detach(), labels.squeeze(1))
                ppv_values = cal_ppv(bin_outputs.detach(), labels.squeeze(1))
                after_vals['dice'].append(safe_value(dice_values[tumor_idx]))
                after_vals['hd95'].append(safe_value(hd95_values[tumor_idx]))
                after_vals['iou'].append(safe_value(iou_values[tumor_idx]))
                after_vals['pa'].append(safe_value(pa_values[tumor_idx]))
                after_vals['rve'].append(safe_value(rve_values[tumor_idx]))
                after_vals['sen'].append(safe_value(sen_values[tumor_idx]))
                after_vals['ppv'].append(safe_value(ppv_values[tumor_idx]))
        after_mean = {k: aggregate_mean_std_1d(v)[0] for k, v in after_vals.items()}
        after_std = {k: aggregate_mean_std_1d(v)[1] for k, v in after_vals.items()}
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = 'nnUNet' if args.model_type == 'nnunet' else 'UNet3D'
        adapted_model_path = None
        if args.tent_save:
            adapted_model_path = os.path.join(weights_dir, f'UNet3d_tent_CT.pth')
            torch.save(tented_model.model.state_dict(), adapted_model_path)
            print(f'✅ 已保存测试适应后的模型权重: {adapted_model_path}')

        def line_pair(metric_name_cn, key, higher_is_better=True, nd=4, nd_std=4):
            b_mean, b_std = (before_mean[key], before_std[key])
            a_mean, a_std = (after_mean[key], after_std[key])
            delta = format_delta(a_mean, b_mean, higher_is_better=higher_is_better, decimals=nd)
            return f'{metric_name_cn:<12} Before: {b_mean:.{nd}f} ± {b_std:.{nd_std}f} | After: {a_mean:.{nd}f} ± {a_std:.{nd_std}f} | Δ: {delta}'
        lines = ['=' * 40, f'测试时间: {timestamp}', '测试配置:', f'- 图像模态: {args.img}', f'- 模型类型: {model_name}', f'- 模型路径: {best_model_path}', f'- 测试数据: {args.target_dir}', f'- 图像目录: {image_dir}', f'- 掩码目录: {mask_dir}', f'- TENT 学习率: {args.lr}', f'- adapt_steps: {args.adapt_steps}', f'- Episodic 模式: {args.episodic}', f'- 冻结BN统计: {args.freeze_bn_stats}', f'- 关闭Dropout: {args.disable_dropout}', f'- 背景通道(bg_channel): {args.bg_channel}', f'- 正类标签(positive_labels): {args.positive_labels}', f'- 是否保存适应后权重: {args.tent_save}', f'- 适应后权重路径: {(adapted_model_path if adapted_model_path else 'N/A')}', '', '== 指标对比（仅肿瘤）：Before TTA  vs  After TTA  ==']
        lines.append(line_pair('Dice', 'dice', higher_is_better=True))
        lines.append(line_pair('HD95(mm)', 'hd95', higher_is_better=False, nd=2, nd_std=2))
        lines.append(line_pair('IoU', 'iou', higher_is_better=True))
        lines.append(line_pair('PA', 'pa', higher_is_better=True))
        lines.append(line_pair('RVE', 'rve', higher_is_better=False))
        lines.append(line_pair('Sensitivity', 'sen', higher_is_better=True))
        lines.append(line_pair('PPV', 'ppv', higher_is_better=True))
        lines.append('=' * 40)
        report = '\n'.join(lines)
        result_file = os.path.join(result_dir, f'{model_name}_{args.img}_{timestamp}.txt')
        with open(result_file, 'w') as f:
            f.write(report)
        print(report)
        print(f'✅ 结果已保存到: {result_file}')
        return True
    except Exception as e:
        error_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        error_msg = f'\n🔥 测试失败\n图像模态: {args.img}\n模型类型: {args.model_type}\n错误信息: {str(e)}\n追踪信息:\n{traceback.format_exc()}'
        print(error_msg)
        os.makedirs(result_dir, exist_ok=True)
        error_log = os.path.join(result_dir, 'test_errors.log')
        with open(error_log, 'a') as f:
            f.write(f'[{error_timestamp}] {error_msg}\n')
        return False

def main():
    parser = argparse.ArgumentParser(description='Test on target dataset with TENT (tumor-only reporting)')
    parser.add_argument('--checkpoint', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth', help='预训练权重路径，或 "default" 使用模型默认路径')
    parser.add_argument('--target_dir', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB', help='数据集根目录（包含 image/ 与 mask/ 子目录）')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='./target_test_results')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['unet3d', 'nnunet'])
    parser.add_argument('--lr', type=float, default=1e-05, help='TENT 学习率（稳健推荐：1e-5）')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号')
    parser.add_argument('--tent_save', action='store_true', default=True, help='保存适应后的模型权重')
    parser.add_argument('--tent_results_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints/tta_results', help='TENT 结果与适应后权重保存目录')
    parser.add_argument('--adapt_steps', type=int, default=2, help='每个batch的TENT适应步数（建议2-4之间）')
    parser.add_argument('--episodic', default=False, help='开启 episodic：每个batch/病例前重置至源模型并清空优化器状态')
    parser.add_argument('--freeze_bn_stats', action='store_true', help='冻结 BN running stats，仅更新 affine（γ/β）')
    parser.add_argument('--disable_dropout', action='store_true', help='关闭 Dropout')
    parser.add_argument('--img', type=str, default='ct', help='数据模态标识，仅用于打印')
    parser.add_argument('--spacing', type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument('--intensity_range', type=float, nargs=2, default=(-200, 400))
    parser.add_argument('--bg_channel', type=int, default=0, help='模型输出中背景通道索引（用于将输出合并为二类）')
    parser.add_argument('--positive_labels', type=str, default='1', help="把哪些标签 id 视为肿瘤，逗号分隔。如 '1' 或 '2' 或 '1,2'。若原始标签即 0/1，保持默认即可。")
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--mask_dir', type=str, default=None)
    parser.add_argument('--dataset_type', type=str, default=None, help='数据集类型（例如 CT）')
    args = parser.parse_args()
    if not args.episodic:
        args.episodic = True
    if not args.freeze_bn_stats:
        args.freeze_bn_stats = True
    if not args.disable_dropout:
        args.disable_dropout = True
    try:
        args.positive_labels = [int(x) for x in args.positive_labels.split(',') if x.strip() != '']
    except Exception:
        raise ValueError("positive_labels 参数格式错误，应为逗号分隔的整数，如 '1' 或 '2' 或 '1,2'")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  使用设备: {device}')
    print(f'📌 正类标签(positive_labels): {args.positive_labels} | 背景通道(bg_channel): {args.bg_channel}')
    print(f'🔧 默认设置：episodic={args.episodic}, freeze_bn_stats={args.freeze_bn_stats}, disable_dropout={args.disable_dropout}, lr={args.lr}, adapt_steps={args.adapt_steps}, batch_size={args.batch_size}')
    test_on_target(args, device)
if __name__ == '__main__':
    main()
