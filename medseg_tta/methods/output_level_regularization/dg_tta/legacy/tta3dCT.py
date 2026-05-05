import argparse
import os
import glob
import datetime
import traceback
from typing import Tuple, Optional, Dict, List
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, Resized, NormalizeIntensityd, ToTensord
from nnunet import PlainConvUNet
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
from augmentation_utils import get_disp_field, get_rand_affine

def _auto_select_subfolder(dataset_path: str, dataset_type: str) -> str:
    if not os.path.exists(dataset_path):
        return 'CT_' if dataset_type == 'CT' else ''
    try:
        subs = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    except PermissionError:
        return ''
    if not subs:
        return ''
    underscore_folders = sorted([f for f in subs if f.endswith('_')])
    return underscore_folders[0] if underscore_folders else sorted(subs)[0]

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

def get_ct_test_loader(image_dir: str=None, mask_dir: str=None, dataset_type: str=None, subfolder: str=None, base_dir: str='/home/yuwenjing/data/tta_dataset', target_dir: str=None, batch_size: int=1, num_workers: int=2, image_size: Tuple[int, int, int]=(128, 128, 128), spacing: Tuple[float, float, float]=(1.0, 1.0, 1.0), intensity_range: Tuple[float, float]=(-200, 400), positive_labels: Optional[List[int]]=None) -> Tuple[DataLoader, str]:
    image_dir, mask_dir = resolve_dirs(target_dir, image_dir, mask_dir)
    if image_dir is None or mask_dir is None:
        if dataset_type is not None:
            image_dir, mask_dir = get_dataset_paths(dataset_type, base_dir, subfolder)
        else:
            raise ValueError('Please provide either (image_dir & mask_dir) or target_dir, or set dataset_type to use base_dir mapping.')
    ds = CTDataset3D(image_dir=image_dir, mask_dir=mask_dir, phase='test', image_size=image_size, spacing=spacing, intensity_range=intensity_range, normalize=True, positive_labels=positive_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return (loader, get_dataset_type_from_path(image_dir))

def safe_value(val):
    return val.item() if isinstance(val, torch.Tensor) else val

def soft_dice_loss(smp_a, smp_b):
    B, _, D, H, W = smp_a.shape
    d = 2
    nominator = (2.0 * smp_a * smp_b).reshape(B, -1, D * H * W).mean(2)
    denominator = 1 / d * ((smp_a + smp_b) ** d).reshape(B, -1, D * H * W).mean(2)
    if denominator.sum() == 0.0:
        dice = nominator * 0.0 + 1.0
    else:
        dice = nominator / denominator
    return dice

def freeze_bn_statistics(model):
    print('📊 冻结所有 BatchNorm 统计量（仅用当前 batch 统计）')
    frozen = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm3d,)):
            m.track_running_stats = False
            if hasattr(m, 'running_mean') and m.running_mean is not None:
                m.running_mean = m.running_mean.detach()
            if hasattr(m, 'running_var') and m.running_var is not None:
                m.running_var = m.running_var.detach()
            frozen += 1
    print(f'✅ 已冻结 {frozen} 个 BN3d 层')
    return model

def freeze_non_bn_parameters(model):
    print('❄️ 冻结所有非 BN 参数，仅更新 BN 仿射（γ/β）')
    frozen_params = 0
    total_params = 0
    bn_param_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm3d):
            bn_param_names.add(f'{name}.weight')
            bn_param_names.add(f'{name}.bias')
    for name, param in model.named_parameters():
        total_params += 1
        if name in bn_param_names:
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen_params += 1
    print(f'✅ 已冻结 {frozen_params}/{total_params} 个参数')
    return model

def compute_consistency_loss(model, imgs, device):
    batch_size, _, *patch_size = imgs.size()
    identity_grid = F.affine_grid(torch.eye(4, device=device).repeat(batch_size, 1, 1)[:, :3], [batch_size, 1] + patch_size, align_corners=False)
    R_a, R_a_inv = get_rand_affine(batch_size, flip=False, device=device)
    affine_grid_a = F.affine_grid(R_a, [batch_size, 1] + patch_size, align_corners=False)
    grid_deform_a, grid_deform_a_inv = get_disp_field(batch_size, patch_size, device=device)
    composite_grid_a = identity_grid + (affine_grid_a - identity_grid) + grid_deform_a
    x_a = F.grid_sample(imgs, composite_grid_a, padding_mode='border', align_corners=False)
    R_b, R_b_inv = get_rand_affine(batch_size, flip=False, device=device)
    affine_grid_b = F.affine_grid(R_b, [batch_size, 1] + patch_size, align_corners=False)
    grid_deform_b, grid_deform_b_inv = get_disp_field(batch_size, patch_size, device=device)
    composite_grid_b = identity_grid + (affine_grid_b - identity_grid) + grid_deform_b
    x_b = F.grid_sample(imgs, composite_grid_b, padding_mode='border', align_corners=False)

    def _forward(x):
        out = model(x)
        return out[0] if isinstance(out, tuple) else out
    pred_a = _forward(x_a)
    pred_b = _forward(x_b)

    def _inverse_transform(pred, R_inv, deform_inv):
        affine_inv = F.affine_grid(R_inv, [batch_size, 1] + patch_size, align_corners=False)
        grid_inv = identity_grid + (affine_inv - identity_grid) + deform_inv
        return F.grid_sample(pred, grid_inv, align_corners=False)
    aligned_a = _inverse_transform(pred_a, R_a_inv, grid_deform_a_inv)
    aligned_b = _inverse_transform(pred_b, R_b_inv, grid_deform_b_inv)
    sm_a = aligned_a.softmax(1)
    sm_b = aligned_b.softmax(1)
    loss = 1 - soft_dice_loss(sm_a, sm_b)[:, 1:].mean()
    return loss

def get_model(model_type: str, device: torch.device):
    if model_type.lower() == 'nnunet':
        print('📋 加载 nnUNet 模型架构')
        model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
        return model
    elif model_type.lower() == 'unet3d':
        print('📋 加载 UNet3D 模型架构')
        from unet3d import UNet3dCT
        model = UNet3dCT().to(device)
        return model
    else:
        raise ValueError(f"不支持的模型类型: {model_type}（可选 'nnunet' 或 'unet3d'）")

def merge_logits_to_binary(logits: torch.Tensor, bg_channel: int=0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    p_bg = probs[:, bg_channel:bg_channel + 1]
    p_tumor = (probs.sum(dim=1, keepdim=True) - p_bg).clamp(min=0.0, max=1.0)
    return torch.cat([p_bg, p_tumor], dim=1)

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 CT 模态 TTA 测试 | 模型: {args.model_type}')
    print(f'{'=' * 40}\n')
    try:
        model = get_model(args.model_type, device)
        model_path = args.model_path
        if model_path == 'default':
            model_path = '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth' if args.model_type.lower() == 'nnunet' else '/home/yuwenjing/DeepLearning_ywj/tta/unet3d_best.pth'
        print(f'📦 加载模型权重: {model_path}')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'未找到模型权重文件: {model_path}')
        state = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        freeze_bn_statistics(model)
        if args.freeze_other:
            freeze_non_bn_parameters(model)
        if args.eval_mode:
            model.eval()
        else:
            model.train()
        print(f'🔧 模式: {('eval' if args.eval_mode else 'train')}')
        optim_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = optim.Adam(optim_params, lr=args.lr)
        test_loader, _ = get_ct_test_loader(image_dir=args.image_dir, mask_dir=args.mask_dir, target_dir=args.target_root, dataset_type='CT' if args.target_root is None else None, batch_size=args.batch_test, num_workers=args.num_workers, image_size=(args.image_size, args.image_size, args.image_size), spacing=tuple(args.spacing), intensity_range=tuple(args.intensity_range), positive_labels=args.positive_labels)
        K = 2
        all_dice = [[] for _ in range(K)]
        all_hd95 = [[] for _ in range(K)]
        all_IoU = [[] for _ in range(K)]
        all_pa = [[] for _ in range(K)]
        all_RVE = [[] for _ in range(K)]
        all_sensitivity = [[] for _ in range(K)]
        all_ppv = [[] for _ in range(K)]
        tumor_idx = 1
        for batch in tqdm(test_loader, desc='TTA 适应'):
            if len(batch) == 3:
                imgs, labels, _ = batch
            else:
                imgs, labels = batch[:2]
            imgs, labels = (imgs.to(device), labels.to(device))
            for _ in range(args.adapt_steps):
                loss = compute_consistency_loss(model, imgs, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                logits = model(imgs)
                logits = logits[0] if isinstance(logits, tuple) else logits
                bin_probs = merge_logits_to_binary(logits, bg_channel=args.bg_channel)
                dice_vals = cal_dice(bin_probs, labels.squeeze(1))
                hd95_vals = cal_hd95(bin_probs, labels.squeeze(1))
                iou_vals = IoU(bin_probs, labels.squeeze(1))
                pa_vals = PA(bin_probs, labels.squeeze(1), K)
                rve_vals = cal_RVE(bin_probs, labels.squeeze(1))
                sen_vals = cal_sensitivity(bin_probs, labels.squeeze(1))
                ppv_vals = cal_ppv(bin_probs, labels.squeeze(1))
                for i in range(K):
                    all_dice[i].append(safe_value(dice_vals[i]))
                    all_hd95[i].append(safe_value(hd95_vals[i]))
                    all_IoU[i].append(safe_value(iou_vals[i]))
                    all_pa[i].append(safe_value(pa_vals[i]))
                    all_RVE[i].append(safe_value(rve_vals[i]))
                    all_sensitivity[i].append(safe_value(sen_vals[i]))
                    all_ppv[i].append(safe_value(ppv_vals[i]))

        def ms(x):
            return (float(np.mean(x)) if len(x) else 0.0, float(np.std(x)) if len(x) else 0.0)
        mean_dice = [ms(all_dice[i])[0] for i in range(K)]
        std_dice = [ms(all_dice[i])[1] for i in range(K)]
        mean_hd95 = [ms(all_hd95[i])[0] for i in range(K)]
        std_hd95 = [ms(all_hd95[i])[1] for i in range(K)]
        mean_IoU = [ms(all_IoU[i])[0] for i in range(K)]
        std_IoU = [ms(all_IoU[i])[1] for i in range(K)]
        mean_pa = [ms(all_pa[i])[0] for i in range(K)]
        std_pa = [ms(all_pa[i])[1] for i in range(K)]
        mean_RVE = [ms(all_RVE[i])[0] for i in range(K)]
        std_RVE = [ms(all_RVE[i])[1] for i in range(K)]
        mean_sen = [ms(all_sensitivity[i])[0] for i in range(K)]
        std_sen = [ms(all_sensitivity[i])[1] for i in range(K)]
        mean_ppv = [ms(all_ppv[i])[0] for i in range(K)]
        std_ppv = [ms(all_ppv[i])[1] for i in range(K)]
        t = tumor_idx
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result = f'测试配置:\n  模型类型: {args.model_type}\n  模型路径: {model_path}\n  适应步数: {args.adapt_steps}\n  学习率:   {args.lr}\n  图像尺寸: {args.image_size}^3, spacing={tuple(args.spacing)}\n  CT 强度范围: {tuple(args.intensity_range)}\n\n[Tumor] 指标均值 ± 标准差：\n  Dice        : {mean_dice[t]:.4f} ± {std_dice[t]:.4f}\n  HD95 (mm)   : {mean_hd95[t]:.2f} ± {std_hd95[t]:.2f}\n  IoU         : {mean_IoU[t]:.4f} ± {std_IoU[t]:.4f}\n  PA          : {mean_pa[t]:.4f} ± {std_pa[t]:.4f}\n  RVE         : {mean_RVE[t]:.4f} ± {std_RVE[t]:.4f}\n  Sensitivity : {mean_sen[t]:.4f} ± {std_sen[t]:.4f}\n  PPV         : {mean_ppv[t]:.4f} ± {std_ppv[t]:.4f}\n'
        print('\n' + result)
        result_dir = os.path.join(args.checkpoint_dir, f'{args.model_type}_CT_results')
        os.makedirs(result_dir, exist_ok=True)
        out_path = os.path.join(result_dir, f'tta_results_{ts}.txt')
        with open(out_path, 'w') as f:
            f.write(result)
        print(f'✅ 结果已保存: {out_path}')
        return True
    except Exception as e:
        print(f'❌ 测试失败: {str(e)}')
        traceback.print_exc()
        return False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CT 模态的一致性 TTA（保持算法主题逻辑不变）')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB', help='目标数据集根目录（包含 image/ 与 mask/ 子目录）')
    parser.add_argument('--image_dir', type=str, default=None, help='可显式指定图像目录')
    parser.add_argument('--mask_dir', type=str, default=None, help='可显式指定标注目录')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/DG-TTA/checkpoints', help='保存测试结果的根目录')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['nnunet', 'unet3d'])
    parser.add_argument('--model_path', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth', help='模型权重路径；设为 "default" 则按模型类型选择内置默认路径')
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--adapt_steps', type=int, default=4)
    parser.add_argument('--eval_mode', action='store_true', help='若给定，则以 eval() 模式运行（默认 train()）')
    parser.add_argument('--freeze_other', action='store_true', help='若给定，仅更新 BN 仿射参数')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_test', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--spacing', type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument('--intensity_range', type=float, nargs=2, default=(-200, 400))
    parser.add_argument('--bg_channel', type=int, default=0, help='背景通道索引（并到二类时使用）')
    parser.add_argument('--positive_labels', type=str, default='1', help="哪些标签 id 视作肿瘤（前景），逗号分隔，如 '1' 或 '1,2'")
    args = parser.parse_args()
    try:
        args.positive_labels = [int(x) for x in args.positive_labels.split(',') if x.strip() != '']
    except Exception:
        raise ValueError("positive_labels 参数格式错误，应为逗号分隔的整数，如 '1' 或 '1,2'")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    test_on_target(args, device)
