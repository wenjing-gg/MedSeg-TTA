import argparse
import os
import glob
import datetime
import traceback
from typing import Tuple, Optional, Dict, List
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, Resized, NormalizeIntensityd, ToTensord
from nnunet import PlainConvUNet
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
from utils.tent import configure_model, collect_params, Tent

def get_dataset_type_from_path(data_path: str) -> str:
    data_path = data_path.replace('\\', '/').lower()
    if 'tta-3dct' in data_path or 'tta-ct' in data_path or 'ct' in data_path:
        return 'CT'
    return 'CT'

def get_dataset_paths(dataset_type: str, base_dir: str='/home/yuwenjing/data/tta_dataset', subfolder: Optional[str]=None) -> Tuple[str, str]:
    dataset_mapping = {'CT': 'TTA-3DCT'}
    if dataset_type not in dataset_mapping:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')
    dataset_folder = dataset_mapping[dataset_type]
    dataset_path = os.path.join(base_dir, dataset_folder)
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

def dia2mask(diameter: int, shape):
    if isinstance(shape, (list, tuple)):
        if len(shape) == 5:
            D, H, W = (shape[2], shape[3], shape[4])
        elif len(shape) == 4:
            D, H, W = (shape[1], shape[2], shape[3])
        else:
            D, H, W = shape
    else:
        raise ValueError('shape must be tuple/list')
    mask = np.zeros((D, H, W), dtype=np.float32)
    r = max(int(diameter // 2), 0)
    cz, cy, cx = (D // 2, H // 2, W // 2)
    for z in range(max(0, cz - r), min(D, cz + r + 1)):
        for y in range(max(0, cy - r), min(H, cy + r + 1)):
            for x in range(max(0, cx - r), min(W, cx + r + 1)):
                if ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) ** 0.5 <= r:
                    mask[z, y, x] = 1.0
    return mask

def get_saclick(outputs: torch.Tensor, masks: torch.Tensor, diameter_rates: List[float], unbiased: bool=False):
    B, C, D, H, W = outputs.shape
    device = outputs.device
    predict = (outputs.argmax(dim=1, keepdim=True) > 0).float()
    mask = (masks > 0).float()
    sphere_list, fake_rate_list = ([], [])
    for i in range(B):
        p = predict[i, 0].detach().cpu().numpy()
        m = mask[i, 0].detach().cpu().numpy()
        sizes = []
        x_proj = np.max(p, axis=(0, 1))
        sizes.append(int(np.sum(x_proj > 0)) or p.shape[2])
        y_proj = np.max(p, axis=(0, 2))
        sizes.append(int(np.sum(y_proj > 0)) or p.shape[1])
        z_proj = np.max(p, axis=(1, 2))
        sizes.append(int(np.sum(z_proj > 0)) or p.shape[0])
        min_box = max(1, min(sizes))
        diameter = max(min(int(min_box * diameter_rates[0]), int(min_box ** 2 * diameter_rates[1])), 1)
        sphere = torch.zeros_like(mask[i:i + 1])
        if np.sum(m) > 0:
            idx = np.nonzero(m)
            cz, cy, cx = (int(np.mean(idx[0])), int(np.mean(idx[1])), int(np.mean(idx[2])))
        else:
            cz, cy, cx = (D // 2, H // 2, W // 2)
        r = diameter // 2
        for zz in range(max(0, cz - r), min(D, cz + r + 1)):
            for yy in range(max(0, cy - r), min(H, cy + r + 1)):
                for xx in range(max(0, cx - r), min(W, cx + r + 1)):
                    if ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) ** 0.5 <= r:
                        sphere[0, 0, zz, yy, xx] = 1.0
        if unbiased:
            sphere_real = sphere * mask[i:i + 1]
            fake_rate = (sphere.sum() - sphere_real.sum()) / (sphere.sum() + 1e-08)
            sphere_list.append(sphere_real)
        else:
            sphere_real = sphere * mask[i:i + 1]
            fake_rate = (sphere.sum() - sphere_real.sum()) / (sphere.sum() + 1e-08)
            sphere_list.append(sphere)
        fake_rate_list.append(fake_rate.item())
    spheres = torch.cat(sphere_list, dim=0)
    avg_fake_rate = float(np.mean(fake_rate_list))
    expanded_spheres = torch.zeros((B, C, D, H, W), device=device, dtype=spheres.dtype)
    for ci in range(C):
        expanded_spheres[:, ci:ci + 1] = spheres
    return (expanded_spheres, avg_fake_rate)

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

def build_tent_model(model, args):
    model = configure_model(model)
    params, _ = collect_params(model)
    optimizer = optim.Adam(params, lr=args.lr)
    tented_model = Tent(model, optimizer)
    return tented_model

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 CT 模态 SaTTCA 测试 | 模型: {args.model_type}')
    print(f'{'=' * 40}\n')
    try:
        model = get_model(args.model_type, device)
        model_path = args.model_path
        if model_path == 'default':
            model_path = '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth' if args.model_type.lower() == 'nnunet' else '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth'
        print(f'📦 加载模型权重: {model_path}')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'未找到模型权重文件: {model_path}')
        state = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        tent_model = build_tent_model(model, args)
        test_loader, _ = get_ct_test_loader(image_dir=args.image_dir, mask_dir=args.mask_dir, target_dir=args.target_root, dataset_type='CT' if args.target_root is None else None, batch_size=args.batch_test, num_workers=args.num_workers, image_size=(args.image_size, args.image_size, args.image_size), spacing=tuple(args.spacing), intensity_range=tuple(args.intensity_range), positive_labels=args.positive_labels)
        K = 2
        all_dice = [[] for _ in range(K)]
        all_hd95 = [[] for _ in range(K)]
        all_IoU = [[] for _ in range(K)]
        all_pa = [[] for _ in range(K)]
        all_RVE = [[] for _ in range(K)]
        all_sensitivity = [[] for _ in range(K)]
        all_ppv = [[] for _ in range(K)]
        metric_rows = []
        loss_log = {'dice_loss': [], 'bce_loss': [], 'total_loss': []}
        tumor_idx = 1
        for batch in tqdm(test_loader, desc='SaTTCA 推理+适应'):
            if len(batch) == 3:
                imgs, labels, filenames = batch
            else:
                imgs, labels = batch[:2]
                filenames = [f'case_{i}' for i in range(imgs.size(0))]
            imgs, labels = (imgs.to(device), labels.to(device))
            with torch.no_grad():
                model.eval()
                logits0 = model(imgs)
                logits0 = logits0[0] if isinstance(logits0, tuple) else logits0
            sphere, fake_rate = get_saclick(logits0, labels, [1.0, 0.05], unbiased=False)
            outputs, loss_dict, _ = tent_model([imgs, sphere])
            bin_probs = merge_logits_to_binary(outputs, bg_channel=args.bg_channel)
            dice_vals = cal_dice(bin_probs, labels.squeeze(1))
            hd95_vals = cal_hd95(bin_probs, labels.squeeze(1))
            iou_vals = IoU(bin_probs, labels.squeeze(1))
            pa_vals = PA(bin_probs, labels.squeeze(1), K)
            rve_vals = cal_RVE(bin_probs, labels.squeeze(1))
            sen_vals = cal_sensitivity(bin_probs, labels.squeeze(1))
            ppv_vals = cal_ppv(bin_probs, labels.squeeze(1))
            for lk in ['dice_loss', 'bce_loss', 'total_loss']:
                if lk in loss_dict:
                    v = loss_dict[lk]
                    v = v.detach().item() if torch.is_tensor(v) else float(v)
                    loss_log[lk].append(v)
            for i in range(K):
                all_dice[i].append(safe_value(dice_vals[i]))
                all_hd95[i].append(safe_value(hd95_vals[i]))
                all_IoU[i].append(safe_value(iou_vals[i]))
                all_pa[i].append(safe_value(pa_vals[i]))
                all_RVE[i].append(safe_value(rve_vals[i]))
                all_sensitivity[i].append(safe_value(sen_vals[i]))
                all_ppv[i].append(safe_value(ppv_vals[i]))
            for j, fname in enumerate(filenames):
                metric_rows.append({'file_id': fname, 'dice_tumor': float(dice_vals[tumor_idx]), 'hd95_tumor': float(hd95_vals[tumor_idx]), 'iou_tumor': float(iou_vals[tumor_idx]), 'pa_tumor': float(pa_vals[tumor_idx]), 'rve_tumor': float(rve_vals[tumor_idx]), 'sen_tumor': float(sen_vals[tumor_idx]), 'ppv_tumor': float(ppv_vals[tumor_idx]), 'fake_rate': fake_rate})

        def ms(x):
            x = np.asarray(x, dtype=np.float32)
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
        t = 1
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result = f'测试配置:\n  模型类型: {args.model_type}\n  模型路径: {model_path}\n  学习率:   {args.lr}\n  图像尺寸: {args.image_size}^3, spacing={tuple(args.spacing)}\n  CT 强度范围: {tuple(args.intensity_range)}\n\n[Tumor] 指标均值 ± 标准差：\n  Dice        : {mean_dice[t]:.4f} ± {std_dice[t]:.4f}\n  HD95 (mm)   : {mean_hd95[t]:.2f} ± {std_hd95[t]:.2f}\n  IoU         : {mean_IoU[t]:.4f} ± {std_IoU[t]:.4f}\n  PA          : {mean_pa[t]:.4f} ± {std_pa[t]:.4f}\n  RVE         : {mean_RVE[t]:.4f} ± {std_RVE[t]:.4f}\n  Sensitivity : {mean_sen[t]:.4f} ± {std_sen[t]:.4f}\n  PPV         : {mean_ppv[t]:.4f} ± {std_ppv[t]:.4f}\n'
        print('\n' + result)
        result_dir = os.path.join(args.checkpoint_dir, f'{args.model_type}_CT_SaTTCA')
        os.makedirs(result_dir, exist_ok=True)
        out_txt = os.path.join(result_dir, f'sattca_results_{ts}.txt')
        with open(out_txt, 'w') as f:
            f.write(result)
        df_detail = pd.DataFrame(metric_rows)
        out_csv = os.path.join(result_dir, f'sattca_sample_metrics_{ts}.csv')
        df_detail.to_csv(out_csv, index=False)
        summary = {'metric': ['dice', 'hd95', 'iou', 'pa', 'rve', 'sensitivity', 'ppv'], 'mean': [mean_dice[t], mean_hd95[t], mean_IoU[t], mean_pa[t], mean_RVE[t], mean_sen[t], mean_ppv[t]], 'std': [std_dice[t], std_hd95[t], std_IoU[t], std_pa[t], std_RVE[t], std_sen[t], std_ppv[t]]}
        df_sum = pd.DataFrame(summary)
        out_sum = os.path.join(result_dir, f'sattca_summary_{ts}.csv')
        df_sum.to_csv(out_sum, index=False)
        print(f'✅ 结果已保存: {out_txt}')
        print(f'📄 样本级指标: {out_csv}')
        print(f'📊 摘要统计  : {out_sum}')
        tta_weights_dir = '/home/yuwenjing/DeepLearning_ywj/tta/SaTTCA/checkpoints/tta_results/unet3d_tta_weights/all'
        os.makedirs(tta_weights_dir, exist_ok=True)
        tta_weights_path = os.path.join(tta_weights_dir, 'unet3d_SaTTCA_CT.pth')
        adapted_model = tent_model.model
        torch.save({'model_state_dict': adapted_model.state_dict(), 'timestamp': ts, 'model_type': args.model_type, 'original_model_path': model_path, 'lr': args.lr, 'image_size': args.image_size, 'spacing': tuple(args.spacing), 'intensity_range': tuple(args.intensity_range), 'mean_dice_tumor': mean_dice[t], 'mean_hd95_tumor': mean_hd95[t]}, tta_weights_path)
        print(f'💾 适应后模型权重已保存: {tta_weights_path}')
        return True
    except Exception as e:
        print(f'❌ 测试失败: {str(e)}')
        traceback.print_exc()
        return False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CT 测试时域适应（SaTTCA：TENT + Sphere-click）')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB', help='目标数据集根目录（包含 image/ 与 mask/ 子目录）')
    parser.add_argument('--image_dir', type=str, default=None, help='可显式指定图像目录')
    parser.add_argument('--mask_dir', type=str, default=None, help='可显式指定标注目录')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/SaTTCA/checkpoints', help='保存测试结果的根目录')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['nnunet', 'unet3d'])
    parser.add_argument('--model_path', type=str, default='default', help='模型权重路径；"default" 则按模型类型选择内置默认路径')
    parser.add_argument('--lr', type=float, default=1e-06)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_test', type=int, default=2)
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
