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
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, Resized, NormalizeIntensityd, ToTensord
from unet3d import UNet3d_PLS_FAS_CT
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv

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
        data_dict = self.transforms(data_dict)
        image = data_dict['image']
        label = data_dict['label']
        filename = data_dict['image_name']
        image = image.float() if isinstance(image, torch.Tensor) else torch.tensor(image, dtype=torch.float32)
        label = label.long() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)
        label = binarize_label_tensor(label, self.positive_labels)
        return (image, label, filename)

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

class TTATrainer:

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.result_dir = os.path.join(args.checkpoint_dir, 'ProSFDA_results_single')
        self.weights_dir = os.path.join(self.result_dir, 'weights')
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        self.model = None
        self.pretrained_bn = {}
        self.pls_optimizer = None
        self.fas_optimizer = None
        self.pls_losses = []
        self.fas_losses = []

    def initialize_and_load(self):
        print('🔧 构建 UNet3d_PLS_FAS_CT 并加载预训练...')
        patch = (32, 128, 128)
        self.model = UNet3d_PLS_FAS_CT(patch_size=patch, pretrained_path=None).to(self.device)
        ckpt_path = None if self.args.model_path == 'default' else self.args.model_path
        if ckpt_path is None:
            ckpt_path = '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth'
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = self._extract_state_dict_for_load(ckpt)
        load_info = self.model.load_state_dict(state_dict, strict=False)
        try:
            missing, unexpected = load_info
            print(f'✅ 权重加载完成：missing={len(missing)}, unexpected={len(unexpected)}')
        except Exception:
            print('✅ 权重加载完成（无详细缺失/意外键可用）')
        self._save_pretrained_bn_stats()
        print(f'✅ 预训练 BN 统计条目: {len(self.pretrained_bn)}')

    def _extract_state_dict_for_load(self, checkpoint):
        if isinstance(checkpoint, dict):
            for k in ['model_state_dict', 'state_dict', 'network', 'model', 'net']:
                if k in checkpoint and isinstance(checkpoint[k], dict):
                    return checkpoint[k]
            if all((isinstance(v, (torch.Tensor, torch.nn.Parameter)) for v in checkpoint.values())):
                return checkpoint
        return checkpoint

    def _save_pretrained_bn_stats(self):
        self.pretrained_bn = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm3d):
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    self.pretrained_bn[f'{name}.running_mean'] = module.running_mean.clone().detach()
                if hasattr(module, 'running_var') and module.running_var is not None:
                    self.pretrained_bn[f'{name}.running_var'] = module.running_var.clone().detach()
        if not self.pretrained_bn:
            raise RuntimeError('未捕获到任何 BN 统计（running_mean/var）用于 PLS 对齐。')

    def setup_training(self):
        assert hasattr(self.model, 'data_prompt_param'), '模型缺少 data_prompt_param'
        for n, p in self.model.named_parameters():
            p.requires_grad = 'data_prompt' in n
        self.pls_optimizer = optim.Adam([self.model.data_prompt_param], lr=self.args.lr)
        print(f'✅ PLS 优化器就绪：仅 data_prompt，lr={self.args.lr}')
        fas_params: List[torch.nn.Parameter] = []
        for n, p in self.model.named_parameters():
            is_norm = any((k in n.lower() for k in ['norm', 'bn', 'batchnorm', 'batch_norm', 'groupnorm', 'layernorm', 'instancenorm']))
            if is_norm:
                fas_params.append(p)
        if not fas_params:
            norm_module_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
            for mod_name, mod in self.model.named_modules():
                if isinstance(mod, norm_module_types):
                    for p in mod.parameters(recurse=True):
                        fas_params.append(p)
        unique_ids = set()
        filtered_params = []
        prompt_id = id(self.model.data_prompt_param)
        for p in fas_params:
            if id(p) == prompt_id:
                continue
            if id(p) not in unique_ids:
                unique_ids.add(id(p))
                filtered_params.append(p)
        fas_params = filtered_params
        selected_ids = {id(p) for p in fas_params}
        for n, p in self.model.named_parameters():
            p.requires_grad = id(p) in selected_ids
        self.model.data_prompt_param.requires_grad = False
        if not fas_params:
            raise RuntimeError('FAS 阶段未找到可训练的归一化层参数')
        fas_count = sum((p.numel() for p in fas_params))
        if fas_count > 10000:
            fas_lr = self.args.lr * 0.01
        elif fas_count > 1000:
            fas_lr = self.args.lr * 0.05
        else:
            fas_lr = self.args.lr * 0.1
        self.fas_optimizer = optim.Adam(fas_params, lr=fas_lr)
        fas_param_ids = {id(p) for p in fas_params}
        chosen_names = []
        for n, p in self.model.named_parameters():
            if id(p) in fas_param_ids:
                try:
                    shape_str = tuple(p.shape)
                except Exception:
                    shape_str = '<?>'
                chosen_names.append(f'{n} {shape_str}')
        print(f'✅ FAS 优化器就绪：参数数={fas_count:,}，组内张量={len(fas_params)}，lr={fas_lr}')
        print('   - 适应参数示例（最多前10条）：')
        for s in chosen_names[:10]:
            print('     •', s)

    def compute_pls_loss(self, bn_features: List, max_layers: int):
        preb = {}
        for name, tensor in self.pretrained_bn.items():
            if name.endswith('.running_mean'):
                k = name[:-13]
                preb.setdefault(k, {})['mean'] = tensor
            elif name.endswith('.running_var'):
                k = name[:-12]
                preb.setdefault(k, {})['var'] = tensor
        total = 0.0
        used = 0
        layer_names = list(preb.keys())
        for i, f in enumerate(bn_features[:min(max_layers, len(layer_names))]):
            if not hasattr(f, 'features') or f.features is None:
                continue
            if not f.features.requires_grad:
                continue
            cur_mean = f.features.mean(dim=(0, 2, 3, 4))
            cur_var = f.features.var(dim=(0, 2, 3, 4), unbiased=False)
            key = layer_names[i]
            tgt_mean = preb[key].get('mean', None)
            tgt_var = preb[key].get('var', None)
            if tgt_mean is None or tgt_var is None:
                continue
            if cur_mean.shape != tgt_mean.shape or cur_var.shape != tgt_var.shape:
                continue
            mean_loss = F.l1_loss(cur_mean, tgt_mean.detach())
            var_loss = F.l1_loss(cur_var, tgt_var.detach())
            total = total + (mean_loss + self.args.alpha * var_loss)
            used += 1
        if used == 0:
            total = total + self.args.alpha * torch.norm(self.model.data_prompt_param, p=2)
        return total

    def train_pls_phase(self, loader: DataLoader):
        print('\n🎯 阶段一：PLS（BN 对齐，仅 data_prompt）')
        self.model.train()
        pls_epochs = max(1, self.args.tta_epochs // 2)
        for ep in range(pls_epochs):
            ep_losses = []
            valid = 0
            pbar = tqdm(loader, desc=f'PLS Epoch {ep + 1}/{pls_epochs}')
            for bidx, (imgs, _, *_) in enumerate(pbar):
                imgs = imgs.to(self.device)
                self.pls_optimizer.zero_grad()
                seg, bn_f = self.model(imgs, training=True, gfeat=False)
                valid_feats = [f for f in bn_f if hasattr(f, 'features') and f.features is not None and f.features.requires_grad]
                if not valid_feats:
                    continue
                loss = self.compute_pls_loss(valid_feats, self.args.bn_layers)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.model.data_prompt_param], max_norm=1.0)
                self.pls_optimizer.step()
                ep_losses.append(loss.item())
                valid += 1
                pbar.set_postfix({'BN_Loss': f'{loss.item():.4f}', 'Prompt_Norm': f'{self.model.data_prompt_param.norm().item():.4f}', 'Valid': f'{valid}'})
            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            self.pls_losses.append(avg_loss)
            print(f'PLS Epoch {ep + 1}: Loss={avg_loss:.4f}, Prompt‖·‖={self.model.data_prompt_param.norm().item():.6f}')
        print(f'✅ PLS 完成，最终 Prompt 范数: {self.model.data_prompt_param.norm().item():.6f}')

    def _forward_with_zero_prompt_no_grad(self, imgs):
        old = self.model.data_prompt_param.data.clone()
        try:
            self.model.data_prompt_param.data.zero_()
            with torch.no_grad():
                _, gfeat_orig = self.model(imgs, training=False, gfeat=True)
        finally:
            self.model.data_prompt_param.data.copy_(old)
        return gfeat_orig

    def train_fas_phase(self, loader: DataLoader):
        print('\n🎯 阶段二：FAS（全局特征对齐，仅归一化层）')
        self.model.train()
        self.model.data_prompt_param.requires_grad = False
        fas_epochs = self.args.tta_epochs - max(1, self.args.tta_epochs // 2)
        fas_epochs = max(1, fas_epochs)
        for ep in range(fas_epochs):
            ep_losses = []
            valid = 0
            pbar = tqdm(loader, desc=f'FAS Epoch {ep + 1}/{fas_epochs}')
            for bidx, (imgs, _, *_) in enumerate(pbar):
                imgs = imgs.to(self.device)
                self.fas_optimizer.zero_grad()
                seg_mixed, gfeat_mixed = self.model(imgs, training=False, gfeat=True)
                gfeat_orig = self._forward_with_zero_prompt_no_grad(imgs)
                align_loss = F.l1_loss(gfeat_mixed, gfeat_orig.detach())
                cons_loss = F.mse_loss(seg_mixed, seg_mixed.detach())
                loss = align_loss + self.args.gamma * cons_loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fas_optimizer.param_groups[0]['params'], max_norm=1.0)
                self.fas_optimizer.step()
                ep_losses.append(loss.item())
                valid += 1
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Align': f'{align_loss.item():.4f}', 'Valid': f'{valid}'})
            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            self.fas_losses.append(avg_loss)
            print(f'FAS Epoch {ep + 1}: Loss={avg_loss:.4f}, Valid={valid}')
        print('✅ FAS 完成！')

    def evaluate(self, loader: DataLoader):
        print('\n🧪 开始评估（CT 二类：背景/肿瘤）...')
        self.model.eval()
        K = 2
        all_dice = [[] for _ in range(K)]
        all_hd95 = [[] for _ in range(K)]
        all_iou = [[] for _ in range(K)]
        all_pa = [[] for _ in range(K)]
        all_rve = [[] for _ in range(K)]
        all_sen = [[] for _ in range(K)]
        all_ppv = [[] for _ in range(K)]
        with torch.no_grad():
            for imgs, labels, *_ in tqdm(loader, desc='评估中'):
                imgs, labels = (imgs.to(self.device), labels.to(self.device))
                probs = self.model(imgs)
                dice = cal_dice(probs, labels.squeeze(1))
                hd95 = cal_hd95(probs, labels.squeeze(1))
                iou = IoU(probs, labels.squeeze(1))
                pa = PA(probs, labels.squeeze(1), K)
                rve = cal_RVE(probs, labels.squeeze(1))
                sen = cal_sensitivity(probs, labels.squeeze(1))
                ppv = cal_ppv(probs, labels.squeeze(1))
                for i in range(K):
                    all_dice[i].append(safe_value(dice[i]))
                    all_hd95[i].append(safe_value(hd95[i]))
                    all_iou[i].append(safe_value(iou[i]))
                    all_pa[i].append(safe_value(pa[i]))
                    all_rve[i].append(safe_value(rve[i]))
                    all_sen[i].append(safe_value(sen[i]))
                    all_ppv[i].append(safe_value(ppv[i]))

        def ms(x):
            x = np.asarray(x, dtype=np.float32)
            return (float(x.mean()) if len(x) else 0.0, float(x.std()) if len(x) else 0.0)
        stats = {'dice': {'mean': [ms(v)[0] for v in all_dice], 'std': [ms(v)[1] for v in all_dice]}, 'hd95': {'mean': [ms(v)[0] for v in all_hd95], 'std': [ms(v)[1] for v in all_hd95]}, 'iou': {'mean': [ms(v)[0] for v in all_iou], 'std': [ms(v)[1] for v in all_iou]}, 'pa': {'mean': [ms(v)[0] for v in all_pa], 'std': [ms(v)[1] for v in all_pa]}, 'rve': {'mean': [ms(v)[0] for v in all_rve], 'std': [ms(v)[1] for v in all_rve]}, 'sensitivity': {'mean': [ms(v)[0] for v in all_sen], 'std': [ms(v)[1] for v in all_sen]}, 'ppv': {'mean': [ms(v)[0] for v in all_ppv], 'std': [ms(v)[1] for v in all_ppv]}}
        return stats

    def save_results(self, stats, model_path):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.splitext(os.path.basename(model_path if model_path != 'default' else '/home/yuwenjing/DeepLearning_ywj/tta/unet3d_best.pth'))[0]
        out_w = os.path.join(self.weights_dir, f'UNet3d_ProSFDA_CT.pth')
        torch.save({'model_state_dict': self.model.state_dict(), 'data_prompt': self.model.data_prompt_param, 'losses': {'pls': self.pls_losses, 'fas': self.fas_losses}, 'final_prompt_norm': self.model.data_prompt_param.norm().item(), 'config': vars(self.args)}, out_w)
        t = 1
        report = f'\n{'=' * 70}\nUNet3d_PLS_FAS_CT ProSFDA（单模型）- CT 二类 结果\n{'=' * 70}\n时间: {ts}\n预训练: {model_path}\n保存权重: {out_w}\n\n训练参数:\n- 适应轮数: {self.args.tta_epochs}（PLS≈{max(1, self.args.tta_epochs // 2)}, FAS≈{self.args.tta_epochs - max(1, self.args.tta_epochs // 2)}）\n- 学习率: PLS={self.args.lr}, FAS={self.args.lr * 0.1}\n- Alpha(BN): {self.args.alpha}\n- Gamma(FAS): {self.args.gamma}\n- BN层数: {self.args.bn_layers}\n- 最终 Prompt 范数: {self.model.data_prompt_param.norm().item():.6f}\n\n[Tumor] 指标均值 ± 标准差：\n- Dice        : {stats['dice']['mean'][t]:.4f} ± {stats['dice']['std'][t]:.4f}\n- HD95 (mm)   : {stats['hd95']['mean'][t]:.2f} ± {stats['hd95']['std'][t]:.2f}\n- IoU         : {stats['iou']['mean'][t]:.4f} ± {stats['iou']['std'][t]:.4f}\n- PA          : {stats['pa']['mean'][t]:.4f} ± {stats['pa']['std'][t]:.4f}\n- RVE         : {stats['rve']['mean'][t]:.4f} ± {stats['rve']['std'][t]:.4f}\n- Sensitivity : {stats['sensitivity']['mean'][t]:.4f} ± {stats['sensitivity']['std'][t]:.4f}\n- PPV         : {stats['ppv']['mean'][t]:.4f} ± {stats['ppv']['std'][t]:.4f}\n{'=' * 70}\n'
        out_txt = os.path.join(self.result_dir, f'{model_name}_CT_ProSFDA_single_{ts}.txt')
        with open(out_txt, 'w') as f:
            f.write(report)
        print(report)
        print(f'✅ 已保存：\n  - 权重: {out_w}\n  - 报告: {out_txt}')
        return out_w

def test_on_target(args, device):
    print(f'\n{'=' * 60}\n🧪 ProSFDA（单模型）: CT\n{'=' * 60}\n')
    trainer = TTATrainer(args, device)
    trainer.initialize_and_load()
    trainer.setup_training()
    test_loader, _ = get_ct_test_loader(image_dir=args.image_dir, mask_dir=args.mask_dir, target_dir=args.target_root, dataset_type='CT' if args.target_root is None else None, subfolder=args.subfolder, base_dir=args.base_dir, batch_size=args.batch_test, num_workers=args.num_workers, image_size=(args.image_size, args.image_size, args.image_size), spacing=tuple(args.spacing), intensity_range=tuple(args.intensity_range), positive_labels=args.positive_labels)
    print(f'📊 批次数: {len(test_loader)}')
    trainer.train_pls_phase(test_loader)
    trainer.train_fas_phase(test_loader)
    stats = trainer.evaluate(test_loader)
    trainer.save_results(stats, args.model_path)
    return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CT ProSFDA（单模型 UNet3d_PLS_FAS_CT）')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB', help='目标数据集根目录（含 image/ 与 mask/）')
    parser.add_argument('--image_dir', type=str, default=None, help='显式指定图像目录')
    parser.add_argument('--mask_dir', type=str, default=None, help='显式指定标注目录')
    parser.add_argument('--base_dir', type=str, default='/home/yuwenjing/data/tta_dataset', help='当未指定 image_dir/mask_dir 时，配合 dataset_type 使用')
    parser.add_argument('--subfolder', type=str, default=None, help='可选子目录（不再自动选择）')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/ProSFDA/checkpoints', help='保存适应结果与权重的根目录')
    parser.add_argument('--model_path', type=str, default='default', help='预训练权重路径；"default" 则使用内置路径')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率（PLS 基准，FAS=lr*0.1）')
    parser.add_argument('--tta_epochs', type=int, default=30, help='总适应轮数')
    parser.add_argument('--alpha', type=float, default=0.01, help='BN 对齐中 var 的权重')
    parser.add_argument('--gamma', type=float, default=0.1, help='FAS 一致性项权重')
    parser.add_argument('--bn_layers', type=int, default=12, help='每步用于 BN 对齐的层数上限')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_test', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--spacing', type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument('--intensity_range', type=float, nargs=2, default=(-200, 400))
    parser.add_argument('--positive_labels', type=str, default='1', help="哪些标签 id 视作肿瘤（前景），逗号分隔，如 '1' 或 '1,2'")
    args = parser.parse_args()
    try:
        args.positive_labels = [int(x) for x in args.positive_labels.split(',') if x.strip() != '']
    except Exception:
        raise ValueError("positive_labels 参数格式错误，应为逗号分隔的整数，如 '1' 或 '1,2'")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️ 设备: {device} | TTA: epochs={args.tta_epochs}, lr={args.lr}')
    try:
        ok = test_on_target(args, device)
        if ok:
            print('✅ ProSFDA（单模型，CT）完成')
    except Exception as e:
        print(f'❌ 失败: {e}')
        print(traceback.format_exc())
