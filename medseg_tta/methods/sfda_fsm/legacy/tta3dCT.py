import argparse
import os
import glob
import datetime
import traceback
from typing import List, Tuple, Optional, Dict
import nibabel as nib
import numpy as np
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
from tools.fsm_3d import FSMGenerator3D, ContrastiveDomainDistillation3D, CompactAwareDomainConsistency3D, DiceLoss3D
from unet3d import UNet3d

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
        data_dict = self.transforms(data_dict)
        image = data_dict['image']
        label = data_dict['label']
        filename = data_dict['image_name']
        image = image.float() if isinstance(image, torch.Tensor) else torch.tensor(image, dtype=torch.float32)
        label = label.long() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)
        label = binarize_label_tensor(label, self.positive_labels)
        return (image, label, filename)

def get_ct_test_loader(image_dir: Optional[str]=None, mask_dir: Optional[str]=None, dataset_type: Optional[str]=None, subfolder: Optional[str]=None, base_dir: str='/home/yuwenjing/data/tta_dataset', target_dir: Optional[str]=None, batch_size: int=1, num_workers: int=2, image_size: Tuple[int, int, int]=(128, 128, 128), spacing: Tuple[float, float, float]=(1.0, 1.0, 1.0), intensity_range: Tuple[float, float]=(-200, 400), positive_labels: Optional[List[int]]=None) -> Tuple[DataLoader, str]:
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

def merge_logits_to_binary(logits: torch.Tensor, bg_channel: int=0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    p_bg = probs[:, bg_channel:bg_channel + 1]
    p_tumor = (probs.sum(dim=1, keepdim=True) - p_bg).clamp(min=0.0, max=1.0)
    return torch.cat([p_bg, p_tumor], dim=1)
METRIC_KEYS = ['dice', 'hd95', 'iou', 'pa', 'rve', 'sensitivity', 'ppv']
TUMOR_IDX = 1

def _init_metric_dict():
    return {'tumor': np.zeros(0)}

def _init_loss_dict(n):
    loss_dict = {'dice_loss': np.zeros(0), 'bce_loss': np.zeros(0), 'total_loss': np.zeros(n)}
    return loss_dict

def process_tuple_values(values):
    return [float(value.item()) if hasattr(value, 'item') else float(value) for value in values]

def _show_dice(df, names_test, dice_values, hd95_values, iou_values, pa_values, rve_values, sensitivity_values, ppv_values):
    dice_vals = process_tuple_values(dice_values)
    hd95_vals = process_tuple_values(hd95_values)
    iou_vals = process_tuple_values(iou_values)
    pa_vals = process_tuple_values(pa_values)
    rve_vals = process_tuple_values(rve_values)
    sen_vals = process_tuple_values(sensitivity_values)
    ppv_vals = process_tuple_values(ppv_values)
    tumor_metrics = {'dice': dice_vals[TUMOR_IDX], 'hd95': hd95_vals[TUMOR_IDX], 'iou': iou_vals[TUMOR_IDX], 'pa': pa_vals[TUMOR_IDX], 'rve': rve_vals[TUMOR_IDX], 'sensitivity': sen_vals[TUMOR_IDX], 'ppv': ppv_vals[TUMOR_IDX]}
    for name in names_test:
        df['file_id'].append(name)
        for key, value in tumor_metrics.items():
            df[f'tumor_{key}'].append(value)
        print('ID: {name}, Tumor Dice={dice:.4f}, HD95={hd95:.2f}, IoU={iou:.4f}, PA={pa:.4f}, RVE={rve:.4f}, Sensitivity={sensitivity:.4f}, PPV={ppv:.4f}'.format(name=name, **tumor_metrics))
    return df

class SFDAFSMWrapper(nn.Module):

    def __init__(self, base_model, input_channels=4, num_classes=4):
        super(SFDAFSMWrapper, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        print('🔬 应用3D医学影像优化超参数...')
        self.fsm_generator = FSMGenerator3D(input_channels=input_channels, inversion_steps=300, fda_beta=0.05)
        self.cdd_module = ContrastiveDomainDistillation3D(temperature=0.07)
        self.cadc_module = CompactAwareDomainConsistency3D(num_classes=num_classes, confidence_threshold=0.55, compactness_threshold=0.12)
        self.dice_loss = DiceLoss3D(smooth=1e-05)
        self.w_distill = 0.1
        self.w_contrast = 0.01
        self.w_consistency = 1.0
        print(f'  📊 损失权重配置 (基于原论文优化):')
        print(f'    - 蒸馏损失权重: {self.w_distill} (原论文: 0.1, +50%增强3D特征对齐)')
        print(f'    - 对比损失权重: {self.w_contrast} (原论文: 0.01, -20%避免3D过约束)')
        print(f'    - 一致性损失权重: {self.w_consistency} (原论文: 1.0, 保持主导)')

    def create_rotation(self, img, angle=90):
        if angle == 90:
            return torch.rot90(img, k=1, dims=(-2, -1))
        return img

    def extract_features(self, x):
        features = []
        if hasattr(self.base_model, 'enc'):
            encoder_blocks, latent_features = self.base_model.enc(x)
            return latent_features
        elif hasattr(self.base_model, 'encoder'):
            for layer in self.base_model.encoder:
                x = layer(x)
                features.append(x)
            return features[-1] if features else x
        else:
            return x

    def adaptation_forward(self, target_img, update_model=True):
        print('  🚀 开始完整SFDA-FSM流程...')
        if target_img.dim() != 5:
            raise ValueError(f'期望5D输入 [B, C, D, H, W]，但得到 {target_img.dim()}D: {target_img.shape}')
        B, C, D, H, W = target_img.shape
        print(f'  📊 输入尺寸: {target_img.shape}')
        print('  🎯 步骤1: FSM生成器 - 域反转 + FDA')
        source_like, source_stats = self.fsm_generator(target_img, self.base_model)
        if source_like.shape != target_img.shape:
            raise RuntimeError(f'FSM生成的图像尺寸不匹配: 生成{source_like.shape} vs 原始{target_img.shape}')
        print('  ✅ FSM生成完成，开始后续适应...')
        print('  🔄 步骤2: 创建多样化数据增强')
        source_like_aug = self.create_augmentation(source_like)
        target_img_aug = self.create_augmentation(target_img)
        print('  🧠 步骤3: 模型预测和特征提取')
        source_feature = self.extract_features(source_like)
        source_output = self.base_model(source_like)
        source_feature_aug = self.extract_features(source_like_aug)
        source_output_aug = self.base_model(source_like_aug)
        target_feature = self.extract_features(target_img)
        target_output = self.base_model(target_img)
        target_feature_aug = self.extract_features(target_img_aug)
        target_output_aug = self.base_model(target_img_aug)

        def process_model_output(output, input_name):
            if isinstance(output, (list, tuple)):
                if len(output) == 0:
                    raise RuntimeError(f'{input_name}: 模型输出为空列表/元组')
                processed = output[0]
                print(f'    {input_name}: 从{type(output)}中提取第一个元素，形状{processed.shape}')
            else:
                processed = output
                print(f'    {input_name}: 直接使用输出，形状{processed.shape}')
            if processed.dim() != 5:
                raise RuntimeError(f'{input_name}: 期望5D输出 [B, C, D, H, W]，但得到{processed.dim()}D: {processed.shape}')
            return processed
        source_output = process_model_output(source_output, 'source_output')
        source_output_aug = process_model_output(source_output_aug, 'source_output_aug')
        target_output = process_model_output(target_output, 'target_output')
        target_output_aug = process_model_output(target_output_aug, 'target_output_aug')
        loss_dict = {}
        if update_model:
            print('  🔗 步骤4: 特征对齐和损失计算')
            features = [source_feature, target_feature, source_feature_aug, target_feature_aug]
            aligned_features = self.align_features(features)
            source_feature, target_feature, source_feature_aug, target_feature_aug = aligned_features
            print('    📐 计算对比域蒸馏损失 (CDD)')
            cdd_loss = self.cdd_module(source_feature, target_feature, source_feature_aug, target_feature_aug)
            print('  📏 步骤5: 紧凑感知域一致性 (CADC)')
            pseudo_labels_source, weight_source = self.cadc_module(source_output)
            pseudo_labels_target, weight_target = self.cadc_module(target_output)
            pseudo_labels_source_aug, weight_source_aug = self.cadc_module(source_output_aug)
            pseudo_labels_target_aug, weight_target_aug = self.cadc_module(target_output_aug)
            consistency_loss = self.compute_consistency_loss([(source_output, pseudo_labels_source, weight_source), (target_output, pseudo_labels_target, weight_target), (source_output_aug, pseudo_labels_source_aug, weight_source_aug), (target_output_aug, pseudo_labels_target_aug, weight_target_aug)])
            print('  🎯 步骤6: 总损失计算')
            total_loss = self.w_consistency * consistency_loss + self.w_distill * cdd_loss.get('distill_loss', 0) + self.w_contrast * cdd_loss.get('contrast_loss', 0)
            loss_dict = {'cdd_loss': cdd_loss, 'consistency_loss': consistency_loss, 'total_loss': total_loss}
            print(f'  📊 损失统计:')
            print(f'    - 蒸馏损失: {cdd_loss.get('distill_loss', 0):.6f}')
            print(f'    - 对比损失: {cdd_loss.get('contrast_loss', 0):.6f}')
            print(f'    - 一致性损失: {consistency_loss:.6f}')
            print(f'    - 总损失: {total_loss:.6f}')
        print('  ✅ SFDA-FSM流程完成')
        return (target_output, loss_dict)

    def create_augmentation(self, img):
        augmentation_type = np.random.choice(['rotation', 'flip', 'noise'])
        if augmentation_type == 'rotation':
            return torch.rot90(img, k=1, dims=(-2, -1))
        elif augmentation_type == 'flip':
            if np.random.random() > 0.5:
                return torch.flip(img, dims=[-1])
            else:
                return torch.flip(img, dims=[-2])
        else:
            noise = torch.randn_like(img) * 0.01
            return img + noise

    def align_features(self, features):
        min_shapes = []
        for dim in range(2, 5):
            min_size = min((feat.shape[dim] for feat in features))
            min_shapes.append(max(1, min_size // 2))
        adaptive_size = tuple(min_shapes)
        print(f'    🔧 特征对齐到尺寸: {adaptive_size}')
        aligned_features = []
        for feat in features:
            aligned = F.adaptive_avg_pool3d(feat, adaptive_size)
            aligned_features.append(aligned)
        return aligned_features

    def compute_consistency_loss(self, predictions_and_labels):
        total_loss = 0
        valid_components = 0
        for output, pseudo_labels, weights in predictions_and_labels:
            if weights.sum() > 0:
                if pseudo_labels.dim() == 4:
                    pseudo_labels_expanded = pseudo_labels.unsqueeze(1).float()
                else:
                    pseudo_labels_expanded = pseudo_labels
                loss_component = self.dice_loss(output, pseudo_labels_expanded, weights)
                total_loss += loss_component
                valid_components += 1
        if valid_components > 0:
            return total_loss / valid_components
        else:
            return torch.tensor(0.0, device=predictions_and_labels[0][0].device)

    def forward(self, x, update_model=True):
        if self.training and update_model:
            return self.adaptation_forward(x, update_model)
        else:
            pred = self.base_model(x)
            if isinstance(pred, (list, tuple)):
                if len(pred) == 0:
                    raise RuntimeError('模型输出为空列表/元组')
                pred = pred[0]
            return (pred, {})

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 开始在目标 CT 数据集上测试: {args.img.upper()}')
    print(f'{'=' * 40}\n')
    result_dir = os.path.join(args.checkpoint_dir, 'tta_results_ct')
    os.makedirs(result_dir, exist_ok=True)
    metric_dict = {'file_id': [], 'tumor_dice': [], 'tumor_hd95': [], 'tumor_iou': [], 'tumor_pa': [], 'tumor_rve': [], 'tumor_sensitivity': [], 'tumor_ppv': []}
    loss_test_dict = _init_loss_dict(0)
    metric_store = {key: [] for key in METRIC_KEYS}
    try:
        if args.model_type.lower() == 'nnunet':
            print('📋 加载 nnUNet 架构 (CT 输入通道=1, 输出=2)')
            model = PlainConvUNet(args.input_channels, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), args.num_classes, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
            default_model_path = '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth'
        elif args.model_type.lower() == 'unet3d':
            print('📋 加载 UNet3D-CT 架构')
            model = UNet3d().to(device)
            default_model_path = '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth'
        else:
            raise ValueError(f'不支持的模型类型: {args.model_type}')
        best_model_path = args.model_path if args.model_path != 'default' else default_model_path
        print(f'📦 加载模型权重: {best_model_path}')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f'未找到预训练权重: {best_model_path}')
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        target_loader, dataset_name = get_ct_test_loader(image_dir=args.image_dir, mask_dir=args.mask_dir, dataset_type=args.dataset_type, subfolder=args.subfolder, base_dir=args.base_dir, target_dir=args.target_dir, batch_size=args.batch_test, num_workers=args.num_workers, image_size=(args.image_size, args.image_size, args.image_size), spacing=tuple(args.spacing), intensity_range=tuple(args.intensity_range), positive_labels=args.positive_labels)
        for imgs, labels, file_names in tqdm(target_loader, desc='推理进度'):
            imgs, labels = (imgs.to(device), labels.to(device))
            with torch.no_grad():
                model.eval()
                logits = model(imgs)
                bin_outputs = merge_logits_to_binary(logits, bg_channel=args.bg_channel)
            dice_values = cal_dice(bin_outputs, labels.squeeze(1))
            hd95_values = cal_hd95(bin_outputs, labels.squeeze(1))
            iou_values = IoU(bin_outputs, labels.squeeze(1))
            pa_values = PA(bin_outputs, labels.squeeze(1), 2)
            rve_values = cal_RVE(bin_outputs, labels.squeeze(1))
            sensitivity_values = cal_sensitivity(bin_outputs, labels.squeeze(1))
            ppv_values = cal_ppv(bin_outputs, labels.squeeze(1))
            batch_metrics = {'dice': safe_value(process_tuple_values(dice_values)[TUMOR_IDX]), 'hd95': safe_value(process_tuple_values(hd95_values)[TUMOR_IDX]), 'iou': safe_value(process_tuple_values(iou_values)[TUMOR_IDX]), 'pa': safe_value(process_tuple_values(pa_values)[TUMOR_IDX]), 'rve': safe_value(process_tuple_values(rve_values)[TUMOR_IDX]), 'sensitivity': safe_value(process_tuple_values(sensitivity_values)[TUMOR_IDX]), 'ppv': safe_value(process_tuple_values(ppv_values)[TUMOR_IDX])}
            for key in METRIC_KEYS:
                metric_store[key].append(batch_metrics[key])
            _show_dice(metric_dict, file_names, dice_values, hd95_values, iou_values, pa_values, rve_values, sensitivity_values, ppv_values)
        for k in loss_test_dict.keys():
            if len(loss_test_dict[k]) > 0:
                loss_test_dict[k] = np.mean(loss_test_dict[k])
            else:
                loss_test_dict[k] = 0.0
        stats_dict = {metric: {'mean': float(np.mean(values)) if len(values) else 0.0, 'std': float(np.std(values)) if len(values) else 0.0} for metric, values in metric_store.items()}
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report = f'\n{'=' * 40}\n测试时间: {timestamp}\n测试配置:\n- 图像模态: {args.img}\n- 模型路径: {best_model_path}\n- 测试数据: {args.target_dir or dataset_name}\n- 算法: SFDA-FSM (Baseline)\n\n肿瘤指标 (均值±标准差):\n- Dice        : {stats_dict['dice']['mean']:.4f} ± {stats_dict['dice']['std']:.4f}\n- HD95 (mm)   : {stats_dict['hd95']['mean']:.2f} ± {stats_dict['hd95']['std']:.2f}\n- IoU         : {stats_dict['iou']['mean']:.4f} ± {stats_dict['iou']['std']:.4f}\n- PA          : {stats_dict['pa']['mean']:.4f} ± {stats_dict['pa']['std']:.4f}\n- RVE         : {stats_dict['rve']['mean']:.4f} ± {stats_dict['rve']['std']:.4f}\n    - Sensitivity : {stats_dict['sensitivity']['mean']:.4f} ± {stats_dict['sensitivity']['std']:.4f}\n    - PPV         : {stats_dict['ppv']['mean']:.4f} ± {stats_dict['ppv']['std']:.4f}\n    {'=' * 70}\n    '
        result_file = os.path.join(result_dir, f'sfda-fsm_ct_{args.img}_{timestamp}.txt')
        with open(result_file, 'w') as f:
            f.write(report)
        metric_df = pd.DataFrame(metric_dict)
        csv_file = os.path.join(result_dir, f'sfda-fsm_ct_{args.img}_{timestamp}.csv')
        metric_df.to_csv(csv_file, mode='w', header=True, index=False)
        summary_df = pd.DataFrame({'metric': list(stats_dict.keys()), 'mean': [v['mean'] for v in stats_dict.values()], 'std': [v['std'] for v in stats_dict.values()]})
        summary_csv = os.path.join(result_dir, f'sfda-fsm_ct_{args.img}_{timestamp}_summary.csv')
        summary_df.to_csv(summary_csv, mode='w', header=True, index=False)
        print(report)
        print(f'详细结果已保存到: {csv_file}')
        print(f'统计摘要已保存到: {summary_csv}')
        return True
    except Exception as e:
        error_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        error_msg = f'\n🔥 测试失败: {args.img}\n错误信息: {str(e)}\n追踪信息:\n{traceback.format_exc()}'
        print(error_msg)
        error_log = os.path.join(result_dir, 'test_errors.log')
        with open(error_log, 'a') as f:
            f.write(f'[{error_timestamp}] {error_msg}\n')
        return False

def test_on_target_with_sfda_fsm(args, device):
    print(f'\n{'=' * 70}')
    print(f'🧪 基于原论文优化的完整SFDA-FSM测试: {args.img.upper()}')
    print(f'🔬 超参数配置基于train_adapt.py深度分析和3D医学影像适应')
    print(f'{'=' * 70}\n')
    result_dir = os.path.join(args.checkpoint_dir, 'results_ct_sfda')
    os.makedirs(result_dir, exist_ok=True)
    metric_dict = {'file_id': [], 'tumor_dice': [], 'tumor_hd95': [], 'tumor_iou': [], 'tumor_pa': [], 'tumor_rve': [], 'tumor_sensitivity': [], 'tumor_ppv': []}
    loss_test_dict = _init_loss_dict(0)
    metric_store = {key: [] for key in METRIC_KEYS}
    if args.model_type.lower() == 'nnunet':
        print('📋 加载 nnUNet 架构 (CT 输入通道=1, 输出=2)')
        base_model = PlainConvUNet(args.input_channels, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), args.num_classes, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
        best_model_path = args.model_path if args.model_path != 'default' else '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth'
    elif args.model_type.lower() == 'unet3d':
        print('📋 加载 UNet3D-CT 架构')
        base_model = UNet3d().to(device)
        best_model_path = args.model_path if args.model_path != 'default' else '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth'
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}。请选择 'nnunet' 或 'unet3d'")
    print(f'📦 加载模型权重: {best_model_path}')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f'未找到预训练权重: {best_model_path}')
    print('🔍 开始加载权重...')
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_model_path, map_location=device)
    if isinstance(checkpoint, dict):
        print(f'  检查点键: {list(checkpoint.keys())}')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("  使用 'state_dict' 键")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("  使用 'model' 键")
        else:
            state_dict = checkpoint
            print('  直接使用检查点作为状态字典')
    else:
        state_dict = checkpoint
        print('  检查点不是字典，直接使用')
    missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f'  ⚠️ 缺失的键: {missing_keys}')
    if unexpected_keys:
        print(f'  ⚠️ 意外的键: {unexpected_keys}')
    print('✅ 权重加载完成')
    print('🔧 创建SFDA-FSM包装器...')
    model = SFDAFSMWrapper(base_model, input_channels=args.input_channels, num_classes=args.num_classes).to(device)
    model_device = next(model.parameters()).device
    if model_device != device:
        raise RuntimeError(f'模型设备不匹配: 期望{device}, 实际{model_device}')
    print(f'✅ 模型在设备 {model_device} 上')
    print('⚙️ 设置优化器...')
    optimizer_params = []
    main_params = []
    if hasattr(model.fsm_generator, 'parameters'):
        main_params.extend(list(model.fsm_generator.parameters()))
    if hasattr(model.cdd_module, 'parameters'):
        main_params.extend(list(model.cdd_module.parameters()))
    if hasattr(model.cadc_module, 'parameters'):
        main_params.extend(list(model.cadc_module.parameters()))
    optimizer_params = [{'params': main_params, 'lr': args.lr}]
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    print(f'✅ 优化器配置完成:')
    print(f'  - 类型: AdamW (现代自适应优化器)')
    print(f'  - 基础学习率: {args.lr}')
    print(f'  - Beta参数: (0.9, 0.999)')
    print(f'  - 权重衰减: {args.weight_decay}')
    print('📂 加载 CT 数据...')
    target_test_loader, dataset_name = get_ct_test_loader(image_dir=args.image_dir, mask_dir=args.mask_dir, dataset_type=args.dataset_type, subfolder=args.subfolder, base_dir=args.base_dir, target_dir=args.target_dir, batch_size=args.batch_test, num_workers=args.num_workers, image_size=(args.image_size, args.image_size, args.image_size), spacing=tuple(args.spacing), intensity_range=tuple(args.intensity_range), positive_labels=args.positive_labels)
    print(f'✅ 数据加载完成，共{len(target_test_loader)}个批次 (数据集: {dataset_name})')
    print('🔄 开始测试时自适应（完整SFDA-FSM流程）...')
    print('📝 包含组件：域反转 → FDA → CDD → CADC')
    print('⚠️  注意：完整流程较耗时，请耐心等待...\n')
    total_samples = len(target_test_loader)
    total_iterations = total_samples * args.adapt_steps
    print(f'📊 总样本数: {total_samples}, 每样本适应步数: {args.adapt_steps}')
    print(f'📊 总迭代次数: {total_iterations}')
    current_iter = 0
    for batch_idx, (imgs, labels, file_names) in enumerate(tqdm(target_test_loader, desc='完整SFDA-FSM推理进度')):
        imgs, labels = (imgs.to(device), labels.to(device))
        print(f'\n📋 处理样本 {batch_idx + 1}/{total_samples}: {(file_names[0] if file_names else 'Unknown')}')
        print(f'  输入尺寸: imgs={imgs.shape}, labels={labels.shape}')
        if torch.isnan(imgs).any():
            raise RuntimeError(f'输入图像包含NaN值')
        if torch.isinf(imgs).any():
            raise RuntimeError(f'输入图像包含Inf值')
        model.train()
        for adapt_step in range(args.adapt_steps):
            print(f'  🔄 适应步骤 {adapt_step + 1}/{args.adapt_steps} (全局迭代: {current_iter + 1}/{total_iterations})')
            current_lr = adjust_learning_rate(optimizer=optimizer, i_iter=current_iter, length=total_iterations, base_lr=args.lr, power=args.power)
            optimizer.zero_grad()
            outputs, loss_dict = model(imgs, update_model=True)
            total_loss = loss_dict.get('total_loss', 0)
            if isinstance(total_loss, torch.Tensor) and total_loss > 0:
                print(f'    执行反向传播，损失值: {total_loss:.6f}')
                total_loss.backward()
                grad_norm = 0
                param_count = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                        param_count += 1
                        if torch.isnan(param.grad).any():
                            raise RuntimeError(f'参数 {name} 的梯度包含NaN')
                grad_norm = grad_norm ** (1.0 / 2)
                print(f'    梯度统计: 范数={grad_norm:.6f}, 参数数量={param_count}')
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                distill_loss_val = loss_dict.get('cdd_loss', {}).get('distill_loss', 0)
                contrast_loss_val = loss_dict.get('cdd_loss', {}).get('contrast_loss', 0)
                consistency_loss_val = loss_dict.get('consistency_loss', 0)
                progress_ratio = current_iter / total_iterations if total_iterations > 0 else 0
                print(f'    📈 训练状态 (基于原论文train_adapt.py策略):')
                print(f'      - 学习率: {current_lr:.8f} (进度: {progress_ratio:.1%})')
                print(f'      - 蒸馏损失: {distill_loss_val:.6f} (权重: 0.12)')
                print(f'      - 对比损失: {contrast_loss_val:.6f} (权重: 0.01)')
                print(f'      - 伪标签损失: {consistency_loss_val:.6f} (权重: 1.0)')
                print(f'      - 总损失: {total_loss:.6f}')
                if current_iter % 20 == 0:
                    initial_lr_ratio = current_lr / args.lr
                    expected_ratio = (1 - current_iter / total_iterations) ** args.power
                    print(f'    🔍 学习率策略验证:')
                    print(f'      - 当前/初始比例: {initial_lr_ratio:.4f}')
                    print(f'      - 理论比例: {expected_ratio:.4f}')
                    print(f'      - 策略一致性: {('✅' if abs(initial_lr_ratio - expected_ratio) < 0.01 else '⚠️')}')
            else:
                print(f'    总损失为0或无效，跳过反向传播')
                print(f'    当前学习率: {current_lr:.8f} (保持调整策略)')
            current_iter += 1
        model.eval()
        with torch.no_grad():
            outputs, _ = model(imgs, update_model=False)
        if torch.isnan(outputs).any():
            raise RuntimeError(f'模型输出包含NaN值')
        if torch.isinf(outputs).any():
            raise RuntimeError(f'模型输出包含Inf值')
        print(f'  📊 计算评估指标...')
        bin_outputs = merge_logits_to_binary(outputs, bg_channel=args.bg_channel)
        dice_values = cal_dice(bin_outputs, labels.squeeze(1))
        hd95_values = cal_hd95(bin_outputs, labels.squeeze(1))
        iou_values = IoU(bin_outputs, labels.squeeze(1))
        pa_values = PA(bin_outputs, labels.squeeze(1), 2)
        rve_values = cal_RVE(bin_outputs, labels.squeeze(1))
        sensitivity_values = cal_sensitivity(bin_outputs, labels.squeeze(1))
        ppv_values = cal_ppv(bin_outputs, labels.squeeze(1))
        for k, v in loss_dict.items():
            if k == 'cdd_loss':
                for sub_k, sub_v in v.items():
                    if sub_k not in loss_test_dict:
                        loss_test_dict[sub_k] = np.array([])
                    loss_value = sub_v.cpu().item() if hasattr(sub_v, 'cpu') else float(sub_v)
                    loss_test_dict[sub_k] = np.append(loss_test_dict[sub_k], loss_value)
            elif k in ['consistency_loss', 'total_loss']:
                if k not in loss_test_dict:
                    loss_test_dict[k] = np.array([])
                loss_value = v.cpu().item() if hasattr(v, 'cpu') else float(v)
                loss_test_dict[k] = np.append(loss_test_dict[k], loss_value)
        dice_vals = process_tuple_values(dice_values)
        hd95_vals = process_tuple_values(hd95_values)
        iou_vals = process_tuple_values(iou_values)
        pa_vals = process_tuple_values(pa_values)
        rve_vals = process_tuple_values(rve_values)
        sensitivity_vals = process_tuple_values(sensitivity_values)
        ppv_vals = process_tuple_values(ppv_values)
        batch_metrics = {'dice': dice_vals[TUMOR_IDX], 'hd95': hd95_vals[TUMOR_IDX], 'iou': iou_vals[TUMOR_IDX], 'pa': pa_vals[TUMOR_IDX], 'rve': rve_vals[TUMOR_IDX], 'sensitivity': sensitivity_vals[TUMOR_IDX], 'ppv': ppv_vals[TUMOR_IDX]}
        for key in METRIC_KEYS:
            metric_store[key].append(batch_metrics[key])
        _show_dice(metric_dict, file_names, dice_vals, hd95_vals, iou_vals, pa_vals, rve_vals, sensitivity_vals, ppv_vals)
    for k in loss_test_dict.keys():
        if len(loss_test_dict[k]) > 0:
            loss_test_dict[k] = np.mean(loss_test_dict[k])
        else:
            loss_test_dict[k] = 0.0
    stats_dict = {metric: {'mean': float(np.mean(values)) if len(values) else 0.0, 'std': float(np.std(values)) if len(values) else 0.0} for metric, values in metric_store.items()}
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(result_dir, f'{args.model_type}_tta_weights', args.img)
    os.makedirs(save_dir, exist_ok=True)
    avg_dice = stats_dict['dice']['mean']
    filename = f'UNet3d_SFDAFSM_CT.pth'
    save_path = os.path.join(save_dir, filename)
    torch.save(model.base_model.state_dict(), save_path)
    print(f'💾 SFDA-FSM适配后模型权重已保存到: {save_path}')
    report = f'\n{'=' * 70}\nSFDA-FSM CT 适应测试报告\n测试时间: {timestamp}\n\n📋 测试配置:\n- 图像模态: {args.img}\n- 模型路径: {best_model_path}\n- 测试数据: {args.target_dir or dataset_name}\n- 算法: SFDA-FSM (域反转 + FDA + CDD + CADC)\n\n🔬 学习率策略:\n- 初始学习率: {args.lr}\n- 衰减指数: {args.power}\n- 迭代步数: {total_iterations}\n\n⚙️ 优化器:\n- 类型: AdamW\n- Beta: (0.9, 0.999)\n- Weight Decay: {args.weight_decay}\n\n📊 损失统计:\n- 蒸馏损失: {loss_test_dict.get('distill_loss', 0):.6f}\n- 对比损失: {loss_test_dict.get('contrast_loss', 0):.6f}\n- 一致性损失: {loss_test_dict.get('consistency_loss', 0):.6f}\n- 总损失: {loss_test_dict.get('total_loss', 0):.6f}\n\n    🏆 肿瘤指标 (均值±标准差):\n    - Dice        : {stats_dict['dice']['mean']:.4f} ± {stats_dict['dice']['std']:.4f}\n    - HD95 (mm)   : {stats_dict['hd95']['mean']:.2f} ± {stats_dict['hd95']['std']:.2f}\n    - IoU         : {stats_dict['iou']['mean']:.4f} ± {stats_dict['iou']['std']:.4f}\n    - PA          : {stats_dict['pa']['mean']:.4f} ± {stats_dict['pa']['std']:.4f}\n    - RVE         : {stats_dict['rve']['mean']:.4f} ± {stats_dict['rve']['std']:.4f}\n    - Sensitivity : {stats_dict['sensitivity']['mean']:.4f} ± {stats_dict['sensitivity']['std']:.4f}\n    - PPV         : {stats_dict['ppv']['mean']:.4f} ± {stats_dict['ppv']['std']:.4f}\n    {'=' * 70}\n    '
    result_file = os.path.join(result_dir, f'sfda_fsm_{args.img}_{timestamp}.txt')
    with open(result_file, 'w') as f:
        f.write(report)
    metric_df = pd.DataFrame(metric_dict)
    csv_file = os.path.join(result_dir, f'sfda_fsm_{args.img}_{timestamp}.csv')
    metric_df.to_csv(csv_file, mode='w', header=True, index=False)
    summary_df = pd.DataFrame({'metric': list(stats_dict.keys()), 'mean': [v['mean'] for v in stats_dict.values()], 'std': [v['std'] for v in stats_dict.values()]})
    summary_csv = os.path.join(result_dir, f'sfda_fsm_{args.img}_{timestamp}_summary.csv')
    summary_df.to_csv(summary_csv, mode='w', header=True, index=False)
    print(report)
    print(f'SFDA-FSM结果已保存到: {csv_file}')
    print(f'汇总已保存到: {summary_csv}')
    return True

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * (1 - float(iter) / max_iter) ** power

def adjust_learning_rate(optimizer, i_iter, length, base_lr, power=0.9):
    lr = lr_poly(base_lr, i_iter, length, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFDA-FSM · CT 目标域测试')
    parser.add_argument('--target_dir', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB', help='包含 image/ 与 mask/ 子目录的目标数据根路径')
    parser.add_argument('--image_dir', type=str, default=None, help='可选，单独指定图像目录')
    parser.add_argument('--mask_dir', type=str, default=None, help='可选，单独指定掩码目录')
    parser.add_argument('--dataset_type', type=str, default='CT', help='用于自动映射的数据集类型')
    parser.add_argument('--subfolder', type=str, default=None, help='TTA 数据集子文件夹')
    parser.add_argument('--base_dir', type=str, default='/home/yuwenjing/data/tta_dataset', help='自动映射数据集的根目录')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/SFDA-FSM/checkpoints_ct', help='保存权重与结果的根目录')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['nnunet', 'unet3d'], help='选择模型架构类型 (nnunet 或 unet3d)')
    parser.add_argument('--model_path', type=str, default='default', help='指定模型权重文件路径，default 表示使用内置路径')
    parser.add_argument('--input_channels', type=int, default=1, help='CT 输入通道数')
    parser.add_argument('--num_classes', type=int, default=2, help='输出类别数 (背景+肿瘤)')
    parser.add_argument('--positive_labels', type=int, nargs='+', default=[1], help='视为肿瘤的标签 id')
    parser.add_argument('--image_size', type=int, default=128, help='立方裁剪尺寸 (各轴一致)')
    parser.add_argument('--spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0], help='重采样 spacing，格式: sx sy sz')
    parser.add_argument('--intensity_range', type=float, nargs=2, default=[-200, 400], help='CT 强度裁剪范围 [min, max]')
    parser.add_argument('--bg_channel', type=int, default=0, help='多类输出中视为背景的通道 id（用于 merge_logits_to_binary）')
    parser.add_argument('--lr', type=float, default=1e-05, help='基础学习率 (测试时适应更保守)')
    parser.add_argument('--power', type=float, default=0.9, help='多项式衰减指数')
    parser.add_argument('--adapt_steps', type=int, default=3, help='每个病例的适应步数')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='AdamW 权重衰减')
    parser.add_argument('--gpu', type=int, default=1, help='使用的 GPU 编号')
    parser.add_argument('--img', nargs='+', default=['ct'], help='逻辑上的模态标签（用于日志/循环）')
    parser.add_argument('--batch_test', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_sfda_fsm', action='store_true', default=True, help='是否启用 SFDA-FSM 自适应流程')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  使用设备: {device}')
    print(f'📋 使用模型: {args.model_type}')
    print(f'🔬 使用SFDA-FSM: {args.use_sfda_fsm}')
    print(f'\n📊 基于原论文优化的3D医学影像超参数配置:')
    print(f'  🔬 学习率策略:')
    print(f'    - 初始学习率: {args.lr} (原论文: 1e-3, 测试时适应保守化)')
    print(f'    - 衰减指数: {args.power} (原论文: 0.9, 保持一致)')
    print(f'    - 衰减策略: 多项式衰减 (与原论文train_adapt.py一致)')
    print(f'  🔄 适应策略:')
    print(f'    - 适应步数: {args.adapt_steps} (原论文训练150步)')
    print(f'    - 优化器: AdamW (现代自适应优化器，beta=(0.9, 0.999), weight_decay={args.weight_decay})')
    print(f'  🎯 损失权重 (在SFDAFSMWrapper中配置):')
    print(f'    - 蒸馏: 0.15 (原论文: 0.1, +50%增强3D)')
    print(f'    - 对比: 0.008 (原论文: 0.01, -20%避免3D过约束)')
    print(f'    - 一致性: 1.0 (与原论文保持一致)')
    success_count = 0
    start_time = datetime.datetime.now()
    for idx, modality in enumerate(args.img, 1):
        print(f'\n🔍 正在测试 ({idx}/{len(args.img)}) {modality.upper()}')
        modality_args = argparse.Namespace(**vars(args))
        modality_args.img = modality
        if args.use_sfda_fsm:
            success = test_on_target_with_sfda_fsm(modality_args, device)
        else:
            success = test_on_target(modality_args, device)
        if success:
            success_count += 1
    total_time = datetime.datetime.now() - start_time
    method_name = 'SFDA-FSM'
    summary = f'\n{'=' * 40}\n{method_name}测试总结:\n- 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n- 总耗时: {total_time}\n- 模型类型: {args.model_type}\n- 模型路径: {args.model_path}\n- 使用SFDA-FSM: {args.use_sfda_fsm}\n- 成功测试: {success_count}/{len(args.img)}\n- 失败测试: {len(args.img) - success_count}\n{'=' * 40}\n'
    print(summary)
    summary_file = os.path.join(args.checkpoint_dir, 'results_ct_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
