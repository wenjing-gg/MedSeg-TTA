import argparse
import os
import glob
import datetime
import traceback
from typing import Tuple, Optional, Dict, List, Sequence, Union, Any, Callable
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, ensure_tuple, fall_back_tuple, look_up_option
from monai.transforms import Resize
from monai.data.meta_tensor import MetaTensor
from monai.utils import convert_data_type, convert_to_dst_type
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, Resized, NormalizeIntensityd, ToTensord
from nnunet import PlainConvUNet
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

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def _get_scan_interval(image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float) -> Tuple[int, ...]:
    if len(image_size) != num_spatial_dims:
        raise ValueError('image coord different from spatial dims.')
    if len(roi_size) != num_spatial_dims:
        raise ValueError('roi coord different from spatial dims.')
    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)

def sliding_window_inference_testfit(inputs: torch.Tensor, roi_size: Union[Sequence[int], int], sw_batch_size: int, predictor: Callable[..., torch.Tensor], ref_model: Callable[..., torch.Tensor], optimizer: Any, loss_function: Any, overlap: float=0.5, mode: Union[BlendMode, str]=BlendMode.CONSTANT, sigma_scale: Union[Sequence[float], float]=0.125, padding_mode: Union[PytorchPadMode, str]=PytorchPadMode.CONSTANT, cval: float=0.0, sw_device: Union[torch.device, str, None]=None, device: Union[torch.device, str, None]=None, progress: bool=False, *args: Any, **kwargs: Any) -> torch.Tensor:
    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError('overlap must be >= 0 and < 1.')
    batch_size, _, *image_size_ = inputs.shape
    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device
    roi_size = fall_back_tuple(roi_size, image_size_)
    image_size = tuple((max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)
    total_slices = num_win * batch_size
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    importance_map_ = compute_importance_map(valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device)
    importance_map_ = convert_data_type(importance_map_, torch.Tensor, device, compute_dtype)[0]
    min_non_zero = max(importance_map_[importance_map_ != 0].min().item(), 0.001)
    importance_map_ = torch.clamp(importance_map_.to(torch.float32), min=min_non_zero).to(compute_dtype)
    output_image_list, count_map_list = ([], [])
    _initialized = False
    slice_iter = tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size)
    for slice_g in slice_iter:
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [[slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win]) for idx in slice_range]
        window_data = torch.cat([convert_data_type(inputs[win_slice], torch.Tensor)[0] for win_slice in unravel_slice]).to(sw_device)
        optimizer.zero_grad()
        seg_prob1 = predictor(window_data, *args, **kwargs)
        with torch.no_grad():
            seg_prob2 = ref_model(window_data, *args, **kwargs).detach()
        high, low = (-1000000000.0, 1000000000.0)
        high_alpha, low_alpha = (0, 0)
        for alpha in range(101):
            mix = alpha / 100 * seg_prob1.detach() + (1 - alpha / 100) * seg_prob2
            score = softmax_entropy(mix).mean()
            if score >= high:
                high, high_alpha = (score, alpha)
            if score <= low:
                low, low_alpha = (score, alpha)
        seg_prob_out = low_alpha / 100 * seg_prob1 + (1 - low_alpha / 100) * seg_prob2
        labels = high_alpha / 100 * seg_prob1 + (1 - high_alpha / 100) * seg_prob2
        labels = torch.sigmoid(labels)
        weight1 = 2 * torch.abs(0.5 - labels).detach()
        weight2 = torch.sigmoid(seg_prob1)
        weight2 = 1 - 2 * torch.abs(0.5 - weight2)
        weight2 = weight2.detach()
        labels[labels > 0.95] = 1.0
        labels[labels <= 0.95] = 0.0
        loss = loss_function(seg_prob1, labels.detach())
        loss = torch.mean(weight1 * weight2 * loss)
        loss.backward()
        optimizer.step()
        seg_prob = seg_prob_out.to(device)
        zoom_scale = []
        for axis, (img_s_i, out_w_i, in_w_i) in enumerate(zip(image_size, seg_prob.shape[2:], window_data.shape[2:])):
            zoom_scale.append(out_w_i / float(in_w_i))
        if not _initialized:
            out_ch = seg_prob.shape[1]
            out_shape = [batch_size, out_ch] + [int(image_size_d * z) for image_size_d, z in zip(image_size, zoom_scale)]
            output_image_list.append(torch.zeros(out_shape, dtype=compute_dtype, device=device))
            count_map_list.append(torch.zeros([1, 1] + out_shape[2:], dtype=compute_dtype, device=device))
            _initialized = True
        resizer = Resize(spatial_size=seg_prob.shape[2:], mode='nearest', anti_aliasing=False)
        for idx, original_idx in zip(slice_range, unravel_slice):
            original_idx_zoom = list(original_idx)
            for axis in range(2, len(original_idx_zoom)):
                zoomed_start = original_idx[axis].start * zoom_scale[axis - 2]
                zoomed_end = original_idx[axis].stop * zoom_scale[axis - 2]
                original_idx_zoom[axis] = slice(int(zoomed_start), int(zoomed_end), None)
            importance_map_zoom = resizer(importance_map_.unsqueeze(0))[0].to(compute_dtype)
            output_image_list[0][original_idx_zoom] += importance_map_zoom * seg_prob[idx - slice_g]
            count_map_list[0][original_idx_zoom] += importance_map_zoom.unsqueeze(0).unsqueeze(0).expand(count_map_list[0][original_idx_zoom].shape)
    output_image = (output_image_list[0] / count_map_list[0]).to(compute_dtype)
    zoom_scale = [out_dim / roi_dim for out_dim, roi_dim in zip(output_image.shape[2:], roi_size)]
    final_slicing = []
    num_spatial_dims = len(image_size_)
    for sp in range(num_spatial_dims):
        start = pad_size[sp * 2]
        stop = image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2]
        slice_dim = slice(int(round(start * zoom_scale[num_spatial_dims - sp - 1])), int(round(stop * zoom_scale[num_spatial_dims - sp - 1])))
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    output_image = output_image[final_slicing]
    if isinstance(inputs, MetaTensor):
        output_image = convert_to_dst_type(output_image, inputs, device=device)[0]
    return output_image

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 CT 模态 TestFit 测试 | 模型: {args.model_type}')
    print(f'{'=' * 40}\n')
    try:
        model = get_model(args.model_type, device)
        ref_model = get_model(args.model_type, device)
        model_path = args.model_path
        if model_path == 'default':
            model_path = '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth' if args.model_type.lower() == 'nnunet' else '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth'
        print(f'📦 加载模型权重: {model_path}')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'未找到模型权重文件: {model_path}')
        state = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        ref_model.load_state_dict(state)
        model.train()
        ref_model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.tta_lr, momentum=0.9)
        loss_function = nn.BCEWithLogitsLoss(reduction='none')
        test_loader, _ = get_ct_test_loader(image_dir=args.image_dir, mask_dir=args.mask_dir, target_dir=args.target_root, dataset_type='CT' if args.target_root is None else None, batch_size=args.batch_test, num_workers=args.num_workers, image_size=(args.image_size, args.image_size, args.image_size), spacing=tuple(args.spacing), intensity_range=tuple(args.intensity_range), positive_labels=args.positive_labels)
        K = 2
        tumor_idx = 1
        rows = []
        all_dice = [[] for _ in range(K)]
        all_hd95 = [[] for _ in range(K)]
        all_IoU = [[] for _ in range(K)]
        all_pa = [[] for _ in range(K)]
        all_RVE = [[] for _ in range(K)]
        all_sen = [[] for _ in range(K)]
        all_ppv = [[] for _ in range(K)]
        for batch in tqdm(test_loader, desc='TestFit 推理+自适应'):
            if len(batch) == 3:
                imgs, labels, filenames = batch
            else:
                imgs, labels = batch[:2]
                filenames = [f'case_{i}' for i in range(imgs.size(0))]
            imgs, labels = (imgs.to(device), labels.to(device))
            logits = sliding_window_inference_testfit(inputs=imgs, roi_size=tuple(args.roi_size), sw_batch_size=1, predictor=model, ref_model=ref_model, optimizer=optimizer, loss_function=loss_function, overlap=args.overlap, progress=args.show_progress)
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
                all_sen[i].append(safe_value(sen_vals[i]))
                all_ppv[i].append(safe_value(ppv_vals[i]))
            for j, fname in enumerate(filenames):
                rows.append({'file_id': fname, 'dice_tumor': float(dice_vals[tumor_idx]), 'hd95_tumor': float(hd95_vals[tumor_idx]), 'iou_tumor': float(iou_vals[tumor_idx]), 'pa_tumor': float(pa_vals[tumor_idx]), 'rve_tumor': float(rve_vals[tumor_idx]), 'sen_tumor': float(sen_vals[tumor_idx]), 'ppv_tumor': float(ppv_vals[tumor_idx])})

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
        mean_sen = [ms(all_sen[i])[0] for i in range(K)]
        std_sen = [ms(all_sen[i])[1] for i in range(K)]
        mean_ppv = [ms(all_ppv[i])[0] for i in range(K)]
        std_ppv = [ms(all_ppv[i])[1] for i in range(K)]
        t = 1
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result = f'测试配置:\n  模型类型: {args.model_type}\n  模型路径: {model_path}\n  TestFit 学习率: {args.tta_lr}\n  ROI 大小: {tuple(args.roi_size)}, overlap={args.overlap}\n  图像尺寸: {args.image_size}^3, spacing={tuple(args.spacing)}\n  CT 强度范围: {tuple(args.intensity_range)}\n\n[Tumor] 指标均值 ± 标准差：\n  Dice        : {mean_dice[t]:.4f} ± {std_dice[t]:.4f}\n  HD95 (mm)   : {mean_hd95[t]:.2f} ± {std_hd95[t]:.2f}\n  IoU         : {mean_IoU[t]:.4f} ± {std_IoU[t]:.4f}\n  PA          : {mean_pa[t]:.4f} ± {std_pa[t]:.4f}\n  RVE         : {mean_RVE[t]:.4f} ± {std_RVE[t]:.4f}\n  Sensitivity : {mean_sen[t]:.4f} ± {std_sen[t]:.4f}\n  PPV         : {mean_ppv[t]:.4f} ± {std_ppv[t]:.4f}\n'
        print('\n' + result)
        result_dir = os.path.join(args.checkpoint_dir, f'{args.model_type}_CT_TestFit')
        os.makedirs(result_dir, exist_ok=True)
        out_txt = os.path.join(result_dir, f'testfit_results_{ts}.txt')
        with open(out_txt, 'w') as f:
            f.write(result)
        df_detail = pd.DataFrame(rows)
        out_csv = os.path.join(result_dir, f'testfit_sample_metrics_{ts}.csv')
        df_detail.to_csv(out_csv, index=False)
        summary = {'metric': ['dice', 'hd95', 'iou', 'pa', 'rve', 'sensitivity', 'ppv'], 'mean': [mean_dice[t], mean_hd95[t], mean_IoU[t], mean_pa[t], mean_RVE[t], mean_sen[t], mean_ppv[t]], 'std': [std_dice[t], std_hd95[t], std_IoU[t], std_pa[t], std_RVE[t], std_sen[t], std_ppv[t]]}
        df_sum = pd.DataFrame(summary)
        out_sum = os.path.join(result_dir, f'testfit_summary_{ts}.csv')
        df_sum.to_csv(out_sum, index=False)
        print(f'✅ 结果已保存: {out_txt}')
        print(f'📄 样本级指标: {out_csv}')
        print(f'📊 摘要统计  : {out_sum}')
        return True
    except Exception as e:
        print(f'❌ 测试失败: {str(e)}')
        traceback.print_exc()
        return False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CT 测试时域自适应（TestFit：滑动窗口+窗口级自适应）')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB', help='目标数据集根目录（包含 image/ 与 mask/ 子目录）')
    parser.add_argument('--image_dir', type=str, default=None, help='可显式指定图像目录')
    parser.add_argument('--mask_dir', type=str, default=None, help='可显式指定标注目录')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/TestFit/checkpoints', help='保存测试结果的根目录')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['nnunet', 'unet3d'])
    parser.add_argument('--model_path', type=str, default='default', help='模型权重路径；"default" 则按模型类型选择内置默认路径')
    parser.add_argument('--tta_lr', type=float, default=0.01, help='TestFit 学习率（窗口级适应）')
    parser.add_argument('--roi_size', type=int, nargs=3, default=(128, 128, 128), help='滑动窗口 ROI 尺寸')
    parser.add_argument('--overlap', type=float, default=0.5, help='滑窗重叠率 0~1')
    parser.add_argument('--show_progress', action='store_true', help='显示滑窗进度条')
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
