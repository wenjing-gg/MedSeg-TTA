import argparse
import os
import datetime
import traceback
import torch
import torch.optim as optim
from tqdm import tqdm
from nnunet import PlainConvUNet
from utils_brats_all import get_data_loader
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.inferers import sliding_window_inference
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, ensure_tuple, fall_back_tuple, look_up_option
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from monai.transforms import Resize
from monai.data.meta_tensor import MetaTensor
from monai.utils import convert_data_type, convert_to_dst_type

def safe_value(val):
    if isinstance(val, torch.Tensor):
        return val.item()
    return val

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
    softmax = nn.Softmax(dim=1)
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
    slice_range_iter = tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size)
    for slice_g in slice_range_iter:
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [[slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win]) for idx in slice_range]
        window_data = torch.cat([convert_data_type(inputs[win_slice], torch.Tensor)[0] for win_slice in unravel_slice]).to(sw_device)
        optimizer.zero_grad()
        seg_prob1 = predictor(window_data, *args, **kwargs)
        with torch.no_grad():
            seg_prob2 = ref_model(window_data, *args, **kwargs)
            seg_prob2 = seg_prob2.detach()
        high = -10000
        low = 10000
        high_alpha = 0
        low_alpha = 0
        for alpha in range(101):
            temp = alpha / 100 * seg_prob1.detach() + (1 - alpha / 100) * seg_prob2
            score = softmax_entropy(temp).mean(0)
            score = torch.mean(score)
            if score >= high:
                high = score
                high_alpha = alpha
            if score <= low:
                low = score
                low_alpha = alpha
        seg_prob_out = low_alpha / 100 * seg_prob1 + (1 - low_alpha / 100) * seg_prob2
        labels = high_alpha / 100 * seg_prob1 + (1 - high_alpha / 100) * seg_prob2
        labels = torch.sigmoid(labels)
        weight1 = labels.clone()
        weight1 = 2 * torch.abs(0.5 - weight1)
        weight1 = weight1.detach()
        weight2 = seg_prob1.clone()
        weight2 = torch.sigmoid(weight2)
        weight2 = 2 * torch.abs(0.5 - weight2)
        weight2 = 1 - weight2
        weight2 = weight2.detach()
        labels[torch.where(labels > 0.95)] = 1.0
        labels[torch.where(labels <= 0.95)] = 0.0
        loss = loss_function(seg_prob1, labels.detach())
        loss = torch.mean(weight1 * weight2 * loss)
        loss.backward()
        optimizer.step()
        seg_prob = seg_prob_out.to(device)
        zoom_scale = []
        for axis, (img_s_i, out_w_i, in_w_i) in enumerate(zip(image_size, seg_prob.shape[2:], window_data.shape[2:])):
            zoom_scale.append(out_w_i / float(in_w_i))
        if not _initialized:
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + [int(image_size_d * zoom_scale_d) for image_size_d, zoom_scale_d in zip(image_size, zoom_scale)]
            output_image_list.append(torch.zeros(output_shape, dtype=compute_dtype, device=device))
            count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
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
    if torch.isnan(output_image).any() or torch.isinf(output_image).any():
        print('警告: 滑动窗口推理结果包含NaN或Inf。')
    zoom_scale = [seg_prob_map_shape_d / roi_size_d for seg_prob_map_shape_d, roi_size_d in zip(output_image.shape[2:], roi_size)]
    final_slicing = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        slice_dim = slice(int(round(slice_dim.start * zoom_scale[num_spatial_dims - sp - 1])), int(round(slice_dim.stop * zoom_scale[num_spatial_dims - sp - 1])))
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    output_image = output_image[final_slicing]
    if isinstance(inputs, MetaTensor):
        output_image = convert_to_dst_type(output_image, inputs, device=device)[0]
    return output_image

def get_model(model_type, device):
    if model_type.lower() == 'nnunet':
        print(f'📋 加载 nnUNet 模型架构')
        model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
        return model
    elif model_type.lower() == 'unet3d':
        print(f'📋 加载 UNet3D 模型架构')
        from unet3d import UNet3d
        model = UNet3d().to(device)
        return model
    else:
        raise ValueError(f"不支持的模型类型: {model_type}。请选择 'nnunet' 或 'unet3d'")

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 开始在目标数据集上测试模态: {args.img.upper()} (使用测试时域自适应)')
    print(f'🧩 使用模型类型: {args.model_type}')
    print(f'{'=' * 40}\n')
    try:
        result_dir = '/home/yuwenjing/DeepLearning_ywj/tta/Testfit/checkpoints/tta_results'
        os.makedirs(result_dir, exist_ok=True)
        model = get_model(args.model_type, device)
        ref_model = get_model(args.model_type, device)
        if args.model_path and args.model_path != 'default':
            best_model_path = args.model_path
        elif args.model_type.lower() == 'nnunet':
            best_model_path = '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth'
        elif args.model_type.lower() == 'unet3d':
            best_model_path = '/home/yuwenjing/DeepLearning_ywj/tta/unet3d_best.pth'
        else:
            raise ValueError(f'不支持的模型类型: {args.model_type}')
        print(f'📦 加载模型权重: {best_model_path}')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f'未找到预训练权重: {best_model_path}')
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        ref_model.load_state_dict(state_dict)
        model.train()
        ref_model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.tta_lr)
        loss_function = nn.BCEWithLogitsLoss(reduction='none')
        _, target_test_loader = get_data_loader(source_root=args.source_root, target_root=args.target_root, batch_train=args.batch_test, batch_test=args.batch_test, nw=args.num_workers, img=args.img, mode='source_to_target')
        dice_samples = [[] for _ in range(3)]
        hd95_samples = [[] for _ in range(3)]
        iou_samples = [[] for _ in range(3)]
        pa_samples = [[] for _ in range(3)]
        rve_samples = [[] for _ in range(3)]
        sensitivity_samples = [[] for _ in range(3)]
        ppv_samples = [[] for _ in range(3)]

        def inference(input_tensor):
            return sliding_window_inference_testfit(inputs=input_tensor, roi_size=(128, 128, 128), sw_batch_size=1, predictor=model, ref_model=ref_model, optimizer=optimizer, loss_function=loss_function, overlap=0.5, progress=args.show_progress)
        for imgs, labels, *_ in tqdm(target_test_loader, desc='测试时域自适应推理中'):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = inference(imgs)
            dice_values = cal_dice(outputs, labels.squeeze(1))
            hd95_values = cal_hd95(outputs, labels.squeeze(1))
            IoU_values = IoU(outputs, labels.squeeze(1))
            pa_values = PA(outputs, labels.squeeze(1), 4)
            RVE_values = cal_RVE(outputs, labels.squeeze(1))
            sensitivity_values = cal_sensitivity(outputs, labels.squeeze(1))
            ppv_values = cal_ppv(outputs, labels.squeeze(1))
            for i in range(3):
                dice_samples[i].append(safe_value(dice_values[i]))
                hd95_samples[i].append(safe_value(hd95_values[i]))
                iou_samples[i].append(safe_value(IoU_values[i]))
                pa_samples[i].append(safe_value(pa_values[i]))
                rve_samples[i].append(safe_value(RVE_values[i]))
                sensitivity_samples[i].append(safe_value(sensitivity_values[i]))
                ppv_samples[i].append(safe_value(ppv_values[i]))
        dice_samples = [np.array(samples) for samples in dice_samples]
        hd95_samples = [np.array(samples) for samples in hd95_samples]
        iou_samples = [np.array(samples) for samples in iou_samples]
        pa_samples = [np.array(samples) for samples in pa_samples]
        rve_samples = [np.array(samples) for samples in rve_samples]
        sensitivity_samples = [np.array(samples) for samples in sensitivity_samples]
        ppv_samples = [np.array(samples) for samples in ppv_samples]
        dice_means = [np.mean(samples) for samples in dice_samples]
        dice_stds = [np.std(samples) for samples in dice_samples]
        hd95_means = [np.mean(samples) for samples in hd95_samples]
        hd95_stds = [np.std(samples) for samples in hd95_samples]
        iou_means = [np.mean(samples) for samples in iou_samples]
        iou_stds = [np.std(samples) for samples in iou_samples]
        pa_means = [np.mean(samples) for samples in pa_samples]
        pa_stds = [np.std(samples) for samples in pa_samples]
        rve_means = [np.mean(samples) for samples in rve_samples]
        rve_stds = [np.std(samples) for samples in rve_samples]
        sensitivity_means = [np.mean(samples) for samples in sensitivity_samples]
        sensitivity_stds = [np.std(samples) for samples in sensitivity_samples]
        ppv_means = [np.mean(samples) for samples in ppv_samples]
        ppv_stds = [np.std(samples) for samples in ppv_samples]
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report = f'\n{'=' * 40}\n测试时间: {timestamp}\n测试配置:\n- 模型类型: {args.model_type}\n- 图像模态: {args.img}\n- 模型路径: {best_model_path}\n- 测试数据: {args.target_root}\n- 使用测试时域自适应: 是\n- 测试时学习率: {args.tta_lr}\n- 测试样本数: {len(dice_samples[0])}\n\n性能指标 (均值±标准差):\nDice:\n  ET: {dice_means[0]:.4f}±{dice_stds[0]:.4f}\n  TC: {dice_means[1]:.4f}±{dice_stds[1]:.4f}\n  WT: {dice_means[2]:.4f}±{dice_stds[2]:.4f}\nHD95(mm):\n  ET: {hd95_means[0]:.2f}±{hd95_stds[0]:.2f}\n  TC: {hd95_means[1]:.2f}±{hd95_stds[1]:.2f}\n  WT: {hd95_means[2]:.2f}±{hd95_stds[2]:.2f}\nIoU:\n  ET: {iou_means[0]:.4f}±{iou_stds[0]:.4f}\n  TC: {iou_means[1]:.4f}±{iou_stds[1]:.4f}\n  WT: {iou_means[2]:.4f}±{iou_stds[2]:.4f}\nPA:\n  ET: {pa_means[0]:.4f}±{pa_stds[0]:.4f}\n  TC: {pa_means[1]:.4f}±{pa_stds[1]:.4f}\n  WT: {pa_means[2]:.4f}±{pa_stds[2]:.4f}\nRVE:\n  ET: {rve_means[0]:.4f}±{rve_stds[0]:.4f}\n  TC: {rve_means[1]:.4f}±{rve_stds[1]:.4f} \n  WT: {rve_means[2]:.4f}±{rve_stds[2]:.4f}\nSensitivity:\n  ET: {sensitivity_means[0]:.4f}±{sensitivity_stds[0]:.4f}\n  TC: {sensitivity_means[1]:.4f}±{sensitivity_stds[1]:.4f}\n  WT: {sensitivity_means[2]:.4f}±{sensitivity_stds[2]:.4f}\nPPV:\n  ET: {ppv_means[0]:.4f}±{ppv_stds[0]:.4f}\n  TC: {ppv_means[1]:.4f}±{ppv_stds[1]:.4f}\n  WT: {ppv_means[2]:.4f}±{ppv_stds[2]:.4f}\n{'=' * 40}\n'
        result_file = os.path.join(result_dir, f'{args.model_type}_tta_test_{args.img}_{timestamp}.txt')
        with open(result_file, 'w') as f:
            f.write(report)
        print(report)
        return True
    except Exception as e:
        error_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        error_msg = f'\n🔥 测试时域自适应失败: {args.model_type}_{args.img}\n错误信息: {str(e)}\n追踪信息:\n{traceback.format_exc()}'
        print(error_msg)
        error_log = os.path.join(result_dir, 'tta_test_errors.log')
        with open(error_log, 'a') as f:
            f.write(f'[{error_timestamp}] {error_msg}\n')
        return False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试时域自适应测试脚本')
    parser.add_argument('--source_root', type=str, default='/home/yuwenjing/data/BraTS2024')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/BraTS-SSA', help='目标数据集根目录路径')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints', help='包含预训练权重的检查点目录')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['nnunet', 'unet3d'], help='选择模型架构类型 (nnunet 或 unet3d)')
    parser.add_argument('--model_path', type=str, default='default', help='指定模型权重文件的完整路径，使用default则根据model_type自动选择')
    parser.add_argument('--lr', type=float, default=1e-05, help='常规学习率')
    parser.add_argument('--tta_lr', type=float, default=1e-05, help='测试时域自适应学习率')
    parser.add_argument('--gpu', type=int, default=3, help='使用GPU编号')
    parser.add_argument('--img', default=['all'], help='测试模态')
    parser.add_argument('--batch_test', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--show_progress', action='store_true', help='显示滑动窗口推理进度')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  使用设备: {device}')
    print(f'📋 使用模型: {args.model_type}')
    success_count = 0
    start_time = datetime.datetime.now()
    for idx, modality in enumerate(args.img, 1):
        print(f'\n🔍 正在测试 ({idx}/{len(args.img)}) {modality.upper()}')
        modality_args = argparse.Namespace(**vars(args))
        modality_args.img = modality
        if test_on_target(modality_args, device):
            success_count += 1
    total_time = datetime.datetime.now() - start_time
    summary = f'\n{'=' * 40}\n测试总结:\n- 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n- 总耗时: {total_time}\n- 模型类型: {args.model_type}\n- 成功测试: {success_count}/{len(args.img)}\n- 失败测试: {len(args.img) - success_count}\n- 使用测试时域自适应: 是\n{'=' * 40}\n'
    print(summary)
    result_dir = '/home/yuwenjing/DeepLearning_ywj/tta/Testfit/checkpoints/tta_results'
    summary_file = os.path.join(result_dir, f'{args.model_type}_tta_test_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
