import argparse
import os
import datetime
import traceback
from typing import Dict, List, Union, Sequence, Callable, Any
import pickle
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from nnunet2d import PlainConvUNet2D
from unet2d import UNet2d
from dataset2D import MedicalImageDataset2D
from train_source2D import calculate_all_metrics

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def safe_value(val):
    if isinstance(val, torch.Tensor):
        return val.item()
    return float(val)

def robust_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        print('[Warning] torch.load failed with weights_only=True – retrying with weights_only=False (trusted checkpoint).')
        return torch.load(path, map_location=map_location, weights_only=False)

def extract_state_dict(obj):
    if isinstance(obj, dict):
        if 'state_dict' in obj:
            return obj['state_dict']
        if 'model_state_dict' in obj:
            return obj['model_state_dict']
    return obj

def testfit_inference_2d(inputs: torch.Tensor, predictor: Callable[..., torch.Tensor], ref_model: Callable[..., torch.Tensor], optimizer: Any, loss_function: Any, patch_size: int=256, overlap: float=0.25, device: Union[torch.device, str, None]=None) -> torch.Tensor:
    if device is None:
        device = inputs.device
    B, C, H, W = inputs.shape
    if H <= patch_size and W <= patch_size:
        return testfit_single_patch_2d(inputs, predictor, ref_model, optimizer, loss_function)
    stride = int(patch_size * (1 - overlap))
    h_patches = max(1, (H - patch_size) // stride + 1)
    w_patches = max(1, (W - patch_size) // stride + 1)
    output = torch.zeros_like(inputs)
    count_map = torch.zeros((B, 1, H, W), device=device)
    for h_idx in range(h_patches):
        for w_idx in range(w_patches):
            h_start = min(h_idx * stride, H - patch_size)
            w_start = min(w_idx * stride, W - patch_size)
            h_end = h_start + patch_size
            w_end = w_start + patch_size
            patch = inputs[:, :, h_start:h_end, w_start:w_end]
            patch_output = testfit_single_patch_2d(patch, predictor, ref_model, optimizer, loss_function)
            output[:, :, h_start:h_end, w_start:w_end] += patch_output
            count_map[:, :, h_start:h_end, w_start:w_end] += 1
    output = output / count_map.clamp(min=1)
    return output

def testfit_single_patch_2d(patch: torch.Tensor, predictor: Callable[..., torch.Tensor], ref_model: Callable[..., torch.Tensor], optimizer: Any, loss_function: Any) -> torch.Tensor:
    optimizer.zero_grad()
    seg_prob1 = predictor(patch)
    with torch.no_grad():
        seg_prob2 = ref_model(patch)
        seg_prob2 = seg_prob2.detach()
    high = -10000
    low = 10000
    high_alpha = 0
    low_alpha = 0
    for alpha in range(101):
        alpha_val = alpha / 100.0
        temp = alpha_val * seg_prob1.detach() + (1 - alpha_val) * seg_prob2
        score = softmax_entropy(temp).mean()
        if score >= high:
            high = score
            high_alpha = alpha
        if score <= low:
            low = score
            low_alpha = alpha
    low_alpha_val = low_alpha / 100.0
    high_alpha_val = high_alpha / 100.0
    seg_prob_out = low_alpha_val * seg_prob1 + (1 - low_alpha_val) * seg_prob2
    labels = high_alpha_val * seg_prob1 + (1 - high_alpha_val) * seg_prob2
    labels = torch.sigmoid(labels)
    weight1 = labels.clone()
    weight1 = 2 * torch.abs(0.5 - weight1)
    weight1 = weight1.detach()
    weight2 = seg_prob1.clone()
    weight2 = torch.sigmoid(weight2)
    weight2 = 2 * torch.abs(0.5 - weight2)
    weight2 = 1 - weight2
    weight2 = weight2.detach()
    labels[labels > 0.9] = 1.0
    labels[labels <= 0.9] = 0.0
    loss = loss_function(seg_prob1, labels.detach())
    loss = torch.mean(weight1 * weight2 * loss)
    loss.backward()
    optimizer.step()
    return seg_prob_out.detach()

def build_test_loader(target_dir: str, batch_size: int, num_workers: int, image_size: int) -> DataLoader:
    image_dir = os.path.join(target_dir, 'image')
    mask_dir = os.path.join(target_dir, 'mask')
    if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
        raise FileNotFoundError(f'Expect image/ & mask/ inside {target_dir}')
    dataset = MedicalImageDataset2D(image_dir=image_dir, mask_dir=mask_dir, phase='test', image_size=(image_size, image_size), normalize=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

def test_on_target(args, device):
    print('\n' + '=' * 40)
    print(f'🧪 开始在目标域上测试数据集: {os.path.basename(args.target_dir)}')
    print('=' * 40 + '\n')
    result_dir = os.path.join(args.checkpoint_dir, 'tta2d_results')
    weights_dir = os.path.join(result_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    if args.model_type == 'nnunet2d':
        model = PlainConvUNet2D(input_channels=1, n_stages=5, features_per_stage=(32, 64, 128, 256, 512), kernel_sizes=3, strides=(1, 2, 2, 2, 2), n_conv_per_stage=2, num_classes=args.num_classes, n_conv_per_stage_decoder=2, deep_supervision=False).to(device)
        default_ckpt = os.path.join(args.checkpoint_dir, 'nnunet2d_best.pth')
        print('已选择 PlainConvUNet2D 架构')
    else:
        model = UNet2d(in_channels=1, n_classes=args.num_classes).to(device)
        default_ckpt = os.path.join(args.checkpoint_dir, 'unet2d_best.pth')
        print('已选择 UNet2d 架构')
    ckpt_path = default_ckpt if args.model_path == 'default' else args.model_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'未找到预训练权重: {ckpt_path}')
    print(f'加载模型权重: {ckpt_path}')
    checkpoint_obj = robust_torch_load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(checkpoint_obj)
    model.load_state_dict(state_dict, strict=True)
    if args.model_type == 'nnunet2d':
        ref_model = PlainConvUNet2D(input_channels=1, n_stages=5, features_per_stage=(32, 64, 128, 256, 512), kernel_sizes=3, strides=(1, 2, 2, 2, 2), n_conv_per_stage=2, num_classes=args.num_classes, n_conv_per_stage_decoder=2, deep_supervision=False).to(device)
    else:
        ref_model = UNet2d(in_channels=1, n_classes=args.num_classes).to(device)
    ref_model.load_state_dict(state_dict, strict=True)
    ref_model.eval()
    model.train()
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    test_loader = build_test_loader(target_dir=args.target_dir, batch_size=args.batch_test, num_workers=args.num_workers, image_size=args.image_size)
    metric_lists: Dict[str, List[float]] = {k: [] for k in ['dice', 'iou', 'sensitivity', 'ppv', 'hd95']}
    print('开始TestFit测试时适应推理...')
    for imgs, labels, _ in tqdm(test_loader, desc='TestFit推理进度'):
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = testfit_inference_2d(inputs=imgs, predictor=model, ref_model=ref_model, optimizer=optimizer, loss_function=loss_function, patch_size=args.image_size, overlap=0.25, device=device)
        for i in range(imgs.shape[0]):
            m = calculate_all_metrics(outputs[i:i + 1], labels[i:i + 1])
            for k in metric_lists:
                metric_lists[k].append(safe_value(m[k]))
    metric_mean = {k: float(np.mean(v)) for k, v in metric_lists.items()}
    metric_std = {k: float(np.std(v)) for k, v in metric_lists.items()}
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    adapted_path = os.path.join(weights_dir, f'{model_tag}_testfit2d_adapted_{timestamp}.pth')
    torch.save(model.state_dict(), adapted_path)
    print(f'✅ 已保存TestFit适应后的模型权重: {adapted_path}')
    report_lines = ['=' * 40, f'TestFit 2D 测试时适应结果', '=' * 40, f'测试时间: {timestamp}', f'目标数据集: {args.target_dir}', f'模型架构: {args.model_type}', f'原始权重: {ckpt_path}', f'TestFit适应权重: {adapted_path}', f'学习率: {args.lr}', '', 'TestFit方法说明:', '- 基于熵优化的测试时适应', '- 使用原始模型作为参考', '- 每个patch进行独立适应', '', '性能指标 (均值 ± 标准差):']
    for k in metric_lists:
        report_lines.append(f'{k.upper()}: {metric_mean[k]:.4f} ± {metric_std[k]:.4f}')
    report_lines.append('=' * 40)
    report = '\n'.join(report_lines)
    result_file = os.path.join(result_dir, f'{model_tag}_testfit2d_{timestamp}.txt')
    with open(result_file, 'w') as f:
        f.write(report)
    print(report)
    return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D TestFit Test-Time Adaptation Evaluation Script')
    parser.add_argument('--target_dir', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-2DPATH/Glas_processed', help='包含 image/ 和 mask/ 的目标域文件夹')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='保存 / 查找权重的目录')
    parser.add_argument('--model_path', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoint_PATH/unet2d_best_PATH.pth', help='自定义权重路径，`default` 表示自动选择')
    parser.add_argument('--model_type', type=str, default='unet2d', choices=['unet2d', 'nnunet2d'], help='模型架构类型')
    parser.add_argument('--num_classes', type=int, default=2, help='输出类别数 (含背景)')
    parser.add_argument('--lr', type=float, default=0.001, help='TestFit适应阶段学习率')
    parser.add_argument('--batch_test', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256, help='测试时图像缩放大小')
    parser.add_argument('--gpu', type=int, default=2, help='GPU 编号 (-1 表示 CPU)')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f'🖥️  使用设备: {device}')
    try:
        test_on_target(args, device)
    except Exception as e:
        err_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        print('🔥 运行失败:', str(e))
        traceback.print_exc()
        err_dir = os.path.join(args.checkpoint_dir, 'tta2d_results')
        os.makedirs(err_dir, exist_ok=True)
        with open(os.path.join(err_dir, 'testfit2d_errors.log'), 'a') as f:
            f.write(f'[{err_time}] {traceback.format_exc()}\n')
