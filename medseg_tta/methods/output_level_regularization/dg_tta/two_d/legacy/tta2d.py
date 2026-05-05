import argparse
import os
import datetime
import traceback
from typing import Dict, List, Tuple
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from unet2d import UNet2d
from dataset2D import MedicalImageDataset2D
from train_source2D import calculate_all_metrics
from augmentation_utils_2d import get_2d_disp_field, get_2d_rand_affine, create_2d_consistency_augmentation, apply_2d_inverse_transform

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

def freeze_bn_statistics(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            module.track_running_stats = False
            if hasattr(module, 'running_mean'):
                module.running_mean.requires_grad = False
            if hasattr(module, 'running_var'):
                module.running_var.requires_grad = False
    return model

def freeze_non_bn_parameters(model: nn.Module) -> nn.Module:
    for name, param in model.named_parameters():
        is_bn_param = name.endswith('.1.weight') or name.endswith('.1.bias') or name.endswith('.5.weight') or name.endswith('.5.bias')
        if not is_bn_param:
            param.requires_grad = False
    return model

@torch.jit.script
def soft_dice_loss_2d(smp_a: torch.Tensor, smp_b: torch.Tensor) -> torch.Tensor:
    B, C, H, W = smp_a.shape
    intersection = (smp_a * smp_b).sum((-2, -1))
    union = smp_a.sum((-2, -1)) + smp_b.sum((-2, -1))
    dice = 2.0 * intersection / (union + 1e-08)
    return dice

def compute_2d_consistency_loss(model: nn.Module, imgs: torch.Tensor, device: str, strength: float=0.1) -> torch.Tensor:
    x_a, x_b, transform_a_inv, transform_b_inv = create_2d_consistency_augmentation(imgs, strength=strength, device=device)

    def _forward(x):
        out = model(x)
        return out[0] if isinstance(out, tuple) else out
    pred_a = _forward(x_a)
    pred_b = _forward(x_b)
    aligned_a = apply_2d_inverse_transform(pred_a, transform_a_inv, device)
    aligned_b = apply_2d_inverse_transform(pred_b, transform_b_inv, device)
    eps = 1e-08
    p = (aligned_a.softmax(1) + eps).clamp(max=1.0)
    q = (aligned_b.softmax(1) + eps).clamp(max=1.0)
    loss_pq = F.kl_div(p.log(), q, reduction='batchmean')
    loss_qp = F.kl_div(q.log(), p, reduction='batchmean')
    loss = 0.5 * (loss_pq + loss_qp)
    return loss

def merge_logits_to_binary_2d(logits: torch.Tensor, bg_channel: int=0) -> torch.Tensor:
    probs = logits.softmax(1)
    p_bg = probs[:, bg_channel:bg_channel + 1]
    p_tumor = probs[:, bg_channel + 1:].sum(1, keepdim=True)
    return torch.cat([p_bg, p_tumor], dim=1)

def build_test_loader(target_dir: str, batch_size: int, num_workers: int, image_size: int) -> DataLoader:
    image_dir = os.path.join(target_dir, 'image')
    mask_dir = os.path.join(target_dir, 'mask')
    if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
        raise FileNotFoundError(f'Expect image/ & mask/ inside {target_dir}')
    dataset = MedicalImageDataset2D(image_dir=image_dir, mask_dir=mask_dir, phase='test', image_size=(image_size, image_size), normalize=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

def test_on_target(args, device):
    print('\n' + '=' * 40)
    print(f'🧪 2D DGTTA 测试 | 目标域: {os.path.basename(args.target_dir)}')
    print(f'🔧 模型类型: {args.model_type}')
    print('=' * 40 + '\n')
    result_dir = os.path.join(args.checkpoint_dir, 'tta2d_results')
    weights_dir = os.path.join(result_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    if args.model_type == 'nnunet2d':
        print('❌ nnunet2d 模型暂时不可用，请使用 unet2d')
        return False
    else:
        model = UNet2d(in_channels=1, n_classes=args.num_classes).to(device)
        default_ckpt = os.path.join(args.checkpoint_dir, 'unet2d_best.pth')
        print('已选择 UNet2d 架构')
    ckpt_path = default_ckpt if args.model_path == 'default' else args.model_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'未找到预训练权重: {ckpt_path}')
    print(f'📦 加载模型权重: {ckpt_path}')
    checkpoint_obj = robust_torch_load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(checkpoint_obj)
    model.load_state_dict(state_dict, strict=True)
    print('🔍 开始适应前评估...')
    model.eval()
    test_loader = build_test_loader(target_dir=args.target_dir, batch_size=args.batch_test, num_workers=args.num_workers, image_size=args.image_size)
    before_metrics: Dict[str, List[float]] = {k: [] for k in ['dice', 'iou', 'sensitivity', 'ppv', 'hd95']}
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='适应前评估')):
            if len(batch) == 3:
                imgs, labels, _ = batch
            else:
                imgs, labels = batch[:2]
            imgs, labels = (imgs.to(device), labels.to(device))
            outputs = model(imgs)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            for i in range(imgs.shape[0]):
                m = calculate_all_metrics(outputs[i:i + 1], labels[i:i + 1])
                for k in before_metrics:
                    before_metrics[k].append(safe_value(m[k]))
    before_mean = {k: float(np.mean(v)) for k, v in before_metrics.items()}
    before_std = {k: float(np.std(v)) for k, v in before_metrics.items()}
    print('📊 适应前性能指标:')
    for k in before_metrics:
        print(f'  {k.upper()}: {before_mean[k]:.4f} ± {before_std[k]:.4f}')
    print('\n🔒 应用DGTTA参数冻结策略...')
    model = freeze_bn_statistics(model)
    if args.freeze_other:
        model = freeze_non_bn_parameters(model)
        print('已冻结非BN参数')
    if args.eval_mode:
        model.eval()
        print('🔧 模式: eval')
    else:
        model.train()
        print('🔧 模式: train (DGTTA推荐)')
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)):
            module.eval()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            module.track_running_stats = False
    bn_affine_params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if getattr(m, 'weight', None) is not None and m.weight.requires_grad:
                bn_affine_params.append(m.weight)
            if getattr(m, 'bias', None) is not None and m.bias.requires_grad:
                bn_affine_params.append(m.bias)
    all_trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if len(bn_affine_params) > 0:
        optim_params = bn_affine_params
        print(f'📊 使用BN仿射参数进行适应，可训练参数数量: {len(optim_params)}')
    elif len(all_trainable_params) > 0:
        optim_params = all_trainable_params
        print(f'📊 未找到BN仿射参数，使用所有可训练参数，数量: {len(optim_params)}')
    else:
        print('⚠️  警告：没有找到可训练参数，强制解冻BN仿射参数')
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if getattr(m, 'weight', None) is not None:
                    m.weight.requires_grad = True
                if getattr(m, 'bias', None) is not None:
                    m.bias.requires_grad = True
        optim_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        print(f'📊 强制解冻后，可训练参数数量: {len(optim_params)}')
    optimizer = optim.Adam(optim_params, lr=args.lr)
    bn_affine_snapshots = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            w0 = m.weight.detach().clone() if getattr(m, 'weight', None) is not None else None
            b0 = m.bias.detach().clone() if getattr(m, 'bias', None) is not None else None
            bn_affine_snapshots.append((m, w0, b0))
    if len(bn_affine_snapshots) == 0:
        print('⚠️  警告：未找到BN层，将跳过BN仿射正则化')
    print(f'\n🔄 开始DGTTA适应 (每批次{args.adapt_steps}步)...')
    after_metrics: Dict[str, List[float]] = {k: [] for k in ['dice', 'iou', 'sensitivity', 'ppv', 'hd95']}
    for batch_idx, batch in enumerate(tqdm(test_loader, desc='DGTTA适应进度')):
        if len(batch) == 3:
            imgs, labels, _ = batch
        else:
            imgs, labels = batch[:2]
        imgs, labels = (imgs.to(device), labels.to(device))
        for step in range(args.adapt_steps):
            loss = compute_2d_consistency_loss(model, imgs, device, strength=args.transform_strength)
            if args.bn_l2_reg > 0 and len(bn_affine_snapshots) > 0:
                reg = 0.0
                for m, w0, b0 in bn_affine_snapshots:
                    if getattr(m, 'weight', None) is not None and w0 is not None:
                        reg = reg + (m.weight - w0).pow(2).mean()
                    if getattr(m, 'bias', None) is not None and b0 is not None:
                        reg = reg + (m.bias - b0).pow(2).mean()
                loss = loss + args.bn_l2_reg * reg
            optimizer.zero_grad()
            loss.backward()
            if len(optim_params) > 0:
                torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
            optimizer.step()
        with torch.no_grad():
            outputs = model(imgs)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            for i in range(imgs.shape[0]):
                m = calculate_all_metrics(outputs[i:i + 1], labels[i:i + 1])
                for k in after_metrics:
                    after_metrics[k].append(safe_value(m[k]))
    after_mean = {k: float(np.mean(v)) for k, v in after_metrics.items()}
    after_std = {k: float(np.std(v)) for k, v in after_metrics.items()}
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    adapted_path = os.path.join(weights_dir, f'{model_tag}_dgtta2d_adapted_{timestamp}.pth')
    torch.save(model.state_dict(), adapted_path)
    print(f'✅ 已保存DGTTA适应后的模型权重: {adapted_path}')
    print('\n' + '=' * 60)
    print('📈 DGTTA 适应前后性能对比')
    print('=' * 60)
    report_lines = ['=' * 60, f'2D DGTTA 测试报告', f'测试时间: {timestamp}', f'目标数据集: {args.target_dir}', f'模型架构: {args.model_type}', f'原始权重: {ckpt_path}', f'适应权重: {adapted_path}', f'适应步数: {args.adapt_steps}', f'学习率: {args.lr}', f'冻结策略: {'BN统计量' + (' + 其他参数' if args.freeze_other else '')}', '', '性能指标对比 (均值 ± 标准差):', '指标名称        适应前              适应后              提升', '-' * 60]
    improvements = {}
    for k in before_metrics:
        before_val = before_mean[k]
        after_val = after_mean[k]
        improvement = after_val - before_val
        improvements[k] = improvement
        report_lines.append(f'{k.upper():<12} {before_val:.4f} ± {before_std[k]:.4f}  {after_val:.4f} ± {after_std[k]:.4f}  {improvement:+.4f}')
    avg_improvement = np.mean(list(improvements.values()))
    report_lines.extend(['-' * 60, f'平均提升: {avg_improvement:+.4f}', '=' * 60])
    report = '\n'.join(report_lines)
    print(report)
    result_file = os.path.join(result_dir, f'{model_tag}_dgtta2d_comparison_{timestamp}.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print('\n🔍 详细分析:')
    for k, improvement in improvements.items():
        if improvement > 0:
            print(f'  ✅ {k.upper()}: 提升 {improvement:+.4f}')
        elif improvement < 0:
            print(f'  ❌ {k.upper()}: 下降 {improvement:+.4f}')
        else:
            print(f'  ➖ {k.upper()}: 无变化')
    if avg_improvement > 0:
        print(f'\n🎉 DGTTA算法整体有效，平均提升: {avg_improvement:+.4f}')
    else:
        print(f'\n⚠️  DGTTA算法效果不佳，平均下降: {avg_improvement:+.4f}')
        print('建议检查以下方面:')
        print('  1. 学习率是否过高或过低')
        print('  2. 适应步数是否合适')
        print('  3. 参数冻结策略是否正确')
        print('  4. 数据集是否适合当前模型')
    return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D DGTTA (Domain Generalization Test-Time Adaptation) Evaluation Script')
    parser.add_argument('--target_dir', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-2DPATH/Glas_processed', help='包含 image/ 和 mask/ 的目标域文件夹')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='保存 / 查找权重的目录')
    parser.add_argument('--model_path', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoint_PATH/unet2d_best_PATH.pth', help='自定义权重路径，`default` 表示自动选择')
    parser.add_argument('--model_type', type=str, default='unet2d', choices=['unet2d', 'nnunet2d'], help='模型架构类型')
    parser.add_argument('--num_classes', type=int, default=2, help='输出类别数 (含背景)')
    parser.add_argument('--lr', type=float, default=1e-06, help='DGTTA适应阶段学习率')
    parser.add_argument('--adapt_steps', type=int, default=40, help='每批次的DGTTA适应步数')
    parser.add_argument('--freeze_other', action='store_true', help='是否冻结非BN参数')
    parser.add_argument('--eval_mode', action='store_true', help='是否使用eval模式（DGTTA推荐使用train模式）')
    parser.add_argument('--transform_strength', type=float, default=0.1, help='一致性空间变换强度，建议0.05~0.2')
    parser.add_argument('--bn_l2_reg', type=float, default=0.0, help='BN仿射参数L2正则系数，建议0~1e-3')
    parser.add_argument('--batch_test', type=int, default=64, help='测试批次大小')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256, help='测试时图像缩放大小')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号 (-1 表示 CPU)')
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
        with open(os.path.join(err_dir, 'dgtta2d_errors.log'), 'a', encoding='utf-8') as f:
            f.write(f'[{err_time}] {traceback.format_exc()}\n')
