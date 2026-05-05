import argparse
import os
import datetime
import traceback
from typing import Dict, List, Tuple, Optional
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
try:
    from nnunet2d import PlainConvUNet2D
    NNUNET2D_AVAILABLE = True
except ImportError:
    print('[Warning] nnunet2d not available, PlainConvUNet2D will not be available')
    NNUNET2D_AVAILABLE = False
try:
    from unet2d import UNet2d
    UNET2D_AVAILABLE = True
except ImportError:
    print('[Warning] unet2d not available, UNet2d will not be available')
    UNET2D_AVAILABLE = False
from dataset2D import MedicalImageDataset2D
from train_source2D import calculate_all_metrics
from utils.tent import configure_model, collect_params, Tent
from utils.utils import get_elliptic

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

def build_test_loader(target_dir: str, batch_size: int, num_workers: int, image_size: int) -> DataLoader:
    image_dir = os.path.join(target_dir, 'image')
    mask_dir = os.path.join(target_dir, 'mask')
    if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
        raise FileNotFoundError(f'Expect image/ & mask/ inside {target_dir}')
    dataset = MedicalImageDataset2D(image_dir=image_dir, mask_dir=mask_dir, phase='test', image_size=(image_size, image_size), normalize=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

class PureSaTTCA2D:

    def __init__(self, args):
        self.args = args
        self.scale_candidates = [0.6, 0.8, 1.0, 1.2, 1.5]

    def _get_prediction_mask(self, logits):
        if logits.shape[1] == 2:
            pred_mask = (logits.argmax(1) == 1).float()
        else:
            pred_mask = (torch.sigmoid(logits) > 0.5).float()
        return pred_mask

    def _estimate_lesion_diameter(self, pred_mask):
        area = pred_mask.sum().item()
        if area == 0:
            return 20
        equivalent_diameter = 2.0 * np.sqrt(area / np.pi)
        return max(10, int(equivalent_diameter))

    def _generate_click_mask(self, center_y, center_x, diameter, H, W, device):
        radius = diameter / 2.0
        y_coords = torch.arange(H, device=device).float()
        x_coords = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distance = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        click_mask = (distance <= radius).float()
        return click_mask.unsqueeze(0).unsqueeze(0)

    def _find_lesion_center(self, pred_mask, gt_mask=None):
        H, W = pred_mask.shape
        if gt_mask is not None and gt_mask.sum() > 0:
            gt_coords = torch.nonzero(gt_mask, as_tuple=False).float()
            center_y = gt_coords[:, 0].mean().item()
            center_x = gt_coords[:, 1].mean().item()
            return (center_y, center_x)
        if pred_mask.sum() > 0:
            pred_coords = torch.nonzero(pred_mask, as_tuple=False).float()
            center_y = pred_coords[:, 0].mean().item()
            center_x = pred_coords[:, 1].mean().item()
            return (center_y, center_x)
        return (H / 2.0, W / 2.0)

    def _evaluate_click_quality(self, click_mask, pred_mask):
        intersection = (click_mask * pred_mask).sum().item()
        click_area = click_mask.sum().item()
        pred_area = pred_mask.sum().item()
        if click_area == 0 or pred_area == 0:
            return 0.0
        overlap_ratio = intersection / click_area
        coverage_ratio = intersection / pred_area
        alpha, beta = (0.7, 0.3)
        score = alpha * overlap_ratio + beta * coverage_ratio
        return score

    def _select_optimal_scale(self, base_diameter, center_y, center_x, pred_mask, H, W, device):
        best_score = -1
        best_click_mask = None
        best_diameter = base_diameter
        for scale in self.scale_candidates:
            candidate_diameter = int(base_diameter * scale)
            click_mask = self._generate_click_mask(center_y, center_x, candidate_diameter, H, W, device)
            score = self._evaluate_click_quality(click_mask[0, 0], pred_mask)
            if score > best_score:
                best_score = score
                best_click_mask = click_mask
                best_diameter = candidate_diameter
        return (best_click_mask, best_diameter, best_score)

    def generate_pseudo_labels(self, logits, labels):
        B, C, H, W = logits.shape
        device = logits.device
        pred_masks = self._get_prediction_mask(logits)
        pseudo_labels = torch.zeros_like(logits)
        diameters = []
        scores = []
        for i in range(B):
            pred_mask = pred_masks[i]
            gt_mask = labels[i] if labels is not None else None
            base_diameter = self._estimate_lesion_diameter(pred_mask)
            center_y, center_x = self._find_lesion_center(pred_mask, gt_mask)
            best_click_mask, best_diameter, best_score = self._select_optimal_scale(base_diameter, center_y, center_x, pred_mask, H, W, device)
            if C == 2:
                pseudo_labels[i, 1:2] = best_click_mask[0]
            else:
                pseudo_labels[i] = best_click_mask[0]
            diameters.append(best_diameter)
            scores.append(best_score)
        meta_info = {'mean_diameter': np.mean(diameters), 'mean_score': np.mean(scores), 'diameters': diameters, 'scores': scores}
        return (pseudo_labels, meta_info)

def build_tent_model_2d(model, args):
    model = configure_model(model)
    params, names = collect_params(model)
    if len(params) == 0:
        raise RuntimeError('未收集到可适应参数(BatchNorm γ/β)，请确认网络包含BN层。')
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    tent_model = Tent(model, optimizer, steps=args.tent_steps, episodic=args.episodic, entropy=False, use_adaptive_loss=False)
    print(f'TENT 可更新参数数: {len(params)}')
    return tent_model

def test_on_target(args, device):
    print('\n' + '=' * 50)
    print(f'🧪 Pure SaTTCA 2D 测试 | 目标域: {os.path.basename(args.target_dir)}')
    print('=' * 50 + '\n')
    result_dir = os.path.join(args.checkpoint_dir, 'sattca2d_results')
    weights_dir = os.path.join(result_dir, 'weights')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    if args.model_type == 'nnunet2d':
        if not NNUNET2D_AVAILABLE:
            raise ImportError('PlainConvUNet2D not available.')
        model = PlainConvUNet2D(input_channels=1, n_stages=5, features_per_stage=(32, 64, 128, 256, 512), kernel_sizes=3, strides=(1, 2, 2, 2, 2), n_conv_per_stage=2, num_classes=args.num_classes, n_conv_per_stage_decoder=2, deep_supervision=False).to(device)
        default_ckpt = os.path.join(args.checkpoint_dir, 'nnunet2d_best.pth')
    else:
        if not UNET2D_AVAILABLE:
            raise ImportError('UNet2d not available.')
        model = UNet2d(in_channels=1, n_classes=args.num_classes).to(device)
        default_ckpt = os.path.join(args.checkpoint_dir, 'unet2d_best.pth')
    ckpt_path = default_ckpt if args.model_path == 'default' else args.model_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'未找到预训练权重: {ckpt_path}')
    print(f'加载模型权重: {ckpt_path}')
    checkpoint_obj = robust_torch_load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(checkpoint_obj)
    model.load_state_dict(state_dict, strict=True)
    tent_model = build_tent_model_2d(model, args)
    sattca_generator = PureSaTTCA2D(args)
    print('已构建Pure SaTTCA模型')
    test_loader = build_test_loader(target_dir=args.target_dir, batch_size=args.batch_test, num_workers=args.num_workers, image_size=args.image_size)
    metric_lists: Dict[str, List[float]] = {k: [] for k in ['dice', 'iou', 'sensitivity', 'ppv', 'hd95']}
    metric_rows = []
    adaptation_stats = []
    for batch_idx, (imgs, labels, filenames) in enumerate(tqdm(test_loader, desc='SaTTCA推理')):
        imgs = imgs.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(imgs)
        pseudo_labels, meta_info = sattca_generator.generate_pseudo_labels(logits.detach(), labels)
        if pseudo_labels.sum() > 0:
            adapted_logits, loss_dict, _ = tent_model([imgs, pseudo_labels])
        else:
            adapted_logits = logits
            loss_dict = {'total_loss': 0.0}
        adapted_metrics = calculate_all_metrics(adapted_logits, labels)
        for k in metric_lists:
            metric_lists[k].append(safe_value(adapted_metrics[k]))
        adaptation_stats.append({'batch_idx': batch_idx, 'mean_diameter': meta_info.get('mean_diameter', 0), 'mean_score': meta_info.get('mean_score', 0), 'pseudo_label_area': pseudo_labels.sum().item(), 'loss': safe_value(loss_dict.get('total_loss', 0))})
        for j, fname in enumerate(filenames):
            metric_rows.append({'file_id': fname, 'dice': safe_value(adapted_metrics['dice']), 'iou': safe_value(adapted_metrics['iou']), 'sensitivity': safe_value(adapted_metrics['sensitivity']), 'ppv': safe_value(adapted_metrics['ppv']), 'hd95': safe_value(adapted_metrics['hd95'])})
        if args.debug and batch_idx < 5:
            print(f'Batch {batch_idx}: 直径={meta_info.get('mean_diameter', 0):.1f}, 得分={meta_info.get('mean_score', 0):.3f}, 适应Dice={adapted_metrics['dice']:.4f}')
    adapted_mean = {k: float(np.mean(v)) for k, v in metric_lists.items()}
    adapted_std = {k: float(np.std(v)) for k, v in metric_lists.items()}
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    adapted_path = os.path.join(weights_dir, f'unet2d_SaTTCA_PH2.pth')
    torch.save(model.state_dict(), adapted_path)
    report_lines = ['=' * 60, f'Pure SaTTCA 2D 测试结果', f'测试时间: {timestamp}', f'目标数据集: {args.target_dir}', f'模型: {args.model_type}', f'学习率: {args.lr}', f'适应步数: {args.tent_steps}', f'图像尺寸: {args.image_size}x{args.image_size}', '', '适应后性能:']
    for k in adapted_mean:
        report_lines.append(f'  {k.upper()}: {adapted_mean[k]:.4f} ± {adapted_std[k]:.4f}')
    report_lines.append('')
    report_lines.append(f'平均伪标签直径: {np.mean([s['mean_diameter'] for s in adaptation_stats]):.1f}')
    report_lines.append(f'平均点击得分: {np.mean([s['mean_score'] for s in adaptation_stats]):.3f}')
    report_lines.append('=' * 60)
    report = '\n'.join(report_lines)
    print(report)
    result_file = os.path.join(result_dir, f'{model_tag}_sattca2d_report_{timestamp}.txt')
    with open(result_file, 'w') as f:
        f.write(report)
    df_detail = pd.DataFrame(metric_rows)
    csv_file = os.path.join(result_dir, f'{model_tag}_sattca2d_detailed_{timestamp}.csv')
    df_detail.to_csv(csv_file, index=False)
    df_stats = pd.DataFrame(adaptation_stats)
    stats_file = os.path.join(result_dir, f'{model_tag}_sattca2d_stats_{timestamp}.csv')
    df_stats.to_csv(stats_file, index=False)
    print(f'📄 详细结果: {csv_file}')
    print(f'📊 适应统计: {stats_file}')
    print(f'🔧 适应后权重: {adapted_path}')
    return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pure SaTTCA 2D Test-Time Adaptation')
    parser.add_argument('--target_dir', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-2Ddermoscopy/PH2', help='目标域数据文件夹')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='权重保存/读取目录')
    parser.add_argument('--model_path', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoint_dermoscopy/unet2d_best_dermoscopy.pth', help='预训练权重路径')
    parser.add_argument('--model_type', type=str, default='unet2d', choices=['unet2d', 'nnunet2d'], help='模型架构')
    parser.add_argument('--num_classes', type=int, default=2, help='类别数')
    parser.add_argument('--lr', type=float, default=5e-06, help='学习率')
    parser.add_argument('--tent_steps', type=int, default=20, help='每批次适应步数')
    parser.add_argument('--episodic', action='store_true', help='使用episodic模式')
    parser.add_argument('--batch_test', type=int, default=8, help='测试批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器worker数')
    parser.add_argument('--image_size', type=int, default=256, help='图像大小')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f'🖥️  使用设备: {device}')
    print(f'[SaTTCA设置] 学习率: {args.lr}, 适应步数: {args.tent_steps}, Episodic: {args.episodic}')
    try:
        test_on_target(args, device)
    except Exception as e:
        print('🔥 运行失败:', str(e))
        traceback.print_exc()
