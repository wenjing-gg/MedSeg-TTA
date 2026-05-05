import argparse
import os
import datetime
import traceback
from typing import Dict, List
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from nnunet2d import PlainConvUNet2D
from unet2d import UNet2d
from dataset2D import MedicalImageDataset2D
from train_source2D import calculate_all_metrics
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'GraTa-master'))
from custom_optimizers.grata import GraTa
from dataloaders.aug import augmentation_spatial, augmentation_strong_style, augmentation_weak_style
from dataloaders.normalize import normalize_image_to_0_1

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

def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f'{nm}.{np}')
    return params

class GraTaCompatibleModel(nn.Module):

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        if hasattr(base_model, 'encoder'):
            feature_dim = self._get_encoder_output_dim()
        else:
            feature_dim = self._get_last_conv_dim()
        self.recon_head = nn.Conv2d(feature_dim, 1, 1)
        self.supres_head = nn.ConvTranspose2d(feature_dim, 1, 4, 2, 1)
        self.denoise_head = nn.Conv2d(feature_dim, 1, 1)
        self.rotate_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(feature_dim, 6))

    def _get_encoder_output_dim(self):
        if hasattr(self.base_model, 'encoder'):
            return 512
        return 256

    def _get_last_conv_dim(self):
        for module in reversed(list(self.base_model.modules())):
            if isinstance(module, nn.Conv2d):
                return module.out_channels
        return 256

    def forward(self, x, rec=False, sup=False, den=False, rot=False):
        if hasattr(self.base_model, 'encoder') and hasattr(self.base_model, 'decoder'):
            try:
                enc_features = self.base_model.encoder(x)
                if isinstance(enc_features, (list, tuple)):
                    bottleneck_features = enc_features[-1]
                else:
                    bottleneck_features = enc_features
            except:
                pred_logit = self.base_model(x)
                bottleneck_features = torch.mean(pred_logit, dim=[2, 3], keepdim=True)
                bottleneck_features = bottleneck_features.expand(-1, -1, x.shape[2] // 8, x.shape[3] // 8)
        else:
            pred_logit = self.base_model(x)
            bottleneck_features = torch.mean(pred_logit, dim=[2, 3], keepdim=True)
            bottleneck_features = bottleneck_features.expand(-1, -1, x.shape[2] // 8, x.shape[3] // 8)
        if 'pred_logit' not in locals():
            pred_logit = self.base_model(x)
        if rec:
            recon_output = self.recon_head(bottleneck_features)
            recon_output = F.interpolate(recon_output, size=x.shape[2:], mode='bilinear', align_corners=True)
            return (recon_output, pred_logit, bottleneck_features)
        elif sup:
            supres_output = self.supres_head(bottleneck_features)
            supres_output = F.interpolate(supres_output, size=x.shape[2:], mode='bilinear', align_corners=True)
            return (supres_output, pred_logit, bottleneck_features)
        elif den:
            denoise_output = self.denoise_head(bottleneck_features)
            denoise_output = F.interpolate(denoise_output, size=x.shape[2:], mode='bilinear', align_corners=True)
            return (denoise_output, pred_logit, bottleneck_features)
        elif rot:
            rotate_output = self.rotate_head(bottleneck_features)
            return (rotate_output, pred_logit, bottleneck_features)
        else:
            return (pred_logit, bottleneck_features)

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
    model = GraTaCompatibleModel(model)
    params = collect_params(model)
    if args.optimizer == 'SGD':
        base_optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, nesterov=True)
    elif args.optimizer == 'Adam':
        base_optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'AdamW':
        base_optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(args.beta1, args.beta2))
    else:
        base_optimizer = torch.optim.AdamW(params, lr=args.lr)
    optimizer = GraTa(params, base_optimizer, model, device=device)
    test_loader = build_test_loader(target_dir=args.target_dir, batch_size=args.batch_test, num_workers=args.num_workers, image_size=args.image_size)
    metric_lists: Dict[str, List[float]] = {k: [] for k in ['dice', 'iou', 'sensitivity', 'ppv', 'hd95']}
    for batch_idx, (imgs, labels, _) in enumerate(tqdm(test_loader, desc='TTA进度')):
        imgs = imgs.to(device)
        labels = labels.to(device)
        data_dict = {'data': imgs.cpu().numpy(), 'mask': labels.cpu().numpy()}
        model.train()
        model.requires_grad_(False)
        for nm, m in model.named_modules():
            if args.aux_loss in nm or args.pse_loss in nm:
                m.requires_grad_(True)
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        optimizer.base_optimizer.zero_grad()
        optimizer.step(data_dict, args.aux_loss, args.pse_loss)
        model.eval()
        with torch.no_grad():
            model_output = model(imgs)
            if isinstance(model_output, tuple):
                outputs = model_output[0]
            else:
                outputs = model_output
            for i in range(imgs.shape[0]):
                m = calculate_all_metrics(outputs[i:i + 1], labels[i:i + 1])
                for k in metric_lists:
                    metric_lists[k].append(safe_value(m[k]))
    metric_mean = {k: float(np.mean(v)) for k, v in metric_lists.items()}
    metric_std = {k: float(np.std(v)) for k, v in metric_lists.items()}
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    adapted_path = os.path.join(weights_dir, f'{model_tag}_grata_adapted_{timestamp}.pth')
    torch.save(model.state_dict(), adapted_path)
    print(f'✅ 已保存GraTa适应后的模型权重: {adapted_path}')
    report_lines = ['=' * 40, f'测试时间: {timestamp}', f'目标数据集: {args.target_dir}', f'模型架构: {args.model_type}', f'TTA方法: GraTa', f'辅助损失: {args.aux_loss}', f'伪标签损失: {args.pse_loss}', f'原始权重: {ckpt_path}', f'适应权重: {adapted_path}', '\n性能指标 (均值 ± 标准差):']
    for k in metric_lists:
        report_lines.append(f'{k.upper()}: {metric_mean[k]:.4f} ± {metric_std[k]:.4f}')
    report_lines.append('=' * 40)
    report = '\n'.join(report_lines)
    result_file = os.path.join(result_dir, f'{model_tag}_grata_{timestamp}.txt')
    with open(result_file, 'w') as f:
        f.write(report)
    print(report)
    return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Test-Time Adaptation with GraTa')
    parser.add_argument('--target_dir', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-2DPATH/Glas_processed', help='包含 image/ 和 mask/ 的目标域文件夹')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='保存 / 查找权重的目录')
    parser.add_argument('--model_path', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoint_PATH/unet2d_best_PATH.pth', help='自定义权重路径，`default` 表示自动选择')
    parser.add_argument('--model_type', type=str, default='unet2d', choices=['unet2d', 'nnunet2d'], help='模型架构类型')
    parser.add_argument('--num_classes', type=int, default=2, help='输出类别数 (含背景)')
    parser.add_argument('--aux_loss', type=str, default='ent', choices=['consis', 'ent', 'recon', 'rotate', 'supres', 'denoise'], help='辅助损失类型')
    parser.add_argument('--pse_loss', type=str, default='consis', choices=['consis', 'ent', 'recon', 'rotate', 'supres', 'denoise'], help='伪标签损失类型')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW'], help='基础优化器类型')
    parser.add_argument('--lr', type=float, default=0.0001, help='适应阶段学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--batch_test', type=int, default=32)
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
        with open(os.path.join(err_dir, 'tta2d_errors.log'), 'a') as f:
            f.write(f'[{err_time}] {traceback.format_exc()}\n')
