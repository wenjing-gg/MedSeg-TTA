import argparse
import os
import datetime
import traceback
from typing import Dict, List
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from prosfda.models.unet_pls import UNet_PLS
from prosfda.models.unet_fas import UNet_FAS
from prosfda.utils.mix_prompt import mix_data_prompt
from prosfda.utils.fourier import FDA_source_to_target_np
from prosfda.datasets.utils.transform import fourier_augmentation
from prosfda.loss_functions.bn_loss import layer_1_loss
from nnunet2d import PlainConvUNet2D
from unet2d import UNet2d
from dataset2D import MedicalImageDataset2D
from train_source2D import calculate_all_metrics

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

def prosfda_test_time_adaptation(model, data_loader, device, args, data_prompt=None, pretrained_params=None):
    print('🔄 开始ProSFDA测试时域适应...')
    model.train()
    if args.model_type == 'prosfda_pls':
        if hasattr(model, 'data_prompt'):
            params_to_optimize = [model.data_prompt]
        else:
            params_to_optimize = model.parameters()
    elif args.model_type == 'prosfda_fas':
        params_to_optimize = model.parameters()
    else:
        params_to_optimize = model.parameters()
    optimizer = optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=0.0001)
    adaptation_losses = []
    for batch_idx, (imgs, labels, _) in enumerate(tqdm(data_loader, desc='TTA适应进度')):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        total_loss = 0.0
        if args.use_fourier_aug:
            imgs_np = imgs.cpu().numpy()
            fda_imgs_np = fourier_augmentation(imgs_np, fda_beta=args.fda_beta)
            fda_imgs = torch.from_numpy(fda_imgs_np).float().to(device)
            if args.model_type == 'prosfda_pls':
                outputs_orig, global_f_orig = model(imgs, rfeat=True)
                outputs_fda, global_f_fda = model(fda_imgs, rfeat=True)
                consistency_loss = F.l1_loss(global_f_fda, global_f_orig.detach()) + F.l1_loss(global_f_orig, global_f_fda.detach())
                if pretrained_params is not None:
                    bn_loss = layer_1_loss(model, pretrained_params, [global_f_orig], alpha=0.01)
                    total_loss += consistency_loss + 0.01 * bn_loss
                else:
                    total_loss += consistency_loss
            elif args.model_type == 'prosfda_fas':
                if data_prompt is not None:
                    outputs_orig, global_f_orig = model(mix_data_prompt(imgs, data_prompt), rfeat=True)
                    outputs_fda, global_f_fda = model(mix_data_prompt(fda_imgs, data_prompt), rfeat=True)
                else:
                    outputs_orig, global_f_orig = model(imgs, rfeat=True)
                    outputs_fda, global_f_fda = model(fda_imgs, rfeat=True)
                compact_loss = F.l1_loss(global_f_orig, global_f_fda.detach())
                seg_loss = F.binary_cross_entropy_with_logits(outputs_orig[:, 0], (labels[:, 0] > 0).float()) + F.binary_cross_entropy_with_logits(outputs_fda[:, 0], (labels[:, 0] > 0).float())
                total_loss = seg_loss + 0.1 * compact_loss
            else:
                outputs_orig = model(imgs)
                outputs_fda = model(fda_imgs)
                consistency_loss = F.mse_loss(outputs_fda, outputs_orig.detach())
                total_loss = consistency_loss
        else:
            if args.model_type == 'prosfda_pls':
                outputs_orig = model(imgs)
            elif args.model_type == 'prosfda_fas' and data_prompt is not None:
                outputs_orig = model(mix_data_prompt(imgs, data_prompt))
            else:
                outputs_orig = model(imgs)
            probs = F.softmax(outputs_orig, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-08), dim=1)
            total_loss = torch.mean(entropy)
        total_loss.backward()
        optimizer.step()
        adaptation_losses.append(total_loss.item())
        if batch_idx >= args.max_adapt_steps:
            break
    avg_loss = np.mean(adaptation_losses)
    print(f'✅ 适应完成，平均损失: {avg_loss:.6f}')
    model.eval()
    return avg_loss

def test_on_target(args, device):
    print('\n' + '=' * 40)
    print(f'🧪 开始ProSFDA测试时域适应: {os.path.basename(args.target_dir)}')
    print('=' * 40 + '\n')
    result_dir = os.path.join(args.checkpoint_dir, 'prosfda_tta_results')
    weights_dir = os.path.join(result_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    data_prompt = None
    pretrained_params = None
    if args.model_type == 'prosfda_pls':
        model = UNet_PLS(args.pretrained_model_path, patch_size=(args.image_size, args.image_size))
        model = model.to(device)
        default_ckpt = args.model_path
        print('已选择 ProSFDA PLS 架构')
        if args.pretrained_model_path:
            pretrained_params = robust_torch_load(args.pretrained_model_path, map_location=device)
    elif args.model_type == 'prosfda_fas':
        model = UNet_FAS()
        model = model.to(device)
        if args.prompt_model_path:
            prompt_params = robust_torch_load(args.prompt_model_path, map_location=device)
            data_prompt = prompt_params['model_state_dict']['data_prompt'].to(device)
            print(f'已加载prompt参数: {args.prompt_model_path}')
        default_ckpt = args.model_path
        print('已选择 ProSFDA FAS 架构')
    elif args.model_type == 'nnunet2d':
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
    test_loader = build_test_loader(target_dir=args.target_dir, batch_size=args.batch_test, num_workers=args.num_workers, image_size=args.image_size)
    if args.enable_tta:
        adaptation_loss = prosfda_test_time_adaptation(model, test_loader, device, args, data_prompt, pretrained_params)
    else:
        adaptation_loss = 0.0
        print('⚠️ 跳过测试时适应')
    metric_lists: Dict[str, List[float]] = {k: [] for k in ['dice', 'iou', 'sensitivity', 'ppv', 'hd95']}
    model.eval()
    with torch.no_grad():
        for imgs, labels, _ in tqdm(test_loader, desc='推理进度'):
            imgs = imgs.to(device)
            labels = labels.to(device)
            if args.model_type == 'prosfda_pls':
                outputs = model(imgs)
            elif args.model_type == 'prosfda_fas' and data_prompt is not None:
                outputs = model(mix_data_prompt(imgs, data_prompt))
            else:
                outputs = model(imgs)
            for i in range(imgs.shape[0]):
                m = calculate_all_metrics(outputs[i:i + 1], labels[i:i + 1])
                for k in metric_lists:
                    metric_lists[k].append(safe_value(m[k]))
    metric_mean = {k: float(np.mean(v)) for k, v in metric_lists.items()}
    metric_std = {k: float(np.std(v)) for k, v in metric_lists.items()}
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    adapted_path = os.path.join(weights_dir, f'{model_tag}_prosfda_adapted_{timestamp}.pth')
    torch.save(model.state_dict(), adapted_path)
    print(f'✅ 已保存ProSFDA适应后的模型权重: {adapted_path}')
    report_lines = ['=' * 40, f'ProSFDA测试时间: {timestamp}', f'目标数据集: {args.target_dir}', f'模型架构: {args.model_type}', f'原始权重: {ckpt_path}', f'适应权重: {adapted_path}', f'适应损失: {adaptation_loss:.6f}', f'Fourier增强: {('启用' if args.use_fourier_aug else '禁用')}', '\n性能指标 (均值 ± 标准差):']
    for k in metric_lists:
        report_lines.append(f'{k.upper()}: {metric_mean[k]:.4f} ± {metric_std[k]:.4f}')
    report_lines.append('=' * 40)
    report = '\n'.join(report_lines)
    result_file = os.path.join(result_dir, f'{model_tag}_prosfda_{timestamp}.txt')
    with open(result_file, 'w') as f:
        f.write(report)
    print(report)
    return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ProSFDA 2D Test-Time Adaptation Evaluation Script')
    parser.add_argument('--target_dir', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-2DPATH/Glas_processed', help='包含 image/ 和 mask/ 的目标域文件夹')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='保存 / 查找权重的目录')
    parser.add_argument('--model_path', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoint_PATH/unet2d_best_PATH.pth', help='自定义权重路径')
    parser.add_argument('--pretrained_model_path', type=str, default='', help='ProSFDA预训练模型路径')
    parser.add_argument('--prompt_model_path', type=str, default='', help='ProSFDA prompt模型路径')
    parser.add_argument('--model_type', type=str, default='unet2d', choices=['unet2d', 'nnunet2d', 'prosfda_pls', 'prosfda_fas'], help='模型架构类型')
    parser.add_argument('--num_classes', type=int, default=2, help='输出类别数 (含背景)')
    parser.add_argument('--lr', type=float, default=1e-06, help='适应阶段学习率')
    parser.add_argument('--enable_tta', default=True, help='启用测试时适应')
    parser.add_argument('--use_fourier_aug', default=True, help='启用Fourier域增强')
    parser.add_argument('--fda_beta', type=float, default=0.15, help='Fourier域增强强度')
    parser.add_argument('--max_adapt_steps', type=int, default=50, help='最大适应步数')
    parser.add_argument('--batch_test', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256, help='测试时图像缩放大小')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号 (-1 表示 CPU)')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f'🖥️  使用设备: {device}')
    print(f'🔬 ProSFDA模式: {args.model_type}')
    print(f'🔄 测试时适应: {('启用' if args.enable_tta else '禁用')}')
    try:
        test_on_target(args, device)
    except Exception as e:
        err_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        print('🔥 运行失败:', str(e))
        traceback.print_exc()
        err_dir = os.path.join(args.checkpoint_dir, 'prosfda_tta_results')
        os.makedirs(err_dir, exist_ok=True)
        with open(os.path.join(err_dir, 'prosfda_tta_errors.log'), 'a') as f:
            f.write(f'[{err_time}] {traceback.format_exc()}\n')
