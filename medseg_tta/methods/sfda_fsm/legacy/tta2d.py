import argparse
import os, datetime, traceback, pickle, time
from typing import Dict, List, Tuple, Any
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from nnunet2d import PlainConvUNet2D
from unet2d import UNet2d
from dataset2D import MedicalImageDataset2D
from train_source2D import calculate_all_metrics

class ContrastiveDataset2D(Dataset):

    def __init__(self, img_dir, msk_dir, phase='train', image_size=(256, 256), normalize=True):
        self.dataset = MedicalImageDataset2D(img_dir, msk_dir, phase, image_size, normalize)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, name = self.dataset[idx]
        image_rotate = torch.rot90(image, k=2, dims=[1, 2])
        label_rotate = torch.rot90(label, k=2, dims=[0, 1])
        data = {'source_image': image, 'source_image_rotate': image_rotate, 'target_image': image, 'target_image_rotate': image_rotate, 'label': label, 'label_rotate': label_rotate}
        weight = torch.ones_like(label).float()
        return (data, weight, name)

class FeatureDistillationLoss(nn.Module):

    def __init__(self):
        super(FeatureDistillationLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, features_a, features_b):
        return self.criterion(features_a, features_b)

class SegmentationLoss(nn.Module):

    def __init__(self, num_classes):
        super(SegmentationLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, outputs, targets, weight=None):
        loss = F.cross_entropy(outputs, targets.squeeze(1).long(), reduction='none')
        if weight is not None:
            loss = loss * weight
        return loss.mean()

def extract_ampl_phase(fft_im):
    fft_amp = fft_im[:, :, :, :, 0] ** 2 + fft_im[:, :, :, :, 1] ** 2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return (fft_amp, fft_pha)

def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    b = np.floor(np.amin((h, w)) * L).astype(int)
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]
    amp_src[:, :, 0:b, w - b:w] = amp_trg[:, :, 0:b, w - b:w]
    amp_src[:, :, h - b:h, 0:b] = amp_trg[:, :, h - b:h, 0:b]
    amp_src[:, :, h - b:h, w - b:w] = amp_trg[:, :, h - b:h, w - b:w]
    return amp_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    fft_src = torch.rfft(src_img.clone(), signal_ndim=2, onesided=False)
    fft_trg = torch.rfft(trg_img.clone(), signal_ndim=2, onesided=False)
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float, device=src_img.device)
    fft_src_[:, :, :, :, 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:, :, :, :, 1] = torch.sin(pha_src.clone()) * amp_src_.clone()
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft(fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH, imgW])
    return src_in_trg

class ModelWithFeatures(nn.Module):

    def __init__(self, base_model):
        super(ModelWithFeatures, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        output = self.base_model(x)
        if not self.training:
            return output
        if isinstance(self.base_model, UNet2d):
            features = self.base_model.get_features(x)
            bottleneck_feature = features[-1]
        else:
            bottleneck_feature = None
        return (bottleneck_feature, output)

def lr_poly(base_lr, iter, max_iter, power=0.9):
    return base_lr * (1 - float(iter) / max_iter) ** power

def adjust_learning_rate(optimizer, i_iter, length, base_lr=0.0001, power=0.9):
    lr = lr_poly(base_lr, i_iter, length, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def extract_state_dict(obj):
    if isinstance(obj, dict):
        if 'state_dict' in obj:
            return obj['state_dict']
        if 'model_state_dict' in obj:
            return obj['model_state_dict']
    return obj

def safe_value(v):
    return v.item() if isinstance(v, torch.Tensor) else float(v)

def build_test_loader(target_dir, batch_size, num_workers, image_size):
    img_dir, msk_dir = (os.path.join(target_dir, 'image'), os.path.join(target_dir, 'mask'))
    if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
        raise FileNotFoundError('expect image/ & mask/ in ' + target_dir)
    ds = MedicalImageDataset2D(img_dir, msk_dir, phase='test', image_size=(image_size, image_size), normalize=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

def build_adaptation_loader(target_dir, batch_size, num_workers, image_size):
    img_dir, msk_dir = (os.path.join(target_dir, 'image'), os.path.join(target_dir, 'mask'))
    if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
        raise FileNotFoundError('expect image/ & mask/ in ' + target_dir)
    ds = ContrastiveDataset2D(img_dir, msk_dir, phase='train', image_size=(image_size, image_size), normalize=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

def evaluate_model(model, loader, device, desc='评估'):
    model.eval()
    metrics = {k: [] for k in ['dice', 'iou', 'sensitivity', 'ppv', 'hd95']}
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc=desc):
            imgs, labels = (imgs.to(device), labels.to(device))
            out = model(imgs)
            if isinstance(out, tuple) and len(out) == 2:
                out = out[1]
            for i in range(imgs.size(0)):
                m = calculate_all_metrics(out[i:i + 1], labels[i:i + 1])
                for k in metrics:
                    metrics[k].append(safe_value(m[k]))
    mean = {k: float(np.mean(v)) for k, v in metrics.items()}
    std = {k: float(np.std(v)) for k, v in metrics.items()}
    return (mean, std)

def print_metrics_comparison(original_mean, original_std, adapted_mean, adapted_std):
    print('\n' + '=' * 60)
    print('📊 适应前后性能对比')
    print('=' * 60)
    print(f'{'指标':<12} {'原始模型':<20} {'适应后模型':<20} {'改进幅度':<15}')
    print('-' * 60)
    for metric in ['dice', 'iou', 'sensitivity', 'ppv', 'hd95']:
        orig_val = original_mean[metric]
        adapt_val = adapted_mean[metric]
        improvement = adapt_val - orig_val
        improvement_sign = '↓' if metric == 'hd95' and improvement < 0 else '↑' if improvement > 0 else '→'
        print(f'{metric.upper():<12} {orig_val:.4f}±{original_std[metric]:.4f}   {adapt_val:.4f}±{adapted_std[metric]:.4f}   {improvement:+.4f} {improvement_sign}')
    print('=' * 60)

def test_on_target_sfda_fsm(args, device):
    print('🔄 加载预训练模型...')
    if args.model_type == 'nnunet2d':
        base_model = PlainConvUNet2D(input_channels=1, n_stages=5, features_per_stage=(32, 64, 128, 256, 512), kernel_sizes=3, strides=(1, 2, 2, 2, 2), n_conv_per_stage=2, num_classes=args.num_classes, n_conv_per_stage_decoder=2, deep_supervision=False).to(device)
    else:
        base_model = UNet2d(in_channels=1, n_classes=args.num_classes).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    base_model.load_state_dict(extract_state_dict(ckpt), strict=True)
    model = ModelWithFeatures(base_model).to(device)
    print('📂 构建数据加载器...')
    test_loader = build_test_loader(target_dir=args.target_dir, batch_size=args.batch_test, num_workers=args.num_workers, image_size=args.image_size)
    adapt_loader = build_adaptation_loader(target_dir=args.target_dir, batch_size=args.batch_test, num_workers=args.num_workers, image_size=args.image_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)
    distill_criterion = FeatureDistillationLoss().to(device)
    source_criterion = SegmentationLoss(num_classes=args.num_classes).to(device)
    print('⚙️ 开始SFDA-FSM测试时适应...')
    model.train()
    metric_lists = {k: [] for k in ['dice', 'iou', 'sensitivity', 'ppv', 'hd95']}
    for epoch in range(args.num_steps):
        seg_loss = 0
        dis_loss = 0
        con_loss = 0
        tic = time.time()
        model.train()
        for i_iter, batch in enumerate(adapt_loader):
            data, weight, _ = batch
            source_image = data['source_image'].to(device)
            source_image_rotate = data['source_image_rotate'].to(device)
            target_image = data['target_image'].to(device)
            target_image_rotate = data['target_image_rotate'].to(device)
            label = data['label'].to(device).unsqueeze(1)
            label_rotate = data['label_rotate'].to(device).unsqueeze(1)
            weight = weight.to(device)
            source_feature, source_output = model(source_image)
            source_feature_rotate, source_output_rotate = model(source_image_rotate)
            target_feature, target_output = model(target_image)
            target_feature_rotate, target_output_rotate = model(target_image_rotate)
            loss_distill = distill_criterion(distill_criterion(source_feature, source_feature_rotate), distill_criterion(target_feature, target_feature_rotate))
            loss_contrast = distill_criterion(source_feature, target_feature) + distill_criterion(distill_criterion(source_feature, target_feature_rotate), distill_criterion(source_feature, source_feature_rotate))
            loss_pseudo = source_criterion(source_output, label, weight=weight) + source_criterion(target_output, label, weight=weight) + source_criterion(source_output_rotate, label_rotate, weight=weight) + source_criterion(target_output_rotate, label_rotate, weight=weight)
            loss_total = loss_pseudo + args.w_distill * loss_distill + args.w_contrast * loss_contrast
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            seg_loss += loss_pseudo.item()
            dis_loss += loss_distill.item()
            con_loss += loss_contrast.item()
            lr = adjust_learning_rate(optimizer=optimizer, i_iter=i_iter + epoch * len(adapt_loader), length=len(adapt_loader) * args.num_steps, base_lr=args.lr)
        batch_time = time.time() - tic
        print('Epoch: [{}/{}], Time: {:.2f}, lr: {:.6f}, Seg Loss: {:.6f}, Dis Loss: {:.6f}, Con Loss: {:.6f}'.format(epoch + 1, args.num_steps, batch_time, lr, seg_loss / len(adapt_loader), dis_loss / len(adapt_loader), con_loss / len(adapt_loader)))
        if (epoch + 1) % args.eval_interval == 0:
            epoch_mean, epoch_std = evaluate_model(model, test_loader, device, f'Epoch {epoch + 1}/{args.num_steps} 评估')
            for k in metric_lists:
                metric_lists[k].append(epoch_mean[k])
            print(f'Epoch {epoch + 1} 评估: Dice={epoch_mean['dice']:.4f}, IoU={epoch_mean['iou']:.4f}')
    print('🔍 评估适应后模型性能...')
    adapt_mean, adapt_std = evaluate_model(model, test_loader, device, '适应后模型评估')
    if args.save_model:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        parts = args.target_dir.split('/')
        dataset_name = parts[5]
        save_path = os.path.join(args.checkpoint_dir, f'unet2d_sfda_fsm_Glas.pth')
        torch.save(model.base_model.state_dict(), save_path)
        print(f'✅ 适应后权重已保存: {save_path}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFDA-FSM Test-Time Adaptation Script')
    parser.add_argument('--target_dir', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-2DPATH/Glas_processed')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--model_path', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoint_PATH/unet2d_best_PATH.pth')
    parser.add_argument('--model_type', type=str, default='unet2d', choices=['unet2d', 'nnunet2d'])
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001, help='适应学习率')
    parser.add_argument('--num_steps', type=int, default=5, help='适应epoch数')
    parser.add_argument('--eval_interval', type=int, default=1, help='每N个epoch评估一次')
    parser.add_argument('--w_distill', type=float, default=0.1, help='蒸馏损失权重')
    parser.add_argument('--w_contrast', type=float, default=0.1, help='对比损失权重')
    parser.add_argument('--batch_test', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--save_model', action='store_true', default=True, help='是否保存适应后的模型')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    try:
        print('🔬 使用 SFDA-FSM 算法进行测试时域适应')
        test_on_target_sfda_fsm(args, device)
    except Exception as e:
        err_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        print('🔥 运行失败:', str(e))
        traceback.print_exc()
