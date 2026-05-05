import argparse
import os
import math
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils_brats_all import get_data_loader
from metrics import cal_dice, cal_hd95, IoU, cal_RVE, cal_sensitivity, cal_ppv
from loss import CombinedLoss
from nnunet import PlainConvUNet

def check_labels(labels, num_classes):
    if (labels < 0).any() or (labels >= num_classes).any():
        illegal_values = labels[(labels < 0) | (labels >= num_classes)]
        print(f'发现非法标签值: {illegal_values.unique()}, 有效范围应为 [0, {num_classes - 1}]')
        raise ValueError('标签值超出类别范围')

def train(args):
    warmup_ratio = args.warmup_ratio
    min_lr = args.min_lr
    source_root = args.source_root
    target_root = args.target_root
    batch_train = args.batch_train
    batch_test = args.batch_test
    num_workers = args.num_workers
    num_epochs = args.epochs
    learning_rate = args.lr
    img_type = args.img
    dataset_mode = args.mode
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')
    train_loader, test_loader = get_data_loader(source_root, target_root, batch_train, batch_test, nw=num_workers, img=img_type, mode=dataset_mode)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
    class_weights = torch.tensor([0.01, 1.0, 1.0, 1.0], device=device)
    criterion = CombinedLoss(class_weights=class_weights, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / learning_rate, cosine_factor)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    patience = 30
    best_epoch = 0
    best_avg_dice = -1.0
    best_avg_hd95 = float('inf')
    best_avg_IoU = -1.0
    best_avg_RVE = -1.0
    best_avg_sensitivity = -1.0
    best_avg_ppv = -1.0
    global_step = 0
    for epoch in range(num_epochs):
        if epoch - best_epoch > patience:
            print('Early stopping!')
            break
        model.train()
        epoch_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for i, (imgs, labels, *_) in enumerate(train_pbar):
            imgs, labels = (imgs.to(device), labels.to(device))
            check_labels(labels, num_classes=4)
            optimizer.zero_grad()
            outputs = model.forward(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            current_lr = optimizer.param_groups[0]['lr']
            epoch_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{epoch_loss / (i + 1):.4f}', 'lr': f'{current_lr:.2e}'})
        avg_loss = epoch_loss / len(train_loader)
        tqdm.write(f'\nEpoch [{epoch + 1}/{num_epochs}] Avg Train Loss: {avg_loss:.4f}')
        model.eval()
        test_loss = 0.0
        total_dice = [0.0, 0.0, 0.0]
        total_hd95 = [0.0, 0.0, 0.0]
        total_IoU = [0.0, 0.0, 0.0]
        total_RVE = [0.0, 0.0, 0.0]
        total_sensitivity = [0.0, 0.0, 0.0]
        total_ppv = [0.0, 0.0, 0.0]
        metric_count = 0
        test_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        with torch.no_grad():
            for imgs, labels, *_ in test_pbar:
                imgs, labels = (imgs.to(device), labels.to(device))
                outputs = model.forward(imgs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                labels_squeezed = labels.squeeze(1)
                dice1, dice2, dice3 = cal_dice(outputs, labels_squeezed)
                hd95_ET, hd95_TC, hd95_WT = cal_hd95(outputs, labels_squeezed)
                IoU_ET, IoU_TC, IoU_WT = IoU(outputs, labels_squeezed)
                RVE_ET, RVE_TC, RVE_WT = cal_RVE(outputs, labels_squeezed)
                sensitivity_ET, sensitivity_TC, sensitivity_WT = cal_sensitivity(outputs, labels_squeezed)
                ppv_ET, ppv_TC, ppv_WT = cal_ppv(outputs, labels_squeezed)
                total_dice[0] += dice1.item()
                total_dice[1] += dice2.item()
                total_dice[2] += dice3.item()
                total_hd95[0] += hd95_ET
                total_hd95[1] += hd95_TC
                total_hd95[2] += hd95_WT
                total_IoU[0] += IoU_ET
                total_IoU[1] += IoU_TC
                total_IoU[2] += IoU_WT
                total_RVE[0] += RVE_ET
                total_RVE[1] += RVE_TC
                total_RVE[2] += RVE_WT
                total_sensitivity[0] += sensitivity_ET
                total_sensitivity[1] += sensitivity_TC
                total_sensitivity[2] += sensitivity_WT
                total_ppv[0] += ppv_ET
                total_ppv[1] += ppv_TC
                total_ppv[2] += ppv_WT
                metric_count += 1
                test_pbar.set_postfix({'val_loss': f'{test_loss / metric_count:.4f}', 'dice': f'{dice1.item():.2f}/{dice2.item():.2f}/{dice3.item():.2f}'})
        avg_test_loss = test_loss / len(test_loader)
        avg_dice = [d / metric_count for d in total_dice]
        avg_hd95 = [h / metric_count for h in total_hd95]
        avg_IoU = [i / metric_count for i in total_IoU]
        avg_RVE = [r / metric_count for r in total_RVE]
        avg_sensitivity = [s / metric_count for s in total_sensitivity]
        avg_ppv = [p / metric_count for p in total_ppv]
        current_avg_dice = sum(avg_dice) / 3
        current_avg_hd95 = sum(avg_hd95) / 3
        if current_avg_dice > best_avg_dice or (current_avg_dice == best_avg_dice and current_avg_hd95 < best_avg_hd95):
            best_avg_dice = current_avg_dice
            best_avg_hd95 = current_avg_hd95
            best_epoch = epoch
            best_model_path = os.path.join(checkpoint_dir, 'nnunet_best.pth')
            torch.save(model.state_dict(), best_model_path)
            log_content = f'[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] '
            log_content += f'img={img_type} mode={dataset_mode}\n'
            log_content += f'Epoch {epoch + 1}, Best Dice: {best_avg_dice:.4f}, Best HD95: {best_avg_hd95:.4f}\n'
            log_content += f'Dice (ET/TC/WT): {avg_dice[0]:.4f}/{avg_dice[1]:.4f}/{avg_dice[2]:.4f}\n'
            log_content += f'HD95 (ET/TC/WT): {avg_hd95[0]:.4f}/{avg_hd95[1]:.4f}/{avg_hd95[2]:.4f}\n'
            log_content += f'IoU (ET/TC/WT): {avg_IoU[0]:.4f}/{avg_IoU[1]:.4f}/{avg_IoU[2]:.4f}\n'
            log_content += f'RVE (ET/TC/WT): {avg_RVE[0]:.4f}/{avg_RVE[1]:.4f}/{avg_RVE[2]:.4f}\n'
            log_content += f'Sensitivity (ET/TC/WT): {avg_sensitivity[0]:.4f}/{avg_sensitivity[1]:.4f}/{avg_sensitivity[2]:.4f}\n'
            log_content += f'PPV (ET/TC/WT): {avg_ppv[0]:.4f}/{avg_ppv[1]:.4f}/{avg_ppv[2]:.4f}\n'
            log_content += '-' * 40 + '\n'
            log_file = os.path.join(checkpoint_dir, 'best_metrics.txt')
            with open(log_file, 'a') as f:
                f.write(log_content)
            tqdm.write(f'🌟 New best model saved (Avg Dice: {current_avg_dice:.4f}, Avg HD95: {current_avg_hd95:.4f})\n')

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 开始在目标数据集上测试模态: {args.img.upper()}')
    print(f'{'=' * 40}\n')
    model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
    best_model_path = os.path.join(args.checkpoint_dir, 'nnunet_best.pth')
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    source_root = args.source_root
    target_root = args.target_root
    batch_train = args.batch_train
    batch_test = args.batch_test
    num_workers = args.num_workers
    img_type = args.img
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')
    train_loader, target_test_loader = get_data_loader(source_root, target_root, batch_train, batch_test, nw=num_workers, img=img_type, mode='source_to_target')
    total_dice = [0.0, 0.0, 0.0]
    total_hd95 = [0.0, 0.0, 0.0]
    metric_count = 0
    with torch.no_grad():
        for imgs, labels, *_ in tqdm(target_test_loader, desc='测试目标数据集'):
            imgs, labels = (imgs.to(device), labels.to(device))
            outputs = model.forward(imgs)
            labels_squeezed = labels.squeeze(1)
            dice1, dice2, dice3 = cal_dice(outputs, labels_squeezed)
            hd95_ET, hd95_TC, hd95_WT = cal_hd95(outputs, labels_squeezed)
            total_dice[0] += dice1.item()
            total_dice[1] += dice2.item()
            total_dice[2] += dice3.item()
            total_hd95[0] += hd95_ET
            total_hd95[1] += hd95_TC
            total_hd95[2] += hd95_WT
            metric_count += 1
    if metric_count == 0:
        raise ValueError('未加载到任何测试数据，请检查数据路径和模式设置')
    avg_dice = [value / metric_count for value in total_dice]
    avg_hd95 = [value / metric_count for value in total_hd95]
    log_content = f'\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n'
    log_content += f'目标数据集测试结果 ({args.img}):\n'
    log_content += f'Dice (ET/TC/WT): {avg_dice[0]:.4f}/{avg_dice[1]:.4f}/{avg_dice[2]:.4f}\n'
    log_content += f'HD95 (ET/TC/WT): {avg_hd95[0]:.4f}/{avg_hd95[1]:.4f}/{avg_hd95[2]:.4f}\n'
    log_content += '=' * 40 + '\n'
    result_file = os.path.join(args.checkpoint_dir, 'target_test_results.txt')
    with open(result_file, 'a') as f:
        f.write(log_content)
    print(log_content)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UNet3d with Warmup+CosineAnnealing')
    parser.add_argument('--source_root', type=str, default='/home/yuwenjing/data/BraTS2024')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/BraTS-SSA')
    parser.add_argument('--mode', type=str, default='source_to_source')
    parser.add_argument('--batch_train', type=int, default=4)
    parser.add_argument('--batch_test', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--min_lr', type=float, default=1e-07)
    args = parser.parse_args()
    modalities = ['all']
    checkpoint_dir_root = args.checkpoint_dir
    for current_img in modalities:
        print(f'\n{'=' * 40}')
        print(f'🚀 开始训练图像模态: {current_img.upper()}')
        print(f'{'=' * 40}\n')
        args.img = current_img
        modality_checkpoint_dir = os.path.join(checkpoint_dir_root, current_img)
        os.makedirs(modality_checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = modality_checkpoint_dir
        train(args)
