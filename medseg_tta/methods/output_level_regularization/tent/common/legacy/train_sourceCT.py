import os
import math
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
from dataset_CT import get_ct_data_loaders
from unet3d import UNet3d

def compute_hausdorff_distance_95_3d(pred_mask, target_mask):
    try:
        pred_points = np.argwhere(pred_mask)
        target_points = np.argwhere(target_mask)
        if len(pred_points) == 0 and len(target_points) == 0:
            return 0.0
        elif len(pred_points) == 0 or len(target_points) == 0:
            return 373.1287
        distances_pred_to_target = []
        for pred_point in pred_points:
            min_dist = np.min(np.sqrt(np.sum((target_points - pred_point) ** 2, axis=1)))
            distances_pred_to_target.append(min_dist)
        distances_target_to_pred = []
        for target_point in target_points:
            min_dist = np.min(np.sqrt(np.sum((pred_points - target_point) ** 2, axis=1)))
            distances_target_to_pred.append(min_dist)
        all_distances = distances_pred_to_target + distances_target_to_pred
        hd95 = np.percentile(all_distances, 95)
        return float(hd95)
    except Exception:
        return 373.1287

class SegmentationLoss3D(nn.Module):

    def __init__(self, ce_weight=1.0, dice_weight=1.0, class_weights=None, smooth=1e-06, ignore_background=True):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    def dice_loss(self, pred_probs, targets):
        batch_size, num_classes = (pred_probs.shape[0], pred_probs.shape[1])
        targets_one_hot = torch.zeros_like(pred_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
        dice_scores = []
        start_idx = 1 if self.ignore_background else 0
        for class_idx in range(start_idx, num_classes):
            pred_class = pred_probs[:, class_idx, :, :, :]
            target_class = targets_one_hot[:, class_idx, :, :, :]
            intersection = torch.sum(pred_class * target_class, dim=(1, 2, 3))
            pred_sum = torch.sum(pred_class, dim=(1, 2, 3))
            target_sum = torch.sum(target_class, dim=(1, 2, 3))
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            dice_scores.append(dice)
        if len(dice_scores) > 0:
            dice_scores = torch.stack(dice_scores, dim=1)
            mean_dice = torch.mean(dice_scores)
            return 1.0 - mean_dice
        else:
            return torch.tensor(0.0, device=pred_probs.device, requires_grad=True)

    def forward(self, logits, targets):
        if targets.dim() == 5:
            targets = targets.squeeze(1)
        ce_loss = self.ce_loss(logits, targets.long())
        pred_probs = torch.softmax(logits, dim=1)
        dice_loss = self.dice_loss(pred_probs, targets)
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return total_loss

    def get_individual_losses(self, logits, targets):
        if targets.dim() == 5:
            targets = targets.squeeze(1)
        ce_loss = self.ce_loss(logits, targets.long())
        pred_probs = torch.softmax(logits, dim=1)
        dice_loss = self.dice_loss(pred_probs, targets)
        return {'ce_loss': ce_loss.item(), 'dice_loss': dice_loss.item(), 'total_loss': (self.ce_weight * ce_loss + self.dice_weight * dice_loss).item()}

def calculate_dice_score_3d(pred: torch.Tensor, target: torch.Tensor, smooth: float=1e-06) -> float:
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    batch_size = pred.shape[0]
    dice_scores = []
    for i in range(batch_size):
        pred_sample = pred_binary[i].flatten()
        target_sample = target_binary[i].flatten()
        intersection = (pred_sample * target_sample).sum()
        union = pred_sample.sum() + target_sample.sum()
        if union == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice = dice.item() if hasattr(dice, 'item') else float(dice)
        dice_scores.append(dice)
    return np.mean(dice_scores)

def calculate_iou_3d(pred: torch.Tensor, target: torch.Tensor, smooth: float=1e-06) -> float:
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    batch_size = pred.shape[0]
    iou_scores = []
    for i in range(batch_size):
        pred_sample = pred_binary[i].flatten()
        target_sample = target_binary[i].flatten()
        intersection = (pred_sample * target_sample).sum()
        union = pred_sample.sum() + target_sample.sum() - intersection
        if union == 0:
            iou = 1.0
        else:
            iou = (intersection + smooth) / (union + smooth)
            iou = iou.item() if hasattr(iou, 'item') else float(iou)
        iou_scores.append(iou)
    return np.mean(iou_scores)

def calculate_sensitivity_3d(pred: torch.Tensor, target: torch.Tensor, threshold: float=0.5, eps: float=1e-06) -> float:
    if pred.dim() == 5 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.dim() == 5 and target.shape[1] == 1:
        target = target[:, 0]
    pred_bin = pred > threshold
    target_bin = target > 0.5
    tp = (pred_bin & target_bin).sum(dim=(1, 2, 3)).float()
    fn = (~pred_bin & target_bin).sum(dim=(1, 2, 3)).float()
    sens = (tp + eps) / (tp + fn + eps)
    return sens.mean().item()

def calculate_ppv_3d(pred: torch.Tensor, target: torch.Tensor, smooth: float=1e-06) -> float:
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    batch_size = pred.shape[0]
    ppv_scores = []
    for i in range(batch_size):
        pred_sample = pred_binary[i].flatten()
        target_sample = target_binary[i].flatten()
        true_positive = (pred_sample * target_sample).sum()
        predicted_positive = pred_sample.sum()
        if predicted_positive == 0:
            ppv = 0.0
        else:
            ppv = true_positive / predicted_positive
            ppv = ppv.item() if hasattr(ppv, 'item') else float(ppv)
        ppv_scores.append(ppv)
    return np.mean(ppv_scores)

def calculate_hd95_3d(pred: torch.Tensor, target: torch.Tensor) -> float:
    try:
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()
        pred_np = pred_binary.squeeze().cpu().numpy().astype(bool)
        target_np = target_binary.squeeze().cpu().numpy().astype(bool)
        if not np.any(pred_np) and (not np.any(target_np)):
            return 0.0
        elif not np.any(pred_np) or not np.any(target_np):
            return 373.1287
        hd95 = compute_hausdorff_distance_95_3d(pred_np, target_np)
        return float(hd95)
    except Exception as e:
        print(f'HD95 calculation failed: {e}')
        return 373.1287

def calculate_all_metrics_3d(pred: torch.Tensor, target: torch.Tensor) -> dict:
    metrics = {}
    if pred.shape[1] == 2:
        pred_probs = torch.softmax(pred, dim=1)
        pred_foreground = pred_probs[:, 1:2]
    else:
        pred_foreground = torch.sigmoid(pred)
    if target.dim() == 4:
        target_foreground = target.unsqueeze(1).float()
    else:
        target_foreground = target.float()
    metrics['dice'] = calculate_dice_score_3d(pred_foreground, target_foreground)
    metrics['iou'] = calculate_iou_3d(pred_foreground, target_foreground)
    metrics['sensitivity'] = calculate_sensitivity_3d(pred_foreground, target_foreground)
    metrics['ppv'] = calculate_ppv_3d(pred_foreground, target_foreground)
    hd95_values = []
    batch_size = pred_foreground.shape[0]
    for i in range(batch_size):
        hd95_val = calculate_hd95_3d(pred_foreground[i], target_foreground[i])
        hd95_values.append(hd95_val)
    metrics['hd95'] = np.mean(hd95_values)
    return metrics

def train_epoch_3d(model: nn.Module, train_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, device: torch.device, epoch: int, total_epochs: int) -> tuple:
    model.train()
    epoch_loss = 0.0
    metrics_list = {'dice': [], 'iou': [], 'sensitivity': [], 'ppv': [], 'hd95': []}
    num_batches = len(train_loader)
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs} [Train]', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    for i, (images, masks, _) in enumerate(train_pbar):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            batch_metrics = calculate_all_metrics_3d(outputs, masks)
            for key in metrics_list:
                metrics_list[key].append(batch_metrics[key])
        current_lr = optimizer.param_groups[0]['lr']
        epoch_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{epoch_loss / (i + 1):.4f}', 'dice': f'{batch_metrics['dice']:.3f}', 'lr': f'{current_lr:.2e}'})
    avg_loss = epoch_loss / num_batches
    avg_metrics = {}
    for key in metrics_list:
        avg_metrics[key] = np.mean(metrics_list[key])
        avg_metrics[f'{key}_std'] = np.std(metrics_list[key])
    return (avg_loss, avg_metrics)

def validate_epoch_3d(model: nn.Module, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device, epoch: int, total_epochs: int) -> tuple:
    model.eval()
    val_loss = 0.0
    metrics_list = {'dice': [], 'iou': [], 'sensitivity': [], 'ppv': [], 'hd95': []}
    num_batches = len(val_loader)
    val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{total_epochs} [Val]', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    with torch.no_grad():
        for i, (images, masks, _) in enumerate(val_pbar):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            batch_metrics = calculate_all_metrics_3d(outputs, masks)
            val_loss += loss.item()
            for key in metrics_list:
                metrics_list[key].append(batch_metrics[key])
            val_pbar.set_postfix({'val_loss': f'{val_loss / (i + 1):.4f}', 'dice': f'{batch_metrics['dice']:.3f}', 'iou': f'{batch_metrics['iou']:.3f}', 'sens': f'{batch_metrics['sensitivity']:.3f}', 'ppv': f'{batch_metrics['ppv']:.3f}'})
    avg_loss = val_loss / num_batches
    avg_metrics = {}
    for key in metrics_list:
        avg_metrics[key] = np.mean(metrics_list[key])
        avg_metrics[f'{key}_std'] = np.std(metrics_list[key])
    return (avg_loss, avg_metrics)

def test_model_3d(model: nn.Module, test_loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device, dataset_type: str) -> dict:
    model.eval()
    test_loss = 0.0
    metrics_list = {'dice': [], 'iou': [], 'sensitivity': [], 'ppv': [], 'hd95': []}
    num_batches = len(test_loader)
    print(f'\nEvaluating model on test set ({dataset_type})...')
    test_pbar = tqdm(test_loader, desc='Testing', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    with torch.no_grad():
        for i, (images, masks, _) in enumerate(test_pbar):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            batch_metrics = calculate_all_metrics_3d(outputs, masks)
            test_loss += loss.item()
            for key in metrics_list:
                metrics_list[key].append(batch_metrics[key])
            test_pbar.set_postfix({'test_loss': f'{test_loss / (i + 1):.4f}', 'dice': f'{batch_metrics['dice']:.3f}', 'iou': f'{batch_metrics['iou']:.3f}', 'sens': f'{batch_metrics['sensitivity']:.3f}', 'ppv': f'{batch_metrics['ppv']:.3f}'})
    avg_loss = test_loss / num_batches
    avg_metrics = {}
    for key in metrics_list:
        avg_metrics[key] = np.mean(metrics_list[key])
        avg_metrics[f'{key}_std'] = np.std(metrics_list[key])
    print(f'\nTest Results ({dataset_type}):')
    print(f'  Test Loss: {avg_loss:.4f}')
    print(f'  Dice Score: {avg_metrics['dice']:.4f} ± {avg_metrics['dice_std']:.4f}')
    print(f'  IoU: {avg_metrics['iou']:.4f} ± {avg_metrics['iou_std']:.4f}')
    print(f'  Sensitivity: {avg_metrics['sensitivity']:.4f} ± {avg_metrics['sensitivity_std']:.4f}')
    print(f'  PPV (Precision): {avg_metrics['ppv']:.4f} ± {avg_metrics['ppv_std']:.4f}')
    print(f'  HD95: {avg_metrics['hd95']:.2f} ± {avg_metrics['hd95_std']:.2f}')
    return avg_metrics

def train_3d(args):
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if hasattr(args, 'dataset_type') and args.dataset_type:
        train_loader, test_loader, dataset_type = get_ct_data_loaders(dataset_type=args.dataset_type, subfolder=getattr(args, 'subfolder', None), base_dir=getattr(args, 'base_dir', '/home/yuwenjing/data/tta_dataset'), batch_size_train=args.batch_train, batch_size_val=getattr(args, 'batch_test', args.batch_train), num_workers=args.num_workers, train_split=args.train_split, image_size=(args.image_size, args.image_size, args.image_size), spacing=getattr(args, 'spacing', (1.0, 1.0, 1.0)), intensity_range=getattr(args, 'intensity_range', (-200, 400)), cache_rate=getattr(args, 'cache_rate', 0.0))
    else:
        train_loader, test_loader, dataset_type = get_ct_data_loaders(image_dir=args.image_dir, mask_dir=args.mask_dir, batch_size_train=args.batch_train, batch_size_val=getattr(args, 'batch_test', args.batch_train), num_workers=args.num_workers, train_split=args.train_split, image_size=(args.image_size, args.image_size, args.image_size), spacing=getattr(args, 'spacing', (1.0, 1.0, 1.0)), intensity_range=getattr(args, 'intensity_range', (-200, 400)), cache_rate=getattr(args, 'cache_rate', 0.0))
    print(f'Dataset: {dataset_type}')
    print(f'Training batches: {len(train_loader)}')
    print(f'Test batches: {len(test_loader)}')
    model = UNet3d(in_chns=1, n_classes=2).to(device)
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'Model parameters: {total_params:,} total, {trainable_params:,} trainable')
    class_weights = torch.tensor([0.2, 0.8], device=device)
    criterion = SegmentationLoss3D(ce_weight=1.0, dice_weight=1.0, class_weights=class_weights, smooth=1e-06, ignore_background=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(args.min_lr / args.lr, cosine_factor)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    print(f'\nStarting training for {args.epochs} epochs...')
    print(f'Early stopping patience: {args.patience}')
    for epoch in range(args.epochs):
        if patience_counter >= args.patience:
            print(f'\nEarly stopping triggered after {epoch} epochs (patience: {args.patience})')
            break
        train_loss, train_metrics = train_epoch_3d(model, train_loader, criterion, optimizer, scheduler, device, epoch, args.epochs)
        test_loss, test_metrics = validate_epoch_3d(model, test_loader, criterion, device, epoch, args.epochs)
        print(f'Epoch [{epoch + 1}/{args.epochs}] Results:')
        print(f'  Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f} ± {train_metrics['dice_std']:.4f}')
        print(f'  Test  - Loss: {test_loss:.4f}')
        print(f'    Dice: {test_metrics['dice']:.4f} ± {test_metrics['dice_std']:.4f}')
        print(f'    IoU: {test_metrics['iou']:.4f} ± {test_metrics['iou_std']:.4f}')
        print(f'    Sens: {test_metrics['sensitivity']:.4f} ± {test_metrics['sensitivity_std']:.4f}')
        print(f'    PPV: {test_metrics['ppv']:.4f} ± {test_metrics['ppv_std']:.4f}')
        print(f'    HD95: {test_metrics['hd95']:.2f} ± {test_metrics['hd95_std']:.2f}')
        test_dice = test_metrics['dice']
        if test_dice > best_dice:
            best_dice = test_dice
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_path = os.path.join(args.checkpoint_dir, f'unet3d_best_{dataset_type}.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_dice': best_dice, 'test_loss': test_loss, 'test_metrics': test_metrics, 'train_metrics': train_metrics, 'args': vars(args)}, best_model_path)
            log_content = f'[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n'
            log_content += f'New Best Model - Epoch {epoch + 1}\n'
            log_content += f'Dataset: {dataset_type}\n'
            log_content += f'Test Dice: {test_metrics['dice']:.4f} ± {test_metrics['dice_std']:.4f}\n'
            log_content += f'Test IoU: {test_metrics['iou']:.4f} ± {test_metrics['iou_std']:.4f}\n'
            log_content += f'Test Sensitivity: {test_metrics['sensitivity']:.4f} ± {test_metrics['sensitivity_std']:.4f}\n'
            log_content += f'Test PPV: {test_metrics['ppv']:.4f} ± {test_metrics['ppv_std']:.4f}\n'
            log_content += f'Test HD95: {test_metrics['hd95']:.2f} ± {test_metrics['hd95_std']:.2f}\n'
            log_content += f'Test Loss: {test_loss:.4f}\n'
            log_content += '-' * 50 + '\n'
            log_file = os.path.join(args.checkpoint_dir, f'training_log_{dataset_type}.txt')
            with open(log_file, 'a') as f:
                f.write(log_content)
            print(f'🏆 New best model saved! Test Dice: {test_dice:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{args.patience}')
    print(f'\n{'=' * 60}')
    print('Training completed!')
    print(f'Best model from epoch {best_epoch} with Test Dice: {best_dice:.4f}')
    best_model_path = os.path.join(args.checkpoint_dir, f'unet3d_best_{dataset_type}.pth')
    if os.path.exists(best_model_path):
        print('\nLoading best model for final evaluation...')
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        final_test_metrics = test_model_3d(model, test_loader, criterion, device, dataset_type)
        final_log_content = f'\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\nFINAL TEST RESULTS - {dataset_type}\nBest Epoch: {best_epoch}\nFinal Test Dice: {final_test_metrics['dice']:.4f} ± {final_test_metrics['dice_std']:.4f}\nFinal Test IoU: {final_test_metrics['iou']:.4f} ± {final_test_metrics['iou_std']:.4f}\nFinal Test Sensitivity: {final_test_metrics['sensitivity']:.4f} ± {final_test_metrics['sensitivity_std']:.4f}\nFinal Test PPV: {final_test_metrics['ppv']:.4f} ± {final_test_metrics['ppv_std']:.4f}\nFinal Test HD95: {final_test_metrics['hd95']:.2f} ± {final_test_metrics['hd95_std']:.2f}\n{'=' * 60}\n'
        log_file = os.path.join(args.checkpoint_dir, f'training_log_{dataset_type}.txt')
        with open(log_file, 'a') as f:
            f.write(final_log_content)
        print(f'\nTraining and evaluation completed for {dataset_type} dataset!')
        print(f'Best model saved to: {best_model_path}')
        print(f'Training log saved to: {log_file}')
    else:
        print(f'Warning: Best model file not found: {best_model_path}')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='3D CT Segmentation Training')
    parser.add_argument('--dataset_type', type=str, default='CT', help='Dataset type')
    parser.add_argument('--base_dir', type=str, default='/home/yuwenjing/data/tta_dataset', help='Base directory')
    parser.add_argument('--image_dir', type=str, help='Image directory (if not using dataset_type)')
    parser.add_argument('--mask_dir', type=str, help='Mask directory (if not using dataset_type)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_train', type=int, default=2, help='Training batch size')
    parser.add_argument('--batch_test', type=int, default=2, help='Test batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-05, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--min_lr', type=float, default=1e-06, help='Minimum learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--train_split', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--image_size', type=int, default=96, help='Image size (cubic)')
    parser.add_argument('--spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0], help='Voxel spacing')
    parser.add_argument('--intensity_range', type=float, nargs=2, default=[-200, 400], help='Intensity range')
    parser.add_argument('--cache_rate', type=float, default=0.0, help='Cache rate for MONAI')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_3d', help='Checkpoint directory')
    args = parser.parse_args()
    args.spacing = tuple(args.spacing)
    args.intensity_range = tuple(args.intensity_range)
    train_3d(args)
