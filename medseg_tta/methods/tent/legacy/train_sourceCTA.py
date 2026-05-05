import os
import math
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from dataset_CTA import get_cta_data_loaders
from nnunet import PlainConvUNet as nnunet

def hd95_surface_mm(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    if not pred.any() and (not gt.any()):
        return 0.0
    if not pred.any() or not gt.any():
        return 373.1287
    struct = np.ones((3, 3, 3), dtype=bool)
    pred_surf = pred ^ binary_erosion(pred, structure=struct, border_value=0)
    gt_surf = gt ^ binary_erosion(gt, structure=struct, border_value=0)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)
    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    d_gt_to_pred = dt_pred[gt_surf]
    d_pred_to_gt = dt_gt[pred_surf]
    hd95 = max(np.percentile(d_gt_to_pred, 95), np.percentile(d_pred_to_gt, 95))
    return float(hd95)

class SegmentationLoss3D(nn.Module):

    def __init__(self, ce_weight: float=1.0, dice_weight: float=1.0, class_weights: torch.Tensor | None=None, smooth: float=1e-06, ignore_background: bool=True):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    def dice_loss(self, pred_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C = pred_probs.shape[:2]
        targets_one_hot = torch.zeros_like(pred_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        start_idx = 1 if self.ignore_background else 0
        dice_losses = []
        for c in range(start_idx, C):
            p = pred_probs[:, c]
            t = targets_one_hot[:, c]
            inter = torch.sum(p * t, dim=(1, 2, 3))
            psum = torch.sum(p, dim=(1, 2, 3))
            tsum = torch.sum(t, dim=(1, 2, 3))
            dice_c = (2 * inter + self.smooth) / (psum + tsum + self.smooth)
            dice_losses.append(1 - dice_c)
        if not dice_losses:
            return torch.tensor(0.0, device=pred_probs.device)
        return torch.mean(torch.stack(dice_losses, dim=0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 5:
            targets = targets.squeeze(1)
        ce = self.ce_loss(logits, targets.long())
        probs = torch.softmax(logits, dim=1)
        dl = self.dice_loss(probs, targets)
        return self.ce_weight * ce + self.dice_weight * dl

    def get_individual_losses(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        if targets.dim() == 5:
            targets = targets.squeeze(1)
        ce = self.ce_loss(logits, targets.long()).item()
        probs = torch.softmax(logits, dim=1)
        dl = self.dice_loss(probs, targets).item()
        return {'ce_loss': ce, 'dice_loss': dl, 'total_loss': self.ce_weight * ce + self.dice_weight * dl}

def _binary_metrics(pred_bin: torch.Tensor, tgt_bin: torch.Tensor, eps: float=1e-06) -> dict:
    B = pred_bin.shape[0]
    dice_list, iou_list, sens_list, ppv_list = ([], [], [], [])
    for i in range(B):
        p = pred_bin[i].flatten()
        t = tgt_bin[i].flatten()
        tp = (p & t).sum().float()
        fp = (p & ~t).sum().float()
        fn = (~p & t).sum().float()
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        sens = (tp + eps) / (tp + fn + eps)
        ppv = (tp + eps) / (tp + fp + eps)
        dice_list.append(dice.item())
        iou_list.append(iou.item())
        sens_list.append(sens.item())
        ppv_list.append(ppv.item())
    return {'dice': float(np.mean(dice_list)), 'iou': float(np.mean(iou_list)), 'sensitivity': float(np.mean(sens_list)), 'ppv': float(np.mean(ppv_list))}

def _hd95_for_class_mm(pred_bin: torch.Tensor, tgt_bin: torch.Tensor, spacing) -> float:
    vals = []
    B = pred_bin.shape[0]
    for i in range(B):
        p = pred_bin[i].cpu().numpy().astype(bool)
        t = tgt_bin[i].cpu().numpy().astype(bool)
        vals.append(hd95_surface_mm(p, t, spacing=spacing))
    return float(np.mean(vals)) if vals else 373.1287

def calculate_all_metrics_3d_multiclass_mm(logits: torch.Tensor, target: torch.Tensor, spacing=(1.0, 1.0, 1.0), num_classes: int=3) -> dict:
    if target.dim() == 5:
        target = target[:, 0]
    probs = torch.softmax(logits, dim=1)
    pred_lbl = torch.argmax(probs, dim=1)
    metrics = {}
    per_class = {}
    fg_classes = [1, 2]
    dice_mean, iou_mean, sens_mean, ppv_mean, hd95_mean = ([], [], [], [], [])
    for c in fg_classes:
        pred_bin = pred_lbl == c
        tgt_bin = target == c
        m = _binary_metrics(pred_bin, tgt_bin)
        hd = _hd95_for_class_mm(pred_bin, tgt_bin, spacing)
        name = 'TL' if c == 1 else 'FL'
        per_class[f'dice_{name}'] = m['dice']
        per_class[f'iou_{name}'] = m['iou']
        per_class[f'sensitivity_{name}'] = m['sensitivity']
        per_class[f'ppv_{name}'] = m['ppv']
        per_class[f'hd95_{name}'] = hd
        dice_mean.append(m['dice'])
        iou_mean.append(m['iou'])
        sens_mean.append(m['sensitivity'])
        ppv_mean.append(m['ppv'])
        hd95_mean.append(hd)
    metrics.update(per_class)
    metrics['dice_mean'] = float(np.mean(dice_mean))
    metrics['iou_mean'] = float(np.mean(iou_mean))
    metrics['sensitivity_mean'] = float(np.mean(sens_mean))
    metrics['ppv_mean'] = float(np.mean(ppv_mean))
    metrics['hd95_mean'] = float(np.mean(hd95_mean))
    return metrics

def train_epoch_3d(model, train_loader, criterion, optimizer, scheduler, device, epoch, total_epochs, spacing):
    model.train()
    epoch_loss = 0.0
    total = {'dice_TL': 0.0, 'dice_FL': 0.0, 'iou_TL': 0.0, 'iou_FL': 0.0, 'sensitivity_TL': 0.0, 'sensitivity_FL': 0.0, 'ppv_TL': 0.0, 'ppv_FL': 0.0, 'hd95_TL': 0.0, 'hd95_FL': 0.0}
    n = len(train_loader)
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs} [Train]', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    for i, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            m = calculate_all_metrics_3d_multiclass_mm(outputs, masks, spacing=spacing, num_classes=3)
            for k in total:
                total[k] += m[k]
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{epoch_loss / (i + 1):.4f}', 'TL': f'{m['dice_TL']:.3f}', 'FL': f'{m['dice_FL']:.3f}', 'lr': f'{optimizer.param_groups[0]['lr']:.2e}'})
    avg_loss = epoch_loss / n
    avg_metrics = {k: v / n for k, v in total.items()}
    return (avg_loss, avg_metrics)

def validate_epoch_3d(model, val_loader, criterion, device, epoch, total_epochs, spacing):
    model.eval()
    val_loss = 0.0
    total = {'dice_TL': 0.0, 'dice_FL': 0.0, 'iou_TL': 0.0, 'iou_FL': 0.0, 'sensitivity_TL': 0.0, 'sensitivity_FL': 0.0, 'ppv_TL': 0.0, 'ppv_FL': 0.0, 'hd95_TL': 0.0, 'hd95_FL': 0.0}
    n = len(val_loader)
    pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{total_epochs} [Val]', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    with torch.no_grad():
        for i, (images, masks, _) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            m = calculate_all_metrics_3d_multiclass_mm(outputs, masks, spacing=spacing, num_classes=3)
            val_loss += loss.item()
            for k in total:
                total[k] += m[k]
            pbar.set_postfix({'val_loss': f'{val_loss / (i + 1):.4f}', 'TL': f'{m['dice_TL']:.3f}', 'FL': f'{m['dice_FL']:.3f}', 'iou_TL': f'{m['iou_TL']:.3f}', 'iou_FL': f'{m['iou_FL']:.3f}'})
    avg_loss = val_loss / n
    avg_metrics = {k: v / n for k, v in total.items()}
    return (avg_loss, avg_metrics)

def test_model_3d(model, test_loader, criterion, device, dataset_type: str, spacing):
    model.eval()
    test_loss = 0.0
    total = {'dice_TL': 0.0, 'dice_FL': 0.0, 'iou_TL': 0.0, 'iou_FL': 0.0, 'sensitivity_TL': 0.0, 'sensitivity_FL': 0.0, 'ppv_TL': 0.0, 'ppv_FL': 0.0, 'hd95_TL': 0.0, 'hd95_FL': 0.0}
    n = len(test_loader)
    print(f'\nEvaluating model on test set ({dataset_type})...')
    pbar = tqdm(test_loader, desc='Testing', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    with torch.no_grad():
        for i, (images, masks, _) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            m = calculate_all_metrics_3d_multiclass_mm(outputs, masks, spacing=spacing, num_classes=3)
            test_loss += loss.item()
            for k in total:
                total[k] += m[k]
            pbar.set_postfix({'test_loss': f'{test_loss / (i + 1):.4f}', 'TL': f'{m['dice_TL']:.3f}', 'FL': f'{m['dice_FL']:.3f}', 'iou_TL': f'{m['iou_TL']:.3f}', 'iou_FL': f'{m['iou_FL']:.3f}'})
    avg_loss = test_loss / n
    avg = {k: v / n for k, v in total.items()}
    print(f'\nTest Results ({dataset_type}):')
    print(f'  Test Loss: {avg_loss:.4f}')
    print(f'\n  True Lumen (TL) Metrics:')
    print(f'    Dice: {avg['dice_TL']:.4f}')
    print(f'    IoU:  {avg['iou_TL']:.4f}')
    print(f'    Sens: {avg['sensitivity_TL']:.4f}')
    print(f'    PPV:  {avg['ppv_TL']:.4f}')
    print(f'    HD95: {avg['hd95_TL']:.2f} mm')
    print(f'\n  False Lumen (FL) Metrics:')
    print(f'    Dice: {avg['dice_FL']:.4f}')
    print(f'    IoU:  {avg['iou_FL']:.4f}')
    print(f'    Sens: {avg['sensitivity_FL']:.4f}')
    print(f'    PPV:  {avg['ppv_FL']:.4f}')
    print(f'    HD95: {avg['hd95_FL']:.2f} mm')
    return avg

def train_3d(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    train_loader, val_loader, dataset_type = get_cta_data_loaders(image_dir=args.image_dir, mask_dir=args.mask_dir, batch_size_train=args.batch_train, batch_size_val=getattr(args, 'batch_test', args.batch_train), num_workers=args.num_workers, train_split=args.train_split, image_size=(args.image_size, args.image_size, args.image_size), spacing=tuple(args.spacing), window_level=args.window_level, window_width=args.window_width, normalize=True, if_flt=bool(args.if_flt))
    print(f'Dataset: {dataset_type}')
    print(f'Training batches: {len(train_loader)}')
    print(f'Validation batches: {len(val_loader)}')
    model = nnunet(1, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 3, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'Model parameters: {total_params:,} total, {trainable_params:,} trainable')
    class_weights = None
    if args.class_weights is not None and len(args.class_weights) == 3:
        class_weights = torch.tensor(args.class_weights, device=device, dtype=torch.float32)
    criterion = SegmentationLoss3D(ce_weight=1.0, dice_weight=1.0, class_weights=class_weights, smooth=1e-06, ignore_background=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(args.min_lr / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    spacing_mm = tuple(args.spacing)
    print(f'\nStarting training for {args.epochs} epochs...')
    print(f'Early stopping patience: {args.patience}')
    for epoch in range(args.epochs):
        if patience_counter >= args.patience:
            print(f'\nEarly stopping triggered after {epoch} epochs (patience: {args.patience})')
            break
        train_loss, train_metrics = train_epoch_3d(model, train_loader, criterion, optimizer, scheduler, device, epoch, args.epochs, spacing=spacing_mm)
        val_loss, val_metrics = validate_epoch_3d(model, val_loader, criterion, device, epoch, args.epochs, spacing=spacing_mm)
        print(f'Epoch [{epoch + 1}/{args.epochs}] Results:')
        print(f'  Train - Loss: {train_loss:.4f}')
        print(f'    TL: Dice={train_metrics['dice_TL']:.4f}, IoU={train_metrics['iou_TL']:.4f}, HD95={train_metrics['hd95_TL']:.2f}mm')
        print(f'    FL: Dice={train_metrics['dice_FL']:.4f}, IoU={train_metrics['iou_FL']:.4f}, HD95={train_metrics['hd95_FL']:.2f}mm')
        print(f'  Val   - Loss: {val_loss:.4f}')
        print(f'    TL: Dice={val_metrics['dice_TL']:.4f}, IoU={val_metrics['iou_TL']:.4f}, Sens={val_metrics['sensitivity_TL']:.4f}, PPV={val_metrics['ppv_TL']:.4f}, HD95={val_metrics['hd95_TL']:.2f}mm')
        print(f'    FL: Dice={val_metrics['dice_FL']:.4f}, IoU={val_metrics['iou_FL']:.4f}, Sens={val_metrics['sensitivity_FL']:.4f}, PPV={val_metrics['ppv_FL']:.4f}, HD95={val_metrics['hd95_FL']:.2f}mm')
        cur_dice = (val_metrics['dice_TL'] + val_metrics['dice_FL']) / 2.0
        if cur_dice > best_dice:
            best_dice = cur_dice
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_path = os.path.join(args.checkpoint_dir, f'unet3d_best_CTA.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_dice': best_dice, 'val_loss': val_loss, 'val_metrics': val_metrics, 'train_metrics': train_metrics, 'args': vars(args)}, best_model_path)
            log_content = f'[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\nNew Best Model - Epoch {epoch + 1}\nDataset: CTA\nVal Loss: {val_loss:.4f}\nTrue Lumen (TL):\n  Dice: {val_metrics['dice_TL']:.4f}\n  IoU:  {val_metrics['iou_TL']:.4f}\n  Sens: {val_metrics['sensitivity_TL']:.4f}\n  PPV:  {val_metrics['ppv_TL']:.4f}\n  HD95: {val_metrics['hd95_TL']:.2f} mm\nFalse Lumen (FL):\n  Dice: {val_metrics['dice_FL']:.4f}\n  IoU:  {val_metrics['iou_FL']:.4f}\n  Sens: {val_metrics['sensitivity_FL']:.4f}\n  PPV:  {val_metrics['ppv_FL']:.4f}\n  HD95: {val_metrics['hd95_FL']:.2f} mm\nAvg Dice: {best_dice:.4f}\n{'-' * 50}\n'
            with open(os.path.join(args.checkpoint_dir, 'training_log_CTA.txt'), 'a') as f:
                f.write(log_content)
            print(f'🏆 New best model saved! Val Dice(mean): {best_dice:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{args.patience}')
    print(f'\n{'=' * 60}')
    print('Training completed!')
    print(f'Best epoch: {best_epoch} | Best Val Dice(mean): {best_dice:.4f}')
    best_model_path = os.path.join(args.checkpoint_dir, 'unet3d_best_CTA.pth')
    if os.path.exists(best_model_path):
        print('\nLoading best model for final evaluation on validation set...')
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        final_metrics = test_model_3d(model, val_loader, criterion, device, 'CTA', spacing=spacing_mm)
        final_log = f'\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\nFINAL EVAL RESULTS - CTA (on validation set)\nBest Epoch: {best_epoch}\nTrue Lumen (TL):\n  Dice: {final_metrics['dice_TL']:.4f}\n  IoU:  {final_metrics['iou_TL']:.4f}\n  Sens: {final_metrics['sensitivity_TL']:.4f}\n  PPV:  {final_metrics['ppv_TL']:.4f}\n  HD95: {final_metrics['hd95_TL']:.2f} mm\nFalse Lumen (FL):\n  Dice: {final_metrics['dice_FL']:.4f}\n  IoU:  {final_metrics['iou_FL']:.4f}\n  Sens: {final_metrics['sensitivity_FL']:.4f}\n  PPV:  {final_metrics['ppv_FL']:.4f}\n  HD95: {final_metrics['hd95_FL']:.2f} mm\n{'=' * 60}\n'
        with open(os.path.join(args.checkpoint_dir, 'training_log_CTA.txt'), 'a') as f:
            f.write(final_log)
        print('\nEvaluation completed.')
        print(f'Best model: {best_model_path}')
    else:
        print(f'Warning: Best model file not found: {best_model_path}')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='3D CTA Segmentation (3 classes: BG/TL/FL)')
    parser.add_argument('--image_dir', type=str, default='/home/yuwenjing/data/imageTBAD', help='CTA image directory')
    parser.add_argument('--mask_dir', type=str, default='/home/yuwenjing/data/imageTBAD', help='CTA label directory')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_train', type=int, default=4)
    parser.add_argument('--batch_test', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-07)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--train_split', type=float, default=0.9)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0], help='(z,y,x) mm')
    parser.add_argument('--window_level', type=float, default=1024.0)
    parser.add_argument('--window_width', type=float, default=4095.0)
    parser.add_argument('--if_flt', type=int, default=0, help='1=不处理; 0=将标签4置0')
    parser.add_argument('--class_weights', type=float, nargs='+', default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_3d')
    args = parser.parse_args()
    train_3d(args)
