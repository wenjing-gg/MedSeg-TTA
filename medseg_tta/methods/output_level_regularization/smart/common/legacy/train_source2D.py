import os
import math
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np

# 导入自定义模块
from dataset2D import get_data_loaders_2d
from unet2d import UNet2d


def compute_hausdorff_distance_95(pred_mask, target_mask):
    """
    计算95%豪斯多夫距离的简化实现
    """
    try:
        # 获取边界点
        pred_points = np.argwhere(pred_mask)
        target_points = np.argwhere(target_mask)

        if len(pred_points) == 0 and len(target_points) == 0:
            return 0.0
        elif len(pred_points) == 0 or len(target_points) == 0:
            return 373.1287  # 大值表示完全不匹配

        # 计算从pred到target的距离
        distances_pred_to_target = []
        for pred_point in pred_points:
            min_dist = np.min(np.sqrt(np.sum((target_points - pred_point) ** 2, axis=1)))
            distances_pred_to_target.append(min_dist)

        # 计算从target到pred的距离
        distances_target_to_pred = []
        for target_point in target_points:
            min_dist = np.min(np.sqrt(np.sum((pred_points - target_point) ** 2, axis=1)))
            distances_target_to_pred.append(min_dist)

        # 合并所有距离
        all_distances = distances_pred_to_target + distances_target_to_pred

        # 计算95%分位数
        hd95 = np.percentile(all_distances, 95)
        return float(hd95)

    except Exception:
        return 373.1287


class SegmentationLoss2D(nn.Module):
    """
    自定义2D分割损失函数，结合交叉熵损失和Dice损失
    专门为2D胸部X光分割任务设计
    """
    def __init__(self,
                 ce_weight=1.0,
                 dice_weight=1.0,
                 class_weights=None,
                 smooth=1e-6,
                 ignore_background=True):
        """
        Args:
            ce_weight: 交叉熵损失权重
            dice_weight: Dice损失权重
            class_weights: 类别权重，用于处理类别不平衡
            smooth: 平滑项，避免除零
            ignore_background: 是否在Dice计算中忽略背景类
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ignore_background = ignore_background

        # 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    def dice_loss(self, pred_probs, targets):
        """
        计算Dice损失
        Args:
            pred_probs: 预测概率 [B, C, H, W]
            targets: 目标标签 [B, H, W]
        """
        batch_size, num_classes = pred_probs.shape[0], pred_probs.shape[1]

        # 将targets转换为one-hot编码 [B, C, H, W]
        targets_one_hot = torch.zeros_like(pred_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)

        dice_scores = []

        # 计算每个类别的Dice分数
        start_idx = 1 if self.ignore_background else 0
        for class_idx in range(start_idx, num_classes):
            pred_class = pred_probs[:, class_idx, :, :]  # [B, H, W]
            target_class = targets_one_hot[:, class_idx, :, :]  # [B, H, W]

            # 计算交集和并集
            intersection = torch.sum(pred_class * target_class, dim=(1, 2))  # [B]
            pred_sum = torch.sum(pred_class, dim=(1, 2))  # [B]
            target_sum = torch.sum(target_class, dim=(1, 2))  # [B]

            # Dice系数
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            dice_scores.append(dice)

        if len(dice_scores) > 0:
            # 计算所有类别的平均Dice分数
            dice_scores = torch.stack(dice_scores, dim=1)  # [B, num_classes-1] or [B, num_classes]
            mean_dice = torch.mean(dice_scores)
            return 1.0 - mean_dice
        else:
            return torch.tensor(0.0, device=pred_probs.device, requires_grad=True)

    def forward(self, logits, targets):
        """
        前向传播
        Args:
            logits: 模型输出 [B, C, H, W]
            targets: 目标标签 [B, H, W] 或 [B, 1, H, W]
        """
        # 确保targets的形状正确
        if targets.dim() == 4:
            targets = targets.squeeze(1)  # [B, H, W]

        # 计算交叉熵损失
        ce_loss = self.ce_loss(logits, targets.long())

        # 计算Dice损失
        pred_probs = torch.softmax(logits, dim=1)
        dice_loss = self.dice_loss(pred_probs, targets)

        # 组合损失
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return total_loss

    def get_individual_losses(self, logits, targets):
        """
        返回各个损失分量，用于监控
        """
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        ce_loss = self.ce_loss(logits, targets.long())
        pred_probs = torch.softmax(logits, dim=1)
        dice_loss = self.dice_loss(pred_probs, targets)

        return {
            'ce_loss': ce_loss.item(),
            'dice_loss': dice_loss.item(),
            'total_loss': (self.ce_weight * ce_loss + self.dice_weight * dice_loss).item()
        }


# 2D分割指标计算函数
def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """计算2D Dice分数"""
    # pred已经是概率，直接二值化
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()

    # 计算批次中所有样本的Dice，然后取平均
    batch_size = pred.shape[0]
    dice_scores = []
    
    for i in range(batch_size):
        pred_sample = pred_binary[i].flatten()
        target_sample = target_binary[i].flatten()
        
        intersection = (pred_sample * target_sample).sum()
        union = pred_sample.sum() + target_sample.sum()
        
        if union == 0:
            # 修正：如果都是背景，Dice应该为1（完美匹配）
            dice = 1.0
        else:
            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice = dice.item() if hasattr(dice, 'item') else float(dice)
        
        dice_scores.append(dice)
    
    return np.mean(dice_scores)


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """计算2D IoU分数"""
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
            # 修正：如果都是背景，IoU应该为1（完美匹配）
            iou = 1.0
        else:
            iou = (intersection + smooth) / (union + smooth)
            iou = iou.item() if hasattr(iou, 'item') else float(iou)
        
        iou_scores.append(iou)
    
    return np.mean(iou_scores)


def calculate_sensitivity(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    Sensitivity / Recall / TPR for 2-D masks.

    Args
    ----
    pred : tensor
        * shape: [B, 1, H, W] **probabilities** (0‒1) -or- [B, H, W] binary.
    target : tensor
        * same shape as `pred` or without channel dim; binary {0,1}.
    threshold : float
        Binarisation threshold for `pred`.
    eps : float
        Small number to avoid division-by-zero.
    """
    # -------- shape unify --------
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]          # -> [B, H, W]
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0]      # -> [B, H, W]

    # -------- binarise --------
    pred_bin   = (pred > threshold)          # bool
    target_bin = (target > 0.5)              # bool  (GT 已是 0/1 时保持不变)

    # -------- TP & FN per-sample --------
    tp = (pred_bin & target_bin).sum(dim=(1, 2)).float()     # [B]
    fn = (~pred_bin & target_bin).sum(dim=(1, 2)).float()    # [B]

    # -------- sensitivity --------
    sens = (tp + eps) / (tp + fn + eps)      # [B]
    return sens.mean().item()


def calculate_ppv(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """计算2D PPV (Precision)"""
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
            # 修正：如果没有预测为正，PPV应该是未定义的，设为0更合理
            ppv = 0.0
        else:
            ppv = true_positive / predicted_positive
            ppv = ppv.item() if hasattr(ppv, 'item') else float(ppv)
        
        ppv_scores.append(ppv)
    
    return np.mean(ppv_scores)


def calculate_hd95(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算2D HD95距离"""
    try:
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()

        # 转换为numpy数组，去掉batch和channel维度
        pred_np = pred_binary.squeeze().cpu().numpy().astype(bool)
        target_np = target_binary.squeeze().cpu().numpy().astype(bool)

        # 检查是否有前景像素
        if not np.any(pred_np) and not np.any(target_np):
            return 0.0  # 都为空，完美匹配
        elif not np.any(pred_np) or not np.any(target_np):
            return 373.1287  # 一个为空一个不为空，返回大值

        # 计算HD95
        hd95 = compute_hausdorff_distance_95(pred_np, target_np)
        return float(hd95)

    except Exception as e:
        # 如果计算失败，返回默认值
        print(f"HD95 calculation failed: {e}")
        return 373.1287


def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """计算所有2D指标"""
    metrics = {}

    # 对于二分类，取前景类别的输出
    if pred.shape[1] == 2:  # [B, 2, H, W]
        # 使用softmax获取概率，然后取前景类
        pred_probs = torch.softmax(pred, dim=1)
        pred_foreground = pred_probs[:, 1:2]  # 取前景类概率 [B, 1, H, W]
    else:  # [B, 1, H, W]
        pred_foreground = torch.sigmoid(pred)  # 应用sigmoid获取概率

    # 确保target格式正确
    if target.dim() == 3:  # [B, H, W]
        target_foreground = target.unsqueeze(1).float()  # [B, 1, H, W]
    else:  # [B, 1, H, W]
        target_foreground = target.float()

    # 计算各项指标
    metrics['dice'] = calculate_dice_score(pred_foreground, target_foreground)
    metrics['iou'] = calculate_iou(pred_foreground, target_foreground)
    metrics['sensitivity'] = calculate_sensitivity(pred_foreground, target_foreground)
    metrics['ppv'] = calculate_ppv(pred_foreground, target_foreground)

    # HD95需要逐个样本计算然后平均
    hd95_values = []
    batch_size = pred_foreground.shape[0]
    for i in range(batch_size):
        hd95_val = calculate_hd95(pred_foreground[i], target_foreground[i])
        hd95_values.append(hd95_val)

    metrics['hd95'] = np.mean(hd95_values)

    return metrics


def train_epoch(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler,
                device: torch.device,
                epoch: int,
                total_epochs: int) -> tuple:
    """训练一个epoch"""
    model.train()
    epoch_loss = 0.0
    total_metrics = {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'ppv': 0.0, 'hd95': 0.0}
    num_batches = len(train_loader)

    train_pbar = tqdm(train_loader,
                     desc=f'Epoch {epoch+1}/{total_epochs} [Train]',
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    for i, (images, masks, _) in enumerate(train_pbar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, masks)

        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 计算指标 - 每个batch都计算
        with torch.no_grad():
            batch_metrics = calculate_all_metrics(outputs, masks)
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]

        current_lr = optimizer.param_groups[0]['lr']
        epoch_loss += loss.item()

        train_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{epoch_loss/(i+1):.4f}",
            'dice': f"{batch_metrics['dice']:.3f}",
            'lr': f"{current_lr:.2e}"
        })

    # 计算平均指标
    avg_loss = epoch_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

    return avg_loss, avg_metrics


def validate_epoch(model: nn.Module,
                  val_loader: torch.utils.data.DataLoader,
                  criterion: nn.Module,
                  device: torch.device,
                  epoch: int,
                  total_epochs: int) -> tuple:
    """验证一个epoch"""
    model.eval()
    val_loss = 0.0
    total_metrics = {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'ppv': 0.0, 'hd95': 0.0}
    num_batches = len(val_loader)

    val_pbar = tqdm(val_loader,
                   desc=f'Epoch {epoch+1}/{total_epochs} [Val]',
                   bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    with torch.no_grad():
        for i, (images, masks, _) in enumerate(val_pbar):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, masks)

            # 计算指标
            batch_metrics = calculate_all_metrics(outputs, masks)

            val_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]

            val_pbar.set_postfix({
                'val_loss': f"{val_loss/(i+1):.4f}",
                'dice': f"{batch_metrics['dice']:.3f}",
                'iou': f"{batch_metrics['iou']:.3f}",
                'sens': f"{batch_metrics['sensitivity']:.3f}",
                'ppv': f"{batch_metrics['ppv']:.3f}"
            })

    # 计算平均指标
    avg_loss = val_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

    return avg_loss, avg_metrics


def test_model(model: nn.Module,
               test_loader: torch.utils.data.DataLoader,
               criterion: nn.Module,
               device: torch.device,
               dataset_type: str) -> dict:
    """在测试集上评估模型"""
    model.eval()
    test_loss = 0.0
    total_metrics = {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'ppv': 0.0, 'hd95': 0.0}
    num_batches = len(test_loader)

    print(f"\nEvaluating model on test set ({dataset_type})...")
    test_pbar = tqdm(test_loader,
                    desc='Testing',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    with torch.no_grad():
        for i, (images, masks, _) in enumerate(test_pbar):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, masks)

            # 计算指标
            batch_metrics = calculate_all_metrics(outputs, masks)

            test_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]

            test_pbar.set_postfix({
                'test_loss': f"{test_loss/(i+1):.4f}",
                'dice': f"{batch_metrics['dice']:.3f}",
                'iou': f"{batch_metrics['iou']:.3f}",
                'sens': f"{batch_metrics['sensitivity']:.3f}",
                'ppv': f"{batch_metrics['ppv']:.3f}"
            })

    # 计算平均指标
    avg_loss = test_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

    # 打印测试结果
    print(f"\nTest Results ({dataset_type}):")
    print(f"  Test Loss: {avg_loss:.4f}")
    print(f"  Dice Score: {avg_metrics['dice']:.4f}")
    print(f"  IoU: {avg_metrics['iou']:.4f}")
    print(f"  Sensitivity: {avg_metrics['sensitivity']:.4f}")
    print(f"  PPV (Precision): {avg_metrics['ppv']:.4f}")
    print(f"  HD95: {avg_metrics['hd95']:.2f}")

    return avg_metrics


def train(args):
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
 
    # 创建数据加载器
    if hasattr(args, 'dataset_type') and args.dataset_type:
        # 使用数据集类型
        train_loader, test_loader, dataset_type = get_data_loaders_2d(
            dataset_type=args.dataset_type,
            subfolder=getattr(args, 'subfolder', None),
            base_dir=getattr(args, 'base_dir', r"/home/yuwenjing/data/tta_dataset"),
            batch_size_train=args.batch_train,
            batch_size_val=getattr(args, 'batch_test', args.batch_train),
            num_workers=args.num_workers,
            train_split=args.train_split,
            image_size=(args.image_size, args.image_size)
        )
    else:
        # 使用具体路径
        train_loader, test_loader, dataset_type = get_data_loaders_2d(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            batch_size_train=args.batch_train,
            batch_size_val=getattr(args, 'batch_test', args.batch_train),
            num_workers=args.num_workers,
            train_split=args.train_split,
            image_size=(args.image_size, args.image_size)
        )

    print(f"Dataset: {dataset_type}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 创建模型
    model = UNet2d(in_channels=1, n_classes=2).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # 修正：调整类别权重，使其更平衡
    class_weights = torch.tensor([0.2, 0.8], device=device)  # 稍微降低前景权重
    criterion = SegmentationLoss2D(
        ce_weight=1.0,
        dice_weight=1.0,
        class_weights=class_weights,
        smooth=1e-6,
        ignore_background=False
    )

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 创建学习率调度器
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(args.min_lr / args.lr, cosine_factor)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 训练循环
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Early stopping patience: {args.patience}")

    for epoch in range(args.epochs):
        # 早停检查
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (patience: {args.patience})")
            break

        # 训练
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, args.epochs
        )

        # 在测试集上验证
        test_loss, test_metrics = validate_epoch(
            model, test_loader, criterion, device, epoch, args.epochs
        )

        # 打印epoch结果
        print(f"Epoch [{epoch+1}/{args.epochs}] Results:")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f}, Dice: {test_metrics['dice']:.4f}, "
              f"IoU: {test_metrics['iou']:.4f}, Sens: {test_metrics['sensitivity']:.4f}, "
              f"PPV: {test_metrics['ppv']:.4f}, HD95: {test_metrics['hd95']:.2f}")

        # 保存最佳模型（基于测试集Dice分数）
        test_dice = test_metrics['dice']
        if test_dice > best_dice:
            best_dice = test_dice
            best_epoch = epoch + 1
            patience_counter = 0

            # 保存模型
            best_model_path = os.path.join(args.checkpoint_dir, f"unet2d_best_{dataset_type}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'test_loss': test_loss,
                'test_metrics': test_metrics,
                'train_metrics': train_metrics,
                'args': vars(args)
            }, best_model_path)

            # 记录最佳结果
            log_content = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
            log_content += f"New Best Model - Epoch {epoch+1}\n"
            log_content += f"Dataset: {dataset_type}\n"
            log_content += f"Test Dice: {test_metrics['dice']:.4f}\n"
            log_content += f"Test IoU: {test_metrics['iou']:.4f}\n"
            log_content += f"Test Sensitivity: {test_metrics['sensitivity']:.4f}\n"
            log_content += f"Test PPV: {test_metrics['ppv']:.4f}\n"
            log_content += f"Test HD95: {test_metrics['hd95']:.2f}\n"
            log_content += f"Test Loss: {test_loss:.4f}\n"
            log_content += "-" * 50 + "\n"

            log_file = os.path.join(args.checkpoint_dir, f"training_log_{dataset_type}.txt")
            with open(log_file, 'a') as f:
                f.write(log_content)

            print(f"🏆 New best model saved! Test Dice: {test_dice:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")

    # ----------------------------------------------
    # 训练完成后，加载最佳模型并进行最终测试
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best model from epoch {best_epoch} with Test Dice: {best_dice:.4f}")

    best_model_path = os.path.join(args.checkpoint_dir,
                                f"unet2d_best_{dataset_type}.pth")

    if os.path.exists(best_model_path):
        print("\nLoading best model for final evaluation...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 最终测试评估
        final_test_metrics = test_model(model, test_loader,
                                        criterion, device, dataset_type)

        # 保存最终结果
        final_log_content  = (f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n"
                            f"FINAL TEST RESULTS - {dataset_type}\n"
                            f"Best Epoch: {best_epoch}\n"
                            f"Final Test Dice: {final_test_metrics['dice']:.4f}\n"
                            f"Final Test IoU: {final_test_metrics['iou']:.4f}\n"
                            f"Final Test Sensitivity: {final_test_metrics['sensitivity']:.4f}\n"
                            f"Final Test PPV: {final_test_metrics['ppv']:.4f}\n"
                            f"Final Test HD95: {final_test_metrics['hd95']:.2f}\n"
                            f"{'='*60}\n")

        log_file = os.path.join(args.checkpoint_dir,
                                f"training_log_{dataset_type}.txt")
        with open(log_file, 'a') as f:
            f.write(final_log_content)

        print(f"\nTraining and evaluation completed for {dataset_type} dataset!")
        print(f"Best model saved to: {best_model_path}")
        print(f"Training log saved to: {log_file}")
    else:
        # 若未找到最佳模型权重
        print(f"Warning: Best model file not found: {best_model_path}")
    # ----------------------------------------------

