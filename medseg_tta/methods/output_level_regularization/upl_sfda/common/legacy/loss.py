import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='micro'):
        super().__init__()
        if isinstance(smooth, int) and smooth > 1:
            smooth = 1e-6
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets, one_hot=None):
        # 自动获取输入数据所在的设备
        device = logits.device
        
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        
        # 确保one_hot编码在正确设备上
        if targets.dim() == logits.dim():
            targets_onehot = targets.to(device).float()
        else:
            targets = targets.to(device).long()
            targets_onehot = F.one_hot(targets, num_classes).float()
            dims = list(range(targets_onehot.dim()))
            targets_onehot = targets_onehot.permute(0, dims[-1], *dims[1:-1]).to(device)

        intersection = torch.sum(probs * targets_onehot, dim=(2, 3, 4))
        union = torch.sum(probs + targets_onehot, dim=(2, 3, 4))

        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == 'macro':
            dice_loss = 1.0 - torch.mean(dice_scores)
        elif self.reduction == 'micro':
            valid_classes = torch.sum(targets_onehot, dim=(2, 3, 4)) > 0
            dice_loss = 1.0 - torch.sum(dice_scores * valid_classes) / (torch.sum(valid_classes) + 1e-8)
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        return dice_loss

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.3, dice_weight=0.7, dice_reduction='macro', class_weights=None, device=None):
        super().__init__()
        
        # 自动检测设备（优先使用传入的device参数）
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 转换class_weights到指定设备
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(reduction=dice_reduction)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        targets = targets.to(self.device).squeeze(dim=1).long()
        
        logits = logits.to(self.device)
        
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

    def to(self, device):
        # 重写to方法以保持设备同步
        self.device = device
        return super().to(device)


if __name__ == "__main__":
    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 模拟数据（自动传输到设备）
    batch_size = 2
    num_classes = 4
    spatial_size = (64, 128, 128)
    
    logits = torch.randn(batch_size, num_classes, *spatial_size).to(device)
    targets = torch.randint(0, num_classes, (batch_size, *spatial_size)).to(device)
    
    # 初始化损失函数（显式指定设备）
    class_weights = torch.tensor([0.1, 1.0, 2.0, 3.0], device=device)
    
    criterion = CombinedLoss(
        ce_weight=1.0,
        dice_weight=0.5,
        dice_reduction='macro',
        class_weights=class_weights,
        device=device
    )
    
    # 计算损失
    loss = criterion(logits, targets)
    print(f"Total Loss: {loss.item():.4f}")
