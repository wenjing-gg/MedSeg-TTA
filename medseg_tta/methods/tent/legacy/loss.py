import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-06, reduction='micro'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        device = logits.device
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).to(device).permute(0, 4, 1, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dim=(2, 3, 4))
        union = torch.sum(probs + targets_onehot, dim=(2, 3, 4))
        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)
        if self.reduction == 'macro':
            dice_loss = 1.0 - torch.mean(dice_scores)
        elif self.reduction == 'micro':
            valid_classes = torch.sum(targets_onehot, dim=(2, 3, 4)) > 0
            dice_loss = 1.0 - torch.sum(dice_scores * valid_classes) / (torch.sum(valid_classes) + 1e-08)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}')
        return dice_loss

class CombinedLoss(nn.Module):

    def __init__(self, ce_weight=1, dice_weight=0.5, dice_reduction='macro', class_weights=None, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(reduction=dice_reduction)
        self.log_ce_weight = nn.Parameter(torch.tensor(float(ce_weight)).log())
        self.log_dice_weight = nn.Parameter(torch.tensor(float(dice_weight)).log())

    def forward(self, logits, targets):
        targets = targets.to(self.device).squeeze(dim=1).long()
        logits = logits.to(self.device)
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        ce_weight = torch.exp(self.log_ce_weight)
        dice_weight = torch.exp(self.log_dice_weight)
        return ce_weight * ce_loss + dice_weight * dice_loss

    def get_weights(self):
        ce_weight = torch.exp(self.log_ce_weight).item()
        dice_weight = torch.exp(self.log_dice_weight).item()
        sum_weights = ce_weight + dice_weight
        return {'ce_weight': ce_weight, 'dice_weight': dice_weight, 'ce_weight_norm': ce_weight / sum_weights, 'dice_weight_norm': dice_weight / sum_weights}

    def to(self, device):
        self.device = device
        return super().to(device)
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    batch_size = 2
    num_classes = 4
    spatial_size = (64, 128, 128)
    logits = torch.randn(batch_size, num_classes, *spatial_size).to(device)
    targets = torch.randint(0, num_classes, (batch_size, *spatial_size)).to(device)
    class_weights = torch.tensor([0.1, 1.0, 2.0, 3.0], device=device)
    criterion = CombinedLoss(ce_weight=1.0, dice_weight=0.5, dice_reduction='macro', class_weights=class_weights, device=device)
    loss = criterion(logits, targets)
    print(f'Total Loss: {loss.item():.4f}')
