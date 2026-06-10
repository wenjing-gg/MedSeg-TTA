import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='micro'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        # 自动获取输入数据所在的设备
        device = logits.device
        #raise ValueError(logits.shape,targets.shape)
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        # 确保one_hot编码在正确设备上
        targets_onehot = F.one_hot(targets, num_classes).to(device).permute(0, 4, 1, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dim=(2, 3, 4))
        union = torch.sum(probs + targets_onehot, dim=(2, 3, 4))
        #print(intersection, union)
        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == 'macro':
            dice_loss = 1.0 - torch.mean(dice_scores)
        elif self.reduction == 'micro':
            valid_classes = torch.sum(targets_onehot, dim=(2, 3, 4)) > 0
            dice_loss = 1.0 - torch.sum(dice_scores * valid_classes) / (torch.sum(valid_classes) + 1e-8)
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        return dice_loss
# 先定义Focal Loss类（支持类别权重）
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 这里传入之前的class_weights（tensor形式）
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, 
                                weight=self.alpha, 
                                reduction='none')  # 先计算基础CE
        pt = torch.exp(-ce_loss)  # 计算概率 p_t
        focal_loss = (1 - pt)**self.gamma * ce_loss  # 调制因子
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
'''
# 使用方式（替换原来的self.ce）
self.ce = FocalLoss(alpha=class_weights, gamma=2)  # gamma可调节（常用2-5）'''

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.3, dice_weight=0.7, dice_reduction='macro', class_weights=None, device=None):
        super().__init__()
        
        # 自动检测设备（优先使用传入的device参数）
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 转换class_weights到指定设备
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        #self.ce = FocalLoss(alpha=class_weights, gamma=2)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(reduction=dice_reduction)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        #targets = torch.argmax(targets,dim = 1).to(self.device).squeeze(dim=1).long()
        targets = targets.to(self.device).squeeze(dim=1).long()
        logits = logits.to(self.device)
        #print(torch.unique(logits),torch.unique(targets))
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

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07, spatial_reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.spatial_reduction = spatial_reduction

    def spatial_similarity(self, feat1, feat2):
        """计算三维特征空间相似度"""
        # 输入形状: (D1, H1, W1), (D2, H2, W2)
        # 展平空间维度
        feat1_flat = feat1.view(feat1.size(0), -1)  # (C, D*H*W)
        feat2_flat = feat2.view(feat2.size(0), -1)  # (C, D*H*W)
        return torch.mm(feat1_flat.T, feat2_flat)  # (D1*H1*W1, D2*H2*W2)

    def forward(self, batch_features, batch_labels, memory_bank):
        device = 'cuda:1'
        """
        Args:
            batch_features (Tensor): 当前批次特征 (B, C=128, D=128, H=128, W=128)
            batch_labels (Tensor): 当前标签 (B,)
            memory_bank (MemoryBank3D): 存储三维特征的记忆库
        """
        batch_features = batch_features.to(device)
        batch_labels = batch_labels
        C, D, H, W = batch_features.shape
        batch_features = batch_features.view(C, -1)
        device_2 = batch_features.device
        
        # 获取记忆库特征和标签
        mem_features = memory_bank.features.to(device)  # (M, C, D, H, W)
        mem_labels = memory_bank.labels.to(device)      # (M,)
        M = mem_features.size(0)
        mem_features = mem_features.view(M, C, -1)
        print(f"当前显存占用5: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        # 初始化损失容器
        total_loss = 0.0

        # 计算与所有记忆样本的相似度
        similarities = []
        similarities = torch.einsum('cd,mcp->mdp', batch_features, mem_features)
        similarities.append(sim_matrix.mean(dim = (2,3)))
        
        print(f"当前显存占用6: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        # 构建正样本掩码
        pos_mask = (mem_labels == batch_labels)  # (M,)

        # 计算InfoNCE
        exp_sim = torch.exp(similarities / self.temperature)
        numerator = exp_sim * pos_mask.sum(dim = 1)
        denominator = exp_sim.sum(dim = 1)


        loss = -torch.log(numerator / (denominator + 1e-8))
        total_loss += loss
        print(f"当前显存占用7: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        return total_loss.to(device_2)

def hybrid_loss(pred, target, alpha=0.5):
    # Dice Loss
    dice = DiceLoss()
    dice_loss = dice(pred, target)
    
    # HD95 Loss（基于距离变换）
    gt_distance_map = kornia.contrib.distance_transform(target)
    pred_edges = kornia.filters.sobel(pred)
    edge_coords = torch.nonzero(pred_edges.squeeze())
    edge_distances = gt_distance_map[edge_coords[:, 0], edge_coords[:, 1]]
    hd95 = torch.quantile(edge_distances, 0.95) if len(edge_distances) > 0 else 0.0
    
    # 总损失
    total_loss = dice_loss + alpha * hd95
    return total_loss
