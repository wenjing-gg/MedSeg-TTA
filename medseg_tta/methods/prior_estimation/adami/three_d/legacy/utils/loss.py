import torch
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
import numpy as np
from torch import einsum

def LS_cross_entropy_loss(outputs, gt, epsilon = 0, reduction = 'mean'):
    log_ = F.log_softmax(outputs, dim=1)
    gt = gt.unsqueeze(dim=1)
    gt_bin = torch.zeros(outputs.shape).cuda().scatter_(1, gt, 1)
    gt_bin = (1 - epsilon) * gt_bin + (1 - gt_bin) * epsilon / 6
    loss = torch.sum(-log_ * gt_bin, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()


def distillation_loss(t, s, T=1.1):
    prob_t = F.softmax(t/T, dim=1)
    log_prob_s = F.log_softmax(s, dim=1)
    dist_loss = -(prob_t*log_prob_s).sum(dim=1).mean()
    return dist_loss
    

def calc_dice_loss(logits, label, num_classes, eps):

    label = torch.zeros(num_classes, label.shape[1]).cuda().scatter_(0, label, 1)
    label = label[1:, :]

    prob = torch.softmax(logits, dim=0)
    prob = prob[1:, :]
    
    intersection = label * prob
    loss = 1 - (2. * intersection.sum() + eps) / (prob.sum() + label.sum() + eps) 

    return loss


def dice_loss(outputs, label):
    """
    计算3D数据的Dice损失
    inputs:
        outputs: 模型输出，形状为 [b, c, d, h, w]
        label: 标签，形状为 [b, d, h, w]
    """
    # 1. 对输出应用softmax获取概率
    prob = torch.softmax(outputs, dim=1)  # [b, c, d, h, w]
    
    # 2. 处理标签维度并进行one-hot编码
    label = rearrange(label, 'b d h w -> b 1 d h w')  # [b, 1, d, h, w]
    b, _, d, h, w = label.shape
    label_onehot = torch.zeros(b, prob.shape[1], d, h, w).cuda()  # [b, c, d, h, w]
    label_onehot = label_onehot.scatter_(1, label.long(), 1.0)
    
    # 3. 忽略背景类（索引0），只计算前景类
    prob = prob[:, 1:, :, :, :]  # [b, c-1, d, h, w]
    label_onehot = label_onehot[:, 1:, :, :, :]  # [b, c-1, d, h, w]
    
    # 4. 调整维度以便计算
    prob = rearrange(prob, 'b c d h w -> b (c d h w)')  # [b, (c-1)*d*h*w]
    label_onehot = rearrange(label_onehot, 'b c d h w -> b (c d h w)')  # [b, (c-1)*d*h*w]
    
    # 5. 计算交集、概率和、标签和
    intersection = torch.sum(prob * label_onehot, dim=1)  # [b]
    prob_sum = torch.sum(prob, dim=1)  # [b]
    label_sum = torch.sum(label_onehot, dim=1)  # [b]
    
    # 6. 计算Dice系数和损失
    eps = 1e-3
    dice = (2. * intersection + eps) / (prob_sum + label_sum + eps)
    loss = 1 - dice.mean()  # 对批次求平均
    
    return loss
class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, eps=1e-3):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits, label):  
        # 输入维度假设: logits=(b, c, d, h, w), label=(b, d, h, w)
        b,  d, h, w = logits.shape
        
        # 1. 处理标签维度
        label = rearrange(label, 'b d h w -> b 1 d h w')  # (b,1,d,h,w)
        label = torch.zeros(b, self.num_classes, d, h, w).cuda().scatter_(1, label, 1)  # one-hot编码
        
        # 2. 过滤忽略索引
        ignore_mask = (label == self.ignore_index).any(dim=1, keepdim=True)  # (b,1,d,h,w)
        valid_label = label.masked_fill(ignore_mask, 0)
        valid_logits = logits.masked_fill(ignore_mask.expand_as(logits), -1e9)
        
        # 3. 计算Dice损失（仅对有效区域）
        prob = torch.softmax(valid_logits, dim=1)  # (b,c,d,h,w)
        intersection = valid_label * prob  # (b,c,d,h,w)
        
        # 4. 求和并计算Dice系数
        intersection_sum = intersection.sum(dim=(2, 3, 4))  # (b,c)
        prob_sum = prob.sum(dim=(2, 3, 4))  # (b,c)
        label_sum = valid_label.sum(dim=(2, 3, 4))  # (b,c)
        
        dice = (2. * intersection_sum + self.eps) / (prob_sum + label_sum + self.eps)
        loss = 1 - dice.mean()  # 对类别和批次求平均
        
        return loss

# class DiceLoss(nn.Module):
#     def __init__(self, num_classes, ignore_index=255, eps=1e-3):
#         super(DiceLoss, self).__init__()
#         self.num_classes = num_classes
#         self.eps = eps
#         self.ignore_index=ignore_index
#     def forward(self, logits, label):  

#         label = rearrange(label, 'b h w -> 1 (b h w)')
#         ind = torch.where(label!=self.ignore_index)[1]
#         label = torch.index_select(label, 1, ind)

#         logits = rearrange(logits, 'b c h w -> c (b h w)')
#         logits = torch.index_select(logits, 1, ind)

#         loss = calc_dice_loss(logits, label, self.num_classes, self.eps)

#         return loss


def masked_ce_dc(outputs, label, mask):
    outputs = rearrange(outputs, 'b c h w -> c (b h w)')
    label = rearrange(label, 'b h w -> 1 (b h w)')
    mask = rearrange(mask, 'b h w -> (b h w)')
    ind = torch.where(mask)[0]
    
    outputs = torch.index_select(outputs, 1, ind)
    label = torch.index_select(label, 1, ind)
    outputs = rearrange(outputs, 'c l -> 1 c l 1')
    label = rearrange(label, '1 l -> 1 l 1')
    loss_ce = F.cross_entropy(outputs, label)
    loss_dc = dice_loss(outputs, label)
    return loss_ce + loss_dc

class DeepInfoMaxLoss(nn.Module): #互信息最大化损失
    def __init__(self, type="concat"):
        super().__init__()
        if type=="concat":
            self.global_d = GlobalDiscriminator(sz=64+64)#
        elif type=="dot":
            self.global_d = GlobalDiscriminatorDot(sz=64)
        elif type=="conv":
            self.global_d = GlobalDiscriminatorConv(sz=7+7)

    def forward(self, proto_label_pos, proto_label_neg, proto_unlabel_pos):
        Ej = -F.softplus(-self.global_d(proto_unlabel_pos, proto_label_pos)).mean() #pos pair
        Em = F.softplus(self.global_d(proto_unlabel_pos, proto_label_neg)).mean() #neg pair
        LOSS = (Em - Ej)
        
        return LOSS
# Infomax Loss =========================================
class GlobalDiscriminator(nn.Module):
    def __init__(self, sz):
        super(GlobalDiscriminator, self).__init__()
        self.l0 = nn.Linear(sz, 512).cuda()
        self.l1 = nn.Linear(512, 512).cuda()
        self.l2 = nn.Linear(512, 1).cuda() #全连接层

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)   
        h = F.relu(self.l0(x))      
        h = F.relu(self.l1(h))
        return self.l2(h)

class MGlobalDiscriminator(nn.Module):
    def __init__(self, sz):
        super(MGlobalDiscriminator, self).__init__()
        self.l0 = nn.Linear(sz, 128).cuda()
        self.l1 = nn.Linear(128, 128).cuda()
        self.l2 = nn.Linear(128, 1).cuda() #全连接层
        self.st = nn.InstanceNorm2d(64).cuda()
        self.c = nn.Conv2d(128,64,kernel_size=3,padding=1).cuda()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1) #128
        x = F.relu(self.st(self.c(x))) #64   
        h = F.relu(self.l0(x))      
        h = F.relu(self.l1(h))
        return self.l2(h)
    
class GlobalDiscriminatorConv(nn.Module):
    def __init__(self, sz):
        super(GlobalDiscriminatorConv, self).__init__()
        self.l0 = nn.Conv2d(sz, 64, kernel_size=1).cuda()
        self.l1 = nn.Conv2d(64, 64, kernel_size=1).cuda()
        self.l2 = nn.Conv2d(64, 1, kernel_size=1).cuda()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return self.l2(h)    
    
class PriorDiscriminator(nn.Module):
    def __init__(self, sz):
        super(PriorDiscriminator, self).__init__()
        self.l0 = nn.Linear(sz, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))    
        
class MILinearBlock(nn.Module):
    def __init__(self, feature_sz, units=2048, bln=True):
        super(MILinearBlock, self).__init__()
        # Pre-dot product encoder for "Encode and Dot" arch for 1D feature maps
        self.feature_nonlinear = nn.Sequential(
            nn.Linear(feature_sz, units, bias=False),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Linear(units, units),
        ).cuda()
        self.feature_shortcut = nn.Linear(feature_sz, units).cuda()
        self.feature_block_ln = nn.LayerNorm(units).cuda()

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((units, feature_sz), dtype=np.bool)
        for i in range(feature_sz):
            eye_mask[i, i] = 1

        self.feature_shortcut.weight.data.uniform_(-0.01, 0.01).cuda()
        self.feature_shortcut.weight.data.masked_fill_(
            torch.tensor(eye_mask).cuda(), 1.0).cuda()
        self.bln = bln

    def forward(self, feat):
        f = self.feature_nonlinear(feat) + self.feature_shortcut(feat)
        if self.bln:
            f = self.feature_block_ln(f)

        return f
    
class GlobalDiscriminatorDot(nn.Module):
    def __init__(self, sz, units=2048, bln=True):
        super(GlobalDiscriminatorDot, self).__init__()
        self.block_a = MILinearBlock(sz, units=units, bln=bln).cuda()
        self.block_b = MILinearBlock(sz, units=units, bln=bln).cuda()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).cuda()

    def forward(
        self,
        features1=None,
        features2=None,
    ):

        # Computer cross modal loss
        feat1 = self.block_a(features1)
        feat2 = self.block_b(features2)

        feat1, feat2 = map(lambda t: F.normalize(
            t, p=2, dim=-1), (feat1, feat2))

        # ## Method 1
        # # Dot product and sum
        # o = torch.sum(feat1 * feat2, dim=1) * self.temperature.exp()

        # ## Method 2
        # o = self.cos(feat1, feat2) * self.temperature.exp()

        # Method 3
        o = einsum("n d, n d -> n", feat1, feat2) * self.temperature.exp()

        return o 

class LeastSquaresGenerativeAdversarialLoss(nn.Module):
    """
    Loss for `Least Squares Generative Adversarial Network (LSGAN) <https://arxiv.org/abs/1611.04076>`_

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - prediction (tensor): unnormalized discriminator predictions
        - real (bool): if the ground truth label is for real images or fake images. Default: true

    .. warning::
        Do not use sigmoid as the last layer of Discriminator.

    """
    def __init__(self, reduction='mean'):
        super(LeastSquaresGenerativeAdversarialLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, prediction, real=True):
        if real:
            label = torch.ones_like(prediction)
        else:
            label = torch.zeros_like(prediction)
        return self.mse_loss(prediction, label) 

