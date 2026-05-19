import numpy as np
from medpy.metric import hd95 as hd95_medpy
from torch import Tensor
import torch
import torch.nn as nn


def dice(output, target, eps=1e-5):
    eps=1e-5
    inter = torch.sum(output * target,dim=(1,2,-1)) + eps
    union = torch.sum(output,dim=(1,2,-1)) + torch.sum(target,dim=(1,2,-1)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice


def cal_dice(output, target):
    '''
    Calculate three Dice coefficients for different classes:
    - Dice1 (ET): label4 (replaced with 3)
    - Dice2 (TC): labels 1 and 3
    - Dice3 (WT): labels 1, 2, and 3

    Parameters:
    - output: (b, num_class, d, h, w)aa
    - target: (b, d, h, w)
    '''
    output = torch.argmax(output, dim=1)
    #print('argout',output)
    target = target.long()
    et_out = torch.any((output == 3).long())
    et_tar = torch.any((target == 3).long())
    tc_out = torch.any(((output == 1) | (output == 3)).long())
    tc_tar = torch.any(((target == 1) | (target == 3)).long())
    wt_out = torch.any((output != 0).long())
    wt_tar = torch.any((target != 0).long())
    if et_out and et_tar:
        dice1 = dice((output == 3).long(), (target == 3).long())
    elif et_tar:
        dice1 = 0.
    elif et_out:
        dice1 = 0.
    else:
        dice1 = 1.
        
    if tc_out and tc_tar:
        dice2 = dice(((output == 1) | (output == 3)).long(), ((target == 1) | (target == 3)).long())
    elif tc_tar:
        dice2 = 0.
    elif tc_out:
        dice2 = 0.
    else:
        dice2 = 1.

    if wt_out and wt_tar:
        dice3 = dice((output != 0).long(), (target != 0).long())
    elif wt_tar:
        dice3 = 0.
    elif wt_out:
        dice3 = 0.
    else:
        dice3 = 1.


    return dice1, dice2, dice3

def Dice(pred, gt, smooth=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
 
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
 
    pred = activation_fn(pred)
 
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / N





def cal_hd95(output: Tensor, target: Tensor, spacing=None):
    output = torch.argmax(output, dim=1)
    target = target.float()

    hd95_ec = compute_hd95((output == 3).float(), (target == 3).float())
    hd95_co = compute_hd95(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    hd95_wt = compute_hd95((output != 0).float(), (target != 0).float())

    return hd95_ec, hd95_co, hd95_wt

def cal_hd95_np(output: Tensor, target: Tensor, spacing=None):
    output = torch.argmax(output, dim=1)
    target = target.float()

    hd95_ec = compute_hd95_np((output == 3).float(), (target == 3).float())
    hd95_co = compute_hd95_np(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    hd95_wt = compute_hd95_np((output != 0).float(), (target != 0).float())

    return hd95_ec, hd95_co, hd95_wt

def compute_hd95_np(pred, gt, spacing=None):
    # 将 PyTorch 张量转换为 NumPy 数组
    pred_np = pred.cpu().numpy().astype(bool)
    gt_np = gt.cpu().numpy().astype(bool)

    try:
        hd = hd95_medpy(pred_np, gt_np, voxelspacing=spacing)
    except:
        # 如果 gt 全为零，设置 hd 为 0，否则设置为一个较大的值
        hd = 373.1287 if np.any(gt_np) else 0.0

    return hd

def compute_hd95(pred, gt, spacing=None):
    # 将 PyTorch 张量转换为 NumPy 数组
    pred_np = pred.cpu().numpy().astype(bool)
    gt_np = gt.cpu().numpy().astype(bool)

    try:
        hd = hd95_medpy(pred_np, gt_np, voxelspacing=spacing)
    except:
        # 如果 gt 全为零，设置 hd 为 0，否则设置为一个较大的值
        hd = 373.1287 if np.any(gt_np) else 0.0

    return hd

from medpy.metric.binary import jc
def IoU(output, target):
    output = torch.argmax(output, dim=1)
    # 将PyTorch张量转换为NumPy数组
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    
    jc_ec = jc((output_np == 3), (target_np == 3))
    jc_co = jc(((output_np == 1) | (output_np == 3)), ((target_np == 1) | (target_np == 3)))
    jc_wt = jc((output_np != 0), (target_np != 0))
    return jc_ec, jc_co, jc_wt

def genConfusionMatrix(imgPredict, imgLabel, numClass):
    # 将PyTorch张量转换为NumPy数组，并确保是整数类型
    imgPredict = imgPredict.long().cpu().numpy()
    imgLabel = imgLabel.long().cpu().numpy()
    
    mask = (imgLabel > 0) & (imgLabel < numClass)
    label = numClass * imgLabel[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass**2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix

def PA(output, target, numClass):
    # 返回所有类别的总体像素精度
    #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
    output = torch.argmax(output, dim=1)
    
    # 使用long()而非float()确保整数类型
    cf_ec = genConfusionMatrix((output == 3).long(), (target == 3).long(), numClass)
    cf_co = genConfusionMatrix(((output == 1) | (output == 3)).long(), ((target == 1) | (target == 3)).long(), numClass)
    cf_wt = genConfusionMatrix((output != 0).long(), (target != 0).long(), numClass)
    
    # 添加除零保护
    pa_ec = np.diag(cf_ec).sum() / cf_ec.sum() if cf_ec.sum() > 0 else 0.0
    pa_co = np.diag(cf_co).sum() / cf_co.sum() if cf_co.sum() > 0 else 0.0
    pa_wt = np.diag(cf_wt).sum() / cf_wt.sum() if cf_wt.sum() > 0 else 0.0
    
    return pa_ec, pa_co, pa_wt

def RVE(output, target):
    s_v = output.sum()
    g_v = target.sum()
    
    # 处理目标体积为零的情况
    if g_v == 0:
        # 如果预测也为零，则完美匹配（返回0）
        # 如果预测不为零，则完全错误（返回1）
        return 1.0 if s_v > 0 else 0.0
    
    # 正常情况计算相对体积误差
    rve = abs(s_v - g_v) / g_v
    return rve

def cal_RVE(output, target):
    output = torch.argmax(output, dim=1)
    cf_ec = RVE((output == 3).float(), (target == 3).float())
    cf_co = RVE(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    cf_wt = RVE((output != 0).float(), (target != 0).float())
    return cf_ec, cf_co, cf_wt

def sensitivity(output, target):
    smooth = 1e-5
 
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
 
    intersection = (output * target).sum()
 
    return (intersection + smooth) / \
        (target.sum() + smooth)
def cal_sensitivity(output, target):
    output = torch.argmax(output, dim=1)
    se_ec = sensitivity((output == 3).float(), (target == 3).float())
    se_co = sensitivity(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    se_wt = sensitivity((output != 0).float(), (target != 0).float())
    return se_ec, se_co, se_wt

def ppv(output, target, smooth=1e-5):
    if isinstance(output, torch.Tensor):
        output = output.data.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()
    return (intersection + smooth) / (output.sum() + smooth)

def cal_ppv(output, target):
    output = torch.argmax(output, dim=1).cpu().numpy()  # 确保是 numpy 数组
    target = target.cpu().numpy()  # 确保是 numpy 数组

    # 计算 PPV for EC (class 3)
    ppv_ec = ppv((output == 3).astype(float), (target == 3).astype(float))

    # 计算 PPV for CO (classes 1 and 3)
    ppv_co = ppv(
        ((output == 1) | (output == 3)).astype(float),
        ((target == 1) | (target == 3)).astype(float)
    )

    # 计算 PPV for WT (all non-zero classes)
    ppv_wt = ppv((output != 0).astype(float), (target != 0).astype(float))

    return ppv_ec, ppv_co, ppv_wt
def main():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(49)
    np.random.seed(49)

    # 创建示例数据
    batch_size = 1
    num_classes = 4  
    channel = 4
    depth = 4
    height = 4
    width = 4

    # 随机生成模型输出 (batch_size, num_classes, depth, height, width)
    output = torch.randn(batch_size, channel, depth, height, width)
    print('out',output)
    # 随机生成真实标签 (batch_size, depth, height, width)
    label = torch.randint(0, 4, (batch_size, depth, height, width))
    print('label',label)

    # 计算 Dice 系数
    dice1, dice2, dice3 = cal_dice(output, label)
    print(f"Dice1 (ET): {dice1:.4f}")
    print(f"Dice2 (TC): {dice2:.4f}")
    print(f"Dice3 (WT): {dice3:.4f}")

    # 计算 HD95
    hd95_ec, hd95_co, hd95_wt = cal_hd95_np(output, label)#除此处测试外，使用cal_hd95(output, label)
    print(f"HD95 EC: {hd95_ec:.4f}")
    print(f"HD95 CO: {hd95_co:.4f}")
    print(f"HD95 WT: {hd95_wt:.4f}")
       
    jc1, jc2, jc3 = IoU(output, label)
    print(f"IoU EC: {jc1:.4f}")
    print(f"IoU CO: {jc2:.4f}")
    print(f"IoU WT: {jc3:.4f}")

    pa_ec, pa_co, pa_wt = PA(output, label,num_classes)
    print(f"PA EC: {pa_ec:.4f}")
    print(f"PA CO: {pa_co:.4f}")
    print(f"PA WT: {pa_wt:.4f}")

    rve_ec, rve_co, rve_wt = cal_RVE(output, label)
    print(f"RVE EC: {rve_ec:.4f}")
    print(f"RVE CO: {rve_co:.4f}")
    print(f"RVE WT: {rve_wt:.4f}")

    se_ec, se_co, se_wt = cal_sensitivity(output, label)
    print(f"sensitivity EC: {se_ec:.4f}")
    print(f"sensitivity CO: {se_co:.4f}")
    print(f"sensitivity WT: {se_wt:.4f}")

    ppv_ec, ppv_co, ppv_wt = cal_ppv(output, label)
    print(f"ppv EC: {ppv_ec:.4f}")
    print(f"ppv CO: {ppv_co:.4f}")
    print(f"ppv WT: {ppv_wt:.4f}")

if __name__ == "__main__":
    main()