import argparse
import os
import datetime
import traceback
import torch
import torch.optim as optim
from tqdm import tqdm
from nnunet import PlainConvUNet
from utils_brats_all import get_data_loader
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
import torch.nn as nn
import torch.nn.functional as F
from augmentation_utils import get_disp_field, get_rand_affine
import numpy as np

def safe_value(val):
    if isinstance(val, torch.Tensor):
        return val.item()
    return val

def soft_dice_loss(smp_a, smp_b):
    B, _, D, H, W = smp_a.shape
    d = 2
    nominator = (2.0 * smp_a * smp_b).reshape(B, -1, D * H * W).mean(2)
    denominator = 1 / d * ((smp_a + smp_b) ** d).reshape(B, -1, D * H * W).mean(2)
    if denominator.sum() == 0.0:
        dice = nominator * 0.0 + 1.0
    else:
        dice = nominator / denominator
    return dice

def freeze_bn_statistics(model):
    print('📊 冻结所有BatchNorm层的统计量')
    frozen_modules = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.track_running_stats = False
            module.running_mean = module.running_mean.detach()
            module.running_var = module.running_var.detach()
            frozen_modules += 1
    print(f'✅ 已冻结 {frozen_modules} 个BatchNorm层')
    return model

def freeze_non_bn_parameters(model):
    print('❄️ 冻结所有非BN参数')
    frozen_params = 0
    total_params = 0
    bn_param_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm3d):
            bn_param_names.add(f'{name}.weight')
            bn_param_names.add(f'{name}.bias')
    for name, param in model.named_parameters():
        total_params += 1
        if name in bn_param_names:
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen_params += 1
    print(f'✅ 已冻结 {frozen_params}/{total_params} 个参数')
    return model

def compute_consistency_loss(model, imgs, device):
    batch_size, _, *patch_size = imgs.size()
    identity_grid = F.affine_grid(torch.eye(4, device=device).repeat(batch_size, 1, 1)[:, :3], [batch_size, 1] + patch_size, align_corners=False)
    R_a, R_a_inv = get_rand_affine(batch_size, flip=False, device=device)
    affine_grid_a = F.affine_grid(R_a, [batch_size, 1] + patch_size, align_corners=False)
    grid_deform_a, grid_deform_a_inv = get_disp_field(batch_size, patch_size, device=device)
    composite_grid_a = identity_grid + (affine_grid_a - identity_grid) + grid_deform_a
    x_a = F.grid_sample(imgs, composite_grid_a, padding_mode='border', align_corners=False)
    R_b, R_b_inv = get_rand_affine(batch_size, flip=False, device=device)
    affine_grid_b = F.affine_grid(R_b, [batch_size, 1] + patch_size, align_corners=False)
    grid_deform_b, grid_deform_b_inv = get_disp_field(batch_size, patch_size, device=device)
    composite_grid_b = identity_grid + (affine_grid_b - identity_grid) + grid_deform_b
    x_b = F.grid_sample(imgs, composite_grid_b, padding_mode='border', align_corners=False)

    def _forward(x):
        output = model(x)
        return output[0] if isinstance(output, tuple) else output
    pred_a = _forward(x_a)
    pred_b = _forward(x_b)

    def _inverse_transform(pred, R_inv, deform_inv):
        affine_inv = F.affine_grid(R_inv, [batch_size, 1] + patch_size, align_corners=False)
        grid_inv = identity_grid + (affine_inv - identity_grid) + deform_inv
        return F.grid_sample(pred, grid_inv, align_corners=False)
    aligned_a = _inverse_transform(pred_a, R_a_inv, grid_deform_a_inv)
    aligned_b = _inverse_transform(pred_b, R_b_inv, grid_deform_b_inv)
    sm_a = aligned_a.softmax(1)
    sm_b = aligned_b.softmax(1)
    loss = 1 - soft_dice_loss(sm_a, sm_b)[:, 1:].mean()
    return loss

def get_model(model_type, device):
    if model_type.lower() == 'nnunet':
        print(f'📋 加载 nnUNet 模型架构')
        model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
        return model
    elif model_type.lower() == 'unet3d':
        print(f'📋 加载 UNet3D 模型架构')
        from unet3d import UNet3d
        model = UNet3d().to(device)
        return model
    else:
        raise ValueError(f"不支持的模型类型: {model_type}。请选择 'nnunet' 或 'unet3d'")

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 开始在目标数据集上测试模态: {args.img.upper()}')
    print(f'🧩 使用模型类型: {args.model_type}')
    print(f'{'=' * 40}\n')
    try:
        model = get_model(args.model_type, device)
        if args.model_path and args.model_path != 'default':
            model_path = args.model_path
        elif args.model_type.lower() == 'nnunet':
            model_path = '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth'
        elif args.model_type.lower() == 'unet3d':
            model_path = '/home/yuwenjing/DeepLearning_ywj/tta/unet3d_best.pth'
        else:
            raise ValueError(f'不支持的模型类型: {args.model_type}')
        print(f'📦 加载模型权重: {model_path}')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'未找到模型权重文件: {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=device))
        for param in model.parameters():
            if param.device != device:
                param.data = param.data.to(device)
        print(f'🔍 模型参数设备检查: {next(model.parameters()).device}')
        model = freeze_bn_statistics(model)
        if args.freeze_other:
            model = freeze_non_bn_parameters(model)
        model.train() if not args.eval_mode else model.eval()
        print(f'🔧 模型模式: {('train' if not args.eval_mode else 'eval')}')
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        _, test_loader = get_data_loader(source_root=args.source_root, target_root=args.target_root, batch_train=args.batch_test, batch_test=args.batch_test, nw=args.num_workers, img=args.img, mode='source_to_target')
        all_dice = [[] for _ in range(3)]
        all_hd95 = [[] for _ in range(3)]
        all_IoU = [[] for _ in range(3)]
        all_pa = [[] for _ in range(3)]
        all_RVE = [[] for _ in range(3)]
        all_sensitivity = [[] for _ in range(3)]
        all_ppv = [[] for _ in range(3)]
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='TTA适应')):
            if len(batch) == 3:
                imgs, labels, _ = batch
            elif len(batch) >= 2:
                imgs, labels = batch[:2]
            else:
                raise ValueError('批次数据格式不正确')
            imgs, labels = (imgs.to(device), labels.to(device))
            for _ in range(args.adapt_steps):
                loss = compute_consistency_loss(model, imgs, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                outputs = model(imgs)[0] if isinstance(model(imgs), tuple) else model(imgs)
                dice_values = cal_dice(outputs, labels.squeeze(1))
                hd95_values = cal_hd95(outputs, labels.squeeze(1))
                IoU_values = IoU(outputs, labels.squeeze(1))
                pa_values = PA(outputs, labels.squeeze(1), 4)
                RVE_values = cal_RVE(outputs, labels.squeeze(1))
                sensitivity_values = cal_sensitivity(outputs, labels.squeeze(1))
                ppv_values = cal_ppv(outputs, labels.squeeze(1))
                for i in range(3):
                    all_dice[i].append(safe_value(dice_values[i]))
                    all_hd95[i].append(safe_value(hd95_values[i]))
                    all_IoU[i].append(safe_value(IoU_values[i]))
                    all_pa[i].append(safe_value(pa_values[i]))
                    all_RVE[i].append(safe_value(RVE_values[i]))
                    all_sensitivity[i].append(safe_value(sensitivity_values[i]))
                    all_ppv[i].append(safe_value(ppv_values[i]))
        mean_dice = [np.mean(all_dice[i]) for i in range(3)]
        std_dice = [np.std(all_dice[i]) for i in range(3)]
        mean_hd95 = [np.mean(all_hd95[i]) for i in range(3)]
        std_hd95 = [np.std(all_hd95[i]) for i in range(3)]
        mean_IoU = [np.mean(all_IoU[i]) for i in range(3)]
        std_IoU = [np.std(all_IoU[i]) for i in range(3)]
        mean_pa = [np.mean(all_pa[i]) for i in range(3)]
        std_pa = [np.std(all_pa[i]) for i in range(3)]
        mean_RVE = [np.mean(all_RVE[i]) for i in range(3)]
        std_RVE = [np.std(all_RVE[i]) for i in range(3)]
        mean_sensitivity = [np.mean(all_sensitivity[i]) for i in range(3)]
        std_sensitivity = [np.std(all_sensitivity[i]) for i in range(3)]
        mean_ppv = [np.mean(all_ppv[i]) for i in range(3)]
        std_ppv = [np.std(all_ppv[i]) for i in range(3)]
        save_dir = os.path.join(args.checkpoint_dir, f'{args.model_type}_tta_weights', args.img)
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        avg_dice = np.mean(mean_dice)
        filename = f'{args.model_type}_DGTTA_SSA.pth'
        save_path = os.path.join(save_dir, filename)
        torch.save(model.state_dict(), save_path)
        print(f'💾 TTA后模型权重已保存到: {save_path}')
        result = f'\n测试配置:\n  模型类型: {args.model_type}\n  模型路径: {model_path}\n  测试模态: {args.img}\n  适应步数: {args.adapt_steps}\n  学习率: {args.lr}\n\nDice:\n  ET: {mean_dice[0]:.4f} ± {std_dice[0]:.4f}\n  TC: {mean_dice[1]:.4f} ± {std_dice[1]:.4f}\n  WT: {mean_dice[2]:.4f} ± {std_dice[2]:.4f}\nHD95:\n  ET: {mean_hd95[0]:.2f} ± {std_hd95[0]:.2f}mm\n  TC: {mean_hd95[1]:.2f} ± {std_hd95[1]:.2f}mm\n  WT: {mean_hd95[2]:.2f} ± {std_hd95[2]:.2f}mm\nIoU:\n  ET: {mean_IoU[0]:.4f} ± {std_IoU[0]:.4f}\n  TC: {mean_IoU[1]:.4f} ± {std_IoU[1]:.4f}\n  WT: {mean_IoU[2]:.4f} ± {std_IoU[2]:.4f}\nPA:\n  ET: {mean_pa[0]:.4f} ± {std_pa[0]:.4f}\n  TC: {mean_pa[1]:.4f} ± {std_pa[1]:.4f}\n  WT: {mean_pa[2]:.4f} ± {std_pa[2]:.4f}\nRVE:\n  ET: {mean_RVE[0]:.4f} ± {std_RVE[0]:.4f}\n  TC: {mean_RVE[1]:.4f} ± {std_RVE[1]:.4f}\n  WT: {mean_RVE[2]:.4f} ± {std_RVE[2]:.4f}\nSensitivity:\n  ET: {mean_sensitivity[0]:.4f} ± {std_sensitivity[0]:.4f}\n  TC: {mean_sensitivity[1]:.4f} ± {std_sensitivity[1]:.4f}\n  WT: {mean_sensitivity[2]:.4f} ± {std_sensitivity[2]:.4f}\nPPV:\n  ET: {mean_ppv[0]:.4f} ± {std_ppv[0]:.4f}\n  TC: {mean_ppv[1]:.4f} ± {std_ppv[1]:.4f}\n  WT: {mean_ppv[2]:.4f} ± {std_ppv[2]:.4f}\n        '
        print(result)
        result_dir = os.path.join(args.checkpoint_dir, f'{args.model_type}_results', args.img)
        os.makedirs(result_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(result_dir, f'tta_results_{timestamp}.txt'), 'w') as f:
            f.write(result)
            f.write('\n\n详细统计数据:\n')
            f.write(f'样本数: {len(all_dice[0])}\n')
            for i, region in enumerate(['ET', 'TC', 'WT']):
                f.write(f'\n{region} 区域详细指标:\n')
                f.write(f'Dice 均值: {mean_dice[i]:.4f}, 标准差: {std_dice[i]:.4f}\n')
                f.write(f'HD95 均值: {mean_hd95[i]:.2f}mm, 标准差: {std_hd95[i]:.2f}mm\n')
                f.write(f'IoU 均值: {mean_IoU[i]:.4f}, 标准差: {std_IoU[i]:.4f}\n')
        return True
    except Exception as e:
        print(f'❌ 测试失败: {str(e)}')
        traceback.print_exc()
        return False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='医学图像域适应测试脚本')
    parser.add_argument('--source_root', type=str, default='/home/yuwenjing/data/BraTS2024')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/BraTS-SSA', help='目标数据集根目录路径')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/DG-TTA/checkpoints')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['nnunet', 'unet3d'], help='选择模型架构类型 (nnunet 或 unet3d)')
    parser.add_argument('--model_path', type=str, default='default', help='指定模型权重文件的完整路径，使用default则根据model_type自动选择')
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--adapt_steps', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--img', type=str, default='all')
    parser.add_argument('--batch_test', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_mode', type=bool, default=False)
    parser.add_argument('--freeze_other', type=bool, default=False)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    test_on_target(args, device)
