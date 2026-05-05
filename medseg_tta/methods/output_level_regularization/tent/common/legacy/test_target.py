import argparse
import os
import datetime
import traceback
import torch
import numpy as np
from tqdm import tqdm
from unet3d import UNet3d
from utils_brats_all import get_data_loader
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
import torch.nn as nn
import tent
import torch.optim as optim

def safe_value(val):
    if isinstance(val, torch.Tensor):
        return val.item()
    return val

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 开始在目标数据集上测试模态: {args.img.upper()}')
    print(f'{'=' * 40}\n')
    try:
        model = UNet3d().to(device)
        print(f'已选择 UNet3d 模型架构')
        default_model_path = os.path.join('/home/yuwenjing/DeepLearning_ywj/tta/unet3d_best_MRI.pth')
        if args.model_path and args.model_path != 'default':
            best_model_path = args.model_path
        else:
            best_model_path = default_model_path
        best_model_path = '/home/yuwenjing/DeepLearning_ywj/tta/unet3d_best.pth'
        print(f'加载模型权重: {best_model_path}')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f'未找到预训练权重: {best_model_path}')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        for param in model.parameters():
            param.data = param.data.to(device)
        model = tent.configure_model(model)
        for name, param in model.named_parameters():
            if param.device != device:
                print(f'警告: 参数 {name} 在 {param.device} 上，而不是 {device}')
                param.data = param.data.to(device)
        params, param_names = tent.collect_params(model)
        optimizer = optim.Adam(params, lr=args.lr)
        tented_model = tent.Tent(model, optimizer)
        _, target_test_loader = get_data_loader(source_root=args.source_root, target_root=args.target_root, batch_train=args.batch_test, batch_test=args.batch_test, nw=args.num_workers, img=args.img, mode='source_to_target')
        all_dice_values = [[] for _ in range(3)]
        all_hd95_values = [[] for _ in range(3)]
        all_IoU_values = [[] for _ in range(3)]
        all_pa_values = [[] for _ in range(3)]
        all_RVE_values = [[] for _ in range(3)]
        all_sensitivity_values = [[] for _ in range(3)]
        all_ppv_values = [[] for _ in range(3)]
        with torch.no_grad():
            for imgs, labels, *_ in tqdm(target_test_loader, desc='推理进度'):
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model.forward(imgs)
                dice_values = cal_dice(outputs, labels.squeeze(1))
                hd95_values = cal_hd95(outputs, labels.squeeze(1))
                IoU_values = IoU(outputs, labels.squeeze(1))
                pa_values = PA(outputs, labels.squeeze(1), 4)
                RVE_values = cal_RVE(outputs, labels.squeeze(1))
                sensitivity_values = cal_sensitivity(outputs, labels.squeeze(1))
                ppv_values = cal_ppv(outputs, labels.squeeze(1))
                for i in range(3):
                    all_dice_values[i].append(safe_value(dice_values[i]))
                    all_hd95_values[i].append(safe_value(hd95_values[i]))
                    all_IoU_values[i].append(safe_value(IoU_values[i]))
                    all_pa_values[i].append(safe_value(pa_values[i]))
                    all_RVE_values[i].append(safe_value(RVE_values[i]))
                    all_sensitivity_values[i].append(safe_value(sensitivity_values[i]))
                    all_ppv_values[i].append(safe_value(ppv_values[i]))
        mean_dice = [np.mean(values) for values in all_dice_values]
        std_dice = [np.std(values) for values in all_dice_values]
        mean_hd95 = [np.mean(values) for values in all_hd95_values]
        std_hd95 = [np.std(values) for values in all_hd95_values]
        mean_IoU = [np.mean(values) for values in all_IoU_values]
        std_IoU = [np.std(values) for values in all_IoU_values]
        mean_pa = [np.mean(values) for values in all_pa_values]
        std_pa = [np.std(values) for values in all_pa_values]
        mean_RVE = [np.mean(values) for values in all_RVE_values]
        std_RVE = [np.std(values) for values in all_RVE_values]
        mean_sensitivity = [np.mean(values) for values in all_sensitivity_values]
        std_sensitivity = [np.std(values) for values in all_sensitivity_values]
        mean_ppv = [np.mean(values) for values in all_ppv_values]
        std_ppv = [np.std(values) for values in all_ppv_values]
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = 'nnUNet' if args.model_type == 'nnunet' else 'UNet3D'
        model_filename = os.path.basename(best_model_path)
        report = f'\n{'=' * 40}\n测试时间: {timestamp}\n测试配置:\n- 图像模态: {args.img}\n- 模型类型: {model_name}\n- 模型路径: {best_model_path}\n- 测试数据: {args.target_root}\n\n性能指标:\nDice (均值 ± 标准差):\n  ET: {mean_dice[0]:.4f} ± {std_dice[0]:.4f}\n  TC: {mean_dice[1]:.4f} ± {std_dice[1]:.4f}\n  WT: {mean_dice[2]:.4f} ± {std_dice[2]:.4f}\nHD95(mm) (均值 ± 标准差):\n  ET: {mean_hd95[0]:.2f} ± {std_hd95[0]:.2f}\n  TC: {mean_hd95[1]:.2f} ± {std_hd95[1]:.2f}\n  WT: {mean_hd95[2]:.2f} ± {std_hd95[2]:.2f}\nIoU (均值 ± 标准差):\n  ET: {mean_IoU[0]:.4f} ± {std_IoU[0]:.4f}\n  TC: {mean_IoU[1]:.4f} ± {std_IoU[1]:.4f}\n  WT: {mean_IoU[2]:.4f} ± {std_IoU[2]:.4f}\nPA (均值 ± 标准差):\n  ET: {mean_pa[0]:.4f} ± {std_pa[0]:.4f}\n  TC: {mean_pa[1]:.4f} ± {std_pa[1]:.4f}\n  WT: {mean_pa[2]:.4f} ± {std_pa[2]:.4f}\nRVE (均值 ± 标准差):\n  ET: {mean_RVE[0]:.4f} ± {std_RVE[0]:.4f}\n  TC: {mean_RVE[1]:.4f} ± {std_RVE[1]:.4f}\n  WT: {mean_RVE[2]:.4f} ± {std_RVE[2]:.4f}\nSensitivity (均值 ± 标准差):\n  ET: {mean_sensitivity[0]:.4f} ± {std_sensitivity[0]:.4f}\n  TC: {mean_sensitivity[1]:.4f} ± {std_sensitivity[1]:.4f}\n  WT: {mean_sensitivity[2]:.4f} ± {std_sensitivity[2]:.4f}\nPPV (均值 ± 标准差):\n  ET: {mean_ppv[0]:.4f} ± {std_ppv[0]:.4f}\n  TC: {mean_ppv[1]:.4f} ± {std_ppv[1]:.4f}\n  WT: {mean_ppv[2]:.4f} ± {std_ppv[2]:.4f}\n{'=' * 40}\n'
        result_file = os.path.join(args.checkpoint_dir, f'test_{args.img}_{args.model_type}_{timestamp}.txt')
        with open(result_file, 'w') as f:
            f.write(report)
        print(report)
        return True
    except Exception as e:
        error_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        error_msg = f'\n🔥 测试失败: {args.img}\n模型类型: {args.model_type}\n错误信息: {str(e)}\n追踪信息:\n{traceback.format_exc()}'
        print(error_msg)
        error_log = os.path.join(args.checkpoint_dir, 'test_errors.log')
        with open(error_log, 'a') as f:
            f.write(f'[{error_timestamp}] {error_msg}\n')
        return False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='目标数据集测试脚本')
    parser.add_argument('--source_root', type=str, default='/home/yuwenjing/data/BraTS2024')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/BraTS2024/train', help='目标数据集根目录路径')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints', help='包含预训练权重的检查点目录')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['nnunet', 'unet3d'], help='选择模型架构类型 (nnunet 或 unet3d)')
    parser.add_argument('--model_path', type=str, default='default', help='指定模型权重文件的完整路径，使用default则根据model_type自动选择')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--gpu', type=int, default=2, help='使用GPU编号')
    parser.add_argument('--img', default=['all'], help='测试模态')
    parser.add_argument('--batch_test', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  使用设备: {device}')
    print(f'📋 使用模型: {args.model_type}')
    print(f'📦 模型路径: {args.model_path}')
    success_count = 0
    start_time = datetime.datetime.now()
    for idx, modality in enumerate(args.img, 1):
        print(f'\n🔍 正在测试 ({idx}/{len(args.img)}) {modality.upper()}')
        modality_args = argparse.Namespace(**vars(args))
        modality_args.img = modality
        if test_on_target(modality_args, device):
            success_count += 1
    total_time = datetime.datetime.now() - start_time
    summary = f'\n{'=' * 40}\n测试总结:\n- 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n- 总耗时: {total_time}\n- 模型类型: {args.model_type}\n- 成功测试: {success_count}/{len(args.img)}\n- 失败测试: {len(args.img) - success_count}\n{'=' * 40}\n'
    print(summary)
    summary_file = os.path.join(args.checkpoint_dir, f'test_summary_{args.model_type}_{start_time.strftime('%Y%m%d_%H%M%S')}.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
