import argparse
import os
import datetime
import traceback
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from nnunet import PlainConvUNet
from unet3d import UNet3d
from utils_brats_all import get_data_loader
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
import tent
import torch.nn as nn

def safe_value(val):
    if isinstance(val, torch.Tensor):
        return val.item()
    return val

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 开始在目标数据集上测试模态: {args.img.upper()}')
    print(f'{'=' * 40}\n')
    try:
        result_dir = '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints/tta_results'
        os.makedirs(result_dir, exist_ok=True)
        weights_dir = os.path.join(result_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        if args.model_type == 'nnunet':
            model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
            print(f'已选择 nnUNet 模型架构')
            default_model_path = '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth'
        else:
            model = UNet3d(in_chns=4, n_classes=4).to(device)
            print(f'已选择 UNet3d 模型架构')
            default_model_path = '/home/yuwenjing/DeepLearning_ywj/tta/unet3d_best.pth'
        if args.model_path and args.model_path != 'default':
            best_model_path = args.model_path
        else:
            best_model_path = default_model_path
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
        all_dice = [[] for _ in range(3)]
        all_hd95 = [[] for _ in range(3)]
        all_IoU = [[] for _ in range(3)]
        all_pa = [[] for _ in range(3)]
        all_RVE = [[] for _ in range(3)]
        all_sensitivity = [[] for _ in range(3)]
        all_ppv = [[] for _ in range(3)]
        with torch.no_grad():
            for imgs, labels, *_ in tqdm(target_test_loader, desc='推理进度'):
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = tented_model(imgs)
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
        dice_mean = [np.mean(vals) for vals in all_dice]
        dice_std = [np.std(vals) for vals in all_dice]
        hd95_mean = [np.mean(vals) for vals in all_hd95]
        hd95_std = [np.std(vals) for vals in all_hd95]
        IoU_mean = [np.mean(vals) for vals in all_IoU]
        IoU_std = [np.std(vals) for vals in all_IoU]
        pa_mean = [np.mean(vals) for vals in all_pa]
        pa_std = [np.std(vals) for vals in all_pa]
        RVE_mean = [np.mean(vals) for vals in all_RVE]
        RVE_std = [np.std(vals) for vals in all_RVE]
        sensitivity_mean = [np.mean(vals) for vals in all_sensitivity]
        sensitivity_std = [np.std(vals) for vals in all_sensitivity]
        ppv_mean = [np.mean(vals) for vals in all_ppv]
        ppv_std = [np.std(vals) for vals in all_ppv]
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = os.path.basename(best_model_path)
        model_name = os.path.splitext(model_filename)[0]
        adapted_model_path = os.path.join(weights_dir, f'{model_name}_{args.img}_tta_adapted_SSA.pth')
        torch.save(tented_model.model.state_dict(), adapted_model_path)
        print(f'✅ 已保存测试适应后的模型权重: {adapted_model_path}')
        report = f'\n{'=' * 40}\n测试时间: {timestamp}\n测试配置:\n- 图像模态: {args.img}\n- 模型类型: {args.model_type}\n- 模型路径: {best_model_path}\n- 测试数据: {args.target_root}\n- 适应后模型保存路径: {adapted_model_path}\n\n性能指标 (格式: 均值±标准差):\nDice:\n  ET: {dice_mean[0]:.4f}±{dice_std[0]:.4f}\n  TC: {dice_mean[1]:.4f}±{dice_std[1]:.4f}\n  WT: {dice_mean[2]:.4f}±{dice_std[2]:.4f}\nHD95(mm):\n  ET: {hd95_mean[0]:.2f}±{hd95_std[0]:.2f}\n  TC: {hd95_mean[1]:.2f}±{hd95_std[1]:.2f}\n  WT: {hd95_mean[2]:.2f}±{hd95_std[2]:.2f}\nIoU:\n  ET: {IoU_mean[0]:.4f}±{IoU_std[0]:.4f}\n  TC: {IoU_mean[1]:.4f}±{IoU_std[1]:.4f}\n  WT: {IoU_mean[2]:.4f}±{IoU_std[2]:.4f}\nPA:\n  ET: {pa_mean[0]:.4f}±{pa_std[0]:.4f}\n  TC: {pa_mean[1]:.4f}±{pa_std[1]:.4f}\n  WT: {pa_mean[2]:.4f}±{pa_std[2]:.4f}\nRVE:\n  ET: {RVE_mean[0]:.4f}±{RVE_std[0]:.4f}\n  TC: {RVE_mean[1]:.4f}±{RVE_std[1]:.4f}\n  WT: {RVE_mean[2]:.4f}±{RVE_std[2]:.4f}\nSensitivity:\n  ET: {sensitivity_mean[0]:.4f}±{sensitivity_std[0]:.4f}\n  TC: {sensitivity_mean[1]:.4f}±{sensitivity_std[1]:.4f}\n  WT: {sensitivity_mean[2]:.4f}±{sensitivity_std[2]:.4f}\nPPV:\n  ET: {ppv_mean[0]:.4f}±{ppv_std[0]:.4f}\n  TC: {ppv_mean[1]:.4f}±{ppv_std[1]:.4f}\n  WT: {ppv_mean[2]:.4f}±{ppv_std[2]:.4f}\n{'=' * 40}\n'
        result_file = os.path.join(result_dir, f'{model_name}_{args.img}_{timestamp}.txt')
        with open(result_file, 'w') as f:
            f.write(report)
        print(report)
        return True
    except Exception as e:
        error_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        error_msg = f'\n🔥 测试失败: {args.img}\n错误信息: {str(e)}\n追踪信息:\n{traceback.format_exc()}'
        print(error_msg)
        error_log = os.path.join(result_dir, 'test_errors.log')
        with open(error_log, 'a') as f:
            f.write(f'[{error_timestamp}] {error_msg}\n')
        return False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='目标数据集测试脚本')
    parser.add_argument('--source_root', type=str, default='/home/yuwenjing/data/BraTS2024')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/BraTS-SSA', help='目标数据集根目录路径')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints', help='包含预训练权重的检查点目录')
    parser.add_argument('--model_path', type=str, default='default', help='指定模型权重文件的完整路径，使用default则根据model_type自动选择')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['nnunet', 'unet3d'], help='选择模型架构类型 (nnunet 或 unet3d)')
    parser.add_argument('--lr', type=float, default=1e-06)
    parser.add_argument('--gpu', type=int, default=3, help='使用GPU编号')
    parser.add_argument('--img', default=['all'], help='测试2模态')
    parser.add_argument('--batch_test', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  使用设备: {device}')
    print(f'📋 使用模型: {args.model_type}')
    print(f'📦 模型权重路径: {args.model_path}')
    success_count = 0
    start_time = datetime.datetime.now()
    for idx, modality in enumerate(args.img, 1):
        print(f'\n🔍 正在测试 ({idx}/{len(args.img)}) {modality.upper()}')
        modality_args = argparse.Namespace(**vars(args))
        modality_args.img = modality
        if test_on_target(modality_args, device):
            success_count += 1
    total_time = datetime.datetime.now() - start_time
    summary = f'\n{'=' * 40}\n测试总结:\n- 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n- 总耗时: {total_time}\n- 模型类型: {args.model_type}\n- 模型路径: {args.model_path}\n- 成功测试: {success_count}/{len(args.img)}\n- 失败测试: {len(args.img) - success_count}\n{'=' * 40}\n'
    print(summary)
    result_dir = '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints/tta_results'
    model_name = os.path.basename(args.model_path).split('.')[0]
    summary_file = os.path.join(result_dir, f'{model_name}_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
