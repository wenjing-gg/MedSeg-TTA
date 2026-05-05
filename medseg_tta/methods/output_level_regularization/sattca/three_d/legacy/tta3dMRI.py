import argparse
import os
import datetime
import traceback
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from nnunet import PlainConvUNet
from utils_brats_all import get_data_loader
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
import torch.nn as nn
from utils.tent import configure_model, collect_params, Tent

def dia2mask_brats(diameter, shape):
    if len(shape) == 5:
        depth, height, width = (shape[2], shape[3], shape[4])
    elif len(shape) == 4:
        depth, height, width = (shape[1], shape[2], shape[3])
    else:
        depth, height, width = shape
    mask = np.zeros((depth, height, width))
    radius = diameter // 2
    center_z, center_y, center_x = (depth // 2, height // 2, width // 2)
    for z in range(max(0, center_z - radius), min(depth, center_z + radius + 1)):
        for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                dist = ((z - center_z) ** 2 + (y - center_y) ** 2 + (x - center_x) ** 2) ** 0.5
                if dist <= radius:
                    mask[z, y, x] = 1
    return mask

def get_saclick_brats(outputs, masks, diameter_rates, unbiased=False):
    batch_size = outputs.shape[0]
    device = outputs.device
    predict = (outputs.argmax(dim=1, keepdim=True) > 0).float()
    mask = (masks > 0).float()
    sphere_list = []
    fake_rate_list = []
    for i in range(batch_size):
        p = predict[i, 0].cpu().numpy()
        m = mask[i, 0].cpu().numpy()
        boxes = []
        x_proj = np.max(p, axis=(0, 1))
        x_size = np.sum(x_proj > 0)
        boxes.append(x_size if x_size > 0 else p.shape[2])
        y_proj = np.max(p, axis=(0, 2))
        y_size = np.sum(y_proj > 0)
        boxes.append(y_size if y_size > 0 else p.shape[1])
        z_proj = np.max(p, axis=(1, 2))
        z_size = np.sum(z_proj > 0)
        boxes.append(z_size if z_size > 0 else p.shape[0])
        min_box = min(boxes)
        diameter = max(min(int(min_box * diameter_rates[0]), int(min_box ** 2 * diameter_rates[1])), 1)
        sphere = torch.zeros_like(mask[i:i + 1])
        if np.sum(m) > 0:
            indices = np.nonzero(m)
            center_z = int(np.mean(indices[0]))
            center_y = int(np.mean(indices[1]))
            center_x = int(np.mean(indices[2]))
        else:
            center_z = p.shape[0] // 2
            center_y = p.shape[1] // 2
            center_x = p.shape[2] // 2
        d, h, w = sphere.shape[2:]
        radius = diameter // 2
        for zi in range(max(0, center_z - radius), min(d, center_z + radius + 1)):
            for yi in range(max(0, center_y - radius), min(h, center_y + radius + 1)):
                for xi in range(max(0, center_x - radius), min(w, center_x + radius + 1)):
                    dist = ((zi - center_z) ** 2 + (yi - center_y) ** 2 + (xi - center_x) ** 2) ** 0.5
                    if dist <= radius:
                        sphere[0, 0, zi, yi, xi] = 1.0
        if unbiased:
            sphere_real = sphere * mask[i:i + 1]
            fake_rate = (sphere.sum() - sphere_real.sum()) / (sphere.sum() + 1e-08)
            sphere_list.append(sphere_real)
        else:
            sphere_real = sphere * mask[i:i + 1]
            fake_rate = (sphere.sum() - sphere_real.sum()) / (sphere.sum() + 1e-08)
            sphere_list.append(sphere)
        fake_rate_list.append(fake_rate.item())
    spheres = torch.cat(sphere_list, dim=0)
    avg_fake_rate = np.mean(fake_rate_list)
    num_classes = outputs.shape[1]
    expanded_spheres = torch.zeros((batch_size, num_classes, *outputs.shape[2:]), device=device)
    for i in range(num_classes):
        expanded_spheres[:, i:i + 1] = spheres
    return (expanded_spheres, avg_fake_rate)

def _tent_model(model, args):
    model = configure_model(model)
    params, _ = collect_params(model)
    optimizer = optim.Adam(params, lr=args.lr)
    tented_model = Tent(model, optimizer)
    return tented_model

def _init_metric_dict():
    metric_dict = {'et': np.zeros(0), 'tc': np.zeros(0), 'wt': np.zeros(0), 'total': np.zeros(0)}
    return metric_dict

def _init_loss_dict(n):
    loss_dict = {'dice_loss': np.zeros(0), 'bce_loss': np.zeros(0), 'total_loss': np.zeros(n)}
    return loss_dict

def process_tuple_values(values):
    return [float(value.item()) if hasattr(value, 'item') else float(value) for value in values]

def _show_dice(df, names_test, dice_values, hd95_values, iou_values, pa_values, rve_values, sensitivity_values, ppv_values):
    for i, name in enumerate(names_test):
        df['file_id'].append(name)
        et_dice, tc_dice, wt_dice = process_tuple_values(dice_values)
        et_hd95, tc_hd95, wt_hd95 = process_tuple_values(hd95_values)
        et_iou, tc_iou, wt_iou = process_tuple_values(iou_values)
        et_pa, tc_pa, wt_pa = process_tuple_values(pa_values)
        et_rve, tc_rve, wt_rve = process_tuple_values(rve_values)
        et_sensitivity, tc_sensitivity, wt_sensitivity = process_tuple_values(sensitivity_values)
        et_ppv, tc_ppv, wt_ppv = process_tuple_values(ppv_values)
        df['et_dice'].append(et_dice)
        df['tc_dice'].append(tc_dice)
        df['wt_dice'].append(wt_dice)
        df['et_hd95'].append(et_hd95)
        df['tc_hd95'].append(tc_hd95)
        df['wt_hd95'].append(wt_hd95)
        df['et_iou'].append(et_iou)
        df['tc_iou'].append(tc_iou)
        df['wt_iou'].append(wt_iou)
        df['et_pa'].append(et_pa)
        df['tc_pa'].append(tc_pa)
        df['wt_pa'].append(wt_pa)
        df['et_rve'].append(et_rve)
        df['tc_rve'].append(tc_rve)
        df['wt_rve'].append(wt_rve)
        df['et_sensitivity'].append(et_sensitivity)
        df['tc_sensitivity'].append(tc_sensitivity)
        df['wt_sensitivity'].append(wt_sensitivity)
        df['et_ppv'].append(et_ppv)
        df['tc_ppv'].append(tc_ppv)
        df['wt_ppv'].append(wt_ppv)
        print(f'ID: {name}, Dice: ET={et_dice:.4f}, TC={tc_dice:.4f}, WT={wt_dice:.4f}, HD95: ET={et_hd95:.2f}, TC={tc_hd95:.2f}, WT={wt_hd95:.2f}, IoU: ET={et_iou:.4f}, TC={tc_iou:.4f}, WT={wt_iou:.4f}, PA: ET={et_pa:.4f}, TC={tc_pa:.4f}, WT={wt_pa:.4f}, RVE: ET={et_rve:.4f}, TC={tc_rve:.4f}, WT={wt_rve:.4f},Sensitivity: ET={et_sensitivity:.4f}, TC={tc_sensitivity:.4f}, WT={wt_sensitivity:.4f}, PPV: ET={et_ppv:.4f}, TC={tc_ppv:.4f}, WT={wt_ppv:.4f}')
    return df

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 开始在目标数据集上测试模态: {args.img.upper()}')
    print(f'{'=' * 40}\n')
    try:
        result_dir = '/home/yuwenjing/DeepLearning_ywj/tta/SaTTCA/checkpoints/tta_results'
        os.makedirs(result_dir, exist_ok=True)
        metric_dict = {'file_id': [], 'et_dice': [], 'tc_dice': [], 'wt_dice': [], 'et_hd95': [], 'tc_hd95': [], 'wt_hd95': [], 'et_iou': [], 'tc_iou': [], 'wt_iou': [], 'et_pa': [], 'tc_pa': [], 'wt_pa': [], 'et_rve': [], 'tc_rve': [], 'wt_rve': [], 'et_sensitivity': [], 'tc_sensitivity': [], 'wt_sensitivity': [], 'et_ppv': [], 'tc_ppv': [], 'wt_ppv': []}
        loss_test_dict = _init_loss_dict(0)
        dice_test_dict = _init_metric_dict()
        hd95_test_dict = _init_metric_dict()
        iou_test_dict = _init_metric_dict()
        pa_test_dict = _init_metric_dict()
        rve_test_dict = _init_metric_dict()
        sensitivity_test_dict = _init_metric_dict()
        ppv_test_dict = _init_metric_dict()
        if args.model_type.lower() == 'nnunet':
            print(f'📋 加载 nnUNet 模型架构')
            model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
            if args.model_path and args.model_path != 'default':
                best_model_path = args.model_path
            else:
                best_model_path = '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth'
        elif args.model_type.lower() == 'unet3d':
            print(f'📋 加载 UNet3D 模型架构')
            from unet3d import UNet3d
            model = UNet3d().to(device)
            if args.model_path and args.model_path != 'default':
                best_model_path = args.model_path
            else:
                best_model_path = '/home/yuwenjing/DeepLearning_ywj/tta/unet3d_best.pth'
        else:
            raise ValueError(f"不支持的模型类型: {args.model_type}。请选择 'nnunet' 或 'unet3d'")
        print(f'📦 加载模型权重: {best_model_path}')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f'未找到预训练权重: {best_model_path}')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        for param in model.parameters():
            if param.device != device:
                param.data = param.data.to(device)
        tent_model = _tent_model(model, args)
        for param in tent_model.model.parameters():
            if param.device != device:
                param.data = param.data.to(device)
        print(f'🔍 模型参数设备检查: {next(model.parameters()).device}')
        print(f'🔍 TTA模型参数设备检查: {next(tent_model.model.parameters()).device}')
        _, target_test_loader = get_data_loader(source_root=args.source_root, target_root=args.target_root, batch_train=args.batch_test, batch_test=args.batch_test, nw=args.num_workers, img=args.img, mode='source_to_target')
        for batch in tqdm(target_test_loader, desc='推理进度'):
            if len(batch) == 3:
                imgs, labels, file_names = batch
            else:
                imgs, labels = batch[:2]
                file_names = [f'sample_{i}' for i in range(len(batch[0]))]
            imgs, labels = (imgs.to(device), labels.to(device))
            with torch.no_grad():
                model.eval()
                outputs = model(imgs)
            sphere, fake_rate = get_saclick_brats(outputs, labels, [0.8, 0.02], unbiased=False)
            outputs, loss_dict, _ = tent_model([imgs, sphere])
            dice_values = cal_dice(outputs, labels.squeeze(1))
            hd95_values = cal_hd95(outputs, labels.squeeze(1))
            iou_values = IoU(outputs, labels.squeeze(1))
            pa_values = PA(outputs, labels.squeeze(1), 4)
            rve_values = cal_RVE(outputs, labels.squeeze(1))
            sensitivity_values = cal_sensitivity(outputs, labels.squeeze(1))
            ppv_values = cal_ppv(outputs, labels.squeeze(1))
            for k in loss_dict.keys():
                loss_test_dict[k] = np.append(loss_test_dict[k], loss_dict[k].cpu().item())
            dice_values = process_tuple_values(dice_values)
            hd95_values = process_tuple_values(hd95_values)
            iou_values = process_tuple_values(iou_values)
            pa_values = process_tuple_values(pa_values)
            rve_values = process_tuple_values(rve_values)
            sensitivity_values = process_tuple_values(sensitivity_values)
            ppv_values = process_tuple_values(ppv_values)
            for key, value in zip(['dice', 'hd95', 'iou', 'pa', 'rve', 'sensitivity', 'ppv'], [dice_values, hd95_values, iou_values, pa_values, rve_values, sensitivity_values, ppv_values]):
                for idx, region in enumerate(['et', 'tc', 'wt']):
                    eval(f'{key}_test_dict')[region] = np.append(eval(f'{key}_test_dict')[region], value[idx])
                eval(f'{key}_test_dict')['total'] = np.append(eval(f'{key}_test_dict')['total'], np.mean(value))
            _show_dice(metric_dict, file_names, dice_values, hd95_values, iou_values, pa_values, rve_values, sensitivity_values, ppv_values)
        for k in loss_test_dict.keys():
            loss_test_dict[k] = np.mean(loss_test_dict[k])
        stats_dict = {}
        for name, dic in [('dice', dice_test_dict), ('hd95', hd95_test_dict), ('iou', iou_test_dict), ('pa', pa_test_dict), ('rve', rve_test_dict), ('sensitivity', sensitivity_test_dict), ('ppv', ppv_test_dict)]:
            stats_dict[name] = {}
            for key in dic.keys():
                if len(dic[key]) > 0:
                    stats_dict[name][f'{key}_mean'] = np.mean(dic[key])
                    stats_dict[name][f'{key}_std'] = np.std(dic[key])
                else:
                    stats_dict[name][f'{key}_mean'] = 0
                    stats_dict[name][f'{key}_std'] = 0
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report = f'\n        {'=' * 40}\n        测试时间: {timestamp}\n        测试配置:\n        - 图像模态: {args.img}\n        - 模型路径: {best_model_path}\n        - 测试数据: {args.target_root}\n        - 算法: SaTTCA\n\n        损失:\n        - BCE Loss: {loss_test_dict['bce_loss']:.4f}\n        - Dice Loss: {loss_test_dict.get('dice_loss', 'N/A'):.4f}\n        - Total Loss: {loss_test_dict.get('total_loss', 'N/A'):.4f}\n\n        性能指标:\n    Dice (均值±标准差):\n        ET: {stats_dict['dice']['et_mean']:.4f} ± {stats_dict['dice']['et_std']:.4f}\n        TC: {stats_dict['dice']['tc_mean']:.4f} ± {stats_dict['dice']['tc_std']:.4f}\n        WT: {stats_dict['dice']['wt_mean']:.4f} ± {stats_dict['dice']['wt_std']:.4f}\n    HD95(mm) (均值±标准差):\n        ET: {stats_dict['hd95']['et_mean']:.2f} ± {stats_dict['hd95']['et_std']:.2f}\n        TC: {stats_dict['hd95']['tc_mean']:.2f} ± {stats_dict['hd95']['tc_std']:.2f}\n        WT: {stats_dict['hd95']['wt_mean']:.2f} ± {stats_dict['hd95']['wt_std']:.2f}\n    IoU (均值±标准差):\n        ET: {stats_dict['iou']['et_mean']:.4f} ± {stats_dict['iou']['et_std']:.4f}\n        TC: {stats_dict['iou']['tc_mean']:.4f} ± {stats_dict['iou']['tc_std']:.4f}\n        WT: {stats_dict['iou']['wt_mean']:.4f} ± {stats_dict['iou']['wt_std']:.4f}\n    PA (均值±标准差):\n        ET: {stats_dict['pa']['et_mean']:.4f} ± {stats_dict['pa']['et_std']:.4f}\n        TC: {stats_dict['pa']['tc_mean']:.4f} ± {stats_dict['pa']['tc_std']:.4f}\n        WT: {stats_dict['pa']['wt_mean']:.4f} ± {stats_dict['pa']['wt_std']:.4f}\n    RVE (均值±标准差):\n        ET: {stats_dict['rve']['et_mean']:.4f} ± {stats_dict['rve']['et_std']:.4f}\n        TC: {stats_dict['rve']['tc_mean']:.4f} ± {stats_dict['rve']['tc_std']:.4f}\n        WT: {stats_dict['rve']['wt_mean']:.4f} ± {stats_dict['rve']['wt_std']:.4f}\n    Sensitivity (均值±标准差):\n        ET: {stats_dict['sensitivity']['et_mean']:.4f} ± {stats_dict['sensitivity']['et_std']:.4f}\n        TC: {stats_dict['sensitivity']['tc_mean']:.4f} ± {stats_dict['sensitivity']['tc_std']:.4f}\n        WT: {stats_dict['sensitivity']['wt_mean']:.4f} ± {stats_dict['sensitivity']['wt_std']:.4f}\n    PPV (均值±标准差):\n        ET: {stats_dict['ppv']['et_mean']:.4f} ± {stats_dict['ppv']['et_std']:.4f}\n        TC: {stats_dict['ppv']['tc_mean']:.4f} ± {stats_dict['ppv']['tc_std']:.4f}\n        WT: {stats_dict['ppv']['wt_mean']:.4f} ± {stats_dict['ppv']['wt_std']:.4f}\n        {'=' * 40}\n        '
        summary_stats = {'metric': [], 'region': [], 'mean': [], 'std': []}
        for metric in ['dice', 'hd95', 'iou', 'pa', 'rve', 'sensitivity', 'ppv']:
            for region in ['et', 'tc', 'wt']:
                summary_stats['metric'].append(metric)
                summary_stats['region'].append(region)
                summary_stats['mean'].append(stats_dict[metric][f'{region}_mean'])
                summary_stats['std'].append(stats_dict[metric][f'{region}_std'])
        save_dir = os.path.join(result_dir, f'{args.model_type}_tta_weights', args.img)
        os.makedirs(save_dir, exist_ok=True)
        avg_dice = np.mean([stats_dict['dice']['et_mean'], stats_dict['dice']['tc_mean'], stats_dict['dice']['wt_mean']])
        filename = f'{args.model_type}_SaTTCA_SSA.pth'
        save_path = os.path.join(save_dir, filename)
        torch.save(tent_model.model.state_dict(), save_path)
        print(f'💾 SaTTCA后模型权重已保存到: {save_path}')
        result_file = os.path.join(result_dir, f'sattca_{args.img}_{timestamp}.txt')
        with open(result_file, 'w') as f:
            f.write(report)
        metric_df = pd.DataFrame(metric_dict)
        csv_file = os.path.join(result_dir, f'sattca_{args.img}_{timestamp}.csv')
        metric_df.to_csv(csv_file, mode='w', header=True, index=False)
        summary_df = pd.DataFrame(summary_stats)
        summary_csv = os.path.join(result_dir, f'sattca_{args.img}_{timestamp}_summary.csv')
        summary_df.to_csv(summary_csv, mode='w', header=True, index=False)
        print(report)
        print(f'详细结果已保存到: {csv_file}')
        print(f'统计摘要已保存到: {summary_csv}')
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
    parser = argparse.ArgumentParser(description='SaTTCA 算法在BraTS数据集上的测试')
    parser.add_argument('--source_root', type=str, default='/home/yuwenjing/data/BraTS2024')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/BraTS-SSA', help='目标数据集根目录路径')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/SaTTCA/checkpoints', help='包含预训练权重的检查点目录')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['nnunet', 'unet3d'], help='选择模型架构类型 (nnunet 或 unet3d)')
    parser.add_argument('--model_path', type=str, default='default', help='指定模型权重文件的完整路径，使用default则根据model_type自动选择')
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--gpu', type=int, default=3, help='使用GPU编号')
    parser.add_argument('--img', default=['all'], help='测试模态')
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
    summary = f'\n{'=' * 40}\nSaTTCA测试总结:\n- 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n- 总耗时: {total_time}\n- 模型类型: {args.model_type}\n- 模型路径: {args.model_path}\n- 成功测试: {success_count}/{len(args.img)}\n- 失败测试: {len(args.img) - success_count}\n{'=' * 40}\n'
    print(summary)
    result_dir = '/home/yuwenjing/DeepLearning_ywj/tta/SaTTCA/checkpoints/tta_results'
    summary_file = os.path.join(result_dir, f'sattca_summary_{args.model_type}_{start_time.strftime('%Y%m%d_%H%M%S')}.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
