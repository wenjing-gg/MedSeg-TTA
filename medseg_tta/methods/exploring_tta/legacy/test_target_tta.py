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
import pickle
from copy import deepcopy
from test_time_adaptation import adaptation_base, tent, hist_matching, entropy_KL, filter_inspect_utils

def load_base_model(model, device):
    return model

def load_hist_match_model(model, device, volume_to_match_to):
    return hist_matching.HistMatching(model, volume_to_match_to)

def load_tent_model(model, device, lr=0.0001, steps=1):
    model = adaptation_base.configure_model(model)
    params, _ = adaptation_base.collect_batch_norm_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    tented_model = tent.Tent(model, optimizer, steps=steps).to(device)
    return tented_model

def load_entropy_KL_model(model, device, lr=0.0001, lambd=1.0, steps=1, atlas_labels_path=None):
    if atlas_labels_path is None:
        raise ValueError('Atlas labels path is required for Entropy+KL method')
    model = adaptation_base.configure_model(model)
    params, _ = adaptation_base.collect_batch_norm_params(model)
    optimizer = torch.optim.Adam(params, lr=lr)
    entropy_KL_model = entropy_KL.EntropyKL(model, optimizer, atlas_labels_path, lambd=lambd, steps=steps).to(device)
    return entropy_KL_model

def load_source_data_activations(source_data_activations_path):
    if os.path.exists(source_data_activations_path):
        print('Source data activations already exist, loading them...', flush=True)
        with open(source_data_activations_path, 'rb') as f:
            source_data_activations = pickle.load(f)
    else:
        raise ValueError('Source data activations do not exist.')
    return source_data_activations

def load_filter_inspector_model(model, device, subject_list, lr=0.0001, steps=1, source_data_activations_path=None, num_to_update=1, week_num=21, force_include_batch_norm=False, use_KL=False, lambd=1.0, atlas_labels_path=None):
    if source_data_activations_path is None:
        raise ValueError('Source data activations path is required for filter inspection method')
    filter_inspect_config = {'week_num': week_num, 'steps': steps, 'lr': lr, 'num_to_update': num_to_update, 'subject_list': subject_list, 'device': device, 'filter_inspect_mode': 'Taylor', 'force_include_batch_norm': force_include_batch_norm, 'use_KL': use_KL, 'hemisphere_split': False, 'lambda': lambd, 'atlas_labels_path': atlas_labels_path}
    source_data_activations = load_source_data_activations(source_data_activations_path)
    filter_inspector = filter_inspect_utils.create_filter_inspector(model, use_cuda=torch.cuda.is_available())
    filter_inspect_model = filter_inspect_utils.configure_filter_inspect(filter_inspector.unet, filter_inspector, None, subject_list, source_data_activations, filter_inspect_config)
    return filter_inspect_model

def create_tta_model(model, args, device):
    tta_method = getattr(args, 'tta_method', 'tent').lower()
    try:
        if tta_method == 'none' or tta_method == 'baseline':
            print('🔧 使用基线模型 (无适应)')
            return model
        elif tta_method == 'tent':
            print('🔧 创建TENT适应模型')
            return load_tent_model(model, device, lr=args.lr, steps=getattr(args, 'tta_steps', 1))
        elif tta_method == 'entropy_kl':
            print('🔧 创建Entropy+KL适应模型')
            atlas_path = getattr(args, 'atlas_labels_path', None)
            lambd = getattr(args, 'kl_lambda', 1.0)
            return load_entropy_KL_model(model, device, lr=args.lr, lambd=lambd, steps=getattr(args, 'tta_steps', 1), atlas_labels_path=atlas_path)
        elif tta_method == 'hist_matching':
            print('🔧 创建直方图匹配模型')
            volume_path = getattr(args, 'reference_volume_path', None)
            if volume_path is None:
                raise ValueError('Reference volume path is required for histogram matching')
            reference_volume = torch.load(volume_path)
            return load_hist_match_model(model, device, reference_volume)
        elif tta_method == 'filter_inspect':
            print('🔧 创建滤波器检查模型')
            subject_list = getattr(args, 'subject_list', [])
            source_activations_path = getattr(args, 'source_data_activations_path', None)
            return load_filter_inspector_model(model, device, subject_list, lr=args.lr, steps=getattr(args, 'tta_steps', 1), source_data_activations_path=source_activations_path, num_to_update=getattr(args, 'num_filters_to_update', 1), force_include_batch_norm=getattr(args, 'force_include_batch_norm', False), use_KL=getattr(args, 'use_KL', False), lambd=getattr(args, 'kl_lambda', 1.0), atlas_labels_path=getattr(args, 'atlas_labels_path', None))
        else:
            raise ValueError(f'Unknown TTA method: {tta_method}. Supported methods: none, tent, entropy_kl, hist_matching, filter_inspect')
    except Exception as e:
        print(f'❌ 创建TTA模型失败: {e}')
        print('🔄 回退到基线模型')
        return model

def _tent_model(model, args):
    return create_tta_model(model, args, model.device if hasattr(model, 'device') else torch.device('cuda:0'))

def validate_tta_args(args):
    if args.tta_method == 'entropy_kl':
        if args.atlas_labels_path is None:
            raise ValueError('Entropy+KL method requires --atlas_labels_path argument')
        if not os.path.exists(args.atlas_labels_path):
            raise FileNotFoundError(f'Atlas labels file not found: {args.atlas_labels_path}')
    elif args.tta_method == 'hist_matching':
        if args.reference_volume_path is None:
            raise ValueError('Histogram matching method requires --reference_volume_path argument')
        if not os.path.exists(args.reference_volume_path):
            raise FileNotFoundError(f'Reference volume file not found: {args.reference_volume_path}')
    elif args.tta_method == 'filter_inspect':
        if args.source_data_activations_path is None:
            raise ValueError('Filter inspection method requires --source_data_activations_path argument')
        if not os.path.exists(args.source_data_activations_path):
            raise FileNotFoundError(f'Source data activations file not found: {args.source_data_activations_path}')
    if args.lr <= 0:
        raise ValueError('Learning rate must be positive')
    if args.tta_steps <= 0:
        raise ValueError('TTA steps must be positive')
    if args.kl_lambda < 0:
        raise ValueError('KL lambda must be non-negative')

def _init_metric_dict():
    metric_dict = {'et': np.zeros(0), 'tc': np.zeros(0), 'wt': np.zeros(0), 'total': np.zeros(0)}
    return metric_dict

def _init_loss_dict(n):
    loss_dict = {'dice_loss': np.zeros(0), 'bce_loss': np.zeros(0), 'total_loss': np.zeros(n)}
    return loss_dict

def process_tuple_values(values):
    return [float(value.item()) if hasattr(value, 'item') else float(value) for value in values]

def _show_dice(df, names_test, dice_values, hd95_values, iou_values, pa_values, rve_values, sensitivity_values, ppv_values):
    for name in names_test:
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
        result_dir = '/home/yuwenjing/DeepLearning_ywj/tta/ExploringTTA/checkpoints/tta_results'
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
        print(f'🔧 创建TTA模型，方法: {getattr(args, 'tta_method', 'tent')}')
        tta_model = create_tta_model(model, args, device)
        if hasattr(tta_model, 'model'):
            for param in tta_model.model.parameters():
                if param.device != device:
                    param.data = param.data.to(device)
            print(f'🔍 TTA模型参数设备检查: {next(tta_model.model.parameters()).device}')
        else:
            for param in tta_model.parameters():
                if param.device != device:
                    param.data = param.data.to(device)
            print(f'🔍 TTA模型参数设备检查: {next(tta_model.parameters()).device}')
        print(f'🔍 原始模型参数设备检查: {next(model.parameters()).device}')
        _, target_test_loader = get_data_loader(source_root=args.source_root, target_root=args.target_root, batch_train=args.batch_test, batch_test=args.batch_test, nw=args.num_workers, img=args.img, mode='source_to_target')
        for batch in tqdm(target_test_loader, desc='推理进度'):
            if len(batch) == 3:
                imgs, labels, file_names = batch
            else:
                imgs, labels = batch[:2]
                file_names = [f'sample_{i}' for i in range(len(batch[0]))]
            imgs, labels = (imgs.to(device), labels.to(device))
            tta_method = getattr(args, 'tta_method', 'tent').lower()
            if tta_method == 'none' or tta_method == 'baseline':
                with torch.no_grad():
                    tta_model.eval()
                    outputs = tta_model(imgs)
            elif hasattr(tta_model, 'forward'):
                outputs = tta_model(imgs)
            else:
                with torch.no_grad():
                    outputs = tta_model(imgs)
            dice_values = cal_dice(outputs, labels.squeeze(1))
            hd95_values = cal_hd95(outputs, labels.squeeze(1))
            iou_values = IoU(outputs, labels.squeeze(1))
            pa_values = PA(outputs, labels.squeeze(1), 4)
            rve_values = cal_RVE(outputs, labels.squeeze(1))
            sensitivity_values = cal_sensitivity(outputs, labels.squeeze(1))
            ppv_values = cal_ppv(outputs, labels.squeeze(1))
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
        report = f'\n        {'=' * 40}\n        测试时间: {timestamp}\n        测试配置:\n        - 图像模态: {args.img}\n        - 模型类型: {args.model_type}\n        - 模型路径: {best_model_path}\n        - 测试数据: {args.target_root}\n        - TTA方法: {getattr(args, 'tta_method', 'tent')}\n        - 学习率: {args.lr}\n        - 适应步数: {getattr(args, 'tta_steps', 1)}\n        - 算法: ExploringTTA\n\n        损失:\n        - BCE Loss: {loss_test_dict['bce_loss']:.4f}\n        - Dice Loss: {loss_test_dict.get('dice_loss', 'N/A'):.4f}\n        - Total Loss: {loss_test_dict.get('total_loss', 'N/A'):.4f}\n\n        性能指标:\n    Dice (均值±标准差):\n        ET: {stats_dict['dice']['et_mean']:.4f} ± {stats_dict['dice']['et_std']:.4f}\n        TC: {stats_dict['dice']['tc_mean']:.4f} ± {stats_dict['dice']['tc_std']:.4f}\n        WT: {stats_dict['dice']['wt_mean']:.4f} ± {stats_dict['dice']['wt_std']:.4f}\n    HD95(mm) (均值±标准差):\n        ET: {stats_dict['hd95']['et_mean']:.2f} ± {stats_dict['hd95']['et_std']:.2f}\n        TC: {stats_dict['hd95']['tc_mean']:.2f} ± {stats_dict['hd95']['tc_std']:.2f}\n        WT: {stats_dict['hd95']['wt_mean']:.2f} ± {stats_dict['hd95']['wt_std']:.2f}\n    IoU (均值±标准差):\n        ET: {stats_dict['iou']['et_mean']:.4f} ± {stats_dict['iou']['et_std']:.4f}\n        TC: {stats_dict['iou']['tc_mean']:.4f} ± {stats_dict['iou']['tc_std']:.4f}\n        WT: {stats_dict['iou']['wt_mean']:.4f} ± {stats_dict['iou']['wt_std']:.4f}\n    PA (均值±标准差):\n        ET: {stats_dict['pa']['et_mean']:.4f} ± {stats_dict['pa']['et_std']:.4f}\n        TC: {stats_dict['pa']['tc_mean']:.4f} ± {stats_dict['pa']['tc_std']:.4f}\n        WT: {stats_dict['pa']['wt_mean']:.4f} ± {stats_dict['pa']['wt_std']:.4f}\n    RVE (均值±标准差):\n        ET: {stats_dict['rve']['et_mean']:.4f} ± {stats_dict['rve']['et_std']:.4f}\n        TC: {stats_dict['rve']['tc_mean']:.4f} ± {stats_dict['rve']['tc_std']:.4f}\n        WT: {stats_dict['rve']['wt_mean']:.4f} ± {stats_dict['rve']['wt_std']:.4f}\n    Sensitivity (均值±标准差):\n        ET: {stats_dict['sensitivity']['et_mean']:.4f} ± {stats_dict['sensitivity']['et_std']:.4f}\n        TC: {stats_dict['sensitivity']['tc_mean']:.4f} ± {stats_dict['sensitivity']['tc_std']:.4f}\n        WT: {stats_dict['sensitivity']['wt_mean']:.4f} ± {stats_dict['sensitivity']['wt_std']:.4f}\n    PPV (均值±标准差):\n        ET: {stats_dict['ppv']['et_mean']:.4f} ± {stats_dict['ppv']['et_std']:.4f}\n        TC: {stats_dict['ppv']['tc_mean']:.4f} ± {stats_dict['ppv']['tc_std']:.4f}\n        WT: {stats_dict['ppv']['wt_mean']:.4f} ± {stats_dict['ppv']['wt_std']:.4f}\n        {'=' * 40}\n        '
        summary_stats = {'metric': [], 'region': [], 'mean': [], 'std': []}
        for metric in ['dice', 'hd95', 'iou', 'pa', 'rve', 'sensitivity', 'ppv']:
            for region in ['et', 'tc', 'wt']:
                summary_stats['metric'].append(metric)
                summary_stats['region'].append(region)
                summary_stats['mean'].append(stats_dict[metric][f'{region}_mean'])
                summary_stats['std'].append(stats_dict[metric][f'{region}_std'])
        tta_method_name = getattr(args, 'tta_method', 'tent')
        result_file = os.path.join(result_dir, f'ExploringTTA_{tta_method_name}_{args.img}_{timestamp}.txt')
        with open(result_file, 'w') as f:
            f.write(report)
        metric_df = pd.DataFrame(metric_dict)
        csv_file = os.path.join(result_dir, f'ExploringTTA_{tta_method_name}_{args.img}_{timestamp}.csv')
        metric_df.to_csv(csv_file, mode='w', header=True, index=False)
        summary_df = pd.DataFrame(summary_stats)
        summary_csv = os.path.join(result_dir, f'ExploringTTA_{tta_method_name}_{args.img}_{timestamp}_summary.csv')
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
    parser = argparse.ArgumentParser(description='ExploringTTA 算法在BraTS数据集上的测试')
    parser.add_argument('--source_root', type=str, default='/home/yuwenjing/data/BraTS2024')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/BraTS-PED2023/Train', help='目标数据集根目录路径')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/ExploringTTA/checkpoints', help='包含预训练权重的检查点目录')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['nnunet', 'unet3d'], help='选择模型架构类型 (nnunet 或 unet3d)')
    parser.add_argument('--model_path', type=str, default='default', help='指定模型权重文件的完整路径，使用default则根据model_type自动选择')
    parser.add_argument('--tta_method', type=str, default='tent', choices=['none', 'baseline', 'tent', 'entropy_kl', 'hist_matching', 'filter_inspect'], help='选择测试时适应方法')
    parser.add_argument('--lr', type=float, default=0.001, help='适应学习率')
    parser.add_argument('--tta_steps', type=int, default=1, help='每个样本的适应步数')
    parser.add_argument('--kl_lambda', type=float, default=1.0, help='KL散度损失权重')
    parser.add_argument('--atlas_labels_path', type=str, default=None, help='Atlas标签文件路径 (Entropy+KL方法需要)')
    parser.add_argument('--reference_volume_path', type=str, default=None, help='参考体积文件路径 (直方图匹配方法需要)')
    parser.add_argument('--source_data_activations_path', type=str, default=None, help='源数据激活文件路径 (滤波器检查方法需要)')
    parser.add_argument('--num_filters_to_update', type=int, default=1, help='要更新的滤波器数量')
    parser.add_argument('--force_include_batch_norm', action='store_true', help='强制包含批归一化层')
    parser.add_argument('--use_KL', action='store_true', help='在滤波器检查中使用KL散度')
    parser.add_argument('--gpu', type=int, default=1, help='使用GPU编号')
    parser.add_argument('--img', default=['all'], help='测试模态')
    parser.add_argument('--batch_test', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    try:
        validate_tta_args(args)
        print('✅ 参数验证通过')
    except (ValueError, FileNotFoundError) as e:
        print(f'❌ 参数验证失败: {e}')
        exit(1)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  使用设备: {device}')
    print(f'📋 使用模型: {args.model_type}')
    print(f'📦 模型权重路径: {args.model_path}')
    print(f'🔧 TTA方法: {args.tta_method}')
    print(f'📊 学习率: {args.lr}')
    print(f'🔄 适应步数: {args.tta_steps}')
    if args.tta_method == 'entropy_kl':
        print(f'🎯 Atlas标签路径: {args.atlas_labels_path}')
        print(f'⚖️  KL权重: {args.kl_lambda}')
    elif args.tta_method == 'hist_matching':
        print(f'📊 参考体积路径: {args.reference_volume_path}')
    elif args.tta_method == 'filter_inspect':
        print(f'🔍 源数据激活路径: {args.source_data_activations_path}')
        print(f'🎛️  更新滤波器数: {args.num_filters_to_update}')
        print(f'🔧 强制包含BN: {args.force_include_batch_norm}')
        print(f'📈 使用KL: {args.use_KL}')
    success_count = 0
    start_time = datetime.datetime.now()
    for idx, modality in enumerate(args.img, 1):
        print(f'\n🔍 正在测试 ({idx}/{len(args.img)}) {modality.upper()}')
        modality_args = argparse.Namespace(**vars(args))
        modality_args.img = modality
        if test_on_target(modality_args, device):
            success_count += 1
    total_time = datetime.datetime.now() - start_time
    summary = f'\n{'=' * 40}\nExploringTTA测试总结:\n- 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n- 总耗时: {total_time}\n- 模型类型: {args.model_type}\n- 模型路径: {args.model_path}\n- TTA方法: {args.tta_method}\n- 学习率: {args.lr}\n- 适应步数: {args.tta_steps}\n- 成功测试: {success_count}/{len(args.img)}\n- 失败测试: {len(args.img) - success_count}\n{'=' * 40}\n'
    print(summary)
    result_dir = '/home/yuwenjing/DeepLearning_ywj/tta/ExploringTTA/checkpoints/tta_results'
    summary_file = os.path.join(result_dir, f'ExploringTTA_summary_{args.tta_method}_{args.model_type}_{start_time.strftime('%Y%m%d_%H%M%S')}.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
