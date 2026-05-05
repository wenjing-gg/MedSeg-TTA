import argparse
import os
import datetime
import traceback
import torch
from tqdm import tqdm
from nnunet import PlainConvUNet
from dataloader import get_data_loader
from metrics import cal_dice, cal_hd95
import torch.nn as nn

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

def test_on_target(args, device):
    print(f'\n{'=' * 40}')
    print(f'🧪 开始在目标数据集上测试模态: {args.img.upper()}')
    print(f'{'=' * 40}\n')
    try:
        model = PlainConvUNet(1, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
        best_model_path = os.path.join(args.checkpoint_dir, args.img, 'nnunet_best.pth')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f'未找到预训练权重: {best_model_path}')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        _, target_test_loader = get_data_loader(source_root=args.source_root, target_root=args.target_root, batch_train=args.batch_test, batch_test=args.batch_test, nw=args.num_workers, img=args.img, mode='source_to_target')
        total_dice = [0.0] * 3
        total_hd95 = [0.0] * 3
        metric_count = 0
        with torch.no_grad():
            for imgs, labels, *_ in tqdm(target_test_loader, desc='推理进度'):
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model.forward(imgs)
                dice_values = cal_dice(outputs, labels.squeeze(1))
                hd95_values = cal_hd95(outputs, labels.squeeze(1))
                for i in range(3):
                    total_dice[i] += dice_values[i].item()
                    total_hd95[i] += hd95_values[i]
                metric_count += 1
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report = f'\n{'=' * 40}\n测试时间: {timestamp}\n测试配置:\n- 图像模态: {args.img}\n- 模型路径: {best_model_path}\n- 测试数据: {args.target_root}\n\n性能指标:\nDice:\n  ET: {total_dice[0] / metric_count:.4f}\n  TC: {total_dice[1] / metric_count:.4f}\n  WT: {total_dice[2] / metric_count:.4f}\n\nHD95(mm):\n  ET: {total_hd95[0] / metric_count:.2f}\n  TC: {total_hd95[1] / metric_count:.2f}\n  WT: {total_hd95[2] / metric_count:.2f}\n{'=' * 40}\n'
        result_file = os.path.join(args.checkpoint_dir, f'test_{args.img}_{timestamp}.txt')
        with open(result_file, 'w') as f:
            f.write(report)
        print(report)
        return True
    except Exception as e:
        error_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        error_msg = f'\n🔥 测试失败: {args.img}\n错误信息: {str(e)}\n追踪信息:\n{traceback.format_exc()}'
        print(error_msg)
        error_log = os.path.join(args.checkpoint_dir, 'test_errors.log')
        with open(error_log, 'a') as f:
            f.write(f'[{error_timestamp}] {error_msg}\n')
        return False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='目标数据集测试脚本')
    parser.add_argument('--source_root', type=str, default='/home/yuwenjing/data/BraTS2024')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/BraTS-SSA', help='目标数据集根目录路径')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints', help='包含预训练权重的检查点目录')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--gpu', type=int, default=0, help='使用GPU编号')
    parser.add_argument('--img', default=['t1c', 't1n', 't2f', 't2w'], help='测试模态')
    parser.add_argument('--batch_test', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  使用设备: {device}')
    success_count = 0
    start_time = datetime.datetime.now()
    for idx, modality in enumerate(args.img, 1):
        print(f'\n🔍 正在测试 ({idx}/{len(args.img)}) {modality.upper()}')
        modality_args = argparse.Namespace(**vars(args))
        modality_args.img = modality
        if test_on_target(modality_args, device):
            success_count += 1
    total_time = datetime.datetime.now() - start_time
    summary = f'\n{'=' * 40}\n测试总结:\n- 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n- 总耗时: {total_time}\n- 成功测试: {success_count}/{len(args.img)}\n- 失败测试: {len(args.img) - success_count}\n{'=' * 40}\n'
    print(summary)
    summary_file = os.path.join(args.checkpoint_dir, f'test_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
