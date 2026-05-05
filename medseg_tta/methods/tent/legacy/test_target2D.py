import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datetime
from dataset2D import MedicalImageDataset2D, get_dataset_type_from_path
from unet2d import UNet2d
from train_source2D import SegmentationLoss2D, calculate_all_metrics, test_model

def create_test_dataset(test_dir: str, image_size: tuple=(256, 256)) -> MedicalImageDataset2D:
    image_dir = os.path.join(test_dir, 'image')
    mask_dir = os.path.join(test_dir, 'mask')
    if not os.path.exists(image_dir):
        raise ValueError(f'Image directory does not exist: {image_dir}')
    if not os.path.exists(mask_dir):
        raise ValueError(f'Mask directory does not exist: {mask_dir}')
    test_dataset = MedicalImageDataset2D(image_dir=image_dir, mask_dir=mask_dir, phase='test', image_size=image_size, normalize=True)
    return test_dataset

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise ValueError(f'Checkpoint file does not exist: {checkpoint_path}')
    print(f'Loading checkpoint from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet2d(in_channels=1, n_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'✓ Model loaded successfully!')
    print(f'  Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}')
    print(f'  Best Dice score: {checkpoint.get('best_dice', 'Unknown'):.4f}')
    if 'args' in checkpoint:
        print(f'  Training args: {checkpoint['args']}')
    return model

def test_on_target_domain(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device, target_name: str) -> dict:
    model.eval()
    test_loss = 0.0
    total_metrics = {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'ppv': 0.0, 'hd95': 0.0}
    num_batches = len(test_loader)
    print(f'\nTesting on target domain: {target_name}')
    print(f'Number of test batches: {num_batches}')
    test_pbar = tqdm(test_loader, desc=f'Testing on {target_name}', bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
    individual_results = []
    with torch.no_grad():
        for i, (images, masks, filenames) in enumerate(test_pbar):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            batch_metrics = calculate_all_metrics(outputs, masks)
            test_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
            batch_size = images.shape[0]
            for j in range(batch_size):
                single_output = outputs[j:j + 1]
                single_mask = masks[j:j + 1]
                single_metrics = calculate_all_metrics(single_output, single_mask)
                individual_results.append({'filename': filenames[j], 'dice': single_metrics['dice'], 'iou': single_metrics['iou'], 'sensitivity': single_metrics['sensitivity'], 'ppv': single_metrics['ppv'], 'hd95': single_metrics['hd95']})
            test_pbar.set_postfix({'loss': f'{test_loss / (i + 1):.4f}', 'dice': f'{batch_metrics['dice']:.3f}', 'iou': f'{batch_metrics['iou']:.3f}', 'sens': f'{batch_metrics['sensitivity']:.3f}', 'ppv': f'{batch_metrics['ppv']:.3f}'})
    avg_loss = test_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
    print(f'\n{'=' * 60}')
    print(f'TARGET DOMAIN TEST RESULTS - {target_name}')
    print(f'{'=' * 60}')
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Dice Score: {avg_metrics['dice']:.4f} ± {np.std([r['dice'] for r in individual_results]):.4f}')
    print(f'IoU: {avg_metrics['iou']:.4f} ± {np.std([r['iou'] for r in individual_results]):.4f}')
    print(f'Sensitivity: {avg_metrics['sensitivity']:.4f} ± {np.std([r['sensitivity'] for r in individual_results]):.4f}')
    print(f'PPV (Precision): {avg_metrics['ppv']:.4f} ± {np.std([r['ppv'] for r in individual_results]):.4f}')
    print(f'HD95: {avg_metrics['hd95']:.2f} ± {np.std([r['hd95'] for r in individual_results]):.2f}')
    print(f'Number of test samples: {len(individual_results)}')
    dice_scores = [r['dice'] for r in individual_results]
    best_idx = np.argmax(dice_scores)
    worst_idx = np.argmin(dice_scores)
    print(f'\nBest performing sample:')
    print(f'  File: {individual_results[best_idx]['filename']}')
    print(f'  Dice: {individual_results[best_idx]['dice']:.4f}')
    print(f'\nWorst performing sample:')
    print(f'  File: {individual_results[worst_idx]['filename']}')
    print(f'  Dice: {individual_results[worst_idx]['dice']:.4f}')
    result = {'avg_metrics': avg_metrics, 'avg_loss': avg_loss, 'individual_results': individual_results, 'summary_stats': {'dice_mean': avg_metrics['dice'], 'dice_std': np.std(dice_scores), 'dice_min': np.min(dice_scores), 'dice_max': np.max(dice_scores), 'num_samples': len(individual_results)}}
    return result

def save_test_results(results: dict, checkpoint_path: str, target_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(output_dir, f'target_test_results_{target_name}_{timestamp}.txt')
    with open(result_file, 'w') as f:
        f.write(f'TARGET DOMAIN TEST RESULTS\n')
        f.write(f'{'=' * 60}\n')
        f.write(f'Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n')
        f.write(f'Checkpoint: {checkpoint_path}\n')
        f.write(f'Target Domain: {target_name}\n')
        f.write(f'Number of Samples: {results['summary_stats']['num_samples']}\n')
        f.write(f'\nAVERAGE METRICS:\n')
        f.write(f'Test Loss: {results['avg_loss']:.4f}\n')
        f.write(f'Dice Score: {results['avg_metrics']['dice']:.4f} ± {results['summary_stats']['dice_std']:.4f}\n')
        f.write(f'IoU: {results['avg_metrics']['iou']:.4f}\n')
        f.write(f'Sensitivity: {results['avg_metrics']['sensitivity']:.4f}\n')
        f.write(f'PPV: {results['avg_metrics']['ppv']:.4f}\n')
        f.write(f'HD95: {results['avg_metrics']['hd95']:.2f}\n')
        f.write(f'\nDICE STATISTICS:\n')
        f.write(f'Mean: {results['summary_stats']['dice_mean']:.4f}\n')
        f.write(f'Std: {results['summary_stats']['dice_std']:.4f}\n')
        f.write(f'Min: {results['summary_stats']['dice_min']:.4f}\n')
        f.write(f'Max: {results['summary_stats']['dice_max']:.4f}\n')
        f.write(f'\nINDIVIDUAL RESULTS:\n')
        f.write(f'{'Filename':<30} {'Dice':<8} {'IoU':<8} {'Sens':<8} {'PPV':<8} {'HD95':<8}\n')
        f.write(f'{'-' * 80}\n')
        for result in results['individual_results']:
            f.write(f'{result['filename']:<30} {result['dice']:<8.4f} {result['iou']:<8.4f} {result['sensitivity']:<8.4f} {result['ppv']:<8.4f} {result['hd95']:<8.2f}\n')
    print(f'\n✓ Test results saved to: {result_file}')
    return result_file

def main():
    parser = argparse.ArgumentParser(description='Test pre-trained model on target domain')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file (e.g., /path/to/unet2d_best_CXR.pth)')
    parser.add_argument('--target_dir', type=str, required=True, help='Target domain directory (e.g., /path/to/TTA-2DCXR/Montgomery)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading (default: 4)')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for testing (default: 256)')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Output directory for test results (default: ./test_results)')
    args = parser.parse_args()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    target_name = os.path.basename(args.target_dir.rstrip('/'))
    print(f'Target domain: {target_name}')
    try:
        print(f'\nCreating test dataset from: {args.target_dir}')
        test_dataset = create_test_dataset(test_dir=args.target_dir, image_size=(args.image_size, args.image_size))
        print(f'✓ Test dataset created successfully!')
        print(f'  Dataset type: {test_dataset.dataset_type}')
        print(f'  Number of test samples: {len(test_dataset)}')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        model = load_model_from_checkpoint(args.checkpoint, device)
        criterion = SegmentationLoss2D(ce_weight=1.0, dice_weight=1.0, class_weights=None, smooth=1e-06, ignore_background=False)
        results = test_on_target_domain(model=model, test_loader=test_loader, criterion=criterion, device=device, target_name=target_name)
        save_test_results(results=results, checkpoint_path=args.checkpoint, target_name=target_name, output_dir=args.output_dir)
        print(f'\n🎉 Target domain testing completed successfully!')
    except Exception as e:
        print(f'❌ Error during testing: {e}')
        import traceback
        traceback.print_exc()
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print('Running with default parameters...')
        default_args = ['--checkpoint', '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoint_PATH/unet2d_best_PATH.pth', '--target_dir', '/home/yuwenjing/data/tta_dataset/TTA-2DPATH/CRAG_processed', '--batch_size', '64', '--num_workers', '2', '--image_size', '256', '--output_dir', './target_test_results']
        sys.argv.extend(default_args)
    main()
