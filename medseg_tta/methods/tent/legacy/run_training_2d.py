import os
import sys
import argparse
from train_source2D import train

def main():
    parser = argparse.ArgumentParser(description='Train 2D UNet for Medical Image Segmentation')
    parser.add_argument('--subfolder', type=str, help='Subfolder name (optional, auto-selects folder with underscore suffix if not specified)')
    parser.add_argument('--base_dir', type=str, default='/home/yuwenjing/data/tta_dataset', help='Base directory containing all datasets')
    parser.add_argument('--batch_train', type=int, default=64, help='Training batch size')
    parser.add_argument('--batch_test', type=int, default=64, help='Test batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-05, help='Weight decay')
    parser.add_argument('--train_split', type=float, default=0.9, help='Training data split ratio (9:1 for train:test)')
    parser.add_argument('--image_size', type=int, default=256, help='Input image size')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--min_lr', type=float, default=1e-06, help='Minimum learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_2d', help='Directory to save checkpoints')
    args = parser.parse_args()
    dataset_types = ['PATH']
    print(f'Training datasets: {', '.join(dataset_types)}')
    for dataset_type in dataset_types:
        args.dataset_type = dataset_type
        args.checkpoint_dir = f'checkpoint_{dataset_type}'
        print(f'\n{'=' * 60}')
        print(f'Starting training for dataset: {dataset_type}')
        print(f'Checkpoint directory: {args.checkpoint_dir}')
        print(f'{'=' * 60}\n')
        if not os.path.exists(args.base_dir):
            print(f'Error: Base directory not found: {args.base_dir}')
            continue
        from dataset2D import get_dataset_paths
        try:
            image_dir, mask_dir = get_dataset_paths(args.dataset_type, args.base_dir, args.subfolder)
            print(f'Dataset paths resolved:')
            print(f'  Image directory: {image_dir}')
            print(f'  Mask directory: {mask_dir}')
            if not os.path.exists(image_dir):
                print(f'Error: Image directory not found: {image_dir}')
                continue
            if not os.path.exists(mask_dir):
                print(f'Error: Mask directory not found: {mask_dir}')
                continue
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            print(f'  Found {len(image_files)} image files')
            print(f'  Found {len(mask_files)} mask files')
            if len(image_files) == 0:
                print(f'Error: No image files found in {image_dir}')
                continue
            if len(mask_files) == 0:
                print(f'Error: No mask files found in {mask_dir}')
                continue
        except ValueError as e:
            print(f'Error: {e}')
            continue
        print(f'Starting training with dataset type: {args.dataset_type}')
        print(f'Train/Test split: {args.train_split:.1f}/{1 - args.train_split:.1f}')
        try:
            train(args)
            print(f'\nTraining completed for dataset: {args.dataset_type}')
        except Exception as e:
            print(f'Error during training for dataset {args.dataset_type}: {e}')
            continue
    print(f'\n{'=' * 60}')
    print('All specified datasets training completed!')
    print(f'{'=' * 60}')
if __name__ == '__main__':
    main()
