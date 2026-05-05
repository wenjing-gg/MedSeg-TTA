import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

def extract_patches(image_array, patch_size=256, stride=None, min_foreground_ratio=0.01):
    if stride is None:
        stride = patch_size
    h, w = image_array.shape[:2]
    patches = []
    positions = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image_array[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            patches.append(patch)
            positions.append((y, x))
    return (patches, positions)

def convert_multiclass_to_binary(mask_array, background_value=0):
    binary_mask = np.zeros_like(mask_array, dtype=np.uint8)
    binary_mask[mask_array != background_value] = 255
    return binary_mask

def is_valid_patch_pair(image_patch, mask_patch, min_foreground_ratio=0.01):
    img_mean = np.mean(image_patch)
    if img_mean < 10 or img_mean > 245:
        return False
    foreground_pixels = np.sum(mask_patch > 0)
    total_pixels = mask_patch.size
    foreground_ratio = foreground_pixels / total_pixels
    return foreground_ratio >= min_foreground_ratio

def process_dataset(image_dir, mask_dir, output_dir, patch_size=256, stride=None, min_foreground_ratio=0.01, include_background_patches=False):
    if stride is None:
        stride = patch_size // 2
    output_image_dir = os.path.join(output_dir, 'image')
    output_mask_dir = os.path.join(output_dir, 'mask')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    image_files = []
    for fmt in supported_formats:
        image_files.extend(Path(image_dir).glob(f'*{fmt}'))
        image_files.extend(Path(image_dir).glob(f'*{fmt.upper()}'))
    image_files = sorted([f.name for f in image_files])
    print(f'找到 {len(image_files)} 个图像文件')
    print(f'Patch尺寸: {patch_size}×{patch_size}')
    print(f'滑动步长: {stride}')
    print(f'最小前景比例: {min_foreground_ratio}')
    print(f'包含背景patch: {include_background_patches}')
    total_patches = 0
    valid_patches = 0
    skipped_files = []
    for img_file in tqdm(image_files, desc='处理图像'):
        try:
            img_path = os.path.join(image_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            mask_file = None
            possible_mask_names = [f'{base_name}.png', f'{base_name}.jpg', f'{base_name}.jpeg', f'{base_name}.bmp', f'{base_name}.tif', f'{base_name}.tiff', f'{base_name}_mask.png', f'{base_name}_segmentation.png']
            for mask_name in possible_mask_names:
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    mask_file = mask_name
                    break
            if mask_file is None:
                print(f'警告: 找不到 {img_file} 对应的掩码文件')
                skipped_files.append(img_file)
                continue
            image = Image.open(img_path).convert('L')
            mask = Image.open(os.path.join(mask_dir, mask_file)).convert('L')
            image_array = np.array(image)
            mask_array = np.array(mask)
            print(f'\n处理 {img_file}:')
            print(f'  原始尺寸: {image_array.shape}')
            print(f'  掩码唯一值: {np.unique(mask_array)}')
            if image_array.shape != mask_array.shape:
                print(f'  错误: 图像和掩码尺寸不匹配 {image_array.shape} vs {mask_array.shape}')
                skipped_files.append(img_file)
                continue
            binary_mask = convert_multiclass_to_binary(mask_array)
            print(f'  转换后掩码唯一值: {np.unique(binary_mask)}')
            image_patches, positions = extract_patches(image_array, patch_size, stride)
            mask_patches, _ = extract_patches(binary_mask, patch_size, stride)
            print(f'  提取到 {len(image_patches)} 个patch')
            file_valid_patches = 0
            for i, (img_patch, mask_patch, pos) in enumerate(zip(image_patches, mask_patches, positions)):
                total_patches += 1
                if include_background_patches or is_valid_patch_pair(img_patch, mask_patch, min_foreground_ratio):
                    y, x = pos
                    patch_name = f'{base_name}_patch_{y:04d}_{x:04d}'
                    img_patch_pil = Image.fromarray(img_patch, mode='L')
                    img_patch_pil.save(os.path.join(output_image_dir, f'{patch_name}.png'))
                    mask_patch_pil = Image.fromarray(mask_patch, mode='L')
                    mask_patch_pil.save(os.path.join(output_mask_dir, f'{patch_name}.png'))
                    valid_patches += 1
                    file_valid_patches += 1
            print(f'  保存了 {file_valid_patches} 个有效patch')
        except Exception as e:
            print(f'处理 {img_file} 时出错: {e}')
            skipped_files.append(img_file)
            continue
    print(f'\n处理完成!')
    print(f'总patch数: {total_patches}')
    print(f'有效patch数: {valid_patches}')
    print(f'跳过的文件数: {len(skipped_files)}')
    if skipped_files:
        print(f'跳过的文件: {skipped_files}')
    report_path = os.path.join(output_dir, 'processing_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'PATH数据集预处理报告\n')
        f.write(f'================\n')
        f.write(f'输入图像目录: {image_dir}\n')
        f.write(f'输入掩码目录: {mask_dir}\n')
        f.write(f'输出目录: {output_dir}\n')
        f.write(f'Patch尺寸: {patch_size}×{patch_size}\n')
        f.write(f'滑动步长: {stride}\n')
        f.write(f'最小前景比例: {min_foreground_ratio}\n')
        f.write(f'包含背景patch: {include_background_patches}\n')
        f.write(f'总patch数: {total_patches}\n')
        f.write(f'有效patch数: {valid_patches}\n')
        f.write(f'跳过的文件数: {len(skipped_files)}\n')
        if skipped_files:
            f.write(f'跳过的文件:\n')
            for file in skipped_files:
                f.write(f'  - {file}\n')
    print(f'处理报告已保存到: {report_path}')

def main():
    parser = argparse.ArgumentParser(description='预处理PATH数据集')
    parser.add_argument('--image_dir', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-2DPATH/Glas/image', help='输入图像目录')
    parser.add_argument('--mask_dir', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-2DPATH/Glas/mask', help='输入掩码目录')
    parser.add_argument('--output_dir', type=str, default='/home/yuwenjing/data/tta_dataset/TTA-2DPATH/Glas_processed', help='输出目录')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch尺寸')
    parser.add_argument('--stride', type=int, default=None, help='滑动步长 (默认为patch_size//2)')
    parser.add_argument('--min_foreground_ratio', type=float, default=0.01, help='最小前景比例')
    parser.add_argument('--include_background', action='store_true', help='包含纯背景patch')
    args = parser.parse_args()
    if not os.path.exists(args.image_dir):
        print(f'错误: 图像目录不存在: {args.image_dir}')
        return
    if not os.path.exists(args.mask_dir):
        print(f'错误: 掩码目录不存在: {args.mask_dir}')
        return
    process_dataset(image_dir=args.image_dir, mask_dir=args.mask_dir, output_dir=args.output_dir, patch_size=args.patch_size, stride=args.stride, min_foreground_ratio=args.min_foreground_ratio, include_background_patches=args.include_background)
if __name__ == '__main__':
    main()
