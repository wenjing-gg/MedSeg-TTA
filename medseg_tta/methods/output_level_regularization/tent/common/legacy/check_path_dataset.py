import os
import numpy as np
from PIL import Image
from pathlib import Path

def analyze_dataset(image_dir, mask_dir):
    print(f'分析数据集:')
    print(f'图像目录: {image_dir}')
    print(f'掩码目录: {mask_dir}')
    print('-' * 60)
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    image_files = []
    for fmt in supported_formats:
        image_files.extend(Path(image_dir).glob(f'*{fmt}'))
        image_files.extend(Path(image_dir).glob(f'*{fmt.upper()}'))
    image_files = sorted([f.name for f in image_files])
    print(f'找到 {len(image_files)} 个图像文件')
    if len(image_files) == 0:
        print('没有找到图像文件!')
        return
    sample_count = min(5, len(image_files))
    for i, img_file in enumerate(image_files[:sample_count]):
        print(f'\n文件 {i + 1}: {img_file}')
        try:
            img_path = os.path.join(image_dir, img_file)
            image = Image.open(img_path)
            img_array = np.array(image)
            print(f'  图像尺寸: {img_array.shape}')
            print(f'  图像模式: {image.mode}')
            print(f'  像素值范围: {img_array.min()} - {img_array.max()}')
            base_name = os.path.splitext(img_file)[0]
            mask_file = None
            possible_mask_names = [f'{base_name}.png', f'{base_name}.jpg', f'{base_name}.jpeg', f'{base_name}.bmp', f'{base_name}.tif', f'{base_name}.tiff', f'{base_name}_mask.png', f'{base_name}_segmentation.png']
            for mask_name in possible_mask_names:
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    mask_file = mask_name
                    break
            if mask_file:
                print(f'  对应掩码: {mask_file}')
                mask_path = os.path.join(mask_dir, mask_file)
                mask = Image.open(mask_path)
                mask_array = np.array(mask)
                print(f'  掩码尺寸: {mask_array.shape}')
                print(f'  掩码模式: {mask.mode}')
                print(f'  掩码唯一值: {np.unique(mask_array)}')
                if len(np.unique(mask_array)) > 1:
                    foreground_pixels = np.sum(mask_array > 0)
                    total_pixels = mask_array.size
                    foreground_ratio = foreground_pixels / total_pixels
                    print(f'  前景比例: {foreground_ratio:.3f}')
                else:
                    print(f'  前景比例: 0.000 (纯背景)')
                if img_array.shape[:2] != mask_array.shape[:2]:
                    print(f'  ⚠️  警告: 图像和掩码尺寸不匹配!')
            else:
                print(f'  ❌ 找不到对应的掩码文件')
        except Exception as e:
            print(f'  ❌ 处理文件时出错: {e}')

def main():
    image_dir = '/home/yuwenjing/data/tta_dataset/TTA-2DPATH/CRAG_/image'
    mask_dir = '/home/yuwenjing/data/tta_dataset/TTA-2DPATH/CRAG_/mask'
    if not os.path.exists(image_dir):
        print(f'错误: 图像目录不存在: {image_dir}')
        return
    if not os.path.exists(mask_dir):
        print(f'错误: 掩码目录不存在: {mask_dir}')
        return
    analyze_dataset(image_dir, mask_dir)
if __name__ == '__main__':
    main()
