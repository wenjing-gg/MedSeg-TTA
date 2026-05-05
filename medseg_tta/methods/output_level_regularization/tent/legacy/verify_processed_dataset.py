import os
import numpy as np
from PIL import Image
import random

def verify_processed_dataset(image_dir, mask_dir, sample_count=10):
    print(f'验证处理后的数据集:')
    print(f'图像目录: {image_dir}')
    print(f'掩码目录: {mask_dir}')
    print('-' * 60)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    print(f'图像文件数: {len(image_files)}')
    print(f'掩码文件数: {len(mask_files)}')
    if len(image_files) != len(mask_files):
        print('⚠️ 警告: 图像和掩码文件数量不匹配!')
        return
    sample_files = random.sample(image_files, min(sample_count, len(image_files)))
    foreground_ratios = []
    size_consistency = True
    value_consistency = True
    for i, img_file in enumerate(sample_files):
        print(f'\n样本 {i + 1}: {img_file}')
        try:
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)
            if not os.path.exists(mask_path):
                print(f'  ❌ 找不到对应的掩码文件')
                continue
            image = Image.open(img_path)
            mask = Image.open(mask_path)
            img_array = np.array(image)
            mask_array = np.array(mask)
            print(f'  图像尺寸: {img_array.shape}')
            print(f'  掩码尺寸: {mask_array.shape}')
            if img_array.shape[:2] != mask_array.shape[:2]:
                print(f'  ⚠️ 图像和掩码尺寸不匹配')
                size_consistency = False
            if img_array.shape[:2] != (256, 256):
                print(f'  ⚠️ patch尺寸不是256×256')
                size_consistency = False
            unique_mask_values = np.unique(mask_array)
            print(f'  掩码唯一值: {unique_mask_values}')
            if not np.array_equal(unique_mask_values, [0]) and (not np.array_equal(unique_mask_values, [0, 255])):
                if len(unique_mask_values) > 2 or (len(unique_mask_values) == 2 and (not (0 in unique_mask_values and 255 in unique_mask_values))):
                    print(f'  ⚠️ 掩码值不是预期的二值 (0, 255)')
                    value_consistency = False
            if len(unique_mask_values) > 1:
                foreground_pixels = np.sum(mask_array > 0)
                total_pixels = mask_array.size
                foreground_ratio = foreground_pixels / total_pixels
                foreground_ratios.append(foreground_ratio)
                print(f'  前景比例: {foreground_ratio:.3f}')
            else:
                print(f'  前景比例: 0.000 (纯背景)')
                foreground_ratios.append(0.0)
            img_mean = np.mean(img_array)
            img_std = np.std(img_array)
            print(f'  图像均值: {img_mean:.1f}, 标准差: {img_std:.1f}')
        except Exception as e:
            print(f'  ❌ 处理文件时出错: {e}')
    print(f'\n{'=' * 60}')
    print('验证总结:')
    print(f'尺寸一致性: {('✅ 通过' if size_consistency else '❌ 失败')}')
    print(f'掩码值一致性: {('✅ 通过' if value_consistency else '❌ 失败')}')
    if foreground_ratios:
        print(f'前景比例统计:')
        print(f'  平均: {np.mean(foreground_ratios):.3f}')
        print(f'  最小: {np.min(foreground_ratios):.3f}')
        print(f'  最大: {np.max(foreground_ratios):.3f}')
        print(f'  标准差: {np.std(foreground_ratios):.3f}')
    avg_foreground = np.mean(foreground_ratios) if foreground_ratios else 0
    if avg_foreground < 0.01:
        print('\n💡 建议: 前景比例较低，考虑调整min_foreground_ratio参数')
    elif avg_foreground > 0.5:
        print('\n💡 建议: 前景比例较高，数据质量良好')
    else:
        print('\n💡 建议: 前景比例适中，适合训练')

def main():
    image_dir = '/home/yuwenjing/data/tta_dataset/TTA-2DPATH/CRAG_processed/image'
    mask_dir = '/home/yuwenjing/data/tta_dataset/TTA-2DPATH/CRAG_processed/mask'
    if not os.path.exists(image_dir):
        print(f'错误: 图像目录不存在: {image_dir}')
        return
    if not os.path.exists(mask_dir):
        print(f'错误: 掩码目录不存在: {mask_dir}')
        return
    verify_processed_dataset(image_dir, mask_dir, sample_count=20)
if __name__ == '__main__':
    main()
