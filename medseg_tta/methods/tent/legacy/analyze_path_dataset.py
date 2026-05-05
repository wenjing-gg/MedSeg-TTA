import os
import numpy as np
from PIL import Image
from collections import Counter

def analyze_single_image(image_path):
    try:
        img_pil = Image.open(image_path)
        img_array = np.array(img_pil)
        info = {'path': image_path, 'pil_mode': img_pil.mode, 'pil_size': img_pil.size, 'array_shape': img_array.shape, 'array_dtype': str(img_array.dtype), 'pixel_range': (img_array.min(), img_array.max()), 'mean_value': np.mean(img_array), 'file_size_kb': os.path.getsize(image_path) / 1024, 'status': 'OK'}
        if len(img_array.shape) == 3 and img_array.shape[2] > 4:
            info['status'] = 'WARNING: Too many channels'
        elif img_array.size == 0:
            info['status'] = 'ERROR: Empty image'
        return info
    except Exception as e:
        return {'path': image_path, 'status': f'ERROR: {str(e)}', 'error': str(e)}

def analyze_single_mask(mask_path):
    try:
        mask_pil = Image.open(mask_path)
        mask_array = np.array(mask_pil.convert('L'))
        unique_values, counts = np.unique(mask_array, return_counts=True)
        value_stats = dict(zip(unique_values, counts))
        if len(unique_values) >= 2:
            background_ratio = counts[0] / mask_array.size * 100
            foreground_ratio = 100 - background_ratio
        else:
            background_ratio = 100 if unique_values[0] == 0 else 0
            foreground_ratio = 100 - background_ratio
        info = {'path': mask_path, 'pil_mode': mask_pil.mode, 'pil_size': mask_pil.size, 'array_shape': mask_array.shape, 'array_dtype': str(mask_array.dtype), 'unique_values': unique_values.tolist(), 'value_counts': value_stats, 'background_ratio': background_ratio, 'foreground_ratio': foreground_ratio, 'is_binary': len(unique_values) <= 2, 'file_size_kb': os.path.getsize(mask_path) / 1024, 'status': 'OK'}
        if len(unique_values) > 10:
            info['status'] = 'WARNING: Too many unique values for a mask'
        elif mask_array.max() > 255:
            info['status'] = 'WARNING: Pixel values > 255'
        elif len(unique_values) == 1 and unique_values[0] == 0:
            info['status'] = 'WARNING: Pure background mask'
        elif np.any(unique_values > 1) and np.any((unique_values > 1) & (unique_values < 255)):
            info['status'] = 'WARNING: Non-binary mask with intermediate values'
        return info
    except Exception as e:
        return {'path': mask_path, 'status': f'ERROR: {str(e)}', 'error': str(e)}

def find_matching_files(image_dir, mask_dir):
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
    mask_files = []
    for ext in image_extensions:
        mask_files.extend([f for f in os.listdir(mask_dir) if f.lower().endswith(ext)])
    matched_pairs = []
    unmatched_images = []
    unmatched_masks = []
    for img_file in sorted(image_files):
        base_name = os.path.splitext(img_file)[0]
        possible_mask_names = [img_file, f'{base_name}.png', f'{base_name}.jpg', f'{base_name}.jpeg', f'{base_name}.tif', f'{base_name}.tiff', f'{base_name}_mask.png', f'{base_name}_segmentation.png', f'{base_name}-1.png']
        matched = False
        for possible_name in possible_mask_names:
            if possible_name in mask_files:
                matched_pairs.append((img_file, possible_name))
                matched = True
                break
        if not matched:
            unmatched_images.append(img_file)
    matched_mask_files = [pair[1] for pair in matched_pairs]
    unmatched_masks = [f for f in mask_files if f not in matched_mask_files]
    return (matched_pairs, unmatched_images, unmatched_masks)

def analyze_dataset(image_dir, mask_dir, max_samples=20):
    print(f'分析数据集:')
    print(f'  图像目录: {image_dir}')
    print(f'  掩码目录: {mask_dir}')
    print('=' * 80)
    if not os.path.exists(image_dir):
        print(f'❌ 错误: 图像目录不存在: {image_dir}')
        return
    if not os.path.exists(mask_dir):
        print(f'❌ 错误: 掩码目录不存在: {mask_dir}')
        return
    print('🔍 查找匹配的图像-掩码对...')
    matched_pairs, unmatched_images, unmatched_masks = find_matching_files(image_dir, mask_dir)
    print(f'📊 文件统计:')
    print(f'  ✅ 匹配的图像-掩码对: {len(matched_pairs)}')
    print(f'  ❌ 未匹配的图像: {len(unmatched_images)}')
    print(f'  ❌ 未匹配的掩码: {len(unmatched_masks)}')
    print()
    if unmatched_images:
        print(f'⚠️  未匹配的图像文件 (前10个):')
        for img in unmatched_images[:10]:
            print(f'    {img}')
        if len(unmatched_images) > 10:
            print(f'    ... 还有 {len(unmatched_images) - 10} 个')
        print()
    if unmatched_masks:
        print(f'⚠️  未匹配的掩码文件 (前10个):')
        for mask in unmatched_masks[:10]:
            print(f'    {mask}')
        if len(unmatched_masks) > 10:
            print(f'    ... 还有 {len(unmatched_masks) - 10} 个')
        print()
    if not matched_pairs:
        print('❌ 没有找到匹配的图像-掩码对!')
        return
    print(f'🔬 详细分析前 {min(max_samples, len(matched_pairs))} 个匹配对:')
    print('-' * 80)
    image_issues = []
    mask_issues = []
    size_mismatches = []
    pure_background_masks = []
    for i, (img_file, mask_file) in enumerate(matched_pairs[:max_samples]):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        print(f'\n📁 样本 {i + 1}: {img_file} <-> {mask_file}')
        img_info = analyze_single_image(img_path)
        print(f'  🖼️  图像: {img_info.get('pil_size', 'N/A')} {img_info.get('pil_mode', 'N/A')} 范围:[{img_info.get('pixel_range', 'N/A')}] 状态:{img_info.get('status', 'N/A')}')
        if 'ERROR' in img_info.get('status', ''):
            image_issues.append(img_file)
        mask_info = analyze_single_mask(mask_path)
        unique_vals = mask_info.get('unique_values', [])
        bg_ratio = mask_info.get('background_ratio', 0)
        fg_ratio = mask_info.get('foreground_ratio', 0)
        print(f'  🎭 掩码: {mask_info.get('pil_size', 'N/A')} {mask_info.get('pil_mode', 'N/A')} 唯一值:{unique_vals} 背景:{bg_ratio:.1f}% 前景:{fg_ratio:.1f}% 状态:{mask_info.get('status', 'N/A')}')
        if 'ERROR' in mask_info.get('status', ''):
            mask_issues.append(mask_file)
        elif 'Pure background' in mask_info.get('status', ''):
            pure_background_masks.append(mask_file)
        img_size = img_info.get('pil_size')
        mask_size = mask_info.get('pil_size')
        if img_size and mask_size and (img_size != mask_size):
            size_mismatches.append((img_file, mask_file, img_size, mask_size))
            print(f'  ⚠️  尺寸不匹配: 图像{img_size} vs 掩码{mask_size}')
    print(f'\n{'=' * 80}')
    print('📋 分析汇总:')
    if image_issues:
        print(f'❌ 图像文件问题: {len(image_issues)} 个')
        for issue in image_issues[:5]:
            print(f'    {issue}')
        if len(image_issues) > 5:
            print(f'    ... 还有 {len(image_issues) - 5} 个')
    if mask_issues:
        print(f'❌ 掩码文件问题: {len(mask_issues)} 个')
        for issue in mask_issues[:5]:
            print(f'    {issue}')
        if len(mask_issues) > 5:
            print(f'    ... 还有 {len(mask_issues) - 5} 个')
    if size_mismatches:
        print(f'❌ 尺寸不匹配: {len(size_mismatches)} 对')
        for img, mask, img_sz, mask_sz in size_mismatches[:3]:
            print(f'    {img} {img_sz} <-> {mask} {mask_sz}')
        if len(size_mismatches) > 3:
            print(f'    ... 还有 {len(size_mismatches) - 3} 对')
    if pure_background_masks:
        print(f'⚠️  纯背景掩码: {len(pure_background_masks)} 个')
        for bg_mask in pure_background_masks[:5]:
            print(f'    {bg_mask}')
        if len(pure_background_masks) > 5:
            print(f'    ... 还有 {len(pure_background_masks) - 5} 个')
    print(f'\n💡 建议:')
    if len(matched_pairs) == 0:
        print('   - 请检查文件命名规则，确保图像和掩码能够正确匹配')
    elif len(unmatched_images) > 0:
        print(f'   - 有 {len(unmatched_images)} 个图像文件没有对应的掩码')
    elif len(size_mismatches) > 0:
        print('   - 请确保图像和掩码具有相同的尺寸')
    elif len(pure_background_masks) > len(matched_pairs) * 0.5:
        print('   - 超过50%的掩码是纯背景，这可能导致训练指标异常')
    else:
        print('   - 数据集看起来基本正常!')
    print(f'\n✅ 分析完成!')

def main():
    image_dir = '/home/yuwenjing/data/tta_dataset/TTA-2DPATH/CRAG_/image'
    mask_dir = '/home/yuwenjing/data/tta_dataset/TTA-2DPATH/CRAG_/mask'
    analyze_dataset(image_dir, mask_dir, max_samples=30)
if __name__ == '__main__':
    main()
