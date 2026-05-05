import os
import numpy as np
from PIL import Image

def check_specific_mask(mask_path):
    print(f'检查掩码文件: {mask_path}')
    print('=' * 60)
    if not os.path.exists(mask_path):
        print(f'❌ 文件不存在: {mask_path}')
        return
    try:
        mask_pil = Image.open(mask_path)
        print(f'原始模式: {mask_pil.mode}')
        print(f'原始大小: {mask_pil.size}')
        mask_gray = mask_pil.convert('L')
        mask_array = np.array(mask_gray)
        print(f'数组形状: {mask_array.shape}')
        print(f'数据类型: {mask_array.dtype}')
        print(f'像素值范围: {mask_array.min()} - {mask_array.max()}')
        unique_values, counts = np.unique(mask_array, return_counts=True)
        print(f'唯一值数量: {len(unique_values)}')
        print(f'唯一值: {unique_values}')
        print(f'\n像素值分布:')
        total_pixels = mask_array.size
        for val, count in zip(unique_values, counts):
            percentage = count / total_pixels * 100
            print(f'  值 {val:3d}: {count:8d} 像素 ({percentage:6.2f}%)')
        print(f'\n掩码分析:')
        if len(unique_values) == 1:
            if unique_values[0] == 0:
                print('  🔍 类型: 纯背景掩码 (所有像素值都是0)')
                print('  ⚠️  警告: 这会导致训练时Dice等指标为1.0')
            else:
                print(f'  🔍 类型: 纯前景掩码 (所有像素值都是{unique_values[0]})')
        elif len(unique_values) == 2:
            print('  🔍 类型: 二值掩码')
            if 0 in unique_values:
                fg_val = unique_values[1] if unique_values[0] == 0 else unique_values[0]
                print(f'  📊 背景值: 0, 前景值: {fg_val}')
            else:
                print(f'  ⚠️  警告: 二值掩码但不包含0值')
        else:
            print(f'  🔍 类型: 多类掩码 ({len(unique_values)}个类别)')
            print('  ⚠️  警告: 对于二分类分割，这可能不正确')
            if np.all(unique_values >= 0) and len(unique_values) > 10:
                print('  💡 可能是实例分割掩码(每个实例有不同的ID)')
    except Exception as e:
        print(f'❌ 读取文件失败: {e}')

def main():
    mask_path = '/home/yuwenjing/data/tta_dataset/TTA-2DPATH/CRAG_/mask/train_1.png'
    check_specific_mask(mask_path)
    print(f'\n{'=' * 60}')
    print('💡 问题分析:')
    print('根据上面的分析结果，PATH数据集的掩码有以下问题:')
    print('1. 掩码不是标准的二值掩码(0和1)，而是多类/实例分割掩码')
    print('2. 掩码包含多个唯一值(如1,2,3,4,5等)，每个值可能代表一个不同的实例')
    print('3. 当前的训练代码假设输入是二值掩码，但实际是多类掩码')
    print('4. 这导致指标计算异常，因为算法将多个实例都当作前景处理')
    print('\n解决方案:')
    print('1. 将多类掩码转换为二值掩码: 将所有非0值设为1')
    print('2. 或者调整模型为多类分割')
    print('3. 或者将多类掩码处理为实例分割任务')
if __name__ == '__main__':
    main()
