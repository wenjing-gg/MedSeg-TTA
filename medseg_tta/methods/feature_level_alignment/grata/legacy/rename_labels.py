import os
import shutil

def rename_label_files(directory_path):
    if not os.path.exists(directory_path):
        print(f'❌ 目录不存在: {directory_path}')
        return
    if not os.path.isdir(directory_path):
        print(f'❌ 路径不是目录: {directory_path}')
        return
    print(f'📁 正在处理目录: {directory_path}')
    total_files = 0
    renamed_files = 0
    skipped_files = 0
    for filename in os.listdir(directory_path):
        total_files += 1
        if '_label' in filename:
            new_filename = filename.replace('_label', '')
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)
            if os.path.exists(new_path):
                print(f'⚠️  跳过 {filename} -> {new_filename} (目标文件已存在)')
                skipped_files += 1
                continue
            try:
                os.rename(old_path, new_path)
                print(f'✅ 重命名成功: {filename} -> {new_filename}')
                renamed_files += 1
            except Exception as e:
                print(f'❌ 重命名失败: {filename} -> {new_filename}, 错误: {e}')
                skipped_files += 1
        else:
            print(f"⏭️  跳过 {filename} (无 '_label' 后缀)")
            skipped_files += 1
    print('\n' + '=' * 50)
    print('📊 处理结果统计:')
    print(f'总文件数: {total_files}')
    print(f'成功重命名: {renamed_files}')
    print(f'跳过文件: {skipped_files}')
    print('=' * 50)

def main():
    target_directory = 'E:\\tta_dataset\\TTA-2DPATH\\Cell—seg\\mask'
    print('🔧 文件重命名工具')
    print("功能: 去掉文件名中的 '_label' 后缀")
    print(f'目标目录: {target_directory}')
    response = input('\n是否继续执行重命名操作? (y/n): ').strip().lower()
    if response not in ['y', 'yes', '是', 'Y']:
        print('❌ 操作已取消')
        return
    rename_label_files(target_directory)
    print('\n🎉 重命名操作完成！')
if __name__ == '__main__':
    main()
