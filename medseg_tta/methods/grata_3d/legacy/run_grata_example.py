import argparse
import os
import sys

def print_banner():
    print('=' * 80)
    print('🧠 GraTa算法 - 3D医学图像分割测试时适应')
    print('   Gradient alignment-based Test-time adaptation for 3D Medical Image Segmentation')
    print('=' * 80)

def print_algorithm_info():
    print('\n📋 算法信息:')
    print('   - 算法名称: GraTa (Gradient alignment-based Test-time adaptation)')
    print('   - 原始论文: https://arxiv.org/pdf/2408.07343')
    print('   - 适配任务: 从2D眼底图像分割 → 3D脑肿瘤分割(BraTS)')
    print('   - 核心特点: 双损失机制 + 梯度对齐 + BatchNorm适应')

def print_usage_examples():
    print('\n🚀 使用示例:')
    print('\n1️⃣ 基础使用 (推荐配置):')
    print('   python test_target_tta.py \\')
    print('       --model_type unet3d \\')
    print('       --target_root /path/to/BraTS/data \\')
    print('       --lr 1e-4 \\')
    print('       --aux_loss ent \\')
    print('       --pse_loss consis \\')
    print('       --optimizer Adam')
    print('\n2️⃣ 使用nnUNet模型:')
    print('   python test_target_tta.py \\')
    print('       --model_type nnunet \\')
    print('       --model_path /path/to/nnunet_best.pth \\')
    print('       --target_root /path/to/BraTS/data \\')
    print('       --lr 5e-5')
    print('\n3️⃣ 不同损失组合:')
    print('   # 一致性 + 熵')
    print('   python test_target_tta.py --aux_loss consis --pse_loss ent')
    print('   # 熵 + 一致性 (推荐)')
    print('   python test_target_tta.py --aux_loss ent --pse_loss consis')
    print('\n4️⃣ 调整学习率:')
    print('   # 较低学习率 (稳定)')
    print('   python test_target_tta.py --lr 1e-5')
    print('   # 较高学习率 (快速适应)')
    print('   python test_target_tta.py --lr 1e-3')

def print_parameter_guide():
    print('\n⚙️ 参数调优指南:')
    print('\n📊 学习率 (--lr):')
    print('   - 1e-5: 保守，适合稳定的适应')
    print('   - 1e-4: 推荐，平衡速度和稳定性')
    print('   - 1e-3: 激进，快速适应但可能不稳定')
    print('\n🎯 损失组合:')
    print('   - aux_loss=ent, pse_loss=consis: 推荐，熵作辅助，一致性作伪标签')
    print('   - aux_loss=consis, pse_loss=ent: 备选，一致性作辅助，熵作伪标签')
    print('\n🔧 优化器:')
    print('   - Adam: 推荐，自适应学习率')
    print('   - SGD: 传统，需要仔细调参')
    print('\n💾 批次大小:')
    print('   - batch_test=1: 内存友好，逐个样本适应')
    print('   - batch_test=2-4: 平衡内存和效率')

def print_expected_results():
    print('\n📈 预期结果:')
    print('   - 输出文件: GraTa_{modality}_{timestamp}.csv')
    print('   - 统计摘要: GraTa_{modality}_{timestamp}_summary.csv')
    print('   - 详细报告: GraTa_{modality}_{timestamp}.txt')
    print('\n📊 性能指标:')
    print('   - Dice系数: 分割重叠度')
    print('   - HD95: 95%豪斯多夫距离')
    print('   - IoU: 交并比')
    print('   - PA: 像素准确率')
    print('   - Sensitivity: 敏感性')
    print('   - PPV: 阳性预测值')

def print_troubleshooting():
    print('\n🛠️ 故障排除:')
    print('\n❌ 内存不足:')
    print('   - 减小 --batch_test 到 1')
    print('   - 使用更小的图像尺寸')
    print('   - 检查GPU内存使用')
    print('\n❌ 收敛不稳定:')
    print('   - 降低学习率 --lr 1e-5')
    print('   - 尝试不同损失组合')
    print('   - 检查数据预处理')
    print('\n❌ 性能下降:')
    print('   - 验证模型权重路径')
    print('   - 检查数据格式匹配')
    print('   - 监控适应过程')

def run_example_command():
    print('\n🎯 运行示例命令:')
    if not os.path.exists('test_target_tta.py'):
        print('❌ 未找到 test_target_tta.py，请确保在正确目录')
        return False
    example_cmd = ['python', 'test_target_tta.py', '--model_type', 'unet3d', '--target_root', '/path/to/your/BraTS/data', '--lr', '1e-4', '--aux_loss', 'ent', '--pse_loss', 'consis', '--optimizer', 'Adam', '--batch_test', '2', '--gpu', '0']
    print('📝 示例命令 (请修改数据路径):')
    print(' '.join(example_cmd))
    print('\n⚠️ 注意:')
    print("   1. 请将 '/path/to/your/BraTS/data' 替换为实际的数据路径")
    print('   2. 确保已安装必要依赖: torch, numpy, pandas, tqdm')
    print('   3. 确保有可用的GPU (或设置 --gpu -1 使用CPU)')
    return True

def main():
    parser = argparse.ArgumentParser(description='GraTa算法使用指南')
    parser.add_argument('--run', action='store_true', help='显示运行示例')
    parser.add_argument('--guide', action='store_true', help='显示完整指南')
    args = parser.parse_args()
    print_banner()
    print_algorithm_info()
    if args.guide or len(sys.argv) == 1:
        print_usage_examples()
        print_parameter_guide()
        print_expected_results()
        print_troubleshooting()
    if args.run:
        run_example_command()
    print('\n' + '=' * 80)
    print('📚 更多信息请查看: README_GraTa_Integration.md')
    print('🧪 功能测试请运行: python test_grata_integration.py')
    print('🚀 实际测试请运行: python test_target_tta.py')
    print('=' * 80)
if __name__ == '__main__':
    main()
