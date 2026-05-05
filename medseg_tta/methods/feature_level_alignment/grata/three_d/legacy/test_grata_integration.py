import torch
import torch.nn as nn
import numpy as np
from grata_wrapper import create_grata_model, get_default_grata_config
from grata_3d import collect_params_3d, configure_model_3d

class SimpleUNet3D(nn.Module):

    def __init__(self, in_channels=4, out_channels=4):
        super(SimpleUNet3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(32, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.pool(x1)
        x2 = self.relu(self.bn2(self.conv2(x2)))
        x3 = self.upsample(x2)
        x3 = self.relu(self.bn3(self.conv3(x3)))
        out = self.conv4(x3)
        return out

def test_grata_integration():
    print('🧪 开始测试GraTa算法集成...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'📱 使用设备: {device}')
    model = SimpleUNet3D(in_channels=4, out_channels=4).to(device)
    print(f'🏗️  创建简单3D UNet模型')
    params = collect_params_3d(model)
    print(f'📊 收集到BatchNorm3d参数数量: {len(params)}')
    model = configure_model_3d(model)
    print(f'⚙️  模型配置完成')
    config = get_default_grata_config()
    config.lr = 0.0001
    print(f'📋 GraTa配置: lr={config.lr}, aux_loss={config.aux_loss}, pse_loss={config.pse_loss}')
    try:
        grata_model = create_grata_model(model, config, device)
        print(f'✅ GraTa模型创建成功')
    except Exception as e:
        print(f'❌ GraTa模型创建失败: {e}')
        return False
    batch_size = 2
    channels = 4
    depth = 32
    height = 64
    width = 64
    test_imgs = torch.randn(batch_size, channels, depth, height, width).to(device)
    print(f'📦 创建测试数据: {test_imgs.shape}')
    try:
        with torch.no_grad():
            outputs_no_adapt = grata_model.predict_only(test_imgs)
        print(f'✅ 无适应预测成功: {outputs_no_adapt.shape}')
    except Exception as e:
        print(f'❌ 无适应预测失败: {e}')
        return False
    try:
        outputs_with_adapt = grata_model.adapt_and_predict(test_imgs)
        print(f'✅ 适应+预测成功: {outputs_with_adapt.shape}')
    except Exception as e:
        print(f'❌ 适应+预测失败: {e}')
        return False
    expected_shape = (batch_size, 4, depth, height, width)
    if outputs_with_adapt.shape == expected_shape:
        print(f'✅ 输出形状正确: {outputs_with_adapt.shape}')
    else:
        print(f'❌ 输出形状错误: 期望{expected_shape}, 实际{outputs_with_adapt.shape}')
        return False
    print('🔄 测试多次适应...')
    for i in range(3):
        try:
            outputs = grata_model.adapt_and_predict(test_imgs)
            print(f'   第{i + 1}次适应成功')
        except Exception as e:
            print(f'❌ 第{i + 1}次适应失败: {e}')
            return False
    print('🎉 所有测试通过！GraTa算法集成成功！')
    return True

def test_different_configs():
    print('\n🔧 测试不同的GraTa配置...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet3D().to(device)
    test_imgs = torch.randn(1, 4, 16, 32, 32).to(device)
    configs = [{'aux_loss': 'ent', 'pse_loss': 'consis', 'optimizer': 'Adam'}, {'aux_loss': 'consis', 'pse_loss': 'ent', 'optimizer': 'Adam'}, {'aux_loss': 'ent', 'pse_loss': 'consis', 'optimizer': 'SGD'}]
    for i, config_dict in enumerate(configs):
        print(f'\n📋 测试配置 {i + 1}: {config_dict}')
        config = get_default_grata_config()
        for key, value in config_dict.items():
            setattr(config, key, value)
        try:
            grata_model = create_grata_model(model, config, device)
            outputs = grata_model.adapt_and_predict(test_imgs)
            print(f'✅ 配置 {i + 1} 测试成功')
        except Exception as e:
            print(f'❌ 配置 {i + 1} 测试失败: {e}')
            return False
    print('✅ 所有配置测试通过！')
    return True
if __name__ == '__main__':
    print('=' * 60)
    print('GraTa算法集成测试')
    print('=' * 60)
    success1 = test_grata_integration()
    success2 = test_different_configs()
    print('\n' + '=' * 60)
    if success1 and success2:
        print('🎉 所有测试通过！GraTa算法已成功集成到3D医学图像分割任务中！')
        print('📝 可以运行 test_target_tta.py 进行实际的BraTS数据集测试')
    else:
        print('❌ 测试失败，请检查代码')
    print('=' * 60)
