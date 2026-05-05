from typing import Union, Type, List, Tuple
import torch
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple

class SaveFeatures3D:

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()

def mix_data_prompt_3d(x, data_prompt):
    b, c, d, h, w = x.shape
    prompt_c, prompt_d, prompt_h, prompt_w = data_prompt.shape
    if (prompt_c, prompt_d, prompt_h, prompt_w) != (c, d, h, w):
        data_prompt_resized = F.interpolate(data_prompt.unsqueeze(0), size=(d, h, w), mode='trilinear', align_corners=False).squeeze(0)
    else:
        data_prompt_resized = data_prompt
    mixed_x = x + data_prompt_resized.expand(b, -1, -1, -1, -1)
    return mixed_x

class PlainConvUNet_PLS(nn.Module):

    def __init__(self, input_channels: int, n_stages: int, features_per_stage: Union[int, List[int], Tuple[int, ...]], conv_op: Type[_ConvNd], kernel_sizes: Union[int, List[int], Tuple[int, ...]], strides: Union[int, List[int], Tuple[int, ...]], n_conv_per_stage: Union[int, List[int], Tuple[int, ...]], num_classes: int, n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], conv_bias: bool=False, norm_op: Union[None, Type[nn.Module]]=None, norm_op_kwargs: dict=None, dropout_op: Union[None, Type[_DropoutNd]]=None, dropout_op_kwargs: dict=None, nonlin: Union[None, Type[torch.nn.Module]]=None, nonlin_kwargs: dict=None, deep_supervision: bool=False, nonlin_first: bool=False, patch_size: Tuple[int, ...]=(32, 128, 128)):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * 2 ** i for i in range(n_stages)]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first)
        data_prompt = torch.zeros((input_channels, *patch_size))
        self.data_prompt = nn.Parameter(data_prompt)
        self.bn_f = []
        self._register_feature_hooks()
        print(f'🔧 nnUNet-PLS初始化完成:')
        print(f'   - 数据提示形状: {self.data_prompt.shape}')
        print(f'   - 注册特征钩子数量: {len(self.bn_f)}')

    def _register_feature_hooks(self):
        try:

            def register_conv_hooks(module, prefix=''):
                for name, child in module.named_children():
                    if isinstance(child, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                        self.bn_f.append(SaveFeatures3D(child))
                        print(f'     - 注册钩子: {prefix}.{name}')
                    else:
                        register_conv_hooks(child, f'{prefix}.{name}' if prefix else name)
            register_conv_hooks(self.encoder, 'encoder')
            register_conv_hooks(self.decoder, 'decoder')
            print(f'   - 成功注册 {len(self.bn_f)} 个特征钩子')
        except Exception as e:
            print(f'⚠️ 特征钩子注册失败: {e}')
            try:
                if hasattr(self.encoder, 'conv_op'):
                    dummy_conv = self.encoder.conv_op(4, 32, 3, padding=1)
                    self.bn_f.append(SaveFeatures3D(dummy_conv))
                    print(f'   - 创建了 {len(self.bn_f)} 个备用钩子')
            except:
                print(f'   - 钩子注册完全失败，将使用空列表')

    def forward(self, x, training=False):
        mixed_x = mix_data_prompt_3d(x, self.data_prompt)
        skips = self.encoder(mixed_x)
        decoder_output = self.decoder(skips)
        if isinstance(decoder_output, (list, tuple)):
            logits = decoder_output[0]
        else:
            logits = decoder_output
        if hasattr(logits, 'as_tensor'):
            logits = logits.as_tensor()
        if training:
            return (logits, self.bn_f)
        else:
            return logits

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), 'just give the image size without color/feature channels or batch channel.'
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    def close(self):
        for sf in self.bn_f:
            sf.remove()

    @staticmethod
    def initialize(module):
        InitWeights_He(0.01)(module)

class nnUNet_PLS(nn.Module):

    def __init__(self, pretrained_path=None, patch_size=(32, 128, 128), input_channels: int=4, n_stages: int=6, features_per_stage: Tuple[int, ...]=(32, 64, 128, 256, 320, 320), conv_op: Type[_ConvNd]=nn.Conv3d, kernel_sizes: Union[int, List[int], Tuple[int, ...]]=3, strides: Tuple[int, ...]=(1, 2, 2, 2, 2, 2), n_conv_per_stage: Tuple[int, ...]=(2, 2, 2, 2, 2, 2), num_classes: int=4, n_conv_per_stage_decoder: Tuple[int, ...]=(2, 2, 2, 2, 2), conv_bias: bool=False, norm_op: Union[None, Type[nn.Module]]=nn.BatchNorm3d, deep_supervision: bool=True):
        super().__init__()
        self.unet = PlainConvUNet_PLS(input_channels=input_channels, n_stages=n_stages, features_per_stage=features_per_stage, conv_op=conv_op, kernel_sizes=kernel_sizes, strides=strides, n_conv_per_stage=n_conv_per_stage, num_classes=num_classes, n_conv_per_stage_decoder=n_conv_per_stage_decoder, conv_bias=conv_bias, norm_op=norm_op, nonlin=nn.ReLU, deep_supervision=deep_supervision, patch_size=patch_size)
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        self._freeze_unet_parameters()
        print(f'🔧 nnUNet-PLS模型设置完成:')
        print(f'   - 可训练参数: data_prompt ({self.unet.data_prompt.numel()} 个参数)')
        print(f'   - 冻结参数: UNet backbone')

    def _load_pretrained_weights(self, pretrained_path):
        try:
            pretrained_params = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in pretrained_params:
                self.unet.load_state_dict(pretrained_params['model_state_dict'], strict=False)
            else:
                self.unet.load_state_dict(pretrained_params, strict=False)
            print(f'✅ 成功加载预训练权重: {pretrained_path}')
        except Exception as e:
            print(f'⚠️ 预训练权重加载失败: {e}')

    def _freeze_unet_parameters(self):
        for name, param in self.unet.named_parameters():
            if 'data_prompt' not in name:
                param.requires_grad = False

    def forward(self, x, training=False):
        return self.unet(x, training=training)

    def close(self):
        self.unet.close()

    @property
    def data_prompt(self):
        return self.unet.data_prompt

class PlainConvUNet_FAS(nn.Module):

    def __init__(self, input_channels: int, n_stages: int, features_per_stage: Union[int, List[int], Tuple[int, ...]], conv_op: Type[_ConvNd], kernel_sizes: Union[int, List[int], Tuple[int, ...]], strides: Union[int, List[int], Tuple[int, ...]], n_conv_per_stage: Union[int, List[int], Tuple[int, ...]], num_classes: int, n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], conv_bias: bool=False, norm_op: Union[None, Type[nn.Module]]=None, norm_op_kwargs: dict=None, dropout_op: Union[None, Type[_DropoutNd]]=None, dropout_op_kwargs: dict=None, nonlin: Union[None, Type[torch.nn.Module]]=None, nonlin_kwargs: dict=None, deep_supervision: bool=False, nonlin_first: bool=False):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * 2 ** i for i in range(n_stages)]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.features_per_stage = features_per_stage
        self.last_feature_dim = features_per_stage[-1] if isinstance(features_per_stage, (list, tuple)) else features_per_stage
        self.global_features = None
        self._register_global_hook()
        print(f'🔧 nnUNet-FAS初始化完成:')
        print(f'   - 全局特征提取器已注册')
        print(f'   - 预期特征维度: {self.last_feature_dim}')

    def _register_global_hook(self):
        try:

            def find_last_conv(module):
                last_conv = None
                for name, child in module.named_modules():
                    if isinstance(child, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                        last_conv = child
                return last_conv
            last_conv = find_last_conv(self.encoder)
            if last_conv is not None:
                self.global_features = SaveFeatures3D(last_conv)
                print(f'   - 全局特征钩子注册成功 (在最后一个卷积层)')
                return
            if hasattr(self.encoder, 'stages') and len(self.encoder.stages) > 0:
                last_stage = self.encoder.stages[-1]
                self.global_features = SaveFeatures3D(last_stage)
                print(f'   - 全局特征钩子注册成功 (在最后一个stage)')
                return
            print(f'⚠️ 未找到合适的层注册全局特征钩子，将使用备用方案')
        except Exception as e:
            print(f'⚠️ 全局特征钩子注册失败: {e}')

    def forward(self, x, gfeat=False):
        skips = self.encoder(x)
        decoder_output = self.decoder(skips)
        if isinstance(decoder_output, (list, tuple)):
            logits = decoder_output[0]
        else:
            logits = decoder_output
        if hasattr(logits, 'as_tensor'):
            logits = logits.as_tensor()
        if not gfeat:
            return logits
        else:
            global_feat = None
            if self.global_features is not None and hasattr(self.global_features, 'features'):
                try:
                    global_feat = self.global_pool(self.global_features.features)
                    global_feat = global_feat.view(global_feat.size(0), -1)
                except Exception as e:
                    print(f'⚠️ 钩子特征提取失败: {e}')
            if global_feat is None and isinstance(skips, (list, tuple)) and (len(skips) > 0):
                try:
                    last_feat = self.global_pool(skips[-1])
                    global_feat = last_feat.view(last_feat.size(0), -1)
                except Exception as e:
                    print(f'⚠️ 跳跃连接特征提取失败: {e}')
            if global_feat is None:
                try:
                    pooled_logits = self.global_pool(logits)
                    global_feat = pooled_logits.view(pooled_logits.size(0), -1)
                except Exception as e:
                    print(f'⚠️ 输出特征提取失败: {e}')
            if global_feat is None:
                batch_size = x.size(0)
                global_feat = torch.zeros(batch_size, self.last_feature_dim).to(x.device)
                print(f'⚠️ 使用零特征作为备用方案')
            return (logits, global_feat)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), 'just give the image size without color/feature channels or batch channel.'
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    def close(self):
        if self.global_features is not None:
            self.global_features.remove()

    @staticmethod
    def initialize(module):
        InitWeights_He(0.01)(module)

class nnUNet_FAS(nn.Module):

    def __init__(self, resnet='nnunet', num_classes=4, pretrained=False, input_channels: int=4, n_stages: int=6, features_per_stage: Tuple[int, ...]=(32, 64, 128, 256, 320, 320), conv_op: Type[_ConvNd]=nn.Conv3d, kernel_sizes: Union[int, List[int], Tuple[int, ...]]=3, strides: Tuple[int, ...]=(1, 2, 2, 2, 2, 2), n_conv_per_stage: Tuple[int, ...]=(2, 2, 2, 2, 2, 2), n_conv_per_stage_decoder: Tuple[int, ...]=(2, 2, 2, 2, 2), conv_bias: bool=False, norm_op: Union[None, Type[nn.Module]]=nn.BatchNorm3d, deep_supervision: bool=True):
        super().__init__()
        self.unet = PlainConvUNet_FAS(input_channels=input_channels, n_stages=n_stages, features_per_stage=features_per_stage, conv_op=conv_op, kernel_sizes=kernel_sizes, strides=strides, n_conv_per_stage=n_conv_per_stage, num_classes=num_classes, n_conv_per_stage_decoder=n_conv_per_stage_decoder, conv_bias=conv_bias, norm_op=norm_op, nonlin=nn.ReLU, deep_supervision=deep_supervision)
        self.num_classes = num_classes
        print(f'🔧 nnUNet-FAS模型设置完成:')
        print(f'   - 输入通道: {input_channels}')
        print(f'   - 输出类别: {num_classes}')
        print(f'   - 支持全局特征提取用于FAS对齐')

    def forward(self, x, gfeat=False):
        return self.unet(x, gfeat=gfeat)

    def close(self):
        self.unet.close()

class PlainConvUNet(nn.Module):

    def __init__(self, input_channels: int, n_stages: int, features_per_stage: Union[int, List[int], Tuple[int, ...]], conv_op: Type[_ConvNd], kernel_sizes: Union[int, List[int], Tuple[int, ...]], strides: Union[int, List[int], Tuple[int, ...]], n_conv_per_stage: Union[int, List[int], Tuple[int, ...]], num_classes: int, n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], conv_bias: bool=False, norm_op: Union[None, Type[nn.Module]]=None, norm_op_kwargs: dict=None, dropout_op: Union[None, Type[_DropoutNd]]=None, dropout_op_kwargs: dict=None, nonlin: Union[None, Type[torch.nn.Module]]=None, nonlin_kwargs: dict=None, deep_supervision: bool=False, nonlin_first: bool=False):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, f'n_conv_per_stage must have as many entries as we have resolution stages. here: {n_stages}. n_conv_per_stage: {n_conv_per_stage}'
        assert len(n_conv_per_stage_decoder) == n_stages - 1, f'n_conv_per_stage_decoder must have one less entries as we have resolution stages. here: {n_stages} stages, so it should have {n_stages - 1} entries. n_conv_per_stage_decoder: {n_conv_per_stage_decoder}'
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first)

    def forward(self, x):
        skips = self.encoder(x)
        decoder_output = self.decoder(skips)
        if isinstance(decoder_output, (list, tuple)):
            logits = decoder_output[0]
        else:
            logits = decoder_output
        if hasattr(logits, 'as_tensor'):
            logits = logits.as_tensor()
        return logits

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), 'just give the image size without color/feature channels or batch channel. Do not give input_size=(b, c, x, y(, z)). Give input_size=(x, y(, z))!'
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(0.01)(module)

class ResidualEncoderUNet(nn.Module):

    def __init__(self, input_channels: int, n_stages: int, features_per_stage: Union[int, List[int], Tuple[int, ...]], conv_op: Type[_ConvNd], kernel_sizes: Union[int, List[int], Tuple[int, ...]], strides: Union[int, List[int], Tuple[int, ...]], n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]], num_classes: int, n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], conv_bias: bool=False, norm_op: Union[None, Type[nn.Module]]=None, norm_op_kwargs: dict=None, dropout_op: Union[None, Type[_DropoutNd]]=None, dropout_op_kwargs: dict=None, nonlin: Union[None, Type[torch.nn.Module]]=None, nonlin_kwargs: dict=None, deep_supervision: bool=False, block: Union[Type[BasicBlockD], Type[BottleneckD]]=BasicBlockD, bottleneck_channels: Union[int, List[int], Tuple[int, ...]]=None, stem_channels: int=None):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, f'n_blocks_per_stage must have as many entries as we have resolution stages. here: {n_stages}. n_blocks_per_stage: {n_blocks_per_stage}'
        assert len(n_conv_per_stage_decoder) == n_stages - 1, f'n_conv_per_stage_decoder must have one less entries as we have resolution stages. here: {n_stages} stages, so it should have {n_stages - 1} entries. n_conv_per_stage_decoder: {n_conv_per_stage_decoder}'
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels, return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), 'just give the image size without color/feature channels or batch channel. Do not give input_size=(b, c, x, y(, z)). Give input_size=(x, y(, z))!'
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(0.01)(module)
        init_last_bn_before_add_to_0(module)

class ResidualUNet(nn.Module):

    def __init__(self, input_channels: int, n_stages: int, features_per_stage: Union[int, List[int], Tuple[int, ...]], conv_op: Type[_ConvNd], kernel_sizes: Union[int, List[int], Tuple[int, ...]], strides: Union[int, List[int], Tuple[int, ...]], n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]], num_classes: int, n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], conv_bias: bool=False, norm_op: Union[None, Type[nn.Module]]=None, norm_op_kwargs: dict=None, dropout_op: Union[None, Type[_DropoutNd]]=None, dropout_op_kwargs: dict=None, nonlin: Union[None, Type[torch.nn.Module]]=None, nonlin_kwargs: dict=None, deep_supervision: bool=False, block: Union[Type[BasicBlockD], Type[BottleneckD]]=BasicBlockD, bottleneck_channels: Union[int, List[int], Tuple[int, ...]]=None, stem_channels: int=None):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, f'n_blocks_per_stage must have as many entries as we have resolution stages. here: {n_stages}. n_blocks_per_stage: {n_blocks_per_stage}'
        assert len(n_conv_per_stage_decoder) == n_stages - 1, f'n_conv_per_stage_decoder must have one less entries as we have resolution stages. here: {n_stages} stages, so it should have {n_stages - 1} entries. n_conv_per_stage_decoder: {n_conv_per_stage_decoder}'
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels, return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), 'just give the image size without color/feature channels or batch channel. Do not give input_size=(b, c, x, y(, z)). Give input_size=(x, y(, z))!'
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(0.01)(module)
        init_last_bn_before_add_to_0(module)
if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))
    model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)
    if False:
        import hiddenlayer as hl
        g = hl.build_graph(model, data, transforms=None)
        g.save('network_architecture.pdf')
        del g
    print(model.compute_conv_feature_map_size(data.shape[2:]))
    data = torch.rand((1, 4, 512, 512))
    model = PlainConvUNet(4, 8, (32, 64, 125, 256, 512, 512, 512, 512), nn.Conv2d, 3, (1, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2, 2, 2), False, nn.BatchNorm2d, None, None, None, nn.ReLU, deep_supervision=True)
    if False:
        import hiddenlayer as hl
        g = hl.build_graph(model, data, transforms=None)
        g.save('network_architecture.pdf')
        del g
    print(model.compute_conv_feature_map_size(data.shape[2:]))
