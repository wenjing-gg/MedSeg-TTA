from __future__ import annotations
from typing import List, Tuple, Type, Union
import torch
from torch import nn
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.conv import _ConvNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0
Conv2d = nn.Conv2d
BatchNorm2d: Type[nn.Module] = nn.BatchNorm2d
Dropout2d: Type[_DropoutNd] = nn.Dropout2d

class PlainConvUNet2D(nn.Module):

    def __init__(self, input_channels: int, n_stages: int, features_per_stage: Union[int, List[int], Tuple[int, ...]], kernel_sizes: Union[int, List[int], Tuple[int, ...]], strides: Union[int, List[int], Tuple[int, ...]], n_conv_per_stage: Union[int, List[int], Tuple[int, ...]], num_classes: int, n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], conv_bias: bool=False, norm_op: Type[nn.Module] | None=BatchNorm2d, norm_op_kwargs: dict | None=None, dropout_op: Type[_DropoutNd] | None=Dropout2d, dropout_op_kwargs: dict | None=None, nonlin: Type[nn.Module] | None=nn.ReLU, nonlin_kwargs: dict | None=None, deep_supervision: bool=False, nonlin_first: bool=False) -> None:
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, Conv2d, kernel_sizes, strides, n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        decoder_output = self.decoder(skips)
        logits = decoder_output[0] if isinstance(decoder_output, (list, tuple)) else decoder_output
        return logits.as_tensor() if hasattr(logits, 'as_tensor') else logits

    def compute_conv_feature_map_size(self, input_size: Tuple[int, int]) -> int:
        assert len(input_size) == 2, 'Input size must be (H, W) for 2‑D networks. Got {}'.format(input_size)
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module: nn.Module) -> None:
        InitWeights_He(0.01)(module)

class ResidualEncoderUNet2D(nn.Module):

    def __init__(self, input_channels: int, n_stages: int, features_per_stage: Union[int, List[int], Tuple[int, ...]], kernel_sizes: Union[int, List[int], Tuple[int, ...]], strides: Union[int, List[int], Tuple[int, ...]], n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]], num_classes: int, n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], conv_bias: bool=False, norm_op: Type[nn.Module] | None=BatchNorm2d, norm_op_kwargs: dict | None=None, dropout_op: Type[_DropoutNd] | None=Dropout2d, dropout_op_kwargs: dict | None=None, nonlin: Type[nn.Module] | None=nn.ReLU, nonlin_kwargs: dict | None=None, deep_supervision: bool=False, block: Type[BasicBlockD] | Type[BottleneckD]=BasicBlockD, bottleneck_channels: Union[int, List[int], Tuple[int, ...]] | None=None, stem_channels: int | None=None) -> None:
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, Conv2d, kernel_sizes, strides, n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels, return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size: Tuple[int, int]) -> int:
        assert len(input_size) == 2, 'Input size must be (H, W) for 2‑D networks.'
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module: nn.Module) -> None:
        InitWeights_He(0.01)(module)
        init_last_bn_before_add_to_0(module)

class ResidualUNet2D(nn.Module):

    def __init__(self, input_channels: int, n_stages: int, features_per_stage: Union[int, List[int], Tuple[int, ...]], kernel_sizes: Union[int, List[int], Tuple[int, ...]], strides: Union[int, List[int], Tuple[int, ...]], n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]], num_classes: int, n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], conv_bias: bool=False, norm_op: Type[nn.Module] | None=BatchNorm2d, norm_op_kwargs: dict | None=None, dropout_op: Type[_DropoutNd] | None=Dropout2d, dropout_op_kwargs: dict | None=None, nonlin: Type[nn.Module] | None=nn.ReLU, nonlin_kwargs: dict | None=None, deep_supervision: bool=False, block: Type[BasicBlockD] | Type[BottleneckD]=BasicBlockD, bottleneck_channels: Union[int, List[int], Tuple[int, ...]] | None=None, stem_channels: int | None=None) -> None:
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, Conv2d, kernel_sizes, strides, n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels, return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size: Tuple[int, int]) -> int:
        assert len(input_size) == 2, 'Input size must be (H, W) for 2‑D networks.'
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module: nn.Module) -> None:
        InitWeights_He(0.01)(module)
        init_last_bn_before_add_to_0(module)
if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model_plain = PlainConvUNet2D(input_channels=3, n_stages=5, features_per_stage=(32, 64, 128, 256, 512), kernel_sizes=3, strides=(1, 2, 2, 2, 2), n_conv_per_stage=2, num_classes=4, n_conv_per_stage_decoder=2, deep_supervision=True)
    print('Plain U‑Net out:', model_plain(x).shape)
    model_re = ResidualEncoderUNet2D(input_channels=3, n_stages=5, features_per_stage=(32, 64, 128, 256, 512), kernel_sizes=3, strides=(1, 2, 2, 2, 2), n_blocks_per_stage=2, num_classes=4, n_conv_per_stage_decoder=2)
    print('Residual‑Encoder U‑Net out:', model_re(x).shape)
    model_res = ResidualUNet2D(input_channels=3, n_stages=5, features_per_stage=(32, 64, 128, 256, 512), kernel_sizes=3, strides=(1, 2, 2, 2, 2), n_blocks_per_stage=2, num_classes=4, n_conv_per_stage_decoder=2)
    print('Residual U‑Net out:', model_res(x).shape)
