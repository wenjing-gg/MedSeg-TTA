import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

class _ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True, actv=nn.LeakyReLU):
        super().__init__()
        self._conv = nn.Conv3d(in_channels, out_channels, kernel, stride, padding, dilation, groups, bias)
        self._norm = nn.InstanceNorm3d(out_channels, affine=True)
        if actv in (nn.ReLU, nn.LeakyReLU):
            self._actv = actv(inplace=True)
        elif actv is not None:
            self._actv = actv()
        else:
            self._actv = actv

    def forward(self, x):
        output = self._norm(self._conv(x))
        if self._actv is not None:
            output = self._actv(output)
        return output

class _DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers, down_stride=1):
        super().__init__()
        self._conv_layers = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self._conv_layers.add_module(f'_conv_layer_{i}', _ConvLayer(in_channels, out_channels, 3, down_stride, 1))
            else:
                self._conv_layers.add_module(f'_conv_layer_{i}', _ConvLayer(out_channels, out_channels, 3, padding=1))

    def forward(self, x):
        output = self._conv_layers(x)
        return output

class _UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers, up_stride=2):
        super().__init__()
        self._upsample = nn.ConvTranspose3d(in_channels, out_channels, up_stride, up_stride, bias=False)
        self._conv_layers = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self._conv_layers.add_module(f'_conv_layer_{i}', _ConvLayer(2 * out_channels, out_channels, 3, padding=1))
            else:
                self._conv_layers.add_module(f'_conv_layer_{i}', _ConvLayer(out_channels, out_channels, 3, padding=1))

    def forward(self, up_feat, down_feat):
        up_feat = self._upsample(up_feat)
        feat = torch.cat((up_feat, down_feat), dim=1)
        output = self._conv_layers(feat)
        return output

class UNet(nn.Module):

    def __init__(self, in_shape=(64, 96, 96), in_channels=1, num_classes=1, num_layers=2, down_strides64=(1, 2, 2, 2, 2, (1, 2, 2)), up_strides64=((1, 2, 2), 2, 2, 2, 2), down_strides32=(1, 1, 2, 2, 2, (1, 2, 2)), up_strides32=((1, 2, 2), 2, 2, 2, 1), down_strides16=(1, 1, 1, 2, 2, (1, 2, 2)), up_strides16=((1, 2, 2), 2, 2, 1, 1), down_channels=(32, 64, 128, 256, 320, 320), up_channels=(320, 256, 256, 128, 32)):
        super().__init__()
        self._down_path64 = nn.ModuleList()
        for i in range(len(down_channels)):
            self._down_path64.append(_DownBlock(in_channels, down_channels[i], num_layers, down_strides64[i]))
            if i in [1, 2]:
                in_channels = 2 * down_channels[i]
            else:
                in_channels = down_channels[i]
        self._up_path64 = nn.ModuleList()
        for i in range(len(up_channels)):
            self._up_path64.append(_UpBlock(in_channels, up_channels[i], num_layers, up_strides64[i]))
            in_channels = up_channels[i]
        self._down_path32 = nn.ModuleList()
        in_channels = 1
        for i in range(2):
            self._down_path32.append(_DownBlock(in_channels, down_channels[i], num_layers, down_strides32[i]))
            in_channels = down_channels[i]
        self._up_path32 = nn.ModuleList()
        for i in range(-2, -1):
            self._up_path32.append(_UpBlock(up_channels[i], up_channels[i + 1], num_layers, up_strides32[i + 1]))
        self._down_path16 = nn.ModuleList()
        in_channels = 1
        for i in range(3):
            self._down_path16.append(_DownBlock(in_channels, down_channels[i], num_layers, down_strides16[i]))
            in_channels = down_channels[i]
        self._up_path16 = nn.ModuleList()
        for i in range(-3, -1):
            if i == -3:
                self._up_path16.append(_UpBlock(up_channels[i], int(up_channels[i + 1] / 2), num_layers, up_strides16[i + 1]))
            else:
                self._up_path16.append(_UpBlock(int(up_channels[i] / 2), up_channels[i + 1], num_layers, up_strides16[i + 1]))
        self._output_layer = nn.Conv3d(up_channels[-1], num_classes, 1, bias=False)

    def forward(self, x):
        down_feats16 = {'down_0': x[2]}
        for i in range(len(self._down_path16)):
            down_feats16[f'down_{i + 1}'] = self._down_path16[i](down_feats16[f'down_{i}'])
        down_feats32 = {'down_0': x[1]}
        for i in range(len(self._down_path32)):
            down_feats32[f'down_{i + 1}'] = self._down_path32[i](down_feats32[f'down_{i}'])
        down_feats64 = {'down_0': x[0]}
        for i in range(len(self._down_path64)):
            if i == 2:
                down_feats64[f'down_{i}'] = torch.cat((down_feats64[f'down_{i}'], down_feats32[f'down_{i}']), dim=1)
            elif i == 3:
                down_feats64[f'down_{i}'] = torch.cat((down_feats64[f'down_{i}'], down_feats16[f'down_{i}']), dim=1)
            down_feats64[f'down_{i + 1}'] = self._down_path64[i](down_feats64[f'down_{i}'])
        up_feats64 = {'up_0': down_feats64[f'down_{len(self._down_path64)}']}
        for i in range(len(self._up_path64)):
            up_feats64[f'up_{i + 1}'] = self._up_path64[i](up_feats64[f'up_{i}'], down_feats64[f'down_{len(self._down_path64) - i - 1}'])
        output64 = self._output_layer(up_feats64[f'up_{len(self._up_path64)}'])
        up_feats32 = {'up_0': down_feats64[f'down_{len(self._down_path32)}']}
        for i in range(len(self._up_path32)):
            up_feats32[f'up_{i + 1}'] = self._up_path32[i](up_feats32[f'up_{i}'], down_feats32[f'down_{len(self._down_path32) - i - 1}'])
        output32 = self._output_layer(up_feats32[f'up_{len(self._up_path32)}'])
        up_feats16 = {'up_0': down_feats64[f'down_{len(self._down_path16)}']}
        for i in range(len(self._up_path16)):
            up_feats16[f'up_{i + 1}'] = self._up_path16[i](up_feats16[f'up_{i}'], down_feats16[f'down_{len(self._down_path16) - i - 1}'])
        output16 = self._output_layer(up_feats16[f'up_{len(self._up_path16)}'])
        output = [output64, output32, output16]
        return output
