from torch import nn
import torch
import torch.nn.functional as F

class SaveFeatures3D:

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()

class UnetBlock3D(nn.Module):

    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv3d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose3d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm3d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))

class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet101_3d(pretrained=False, **kwargs):
    model = ResNet3D(Bottleneck3D, [3, 4, 23, 3], **kwargs)
    return model

class UNet3D(nn.Module):

    def __init__(self, resnet='resnet101', num_classes=2, pretrained=False):
        super().__init__()
        cut = 8
        if resnet != 'resnet101':
            raise Exception('This model only supports resnet101')
        base_model = resnet101_3d(pretrained=pretrained)
        base_layers = list(base_model.children())[:cut]
        self.rn = nn.Sequential(*base_layers)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures3D(self.rn[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock3D(2048, 1024, 1024)
        self.up2 = UnetBlock3D(1024, 512, 1024)
        self.up3 = UnetBlock3D(1024, 256, 512)
        self.up4 = UnetBlock3D(512, 64, 256)
        self.up5 = nn.ConvTranspose3d(256, self.num_classes, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        output = self.up5(x)
        return output

    def close(self):
        for sf in self.sfs:
            sf.remove()

def mix_data_prompt_3d(x, data_prompt):
    b, c, d, h, w = x.shape
    prompt_d, prompt_h, prompt_w = data_prompt.shape[1:]
    if (prompt_d, prompt_h, prompt_w) != (d, h, w):
        data_prompt_resized = F.interpolate(data_prompt.unsqueeze(0), size=(d, h, w), mode='trilinear', align_corners=False).squeeze(0)
    else:
        data_prompt_resized = data_prompt
    mixed_x = x + data_prompt_resized.expand(b, -1, -1, -1, -1)
    return mixed_x

class UNet3D_PLS(nn.Module):

    def __init__(self, pretrained_path=None, patch_size=(32, 512, 512), resnet='resnet101', num_classes=2, pretrained=False):
        super().__init__()
        data_prompt = torch.zeros((1, *patch_size))
        self.data_prompt = nn.Parameter(data_prompt)
        self.unet = UNet3D(resnet=resnet, num_classes=num_classes, pretrained=pretrained)
        if pretrained_path:
            try:
                pretrained_params = torch.load(pretrained_path, map_location='cpu')
                if 'model_state_dict' in pretrained_params:
                    self._load_2d_weights_to_3d(pretrained_params['model_state_dict'])
                else:
                    self._load_2d_weights_to_3d(pretrained_params)
            except Exception as e:
                print(f'Warning: Could not load pretrained weights: {e}')
        self.bn_f = []
        try:
            if len(self.unet.rn) > 0:
                self.bn_f.append(SaveFeatures3D(self.unet.rn[0]))
            for layer_idx in range(4, min(8, len(self.unet.rn))):
                layer = self.unet.rn[layer_idx]
                if hasattr(layer, '__len__') and len(layer) > 0:
                    for block_idx in range(len(layer)):
                        block = layer[block_idx]
                        if hasattr(block, 'conv1'):
                            self.bn_f.append(SaveFeatures3D(block.conv1))
                        if hasattr(block, 'conv2'):
                            self.bn_f.append(SaveFeatures3D(block.conv2))
                        if hasattr(block, 'conv3'):
                            self.bn_f.append(SaveFeatures3D(block.conv3))
                        if hasattr(block, 'downsample') and block.downsample is not None:
                            if len(block.downsample) > 0:
                                self.bn_f.append(SaveFeatures3D(block.downsample[0]))
            if hasattr(self.unet, 'up1'):
                self.bn_f.append(SaveFeatures3D(self.unet.up1.tr_conv))
                self.bn_f.append(SaveFeatures3D(self.unet.up1.x_conv))
            if hasattr(self.unet, 'up2'):
                self.bn_f.append(SaveFeatures3D(self.unet.up2.tr_conv))
                self.bn_f.append(SaveFeatures3D(self.unet.up2.x_conv))
            if hasattr(self.unet, 'up3'):
                self.bn_f.append(SaveFeatures3D(self.unet.up3.tr_conv))
                self.bn_f.append(SaveFeatures3D(self.unet.up3.x_conv))
            if hasattr(self.unet, 'up4'):
                self.bn_f.append(SaveFeatures3D(self.unet.up4.tr_conv))
                self.bn_f.append(SaveFeatures3D(self.unet.up4.x_conv))
        except Exception as e:
            print(f'Warning: Could not register all feature hooks: {e}')
        for name, param in self.unet.named_parameters():
            param.requires_grad = False

    def _load_2d_weights_to_3d(self, state_dict_2d):
        state_dict_3d = {}
        for name, param in state_dict_2d.items():
            try:
                if 'conv' in name and param.dim() == 4:
                    if param.shape[2] > 1:
                        param_3d = param.unsqueeze(2).repeat(1, 1, param.shape[2], 1, 1) / param.shape[2]
                    else:
                        param_3d = param.unsqueeze(2)
                    state_dict_3d[name] = param_3d
                else:
                    state_dict_3d[name] = param
            except Exception as e:
                print(f'Warning: Could not convert weight {name}: {e}')
        try:
            self.unet.load_state_dict(state_dict_3d, strict=False)
        except Exception as e:
            print(f'Warning: Could not load state dict: {e}')

    def forward(self, x, training=False):
        mixed_x = mix_data_prompt_3d(x, self.data_prompt)
        output = self.unet(mixed_x)
        if training:
            return (output, self.bn_f)
        else:
            return output
