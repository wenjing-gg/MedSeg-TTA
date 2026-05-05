from scipy import ndimage
import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from torch.nn import init
import torch.nn.functional as F
import torch

class SaveFeatures3D:

    def __init__(self, module):
        self.features = None
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

def count_labels(pseudo_lab):
    pseudo_lab = pseudo_lab.view(-1)
    counts = torch.bincount(pseudo_lab, minlength=4)
    label_counts = {i: counts[i].item() for i in range(len(counts))}
    return label_counts

def get_scheduler(optimizer):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def get_largest_component(image):
    dim = len(image.shape)
    if image.sum() == 0:
        return image
    if dim == 2:
        s = ndimage.generate_binary_structure(2, 1)
    elif dim == 3:
        s = ndimage.generate_binary_structure(3, 1)
    else:
        raise ValueError('the dimension number should be 2 or 3')
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return output

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm3d(out_channels), nn.LeakyReLU(), nn.Dropout(dropout_p), nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm3d(out_channels), nn.LeakyReLU())

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d(2), ConvBlock(in_channels, out_channels, dropout_p))

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, trilinear=True):
        super(UpBlock, self).__init__()
        self.trilinear = trilinear
        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.trilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Encoder(nn.Module):

    def __init__(self, in_chns, n_classes, ft_chns, dropout_p):
        super().__init__()
        self.in_chns = in_chns
        self.ft_chns = ft_chns
        self.n_class = n_classes
        self.dropout = dropout_p
        self.down_path = nn.ModuleList()
        self.down_path.append(ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0]))
        self.down_path.append(DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[0]))
        self.down_path.append(DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[0]))
        self.down_path.append(DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[0]))
        if len(self.ft_chns) == 5:
            self.down_path.append(DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[0]))

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
        return (blocks, x)

class Decoder(nn.Module):

    def __init__(self, in_chns, n_classes, ft_chns, dropout_p, trilinear):
        super().__init__()
        self.in_chns = in_chns
        self.ft_chns = ft_chns
        self.n_class = n_classes
        self.dropout = dropout_p
        self.trilinear = trilinear
        self.up_path = nn.ModuleList()
        if len(self.ft_chns) == 5:
            self.up_path.append(UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=self.dropout[1], trilinear=self.trilinear))
        self.up_path.append(UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=self.dropout[0], trilinear=self.trilinear))
        self.up_path.append(UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=self.dropout[0], trilinear=self.trilinear))
        self.up_path.append(UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=self.dropout[0], trilinear=self.trilinear))
        self.last = nn.Conv3d(self.ft_chns[0], self.n_class, kernel_size=1)

    def forward(self, x, blocks):
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x)

class UNet3d(nn.Module):

    def __init__(self):
        super(UNet3d, self).__init__()
        in_chns = 4
        n_classes = 4
        ft_chns = [16, 32, 64, 128, 256]
        dropout_p = [0, 0.5]
        self.enc = Encoder(in_chns, n_classes, ft_chns, dropout_p)
        self.aux_dec1 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec2 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec3 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec4 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)

    def forward(self, x):
        blocks, latent_A = self.enc(x)
        self.aux_seg_1 = self.aux_dec1(latent_A, blocks).softmax(1)
        return self.aux_seg_1

class UNet3d_PLS(nn.Module):

    def __init__(self, patch_size=(32, 128, 128)):
        super(UNet3d_PLS, self).__init__()
        in_chns = 4
        n_classes = 4
        ft_chns = [16, 32, 64, 128, 256]
        dropout_p = [0, 0.5]
        self.enc = Encoder(in_chns, n_classes, ft_chns, dropout_p)
        self.aux_dec1 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec2 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec3 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec4 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        data_prompt = torch.zeros((in_chns, *patch_size))
        self.data_prompt = nn.Parameter(data_prompt)
        self.bn_f = []
        self._register_feature_hooks()
        print(f'🔧 UNet3d-PLS初始化完成:')
        print(f'   - 数据提示形状: {self.data_prompt.shape}')
        print(f'   - 注册特征钩子数量: {len(self.bn_f)}')

    def _register_feature_hooks(self):
        try:
            for i, down_block in enumerate(self.enc.down_path):
                if hasattr(down_block, 'conv_conv'):
                    for layer in down_block.conv_conv:
                        if isinstance(layer, nn.Conv3d):
                            self.bn_f.append(SaveFeatures3D(layer))
                elif hasattr(down_block, 'maxpool_conv'):
                    conv_block = down_block.maxpool_conv[1]
                    for layer in conv_block.conv_conv:
                        if isinstance(layer, nn.Conv3d):
                            self.bn_f.append(SaveFeatures3D(layer))
            for decoder in [self.aux_dec1, self.aux_dec2, self.aux_dec3, self.aux_dec4]:
                for up_block in decoder.up_path:
                    if hasattr(up_block, 'conv'):
                        for layer in up_block.conv.conv_conv:
                            if isinstance(layer, nn.Conv3d):
                                self.bn_f.append(SaveFeatures3D(layer))
                self.bn_f.append(SaveFeatures3D(decoder.last))
            print(f'   - 成功注册 {len(self.bn_f)} 个特征钩子')
        except Exception as e:
            print(f'⚠️ 特征钩子注册失败: {e}')

    def forward(self, x, training=False):
        mixed_x = mix_data_prompt_3d(x, self.data_prompt)
        blocks, latent_A = self.enc(mixed_x)
        aux_seg_1 = self.aux_dec1(latent_A, blocks).softmax(1)
        if training:
            return (aux_seg_1, self.bn_f)
        else:
            return aux_seg_1

    def close(self):
        for sf in self.bn_f:
            sf.remove()

class UNet3d_PLSCT(nn.Module):

    def __init__(self, patch_size=(32, 128, 128)):
        super(UNet3d_PLS, self).__init__()
        in_chns = 1
        n_classes = 2
        ft_chns = [16, 32, 64, 128, 256]
        dropout_p = [0, 0.5]
        self.enc = Encoder(in_chns, n_classes, ft_chns, dropout_p)
        self.aux_dec1 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec2 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec3 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec4 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        data_prompt = torch.zeros((in_chns, *patch_size))
        self.data_prompt = nn.Parameter(data_prompt)
        self.bn_f = []
        self._register_feature_hooks()
        print(f'🔧 UNet3d-PLS初始化完成:')
        print(f'   - 数据提示形状: {self.data_prompt.shape}')
        print(f'   - 注册特征钩子数量: {len(self.bn_f)}')

    def _register_feature_hooks(self):
        try:
            for i, down_block in enumerate(self.enc.down_path):
                if hasattr(down_block, 'conv_conv'):
                    for layer in down_block.conv_conv:
                        if isinstance(layer, nn.Conv3d):
                            self.bn_f.append(SaveFeatures3D(layer))
                elif hasattr(down_block, 'maxpool_conv'):
                    conv_block = down_block.maxpool_conv[1]
                    for layer in conv_block.conv_conv:
                        if isinstance(layer, nn.Conv3d):
                            self.bn_f.append(SaveFeatures3D(layer))
            for decoder in [self.aux_dec1, self.aux_dec2, self.aux_dec3, self.aux_dec4]:
                for up_block in decoder.up_path:
                    if hasattr(up_block, 'conv'):
                        for layer in up_block.conv.conv_conv:
                            if isinstance(layer, nn.Conv3d):
                                self.bn_f.append(SaveFeatures3D(layer))
                self.bn_f.append(SaveFeatures3D(decoder.last))
            print(f'   - 成功注册 {len(self.bn_f)} 个特征钩子')
        except Exception as e:
            print(f'⚠️ 特征钩子注册失败: {e}')

    def forward(self, x, training=False):
        mixed_x = mix_data_prompt_3d(x, self.data_prompt)
        blocks, latent_A = self.enc(mixed_x)
        aux_seg_1 = self.aux_dec1(latent_A, blocks).softmax(1)
        if training:
            return (aux_seg_1, self.bn_f)
        else:
            return aux_seg_1

    def close(self):
        for sf in self.bn_f:
            sf.remove()

class UNet3d_FAS(nn.Module):

    def __init__(self):
        super(UNet3d_FAS, self).__init__()
        in_chns = 4
        n_classes = 4
        ft_chns = [16, 32, 64, 128, 256]
        dropout_p = [0, 0.5]
        self.enc = Encoder(in_chns, n_classes, ft_chns, dropout_p)
        self.aux_dec1 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec2 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec3 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec4 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.last_feature_dim = ft_chns[-1]
        self.global_features = None
        self._register_global_hook()
        print(f'🔧 UNet3d-FAS初始化完成:')
        print(f'   - 全局特征提取器已注册')
        print(f'   - 预期特征维度: {self.last_feature_dim}')

    def _register_global_hook(self):
        try:
            last_down_block = self.enc.down_path[-1]
            if hasattr(last_down_block, 'conv_conv'):
                self.global_features = SaveFeatures3D(last_down_block)
            elif hasattr(last_down_block, 'maxpool_conv'):
                self.global_features = SaveFeatures3D(last_down_block.maxpool_conv[1])
            print(f'   - 全局特征钩子注册成功')
        except Exception as e:
            print(f'⚠️ 全局特征钩子注册失败: {e}')

    def forward(self, x, gfeat=False):
        blocks, latent_A = self.enc(x)
        aux_seg_1 = self.aux_dec1(latent_A, blocks).softmax(1)
        if not gfeat:
            return aux_seg_1
        else:
            global_feat = None
            if self.global_features is not None and hasattr(self.global_features, 'features'):
                try:
                    global_feat = self.global_pool(self.global_features.features)
                    global_feat = global_feat.view(global_feat.size(0), -1)
                except Exception as e:
                    print(f'⚠️ 钩子特征提取失败: {e}')
            if global_feat is None:
                try:
                    global_feat = self.global_pool(latent_A)
                    global_feat = global_feat.view(global_feat.size(0), -1)
                except Exception as e:
                    print(f'⚠️ 编码器特征提取失败: {e}')
            if global_feat is None:
                batch_size = x.size(0)
                global_feat = torch.zeros(batch_size, self.last_feature_dim).to(x.device)
                print(f'⚠️ 使用零特征作为备用方案')
            return (aux_seg_1, global_feat)

    def close(self):
        if self.global_features is not None:
            self.global_features.remove()

class UNet3d_FASCT(nn.Module):

    def __init__(self):
        super(UNet3d_FAS, self).__init__()
        in_chns = 1
        n_classes = 2
        ft_chns = [16, 32, 64, 128, 256]
        dropout_p = [0, 0.5]
        self.enc = Encoder(in_chns, n_classes, ft_chns, dropout_p)
        self.aux_dec1 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec2 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec3 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec4 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.last_feature_dim = ft_chns[-1]
        self.global_features = None
        self._register_global_hook()
        print(f'🔧 UNet3d-FAS初始化完成:')
        print(f'   - 全局特征提取器已注册')
        print(f'   - 预期特征维度: {self.last_feature_dim}')

    def _register_global_hook(self):
        try:
            last_down_block = self.enc.down_path[-1]
            if hasattr(last_down_block, 'conv_conv'):
                self.global_features = SaveFeatures3D(last_down_block)
            elif hasattr(last_down_block, 'maxpool_conv'):
                self.global_features = SaveFeatures3D(last_down_block.maxpool_conv[1])
            print(f'   - 全局特征钩子注册成功')
        except Exception as e:
            print(f'⚠️ 全局特征钩子注册失败: {e}')

    def forward(self, x, gfeat=False):
        blocks, latent_A = self.enc(x)
        aux_seg_1 = self.aux_dec1(latent_A, blocks).softmax(1)
        if not gfeat:
            return aux_seg_1
        else:
            global_feat = None
            if self.global_features is not None and hasattr(self.global_features, 'features'):
                try:
                    global_feat = self.global_pool(self.global_features.features)
                    global_feat = global_feat.view(global_feat.size(0), -1)
                except Exception as e:
                    print(f'⚠️ 钩子特征提取失败: {e}')
            if global_feat is None:
                try:
                    global_feat = self.global_pool(latent_A)
                    global_feat = global_feat.view(global_feat.size(0), -1)
                except Exception as e:
                    print(f'⚠️ 编码器特征提取失败: {e}')
            if global_feat is None:
                batch_size = x.size(0)
                global_feat = torch.zeros(batch_size, self.last_feature_dim).to(x.device)
                print(f'⚠️ 使用零特征作为备用方案')
            return (aux_seg_1, global_feat)

    def close(self):
        if self.global_features is not None:
            self.global_features.remove()

class UNet3d_PLS_FAS(nn.Module):

    def __init__(self, patch_size=(32, 128, 128), pretrained_path=None):
        super(UNet3d_PLS_FAS, self).__init__()
        in_chns = 4
        n_classes = 4
        ft_chns = [16, 32, 64, 128, 256]
        dropout_p = [0, 0.5]
        self.enc = Encoder(in_chns, n_classes, ft_chns, dropout_p)
        self.aux_dec1 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec2 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec3 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec4 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        data_prompt = torch.zeros((in_chns, *patch_size))
        self.data_prompt = nn.Parameter(data_prompt)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.last_feature_dim = ft_chns[-1]
        self.bn_f = []
        self.global_features = None
        self._register_feature_hooks()
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        self._freeze_unet_parameters()
        print(f'🔧 UNet3d-PLS-FAS模型设置完成:')
        print(f'   - 数据提示形状: {self.data_prompt.shape}')
        print(f'   - PLS特征钩子: {len(self.bn_f)} 个')
        print(f'   - FAS特征维度: {self.last_feature_dim}')
        print(f'   - 可训练参数: data_prompt ({self.data_prompt.numel()} 个参数)')

    def _load_pretrained_weights(self, pretrained_path):
        try:
            pretrained_params = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in pretrained_params:
                self.load_state_dict(pretrained_params['model_state_dict'], strict=False)
            else:
                self.load_state_dict(pretrained_params, strict=False)
            print(f'✅ 成功加载预训练权重: {pretrained_path}')
        except Exception as e:
            print(f'⚠️ 预训练权重加载失败: {e}')

    def _freeze_unet_parameters(self):
        for name, param in self.named_parameters():
            if 'data_prompt' not in name:
                param.requires_grad = False

    def _register_feature_hooks(self):
        try:
            for i, down_block in enumerate(self.enc.down_path):
                if hasattr(down_block, 'conv_conv'):
                    for layer in down_block.conv_conv:
                        if isinstance(layer, nn.Conv3d):
                            self.bn_f.append(SaveFeatures3D(layer))
                elif hasattr(down_block, 'maxpool_conv'):
                    conv_block = down_block.maxpool_conv[1]
                    for layer in conv_block.conv_conv:
                        if isinstance(layer, nn.Conv3d):
                            self.bn_f.append(SaveFeatures3D(layer))
            for up_block in self.aux_dec1.up_path:
                if hasattr(up_block, 'conv'):
                    for layer in up_block.conv.conv_conv:
                        if isinstance(layer, nn.Conv3d):
                            self.bn_f.append(SaveFeatures3D(layer))
            self.bn_f.append(SaveFeatures3D(self.aux_dec1.last))
            last_down_block = self.enc.down_path[-1]
            if hasattr(last_down_block, 'conv_conv'):
                self.global_features = SaveFeatures3D(last_down_block)
            elif hasattr(last_down_block, 'maxpool_conv'):
                self.global_features = SaveFeatures3D(last_down_block.maxpool_conv[1])
            print(f'   - PLS钩子: {len(self.bn_f)} 个')
            print(f'   - FAS钩子: 已注册')
        except Exception as e:
            print(f'⚠️ 特征钩子注册失败: {e}')

    def forward(self, x, training=False, gfeat=False):
        mixed_x = mix_data_prompt_3d(x, self.data_prompt)
        blocks, latent_A = self.enc(mixed_x)
        aux_seg_1 = self.aux_dec1(latent_A, blocks).softmax(1)
        if training and gfeat:
            global_feat = self._extract_global_features(latent_A, x.device)
            return (aux_seg_1, self.bn_f, global_feat)
        elif training:
            return (aux_seg_1, self.bn_f)
        elif gfeat:
            global_feat = self._extract_global_features(latent_A, x.device)
            return (aux_seg_1, global_feat)
        else:
            return aux_seg_1

    def _extract_global_features(self, latent_A, device):
        global_feat = None
        if self.global_features is not None and hasattr(self.global_features, 'features'):
            try:
                global_feat = self.global_pool(self.global_features.features)
                global_feat = global_feat.view(global_feat.size(0), -1)
            except Exception as e:
                print(f'⚠️ 钩子特征提取失败: {e}')
        if global_feat is None:
            try:
                global_feat = self.global_pool(latent_A)
                global_feat = global_feat.view(global_feat.size(0), -1)
            except Exception as e:
                print(f'⚠️ 编码器特征提取失败: {e}')
        if global_feat is None:
            batch_size = latent_A.size(0)
            global_feat = torch.zeros(batch_size, self.last_feature_dim).to(device)
        return global_feat

    def close(self):
        for sf in self.bn_f:
            sf.remove()
        if self.global_features is not None:
            self.global_features.remove()

    @property
    def data_prompt_param(self):
        return self.data_prompt

class UNet3d_PLS_FAS_CT(nn.Module):

    def __init__(self, patch_size=(32, 128, 128), pretrained_path=None):
        super(UNet3d_PLS_FAS_CT, self).__init__()
        in_chns = 1
        n_classes = 2
        ft_chns = [16, 32, 64, 128, 256]
        dropout_p = [0, 0.5]
        self.enc = Encoder(in_chns, n_classes, ft_chns, dropout_p)
        self.aux_dec1 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec2 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec3 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        self.aux_dec4 = Decoder(in_chns, n_classes, ft_chns, dropout_p, trilinear=True)
        data_prompt = torch.zeros((in_chns, *patch_size))
        self.data_prompt = nn.Parameter(data_prompt)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.last_feature_dim = ft_chns[-1]
        self.bn_f = []
        self.global_features = None
        self._register_feature_hooks()
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        self._freeze_unet_parameters()
        print(f'🔧 UNet3d-PLS-FAS模型设置完成:')
        print(f'   - 数据提示形状: {self.data_prompt.shape}')
        print(f'   - PLS特征钩子: {len(self.bn_f)} 个')
        print(f'   - FAS特征维度: {self.last_feature_dim}')
        print(f'   - 可训练参数: data_prompt ({self.data_prompt.numel()} 个参数)')

    def _load_pretrained_weights(self, pretrained_path):
        try:
            pretrained_params = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in pretrained_params:
                self.load_state_dict(pretrained_params['model_state_dict'], strict=False)
            else:
                self.load_state_dict(pretrained_params, strict=False)
            print(f'✅ 成功加载预训练权重: {pretrained_path}')
        except Exception as e:
            print(f'⚠️ 预训练权重加载失败: {e}')

    def _freeze_unet_parameters(self):
        for name, param in self.named_parameters():
            if 'data_prompt' not in name:
                param.requires_grad = False

    def _register_feature_hooks(self):
        try:
            for i, down_block in enumerate(self.enc.down_path):
                if hasattr(down_block, 'conv_conv'):
                    for layer in down_block.conv_conv:
                        if isinstance(layer, nn.Conv3d):
                            self.bn_f.append(SaveFeatures3D(layer))
                elif hasattr(down_block, 'maxpool_conv'):
                    conv_block = down_block.maxpool_conv[1]
                    for layer in conv_block.conv_conv:
                        if isinstance(layer, nn.Conv3d):
                            self.bn_f.append(SaveFeatures3D(layer))
            for up_block in self.aux_dec1.up_path:
                if hasattr(up_block, 'conv'):
                    for layer in up_block.conv.conv_conv:
                        if isinstance(layer, nn.Conv3d):
                            self.bn_f.append(SaveFeatures3D(layer))
            self.bn_f.append(SaveFeatures3D(self.aux_dec1.last))
            last_down_block = self.enc.down_path[-1]
            if hasattr(last_down_block, 'conv_conv'):
                self.global_features = SaveFeatures3D(last_down_block)
            elif hasattr(last_down_block, 'maxpool_conv'):
                self.global_features = SaveFeatures3D(last_down_block.maxpool_conv[1])
            print(f'   - PLS钩子: {len(self.bn_f)} 个')
            print(f'   - FAS钩子: 已注册')
        except Exception as e:
            print(f'⚠️ 特征钩子注册失败: {e}')

    def forward(self, x, training=False, gfeat=False):
        mixed_x = mix_data_prompt_3d(x, self.data_prompt)
        blocks, latent_A = self.enc(mixed_x)
        aux_seg_1 = self.aux_dec1(latent_A, blocks).softmax(1)
        if training and gfeat:
            global_feat = self._extract_global_features(latent_A, x.device)
            return (aux_seg_1, self.bn_f, global_feat)
        elif training:
            return (aux_seg_1, self.bn_f)
        elif gfeat:
            global_feat = self._extract_global_features(latent_A, x.device)
            return (aux_seg_1, global_feat)
        else:
            return aux_seg_1

    def _extract_global_features(self, latent_A, device):
        global_feat = None
        if self.global_features is not None and hasattr(self.global_features, 'features'):
            try:
                global_feat = self.global_pool(self.global_features.features)
                global_feat = global_feat.view(global_feat.size(0), -1)
            except Exception as e:
                print(f'⚠️ 钩子特征提取失败: {e}')
        if global_feat is None:
            try:
                global_feat = self.global_pool(latent_A)
                global_feat = global_feat.view(global_feat.size(0), -1)
            except Exception as e:
                print(f'⚠️ 编码器特征提取失败: {e}')
        if global_feat is None:
            batch_size = latent_A.size(0)
            global_feat = torch.zeros(batch_size, self.last_feature_dim).to(device)
        return global_feat

    def close(self):
        for sf in self.bn_f:
            sf.remove()
        if self.global_features is not None:
            self.global_features.remove()

    @property
    def data_prompt_param(self):
        return self.data_prompt
