from scipy import ndimage
import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from torch.nn import init
import monai.losses as losses
from torchsummary import summary
import torch.nn.functional as F
import torch

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

class UNet3dCT(nn.Module):

    def __init__(self, in_chns=1, n_classes=2, ft_chns=None, dropout_p=None):
        super(UNet3dCT, self).__init__()
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
