import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True), nn.Dropout2d(dropout_p), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class DownBlock2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(kernel_size=2), ConvBlock2d(in_channels, out_channels, dropout_p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UpBlock2d(nn.Module):

    def __init__(self, in_channels1: int, in_channels2: int, out_channels: int, dropout_p: float, bilinear: bool=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock2d(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Encoder2d(nn.Module):

    def __init__(self, in_channels: int, ft_chns: list[int], dropout_p: list[float]):
        super().__init__()
        self.down_path = nn.ModuleList([ConvBlock2d(in_channels, ft_chns[0], dropout_p[0]), DownBlock2d(ft_chns[0], ft_chns[1], dropout_p[0]), DownBlock2d(ft_chns[1], ft_chns[2], dropout_p[0]), DownBlock2d(ft_chns[2], ft_chns[3], dropout_p[0]), DownBlock2d(ft_chns[3], ft_chns[4], dropout_p[0])])

    def forward(self, x: torch.Tensor):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
        return (blocks, x)

class Decoder2d(nn.Module):

    def __init__(self, ft_chns: list[int], dropout_p: list[float], n_classes: int=4, bilinear: bool=True):
        super().__init__()
        self.up_path = nn.ModuleList([UpBlock2d(ft_chns[4], ft_chns[3], ft_chns[3], dropout_p[1], bilinear), UpBlock2d(ft_chns[3], ft_chns[2], ft_chns[2], dropout_p[0], bilinear), UpBlock2d(ft_chns[2], ft_chns[1], ft_chns[1], dropout_p[0], bilinear), UpBlock2d(ft_chns[1], ft_chns[0], ft_chns[0], dropout_p[0], bilinear)])
        self.last = nn.Conv2d(ft_chns[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, blocks: list[torch.Tensor]) -> torch.Tensor:
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x)

class UNet2d(nn.Module):

    def __init__(self, in_channels: int=1, n_classes: int=2):
        super().__init__()
        ft_chns = [16, 32, 64, 128, 256]
        dropout_p = [0.0, 0.5]
        self.encoder = Encoder2d(in_channels, ft_chns, dropout_p)
        self.decoder = Decoder2d(ft_chns, dropout_p, n_classes, bilinear=True)
        self.n_classes = n_classes

    def get_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for i, down in enumerate(self.encoder.down_path):
            x = down(x)
            features.append(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks, latent = self.encoder(x)
        logits = self.decoder(latent, blocks)
        return logits
