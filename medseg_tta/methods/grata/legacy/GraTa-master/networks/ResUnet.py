from torch import nn
import torch
from networks.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F

class UpsampleBlock(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(c_in, c_out, 1)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        out = self.up(x)
        out = self.bn(F.relu(self.conv(out)))
        return out

class UnetBlock(nn.Module):

    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))

class ResUnet(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
            feature_channels = [64, 256, 512, 1024, 2048]
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,resnet101 and resnet152')
        self.res = base_model(pretrained=pretrained)
        self.num_classes = num_classes
        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)
        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)
        self.seg_head = nn.Conv2d(32, self.num_classes, 1)
        aux_branch_base_feature = 256
        self.recon1 = UpsampleBlock(feature_channels[4], aux_branch_base_feature)
        self.recon2 = UpsampleBlock(aux_branch_base_feature, aux_branch_base_feature // 2)
        self.recon3 = UpsampleBlock(aux_branch_base_feature // 2, aux_branch_base_feature // 4)
        self.recon4 = UpsampleBlock(aux_branch_base_feature // 4, aux_branch_base_feature // 8)
        self.recon5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.recon_head = nn.Conv2d(aux_branch_base_feature // 8, 3, 1)
        self.denoise1 = UpsampleBlock(feature_channels[4], aux_branch_base_feature)
        self.denoise2 = UpsampleBlock(aux_branch_base_feature, aux_branch_base_feature // 2)
        self.denoise3 = UpsampleBlock(aux_branch_base_feature // 2, aux_branch_base_feature // 4)
        self.denoise4 = UpsampleBlock(aux_branch_base_feature // 4, aux_branch_base_feature // 8)
        self.denoise5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.denoise_head = nn.Conv2d(aux_branch_base_feature // 8, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rotate_head = nn.Sequential(nn.Linear(feature_channels[4], aux_branch_base_feature), nn.LeakyReLU(0.2), nn.Linear(aux_branch_base_feature, aux_branch_base_feature // 4), nn.LeakyReLU(0.2), nn.Linear(aux_branch_base_feature // 4, 6))
        self.supres1 = UpsampleBlock(feature_channels[4], aux_branch_base_feature)
        self.supres2 = UpsampleBlock(aux_branch_base_feature, aux_branch_base_feature // 2)
        self.supres3 = UpsampleBlock(aux_branch_base_feature // 2, aux_branch_base_feature // 4)
        self.supres4 = UpsampleBlock(aux_branch_base_feature // 4, aux_branch_base_feature // 8)
        self.supres5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.supres_head = nn.Sequential(nn.Conv2d(aux_branch_base_feature // 8, aux_branch_base_feature // 8, 5, 1, 2), nn.Tanh(), nn.Conv2d(aux_branch_base_feature // 8, aux_branch_base_feature // 8, 3, 1, 1), nn.Tanh(), nn.Conv2d(aux_branch_base_feature // 8, 3 * 2 ** 2, 3, 1, 1), nn.PixelShuffle(2))

    def forward(self, input_x, rec=False, rot=False, den=False, sup=False):
        outputs = []
        x, sfs = self.res(input_x)
        x = F.relu(x)
        if rec:
            recon_x = self.recon1(x)
            recon_x = self.recon2(recon_x)
            recon_x = self.recon3(recon_x)
            recon_x = self.recon4(recon_x)
            recon_x = self.recon_head(self.recon5(recon_x))
            outputs.append(recon_x)
        if rot:
            rot_output = self.rotate_head(self.pool(x).view(x.shape[0], -1))
            outputs.append(rot_output)
        if den:
            denoise_x = self.denoise1(x)
            denoise_x = self.denoise2(denoise_x)
            denoise_x = self.denoise3(denoise_x)
            denoise_x = self.denoise4(denoise_x)
            denoise_x = self.denoise_head(self.denoise5(denoise_x))
            outputs.append(denoise_x)
        if sup:
            supres_x = self.supres1(x)
            supres_x = self.supres2(supres_x)
            supres_x = self.supres3(supres_x)
            supres_x = self.supres4(supres_x)
            supres_x = self.supres_head(self.supres5(supres_x))
            outputs.append(supres_x)
        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)
        head_input = F.relu(self.bnout(x))
        seg_output = self.seg_head(head_input)
        outputs.append(seg_output)
        outputs.append(sfs[-1])
        return outputs

    def close(self):
        for sf in self.sfs:
            sf.remove()
if __name__ == '__main__':
    model = ResUnet(resnet='resnet34', num_classes=2, pretrained=False)
    print(model.res)
    model.cuda().eval()
    input = torch.rand(2, 3, 512, 512).cuda()
    seg_output, x_iw_list, iw_loss = model(input)
    print(seg_output.size())
