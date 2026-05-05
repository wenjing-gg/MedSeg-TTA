import os
import shutil
import tempfile
import time
import pdb
import torch.nn as nn
import matplotlib.pyplot as plt
import monai
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric, MAEMetric
from monai.networks.nets import SegResNet
from monai.apps.utils import get_logger
from monai.data.meta_tensor import MetaTensor
from monai.inferers.merger import AvgMerger, Merger
from monai.inferers.splitter import Splitter
from monai.inferers.utils import compute_importance_map, sliding_window_inference
from monai.utils import BlendMode, PatchKeys, PytorchPadMode, ensure_tuple, optional_import
from monai.visualize import CAM, GradCAM, GradCAMpp
import torch.nn.functional as F
from monai.transforms import Resize
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from monai.utils import BlendMode, PytorchPadMode, convert_data_type, convert_to_dst_type, ensure_tuple, fall_back_tuple, look_up_option, optional_import
from monai.transforms import Activations, Activationsd, AsDiscrete, AsDiscreted, Compose, Invertd, LoadImaged, MapTransform, NormalizeIntensityd, Orientationd, RandFlipd, RandScaleIntensityd, RandShiftIntensityd, RandSpatialCropd, Spacingd, EnsureTyped, EnsureChannelFirstd
from monai.utils import set_determinism
import torch
print_config()
root_dir = '/home/yz/Documents/Heart/'
print(root_dir)
set_determinism(seed=0)

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
from monai.utils import set_determinism
import torch
import logging
import torch.nn as nn
import numpy as np
import torch._utils
import torch.nn.functional as F
print_config()
root_dir = '/home/yz/Documents/Heart/'
print(root_dir)
set_determinism(seed=0)
from yacs.config import CfgNode as CN
HRNET_18 = CN()
HRNET_18.FINAL_CONV_KERNEL = 1
HRNET_18.STAGE1 = CN()
HRNET_18.STAGE1.NUM_MODULES = 1
HRNET_18.STAGE1.NUM_BRANCHES = 1
HRNET_18.STAGE1.NUM_BLOCKS = [4]
HRNET_18.STAGE1.NUM_CHANNELS = [64]
HRNET_18.STAGE1.BLOCK = 'BOTTLENECK'
HRNET_18.STAGE1.FUSE_METHOD = 'SUM'
HRNET_18.STAGE2 = CN()
HRNET_18.STAGE2.NUM_MODULES = 1
HRNET_18.STAGE2.NUM_BRANCHES = 2
HRNET_18.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_18.STAGE2.NUM_CHANNELS = [18, 36]
HRNET_18.STAGE2.BLOCK = 'BOTTLENECK'
HRNET_18.STAGE2.FUSE_METHOD = 'SUM'
HRNET_18.STAGE3 = CN()
HRNET_18.STAGE3.NUM_MODULES = 4
HRNET_18.STAGE3.NUM_BRANCHES = 3
HRNET_18.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_18.STAGE3.NUM_CHANNELS = [18, 36, 72]
HRNET_18.STAGE3.BLOCK = 'BOTTLENECK'
HRNET_18.STAGE3.FUSE_METHOD = 'SUM'
HRNET_18.STAGE4 = CN()
HRNET_18.STAGE4.NUM_MODULES = 3
HRNET_18.STAGE4.NUM_BRANCHES = 4
HRNET_18.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_18.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
HRNET_18.STAGE4.BLOCK = 'BOTTLENECK'
HRNET_18.STAGE4.FUSE_METHOD = 'SUM'
BN_MOMENTUM = 0.1
ALIGN_CORNERS = None
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    relu_inplace = True

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion, momentum=BN_MOMENTUM)
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
        out = out + residual
        out = self.relu(out)
        return out

class HighResolutionModule(nn.Module):
    relu_inplace = True

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv3d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), nn.BatchNorm3d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv3d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm3d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv3d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm3d(num_outchannels_conv3x3, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    z_output = x[i].shape[-3]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[z_output, height_output, width_output], mode='trilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse
blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

class HighResolutionNet(nn.Module):

    def __init__(self):
        super(HighResolutionNet, self).__init__()
        ALIGN_CORNERS = None
        relu_inplace = True
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.stage1_cfg = HRNET_18.STAGE1
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels
        self.stage2_cfg = HRNET_18.STAGE2
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = HRNET_18.STAGE3
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = HRNET_18.STAGE4
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)
        last_inp_channels = int(np.sum(pre_stage_channels))
        self.last_layer = nn.Sequential(nn.Conv3d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm3d(last_inp_channels), nn.ReLU(inplace=True), nn.Conv3d(in_channels=last_inp_channels, out_channels=1, kernel_size=1, stride=1, padding=0))

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv3d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm3d(num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv3d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm3d(outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return (nn.Sequential(*modules), num_inchannels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        x0_h, x0_w, x0_z = (x[0].size(2), x[0].size(3), x[0].size(4))
        x1 = F.interpolate(x[1], size=(x0_h, x0_w, x0_z), mode='trilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w, x0_z), mode='trilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w, x0_z), mode='trilinear', align_corners=ALIGN_CORNERS)
        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        x = F.interpolate(x, size=(int(x0_h * 4), int(x0_w * 4), int(4 * x0_z)), mode='trilinear', align_corners=ALIGN_CORNERS)
        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info('=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def _get_scan_interval(image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float) -> Tuple[int, ...]:
    if len(image_size) != num_spatial_dims:
        raise ValueError('image coord different from spatial dims.')
    if len(roi_size) != num_spatial_dims:
        raise ValueError('roi coord different from spatial dims.')
    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)

def sliding_window_inference_testfit(inputs: torch.Tensor, roi_size: Union[Sequence[int], int], sw_batch_size: int, predictor: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]], OGNet: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]], optimizer: Any, loss_function: Any, overlap: float=0.25, mode: Union[BlendMode, str]=BlendMode.CONSTANT, sigma_scale: Union[Sequence[float], float]=0.125, padding_mode: Union[PytorchPadMode, str]=PytorchPadMode.CONSTANT, cval: float=0.0, sw_device: Union[torch.device, str, None]=None, device: Union[torch.device, str, None]=None, progress: bool=False, roi_weight_map: Optional[torch.Tensor]=None, process_fn: Optional[Callable]=None, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
    softmax = torch.nn.Softmax(dim=1)
    avgpool3d = torch.nn.AvgPool3d(2, stride=1)
    avgpool3d_pad1 = torch.nn.AvgPool3d(2, stride=1, padding=1)
    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError('overlap must be >= 0 and < 1.')
    batch_size, _, *image_size_ = inputs.shape
    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device
    roi_size = fall_back_tuple(roi_size, image_size_)
    image_size = tuple((max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)
    total_slices = num_win * batch_size
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and roi_weight_map is not None:
        importance_map_ = roi_weight_map
    else:
        try:
            importance_map_ = compute_importance_map(valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device)
        except BaseException as e:
            raise RuntimeError("Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'.") from e
    importance_map_ = convert_data_type(importance_map_, torch.Tensor, device, compute_dtype)[0]
    min_non_zero = max(importance_map_[importance_map_ != 0].min().item(), 0.001)
    importance_map_ = torch.clamp(importance_map_.to(torch.float32), min=min_non_zero).to(compute_dtype)
    dict_key, output_image_list, count_map_list = (None, [], [])
    _initialized_ss = -1
    is_tensor_output = True
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [[slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win]) for idx in slice_range]
        window_data = torch.cat([convert_data_type(inputs[win_slice], torch.Tensor)[0] for win_slice in unravel_slice]).to(sw_device)
        optimizer.zero_grad()
        seg_prob1 = predictor(torch.nn.functional.interpolate(avgpool3d(window_data), size=[window_data.shape[2], window_data.shape[3], window_data.shape[4]], mode='trilinear'), *args, **kwargs)
        with torch.no_grad():
            seg_prob2 = OGNet(window_data, *args, **kwargs)
            seg_prob2 = seg_prob2.detach()
        high = -10000
        low = 10000
        high_alpha = 0
        low_alpha = 0
        for alpha in range(101):
            temp = alpha / 100 * seg_prob1.detach() + (1 - alpha / 100) * seg_prob2
            score = softmax_entropy(temp).mean(0)
            score = torch.mean(score)
            if score >= high:
                high = score
                high_alpha = alpha
            if score <= low:
                low = score
                low_alpha = alpha
        seg_prob_out = low_alpha / 100 * seg_prob1 + (1 - low_alpha / 100) * seg_prob2
        labels = high_alpha / 100 * seg_prob1 + (1 - high_alpha / 100) * seg_prob2
        labels = torch.sigmoid(labels)
        weight1 = labels.clone()
        weight1 = 2 * torch.abs(0.5 - weight1)
        weight1 = weight1.detach()
        weight2 = seg_prob1.clone()
        weight2 = torch.sigmoid(weight2)
        weight2 = 2 * torch.abs(0.5 - weight2)
        weight2 = 1 - weight2
        weight2 = weight2.detach()
        labels[torch.where(labels > 0.95)] = 1.0
        labels[torch.where(labels <= 0.95)] = 0.0
        loss = loss_function(seg_prob1, labels.detach())
        loss = torch.mean(weight1 * weight2 * loss)
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        seg_prob_tuple: Tuple[torch.Tensor, ...]
        if isinstance(seg_prob_out, torch.Tensor):
            seg_prob_tuple = (seg_prob_out,)
        elif isinstance(seg_prob_out, Mapping):
            if dict_key is None:
                dict_key = sorted(seg_prob_out.keys())
            seg_prob_tuple = tuple((seg_prob_out[k] for k in dict_key))
            is_tensor_output = False
        else:
            seg_prob_tuple = ensure_tuple(seg_prob_out)
            is_tensor_output = False
        if process_fn:
            seg_prob_tuple, importance_map = process_fn(seg_prob_tuple, window_data, importance_map_)
        else:
            importance_map = importance_map_
        for ss, seg_prob in enumerate(seg_prob_tuple):
            seg_prob = seg_prob.to(device)
            zoom_scale = []
            for axis, (img_s_i, out_w_i, in_w_i) in enumerate(zip(image_size, seg_prob.shape[2:], window_data.shape[2:])):
                _scale = out_w_i / float(in_w_i)
                if not (img_s_i * _scale).is_integer():
                    warnings.warn(f'For spatial axis: {axis}, output[{ss}] will have non-integer shape. Spatial zoom_scale between output[{ss}] and input is {_scale}. Please pad inputs.')
                zoom_scale.append(_scale)
            if _initialized_ss < ss:
                output_classes = seg_prob.shape[1]
                output_shape = [batch_size, output_classes] + [int(image_size_d * zoom_scale_d) for image_size_d, zoom_scale_d in zip(image_size, zoom_scale)]
                output_image_list.append(torch.zeros(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                _initialized_ss += 1
            resizer = Resize(spatial_size=seg_prob.shape[2:], mode='nearest', anti_aliasing=False)
            for idx, original_idx in zip(slice_range, unravel_slice):
                original_idx_zoom = list(original_idx)
                for axis in range(2, len(original_idx_zoom)):
                    zoomed_start = original_idx[axis].start * zoom_scale[axis - 2]
                    zoomed_end = original_idx[axis].stop * zoom_scale[axis - 2]
                    if not zoomed_start.is_integer() or not zoomed_end.is_integer():
                        warnings.warn(f'For axis-{axis - 2} of output[{ss}], the output roi range is not int. Input roi range is ({original_idx[axis].start}, {original_idx[axis].stop}). Spatial zoom_scale between output[{ss}] and input is {zoom_scale[axis - 2]}. Corresponding output roi range is ({zoomed_start}, {zoomed_end}).\nPlease change overlap ({overlap}) or roi_size ({roi_size[axis - 2]}) for axis-{axis - 2}. Tips: if overlap*roi_size*zoom_scale is an integer, it usually works.')
                    original_idx_zoom[axis] = slice(int(zoomed_start), int(zoomed_end), None)
                importance_map_zoom = resizer(importance_map.unsqueeze(0))[0].to(compute_dtype)
                output_image_list[ss][original_idx_zoom] += importance_map_zoom * seg_prob[idx - slice_g]
                count_map_list[ss][original_idx_zoom] += importance_map_zoom.unsqueeze(0).unsqueeze(0).expand(count_map_list[ss][original_idx_zoom].shape)
    for ss in range(len(output_image_list)):
        output_image_list[ss] = (output_image_list[ss] / count_map_list.pop(0)).to(compute_dtype)
    for ss, output_i in enumerate(output_image_list):
        if torch.isnan(output_i).any() or torch.isinf(output_i).any():
            warnings.warn('Sliding window inference results contain NaN or Inf.')
        zoom_scale = [seg_prob_map_shape_d / roi_size_d for seg_prob_map_shape_d, roi_size_d in zip(output_i.shape[2:], roi_size)]
        final_slicing: List[slice] = []
        for sp in range(num_spatial_dims):
            slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
            slice_dim = slice(int(round(slice_dim.start * zoom_scale[num_spatial_dims - sp - 1])), int(round(slice_dim.stop * zoom_scale[num_spatial_dims - sp - 1])))
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_i.shape):
            final_slicing.insert(0, slice(None))
        output_image_list[ss] = output_i[final_slicing]
    if dict_key is not None:
        final_output = dict(zip(dict_key, output_image_list))
    else:
        final_output = tuple(output_image_list)
    final_output = final_output[0] if is_tensor_output else final_output
    if isinstance(inputs, MetaTensor):
        final_output = convert_to_dst_type(final_output, inputs, device=device)[0]
    return final_output

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 1)
            d[key] = torch.stack(result, axis=0).float()
        return d
train_transform = Compose([LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys='image'), EnsureTyped(keys=['image', 'label']), ConvertToMultiChannelBasedOnBratsClassesd(keys='label'), Orientationd(keys=['image', 'label'], axcodes='RAS'), Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')), RandSpatialCropd(keys=['image', 'label'], roi_size=[224, 224, 144], random_size=False), RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0), RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1), RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2), NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True), RandScaleIntensityd(keys='image', factors=0.1, prob=1.0), RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0)])
val_transform = Compose([LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys='image'), EnsureTyped(keys=['image', 'label']), ConvertToMultiChannelBasedOnBratsClassesd(keys='label'), Orientationd(keys=['image', 'label'], axcodes='RAS'), Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')), NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True)])
train_ds = DecathlonDataset(root_dir=root_dir, task='Task02_Heart', transform=train_transform, section='training', download=True, cache_rate=0.0, num_workers=1)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
val_ds = DecathlonDataset(root_dir=root_dir, task='Task02_Heart', transform=val_transform, section='validation', download=False, cache_rate=0.0, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
max_epochs = 300
val_interval = 1
VAL_AMP = True
device = torch.device('cuda:0')
loss_function = nn.BCEWithLogitsLoss(reduction='none')
dice_metric = DiceMetric(include_background=True, reduction='mean')
dice_metric_batch = DiceMetric(include_background=True, reduction='mean_batch')
asd_metric = HausdorffDistanceMetric(include_background=True, reduction='mean')
asd_metric_batch = SurfaceDistanceMetric(include_background=True, reduction='mean_batch')
hd_metric = HausdorffDistanceMetric(include_background=True, reduction='mean')
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
mse = MAEMetric(reduction='mean')

def inference(input, net, OGNet, optimizer, loss_function):

    def _compute(input, net, OGNet, optimizer, loss_function):
        return sliding_window_inference_testfit(inputs=input, roi_size=(240, 240, 160), sw_batch_size=1, predictor=net, OGNet=OGNet, optimizer=optimizer, loss_function=loss_function, overlap=0.5)
    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input, net, OGNet, optimizer, loss_function)
    else:
        return _compute(input, net, OGNet, optimizer, loss_function)
device = torch.device('cuda:0')

def get_net():
    net = HighResolutionNet()
    return net
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True
val_org_transforms = Compose([LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image']), ConvertToMultiChannelBasedOnBratsClassesd(keys='label'), Orientationd(keys=['image'], axcodes='RAS'), Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'), NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True)])
val_org_ds = DecathlonDataset(root_dir=root_dir, task='Task02_Heart', transform=val_org_transforms, section='validation', download=False, num_workers=1, cache_num=0)
val_org_loader = DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=1)
post_transforms = Compose([Invertd(keys='pred', transform=val_org_transforms, orig_keys='image', meta_keys='pred_meta_dict', orig_meta_keys='image_meta_dict', meta_key_postfix='meta_dict', nearest_interp=False, to_tensor=True, device='cpu'), Activationsd(keys='pred', sigmoid=True), AsDiscreted(keys='pred', threshold=0.5)])
if 1:
    net = get_net().to(device)
    net.load_state_dict(torch.load(os.path.join(root_dir, 'model_hrnet_heart.pth')))
    net.train()
    OGNet = get_net().to(device)
    OGNet.load_state_dict(torch.load(os.path.join(root_dir, 'model_hrnet_heart.pth')))
    OGNet.eval()
    optimizer = torch.optim.SGD(net.parameters(), 1e-05)
if 1:
    for val_data in val_org_loader:
        val_inputs = val_data['image'].to(device)
        val_data['pred'] = inference(val_inputs, net, OGNet, optimizer, loss_function)
        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
        val_outputs, val_labels = from_engine(['pred', 'label'])(val_data)
        dice_metric(y_pred=val_outputs, y=val_labels)
        dice_metric_batch(y_pred=val_outputs, y=val_labels)
        asd_metric(y_pred=val_outputs, y=val_labels)
    metric_org = dice_metric.aggregate().item()
    metric_batch_org = dice_metric_batch.aggregate()
    asd_metric_org = asd_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()
    asd_metric.reset()
metric_tc = metric_batch_org[0].item()
print('Metric on original image spacing: ', metric_org)
