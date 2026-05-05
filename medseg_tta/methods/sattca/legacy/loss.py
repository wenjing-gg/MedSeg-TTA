import torch
import torch.nn as nn
import torch.nn.functional as F

def _binary_dice_loss(output, target):
    smooth = 1e-08
    intersection = (output * target).sum()
    dice_loss = 1 - 2 * intersection / (output.sum() + target.sum() + smooth)
    return dice_loss

class BinaryDiceLoss(nn.Module):

    def forward(self, output, target):
        output = output.sigmoid()
        dice_loss = _binary_dice_loss(output, target)
        return dice_loss

class BinaryDiceTestLoss(nn.Module):

    def forward(self, output, target):
        center_value = output[target == 1]
        if target.sum() == 0:
            return (output.new_tensor(0.0), 0.0)
        output = output.sigmoid()
        dice_loss = _binary_dice_loss(output, target)
        cv = center_value.mean().item() if center_value.numel() > 0 else 0.0
        return (dice_loss, cv)

class SegLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.seg_losses = nn.ModuleDict()
        self.seg_losses.update([('dice_loss', BinaryDiceLoss())])
        self.seg_losses.update([('bce_loss', nn.BCEWithLogitsLoss())])
        self.weight = [1, 0.5]

    def forward(self, outputs, targets):
        dice_loss = self.seg_losses['dice_loss'](outputs, targets) * self.weight[0]
        bce_loss = self.seg_losses['bce_loss'](outputs, targets) * self.weight[1]
        loss_dict = {'dice_loss': dice_loss, 'bce_loss': bce_loss}
        total_loss = sum([loss_dict[k] for k in loss_dict.keys()])
        loss_dict['total_loss'] = total_loss
        return loss_dict

class TestLoss(nn.Module):

    def __init__(self, bce_bg_weight: float=0.2):
        super().__init__()
        self.dice_module = BinaryDiceTestLoss()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.weight = [1.0, 0.5]
        self.bg_weight = bce_bg_weight

    def forward(self, outputs, targets):
        if outputs.shape[1] == 2:
            out_fg = outputs[:, 1:2]
            tgt_fg = targets[:, 1:2]
        else:
            out_fg = outputs
            tgt_fg = targets
        dice_loss, center_value = self.dice_module(out_fg, tgt_fg)
        dice_loss = dice_loss * self.weight[0]
        weights = self.bg_weight + (1 - self.bg_weight) * tgt_fg
        bce_map = self.bce(out_fg, tgt_fg)
        bce_loss = (bce_map * weights).mean() * self.weight[1]
        loss_dict = {'dice_loss': dice_loss, 'bce_loss': bce_loss}
        total_loss = dice_loss + bce_loss
        loss_dict['total_loss'] = total_loss
        return (loss_dict, center_value)

class AdaptiveTestLoss(TestLoss):

    def __init__(self, bce_bg_weight: float=0.2):
        super().__init__(bce_bg_weight=bce_bg_weight)

    def forward(self, outputs, targets, init_outputs=None, bn_param_pairs=None, consistency_weight: float=0.0, reg_weight: float=0.0, scale_weight: float=1.0, boundary_mask: torch.Tensor=None, boundary_factor: float=1.0, diff_mask_weight: float=0.0, use_diff_mask_loss: bool=False, diff_mask_boundary_only: bool=False):
        loss_dict, center_value = super().forward(outputs, targets)
        if boundary_mask is not None and boundary_factor != 1.0:
            if outputs.shape[1] == 2:
                fg_logits = outputs[:, 1:2]
                fg_targets = targets[:, 1:2]
            else:
                fg_logits = outputs
                fg_targets = targets
            with torch.no_grad():
                bmask = boundary_mask
                if bmask.shape[0] != outputs.shape[0]:
                    bmask = bmask[:outputs.shape[0]]
            fg_prob = torch.sigmoid(fg_logits)
            smooth = 1e-08
            w = torch.ones_like(fg_prob)
            w = w + (boundary_factor - 1.0) * bmask
            intersection = (w * fg_prob * fg_targets).sum()
            denom = (w * (fg_prob + fg_targets)).sum() + smooth
            dice_w = 1 - 2 * intersection / denom
            bce_map = torch.nn.functional.binary_cross_entropy_with_logits(fg_logits, fg_targets, reduction='none')
            bce_w = (bce_map * w).sum() / (w.sum() + 1e-06)
            loss_dict['dice_loss'] = dice_w * self.weight[0]
            loss_dict['bce_loss'] = bce_w * self.weight[1]
            loss_dict['total_loss'] = loss_dict['dice_loss'] + loss_dict['bce_loss']
        if scale_weight != 1.0:
            loss_dict['dice_loss'] = loss_dict['dice_loss'] * scale_weight
            loss_dict['bce_loss'] = loss_dict['bce_loss'] * scale_weight
            loss_dict['total_loss'] = loss_dict['dice_loss'] + loss_dict['bce_loss']
        if consistency_weight > 0.0 and init_outputs is not None:
            if outputs.shape[1] == 2:
                out_fg = torch.sigmoid(outputs[:, 1:2])
                init_fg = torch.sigmoid(init_outputs[:, 1:2].detach())
            else:
                out_fg = torch.sigmoid(outputs)
                init_fg = torch.sigmoid(init_outputs.detach())
            consistency_loss = torch.mean((out_fg - init_fg) ** 2)
            loss_dict['consistency_loss'] = consistency_loss * consistency_weight
        else:
            loss_dict['consistency_loss'] = outputs.new_tensor(0.0)
        if use_diff_mask_loss and init_outputs is not None and (diff_mask_weight > 0):
            if outputs.shape[1] == 2:
                cur_logits = outputs[:, 1:2]
                init_logits = init_outputs[:, 1:2].detach()
            else:
                cur_logits = outputs
                init_logits = init_outputs.detach()
            cur_bin = (torch.sigmoid(cur_logits) > 0.5).float()
            init_bin = (torch.sigmoid(init_logits) > 0.5).float()
            diff_mask = (cur_bin - init_bin).abs()
            if diff_mask_boundary_only and boundary_mask is not None:
                diff_mask = diff_mask * boundary_mask
            if diff_mask.sum() > 0:
                if outputs.shape[1] == 2:
                    target_fg = targets[:, 1:2]
                else:
                    target_fg = targets
                bce_map_full = torch.nn.functional.binary_cross_entropy_with_logits(cur_logits, target_fg, reduction='none')
                diff_loss = (bce_map_full * diff_mask).sum() / (diff_mask.sum() + 1e-06)
                loss_dict['diff_mask_loss'] = diff_loss * diff_mask_weight
            else:
                loss_dict['diff_mask_loss'] = outputs.new_tensor(0.0)
        else:
            loss_dict['diff_mask_loss'] = outputs.new_tensor(0.0)
        if reg_weight > 0.0 and bn_param_pairs is not None and (len(bn_param_pairs) > 0):
            reg_accum = outputs.new_tensor(0.0)
            for current, orig in bn_param_pairs:
                reg_accum = reg_accum + (current - orig).pow(2).mean()
            loss_dict['reg_loss'] = reg_accum * reg_weight
        else:
            loss_dict['reg_loss'] = outputs.new_tensor(0.0)
        loss_dict['total_loss'] = loss_dict['total_loss'] + loss_dict['consistency_loss'] + loss_dict['reg_loss'] + loss_dict['diff_mask_loss']
        return (loss_dict, center_value)

class TestPointLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.seg_losses = nn.ModuleDict()
        self.seg_losses.update([('dice_loss', BinaryDiceLoss())])
        self.seg_losses.update([('bce_loss', nn.BCEWithLogitsLoss())])
        self.weight = [1, 0.5]

    def forward(self, outputs, targets, input_shape):
        fliter = torch.zeros_like(outputs) - 100000
        fliter[targets == 1] = 0
        fliter[:, :, int(input_shape[2] / 2), int(input_shape[3] / 2), int(input_shape[4] / 2)] = 0
        outputs = outputs + fliter
        dice_loss = self.seg_losses['dice_loss'](outputs, targets) * self.weight[0]
        bce_loss = self.seg_losses['bce_loss'](outputs, targets) * self.weight[1]
        loss_dict = {'dice_loss': dice_loss, 'bce_loss': bce_loss}
        total_loss = sum([loss_dict[k] for k in loss_dict.keys()])
        loss_dict['total_loss'] = total_loss
        return loss_dict

def region_loss(input: torch.Tensor, target: torch.Tensor, exclude_bg: bool=False) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not len(input.shape) == 4:
        raise ValueError(f'Invalid input shape, we expect BxNxHxW.             Got: {input.shape}')
    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f'input and target shapes must be the same.             Got: {input.shape} and {target.shape}')
    if not input.device == target.device:
        raise ValueError(f'input and target must be in the same device.             Got: {input.device} and {target.device}')
    input_soft: torch.Tensor = F.sigmoid(input)
    rl = (target * (1 - input_soft) + (1 - target) * input_soft).sum(dim=(2, 3))
    rl = rl / input[0, 0].numel()
    offset = 1 if exclude_bg else 0
    return rl.mean()

class RegionLoss(nn.Module):

    def __init__(self, exclude_bg: bool=True) -> None:
        super().__init__()
        self.exclude_bg = exclude_bg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return region_loss(input, target, self.exclude_bg)

def shape_loss(input: torch.Tensor, target: torch.Tensor, distance_maps: torch.Tensor, exclude_bg: bool=False) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not len(input.shape) == 4:
        raise ValueError(f'Invalid input shape, we expect BxNxHxW.             Got: {input.shape}')
    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f'input and target shapes must be the same.             Got: {input.shape} and {target.shape}')
    if not input.device == target.device:
        raise ValueError(f'input and target must be in the same device.             Got: {input.device} and {target.device}')
    input_soft: torch.Tensor = F.sigmoid(input)
    sl = (distance_maps - input_soft).abs().sum(dim=(2, 3))
    sl = sl / input[0, 0].numel()
    return sl.mean()

class ShapeLoss(nn.Module):

    def __init__(self, exclude_bg: bool=True) -> None:
        super().__init__()
        self.exclude_bg = exclude_bg

    def forward(self, input: torch.Tensor, target: torch.Tensor, distance_map: torch.Tensor) -> torch.Tensor:
        return shape_loss(input, target, distance_map, self.exclude_bg)

class SegReShLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.seg_losses = nn.ModuleDict()
        self.seg_losses.update([('dice_loss', BinaryDiceLoss())])
        self.seg_losses.update([('bce_loss', nn.BCEWithLogitsLoss())])
        self.seg_losses.update([('shape_loss', ShapeLoss(exclude_bg=False))])
        self.seg_losses.update([('region_loss', RegionLoss(exclude_bg=False))])
        self.weight = [1, 0.5, 0.5, 0.5]

    def forward(self, outputs, targets, distance_maps):
        dice_loss = self.seg_losses['dice_loss'](outputs, targets) * self.weight[0]
        bce_loss = self.seg_losses['bce_loss'](outputs, targets) * self.weight[1]
        shape_loss = self.seg_losses['shape_loss'](outputs, targets, distance_maps) * self.weight[2]
        region_loss = self.seg_losses['region_loss'](outputs, targets) * self.weight[2]
        loss_dict = {'dice_loss': dice_loss, 'bce_loss': bce_loss, 'shape_loss': shape_loss, 'region_loss': region_loss}
        total_loss = sum([loss_dict[k] for k in loss_dict.keys()])
        loss_dict['total_loss'] = total_loss
        return loss_dict
