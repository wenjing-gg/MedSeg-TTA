import numpy as np
import torch

def get_dice_threshold(output, mask, threshold=0.5):
    smooth = 1e-06
    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    intersection = (output * mask).sum()
    dice = (2.0 * intersection + smooth) / (output.sum() + mask.sum() + smooth)
    return dice

def get_hard_dice(outputs, masks, std=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    dice_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        dice_list.append(get_dice_threshold(output, mask, threshold=0.5))
    if not std:
        return np.mean(dice_list)
    else:
        return (np.mean(dice_list), np.std(dice_list), dice_list)
