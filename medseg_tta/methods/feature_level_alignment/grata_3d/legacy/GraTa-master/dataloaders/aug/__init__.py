from .slaug import LocationScaleAugmentation
from .spatial_aug import RandomRotate, RandomFlip
from .style_aug import get_strong_style_transform, get_weak_style_transform, fourier_augmentation
import random

def augmentation_strong_style(data):
    style_trans = get_strong_style_transform()
    Bezier_curve = LocationScaleAugmentation(vrange=(0.0, 1.0), background_threshold=0.01)
    data_aug = style_trans(**data)
    img = data_aug['data']
    return img

def augmentation_weak_style(data):
    style_trans = get_weak_style_transform()
    data_aug = style_trans(**data)
    img = data_aug['data']
    return img

def augmentation_spatial(x):
    augmentors = [RandomRotate(p=0.5), RandomFlip(p=0.5)]
    spatial_factors = []
    for aug in augmentors:
        x, factor = aug.forward(x)
        spatial_factors.append(factor)
    return (x, spatial_factors, augmentors)
