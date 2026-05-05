import torch
import numpy as np
from dataloaders.aug.fourier import FDA_source_to_target_np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform

def fourier_augmentation(data, fda_beta=0.15):
    this_fda_beta = round(np.random.random() * fda_beta, 2)
    lowf_batch = np.random.permutation(data)
    fda_data = FDA_source_to_target_np(data, lowf_batch, L=this_fda_beta)
    return fda_data

def augment_lowfreq(input, beta=0.01, target_lowfreq=None, t=1.0):
    batch, _, imgH, imgW = input.size()
    lowfreq_H, lowfreq_W = (int(imgH * beta), int(imgW * beta))
    padding_H, padding_W = ((imgH - lowfreq_H) // 2, (imgW - lowfreq_W) // 2)
    fft = torch.fft.fft2(input.clone(), dim=(-2, -1))
    amp_src, pha_src = (torch.abs(fft), torch.angle(fft))
    amp_src = torch.fft.fftshift(amp_src)
    low_freq = amp_src[:, :, padding_H:padding_H + lowfreq_H, padding_W:padding_W + lowfreq_W]
    if target_lowfreq is None:
        return (input, low_freq)
    else:
        target_lowfreq = torch.cat((target_lowfreq, low_freq), dim=0)
    aug_lowfreq = torch.normal(mean=target_lowfreq.mean(dim=0).repeat(batch, 1, 1, 1), std=target_lowfreq.std(dim=0).repeat(batch, 1, 1, 1) / t)
    amp_src[:, :, padding_H:padding_H + lowfreq_H, padding_W:padding_W + lowfreq_W] = aug_lowfreq
    amp_src = torch.fft.ifftshift(amp_src)
    real = torch.cos(pha_src) * amp_src
    imag = torch.sin(pha_src) * amp_src
    fft_src_ = torch.complex(real=real, imag=imag)
    src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
    return (src_in_trg, target_lowfreq)

def get_strong_style_transform():
    trans = []
    trans.append(BrightnessMultiplicativeTransform((0.5, 1.5), per_channel=True, p_per_sample=0.75))
    trans.append(ContrastAugmentationTransform((0.5, 1.5), per_channel=True, p_per_sample=0.75))
    trans.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.75))
    trans.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.5))
    trans.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True, p_per_channel=0.5, p_per_sample=0.5))
    trans = Compose(trans)
    return trans

def get_weak_style_transform():
    trans = []
    trans.append(BrightnessMultiplicativeTransform((0.75, 1.25), per_channel=True, p_per_sample=0.25))
    trans.append(ContrastAugmentationTransform((0.75, 1.25), per_channel=True, p_per_sample=0.25))
    trans = Compose(trans)
    return trans
