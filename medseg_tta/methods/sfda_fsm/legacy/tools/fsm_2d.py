import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

class FSMGenerator2D(nn.Module):

    def __init__(self, input_channels: int=1, inversion_steps: int=100, fda_beta: float=0.05):
        super().__init__()
        self.input_channels = input_channels
        self.inversion_steps = inversion_steps
        self.fda_beta = fda_beta
        self.inversion_lr = 0.01
        self.noise_std = 0.1

    def forward(self, target_imgs: torch.Tensor, model: nn.Module) -> Tuple[torch.Tensor, Dict]:
        batch_size, channels, height, width = target_imgs.shape
        device = target_imgs.device
        source_like_imgs = self._domain_inversion(target_imgs, model)
        source_like_imgs = self._fourier_domain_adaptation(source_like_imgs, target_imgs)
        source_stats = {'mean': torch.mean(source_like_imgs, dim=(2, 3), keepdim=True), 'std': torch.std(source_like_imgs, dim=(2, 3), keepdim=True), 'min': torch.min(source_like_imgs.view(batch_size, channels, -1), dim=2)[0].unsqueeze(-1).unsqueeze(-1), 'max': torch.max(source_like_imgs.view(batch_size, channels, -1), dim=2)[0].unsqueeze(-1).unsqueeze(-1)}
        return (source_like_imgs, source_stats)

    def _domain_inversion(self, target_imgs: torch.Tensor, model: nn.Module) -> torch.Tensor:
        source_like_imgs = target_imgs.clone()
        noise = torch.randn_like(source_like_imgs) * self.noise_std
        source_like_imgs = source_like_imgs + noise
        brightness_factor = 0.8 + 0.4 * torch.rand(source_like_imgs.shape[0], 1, 1, 1).to(source_like_imgs.device)
        contrast_factor = 0.8 + 0.4 * torch.rand(source_like_imgs.shape[0], 1, 1, 1).to(source_like_imgs.device)
        source_like_imgs = source_like_imgs * contrast_factor + brightness_factor * 0.1
        source_like_imgs = torch.clamp(source_like_imgs, 0, 1)
        return source_like_imgs

    def _fourier_domain_adaptation(self, source_imgs: torch.Tensor, target_imgs: torch.Tensor) -> torch.Tensor:
        adapted_imgs = []
        for i in range(source_imgs.shape[0]):
            src_img = source_imgs[i].cpu().numpy()
            tgt_img = target_imgs[i].cpu().numpy()
            adapted_channels = []
            for c in range(src_img.shape[0]):
                adapted_channel = self._fda_transform(src_img[c], tgt_img[c])
                adapted_channels.append(adapted_channel)
            adapted_img = np.stack(adapted_channels, axis=0)
            adapted_imgs.append(torch.from_numpy(adapted_img).float())
        return torch.stack(adapted_imgs).to(source_imgs.device)

    def _fda_transform(self, src_img: np.ndarray, tgt_img: np.ndarray) -> np.ndarray:
        fft_src = np.fft.fft2(src_img)
        fft_tgt = np.fft.fft2(tgt_img)
        amp_src = np.abs(fft_src)
        amp_tgt = np.abs(fft_tgt)
        phase_src = np.angle(fft_src)
        h, w = src_img.shape
        b = np.floor(np.amin((h, w)) * self.fda_beta).astype(int)
        mask = np.zeros((h, w))
        mask[h // 2 - b:h // 2 + b, w // 2 - b:w // 2 + b] = 1
        amp_mixed = amp_src * (1 - mask) + amp_tgt * mask
        fft_mixed = amp_mixed * np.exp(1j * phase_src)
        adapted_img = np.real(np.fft.ifft2(fft_mixed))
        return adapted_img

class ContrastiveDomainDistillation2D(nn.Module):

    def __init__(self, temperature: float=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, source_preds: torch.Tensor, target_preds: torch.Tensor, source_preds_aug: torch.Tensor, target_preds_aug: torch.Tensor) -> Dict[str, torch.Tensor]:
        distill_loss = self._compute_distillation_loss(source_preds, target_preds)
        contrast_loss = self._compute_contrastive_loss(source_preds, target_preds, source_preds_aug, target_preds_aug)
        return {'distill_loss': distill_loss, 'contrast_loss': contrast_loss}

    def _compute_distillation_loss(self, source_preds: torch.Tensor, target_preds: torch.Tensor) -> torch.Tensor:
        source_prob = F.softmax(source_preds / self.temperature, dim=1)
        target_log_prob = F.log_softmax(target_preds / self.temperature, dim=1)
        kl_loss = F.kl_div(target_log_prob, source_prob, reduction='batchmean')
        return kl_loss * self.temperature ** 2

    def _compute_contrastive_loss(self, source_preds: torch.Tensor, target_preds: torch.Tensor, source_preds_aug: torch.Tensor, target_preds_aug: torch.Tensor) -> torch.Tensor:
        source_sim = F.cosine_similarity(source_preds.view(source_preds.size(0), -1), source_preds_aug.view(source_preds_aug.size(0), -1), dim=1)
        target_sim = F.cosine_similarity(target_preds.view(target_preds.size(0), -1), target_preds_aug.view(target_preds_aug.size(0), -1), dim=1)
        contrast_loss = -torch.mean(source_sim) - torch.mean(target_sim)
        return contrast_loss

class CompactAwareDomainConsistency2D(nn.Module):

    def __init__(self, num_classes: int, confidence_threshold: float=0.7, compactness_threshold: float=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.compactness_threshold = compactness_threshold

    def forward(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_classes, height, width = predictions.shape
        probs = F.softmax(predictions, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        confidence_mask = max_probs > self.confidence_threshold
        compactness_mask = self._compute_compactness_mask(probs)
        pseudo_mask = confidence_mask & compactness_mask
        return (pseudo_labels, pseudo_mask)

    def _compute_compactness_mask(self, probs: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes, height, width = probs.shape
        compactness_mask = torch.ones(batch_size, height, width, dtype=torch.bool, device=probs.device)
        for b in range(batch_size):
            for c in range(num_classes):
                prob_map = probs[b, c]
                kernel_size = 3
                pad = kernel_size // 2
                local_mean = F.avg_pool2d(prob_map.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=pad).squeeze()
                local_var = F.avg_pool2d((prob_map.unsqueeze(0).unsqueeze(0) - local_mean.unsqueeze(0).unsqueeze(0)) ** 2, kernel_size=kernel_size, stride=1, padding=pad).squeeze()
                class_compact_mask = local_var < self.compactness_threshold
                max_class = torch.argmax(probs[b], dim=0)
                class_mask = max_class == c
                compactness_mask[b] = compactness_mask[b] & (class_compact_mask | ~class_mask)
        return compactness_mask
