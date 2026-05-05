import torch
import torch.nn as nn, numpy as np
from skimage.exposure import match_histograms

class HistMatching(nn.Module):

    def __init__(self, model, base_volume):
        super().__init__()
        self.model = model
        self.base_volume = base_volume.numpy()

    def forward(self, x: torch.Tensor):
        device = x.device
        x = x.cpu().numpy()
        matched_vol = np.zeros_like(x)
        for i in range(x.shape[0]):
            matched_vol[i] = match_histograms(x[i], self.base_volume)
        matched_vol = torch.from_numpy(matched_vol).to(device)
        return self.model(matched_vol)
