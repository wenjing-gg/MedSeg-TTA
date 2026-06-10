import torch
import torch.nn as nn
import numpy as np
import torch
from autoaugment import LearnableImageNetPolicy

class Augmentmodel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.aug_part = LearnableImageNetPolicy()
        self.model = model
    def forward(self, img, aug = 0):
        if aug == 1:
            #print(img.shape)
            aug_img, weight, _ = self.aug_part(img)
            aug_img = aug_img.transpose(0, 1).contiguous().view(-1, 1, 128, 128, 128)
            aug_img = aug_img.squeeze()
            out =  self.model.forward(aug_img)
            return out, weight
        else:
            out =  self.model.forward(img)
            return out