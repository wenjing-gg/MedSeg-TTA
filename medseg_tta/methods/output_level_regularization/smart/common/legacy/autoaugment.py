import numpy as np
import torch
import torch.nn as nn
import random
from torchvision.transforms.functional import affine
from ops import *
class LearnableImageNetPolicy(nn.Module):
    """ 同时应用所有25个子策略并返回可学习权重 """
    def __init__(self, fillcolor=(128, 128, 128)):
        super().__init__()
        self.policies = nn.ModuleList([
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.8, "equalize", 8, 1.0, "gaussnoise", 6, fillcolor),
            SubPolicy(0.5, "posterize", 7, 1.0, "gaussnoise", 6, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "gaussnoise", 6, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "gaussnoise", 6, fillcolor),
            SubPolicy(1.0, "contrast", 8, 1.0, "gaussnoise", 6, fillcolor),
            SubPolicy(1.0, "gaussnoise", 6, 1.0, "gaussnoise", 6, fillcolor)
        ])
        self.weights = nn.Parameter(torch.ones(len(self.policies)))

    def forward(self, img):
        augmented_list = []
        selected_indices = random.sample(range(len(self.policies)), 5)
        selected_policies = [self.policies[i] for i in selected_indices]
        selected_weights = self.weights[selected_indices]
        for i,policy in enumerate(selected_policies):
            augmented_img = policy(img)
            augmented_list.append(augmented_img)
            #print(i,augmented_img.shape)
        augmented_tensor = torch.stack(augmented_list)
        
        normalized_weights = torch.softmax(selected_weights, dim=0)
        
        return augmented_tensor, normalized_weights, selected_indices

class SubPolicy(nn.Module):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        super().__init__()
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
            "gaussnoise": [0] * 10
        }

        func = {
            "shearX": ShearX(fillcolor=fillcolor),
            "shearY": ShearY(fillcolor=fillcolor),
            "translateX": TranslateX(fillcolor=fillcolor),
            "translateY": TranslateY(fillcolor=fillcolor),
            "rotate": Rotate(),
            "color": Color(),
            "posterize": Posterize(),
            "solarize": Solarize(),
            "contrast": Contrast(),
            "sharpness": Sharpness(),
            "brightness": Brightness(),
            "autocontrast": AutoContrast(),
            "equalize": Equalize(),
            "invert": Invert(),
            "gaussnoise": gaussnoise()
        }
        
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def forward(self, x):
        '''if isinstance(x, torch.Tensor):
            # 转换 MetaTensor 到普通张量
            x = x.as_subclass(torch.Tensor)
            # 张量转 PIL 图像
            x_np = x.cpu().detach().numpy()
            x_np = (x_np * 255).astype(np.uint8)
            x_np = np.transpose(x_np, (1, 2, 0))
            x = Image.fromarray(x_np)'''
        if random.random() < self.p1:
            x = self.operation1(x, self.magnitude1)
        if random.random() < self.p2:
            x = self.operation2(x, self.magnitude2)
        return x