import torch
import torchvision
import torch.nn.functional as F
import math
import random

class ShearX(object):
    def __init__(self, fillcolor=0.5):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        direction = random.choice([-1, 1])
        shear_factor = magnitude * direction
        # 3D仿射矩阵 (1, 3, 4)
        affine_matrix = torch.tensor([
            [1, shear_factor, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).to(x.device)
        
        grid = F.affine_grid(affine_matrix, x.size(), align_corners=False)
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

class ShearY(object):
    def __init__(self, fillcolor=0.5):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        direction = random.choice([-1, 1])
        shear_factor = magnitude * direction
        
        affine_matrix = torch.tensor([
            [1, 0, 0, 0],
            [shear_factor, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).to(x.device)
        
        grid = F.affine_grid(affine_matrix, x.size(), align_corners=False)
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

class TranslateX(object):
    def __init__(self, fillcolor=0.5):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        direction = random.choice([-1, 1])
        W = x.size()[4]  # 宽度维度
        tx = 2 * magnitude * direction  # 归一化平移量
        
        affine_matrix = torch.tensor([
            [1, 0, 0, tx],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).to(x.device)
        
        grid = F.affine_grid(affine_matrix, x.size(), align_corners=False)
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

class TranslateY(object):
    def __init__(self, fillcolor=0.5):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        direction = random.choice([-1, 1])
        H = x.size()[3]  # 高度维度
        ty = 2 * magnitude * direction  # 归一化平移量
        
        affine_matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, ty],
            [0, 0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).to(x.device)
        
        grid = F.affine_grid(affine_matrix, x.size(), align_corners=False)
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

class Rotate:
    def __init__(self, fillcolor=0.5):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        # 生成随机旋转方向
        direction = random.choice([-1, 1])
        angle = magnitude * direction

        # 定义填充值
        fill_value = 128.0 / 255.0  # 转换为 [0, 1] 范围的值

        # 初始化旋转后的张量
        rotated_x = torch.zeros_like(x)

        # 对每个深度片进行旋转
        for d in range(x.size(2)):
            # 提取深度片
            img = x[:, :, d, :, :].squeeze(0)  # 去掉深度维度和批次维度

            # 应用旋转
            rotated_img = torchvision.transforms.functional.rotate(img, angle, fill=fill_value)

            # 将旋转后的深度片放回张量
            rotated_x[:, :, d, :, :] = rotated_img.unsqueeze(0)  # 恢复批次维度

        return rotated_x
        
# ---------------------- 颜色调整类（支持批量处理）---------------------
class Color:
    def __call__(self, x, magnitude):
        direction = torch.rand(x.size(0), device=x.device) * 2 - 1  # [-1,1) 均匀分布
        factor = 1 + magnitude * direction.view(-1, 1, 1, 1, 1)
        return torch.clamp(x * factor, 0, 1)  # 保持颜色在 [0,1]

class Contrast:
    def __call__(self, x, magnitude):
        direction = torch.rand(x.size(0), device=x.device) * 2 - 1
        factor = 1 + magnitude * direction.view(-1, 1, 1, 1, 1)
        mean = x.mean(dim=(2,3,4), keepdim=True)
        return torch.clamp(mean + (x - mean) * factor, 0, 1)

class Brightness:
    def __call__(self, x, magnitude):
        direction = torch.rand(x.size(0), device=x.device) * 2 - 1
        delta = magnitude * direction.view(-1, 1, 1, 1, 1)
        return torch.clamp(x + delta, 0, 1)

class Sharpness:
    def __call__(self, x, magnitude):
        # 生成随机锐化方向
        direction = random.choice([-1, 1])
        sharpness_factor = 1 + magnitude * direction

        # 创建一个简单的锐化滤波器（3x3x3）
        kernel = torch.tensor([
            [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
            [[0, -1, 0], [-1, 9, -1], [0, -1, 0]],
            [[0, 0, 0], [0, -1, 0], [0, 0, 0]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 扩展核以适应多个通道
        kernel = kernel.expand(x.size(1), -1, -1, -1, -1)

        # 将输入张量移动到与核相同的设备上
        kernel = kernel.to(x.device)

        # 应用锐化操作
        sharpened = F.conv3d(x, kernel, padding=1, groups=x.size(1))

        # 确保输出张量的值在合理范围内
        sharpened = torch.clamp(sharpened, min=0.0, max=1.0)

        return sharpened

    
# ---------------------- 像素级操作（批量处理）---------------------
class Posterize:
    def __call__(self, x, magnitude):
        x_uint8 = (x * 255).byte()
        mask = 0xFF << (8 - magnitude)
        return (x_uint8 & mask).float() / 255.0

class Solarize:
    def __call__(self, x, magnitude):
        threshold = 1.0 - magnitude
        return torch.where(x > threshold, 1.0 - x, x)

class AutoContrast:
    def __call__(self, x, magnitude):
        min_val = x.amin(dim=(2,3,4), keepdim=True)
        max_val = x.amax(dim=(2,3,4), keepdim=True)
        scale = 1.0 / (max_val - min_val + 1e-5)
        return (x - min_val) * scale

class Equalize:
    def __call__(self, x, magnitude):
        # 直方图均衡化（逐通道处理）
        B, C, D, H, W = x.shape
        x_eq = x.clone()
        for b in range(B):
            for c in range(C):
                for d in range(D):
                    img = x[b, c, d]  # (H, W)
                    hist = torch.histc(img * 255, bins=256, min=0, max=255)
                    cdf = hist.cumsum(dim=0)
                    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-5)
                    x_eq[b, c, d] = cdf[(img * 255).clamp(0, 255).long()] / 255.0
        return x_eq

class Invert:
    def __call__(self, x, magnitude):
        return 1.0 - x

class gaussnoise:
    def __call__(self, x, magnitude):
        noise = torch.randn_like(x) * 0.01
        return x + noise