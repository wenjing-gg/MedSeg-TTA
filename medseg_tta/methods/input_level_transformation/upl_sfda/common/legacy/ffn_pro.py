import numpy as np
import torch
import scipy.ndimage
import os
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

class SpatialMotionSimLayer:
    def __init__(self,
                 max_translation=5,
                 apply_blur=True,
                 blur_sigma_range=(0.5, 1.5),
                 apply_ghosting=False,
                 ghost_shift=3,
                 ghost_intensity=0.3,
                 p=1.0):
        self.max_translation = max_translation
        self.apply_blur = apply_blur
        self.blur_sigma_range = blur_sigma_range
        self.apply_ghosting = apply_ghosting
        self.ghost_shift = ghost_shift
        self.ghost_intensity = ghost_intensity
        self.p = p

    def random_translate(self, image):
        shifts = np.random.randint(-self.max_translation, self.max_translation + 1, size=3)
        return scipy.ndimage.shift(image, shift=shifts, order=1, mode='nearest')

    def random_blur(self, image):
        sigma = np.random.uniform(*self.blur_sigma_range)
        return scipy.ndimage.gaussian_filter(image, sigma=sigma)

    def apply_ghost(self, image):
        ghost = np.roll(image, shift=self.ghost_shift, axis=2)  # W axis
        return image * (1 - self.ghost_intensity) + ghost * self.ghost_intensity

    def __call__(self, tensor_5d):  # [B, C, D, H, W]
        if not isinstance(tensor_5d, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        device = tensor_5d.device
        dtype = tensor_5d.dtype
        b, c, d, h, w = tensor_5d.shape

        tensor_np = tensor_5d.detach().cpu().numpy()
        output = np.empty_like(tensor_np)

        for i in range(b):
            for j in range(c):
                img = np.copy(tensor_np[i, j])  # shape: [D, H, W]
                if np.random.rand() < self.p:
                    img = self.random_translate(img)
                    if self.apply_blur:
                        img = self.random_blur(img)
                    if self.apply_ghosting:
                        img = self.apply_ghost(img)
                output[i, j] = img

        return torch.from_numpy(output).to(dtype=dtype, device=device)

def process_images():
    # 定义源目录和目标目录
    src_dir = "/root/autodl-tmp/BraTS2024/test"
    dst_dir = "/root/autodl-tmp/BraTS2024/test_ffn"
    
    # 创建目标目录(如果不存在)
    os.makedirs(dst_dir, exist_ok=True)
    
    # 初始化SpatialMotionSimLayer
    sim_layer = SpatialMotionSimLayer(max_translation=2, apply_blur=True,blur_sigma_range=(0.3, 0.7),apply_ghosting=False,p=0.8)
    
    # 获取源目录中的所有nii.gz文件
    files = list(Path(src_dir).glob('**/*.nii.gz'))
    
    print(f"找到 {len(files)} 个文件待处理")
    
    # 处理每个文件
    for file_path in tqdm(files):
        # 构建目标文件路径，保持相对路径结构
        rel_path = file_path.relative_to(src_dir)
        dst_path = Path(dst_dir) / rel_path
        
        # 确保目标目录存在
        os.makedirs(dst_path.parent, exist_ok=True)
        
        # 加载NIfTI文件
        img = nib.load(str(file_path))
        data = img.get_fdata()
        
        # 为了符合模型的输入要求，调整维度
        # 转换为 [B, C, D, H, W] 格式
        data_tensor = torch.from_numpy(data).float()
        
        # 添加批次和通道维度
        if len(data_tensor.shape) == 3:  # [D, H, W]
            data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        elif len(data_tensor.shape) == 4:  # [C, D, H, W]
            data_tensor = data_tensor.unsqueeze(0)  # [1, C, D, H, W]
        
        # 应用SpatialMotionSimLayer
        processed_tensor = sim_layer(data_tensor)
        
        # 转回原始维度
        if len(data.shape) == 3:
            processed_data = processed_tensor.squeeze(0).squeeze(0).numpy()
        else:
            processed_data = processed_tensor.squeeze(0).numpy()
        
        # 创建新的NIfTI文件并保存
        new_img = nib.Nifti1Image(processed_data, img.affine, img.header)
        nib.save(new_img, str(dst_path))
        
    print(f"所有图像已处理并保存到 {dst_dir}")

if __name__ == "__main__":
    process_images()