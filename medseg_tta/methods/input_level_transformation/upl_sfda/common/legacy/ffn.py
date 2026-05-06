import numpy as np
import torch
import scipy.ndimage

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
