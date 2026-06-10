import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple


def get_dataset_type_from_path(data_path: str) -> str:
    """
    从数据路径中提取数据集类型
    Args:
        data_path: 数据路径，如 "E:/tta_dataset/TTA-2DCXR/Shenzhen_/image"
    Returns:
        数据集类型: "CXR", "dermoscopy", "PATH", "US"
    """
    data_path = data_path.replace('\\', '/').lower()

    if 'tta-2dcxr' in data_path:
        return 'CXR'
    elif 'tta-2ddermoscopy' in data_path:
        return 'dermoscopy'
    elif 'tta-2dpath' in data_path:
        return 'PATH'
    elif 'tta-2dus' in data_path:
        return 'US'
    else:
        # 默认返回CXR
        return 'CXR'


def get_dataset_paths(dataset_type: str, base_dir: str = r"E:\tta_dataset", subfolder: str = None) -> Tuple[str, str]:
    """
    根据数据集类型获取图像和掩码路径
    Args:
        dataset_type: 数据集类型 ("CXR", "dermoscopy", "PATH", "US")
        base_dir: 基础目录路径
        subfolder: 指定子文件夹，如果为None则自动选择带下划线后缀的文件夹
    Returns:
        (image_dir, mask_dir) 元组
    """
    dataset_mapping = {
        'CXR': 'TTA-2DCXR',
        'dermoscopy': 'TTA-2Ddermoscopy',
        'PATH': 'TTA-2DPATH',
        'US': 'TTA-2DUS'
    }

    if dataset_type not in dataset_mapping:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                        f"Supported types: {list(dataset_mapping.keys())}")

    dataset_folder = dataset_mapping[dataset_type]
    dataset_path = os.path.join(base_dir, dataset_folder)

    # 如果没有指定子文件夹，自动选择
    if subfolder is None:
        subfolder = _auto_select_subfolder(dataset_path, dataset_type)

    if subfolder:
        image_dir = os.path.join(dataset_path, subfolder, 'image')
        mask_dir = os.path.join(dataset_path, subfolder, 'mask')
    else:
        image_dir = os.path.join(dataset_path, 'image')
        mask_dir = os.path.join(dataset_path, 'mask')

    return image_dir, mask_dir


def _auto_select_subfolder(dataset_path: str, dataset_type: str) -> str:
    """
    自动选择子文件夹，优先选择带下划线后缀的文件夹
    Args:
        dataset_path: 数据集路径
        dataset_type: 数据集类型
    Returns:
        选择的子文件夹名称，如果没有子文件夹则返回空字符串
    """
    if not os.path.exists(dataset_path):
        # 如果数据集路径不存在，返回默认值
        if dataset_type == 'CXR':
            return 'Shenzhen_'
        else:
            return ''

    # 获取所有子文件夹
    try:
        subfolders = [f for f in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, f))]
    except PermissionError:
        return ''

    if not subfolders:
        return ''

    # 优先选择带下划线后缀的文件夹
    underscore_folders = [f for f in subfolders if f.endswith('_')]
    
    if underscore_folders:
        # 如果有多个带下划线的文件夹，选择第一个（按字母顺序）
        underscore_folders.sort()
        return underscore_folders[0]

    # 如果没有带下划线的文件夹，选择第一个文件夹
    subfolders.sort()
    return subfolders[0]


class MedicalImageDataset2D(Dataset):
    """
    2D医学图像分割数据集
    支持多种医学图像数据集：
    - TTA-2DCXR: 胸部X光图像
    - TTA-2Ddermoscopy: 皮肤镜图像
    - TTA-2DPATH: 病理图像
    - TTA-2DUS: 超声图像
    """

    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 phase: str = 'train',
                 image_size: Tuple[int, int] = (256, 256),
                 normalize: bool = True):
        """
        Args:
            image_dir: 图像文件夹路径
            mask_dir: 掩码文件夹路径
            phase: 'train' 或 'val' 或 'test'
            image_size: 图像尺寸 (height, width)
            normalize: 是否对图像进行归一化
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.phase = phase
        self.image_size = image_size
        self.normalize = normalize

        # 自动检测数据集类型
        self.dataset_type = get_dataset_type_from_path(image_dir)

        # 支持的图像文件扩展名
        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

        # 获取所有图像文件名
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")
        
        self.image_files = []
        for ext in self.supported_extensions:
            self.image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        self.image_files.sort()  # 确保顺序一致

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {image_dir}")

        # 验证对应的掩码文件是否存在
        self.valid_files = []
        for img_file in self.image_files:
            # 尝试不同的掩码文件名匹配策略
            mask_candidates = [
                img_file,  # 相同文件名
                os.path.splitext(img_file)[0] + '.png',  # 同名但.png扩展名
                os.path.splitext(img_file)[0] + '.jpg',  # 同名但.jpg扩展名
                os.path.splitext(img_file)[0] + '_lesion.bmp',  # 同名但.bmp扩展名
                os.path.splitext(img_file)[0] + '-1.tif',  # 同名但.tif扩展名
                os.path.splitext(img_file)[0] + '.tiff', # 同名但.tiff扩展名
            ]
            
            for mask_file in mask_candidates:
                mask_path = os.path.join(mask_dir, mask_file)
                #raise ValueError(mask_file)
                if os.path.exists(mask_path):
                    self.valid_files.append(img_file)
                    break

        if len(self.valid_files) == 0:
            raise ValueError(f"No valid image-mask pairs found. Image dir: {image_dir}, Mask dir: {mask_dir}")

        # 定义数据增强
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        """根据阶段返回相应的数据变换"""
        if self.phase == 'train':
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])

    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        try:
            image = Image.open(image_path).convert('L')  # 转换为灰度图
            image = np.array(image)
            return image
        except Exception as e:
            raise ValueError(f"Cannot load image: {image_path}, Error: {e}")

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """加载掩码"""
        try:
            mask = Image.open(mask_path).convert('L')  # 转换为灰度图
            mask = np.array(mask)

            # 将掩码值转换为类别标签
            # 假设掩码中0为背景，255为前景（肺部区域）
            mask = (mask > 127).astype(np.uint8)  # 二值化：0为背景，1为前景
            return mask
        except Exception as e:
            raise ValueError(f"Cannot load mask: {mask_path}, Error: {e}")

    def _find_mask_path(self, img_file: str) -> str:
        """查找对应的掩码文件路径"""
        base_name = os.path.splitext(img_file)[0]
        img_ext = os.path.splitext(img_file)[1]
        
        # 掩码文件候选列表：只考虑-1和_segmentation后缀
        mask_candidates = [
            # 1. 完全相同的文件名
            img_file,
            
            # 2. 相同基础名称但不同扩展名
            base_name + '.png',
            base_name + '.jpg',
            base_name + '.jpeg',
            base_name + '.bmp',
            base_name + '.tif',
            base_name + '.tiff',
            
            # 3. 基础名称加-1后缀
            base_name + '-1.png',
            base_name + '-1.jpg',
            base_name + '-1.jpeg',
            base_name + '_lesion.bmp',
            base_name + '-1.tif',
            base_name + '-1.tiff',
            
            # 4. 基础名称加_segmentation后缀
            base_name + '_segmentation.png',
            base_name + '_segmentation.jpg',
            base_name + '_segmentation.jpeg',
            base_name + '_segmentation.bmp',
            base_name + '_segmentation.tif',
            base_name + '_segmentation.tiff',
        ]
        
        # 查找匹配的掩码文件
        for mask_file in mask_candidates:
            mask_path = os.path.join(self.mask_dir, mask_file)
            if os.path.exists(mask_path):
                return mask_path
        
        raise ValueError(f"Cannot find mask file for image: {img_file}")

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            image: 形状为 (1, H, W) 的图像张量
            mask: 形状为 (H, W) 的掩码张量
            filename: 文件名
        """
        filename = self.valid_files[idx]

        # 加载图像和掩码
        image_path = os.path.join(self.image_dir, filename)
        mask_path = self._find_mask_path(filename)

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        # 转换为PIL图像以便使用transforms
        image_pil = Image.fromarray(image)
        # 修正：掩码转换需要乘以255才能正确处理
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

        # 应用变换
        if self.phase == 'train':
            # 对于训练阶段，需要对图像和掩码应用相同的几何变换
            seed = np.random.randint(2147483647)

            # 图像变换
            np.random.seed(seed)
            torch.manual_seed(seed)
            image_tensor = self.transforms(image_pil)

            # 掩码变换（只应用几何变换，不应用颜色变换）
            np.random.seed(seed)
            torch.manual_seed(seed)
            mask_transforms = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
            mask_tensor = mask_transforms(mask_pil)
        else:
            image_tensor = self.transforms(image_pil)
            mask_transforms = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
            mask_tensor = mask_transforms(mask_pil)

        # 修正：确保掩码在0-1范围内，然后转换为长整型
        mask_tensor = (mask_tensor > 0.5).long().squeeze(0)

        # 如果需要归一化图像
        if self.normalize:
            image_tensor = (image_tensor - image_tensor.mean()) / (image_tensor.std() + 1e-8)

        return image_tensor, mask_tensor, filename


# 修正的数据集包装类，用于正确设置phase
class DatasetWithPhase(Dataset):
    def __init__(self, subset_dataset, phase):
        self.subset_dataset = subset_dataset
        self.phase = phase
        
        # 获取原始数据集（从Subset中）
        self.original_dataset = subset_dataset.dataset
        
        # 设置原始数据集的phase
        self.original_dataset.phase = phase
        
        # 重新生成transforms
        self.original_dataset.transforms = self.original_dataset._get_transforms()
    
    def __len__(self):
        return len(self.subset_dataset)
    
    def __getitem__(self, idx):
        return self.subset_dataset[idx]


def get_data_loaders_2d(image_dir: str = None,
                       mask_dir: str = None,
                       dataset_type: str = None,
                       subfolder: str = None,
                       base_dir: str = r"E:\tta_dataset",
                       batch_size_train: int = 8,
                       batch_size_val: int = 4,
                       num_workers: int = 4,
                       train_split: float = 0.9,
                       image_size: Tuple[int, int] = (256, 256)) -> Tuple[DataLoader, DataLoader, str]:
    """
    创建训练和验证数据加载器

    Args:
        image_dir: 图像文件夹路径（可选，如果提供dataset_type则自动生成）
        mask_dir: 掩码文件夹路径（可选，如果提供dataset_type则自动生成）
        dataset_type: 数据集类型 ("CXR", "dermoscopy", "PATH", "US")
        subfolder: 子文件夹名称（可选，如果为None则自动选择带下划线后缀的文件夹）
        base_dir: 基础目录路径
        batch_size_train: 训练批次大小
        batch_size_val: 验证批次大小
        num_workers: 数据加载器工作进程数
        train_split: 训练集比例
        image_size: 图像尺寸

    Returns:
        train_loader, val_loader, dataset_type
    """
    # 如果提供了dataset_type，自动生成路径
    if dataset_type is not None:
        image_dir, mask_dir = get_dataset_paths(dataset_type, base_dir, subfolder)
    elif image_dir is not None:
        # 从路径中自动检测数据集类型
        dataset_type = get_dataset_type_from_path(image_dir)
    else:
        raise ValueError("Either provide image_dir/mask_dir or dataset_type")
    
    # 创建完整数据集
    full_dataset = MedicalImageDataset2D(
        image_dir=image_dir,
        mask_dir=mask_dir,
        phase='train',
        image_size=image_size
    )

    # 划分训练集和验证集
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    train_indices, val_indices = torch.utils.data.random_split(
        range(total_size), [train_size, val_size]
    )

    # 创建训练和验证数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices.indices)

    # 使用包装类来正确设置phase
    train_dataset = DatasetWithPhase(train_dataset, 'train')
    val_dataset = DatasetWithPhase(val_dataset, 'val')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, dataset_type


def main():
    """
    主函数，用于测试数据加载器的功能
    """
    print("Testing Medical Image Dataset 2D...")
    
    # 测试1: 使用示例路径创建数据集
    try:
        # 这里使用示例路径，实际使用时需要替换为真实路径
        import tempfile
        test_image_dir = os.path.join(tempfile.gettempdir(), "test_images")
        test_mask_dir = os.path.join(tempfile.gettempdir(), "test_masks")
        
        # 创建测试目录和文件（如果不存在）
        os.makedirs(test_image_dir, exist_ok=True)
        os.makedirs(test_mask_dir, exist_ok=True)
        
        # 创建一些测试图像和掩码
        for i in range(5):
            # 创建随机测试图像
            test_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            test_mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
            
            img_path = os.path.join(test_image_dir, f"test_{i:03d}.png")
            mask_path = os.path.join(test_mask_dir, f"test_{i:03d}.png")
            
            Image.fromarray(test_img).save(img_path)
            Image.fromarray(test_mask).save(mask_path)
        
        print(f"Created test dataset with {len(os.listdir(test_image_dir))} images")
        
        # 测试数据集创建
        dataset = MedicalImageDataset2D(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            phase='train',
            image_size=(128, 128),
            normalize=True
        )
        
        print(f"Dataset created successfully with {len(dataset)} samples")
        print(f"Dataset type detected: {dataset.dataset_type}")
        
        # 测试数据加载
        sample_image, sample_mask, sample_filename = dataset[0]
        print(f"Sample loaded: {sample_filename}")
        print(f"Image shape: {sample_image.shape}, dtype: {sample_image.dtype}")
        print(f"Mask shape: {sample_mask.shape}, dtype: {sample_mask.dtype}")
        print(f"Image range: [{sample_image.min():.4f}, {sample_image.max():.4f}]")
        print(f"Mask range: [{sample_mask.min()}, {sample_mask.max()}]")
        
        # 测试数据加载器创建
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        
        # 测试批次加载
        for batch_idx, (images, masks, filenames) in enumerate(train_loader):
            print(f"Batch {batch_idx}: images shape {images.shape}, masks shape {masks.shape}")
            if batch_idx >= 1:  # 只测试前两个批次
                break
        
        print("Basic dataset test passed!")
        
        # 清理测试文件
        import shutil
        shutil.rmtree(test_image_dir)
        shutil.rmtree(test_mask_dir)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 测试数据集类型检测
    print("\nTesting dataset type detection...")
    test_paths = [
        "/home/yuwenjing/data/tta_dataset/TTA-2DCXR/Shenzhen_/image",
        "/home/yuwenjing/data/tta_dataset/TTA-2Ddermoscopy/ISIC_/image",
        "/home/yuwenjing/data/tta_dataset/TTA-2DPATH/path_/image",
        "/home/yuwenjing/data/tta_dataset/TTA-2DUS/us_/image",
        "/some/unknown/path"
    ]
    
    for path in test_paths:
        detected_type = get_dataset_type_from_path(path)
        print(f"Path: {path} -> Type: {detected_type}")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()


