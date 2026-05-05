import os
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import glob
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld, RandRotate90d, RandFlipd, RandGaussianNoised, RandAdjustContrastd, RandShiftIntensityd, ToTensord, Resized, NormalizeIntensityd

class CTADataset3D(Dataset):

    def __init__(self, image_dir: str, mask_dir: str, phase: str='train', image_size: Tuple[int, int, int]=(128, 128, 128), spacing: Tuple[float, float, float]=(1.0, 1.0, 1.0), window_level: float=1024.0, window_width: float=4095.0, normalize: bool=True, if_flt: bool=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.phase = phase
        self.image_size = image_size
        self.spacing = spacing
        self.window_level = float(window_level)
        self.window_width = float(window_width)
        self.normalize = normalize
        self.if_flt = if_flt
        self.supported_extensions = ['.nii.gz', '.nii', '.mha', '.mhd']
        if not os.path.exists(image_dir):
            raise ValueError(f'Image directory does not exist: {image_dir}')
        if not os.path.exists(mask_dir):
            raise ValueError(f'Mask directory does not exist: {mask_dir}')
        self.data_dicts = self._collect_data_pairs()
        if len(self.data_dicts) == 0:
            raise ValueError('No valid image-mask pairs found!')
        self.a_min = self.window_level - self.window_width / 2.0
        self.a_max = self.window_level + self.window_width / 2.0
        self.transforms = self._get_transforms()

    def _collect_data_pairs(self) -> List[Dict[str, str]]:
        data_dicts: List[Dict[str, str]] = []
        image_files: List[str] = []
        for ext in self.supported_extensions:
            image_files.extend(glob.glob(os.path.join(self.image_dir, f'*_image{ext}')))
        image_files.sort()
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            label_name = img_name.replace('_image', '_label')
            mask_path = os.path.join(self.mask_dir, label_name)
            if os.path.exists(mask_path) and self._is_valid_file(img_path) and self._is_valid_file(mask_path):
                data_dicts.append({'image': img_path, 'label': mask_path, 'image_name': img_name})
            else:
                print(f'[Warning] Skip invalid or missing pair: {img_name} -> {label_name}')
        print(f'Found {len(data_dicts)} valid CTA image-label pairs')
        return data_dicts

    def _is_valid_file(self, file_path: str) -> bool:
        try:
            if file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
                _ = nib.load(file_path).get_fdata()
            return True
        except Exception:
            return False

    def _get_transforms(self):
        common = [LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']), Orientationd(keys=['image', 'label'], axcodes='RAS'), Spacingd(keys=['image', 'label'], pixdim=self.spacing, mode=('bilinear', 'nearest')), ScaleIntensityRanged(keys=['image'], a_min=self.a_min, a_max=self.a_max, b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=['image', 'label'], source_key='image'), Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=('trilinear', 'nearest'))]
        if self.normalize:
            common.append(NormalizeIntensityd(keys=['image'], nonzero=True))
        if self.phase == 'train':
            aug = [RandRotate90d(keys=['image', 'label'], prob=0.3, max_k=3), RandFlipd(keys=['image', 'label'], spatial_axis=[0], prob=0.3), RandFlipd(keys=['image', 'label'], spatial_axis=[1], prob=0.3), RandFlipd(keys=['image', 'label'], spatial_axis=[2], prob=0.3), RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.1), RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.8, 1.2)), RandShiftIntensityd(keys=['image'], prob=0.3, offsets=0.1)]
        else:
            aug = []
        tail = [ToTensord(keys=['image', 'label'])]
        return Compose(common + aug + tail)

    def __len__(self) -> int:
        return len(self.data_dicts)

    def __getitem__(self, idx: int):
        data_dict = self.data_dicts[idx].copy()
        try:
            data_dict = self.transforms(data_dict)
        except Exception as e:
            print(f'Error applying transforms to {data_dict.get('image_name', 'unknown')}: {e}')
            raise e
        if isinstance(data_dict, list):
            data_dict = data_dict[0]
        image = data_dict['image']
        label = data_dict['label']
        filename = data_dict['image_name']
        image = image.float() if isinstance(image, torch.Tensor) else torch.tensor(image, dtype=torch.float32)
        label = label.long() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)
        if label.ndim == 4 and label.shape[0] == 1:
            label = label.squeeze(0)
        if not self.if_flt:
            label = label.clone()
            label[label == 3] = 0
            label[label == 4] = 0
        label = torch.clamp(label, 0, 2)
        return (image, label, filename)

def get_cta_data_loaders(image_dir: str, mask_dir: str, batch_size_train: int=2, batch_size_val: int=2, num_workers: int=4, train_split: float=0.8, image_size: Tuple[int, int, int]=(128, 128, 128), spacing: Tuple[float, float, float]=(1.0, 1.0, 1.0), window_level: float=1024.0, window_width: float=4095.0, normalize: bool=True, if_flt: bool=True):
    print(f'\n{'=' * 60}')
    print('Initializing CTA Dataset...')
    print(f'{'=' * 60}')
    temp_dataset = CTADataset3D(image_dir=image_dir, mask_dir=mask_dir, phase='train', image_size=image_size, spacing=spacing, window_level=window_level, window_width=window_width, normalize=normalize, if_flt=if_flt)
    total_size = len(temp_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_dicts = [temp_dataset.data_dicts[i] for i in train_indices]
    val_dicts = [temp_dataset.data_dicts[i] for i in val_indices]
    print(f'\n{'=' * 60}')
    print(f'Dataset Split Summary:')
    print(f'  Total samples: {total_size}')
    print(f'  Training samples: {train_size} ({train_split * 100:.1f}%)')
    print(f'  Validation samples: {val_size} ({(1 - train_split) * 100:.1f}%)')
    print(f'{'=' * 60}\n')
    train_dataset = CTADataset3D.__new__(CTADataset3D)
    train_dataset.image_dir = image_dir
    train_dataset.mask_dir = mask_dir
    train_dataset.phase = 'train'
    train_dataset.image_size = image_size
    train_dataset.spacing = spacing
    train_dataset.window_level = window_level
    train_dataset.window_width = window_width
    train_dataset.normalize = normalize
    train_dataset.if_flt = if_flt
    train_dataset.supported_extensions = temp_dataset.supported_extensions
    train_dataset.data_dicts = train_dicts
    train_dataset.a_min = temp_dataset.a_min
    train_dataset.a_max = temp_dataset.a_max
    train_dataset.transforms = temp_dataset.transforms
    val_dataset = CTADataset3D.__new__(CTADataset3D)
    val_dataset.image_dir = image_dir
    val_dataset.mask_dir = mask_dir
    val_dataset.phase = 'val'
    val_dataset.image_size = image_size
    val_dataset.spacing = spacing
    val_dataset.window_level = window_level
    val_dataset.window_width = window_width
    val_dataset.normalize = normalize
    val_dataset.if_flt = if_flt
    val_dataset.supported_extensions = temp_dataset.supported_extensions
    val_dataset.data_dicts = val_dicts
    val_dataset.a_min = temp_dataset.a_min
    val_dataset.a_max = temp_dataset.a_max
    val_dataset.transforms = val_dataset._get_transforms()
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    print(f'DataLoaders created:')
    print(f'  Train batches: {len(train_loader)} (batch_size={batch_size_train})')
    print(f'  Val batches: {len(val_loader)} (batch_size={batch_size_val})')
    print(f'{'=' * 60}\n')
    return (train_loader, val_loader, 'CTA')

def main():
    base = '/home/yuwenjing/data/imageTBAD'
    image_dir = base
    mask_dir = base
    print('Testing CTA dataset with WL=1024, WW=4095 ...')
    try:
        dataset = CTADataset3D(image_dir=image_dir, mask_dir=mask_dir, phase='train', image_size=(96, 96, 96), spacing=(1.0, 1.0, 1.0), window_level=1024.0, window_width=4095.0, normalize=True, if_flt=False)
        print(f'✓ Dataset created. Samples: {len(dataset)}')
        if len(dataset) > 0:
            img, msk, fn = dataset[0]
            print(f'✓ Sample loaded: {fn}')
            print(f'  Image shape: {img.shape}, range: [{img.min():.4f}, {img.max():.4f}]')
            print(f'  Mask shape: {msk.shape}, unique: {torch.unique(msk)}')
        print('\n✓ Building loaders ...')
        tl, vl, dtype = get_cta_data_loaders(image_dir=image_dir, mask_dir=mask_dir, batch_size_train=1, batch_size_val=1, num_workers=0, train_split=0.8, image_size=(96, 96, 96), window_level=1024.0, window_width=4095.0, normalize=True, if_flt=False)
        print(f'  Detected type: {dtype}')
        print(f'  Train batches: {len(tl)} | Val batches: {len(vl)}')
        for b, (images, masks, names) in enumerate(tl):
            print(f'  Train batch {b}: images {images.shape}, masks {masks.shape}')
            print(f'  Filenames: {names}')
            break
        for b, (images, masks, names) in enumerate(vl):
            print(f'  Val batch {b}: images {images.shape}, masks {masks.shape}')
            print(f'  Filenames: {names}')
            break
        print('\n🎉 CTA dataset test passed!')
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
    print('\nTest completed!')
if __name__ == '__main__':
    main()
