import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import glob
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld, RandRotate90d, RandFlipd, RandGaussianNoised, RandAdjustContrastd, RandShiftIntensityd, ToTensord, Resized, NormalizeIntensityd

def get_dataset_type_from_path(data_path: str) -> str:
    data_path = data_path.replace('\\', '/').lower()
    if 'tta-3dct' in data_path or 'tta-ct' in data_path or 'ct' in data_path:
        return 'CT'
    return 'CT'

def get_dataset_paths(dataset_type: str, base_dir: str='/home/yuwenjing/data/tta_dataset', subfolder: str=None) -> Tuple[str, str]:
    dataset_mapping = {'CT': 'TTA-3DCT'}
    if dataset_type not in dataset_mapping:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')
    dataset_folder = dataset_mapping[dataset_type]
    dataset_path = os.path.join(base_dir, dataset_folder)
    if subfolder is None:
        subfolder = _auto_select_subfolder(dataset_path, dataset_type)
    if subfolder:
        image_dir = os.path.join(dataset_path, subfolder, 'image')
        mask_dir = os.path.join(dataset_path, subfolder, 'mask')
    else:
        image_dir = os.path.join(dataset_path, 'image')
        mask_dir = os.path.join(dataset_path, 'mask')
    return (image_dir, mask_dir)

def _auto_select_subfolder(dataset_path: str, dataset_type: str) -> str:
    if not os.path.exists(dataset_path):
        return 'CT_' if dataset_type == 'CT' else ''
    try:
        subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    except PermissionError:
        return ''
    if not subfolders:
        return ''
    underscore_folders = sorted([f for f in subfolders if f.endswith('_')])
    return underscore_folders[0] if underscore_folders else sorted(subfolders)[0]

class CTDataset3D(Dataset):

    def __init__(self, image_dir: str, mask_dir: str, phase: str='train', image_size: Tuple[int, int, int]=(128, 128, 128), spacing: Tuple[float, float, float]=(1.0, 1.0, 1.0), intensity_range: Tuple[float, float]=(-200, 400), normalize: bool=True, cache_rate: float=0.0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.phase = phase
        self.image_size = image_size
        self.spacing = spacing
        self.intensity_range = intensity_range
        self.normalize = normalize
        self.cache_rate = cache_rate
        self.supported_extensions = ['.nii.gz', '.nii', '.mha', '.mhd']
        if not os.path.exists(image_dir):
            raise ValueError(f'Image directory does not exist: {image_dir}')
        if not os.path.exists(mask_dir):
            raise ValueError(f'Mask directory does not exist: {mask_dir}')
        self.data_dicts = self._collect_data_pairs()
        if len(self.data_dicts) == 0:
            raise ValueError('No valid image-mask pairs found!')
        print(f'Found {len(self.data_dicts)} valid CT image-mask pairs for {phase} phase')
        self.transforms = self._get_transforms()

    def _collect_data_pairs(self) -> List[Dict[str, str]]:
        data_dicts = []
        image_files = []
        for ext in self.supported_extensions:
            pattern = os.path.join(self.image_dir, f'*{ext}')
            image_files.extend(glob.glob(pattern))
        image_files.sort()
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            base_name = self._get_base_name(img_name)
            mask_path = self._find_mask_path(base_name)
            if mask_path and self._is_valid_file(img_path) and self._is_valid_file(mask_path):
                data_dicts.append({'image': img_path, 'label': mask_path, 'image_name': img_name})
            else:
                print(f'[Warning] Skip invalid pair: {img_name}')
        return data_dicts

    def _get_base_name(self, filename: str) -> str:
        for ext in self.supported_extensions:
            if filename.endswith(ext):
                return filename[:-len(ext)]
        return os.path.splitext(filename)[0]

    def _find_mask_path(self, base_name: str) -> Optional[str]:
        if base_name.endswith('-image'):
            liver_base = base_name[:-6] + '-liver_mask'
            patterns = [liver_base]
        else:
            patterns = [base_name, f'{base_name}_seg', f'{base_name}_segmentation', f'{base_name}_mask', f'{base_name}_label', f'{base_name}_gt', f'{base_name}-liver_mask', f'{base_name}-mask']
        for pattern in patterns:
            for ext in self.supported_extensions:
                mask_path = os.path.join(self.mask_dir, f'{pattern}{ext}')
                if os.path.exists(mask_path):
                    return mask_path
        return None

    def _is_valid_file(self, file_path: str) -> bool:
        try:
            if file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
                img = nib.load(file_path)
                _ = img.get_fdata()
            return True
        except Exception:
            return False

    def _get_transforms(self):
        if self.phase == 'train':
            transforms_list = [LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']), Orientationd(keys=['image', 'label'], axcodes='RAS'), Spacingd(keys=['image', 'label'], pixdim=self.spacing, mode=('bilinear', 'nearest')), ScaleIntensityRanged(keys=['image'], a_min=self.intensity_range[0], a_max=self.intensity_range[1], b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=['image', 'label'], source_key='image'), Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=('trilinear', 'nearest')), RandRotate90d(keys=['image', 'label'], prob=0.3, max_k=3), RandFlipd(keys=['image', 'label'], spatial_axis=[0], prob=0.3), RandFlipd(keys=['image', 'label'], spatial_axis=[1], prob=0.3), RandFlipd(keys=['image', 'label'], spatial_axis=[2], prob=0.3), RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.1), RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.8, 1.2)), RandShiftIntensityd(keys=['image'], prob=0.3, offsets=0.1), ToTensord(keys=['image', 'label'])]
            if self.normalize:
                transforms_list.insert(-1, NormalizeIntensityd(keys=['image'], nonzero=True))
            return Compose(transforms_list)
        else:
            transforms_list = [LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']), Orientationd(keys=['image', 'label'], axcodes='RAS'), Spacingd(keys=['image', 'label'], pixdim=self.spacing, mode=('bilinear', 'nearest')), ScaleIntensityRanged(keys=['image'], a_min=self.intensity_range[0], a_max=self.intensity_range[1], b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=['image', 'label'], source_key='image'), Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=('trilinear', 'nearest')), ToTensord(keys=['image', 'label'])]
            if self.normalize:
                transforms_list.insert(-1, NormalizeIntensityd(keys=['image'], nonzero=True))
            return Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.data_dicts)

    def __getitem__(self, idx: int):
        data_dict = self.data_dicts[idx].copy()
        try:
            data_dict = self.transforms(data_dict)
        except Exception as e:
            print(f'Error applying transforms to {data_dict['image_name']}: {e}')
            raise e
        if isinstance(data_dict, list):
            data_dict = data_dict[0]
        image = data_dict['image']
        label = data_dict['label']
        filename = data_dict['image_name']
        if isinstance(image, torch.Tensor):
            image = image.float()
        else:
            image = torch.tensor(image, dtype=torch.float32)
        if isinstance(label, torch.Tensor):
            label = label.long()
        else:
            label = torch.tensor(label, dtype=torch.long)
        if label.ndim == 4 and label.shape[0] == 1:
            label = label.squeeze(0)
        return (image, label, filename)

class DatasetWithPhase(Dataset):

    def __init__(self, subset_dataset, phase):
        self.subset_dataset = subset_dataset
        self.phase = phase
        self.original_dataset = subset_dataset.dataset
        self.original_dataset.phase = phase
        self.original_dataset.transforms = self.original_dataset._get_transforms()

    def __len__(self):
        return len(self.subset_dataset)

    def __getitem__(self, idx):
        return self.subset_dataset[idx]

def get_ct_data_loaders(image_dir: str=None, mask_dir: str=None, dataset_type: str=None, subfolder: str=None, base_dir: str='/home/yuwenjing/data/tta_dataset', batch_size_train: int=2, batch_size_val: int=2, num_workers: int=4, train_split: float=0.8, image_size: Tuple[int, int, int]=(128, 128, 128), spacing: Tuple[float, float, float]=(1.0, 1.0, 1.0), intensity_range: Tuple[float, float]=(-200, 400), cache_rate: float=0.0) -> Tuple[DataLoader, DataLoader, str]:
    if dataset_type is not None:
        image_dir, mask_dir = get_dataset_paths(dataset_type, base_dir, subfolder)
    elif image_dir is not None:
        dataset_type = get_dataset_type_from_path(image_dir)
    else:
        raise ValueError('Either provide image_dir/mask_dir or dataset_type')
    full_dataset = CTDataset3D(image_dir=image_dir, mask_dir=mask_dir, phase='train', image_size=image_size, spacing=spacing, intensity_range=intensity_range, cache_rate=cache_rate)
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_data_dicts = [full_dataset.data_dicts[i] for i in train_indices]
    val_data_dicts = [full_dataset.data_dicts[i] for i in val_indices]
    train_dataset = CTDataset3D(image_dir=image_dir, mask_dir=mask_dir, phase='train', image_size=image_size, spacing=spacing, intensity_range=intensity_range, cache_rate=cache_rate)
    train_dataset.data_dicts = train_data_dicts
    val_dataset = CTDataset3D(image_dir=image_dir, mask_dir=mask_dir, phase='val', image_size=image_size, spacing=spacing, intensity_range=intensity_range, cache_rate=cache_rate)
    val_dataset.data_dicts = val_data_dicts
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return (train_loader, val_loader, dataset_type)

def main():
    TEST_DATASET = 'CT'
    BASE_DIR = '/home/yuwenjing/data/tta_dataset'
    print(f'Testing {TEST_DATASET} Dataset...')
    dataset_info = {'CT': {'full_name': 'TTA-3DCT', 'description': 'CT 3D images'}}
    if TEST_DATASET not in dataset_info:
        print(f'❌ Unsupported dataset type: {TEST_DATASET}')
        print(f'   Supported types: {list(dataset_info.keys())}')
        return
    info = dataset_info[TEST_DATASET]
    print(f'Testing {info['full_name']} - {info['description']}')
    try:
        dataset_base_path = os.path.join(BASE_DIR, info['full_name'])
        if os.path.exists(dataset_base_path):
            print(f'✓ Base path exists: {dataset_base_path}')
            subfolders = [f for f in os.listdir(dataset_base_path) if os.path.isdir(os.path.join(dataset_base_path, f))]
            print(f'Available subfolders: {subfolders}')
            image_dir, mask_dir = get_dataset_paths(dataset_type=TEST_DATASET, base_dir=BASE_DIR, subfolder=None)
            print(f'\nAuto-generated paths:')
            print(f'  Image dir: {image_dir}')
            print(f'  Mask dir: {mask_dir}')
            image_exists = os.path.exists(image_dir)
            mask_exists = os.path.exists(mask_dir)
            print(f'  Image dir exists: {image_exists}')
            print(f'  Mask dir exists: {mask_exists}')
            if image_exists and mask_exists:
                print('\n✓ Both directories exist, testing dataset creation...')
                image_files = []
                mask_files = []
                for ext in ['.nii.gz', '.nii', '.mha', '.mhd']:
                    image_files.extend([f for f in os.listdir(image_dir) if f.endswith(ext)])
                    mask_files.extend([f for f in os.listdir(mask_dir) if f.endswith(ext)])
                print(f'  Found {len(image_files)} image files')
                print(f'  Found {len(mask_files)} mask files')
                if len(image_files) > 0:
                    print(f'  Sample image files: {image_files[:3]}')
                if len(mask_files) > 0:
                    print(f'  Sample mask files: {mask_files[:3]}')
                dataset = CTDataset3D(image_dir=image_dir, mask_dir=mask_dir, phase='train', image_size=(96, 96, 96), spacing=(1.0, 1.0, 1.0), intensity_range=(-200, 400), normalize=True, cache_rate=0.0)
                print(f'\n✓ Dataset created successfully!')
                print(f'  Number of valid samples: {len(dataset)}')
                if len(dataset) > 0:
                    sample_image, sample_mask, sample_filename = dataset[0]
                    print(f'\n✓ Sample loaded: {sample_filename}')
                    print(f'  Image shape: {sample_image.shape}')
                    print(f'  Mask shape: {sample_mask.shape}')
                    print(f'  Image range: [{sample_image.min():.4f}, {sample_image.max():.4f}]')
                    print(f'  Mask unique values: {torch.unique(sample_mask)}')
                    print('\n✓ Testing data loaders...')
                    train_loader, val_loader, detected_type = get_ct_data_loaders(dataset_type=TEST_DATASET, base_dir=BASE_DIR, batch_size_train=1, batch_size_val=1, num_workers=0, train_split=0.8, image_size=(96, 96, 96), cache_rate=0.0)
                    print(f'✓ Data loaders created successfully!')
                    print(f'  Detected type: {detected_type}')
                    print(f'  Train batches: {len(train_loader)}')
                    print(f'  Val batches: {len(val_loader)}')
                    print('\n✓ Testing training batch...')
                    for batch_idx, (images, masks, filenames) in enumerate(train_loader):
                        print(f'  Batch {batch_idx}: images {images.shape}, masks {masks.shape}')
                        print(f'  Filenames: {filenames}')
                        break
                    print('\n✓ Testing validation batch...')
                    for batch_idx, (images, masks, filenames) in enumerate(val_loader):
                        print(f'  Batch {batch_idx}: images {images.shape}, masks {masks.shape}')
                        print(f'  Filenames: {filenames}')
                        break
                    print(f'\n🎉 {TEST_DATASET} dataset test passed!')
                else:
                    print('❌ Dataset is empty!')
            else:
                print("❌ Required directories don't exist:")
                if not image_exists:
                    print(f'  Missing image directory: {image_dir}')
                if not mask_exists:
                    print(f'  Missing mask directory: {mask_dir}')
        else:
            print(f'❌ Base path does not exist: {dataset_base_path}')
            print('Please check if the dataset path is correct.')
    except Exception as e:
        print(f'❌ Error during {TEST_DATASET} dataset testing: {e}')
        import traceback
        traceback.print_exc()
    print('\nTest completed!')
if __name__ == '__main__':
    main()
