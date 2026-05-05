import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple

def get_dataset_type_from_path(data_path: str) -> str:
    data_path = data_path.replace('\\', '/').lower()
    if 'tta-2dcxr' in data_path:
        return 'CXR'
    elif 'tta-2ddermoscopy' in data_path:
        return 'dermoscopy'
    elif 'tta-2dpath' in data_path:
        return 'PATH'
    elif 'tta-2dus' in data_path:
        return 'US'
    elif 'tta-2doct':
        return 'OCT'

def get_dataset_paths(dataset_type: str, base_dir: str='E:\\tta_dataset', subfolder: str=None) -> Tuple[str, str]:
    dataset_mapping = {'CXR': 'TTA-2DCXR', 'dermoscopy': 'TTA-2Ddermoscopy', 'PATH': 'TTA-2DPATH', 'US': 'TTA-2DUS', 'OCT': 'TTA-2DOCT'}
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
        return 'Shenzhen_' if dataset_type == 'CXR' else ''
    try:
        subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    except PermissionError:
        return ''
    if not subfolders:
        return ''
    if dataset_type == 'PATH':
        processed_folders = [f for f in subfolders if 'processed' in f.lower()]
        if processed_folders:
            return sorted(processed_folders)[0]
    underscore_folders = sorted([f for f in subfolders if f.endswith('_')])
    if underscore_folders:
        return underscore_folders[0]
    return sorted(subfolders)[0]

class MedicalImageDataset2D(Dataset):

    def __init__(self, image_dir: str, mask_dir: str, phase: str='train', image_size: Tuple[int, int]=(256, 256), normalize: bool=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.phase = phase
        self.image_size = image_size
        self.normalize = normalize
        self.dataset_type = get_dataset_type_from_path(image_dir)
        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        if not os.path.exists(image_dir):
            raise ValueError(f'Image directory does not exist: {image_dir}')
        self.image_files = []
        for ext in self.supported_extensions:
            self.image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        self.image_files.sort()
        if len(self.image_files) == 0:
            raise ValueError(f'No image files found in {image_dir}')
        self.valid_files = []
        matched_pairs = []
        for img_file in self.image_files:
            image_path = os.path.join(self.image_dir, img_file)
            if not self._is_valid_image(image_path):
                print(f'[Warning] Skip corrupted image: {img_file}')
                continue
            try:
                mask_path = self._find_mask_path(img_file)
            except ValueError:
                continue
            if not self._is_valid_image(mask_path):
                print(f'[Warning] Skip corrupted mask: {os.path.basename(mask_path)}')
                continue
            self.valid_files.append(img_file)
            matched_pairs.append((img_file, os.path.basename(mask_path)))
        if len(self.valid_files) == 0:
            raise ValueError('No valid image-mask pairs after filtering corrupt files!')
        print(f'Found {len(matched_pairs)} image-mask pairs (after filtering).')
        for img, msk in matched_pairs[:5]:
            print(f'  {img} -> {msk}')
        if len(matched_pairs) > 5:
            print(f'  ... and {len(matched_pairs) - 5} more')
        self.transforms = self._get_transforms()

    @staticmethod
    def _is_valid_image(path: str) -> bool:
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception:
            return False

    def _get_transforms(self):
        if self.phase == 'train':
            return transforms.Compose([transforms.Resize(self.image_size), transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10), transforms.ColorJitter(0.2, 0.2), transforms.ToTensor()])
        else:
            return transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor()])

    def _load_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert('L')
            return np.asarray(image)
        except Exception as e:
            raise ValueError(f'Cannot load image: {image_path}, Error: {e}')

    def _load_mask(self, mask_path: str) -> np.ndarray:
        try:
            mask = Image.open(mask_path).convert('L')
            mask = np.asarray(mask)
            mask = (mask > 127).astype(np.uint8)
            return mask
        except Exception as e:
            raise ValueError(f'Cannot load mask: {mask_path}, Error: {e}')

    def _find_mask_path(self, img_file: str) -> str:
        base_name, _ = os.path.splitext(img_file)
        candidates = [img_file, *[f'{base_name}{ext}' for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')], *[f'{base_name}-1{ext}' for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')], *[f'{base_name}_segmentation{ext}' for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]]
        for mask_file in candidates:
            mask_path = os.path.join(self.mask_dir, mask_file)
            if os.path.exists(mask_path):
                return mask_path
        all_masks = [f for f in os.listdir(self.mask_dir) if f.lower().endswith(tuple(self.supported_extensions))]
        for mf in all_masks:
            if base_name in mf:
                return os.path.join(self.mask_dir, mf)
        raise ValueError(f'Cannot find mask for {img_file}')

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int):
        filename = self.valid_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        mask_path = self._find_mask_path(filename)
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        if self.phase == 'train':
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            np.random.seed(seed)
            image_tensor = self.transforms(image_pil)
            torch.manual_seed(seed)
            np.random.seed(seed)
            mask_tensor = transforms.Compose([transforms.Resize(self.image_size, transforms.InterpolationMode.NEAREST), transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10, transforms.InterpolationMode.NEAREST), transforms.ToTensor()])(mask_pil)
        else:
            image_tensor = self.transforms(image_pil)
            mask_tensor = transforms.Compose([transforms.Resize(self.image_size, transforms.InterpolationMode.NEAREST), transforms.ToTensor()])(mask_pil)
        mask_tensor = (mask_tensor > 0.5).long().squeeze(0)
        if self.normalize:
            image_tensor = (image_tensor - image_tensor.mean()) / (image_tensor.std() + 1e-08)
        return (image_tensor, mask_tensor, filename)

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

def get_data_loaders_2d(image_dir: str=None, mask_dir: str=None, dataset_type: str=None, subfolder: str=None, base_dir: str='E:\\tta_dataset', batch_size_train: int=8, batch_size_val: int=4, num_workers: int=4, train_split: float=0.9, image_size: Tuple[int, int]=(256, 256)) -> Tuple[DataLoader, DataLoader, str]:
    if dataset_type is not None:
        image_dir, mask_dir = get_dataset_paths(dataset_type, base_dir, subfolder)
    elif image_dir is not None:
        dataset_type = get_dataset_type_from_path(image_dir)
    else:
        raise ValueError('Either provide image_dir/mask_dir or dataset_type')
    full_dataset = MedicalImageDataset2D(image_dir=image_dir, mask_dir=mask_dir, phase='train', image_size=image_size)
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    train_indices, val_indices = torch.utils.data.random_split(range(total_size), [train_size, val_size])
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices.indices)
    train_dataset = DatasetWithPhase(train_dataset, 'train')
    val_dataset = DatasetWithPhase(val_dataset, 'val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return (train_loader, val_loader, dataset_type)

def main():
    TEST_DATASET = 'dermoscopy'
    BASE_DIR = '/home/yuwenjing/data/tta_dataset'
    print(f'Testing {TEST_DATASET} Dataset...')
    dataset_info = {'CXR': {'full_name': 'TTA-2DCXR', 'description': 'Chest X-Ray images'}, 'dermoscopy': {'full_name': 'TTA-2Ddermoscopy', 'description': 'Dermoscopy images'}, 'PATH': {'full_name': 'TTA-2DPATH', 'description': 'Pathology images'}, 'US': {'full_name': 'TTA-2DUS', 'description': 'Ultrasound images'}, 'OCT': {'full_name': 'TTA-2DOCT', 'description': 'Optical Coherence Tomography images'}}
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
                image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
                mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
                print(f'  Found {len(image_files)} image files')
                print(f'  Found {len(mask_files)} mask files')
                if len(image_files) > 0:
                    print(f'  Sample image files: {image_files[:3]}')
                if len(mask_files) > 0:
                    print(f'  Sample mask files: {mask_files[:3]}')
                dataset = MedicalImageDataset2D(image_dir=image_dir, mask_dir=mask_dir, phase='train', image_size=(256, 256), normalize=True)
                print(f'\n✓ Dataset created successfully!')
                print(f'  Dataset type: {dataset.dataset_type}')
                print(f'  Number of valid samples: {len(dataset)}')
                if len(dataset) > 0:
                    sample_image, sample_mask, sample_filename = dataset[0]
                    print(f'\n✓ Sample loaded: {sample_filename}')
                    print(f'  Image shape: {sample_image.shape}')
                    print(f'  Mask shape: {sample_mask.shape}')
                    print(f'  Image range: [{sample_image.min():.4f}, {sample_image.max():.4f}]')
                    print(f'  Mask unique values: {torch.unique(sample_mask)}')
                    print('\n✓ Testing data loaders...')
                    train_loader, val_loader, detected_type = get_data_loaders_2d(dataset_type=TEST_DATASET, base_dir=BASE_DIR, batch_size_train=2, batch_size_val=2, num_workers=0, train_split=0.8, image_size=(256, 256))
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
