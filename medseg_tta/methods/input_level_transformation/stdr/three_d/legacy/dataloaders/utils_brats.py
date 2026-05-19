##load data
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import yaml
from torch.utils.data import DataLoader
import random
import configparser
from torch.nn import init
from scipy import ndimage
#import SimpleITK as sitk 
import torch.nn as nn
#import qtlib

import monai.transforms as transforms
import nibabel as nib
from monai.transforms.transform import MapTransform
from os.path import join
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
def resize_and_normalize(image, label, target_shape, num_class):
    """
    动态调整图像和标签的大小，并对图像进行归一化。

    参数:
    - image: 输入图像，形状为 [D, H, W]，每个像素是灰度值。
    - label: 输入标签，形状为 [D, H, W]，每个像素是分类标签。
    - target_shape: 目标形状，一个三元组 (target_d, target_h, target_w)。
    - num_class: 分类数。

    返回:
    - resized_image: 调整大小后的图像。
    - resized_label: 调整大小后的标签。
    """
    # 确保输入是 NumPy 数组
    image = np.asarray(image, dtype=np.float32)
    label = np.asarray(label, dtype=np.int64)

    # 获取输入图像的形状
    d, h, w = image.shape
    target_d, target_h, target_w = target_shape

    # 裁剪或填充图像和标签
    def crop_or_pad(volume, target_size):
        c, D, H, W = volume.shape
        target_d, target_h, target_w = target_size

        # 实际裁剪尺寸
        crop_d = min(target_d, D)
        crop_h = min(target_h, H)
        crop_w = min(target_w, W)

        # 随机起始位置
        sx = (D - crop_d) // 2
        sy = (H - crop_h) // 2
        sz = (W - crop_w) // 2

        # 执行裁剪
        volume_crop = volume[:, sx:sx + crop_d, sy:sy + crop_h, sz:sz + crop_w]

        # 填充到目标尺寸
        if any([crop_d < target_d, crop_h < target_h, crop_w < target_w]):
            pad_d = max(0, target_d - crop_d)
            pad_h = max(0, target_h - crop_h)
            pad_w = max(0, target_w - crop_w)
            volume_crop = np.pad(volume_crop, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

        return volume_crop

    # 调整图像大小
    image = image[np.newaxis, :, :, :]  # 添加通道维度 [1, D, H, W]
    resized_image = crop_or_pad(image, target_shape)
    resized_image = resized_image.squeeze(0)  # 去掉通道维度 [D, H, W]

    # 调整标签大小
    label = label[np.newaxis, :, :, :]  # 添加通道维度 [1, D, H, W]
    resized_label = crop_or_pad(label, target_shape)
    resized_label = resized_label.squeeze(0)  # 去掉通道维度 [D, H, W]

    # 归一化图像
    def normalize(volume):
        non_zero_mask = volume > 0
        if non_zero_mask.any():
            mean = volume[non_zero_mask].mean()
            std = volume[non_zero_mask].std()
            volume = (volume - mean) / (std + 1e-8)
        return volume

    resized_image = normalize(resized_image)

    # 转换为 PyTorch 张量
    resized_image = torch.as_tensor(resized_image, dtype=torch.float32)
    resized_label = torch.as_tensor(resized_label, dtype=torch.int64)

    return resized_image, resized_label
def is_int(val_str):
    start_digit = 0
    if(val_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if(str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(val_str):
    flag = False
    if('.' in val_str and len(val_str.split('.'))==2 and not('./' in val_str)):
        if(is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in val_str and val_str[0] != 'e' and len(val_str.split('e'))==2):
        if(is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False       
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str.lower() =='true' or var_str.lower() == 'false'):
        return True
    else:
        return False
    
def parse_bool(var_str):
    if(var_str.lower() =='true'):
        return True
    else:
        return False
     
def is_list(val_str):
    if(val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False

def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        elif(item.lower() == 'none'):
            output.append(None)
        else:
            output.append(item)
    return output
    
def parse_value_from_string(val_str):
    if(is_int(val_str)):
        val = int(val_str)
    elif(is_float(val_str)):
        val = float(val_str)
    elif(is_list(val_str)):
        val = parse_list(val_str)
    elif(is_bool(val_str)):
        val = parse_bool(val_str)
    elif(val_str.lower() == 'none'):
        val = None
    else:
        val = val_str
    return val

def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if(len(val_str)>0):
                val = parse_value_from_string(val_str)
                output[section][key] = val
            else:
                val = None
            print(section, key, val_str, val)
    return output


def load_npz(path):
    img = np.load(path)['arr_0']
    gt = np.load(path)['arr_1']
    return img, gt
    
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream,Loader=yaml.FullLoader)

def set_random(seed_id=1234):
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)   #for cpu
    torch.cuda.manual_seed_all(seed_id) #for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# config setting
def is_int(val_str):
    start_digit = 0
    if(val_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if(str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(val_str):
    flag = False
    if('.' in val_str and len(val_str.split('.'))==2 and not('./' in val_str)):
        if(is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in val_str and val_str[0] != 'e' and len(val_str.split('e'))==2):
        if(is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False       
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str.lower() =='true' or var_str.lower() == 'false'):
        return True
    else:
        return False
    
def parse_bool(var_str):
    if(var_str.lower() =='true'):
        return True
    else:
        return False
     
def is_list(val_str):
    if(val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False

def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        elif(item.lower() == 'none'):
            output.append(None)
        else:
            output.append(item)
    return output

def parse_value_from_string(val_str):
    if(is_int(val_str)):
        val = int(val_str)
    elif(is_float(val_str)):
        val = float(val_str)
    elif(is_list(val_str)):
        val = parse_list(val_str)
    elif(is_bool(val_str)):
        val = parse_bool(val_str)
    elif(val_str.lower() == 'none'):
        val = None
    else:
        val = val_str
    return val

def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if(len(val_str)>0):
                val = parse_value_from_string(val_str)
                output[section][key] = val
            else:
                val = None
            print(section, key, val_str, val)
    return output

class UnpairedDataset(Dataset):
    #get unpaired dataset, such as MR-CT dataset
    def __init__(self,A_path,B_path):
        listA = os.listdir(A_path)
        listB = os.listdir(B_path)
        self.listA = [os.path.join(A_path,k) for k in listA]
        self.listB = [os.path.join(B_path,k) for k in listB]
        self.Asize = len(self.listA)
        self.Bsize = len(self.listB)
        self.dataset_size = max(self.Asize,self.Bsize)
        
    def __getitem__(self,index):
        if self.Asize == self.dataset_size:
            A,A_gt = load_npz(self.listA[index])
            B,B_gt = load_npz(self.listB[random.randint(0, self.Bsize - 1)])
        else :
            B,B_gt = load_npz(self.listB[index])
            A,A_gt = load_npz(self.listA[random.randint(0, self.Asize - 1)])


        A = torch.from_numpy(A.copy()).unsqueeze(0).float()
        A_gt = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()
        B = torch.from_numpy(B.copy()).unsqueeze(0).float()
        B_gt = torch.from_numpy(B_gt.copy()).unsqueeze(0).float()
        return A,A_gt,B,B_gt
        
    def __len__(self):
        return self.dataset_size

def crop_depth(img,lab,phase = 'train'):
    D,H,W = img.shape
    if D > 10:
        if phase == 'train':
            target_ssh = np.random.randint(0, int(D-10), 1)[0]
            zero_img = img[target_ssh:target_ssh+10,:,:]
            zero_lab = lab[target_ssh:target_ssh+10,:,:]
        elif phase == 'valid':
            zero_img,zero_lab = img,lab
        elif phase == 'feta':
            sample_indices = np.random.choice(D, size=10, replace=False)
            zero_img = np.zeros((10,H,W))
            zero_lab = np.zeros((10,H,W))
            for i, index in enumerate(sample_indices):
                zero_img[i] = img[index]
                zero_lab[i] = lab[index]
    else:
        zero_img = np.zeros((10,H,W))
        zero_lab = np.zeros((10,H,W))
        zero_img[0:D,:,:] = img
        zero_lab[0:D,:,:] = lab
    return zero_img,zero_lab

def winadj_mri(array):
    v0 = np.percentile(array, 1)
    v1 = np.percentile(array, 99)
    array[array < v0] = v0    
    array[array > v1] = v1  
    v0 = array.min() 
    v1 = array.max() 
    array = (array - v0) / (v1 - v0) * 2.0 - 1.0
    return array

def resize(img,lab):
    D,H,W = img.shape
    zoom = [64/D,64/H,64/W]
    img=ndimage.zoom(img,zoom,order=2)
    lab=ndimage.zoom(lab,zoom,order=0)
    return img,lab
####################################################################################################
# transforms
def nib_load(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError
    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data
class RobustZScoreNormalization(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key] > 0

            lower = np.percentile(d[key][mask], 0.2)
            upper = np.percentile(d[key][mask], 99.8)

            d[key][mask & (d[key] < lower)] = lower
            d[key][mask & (d[key] > upper)] = upper

            y = d[key][mask]
            d[key] -= y.mean()
            d[key] /= y.std()+0.01

        return d

def get_brats2021_base_transform():
    base_transform = [

        #transforms.EnsureChannelFirstd(keys=['flair', 't1', 't1ce', 't2', 'label'],channel_dim=-1),      
        #transforms.Orientationd(keys=[, 'label'], axcodes="RAS"),  #将图像的方向统一为指定的坐标系（RAS：Right-Anterior-Superior）
        
        #RobustZScoreNormalization(keys=['image']),#对图像进行鲁棒的 Z 分数归一化（减去均值，除以标准差）
        transforms.SpatialPadD(keys=["image", "label"], spatial_size=(218,218,218), method='symmetric', mode='constant'),
        transforms.Resized(keys=['label'], spatial_size=(128,128,128),mode='nearest'),
        transforms.Resized(keys=['image'], spatial_size=(128,128,128),mode='trilinear'),

        transforms.NormalizeIntensityd(keys=['image'], nonzero=True, dtype=np.float32)


 

        #transforms.ConcatItemsd(keys=['image','image','image','image'], name='image', dim=0),#将多个模态的图像合并为一个多通道图像。
        #transforms.DeleteItemsd(keys=['image','image','image','image'])#删除单个模态的图像，因为它们已经被合并到多通道图像中。
        
    ]
    return base_transform

def get_brats2021_train_target_transform():
    base_transform = get_brats2021_base_transform()
    data_aug = [

        # spatial aug
        #transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=0),
        #transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=1),
        #transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=2),

        # intensity aug
        #随机添加高斯噪声。
        #transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
        #随机对图像进行高斯平滑。
        transforms.RandGaussianSmoothd(
            keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        #随机调整图像对比度。
        transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),

        # other stuff
        #确保数据类型正确。
        transforms.EnsureTyped(keys=["image", 'label']),
    ]
    return transforms.Compose(base_transform + data_aug)


def get_brats2021_train_transform():
    base_transform = get_brats2021_base_transform()
    
    data_aug = [

        # spatial aug
        #transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=0),
        #transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=1),
        #transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=2),

        # intensity aug
        #随机添加高斯噪声。
        #transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
        #随机对图像进行高斯平滑。
        transforms.RandGaussianSmoothd(
            keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        #随机调整图像对比度。
        transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),

        # other stuff
        #确保数据类型正确。
        transforms.EnsureTyped(keys=["image", 'label']),
    ]
    return transforms.Compose(base_transform + data_aug)


def get_brats2021_infer_transform():
    base_transform = get_brats2021_base_transform()
    infer_transform = [transforms.EnsureTyped(keys=["image", 'label'])]
    return transforms.Compose(base_transform + infer_transform)


####################################################################################################
# dataset

class niiDataset(Dataset):
    def __init__(self, image_path,img,phase = 'test'):
        self.img_path = image_path
        self.img = img
        self.phase  =phase
        self.case_names = os.listdir(image_path)
        if phase == 'train':
            self.transforms = get_brats2021_train_transform()
        if phase == 'target':
            self.transforms = get_brats2021_train_target_transform()
        else:
            self.transforms = get_brats2021_infer_transform()

            
    def __getitem__(self, index):
        name = self.case_names[index]
        base_dir = join(self.img_path,name,name)
        imga = np.array(nib_load(base_dir + f'-{self.img}.nii.gz'),dtype='float32')
        # t1c = np.array(nib_load(base_dir + '-t1c.nii.gz'),dtype='float32')
        # t1n = np.array(nib_load(base_dir + '-t1n.nii.gz'),dtype='float32')
        # t2w = np.array(nib_load(base_dir + '-t2w.nii.gz'),dtype='float32')
        # t2f = np.array(nib_load(base_dir + '-t2f.nii.gz'),dtype='float32')
        mask = np.array(nib_load(base_dir + '-seg.nii.gz'),dtype='float32')
        # t1c = np.expand_dims(t1c, axis=0)  # 添加通道维度 [1, H, W, D]
        # t1n = np.expand_dims(t1n, axis=0)
        # t2w = np.expand_dims(t2w, axis=0)
        # t2f = np.expand_dims(t2f, axis=0)
        imga = np.expand_dims(imga, axis=0)
        mask = np.expand_dims(mask, axis=0)
      
        mask[mask==4] = 0
        
        
        if self.phase == 'train' or self.phase == 'target':
            #item = self.transforms({'t1c':t1c, 't1n':t1n, 't2w':t2w, 't2f':t2f, 'label':mask})
            item = self.transforms({'image':imga, 'label':mask})
            item['name'] = name
            
            #item = item[0]
        else:
            #item = self.transforms({'t1c':t1c, 't1n':t1n, 't2w':t2w, 't2f':t2f, 'label':mask})
            item = self.transforms({'image':imga, 'label':mask})
            
        item['idx'] = index
        #return item['image'],item['label'],index,name
        return item
    

           
    def __len__(self):
        return len(self.case_names)


def one_hot_encode(input_tensor):
    if len(input_tensor.shape) == 4:
        a,b,c,d = input_tensor.shape
    elif len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    for i in range(4):
        tmp = (input_tensor==i) * torch.ones_like(input_tensor)
        tensor_list.append(tmp)
    output_tensor = torch.cat(tensor_list,dim=1)
    return output_tensor.float()

def get_largest_component(image):
    dim = len(image.shape)
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label[0], np.uint8)
    return output


def tensor_rot_90(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3).transpose(2, 3)
    else:
	    return x.flip(2).transpose(1, 2)
def tensor_rot_180(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3).flip(2)
    else:
	    return x.flip(2).flip(1)
def tensor_flip_2(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(2)
    else:
	    return x.flip(1)
def tensor_flip_3(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.flip(3)
    else:
	    return x.flip(2)

def tensor_rot_270(x):
    x_shape = list(x.shape)
    if(len(x_shape) == 4):
        return x.transpose(2, 3).flip(3)
    else:
        return x.transpose(1, 2).flip(2)
    
def rotate_single_random(img):
    x_shape = list(img.shape)
    if(len(x_shape) == 5):
        [N, C, D, H, W] = x_shape
        new_shape = [N*D, C, H, W]
        x = torch.transpose(img, 1, 2)
        img = torch.reshape(x, new_shape)
    label = np.random.randint(0, 4, 1)[0]
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    else:
        img = img
    return img,label

def rotate_single_with_label(img, label):
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    else:
        img = img
    return img

def random_rotate(A,A_gt):
    target_ssh = np.random.randint(0, 8, 1)[0]
    A = rotate_single_with_label(A, target_ssh)
    A_gt = rotate_single_with_label(A_gt, target_ssh)
    return A,A_gt

def rotate_4(img):
    # target_ssh = np.random.randint(0, 4, 1)[0]
    A_1 = rotate_single_with_label(img, 1)
    A_2 = rotate_single_with_label(img, 2)
    A_3 = rotate_single_with_label(img, 3)
    return A_1,A_2,A_3



def get_data_loader(source_root,target_root,train_path,test_path,batch_train,batch_test,nw = 4,img='t2f',mode='source_to_source'):
    if mode == 'source_to_source':
        train_img = os.path.join(source_root,train_path)
        test_img = os.path.join(source_root,test_path)
    elif mode == 'source_to_target':
        train_img = os.path.join(source_root,train_path)
        test_img = os.path.join(target_root,'')
    elif mode == 'target_to_target':
        train_img = os.path.join(target_root,'')
        test_img = os.path.join(target_root,'')
    else:
        raise ValueError("mode should be 'source_to_source' or 'source_to_target' or 'target_to_target")


    # 假设 train_img 和 train_lab 是训练集的图像和标签
    train_dataset = niiDataset(train_img,img=img, phase='train')
    test_dataset = niiDataset(test_img,img=img,phase='test')
  

   
    # 创建新的 DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=batch_train, num_workers=nw,shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_test, num_workers=nw,shuffle=False, drop_last=False)
    
    print("数据加载完成")
    print("len of train dataset:" + str(len(train_dataset)))
    print("len of test dataset:" + str(len(test_dataset)))

    return train_loader,test_loader

def count_label_values(label):
    """
    统计label中一个体素能有几种取值。

    参数:
    - label: 输入标签，形状为 [D, H, W] 或 [C, D, H, W]，每个像素是分类标签。

    返回:
    - unique_values: 标签中所有唯一的值。
    - counts: 每个唯一值对应的计数。
    """
    # 确保输入是 NumPy 数组
    label = np.asarray(label)
    
    # 如果标签有通道维度，去掉通道维度
    #if len(label.shape) == 4:
        #label = label.squeeze(0)
    
    # 获取所有唯一的值及其计数
    unique_values, counts = np.unique(label, return_counts=True)
    
    return unique_values, counts

if __name__ == '__main__':
    
    #data_root_brats = '/root/autodl-tmp/BraTS-SSA' # 测试域：/root/autodl-tmp/BraTS-SSA
    batch_train = 2
    batch_test = 1
    num_workers = 0
    source_root = '/root/autodl-tmp/BraTS2024'
    target_root = '/root/autodl-tmp/BraTS-SSA'
    train_path = 'train'
    test_path = 'test'
    mode = 'source_to_source'
    img = 't2f'
    train_loader,test_loader = get_data_loader(source_root,target_root,
                                               train_path,test_path,
                                               batch_train,batch_test,
                                               nw = num_workers,
                                               img=img,mode=mode)
    #get_data_loader(source_root,target_root,train_path,
    #               test_path,batch_train,batch_test,nw = 4,img='t2f',mode='source_to_source')
    print("数据加载完成")
    
    # 获取一个训练样本
    trainimg, train_label, _, _ = next(iter(train_loader))

    
    # 统计label中的取值
    unique_values, counts = count_label_values(train_label.numpy())
    
    print("Label中一个体素能有的取值及其数量：")
    for value, count in zip(unique_values, counts):
        print(f"取值: {value}, 数量: {count}")
