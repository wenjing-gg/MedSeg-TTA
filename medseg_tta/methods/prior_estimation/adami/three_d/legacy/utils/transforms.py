import random
from numpy import dtype
import torchvision.transforms.functional as TF
from einops import repeat, rearrange
from elasticdeform import deform_random_grid
import torch

def random_flip_rotate(img, label, p=0.5, degrees=15):
    if random.random() < p:
        img = TF.hflip(img)
        label = TF.hflip(label)
        
    if random.random() < p:
        angle = random.randint(-degrees, degrees)
        img = TF.rotate(img, angle, fill=0)
        label = TF.rotate(label, angle)

    return img, label


def random_flip_rotate_aug(img, img_aug, label, p=0.5, degrees=15):

    if random.random() < p:
        img = TF.hflip(img)
        label = TF.hflip(label)
        img_aug = TF.hflip(img_aug)  

    if random.random() < p:
        angle = random.randint(-degrees, degrees)
        img = TF.rotate(img, angle, fill=0)
        label = TF.rotate(label, angle)
        img_aug = TF.rotate(img_aug, angle, fill=0)

    return img, img_aug, label



def random_flip_rotate_elastic_deform(img, label, num_classes=7, p=0.5, degrees=15):
    
    if random.random() < p:
        img, label = deform_random_grid([img, label], sigma=1, points=3, axis=(1, 2))
        label[label>(num_classes-1)] = num_classes-1
        label[label<0] = 0

    img, label = to_tensor(img, label)

    if random.random() < p:
        img = TF.hflip(img)
        label = TF.hflip(label)
        
    if random.random() < p:
        angle = random.randint(-degrees, degrees)
        img = TF.rotate(img, angle, fill=-1)
        label = TF.rotate(label, angle)

    return img, label    


def img_to_tensor(img):
    img = torch.from_numpy(img.copy())
    img = repeat(img, '1 h w -> 3 h w')
    return img.to(dtype=torch.float32).div(255)

def label_to_tensor(label):
    label = torch.from_numpy(label.copy())
    return label.to(dtype=torch.long)

def to_tensor(img, label):
    img = img_to_tensor(img)
    label = label_to_tensor(label)
    return img, label

def normalize(img):
    img = TF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return img
    
def denormalize(img):
    img = img * 0.5 + 0.5
    return img
