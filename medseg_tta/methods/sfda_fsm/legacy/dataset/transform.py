import torch
import torchvision.transforms.functional as F
import scipy
import random
from PIL import Image
import numpy as np
import cv2
from skimage import transform as tf
import numbers

class ToTensor(object):

    def __call__(self, data):
        image, label = (data['image'], data['label'])
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}

class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = (data['image'], data['label'])
        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=Image.NEAREST)}

class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = (data['image'], data['label'])
        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}
        return {'image': image, 'label': label}

class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = (data['image'], data['label'])
        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}
        return {'image': image, 'label': label}

class RandomRotation(object):

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number, it must be positive.')
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence, it must be of len 2.')
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, data):
        image, label = (data['image'], data['label'])
        if random.random() < 0.5:
            angle = self.get_params(self.degrees)
            return {'image': F.rotate(image, angle, self.resample, self.expand, self.center), 'label': F.rotate(label, angle, self.resample, self.expand, self.center)}
        return {'image': image, 'label': label}

class RandomZoom(object):

    def __init__(self, zoom=(0.8, 1.2)):
        self.min, self.max = (zoom[0], zoom[1])

    def __call__(self, data):
        image, label = (data['image'], data['label'])
        if random.random() < 0.5:
            image = np.array(image)
            label = np.array(label)
            zoom = random.uniform(self.min, self.max)
            zoom_image = clipped_zoom(image, zoom)
            zoom_label = clipped_zoom(label, zoom)
            zoom_image = Image.fromarray(zoom_image.astype('uint8'), 'RGB')
            zoom_label = Image.fromarray(zoom_label.astype('uint8'), 'L')
            return {'image': zoom_image, 'label': zoom_label}
        return {'image': image, 'label': label}

def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    if zoom_factor < 1:
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = scipy.ndimage.zoom(img, zoom_tuple, **kwargs)
    elif zoom_factor > 1:
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        zoom_in = scipy.ndimage.zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
        if zoom_in.shape[0] >= h:
            zoom_top = (zoom_in.shape[0] - h) // 2
            sh = h
            out_top = 0
            oh = h
        else:
            zoom_top = 0
            sh = zoom_in.shape[0]
            out_top = (h - zoom_in.shape[0]) // 2
            oh = zoom_in.shape[0]
        if zoom_in.shape[1] >= w:
            zoom_left = (zoom_in.shape[1] - w) // 2
            sw = w
            out_left = 0
            ow = w
        else:
            zoom_left = 0
            sw = zoom_in.shape[1]
            out_left = (w - zoom_in.shape[1]) // 2
            ow = zoom_in.shape[1]
        out = np.zeros_like(img)
        out[out_top:out_top + oh, out_left:out_left + ow] = zoom_in[zoom_top:zoom_top + sh, zoom_left:zoom_left + sw]
    else:
        out = img
    return out

class Translation(object):

    def __init__(self, translation):
        self.translation = translation

    def __call__(self, data):
        image, label = (data['image'], data['label'])
        if random.random() < 0.5:
            image = np.array(image)
            label = np.array(label)
            rows, cols, _ = image.shape
            translation = random.uniform(0, self.translation)
            tr_x = translation / 2
            tr_y = translation / 2
            Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
            translate_image = cv2.warpAffine(image, Trans_M, (cols, rows))
            translate_label = cv2.warpAffine(label, Trans_M, (cols, rows))
            translate_image = Image.fromarray(translate_image.astype('uint8'), 'RGB')
            translate_label = Image.fromarray(translate_label.astype('uint8'), 'L')
            return {'image': translate_image, 'label': translate_label}
        return {'image': image, 'label': label}

class RandomCrop(object):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return (0, 0, h, w)
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return (i, j, th, tw)

    def __call__(self, data):
        img, label = (data['image'], data['label'])
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        label = F.crop(label, i, j, h, w)
        return {'image': img, 'label': label}

class Normalization(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = (sample['image'], sample['label'])
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}
