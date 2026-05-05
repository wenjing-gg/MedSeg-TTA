import math
import os
import sys
import torch
sys.path.append('/data/storage/lizhihao/workspace')
import numpy as np
import pandas as pd

def dia2mask(diameter, shape):
    [shape_z, shape_y, shape_x] = shape
    [center_z, center_y, center_x] = [int(shape_z / 2), int(shape_y / 2), int(shape_x / 2)]
    z = np.array(list(range(shape_z))).reshape([shape_z, 1, 1])
    y = np.array(list(range(shape_y))).reshape([1, shape_y, 1])
    x = np.array(list(range(shape_x))).reshape([1, 1, shape_x])
    mask = (z - center_z) ** 2 + (y - center_y) ** 2 + (x - center_x) ** 2 <= (diameter / 2) ** 2
    return mask.astype(np.uint8)

def dia2mask_2d(diameter, shape):
    [shape_y, shape_x] = shape
    [center_y, center_x] = [int(shape_y / 2), int(shape_x / 2)]
    y = np.array(list(range(shape_y))).reshape([shape_y, 1])
    x = np.array(list(range(shape_x))).reshape([1, shape_x])
    mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= (diameter / 2) ** 2
    return mask.astype(np.uint8)

def box2mask(center_point, bbox, shape):
    [center_z, center_y, center_x] = center_point
    [bbox_z, bbox_y, bbox_x] = list(bbox)
    [range_z, range_y, range_x] = [int(bbox_z / 2), int(bbox_y / 2), int(bbox_x / 2)]
    mask = np.zeros(shape)
    mask[center_z - range_z:center_z - range_z + bbox_z, center_y - range_y:center_y - range_y + bbox_y, center_x - range_x:center_x - range_x + bbox_x] = 1
    return mask.astype(np.uint8)

def make_ws_label(seg, ws_type):
    if ws_type == 'bbox':
        mapping = {0: [1, 1, 32], 1: [0, 1, 64], 2: [0, 0, 64]}
        boxes = []
        for i in range(3):
            dims = mapping[i]
            length = np.sum(np.max(np.max(seg, dims[0]), dims[1]))
            boxes.append(length)
        return tuple(boxes)
    else:
        area = np.sum(np.max(seg, 0))
        diameter = math.sqrt(area / math.pi) * 2
        return diameter

def add_dia(df_metric, save_path):
    base_path = '/data/storage/lizhihao/data/data/'
    csv_path = '/data/storage/lizhihao/workspace/ULTra/code/docs/files_shuffle_mask64_1.csv'
    df_all = pd.read_csv(csv_path)
    if not 'size' in df_metric.keys():
        df_metric.insert(6, 'size', np.nan)
    for i, row in df_metric.iterrows():
        dataset = row['file_id'].split('_')[0]
        path = base_path + dataset + '/npz_norm_crop64/' + row['file_id'] + '.npz'
        size_code = df_all.loc[df_all['path'] == path]['size']
        df_metric.iloc[i, 6] = size_code.item()
    df_metric.to_csv(save_path, mode='w', header=True, index=False)

def get_saclick(predict, mask, rate, unbiased='True'):
    predict = (predict > 0).float()
    mapping = {0: [1, 1], 1: [0, 1], 2: [0, 0]}
    boxes = []
    sphere = torch.zeros_like(predict)
    predict = predict.squeeze()
    for i in range(3):
        dims = mapping[i]
        length = predict.max(dims[0])[0].max(dims[1])[0].sum(0)
        boxes.append(length)
    boxes = torch.stack(boxes).unsqueeze(0)
    diameter = max(min(int(boxes.min() * rate[0]), int(boxes.min() ** 2 * rate[1])), 1)
    shape = predict.shape
    sphere[0, 0] = torch.from_numpy(dia2mask(diameter, shape))
    if unbiased:
        sphere_real = sphere * mask
        fake_rate = (sphere - sphere_real).sum() / sphere.sum()
        return (sphere_real, fake_rate.item())
    else:
        sphere_real = sphere * mask
        fake_rate = (sphere - sphere_real).sum() / sphere.sum()
        return (sphere, fake_rate.item())

def get_elliptic(predict, mask, rate, unbiased='True'):
    if mask.sum() == 0:
        return (mask, 0)
    predict = (predict > 0).float()
    mapping = {0: [1, 0], 1: [0, 0]}
    boxes = []
    elliptic = torch.zeros_like(predict)
    predict = predict.squeeze()
    for i in range(2):
        dims = mapping[i]
        length = predict.max(1 - i)[0].sum(0)
        boxes.append(length)
    boxes = torch.stack(boxes).unsqueeze(0)
    if rate != 0:
        diameter = max(int(boxes.min() * rate), 1)
    else:
        diameter = 1
    shape = predict.shape
    elliptic[0, 0] = torch.from_numpy(dia2mask_2d(diameter, shape))
    if unbiased:
        sphere_real = elliptic * mask
        fake_rate = (elliptic - sphere_real).sum() / elliptic.sum()
        return (sphere_real, fake_rate.item())
    else:
        sphere_real = elliptic * mask
        fake_rate = (elliptic - sphere_real).sum() / elliptic.sum()
        return (elliptic, fake_rate.item())
