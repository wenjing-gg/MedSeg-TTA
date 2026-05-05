import os
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch import optim
from datetime import datetime
from utils.logger import logger
from utils.metrics import *
from utils.utils import get_saclick
from utils.tent import *
from models.unet import UNet
from data.dataset_test import SaDataset
from data import transforms as aug
from configs import config as cfg

def _set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _load_model(model_path):
    model = UNet(num_classes=1)
    model = model.cuda()
    model = nn.DataParallel(model)
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    return model

def _init_dataloaders(dataset=[1, 2, 3], test_set=(0, 1)):
    file_csv = cfg.csv_path
    transforms_test = [aug.Window(cfg.win_min, cfg.win_max), aug.MinMaxNormalize(cfg.win_min, cfg.win_max), aug.LabelTrans(), aug.ClickTrans()]
    ds_test = SaDataset(file_csv, transforms_test, test_set, dataset, 0)
    dl_test = SaDataset.get_dataloader(ds_test, cfg.batch_size_test, False, cfg.num_workers)
    return dl_test

def _init_loss_dict(n):
    loss_dict = {'dice_loss': np.zeros(0), 'bce_loss': np.zeros(0), 'total_loss': np.zeros(n)}
    return loss_dict

def _init_metric_dict():
    metric_dict = {'lidc': np.zeros(0), 'lndb': np.zeros(0), 'inhouse': np.zeros(0), 'total': np.zeros(0)}
    return metric_dict

def _tent_model(model):
    model = configure_model(model)
    params, _ = collect_params(model)
    optimizer = optim.Adam(params, lr=cfg.test_lr)
    tented_model = Tent(model, optimizer)
    return tented_model

@logger
def _test_epoch(df, model, dataloader):
    loss_test_dict = _init_loss_dict(0)
    dice_test_dict = _init_metric_dict()
    nsd_test_dict = _init_metric_dict()
    iou_test_dict = _init_metric_dict()
    recall_test_dict = _init_metric_dict()
    precision_test_dict = _init_metric_dict()
    tent_model = _tent_model(model)
    for idx, sample in enumerate(dataloader):
        images = sample['images'].cuda()
        masks = sample['masks'].cuda()
        names = sample['names']
        datasets = np.array(sample['datasets'])
        with torch.no_grad():
            model.eval()
            outputs = model(images)
        sphere, fake_rate = get_saclick(outputs, masks, cfg.diameter_rate, unbiased=False)
        outputs, loss_dict, center_value = tent_model([images, sphere])
        dice_dict = _metric_by_label(masks.cpu().detach().numpy(), (outputs > 0).float().cpu().detach().numpy(), datasets, dice_by_label)
        nsd_dict = _metric_by_label(masks.cpu().detach().numpy().astype(np.bool_), (outputs > 0).float().cpu().detach().numpy().astype(np.bool_), datasets, surface_dice)
        iou_dict = _metric_by_label(masks.cpu().detach().numpy(), (outputs > 0).float().cpu().detach().numpy(), datasets, iou_by_label)
        recall_dict = _metric_by_label(masks.cpu().detach().numpy(), (outputs > 0).float().cpu().detach().numpy(), datasets, recall_by_label)
        precision_dict = _metric_by_label(masks.cpu().detach().numpy(), (outputs > 0).float().cpu().detach().numpy(), datasets, precision_by_label)
        for k in loss_dict.keys():
            loss_test_dict[k] = np.append(loss_test_dict[k], loss_dict[k].cpu().item())
        for k in dice_dict.keys():
            dice_test_dict[k] = np.append(dice_test_dict[k], dice_dict[k])
        for k in iou_dict.keys():
            iou_test_dict[k] = np.append(iou_test_dict[k], iou_dict[k])
        for k in nsd_dict.keys():
            nsd_test_dict[k] = np.append(nsd_test_dict[k], nsd_dict[k])
        for k in recall_dict.keys():
            recall_test_dict[k] = np.append(recall_test_dict[k], recall_dict[k])
        for k in precision_dict.keys():
            precision_test_dict[k] = np.append(precision_test_dict[k], precision_dict[k])
        df = _show_dice(df, names, dice_dict['total'], nsd_dict['total'], iou_dict['total'], recall_dict['total'], precision_dict['total'])
    for k in loss_test_dict.keys():
        loss_test_dict[k] = np.mean(loss_test_dict[k])
    for k in dice_test_dict.keys():
        if len(dice_test_dict[k]) > 0:
            dice_test_dict[k] = np.mean(dice_test_dict[k])
        else:
            dice_test_dict[k] = 0
    for k in nsd_test_dict.keys():
        if len(nsd_test_dict[k]) > 0:
            nsd_test_dict[k] = np.mean(nsd_test_dict[k])
        else:
            nsd_test_dict[k] = 0
    return (df, dice_test_dict, nsd_test_dict)

def _log_metrics(loss_dict):
    stats = pd.DataFrame([loss_dict], index=['loss and metric']).T
    print(stats)

def _metric_by_label(y_true, y_pred, dataset_list, metric):
    metric_list = metric(y_true, y_pred)
    metric_lidc_list = metric_list[np.argwhere(dataset_list == 1)]
    metric_lndb_list = metric_list[np.argwhere(dataset_list == 2)]
    metric_inhouse_list = metric_list[np.argwhere(dataset_list == 3)]
    metric_dict = {'lidc': metric_lidc_list, 'lndb': metric_lndb_list, 'inhouse': metric_inhouse_list, 'total': metric_list}
    return metric_dict

def _show_dice(df, names_test, dice_test, nsd_test, iou_test, recall_test, precision_test):
    df['file_id'].extend(names_test)
    df['dcs'].extend(dice_test)
    df['nsd'].extend(nsd_test)
    df['iou'].extend(iou_test)
    df['recall'].extend(recall_test)
    df['precision'].extend(precision_test)
    for name, dice, nsd, iou, recall, precision in zip(names_test, dice_test, nsd_test, iou_test, recall_test, precision_test):
        print('id: ', name, 'dice: ', dice, 'nsd', nsd, 'iou', iou, 'recall', recall, 'precision', precision)
    return df

def main(df, model_path, dataset, testset):
    _set_rng_seed(42)
    model = _load_model(model_path)
    dl_test = _init_dataloaders(dataset, testset)
    df, dice_test_dict, nsd_test_dict = _test_epoch(df, model, dl_test)
    _log_metrics(dice_test_dict)
    _log_metrics(nsd_test_dict)
    return df
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    flag = 'sattc'
    metric_dict = {'file_id': [], 'dcs': [], 'nsd': [], 'iou': [], 'recall': [], 'precision': []}
    metric_dict = main(metric_dict, cfg.model_path, cfg.dataset_list, cfg.test_set)
    save_path = cfg.log_path + flag + '.csv'
    metric_df = pd.DataFrame(metric_dict)
    metric_df.to_csv(save_path, mode='w', header=True, index=False)
