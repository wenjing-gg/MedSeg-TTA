from utils.logger import logger, save_nii
from utils.metrics import dice_by_label
from utils.misc import *
from models.unet import UNet
from loss import SegLoss
from data import transforms as aug
from data.dataset import SaDataset
from configs import config as cfg
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import random
from copy import deepcopy
import os

def _set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _init_models():
    model = UNet(num_classes=1)
    model = model.cuda()
    model = nn.DataParallel(model)
    return model

def _init_dataloaders(dataset):
    file_csv = cfg.csv_path
    transforms_train = [aug.Window(cfg.win_min, cfg.win_max), aug.MinMaxNormalize(cfg.win_min, cfg.win_max), aug.LabelTrans()]
    transforms_val = [aug.Window(cfg.win_min, cfg.win_max), aug.MinMaxNormalize(cfg.win_min, cfg.win_max), aug.LabelTrans()]
    val_set = (0,)
    test_set = (1, 2)
    total = set(range(10))
    train_set = tuple(total ^ set(val_set + test_set))
    ds_train = SaDataset(file_csv, transforms_train, train_set, dataset, 0)
    dl_train = SaDataset.get_dataloader(ds_train, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    ds_val = SaDataset(file_csv, transforms_val, val_set, dataset, 0)
    dl_val = SaDataset.get_dataloader(ds_val, cfg.batch_size, False, cfg.num_workers)
    print('train data number: ', len(ds_train), 'val data number: ', len(ds_val))
    return (dl_train, dl_val)

def _init_loss_dict():
    loss_dict = {'dice_loss': 0, 'bce_loss': 0, 'shape_loss': 0, 'region_loss': 0, 'total_loss': 0}
    return loss_dict

def _init_dice_dict():
    dice_dict = {'dice_lidc': np.zeros(0), 'dice_lndb': np.zeros(0), 'dice_inhouse': np.zeros(0), 'dice': np.zeros(0)}
    return dice_dict

@logger
def _train_epoch(model, dataloader, criterion, optimizer, scheduler, niis_dir, epoch):
    model.train()
    loss_train_dict = _init_loss_dict()
    len_train_dict = deepcopy(loss_train_dict)
    dice_train_dict = _init_dice_dict()
    len_train_dice = deepcopy(dice_train_dict)
    for idx, sample in enumerate(dataloader):
        optimizer.zero_grad()
        images = sample['images'].cuda()
        masks = sample['masks'].cuda()
        distance_maps = sample['distance_maps'].cuda()
        names = sample['names']
        datasets = np.array(sample['datasets'])
        outputs = model(images)
        loss_dict = criterion(outputs, masks)
        loss_dict['total_loss'].backward()
        optimizer.step()
        scheduler.step()
        dice_dict = _dice_by_label(masks.cpu().detach().numpy(), (outputs > 0).float().cpu().detach().numpy(), datasets)
        for k in loss_dict.keys():
            loss_train_dict[k] += loss_dict[k].detach().cpu().item()
            if k != 'total_loss':
                len_train_dict[k] += 1
        for k in dice_dict.keys():
            dice_train_dict[k] = np.append(dice_train_dict[k], dice_dict[k])
            if k != 'dice':
                len_train_dice[k] += 1
    for k in loss_train_dict.keys():
        if k == 'total_loss':
            loss_train_dict[k] /= len(dataloader)
        elif len_train_dict[k] > 0:
            loss_train_dict[k] /= len_train_dict[k]
    for k in dice_train_dict.keys():
        dice_train_dict[k] = np.mean(dice_train_dict[k])
    if epoch % 10 == 9:
        nii = {'name': names, 'img': images.cpu().detach().numpy(), 'seg': masks.cpu().detach().numpy(), 'pre': (outputs > 0).cpu().detach().numpy()}
        _log_nii(niis_dir + '_train/', nii)
    lr = optimizer.param_groups[0]['lr']
    return (loss_train_dict, dice_train_dict, names, lr, dice_dict['dice'])

@logger
@torch.no_grad()
def _eval_epoch(model, dataloader, criterion, niis_dir, epoch):
    model.eval()
    loss_val_dict = _init_loss_dict()
    dice_val_dict = _init_dice_dict()
    len_val_dict = deepcopy(loss_val_dict)
    len_val_dice = deepcopy(dice_val_dict)
    for idx, sample in enumerate(dataloader):
        images = sample['images'].cuda()
        masks = sample['masks'].cuda()
        names = sample['names']
        datasets = np.array(sample['datasets'])
        outputs = model(images)
        loss_dict = criterion(outputs, masks)
        dice_dict = _dice_by_label(masks.cpu().detach().numpy(), (outputs > 0).float().cpu().detach().numpy(), datasets)
        for k in loss_dict.keys():
            loss_val_dict[k] += loss_dict[k].cpu().item()
            if k != 'total_loss':
                len_val_dict[k] += 1
        for k in dice_dict.keys():
            dice_val_dict[k] = np.append(dice_val_dict[k], dice_dict[k])
            if k != 'dice':
                len_val_dice[k] += 1
        if epoch % 10 == 9:
            nii = {'name': names, 'img': images.cpu().detach().numpy(), 'seg': masks.cpu().detach().numpy(), 'pre': (outputs > 0).cpu().detach().numpy()}
    for k in loss_val_dict.keys():
        if k == 'total_loss':
            loss_val_dict[k] /= len(dataloader)
        elif len_val_dict[k] > 0:
            loss_val_dict[k] /= len_val_dict[k]
    for k in dice_val_dict.keys():
        if len(dice_val_dict[k]) > 0:
            dice_val_dict[k] = np.mean(dice_val_dict[k])
        else:
            dice_val_dict[k] = 0
    return (loss_val_dict, dice_val_dict, names, dice_dict['dice'])

def _log_metrics(loss_dicts):
    if len(loss_dicts) == 1:
        index = ['train']
    else:
        index = ['train', 'val']
    stats = pd.DataFrame(loss_dicts, index=index).T
    print(stats)

def _log_tensorboard(tb_writer, epoch, loss_train_dict, tital):
    for k in loss_train_dict.keys():
        tb_writer.add_scalars(f'{tital}/{k}', {'train': loss_train_dict[k]}, epoch)
    tb_writer.flush()

def _log_nii(outpath, nii):
    os.makedirs(outpath, exist_ok=True)
    for i in range(len(nii['name'])):
        name = nii['name'][i]
        img = nii['img'][i][0]
        seg = nii['seg'][i][0]
        pre = nii['pre'][i][0]
        os.makedirs(outpath + name, exist_ok=True)
        img_path = outpath + name + '/' + 'img.nii.gz'
        seg_path = outpath + name + '/' + 'seg.nii.gz'
        pre_path = outpath + name + '/' + 'pre.nii.gz'
        save_nii(np.float32(img), img_path)
        save_nii(np.int8(seg), seg_path)
        save_nii(np.int8(pre), pre_path)

def _dice_by_label(y_true, y_pred, dataset_list):
    dice_list = dice_by_label(y_true, y_pred)
    dice_lidc_list = dice_list[np.argwhere(dataset_list == 1)]
    dice_lndb_list = dice_list[np.argwhere(dataset_list == 2)]
    dice_inhouse_list = dice_list[np.argwhere(dataset_list == 3)]
    dice_dict = {'dice_lidc': dice_lidc_list, 'dice_lndb': dice_lndb_list, 'dice_inhouse': dice_inhouse_list, 'dice': dice_list}
    return dice_dict

def _show_dice(names_train, dice_train, names_val, dice_val):
    print('train sample:')
    for name, dice in zip(names_train, dice_train):
        print('id: ', name, 'dice: ', dice)
    print('test sample:')
    for name, dice in zip(names_val, dice_val):
        print('id: ', name, 'dice: ', dice)

def main(dataset):
    _set_rng_seed(42)
    dl_train, dl_val = _init_dataloaders(dataset)
    model = _init_models()
    criterion = SegLoss()
    optimizer = optim.AdamW(model.parameters(), cfg.max_lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dl_train) * cfg.epochs, cfg.min_lr)
    log_dir = cfg.log_path
    cur_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    print(cur_time)
    log_dir = os.path.join(log_dir, cur_time)
    tb_writer = SummaryWriter(log_dir)
    for i in range(cfg.epochs):
        print(f'Epoch {i}')
        niis_dir = os.path.join(log_dir, 'niis/epoch' + str(i))
        loss_train, dice_train_dict, names_train, lr, dice_train = _train_epoch(model, dl_train, criterion, optimizer, scheduler, niis_dir, i)
        tb_writer.add_scalars('parameter/lr', {'learning_rate': lr}, i)
        _log_metrics([loss_train])
        _log_metrics([dice_train_dict])
        _log_tensorboard(tb_writer, i, loss_train, 'loss')
        _log_tensorboard(tb_writer, i, dice_train_dict, 'dice')
        if i % 10 == 9:
            loss_val, dice_val_dict, names_val, dice_val = _eval_epoch(model, dl_val, criterion, niis_dir, i)
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_{i}.pth'))
            _show_dice(names_train, dice_train, names_val, dice_val)
            _log_metrics([loss_train, loss_val])
            _log_metrics([dice_train_dict, dice_val_dict])
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(cfg.dataset_list)
