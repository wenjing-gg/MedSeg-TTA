import numpy as np
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
EPS = 1e-08

def dice_by_label(y_true, y_pred):
    batch_size = y_true.shape[0]
    dice_list = []
    for i in range(batch_size):
        dice_list.append(_oneclass_dice(y_true[i], y_pred[i]))
    return np.array(dice_list)

def iou_by_label(y_true, y_pred):
    batch_size = y_true.shape[0]
    iou_list = []
    for i in range(batch_size):
        iou_list.append(_oneclass_iou(y_true[i], y_pred[i]))
    return np.array(iou_list)

def recall_by_label(y_true, y_pred):
    batch_size = y_true.shape[0]
    recall_list = []
    for i in range(batch_size):
        recall_list.append(_oneclass_recall(y_true[i], y_pred[i]))
    return np.array(recall_list)

def precision_by_label(y_true, y_pred):
    batch_size = y_true.shape[0]
    precision_list = []
    for i in range(batch_size):
        precision_list.append(_oneclass_precision(y_true[i], y_pred[i]))
    return np.array(precision_list)

def surface_dice(y_pred, y_true):
    batch_size = y_true.shape[0]
    nsd_list = []
    mask_gt, mask_pred = (y_true.astype(np.bool), y_pred.astype(np.bool))
    for i in range(batch_size):
        surface_distances = compute_surface_distances(mask_gt[i, 0], mask_pred[i, 0], spacing_mm=(1, 1, 1))
        ret = compute_surface_dice_at_tolerance(surface_distances, 1)
        nsd_list.append(ret)
    return np.array(nsd_list)

def _oneclass_iou(y_true, y_pred):
    iou = np.logical_and(y_true, y_pred).sum() / (y_true.sum() + y_pred.sum() - np.logical_and(y_true, y_pred).sum() + EPS)
    return iou

def _oneclass_dice(y_true, y_pred):
    dice = 2 * np.logical_and(y_true, y_pred).sum() / (y_true.sum() + y_pred.sum() + EPS)
    return dice

def _oneclass_recall(y_true, y_pred):
    recall = np.logical_and(y_true, y_pred).sum() / (y_true.sum() + EPS)
    return recall

def _oneclass_precision(y_true, y_pred):
    precision = np.logical_and(y_true, y_pred).sum() / (y_pred.sum() + EPS)
    return precision

def dice_score(y_true, y_pred, num_classes):
    dice = np.mean([_oneclass_dice(y_true == i, y_pred == i) for i in range(num_classes)])
    return dice

def foreground_dice_score(y_true, y_pred, num_classes):
    dice = np.mean([_oneclass_dice(y_true == i, y_pred == i) for i in range(1, num_classes + 1)])
    return dice
