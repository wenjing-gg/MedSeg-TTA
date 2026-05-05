import argparse, os, glob, datetime, traceback, pickle
from typing import Tuple, Optional, Dict, List
import numpy as np, nibabel as nib, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, Resized, NormalizeIntensityd, ToTensord
from nnunet import PlainConvUNet
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
from test_time_adaptation import adaptation_base, tent, hist_matching, entropy_KL, filter_inspect_utils

def _auto_subfolder(path: str, dtype: str) -> str:
    if not os.path.exists(path):
        return 'CT_' if dtype == 'CT' else ''
    subs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    if not subs:
        return ''
    cand = sorted([s for s in subs if s.endswith('_')])
    return cand[0] if cand else sorted(subs)[0]

def get_dataset_paths(dtype: str, base: str) -> Tuple[str, str]:
    mapping = {'CT': 'TTA-3DCT'}
    folder = mapping.get(dtype, 'TTA-3DCT')
    root = os.path.join(base, folder)
    sub = _auto_subfolder(root, dtype)
    if sub:
        return (os.path.join(root, sub, 'image'), os.path.join(root, sub, 'mask'))
    return (os.path.join(root, 'image'), os.path.join(root, 'mask'))

def resolve_dirs(tar, img, msk):
    if img and msk:
        return (img, msk)
    if tar:
        return (img or os.path.join(tar, 'image'), msk or os.path.join(tar, 'mask'))
    return (None, None)

def binarize(lbl: torch.Tensor, pos: List[int]):
    if lbl.ndim == 4 and lbl.shape[0] == 1:
        lbl = lbl.squeeze(0)
    mask = torch.zeros_like(lbl, dtype=torch.bool)
    for p in pos:
        mask |= lbl == p
    return mask.long().unsqueeze(0)

class CTDataset3D(Dataset):

    def __init__(self, img_dir, msk_dir, image_size=(128,) * 3, spacing=(1.0, 1.0, 1.0), intensity_range=(-200, 400), pos_labels=None):
        self.img_dir, self.msk_dir = (img_dir, msk_dir)
        self.image_size, self.spacing = (image_size, spacing)
        self.intensity_range = intensity_range
        self.pos_labels = pos_labels or [1]
        exts = ['.nii.gz', '.nii', '.mha', '.mhd']
        self.exts = exts
        self.data = self._pairs()
        self.tf = self._transforms()

    def _pairs(self):
        imgs, pairs = ([], [])
        for e in self.exts:
            imgs += glob.glob(os.path.join(self.img_dir, f'*{e}'))
        for p in sorted(imgs):
            base = os.path.basename(p)
            stem = os.path.splitext(base)[0].split('.nii')[0]
            msk = next((os.path.join(self.msk_dir, stem + suf + e) for suf in ['', '_mask', '-mask', '_seg', '_gt'] for e in self.exts if os.path.exists(os.path.join(self.msk_dir, stem + suf + e))), None)
            if msk:
                pairs.append({'image': p, 'label': msk, 'id': base})
        return pairs

    def _transforms(self):
        t = [LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']), Orientationd(keys=['image', 'label'], axcodes='RAS'), Spacingd(keys=['image', 'label'], pixdim=self.spacing, mode=('bilinear', 'nearest')), ScaleIntensityRanged(keys=['image'], a_min=self.intensity_range[0], a_max=self.intensity_range[1], b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=['image', 'label'], source_key='image'), Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=('trilinear', 'nearest')), NormalizeIntensityd(keys=['image'], nonzero=True), ToTensord(keys=['image', 'label'])]
        return Compose(t)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.tf(self.data[idx].copy())
        img, lbl = (d['image'].float(), d['label'].long())
        lbl = binarize(lbl, self.pos_labels)
        return (img, lbl, d['id'])

def get_loader(tdir, img_dir, msk_dir, **kw):
    img_dir, msk_dir = resolve_dirs(tdir, img_dir, msk_dir)
    if not (img_dir and msk_dir):
        img_dir, msk_dir = get_dataset_paths('CT', kw.get('base', '/home/yuwenjing/data/tta_dataset'))
    ds = CTDataset3D(img_dir, msk_dir, image_size=kw.get('image_size', (128,) * 3), spacing=kw.get('spacing', (1.0, 1.0, 1.0)), intensity_range=kw.get('intensity_range', (-200, 400)), pos_labels=kw.get('positive_labels', [1]))
    return DataLoader(ds, batch_size=kw.get('batch', 1), shuffle=False, num_workers=kw.get('nw', 2), pin_memory=True)

def build_model(kind, dev):
    if kind == 'nnunet':
        return PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(dev)
    from unet3d import UNet3dCT
    return UNet3dCT().to(dev)

def entropy_kl_wrapper(model, dev, lr, lambd, steps, atlas):
    if atlas is None:
        raise FileNotFoundError('atlas_labels_path 不能为空')
    model = adaptation_base.configure_model(model)
    params, _ = adaptation_base.collect_batch_norm_params(model)
    opt = torch.optim.Adam(params, lr=lr)
    return entropy_KL.EntropyKL(model, opt, atlas, lambd=lambd, steps=steps).to(dev)

def make_tta(model, args, dev):
    if args.tta_method in ['none', 'baseline']:
        return model
    if args.tta_method == 'tent':
        model = adaptation_base.configure_model(model)
        params, _ = adaptation_base.collect_batch_norm_params(model)
        return tent.Tent(model, torch.optim.Adam(params, lr=args.lr), steps=args.tta_steps).to(dev)
    if args.tta_method == 'entropy_kl':
        return entropy_kl_wrapper(model, dev, args.lr, args.kl_lambda, args.tta_steps, args.atlas_labels_path)
    if args.tta_method == 'hist_matching':
        ref = torch.load(args.reference_volume_path)
        return hist_matching.HistMatching(model, ref)
    if args.tta_method == 'filter_inspect':
        act = pickle.load(open(args.source_data_activations_path, 'rb'))
        fi = filter_inspect_utils.create_filter_inspector(model, use_cuda=True)
        cfg = {'steps': args.tta_steps, 'lr': args.lr, 'num_to_update': args.num_filters_to_update, 'device': dev, 'filter_inspect_mode': 'Taylor', 'lambda': args.kl_lambda, 'atlas_labels_path': args.atlas_labels_path, 'force_include_batch_norm': args.force_include_batch_norm, 'use_KL': args.use_KL, 'week_num': 21, 'hemisphere_split': False, 'subject_list': args.img}
        return filter_inspect_utils.configure_filter_inspect(fi.unet, fi, None, args.img, act, cfg)
    raise ValueError('Unsupported TTA method')

def merge_logits(logits, bg=0):
    prob = torch.softmax(logits, 1)
    p_bg = prob[:, bg:bg + 1]
    p_t = (prob.sum(1, keepdim=True) - p_bg).clamp(0, 1)
    return torch.cat([p_bg, p_t], 1)

def calc_mean_std(arr_list: List[np.ndarray]) -> Tuple[float, float]:
    v = np.stack(arr_list)
    return (float(v.mean()), float(v.std()))

def test_on_target(a, dev):
    model = build_model(a.model_type, dev)
    ck = a.model_path if a.model_path != 'default' else '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth' if a.model_type == 'nnunet' else '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth'
    state = torch.load(ck, map_location=dev, weights_only=False)
    model.load_state_dict(state['model_state_dict'] if isinstance(state, dict) and 'model_state_dict' in state else state)
    tta = make_tta(model, a, dev)
    loader = get_loader(a.target_root, None, None, image_size=(a.image_size,) * 3, spacing=tuple(a.spacing), intensity_range=tuple(a.intensity_range), batch=a.batch_test, nw=a.num_workers, positive_labels=a.positive_labels)
    metric_buf = {k: [] for k in ['dice', 'hd95', 'iou', 'pa', 'rve', 'sen', 'ppv']}
    for img, lbl, _ in tqdm(loader, desc='Inference'):
        img, lbl = (img.to(dev), lbl.to(dev))
        out = tta(img) if hasattr(tta, 'forward') else tta.model(img)
        if isinstance(out, tuple):
            out = out[0]
        binp = merge_logits(out, a.bg_channel)
        metric_buf['dice'].append(cal_dice(binp, lbl.squeeze(1))[1])
        metric_buf['hd95'].append(cal_hd95(binp, lbl.squeeze(1))[1])
        metric_buf['iou'].append(IoU(binp, lbl.squeeze(1))[1])
        metric_buf['pa'].append(PA(binp, lbl.squeeze(1), 2)[1])
        metric_buf['rve'].append(cal_RVE(binp, lbl.squeeze(1))[1])
        metric_buf['sen'].append(cal_sensitivity(binp, lbl.squeeze(1))[1])
        metric_buf['ppv'].append(cal_ppv(binp, lbl.squeeze(1))[1])
    print(f'\n==== Tumor 通道 结果 (方法: {a.tta_method}) ====')
    for m in metric_buf:
        mu, sigma = calc_mean_std(metric_buf[m])
        if m == 'hd95':
            print(f'{m.upper():4s}: {mu:6.2f} ± {sigma:6.2f}  mm')
        else:
            print(f'{m.upper():4s}: {mu:6.4f} ± {sigma:6.4f}')
    return True
if __name__ == '__main__':
    pa = argparse.ArgumentParser('CT-TTA with Entropy+KL default & mean±std')
    pa.add_argument('--target_root', default='/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB')
    pa.add_argument('--model_type', choices=['nnunet', 'unet3d'], default='unet3d')
    pa.add_argument('--model_path', default='default')
    pa.add_argument('--tta_method', default='tent', choices=['none', 'baseline', 'tent', 'entropy_kl', 'hist_matching', 'filter_inspect'])
    pa.add_argument('--lr', type=float, default=1e-05)
    pa.add_argument('--tta_steps', type=int, default=1)
    pa.add_argument('--kl_lambda', type=float, default=1.0)
    pa.add_argument('--atlas_labels_path', type=str, default=None)
    pa.add_argument('--reference_volume_path', type=str, default=None)
    pa.add_argument('--source_data_activations_path', type=str, default=None)
    pa.add_argument('--num_filters_to_update', type=int, default=1)
    pa.add_argument('--force_include_batch_norm', action='store_true')
    pa.add_argument('--use_KL', action='store_true')
    pa.add_argument('--gpu', type=int, default=0)
    pa.add_argument('--batch_test', type=int, default=1)
    pa.add_argument('--num_workers', type=int, default=2)
    pa.add_argument('--image_size', type=int, default=128)
    pa.add_argument('--spacing', type=float, nargs=3, default=(1.0, 1.0, 1.0))
    pa.add_argument('--intensity_range', type=float, nargs=2, default=(-200, 400))
    pa.add_argument('--bg_channel', type=int, default=0)
    pa.add_argument('--positive_labels', type=str, default='1')
    args = pa.parse_args()
    args.positive_labels = [int(x) for x in args.positive_labels.split(',') if x.strip()]
    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if args.tta_method == 'entropy_kl' and (not args.atlas_labels_path):
        raise FileNotFoundError('必须提供 --atlas_labels_path 才能使用 Entropy+KL')
    test_on_target(args, dev)
