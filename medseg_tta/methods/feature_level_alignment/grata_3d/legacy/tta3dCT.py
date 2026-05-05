import argparse, os, glob, datetime, traceback, pickle, warnings
from typing import List, Tuple, Optional, Dict
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, Resized, NormalizeIntensityd, ToTensord
from nnunet import PlainConvUNet
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
from utils_brats_all import get_data_loader
from grata_wrapper import create_grata_model, get_default_grata_config

def get_dataset_paths(root_dir: str) -> Tuple[str, str]:
    image_dir = os.path.join(root_dir, 'image')
    mask_dir = os.path.join(root_dir, 'mask')
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f'[get_dataset_paths] 未找到图像目录: {image_dir}')
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f'[get_dataset_paths] 未找到标注目录: {mask_dir}')
    return (image_dir, mask_dir)

def binarize(lbl: torch.Tensor, pos=[1]):
    if lbl.ndim == 4 and lbl.shape[0] == 1:
        lbl = lbl.squeeze(0)
    mask = torch.zeros_like(lbl, dtype=torch.bool)
    for p in pos:
        mask |= lbl == p
    return mask.long().unsqueeze(0)

class CTDataset3D(Dataset):

    def __init__(self, img_dir, msk_dir, img_size=(128,) * 3, spacing=(1.0, 1.0, 1.0), inten=(-200, 400), pos=[1]):
        self.img_dir, self.msk_dir = (img_dir, msk_dir)
        self.pos = pos
        self.exts = ['.nii.gz', '.nii', '.mha', '.mhd']
        self.data = self._collect()
        if len(self.data) == 0:
            raise RuntimeError(f'[CTDataset3D] 在 {img_dir} 与 {msk_dir} 下未匹配到任何图像/标注对')
        self.tf = self._tf(img_size, spacing, inten)

    def _collect(self):
        pairs, imgs = ([], [])
        for e in self.exts:
            imgs += glob.glob(os.path.join(self.img_dir, f'*{e}'))
        for p in sorted(imgs):
            base = os.path.basename(p)
            stem = base
            for e in self.exts:
                if stem.endswith(e):
                    stem = stem[:-len(e)]
            msk = None
            for suf in ['', '_mask', '-mask', '_seg', '_gt']:
                for e in self.exts:
                    cand = os.path.join(self.msk_dir, stem + suf + e)
                    if os.path.exists(cand):
                        msk = cand
                        break
                if msk is not None:
                    break
            if msk:
                pairs.append({'image': p, 'label': msk, 'id': base})
        return pairs

    def _tf(self, img_size, spacing, inten):
        return Compose([LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']), Orientationd(keys=['image', 'label'], axcodes='RAS'), Spacingd(keys=['image', 'label'], pixdim=spacing, mode=('bilinear', 'nearest')), ScaleIntensityRanged(keys=['image'], a_min=inten[0], a_max=inten[1], b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=['image', 'label'], source_key='image'), Resized(keys=['image', 'label'], spatial_size=img_size, mode=('trilinear', 'nearest')), NormalizeIntensityd(keys=['image'], nonzero=True), ToTensord(keys=['image', 'label'])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.tf(self.data[idx].copy())
        return (d['image'].float(), binarize(d['label'].long(), self.pos), d['id'])

def get_ct_loader(root, bs, nw, **cfg):
    img_dir, msk_dir = get_dataset_paths(root)
    ds = CTDataset3D(img_dir, msk_dir, img_size=cfg.get('img_size', (128,) * 3), spacing=cfg.get('spacing', (1.0, 1.0, 1.0)), inten=cfg.get('inten', (-200, 400)), pos=cfg.get('pos', [1]))
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

def build_backbone(kind: str, dev):
    if kind == 'nnunet':
        return PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(dev)
    from unet3d import UNet3dCT
    return UNet3dCT().to(dev)

def load_weights(model, path, dev):
    state = torch.load(path, map_location=dev, weights_only=False)
    model.load_state_dict(state['model_state_dict'] if isinstance(state, dict) and 'model_state_dict' in state else state)
    return model

def to_scalar(t):
    return float(t.item()) if hasattr(t, 'item') else float(t)

def metric_buffer():
    return {k: [] for k in ['dice', 'hd95', 'iou', 'pa', 'rve', 'sen', 'ppv']}

def calc_stats(buf, idx):
    arr = np.array(buf)[:, idx]
    return (arr.mean(), arr.std())

def test_once(args, dev):
    backbone = build_backbone(args.model_type, dev)
    ckpt = args.model_path if args.model_path != 'default' else '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth' if args.model_type == 'nnunet' else '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints_CT/unet3d_best_CT.pth'
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'权重不存在: {ckpt}')
    load_weights(backbone, ckpt, dev)
    cfg = get_default_grata_config()
    cfg.lr = args.lr
    cfg.aux_loss = args.aux_loss
    cfg.pse_loss = args.pse_loss
    cfg.optimizer = args.optimizer
    grata = create_grata_model(backbone, cfg, dev)
    loader = get_ct_loader(args.target_root, args.batch_test, args.num_workers, img_size=(128,) * 3, spacing=(1.0, 1.0, 1.0), inten=(-200, 400), pos=[1])
    buf = metric_buffer()
    adapted_model_path = None
    for img, lbl, _ in tqdm(loader, desc='GraTa 推理'):
        img, lbl = (img.to(dev), lbl.to(dev))
        with torch.no_grad():
            out = grata.adapt_and_predict(img)
        pb = torch.softmax(out, 1)
        pb = torch.stack([pb[:, 0], pb[:, 1:].sum(1)], 1)
        buf['dice'].append(cal_dice(pb, lbl.squeeze(1)))
        buf['hd95'].append(cal_hd95(pb, lbl.squeeze(1)))
        buf['iou'].append(IoU(pb, lbl.squeeze(1)))
        buf['pa'].append(PA(pb, lbl.squeeze(1), 2))
        buf['rve'].append(cal_RVE(pb, lbl.squeeze(1)))
        buf['sen'].append(cal_sensitivity(pb, lbl.squeeze(1)))
        buf['ppv'].append(cal_ppv(pb, lbl.squeeze(1)))
    print('\n=== GraTa 结果 (Tumor-ROI) ===')
    out_lines = []
    for m in buf:
        mu, sd = calc_stats([list(map(to_scalar, b)) for b in buf[m]], 1)
        if m == 'hd95':
            line = f'{m.upper():4s}: {mu:6.2f} ± {sd:6.2f} mm'
        else:
            line = f'{m.upper():4s}: {mu:6.4f} ± {sd:6.4f}'
        print(line)
        out_lines.append(line)
    t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.save_adapted:
        weights_dir = os.path.join(args.checkpoint_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        adapted_model_path = os.path.join(weights_dir, f'UNet3d_tent_CT.pth')
        torch.save(grata.get_model().state_dict(), adapted_model_path)
        print(f'\n✅ 已保存 GraTa 适应后的模型权重: {adapted_model_path}')
    if args.save_adapted:
        out_lines.append('')
        out_lines.append(f'保存适应后权重: {args.save_adapted}')
        out_lines.append(f'适应后权重路径: {(adapted_model_path if adapted_model_path else 'N/A')}')
    out_path = os.path.join(args.checkpoint_dir, f'GraTa_CT_{t}.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f'\n✅ 结果保存于: {out_path}')
if __name__ == '__main__':
    pa = argparse.ArgumentParser('GraTa-CT Test-time Adaptation')
    pa.add_argument('--target_root', default='/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB', help='必须是包含 image/ 与 mask/ 的目录')
    pa.add_argument('--checkpoint_dir', default='/home/yuwenjing/DeepLearning_ywj/tta/GraTa-3d/checkpoints')
    pa.add_argument('--model_type', choices=['nnunet', 'unet3d'], default='unet3d')
    pa.add_argument('--model_path', default='default')
    pa.add_argument('--lr', type=float, default=5e-06)
    pa.add_argument('--aux_loss', choices=['ent', 'consis'], default='ent')
    pa.add_argument('--pse_loss', choices=['ent', 'consis'], default='consis')
    pa.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam')
    pa.add_argument('--batch_test', type=int, default=1)
    pa.add_argument('--num_workers', type=int, default=2)
    pa.add_argument('--gpu', type=int, default=0)
    pa.add_argument('--save_adapted', action='store_true', default=True, help='保存 GraTa 适应后的模型权重')
    args = pa.parse_args()
    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    try:
        test_once(args, dev)
    except Exception as e:
        print(f'\n🔥 运行失败: {e}')
        traceback.print_exc()
