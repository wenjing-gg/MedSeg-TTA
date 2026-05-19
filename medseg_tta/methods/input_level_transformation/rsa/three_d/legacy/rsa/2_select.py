import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from medseg_tta.models.nnunet import PlainConvUNet

from dataset.utils_brats_all import get_data_loader
from utils import set_seed, cal_var, cal_dice, find_best

sys.path.append('.')


def get_args():
    script_dir = Path(__file__).resolve().parent
    artifact_dir = script_dir.parent / "artifacts"
    parser = argparse.ArgumentParser(description='RSA selection stage')
    parser.add_argument('--sample_dir', type=str, default=str(artifact_dir / 'translated'))
    parser.add_argument('--source-root', type=str, default=str(artifact_dir / 'source'))
    parser.add_argument('--target-root', type=str, default=str(artifact_dir / 'target'))
    parser.add_argument('--train-path', type=str, default='train')
    parser.add_argument('--test-path', type=str, default='test')
    parser.add_argument('--img', type=str, default='t2f')
    parser.add_argument('--mode', type=str, default='source_to_source')
    parser.add_argument('--save_dir', type=str, default=str(artifact_dir / 'selected'))
    parser.add_argument('--record_path', type=str, default=str(artifact_dir / 'selected' / 'selection_record.json'))
    parser.add_argument('--seg_ckpt_dir', type=str, default=str(artifact_dir / 'checkpoints' / 'nnunet_best.pth'))
    parser.add_argument('--pred_thresh', type=float, default=0.7)
    parser.add_argument('--step_n', type=int, default=2)
    parser.add_argument('--run_n', type=int, default=3)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(Path(args.record_path).parent, exist_ok=True)
    device = torch.device(f"cuda:{args.device}")
    set_seed(args.seed)
    record = {}
    dice_avg = []

    train_dataloader, _ = get_data_loader(
        sample_dir=args.sample_dir,
        source_root=args.source_root,
        target_root=args.target_root,
        train_path=args.train_path,
        test_path=args.test_path,
        batch_train=1,
        batch_test=1,
        nw=1,
        img=args.img,
        mode=args.mode,
    )

    net = PlainConvUNet(1, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4, (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True).to(device)
    checkpoint = torch.load(args.seg_ckpt_dir, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    for sampled_batch in train_dataloader:
        label = sampled_batch["label"].to(device).squeeze().cpu().numpy()
        inp = torch.as_tensor(sampled_batch["samples"], dtype=torch.float32, device=device)
        if inp.ndim == 5:
            inp = inp.squeeze(0)
        if inp.ndim == 4:
            inp = inp.unsqueeze(1)
        sample_id = sampled_batch["sample_id"][0] if isinstance(sampled_batch["sample_id"], list) else sampled_batch["sample_id"]

        masks_pred = net.forward(inp)
        if isinstance(masks_pred, (tuple, list)):
            masks_pred = masks_pred[0]
        masks_pred = (torch.sigmoid(masks_pred) > 0.5).float().squeeze(1).cpu().numpy()
        samples_np = inp.squeeze(1).cpu().numpy()

        samples = np.array_split(samples_np, args.step_n, axis=0)
        masks_pred = np.array_split(masks_pred, args.step_n, axis=0)

        all_var = np.ones((args.step_n,))
        all_dice = np.ones((args.step_n, args.run_n))

        for r, (trans, preds) in enumerate(zip(samples, masks_pred)):
            var = cal_var(preds)
            all_var[r] = var
            for i in range(min(trans.shape[0], args.run_n)):
                pred = preds[i]
                all_dice[r, i] = cal_dice(im1=pred, im2=label)

        best_var, best_pred, best_step, best_run = find_best(all_var, args.pred_thresh, masks_pred)
        if best_var is not None:
            best_dice = cal_dice(best_pred, label)
            best_sample = samples[best_step][best_run]
            dice_avg.append(best_dice)

            record[sample_id] = {
                'best_dice': float(best_dice),
                'best_var': float(best_var),
                'best_step': int(best_step),
                'best_run': int(best_run),
            }
            sio.savemat(
                f'{args.save_dir}/{sample_id}.mat',
                {
                    'sample': best_sample,
                    'pseudo': best_pred,
                }
            )

    if dice_avg:
        print(len(dice_avg))
        print(sum(dice_avg) / len(dice_avg))
    else:
        print(0)
        print(0)

    with open(args.record_path, 'w') as f:
        json.dump(record, f, indent=4)
