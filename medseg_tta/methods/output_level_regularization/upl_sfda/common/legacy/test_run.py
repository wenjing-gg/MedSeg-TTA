from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from metrics import dice_eval


def test(config, upl_model, valid_loader, test_loader, list_data):
    dataset = config["train"]["dataset"]
    if dataset == "fb":
        num_classes = config["network"]["n_classes_fb"]
    else:
        num_classes = config["network"]["n_classes_mms"]

    device = torch.device(f"cuda:{config['train']['gpu']}" if torch.cuda.is_available() else "cpu")
    upl_model.eval()
    scores = []

    with torch.no_grad():
        for xt, xt_label, *_ in tqdm(test_loader, desc="UPL-SFDA 2D eval"):
            xt = xt.to(device)
            xt_label = xt_label.numpy().squeeze().astype(np.uint8)
            output = upl_model.test_with_name(xt)
            output = torch.argmax(output.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)
            one_case = np.asarray(dice_eval(output, xt_label, num_classes)) * 100.0
            scores.append(float(np.mean(one_case)))

    if scores:
        list_data.append(f"avg_dice={float(np.mean(scores)):.4f}")
    return list_data
