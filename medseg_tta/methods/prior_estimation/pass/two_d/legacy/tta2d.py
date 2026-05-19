import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import datetime
import traceback
from typing import Dict, List
import pickle  # <-- for catching UnpicklingError
import copy
import torch.nn as nn
from urils_DLTTA.loss import DiceLoss, entropy_loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import models.moment_tta.losses as moment_tta_losses
# 新的模型结构
import torch
import torch.nn as nn
import torch.nn.functional as F


TENT = ['Weighted_self_entropy_loss',
         {'weights':[1, 10], 'idc':[0, 1], 'act':'sigmoid'}]

TENT_Prostate = ['Weighted_self_entropy_loss',
         {'weights':[1], 'idc':[0], 'act':'sigmoid'}]

RN_w_CR = ['RN_w_CR_loss',
         {'idc':[0, 1], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'3d'}]

RN_w_CR_Prostate = ['RN_w_CR_loss',
         {'idc':[0], 'act':'sigmoid', 'k':4, 'd':4, 'alpha':0.001, 'tag':'3d'}]

LSIZE_RIGA = ['KL_class_ratio_entropy_loss',
         {'weights':[1, 5], 'idc':[0, 1],
         'class_ratio_prior':[0.0708947674981479, 0.01705511685075431], 'act':'sigmoid'}]

LSIZE_Prostate = ['KL_class_ratio_entropy_loss',
         {'weights':[1], 'idc':[0],
         'class_ratio_prior':[0.034565616183810766,0.032456], 'act':'sigmoid_onelabel'}]

LSIZECentroid = ['Constrain_prior_w_self_entropy_loss',
                 {'idc': [0, 1],
                  'weights_se':[0.8, 0.2],'lamb_se':1,
                  'class_ratio_prior':[0.0708947674981479, 0.01705511685075431],
                  'lamb_moment':0.0001, 'temp':1.01,'margin':0,
                  'mom_est':[[254.9747, 255.98499], [253.97656, 255.04263]],
                  'moment_fn':'soft_centroid', 'lamb_consprior':1,
                  'power': 1, 'act':'sigmoid'}]


LSIZEDistCentroid = ['Constrain_prior_w_self_entropy_loss',
                     {'idc':[0, 1],
                      'weights_se':[0.8, 0.2],'lamb_se':1,
                      'class_ratio_prior':[0.0708947674981479, 0.01705511685075431],
                     'lamb_moment':0.0001, 'temp':1.01,'margin':0,
                     'mom_est':[[39.157185, 37.18185 ],[17.430944,18.484102]],
                     'moment_fn':'soft_dist_centroid', 'lamb_consprior':1,
                     'power': 1, 'act':'sigmoid'}]




class ConvBlock2d(nn.Module):
    """Two 2D convolution layers with batch norm, leaky ReLU and dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock2d(nn.Module):
    """2D down‑sampling followed by a ConvBlock2d."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock2d(in_channels, out_channels, dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock2d(nn.Module):
    """Upsampling block that concatenates skip connection and applies a ConvBlock2d."""

    def __init__(
        self,
        in_channels1: int,
        in_channels2: int,
        out_channels: int,
        dropout_p: float,
        bilinear: bool = True,
    ):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2
            )
        self.conv = ConvBlock2d(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: decoder feature map, x2: corresponding encoder feature map
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)

        # Pad if necessary (handles odd input shapes)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder2d(nn.Module):
    """Hierarchical encoder producing feature maps for skip connections."""

    def __init__(self, in_channels: int, ft_chns: list[int], dropout_p: list[float]):
        super().__init__()
        self.down_path = nn.ModuleList(
            [
                ConvBlock2d(in_channels, ft_chns[0], dropout_p[0]),
                DownBlock2d(ft_chns[0], ft_chns[1], dropout_p[0]),
                DownBlock2d(ft_chns[1], ft_chns[2], dropout_p[0]),
                DownBlock2d(ft_chns[2], ft_chns[3], dropout_p[0]),
                DownBlock2d(ft_chns[3], ft_chns[4], dropout_p[0]),
            ]
        )

    def forward(self, x: torch.Tensor):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
        return blocks, x


class Decoder2d(nn.Module):
    """Decoder that reconstructs the segmentation map using encoder skip connections."""

    def __init__(self, ft_chns: list[int], dropout_p: list[float], n_classes: int = 4, bilinear: bool = True):
        super().__init__()
        self.up_path = nn.ModuleList(
            [
                UpBlock2d(ft_chns[4], ft_chns[3], ft_chns[3], dropout_p[1], bilinear),
                UpBlock2d(ft_chns[3], ft_chns[2], ft_chns[2], dropout_p[0], bilinear),
                UpBlock2d(ft_chns[2], ft_chns[1], ft_chns[1], dropout_p[0], bilinear),
                UpBlock2d(ft_chns[1], ft_chns[0], ft_chns[0], dropout_p[0], bilinear),
            ]
        )
        self.last = nn.Conv2d(ft_chns[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, blocks: list[torch.Tensor]) -> torch.Tensor:
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x)


class UNet2d(nn.Module):
    """End‑to‑end 2D U‑Net segmentation network."""

    def __init__(self, in_channels: int = 1, n_classes: int = 2):
        super().__init__()
        ft_chns = [16, 32, 64, 128, 256]
        dropout_p = [0.0, 0.5]  # [encoder dropout, bottleneck dropout]

        self.encoder = Encoder2d(in_channels, ft_chns, dropout_p)
        self.decoder = Decoder2d(ft_chns, dropout_p, n_classes, bilinear=True)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks, latent = self.encoder(x)
        logits = self.decoder(latent, blocks)
        return logits  # 返回原始logits，不应用softmax


# Gradient reversal layer from the DANN paper
class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DAUNet2d(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super().__init__()
        self.unet = UNet2d(in_channels, n_classes)

        # Domain classifier
        self.fc1 = nn.Linear(65536, 4)  # 假设输入图像大小为 256x256，经过编码器后最后一层特征图大小为 8x8
        self.bn1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 1)
        self.rl = nn.ReLU(True)

    def forward(self, inputs, alpha):
        # Segmentation output
        seg_output = self.unet(inputs)

        # Classification output
        _, latent = self.unet.encoder(inputs)
        bottleneck_features = torch.flatten(latent, 1)
        reverse_features = ReverseLayerF.apply(bottleneck_features, alpha)

        domain_prediction = self.fc1(reverse_features)
        domain_prediction = self.bn1(domain_prediction)
        domain_prediction = self.rl(domain_prediction)
        domain_prediction = self.fc2(domain_prediction)

        return seg_output, domain_prediction


# ----------------------- 2‑D dataset & metrics -------------------------------
from dataset2D import MedicalImageDataset2D
from train_source2D import calculate_all_metrics

# -----------------------------------------------------------------------------
# Utility helpers                                                               
# -----------------------------------------------------------------------------

def safe_value(val):
    """Return python float for scalars / 0‑dim tensors, otherwise as‑is."""
    if isinstance(val, torch.Tensor):
        return val.item()
    return float(val)


def robust_torch_load(path: str, map_location):
    """Load a checkpoint while handling the PyTorch>=2.6 `weights_only` change.

    Strategy:
    1. Try `torch.load` with the default (weights_only=True).
    2. If it throws an `UnpicklingError`, **fallback** to `weights_only=False` –
       only do this if you *trust* the checkpoint source.
    """
    try:
        return torch.load(path, map_location=map_location)  # default (weights_only=True)
    except pickle.UnpicklingError:
        print("[Warning] torch.load failed with weights_only=True – retrying with weights_only=False (trusted checkpoint).")
        return torch.load(path, map_location=map_location, weights_only=False)


def extract_state_dict(obj):
    """Given an arbitrary checkpoint object, return the state_dict to load."""
    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"]
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
    return obj  # already a state_dict


def build_test_loader(target_dir: str,
                      batch_size: int,
                      num_workers: int,
                      image_size: int) -> DataLoader:
    """Create a DataLoader for the target domain (expects image/ & mask/)."""
    image_dir = os.path.join(target_dir, "image")
    mask_dir = os.path.join(target_dir, "mask")
    if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
        raise FileNotFoundError(f"Expect image/ & mask/ inside {target_dir}")

    dataset = MedicalImageDataset2D(
        image_dir=image_dir,
        mask_dir=mask_dir,
        phase="test",
        image_size=(image_size, image_size),
        normalize=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_train_loader(target_dir: str,
                       batch_size: int,
                       num_workers: int,
                       image_size: int) -> DataLoader:
    """Create a DataLoader for the target domain (expects image/ & mask/)."""
    image_dir = os.path.join(target_dir, "image")
    mask_dir = os.path.join(target_dir, "mask")
    if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
        raise FileNotFoundError(f"Expect image/ & mask/ inside {target_dir}")

    dataset = MedicalImageDataset2D(
        image_dir=image_dir,
        mask_dir=mask_dir,
        phase="train",
        image_size=(image_size, image_size),
        normalize=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# -----------------------------------------------------------------------------
# Core evaluation routine                                                       
# -----------------------------------------------------------------------------

def test_on_target(args, device, model):
    print("\n" + "=" * 40)
    print(f"🧪 开始在目标域上测试数据集: {os.path.basename(args.target_dir)}")
    print("=" * 40 + "\n")

    # ---------------- Paths --------------------------------------------------
    result_dir = os.path.join(args.checkpoint_dir, "tta2d_results")
    weights_dir = os.path.join(result_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # ---------------- Data ---------------------------------------------------
    test_loader = build_test_loader(
        target_dir=args.target_dir,
        batch_size=args.batch_test,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # ---------------- Eval loop ---------------------------------------------
    metric_lists: Dict[str, List[float]] = {k: [] for k in ["dice", "iou", "sensitivity", "ppv", "hd95"]}
    model.eval()  # Ensure eval mode

    with torch.no_grad():
        for imgs, labels, _ in tqdm(test_loader, desc="推理进度"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            seg_output, _ = model(imgs, 0)  # 测试时 alpha 设为 0
            seg_output = torch.softmax(seg_output, dim=1)

            for i in range(imgs.shape[0]):
                m = calculate_all_metrics(seg_output[i:i + 1], labels[i:i + 1])
                for k in metric_lists:
                    metric_lists[k].append(safe_value(m[k]))

    # ---------------- Stats --------------------------------------------------
    metric_mean = {k: float(np.mean(v)) for k, v in metric_lists.items()}
    metric_std = {k: float(np.std(v)) for k, v in metric_lists.items()}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------- Report -------------------------------------------------
    report_lines = [
        "=" * 40,
        f"测试时间: {timestamp}",
        f"目标数据集: {args.target_dir}",
        f"模型架构: {args.model_type}",
        "\n性能指标 (均值 ± 标准差):",
    ]
    for k in metric_lists:
        report_lines.append(f"{k.upper()}: {metric_mean[k]:.4f} ± {metric_std[k]:.4f}")
    report_lines.append("=" * 40)
    report = "\n".join(report_lines)
    model_tag = 'pass_2D'
    result_file = f"{model_tag}.txt"
    with open(result_file, "a") as f:
        f.write(report)

    print(report)
    return True


def train_on_target(args, device, epoch):
    print("\n" + "=" * 40)
    print(f"🧪 开始在目标域上训练数据集: {os.path.basename(args.target_dir)}")
    print("=" * 40 + "\n")

    # ---------------- Paths --------------------------------------------------
    result_dir = os.path.join("pass_2D", "tta2d_results")
    weights_dir = os.path.join(result_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # ---------------- Model --------------------------------------------------
    model = DAUNet2d(in_channels=1, n_classes=args.num_classes).to(device)

    ckpt_path = args.model_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到预训练权重: {ckpt_path}")

    print(f"加载模型权重: {ckpt_path}")
    checkpoint_obj = robust_torch_load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(checkpoint_obj)
    model.unet.load_state_dict(state_dict, strict=True)

    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=0.0001)
    segmentation_loss = DiceLoss().to(device)
    domain_classifier_loss = nn.BCELoss()
    m = nn.Sigmoid()
    loss_name, loss_params = LSIZE_Prostate
    loss_class = getattr(moment_tta_losses, loss_name)
    loss_fn = loss_class(**loss_params)


    # ---------------- Data ---------------------------------------------------
    train_loader = build_train_loader(
        target_dir=args.target_dir,
        batch_size=args.batch_test,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # ---------------- Eval loop ---------------------------------------------
    for i in range(epoch):
        train_bar = tqdm(train_loader, desc="训练进度")
        len_dataloader = len(train_loader)
        for step, batch in enumerate(train_bar):
            imgs, labels, _ = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            p = float(step + i * len_dataloader) / epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            pred_seg, pred_domain = model(imgs, alpha)
            # loss_seg = segmentation_loss(pred_seg, labels)
            # target_domain_label = pred_domain  
            # m_p = m(pred_domain)
            # 如果有非1的数，就变0
            # 1. 生成掩码（超出范围的位置为 True）
            # mask_tdl = (target_domain_label > 1.0) | (target_domain_label < 0.0)
            # # 2. 用 torch.where 生成新张量（不修改原张量）
            # target_domain_label = torch.where(
            #     mask_tdl, 
            #     torch.tensor(0.0, device=device),  # 超出范围时设为 0
            #     target_domain_label                # 否则保留原值
            # )
            # # 1. 生成掩码（超出范围的位置为 True）
            # mask_mp = (m_p > 1.0) | (m_p < 0.0)
            # # 2. 用 torch.where 生成新张量（不修改原张量）
            # m_p = torch.where(
            #     mask_mp, 
            #     torch.tensor(0.0, device=device),  # 超出范围时设为 0
            #     m_p                                # 否则保留原值
            # )
                        # print(m_p)
            # print(target_domain_label)

            # print(m(pred_domain).shape)
            # print(target_domain_label.shape)
            # loss_dc = domain_classifier_loss(m_p, target_domain_label)

            loss_dc = loss_fn(pred_seg)

            total_loss = loss_dc
            total_loss.backward()
            optimizer.step()

            train_bar.set_description(f"Training: Loss: {total_loss.item():.4f}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = "pass_2D"

        adapted_path = os.path.join(weights_dir, f"{model_tag}_tta2d_{args.dataset}_adapted_{timestamp}.pth")
        torch.save(model.state_dict(), adapted_path)
        print(f"✅ 已保存适应后的模型权重: {adapted_path}")
        if (i+1) % 3 == 0:
            test_on_target(args, device, model)

    return True


# -----------------------------------------------------------------------------
# Main                                                                         
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Test-Time Adaptation (Tent) Evaluation Script")

    # Data paths -------------------------------------------------------------
    parser.add_argument("--dataset", type=str, default="PATH",
                        help="Name of the dataset.")
    parser.add_argument("--target_dir", type=str, default="/home/yuwenjing/data/tta_dataset/TTA-2DPATH/Glas_processed",
                        help="包含 image/ 和 mask/ 的目标域文件夹")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints-pass", help="保存 / 查找权重的目录")
    parser.add_argument("--model_path", type=str, default="/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoint_PATH/unet2d_best_PATH.pth",
                        help="自定义权重路径，`default` 表示自动选择")

    # Model & TTA ------------------------------------------------------------
    parser.add_argument("--model_type", type=str, default="unet2d", choices=["unet2d", "nnunet2d"],
                        help="模型架构类型")
    parser.add_argument("--num_classes", type=int, default=2, help="输出类别数 (含背景)")
    parser.add_argument("--lr", type=float, default=1e-5, help="适应阶段学习率")
    parser.add_argument("--adapt_steps", type=int, default=1, help="每批适应步数 (Tent steps)")
    parser.add_argument("--episodic", action="store_true", help="是否启用 episodic Tent (每批重置)")

    # Dataloader -------------------------------------------------------------
    parser.add_argument("--batch_test", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256, help="测试时图像缩放大小")

    # Runtime ----------------------------------------------------------------
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号 (-1 表示 CPU)")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    epoch = 30
    try:
        train_on_target(args, device, epoch)
    except Exception as e:
        err_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print("🔥 运行失败:", str(e))
        traceback.print_exc()
        err_dir = os.path.join(args.checkpoint_dir, "tta2d_results")
        os.makedirs(err_dir, exist_ok=True)
        with open(os.path.join(err_dir, "tta2d_errors.log"), "a") as f:
            f.write(f"[{err_time}] {traceback.format_exc()}\n")