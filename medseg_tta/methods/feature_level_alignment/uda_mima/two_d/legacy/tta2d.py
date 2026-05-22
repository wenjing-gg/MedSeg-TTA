import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import datetime
import traceback
from typing import Dict, List
import pickle  # <-- for catching UnpicklingError
import copy
from pathlib import Path
import torch.nn as nn
from urils_DLTTA.loss import DiceLoss, entropy_loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 新的模型结构
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from utils_uda import FDA_source_to_target

def to_one(label, num_classes=7):
    # print(label.shape)
    label = rearrange(label, 'b 1 h w -> b 1 h w')
    label = torch.where(label != 0, 1, 0)
    return label.float()

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

    def forward(self, x: torch.Tensor,feat = False) -> torch.Tensor:
        blocks, latent = self.encoder(x)
        logits = self.decoder(latent, blocks)

        if feat:
            return logits,latent

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


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "TTA-2DOCT"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "uda_mima"

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
            seg_output = model(imgs)  # 测试时 alpha 设为 0
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
    model_tag = 'UDA_2D'
    result_file = f"{model_tag}.txt"
    with open(result_file, "a") as f:
        f.write(report)

    print(report)
    return True



class PixelDiscriminator_(nn.Module):
    def __init__(self, input_nc, ndf=128, num_classes=7):
        super(PixelDiscriminator_, self).__init__()

        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls1 = nn.Conv2d(ndf // 2, num_classes, kernel_size=1, stride=1)
        self.cls2 = nn.Conv2d(ndf // 2, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.D(x)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        out = torch.cat((src_out, tgt_out), dim=1)
        return out
    

class PosNeg(nn.Module):
    def __init__(self, input_nc, ndf=64, num_classes=7):
        super(PosNeg, self).__init__()
        self.proto_projection = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True))
        self.proto_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1, 1)),
            nn.Flatten())
        self.proto_D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls = nn.Conv2d(ndf * 2, num_classes, kernel_size=1, stride=1)

    def forward(self, fea, label):
        # 调试输出
        # print(f"Input fea shape: {fea.shape}")
        # print(f"Input label shape: {label.shape}")
        
        # 调整标签尺寸以匹配特征图
        if label.shape[2:] != fea.shape[2:]:  # 比较空间维度 (D, H, W)
            label = F.interpolate(
                label.float(), 
                size=fea.shape[2:],  # 使用特征图的空间维度
                mode='bilinear',    
                align_corners=False
            )
            # print(f"Resized label shape: {label.shape}")
        
        mask_pos = label.cuda()
        mask_neg = 1. - mask_pos
        
        # 应用掩码前确保特征图和掩码维度匹配
        out_pos = self.cls(self.proto_D(fea.cuda())) * mask_pos
        out_neg = self.cls(self.proto_D(fea.cuda())) * mask_neg
        
        return out_pos, out_neg
    

# 深度信息最大化损失（用于互信息计算）
class DeepInfoMaxLoss(nn.Module):
    def __init__(self, type="fc"):
        super(DeepInfoMaxLoss, self).__init__()
        self.type = type

    def forward(self, x, y, z):
        if self.type == "fc":
            return -torch.mean(torch.log(torch.sigmoid(torch.sum(x * y, dim=1)) + 1e-6)) - \
                   torch.mean(torch.log(1 - torch.sigmoid(torch.sum(x * z, dim=1)) + 1e-6))
        else:  # conv
            return -torch.mean(torch.log(torch.sigmoid(torch.sum(x * y, dim=1)) + 1e-6)) - \
                   torch.mean(torch.log(1 - torch.sigmoid(torch.sum(x * z, dim=1)) + 1e-6))

# 熵置信度掩码，用于生成伪标签
def entropy_confidence_mask(logits, th=0.1):
    prob = torch.softmax(logits, dim=1)
    entropy = torch.sum(-prob * torch.log(prob + 1e-10), dim=1).detach()
    mask = entropy.ge(th)
    return mask


def train_on_target(args, device, epoch):
    print("\n" + "=" * 40)
    print(f"🧪 开始在目标域上训练数据集: {os.path.basename(args.target_dir)}")
    print("=" * 40 + "\n")

    # ---------------- Paths --------------------------------------------------
    result_dir = os.path.join("UDA_2D", "tta2d_results")
    weights_dir = os.path.join(result_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # ---------------- Model --------------------------------------------------
    model = UNet2d(in_channels=1, n_classes=args.num_classes).to(device)

    ckpt_path = args.model_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到预训练权重: {ckpt_path}")

    print(f"加载模型权重: {ckpt_path}")
    checkpoint_obj = robust_torch_load(ckpt_path, map_location=device)
    state_dict = extract_state_dict(checkpoint_obj)
    model.load_state_dict(state_dict, strict=True)

    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=0.00001)
    segmentation_loss = DiceLoss().to(device)
    domain_classifier_loss = nn.BCELoss()
    m = nn.Sigmoid()
    D = PixelDiscriminator_(256).cuda()
    MI = PosNeg(256).cuda()
    loss_MI = DeepInfoMaxLoss(type="conv")
    # ---------------- Data ---------------------------------------------------
    train_loader = build_train_loader(
        target_dir=args.target_dir,
        batch_size=args.batch_test,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    source_loader = build_test_loader(
        target_dir=args.source_dir,
        batch_size=args.batch_test,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # ---------------- Eval loop ---------------------------------------------
    for i in range(epoch):
        train_bar = tqdm(train_loader, desc="训练进度")
        len_dataloader = len(train_loader)
        source_iter = enumerate(source_loader)
        for step, batch in enumerate(train_bar):
            imgs, labels, _ = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            _, inputs = source_iter.__next__()  # inputs 是一个 batch，结构如 (src_imgs, src_labels, ...)
            src_img, src_label, *_ = inputs      # 直接解包 batch，得到张量 src_img、src_label
            src_img = src_img.to(device)         # 将张量移动到设备
            src_label = src_label.to(device)

            # p = float(step + i * len_dataloader) / epoch / len_dataloader
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            pred_seg,tgt_fea = model(x = imgs,feat = True)
            mask = entropy_confidence_mask(pred_seg, 0.1)
            tgt_pseudo_label = pred_seg.max(1).indices
            tgt_pseudo_label[torch.where(mask)] = 255


            tgt_img_aug = FDA_source_to_target(imgs, src_img)
            img_aug_min = reduce(tgt_img_aug, 'b c h w -> b c 1 1', 'min')
            img_aug_max = reduce(tgt_img_aug, 'b c h w -> b c 1 1', 'max')
            tgt_img_aug = (tgt_img_aug - img_aug_min) / (img_aug_max - img_aug_min)

            tgt_logits_aug, tgt_fea_aug = model(x = tgt_img_aug, feat = True)
            tgt_pseudo_aug = tgt_logits_aug.max(1).indices
            tgt_pseudo_aug = tgt_pseudo_aug.unsqueeze(1)
            tgt_pseudo_aug_one = to_one(tgt_pseudo_aug, 2)

            tgt_pos, tgt_neg = MI(tgt_fea_aug, tgt_pseudo_aug_one)
            src_pos, src_neg = MI(tgt_fea, tgt_pseudo_aug_one)
            loss_mu = 0.5 * loss_MI(src_pos, src_neg, tgt_pos) + 0.5 * loss_MI(src_neg, src_pos, tgt_neg)

            total_loss = loss_mu
            total_loss.backward()
            optimizer.step()

            train_bar.set_description(f"Training: Loss: {total_loss.item():.4f}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = "UDA_2D"

        adapted_path = os.path.join(weights_dir, f"{model_tag}_tta2d_{args.dataset}_adapted_{timestamp}.pth")
        torch.save(model.state_dict(), adapted_path)
        print(f"✅ 已保存适应后的模型权重: {adapted_path}")

        test_on_target(args, device, model)

    return True


# -----------------------------------------------------------------------------
# Main                                                                         
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Test-Time Adaptation (Tent) Evaluation Script")

    # Data paths -------------------------------------------------------------


    dataset_name = "OCT"
    parser.add_argument("--dataset", type=str, default=dataset_name,
                        help="Name of the dataset.")
    parser.add_argument("--source_dir", type=str, default=str(DEFAULT_DATA_ROOT / "Q2_"),
                        help="包含 image/ 和 mask/ 的目标域文件夹")
    parser.add_argument("--target_dir", type=str, default=str(DEFAULT_DATA_ROOT / "Q1"),
                        help="包含 image/ 和 mask/ 的目标域文件夹")
    parser.add_argument("--checkpoint_dir", type=str, default=str(DEFAULT_CHECKPOINT_ROOT), help="保存 / 查找权重的目录")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_CHECKPOINT_ROOT / f"unet2d_best_{dataset_name}.pth"),
                        help="自定义权重路径，`default` 表示自动选择")

    # Model & TTA ------------------------------------------------------------
    parser.add_argument("--model_type", type=str, default="unet2d", choices=["unet2d", "nnunet2d"],
                        help="模型架构类型")
    parser.add_argument("--num_classes", type=int, default=2, help="输出类别数 (含背景)")
    parser.add_argument("--lr", type=float, default=1e-7, help="适应阶段学习率")
    parser.add_argument("--adapt_steps", type=int, default=1, help="每批适应步数 (Tent steps)")
    parser.add_argument("--episodic", action="store_true", help="是否启用 episodic Tent (每批重置)")

    # Dataloader -------------------------------------------------------------
    parser.add_argument("--batch_test", type=int, default=1)
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
