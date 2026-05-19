import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
from tqdm import tqdm
from utils_u import parse_config, set_random
from utils_brats_all import get_data_loader
from utils.utils import set_requires_grad, setup_seed
from utils.transforms import random_flip_rotate
from utils.fft import FDA_source_to_target
from unet3d_brats import UNet3d
import argparse
from loss_brats import CombinedLoss
from utils.loss import DiceLoss
from metrics import cal_hd95, cal_dice, cal_RVE, IoU, PA, cal_sensitivity, cal_ppv
import torch.nn as nn
from einops import rearrange, reduce

# 设置随机种子保证结果可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 熵置信度掩码，用于生成伪标签
def entropy_confidence_mask(logits, th=0.1):
    prob = torch.softmax(logits, dim=1)
    entropy = torch.sum(-prob * torch.log(prob + 1e-10), dim=1).detach()
    mask = entropy.ge(th)
    return mask

# 将标签转换为one-hot编码
def to_one_hot(label, num_classes=7):
    b, h, w, d = label.shape
    label = rearrange(label, 'b h w d -> b h w d')
    label = label.unsqueeze(1)
    label = torch.zeros(b, num_classes, h, w, d, dtype=torch.int64).cuda().scatter_(1, label, 1)
    return label
def tensor2numpy(tensor_list):
    return [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in tensor_list]
# 对称交叉熵损失函数
def symmetric_cross_entropy(logits, label, alpha=1, beta=1):
    ce_loss = nn.CrossEntropyLoss()(logits, label)
    pred = torch.softmax(logits, dim=1)
    label_one_hot = to_one_hot(label, 7)
    label_one_hot = torch.clamp(label_one_hot.float(), min=1e-4, max=1.0)
    rce_loss = torch.mean(torch.sum(-pred * torch.log(label_one_hot), dim=1))
    return alpha * ce_loss + beta * rce_loss

# 组合损失函数类
class Criterion(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super(Criterion, self).__init__()
        self.dice_loss = DiceLoss(num_classes, ignore_index=ignore_index)

    def forward(self, logits, label):
        return self.dice_loss(logits, label) + nn.CrossEntropyLoss()(logits, label)

# 将标签二值化（用于正负样本分离）
def to_one(label, num_classes=7):
    label = rearrange(label, 'b 1 h w d -> b 1 h w d')
    label = torch.where(label != 0, 1, 0)
    return label.float()

# 正负样本处理模块
class PosNeg(nn.Module):
    def __init__(self, input_nc, ndf=64, num_classes=7):
        super(PosNeg, self).__init__()
        self.proto_projection = nn.Sequential(
            nn.Conv3d(input_nc, ndf, kernel_size=1),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True))
        self.proto_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten())
        self.proto_D = nn.Sequential(
            nn.Conv3d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls = nn.Conv3d(ndf * 2, num_classes, kernel_size=1, stride=1)

    def forward(self, fea, label):
        # 调整标签尺寸以匹配特征图
        if label.shape[2:] != fea.shape[2:]:
            label = torch.nn.functional.interpolate(
                label.float(),
                size=fea.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        mask_pos = label.cuda()
        mask_neg = 1. - mask_pos
        
        # 应用掩码到分类结果
        out_pos = self.cls(self.proto_D(fea.cuda())) * mask_pos
        out_neg = self.cls(self.proto_D(fea.cuda())) * mask_neg
        
        return out_pos, out_neg

# 像素判别器（用于域适应）
class PixelDiscriminator_(nn.Module):
    def __init__(self, input_nc, ndf=128, num_classes=7):
        super(PixelDiscriminator_, self).__init__()
        self.D = nn.Sequential(
            nn.Conv3d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ndf, ndf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls1 = nn.Conv3d(ndf // 2, num_classes, kernel_size=1, stride=1)
        self.cls2 = nn.Conv3d(ndf // 2, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.D(x)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        out = torch.cat((src_out, tgt_out), dim=1)
        return out

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

# 学生模型评估函数
def evaluate_student(model, data_loader, device, num_classes=4):
    model.eval()
    # 使用字典存储所有批次的指标值
    metrics = {
        'dice1': [], 'dice2': [], 'dice3': [],
        'hd95_ec': [], 'hd95_co': [], 'hd95_wt': [],
        'RVE_ec': [], 'RVE_co': [], 'RVE_wt': [],
        'iou_ec': [], 'iou_co': [], 'iou_wt': [],
        'pa_ec': [], 'pa_co': [], 'pa_wt': [],
        'sensitivity_ec': [], 'sensitivity_co': [], 'sensitivity_wt': [],
        'ppv_ec': [], 'ppv_co': [], 'ppv_wt': []
    }
    
    with torch.no_grad():
        for image, label, _, _ in tqdm(data_loader, desc="Validation"):
            image = image.to(device)
            label = label.long().squeeze(1).to(device)
            
            blocks, fea = model.enc(image)
            logits = model.aux_dec1(fea, blocks)
            
            # 计算各项评估指标
            dice1, dice2, dice3 = cal_dice(logits, label)
            hd95_ec, hd95_co, hd95_wt = cal_hd95(logits, label)
            RVE_ec, RVE_co, RVE_wt = cal_RVE(logits, label)
            iou_ec, iou_co, iou_wt = IoU(logits, label)
            pa_ec, pa_co, pa_wt = PA(logits, label, num_classes)
            sensitivity_ec, sensitivity_co, sensitivity_wt = cal_sensitivity(logits, label)
            ppv_ec, ppv_co, ppv_wt = cal_ppv(logits, label)
            
            # 存储每个批次的指标（而非累加）
            metrics['dice1'].append(dice1)
            metrics['dice2'].append(dice2)
            metrics['dice3'].append(dice3)
            metrics['hd95_ec'].append(hd95_ec)
            metrics['hd95_co'].append(hd95_co)
            metrics['hd95_wt'].append(hd95_wt)
            metrics['RVE_ec'].append(RVE_ec)
            metrics['RVE_co'].append(RVE_co)
            metrics['RVE_wt'].append(RVE_wt)
            metrics['iou_ec'].append(iou_ec)
            metrics['iou_co'].append(iou_co)
            metrics['iou_wt'].append(iou_wt)
            metrics['pa_ec'].append(pa_ec)
            metrics['pa_co'].append(pa_co)
            metrics['pa_wt'].append(pa_wt)
            metrics['sensitivity_ec'].append(sensitivity_ec)
            metrics['sensitivity_co'].append(sensitivity_co)
            metrics['sensitivity_wt'].append(sensitivity_wt)
            metrics['ppv_ec'].append(ppv_ec)
            metrics['ppv_co'].append(ppv_co)
            metrics['ppv_wt'].append(ppv_wt)
    
    # 计算均值和标准差（使用numpy）
    import numpy as np
    results = {}
    for key, values in metrics.items():
        values =tensor2numpy(values)
    
        avg = np.mean(values)
        std = np.std(values)
        results[key] = (avg, std)  # 存储格式：(均值, 标准差)
    
    # 整理返回结果（保持原有结构，但每个指标包含均值和标准差）
    return {
        'dice': (
            (results['dice1'][0], results['dice1'][1]),
            (results['dice2'][0], results['dice2'][1]),
            (results['dice3'][0], results['dice3'][1])
        ),
        'hd95': (
            (results['hd95_ec'][0], results['hd95_ec'][1]),
            (results['hd95_co'][0], results['hd95_co'][1]),
            (results['hd95_wt'][0], results['hd95_wt'][1])
        ),
        'rve': (
            (results['RVE_ec'][0], results['RVE_ec'][1]),
            (results['RVE_co'][0], results['RVE_co'][1]),
            (results['RVE_wt'][0], results['RVE_wt'][1])
        ),
        'iou': (
            (results['iou_ec'][0], results['iou_ec'][1]),
            (results['iou_co'][0], results['iou_co'][1]),
            (results['iou_wt'][0], results['iou_wt'][1])
        ),
        'pa': (
            (results['pa_ec'][0], results['pa_ec'][1]),
            (results['pa_co'][0], results['pa_co'][1]),
            (results['pa_wt'][0], results['pa_wt'][1])
        ),
        'sensitivity': (
            (results['sensitivity_ec'][0], results['sensitivity_ec'][1]),
            (results['sensitivity_co'][0], results['sensitivity_co'][1]),
            (results['sensitivity_wt'][0], results['sensitivity_wt'][1])
        ),
        'ppv': (
            (results['ppv_ec'][0], results['ppv_ec'][1]),
            (results['ppv_co'][0], results['ppv_co'][1]),
            (results['ppv_wt'][0], results['ppv_wt'][1])
        )
    }

# 学生模型训练函数
def train_student(config, train_loader, valid_loader):
    print("Starting student model training...")
    
    # 加载配置参数
    exp_name = 'brats'
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化学生模型
    st_model = UNet3d(config).to(device)
    
    # 加载预训练权重（如果需要）
    resume_path = r"/home/zhengjingyuan/JS/DLTTA-3d/experiments/prostate/unet3d_best.pth"
    if resume_path and os.path.exists(resume_path):
        st_model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"Loaded pre-trained model from {resume_path}")
    
    # 定义损失函数和优化器
    class_weights = torch.tensor([0.05, 2.0, 0.1, 0.5], device=device)
    criterion = CombinedLoss(
        ce_weight=2.0,
        dice_weight=3.0,
        dice_reduction='macro',
        class_weights=class_weights,
        device=device
    )
    optimizer = torch.optim.SGD(st_model.parameters(), lr=0.0001, 
                               momentum=0.9, weight_decay=5e-4)
    
    # 域适应相关组件
    MI = PosNeg(256).to(device)
    D = PixelDiscriminator_(256).to(device)
    loss_MI = DeepInfoMaxLoss(type="conv")
    lambda_adv = 0.1
    lambda_D = 0.1
    
    # 训练参数
    num_epochs = 100
    output_dir = os.path.join('output', exp_name)
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 日志文件
    results_file = os.path.join(output_dir, "evaluation_results-PED.txt")
    
    # 最佳模型记录
    best_dice = 0.0
    train_flag = True
    # 开始训练
    for epoch in range(num_epochs):
        if train_flag ==True:
            st_model.train()
            D.train()
            MI.train()
            
            train_loss = 0.0
            progress_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
            source_iter = enumerate(train_loader)
            for i, (tgt_img, tgt_label, src_img, _) in enumerate(progress_bar):
                # 数据准备
                tgt_img = tgt_img.to(device)
                tgt_label = tgt_label.cuda().squeeze(1).long()
                _, inputs = source_iter.__next__()
                src_img, src_label, *_ = inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()
                    # tgt_img, tgt_label = tgt_img.cuda(), tgt_label.cuda().squeeze(1).long()
                
                # 清空梯度
                optimizer.zero_grad()
                
                # 1. 训练学生模型对目标域的分割能力
                blocks, tgt_fea = st_model.enc(tgt_img)
                tgt_logits = st_model.aux_dec1(tgt_fea, blocks)
                
                # 生成伪标签
                mask = entropy_confidence_mask(tgt_logits, 0.1)
                tgt_pseudo_label = tgt_logits.max(1).indices
                tgt_pseudo_label[torch.where(mask)] = 255  # 低置信度像素设为忽略
                
                # 计算分割损失
                # loss_seg = criterion(tgt_logits, tgt_pseudo_label)
                
                # 2. 域适应处理 - 特征对齐
                # FDA域适应
                tgt_img_aug = FDA_source_to_target(tgt_img, src_img)
                img_aug_min = reduce(tgt_img_aug, 'b c h w d -> b c 1 1 1', 'min')
                img_aug_max = reduce(tgt_img_aug, 'b c h w d -> b c 1 1 1', 'max')
                tgt_img_aug = (tgt_img_aug - img_aug_min) / (img_aug_max - img_aug_min)
                
                # 教师模型对增强目标域的输出（假设已预训练）
                blocks_aug, tgt_fea_aug = st_model.enc(tgt_img_aug)
                tgt_logits_aug = st_model.aux_dec1(tgt_fea_aug, blocks_aug)
                tgt_pseudo_aug = tgt_logits_aug.max(1).indices
                tgt_pseudo_aug = tgt_pseudo_aug.unsqueeze(1)
                tgt_pseudo_aug_one = to_one(tgt_pseudo_aug, 4)
                
                # 3. 互信息损失计算
                tgt_pos, tgt_neg = MI(tgt_fea_aug, tgt_pseudo_aug_one)
                src_pos, src_neg = MI(tgt_fea, tgt_pseudo_aug_one)
                loss_mu = 0.5 * loss_MI(src_pos, src_neg, tgt_pos) + 0.5 * loss_MI(src_neg, src_pos, tgt_neg)
                
                # # 4. 对抗损失计算
                # tgt_D_pred = D(tgt_fea_aug)
                # print(tgt_D_pred.shape)
                # tgt_pseudo_label_onehot = to_one_hot(tgt_pseudo_label, 4)
                # print('dada',tgt_pseudo_label_onehot.shape)
                # loss_adv = lambda_adv * nn.CrossEntropyLoss()(
                #     tgt_D_pred, 
                #     torch.cat((tgt_pseudo_label_onehot, torch.zeros_like(tgt_pseudo_label_onehot)), dim=1).argmax(dim=1)
                # )
                
                # 总损失
                total_loss =  loss_mu #+ 0.1 * loss_adv
                total_loss.backward()
                optimizer.step()
                
                progress_bar.desc = f"Epoch {epoch+1}/{num_epochs} Loss: {total_loss.item():.4f}"
                train_loss += total_loss.item()
        
        # 每个epoch结束后进行验证

        if (epoch + 1) % 1 == 0:
            train_flag = True
            print("\nPerforming validation...")
            val_results = evaluate_student(st_model, valid_loader, device)
            
            # 计算平均Dice（仅使用均值）
            avg_dice = sum([val_results['dice'][i][0] for i in range(3)]) / 3
            
            # 保存验证结果（包含均值和标准差）
            with open(results_file, 'a') as f:
                f.write(f"Epoch {epoch+1}/{num_epochs}\n")
                f.write(f"Val Dice: ET {val_results['dice'][0][0]:.4f}±{val_results['dice'][0][1]:.4f}, "
                        f"TC {val_results['dice'][1][0]:.4f}±{val_results['dice'][1][1]:.4f}, "
                        f"WT {val_results['dice'][2][0]:.4f}±{val_results['dice'][2][1]:.4f}\n")
                f.write(f"Val HD95: ET {val_results['hd95'][0][0]:.4f}±{val_results['hd95'][0][1]:.4f}, "
                        f"TC {val_results['hd95'][1][0]:.4f}±{val_results['hd95'][1][1]:.4f}, "
                        f"WT {val_results['hd95'][2][0]:.4f}±{val_results['hd95'][2][1]:.4f}\n")
                f.write(f"Val RVE: ET {val_results['rve'][0][0]:.4f}±{val_results['rve'][0][1]:.4f}, "
                        f"TC {val_results['rve'][1][0]:.4f}±{val_results['rve'][1][1]:.4f}, "
                        f"WT {val_results['rve'][2][0]:.4f}±{val_results['rve'][2][1]:.4f}\n")
                f.write(f"Val IoU: ET {val_results['iou'][0][0]:.4f}±{val_results['iou'][0][1]:.4f}, "
                        f"TC {val_results['iou'][1][0]:.4f}±{val_results['iou'][1][1]:.4f}, "
                        f"WT {val_results['iou'][2][0]:.4f}±{val_results['iou'][2][1]:.4f}\n")
                f.write(f"Val PA: ET {val_results['pa'][0][0]:.4f}±{val_results['pa'][0][1]:.4f}, "
                        f"TC {val_results['pa'][1][0]:.4f}±{val_results['pa'][1][1]:.4f}, "
                        f"WT {val_results['pa'][2][0]:.4f}±{val_results['pa'][2][1]:.4f}\n")
                f.write(f"Val Sensitivity: ET {val_results['sensitivity'][0][0]:.4f}±{val_results['sensitivity'][0][1]:.4f}, "
                        f"TC {val_results['sensitivity'][1][0]:.4f}±{val_results['sensitivity'][1][1]:.4f}, "
                        f"WT {val_results['sensitivity'][2][0]:.4f}±{val_results['sensitivity'][2][1]:.4f}\n")
                f.write(f"Val PPV: ET {val_results['ppv'][0][0]:.4f}±{val_results['ppv'][0][1]:.4f}, "
                        f"TC {val_results['ppv'][1][0]:.4f}±{val_results['ppv'][1][1]:.4f}, "
                        f"WT {val_results['ppv'][2][0]:.4f}±{val_results['ppv'][2][1]:.4f}\n")
    # return {
    #     'dice': (avg_dice1, avg_dice2, avg_dice3),
    #     'hd95': (avg_hd95_ec, avg_hd95_co, avg_hd95_wt),
    #     'rve': (avg_RVE_ec, avg_RVE_co, avg_RVE_wt),
    #     'iou': (avg_iou_ec, avg_iou_co, avg_iou_wt),
    #     'pa': (avg_pa_ec, avg_pa_co, avg_pa_wt),
    #     'sensitivity': (avg_sensitivity_ec, avg_sensitivity_co, avg_sensitivity_wt),
    #     'ppv': (avg_ppv_ec, avg_ppv_co, avg_ppv_wt)
    # }
            
            # 保存最佳模型
            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save(st_model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                print(f"Best model saved with Dice: {best_dice:.4f}")
            
            # 保存当前epoch模型
            torch.save(st_model.state_dict(), os.path.join(model_dir, f"epoch_{epoch+1}.pth"))
    
    print(f"Training completed. Best validation Dice: {best_dice:.4f}")
    return st_model

# 主函数
def main():
    
    # load config
    parser = argparse.ArgumentParser(description='config file')
    default_config = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'common', 'legacy', 'config', 'train3d.cfg')
    )
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to the configuration file')
    args = parser.parse_args()
    config = args.config
    config = parse_config(config)
    list_data = []
    # print(config)
    dataset = 'brats'

    if dataset == 'brats':

        batch_train = 1
        batch_test = 1
        num_workers = 0
        source_root = r"/home/yuwenjing/data/BraTS2024/train"
        # target_root = r"/home/yuwenjing/data/BraTS-SSA"
        target_root = r"/home/yuwenjing/data/BraTS-PED2023/Train"
        train_path = ''
        test_path = ''
        mode = 'source_to_target'
        # mode should be 'source_to_source' or 'source_to_target' or 'target_to_target
        img = 'all'
        train_loader, test_loader = get_data_loader(source_root=source_root,
                                                    target_root=target_root,
                                                    train_path=train_path,
                                                    test_path=test_path,
                                                    batch_train=batch_train,
                                                    batch_test=batch_test,
                                                    nw=num_workers,
                                                    img=img,
                                                    mode=mode)
    
    # 训练学生模型
    train_student(config, train_loader, test_loader)

if __name__ == '__main__':
    main()
