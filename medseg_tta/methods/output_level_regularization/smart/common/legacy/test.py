from unet3d import UNet3d, TAdaBN3D
from autoaugment import LearnableImageNetPolicy
import torch
import os
from metrics import *
from augmodel import Augmentmodel
import numpy as np
from utils_brats_all import get_data_loader, parse_config, set_random
import monai.losses as losses
dice_loss = losses.DiceLoss()
from loss import CombinedLoss
from shape_analysis_functions import cnh_loss,ih_loss
import torch.nn.functional as F
import argparse
from test_targetCT import get_ct_test_loader

def test(config, upl_model, test_loader, exp_name, device, epoch=0):
    def to_float(x):
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
        else:
            return float(x)

    for data_loader in [test_loader]:  # 只对测试集进行评估
        # 保存每个 batch 的 WT 指标
        dice_wt_list = []
        hd95_list_wt = []
        IoU_list_wt = []
        sen_list_wt = []
        ppv_list_wt = []

        with torch.no_grad():
            upl_model.eval()
            skip_num = 0
            for it, (image, label, xt_name) in enumerate(test_loader):
                image = image.to(device, non_blocking=True)
                label = label.long().to(device, non_blocking=True).squeeze(1) 

                aux_seg_1 = upl_model.forward(image)

                # Dice (dice3 = WT)
                dice1, dice2, dice3 = cal_dice(aux_seg_1, label)  
                dice_wt_list.append(to_float(dice3))

                # HD95
                hd95_ec, hd95_co, hd95_wt = cal_hd95(aux_seg_1, label)
                hd95_list_wt.append(to_float(hd95_wt))

                # IoU
                IoU_ec, IoU_co, IoU_wt = IoU(aux_seg_1, label)
                IoU_list_wt.append(to_float(IoU_wt))

                # Sensitivity
                sen_ec, sen_co, sen_wt = cal_sensitivity(aux_seg_1, label)
                sen_list_wt.append(to_float(sen_wt))

                # PPV
                ppv_ec, ppv_co, ppv_wt = cal_ppv(aux_seg_1, label)
                ppv_list_wt.append(to_float(ppv_wt))
                #print(dice3,hd95_wt,IoU_wt,sen_wt,ppv_wt)

        # ===== 计算均值 & 标准差 =====
        avg_dice_wt = np.nanmean(dice_wt_list)
        std_dice_wt = np.nanstd(dice_wt_list)

        avg_hd95_wt = np.nanmean(hd95_list_wt)
        std_hd95_wt = np.nanstd(hd95_list_wt)

        avg_IoU_wt = np.nanmean(IoU_list_wt)
        std_IoU_wt = np.nanstd(IoU_list_wt)

        avg_sen_wt = np.nanmean(sen_list_wt)
        std_sen_wt = np.nanstd(sen_list_wt)

        avg_ppv_wt = np.nanmean(ppv_list_wt)
        std_ppv_wt = np.nanstd(ppv_list_wt)

        # ===== 输出结果 =====
        output_result = []
        output_result.append(f"WT_Dice_mean : {avg_dice_wt}")
        output_result.append(f"WT_Dice_std  : {std_dice_wt}")
        output_result.append(f"WT_HD95_mean : {avg_hd95_wt}")
        output_result.append(f"WT_HD95_std  : {std_hd95_wt}")
        output_result.append(f"WT_IoU_mean  : {avg_IoU_wt}")
        output_result.append(f"WT_IoU_std   : {std_IoU_wt}")
        output_result.append(f"WT_sen_mean  : {avg_sen_wt}")
        output_result.append(f"WT_sen_std   : {std_sen_wt}")
        output_result.append(f"WT_ppv_mean  : {avg_ppv_wt}")
        output_result.append(f"WT_ppv_std   : {std_ppv_wt}")

        results_dir = f"results_comparison/mia"
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'benchmark_CT_3.txt'), 'a') as file:
            for line in output_result:
                file.write(line + "\n")

        # 同时打印到终端
        for line in output_result:
            print(line)

        return avg_dice_wt



                
def get_model_module(model):
    """
    动态获取模型的访问方式，支持单 GPU 和多 GPU。
    """
    return model.module if isinstance(model, torch.nn.DataParallel) else model

def momentum_update_key_encoder(model, momentum_model):
    """
    Momentum update of the key encoder
    """
    # encoder_q -> encoder_k
    for param_q, param_k in zip(
        model.parameters(), momentum_model.parameters()
    ):
        param_k.data = param_k.data * 0.95 + param_q.data * 0.05
    return momentum_model

def total_variation_loss_3d(pred):
    """
    pred: [B, C, D, H, W] - softmax输出
    惩罚预测体积中相邻体素间的跳变
    """
    dz = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
    dy = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
    dx = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
    return (dz.mean() + dy.mean() + dx.mean())

def gradient_loss_3d(pred):
    """
    pred: [B, C, D, H, W] - softmax输出
    用三维一阶差分计算梯度
    """
    dz = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    dx = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    return (dz.abs().mean() + dy.abs().mean() + dx.abs().mean())

def compactness_loss_3d(pred, epsilon=1e-6):
    """
    pred: [B, C, D, H, W] - softmax输出
    近似方式：用TV估算表面积，面积用预测体素求和
    """
    volume = torch.sum(pred, dim=[2, 3, 4]) + epsilon  # [B, C]
    surface = total_variation_loss_3d(pred) * pred.shape[2] * pred.shape[3] * pred.shape[4]
    loss = torch.mean(surface / volume)
    return loss
    
def train(config,train_loader,test_loader,source_model):
    print("train")
    exp_name = config['train']['exp_name']
    dataset = config['train']['dataset']
    device = config['train']['gpu']
    checkpoint = torch.load(source_model)
    upl_model = UNet3d().to(device)
    upl_model.load_state_dict(checkpoint['model_state_dict'])
    momentum_model = UNet3d().to(device)
    momentum_model.load_state_dict(checkpoint['model_state_dict'])
    print('source_model_created')
    class_weights = torch.tensor([1.0, 2.0, 1.5, 1.5], device=device)
    dice_reduction='macro'
    criterion = CombinedLoss(
        ce_weight=3.0,
        dice_weight=2.0,
        dice_reduction=dice_reduction,
        class_weights=class_weights,
        device=device
    )
    dice_loss = losses.DiceLoss(to_onehot_y=True, softmax=True)
    aug_num = 1
    print('source_model_loaded')
    aug_model = Augmentmodel(upl_model).to(device)
    aug_momentum_model = Augmentmodel(momentum_model).to(device)
    #test(config,upl_model,test_loader,exp_name=exp_name,device = device,epoch = 0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, aug_model.parameters()),lr=config['train']['lr'])
    num_epochs = 1000
    best_dice = 0.
    output_dir = "validation_results_ssa-t2f"
    os.makedirs(output_dir,exist_ok=True)
    train_flag = True
    aug_model = aug_model.to(device)
    for epoch in range(num_epochs):
        #print(epoch)
        if train_flag:
            pesudo_labels = []
            with torch.no_grad():
                for i, (B, B_label,  _) in enumerate(train_loader):
                    
                    #raise ValueError(B.shape,B_label.shape)
                    B = B.to(device)
                    pesudo_label = aug_momentum_model(B, aug = 0)
                    pesudo_labels.append(pesudo_label.detach())
            aug_model.train()
            for module in aug_model.modules():
                if isinstance(module, TAdaBN3D) or isinstance(module, LearnableImageNetPolicy):
                    module.train()
                else:
                    module.eval()
            for i, (B, B_label, _) in enumerate(train_loader):
                B_label = B_label.long().squeeze(1).to(device)
                B = B.to(device)
                optimizer.zero_grad() 
                for j in range(aug_num):
                    out = aug_model(B, aug = 0)
                    #out, weight = aug_model(B, aug = 1)
                    #weight = weight.to(device)
                    total_loss = torch.tensor(0.0).to(device)
                    
                    out = aug_model(B, aug = 0)
                   
                    for k in range(out.size(0)):
                        input_i = out[k].unsqueeze(0)
                        #print("input_i:", input_i.shape, "B_label:", B_label.shape, B_label.dtype)
                        loss_dice = dice_loss(input_i, B_label.unsqueeze(1)) 
                        loss_ao = ih_loss(input_i) / 10
                        loss_tu = cnh_loss(input_i) / 5
                        #weighted_loss_i = (loss_ao + loss_tu + loss_dice) * weight[k]
                        weighted_loss_i = (loss_ao + loss_tu + loss_dice)
                        #weighted_loss_i = loss_ao + loss_tu
                        print(loss_dice,loss_ao,loss_tu)
                        total_loss += weighted_loss_i
                    if total_loss != 0:
                        total_loss.backward() 
                optimizer.step()
                
        # # valid for target domain
        if (epoch+1) % 1 == 0:
            current_dice = test(config,upl_model,test_loader,exp_name=exp_name,device = device,epoch = epoch)
            if (current_dice) > best_dice:
                best_dice = current_dice
                model_dir = "/data/birth/cyf/output/wyh_output/tta/new_origin_5/" + str(exp_name )
                os.makedirs(model_dir, exist_ok=True)
                best_epoch = '{}/model-{}-{}-{}.pth'.format(model_dir, 'best', str(epoch), np.round(best_dice,3))
                torch.save(upl_model.state_dict(), best_epoch)
            model_dir = "/data/birth/cyf/output/wyh_output/tta/new_origin_5/" + str(exp_name )
            os.makedirs(model_dir, exist_ok=True)
            best_epoch = '{}/model-{}.pth'.format(model_dir, str(epoch))
            torch.save(upl_model.state_dict(), best_epoch)   
        momentum_model = momentum_update_key_encoder(upl_model,momentum_model)
    if train_flag and (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), '{}/model-{}.pth'.format(model_dir, 'latest'))

    upl_model.load_state_dict(torch.load(best_epoch,map_location='cpu'),strict=True)
    upl_model.eval()
    print("test")
    test(config,upl_model,test_loader,exp_name=exp_name,device = device)
    
def main():
    # load config
    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--config', type=str, default="./config/train3d.cfg",
                        help='Path to the configuration file')
    args = parser.parse_args()
    config = args.config
    config = parse_config(config)
    #source_model = '/home/yuwenjing/DeepLearning_ywj/tta/unet3d_best.pth'
    source_model = '/home/jiangshuo/WYH/SmaRT/unet3d_best_CT.pth'
    batch_train = 1
    batch_test = 1
    num_workers = 0
    source_root = '/FM_data/cyf/tta_data/BraTS2024'
    target_root = '/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB'
    train_path = 'train'
    test_path = 'test'
    mode = 'target_to_target'
    img = 'all'
    train_loader,test_loader = get_data_loader(source_root,target_root,
                                               train_path,test_path,
                                               batch_train,batch_test,
                                               nw = num_workers,
                                               img=img,mode=mode)
    target_test_loader, _ = get_ct_test_loader(
            image_dir='/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB/image',
            mask_dir='/home/yuwenjing/data/tta_dataset/TTA-3DCT/3D-IRCADB/mask',
            target_dir=None,
            dataset_type=None,
            batch_size=batch_train,
            num_workers=num_workers,
            image_size=(128,128,128),
            spacing=(1.0, 1.0, 1.0),
            intensity_range=(-200, 400)
        )
    print("数据加载完成")

    train(config,target_test_loader,target_test_loader,source_model)
        
if __name__ == '__main__':
    
    set_random()
    torch.manual_seed(0.95)
    torch.cuda.manual_seed(0.95) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    main()