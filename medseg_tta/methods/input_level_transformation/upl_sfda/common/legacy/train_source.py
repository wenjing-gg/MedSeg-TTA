from utils import parse_config, set_random
from unet3d import UNet3d
from unet import UNet

import torch
import matplotlib
import os
import argparse
from utils_brats_all import get_data_loader

import numpy as np
from loss import CombinedLoss
from tqdm import tqdm
from metrics import cal_hd95,cal_dice,cal_RVE,IoU,PA,cal_sensitivity,cal_ppv
#torch.cuda.empty_cache()
from metrics import cal_hd95
matplotlib.use('Agg')


import torch
import torch.nn as nn


def train(config, train_loader, valid_loader, list_data):
    # 加载配置参数
    exp_name = config['train']['exp_name']
    dataset = config['train']['dataset']
    num_classes = 4

    # 设置设备
    device = torch.device('cuda:{}'.format(config['train']['gpu']) if torch.cuda.is_available() else 'cpu')

    # 加载模型
    upl_model = UNet3d(config).to(device)
    
    if resume1:
        upl_model.load_state_dict(torch.load('/root/autodl-tmp/JS/UPL-SFDA-BRATS/okkkk-5-me-model/model-19.pth'))
        print('load model')
    
    # 启用梯度跟踪
    for param in upl_model.parameters():
        param.requires_grad = True

    # 定义损失函数和优化器
    class_weights = torch.tensor([0.05, 2.0, 0.1, 0.5], device=device)
    dice_reduction='macro'
    #数量顺序 0,2,3,1
    criterion = CombinedLoss(
        ce_weight=2.0,
        dice_weight=3.0,
        dice_reduction=dice_reduction,
        class_weights=class_weights,
        device=device
    )
    
    
    optimizer = torch.optim.AdamW(upl_model.parameters(), lr=config['train']['lr'])

    # 设置训练超参数
    num_epochs = 1000
    start_epoch = 0
    valid_epochs = config['train'].get('valid_epoch', 1)
    best_dice = 10000
    output_dir = "okkkk-5-me"
    os.makedirs(output_dir, exist_ok=True)
    validation_results_file = os.path.join(output_dir, "dice.txt")

    model_dir = "okkkk-5-me-model"
    os.makedirs(model_dir, exist_ok=True)


    # 定义训练源函数
    def train_source(image, label):
        upl_model.imgA = image.to(device)
        upl_model.labA = label.long().squeeze(1).to(device)
        upl_model.enc_opt.zero_grad()
        upl_model.aux_dec1_opt.zero_grad()
        
        blocks, latent_A = upl_model.enc(upl_model.imgA)
        upl_model.aux_seg_1 = upl_model.aux_dec1(latent_A, blocks)
        
        loss = criterion(upl_model.aux_seg_1, upl_model.labA)
        loss.backward()
        optimizer.step()
        #dice1, dice2, dice3 = cal_dice(upl_model.aux_seg_1, upl_model.labA)

        upl_model.enc_opt.step()
        upl_model.aux_dec1_opt.step()
        
            
        
        
        return loss.item()
    def evaluate_source(image, label):
    # 将模型切换到评估模式
        
        
        # 将数据移动到指定设备
        image = image.to(device)
        label = label.long().squeeze(1).to(device)
        
        # 禁用梯度计算
        with torch.no_grad():
            # 前向传播
            blocks, latent_A = upl_model.enc(image)
            aux_seg_1 = upl_model.aux_dec1(latent_A, blocks)
            
            # 计算损失
            #loss = criterion(aux_seg_1, label)
            
            
            # 计算性能指标
            dice1, dice2, dice3 = cal_dice(aux_seg_1, label)
            hd95_ec, hd95_co, hd95_wt = cal_hd95(aux_seg_1, label)
            # cal_RVE,IoU,PA
            RVE_ec, RVE_co, RVE_wt = cal_RVE(aux_seg_1, label)
            iou_ec, iou_co, iou_wt = IoU(aux_seg_1, label)
            pa_ec, pa_co, pa_wt = PA(aux_seg_1, label, num_classes)
            sensitivity_ec, sensitivity_co, sensitivity_wt = cal_sensitivity(aux_seg_1, label)
            ppv_ec, ppv_co, ppv_wt = cal_ppv(aux_seg_1, label)


        
        # 返回损失和性能指标
        #return dice1.item(), dice2.item(), dice3.item(),hd95_wt, hd95_co, hd95_ec
        return dice1, dice2, dice3,hd95_ec, hd95_co, hd95_wt,RVE_ec, RVE_co, RVE_wt,iou_ec, iou_co, iou_wt,pa_ec, pa_co, pa_wt,sensitivity_ec, sensitivity_co, sensitivity_wt,ppv_ec, ppv_co, ppv_wt
    # 训练主循环
    train_flag = False
    for epoch in range(start_epoch,num_epochs):
        running_loss = 0.0
        train_loss = 0.0
        dice1_train = 0.0
        dice2_train = 0.0
        dice3_train = 0.0
        if train_flag:

            upl_model.train()
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
            # 初始化变量来存储历史平均值
            dice1_total = 0.0
            dice2_total = 0.0
            dice3_total = 0.0
            batch_count = 0

            for i, (image, label, _, C) in enumerate(train_bar):
                # 调用训练源函数
                loss = train_source(image, label)
                
                # 累加损失和dice值
                #running_loss += loss
                # dice1_train += dice1
                # dice2_train += dice2
                # dice3_train += dice3
                
                # 累加dice值和批次计数，用于计算历史平均
                # dice1_total += dice1
                # dice2_total += dice2
                # dice3_total += dice3
                batch_count += 1
                
                # 计算历史平均dice值
                # dice1_avg = dice1_total / batch_count
                # dice2_avg = dice2_total / batch_count
                # dice3_avg = dice3_total / batch_count
                # dice_mean_avg = (dice1_avg + dice2_avg + dice3_avg) / 3
                
                # 更新进度条
                train_bar.desc = f"Epoch [{epoch + 1}/{num_epochs}]] loss:[{loss:.4f}] Training"
            # 计算平均训练指标
            # #train_loss = running_loss / len(train_loader)
            # avg_dice1_train = dice1_train / len(train_loader)
            # avg_dice2_train = dice2_train / len(train_loader)
            # avg_dice3_train = dice3_train / len(train_loader)

            # 保存模型
        if epoch % 1 == 0:
            if train_flag :
                torch.save(upl_model.state_dict(), os.path.join(model_dir, f"model-{epoch}.pth"))
            

        # 验证过程
        if (epoch + 2) % 10 == 0:

            
            upl_model.eval()
            val_loss = 0.0
            dice1_val = 0.0
            dice2_val = 0.0
            dice3_val = 0.0
            RVE_wt_val = 0
            RVE_co_val = 0 
            RVE_ec_val = 0
            iou_wt_val = 0
            iou_co_val = 0
            iou_ec_val = 0
            pa_wt_val = 0
            pa_co_val = 0
            pa_ec_val = 0
            sensitivity_wt_val = 0
            sensitivity_co_val = 0
            sensitivity_ec_val = 0
            ppv_wt_val = 0
            ppv_co_val = 0
            ppv_ec_val = 0
            hd95_list_wt = []
            hd95_list_co = []
            hd95_list_ec = []
            valid_bar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation")
            with torch.no_grad():
                for image, label, _, _  in valid_bar:

                    dice1, dice2, dice3,hd95_ec, hd95_co, hd95_wt,RVE_ec, RVE_co, RVE_wt,iou_ec, iou_co, iou_wt,pa_ec, pa_co, pa_wt,sensitivity_ec, sensitivity_co, sensitivity_wt,ppv_ec, ppv_co, ppv_wt = evaluate_source(image, label)
                    
                    dice1_val += dice1
                    dice2_val += dice2
                    dice3_val += dice3
                    RVE_wt_val += RVE_wt
                    RVE_co_val += RVE_co
                    RVE_ec_val += RVE_ec
                    iou_wt_val += iou_wt
                    iou_co_val += iou_co
                    iou_ec_val += iou_ec
                    pa_wt_val += pa_wt
                    pa_co_val += pa_co
                    pa_ec_val += pa_ec
                    sensitivity_ec_val += sensitivity_ec
                    sensitivity_co_val += sensitivity_co
                    sensitivity_wt_val += sensitivity_wt
                    ppv_ec_val += ppv_ec    
                    ppv_co_val += ppv_co
                    ppv_wt_val += ppv_wt


                    hd95_list_wt.append(hd95_wt)
                    hd95_list_co.append(hd95_co)
                    hd95_list_ec.append(hd95_ec)
                    hd95mean = (hd95_wt + hd95_co + hd95_ec) / 3
                    dice_mean = (dice1 + dice2 + dice3) / 3
                    valid_bar.desc = f"Epoch [{epoch + 1}/{num_epochs}] dice:[{dice_mean}] hd95:[{hd95mean}] Validation"
                    

                avg_dice1_val = dice1_val / len(valid_loader)
                avg_dice2_val = dice2_val / len(valid_loader)
                avg_dice3_val = dice3_val / len(valid_loader)
                #cal_RVE,IoU,PA
                avg_RVE_wt = RVE_wt_val / len(valid_loader)
                avg_RVE_co = RVE_co_val / len(valid_loader)
                avg_RVE_ec = RVE_ec_val / len(valid_loader)
                avg_iou_wt = iou_wt_val / len(valid_loader)
                avg_iou_co = iou_co_val / len(valid_loader)
                avg_iou_ec = iou_ec_val / len(valid_loader)
                avg_pa_wt = pa_wt_val / len(valid_loader)
                avg_pa_co = pa_co_val / len(valid_loader)
                avg_pa_ec = pa_ec_val / len(valid_loader)
                avg_sensitivity_wt = sensitivity_wt_val / len(valid_loader)
                avg_sensitivity_co = sensitivity_co_val / len(valid_loader)
                avg_sensitivity_ec = sensitivity_ec_val / len(valid_loader)
                avg_ppv_wt = ppv_wt_val / len(valid_loader)
                avg_ppv_co = ppv_co_val / len(valid_loader)
                avg_ppv_ec = ppv_ec_val / len(valid_loader)

                avg_hd95_wt = np.nanmean(hd95_list_wt)
                avg_hd95_co = np.nanmean(hd95_list_co)
                avg_hd95_ec = np.nanmean(hd95_list_ec)
                avg_dice = (avg_dice1_val + avg_dice2_val + avg_dice3_val) / 3

                # 保存日志
                with open(validation_results_file, 'a') as f:
                    f.write(f"Epoch: {epoch + 1}/{num_epochs}\n")
                    # if train_flag:
                        
                    #     f.write(f"Train Dice: ET {avg_dice1_train:.3f} TC {avg_dice2_train:.3f} WT {avg_dice3_train:.3f}\n")
                    f.write(f"Val Dice: ET {avg_dice1_val:.3f} TC {avg_dice2_val:.3f} WT {avg_dice3_val:.3f}\n")
                    f.write(f"Val HD95: ET {avg_hd95_ec:.3f} TC {avg_hd95_co:.3f} WT {avg_hd95_wt:.3f}\n\n")
                    f.write(f"Val RVE: ET {avg_RVE_ec:.3f} TC {avg_RVE_co:.3f} WT {avg_RVE_wt:.3f}\n")
                    f.write(f"Val IoU: ET {avg_iou_ec:.3f} TC {avg_iou_co:.3f} WT {avg_iou_wt:.3f}\n")
                    f.write(f"Val PA: ET {avg_pa_ec:.3f} TC {avg_pa_co:.3f} WT {avg_pa_wt:.3f}\n")
                    f.write(f"Val Sensitivity: ET {avg_sensitivity_ec:.3f} TC {avg_sensitivity_co:.3f} WT {avg_sensitivity_wt:.3f}\n")
                    f.write(f"Val PPV: ET {avg_ppv_ec:.3f} TC {avg_ppv_co:.3f} WT {avg_ppv_wt:.3f}\n")

                #train_flag = True
                # 保存最佳模型
                if train_flag ==True:
                    if avg_dice < best_dice:
                        best_dice = avg_dice
                        torch.save(upl_model.state_dict(), os.path.join(model_dir, f"best-model-NEW.pth"))
                train_flag = True
        
        #upl_model.update_lr()  #此处为所有编码器都更新

    print("Training completed successfully!")
    #os.system("shutdown")
def main():
    # load config
    # load config
    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--config', type=str, default="/root/autodl-tmp/JS/UPL-SFDA-BRATS/config/train2d_source.cfg",
                        help='Path to the configuration file')
    args = parser.parse_args()
    config = args.config
    config = parse_config(config)
    list_data = []
    #print(config)
    dataset = config['train']['dataset']
    
    if dataset == 'brats':

        
        batch_train = 4
        batch_test = 4
        num_workers = 4
        source_root = '/root/autodl-tmp/BraTS2024'
        target_root = '/root/autodl-tmp/BraTS-SSA'
        train_path = 'train'
        test_path = 'test'
        mode = 'source_to_source'
        #mode should be 'source_to_source' or 'source_to_target' or 'target_to_target
        img = 'all'
        train_loader,test_loader = get_data_loader(source_root=source_root,
                                               target_root=target_root,
                                               train_path=train_path,
                                               test_path=test_path,
                                               batch_train=batch_train,
                                               batch_test=batch_test,
                                               nw = num_workers,
                                               img=img,
                                               mode=mode)
            
        list_data = train(config,train_loader,test_loader,list_data)
        
if __name__ == '__main__':
    set_random()
    resume1 = True
    #torch.manual_seed(0.95)
    #torch.cuda.manual_seed(0.95) 
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False 
    main()