from utils_u import parse_config, set_random
from unet3d import UNet3d
from unet import UNet
from memory import Memory
import torch
import matplotlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
from utils_brats_all import get_data_loader

import numpy as np
from loss import CombinedLoss
from utils.loss import DiceLoss, entropy_loss
from tqdm import tqdm
from metrics import cal_hd95,cal_dice,cal_RVE,IoU,PA,cal_sensitivity,cal_ppv
#torch.cuda.empty_cache()
from metrics import cal_hd95
matplotlib.use('Agg')

import torch
import torch.nn as nn

# 引入其他代码中的模块和函数
from train.configs import set_configs
# from data.prostate.generate_data import load_prostate
from train.model_trainer_segmentation import ModelTrainerSegmentation
from train.api import API

def train(config, train_loader, valid_loader, list_data):
    print("start")
    # 加载配置参数
    exp_name = config['train']['exp_name']
    dataset = config['train']['dataset']
    num_classes = 4

    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # 加载模型
    upl_model = UNet3d(config)
    
    if resume1:
        upl_model.load_state_dict(torch.load(r"D:\HDU\STORE\BRATS_dataloader\UPL-SFDA-BRATS\checkpoint\UNET3D-ORI.pth",map_location=device))
        print('load model')
    # buffer_size = 20
    # memory = Memory(buffer_size)
    # upl_model.memory = memory
    # upl_model.buffer_size = buffer_size
    # prev_loss = np.inf
    
    # 启用梯度跟踪
    for param in upl_model.parameters():
        param.requires_grad = True
    upl_model.to(device)
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
    num_epochs = 10
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
        upl_model.eval()
        
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
    train_flag = True
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
        if (epoch + 2) % 1 == 0:
            upl_model.eval()
            metrics = {
                'dice1': [], 'dice2': [], 'dice3': [],
                'RVE_wt': [], 'RVE_co': [], 'RVE_ec': [],
                'iou_wt': [], 'iou_co': [], 'iou_ec': [],
                'pa_wt': [], 'pa_co': [], 'pa_ec': [],
                'sensitivity_wt': [], 'sensitivity_co': [], 'sensitivity_ec': [],
                'ppv_wt': [], 'ppv_co': [], 'ppv_ec': [],
                'hd95_wt': [], 'hd95_co': [], 'hd95_ec': []
            }
            
            valid_bar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation")
            with torch.no_grad():
                for image, label, _, _ in valid_bar:
                    # 获取评估指标
                    dice1, dice2, dice3, hd95_ec, hd95_co, hd95_wt, RVE_ec, RVE_co, RVE_wt, iou_ec, iou_co, iou_wt, pa_ec, pa_co, pa_wt, sensitivity_ec, sensitivity_co, sensitivity_wt, ppv_ec, ppv_co, ppv_wt = evaluate_source(image, label)
                    
                    # 存储各项指标
                    metrics['dice1'].append(dice1)
                    metrics['dice2'].append(dice2)
                    metrics['dice3'].append(dice3)
                    metrics['RVE_wt'].append(RVE_wt)
                    metrics['RVE_co'].append(RVE_co)
                    metrics['RVE_ec'].append(RVE_ec)
                    metrics['iou_wt'].append(iou_wt)
                    metrics['iou_co'].append(iou_co)
                    metrics['iou_ec'].append(iou_ec)
                    metrics['pa_wt'].append(pa_wt)
                    metrics['pa_co'].append(pa_co)
                    metrics['pa_ec'].append(pa_ec)
                    metrics['sensitivity_wt'].append(sensitivity_wt)
                    metrics['sensitivity_co'].append(sensitivity_co)
                    metrics['sensitivity_ec'].append(sensitivity_ec)
                    metrics['ppv_wt'].append(ppv_wt)
                    metrics['ppv_co'].append(ppv_co)
                    metrics['ppv_ec'].append(ppv_ec)
                    metrics['hd95_wt'].append(hd95_wt)
                    metrics['hd95_co'].append(hd95_co)
                    metrics['hd95_ec'].append(hd95_ec)
                    
                    # 计算当前批次的平均指标用于进度条显示
                    hd95mean = (hd95_wt + hd95_co + hd95_ec) / 3
                    dice_mean = (dice1 + dice2 + dice3) / 3
                    valid_bar.desc = f"Epoch [{epoch + 1}/{num_epochs}] dice:[{dice_mean}] hd95:[{hd95mean}] Validation"
            
            # 计算所有指标的平均值和标准差
            results = {}
            for key, values in metrics.items():
                avg = np.mean(values)
                std = np.std(values)
                results[key] = (avg, std)
            
            # 计算组合指标
            avg_dice = (results['dice1'][0] + results['dice2'][0] + results['dice3'][0]) / 3
            
            # 保存日志
            with open(validation_results_file, 'a') as f:
                f.write(f"Epoch: {epoch + 1}/{num_epochs}\n")
                f.write(f"Val Dice: ET {results['dice1'][0]:.4f}±{results['dice1'][1]:.4f} TC {results['dice2'][0]:.4f}±{results['dice2'][1]:.4f} WT {results['dice3'][0]:.4f}±{results['dice3'][1]:.4f}\n")
                f.write(f"Val HD95: ET {results['hd95_ec'][0]:.4f}±{results['hd95_ec'][1]:.4f} TC {results['hd95_co'][0]:.4f}±{results['hd95_co'][1]:.4f} WT {results['hd95_wt'][0]:.4f}±{results['hd95_wt'][1]:.4f}\n\n")
                f.write(f"Val RVE: ET {results['RVE_ec'][0]:.4f}±{results['RVE_ec'][1]:.4f} TC {results['RVE_co'][0]:.4f}±{results['RVE_co'][1]:.4f} WT {results['RVE_wt'][0]:.4f}±{results['RVE_wt'][1]:.4f}\n")
                f.write(f"Val IoU: ET {results['iou_ec'][0]:.4f}±{results['iou_ec'][1]:.4f} TC {results['iou_co'][0]:.4f}±{results['iou_co'][1]:.4f} WT {results['iou_wt'][0]:.4f}±{results['iou_wt'][1]:.4f}\n")
                f.write(f"Val PA: ET {results['pa_ec'][0]:.4f}±{results['pa_ec'][1]:.4f} TC {results['pa_co'][0]:.4f}±{results['pa_co'][1]:.4f} WT {results['pa_wt'][0]:.4f}±{results['pa_wt'][1]:.4f}\n")
                f.write(f"Val Sensitivity: ET {results['sensitivity_ec'][0]:.4f}±{results['sensitivity_ec'][1]:.4f} TC {results['sensitivity_co'][0]:.4f}±{results['sensitivity_co'][1]:.4f} WT {results['sensitivity_wt'][0]:.4f}±{results['sensitivity_wt'][1]:.4f}\n")
                f.write(f"Val PPV: ET {results['ppv_ec'][0]:.4f}±{results['ppv_ec'][1]:.4f} TC {results['ppv_co'][0]:.4f}±{results['ppv_co'][1]:.4f} WT {results['ppv_wt'][0]:.4f}±{results['ppv_wt'][1]:.4f}\n")
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

    # 添加测试时域适应方法
import torch.backends.cudnn as cudnn
import random
import copy

def tensor2numpy(tensor_list):
    return [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in tensor_list]
def deterministic(seed):
     cudnn.benchmark = False
     cudnn.deterministic = True
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
def test_time(config,test_data, device, args):
        num_classes =4
        def evaluate_source(image, label):
            # 将模型切换到评估模式
            model_adapt.eval()
            
            # 将数据移动到指定设备
            image = image.to(device)
            label = label.long().squeeze(1).to(device)
            
            # 禁用梯度计算
            with torch.no_grad():
                # 前向传播
                blocks, latent_A = model_adapt.enc(image)
                aux_seg_1 = model_adapt.aux_dec1(latent_A, blocks)
                
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

            # 加载模型
        model = UNet3d(config)
        
        if resume1:
            model.load_state_dict(torch.load(r"/home/zhengjingyuan/JS/DLTTA-3d/experiments/prostate/unet3d_best.pth",map_location=device))
            print('load model')
        # test_set = Prostate(args.target)
        # test_data = DataLoader(test_set, batch_size=1, shuffle=True)
        deterministic(2025)
        metrics = {
            'test_acc': 0,
            'test_loss': 0,
        }
        best_dice = 0.
        dice_buffer = []
        
        model_adapt = copy.deepcopy(model)
        model_adapt.to(device)
        model_adapt.train()
        for m in model_adapt.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        params = model_adapt.parameters()
        optimizer = torch.optim.SGD(params, lr=0.0001)
        criterion = DiceLoss().to(device)
        loss_all = 0
        test_acc = 0.
        num_epochs = 100
        train_flag = True
        for epoch in range(num_epochs):
            loss_all = 0
            test_acc = 0.
            
            # deterministic(2025)
            test_bar = tqdm(test_data, desc=f"Testing")
            for i, (data, target, _, C) in enumerate(test_bar):
                # deterministic(2025)
                data = data.to(device)
                # target = target.squeeze(1).to(device)
                model_adapt.enc_opt.zero_grad()
                model_adapt.aux_dec1_opt.zero_grad()
                blocks, latent_A = model_adapt.enc(data)
                output = model_adapt.aux_dec1(latent_A, blocks)
                
             
                loss_entropy_before = entropy_loss(output, c=4)        
                all_loss = loss_entropy_before
                weight = 1
                all_loss = weight*all_loss
                #print(all_loss)
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                # output = model_adapt(data)    
                  
                # loss = criterion(output, target)
                # loss_all += loss.item()   
                test_bar.set_description(f"Testing: Loss: {all_loss.item():.4f}")
            if (epoch + 2) % 1 == 0:
                    model_adapt.eval()
                    metrics = {
                        'dice1': [], 'dice2': [], 'dice3': [],
                        'RVE_wt': [], 'RVE_co': [], 'RVE_ec': [],
                        'iou_wt': [], 'iou_co': [], 'iou_ec': [],
                        'pa_wt': [], 'pa_co': [], 'pa_ec': [],
                        'sensitivity_wt': [], 'sensitivity_co': [], 'sensitivity_ec': [],
                        'ppv_wt': [], 'ppv_co': [], 'ppv_ec': [],
                        'hd95_wt': [], 'hd95_co': [], 'hd95_ec': []
                    }
                    
                    valid_bar = tqdm(test_data, desc=f"Epoch {epoch + 1}/{num_epochs} Validation")
                    with torch.no_grad():
                        for image, label, _, _ in valid_bar:
                            # 获取评估指标
                            dice1, dice2, dice3, hd95_ec, hd95_co, hd95_wt, RVE_ec, RVE_co, RVE_wt, iou_ec, iou_co, iou_wt, pa_ec, pa_co, pa_wt, sensitivity_ec, sensitivity_co, sensitivity_wt, ppv_ec, ppv_co, ppv_wt = evaluate_source(image, label)
                            
                            # 存储各项指标
                            metrics['dice1'].append(dice1)
                            metrics['dice2'].append(dice2)
                            metrics['dice3'].append(dice3)
                            metrics['RVE_wt'].append(RVE_wt)
                            metrics['RVE_co'].append(RVE_co)
                            metrics['RVE_ec'].append(RVE_ec)
                            metrics['iou_wt'].append(iou_wt)
                            metrics['iou_co'].append(iou_co)
                            metrics['iou_ec'].append(iou_ec)
                            metrics['pa_wt'].append(pa_wt)
                            metrics['pa_co'].append(pa_co)
                            metrics['pa_ec'].append(pa_ec)
                            metrics['sensitivity_wt'].append(sensitivity_wt)
                            metrics['sensitivity_co'].append(sensitivity_co)
                            metrics['sensitivity_ec'].append(sensitivity_ec)
                            metrics['ppv_wt'].append(ppv_wt)
                            metrics['ppv_co'].append(ppv_co)
                            metrics['ppv_ec'].append(ppv_ec)
                            metrics['hd95_wt'].append(hd95_wt)
                            metrics['hd95_co'].append(hd95_co)
                            metrics['hd95_ec'].append(hd95_ec)
                            
                            # 计算当前批次的平均指标用于进度条显示
                            hd95mean = (hd95_wt + hd95_co + hd95_ec) / 3
                            dice_mean = (dice1 + dice2 + dice3) / 3
                            valid_bar.desc = f"Epoch [{epoch + 1}/{num_epochs}] dice:[{dice_mean}] hd95:[{hd95mean}] Validation"
                    
                    # 计算所有指标的平均值和标准差
                    results = {}
                    for key, values in metrics.items():
                        values=tensor2numpy(values)
                        avg = np.mean(values)
                        std = np.std(values)
                        results[key] = (avg, std)
                    
                    # 计算组合指标
                    avg_dice = (results['dice1'][0] + results['dice2'][0] + results['dice3'][0]) / 3
                    validation_results_file ='PED.txt'
                    # 保存日志
                    with open(validation_results_file, 'a') as f:
                        f.write(f"Epoch: {epoch + 1}/{num_epochs}\n")
                        f.write(f"Val Dice: ET {results['dice1'][0]:.4f}±{results['dice1'][1]:.4f} TC {results['dice2'][0]:.4f}±{results['dice2'][1]:.4f} WT {results['dice3'][0]:.4f}±{results['dice3'][1]:.4f}\n")
                        f.write(f"Val HD95: ET {results['hd95_ec'][0]:.4f}±{results['hd95_ec'][1]:.4f} TC {results['hd95_co'][0]:.4f}±{results['hd95_co'][1]:.4f} WT {results['hd95_wt'][0]:.4f}±{results['hd95_wt'][1]:.4f}\n\n")
                        f.write(f"Val RVE: ET {results['RVE_ec'][0]:.4f}±{results['RVE_ec'][1]:.4f} TC {results['RVE_co'][0]:.4f}±{results['RVE_co'][1]:.4f} WT {results['RVE_wt'][0]:.4f}±{results['RVE_wt'][1]:.4f}\n")
                        f.write(f"Val IoU: ET {results['iou_ec'][0]:.4f}±{results['iou_ec'][1]:.4f} TC {results['iou_co'][0]:.4f}±{results['iou_co'][1]:.4f} WT {results['iou_wt'][0]:.4f}±{results['iou_wt'][1]:.4f}\n")
                        f.write(f"Val PA: ET {results['pa_ec'][0]:.4f}±{results['pa_ec'][1]:.4f} TC {results['pa_co'][0]:.4f}±{results['pa_co'][1]:.4f} WT {results['pa_wt'][0]:.4f}±{results['pa_wt'][1]:.4f}\n")
                        f.write(f"Val Sensitivity: ET {results['sensitivity_ec'][0]:.4f}±{results['sensitivity_ec'][1]:.4f} TC {results['sensitivity_co'][0]:.4f}±{results['sensitivity_co'][1]:.4f} WT {results['sensitivity_wt'][0]:.4f}±{results['sensitivity_wt'][1]:.4f}\n")
                        f.write(f"Val PPV: ET {results['ppv_ec'][0]:.4f}±{results['ppv_ec'][1]:.4f} TC {results['ppv_co'][0]:.4f}±{results['ppv_co'][1]:.4f} WT {results['ppv_wt'][0]:.4f}±{results['ppv_wt'][1]:.4f}\n")
                    #train_flag = True
                    # 保存最佳模型
                    if epoch % 5==0:
                        if avg_dice < best_dice:
                            best_dice = avg_dice
                            os.makedirs('save_model_DLTTA', exist_ok=True)
                            torch.save(model_adapt.state_dict(), os.path.join('save_model_DLTTA', f"best-model-NEW.pth"))
                    train_flag = True
        loss = loss_all / len(test_data)
        acc = 1 - loss  
        metrics['test_loss'] = loss
        metrics["test_acc"] = acc
        return metrics

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
    #print(config)
    dataset = 'brats'
    
    if dataset == 'brats':

        
        batch_train = 1
        batch_test = 1
        num_workers = 0
        source_root = r"/home/yuwenjing/data/BraTS-SSA"
        # target_root = r"/home/yuwenjing/data/BraTS-SSA"
        target_root = r"/home/yuwenjing/data/BraTS-PED2023/Train"
        train_path = ''
        test_path = ''
        mode = 'target_to_target'
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
            
        # list_data = train(config,train_loader,test_loader,list_data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_time(config=config,test_data=test_loader,device=device,args=args)
        
if __name__ == '__main__':
    set_random()
    resume1 = True
    #torch.manual_seed(0.95)
    #torch.cuda.manual_seed(0.95) 
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False 
    main()