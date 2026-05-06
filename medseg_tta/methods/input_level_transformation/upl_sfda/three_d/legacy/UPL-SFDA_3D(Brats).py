from utils import parse_config, set_random
from unet3d import UNet3d
import torch

import os
from metrics import cal_hd95,cal_dice,cal_RVE,IoU,PA,cal_sensitivity,cal_ppv
import numpy as np
import argparse
import datetime
from monai.inferers import sliding_window_inference

from tqdm import tqdm
from utils_brats_all import get_data_loader
# from ffn  import SpatialMotionSimLayer
def tensor2numpy(tensor_list):
    return [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in tensor_list]

def inference(input,model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(32,64,64),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    return _compute(input)



def test(config, upl_model, test_loader, exp_name):
    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    num_classes = config['train']['num_classes']
    # sim_layer = SpatialMotionSimLayer(
    #                         max_translation=2,       
    #                         apply_blur=True,
    #                         blur_sigma_range=(0.3, 0.7),  
    #                         apply_ghosting=False,     
    #                         p=0.8               )
    
    
    for data_loader in [test_loader]:  # 只对测试集进行评估
        all_batch_dice = []
        all_batch_assd = []
        all_batch_hd95 = []
        hd95_list_wt = []
        hd95_list_co = []
        hd95_list_ec = []  # 新增 HD95 列表
        output_result = []
        #val_loss = 0.0
        dice1_list = []
        dice2_list = []
        dice3_list = []
        iou_ec_list = []
        iou_co_list = []
        iou_wt_list = []
        rve_ec_list = []
        rve_co_list = []
        rve_wt_list = []
        pa_ec_list = []
        pa_co_list = []
        pa_wt_list = []
        sensitivity_ec_list = []
        sensitivity_co_list = []
        sensitivity_wt_list = []
        ppv_ec_list = []
        ppv_co_list = []
        ppv_wt_list = []
        hd95_ec_list = []
        hd95_co_list = []
        hd95_wt_list = []


        with torch.no_grad():
                upl_model.eval()

                val_loss = 0



                batch_count = 0
                valid_bar = tqdm(test_loader, desc=f"test:")
                for image, label, _, _  in valid_bar:
                    # image = sim_layer(image)
                    image = image.to(device)
                    label = label.long().squeeze(1).to(device)  

                    # 前向传播
                    blocks, latent_A = upl_model.enc(image)
                    aux_seg_1 = upl_model.aux_dec1(latent_A, blocks)
                    #loss = criterion(outputs, label)

                    # 更新验证指标
                    #val_loss += loss.item()
                    dice1, dice2, dice3 = cal_dice(aux_seg_1, label)  # 假设 cal_dice 可以处理多分类
                    hd95_ec, hd95_co, hd95_wt = cal_hd95(aux_seg_1, label)
                    IoU_ec, IoU_co, IoU_wt = IoU(aux_seg_1, label)
                    pa_ec, pa_co, pa_wt = PA(aux_seg_1, label, num_classes)
                    RVE_ec, RVE_co, RVE_wt = cal_RVE(aux_seg_1, label)
                    sensitivity_ec, sensitivity_co, sensitivity_wt = cal_sensitivity(aux_seg_1, label)
                    ppv_ec, ppv_co, ppv_wt = cal_ppv(aux_seg_1, label)
                    #print(hd95(outputs, label))
                    # 在 valid_bar 循环内部，每次 forward 后添加：
                    dice1_list.append(dice1)
                    dice2_list.append(dice2)
                    dice3_list.append(dice3)

                    iou_ec_list.append(IoU_ec)
                    iou_co_list.append(IoU_co)
                    iou_wt_list.append(IoU_wt)

                    rve_ec_list.append(RVE_ec)
                    rve_co_list.append(RVE_co)
                    rve_wt_list.append(RVE_wt)

                    pa_ec_list.append(pa_ec)
                    pa_co_list.append(pa_co)
                    pa_wt_list.append(pa_wt)

                    sensitivity_ec_list.append(sensitivity_ec)
                    sensitivity_co_list.append(sensitivity_co)
                    sensitivity_wt_list.append(sensitivity_wt)

                    ppv_ec_list.append(ppv_ec)
                    ppv_co_list.append(ppv_co)
                    ppv_wt_list.append(ppv_wt)

                    hd95_ec_list.append(hd95_ec)
                    hd95_co_list.append(hd95_co)
                    hd95_wt_list.append(hd95_wt)

                    valid_bar.desc = f"dice:[] hd95:[]"
        dice1_list = tensor2numpy(dice1_list)
        # 将 GPU 上的张量列表转为 NumPy 数值列表
        dice2_list = tensor2numpy(dice2_list)
        dice3_list = tensor2numpy(dice3_list)

        iou_ec_list = tensor2numpy(iou_ec_list)
        iou_co_list = tensor2numpy(iou_co_list)
        iou_wt_list = tensor2numpy(iou_wt_list)

        rve_ec_list = tensor2numpy(rve_ec_list)
        rve_co_list = tensor2numpy(rve_co_list)
        rve_wt_list = tensor2numpy(rve_wt_list)

        pa_ec_list = tensor2numpy(pa_ec_list)
        pa_co_list = tensor2numpy(pa_co_list)
        pa_wt_list = tensor2numpy(pa_wt_list)

        sensitivity_ec_list = tensor2numpy(sensitivity_ec_list)
        sensitivity_co_list = tensor2numpy(sensitivity_co_list)
        sensitivity_wt_list = tensor2numpy(sensitivity_wt_list)

        ppv_ec_list = tensor2numpy(ppv_ec_list)
        ppv_co_list = tensor2numpy(ppv_co_list)
        ppv_wt_list = tensor2numpy(ppv_wt_list)

        hd95_ec_list = tensor2numpy(hd95_ec_list)  # 注意：hd95 可能已经是 float，不影响
        hd95_co_list = tensor2numpy(hd95_co_list)
        hd95_wt_list = tensor2numpy(hd95_wt_list)
        std_dice1 = np.std(dice1_list)
        std_dice2 = np.std(dice2_list)
        std_dice3 = np.std(dice3_list)

        std_iou_ec = np.std(iou_ec_list)
        std_iou_co = np.std(iou_co_list)
        std_iou_wt = np.std(iou_wt_list)

        std_rve_ec = np.std(rve_ec_list)
        std_rve_co = np.std(rve_co_list)
        std_rve_wt = np.std(rve_wt_list)

        std_pa_ec = np.std(pa_ec_list)
        std_pa_co = np.std(pa_co_list)
        std_pa_wt = np.std(pa_wt_list)

        std_sensitivity_ec = np.std(sensitivity_ec_list)
        std_sensitivity_co = np.std(sensitivity_co_list)
        std_sensitivity_wt = np.std(sensitivity_wt_list)

        std_ppv_ec = np.std(ppv_ec_list)
        std_ppv_co = np.std(ppv_co_list)
        std_ppv_wt = np.std(ppv_wt_list)

        std_hd95_ec = np.nanstd(hd95_ec_list)
        std_hd95_co = np.nanstd(hd95_co_list)
        std_hd95_wt = np.nanstd(hd95_wt_list)
        # Dice
        avg_dice1_val = np.mean(dice1_list)
        avg_dice2_val = np.mean(dice2_list)
        avg_dice3_val = np.mean(dice3_list)

        # IoU
        avg_iou_ec_val = np.mean(iou_ec_list)
        avg_iou_co_val = np.mean(iou_co_list)
        avg_iou_wt_val = np.mean(iou_wt_list)

        # RVE
        avg_rve_ec_val = np.mean(rve_ec_list)
        avg_rve_co_val = np.mean(rve_co_list)
        avg_rve_wt_val = np.mean(rve_wt_list)

        # PA
        avg_pa_ec_val = np.mean(pa_ec_list)
        avg_pa_co_val = np.mean(pa_co_list)
        avg_pa_wt_val = np.mean(pa_wt_list)

        # Sensitivity
        avg_sensitivity_ec_val = np.mean(sensitivity_ec_list)
        avg_sensitivity_co_val = np.mean(sensitivity_co_list)
        avg_sensitivity_wt_val = np.mean(sensitivity_wt_list)

        # PPV
        avg_ppv_ec_val = np.mean(ppv_ec_list)
        avg_ppv_co_val = np.mean(ppv_co_list)
        avg_ppv_wt_val = np.mean(ppv_wt_list)

        # HD95
        avg_hd95_ec_val = np.nanmean(hd95_ec_list)
        avg_hd95_co_val = np.nanmean(hd95_co_list)
        avg_hd95_wt_val = np.nanmean(hd95_wt_list)
        dicemean = (avg_dice1_val + avg_dice2_val + avg_dice3_val) / 3
        # 转换为 NumPy 数组并计算统计值
        #日期戳

        output_result.append(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output_result.append("Dice:")
        output_result.append(f"ET : {avg_dice1_val:.3f}±{std_dice1:.3f}")
        output_result.append(f"TC : {avg_dice2_val:.3f}±{std_dice2:.3f}")
        output_result.append(f"WT : {avg_dice3_val:.3f}±{std_dice3:.3f}")
        output_result.append(f"HD95_ET : {avg_hd95_ec_val:.3f}±{std_hd95_ec:.3f}")
        output_result.append(f"HD95_TC : {avg_hd95_co_val:.3f}±{std_hd95_co:.3f}")
        output_result.append(f"HD95_WT : {avg_hd95_wt_val:.3f}±{std_hd95_wt:.3f}")
        output_result.append(f"PA_ET : {avg_pa_ec_val:.3f}±{std_pa_ec:.3f}")
        output_result.append(f"PA_TC : {avg_pa_co_val:.3f}±{std_pa_co:.3f}")
        output_result.append(f"PA_WT : {avg_pa_wt_val:.3f}±{std_pa_wt:.3f}")
        output_result.append(f"PPV_ET : {avg_ppv_ec_val:.3f}±{std_ppv_ec:.3f}")
        output_result.append(f"PPV_TC : {avg_ppv_co_val:.3f}±{std_ppv_co:.3f}")
        output_result.append(f"PPV_WT : {avg_ppv_wt_val:.3f}±{std_ppv_wt:.3f}")
        output_result.append(f"Sensitivity_ET : {avg_sensitivity_ec_val:.3f}±{std_sensitivity_ec:.3f}")
        output_result.append(f"Sensitivity_TC : {avg_sensitivity_co_val:.3f}±{std_sensitivity_co:.3f}")
        output_result.append(f"Sensitivity_WT : {avg_sensitivity_wt_val:.3f}±{std_sensitivity_wt:.3f}")
        output_result.append(f"RVE_ET : {avg_rve_ec_val:.3f}±{std_rve_ec:.3f}")
        output_result.append(f"RVE_TC : {avg_rve_co_val:.3f}±{std_rve_co:.3f}")
        output_result.append(f"RVE_WT : {avg_rve_wt_val:.3f}±{std_rve_wt:.3f}")
        output_result.append(f"IoU_ET : {avg_iou_ec_val:.3f}±{std_iou_ec:.3f}")
        output_result.append(f"IoU_TC : {avg_iou_co_val:.3f}±{std_iou_co:.3f}")
        output_result.append(f"IoU_WT : {avg_iou_wt_val:.3f}±{std_iou_wt:.3f}")
        output_result.append("\n")
        

            
            
            # 保存结果到文件
        results_dir = f"results_ssaokkk-FANG/{exp_name}"
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'result_tar.txt'), 'a') as file:
                for line in output_result:
                    file.write(line + "\n")
        return dicemean
                
def get_model_module(model):
    """
    动态获取模型的访问方式，支持单 GPU 和多 GPU。
    """
    return model.module if isinstance(model, torch.nn.DataParallel) else model
               
def train(config,train_loader,test_loader,source_model):
    # load exp_name
    exp_name = config['train']['exp_name']
    dataset = config['train']['dataset']
    #num_classes=4
    device = torch.device('cuda')
    upl_model = UNet3d(config).to(device)
    print('source_model_created')
    checkpoint = torch.load(source_model)
    print('source_model_loaded')
    upl_model.load_state_dict(checkpoint)
    dec1 = upl_model.aux_dec1.state_dict()
    upl_model.aux_dec2.load_state_dict(dec1)
    upl_model.aux_dec3.load_state_dict(dec1)
    upl_model.aux_dec4.load_state_dict(dec1)
    num_epochs = 60
    best_dice = 0.
    output_dir = "validation_results_ssa-t2f"
    os.makedirs(output_dir,exist_ok=True)
    train_flag = True
    current_dice = test(config,upl_model,test_loader,exp_name=exp_name)
    #if (current_dice) > best_dice:
                #best_dice = current_dice
                #model_dir = "save_model_ssa/" + str(exp_name )
                #os.makedirs(model_dir, exist_ok=True)
                #best_epoch = '{}/model-{}-{}-{}.pth'.format(model_dir, 'best', str(epoch), np.round(best_dice,3))
                #torch.save(upl_model.state_dict(), best_epoch)
    # sim_layer = SpatialMotionSimLayer(
    #                         max_translation=2,       
    #                         apply_blur=True,
    #                         blur_sigma_range=(0.3, 0.7),  
    #                         apply_ghosting=False,     
    #                         p=0.8               )

    for epoch in range(num_epochs):
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        if  train_flag :
            upl_model.train()
            for i, (B,B_label,_,_) in enumerate(train_bar):
                B = B.to(device).detach()
                # B = sim_layer(B)
                upl_model.save_nii(B)
                loss,_ = upl_model.train_target(B)
                #更新tqdm
                train_bar.desc = f"Epoch {epoch + 1}/{num_epochs} loss:{loss}"
        # # valid for target domain
        if (epoch+1) % 1 == 0:
            current_dice = test(config,upl_model,test_loader,exp_name=exp_name)
            if (current_dice) > best_dice:
                best_dice = current_dice
                model_dir = "save_model_ssa-FANG/" + str(exp_name )
                os.makedirs(model_dir, exist_ok=True)
                best_epoch = '{}/model-{}-{}-{}.pth'.format(model_dir, 'best', str(epoch), np.round(best_dice,3))
                torch.save(upl_model.state_dict(), best_epoch)
            
    if train_flag and (epoch+1) % 10 == 0:
        torch.save(upl_model.state_dict(), '{}/model-{}.pth'.format(model_dir, 'latest'))

    upl_model.load_state_dict(torch.load(best_epoch,map_location='cpu'),strict=True)
    upl_model.eval()
    test(config,upl_model,test_loader,exp_name=exp_name)
    
    

def mian():
    # load config
    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--config', type=str, default="./config/train3d.cfg",
                        help='Path to the configuration file')
    args = parser.parse_args()
    config = args.config
    config = parse_config(config)

    source_model = '/root/autodl-tmp/JS/UPL-SFDA-BRATS/okkkk-5-me-model/model-86.pth'

    batch_train = 1
    batch_test = 1
    num_workers = 0
    source_root = "/root/autodl-tmp/JS/test_ffn"
    target_root = "/root/autodl-tmp/JS/test_ffn"
    train_path = 'train'
    test_path = 'test'
    mode = 'target_to_target'
    img = 'all'
    train_loader,test_loader = get_data_loader(source_root,target_root,
                                               train_path,test_path,
                                               batch_train,batch_test,
                                               nw = num_workers,
                                               img=img,mode=mode)
    print("数据加载完成")
    

    train(config,train_loader,test_loader,source_model)
        
if __name__ == '__main__':
    
    set_random()
    torch.manual_seed(0.95)
    torch.cuda.manual_seed(0.95) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    
    mian()