import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from networks.ResUnet import ResUnet
from config import *
from utils.metrics import calculate_metrics
import numpy as np
import argparse
import sys, datetime, time
from torch.utils.data import DataLoader
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.transform import collate_fn_wo_transform
from custom_optimizers.grata import GraTa
torch.set_num_threads(1)

def print_information(config):
    print('Model Root: ', config.path_save_model)
    print('GPUs: ', torch.cuda.device_count())
    print('time: ', config.time_now)
    print('source domain: ', config.Source_Dataset)
    print('target domain: ', config.Target_Dataset)
    print('model: ' + str(config.model_type))
    print('input size: ', config.image_size)
    print('batch size: ', config.batch_size)
    print('optimizer: ', config.optimizer)
    print('lr: ', config.lr)
    print('auxiliary loss: ', config.aux_loss)
    print('pseudo loss: ', config.pse_loss)
    print('***' * 10)

def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f'{nm}.{np}')
    return params

class TrainTTA:

    def __init__(self, config):
        config.time_now = datetime.datetime.now().__format__('%Y%m%d_%H%M%S_%f')
        self.load_model = os.path.join(config.path_save_model, str(config.Source_Dataset))
        self.log_path = os.path.join(config.path_save_log, 'TrainTTA')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.log_path = os.path.join(self.log_path, config.time_now + '.log')
        sys.stdout = Logger(self.log_path, sys.stdout)
        target_test_csv = []
        if config.Target_Dataset != 'REFUGE_Valid':
            target_test_csv.append(config.Target_Dataset + '_train.csv')
            target_test_csv.append(config.Target_Dataset + '_test.csv')
        else:
            target_test_csv.append(config.Target_Dataset + '.csv')
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        target_test_dataset = OPTIC_dataset(config.dataset_root, ts_img_list, ts_label_list, config.image_size, img_normalize=True)
        self.target_test_loader = DataLoader(dataset=target_test_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn_wo_transform, num_workers=config.num_workers)
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.model_type = config.model_type
        self.optimizer = None
        self.optim = config.optimizer
        self.lr = config.lr
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)
        self.device = config.device
        self.aux = config.aux_loss
        self.pse = config.pse_loss
        print_information(config)
        self.build_model()
        self.print_network()

    def build_model(self):
        self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False).to(self.device)
        checkpoint = torch.load(self.load_model + '/' + 'last-' + self.model_type + '.pth')
        self.model.load_state_dict(checkpoint, strict=False)
        para = collect_params(self.model)
        if self.optim == 'SGD':
            base_optimizer = torch.optim.SGD(para, lr=self.lr, momentum=self.momentum, nesterov=True)
        elif self.optim == 'Adam':
            base_optimizer = torch.optim.Adam(para, lr=self.lr, betas=self.betas)
        elif self.optim == 'AdamW':
            base_optimizer = torch.optim.AdamW(para, lr=self.lr, betas=self.betas)
        else:
            raise NotImplementedError('ERROR: no such optimizer {}!'.format(self.optim))
        self.optimizer = GraTa(para, base_optimizer, self.model, device=self.device)

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print('The number of total parameters: {}'.format(num_params))

    def run(self):
        metric_dict = ['Disc_Dice', 'Disc_ASSD', 'Cup_Dice', 'Cup_ASSD']
        metrics_test = [[], [], [], []]
        for batch, data in enumerate(self.target_test_loader):
            x, y = (data['data'], data['mask'])
            x = torch.from_numpy(x).to(dtype=torch.float32).to(self.device)
            y = torch.from_numpy(y).to(dtype=torch.float32).to(self.device)
            self.model.train()
            self.model.requires_grad_(False)
            for nm, m in self.model.named_modules():
                if self.aux in nm or self.pse in nm:
                    m.requires_grad_(True)
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
            self.optimizer.base_optimizer.zero_grad()
            self.optimizer.step(data, self.aux, self.pse)
            with torch.no_grad():
                pred_logit, fea = self.model(x)
            seg_output = torch.sigmoid(pred_logit)
            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), 'The metrics value is not list type.'
                metrics_test[i] += metrics[i]
        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print('Test Metrics Mean: ', print_test_metric_mean)
        test_metrics_y = np.std(metrics_test, axis=1)
        print_test_metric_std = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_std[metric_dict[i]] = test_metrics_y[i]
        print('Test Metrics Std: ', print_test_metric_std)
        return print_test_metric_mean
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aux_loss', type=str, default='ent', help='consis/ent/recon/rotate/supres/denoise')
    parser.add_argument('--pse_loss', type=str, default='consis', help='consis/ent/recon/rotate/supres/denoise')
    parser.add_argument('--Source_Dataset', type=str, help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
    parser.add_argument('--optimizer', type=str, required=False, default='Adam', help='SGD/Adam/AdamW')
    parser.add_argument('--lr', type=float, required=False, default=0.0001)
    parser.add_argument('--momentum', type=float, required=False, default=0.99)
    parser.add_argument('--beta1', type=float, required=False, default=0.9)
    parser.add_argument('--beta2', type=float, required=False, default=0.999)
    parser.add_argument('--weight_decay', type=float, required=False, default=0.0)
    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--model_type', type=str, required=False, default='Res_Unet')
    parser.add_argument('--backbone', type=str, required=False, default='resnet34')
    parser.add_argument('--in_ch', type=int, required=False, default=3)
    parser.add_argument('--out_ch', type=int, required=False, default=2)
    parser.add_argument('--image_size', type=int, required=False, default=512)
    parser.add_argument('--num_workers', type=int, required=False, default=8)
    parser.add_argument('--path_save_model', type=str)
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--path_save_log', type=str, required=False, default='./logs/')
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, required=False, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, required=False, default='cpu')
    config = parser.parse_args()
    targets = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    targets.remove(config.Source_Dataset)
    dice_score = 0
    for config.Target_Dataset in targets:
        TTA = TrainTTA(config)
        metric = TTA.run()
        mean_dice = (metric['Disc_Dice'] + metric['Cup_Dice']) / 2
        dice_score += mean_dice
    print(config.Source_Dataset + ': Dice Mean=' + str(dice_score / len(targets)))
    print('\n\n\n')
