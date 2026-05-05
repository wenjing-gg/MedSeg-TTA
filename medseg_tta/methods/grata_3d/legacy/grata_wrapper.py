import torch
import torch.nn as nn
from grata_3d import GraTa3D, collect_params_3d, configure_model_3d

class GraTaWrapper:

    def __init__(self, model, args, device='cuda:0'):
        self.device = device
        self.model = configure_model_3d(model)
        self.args = args
        self.adaptable_params = collect_params_3d(self.model)
        if hasattr(args, 'optimizer') and args.optimizer == 'SGD':
            self.base_optimizer = torch.optim.SGD(self.adaptable_params, lr=getattr(args, 'lr', 0.0001), momentum=getattr(args, 'momentum', 0.9), nesterov=True)
        else:
            self.base_optimizer = torch.optim.Adam(self.adaptable_params, lr=getattr(args, 'lr', 0.0001), betas=(0.9, 0.999))
        self.optimizer = GraTa3D(self.adaptable_params, self.base_optimizer, self.model, device=device)
        self.aux_loss = getattr(args, 'aux_loss', 'ent')
        self.pse_loss = getattr(args, 'pse_loss', 'consis')
        print(f'🔧 GraTa初始化完成:')
        print(f'   - 可适应参数数量: {len(self.adaptable_params)}')
        print(f'   - 辅助损失: {self.aux_loss}')
        print(f'   - 伪损失: {self.pse_loss}')
        print(f'   - 学习率: {getattr(args, 'lr', 0.0001)}')

    def adapt_and_predict(self, imgs):
        self.model.train()
        self.optimizer.step(imgs, aux=self.aux_loss, pse=self.pse_loss)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(imgs)
        return outputs

    def predict_only(self, imgs):
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(imgs)
        return outputs

    def reset_model(self):
        self.model = configure_model_3d(self.model)

    def get_model(self):
        return self.model

def create_grata_model(model, args, device='cuda:0'):
    return GraTaWrapper(model, args, device)

class GraTaConfig:

    def __init__(self, lr=0.0001, aux_loss='ent', pse_loss='consis', optimizer='Adam', momentum=0.9):
        self.lr = lr
        self.aux_loss = aux_loss
        self.pse_loss = pse_loss
        self.optimizer = optimizer
        self.momentum = momentum

def get_default_grata_config():
    return GraTaConfig(lr=0.0001, aux_loss='ent', pse_loss='consis', optimizer='Adam', momentum=0.9)
