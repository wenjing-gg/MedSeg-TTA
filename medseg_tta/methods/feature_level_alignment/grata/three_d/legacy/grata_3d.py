import torch
import torch.nn as nn
import contextlib
import numpy as np
from torch.distributed import ReduceOp

class GraTa3D(torch.optim.Optimizer):

    def __init__(self, params, base_optimizer, model, adaptive=False, perturb_eps=1e-12, grad_reduce='mean', device='cuda:0', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(GraTa3D, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps
        self.init_lr = self.base_optimizer.param_groups[0]['lr']
        self.device = device
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def perturb_weights_sub(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                self.state[p]['aux_g'] = p.grad.data.clone()
                e_w = p.grad
                p.data.sub_(e_w)

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'old_p' in self.state[p].keys():
                    p.data.copy_(self.state[p]['old_p'])

    @torch.enable_grad()
    def cal_ent_loss(self, imgs):
        self.base_optimizer.zero_grad()
        outputs = self.model(imgs)
        if isinstance(outputs, (list, tuple)):
            pred_logit = outputs[0]
        else:
            pred_logit = outputs
        pred_sigmoid = torch.sigmoid(pred_logit)
        ent = -(pred_sigmoid * torch.log(pred_sigmoid + 1e-06)).sum(1)
        ent_loss = ent.mean()
        ent_loss.backward()
        return ent_loss

    @torch.enable_grad()
    def cal_consis_loss_3d(self, imgs, criterion=torch.nn.BCEWithLogitsLoss()):
        predictions = []
        with torch.no_grad():
            outputs = self.model(imgs)
            if isinstance(outputs, (list, tuple)):
                pred = outputs[0]
            else:
                pred = outputs
            predictions.append(pred.detach().cpu())
        aug_factors = [0, 1, 2, 3]
        for factor in aug_factors:
            imgs_aug = self.apply_3d_augmentation(imgs, factor)
            with torch.no_grad():
                outputs_aug = self.model(imgs_aug)
                if isinstance(outputs_aug, (list, tuple)):
                    pred_aug = outputs_aug[0]
                else:
                    pred_aug = outputs_aug
                pred_aug = self.inverse_3d_augmentation(pred_aug, factor)
                predictions.append(pred_aug.detach().cpu())
        predictions = torch.stack(predictions).sigmoid()
        predictions = predictions.mean(0).to(self.device)
        self.base_optimizer.zero_grad()
        imgs_strong = self.apply_strong_3d_augmentation(imgs)
        outputs_strong = self.model(imgs_strong)
        if isinstance(outputs_strong, (list, tuple)):
            pred_strong = outputs_strong[0]
        else:
            pred_strong = outputs_strong
        consis_loss = criterion(pred_strong, predictions)
        consis_loss.backward()
        return consis_loss

    def apply_3d_augmentation(self, imgs, factor):
        if factor == 0:
            return imgs.flip(-1)
        elif factor == 1:
            return imgs.flip(-2)
        elif factor == 2:
            return imgs.flip(-3)
        elif factor == 3:
            return imgs
        else:
            return imgs

    def inverse_3d_augmentation(self, pred, factor):
        if factor == 0:
            return pred.flip(-1)
        elif factor == 1:
            return pred.flip(-2)
        elif factor == 2:
            return pred.flip(-3)
        elif factor == 3:
            return pred
        else:
            return pred

    def apply_strong_3d_augmentation(self, imgs):
        noise = torch.randn_like(imgs) * 0.1
        return imgs + noise

    @torch.no_grad()
    def get_cosine(self):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'aux_g' not in self.state[p].keys():
                    continue
                inner_prod += torch.sum(self.state[p]['aux_g'] * p.grad.data.clone())
        pse_grad_norm = self._grad_norm()
        aux_grad_norm = self._grad_norm(by='aux_g')
        cosine = inner_prod / (pse_grad_norm * aux_grad_norm + self.perturb_eps)
        return cosine.detach()

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        if not by:
            norm = torch.norm(torch.stack([((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2) for group in self.param_groups for p in group['params'] if p.grad is not None]), p=2)
        else:
            norm = torch.norm(torch.stack([((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2) for group in self.param_groups for p in group['params'] if p.grad is not None and by in self.state[p].keys()]), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def step(self, imgs, aux='ent', pse='consis'):
        losses = {'consis': lambda x: self.cal_consis_loss_3d(x), 'ent': lambda x: self.cal_ent_loss(x)}
        with self.maybe_no_sync():
            aux_loss = losses[aux](imgs)
            self.perturb_weights_sub()
            pse_loss = losses[pse](imgs)
            cosine = self.get_cosine()
            self.unperturb()
        self._sync_grad()
        self.base_optimizer.param_groups[0]['lr'] = self.init_lr * custom_activation(cosine)
        self.base_optimizer.step()

def custom_activation(x):
    return 1 / 4 * (x + 1) ** 2

def collect_params_3d(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm3d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f'{nm}.{np}')
    return params

def configure_model_3d(model):
    model.train()
    model.requires_grad_(False)
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm3d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model
