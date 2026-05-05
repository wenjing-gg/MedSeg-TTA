import torch
import contextlib
from custom_optimizers.spatial_aug import augment
from torch.distributed import ReduceOp
from dataloaders.aug import augmentation_spatial, augmentation_strong_style, augmentation_weak_style
from dataloaders.aug.spatial_aug import Rotate_and_Flip
from dataloaders.aug.style_aug import augment_lowfreq
from dataloaders.normalize import normalize_image_to_0_1

class GraTa(torch.optim.Optimizer):

    def __init__(self, params, base_optimizer, model, adaptive=False, perturb_eps=1e-12, grad_reduce='mean', device='cuda:0', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(GraTa, self).__init__(params, defaults)
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
    def cal_groundtruth_loss(self, data):
        x = torch.from_numpy(data['data']).to(dtype=torch.float32).to(self.device)
        y = torch.from_numpy(data['mask']).to(dtype=torch.float32).to(self.device)
        self.base_optimizer.zero_grad()
        pred_logit, _ = self.model(x)
        pred_sigmoid = torch.sigmoid(pred_logit)
        loss = torch.nn.BCELoss()(pred_sigmoid, y)
        loss.backward()

    @torch.enable_grad()
    def cal_ent_loss(self, data):
        x = torch.from_numpy(data['data']).to(dtype=torch.float32).to(self.device)
        self.base_optimizer.zero_grad()
        pred_logit, _ = self.model(x)
        pred_sigmoid = torch.sigmoid(pred_logit)
        ent = -(pred_sigmoid * torch.log(pred_sigmoid + 1e-06)).sum(1)
        ent_loss = ent.mean()
        ent_loss.backward()
        return ent_loss

    @torch.enable_grad()
    def cal_recon_loss(self, data):
        x = torch.from_numpy(data['data']).to(dtype=torch.float32).to(self.device)
        self.base_optimizer.zero_grad()
        recon_x, pred_logit, _ = self.model(x, rec=True)
        recon_sigmoid = torch.sigmoid(recon_x)
        recon_loss = torch.nn.MSELoss()(recon_sigmoid, x)
        recon_loss.backward()
        return recon_loss

    @torch.enable_grad()
    def cal_supres_loss(self, data):
        x = torch.from_numpy(data['data']).to(dtype=torch.float32).to(self.device)
        small_x = torch.nn.functional.interpolate(x, size=[x.shape[-2] // 2, x.shape[-1] // 2], mode='bilinear', align_corners=True)
        self.base_optimizer.zero_grad()
        supres_x, pred_logit, _ = self.model(small_x, sup=True)
        supres_sigmoid = torch.sigmoid(supres_x)
        supres_loss = torch.nn.MSELoss()(supres_sigmoid, x)
        supres_loss.backward()
        return supres_loss

    @torch.enable_grad()
    def cal_denoise_loss(self, data, mean=0.0, std=0.1):
        x = torch.from_numpy(data['data']).to(dtype=torch.float32).to(self.device)
        noise = torch.randn(x.size(), dtype=torch.float32) * std + mean
        noise_x = x + noise.to(self.device)
        self.base_optimizer.zero_grad()
        recon_x, pred_logit, _ = self.model(noise_x, den=True)
        recon_sigmoid = torch.sigmoid(recon_x)
        recon_loss = torch.nn.MSELoss()(recon_sigmoid, x)
        recon_loss.backward()
        return recon_loss

    @torch.enable_grad()
    def cal_rotate_loss(self, data):
        x = torch.from_numpy(data['data']).to(dtype=torch.float32).to(self.device)
        rotate_label = torch.randint(0, 5 + 1, (x.shape[0],)).to(self.device)
        aug_imgs = []
        for i in range(x.shape[0]):
            aug_imgs.append(augment(x[i], rotate_label[i]))
        aug_imgs = torch.stack(aug_imgs)
        self.base_optimizer.zero_grad()
        rotate_output, pred_logit, _ = self.model(aug_imgs, rot=True)
        rotate_loss = torch.nn.CrossEntropyLoss()(rotate_output, rotate_label)
        rotate_loss.backward()
        return rotate_loss

    @torch.enable_grad()
    def cal_consis_loss(self, data, criterion=torch.nn.BCEWithLogitsLoss()):
        x = torch.from_numpy(data['data']).to(dtype=torch.float32).to(self.device)
        weak_transform = Rotate_and_Flip()
        predictions = []
        with torch.no_grad():
            pred, fea = self.model(x)
            predictions.append(pred.detach().cpu())
        factors = [0, 1, 2, 3, 4]
        for factor in factors:
            x_weak = weak_transform(x, factor)
            with torch.no_grad():
                pred, fea = self.model(x_weak)
                pred = weak_transform.inverse(pred, factor)
                predictions.append(pred.detach().cpu())
        predictions = torch.stack(predictions).sigmoid()
        predictions = predictions.mean(0).to(self.device)
        self.base_optimizer.zero_grad()
        x_aug = augmentation_strong_style(data)
        x_aug = torch.from_numpy(x_aug).to(dtype=torch.float32).to(self.device)
        x_aug = normalize_image_to_0_1(x_aug)
        pred_aug, fea_aug = self.model(x_aug)
        consis_loss = criterion(pred_aug, predictions)
        consis_loss.backward()
        return consis_loss

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
    def step(self, data, aux='ent', pse='consis'):
        losses = {'consis': self.cal_consis_loss, 'ent': self.cal_ent_loss, 'recon': self.cal_recon_loss, 'rotate': self.cal_rotate_loss, 'denoise': self.cal_denoise_loss, 'supres': self.cal_supres_loss}
        with self.maybe_no_sync():
            aux_loss = losses[aux](data)
            self.perturb_weights_sub()
            pse_loss = losses[pse](data)
            cosine = self.get_cosine()
            self.unperturb()
        self._sync_grad()
        self.base_optimizer.param_groups[0]['lr'] = self.init_lr * custom_activation(cosine)
        self.base_optimizer.step()

def linear_activation(x):
    return 1 / 2 * (x + 1)

def softplus(x):
    return torch.log(1 + torch.exp(x))

def custom_activation(x):
    return 1 / 4 * (x + 1) ** 2
