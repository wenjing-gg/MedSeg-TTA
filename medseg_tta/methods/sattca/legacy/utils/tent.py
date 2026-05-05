from copy import deepcopy
from loss import TestLoss, AdaptiveTestLoss
import torch
import torch.nn as nn
import torch.jit

class Tent(nn.Module):

    def __init__(self, model, optimizer, steps=10, entropy=False, episodic=False, use_adaptive_loss: bool=True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.base_loss = TestLoss()
        self.adaptive_loss = AdaptiveTestLoss()
        self.use_adaptive_loss = use_adaptive_loss
        self.steps = steps
        self.entropy = entropy
        assert steps > 0, 'tent requires >= 1 step(s) to forward and update'
        self.episodic = episodic
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
        self.bn_param_clones = []
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                for p in [m.weight, m.bias]:
                    if p is not None:
                        self.bn_param_clones.append((p, p.detach().clone()))

    def forward(self, x):
        if self.episodic:
            self.reset()
        if len(x) == 2:
            imgs, pseudo = x
            init_logits = None
            meta = {}
        elif len(x) == 3:
            imgs, pseudo, init_logits = x
            meta = {}
        else:
            imgs, pseudo, init_logits, meta = x
        cw = meta.get('consistency_weight', 0.0)
        rw = meta.get('reg_weight', 0.0)
        sw = meta.get('scale_weight', 1.0)
        bn_pairs = [(p, q) for p, q in self.bn_param_clones] if rw > 0 else None
        imgs.requires_grad_(True)
        pseudo.requires_grad_(True)
        outputs = None
        loss_dict = {}
        center_value = None
        for _ in range(self.steps):
            if self.entropy:
                outputs = forward_and_adapt(imgs, self.model, self.optimizer)
                loss_dict = {'total_loss': outputs.new_tensor(0.0)}
            else:
                outputs = self.model(imgs)
                if self.use_adaptive_loss and init_logits is not None:
                    loss_dict, center_value = self.adaptive_loss(outputs, pseudo, init_outputs=init_logits, bn_param_pairs=bn_pairs, consistency_weight=cw, reg_weight=rw, scale_weight=sw, boundary_mask=meta.get('boundary_mask', None), boundary_factor=meta.get('boundary_factor', 1.0), diff_mask_weight=meta.get('diff_mask_weight', 0.0), use_diff_mask_loss=meta.get('use_diff_mask_loss', False), diff_mask_boundary_only=meta.get('diff_mask_boundary_only', False))
                else:
                    loss_dict, center_value = self.base_loss(outputs, pseudo)
                loss = loss_dict['total_loss']
                if loss > 1e-07:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        return (outputs, loss_dict, center_value)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception('cannot reset without saved model/optimizer state')
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    return x.sigmoid() * x

@torch.enable_grad()
def forward_and_adapt(x, model, optimizer):
    outputs = model(x)
    loss = sigmoid_entropy(outputs).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

@torch.enable_grad()
def forward_and_click(x, y, model, optimizer, loss_fn):
    outputs = model(x)
    loss_dict, center_value = loss_fn(outputs, y)
    loss = loss_dict['total_loss']
    if loss > 1e-07:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return (outputs, loss_dict, center_value)

def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f'{nm}.{np}')
    return (params, names)

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return (model_state, optimizer_state)

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def check_model(model):
    is_training = model.training
    assert is_training, 'tent needs train mode: call model.train()'
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, 'tent needs params to update: check which require grad'
    assert not has_all_params, 'tent should not update all params: check which require grad'
    has_bn = any([isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)) for m in model.modules()])
    assert has_bn, 'tent needs normalization for its optimization'
