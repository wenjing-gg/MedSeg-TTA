from copy import deepcopy
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.jit
from augmentation_utils import get_disp_field, get_rand_affine

class Tent(nn.Module):

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, 'tent requires >= 1 step(s) to forward and update'
        self.episodic = episodic
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception('cannot reset without saved model/optimizer state')
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

@torch.jit.script
def soft_dice_loss(smp_a, smp_b):
    B, _, D, H, W = smp_a.shape
    d = 2
    nominator = (2.0 * smp_a * smp_b).reshape(B, -1, D * H * W).mean(2)
    denominator = 1 / d * ((smp_a + smp_b) ** d).reshape(B, -1, D * H * W).mean(2)
    if denominator.sum() == 0.0:
        dice = nominator * 0.0 + 1.0
    else:
        dice = nominator / denominator
    return dice

@torch.enable_grad()
def forward_and_adapt(x, model, optimizer):
    device = x.device
    outputs = model(x)
    batch_size, _, *patch_size = x.size()
    identity_grid = F.affine_grid(torch.eye(4, device=device).repeat(batch_size, 1, 1)[:, :3], [batch_size, 1] + patch_size, align_corners=False)
    zero_grid = 0.0 * identity_grid
    grid_a = zero_grid.clone()
    grid_a_inverse = zero_grid.clone()
    R_a, R_a_inv = get_rand_affine(batch_size, flip=False, device=device)
    grid_a = grid_a + (F.affine_grid(R_a, [batch_size, 1] + patch_size, align_corners=False) - identity_grid)
    grid_a_inverse = grid_a_inverse + (F.affine_grid(R_a_inv, [batch_size, 1] + patch_size, align_corners=False) - identity_grid)
    grid_deformable, grid_deformable_inverse = get_disp_field(batch_size, patch_size, factor=0.5, interpolation_factor=5, device=device)
    grid_a = grid_a + grid_deformable
    grid_a_inverse = grid_a_inverse + grid_deformable_inverse
    grid_a = grid_a + identity_grid
    x_a = F.grid_sample(x, grid_a, padding_mode='border', align_corners=False)
    grid_b = zero_grid.clone()
    grid_b_inverse = zero_grid.clone()
    R_b, R_b_inv = get_rand_affine(batch_size, flip=False, device=device)
    grid_b = grid_b + (F.affine_grid(R_b, [batch_size, 1] + patch_size, align_corners=False) - identity_grid)
    grid_b_inverse = grid_b_inverse + (F.affine_grid(R_b_inv, [batch_size, 1] + patch_size, align_corners=False) - identity_grid)
    grid_deformable, grid_deformable_inverse = get_disp_field(batch_size, patch_size, factor=0.5, interpolation_factor=5, device=device)
    grid_b = grid_b + grid_deformable
    grid_b_inverse = grid_b_inverse + grid_deformable_inverse
    grid_b = grid_b + identity_grid
    x_b = F.grid_sample(x, grid_b, padding_mode='border', align_corners=False)
    target_a = model(x_a)
    target_b = model(x_b)
    grid_a_inverse = grid_a_inverse + identity_grid
    target_a = F.grid_sample(target_a, grid_a_inverse, align_corners=False)
    grid_b_inverse = grid_b_inverse + identity_grid
    target_b = F.grid_sample(target_b, grid_b_inverse, align_corners=False)
    common_content_mask = (target_a.sum(1, keepdim=True) > 0.0).float() * (target_b.sum(1, keepdim=True) > 0.0).float()
    sm_a = target_a.softmax(1) * common_content_mask
    sm_b = target_b.softmax(1) * common_content_mask
    loss = 1 - soft_dice_loss(sm_a, sm_b)[:, 1:].mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

def collect_params(model):
    params = []
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)
            names.append(name)
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
    for name, param in model.named_parameters():
        param.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def check_model(model):
    is_training = model.training
    assert is_training, 'tent needs train mode: call model.train()'
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    assert has_any_params, 'tent needs params to update: check which require grad'
    has_bn = any([isinstance(m, nn.BatchNorm3d) for m in model.modules()])
    if has_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):
                assert not m.track_running_stats, 'BatchNorm3d should not track running stats'
