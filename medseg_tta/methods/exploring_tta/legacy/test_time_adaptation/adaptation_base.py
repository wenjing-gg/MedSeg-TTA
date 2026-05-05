from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit

class BaseAdaptation(nn.Module):

    def __init__(self, model, optimizer, loss, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss
        self.steps = steps
        assert steps > 0, 'Requires >= 1 step(s) to forward and update'
        self.episodic = episodic
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, self.loss_fn)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception('cannot reset without saved model/optimizer state')
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, loss_fn):
    outputs = model(x)
    loss = loss_fn(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

def collect_batch_norm_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm3d):
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
        if isinstance(m, nn.BatchNorm3d):
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
    has_bn = any([isinstance(m, nn.BatchNorm3d) for m in model.modules()])
    assert has_bn, 'tent needs normalization for its optimization'
