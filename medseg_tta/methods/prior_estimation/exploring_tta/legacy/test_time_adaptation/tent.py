from copy import deepcopy
import torch, gc
import torch.nn as nn
import torch.jit
from test_time_adaptation import adaptation_base

class Tent(adaptation_base.BaseAdaptation):

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__(model=model, optimizer=optimizer, loss=softmax_entropy, steps=steps, episodic=episodic)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(-(x.softmax(1) * x.log_softmax(1)).sum(1), dim=(1, 2, 3))

@torch.enable_grad()
def forward_and_adapt(x, model, optimizer):
    outputs = model(x)
    loss = softmax_entropy(outputs).sum(0)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    return outputs

def collect_params(model):
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
