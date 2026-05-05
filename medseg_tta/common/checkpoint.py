from pathlib import Path
import torch


def load_state_dict(path, map_location='cpu'):
    obj = torch.load(Path(path), map_location=map_location)
    if isinstance(obj, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
    return obj
