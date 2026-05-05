import torch
import torch.jit
import SimpleITK as sitk
from test_time_adaptation import adaptation_base

class EntropyKL(adaptation_base.BaseAdaptation):

    def __init__(self, model, optimizer, atlas_labels_path, steps=1, lambd=1.0, episodic=False):
        super().__init__(model=model, optimizer=optimizer, loss=lambda x: entropy_KL_loss(x, atlas_labels_path, lambd=lambd), steps=steps, episodic=episodic)

def load_atlas_labels(atlas_labels_path):
    atlas_labels = sitk.GetArrayFromImage(sitk.ReadImage(atlas_labels_path))
    atlas_labels = torch.from_numpy(atlas_labels).long()
    return atlas_labels
'\nCalculate class ratios\nNote the atlas has additional classes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)\nThe model predicts classes 0 to 4 in the order background, choroidPlexus, ventricle, cavum, cerebellum\nIn the atlas, these correspond to classes 4, 5, 2, 3 respectively. See .../struc_index.txt for more info\nNote in both the Atlas and the model, class 0 is the background\n'

def calculate_atlas_class_ratios(label_vol, ordered_class_indices=[4, 5, 2, 3]):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    class_counts = torch.zeros((1, len(ordered_class_indices)), device=device)
    for i in range(len(ordered_class_indices)):
        mask = label_vol == ordered_class_indices[i]
        class_counts[:, i] = torch.sum(mask)
    return class_counts / torch.sum(class_counts, dim=1, keepdim=True)

def calculate_model_class_ratios(preds_vol, num_classes=5, omit_background=False):
    if omit_background:
        num_classes -= 1
    start_index = 1 if omit_background else 0
    class_sum = torch.sum(preds_vol[:, start_index:], dim=(2, 3, 4))
    class_sum = class_sum.unsqueeze(0)
    return class_sum / torch.sum(class_sum, dim=2, keepdim=True)

def calculate_KL_divergence(model_output, atlas_class_ratios, num_classes=5):
    model_class_ratios = calculate_model_class_ratios(model_output, num_classes, omit_background=True)
    eps = 1e-10
    atlas_class_ratios = atlas_class_ratios.unsqueeze(1)
    return torch.sum(atlas_class_ratios * torch.log(atlas_class_ratios / (model_class_ratios + eps) + eps), dim=2)

def entropy_KL_loss(x: torch.Tensor, atlas_labels_path: str, lambd: float=1.0) -> torch.Tensor:
    model_class_ratios = calculate_model_class_ratios(x.softmax(1))
    v_k = torch.pow(model_class_ratios, -1)
    v_k = v_k / torch.sum(v_k, dim=1)
    v_k = v_k[0]
    assert v_k.shape[1] == x.shape[1]
    v_k = v_k.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    mean_softmax_entropy = torch.mean(-(v_k * x.softmax(1) * x.log_softmax(1)).sum(1), dim=(1, 2, 3))
    atlas_labels = load_atlas_labels(atlas_labels_path)
    atlas_class_ratios = calculate_atlas_class_ratios(atlas_labels)
    kl = calculate_KL_divergence(x.softmax(1), atlas_class_ratios)
    return mean_softmax_entropy + lambd * torch.sum(kl, dim=0)
