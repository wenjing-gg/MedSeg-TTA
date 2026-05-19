import torch
from torch import nn
import numpy as np
import random
import os
import shutil

def model_size(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


def split_dataset(IDs, num_train):
    num = len(IDs)
    IDs_list = list(range(0, num))
    IDs_train = random.sample(IDs_list, num_train)
    IDs_val = set(IDs_list).symmetric_difference(IDs_train)
    IDs_val = tuple(IDs_val)
    return [IDs[id] for id in IDs_train], [IDs[id] for id in IDs_val]


### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def adjust_learning_rate(init_lr, optimizer, epoch, n=10):
    lr = init_lr * (0.1 ** (epoch // n))
    # lr = poly_lr(epoch, 15, init_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def poly_lr(epoch, max_epochs, init_lr, exponent=0.9):
    return init_lr * (1 - epoch / max_epochs)**exponent


def get_neighbour_index(mask, patch_size):
    rows, cols = mask.shape
    i, j = np.nonzero(mask)
    i = np.tile(i, (patch_size ** 2, 1))
    j = np.tile(j, (patch_size ** 2, 1))
    add = np.arange(- (patch_size // 2), patch_size // 2 + 1).reshape(-1, 1)
    add_j = np.tile(add, (patch_size, 1))
    add_j = np.tile(add_j, (1, i.shape[1]))
    add_i = np.tile(add, (1, patch_size)).reshape(patch_size ** 2, 1)
    add_i = np.tile(add_i, (1, i.shape[1]))
    i = i + add_i
    j = j + add_j
    i[i < 0] = 0
    j[j < 0] = 0
    i[i >= rows] = rows - 1
    j[j >= cols] = cols - 1
    index = rows * i + j
    return index.transpose(1, 0)


def get_nl_mask(num_rows, num_cols, n_size):
    mask = np.ones([num_rows, num_cols])
    index = get_neighbour_index(mask, n_size)
    output = np.zeros([mask.size, mask.size])
    for i in range(mask.size):
        output[i, index[i, :]] = 1

    return output


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False


def get_atlas(img, label, atlas_img, atlas_label):
    l, r, c = img.shape
    img = np.expand_dims(img, axis=1)
    label = np.expand_dims(label, axis=1)
    output_img = []
    output_label = []
    num_atlas = len(atlas_img)
    for i in range(l):
        tempt_img = []
        tempt_label = []
        for j in range(num_atlas):
            img_atlas_j = atlas_img[j].reshape(atlas_img[j].shape[0], -1)
            ssd = np.sum((img.reshape(l, -1)[i, :] - img_atlas_j) ** 2, axis=1, keepdims=True)
            index = np.argsort(ssd, axis=0)
            tempt_img.append(atlas_img[j][index[0]])
            tempt_label.append(atlas_label[j][index[0]])
            
        tempt_img = np.concatenate(tempt_img, axis=0)
        tempt_label = np.concatenate(tempt_label, axis=0)
        output_img.append(np.expand_dims(tempt_img, axis=0))
        output_label.append(np.expand_dims(tempt_label, axis=0))
    output_img = np.concatenate(output_img, axis=0)
    output_label = np.concatenate(output_label, axis=0)
    output_img = np.concatenate([img, output_img], axis=1)
    output_label = np.concatenate([label, output_label], axis=1)
    return output_img, output_label


def update_meta_parameters(meta_model, model):
    for name, param in meta_model.named_parameters():
        param.data = torch.clone(model.get_parameter(name))


# def load_pretrained(model, path):
#     if os.path.isfile(path):
#         print('Find pretrained param')
#         pretrained_dict = torch.load(path)
#         print("=> loaded pretrained params '{}'".format(path))
#         model_dict = model.state_dict()              
#         pretrained_dict = {k: v for k, v in pretrained_dict.items()
#                            if k in model_dict.keys()}
#         for k, _ in pretrained_dict.items():
#             if k in model_dict.keys():
#                 print(k)
#         model_dict.update(pretrained_dict)
#         model.load_state_dict(model_dict)


def load_pretrained(model, pretrained_dict):
    model_dict = model.state_dict()              
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("=> loaded pretrained params.")

def set_requires_grad(net, requires_grad=False):
    """
    Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    """
    for param in net.parameters():
        param.requires_grad = requires_grad


def copy_code(logdir, runroot):
    tgt_code_dir = os.path.join(logdir, 'code')
    shutil.copytree(runroot, tgt_code_dir)        