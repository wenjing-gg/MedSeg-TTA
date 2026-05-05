import torch, heapq, json, os, gc, pickle
import torch.nn as nn
from torch.utils.data import DataLoader
import torchio as tio
from test_time_adaptation import tent, entropy_KL, filter_inspector as inspector

def create_filter_inspector(net, filter_inspect_mode='Taylor', use_cuda=False):
    print('Creating filter inspector...', flush=True)
    filter_inspector = inspector.FilterInspector(net, mode=filter_inspect_mode, use_cuda=use_cuda)
    print('Done!', flush=True)
    return filter_inspector

def get_model_activations(filter_inspector, subject_list, transform=None, device='cpu'):
    print('Forward pass through data...', flush=True)
    filter_inspector.reset()
    subject_dataset = tio.SubjectsDataset(subject_list, transform=transform)
    dataloader = DataLoader(subject_dataset, batch_size=1, num_workers=0, shuffle=False)
    for subject_batch in dataloader:
        inputs = subject_batch['img'][tio.DATA].to(device).float()
        out = filter_inspector.forward(inputs)
        filter_inspector.unet.zero_grad()
        out.sum().backward()
        del inputs
        del out
        gc.collect()
        torch.cuda.empty_cache()
    print('Normalising activations...', flush=True)
    filter_inspector.normalize_ranks_per_layer()
    print('Done!', flush=True)
    return filter_inspector.get_filter_activations()

def save_source_data_activations(filter_inspector, subject_list, file_name, filter_inspect_mode='Taylor', week_num=21, device='cpu'):
    source_data_activations = get_model_activations(filter_inspector, subject_list, device=device)
    print('Saving activations...', flush=True)
    with open(file_name, 'wb') as f:
        pickle.dump(source_data_activations, f)
    print('Done!', flush=True)

def top_n_layers(dictionary, n, omit_batch_norm=False):
    if omit_batch_norm:
        dictionary = {k: v for k, v in dictionary.items() if not k.endswith('double_conv.1') or k.endswith('double_conv.4')}
    return dict(heapq.nlargest(n, dictionary.items(), key=lambda item: item[1]))

def get_transform_repr(transform):
    argument_vars = {'scales', 'degrees', 'translation', 'std_ranges', 'log_gamma_range'}
    transform_name = transform.__class__.__name__
    return transform_name + '_' + '_'.join([f'{k}_{v}' for k, v in transform.__dict__.items() if k in argument_vars])

def collect_params(model, max_diff_ranks_dict, force_include_batch_norm=False):
    params = []
    names = []
    for nm, m in model.named_modules():
        if nm in max_diff_ranks_dict.keys() or (force_include_batch_norm and isinstance(m, nn.BatchNorm3d)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f'{nm}.{np}')
    return (params, names)

def configure_model(model, max_diff_ranks_dict, force_include_batch_norm=False):
    model.train()
    model.requires_grad_(False)
    for name, module in model.named_modules():
        if name in max_diff_ranks_dict.keys() or (force_include_batch_norm and isinstance(module, nn.BatchNorm3d)):
            module.requires_grad_(True)
            if isinstance(module, nn.BatchNorm3d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
    return model

def configure_filter_inspect(model, filter_inspector, transform, subject_list, source_data_activations, filter_inspect_config):
    week_num = filter_inspect_config['week_num']
    steps = filter_inspect_config['steps']
    learning_rate = filter_inspect_config['lr']
    num_to_update = filter_inspect_config['num_to_update']
    device = filter_inspect_config['device']
    force_include_batch_norm = filter_inspect_config.get('force_include_batch_norm', False)
    save_activations = filter_inspect_config.get('save_activations', False)
    use_KL = filter_inspect_config.get('use_KL', False)
    filter_diff_activations_path = filter_inspect_config.get('filter_diff_activations_path', None)
    atlas_labels_path = filter_inspect_config['atlas_labels_path']
    if use_KL:
        lambd = filter_inspect_config.get('lambda', 1.0)
    if filter_diff_activations_path and os.path.exists(filter_diff_activations_path):
        print('Activations for transform already exist, loading them...', flush=True)
        with open(filter_diff_activations_path, 'r') as f:
            difference_activations = json.load(f)
    else:
        print('Getting transformed data activations...', flush=True)
        transformed_data_activations = get_model_activations(filter_inspector, subject_list, transform, device=device)
        print('Finding difference between activations...', flush=True)
        difference_activations = {}
        for layer_name, rank in source_data_activations.items():
            difference_activations[layer_name] = torch.norm(rank - transformed_data_activations[layer_name], p=2).item()
        if save_activations:
            print(f'Saving activations for {repr(transform)}...', flush=True)
            with open(filter_diff_activations_path, 'w') as f:
                json.dump(difference_activations, f, indent=4, separators=(',', ': '))
            print('Done!', flush=True)
    print(f'Updating {num_to_update} layers out of {len(list(difference_activations.items()))}...', flush=True)
    assert num_to_update > 0, 'No filters to update'
    max_difference_activations = top_n_layers(difference_activations, num_to_update, omit_batch_norm=force_include_batch_norm)
    print(f'Largest activation difference: {list(max_difference_activations.items())}', flush=True)
    print('Configuring model...', flush=True)
    model = configure_model(model, max_difference_activations, force_include_batch_norm=force_include_batch_norm)
    params, _ = collect_params(model, max_difference_activations, force_include_batch_norm=force_include_batch_norm)
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    if use_KL:
        net = entropy_KL.EntropyKL(model, optimizer, atlas_labels_path, steps=steps, lambd=lambd, week_num=week_num).to(device)
    else:
        net = tent.Tent(model, optimizer, steps=steps).to(device)
    return net
