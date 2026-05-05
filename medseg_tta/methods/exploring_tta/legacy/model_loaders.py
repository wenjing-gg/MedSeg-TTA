import torch, pickle, os
from unet_architecture import UNet
from test_time_adaptation import adaptation_base, tent, hist_matching, entropy_KL, filter_inspect_utils

def load_base_model(device, model_weights_path):
    net = torch.nn.DataParallel(UNet(1, 5, min_featuremaps=16, depth=5))
    model = torch.load(model_weights_path, map_location=device)
    net.load_state_dict(model['model_state_dict'])
    return net

def load_hist_match_model(device, model_weights_path, volume_to_match_to):
    net = load_base_model(device, model_weights_path)
    return hist_matching.HistMatching(net, volume_to_match_to)

def load_tent_model(device, model_weights_path, lr, steps):
    net = load_base_model(device, model_weights_path)
    net = adaptation_base.configure_model(net)
    params, _ = adaptation_base.collect_batch_norm_params(net)
    optimizer = torch.optim.Adam(params, lr=lr)
    tented_net = tent.Tent(net, optimizer, steps=steps).to(device)
    return tented_net

def load_entropy_KL_model(device, model_weights_path, lr, lambd, steps, atlas_labels_path):
    net = load_base_model(device, model_weights_path)
    net = adaptation_base.configure_model(net)
    params, _ = adaptation_base.collect_batch_norm_params(net)
    optimizer = torch.optim.Adam(params, lr=lr)
    entropy_KL_net = entropy_KL.EntropyKL(net, optimizer, atlas_labels_path, lambd=lambd, steps=steps).to(device)
    return entropy_KL_net

def load_source_data_activations(source_data_activations_path):
    if os.path.exists(source_data_activations_path):
        print('Source data activations already exist, loading them...', flush=True)
        with open(source_data_activations_path, 'rb') as f:
            source_data_activations = pickle.load(f)
    else:
        raise ValueError('Source data activations do not exist.')
    return source_data_activations

def load_filter_inspector_model(device, model_weights_path, subject_list, lr, steps, source_data_activations_path, num_to_update=1, week_num=21, force_include_batch_norm=False, use_KL=False, lambd=1.0):
    net = load_base_model(device, model_weights_path)
    filter_inspect_config = {'week_num': week_num, 'steps': steps, 'lr': lr, 'num_to_update': num_to_update, 'subject_list': subject_list, 'device': device, 'filter_inspect_mode': 'Taylor', 'force_include_batch_norm': force_include_batch_norm, 'use_KL': use_KL, 'hemisphere_split': False, 'lambda': lambd}
    source_data_activations = load_source_data_activations(source_data_activations_path)
    filter_inspector = filter_inspect_utils.create_filter_inspector(net, use_cuda=torch.cuda.is_available())
    filter_inspect_model = filter_inspect_utils.configure_filter_inspect(filter_inspector.unet, filter_inspector, None, subject_list, source_data_activations, filter_inspect_config)
    return filter_inspect_model
