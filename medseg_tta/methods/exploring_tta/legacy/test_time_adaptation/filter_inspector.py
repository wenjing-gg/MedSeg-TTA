import torch
import torch.nn.functional as F

class FilterInspector:

    def __init__(self, unet, mode='Taylor', use_cuda=False):
        self.unet = unet
        self.mode = mode
        self.use_cuda = use_cuda
        self.return_ranks = None
        self.activation_to_name = {}
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward_step(self, module, x, activation_index, module_name):
        x = module(x)
        if self.mode == 'Taylor':
            if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.BatchNorm3d):
                x.register_hook(self.compute_rank_taylor)
                self.activations.append(x)
                self.activation_to_name[activation_index] = module_name
                activation_index += 1
        elif self.mode == 'Random':
            if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.BatchNorm3d):
                x.register_hook(self.compute_rank_random)
                self.activations.append(x)
                self.activation_to_name[activation_index] = module_name
                activation_index += 1
        elif self.mode == 'L1':
            if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.BatchNorm3d):
                x.register_hook(self.compute_rank_l1)
                self.activations.append(x)
                self.activation_to_name[activation_index] = module_name
                activation_index += 1
        elif self.mode == 'L2':
            if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.BatchNorm3d):
                x.register_hook(self.compute_rank_l2)
                self.activations.append(x)
                self.activation_to_name[activation_index] = module_name
                activation_index += 1
        elif self.mode == 'L1_std':
            if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.BatchNorm3d):
                x.register_hook(self.compute_rank_l1_std)
                self.activations.append(x)
                self.activation_to_name[activation_index] = module_name
                activation_index += 1
        else:
            raise Exception('Unknown mode. Implemented modes: Taylor, Random, L1, L2')
        return (x, activation_index)

    def parse_double_conv_module(self, module, x, activation_index, block_name):
        for layer_name, layer in module._modules['double_conv']._modules.items():
            x, activation_index = self.forward_step(layer, x, activation_index, block_name + '.double_conv.' + layer_name)
        return (x, activation_index)

    def parse_down_module(self, module, x, activation_index, block_name):
        for layer_name, layer in module._modules['maxpool_conv']._modules.items():
            if not isinstance(layer, torch.nn.MaxPool3d) and (not isinstance(layer, torch.nn.Dropout)):
                x, activation_index = self.parse_double_conv_module(layer, x, activation_index, block_name + '.maxpool_conv.' + layer_name)
            else:
                x = layer(x)
        return (x, activation_index)

    def parse_up_module(self, module, x, skip_connection, activation_index, block_name):
        for layer_name, layer in module._modules.items():
            if not isinstance(layer, torch.nn.Upsample) and (not isinstance(layer, torch.nn.ConvTranspose3d)):
                x, activation_index = self.parse_double_conv_module(layer, x, activation_index, block_name + '.' + layer_name)
            else:
                x = layer(x)
                diffX = skip_connection.size()[2] - x.size()[2]
                diffY = skip_connection.size()[3] - x.size()[3]
                diffZ = skip_connection.size()[4] - x.size()[4]
                x = F.pad(x, [diffZ // 2, diffZ - diffZ // 2, diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2])
                x = torch.cat([skip_connection, x], dim=1)
        return (x, activation_index)

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        skip_connections = []
        activation_index = 0
        self.unet.requires_grad_(True)
        for block_name, block in self.unet._modules['module']._modules.items():
            if block_name == 'inc':
                x, activation_index = self.parse_double_conv_module(block, x, activation_index, 'module.' + block_name)
                skip_connections.append(x)
            elif block_name == 'downs':
                for down_index, down in block._modules.items():
                    x, activation_index = self.parse_down_module(down, x, activation_index, 'module.' + block_name + '.' + down_index)
                    skip_connections.append(x)
                skip_connections.pop()
            elif block_name == 'ups':
                for up_index, up in block._modules.items():
                    skip = skip_connections.pop()
                    x, activation_index = self.parse_up_module(up, x, skip, activation_index, 'module.' + block_name + '.' + up_index)
            elif block_name == 'outc':
                for module_name, module in block._modules.items():
                    x, activation_index = self.forward_step(module, x, activation_index, 'module.' + block_name + '.' + module_name)
            else:
                raise Exception('Unknown block name. Implemented blocks: inc, downs, ups, outc')
        return x

    def compute_rank_taylor(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        taylor = activation * grad
        taylor = taylor.mean(dim=(0, 2, 3, 4)).data
        if activation_index not in self.filter_ranks.keys():
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()
        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def compute_rank_l1(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        l1 = torch.norm(activation, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = l1.mean(dim=0).data
        if activation_index not in self.filter_ranks.keys():
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()
        self.filter_ranks[activation_index] += l1.cpu()
        self.grad_index += 1

    def compute_rank_l1_std(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        l1 = torch.norm(activation, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = torch.norm(l1, p=1, dim=2)
        l1 = l1.std(dim=0).data
        if activation_index not in self.filter_ranks.keys():
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()
        self.filter_ranks[activation_index] += l1.cpu()
        self.grad_index += 1

    def compute_rank_l2(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        l2 = torch.norm(activation, p=2, dim=2)
        l2 = torch.norm(l2, p=2, dim=2)
        l2 = torch.norm(l2, p=2, dim=2)
        l2 = l2.mean(dim=0).data
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if activation_index not in self.filter_ranks.keys():
                self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
        self.filter_ranks[activation_index] += l2.cpu()
        self.grad_index += 1

    def compute_rank_random(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        if activation.size()[1] != 1:
            taylor = activation * grad
            taylor = taylor.mean(dim=(0, 2, 3, 4)).data
            random = torch.rand(taylor.size())
        else:
            taylor = activation * grad
            taylor = taylor.mean(dim=(0, 2, 3, 4)).data
            random = torch.ones(taylor.size()) * 100
        if activation_index not in self.filter_ranks.keys():
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            if self.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()
        self.filter_ranks[activation_index] += random.cpu()
        self.grad_index += 1

    def get_filter_activations(self):
        filter_activations = {}
        for i in sorted(self.filter_ranks.keys()):
            filter_activations[self.activation_to_name[i]] = self.filter_ranks[i]
        return filter_activations

    def normalize_ranks_per_layer(self):
        if not self.mode == 'Random':
            for i in self.filter_ranks:
                v = torch.abs(self.filter_ranks[i])
                v = v / torch.sqrt(torch.sum(v * v))
                self.filter_ranks[i] = v.cpu()
        self.return_ranks = self.filter_ranks
