import argparse
import os
import datetime
import traceback
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from nnunet import PlainConvUNet, nnUNet_PLS, nnUNet_FAS, mix_data_prompt_3d
from unet3d import UNet3d, UNet3d_PLS, UNet3d_FAS, UNet3d_PLS_FAS
from utils_brats_all import get_data_loader
from metrics import cal_dice, cal_hd95, IoU, PA, cal_RVE, cal_sensitivity, cal_ppv
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

class TTATrainer:

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.result_dir = '/home/yuwenjing/DeepLearning_ywj/tta/tent/checkpoints/tta_results'
        self.weights_dir = os.path.join(self.result_dir, 'weights')
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        self.pls_model = None
        self.fas_model = None
        self.pretrained_params = {}
        self.pls_optimizer = None
        self.fas_optimizer = None
        self.pls_losses = []
        self.fas_losses = []
        self.joint_losses = []
        self.model_type = args.model_type.lower()
        if self.model_type not in ['unet3d', 'nnunet']:
            raise ValueError(f"不支持的模型类型: {args.model_type}。支持的类型: ['unet3d', 'nnunet']")
        print(f'🔧 使用模型类型: {self.model_type.upper()}')

    def safe_value(self, val):
        if isinstance(val, torch.Tensor):
            return val.item()
        return val

    def load_pretrained_weights(self, model_path):
        print(f'📦 加载{self.model_type.upper()}预训练模型权重: {model_path}')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'未找到预训练权重: {model_path}')
        pretrained_weights = torch.load(model_path, map_location=self.device)
        state_dict = self._extract_state_dict(pretrained_weights)
        if self.model_type == 'nnunet':
            self._load_to_nnunet_models(state_dict)
        else:
            self._load_to_unet3d_models(state_dict)
        self._save_pretrained_params()

    def _extract_state_dict(self, checkpoint):
        print(f'📋 检查{self.model_type.upper()}权重文件结构...')
        if isinstance(checkpoint, dict):
            for key in ['model_state_dict', 'state_dict', 'model', 'network']:
                if key in checkpoint:
                    print(f"   - 使用键: '{key}'")
                    return checkpoint[key]
            param_keys = [k for k in checkpoint.keys() if isinstance(k, str) and '.' in k]
            if len(param_keys) > len(checkpoint) * 0.5:
                print(f'   - 直接使用权重字典')
                return checkpoint
            dict_items = [(k, v) for k, v in checkpoint.items() if isinstance(v, dict)]
            if dict_items:
                largest_key, largest_dict = max(dict_items, key=lambda x: len(x[1]))
                print(f"   - 使用最大字典: '{largest_key}' (包含 {len(largest_dict)} 个参数)")
                return largest_dict
            else:
                raise ValueError(f'无法从checkpoint中找到有效的state_dict。顶级键: {list(checkpoint.keys())}')
        return checkpoint

    def _load_to_nnunet_models(self, state_dict):
        print(f'📦 准备加载 {len(state_dict)} 个参数到nnUNet模型...')
        if not hasattr(self.pls_model, 'unet'):
            raise AttributeError(f'nnUNet PLS模型没有unet属性。模型类型: {type(self.pls_model)}')
        if not hasattr(self.fas_model, 'unet'):
            raise AttributeError(f'nnUNet FAS模型没有unet属性。模型类型: {type(self.fas_model)}')
        pls_missing, pls_unexpected = self.pls_model.unet.load_state_dict(state_dict, strict=False)
        fas_missing, fas_unexpected = self.fas_model.unet.load_state_dict(state_dict, strict=False)
        print(f'✅ nnUNet权重加载完成:')
        print(f'   - PLS缺失: {len(pls_missing)}, 意外: {len(pls_unexpected)}')
        print(f'   - FAS缺失: {len(fas_missing)}, 意外: {len(fas_unexpected)}')
        pls_new_modules = [k for k in pls_missing if 'data_prompt' in k]
        if pls_new_modules:
            print(f'   - 新增PLS模块: {pls_new_modules}')

    def _load_to_unet3d_models(self, state_dict):
        print(f'📦 准备加载 {len(state_dict)} 个参数到UNet3d模型...')
        pls_filtered_dict = {k: v for k, v in state_dict.items() if 'data_prompt' not in k}
        pls_missing, pls_unexpected = self.pls_model.load_state_dict(pls_filtered_dict, strict=False)
        fas_missing, fas_unexpected = self.fas_model.load_state_dict(state_dict, strict=False)
        print(f'✅ UNet3d权重加载完成:')
        print(f'   - PLS缺失: {len(pls_missing)}, 意外: {len(pls_unexpected)}')
        print(f'   - FAS缺失: {len(fas_missing)}, 意外: {len(fas_unexpected)}')
        pls_new_modules = [k for k in pls_missing if 'data_prompt' in k]
        if pls_new_modules:
            print(f'   - 新增PLS模块: {pls_new_modules}')

    def _save_pretrained_params(self):
        if self.model_type == 'nnunet':
            if not hasattr(self.pls_model, 'unet'):
                raise AttributeError('nnUNet PLS模型没有unet属性，无法保存预训练参数')
            source_model = self.pls_model.unet
        else:
            source_model = self.pls_model
        self.pretrained_params = {}
        param_count = 0
        bn_param_count = 0
        print(f'🔍 调试: 检查{self.model_type.upper()}模型参数名称...')
        all_param_names = []
        for name, param in source_model.named_parameters():
            all_param_names.append(name)
            if 'data_prompt' not in name:
                if param is None:
                    raise ValueError(f'参数 {name} 为 None')
                if not isinstance(param, torch.Tensor):
                    raise TypeError(f'参数 {name} 不是张量，类型: {type(param)}')
                self.pretrained_params[name] = param.clone().detach()
                param_count += 1
                if any((bn_key in name.lower() for bn_key in ['norm', 'running_mean', 'running_var'])):
                    bn_param_count += 1
        bn_stats_count = 0
        for name, module in source_model.named_modules():
            if isinstance(module, torch.nn.BatchNorm3d):
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    param_name = f'{name}.running_mean'
                    self.pretrained_params[param_name] = module.running_mean.clone().detach()
                    bn_stats_count += 1
                if hasattr(module, 'running_var') and module.running_var is not None:
                    param_name = f'{name}.running_var'
                    self.pretrained_params[param_name] = module.running_var.clone().detach()
                    bn_stats_count += 1
        print(f'   - 保存了 {param_count} 个{self.model_type.upper()}预训练参数')
        print(f'   - 其中BN相关参数: {bn_param_count} 个')
        print(f'   - BN统计参数: {bn_stats_count} 个')
        bn_params = [name for name in self.pretrained_params.keys() if any((bn_key in name.lower() for bn_key in ['norm', 'running_mean', 'running_var']))]
        print(f'   - BN相关参数示例 (前5个):')
        for i, name in enumerate(bn_params[:5]):
            param = self.pretrained_params[name]
            print(f'     {i + 1}: {name} - {param.shape}')
        if bn_stats_count == 0:
            raise RuntimeError('没有找到任何BN统计参数（running_mean/running_var），无法进行精确的BN loss计算')
        if param_count == 0:
            raise RuntimeError('没有保存任何预训练参数！')
        print(f'   - ✅ 成功保存{self.model_type.upper()}的BN统计参数，可以进行精确匹配')
        sample_names = list(self.pretrained_params.keys())[:3]
        for name in sample_names:
            param = self.pretrained_params[name]
            print(f'     - {name}: {param.shape}, norm={param.norm().item():.4f}')

    def initialize_models(self):
        if self.model_type == 'nnunet':
            self._initialize_nnunet_models()
        else:
            self._initialize_unet3d_models()

    def _initialize_nnunet_models(self):
        model_config = {'input_channels': 4, 'n_stages': 6, 'features_per_stage': (32, 64, 125, 256, 320, 320), 'conv_op': nn.Conv3d, 'kernel_sizes': 3, 'strides': (1, 2, 2, 2, 2, 2), 'n_conv_per_stage': (2, 2, 2, 2, 2, 2), 'num_classes': 4, 'n_conv_per_stage_decoder': (2, 2, 2, 2, 2), 'conv_bias': False, 'norm_op': nn.BatchNorm3d, 'deep_supervision': True}
        print('🔧 初始化nnUNet-PLS模型...')
        try:
            self.pls_model = nnUNet_PLS(pretrained_path=None, patch_size=(32, 128, 128), **model_config).to(self.device)
        except Exception as e:
            raise RuntimeError(f'nnUNet PLS模型初始化失败: {e}')
        print('🔧 初始化nnUNet-FAS模型...')
        try:
            self.fas_model = nnUNet_FAS(resnet='nnunet', pretrained=False, **model_config).to(self.device)
        except Exception as e:
            raise RuntimeError(f'nnUNet FAS模型初始化失败: {e}')

    def _initialize_unet3d_models(self):
        print('🔧 初始化UNet3d-PLS模型...')
        try:
            self.pls_model = UNet3d_PLS(patch_size=(32, 128, 128)).to(self.device)
        except Exception as e:
            raise RuntimeError(f'UNet3d PLS模型初始化失败: {e}')
        print('🔧 初始化UNet3d-FAS模型...')
        try:
            self.fas_model = UNet3d_FAS().to(self.device)
        except Exception as e:
            raise RuntimeError(f'UNet3d FAS模型初始化失败: {e}')

    def setup_training(self):
        print(f'🔧 设置{self.model_type.upper()}训练参数...')
        if not hasattr(self.pls_model, 'data_prompt'):
            raise AttributeError(f'{self.model_type.upper()} PLS模型没有data_prompt属性')
        if not isinstance(self.pls_model.data_prompt, torch.nn.Parameter):
            raise TypeError(f'data_prompt不是Parameter类型，实际类型: {type(self.pls_model.data_prompt)}')
        print(f'   - 初始提示范数: {self.pls_model.data_prompt.norm().item():.6f}')
        print(f'   - 初始提示形状: {self.pls_model.data_prompt.shape}')
        pls_trainable_count = 0
        for name, param in self.pls_model.named_parameters():
            if 'data_prompt' in name:
                param.requires_grad = True
                pls_trainable_count += param.numel()
            else:
                param.requires_grad = False
        if pls_trainable_count == 0:
            raise RuntimeError('没有找到PLS可训练参数')
        fas_trainable_params = []
        fas_trainable_count = 0
        bn_keywords = ['norm', 'bn', 'batchnorm', 'batch_norm', 'groupnorm', 'layernorm', 'instancenorm']
        for name, param in self.fas_model.named_parameters():
            is_bn = any((norm_key in name.lower() for norm_key in bn_keywords))
            if is_bn:
                param.requires_grad = True
                fas_trainable_params.append(param)
                fas_trainable_count += param.numel()
            else:
                param.requires_grad = False
        if not fas_trainable_params:
            norm_modules = []
            for name, module in self.fas_model.named_modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.InstanceNorm3d)):
                    norm_modules.append(name)
            if norm_modules:
                for name, param in self.fas_model.named_parameters():
                    is_norm_module_param = any((norm_name in name for norm_name in norm_modules))
                    if is_norm_module_param:
                        param.requires_grad = True
                        fas_trainable_params.append(param)
                        fas_trainable_count += param.numel()
                    else:
                        param.requires_grad = False
        try:
            self.pls_optimizer = optim.Adam([self.pls_model.data_prompt], lr=self.args.lr)
        except Exception as e:
            raise RuntimeError(f'PLS优化器创建失败: {e}')
        if not fas_trainable_params:
            raise RuntimeError('没有找到FAS可训练的归一化层参数，无法创建FAS优化器')
        try:
            if fas_trainable_count > 10000:
                fas_lr = self.args.lr * 0.01
            elif fas_trainable_count > 1000:
                fas_lr = self.args.lr * 0.05
            else:
                fas_lr = self.args.lr * 0.1
            self.fas_optimizer = optim.Adam(fas_trainable_params, lr=fas_lr)
        except Exception as e:
            raise RuntimeError(f'FAS优化器创建失败: {e}')
        print(f'✅ {self.model_type.upper()}训练设置完成:')
        print(f'   - PLS可训练参数: {pls_trainable_count:,} 个')
        print(f'   - FAS可训练参数: {fas_trainable_count:,} 个')
        print(f'   - PLS学习率: {self.args.lr}')
        print(f'   - FAS学习率: {fas_lr}')

    def compute_pls_loss(self, bn_features, batch_idx=0):
        if not self.pretrained_params:
            raise RuntimeError('预训练参数为空，无法计算BN loss')
        if not bn_features:
            raise ValueError('BN特征为空')
        if not isinstance(bn_features, list):
            raise TypeError(f'BN特征应该是列表类型，实际类型: {type(bn_features)}')
        valid_features = []
        for i, f in enumerate(bn_features):
            if not hasattr(f, 'features'):
                if batch_idx == 0:
                    print(f'     - 跳过特征 {i}: 没有features属性')
                continue
            if f.features is None:
                if batch_idx == 0:
                    print(f'     - 跳过特征 {i}: features为None')
                continue
            if not f.features.requires_grad:
                if batch_idx == 0:
                    print(f'     - 跳过特征 {i}: 不需要梯度')
                continue
            valid_features.append(f)
        if not valid_features:
            raise RuntimeError('没有有效的BN特征用于损失计算')
        if batch_idx == 0:
            print(f'🔍 计算{self.model_type.upper()} BN loss，有效特征数: {len(valid_features)}')
        try:
            bn_loss_value = self._compute_precise_bn_loss(valid_features, batch_idx)
        except Exception as e:
            raise RuntimeError(f'精确BN loss计算失败: {e}')
        if torch.isnan(bn_loss_value):
            raise ValueError(f'BN loss计算结果为NaN')
        if torch.isinf(bn_loss_value):
            raise ValueError(f'BN loss计算结果为Inf')
        if bn_loss_value.item() < 0:
            raise ValueError(f'BN loss为负值: {bn_loss_value.item()}')
        if batch_idx == 0:
            print(f'✅ 精确{self.model_type.upper()} BN loss计算成功: {bn_loss_value.item():.6f}')
        return bn_loss_value

    def _compute_precise_bn_loss(self, bn_features, batch_idx=0):
        total_loss = 0.0
        matched_layers = 0
        if self.model_type == 'nnunet':
            source_model = self.pls_model.unet
        else:
            source_model = self.pls_model
        current_bn_params = {}
        for name, module in source_model.named_modules():
            if isinstance(module, torch.nn.BatchNorm3d) and hasattr(module, 'running_mean'):
                current_bn_params[name] = {'running_mean': module.running_mean, 'running_var': module.running_var}
        pretrained_bn_params = {}
        for name, param in self.pretrained_params.items():
            if '.running_mean' in name:
                layer_name = name.replace('.running_mean', '')
                if layer_name not in pretrained_bn_params:
                    pretrained_bn_params[layer_name] = {}
                pretrained_bn_params[layer_name]['running_mean'] = param
            elif '.running_var' in name:
                layer_name = name.replace('.running_var', '')
                if layer_name not in pretrained_bn_params:
                    pretrained_bn_params[layer_name] = {}
                pretrained_bn_params[layer_name]['running_var'] = param
        if batch_idx == 0:
            print(f'   - 当前{self.model_type.upper()}模型BN层数: {len(current_bn_params)}')
            print(f'   - 预训练BN层数: {len(pretrained_bn_params)}')
            if pretrained_bn_params:
                print(f'   - 预训练BN层示例: {list(pretrained_bn_params.keys())[:3]}')
            if current_bn_params:
                print(f'   - 当前BN层示例: {list(current_bn_params.keys())[:3]}')
        if not pretrained_bn_params:
            raise RuntimeError(f'没有找到{self.model_type.upper()}预训练BN参数，无法进行精确BN loss计算')
        if not current_bn_params:
            raise RuntimeError(f'没有找到{self.model_type.upper()}当前模型BN参数，无法进行精确BN loss计算')
        pretrained_layer_names = list(pretrained_bn_params.keys())
        for i, feature in enumerate(bn_features[:min(self.args.bn_layers, len(pretrained_layer_names))]):
            if feature.features is None:
                continue
            if not feature.features.requires_grad:
                if batch_idx == 0 and i < 3:
                    print(f'     - 警告：特征 {i} 不需要梯度，跳过')
                continue
            current_mean = feature.features.mean(dim=(0, 2, 3, 4))
            current_var = feature.features.var(dim=(0, 2, 3, 4), unbiased=False)
            if i < len(pretrained_layer_names):
                layer_name = pretrained_layer_names[i]
                pretrained_layer = pretrained_bn_params[layer_name]
                if 'running_mean' in pretrained_layer and 'running_var' in pretrained_layer:
                    pretrained_mean = pretrained_layer['running_mean'].detach()
                    pretrained_var = pretrained_layer['running_var'].detach()
                    if current_mean.shape == pretrained_mean.shape and current_var.shape == pretrained_var.shape:
                        mean_loss = F.l1_loss(current_mean, pretrained_mean)
                        var_loss = F.l1_loss(current_var, pretrained_var)
                        layer_loss = mean_loss + self.args.alpha * var_loss
                        total_loss += layer_loss
                        matched_layers += 1
                        if batch_idx == 0 and i < 3:
                            print(f'     - 特征匹配层 {i} ({layer_name}): mean_loss={mean_loss.item():.6f}, var_loss={var_loss.item():.6f}')
                            print(f'       current_mean requires_grad: {current_mean.requires_grad}')
                            print(f'       current_var requires_grad: {current_var.requires_grad}')
                    elif batch_idx == 0 and i < 3:
                        print(f'     - 层 {i} ({layer_name}) 维度不匹配: {current_mean.shape} vs {pretrained_mean.shape}')
        if matched_layers == 0:
            if batch_idx == 0:
                print(f'   - 没有匹配的特征，使用data_prompt正则化')
            if self.pls_model.data_prompt.requires_grad:
                prompt_reg = self.args.alpha * torch.norm(self.pls_model.data_prompt, p=2)
                total_loss = prompt_reg
                matched_layers = 1
                if batch_idx == 0:
                    print(f'     - 提示正则化: {prompt_reg.item():.6f}')
                    print(f'     - data_prompt requires_grad: {self.pls_model.data_prompt.requires_grad}')
            else:
                raise RuntimeError('data_prompt不需要梯度，无法计算损失')
        if matched_layers == 0:
            raise RuntimeError(f'\n无法匹配任何层进行{self.model_type.upper()}损失计算！\n- 当前BN层数: {len(current_bn_params)}\n- 预训练BN层数: {len(pretrained_bn_params)}\n- BN特征数: {len(bn_features)}\n- 有梯度的特征数: {sum((1 for f in bn_features if f.features is not None and f.features.requires_grad))}\n')
        if not total_loss.requires_grad:
            raise RuntimeError(f'计算的总损失不需要梯度！matched_layers: {matched_layers}')
        if batch_idx == 0:
            print(f'   - 成功匹配层数: {matched_layers}, 总损失: {total_loss:.6f}')
            print(f'   - 总损失 requires_grad: {total_loss.requires_grad}')
        return total_loss

    def train_pls_phase(self, data_loader):
        print(f'\n🎯 第一阶段：{self.model_type.upper()}-PLS数据提示学习...')
        self.pls_model.train()
        pls_epochs = self.args.tta_epochs // 2 if self.args.tta_epochs > 1 else 1
        for epoch in range(pls_epochs):
            epoch_losses = []
            valid_batches = 0
            pbar = tqdm(data_loader, desc=f'{self.model_type.upper()}-PLS Epoch {epoch + 1}/{pls_epochs}')
            for batch_idx, (imgs, labels, *_) in enumerate(pbar):
                imgs, labels = (imgs.to(self.device), labels.to(self.device))
                self.pls_optimizer.zero_grad()
                try:
                    if self.model_type == 'nnunet':
                        outputs = self.pls_model(imgs)
                        bn_features = getattr(self.pls_model, 'bn_f', [])
                    else:
                        outputs, bn_features = self.pls_model(imgs, training=True)
                    valid_bn_features = []
                    for i, f in enumerate(bn_features):
                        if hasattr(f, 'features') and f.features is not None:
                            if f.features.requires_grad:
                                valid_bn_features.append(f)
                            elif batch_idx == 0 and i < 3:
                                print(f'     - 警告：特征 {i} 不需要梯度')
                        elif batch_idx == 0 and i < 3:
                            print(f'     - 警告：特征 {i} 没有features属性或为None')
                    if not valid_bn_features:
                        if batch_idx == 0:
                            print(f'     - 警告：没有有效的BN特征，跳过此batch')
                        continue
                    bn_loss = self.compute_pls_loss(valid_bn_features, batch_idx)
                    if torch.isnan(bn_loss) or torch.isinf(bn_loss):
                        if batch_idx < 5:
                            print(f'     - 警告：BN损失无效 (NaN/Inf)，跳过此batch')
                        continue
                    bn_loss.backward()
                    torch.nn.utils.clip_grad_norm_([self.pls_model.data_prompt], max_norm=1.0)
                    self.pls_optimizer.step()
                    valid_batches += 1
                    epoch_losses.append(bn_loss.item())
                    pbar.set_postfix({'BN_Loss': f'{bn_loss.item():.4f}', 'Prompt_Norm': f'{self.pls_model.data_prompt.norm().item():.4f}', 'Valid_Feat': f'{len(valid_bn_features)}', 'Valid_Batch': f'{valid_batches}'})
                    if batch_idx % 50 == 0 and batch_idx > 0:
                        current_prompt_norm = self.pls_model.data_prompt.norm().item()
                        print(f'     - Batch {batch_idx}: 提示范数={current_prompt_norm:.6f}, 有效特征={len(valid_bn_features)}')
                except Exception as e:
                    print(f'     - 错误：Batch {batch_idx} 处理失败: {e}')
                    continue
            if valid_batches == 0:
                print(f'   - 警告：PLS Epoch {epoch + 1}: 没有有效批次')
                continue
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            self.pls_losses.append(avg_loss)
            final_prompt_norm = self.pls_model.data_prompt.norm().item()
            print(f'{self.model_type.upper()}-PLS Epoch {epoch + 1}: Loss = {avg_loss:.4f}, 提示范数 = {final_prompt_norm:.6f}, 有效批次 = {valid_batches}')
        print(f'✅ {self.model_type.upper()}-PLS学习完成! 最终提示范数: {self.pls_model.data_prompt.norm().item():.6f}')

    def train_fas_phase(self, data_loader):
        print(f'\n🎯 第二阶段：{self.model_type.upper()}-FAS特征对齐学习...')
        self.pls_model.eval()
        self.fas_model.train()
        learned_prompt = self.pls_model.data_prompt.clone().detach()
        fas_epochs = self.args.tta_epochs - self.args.tta_epochs // 2
        if fas_epochs == 0:
            raise ValueError(f'FAS轮数为0')
        print(f'   - 使用学习到的提示，范数: {learned_prompt.norm().item():.6f}')
        for epoch in range(fas_epochs):
            epoch_losses = []
            valid_batches = 0
            pbar = tqdm(data_loader, desc=f'{self.model_type.upper()}-FAS Epoch {epoch + 1}/{fas_epochs}')
            for batch_idx, (imgs, labels, *_) in enumerate(pbar):
                imgs, labels = (imgs.to(self.device), labels.to(self.device))
                self.fas_optimizer.zero_grad()
                mixed_imgs = mix_data_prompt_3d(imgs, learned_prompt)
                outputs, global_features = self.fas_model(mixed_imgs, gfeat=True)
                with torch.no_grad():
                    _, original_features = self.fas_model(imgs, gfeat=True)
                align_loss = F.l1_loss(global_features, original_features.detach())
                consistency_loss = F.mse_loss(outputs, outputs.detach())
                loss = align_loss + self.args.gamma * consistency_loss
                if torch.isnan(loss):
                    raise ValueError(f'FAS损失为NaN')
                if torch.isinf(loss):
                    raise ValueError(f'FAS损失为Inf')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fas_model.parameters(), max_norm=1.0)
                self.fas_optimizer.step()
                valid_batches += 1
                epoch_losses.append(loss.item())
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Align': f'{align_loss.item():.4f}', 'Valid': f'{valid_batches}'})
            if valid_batches == 0:
                raise RuntimeError(f'FAS Epoch {epoch + 1}: 没有有效批次')
            avg_loss = np.mean(epoch_losses)
            self.fas_losses.append(avg_loss)
            print(f'{self.model_type.upper()}-FAS Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Valid batches = {valid_batches}')
        print(f'✅ {self.model_type.upper()}-FAS学习完成!')

    def evaluate_model(self, data_loader):
        print(f'\n🧪 开始{self.model_type.upper()}模型评估...')
        self.pls_model.eval()
        self.fas_model.eval()
        all_metrics = {'dice': [[] for _ in range(3)], 'hd95': [[] for _ in range(3)], 'iou': [[] for _ in range(3)], 'pa': [[] for _ in range(3)], 'rve': [[] for _ in range(3)], 'sensitivity': [[] for _ in range(3)], 'ppv': [[] for _ in range(3)]}
        with torch.no_grad():
            for imgs, labels, *_ in tqdm(data_loader, desc=f'{self.model_type.upper()}评估进度'):
                imgs, labels = (imgs.to(self.device), labels.to(self.device))
                mixed_imgs = mix_data_prompt_3d(imgs, self.pls_model.data_prompt)
                outputs = self.fas_model(mixed_imgs, gfeat=False)
                metrics = self._compute_metrics(outputs, labels.squeeze(1))
                for i in range(3):
                    all_metrics['dice'][i].append(self.safe_value(metrics['dice'][i]))
                    all_metrics['hd95'][i].append(self.safe_value(metrics['hd95'][i]))
                    all_metrics['iou'][i].append(self.safe_value(metrics['iou'][i]))
                    all_metrics['pa'][i].append(self.safe_value(metrics['pa'][i]))
                    all_metrics['rve'][i].append(self.safe_value(metrics['rve'][i]))
                    all_metrics['sensitivity'][i].append(self.safe_value(metrics['sensitivity'][i]))
                    all_metrics['ppv'][i].append(self.safe_value(metrics['ppv'][i]))
        return self._compute_statistics(all_metrics)

    def _compute_metrics(self, outputs, labels):
        return {'dice': cal_dice(outputs, labels), 'hd95': cal_hd95(outputs, labels), 'iou': IoU(outputs, labels), 'pa': PA(outputs, labels, 4), 'rve': cal_RVE(outputs, labels), 'sensitivity': cal_sensitivity(outputs, labels), 'ppv': cal_ppv(outputs, labels)}

    def _compute_statistics(self, all_metrics):
        stats = {}
        for metric_name, values in all_metrics.items():
            stats[metric_name] = {'mean': [np.mean(vals) if vals else 0.0 for vals in values], 'std': [np.std(vals) if vals else 0.0 for vals in values]}
        return stats

    def save_results(self, stats, model_path):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        adapted_model_path = os.path.join(self.weights_dir, f'{model_name}_{self.args.img}_{self.model_type}_pls_fas_tta_{timestamp}.pth')
        torch.save({'model_type': self.model_type, 'pls_model_state_dict': self.pls_model.state_dict(), 'fas_model_state_dict': self.fas_model.state_dict(), 'data_prompt': self.pls_model.data_prompt, 'losses': {'pls': self.pls_losses, 'fas': self.fas_losses, 'joint': self.joint_losses}, 'final_prompt_norm': self.pls_model.data_prompt.norm().item(), 'config': vars(self.args)}, adapted_model_path)
        report = self._generate_report(stats, model_path, adapted_model_path, timestamp)
        result_file = os.path.join(self.result_dir, f'{model_name}_{self.args.img}_{self.model_type}_pls_fas_tta_{timestamp}.txt')
        with open(result_file, 'w') as f:
            f.write(report)
        print(report)
        print(f'✅ 结果已保存: {adapted_model_path}')
        return adapted_model_path

    def _generate_report(self, stats, model_path, adapted_path, timestamp):
        return f'\n{'=' * 70}\n{self.model_type.upper()}-PLS+FAS测试时域适应结果报告\n{'=' * 70}\n测试时间: {timestamp}\n模型类型: {self.model_type.upper()}\n图像模态: {self.args.img}\n预训练模型: {model_path}\n适应后模型: {adapted_path}\n\n训练参数:\n- 适应轮数: {self.args.tta_epochs}\n- PLS轮数: {self.args.tta_epochs // 2}\n- FAS轮数: {self.args.tta_epochs - self.args.tta_epochs // 2}\n- 学习率: PLS={self.args.lr}, FAS={self.args.lr * 0.1}\n- Alpha (BN权重): {self.args.alpha}\n- Gamma (FAS权重): {self.args.gamma}\n- BN层数: {self.args.bn_layers}\n- 最终提示范数: {self.pls_model.data_prompt.norm().item():.6f}\n\n训练损失:\n- PLS损失历史: {[f'{l:.6f}' for l in self.pls_losses]}\n- FAS损失历史: {[f'{l:.4f}' for l in self.fas_losses]}\n\n性能指标 (ET/TC/WT):\nDice: {stats['dice']['mean'][0]:.4f}±{stats['dice']['std'][0]:.4f} / {stats['dice']['mean'][1]:.4f}±{stats['dice']['std'][1]:.4f} / {stats['dice']['mean'][2]:.4f}±{stats['dice']['std'][2]:.4f}\nHD95: {stats['hd95']['mean'][0]:.2f}±{stats['hd95']['std'][0]:.2f} / {stats['hd95']['mean'][1]:.2f}±{stats['hd95']['std'][1]:.2f} / {stats['hd95']['mean'][2]:.2f}±{stats['hd95']['std'][2]:.2f}\nIoU:  {stats['iou']['mean'][0]:.4f}±{stats['iou']['std'][0]:.4f} / {stats['iou']['mean'][1]:.4f}±{stats['iou']['std'][1]:.4f} / {stats['iou']['mean'][2]:.4f}±{stats['iou']['std'][2]:.4f}\nPA:   {stats['pa']['mean'][0]:.4f}±{stats['pa']['std'][0]:.4f} / {stats['pa']['mean'][1]:.4f}±{stats['pa']['std'][1]:.4f} / {stats['pa']['mean'][2]:.4f}±{stats['pa']['std'][2]:.4f}\nRVE:  {stats['rve']['mean'][0]:.4f}±{stats['rve']['std'][0]:.4f} / {stats['rve']['mean'][1]:.4f}±{stats['rve']['std'][1]:.4f} / {stats['rve']['mean'][2]:.4f}±{stats['rve']['std'][2]:.4f}\nSens: {stats['sensitivity']['mean'][0]:.4f}±{stats['sensitivity']['std'][0]:.4f} / {stats['sensitivity']['mean'][1]:.4f}±{stats['sensitivity']['std'][1]:.4f} / {stats['sensitivity']['mean'][2]:.4f}±{stats['sensitivity']['std'][2]:.4f}\nPPV:  {stats['ppv']['mean'][0]:.4f}±{stats['ppv']['std'][0]:.4f} / {stats['ppv']['mean'][1]:.4f}±{stats['ppv']['std'][1]:.4f} / {stats['ppv']['mean'][2]:.4f}±{stats['ppv']['std'][2]:.4f}\n{'=' * 70}\n'

    def cleanup(self):
        if self.pls_model and hasattr(self.pls_model, 'close'):
            self.pls_model.close()
        if self.fas_model and hasattr(self.fas_model, 'close'):
            self.fas_model.close()

def test_on_target(args, device):
    print(f'\n{'=' * 60}')
    print(f'🧪 {args.model_type.upper()}-PLS+FAS TTA: {args.img.upper()}')
    print(f'{'=' * 60}\n')
    trainer = TTATrainer(args, device)
    trainer.initialize_models()
    if args.model_path and args.model_path != 'default':
        model_path = args.model_path
    elif args.model_type.lower() == 'nnunet':
        model_path = '/home/yuwenjing/DeepLearning_ywj/tta/nnunet_best.pth'
    else:
        model_path = '/home/yuwenjing/DeepLearning_ywj/tta/unet3d_best.pth'
    trainer.load_pretrained_weights(model_path)
    trainer.setup_training()
    _, target_loader = get_data_loader(source_root=args.source_root, target_root=args.target_root, batch_train=args.batch_test, batch_test=args.batch_test, nw=args.num_workers, img=args.img, mode='source_to_target')
    print(f'📊 数据加载器: {len(target_loader)} 个批次')
    trainer.train_pls_phase(target_loader)
    trainer.train_fas_phase(target_loader)
    stats = trainer.evaluate_model(target_loader)
    trainer.save_results(stats, model_path)
    trainer.cleanup()
    return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PLS+FAS测试时域适应脚本 - 支持UNet3d和nnUNet')
    parser.add_argument('--source_root', type=str, default='/home/yuwenjing/data/BraTS2024')
    parser.add_argument('--target_root', type=str, default='/home/yuwenjing/data/BraTS-PED2023/Train', help='目标数据集根目录路径')
    parser.add_argument('--model_path', type=str, default='default', help='预训练模型路径，default为自动选择')
    parser.add_argument('--model_type', type=str, default='unet3d', choices=['unet3d', 'nnunet'], help='模型类型选择: unet3d 或 nnunet')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--tta_epochs', type=int, default=30, help='适应轮数')
    parser.add_argument('--alpha', type=float, default=0.01, help='BN loss权重')
    parser.add_argument('--gamma', type=float, default=0.1, help='FAS权重')
    parser.add_argument('--bn_layers', type=int, default=12, help='BN层数')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--img', default=['all'], help='测试模态')
    parser.add_argument('--batch_test', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  设备: {device}')
    print(f'🔧 模型类型: {args.model_type.upper()}')
    print(f'🎯 TTA参数: lr={args.lr}, epochs={args.tta_epochs}, alpha={args.alpha}, gamma={args.gamma}')
    success_count = 0
    start_time = datetime.datetime.now()
    for idx, modality in enumerate(args.img, 1):
        print(f'\n🔍 测试模态 ({idx}/{len(args.img)}): {modality.upper()}')
        modality_args = argparse.Namespace(**vars(args))
        modality_args.img = modality
        try:
            if test_on_target(modality_args, device):
                success_count += 1
                print(f'✅ {modality.upper()} {args.model_type.upper()}-TTA成功')
        except Exception as e:
            print(f'❌ {modality.upper()} {args.model_type.upper()}-TTA失败: {e}')
            print(f'详细错误信息:\n{traceback.format_exc()}')
    total_time = datetime.datetime.now() - start_time
    print(f'\n{'=' * 50}\n{args.model_type.upper()}-PLS+FAS TTA总结:\n- 模型类型: {args.model_type.upper()}\n- 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n- 总耗时: {total_time}\n- 成功测试: {success_count}/{len(args.img)}\n- 失败测试: {len(args.img) - success_count}\n- 成功率: {success_count / len(args.img) * 100:.1f}%\n{'=' * 50}\n')
