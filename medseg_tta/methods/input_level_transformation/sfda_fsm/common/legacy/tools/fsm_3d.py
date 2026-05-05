import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FSMGenerator3D(nn.Module):

    def __init__(self, input_channels=4, inversion_steps=200, fda_beta=0.03):
        super(FSMGenerator3D, self).__init__()
        self.input_channels = input_channels
        self.inversion_steps = inversion_steps
        self.fda_beta = fda_beta
        self.feature_scales = [1.0, 0.5, 0.25]

    def forward(self, target_img, source_model):
        print(f'  🎯 开始FSM生成流程...')
        device = target_img.device
        source_like_rough = self.domain_inversion(target_img, source_model)
        source_like_rough = source_like_rough.to(device)
        source_like_refined = self.fda_3d(source_like_rough, target_img)
        source_stats = {'mean': source_like_refined.mean(dim=(2, 3, 4), keepdim=True), 'std': source_like_refined.std(dim=(2, 3, 4), keepdim=True)}
        print(f'  ✅ FSM生成完成')
        return (source_like_refined, source_stats)

    def domain_inversion(self, target_img, source_model):
        B, C, D, H, W = target_img.shape
        device = target_img.device
        print(f'    🔄 域反转开始，输入尺寸: {target_img.shape}, 设备: {device}')
        if torch.isnan(target_img).any():
            raise RuntimeError('目标图像包含NaN值')
        target_mean = target_img.mean(dim=(2, 3, 4), keepdim=True)
        target_std = target_img.std(dim=(2, 3, 4), keepdim=True) + 1e-08
        z = self.initialize_noise(target_img, target_mean, target_std)
        z = z.to(device)
        target_mean = target_mean.to(device)
        target_std = target_std.to(device)
        optimizer = torch.optim.Adam([z], lr=0.01, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.inversion_steps)
        print(f'    🔄 开始多尺度域反转，共{self.inversion_steps}步...')
        best_loss = float('inf')
        best_z = z.clone().detach()
        for step in range(self.inversion_steps):
            optimizer.zero_grad()
            candidate = z * target_std + target_mean
            total_feature_loss = 0
            for scale in self.feature_scales:
                if scale < 1.0:
                    scale_size = tuple((int(s * scale) for s in candidate.shape[2:]))
                    candidate_scaled = F.interpolate(candidate, size=scale_size, mode='trilinear', align_corners=False)
                    target_scaled = F.interpolate(target_img, size=scale_size, mode='trilinear', align_corners=False)
                else:
                    candidate_scaled = candidate
                    target_scaled = target_img
                candidate_scaled = candidate_scaled.to(device)
                target_scaled = target_scaled.to(device)
                source_features = self.extract_features(source_model, candidate_scaled)
                with torch.no_grad():
                    target_features = self.extract_features(source_model, target_scaled)
                source_features = source_features.to(device)
                target_features = target_features.to(device)
                if source_features.shape != target_features.shape:
                    min_size = tuple((min(s, t) for s, t in zip(source_features.shape[2:], target_features.shape[2:])))
                    adaptive_size = tuple((max(1, s // 2) for s in min_size))
                    source_features = F.adaptive_avg_pool3d(source_features, adaptive_size)
                    target_features = F.adaptive_avg_pool3d(target_features, adaptive_size)
                feature_loss = F.mse_loss(source_features, target_features)
                total_feature_loss += feature_loss * scale
            reg_loss = 0.01 * torch.norm(z, p=2) + 0.001 * self.total_variation_loss(candidate)
            total_loss = total_feature_loss + reg_loss
            if torch.isnan(total_loss):
                print(f'    ⚠️ 步骤{step}: 损失为NaN，重新初始化')
                z.data = self.initialize_noise(target_img, target_mean, target_std).data.to(device)
                continue
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                z.clamp_(-3, 3)
            current_loss = total_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                with torch.no_grad():
                    best_z = z.clone().to(device)
            if step % 10 == 0 or step == self.inversion_steps - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f'      步骤{step}: 损失={current_loss:.6f}, 学习率={current_lr:.6f}')
        with torch.no_grad():
            source_like = best_z * target_std + target_mean
        print(f'    ✅ 域反转完成，最佳损失: {best_loss:.6f}')
        return source_like.detach().to(device)

    def initialize_noise(self, target_img, target_mean, target_std):
        device = target_img.device
        noise_ratio = 0.3
        z_random = torch.randn_like(target_img).to(device)
        z_target = ((target_img - target_mean) / target_std).to(device)
        z = noise_ratio * z_random + (1 - noise_ratio) * z_target
        z.requires_grad_(True)
        return z.to(device)

    def total_variation_loss(self, img):
        tv_d = torch.abs(img[:, :, 1:, :, :] - img[:, :, :-1, :, :]).mean()
        tv_h = torch.abs(img[:, :, :, 1:, :] - img[:, :, :, :-1, :]).mean()
        tv_w = torch.abs(img[:, :, :, :, 1:] - img[:, :, :, :, :-1]).mean()
        return tv_d + tv_h + tv_w

    def extract_features(self, source_model, x):
        with torch.no_grad():
            if hasattr(source_model, 'enc'):
                encoder_blocks, latent_features = source_model.enc(x)
                return latent_features
            elif hasattr(source_model, 'encoder'):
                features = x
                for layer in source_model.encoder:
                    features = layer(features)
                return features
            elif hasattr(source_model, 'features'):
                return source_model.features(x)
            else:
                output = source_model(x)
                if isinstance(output, (list, tuple)):
                    return output[0] if len(output) > 0 else x
                else:
                    return output

    def fda_3d(self, source_like, target_img):
        print(f'    🌊 开始改进FDA，输入尺寸: source_like{source_like.shape}')
        device = source_like.device
        target_img = target_img.to(device)
        print(f'    📱 设备统一: source_like在{source_like.device}, target_img在{target_img.device}')
        fft_source = torch.fft.fftn(source_like, dim=(-3, -2, -1))
        fft_target = torch.fft.fftn(target_img, dim=(-3, -2, -1))
        amp_source, pha_source = (torch.abs(fft_source), torch.angle(fft_source))
        amp_target = torch.abs(fft_target)
        amp_source = amp_source.to(device)
        pha_source = pha_source.to(device)
        amp_target = amp_target.to(device)
        _, _, D, H, W = source_like.shape
        center_d, center_h, center_w = (D // 2, H // 2, W // 2)
        adaptive_beta = self.compute_adaptive_beta(amp_source, amp_target)
        d_radius = max(1, int(D * adaptive_beta))
        h_radius = max(1, int(H * adaptive_beta))
        w_radius = max(1, int(W * adaptive_beta))
        print(f'    📊 自适应低频区域: beta={adaptive_beta:.3f}, 半径=({d_radius}, {h_radius}, {w_radius})')
        mask = self.create_smooth_mask(source_like, (center_d, center_h, center_w), (d_radius, h_radius, w_radius))
        mask = mask.to(device)
        print(f'    📱 掩码设备检查: mask在{mask.device}')
        amp_mixed = amp_source * (1 - mask) + amp_target * mask
        fft_mixed = amp_mixed * torch.exp(1j * pha_source)
        source_adapted = torch.fft.ifftn(fft_mixed, dim=(-3, -2, -1)).real
        source_adapted = source_adapted.to(device)
        print(f'    ✅ 改进FDA完成')
        return source_adapted

    def compute_adaptive_beta(self, amp_source, amp_target):
        device = amp_source.device
        amp_target = amp_target.to(device)
        similarity = F.cosine_similarity(amp_source.flatten(2).mean(dim=2), amp_target.flatten(2).mean(dim=2), dim=1).mean()
        adaptive_beta = self.fda_beta * (1.5 - similarity.clamp(0, 1))
        return adaptive_beta.clamp(0.01, 0.1).item()

    def create_smooth_mask(self, input_tensor, center, radius):
        B, C, D, H, W = input_tensor.shape
        device = input_tensor.device
        center_d, center_h, center_w = center
        d_radius, h_radius, w_radius = radius
        print(f'    🎭 创建掩码: 设备={device}, 形状=({B}, {C}, {D}, {H}, {W})')
        d_coords = torch.arange(D, device=device).float() - center_d
        h_coords = torch.arange(H, device=device).float() - center_h
        w_coords = torch.arange(W, device=device).float() - center_w
        d_grid, h_grid, w_grid = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        d_grid = d_grid.to(device)
        h_grid = h_grid.to(device)
        w_grid = w_grid.to(device)
        distance = torch.sqrt((d_grid / d_radius) ** 2 + (h_grid / h_radius) ** 2 + (w_grid / w_radius) ** 2)
        mask = torch.sigmoid(5 * (1 - distance))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1)
        mask = mask.to(device)
        print(f'    🎭 掩码创建完成: 设备={mask.device}')
        return mask

class ContrastiveDomainDistillation3D(nn.Module):

    def __init__(self, temperature=0.1):
        super(ContrastiveDomainDistillation3D, self).__init__()
        self.temperature = temperature

    def forward(self, source_feat, target_feat, source_feat_aug, target_feat_aug):
        source_feat_norm = F.normalize(source_feat.flatten(2).mean(dim=2), p=2, dim=1)
        target_feat_norm = F.normalize(target_feat.flatten(2).mean(dim=2), p=2, dim=1)
        source_aug_norm = F.normalize(source_feat_aug.flatten(2).mean(dim=2), p=2, dim=1)
        target_aug_norm = F.normalize(target_feat_aug.flatten(2).mean(dim=2), p=2, dim=1)
        distill_loss = 1 - F.cosine_similarity(source_feat_norm, target_feat_norm, dim=1).mean()
        pos_sim_source = F.cosine_similarity(source_feat_norm, source_aug_norm, dim=1) / self.temperature
        pos_sim_target = F.cosine_similarity(target_feat_norm, target_aug_norm, dim=1) / self.temperature
        neg_sim = F.cosine_similarity(source_feat_norm, target_feat_norm, dim=1) / self.temperature
        contrast_loss = -torch.log(torch.exp(pos_sim_source) / (torch.exp(pos_sim_source) + torch.exp(neg_sim))).mean() - torch.log(torch.exp(pos_sim_target) / (torch.exp(pos_sim_target) + torch.exp(neg_sim))).mean()
        return {'distill_loss': distill_loss, 'contrast_loss': contrast_loss}

class CompactAwareDomainConsistency3D(nn.Module):

    def __init__(self, num_classes=4, confidence_threshold=0.7, compactness_threshold=0.15):
        super(CompactAwareDomainConsistency3D, self).__init__()
        self.num_classes = num_classes
        if confidence_threshold >= 0.8:
            print(f'      ⚠️ 检测到高置信度阈值 {confidence_threshold:.2f}，针对3D医学影像进行调整')
            self.confidence_threshold = 0.65
            print(f'      🔧 调整为3D医学影像优化的阈值: {self.confidence_threshold:.2f}')
        else:
            self.confidence_threshold = max(0.45, confidence_threshold)
            print(f'      ✅ 使用设定的置信度阈值: {self.confidence_threshold:.2f}')
        if compactness_threshold >= 0.2:
            print(f'      ⚠️ 检测到高紧凑性阈值 {compactness_threshold:.2f}，为3D结构调整')
            self.compactness_threshold = 0.08
            print(f'      🔧 调整为3D结构优化的阈值: {self.compactness_threshold:.2f}')
        else:
            self.compactness_threshold = max(0.05, compactness_threshold)
            print(f'      ✅ 使用设定的紧凑性阈值: {self.compactness_threshold:.2f}')
        self.high_conf_threshold = self.confidence_threshold
        self.medium_conf_threshold = self.confidence_threshold * 0.75
        self.low_conf_threshold = self.confidence_threshold * 0.55
        self.emergency_threshold = self.confidence_threshold * 0.4
        print(f'      📊 多层次置信度策略:')
        print(f'        - 高置信度: {self.high_conf_threshold:.3f} (直接接受)')
        print(f'        - 中等置信度: {self.medium_conf_threshold:.3f} (需紧凑性验证)')
        print(f'        - 低置信度: {self.low_conf_threshold:.3f} (仅大型连通区域)')
        print(f'        - 应急阈值: {self.emergency_threshold:.3f} (避免无有效像素)')

    def forward(self, predictions):
        print(f'      🎯 CADC前向传播开始，输入形状: {predictions.shape}')
        if predictions.dim() != 5:
            raise ValueError(f'期望5D输入 [B, C, D, H, W]，但得到 {predictions.dim()}D: {predictions.shape}')
        if torch.isnan(predictions).any():
            raise RuntimeError('CADC输入包含NaN值')
        if torch.isinf(predictions).any():
            raise RuntimeError('CADC输入包含Inf值')
        pseudo_labels, weights = self.generate_pseudo_labels(predictions)
        if torch.isnan(pseudo_labels).any():
            raise RuntimeError('生成的伪标签包含NaN值')
        if torch.isnan(weights).any():
            raise RuntimeError('生成的权重包含NaN值')
        print(f'      ✅ CADC前向传播完成，输出: pseudo_labels{pseudo_labels.shape}, weights{weights.shape}')
        return (pseudo_labels, weights)

    def compute_compactness_3d(self, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=mask.device)
        volume = mask.sum().float()
        mask_float = mask.float()
        if mask_float.dim() == 3:
            grad_d = torch.abs(mask_float[1:] - mask_float[:-1]).sum()
            grad_h = torch.abs(mask_float[:, 1:] - mask_float[:, :-1]).sum() if mask_float.shape[1] > 1 else 0
            grad_w = torch.abs(mask_float[:, :, 1:] - mask_float[:, :, :-1]).sum() if mask_float.shape[2] > 1 else 0
            surface_area = grad_d + grad_h + grad_w + 1e-08
        else:
            surface_area = mask.sum().float() + 1e-08
        ideal_surface_area = (36 * np.pi) ** (1 / 3) * volume ** (2 / 3) + 1e-08
        sphericity = ideal_surface_area / surface_area
        return sphericity.clamp(0, 1)

    def generate_pseudo_labels(self, predictions):
        print(f'      🏷️  开始3D医学影像伪标签生成，输入形状: {predictions.shape}')
        probs = F.softmax(predictions, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        high_confidence_mask = max_probs > self.high_conf_threshold
        medium_confidence_mask = max_probs > self.medium_conf_threshold
        low_confidence_mask = max_probs > self.low_conf_threshold
        emergency_mask = max_probs > self.emergency_threshold
        weights = torch.zeros_like(pseudo_labels, dtype=torch.float)
        total_valid_pixels = 0
        print(f'      📊 置信度分布统计:')
        print(f'        - 高置信度({self.high_conf_threshold:.3f}): {high_confidence_mask.sum().item()} 像素')
        print(f'        - 中等置信度({self.medium_conf_threshold:.3f}): {medium_confidence_mask.sum().item()} 像素')
        print(f'        - 低置信度({self.low_conf_threshold:.3f}): {low_confidence_mask.sum().item()} 像素')
        print(f'        - 应急水平({self.emergency_threshold:.3f}): {emergency_mask.sum().item()} 像素')
        for class_id in range(self.num_classes):
            class_mask = pseudo_labels == class_id
            class_pixels = class_mask.sum().item()
            if class_pixels == 0:
                continue
            print(f'        📋 类别{class_id}分析: 总像素={class_pixels}')
            high_conf_class = class_mask & high_confidence_mask
            if high_conf_class.sum() > 0:
                weights[high_conf_class] = 1.0
                total_valid_pixels += high_conf_class.sum().item()
                print(f'          ✅ 高置信度: {high_conf_class.sum().item()} 像素 (权重=1.0)')
            medium_conf_class = class_mask & medium_confidence_mask & ~high_confidence_mask
            if medium_conf_class.sum() > 10:
                for b in range(class_mask.shape[0]):
                    batch_mask = medium_conf_class[b]
                    if batch_mask.sum() > 8:
                        compactness = self.compute_compactness_3d(batch_mask)
                        if class_id == 0:
                            threshold = self.compactness_threshold * 0.6
                            weight = 0.8
                        elif class_id in [1, 2]:
                            threshold = self.compactness_threshold * 1.2
                            weight = 0.7
                        else:
                            threshold = self.compactness_threshold
                            weight = 0.75
                        if compactness > threshold:
                            weights[b][batch_mask] = weight
                            total_valid_pixels += batch_mask.sum().item()
                            print(f'          ✅ 中等置信度(类别{class_id}): {batch_mask.sum().item()} 像素 (紧凑性={compactness:.3f}, 权重={weight})')
            low_conf_class = class_mask & low_confidence_mask & ~medium_confidence_mask
            if low_conf_class.sum() > 50:
                for b in range(class_mask.shape[0]):
                    batch_mask = low_conf_class[b]
                    min_volume_threshold = 100 if class_id == 0 else 50
                    if batch_mask.sum() > min_volume_threshold:
                        if class_id == 0:
                            weights[b][batch_mask] = 0.4
                            total_valid_pixels += batch_mask.sum().item()
                            print(f'          ✅ 低置信度背景: {batch_mask.sum().item()} 像素 (权重=0.4)')
                        elif class_id == 3:
                            compactness = self.compute_compactness_3d(batch_mask)
                            if compactness > self.compactness_threshold * 0.4:
                                weights[b][batch_mask] = 0.5
                                total_valid_pixels += batch_mask.sum().item()
                                print(f'          ✅ 低置信度肿瘤: {batch_mask.sum().item()} 像素 (权重=0.5)')
        effective_ratio = total_valid_pixels / pseudo_labels.numel()
        min_effective_ratio = 0.05
        if effective_ratio < min_effective_ratio:
            print(f'      🚨 有效像素过少({effective_ratio:.1%})，启用应急策略...')
            for class_id in range(self.num_classes):
                class_mask = (pseudo_labels == class_id) & emergency_mask
                if class_mask.sum() > 0:
                    zero_weight_mask = class_mask & (weights == 0)
                    if zero_weight_mask.sum() > 0:
                        emergency_weight = 0.2 if class_id == 0 else 0.15
                        weights[zero_weight_mask] = emergency_weight
                        total_valid_pixels += zero_weight_mask.sum().item()
                        print(f'          🆘 应急策略类别{class_id}: {zero_weight_mask.sum().item()} 像素 (权重={emergency_weight})')
        final_ratio = total_valid_pixels / pseudo_labels.numel()
        print(f'      ✅ 3D伪标签生成完成: {total_valid_pixels} 个有效像素 ({final_ratio:.1%})')
        if final_ratio > 0.15:
            quality = '优秀'
        elif final_ratio > 0.08:
            quality = '良好'
        elif final_ratio > 0.03:
            quality = '可接受'
        else:
            quality = '需要调整'
        print(f'      📈 伪标签质量评估: {quality} (有效像素比例: {final_ratio:.1%})')
        return (pseudo_labels, weights)

class DiceLoss3D(nn.Module):

    def __init__(self, smooth=1e-05):
        super(DiceLoss3D, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets, weights=None):
        print(f'        🎲 Dice损失计算: pred={predictions.shape}, target={targets.shape}')
        if weights is not None:
            print(f'        🎲 权重统计: shape={weights.shape}, 非零={(weights > 0).sum().item()}, max={weights.max().item():.4f}')
        if torch.isnan(predictions).any():
            raise RuntimeError('Dice损失输入预测包含NaN值')
        if torch.isnan(targets).any():
            raise RuntimeError('Dice损失输入目标包含NaN值')
        if weights is not None and torch.isnan(weights).any():
            raise RuntimeError('Dice损失输入权重包含NaN值')
        predictions = F.softmax(predictions, dim=1)
        if torch.isnan(predictions).any():
            raise RuntimeError('Softmax后预测包含NaN值')
        if targets.dim() == 5:
            targets = targets.squeeze(1)
        if targets.dim() == 4:
            targets = torch.clamp(targets.long(), 0, predictions.size(1) - 1)
            targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1))
            targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        else:
            targets_one_hot = targets
        if torch.isnan(targets_one_hot).any():
            raise RuntimeError('One-hot编码后目标包含NaN值')
        dice_loss = 0
        num_classes = predictions.size(1)
        valid_classes = 0
        class_losses = []
        for class_id in range(num_classes):
            pred_class = predictions[:, class_id]
            target_class = targets_one_hot[:, class_id]
            if weights is not None:
                pred_class = pred_class * weights
                target_class = target_class * weights
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            if torch.isnan(intersection):
                raise RuntimeError(f'类别{class_id}的交集计算包含NaN值')
            if torch.isnan(union):
                raise RuntimeError(f'类别{class_id}的并集计算包含NaN值')
            if union > self.smooth:
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                if torch.isnan(dice):
                    raise RuntimeError(f'类别{class_id}的Dice系数为NaN')
                class_dice_loss = 1 - dice
                class_losses.append(class_dice_loss)
                valid_classes += 1
                print(f'          类别{class_id}: intersection={intersection.item():.4f}, union={union.item():.4f}, dice={dice.item():.4f}, loss={class_dice_loss.item():.6f}')
            else:
                print(f'          类别{class_id}: 跳过 (union={union.item():.4f} <= {self.smooth})')
        if valid_classes > 0:
            final_loss = sum(class_losses) / valid_classes
            print(f'        🎲 最终Dice损失: {final_loss.item():.6f} (来自 {valid_classes} 个有效类别)')
        else:
            final_loss = torch.tensor(0.0, device=predictions.device)
            print(f'        🎲 无有效类别，Dice损失=0')
        if torch.isnan(final_loss):
            raise RuntimeError('最终Dice损失为NaN')
        return final_loss
