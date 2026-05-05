import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

def get_2d_rf_field(num_batch: int, size_2d: Tuple[int, int], interpolation_factor: int=5, num_fields: int=2, device: str='cpu') -> torch.Tensor:
    rf_field = F.interpolate(F.avg_pool2d(F.avg_pool2d(torch.randn(num_batch, num_fields, size_2d[0] // interpolation_factor, size_2d[1] // interpolation_factor, device=device), interpolation_factor, stride=1, padding=interpolation_factor // 2), interpolation_factor, stride=1, padding=interpolation_factor // 2), size=size_2d, mode='bilinear')
    rf_field -= rf_field.mean((-2, -1), keepdim=True)
    rf_field /= 0.001 + rf_field.view(num_batch * num_fields, -1).std(1).view(num_batch, num_fields, 1, 1)
    return rf_field

def calc_2d_consistent_diffeomorphic_field(disp_field: torch.Tensor, inverse_disp_field: torch.Tensor, time_steps: int=1, ensure_inverse_consistency: bool=True, iter_steps_override: int=None) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = disp_field.size()
    dimension_correction = torch.tensor([H, W], device=disp_field.device).view(1, 2, 1, 1)
    dt = 1.0 / time_steps
    with torch.no_grad():
        identity = F.affine_grid(torch.eye(2, 3).unsqueeze(0), (1, 1, H, W), align_corners=True).permute(0, 3, 1, 2).to(disp_field)
        if ensure_inverse_consistency:
            out_disp_field = (disp_field / dimension_correction / 2 ** time_steps * dt).clone()
            out_inverse_disp_field = (inverse_disp_field / dimension_correction / 2 ** time_steps * dt).clone()
            for _ in range(time_steps if not iter_steps_override else iter_steps_override):
                ds = out_disp_field.clone()
                inverse_ds = out_inverse_disp_field.clone()
                out_disp_field = +0.5 * ds - 0.5 * F.grid_sample(inverse_ds, (identity + ds).permute(0, 2, 3, 1), padding_mode='border', align_corners=True)
                out_inverse_disp_field = +0.5 * inverse_ds - 0.5 * F.grid_sample(ds, (identity + inverse_ds).permute(0, 2, 3, 1), padding_mode='border', align_corners=True)
            out_disp_field = out_disp_field * 2 ** time_steps * dimension_correction
            out_inverse_disp_field = out_inverse_disp_field * 2 ** time_steps * dimension_correction
        else:
            ds_dt = disp_field / dimension_correction / 2 ** time_steps
            inverse_ds_dt = inverse_disp_field / dimension_correction / 2 ** time_steps
            ds = ds_dt * dt
            inverse_ds = inverse_ds_dt * dt
            for _ in range(time_steps if not iter_steps_override else iter_steps_override):
                ds = ds + F.grid_sample(ds, (identity + ds).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
                inverse_ds = inverse_ds + F.grid_sample(inverse_ds, (identity + inverse_ds).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
            out_disp_field = ds * dimension_correction
            out_inverse_disp_field = inverse_ds * dimension_correction
    return (out_disp_field, out_inverse_disp_field)

def get_2d_disp_field(batch_num: int, size_2d: Tuple[int, int], factor: float=0.1, interpolation_factor: int=5, device: str='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    field = get_2d_rf_field(num_batch=batch_num, size_2d=size_2d, device=device)
    STEPS = 5
    disp_field, inverse_disp_field = calc_2d_consistent_diffeomorphic_field(field * factor, torch.zeros_like(field), STEPS, ensure_inverse_consistency=True)
    return (disp_field.permute(0, 2, 3, 1), inverse_disp_field.permute(0, 2, 3, 1))

def get_2d_rand_affine(batch_size: int, strength: float=0.05, flip: bool=False, device: str='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    affine = torch.randn(batch_size, 2, 3, device=device) * strength + torch.eye(2, 3, device=device).unsqueeze(0)
    if flip:
        flip_affine = torch.diag(torch.cat([2 * (torch.rand(2, device=device) > 0.5).float() - 1, torch.tensor([1.0], device=device)]))
        affine = affine @ flip_affine.to(device)
    A = affine[:, :, :2]
    t = affine[:, :, 2:]
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    det = det.unsqueeze(1).unsqueeze(2)
    A_inv = torch.zeros_like(A)
    A_inv[:, 0, 0] = A[:, 1, 1]
    A_inv[:, 0, 1] = -A[:, 0, 1]
    A_inv[:, 1, 0] = -A[:, 1, 0]
    A_inv[:, 1, 1] = A[:, 0, 0]
    A_inv = A_inv / (det + 1e-08)
    t_inv = -torch.bmm(A_inv, t)
    affine_inv = torch.cat([A_inv, t_inv], dim=2)
    return (affine, affine_inv)

def apply_2d_spatial_transform(x: torch.Tensor, affine_matrix: torch.Tensor, disp_field: torch.Tensor=None, padding_mode: str='border') -> torch.Tensor:
    batch_size, _, H, W = x.size()
    identity_grid = F.affine_grid(torch.eye(2, 3, device=x.device).repeat(batch_size, 1, 1), [batch_size, 1, H, W], align_corners=False)
    affine_grid = F.affine_grid(affine_matrix, [batch_size, 1, H, W], align_corners=False)
    if disp_field is not None:
        composite_grid = identity_grid + (affine_grid - identity_grid) + disp_field
    else:
        composite_grid = affine_grid
    transformed = F.grid_sample(x, composite_grid, padding_mode=padding_mode, align_corners=False)
    return transformed

def create_2d_consistency_augmentation(x: torch.Tensor, strength: float=0.05, device: str='cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, _, H, W = x.size()
    affine_a, affine_a_inv = get_2d_rand_affine(batch_size, strength, device=device)
    disp_a, disp_a_inv = get_2d_disp_field(batch_size, (H, W), factor=0.1, device=device)
    x_a = apply_2d_spatial_transform(x, affine_a, disp_a)
    affine_b, affine_b_inv = get_2d_rand_affine(batch_size, strength, device=device)
    disp_b, disp_b_inv = get_2d_disp_field(batch_size, (H, W), factor=0.1, device=device)
    x_b = apply_2d_spatial_transform(x, affine_b, disp_b)
    transform_a_inv = (affine_a_inv, disp_a_inv)
    transform_b_inv = (affine_b_inv, disp_b_inv)
    return (x_a, x_b, transform_a_inv, transform_b_inv)

def apply_2d_inverse_transform(x: torch.Tensor, transform_inv: Tuple[torch.Tensor, torch.Tensor], device: str='cpu') -> torch.Tensor:
    affine_inv, disp_inv = transform_inv
    return apply_2d_spatial_transform(x, affine_inv, disp_inv)
