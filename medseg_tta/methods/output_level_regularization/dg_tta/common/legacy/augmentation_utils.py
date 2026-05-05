import torch
import torch.nn.functional as F
from mind import MIND3D
from gin import gin_aug

def get_rf_field(num_batch, size_3d, interpolation_factor=5, num_fields=3, device='cpu'):
    rf_field = F.interpolate(F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(torch.randn(num_batch, num_fields, size_3d[0] // interpolation_factor, size_3d[1] // interpolation_factor, size_3d[2] // interpolation_factor, device=device), interpolation_factor, stride=1, padding=interpolation_factor // 2), interpolation_factor, stride=1, padding=interpolation_factor // 2), interpolation_factor, stride=1, padding=interpolation_factor // 2), size=size_3d, mode='trilinear')
    rf_field -= rf_field.mean((-3, -2, -1), keepdim=True)
    rf_field /= 0.001 + rf_field.view(num_batch * num_fields, -1).std(1).view(num_batch, num_fields, 1, 1, 1)
    return rf_field

def calc_consistent_diffeomorphic_field(disp_field, inverse_disp_field, time_steps=1, ensure_inverse_consistency=True, iter_steps_override=None):
    B, C, D, H, W = disp_field.size()
    dimension_correction = torch.tensor([D, H, W], device=disp_field.device).view(1, 3, 1, 1, 1)
    dt = 1.0 / time_steps
    with torch.no_grad():
        identity = F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1, D, H, W), align_corners=True).permute(0, 4, 1, 2, 3).to(disp_field)
        if ensure_inverse_consistency:
            out_disp_field = (disp_field / dimension_correction / 2 ** time_steps * dt).clone()
            out_inverse_disp_field = (inverse_disp_field / dimension_correction / 2 ** time_steps * dt).clone()
            for _ in range(time_steps if not iter_steps_override else iter_steps_override):
                ds = out_disp_field.clone()
                inverse_ds = out_inverse_disp_field.clone()
                out_disp_field = +0.5 * ds - 0.5 * F.grid_sample(inverse_ds, (identity + ds).permute(0, 2, 3, 4, 1), padding_mode='border', align_corners=True)
                out_inverse_disp_field = +0.5 * inverse_ds - 0.5 * F.grid_sample(ds, (identity + inverse_ds).permute(0, 2, 3, 4, 1), padding_mode='border', align_corners=True)
            out_disp_field = out_disp_field * 2 ** time_steps * dimension_correction
            out_inverse_disp_field = out_inverse_disp_field * 2 ** time_steps * dimension_correction
        else:
            ds_dt = disp_field / dimension_correction / 2 ** time_steps
            inverse_ds_dt = inverse_disp_field / dimension_correction / 2 ** time_steps
            ds = ds_dt * dt
            inverse_ds = inverse_ds_dt * dt
            for _ in range(time_steps if not iter_steps_override else iter_steps_override):
                ds = ds + F.grid_sample(ds, (identity + ds).permute(0, 2, 3, 4, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
                inverse_ds = inverse_ds + F.grid_sample(inverse_ds, (identity + inverse_ds).permute(0, 2, 3, 4, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
            out_disp_field = ds * dimension_correction
            out_inverse_disp_field = inverse_ds * dimension_correction
    return (out_disp_field, out_inverse_disp_field)

def get_disp_field(batch_num, size_3d, factor=0.1, interpolation_factor=5, device='cpu'):
    field = get_rf_field(num_batch=batch_num, size_3d=size_3d, device=device)
    STEPS = 5
    disp_field, inverse_disp_field = calc_consistent_diffeomorphic_field(field * factor, torch.zeros_like(field), STEPS, ensure_inverse_consistency=True)
    return (disp_field.permute(0, 2, 3, 4, 1), inverse_disp_field.permute(0, 2, 3, 4, 1))

def get_rand_affine(batch_size, strength=0.05, flip=False, device='cpu'):
    affine = torch.cat((torch.randn(batch_size, 3, 4, device=device) * strength + torch.eye(3, 4, device=device).unsqueeze(0), torch.tensor([0, 0, 0, 1], device=device).view(1, 1, 4).repeat(batch_size, 1, 1)), 1)
    if flip:
        flip_affine = torch.diag(torch.cat([2 * (torch.rand(3, device=device) > 0.5).float() - 1, torch.tensor([1.0], device=device)]))
        affine = affine @ flip_affine.to(device)
    return (affine[:, :3], affine.inverse()[:, :3])

def gin_mind_aug(input):
    return MIND3D()(gin_aug(input))
