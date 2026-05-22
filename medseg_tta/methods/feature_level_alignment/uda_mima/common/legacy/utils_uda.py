import torch
import numpy as np

def extract_ampl_phase(fft_im):
    """
    从复数张量中提取振幅和相位
    支持2D和3D傅里叶变换结果
    """
    # 检查是否为3D数据
    is_3d = (len(fft_im.shape) == 5)  # b,c,d,h,w
    
    if is_3d:
        # 3D FFT结果处理
        fft_amp = torch.abs(fft_im)
        fft_pha = torch.angle(fft_im)
    else:
        # 2D FFT结果处理（原始代码）
        fft_amp = torch.abs(fft_im)
        fft_pha = torch.angle(fft_im)
        
    return fft_amp, fft_pha

def low_freq_mutate(amp_src, amp_trg, L=0.1):
    """
    在频域中替换低频振幅部分
    支持2D和3D数据
    """
    # 检查是否为3D数据
    is_3d = (len(amp_src.shape) == 5)  # b,c,d,h,w
    
    if is_3d:
        # 3D数据处理
        _, _, d, h, w = amp_src.size()
        w *= 2  # 考虑rFFT的对称性
        
        # 计算低频区域大小
        b_d = (np.floor(0.5 * d * L)).astype(int)  # 深度方向低频区域
        b_h = (np.floor(0.5 * h * L)).astype(int)  # 高度方向低频区域
        b_w = (np.floor(0.5 * w * L)).astype(int)  # 宽度方向低频区域
        
        if b_d > 0 and b_h > 0 and b_w > 0:
            # 替换3D低频区域
            # 中心区域
            amp_src[:, :, 0:b_d, 0:b_h, 0:b_w] = amp_trg[:, :, 0:b_d, 0:b_h, 0:b_w]
            
            # 其他八个角落区域（根据3D对称性调整）
            # 示例：仅替换部分关键区域，可根据需要扩展
            amp_src[:, :, d-b_d+1:d, 0:b_h, 0:b_w] = amp_trg[:, :, d-b_d+1:d, 0:b_h, 0:b_w]
            amp_src[:, :, 0:b_d, h-b_h+1:h, 0:b_w] = amp_trg[:, :, 0:b_d, h-b_h+1:h, 0:b_w]
            
    else:
        # 2D数据处理（原始代码）
        _, _, h, w = amp_src.size()
        w *= 2  # 考虑rFFT的对称性
        b = (np.floor(0.5 * np.amin((h, w)) * L)).astype(int)
        
        if b > 0:
            # 替换2D低频区域
            amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]  # 左上角
            amp_src[:, :, h-b+1:h, 0:b] = amp_trg[:, :, h-b+1:h, 0:b]  # 左下角
            
    return amp_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    """
    将源图像的低频振幅替换为目标图像的低频振幅
    支持2D和3D图像数据
    """
    # 检查是否为3D数据
    is_3d = (len(src_img.shape) == 5)  # b,c,d,h,w
    
    if is_3d:
        # 3D傅里叶变换
        fft_src = torch.fft.rfftn(src_img.clone(), dim=(-3, -2, -1))  # 3D rFFT
        fft_trg = torch.fft.rfftn(trg_img.clone(), dim=(-3, -2, -1))  # 3D rFFT
    else:
        # 2D傅里叶变换（原始代码）
        fft_src = torch.fft.rfft2(src_img.clone(), dim=(-2, -1))
        fft_trg = torch.fft.rfft2(trg_img.clone(), dim=(-2, -1))
    
    # 提取振幅和相位
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())
    
    # 替换低频振幅
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)
    
    # 重新组合复数
    real = torch.cos(pha_src.clone()) * amp_src_.clone()
    imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    fft_src_ = torch.complex(real=real, imag=imag)
    
    # 逆傅里叶变换
    if is_3d:
        # 3D逆傅里叶变换
        _, _, d, h, w = src_img.size()
        src_in_trg = torch.fft.irfftn(fft_src_, dim=(-3, -2, -1), s=[d, h, w])
    else:
        # 2D逆傅里叶变换（原始代码）
        _, _, h, w = src_img.size()
        src_in_trg = torch.fft.irfft2(fft_src_, dim=(-2, -1), s=[h, w])
    
    return src_in_trg

# 以下是numpy版本的3D支持代码
def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    """
    numpy版本的3D低频振幅替换
    """
    is_3d = (len(amp_src.shape) == 4)  # c,d,h,w
    
    if is_3d:
        a_src = np.fft.fftshift(amp_src, axes=(-3, -2, -1))  # 3D shift
        a_trg = np.fft.fftshift(amp_trg, axes=(-3, -2, -1))  # 3D shift
        
        _, d, h, w = a_src.shape
        b_d = (np.floor(d * L)).astype(int)
        b_h = (np.floor(h * L)).astype(int)
        b_w = (np.floor(w * L)).astype(int)
        
        c_d = np.floor(d/2.0).astype(int)
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)
        
        # 3D低频区域边界
        d1 = c_d - b_d
        d2 = c_d + b_d + 1
        h1 = c_h - b_h
        h2 = c_h + b_h + 1
        w1 = c_w - b_w
        w2 = c_w + b_w + 1
        
        # 替换3D低频区域中心
        a_src[:, d1:d2, h1:h2, w1:w2] = a_trg[:, d1:d2, h1:h2, w1:w2]
        a_src = np.fft.ifftshift(a_src, axes=(-3, -2, -1))
    else:
        # 2D处理（原始代码）
        a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
        a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))
        
        _, h, w = a_src.shape
        b = (np.floor(np.amin((h, w)) * L)).astype(int)
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)
        
        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1
        
        a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
        a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    
    return a_src

def FDA_source_to_target_np(src_img, trg_img, L=0.1):
    """
    numpy版本的3D FDA
    """
    is_3d = (len(src_img.shape) == 5)  # b,c,d,h,w
    
    if is_3d:
        # 3D处理
        src_img_np = src_img
        trg_img_np = trg_img
        
        # 3D傅里叶变换
        fft_src_np = np.fft.fftn(src_img_np, axes=(-3, -2, -1))
        fft_trg_np = np.fft.fftn(trg_img_np, axes=(-3, -2, -1))
        
        # 提取振幅和相位
        amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)
        
        # 替换低频振幅
        amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)
        
        # 重新组合复数并逆变换
        fft_src_ = amp_src_ * np.exp(1j * pha_src)
        src_in_trg = np.fft.ifftn(fft_src_, axes=(-3, -2, -1))
        src_in_trg = np.real(src_in_trg)
    else:
        # 2D处理（原始代码）
        src_img_np = src_img
        trg_img_np = trg_img
        
        # 2D傅里叶变换
        fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
        fft_trg_np = np.fft.fft2(trg_img_np, axes=(-2, -1))
        
        # 提取振幅和相位
        amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)
        
        # 替换低频振幅
        amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)
        
        # 重新组合复数并逆变换
        fft_src_ = amp_src_ * np.exp(1j * pha_src)
        src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
        src_in_trg = np.real(src_in_trg)
    
    return src_in_trg