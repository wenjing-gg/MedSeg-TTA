import numpy as np

def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))
    _, _, h, w = a_src.shape
    b = np.floor(np.amin((h, w)) * L).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1
    a_src[:, :, h1:h2, w1:w2] = a_trg[:, :, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src

def FDA_source_to_target_np(src_img, trg_img, L=0.1):
    if L == 0:
        return src_img
    src_img_np = src_img
    trg_img_np = trg_img
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(trg_img_np, axes=(-2, -1))
    amp_src, pha_src = (np.abs(fft_src_np), np.angle(fft_src_np))
    amp_trg, pha_trg = (np.abs(fft_trg_np), np.angle(fft_trg_np))
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)
    fft_src_ = amp_src_ * np.exp(1j * pha_src)
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)
    return src_in_trg

def low_freq_mutate_np_hfi(amp_src, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    _, _, h, w = a_src.shape
    b = np.floor(np.amin((h, w)) * L).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1
    a_src[:, :, h1:h2, w1:w2] = 0
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src

def FDA_img_to_hfi(src_img, L=0.1):
    assert L >= 0, 'L must be a non-negative value!'
    if L == 0:
        return src_img
    src_img_np = src_img
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    amp_src, pha_src = (np.abs(fft_src_np), np.angle(fft_src_np))
    amp_src_ = low_freq_mutate_np_hfi(amp_src, L=L)
    fft_src_ = amp_src_ * np.exp(1j * pha_src)
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)
    return src_in_trg
