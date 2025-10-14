import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window


def init_kernels(n_fft, window_length, window_type=None, inverse=False):
    if window_type is None:
        window = np.ones(window_length)
    else:
        window = get_window(window_type, window_length, fftbins=True)

    fourier_basis = np.fft.rfft(np.eye(n_fft))[:window_length]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if inverse:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return (
        torch.from_numpy(kernel.astype(np.float32)).contiguous(),
        torch.from_numpy(window[None, :, None].astype(np.float32)).contiguous(),
    )


class STFT(nn.Module):
    def __init__(self, win_length, hop_length, n_fft=None, window_type="hamming"):
        super(STFT, self).__init__()

        if n_fft is None:
            self.n_fft = np.int(2 ** np.ceil(np.log2(win_length)))
        else:
            self.n_fft = n_fft

        kernel, _ = init_kernels(self.n_fft, win_length, window_type)
        self.register_buffer("weight", kernel)

        self.stride = hop_length
        self.win_length = win_length
        self.dim = self.n_fft // 2 + 1

    def forward(self, inputs):
        inputs = F.pad(inputs, [self.n_fft // 2, self.n_fft // 2], mode="reflect")
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        real = outputs[:, : self.dim, :]
        imag = outputs[:, self.dim :, :]
        mags = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)
        return mags, phase


class ISTFT(nn.Module):
    def __init__(self, win_length, hop_length, n_fft=None, window_type="hamming"):
        super(ISTFT, self).__init__()

        if n_fft is None:
            self.n_fft = np.int(2 ** np.ceil(np.log2(win_length)))
        else:
            self.n_fft = n_fft
        kernel, window = init_kernels(self.n_fft, win_length, window_type, inverse=True)

        self.register_buffer("weight", kernel)

        self.window_type = window_type
        self.win_length = win_length
        self.stride = hop_length

        self.pad = (win_length - hop_length) // 2

        self.dim = self.n_fft // 2 + 1
        self.register_buffer("window", window)
        self.register_buffer("enframe", torch.eye(win_length)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """
        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs /= torch.clamp(coff, min=1e-8)
        outputs = outputs[:, :, self.pad : -self.pad]
        return outputs
