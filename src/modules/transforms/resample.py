import torch
import torch.nn as nn
import torchaudio

class Resample(nn.Module):
    r"""Normalize the spectrogram with a minimum db value"""

    def __init__(self, orig_sample_rate=16000, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.orig_sample_rate = orig_sample_rate
        if self.sample_rate == self.orig_sample_rate:
            self.resample = nn.Identity()
        else:
            self.resample = torchaudio.transforms.Resample(orig_freq=self.orig_sample_rate, new_freq=self.sample_rate)

    def forward(self, x):
        x = self.resample(x)
        return x
