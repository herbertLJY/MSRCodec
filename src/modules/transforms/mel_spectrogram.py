import librosa
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from src.modules.common import ISTFT


class NormalizeDB(nn.Module):
    r"""Normalize the spectrogram with a minimum db value"""

    def __init__(self, normalization, min_level_db=-100, max_abs_value=1.0, symmetric=True):
        super().__init__()
        self.min_level_db = min_level_db
        self.max_abs_value = max_abs_value
        self.symmetric = symmetric
        self.normalization = normalization

    def forward(self, spec):
        spec = torch.log10(torch.clamp(spec, min=1e-5))
        if self.normalization:
            spec = torch.clamp((self.min_level_db - 20 * spec) / self.min_level_db, min=0, max=1)
            spec *= self.max_abs_value
            if self.symmetric:
                spec = spec * 2.0 - self.max_abs_value
        return spec

    def inverse(self, spec):
        if self.normalization:
            if self.symmetric:
                spec = (spec + self.max_abs_value) / 2.0
            spec /= self.max_abs_value
            spec = (self.min_level_db - self.min_level_db * spec) / 20
        spec = 10**spec
        return spec


class TacotronMelSpectrogram(nn.Module):
    def __init__(
        self,
        norm="slaney",
        mel_scale="slaney",
        power=1.0,
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        orig_sample_rate=None,
    ):
        super().__init__()

        self.normalizer = NormalizeDB(min_level_db=-100, max_abs_value=1.0, symmetric=True, normalization=True)
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            norm=norm,
            mel_scale=mel_scale,
            power=power,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.orig_sample_rate = orig_sample_rate if orig_sample_rate is not None else sample_rate

        self.griffinlim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=1.0
        )

        if self.orig_sample_rate != self.sample_rate:
            self.resample = torchaudio.transforms.Resample(orig_freq=self.orig_sample_rate, new_freq=self.sample_rate)
        else:
            self.resample = nn.Identity()

    def forward(self, x, with_wav=False):
        x = self.resample(x)

        # Remove uncertainty of stft transformation
        frame_size = x.size(-1) // self.hop_length
        mel = self.mel_stft(x)
        mel = self.normalizer(mel)

        if with_wav:
            return x, mel[..., :frame_size]
        else:
            return mel[..., :frame_size]

    def inverse(self, mel, return_spec=False):
        mel = self.normalizer.inverse(mel)
        inverse_mel_mat = torch.linalg.pinv(self.mel_stft.mel_scale.fb)
        spec = torch.matmul(inverse_mel_mat.unsqueeze(0).transpose(1, 2), mel)
        spec = torch.clamp(spec, min=1e-10)
        if return_spec:
            return spec
        x = self.griffinlim(spec)
        return x
