import torch
import torch.nn as nn


class NormalizeDB(nn.Module):
    r"""Normalize the spectrogram with a minimum db value"""

    def __init__(self, min_level_db, max_abs_value, symmetric, normalization):
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
