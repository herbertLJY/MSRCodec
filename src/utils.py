import torch
import re
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
import librosa
import numpy as np
import logging
import os


def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)]
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids


_whitespace_re = re.compile(r"\s+")


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text




def apply_weight_norm(module: nn.Module):
    """Apply weight normalization module from all the layers."""

    def _apply_weight_norm(m, name="weight"):
        for k, hook in m._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm) and hook.name == name:
                return

        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.utils.weight_norm(m, name=name)
        if isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
            nn.utils.weight_norm(m, name=name, dim=1)

    module.apply(_apply_weight_norm)


def remove_weight_norm(module: nn.Module):
    """Remove weight normalization module from all the layers."""

    def _remove_weight_norm(m):
        try:
            nn.utils.remove_weight_norm(m)
        except ValueError:  # this module didn't have weight norm
            return

    module.apply(_remove_weight_norm)


def load_wav(path, sample_rate=None, normalize=True, return_type="float32"):
    assert return_type in ["float32", "int16"]

    raw_data, sr = librosa.load(path, sr=None)

    raw_data = np.clip(raw_data, -1.0, 1.0)

    if sample_rate is not None and sr != sample_rate:
        raw_data = librosa.resample(raw_data, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    if normalize:
        raw_data = librosa.util.normalize(raw_data) * 0.95

    if return_type == "int16":
        raw_data = ((raw_data + 1) / 2 * 65535.0 - 32768).astype(np.int16)

    return raw_data, sr
    



def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    is_rank_zero = int(os.environ.get("RANK", 0)) == 0

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, getattr(logger, level) if is_rank_zero else lambda msg, *args, **kwargs: None)

    return logger
