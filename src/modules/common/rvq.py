import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn.utils import weight_norm

from einops import rearrange, reduce, repeat
from src.modules.common.vq import VectorQuantize as VectorQuantize_new

def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t



def check_nan(x, n):
    if torch.isnan(x).any():
        print('{} NAN'.format(n))
        return True
    if torch.isinf(x).any():
        print('{} inf'.format(n))
        return True
    return False

        
def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins



def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))



class ResidualVectorQuantize(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, num_quantizers, vq_type='vq', **kwargs):
        super().__init__()
        vq = VectorQuantize_new
        self.layers = nn.ModuleList([vq(**kwargs) for _ in range(num_quantizers)])
        self.num_tokens = self.layers[0].num_tokens
        self.num_codebook = num_quantizers

    def forward(self, x):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            # https://github.com/lucidrains/vector-quantize-pytorch/issues/33
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses.mean()

    def get_embedding_from_code(self, embed_id):
        quantized_out = 0.0
        for idx, layer in enumerate(self.layers):
            quantized = layer.get_embedding_from_code(embed_id[idx])
            quantized_out = quantized_out + quantized
        return quantized_out

