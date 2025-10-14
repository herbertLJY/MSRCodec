import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, dim: int = -1, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__(
            normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(self.dim, -1)
        x = super().forward(x)
        return x.transpose(self.dim, -1)
