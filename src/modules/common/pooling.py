import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPool1D(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=0, ceil_mode=False):
        super(AvgPool1D, self).__init__()
        self.ceil_mode = ceil_mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode)

    def forward(self, x):
        x_len = x.size(-1)
        if x_len % self.stride != 0:
            x = F.pad(x, (0, self.stride - x_len % self.stride))
        return self.pool(x)


class MaxPool1D(AvgPool1D):
    def __init__(self, kernel_size=3, stride=2, padding=0, ceil_mode=False):
        super(MaxPool1D, self).__init__()
        self.ceil_mode = ceil_mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool = nn.MaxPool1d(kernel_size, stride, padding, ceil_mode)

    def forward(self, x):
        x_len = x.size(-1)
        if x_len % self.stride != 0:
            x = F.pad(x, (0, self.stride - x_len % self.stride))
        return self.pool(x)