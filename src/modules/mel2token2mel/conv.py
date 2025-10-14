import math
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_logger

logger = get_logger(__name__)


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = "zero", value: float = 0.0):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def pad2d(
    x: torch.Tensor, paddings: tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]], mode: str = "zero", value: float = 0.0
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    freq_len, time_len = x.shape[-2:]
    padding_time, padding_freq = paddings
    assert min(padding_freq) >= 0 and min(padding_time) >= 0, (padding_time, padding_freq)
    if mode == "reflect":
        max_time_pad, max_freq_pad = max(padding_time), max(padding_freq)
        extra_time_pad = max_time_pad - time_len + 1 if time_len <= max_time_pad else 0
        extra_freq_pad = max_freq_pad - freq_len + 1 if freq_len <= max_freq_pad else 0
        extra_pad = [0, extra_time_pad, 0, extra_freq_pad]
        x = F.pad(x, extra_pad)
        padded = F.pad(x, (*padding_time, *padding_freq), mode, value)
        freq_end = padded.shape[-2] - extra_freq_pad
        time_end = padded.shape[-1] - extra_time_pad
        return padded[..., :freq_end, :time_end]
    else:
        return F.pad(x, (*paddings[0], *paddings[1]), mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def unpad2d(x: torch.Tensor, paddings: tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_time_left, padding_time_right = paddings[0]
    padding_freq_left, padding_freq_right = paddings[1]
    assert min(paddings[0]) >= 0 and min(paddings[1]) >= 0, paddings
    assert (padding_time_left + padding_time_right) <= x.shape[-1] and (
        padding_freq_left + padding_freq_right
    ) <= x.shape[-2]

    freq_end = x.shape[-2] - padding_freq_right
    time_end = x.shape[-1] - padding_time_right
    return x[..., padding_freq_left:freq_end, padding_time_left:time_end]


class SConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            logger.warning(
                "SConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias
        )

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad_mode = pad_mode
        self.padding_total = (self.kernel_size - 1) * self.dilation - (self.stride - 1)

    def forward(self, x):
        extra_padding = get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, self.padding_total)
        padding_right = self.padding_total // 2
        padding_left = self.padding_total - padding_right
        x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)

        x = self.conv(x)
        return x


class SConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)

        self.padding_total = kernel_size - stride

    def forward(self, x):
        y = self.convtr(x)

        padding_right = self.padding_total // 2
        padding_left = self.padding_total - padding_right
        y = unpad1d(y, (padding_left, padding_right))
        return y


def tuple_it(x, num=2):
    if isinstance(x, list):
        return tuple(x[:2])
    elif isinstance(x, int):
        return tuple([x for _ in range(num)])
    else:
        return x


class SConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tp.Union[int, tp.Tuple[int, int]],
        stride: tp.Union[int, tp.Tuple[int, int]] = 1,
        dilation: tp.Union[int, tp.Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        kernel_size, stride, dilation = tuple_it(kernel_size), tuple_it(stride), tuple_it(dilation)

        if max(stride) > 1 and max(dilation) > 1:
            logger.warning(
                "SConv2d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias
        )
        self.pad_mode = pad_mode

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        assert len(x.shape) == 4, x.shape
        padding_total_list: tp.List[int] = []
        extra_padding_list: tp.List[int] = []
        for i, (kernel_size, stride, dilation) in enumerate(zip(self.kernel_size, self.stride, self.dilation)):
            padding_total = (kernel_size - 1) * dilation - (stride - 1)
            if i == 0:
                # no extra padding for frequency dim
                extra_padding = 0
            else:
                extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
            padding_total_list.append(padding_total)
            extra_padding_list.append(extra_padding)

        # Asymmetric padding required for odd strides
        freq_after = padding_total_list[0] // 2
        freq_before = padding_total_list[0] - freq_after + extra_padding_list[0]
        time_after = padding_total_list[1] // 2
        time_before = padding_total_list[1] - time_after + extra_padding_list[1]
        x = pad2d(x, ((time_before, time_after), (freq_before, freq_after)), mode=self.pad_mode)
        x = self.conv(x)
        return x


class SConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tp.Union[int, tp.Tuple[int, int]],
        stride: tp.Union[int, tp.Tuple[int, int]] = 1,
        out_padding: tp.Union[int, tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]]] = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.convtr = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
        )
        if isinstance(out_padding, int):
            self.out_padding = [(out_padding, out_padding), (out_padding, out_padding)]
        else:
            self.out_padding = out_padding

        self.padding_freq_total = kernel_size[0] - stride[0]
        self.padding_time_total = kernel_size[1] - stride[1]

    def forward(self, x):
        y = self.convtr(x)

        (freq_out_pad_left, freq_out_pad_right) = self.out_padding[0]
        (time_out_pad_left, time_out_pad_right) = self.out_padding[1]

        padding_freq_right = self.padding_freq_total // 2
        padding_freq_left = self.padding_freq_total - padding_freq_right
        padding_time_right = self.padding_time_total // 2
        padding_time_left = self.padding_time_total - padding_time_right

        y = unpad2d(
            y,
            (
                (max(padding_time_left - time_out_pad_left, 0), max(padding_time_right - time_out_pad_right, 0)),
                (max(padding_freq_left - freq_out_pad_left, 0), max(padding_freq_right - freq_out_pad_right, 0)),
            ),
        )
        return y
