import numpy as np
import torch.nn as nn

from src.modules.common import Snake1d
from src.modules.mel2token2mel.conv import SConv1d, SConvTranspose1d


def get_activation(activation: str = None, channels=None, **kwargs):
    if activation.lower() == "snake":
        assert channels is not None, "Snake activation needs channel number."
        return Snake1d(channels=channels)
    else:
        act = getattr(nn, activation)
        return act(**kwargs)


class SEANetResNetBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        compress: int = 2,  # from Demucs v3
        true_skip: bool = True,
        conv_group_ratio: int = -1,
    ):
        super(SEANetResNetBlock, self).__init__()

        hidden_channels = channels // compress
        kernel_sizes = (kernel_size, 1)
        dilations = (dilation, 1)

        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = channels if i == 0 else hidden_channels
            out_chs = channels if i == len(kernel_sizes) - 1 else hidden_channels
            block += [
                get_activation(activation, **{**activation_params, "channels": in_chs}),
                SConv1d(
                    in_channels=in_chs,
                    out_channels=out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    groups=min(in_chs, out_chs) // 2 // conv_group_ratio if conv_group_ratio > 0 else 1,
                ),
            ]
        self.block = nn.Sequential(*block)

        if true_skip:
            self.short_cut = nn.Identity()
        else:
            self.short_cut = SConv1d(
                channels,
                channels,
                kernel_size=1,
                groups=channels // 2 // conv_group_ratio if conv_group_ratio > 0 else 1,
            )

    def forward(self, x):
        return self.short_cut(x) + self.block(x)


class SEANetEncoderBlock(nn.Module):
    def __init__(
        self,
        ratio: int,
        in_channels: int,
        out_channels: int,
        residual_kernel_size: int = 3,
        residual_dilations: tuple = (1, 3),
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        compress: int = 2,
        true_skip: bool = True,
        conv_group_ratio: int = -1,
    ):
        super(SEANetEncoderBlock, self).__init__()

        assert ratio >= 1, f"ratio is {ratio}"

        block = []
        for dilation in residual_dilations:
            block += [
                SEANetResNetBlock(
                    channels=in_channels,
                    kernel_size=residual_kernel_size,
                    dilation=dilation,
                    activation=activation,
                    activation_params=activation_params,
                    compress=compress,
                    true_skip=true_skip,
                    conv_group_ratio=conv_group_ratio,
                )
            ]
        block += [
            get_activation(activation, **{**activation_params, "channels": in_channels}),
            SConv1d(in_channels, out_channels, kernel_size=ratio * 2, stride=ratio),
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class SEANetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 128,
        channels: tuple = (32, 64, 128),
        kernel_sizes: tuple = (3, 3),
        ratios: tuple = (2, 2),
        residual_kernel_size: int = 3,
        residual_dilations: tuple = (1, 3),
        double_out_channels: bool = False,
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        add_snake_activation: bool = False,
        compress: int = 2,
        true_skip: bool = True,
        conv_group_ratio: int = -1,
    ):
        super(SEANetEncoder, self).__init__()

        assert len(channels) == len(ratios) + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_rate = int(np.prod(ratios))
        self.ratios = ratios

        first_block = [SConv1d(in_channels, channels[0], kernel_size=kernel_sizes[0])]
        if add_snake_activation:
            first_block += [
                get_activation("snake", **{**activation_params, "channels": channels}),
                SConv1d(channels[0], channels[0], kernel_size=kernel_sizes[0]),
            ]
        self.first_block = nn.Sequential(*first_block)

        mid_block = []
        for i, ratio in enumerate(ratios):
            mid_block += [
                SEANetEncoderBlock(
                    ratio=ratio,
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    residual_kernel_size=residual_kernel_size,
                    residual_dilations=residual_dilations,
                    activation=activation,
                    activation_params=activation_params,
                    compress=compress,
                    true_skip=true_skip,
                    conv_group_ratio=conv_group_ratio,
                )
            ]

        self.mid_block = nn.Sequential(*mid_block)

        last_block = [
            get_activation(activation, **{**activation_params, "channels": channels[i + 1]}),
            SConv1d(
                channels[i + 1], out_channels * 2 if double_out_channels else out_channels, kernel_size=kernel_sizes[1]
            ),
        ]
        self.last_block = nn.Sequential(*last_block)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mid_block(x)
        x = self.last_block(x)
        return x


class SEANetDecoderBlock(nn.Module):
    def __init__(
        self,
        ratio: int,
        in_channels: int,
        out_channels: int,
        residual_kernel_size: int = 3,
        residual_dilations: tuple = (1, 3),
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        compress: int = 2,
        true_skip: bool = True,
        conv_group_ratio: int = -1,
    ):
        super(SEANetDecoderBlock, self).__init__()

        assert ratio >= 1, f"ratio is {ratio}"

        block = [
            get_activation(activation, **{**activation_params, "channels": in_channels}),
            SConvTranspose1d(in_channels, out_channels, kernel_size=ratio * 2, stride=ratio),
        ]
        for dilation in residual_dilations:
            block += [
                SEANetResNetBlock(
                    channels=out_channels,
                    kernel_size=residual_kernel_size,
                    dilation=dilation,
                    activation=activation,
                    activation_params=activation_params,
                    compress=compress,
                    true_skip=true_skip,
                    conv_group_ratio=conv_group_ratio,
                )
            ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class SEANetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 80,
        channels: tuple = (32, 64, 128),
        kernel_sizes: tuple = (3, 3),
        ratios: tuple = (2, 2),
        residual_kernel_size: int = 3,
        residual_dilations: tuple = (1, 3),
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        add_snake_activation: bool = False,
        compress: int = 2,
        true_skip: bool = True,
        conv_group_ratio: int = -1,
        final_activation: str = None,
        final_activation_params: dict = None,
    ):
        super(SEANetDecoder, self).__init__()

        assert len(channels) == len(ratios) + 1
        self.in_channels = in_channels
        self.out_channels = out_channels

        first_block = [SConv1d(in_channels, channels[0], kernel_size=kernel_sizes[0])]
        self.first_block = nn.Sequential(*first_block)

        mid_block = []
        for i, ratio in enumerate(ratios):
            mid_block += [
                SEANetDecoderBlock(
                    ratio=ratio,
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    residual_kernel_size=residual_kernel_size,
                    residual_dilations=residual_dilations,
                    activation=activation,
                    activation_params=activation_params,
                    compress=compress,
                    true_skip=true_skip,
                    conv_group_ratio=conv_group_ratio,
                )
            ]

        self.mid_block = nn.Sequential(*mid_block)

        last_block = [
            get_activation(
                "snake" if add_snake_activation else activation, **{**activation_params, "channels": channels[i + 1]}
            ),
            SConv1d(channels[i + 1], out_channels, kernel_size=kernel_sizes[1]),
        ]

        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            last_block += [final_act(**final_activation_params)]
        self.last_block = nn.Sequential(*last_block)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mid_block(x)
        x = self.last_block(x)
        return x
