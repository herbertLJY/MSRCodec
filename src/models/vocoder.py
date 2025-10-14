import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, use_sin=False):
        super(Upsample, self).__init__()
        self.scale = scale

        self.transposed_conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=scale * 2,
            stride=scale,
            padding=scale // 2 + scale % 2,
            output_padding=scale % 2,
        )
        self.use_sin = use_sin
        if use_sin:
            self.conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        if self.use_sin:
            x = x + torch.sin(x)
            a = self.transposed_conv(x)
            b = self.conv(F.interpolate(x, scale_factor=self.scale))
            x = a + b
        else:
            x = self.transposed_conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self, channels: int = 512, kernel_size: int = 3, dilations: tuple = (1, 3, 5), use_additional_convs: bool = True
    ):
        super(ResidualBlock, self).__init__()

        assert kernel_size % 2 == 1
        self.use_additional_convs = use_additional_convs

        self.convs1 = nn.ModuleList()
        if use_additional_convs:
            self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1 += [
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        padding=(kernel_size - 1) // 2 * dilation,
                    ),
                )
            ]
            if use_additional_convs:
                self.convs2 += [
                    nn.Sequential(
                        nn.LeakyReLU(0.1),
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x


class FreGANGenerator(nn.Module):
    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        dropout=0.1,
        use_rnn=True,
        rnn_type="gru",
        rnn_layers=1,
        rnn_dropout=0.1,
        rnn_bidirectional=True,
        upsample_scales=(2, 4, 4, 5),
        upsample_use_sin=False,
        resblock_kernel_sizes=(3,),
        resblock_dilations=((1, 3, 9),),
        use_additional_convs=True,
        use_residual=True,
        use_weight_norm=True,
    ):
        super(FreGANGenerator, self).__init__()

        # check hyper parameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        self.num_upsamples = len(upsample_scales)
        self.num_blocks = len(resblock_kernel_sizes)
        self.use_rnn = use_rnn
        self.use_residual = use_residual

        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size, 1, padding=kernel_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        if use_rnn:
            rnn_dropout = 0.0 if rnn_layers < 2 else rnn_dropout
            if rnn_type == "gru":
                self.rnn = nn.GRU(
                    channels,
                    channels // 2 if rnn_bidirectional else channels,
                    num_layers=rnn_layers,
                    dropout=rnn_dropout,
                    bidirectional=rnn_bidirectional,
                    batch_first=True,
                )
            else:
                self.rnn = nn.LSTM(
                    channels,
                    channels // 2 if rnn_bidirectional else channels,
                    num_layers=rnn_layers,
                    dropout=rnn_dropout,
                    bidirectional=rnn_bidirectional,
                    batch_first=True,
                )

        self.upsamples = nn.ModuleList()
        self.cond_upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.output_layers = nn.ModuleList()

        self.non_linear = nn.LeakyReLU(0.1)
        cond_channels = channels
        for i in range(len(upsample_scales)):
            self.upsamples += [
                Upsample(
                    in_channels=channels // (2**i),
                    out_channels=channels // (2 ** (i + 1)),
                    scale=upsample_scales[i],
                    use_sin=upsample_use_sin,
                )
            ]

            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        use_additional_convs=use_additional_convs,
                    )
                ]

            if i >= 1 and use_residual:
                self.output_layers.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=upsample_scales[i], mode="nearest"),
                        nn.Conv1d(channels // (2**i), channels // (2 ** (i + 1)), 1),
                        nn.LeakyReLU(0.1),
                    )
                )

                self.cond_upsamples.append(
                    nn.Sequential(
                        Upsample(
                            in_channels=cond_channels, out_channels=channels // (2**i), scale=upsample_scales[i - 1]
                        ),
                        nn.LeakyReLU(0.1),
                    )
                )

                cond_channels = channels // (2**i)

        self.output_conv = nn.Sequential(
            nn.Conv1d(channels // (2 ** (i + 1)), out_channels, kernel_size, padding=kernel_size // 2), nn.Tanh()
        )

        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x):
        x = self.input_conv(x)

        if self.use_rnn:
            x, _ = self.rnn(x.transpose(1, 2))
            x = x.transpose(1, 2)

        condition = x
        output = None
        for i in range(self.num_upsamples):
            if i >= 1 and self.use_residual:
                if output is None:
                    output = self.output_layers[i - 1](x)
                else:
                    output = self.output_layers[i - 1](output)

            if i >= 1 and self.use_residual:
                condition = self.cond_upsamples[i - 1](condition)
                x += condition

            x = self.upsamples[i](x)
            xs = None  # initialize
            for j in range(self.num_blocks):
                if xs is None:
                    xs = self.blocks[i * self.num_blocks + j](x)
                else:
                    xs += self.blocks[i * self.num_blocks + j](x)
            x = xs / self.num_blocks
            x = self.non_linear(x)

            if output is not None and self.use_residual:
                output = output + x

        x = self.output_conv(output if self.use_residual else x)
        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                nn.utils.weight_norm(m)
            if isinstance(m, nn.ConvTranspose1d):
                nn.utils.weight_norm(m, dim=1)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.02)

        self.apply(_reset_parameters)
