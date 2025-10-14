import torch
import torch.nn as nn


class SEBottleneck1D(nn.Module):
    expansion_default = 4

    def __init__(self, inplanes, outplanes, stride=1, expansion=4, groups=1, reduction=8, dialation=1):
        super(SEBottleneck1D, self).__init__()
        planes = max(inplanes // expansion, outplanes // expansion)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        if groups == -1:
            self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False, groups=planes)
        else:
            self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(outplanes)

        out_channel = outplanes
        att_inplanes = max(out_channel // reduction, 8)

        self.att_fc = nn.Sequential(
            nn.Conv1d(out_channel, att_inplanes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(att_inplanes, out_channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if outplanes != inplanes:
            self.downsample = nn.Sequential(nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm1d(outplanes))
        else:
            self.downsample = nn.Identity()

    def SE_block(self, x):
        out_global = torch.mean(x, dim=-1, keepdim=True)
        SE_att_w = self.att_fc(out_global)
        out = x * SE_att_w
        return out

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(1, out.size())
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print(2, out.size())
        out = self.conv3(out)
        out = self.bn3(out)

        out = self.SE_block(out)

        # print(3, out.size())
        residual = self.downsample(residual)

        # print(11, residual.size(), out.size())
        out += residual
        out = self.relu(out)
        # print(out.size())
        return out


class SEResConv1D(nn.Module):
    def __init__(self, in_channels, out_channels=None, 
                channels=[512, 512], layer_numbers=[1, 1]):
        super(SEResConv1D, self).__init__()
        assert len(channels) == len(layer_numbers)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = []
        for i in range(len(layer_numbers)):
            self.channels += [channels[i]] * layer_numbers[i]

        self.convs = nn.ModuleList()

        for i, c in enumerate(self.channels):
            if i == 0:
                self.convs.append(SEBottleneck1D(in_channels, c))
            else:
                self.convs.append(SEBottleneck1D(self.channels[i - 1], c))
        
        self.convs.append(SEBottleneck1D(self.channels[-1], out_channels))


    def forward(self, x):
        for i, conv in enumerate(self.convs):
            # print(i, x.size())
            x = conv(x)
            # print(x.size())
        return x


if __name__ == "__main__":
    model = SEResConv1D(128, 256, channels=[256, 512], layer_numbers=[4, 6])
    x = torch.randn(1, 128, 100)
    y = model(x)
    print(y.size())