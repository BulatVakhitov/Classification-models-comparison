import math

import torch
from torch import nn

from models.SEblock import SEblock


net_param = {
        # 'efficientnet type': (width_coef, depth_coef, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5)
    }


def stochastic_depth(input_tensor, p, mode, training= True):

    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input_tensor

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input_tensor.shape[0]] + [1] * (input_tensor.ndim - 1)
    else:
        size = [1] * input_tensor.ndim
    noise = torch.empty(size, dtype=input_tensor.dtype, device=input_tensor.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input_tensor * noise


def conv_bn_silu(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    groups=1,
    use_batch_norm=True,
    use_activation=True
):

    padding = kernel_size // 2

    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=not use_batch_norm
        )
    ]

    if use_batch_norm:
        layers.append(nn.BatchNorm2d(num_features=out_channels))
    if use_activation:
        layers.append(nn.SiLU())

    return nn.Sequential(*layers)



def round_channels(channels, width, divisor=8):
    channels *= width
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels


class StochasticDepth(nn.Module):

    def __init__(self, p, mode):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x):
        return stochastic_depth(x, self.p, self.mode, self.training)


class InvertedResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expand_ratio,
        se_reduction=0.0,
        se_block=SEblock,
        stochastic_depth_prob = 0.2,):

        super().__init__()

        hidden_channels = int(round(in_channels * expand_ratio))
        self.has_res_connection = (stride == 1 and in_channels == out_channels)

        inv_layers = []
        if expand_ratio != 1:
            inv_layers.append(conv_bn_silu(in_channels, hidden_channels, kernel_size=1))

        #depthwise
        inv_layers.append(
            conv_bn_silu(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_channels
            )
        )

        if se_reduction != 0.0:
            squeeze_channels = in_channels // se_reduction
            inv_layers.append(se_block(hidden_channels, squeeze_channels, nn.SiLU))

        #pointwise
        inv_layers.append(
            conv_bn_silu(
                hidden_channels,
                out_channels,
                kernel_size=1,
                use_activation=False
            )
        )

        if self.has_res_connection:
            inv_layers.append(StochasticDepth(stochastic_depth_prob, mode='row'))

        self.inv_block = nn.Sequential(*inv_layers)


    def forward(self, x):
        output = self.inv_block(x)
        if self.has_res_connection:
            output += x
        return output


class MBConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio,
                 se_reduction,
                 num_repeat):
        super().__init__()

        self.inv_block = InvertedResBlock

        inv_block_layers = [
            self.inv_block(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                expand_ratio,
                se_reduction)
        ]

        for _ in range(1, num_repeat):
            inv_block_layers.append(
                self.inv_block(
                    out_channels,
                    out_channels,
                    kernel_size,
                    1,
                    expand_ratio,
                    se_reduction
                )
            )
        self.inv_block_layers = nn.Sequential(*inv_block_layers)

    def forward(self, x):
        return self.inv_block_layers(x)


class EfficientNet(nn.Module):

    config = [
        #(in_channels, out_channels, kernel_size, stride, expand_ratio, se_reduction, repeats)
        [32,  16,  3, 1, 1, 4, 1],
        [16,  24,  3, 2, 6, 4, 2],
        [24,  40,  5, 2, 6, 4, 2],
        [40,  80,  3, 2, 6, 4, 3],
        [80,  112, 5, 1, 6, 4, 3],
        [112, 192, 5, 2, 6, 4, 4],
        [192, 320, 3, 1, 6, 4, 1]
    ]

    def __init__(self,
                 parameters,
                 layer0_channels=32,
                 num_classes=10,
                 feature_size=1280):
        super().__init__()

        self.mb_block = MBConvBlock
        width = parameters[0]
        depth = parameters[1]
        input_size = parameters[2]
        dropout_rate = parameters[3]

        if width != 1.0:
            layer0_channels = round_channels(layer0_channels, width)
            for conf in self.config:
                conf[0] = round_channels(conf[0], width)
                conf[1] = round_channels(conf[1], width)

        if depth != 1.0:
            for conf in self.config:
                conf[6] = int(math.ceil(conf[6]*depth))

        self.layer0 = conv_bn_silu(
            in_channels=3,
            out_channels=layer0_channels,
            kernel_size=3,
            stride=2
        )

        layers = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, se_reduction, repeats in self.config:
            layers.append(
                self.mb_block(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    expand_ratio,
                    se_reduction,
                    repeats
                )
            )

        self.layers = nn.Sequential(*layers)

        self.last_conv_layer = conv_bn_silu(in_channels=self.config[-1][1],
                                            out_channels=feature_size,
                                            kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool2d((1,1))

        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.fc = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        output = self.layer0(x)
        output = self.layers(output)
        output = self.last_conv_layer(output)

        output = self.pooling(output)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.fc(output)

        return output
