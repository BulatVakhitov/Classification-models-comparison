import torch
from torch import nn
from models.SEblock import SEblock


def conv_block(in_channels,
               out_channels,
               kernel_size,
               stride,
               groups=1,
               use_batch_norm=True,
               use_activation=True,
               max_pool=False):

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
        layers.append(nn.ReLU())
    if max_pool:
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    return nn.Sequential(*layers)


class BottleNeckBlock(nn.Module):

    expansion: int = 4
    se_reduction: int = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_resnet_d: bool,
                 downsample: bool = False,
                 stride: int = 1,
                 groups: int = 1,
                 width: int = 64,
                 downsample_kernel_size: int = 1,
                 use_se: bool = True) -> None:
        super().__init__()

        width = int(out_channels * (width / 64.0)) * groups

        self.conv1 = conv_block(in_channels=in_channels,
                                out_channels=width,
                                kernel_size=1,
                                stride=1)
        self.conv2 = conv_block(in_channels=width,
                                out_channels=width,
                                kernel_size=3,
                                stride=stride,
                                groups=groups)
        self.conv3 = conv_block(in_channels=width,
                                out_channels=out_channels * self.expansion,
                                kernel_size=1,
                                stride=1,
                                use_activation=False)

        if use_se:
            squeeze_channels = out_channels * self.expansion // self.se_reduction
            self.se_block = SEblock(
                in_channels = out_channels * self.expansion,
                squeeze_channels = squeeze_channels
            )
        else:
            self.se_block = nn.Identity()

        if downsample:
            downsample_layers = []
            if use_resnet_d and stride == 2:
                downsample_layers.append(nn.AvgPool2d(kernel_size=2,
                                                      stride=2))
                stride = 1
            downsample_layers.append(conv_block(in_channels,
                                                out_channels * self.expansion,
                                                downsample_kernel_size,
                                                stride,
                                                use_activation=False))

            self.downsample = nn.Sequential(*downsample_layers)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.se_block(output)

        residual = self.downsample(residual)
        output = nn.ReLU()(output + residual)
        return output


class ResBlock(nn.Module):

    expansion: int = 1
    se_reduction: int = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_resnet_d: bool,
                 downsample: bool = False,
                 stride: int = 1,
                 groups: int = 1,
                 width: int = 64,
                 downsample_kernel_size: int = 1,
                 use_se: bool = True) -> None:
        super().__init__()

        if groups != 1 or width != 64:
            raise ValueError("ResBlock only supports groups=1 and width=64. Use BottleNeckBlock instead")

        self.has_downsample = downsample

        self.conv1 = conv_block(in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=stride)
        self.conv2 = conv_block(out_channels,
                                out_channels,
                                kernel_size=3,
                                stride=1)

        if use_se:
            squeeze_channels = out_channels * self.expansion // self.se_reduction
            self.se_block = SEblock(
                in_channels=out_channels * self.expansion, 
                squeeze_channels=squeeze_channels
            )
        else:
            self.se_block = nn.Identity()

        if downsample:
            self.downsample_layers = []

            if use_resnet_d and self.stride == 2:
                self.downsample_layers.append(
                    nn.AvgPool2d(
                        kernel_size=2,
                        stride=2
                    )
                )
                # resnet d - AvgPool2d(kernel_size=2, stride=2) + Conv1x1(stride=1)
                stride = 1

            self.downsample_layers.append(
                conv_block(in_channels,
                           out_channels,
                           kernel_size=downsample_kernel_size,
                           stride=stride,
                           use_activation=False))

            self.downsample = nn.Sequential(*self.downsample_layers)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        output = self.conv1(x)
        output = self.conv2(output)
        output = self.se_block(output)
        residual = self.downsample(residual)
        output = nn.ReLU()(output + residual)
        return output


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 num_layers: int,
                 num_classes: int = 10,
                 groups: int = 1,
                 width: int = 64,
                 use_resnet_c: bool = False,
                 use_resnet_d: bool = False,
                 downsample_kernel_size: int = 1,
                 use_se: bool = True):
        super().__init__()

        self.num_layers = num_layers
        self.use_resnet_d = use_resnet_d
        self.downsample_kernel_size = downsample_kernel_size
        self.block = block
        self.groups = groups
        self.width = width
        self.use_se = use_se

        if self.num_layers == 18:
            self.layers = [2, 2, 2, 2]
        elif self.num_layers in (34, 50):
            self.layers = [3, 4, 6, 3]
        elif self.num_layers == 101:
            self.layers = [3, 4, 23, 3]
        elif self.num_layers == 152:
            self.layers = [3, 4, 36, 3]
        else:
            raise ValueError('num_layers must be 18, 34, 50, 101 or 152')

        if use_resnet_c:
            self.layer0 = nn.Sequential(
                conv_block(
                    in_channels=3,
                    out_channels=64,
                    kernel_size=3,
                    stride=2,
                    use_activation=False
                ),
                conv_block(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    use_activation=False
                ),
                conv_block(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    max_pool=True
                )
            )
        else:
            self.layer0 = nn.Sequential(
                conv_block(
                    in_channels=3,
                    out_channels=64,
                    kernel_size=7,
                    stride=2,
                    max_pool=True
                )
            )

        self.layer1 = self._make_layer(
            in_channels=64,
            out_channels=64,
            repeats=self.layers[0],
            stride=1
        )
        self.layer2 = self._make_layer(
            in_channels=64,
            out_channels=128,
            repeats=self.layers[1],
            stride=2
        )
        self.layer3 = self._make_layer(
            in_channels=128,
            out_channels=256,
            repeats=self.layers[2],
            stride=2
        )
        self.layer4 = self._make_layer(
            in_channels=256,
            out_channels=512,
            repeats=self.layers[3],
            stride=2
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * self.block.expansion, num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int,
        stride: int = 1
    ) -> nn.Sequential:

        downsample = (stride != 1) or (in_channels != out_channels * self.block.expansion)

        if stride == 2 and issubclass(self.block, BottleNeckBlock): # starting from layer2
            in_channels = in_channels * self.block.expansion

        layers = []
        layers.append(
            self.block(
                in_channels = in_channels,
                out_channels = out_channels,
                use_resnet_d = self.use_resnet_d,
                downsample = downsample,
                stride = stride,
                groups = self.groups,
                width = self.width,
                downsample_kernel_size=self.downsample_kernel_size,
                use_se = self.use_se
            )
        )

        in_channels = out_channels * self.block.expansion

        for _ in range(1, repeats):
            layers.append(
                self.block(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    use_resnet_d = self.use_resnet_d,
                    groups = self.groups,
                    width = self.width,
                    downsample_kernel_size = self.downsample_kernel_size,
                    use_se = self.use_se
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet(num_layers, **kwargs) -> ResNet:

    if num_layers > 34:
        block = BottleNeckBlock
    else:
        block = ResBlock

    return ResNet(block, num_layers, **kwargs)


def resnext(num_layers, cardinality, width, **kwargs):

    return ResNet(
        BottleNeckBlock,
        num_layers,
        groups = cardinality,
        width = width,
        **kwargs
    )
