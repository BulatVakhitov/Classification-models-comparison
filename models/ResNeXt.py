import torch
from torch import nn
from typing import Tuple, Dict, List


class ResNeXtBlock(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 cardinality: int, 
                 width: int, 
                 downsample: bool=False, 
                 stride: int=1):
        super().__init__()

        self.expansion = 2
        self.cardinality = cardinality
        self.has_downsample = downsample

        self.inner_width = cardinality * width

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=1, 
                      in_channels=in_channels, 
                      out_channels=self.inner_width, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            nn.BatchNorm2d(num_features=self.inner_width),
            nn.ReLU(),
            nn.Conv2d(kernel_size=3, 
                      in_channels=self.inner_width, 
                      out_channels=self.inner_width, 
                      stride=stride, 
                      groups=cardinality,
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(num_features=self.inner_width),
            nn.ReLU(),
            nn.Conv2d(kernel_size=1, 
                      in_channels=self.inner_width, 
                      out_channels=self.inner_width * self.expansion, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            nn.BatchNorm2d(num_features=self.inner_width * self.expansion)
        )

        if downsample:
            if stride == 2: # layer1 имеет stride=1, из-за этого все ломается
                self.downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, 
                                 stride=2),
                    nn.Conv2d(kernel_size=3, 
                              in_channels=in_channels, 
                              out_channels=self.inner_width*self.expansion, 
                              stride=1, 
                              padding=1),
                    nn.BatchNorm2d(num_features=self.inner_width*self.expansion)
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(kernel_size=3, 
                              in_channels=in_channels, 
                              out_channels=self.inner_width*self.expansion, 
                              stride=1, 
                              padding=1),
                    nn.BatchNorm2d(num_features=self.inner_width*self.expansion)
                )

    def forward(self, x: torch.Tensor):
        residual = x

        output = self.block(x)

        if self.has_downsample:
            residual = self.downsample(residual)
        output = nn.ReLU()(output + residual)
        return output


class ResNeXt(nn.Module):

    def __init__(self, 
                 block: ResNeXtBlock, 
                 cardinality: int, 
                 width: int, 
                 layers: List[int],
                 num_classes: int=10):
        super().__init__()

        self.cardinality = cardinality
        self.width = width
        self.expansion = 2

        self.layers = layers

        self.layer0 = nn.Sequential(
                nn.Conv2d(kernel_size=3, in_channels=3, out_channels=64, stride=2, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU()
            )

        self.layer1 = self._make_layer(block=block, 
                                       in_channels=64, 
                                       n_layers=self.layers[0], 
                                       stride=1)
        self.layer2 = self._make_layer(block=block, 
                                       in_channels=256, 
                                       n_layers=self.layers[1], 
                                       stride=2)
        self.layer3 = self._make_layer(block=block, 
                                       in_channels=512, 
                                       n_layers=self.layers[2], 
                                       stride=2)
        self.layer4 = self._make_layer(block=block, 
                                       in_channels=1024, 
                                       n_layers=self.layers[3], 
                                       stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * self.expansion, num_classes)

    def _make_layer(self, 
                    block: ResNeXtBlock, 
                    in_channels: int, 
                    n_layers: int,
                    stride: int=1) -> nn.Sequential:

        downsample = (stride != 1) or (in_channels != self.cardinality * self.width * self.expansion)

        layers = []
        layers.append(block(in_channels=in_channels, 
                            cardinality=self.cardinality, 
                            width=self.width,
                            downsample=downsample, 
                            stride=stride))

        in_channels = self.cardinality * self.width * self.expansion

        for i in range(1, n_layers):
            layers.append(block(in_channels=in_channels, 
                                cardinality=self.cardinality, 
                                width=self.width))
        self.width *= 2
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