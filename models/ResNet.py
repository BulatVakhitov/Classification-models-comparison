from typing import Tuple, Dict, List

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 num_layers: int,
                 use_resnet_d: bool,
                 downsample: bool=False, 
                 stride: int=1, 
                 downsample_kernel_size: int=1, 
                 downsample_padding: int=0) -> None:
        super().__init__()
        
        self.num_layers = num_layers
        self.has_downsample = downsample
        
        if self.num_layers <= 34:
            self.expansion = 1
        else:
            self.expansion = 4
        
        if self.num_layers <= 34:
            self.conv3x3_1 = nn.Sequential(
                nn.Conv2d(kernel_size=3, 
                          in_channels=in_channels, 
                          out_channels=out_channels, 
                          stride=stride, 
                          padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            )
            self.conv3x3_2 = nn.Sequential(
                nn.Conv2d(kernel_size=3, 
                          in_channels=out_channels, 
                          out_channels=out_channels, 
                          stride=1, 
                          padding=1),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.conv1x1_1 = nn.Sequential(
                nn.Conv2d(kernel_size=1, 
                          in_channels=in_channels, 
                          out_channels=out_channels, 
                          stride=1, 
                          padding=0),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            )
            self.conv3x3_2 = nn.Sequential(
                nn.Conv2d(kernel_size=3, 
                          in_channels=out_channels, 
                          out_channels=out_channels, 
                          stride=stride, 
                          padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            )
            self.conv1x1_3 = nn.Sequential(
                nn.Conv2d(kernel_size=1, 
                          in_channels=out_channels, 
                          out_channels=out_channels*self.expansion, 
                          stride=1, 
                          padding=0),
                nn.BatchNorm2d(num_features=out_channels*self.expansion)
            )
        
        if self.has_downsample:  # TODO починить эту штуку, чтобы не было много if 
            if use_resnet_d:
                if stride == 2: # layer1 имеет stride=1, из-за этого все ломается
                    self.downsample = nn.Sequential(
                        nn.AvgPool2d(kernel_size=2, 
                                     stride=2),
                        nn.Conv2d(kernel_size=downsample_kernel_size, 
                                  in_channels=in_channels, 
                                  out_channels=out_channels*self.expansion, 
                                  stride=1, 
                                  padding=downsample_padding),
                        nn.BatchNorm2d(num_features=out_channels*self.expansion)
                    )
                else:
                    self.downsample = nn.Sequential(
                        nn.Conv2d(kernel_size=downsample_kernel_size, 
                                  in_channels=in_channels, 
                                  out_channels=out_channels*self.expansion, 
                                  stride=1, 
                                  padding=downsample_padding),
                        nn.BatchNorm2d(num_features=out_channels*self.expansion)
                    )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(kernel_size=downsample_kernel_size, 
                              in_channels=in_channels, 
                              out_channels=out_channels*self.expansion, 
                              stride=stride, 
                              padding=downsample_padding),
                    nn.BatchNorm2d(num_features=out_channels*self.expansion)
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        if self.num_layers <= 34:
            output = self.conv3x3_1(x)
            output = self.conv3x3_2(output)
        else:
            output = self.conv1x1_1(x)
            output = self.conv3x3_2(output)
            output = self.conv1x1_3(output)

        if self.has_downsample:
            residual = self.downsample(residual)
        output = nn.ReLU()(output + residual)
        return output
    
    
class ResNet(nn.Module):
    
    def __init__(self, 
                 res_block: ResBlock, 
                 num_layers: int, 
                 num_classes: int=10, 
                 use_resnet_c: bool=False, 
                 use_resnet_d: bool=False, 
                 downsample_kernel_size: int=1, 
                 downsample_padding: int=0):
        super().__init__()

        self.num_layers = num_layers

        if self.num_layers  <= 34:
            self.expansion = 1
        else:
            self.expansion = 4

        if self.num_layers == 18:
            self.layers = [2, 2, 2, 2]
        elif self.num_layers == 34 or self.num_layers == 50:
            self.layers = [3, 4, 6, 3]
        elif self.num_layers == 101:
            self.layers = [3, 4, 23, 3]
        elif self.num_layers == 152:
            self.layers = [3, 4, 36, 3]

        if use_resnet_c:
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
        else:
            self.layer0 = nn.Sequential(
                nn.Conv2d(kernel_size=7, in_channels=3, out_channels=64, stride=2, padding=3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU()
            )



        self.layer1 = self._make_layer(block=res_block, 
                                       in_channels=64, 
                                       out_channels=64, 
                                       n_layers=self.layers[0], 
                                       stride=1, 
                                       use_resnet_d=use_resnet_d, 
                                       downsample_kernel_size=downsample_kernel_size, 
                                       downsample_padding=downsample_padding)
        self.layer2 = self._make_layer(block=res_block, 
                                       in_channels=64 * self.expansion, 
                                       out_channels=128, 
                                       n_layers=self.layers[1], 
                                       stride=2, 
                                       use_resnet_d=use_resnet_d, 
                                       downsample_kernel_size=downsample_kernel_size, 
                                       downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block=res_block, 
                                       in_channels=128*self.expansion, 
                                       out_channels=256, 
                                       n_layers=self.layers[2], 
                                       stride=2, 
                                       use_resnet_d=use_resnet_d, 
                                       downsample_kernel_size=downsample_kernel_size, 
                                       downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block=res_block, 
                                       in_channels=256*self.expansion, 
                                       out_channels=512, 
                                       n_layers=self.layers[3], 
                                       stride=2, 
                                       use_resnet_d=use_resnet_d, 
                                       downsample_kernel_size=downsample_kernel_size, 
                                       downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)


    def _make_layer(self, 
                    block: ResBlock, 
                    in_channels: int, 
                    out_channels: int, 
                    n_layers: int,
                    use_resnet_d: bool,
                    stride: int=1, 
                    downsample_kernel_size: int=1, 
                    downsample_padding: int=0) -> nn.Sequential:

        downsample = (stride != 1) or (in_channels != out_channels) or (self.num_layers > 34)

        layers = []
        layers.append(block(in_channels=in_channels, 
                            out_channels=out_channels, 
                            num_layers=self.num_layers, 
                            stride=stride, 
                            use_resnet_d=use_resnet_d,
                            downsample=downsample, 
                            downsample_kernel_size=downsample_kernel_size, 
                            downsample_padding=downsample_padding))

        in_channels = out_channels * self.expansion

        for i in range(1, n_layers):
            layers.append(block(in_channels=in_channels, 
                                out_channels=out_channels, 
                                num_layers=self.num_layers, 
                                use_resnet_d=use_resnet_d, 
                                downsample_kernel_size=downsample_kernel_size, 
                                downsample_padding=downsample_padding))

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