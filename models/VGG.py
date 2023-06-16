import torch
import torch.nn as nn


class VGG_Block(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 n_convs: int, 
                 stride: int=1, 
                 padding: int=1, 
                 use_batch_norm: bool=False) -> None:
        super().__init__()
        self.layers = [
            nn.Conv2d(
                kernel_size=3, 
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=stride, 
                padding=padding, 
                bias=False
            )
        ]
        if use_batch_norm:
            self.layers.append(nn.BatchNorm2d(num_features=out_channels))
        self.layers.append(nn.ReLU())

        for i in range(1, n_convs):
            self.layers.append(
                nn.Conv2d(
                    kernel_size=3, 
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    stride=stride, 
                    padding=padding, 
                    bias=False
                )
            )
            if use_batch_norm:
                self.layers.append(nn.BatchNorm2d(num_features=out_channels))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGG(nn.Module):
    def __init__(self, 
                 vgg_block: VGG_Block,
                 n_layers: int, 
                 num_classes: int, 
                 use_batch_norm: bool=False) -> None:
        super().__init__()

        if n_layers == 11:
            self.convs_per_block = [1, 1, 2, 2, 2]
        elif n_layers == 13:
            self.convs_per_block = [2, 2, 2, 2, 2]
        elif n_layers == 16:
            self.convs_per_block = [2, 2, 3, 3, 3]
        elif n_layers == 19:
            self.convs_per_block = [2, 2, 4, 4, 4]
        else:
            raise ValueError('n_layers must be 11, 13, 16 or 19')

        self.conv1 = vgg_block(in_channels=3, 
                               out_channels=64, 
                               n_convs=self.convs_per_block[0], 
                               use_batch_norm=use_batch_norm)
        self.conv2 = vgg_block(in_channels=64, 
                               out_channels=128, 
                               n_convs=self.convs_per_block[1], 
                               use_batch_norm=use_batch_norm)
        self.conv3 = vgg_block(in_channels=128, 
                               out_channels=256, 
                               n_convs=self.convs_per_block[2], 
                               use_batch_norm=use_batch_norm)
        self.conv4 = vgg_block(in_channels=256, 
                               out_channels=512, 
                               n_convs=self.convs_per_block[3], 
                               use_batch_norm=use_batch_norm)
        self.conv5 = vgg_block(in_channels=512, 
                               out_channels=512, 
                               n_convs=self.convs_per_block[4], 
                               use_batch_norm=use_batch_norm)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)

        output = output.view(output.size(0), -1)

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output