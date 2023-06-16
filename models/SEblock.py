import torch
from torch import nn


class SEblock(nn.Module):

    def __init__(self, in_channels: int, squeeze_channels: int, activation=nn.ReLU):
        super().__init__()

        self.in_channels = in_channels

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=in_channels, out_features=squeeze_channels)
        self.activation = activation(inplace=True)
        self.fc2 = nn.Linear(in_features=squeeze_channels, out_features=in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:torch.Tensor):
        output = self.global_avg_pool(x)
        output = output.reshape(output.size(0), self.in_channels)
        output = self.fc1(output)
        output = self.activation(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        output = output.reshape((output.size(0), self.in_channels, 1, 1))

        x = torch.mul(x, output)
        return x