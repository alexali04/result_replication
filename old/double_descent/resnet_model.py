"""
Implementation of Resnet18 model

Deep Double Descent: Where Bigger Models and More Data Hurt
https://arxiv.org/abs/1912.02292
"""


import torch
import torch.nn as nn
from torchvision.models import resnet18

class BasicBlock(nn.Module):
    """
    I wanted to implement a Resnet18 myself

    BasicBlock:
        - Convolution1, BatchNorm, ReLU
        - Convolution2, BatchNorm, Residual
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        stride=1, 
        padding=1
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding, biase=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


