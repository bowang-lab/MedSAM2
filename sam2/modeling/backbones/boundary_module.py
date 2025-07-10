from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryHead(nn.Module):
    """
    A lightweight CNN to extract boundary-specific features from the raw image.
    """

    def __init__(self, input_channels: int = 3, output_channels: int = 64):
        super().__init__()
        # A simple series of convolutional blocks
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        # A final 1x1 convolution to set the desired output channel dimension
        self.conv3 = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Boundary feature map.
        """
        # The main image encoder in SAM usually downsamples by 4x for the highest-res FPN feature.
        # We need to match that. Let's assume input image is 1024x1024, and FPN level 0 is 256x256.
        # We can use a stride-2 convolution or max pooling to downsample.
        # Let's check the input size to be robust.
        # A simple downsampling by 4x.
        target_size = (x.shape[-2] // 4, x.shape[-1] // 4)

        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x
