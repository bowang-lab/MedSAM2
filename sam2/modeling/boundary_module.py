import torch
import torch.nn as nn
import torch.nn.functional as F


# class BoundaryHead(nn.Module):
#     """
#     A head to extract boundary-specific features from a high-resolution FPN feature map,
#     and then fuse them back into the original feature map.
#     """

#     def __init__(self, fpn_in_channels: int, boundary_out_channels: int = 64):
#         """
#         Args:
#             fpn_in_channels (int): The number of channels of the input FPN feature map.
#                                    This will be 32 based on your logs for fpn[0].
#             boundary_out_channels (int): The number of channels for the boundary-specific feature map.
#         """
#         super().__init__()
#         # A simple series of convolutional blocks to process the FPN feature
#         # Note: The input channel is now dynamic (fpn_in_channels)
#         self.conv1 = nn.Conv2d(fpn_in_channels, 64, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.ReLU()

#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.relu2 = nn.ReLU()

#         # A final 1x1 convolution to create the boundary logits/features
#         # This can be used for a dedicated boundary loss
#         self.boundary_conv = nn.Conv2d(64, boundary_out_channels, kernel_size=1)

#         # This 1x1 conv will fuse the original FPN features and the new boundary features
#         # It takes concatenated features as input.
#         self.fusion_conv = nn.Conv2d(
#             in_channels=fpn_in_channels + boundary_out_channels,
#             out_channels=fpn_in_channels,  # Fuse back to the original FPN channel dimension
#             kernel_size=1,
#         )

#     def forward(self, fpn_feature: torch.Tensor):
#         """
#         Args:
#             fpn_feature (torch.Tensor): The high-resolution feature map from the encoder's FPN.
#                                         Expected shape: (B, fpn_in_channels, H, W).

#         Returns:
#             A tuple containing:
#             - enhanced_feature (torch.Tensor): The FPN feature enhanced with boundary information.
#                                                Shape: (B, fpn_in_channels, H, W).
#             - boundary_logits (torch.Tensor): The extracted boundary features, for loss calculation.
#                                               Shape: (B, boundary_out_channels, H, W).
#         """
#         # 1. We no longer need to downsample, as we are operating on features directly.
#         # Pass the FPN feature through our small CNN.
#         x = self.relu1(self.bn1(self.conv1(fpn_feature)))
#         x = self.relu2(self.bn2(self.conv2(x)))

#         # 2. Generate the boundary-specific features (logits).
#         boundary_logits = self.boundary_conv(x)

#         # 3. Fuse the boundary features with the original FPN feature.
#         # Concatenate along the channel dimension.
#         fusion_input = torch.cat([fpn_feature, boundary_logits], dim=1)

#         # Apply the fusion convolution.
#         enhanced_feature = self.fusion_conv(fusion_input)

#         return enhanced_feature, boundary_logits


class BoundaryHead(nn.Module):
    def __init__(self, fpn_in_channels: int, boundary_out_channels: int = 64):  # boundary_out_channels is now just for intermediate features
        super().__init__()
        # ... (conv1, bn1, relu1, conv2, bn2, relu2 are unchanged) ...
        self.conv1 = nn.Conv2d(fpn_in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)  # Add a new block
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        # --- MODIFICATION HERE ---
        # This convolution now produces the final single-channel logit for loss calculation.
        self.boundary_conv = nn.Conv2d(64, 1, kernel_size=1)  # Output channel is now 1

        # The fusion logic remains the same, but it should probably use the intermediate features
        # Let's assume for now the main goal is the loss.
        # Fusion part might need adjustment if you want to enhance the main features.
        self.fusion_conv = nn.Conv2d(
            in_channels=fpn_in_channels + 1,  # Fusing with the 1-channel logit
            out_channels=fpn_in_channels,
            kernel_size=1,
        )

    def forward(self, fpn_feature: torch.Tensor):
        x = self.relu1(self.bn1(self.conv1(fpn_feature)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        # This is now a single-channel logit map
        boundary_logits = self.boundary_conv(x)

        # Fusion logic
        fusion_input = torch.cat([fpn_feature, boundary_logits], dim=1)
        enhanced_feature = self.fusion_conv(fusion_input)

        return enhanced_feature, boundary_logits
