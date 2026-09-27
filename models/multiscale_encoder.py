"""STAGE 3 - Multi-Scale CNN Encoder.

Extracts spatial features at three scales (64x64, 32x32, 16x16) and fuses
them into a single feature map suitable for the ConvLSTM stage.

Design (lightweight, 12 GB GPU friendly):
    Scale 1: 64x64,  C1 channels (full resolution detail)
    Scale 2: 32x32,  C2 channels
    Scale 3: 16x16,  C3 channels
Fusion: upsample scales 2,3 back to 64x64, concat with scale 1,
then a 1x1 conv projects to `out_channels` (ConvLSTM input).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv_blocks import make_conv_block, make_downsample_block


class MultiScaleEncoder(nn.Module):
    def __init__(self, in_channels: int = 1,
                 c1: int = 32, c2: int = 64, c3: int = 128,
                 out_channels: int = 64):
        super().__init__()
        self.out_channels = out_channels

        # Scale 1: full resolution, two convs
        self.s1 = nn.Sequential(
            make_conv_block(in_channels, c1),
            make_conv_block(c1, c1),
        )
        # Scale 2: downsample x2 then conv
        self.s2 = nn.Sequential(
            make_downsample_block(c1, c2),   # 64 -> 32
            make_conv_block(c2, c2),
        )
        # Scale 3: downsample again
        self.s3 = nn.Sequential(
            make_downsample_block(c2, c3),   # 32 -> 16
            make_conv_block(c3, c3),
        )

        # Fusion: concat( s1(64x64), up(s2), up(s3) ) -> 1x1 conv projection
        fusion_in = c1 + c2 + c3
        self.fuse = nn.Conv2d(fusion_in, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] with H=W=64 -> returns [B, out_channels, 64, 64]."""
        assert x.ndim == 4, f"expected [B, C, H, W], got {tuple(x.shape)}"

        f1 = self.s1(x)          # [B, c1, 64, 64]
        f2 = self.s2(f1)         # [B, c2, 32, 32]
        f3 = self.s3(f2)         # [B, c3, 16, 16]

        f2_up = torch.nn.functional.interpolate(
            f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f3_up = torch.nn.functional.interpolate(
            f3, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        fused = torch.cat([f1, f2_up, f3_up], dim=1)  # [B, c1+c2+c3, 64, 64]
        return self.fuse(fused)                        # [B, out_ch, 64, 64]