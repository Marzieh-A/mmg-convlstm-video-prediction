"""STAGE 7 - Multi-Scale CNN Decoder (corrected).

The encoder fuses all scales back at 64x64, so the ConvLSTM representation
Z_t lives at 64x64. The decoder therefore REFINES at full resolution:
a few residual-style conv blocks, then a final Conv2d to C channels with
sigmoid (frames are normalized to [0, 1]). No upsampling is needed because
there is no spatial bottleneck; supports C=1 and C=3 unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv_blocks import make_conv_block


class MultiScaleDecoder(nn.Module):
    def __init__(self, in_channels: int = 64,
                 d2: int = 64, d1: int = 32,
                 out_channels: int = 1):
        super().__init__()
        # refinement blocks at the full 64x64 resolution
        self.refine = nn.Sequential(
            make_conv_block(in_channels, d2),
            make_conv_block(d2, d2),
            make_conv_block(d2, d1),
        )
        self.head = nn.Conv2d(d1, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, in_channels, H, W] -> frame [B, out_channels, H, W]."""
        assert z.ndim == 4, f"expected [B, C, H, W], got {tuple(z.shape)}"
        x = self.refine(z)
        return torch.sigmoid(self.head(x))