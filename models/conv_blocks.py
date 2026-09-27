"""STAGE 3 - Basic convolutional building blocks (no attention, no recurrence)."""

from __future__ import annotations

import torch.nn as nn


def make_conv_block(in_ch: int, out_ch: int, kernel_size: int = 3,
                    stride: int = 1, norm: str = "group") -> nn.Sequential:
    """Conv -> Norm -> SiLU. 'same' padding for odd kernels.

    norm: "group" (GroupNorm - batch-size friendly, safe for small batches),
          "batch" (BatchNorm2d), or "none".
    """
    padding = kernel_size // 2
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)]
    if norm == "group":
        num_groups = 8 if out_ch % 8 == 0 else 4 if out_ch % 4 == 0 else 1
        layers.append(nn.GroupNorm(num_groups, out_ch))
    elif norm == "batch":
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


def make_downsample_block(in_ch: int, out_ch: int, kernel_size: int = 4) -> nn.Sequential:
    """Strided-conv downsampling by a factor of 2 (kernel 4, stride 2, pad 1)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, padding=1),
        nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch),
        nn.SiLU(inplace=True),
    )


def make_upsample_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bilinear upsampling x2 followed by a conv (avoids checkerboard artifacts
    that transposed convs can introduce)."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch),
        nn.SiLU(inplace=True),
    )