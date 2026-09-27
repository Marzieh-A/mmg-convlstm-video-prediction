"""
STAGE 4 - Feature-space temporal motion residual.

Given encoded features F_t, define:

    D_t = F_t - F_{t-1}

with:

    D_1 = zeros_like(F_1)

This provides an explicit feature-space representation
of temporal change.

No optical flow, no pretrained motion network,
and no complex motion estimation are used.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MotionResidual(nn.Module):
    """
    Computes feature-space temporal residuals over a sequence.

    Input:
        features: Tensor of shape [B, T, C_f, H, W]

    Output:
        residuals: Tensor of shape [B, T, C_f, H, W]

        The residual at the first timestep is zero:
            residuals[:, 0] = 0
    """

    def __init__(self) -> None:
        super().__init__()

        # This module has no learnable parameters by design.

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal feature differences.

        Args:
            features:
                Feature sequence with shape [B, T, C_f, H, W].

        Returns:
            Temporal residuals with shape [B, T, C_f, H, W].
        """

        if features.ndim != 5:
            raise ValueError(
                "Expected features with shape [B, T, C_f, H, W], "
                f"but got {tuple(features.shape)}."
            )

        # Temporal difference:
        # D_t = F_t - F_{t-1}
        #
        # Shape:
        # [B, T-1, C_f, H, W]
        diffs = features[:, 1:] - features[:, :-1]

        # The first timestep has no previous frame,
        # therefore its temporal residual is initialized to zero.
        #
        # Shape:
        # [B, 1, C_f, H, W]
        first = torch.zeros_like(features[:, 0:1])

        # Concatenate the zero residual with the temporal differences.
        #
        # Final shape:
        # [B, T, C_f, H, W]
        residuals = torch.cat([first, diffs], dim=1)

        return residuals