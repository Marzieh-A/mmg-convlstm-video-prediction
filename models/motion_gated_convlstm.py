"""
STAGE 5 - Motion-Gated ConvLSTM
Central proposed mechanism.

For each timestep t, given encoded features F_t and
motion residual D_t:

    H_t, C_t = ConvLSTMCell(F_t, H_{t-1}, C_{t-1})

    G_t = sigmoid(Conv([H_t, D_t]))

    Z_t = H_t + G_t * D_t

where [H_t, D_t] denotes channel-wise concatenation.

The gate G_t is spatially varying (per-pixel) and
is learned using a convolution.

No attention, no softmax, and no additional recurrent
mechanism are used.

Z_t is the motion-aware temporal representation
passed to the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .convlstm import ConvLSTMCell


class MotionGatedConvLSTM(nn.Module):
    """
    ConvLSTM with a learned spatial motion gate.

    The standard ConvLSTM hidden state H_t is fused with
    the feature-space motion residual D_t through a
    learned spatial gate G_t.

    Input:
        features:
            Tensor of shape [B, T, C_f, H, W]

        residuals:
            Tensor of shape [B, T, C_f, H, W]

    Output:
        fused:
            Motion-aware temporal representation with shape
            [B, T, C_h, H, W]

        gates:
            Learned spatial gates with shape
            [B, T, C_h, H, W]

        final_states:
            Tuple (H_last, C_last), each with shape
            [B, C_h, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        # Standard ConvLSTM cell.
        self.cell = ConvLSTMCell(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        # The gate receives the channel-wise concatenation:
        #
        #     [H_t, D_t]
        #
        # Therefore, its input has:
        #
        #     hidden_channels + hidden_channels
        #
        # = 2 * hidden_channels
        #
        # The gate produces one spatial gate per hidden channel.
        self.gate_conv = nn.Conv2d(
            in_channels=2 * hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self,
        features: torch.Tensor,
        residuals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Process the feature sequence using the motion-gated ConvLSTM.

        Args:
            features:
                Encoded feature sequence with shape
                [B, T, C_f, H, W].

            residuals:
                Feature-space temporal residuals with shape
                [B, T, C_f, H, W].

        Returns:
            fused:
                Motion-aware representation with shape
                [B, T, C_h, H, W].

            gates:
                Spatial motion gates with shape
                [B, T, C_h, H, W].

            (h, c):
                Final ConvLSTM hidden and cell states.
        """

        if features.ndim != 5:
            raise ValueError(
                "Expected features with shape [B, T, C_f, H, W], "
                f"but got {tuple(features.shape)}."
            )

        if residuals.shape != features.shape:
            raise ValueError(
                "Residuals must have the same shape as features. "
                f"Got residuals={tuple(residuals.shape)} and "
                f"features={tuple(features.shape)}."
            )

        batch_size, sequence_length, _, height, width = features.shape

        # Initialize ConvLSTM hidden and cell states.
        h, c = self.cell.init_state(
            batch_size,
            height,
            width,
            features.device,
            features.dtype,
        )

        fused_outputs = []
        gates = []

        # Process the sequence one timestep at a time.
        for t in range(sequence_length):

            # Standard ConvLSTM update:
            #
            # H_t, C_t = ConvLSTMCell(
            #     F_t, H_{t-1}, C_{t-1}
            # )
            h, c = self.cell(
                features[:, t],
                (h, c),
            )

            # Concatenate the current hidden state and
            # motion residual along the channel dimension.
            #
            # [H_t, D_t]
            gate_input = torch.cat(
                [h, residuals[:, t]],
                dim=1,
            )

            # Learn a spatially varying gate:
            #
            # G_t = sigmoid(Conv([H_t, D_t]))
            g_t = torch.sigmoid(
                self.gate_conv(gate_input)
            )

            # Motion-aware representation:
            #
            # Z_t = H_t + G_t * D_t
            z_t = h + g_t * residuals[:, t]

            fused_outputs.append(z_t)
            gates.append(g_t)

        # Stack the outputs over the temporal dimension.
        #
        # [B, T, C_h, H, W]
        fused = torch.stack(
            fused_outputs,
            dim=1,
        )

        # [B, T, C_h, H, W]
        gates = torch.stack(
            gates,
            dim=1,
        )

        return fused, gates, (h, c)