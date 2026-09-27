"""STAGE 2 - Standard ConvLSTM cell and layer.

Implements the standard ConvLSTM equations with convolutional operations
(no flattening of spatial dimensions):

    i_t = sigmoid(W_xi * X_t + W_hi * H_{t-1} + b_i)
    f_t = sigmoid(W_xf * X_t + W_hf * H_{t-1} + b_f)
    o_t = sigmoid(W_xo * X_t + W_ho * H_{t-1} + b_o)
    g_t = tanh(   W_xg * X_t + W_hg * H_{t-1} + b_g)
    C_t = f_t * C_{t-1} + i_t * g_t
    H_t = o_t * tanh(C_t)

Temporal convention across the whole project: [B, T, C, H, W].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell.

    Args:
        in_channels: channels of the input tensor X_t.
        hidden_channels: channels of H_t / C_t.
        kernel_size: spatial kernel size for all convolutions.
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2  # 'same' padding for odd kernels

        # One grouped convolution producing the 4 input-gate projections
        # (Xi, Xf, Xo, Xg). Mathematically identical to 4 separate convs.
        self.conv_x = nn.Conv2d(
            in_channels=in_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        # Hidden-state projection; no bias (biases live in conv_x, which is
        # the standard formulation b_i, b_f, b_o, b_g).
        self.conv_h = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def init_state(self, batch_size: int, height: int, width: int,
                   device: torch.device, dtype: torch.dtype):
        h = torch.zeros(batch_size, self.hidden_channels, height, width,
                        device=device, dtype=dtype)
        c = torch.zeros_like(h)
        return h, c

    def forward(self, x: torch.Tensor, state):
        """x: [B, C_in, H, W]; state = (H_{t-1}, C_{t-1})."""
        h_prev, c_prev = state

        combined = self.conv_x(x) + self.conv_h(h_prev)  # [B, 4*Hc, H, W]
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_channels, dim=1)

        i_t = torch.sigmoid(cc_i)
        f_t = torch.sigmoid(cc_f)
        o_t = torch.sigmoid(cc_o)
        g_t = torch.tanh(cc_g)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class ConvLSTM(nn.Module):
    """Multi-step ConvLSTM layer operating on sequences [B, T, C, H, W].

    Returns:
        outputs: [B, T, hidden_channels, H, W] - H_t for every timestep.
        (H_last, C_last): final states, each [B, hidden_channels, H, W].
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x: torch.Tensor, state=None):
        assert x.ndim == 5, f"expected [B, T, C, H, W], got {tuple(x.shape)}"
        B, T, C, H, W = x.shape

        if state is None:
            h, c = self.cell.init_state(B, H, W, x.device, x.dtype)
        else:
            h, c = state

        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # [B, T, Hc, H, W]
        return outputs, (h, c)
