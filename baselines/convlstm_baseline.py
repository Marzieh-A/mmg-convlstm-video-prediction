"""STAGE 13 - Standard ConvLSTM video-prediction baseline.

A genuine baseline: raw frames -> ConvLSTM (hidden=64) at full 64x64
resolution -> same-style conv refinement head -> sigmoid frame.
NO multi-scale encoder, NO motion residual, NO motion gate.
Capacity is kept approximately comparable to MMG-ConvLSTM so that the
ablation comparison is fair.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.convlstm import ConvLSTM
from models.decoder import MultiScaleDecoder


@dataclass
class BaselineConfig:
    in_channels: int = 1
    hidden_channels: int = 64
    kernel_size: int = 3
    input_len: int = 10
    pred_len: int = 10


class ConvLSTMBaseline(nn.Module):
    def __init__(self, config: BaselineConfig | None = None, **overrides):
        super().__init__()
        cfg = config or BaselineConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown config key: {k}")
            setattr(cfg, k, v)
        self.cfg = cfg

        self.rnn = ConvLSTM(cfg.in_channels, cfg.hidden_channels, cfg.kernel_size)
        self.decoder = MultiScaleDecoder(
            in_channels=cfg.hidden_channels, out_channels=cfg.in_channels)

    def forward(self, inputs: torch.Tensor, pred_len: int | None = None):
        """inputs: [B, T_in, C, H, W] -> predictions [B, pred_len, C, H, W].

        Autoregressive, identical protocol to MMG-ConvLSTM:
        encode observed frames, then feed each PREDICTION back as the next
        input (detached - no gradient through the AR chain).
        """
        T_pred = pred_len if pred_len is not None else self.cfg.pred_len
        assert inputs.ndim == 5, f"expected [B, T, C, H, W], got {tuple(inputs.shape)}"
        B, T_in, C, H, W = inputs.shape

        outputs, (h, c) = self.rnn(inputs)       # [B, T_in, Hc, H, W]
        preds = []

        # first future frame from the last observed hidden state
        x_hat = self.decoder(outputs[:, -1])
        preds.append(x_hat)

        for _ in range(1, T_pred):
            with torch.no_grad():
                _, (h, c) = self.rnn(x_hat.detach().unsqueeze(1), (h, c))
            x_hat = self.decoder(h)
            preds.append(x_hat)

        return torch.stack(preds, dim=1)          # [B, pred_len, C, H, W]