"""STAGE 6+8 - MMG-ConvLSTM: complete model with autoregressive generation.

Pipeline per timestep:
    X_t -> MultiScaleEncoder -> F_t
    D_t = F_t - F_{t-1}  (first frame: zero)
    ConvLSTMCell(F_t, H_{t-1}, C_{t-1}) -> H_t
    G_t = sigmoid(Conv([H_t, D_t]));  Z_t = H_t + G_t * D_t
    MultiScaleDecoder(Z_t) -> X_hat_t

Generation protocol (STAGE 6):
    - Observed frames X_1..X_10 are encoded sequentially.
    - Then X_hat_11 is decoded from the last state; X_hat_11 is re-encoded
      and fed as the next input, and so on until X_hat_20.
    - Evaluation is FULLY autoregressive: no ground-truth future frame is
      ever fed during inference.
    - Training defaults to pure autoregressive prediction on the observed
      context (teacher forcing is NOT implemented unless later requested
      with explicit configuration).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .decoder import MultiScaleDecoder
from .motion_gated_convlstm import MotionGatedConvLSTM
from .motion_module import MotionResidual
from .multiscale_encoder import MultiScaleEncoder


@dataclass
class MMGConfig:
    in_channels: int = 1          # 1 for Moving MNIST, 3 for UCF101
    enc_c1: int = 32
    enc_c2: int = 64
    enc_c3: int = 128
    feat_channels: int = 64       # encoder output / ConvLSTM hidden / decoder input
    kernel_size: int = 3
    input_len: int = 10
    pred_len: int = 10


class MMGConvLSTM(nn.Module):
    def __init__(self, config: MMGConfig | None = None, **overrides):
        super().__init__()
        cfg = config or MMGConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown config key: {k}")
            setattr(cfg, k, v)
        self.cfg = cfg

        self.encoder = MultiScaleEncoder(
            in_channels=cfg.in_channels,
            c1=cfg.enc_c1, c2=cfg.enc_c2, c3=cfg.enc_c3,
            out_channels=cfg.feat_channels,
        )
        self.motion = MotionResidual()          # parameter-free
        self.rnn = MotionGatedConvLSTM(
            in_channels=cfg.feat_channels,
            hidden_channels=cfg.feat_channels,
            kernel_size=cfg.kernel_size,
        )
        self.decoder = MultiScaleDecoder(
            in_channels=cfg.feat_channels,
            out_channels=cfg.in_channels,
        )

    # ---- single-step helpers ----

    def _encode_step(self, x_t: torch.Tensor) -> torch.Tensor:
        """x_t: [B, C, H, W] -> F_t: [B, Cf, H, W]. No grad graph kept
        for re-encoded predictions (detached inside the AR loop)."""
        return self.encoder(x_t)

    def forward(self, inputs: torch.Tensor, pred_len: int | None = None,
                return_intermediates: bool = False):
        """inputs: [B, T_in, C, H, W] observed frames.

        Returns predictions [B, pred_len, C, H, W].
        If return_intermediates=True (debug only), also returns a dict with
        the last fused feature Z_last and final RNN states.
        """
        cfg = self.cfg
        T_pred = pred_len if pred_len is not None else cfg.pred_len
        assert inputs.ndim == 5, f"expected [B, T, C, H, W], got {tuple(inputs.shape)}"
        B, T_in, C, H, W = inputs.shape

        # ---- 1) encode observed frames sequentially ----
        feats = []
        for t in range(T_in):
            feats.append(self.encoder(inputs[:, t]))
        feats = torch.stack(feats, dim=1)            # [B, T_in, Cf, H, W]

        # ---- 2) motion residuals over the observed window ----
        resids = self.motion(feats)                  # [B, T_in, Cf, H, W]

        # ---- 3) run motion-gated ConvLSTM over the observed window ----
        fused, _, (h, c) = self.rnn(feats, resids)   # fused: [B, T_in, Cf, H, W]
        z_last = fused[:, -1]                        # [B, Cf, H, W]

        # ---- 4) autoregressive future generation ----
        preds = []
        x_next = None
        f_prev = feats[:, -1]                        # F_{T_in} for the first residual
        for t in range(T_pred):
            if t == 0:
                # first prediction comes from the last observed state
                x_hat = self.decoder(z_last)
            else:
                # re-encode the previous PREDICTION (detached: no gradient
                # flows through the AR chain - memory-safe on 12 GB)
                with torch.no_grad():
                    f_next = self.encoder(x_next)
                d_next = f_next - f_prev             # D_{T_in + t}
                h, c = self.rnn.cell(f_next, (h, c))
                g_t = torch.sigmoid(self.rnn.gate_conv(torch.cat([h, d_next], dim=1)))
                z_t = h + g_t * d_next
                x_hat = self.decoder(z_t)
                f_prev = f_next.detach()
            preds.append(x_hat)
            x_next = x_hat.detach()                  # prediction becomes next input

        predictions = torch.stack(preds, dim=1)      # [B, pred_len, C, H, W]

        if return_intermediates:
            info = {"h": h, "c": c, "z_last": z_last}
            return predictions, info
        return predictions