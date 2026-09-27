"""STAGE 14 - Ablation variants A/B/C/D sharing identical components.

A: raw frames -> ConvLSTM -> decoder head                    (= baseline)
B: Multi-Scale Encoder -> ConvLSTM -> decoder                (no motion)
C: Encoder -> ConvLSTM + feature residual injection (Z=H+D)  (no gate)
D: Encoder -> Motion-gated ConvLSTM                          (= MMG-ConvLSTM)

All variants use the SAME autoregressive protocol, the SAME decoder style,
and are trained/evaluated under identical conditions (same config, seed,
batch size, epochs, optimizer - enforced in scripts/run_ablation.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.convlstm import ConvLSTM
from models.decoder import MultiScaleDecoder
from models.motion_gated_convlstm import MotionGatedConvLSTM
from models.motion_module import MotionResidual
from models.multiscale_encoder import MultiScaleEncoder


@dataclass
class AblationConfig:
    in_channels: int = 1
    enc_c1: int = 32
    enc_c2: int = 64
    enc_c3: int = 128
    feat_channels: int = 64
    kernel_size: int = 3
    input_len: int = 10
    pred_len: int = 10
    variant: str = "D"          # "A" | "B" | "C" | "D"


class AblationModel(nn.Module):
    def __init__(self, config: AblationConfig | None = None, **overrides):
        super().__init__()
        cfg = config or AblationConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown config key: {k}")
            setattr(cfg, k, v)
        self.cfg = cfg
        assert cfg.variant in ("A", "B", "C", "D"), cfg.variant

        C = cfg.feat_channels
        if cfg.variant == "A":
            self.rnn = ConvLSTM(cfg.in_channels, C, cfg.kernel_size)
        else:
            self.encoder = MultiScaleEncoder(
                cfg.in_channels, cfg.enc_c1, cfg.enc_c2, cfg.enc_c3, C)
            if cfg.variant in ("B", "C"):
                # B: plain ConvLSTM. C: plain ConvLSTM + manual residual
                # injection in forward (NO gate anywhere).
                self.rnn = ConvLSTM(C, C, cfg.kernel_size)
                if cfg.variant == "C":
                    self.motion = MotionResidual()
            else:  # D: full motion-gated ConvLSTM
                self.motion = MotionResidual()
                self.rnn = MotionGatedConvLSTM(C, C, cfg.kernel_size)

        self.decoder = MultiScaleDecoder(C, out_channels=cfg.in_channels)

    def _encode_seq(self, inputs: torch.Tensor) -> torch.Tensor:
        feats = [self.encoder(inputs[:, t]) for t in range(inputs.shape[1])]
        return torch.stack(feats, dim=1)

    def forward(self, inputs: torch.Tensor, pred_len: int | None = None):
        T_pred = pred_len if pred_len is not None else self.cfg.pred_len
        assert inputs.ndim == 5
        v = self.cfg.variant

        if v == "A":
            outputs, (h, c) = self.rnn(inputs)
            z_last = outputs[:, -1]
            f_prev = None
        else:
            feats = self._encode_seq(inputs)
            if v in ("C", "D"):
                resids = self.motion(feats)
            else:
                resids = None

            if v == "D":
                fused, _, (h, c) = self.rnn(feats, resids)
            else:
                fused, (h, c) = self.rnn(feats)
            z_last = fused[:, -1]
            f_prev = feats[:, -1]

        preds = [self.decoder(z_last)]
        x_next = preds[0].detach()

        for _ in range(1, T_pred):
            with torch.no_grad():
                if v == "A":
                    _, (h, c) = self.rnn(x_next.unsqueeze(1), (h, c))
                    z = h
                else:
                    f_next = self.encoder(x_next)
                    if v == "B":
                        _, (h, c) = self.rnn(f_next.unsqueeze(1), (h, c))
                        z = h
                    elif v == "C":
                        d = f_next - f_prev
                        h, c = self.rnn.cell(f_next, (h, c))
                        z = h + d                      # residual, no gate
                    else:  # D
                        d = f_next - f_prev
                        h, c = self.rnn.cell(f_next, (h, c))
                        g = torch.sigmoid(self.rnn.gate_conv(torch.cat([h, d], dim=1)))
                        z = h + g * d
                    f_prev = f_next.detach()
            x_hat = self.decoder(z)
            preds.append(x_hat)
            x_next = x_hat.detach()

        return torch.stack(preds, dim=1)

    def _make_a_step(self):
        return None

    def _make_enc_step(self, f_last):
        return None