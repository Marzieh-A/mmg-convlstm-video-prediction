"""STAGE 9 (rev2) - Losses per literature evidence.

Base protocol (previous experiments): pure MSE.
New protocol: GDL-L1 following Mathieu et al. (ICLR 2016) - the best
non-adversarial loss in their benchmark, including the multi-frame
prediction setting closest to our protocol:
    L = L1(pred, target) + GDL(pred, target)
with alpha = 1 and equal weights (1.0 each) - the paper's settings;
no invented weights.
GDL (eq. 6 of Mathieu et al.): L1 difference of ABSOLUTE image gradients
between prediction and target - directly penalizes blur.
Documented deviation: the paper sums the GDL term; we take the mean for
loss-scale invariance w.r.t. image size (the term ratio is preserved).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Original protocol loss (kept for reproducibility)."""
    assert predictions.shape == targets.shape
    return (predictions - targets).pow(2).mean()


def l1_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.shape == targets.shape
    return (predictions - targets).abs().mean()


def _abs_grads(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute first-order image gradients along H and W (Mathieu et al.
    eq. 6: simplest possible gradient - neighboring pixel differences -
    chosen by the authors for low training cost)."""
    gx = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs()
    gy = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs()
    return gx, gy


def gdl_loss(predictions: torch.Tensor, targets: torch.Tensor,
             alpha: int = 1) -> torch.Tensor:
    """Gradient Difference Loss (Mathieu et al. 2016, eq. 6), alpha=1."""
    assert predictions.shape == targets.shape
    assert alpha == 1, "only alpha=1 is used (paper's GDL-L1 setting)"
    pgx, pgy = _abs_grads(predictions)
    tgx, tgy = _abs_grads(targets)
    # restore full shape by zero-padding the missing last column/row so the
    # term is a true mean over all pixels
    pgx = F.pad(pgx, (0, 1)); pgy = F.pad(pgy, (0, 0, 0, 1))
    tgx = F.pad(tgx, (0, 1)); tgy = F.pad(tgy, (0, 0, 0, 1))
    return ((pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()) / 2.0


def gdl_l1_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The paper's GDL-L1 combination: L1 + GDL, equal weights (1.0)."""
    return l1_loss(predictions, targets) + gdl_loss(predictions, targets)


def composite_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE + L1 + GDL, equal weights.
    Rationale (documented from our stalled-run diagnosis): on mostly-black
    data, L1+GDL alone has a near-zero-gradient attractor at the dark
    constant prediction; the MSE term anchors gradients away from it.
    Family is still Mathieu et al. (ICLR 2016): Lp + GDL."""
    return mse_loss(predictions, targets) + gdl_l1_loss(predictions, targets)

def mae_metric(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MAE - evaluation metric, not a training loss."""
    assert predictions.shape == targets.shape
    return (predictions - targets).abs().mean()

class ReconstructionLoss:
    """kind: 'mse' (base) | 'gdl_l1' | 'composite' (mse+l1+gdl)."""

    def __init__(self, kind: str = "composite"):
        assert kind in ("mse", "gdl_l1", "composite"), kind
        self.kind = kind

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.kind == "mse":
            return mse_loss(predictions, targets)
        if self.kind == "gdl_l1":
            return gdl_l1_loss(predictions, targets)
        return composite_loss(predictions, targets)

class WarmupLoss:
    """Two-phase protocol:
    Phase 1 (epochs < warmup_epochs): pure MSE - proven to learn the
      dynamics on this data (our MSE ablation runs).
    Phase 2: composite MSE+L1+GDL for sharpening.
    Rationale (documented from our diagnosis): composite-from-scratch
      collapses to the constant gray prediction on mostly-black data;
      MSE warmup escapes that attractor first."""

    def __init__(self, warmup_epochs: int = 8):
        self.warmup_epochs = warmup_epochs
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __call__(self, predictions: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        if self.epoch < self.warmup_epochs:
            return mse_loss(predictions, targets)
        return composite_loss(predictions, targets)