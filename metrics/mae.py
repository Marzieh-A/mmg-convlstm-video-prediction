"""STAGE 11 - MAE metric."""

from __future__ import annotations

import torch


def mae(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Elementwise mean absolute error. Lower = better."""
    assert preds.shape == targets.shape
    return (preds - targets).abs().mean()