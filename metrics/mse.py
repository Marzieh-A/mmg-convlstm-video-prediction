"""STAGE 11 - MSE metric (aggregation over a batch is the caller's job)."""

from __future__ import annotations

import torch


def mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Elementwise mean squared error over the full tensor. Lower = better."""
    assert preds.shape == targets.shape
    return (preds - targets).pow(2).mean()