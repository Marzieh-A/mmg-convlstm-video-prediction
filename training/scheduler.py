"""STAGE 10 - LR scheduler factory (conservative defaults)."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def make_scheduler(optimizer: Optimizer, kind: str, epochs: int, lr: float):
    if kind == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    if kind == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    if kind == "none":
        return None
    raise ValueError(f"unknown scheduler: {kind}")