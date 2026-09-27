"""STAGE 11/15 - Horizon-wise metric computation.

Given predictions [B, 10, C, H, W] and targets [B, 10, C, H, W], computes
MSE / MAE / SSIM / LPIPS at each horizon t+1 ... t+10 plus the overall
average. No number is invented - everything is measured here.
"""

from __future__ import annotations

import torch

from metrics.mae import mae
from metrics.mse import mse
from metrics.ssim import ssim


def per_horizon_metrics(preds: torch.Tensor, targets: torch.Tensor,
                        compute_lpips=None) -> dict:
    """compute_lpips: optional callable (injected so tests can skip the
    heavy LPIPS model). Returns {metric: [T values]} plus 'overall_*'."""
    B, T, C, H, W = preds.shape
    out = {"mse": [], "mae": [], "ssim": []}
    if compute_lpips is not None:
        out["lpips"] = []

    for t in range(T):
        p, g = preds[:, t], targets[:, t]
        out["mse"].append(float(mse(p, g)))
        out["mae"].append(float(mae(p, g)))
        out["ssim"].append(float(ssim(p.unsqueeze(1), g.unsqueeze(1))))
        if compute_lpips is not None:
            out["lpips"].append(float(compute_lpips(
                preds[:, t:t + 1], targets[:, t:t + 1])))

    for k in list(out.keys()):
        out[f"overall_{k}"] = sum(out[k]) / T
    return out


def format_horizon_table(m: dict) -> str:
    """Human-readable table: Horizon | MSE | MAE | SSIM | LPIPS."""
    T = len(m["mse"])
    has_lp = "lpips" in m
    header = "Horizon | MSE | MAE | SSIM" + (" | LPIPS" if has_lp else "")
    lines = [header, "-" * len(header)]
    for t in range(T):
        row = (f"t+{t + 1:<3d} | {m['mse'][t]:.6f} | {m['mae'][t]:.6f} | "
               f"{m['ssim'][t]:.6f}")
        if has_lp:
            row += f" | {m['lpips'][t]:.6f}"
        lines.append(row)
    lines.append("-" * len(header))
    row = (f"Overall | {m['overall_mse']:.6f} | {m['overall_mae']:.6f} | "
           f"{m['overall_ssim']:.6f}")
    if has_lp:
        row += f" | {m['overall_lpips']:.6f}"
    lines.append(row)
    return "\n".join(lines)