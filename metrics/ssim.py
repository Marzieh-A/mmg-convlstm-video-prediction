"""STAGE 11 - SSIM (structural similarity), per-frame.

Standard formulation (Wang et al. 2004): 11x11 Gaussian window,
sigma=1.5, K1=0.01, K2=0.03. Works for C=1 and C=3 (channels handled
by depthwise convolution). Higher = better; [0, 1] for [0, 1] inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_C1, _C2 = 0.01 ** 2, 0.03 ** 2
_WINDOW_SIZE, _SIGMA = 11, 1.5


def _gaussian_kernel(size: int = _WINDOW_SIZE, sigma: float = _SIGMA) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)                      # [size, size]


def ssim(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """preds/targets: [B, T, C, H, W] in [0, 1]. Returns scalar SSIM."""
    assert preds.shape == targets.shape
    assert preds.ndim == 5, "expected [B, T, C, H, W]"

    B, T, C, H, W = preds.shape
    win = _gaussian_kernel().to(preds.device, preds.dtype)
    win = win.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)   # [C, 1, size, size]

    p = preds.reshape(B * T, C, H, W)
    t = targets.reshape(B * T, C, H, W)

    pad = _WINDOW_SIZE // 2
    mu_p = F.conv2d(p, win, padding=pad, groups=C)
    mu_t = F.conv2d(t, win, padding=pad, groups=C)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p2 = F.conv2d(p * p, win, padding=pad, groups=C) - mu_p2
    sigma_t2 = F.conv2d(t * t, win, padding=pad, groups=C) - mu_t2
    sigma_pt = F.conv2d(p * t, win, padding=pad, groups=C) - mu_pt

    # numerical safety: variance is mathematically non-negative; float noise
    # can make it slightly negative, which would zero the SSIM denominator
    sigma_p2 = sigma_p2.clamp(min=0.0)
    sigma_t2 = sigma_t2.clamp(min=0.0)

    ssim_map = ((2 * mu_pt + _C1) * (2 * sigma_pt + _C2)) / \
               ((mu_p2 + mu_t2 + _C1) * (sigma_p2 + sigma_t2 + _C2))
    return ssim_map.mean()