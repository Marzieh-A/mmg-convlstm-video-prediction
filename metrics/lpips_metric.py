"""STAGE 11 - LPIPS wrapper.

Rules honored exactly:
  - LPIPS expects 3-channel input in [-1, 1].
  - Moving MNIST (C=1): replicate the single channel to 3 channels ONLY for
    the LPIPS computation; the model/data stay 1-channel.
  - UCF101 (C=3): used directly.
  - [0, 1] inputs are remapped to [-1, 1] before scoring - no silent,
    invalid LPIPS values.
"""

from __future__ import annotations

import torch

_lpips_model = None


def _get_model(device: torch.device):
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device)
        _lpips_model.eval()
        for p in _lpips_model.parameters():
            p.requires_grad_(False)
    return _lpips_model


def _to_lpips_domain(x: torch.Tensor) -> torch.Tensor:
    """[B, T, C, H, W] in [0, 1] -> [B*T, 3, H, W] in [-1, 1]."""
    B, T, C, H, W = x.shape
    assert C in (1, 3), f"unsupported channel count {C}"
    x = x.reshape(B * T, C, H, W)
    if C == 1:
        x = x.repeat(1, 3, 1, 1)         # grayscale -> 3 identical channels
    return x * 2.0 - 1.0                 # [0,1] -> [-1,1]


@torch.no_grad()
def lpips_score(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean LPIPS over all frames. Lower = better."""
    model = _get_model(preds.device)
    p = _to_lpips_domain(preds)
    t = _to_lpips_domain(targets)
    return model(p, t).mean()