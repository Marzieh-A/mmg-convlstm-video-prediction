"""STAGE 11 tests - metrics correctness."""

from __future__ import annotations

import torch

from evaluation.horizon_metrics import format_horizon_table, per_horizon_metrics
from metrics.mae import mae
from metrics.mse import mse
from metrics.ssim import ssim


def test_mse_mae_known():
    p = torch.zeros(1, 1, 1, 2, 2)
    t = torch.ones(1, 1, 1, 2, 2)
    assert mse(p, t).item() == 1.0
    assert mae(p, t).item() == 1.0


def test_ssim_identical_is_one():
    x = torch.rand(2, 4, 1, 64, 64)
    s = ssim(x, x)
    assert abs(s.item() - 1.0) < 1e-5


def test_ssim_anticorrelated_is_low():
    """Black-vs-white frames must score far below 1."""
    x = torch.zeros(1, 1, 1, 64, 64)
    y = torch.ones(1, 1, 1, 64, 64)
    assert ssim(x, y).item() < 0.1


def test_ssim_range_and_rgb():
    x = torch.rand(2, 3, 3, 64, 64)
    s = ssim(x, x.clamp(0, 1))
    assert 0.0 <= s.item() <= 1.0


def test_per_horizon_lengths_and_directions():
    torch.manual_seed(0)
    preds = torch.rand(2, 10, 1, 64, 64)
    targets = torch.rand(2, 10, 1, 64, 64)
    m = per_horizon_metrics(preds, targets)   # no LPIPS in unit tests
    for k in ("mse", "mae", "ssim"):
        assert len(m[k]) == 10
        # overall must be a finite float equal to the mean of the 10 values
        overall = m[f"overall_{k}"]
        assert isinstance(overall, float)
        import math
        assert math.isfinite(overall)
        assert abs(overall - sum(m[k]) / 10) < 1e-9
    assert 0.0 <= m["overall_ssim"] <= 1.0


def test_horizon_table_format():
    torch.manual_seed(0)
    m = per_horizon_metrics(torch.rand(1, 4, 1, 32, 32),
                            torch.rand(1, 4, 1, 32, 32))
    table = format_horizon_table(m)
    assert "t+1" in table and "Overall" in table


def test_lpips_domain_and_channels():
    """LPIPS wrapper: identical images -> near-zero; needs the lpips package."""
    try:
        import lpips  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("lpips package not installed")
    from metrics.lpips_metric import lpips_score
    x = torch.rand(2, 2, 1, 64, 64)
    assert lpips_score(x, x).item() < 0.05     # identical -> ~0
    y = torch.rand(2, 2, 1, 64, 64)
    assert lpips_score(x, y).item() > 0.0