"""STAGE 13 tests - ConvLSTM baseline correctness and comparability."""

from __future__ import annotations

import torch

from baselines.convlstm_baseline import ConvLSTMBaseline
from models.mmg_convlstm import MMGConfig, MMGConvLSTM


def _assert_finite(t: torch.Tensor):
    assert torch.isfinite(t).all(), "NaN or Inf detected"


def test_baseline_shapes_gray():
    torch.manual_seed(0)
    m = ConvLSTMBaseline()
    y = m(torch.randn(2, 10, 1, 64, 64))
    assert y.shape == (2, 10, 1, 64, 64)
    assert y.min() >= 0.0 and y.max() <= 1.0
    _assert_finite(y)


def test_baseline_shapes_rgb():
    torch.manual_seed(0)
    m = ConvLSTMBaseline(in_channels=3)
    y = m(torch.randn(1, 10, 3, 64, 64))
    assert y.shape == (1, 10, 3, 64, 64)
    _assert_finite(y)


def test_baseline_batch_one():
    torch.manual_seed(0)
    m = ConvLSTMBaseline()
    y = m(torch.randn(1, 10, 1, 64, 64))
    assert y.shape == (1, 10, 1, 64, 64)


def test_baseline_has_no_proposed_components():
    """The baseline must not contain encoder/motion/gate modules."""
    m = ConvLSTMBaseline()
    names = [n for n, _ in m.named_modules()]
    assert not any("encoder" in n or "motion" in n or "gate" in n for n in names)


def test_baseline_training_step():
    torch.manual_seed(0)
    m = ConvLSTMBaseline()
    preds = m(torch.rand(2, 10, 1, 64, 64))
    loss = (preds - torch.rand(2, 10, 1, 64, 64)).pow(2).mean()
    loss.backward()
    _assert_finite(loss)
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        _assert_finite(p.grad)


def test_capacity_comparable():
    """Both models' parameter counts are reported (no assertion on which is
    larger - measured in Stage 17 profiling - but they must be same order)."""
    base = sum(p.numel() for p in ConvLSTMBaseline().parameters())
    mmg = sum(p.numel() for p in MMGConvLSTM(MMGConfig(in_channels=1)).parameters())
    ratio = max(base, mmg) / max(1, min(base, mmg))
    print(f"baseline params: {base:,} | MMG params: {mmg:,} | ratio: {ratio:.2f}")
    assert ratio < 5.0   # same order of magnitude, reasonably comparable