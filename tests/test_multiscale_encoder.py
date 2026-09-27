"""STAGE 3 tests - multi-scale encoder shapes, fusion, gradient flow."""

from __future__ import annotations

import torch

from models.multiscale_encoder import MultiScaleEncoder


def _assert_finite(t: torch.Tensor):
    assert torch.isfinite(t).all(), "NaN or Inf detected"


def test_output_shape_moving_mnist():
    torch.manual_seed(0)
    enc = MultiScaleEncoder(in_channels=1, out_channels=64)
    x = torch.randn(2, 1, 64, 64)
    y = enc(x)
    assert y.shape == (2, 64, 64, 64)
    _assert_finite(y)


def test_output_shape_ucf101():
    torch.manual_seed(0)
    enc = MultiScaleEncoder(in_channels=3, out_channels=64)
    x = torch.randn(2, 3, 64, 64)
    y = enc(x)
    assert y.shape == (2, 64, 64, 64)
    _assert_finite(y)


def test_batch_size_one():
    torch.manual_seed(0)
    enc = MultiScaleEncoder(in_channels=1, out_channels=32)
    y = enc(torch.randn(1, 1, 64, 64))
    assert y.shape == (1, 32, 64, 64)


def test_spatial_info_preserved():
    """Encoder must not collapse spatial structure: distinct inputs give
    distinct outputs (no global pooling bottleneck)."""
    torch.manual_seed(0)
    enc = MultiScaleEncoder(in_channels=1, out_channels=16).eval()
    with torch.no_grad():
        a = torch.zeros(1, 1, 64, 64); a[:, :, :32, :] = 1.0
        b = torch.zeros(1, 1, 64, 64); b[:, :, :, :32] = 1.0
        ya, yb = enc(a), enc(b)
    assert not torch.allclose(ya, yb, atol=1e-4)


def test_gradient_flow():
    torch.manual_seed(0)
    enc = MultiScaleEncoder(in_channels=1, out_channels=16)
    x = torch.randn(2, 1, 64, 64, requires_grad=True)
    y = enc(x)
    y.pow(2).mean().backward()
    assert x.grad is not None
    _assert_finite(x.grad)
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        _assert_finite(p.grad)


def test_deterministic_given_weights():
    torch.manual_seed(0)
    enc = MultiScaleEncoder(in_channels=1, out_channels=16).eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(x))