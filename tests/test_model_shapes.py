"""STAGE 6+7 tests - decoder shapes and autoregressive generation protocol."""

from __future__ import annotations

import torch

from models.decoder import MultiScaleDecoder
from models.mmg_convlstm import MMGConfig, MMGConvLSTM


def _assert_finite(t: torch.Tensor):
    assert torch.isfinite(t).all(), "NaN or Inf detected"


# ---------------- STAGE 7: decoder ----------------

def test_decoder_gray():
    torch.manual_seed(0)
    dec = MultiScaleDecoder(in_channels=64, out_channels=1)
    z = torch.randn(2, 64, 64, 64)
    y = dec(z)
    assert y.shape == (2, 1, 64, 64)
    assert y.min() >= 0.0 and y.max() <= 1.0   # sigmoid
    _assert_finite(y)


def test_decoder_rgb():
    torch.manual_seed(0)
    dec = MultiScaleDecoder(in_channels=64, out_channels=3)
    y = dec(torch.randn(2, 64, 64, 64))
    assert y.shape == (2, 3, 64, 64)
    _assert_finite(y)


def test_decoder_gradient_flow():
    torch.manual_seed(0)
    dec = MultiScaleDecoder(in_channels=16, out_channels=1)
    z = torch.randn(2, 16, 32, 32, requires_grad=True)
    dec(z).pow(2).mean().backward()
    assert z.grad is not None
    _assert_finite(z.grad)


# ---------------- STAGE 6: full model, autoregressive ----------------

def test_full_model_moving_mnist_shapes():
    torch.manual_seed(0)
    model = MMGConvLSTM(MMGConfig(in_channels=1))
    x = torch.randn(2, 10, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 10, 1, 64, 64)
    assert y.min() >= 0.0 and y.max() <= 1.0
    _assert_finite(y)


def test_full_model_ucf101_shapes():
    torch.manual_seed(0)
    model = MMGConvLSTM(MMGConfig(in_channels=3))
    x = torch.randn(2, 10, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 10, 3, 64, 64)
    _assert_finite(y)


def test_full_model_batch_one():
    torch.manual_seed(0)
    model = MMGConvLSTM(MMGConfig(in_channels=1))
    y = model(torch.randn(1, 10, 1, 64, 64))
    assert y.shape == (1, 10, 1, 64, 64)
    _assert_finite(y)


def test_predictions_are_autoregressive():
    """Changing a FUTURE ground-truth frame must not exist in the API at all;
    instead we verify the AR chain: perturbing the last observed frame must
    change the first prediction, and the effect must propagate to t+2."""
    torch.manual_seed(0)
    model = MMGConvLSTM(MMGConfig(in_channels=1)).eval()
    x = torch.rand(1, 10, 1, 64, 64)
    with torch.no_grad():
        y1 = model(x)
        x2 = x.clone()
        x2[:, -1] += 0.5                          # perturb last observed frame
        x2 = x2.clamp(0, 1)
        y2 = model(x2)
    assert not torch.allclose(y1[:, 0], y2[:, 0], atol=1e-5)
    assert not torch.allclose(y1[:, 1], y2[:, 1], atol=1e-5)


def test_training_step_backward():
    """One full training step: forward, MSE loss, backward - finite grads."""
    torch.manual_seed(0)
    model = MMGConvLSTM(MMGConfig(in_channels=1))
    x = torch.rand(2, 10, 1, 64, 64)
    target = torch.rand(2, 10, 1, 64, 64)
    preds = model(x)
    loss = (preds - target).pow(2).mean()
    loss.backward()
    _assert_finite(loss)
    n_grads = 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        _assert_finite(p.grad)
        n_grads += 1
    assert n_grads > 0


def test_deterministic_eval():
    torch.manual_seed(0)
    model = MMGConvLSTM(MMGConfig(in_channels=1)).eval()
    x = torch.rand(1, 10, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))