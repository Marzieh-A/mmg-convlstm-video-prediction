"""STAGE 2 tests - ConvLSTM correctness, shapes, gradients, stability."""

from __future__ import annotations

import torch

from models.convlstm import ConvLSTM, ConvLSTMCell


def _assert_finite(t: torch.Tensor):
    assert torch.isfinite(t).all(), "NaN or Inf detected in tensor"


def test_cell_single_step_shapes():
    torch.manual_seed(0)
    cell = ConvLSTMCell(in_channels=1, hidden_channels=16, kernel_size=3)
    x = torch.randn(2, 1, 64, 64)
    h, c = cell.init_state(2, 64, 64, x.device, x.dtype)
    h_t, c_t = cell(x, (h, c))
    assert h_t.shape == (2, 16, 64, 64)
    assert c_t.shape == (2, 16, 64, 64)
    _assert_finite(h_t)
    _assert_finite(c_t)


def test_cell_zero_state_output_bounded():
    """With zero states, H_t = o * tanh(...) must stay in (-1, 1)."""
    torch.manual_seed(0)
    cell = ConvLSTMCell(in_channels=1, hidden_channels=8, kernel_size=3)
    x = torch.randn(2, 1, 32, 32)
    h, c = cell.init_state(2, 32, 32, x.device, x.dtype)
    h_t, c_t = cell(x, (h, c))
    assert h_t.abs().max() < 1.0
    _assert_finite(c_t)


def test_layer_sequence_output_shape():
    torch.manual_seed(0)
    layer = ConvLSTM(in_channels=1, hidden_channels=16, kernel_size=3)
    x = torch.randn(2, 10, 1, 64, 64)  # [B, T, C, H, W]
    outputs, (h_last, c_last) = layer(x)
    assert outputs.shape == (2, 10, 16, 64, 64)
    assert h_last.shape == (2, 16, 64, 64)
    assert c_last.shape == (2, 16, 64, 64)
    _assert_finite(outputs)


def test_arbitrary_spatial_dims():
    """Convolutional cell must work on any H, W."""
    torch.manual_seed(0)
    layer = ConvLSTM(in_channels=3, hidden_channels=8, kernel_size=3)
    x = torch.randn(2, 4, 3, 37, 51)  # non-square, non-64
    outputs, _ = layer(x)
    assert outputs.shape == (2, 4, 8, 37, 51)
    _assert_finite(outputs)


def test_gradient_flow_and_backward():
    torch.manual_seed(0)
    layer = ConvLSTM(in_channels=1, hidden_channels=8, kernel_size=3)
    x = torch.randn(2, 6, 1, 32, 32, requires_grad=True)
    outputs, _ = layer(x)
    loss = outputs.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    _assert_finite(x.grad)
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        _assert_finite(p.grad)


def test_state_persistence_autoregressive():
    """Feeding frames one-by-one with carried state must equal the batch call."""
    torch.manual_seed(1)
    layer = ConvLSTM(in_channels=1, hidden_channels=8, kernel_size=3)
    layer.eval()
    with torch.no_grad():
        x = torch.randn(1, 5, 1, 16, 16)
        out_batch, (h_b, c_b) = layer(x)

        h, c = layer.cell.init_state(1, 16, 16, x.device, x.dtype)
        outs = []
        for t in range(5):
            h, c = layer.cell(x[:, t], (h, c))
            outs.append(h)
        out_step = torch.stack(outs, dim=1)

    assert torch.allclose(out_batch, out_step, atol=1e-6)
    assert torch.allclose(h_b, h, atol=1e-6)
    assert torch.allclose(c_b, c, atol=1e-6)
