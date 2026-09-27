"""STAGE 9 tests - loss correctness (base MSE + GDL-L1 protocol)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.reconstruction import (
    ReconstructionLoss, gdl_l1_loss, gdl_loss, l1_loss, mse_loss,
)


# ---------- base MSE protocol ----------

def test_mse_known_value():
    p = torch.zeros(1, 1, 1, 2, 2)
    t = torch.ones(1, 1, 1, 2, 2)
    assert mse_loss(p, t).item() == 1.0


def test_mse_matches_torch_mse():
    torch.manual_seed(0)
    p, t = torch.rand(2, 3, 1, 16, 16), torch.rand(2, 3, 1, 16, 16)
    assert torch.allclose(mse_loss(p, t),
                          F.mse_loss(p, t))


def test_mse_zero_when_equal():
    t = torch.rand(1, 2, 1, 8, 8)
    assert mse_loss(t, t).item() == 0.0


def test_mse_shape_mismatch_raises():
    p = torch.zeros(1, 1, 1, 8, 8)
    t = torch.zeros(1, 1, 1, 4, 4)
    try:
        mse_loss(p, t)
        raised = False
    except AssertionError:
        raised = True
    assert raised


# ---------- GDL / L1 (Mathieu et al. 2016 protocol) ----------

def test_l1_known_value():
    p = torch.zeros(1, 1, 1, 2, 2)
    t = torch.ones(1, 1, 1, 2, 2)
    assert l1_loss(p, t).item() == 1.0


def test_gdl_zero_when_equal():
    t = torch.rand(2, 4, 1, 16, 16)
    assert gdl_loss(t, t).item() == 0.0


def test_gdl_penalizes_blur():
    """Blurred prediction must score worse than a sharp perturbation,
    even when the blurred one has LOWER pixel-space L1 - the whole point
    of GDL (Mathieu et al. 2016)."""
    torch.manual_seed(0)
    t4 = torch.rand(1, 1, 32, 32)                       # [B, C, H, W]
    blurred4 = F.avg_pool2d(t4, 5, stride=1, padding=2)
    t = t4.unsqueeze(1)                                 # -> [B, T=1, C, H, W]
    blurred = blurred4.unsqueeze(1)
    sharp = t + 0.05 * torch.randn_like(t)
    assert gdl_loss(blurred, t) > gdl_loss(sharp.clamp(0, 1), t)


def test_gdl_l1_combination():
    torch.manual_seed(0)
    p, t = torch.rand(1, 2, 1, 8, 8), torch.rand(1, 2, 1, 8, 8)
    assert torch.allclose(gdl_l1_loss(p, t), l1_loss(p, t) + gdl_loss(p, t))



def test_composite_and_wrapper():
    torch.manual_seed(0)
    p, t = torch.rand(1, 2, 1, 8, 8), torch.rand(1, 2, 1, 8, 8)
    expected = mse_loss(p, t) + l1_loss(p, t) + gdl_loss(p, t)
    assert torch.allclose(ReconstructionLoss("composite")(p, t), expected)

def test_wrapper_kinds():
    torch.manual_seed(0)
    p, t = torch.rand(1, 2, 1, 8, 8), torch.rand(1, 2, 1, 8, 8)
    assert torch.allclose(ReconstructionLoss("mse")(p, t), mse_loss(p, t))
    assert torch.allclose(ReconstructionLoss("gdl_l1")(p, t), gdl_l1_loss(p, t))


# ---------- integration with the model ----------

def test_wrapper_and_gradient():
    torch.manual_seed(0)
    from models.mmg_convlstm import MMGConvLSTM
    m = MMGConvLSTM()
    preds = m(torch.rand(1, 10, 1, 64, 64))
    loss = ReconstructionLoss("gdl_l1")(preds, torch.rand(1, 10, 1, 64, 64))
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_training_loss_on_model_output():
    torch.manual_seed(0)
    from models.mmg_convlstm import MMGConvLSTM
    m = MMGConvLSTM()
    out = m(torch.rand(1, 10, 1, 32, 32))
    assert torch.isfinite(ReconstructionLoss("gdl_l1")(out, torch.rand_like(out)))