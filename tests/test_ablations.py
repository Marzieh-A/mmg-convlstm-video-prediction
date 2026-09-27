"""STAGE 14 tests - all four ablation variants must be shape-correct,
autoregressive-consistent, and trainable with the SAME config."""

from __future__ import annotations

import torch

from models.ablations import AblationConfig, AblationModel


def _assert_finite(t):
    assert torch.isfinite(t).all()


def test_all_variants_shapes():
    torch.manual_seed(0)
    for v in ("A", "B", "C", "D"):
        m = AblationModel(AblationConfig(in_channels=1, variant=v))
        y = m(torch.randn(1, 10, 1, 64, 64))
        assert y.shape == (1, 10, 1, 64, 64), v
        assert y.min() >= 0.0 and y.max() <= 1.0
        _assert_finite(y)


def test_variants_have_expected_components():
    for v, must in (("A", False), ("B", False), ("C", True), ("D", True)):
        m = AblationModel(AblationConfig(variant=v))
        names = [n for n, _ in m.named_modules()]
        has_motion = any("motion" in n for n in names)
        has_gate = any("gate_conv" in n for n in names)
        assert has_motion == must, (v, has_motion)
        assert has_gate == (v == "D"), (v, has_gate)
        if v == "A":
            assert not any("encoder" in n for n in names)
        else:
            assert any("encoder" in n for n in names)


def test_all_variants_backward():
    torch.manual_seed(0)
    for v in ("A", "B", "C", "D"):
        m = AblationModel(AblationConfig(variant=v))
        preds = m(torch.rand(1, 10, 1, 64, 64))
        (preds - torch.rand(1, 10, 1, 64, 64)).pow(2).mean().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, (v, name)
            _assert_finite(p.grad)


def test_variant_D_matches_MMG():
    """D must contain exactly the same architectural components as
    MMG-ConvLSTM (encoder, motion, gated ConvLSTM cell + gate, decoder)."""
    from models.mmg_convlstm import MMGConvLSTM
    m_d = AblationModel(AblationConfig(variant="D"))
    m_mmg = MMGConvLSTM()

    def top_modules(model):
        # top-level children, order-independent
        return {name for name, _ in model.named_children()}

    assert top_modules(m_d) == top_modules(m_mmg)

    # and the RNN inside must be the same class
    assert type(m_d.rnn) is type(m_mmg.rnn)


def test_rgb_variant():
    torch.manual_seed(0)
    m = AblationModel(AblationConfig(in_channels=3, variant="B"))
    y = m(torch.randn(1, 10, 3, 64, 64))
    assert y.shape == (1, 10, 3, 64, 64)
    _assert_finite(y)