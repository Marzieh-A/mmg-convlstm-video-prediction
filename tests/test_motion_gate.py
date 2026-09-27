"""
STAGE 4 + 5 tests
Motion residual and motion-gated ConvLSTM.
"""

from __future__ import annotations

import torch

from models.motion_gated_convlstm import MotionGatedConvLSTM
from models.motion_module import MotionResidual


def _assert_finite(tensor: torch.Tensor) -> None:
    """Check that a tensor contains no NaN or Inf values."""
    assert torch.isfinite(tensor).all(), "NaN or Inf detected"


# ============================================================
# STAGE 4: Motion Residual
# ============================================================


def test_output_shape() -> None:
    """Motion residual must preserve the input shape."""
    torch.manual_seed(0)

    motion_residual = MotionResidual()

    features = torch.randn(2, 10, 64, 64, 64)
    # [B, T, C_f, H, W]

    residuals = motion_residual(features)

    assert residuals.shape == features.shape
    _assert_finite(residuals)


def test_first_frame_is_zero() -> None:
    """The residual of the first timestep must be zero."""
    torch.manual_seed(0)

    motion_residual = MotionResidual()

    features = torch.randn(1, 5, 8, 16, 16)

    residuals = motion_residual(features)

    assert torch.equal(
        residuals[:, 0],
        torch.zeros_like(residuals[:, 0]),
    )


def test_temporal_indexing_correct() -> None:
    """
    D_t must equal F_t - F_{t-1} for every t >= 1
    in zero-based Python indexing.
    """
    torch.manual_seed(0)

    motion_residual = MotionResidual()

    features = torch.randn(2, 6, 4, 8, 8)

    residuals = motion_residual(features)

    for t in range(1, 6):
        expected = features[:, t] - features[:, t - 1]

        assert torch.allclose(
            residuals[:, t],
            expected,
            atol=1e-6,
        )


def test_constant_sequence_gives_zero_motion() -> None:
    """
    A temporally constant sequence must produce zero
    motion residual at every timestep.
    """
    torch.manual_seed(0)

    motion_residual = MotionResidual()

    frame = torch.randn(1, 1, 4, 8, 8)

    features = frame.repeat(1, 7, 1, 1, 1)

    residuals = motion_residual(features)

    assert residuals.abs().max().item() == 0.0


def test_residual_gradient_flow() -> None:
    """Gradients must propagate through MotionResidual."""
    torch.manual_seed(0)

    motion_residual = MotionResidual()

    features = torch.randn(
        2,
        4,
        8,
        16,
        16,
        requires_grad=True,
    )

    residuals = motion_residual(features)

    loss = residuals.pow(2).mean()
    loss.backward()

    assert features.grad is not None
    _assert_finite(features.grad)


def test_residual_batch_size_one() -> None:
    """MotionResidual must support batch size equal to one."""
    motion_residual = MotionResidual()

    features = torch.randn(1, 3, 16, 32, 32)

    residuals = motion_residual(features)

    assert residuals.shape == (1, 3, 16, 32, 32)


# ============================================================
# STAGE 5: Motion-Gated ConvLSTM
# ============================================================


def test_gate_module_output_shapes() -> None:
    """
    MotionGatedConvLSTM must return correctly shaped
    fused representations, gates, and final states.
    """
    torch.manual_seed(0)

    motion_gated_lstm = MotionGatedConvLSTM(
        in_channels=64,
        hidden_channels=64,
        kernel_size=3,
    )

    features = torch.randn(
        2,
        4,
        64,
        64,
        64,
    )

    residuals = torch.randn(
        2,
        4,
        64,
        64,
        64,
    )

    fused, gates, (h_last, c_last) = motion_gated_lstm(
        features,
        residuals,
    )

    assert fused.shape == (2, 4, 64, 64, 64)
    assert gates.shape == (2, 4, 64, 64, 64)

    assert h_last.shape == (2, 64, 64, 64)
    assert c_last.shape == (2, 64, 64, 64)

    _assert_finite(fused)
    _assert_finite(gates)


def test_gate_range_sigmoid() -> None:
    """The sigmoid gate must remain in the range [0, 1]."""
    torch.manual_seed(0)

    motion_gated_lstm = MotionGatedConvLSTM(
        in_channels=8,
        hidden_channels=8,
    )

    features = torch.randn(
        2,
        3,
        8,
        16,
        16,
    ) * 5.0

    residuals = torch.randn(
        2,
        3,
        8,
        16,
        16,
    ) * 5.0

    _, gates, _ = motion_gated_lstm(
        features,
        residuals,
    )

    assert gates.min().item() >= 0.0
    assert gates.max().item() <= 1.0


def test_fusion_matches_formula() -> None:
    """
    Verify the exact fusion equation:

        Z_t = H_t + G_t * D_t

    by independently recomputing the ConvLSTM states
    and gates using the same module parameters.
    """
    torch.manual_seed(0)

    motion_gated_lstm = MotionGatedConvLSTM(
        in_channels=4,
        hidden_channels=4,
        kernel_size=3,
    )

    motion_gated_lstm.eval()

    features = torch.randn(
        1,
        3,
        4,
        8,
        8,
    )

    residuals = torch.randn(
        1,
        3,
        4,
        8,
        8,
    )

    with torch.no_grad():

        fused, gates, _ = motion_gated_lstm(
            features,
            residuals,
        )

        # Independently recompute the ConvLSTM states.
        h, c = motion_gated_lstm.cell.init_state(
            1,
            8,
            8,
            features.device,
            features.dtype,
        )

        for t in range(3):

            h, c = motion_gated_lstm.cell(
                features[:, t],
                (h, c),
            )

            gate_t = torch.sigmoid(
                motion_gated_lstm.gate_conv(
                    torch.cat(
                        [h, residuals[:, t]],
                        dim=1,
                    )
                )
            )

            fused_t = h + gate_t * residuals[:, t]

            assert torch.allclose(
                fused[:, t],
                fused_t,
                atol=1e-6,
            )

            assert torch.allclose(
                gates[:, t],
                gate_t,
                atol=1e-6,
            )


def test_gate_is_spatially_varying() -> None:
    """
    The learned gate must vary spatially.

    A scalar gate would produce the same value at every
    spatial location. The convolutional gate should instead
    produce spatially varying values.
    """
    torch.manual_seed(0)

    motion_gated_lstm = MotionGatedConvLSTM(
        in_channels=8,
        hidden_channels=8,
    )

    features = torch.randn(
        2,
        2,
        8,
        16,
        16,
    )

    residuals = torch.randn(
        2,
        2,
        8,
        16,
        16,
    )

    _, gates, _ = motion_gated_lstm(
        features,
        residuals,
    )

    # Compute standard deviation over spatial dimensions.
    spatial_std = gates.std(dim=(-2, -1))

    # At least one timestep/channel must show
    # meaningful spatial variation.
    assert (spatial_std > 1e-4).any().item()


def test_gate_module_gradient_flow() -> None:
    """
    Gradients must propagate through:
        features
        residuals
        ConvLSTM parameters
        gate parameters
    """
    torch.manual_seed(0)

    motion_gated_lstm = MotionGatedConvLSTM(
        in_channels=4,
        hidden_channels=4,
    )

    features = torch.randn(
        2,
        3,
        4,
        16,
        16,
        requires_grad=True,
    )

    residuals = torch.randn(
        2,
        3,
        4,
        16,
        16,
        requires_grad=True,
    )

    fused, _, _ = motion_gated_lstm(
        features,
        residuals,
    )

    loss = fused.pow(2).mean()
    loss.backward()

    # Input feature gradients.
    assert features.grad is not None
    _assert_finite(features.grad)

    # Motion residual gradients.
    assert residuals.grad is not None
    _assert_finite(residuals.grad)

    # Model parameter gradients.
    for name, parameter in motion_gated_lstm.named_parameters():
        assert parameter.grad is not None, (
            f"No gradient for parameter: {name}"
        )

        _assert_finite(parameter.grad)


def test_gate_module_batch_one_and_small() -> None:
    """The module must support batch size one and small sequences."""
    torch.manual_seed(0)

    motion_gated_lstm = MotionGatedConvLSTM(
        in_channels=4,
        hidden_channels=4,
    )

    features = torch.randn(
        1,
        2,
        4,
        16,
        16,
    )

    residuals = torch.randn(
        1,
        2,
        4,
        16,
        16,
    )

    fused, _, _ = motion_gated_lstm(
        features,
        residuals,
    )

    assert fused.shape == (1, 2, 4, 16, 16)

    _assert_finite(fused)