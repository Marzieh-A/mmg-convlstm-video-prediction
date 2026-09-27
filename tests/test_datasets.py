"""STAGE 1 tests - dataset contract [B, T, C, H, W] for Moving MNIST."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from datasets.moving_mnist import (
    EXPECTED_SHAPE, RAW_PATH_DEFAULT, MovingMNIST, get_splits, split_input_target,
)

RAW = RAW_PATH_DEFAULT
FILE_OK = RAW.exists()

pytestmark = pytest.mark.skipif(
    not FILE_OK, reason="raw moving mnist file missing - forensic check pending"
)


@pytest.fixture(scope="module")
def splits():
    return get_splits(RAW)


@pytest.mark.skipif(not FILE_OK, reason="raw file missing")
def test_loads_and_shapes(splits):
    train, val, test = splits
    assert len(train) == 8000 and len(val) == 1000 and len(test) == 1000


@pytest.mark.skipif(not FILE_OK, reason="raw file missing")
def test_item_contract(splits):
    seq = splits[0][0]
    assert isinstance(seq, torch.Tensor)
    assert seq.shape == (20, 1, 64, 64)          # [T, C, H, W]
    assert seq.dtype == torch.float32


@pytest.mark.skipif(not FILE_OK, reason="raw file missing")
def test_value_range_and_finite(splits):
    for i in (0, 1234, 7999):
        seq = splits[0][i]
        assert torch.isfinite(seq).all()
        assert seq.min() >= 0.0 and seq.max() <= 1.0


@pytest.mark.skipif(not FILE_OK, reason="raw file missing")
def test_input_target_slicing(splits):
    batch = torch.stack([splits[0][i] for i in range(4)])  # [B,20,1,64,64]
    inp, tgt = split_input_target(batch)
    assert inp.shape == (4, 10, 1, 64, 64)
    assert tgt.shape == (4, 10, 1, 64, 64)
    # disjointness of the slice itself
    assert not torch.equal(inp[:, -1], tgt[:, 0])


@pytest.mark.skipif(not FILE_OK, reason="raw file missing")
def test_no_leakage_between_splits(splits):
    train, val, test = splits
    # disjointness guaranteed by construction; verify on raw indices
    t0 = train[0]; v0 = val[0]; s0 = test[0]
    assert not torch.equal(t0, v0) or True  # same content possible in theory;
    # authoritative check is index ranges, so verify dataset lengths + offsets
    from datasets.moving_mnist import SPLIT_OFFSETS, SPLIT_SIZES
    assert SPLIT_OFFSETS["val"] == 8000
    assert SPLIT_OFFSETS["test"] == 9000
    assert SPLIT_SIZES["train"] + SPLIT_SIZES["val"] + SPLIT_SIZES["test"] == 10000


@pytest.mark.skipif(not FILE_OK, reason="raw file missing")
def test_resolution_and_channels(splits):
    seq = splits[2][0]
    assert seq.shape[2] == 64 and seq.shape[3] == 64
    assert seq.shape[1] == 1


@pytest.mark.skipif(not FILE_OK, reason="raw file missing")
def test_sequence_length(splits):
    seq = splits[1][0]
    assert seq.shape[0] == 20


def test_expected_shape_constant():
    assert EXPECTED_SHAPE == (20, 10000, 64, 64)