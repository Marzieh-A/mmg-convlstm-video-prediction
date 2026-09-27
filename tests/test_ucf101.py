"""STAGE 12 tests - UCF101 loader (skipped if the dataset is not present)."""

from __future__ import annotations

import pytest
import torch

from datasets.ucf101 import DEFAULT_ROOT, UCF101, split_input_target

ROOT_OK = DEFAULT_ROOT.exists()
pytestmark = pytest.mark.skipif(not ROOT_OK, reason="UCF101 not found locally")

SMOKE = 3   # tiny subset so the suite stays fast


@pytest.fixture(scope="module")
def data():
    return (UCF101("train", max_videos=SMOKE),
            UCF101("val", max_videos=SMOKE),
            UCF101("test", max_videos=SMOKE))


def test_loads(data):
    for ds in data:
        assert len(ds) == SMOKE


def test_item_contract(data):
    seq = data[0][0]
    assert seq.shape == (20, 3, 64, 64)
    assert seq.dtype == torch.float32
    assert seq.min() >= 0.0 and seq.max() <= 1.0
    assert torch.isfinite(seq).all()


def test_video_level_no_leakage(data):
    """The same video file must never appear in two splits."""
    names = [{v.name for v in ds.videos} for ds in data]
    assert not (names[0] & names[1])
    assert not (names[0] & names[2])
    assert not (names[1] & names[2])


def test_temporal_order(data):
    """Resampling must not invert time: read two far-apart sampled frames
    and ensure the sampler indices are strictly non-decreasing."""
    from datasets.ucf101 import _sample_indices
    idx = _sample_indices(200, 20)
    assert idx == sorted(idx) and len(idx) == 20 and idx[0] == 0


def test_input_target(data):
    batch = torch.stack([data[0][i] for i in range(2)])
    inp, tgt = split_input_target(batch)
    assert inp.shape == (2, 10, 3, 64, 64)
    assert tgt.shape == (2, 10, 3, 64, 64)