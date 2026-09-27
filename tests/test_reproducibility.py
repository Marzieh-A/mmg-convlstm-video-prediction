"""STAGE 18 tests - seeding determinism and run-record completeness."""

from __future__ import annotations

import json

import numpy as np
import torch

from models.ablations import AblationConfig, AblationModel
from utils.reproducibility import build_reproducibility_record, save_run_record
from utils.seed import set_seed


def test_seeding_same_weights():
    """Same seed -> identical draws and identical model outputs.
    Nothing else may touch the RNG between the two seeded blocks."""
    set_seed(1234)
    w1 = torch.rand(3, 3)
    set_seed(1234)
    w2 = torch.rand(3, 3)
    assert torch.equal(w1, w2)

    # two freshly seeded identical models must produce identical outputs
    set_seed(7)
    a = AblationModel(AblationConfig(variant="A")).eval()
    set_seed(7)
    b = AblationModel(AblationConfig(variant="A")).eval()
    x = torch.rand(1, 4, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-6)


def test_cudnn_deterministic_flags():
    set_seed(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    set_seed(0, deterministic=False)
    assert torch.backends.cudnn.benchmark is True
    set_seed(0, deterministic=True)   # restore project default


def test_run_record_content(tmp_path):
    cfg = {"epochs": 5, "batch_size": 4, "lr": 1e-3, "seed": 42}
    p = save_run_record(cfg, 42, tmp_path)
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["seed"] == 42
    assert rec["epochs"] == 5
    assert "torch_version" in rec
    assert "python_version" in rec
    assert "cuda_available" in rec
    if torch.cuda.is_available():
        assert "gpu_name" in rec
        assert "cudnn_version" in rec


def test_numpy_rng_seeded():
    set_seed(99)
    a = np.random.rand(10)
    set_seed(99)
    b = np.random.rand(10)
    assert np.array_equal(a, b)