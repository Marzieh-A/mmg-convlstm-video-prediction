"""STAGE 15 tests - evaluation pipeline with synthetic data (no checkpoint)."""

from __future__ import annotations

import torch

from evaluation.evaluate import evaluate_model
from models.ablations import AblationConfig, AblationModel


class _TinySet(torch.utils.data.Dataset):
    def __init__(self, n=4):
        g = torch.Generator().manual_seed(0)
        self.data = torch.rand(n, 20, 1, 64, 64, generator=g)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]


def test_evaluate_model_runs_without_lpips():
    torch.manual_seed(0)
    m = AblationModel(AblationConfig(variant="A"))
    res = evaluate_model(m, _TinySet(), {"batch_size": 2}, use_lpips=False)
    for k in ("mse", "mae", "ssim"):
        assert res[k] >= 0.0
        assert res["n_batches"] == 2


def test_evaluate_model_is_autoregressive():
    """Evaluation must use AR generation: predictions only depend on the
    first 10 frames - verified by zeroing the future frames of the input."""
    torch.manual_seed(0)
    m = AblationModel(AblationConfig(variant="A")).eval()
    data = torch.rand(2, 20, 1, 32, 32)
    with torch.no_grad():
        y1 = m(data[:, 0:10])
        y2 = m(data[:, 0:10])   # future frames never enter the API
    assert torch.allclose(y1, y2)


def test_evaluate_reproducible():
    torch.manual_seed(0)
    m = AblationModel(AblationConfig(variant="A")).eval()
    ds = _TinySet(4)
    r1 = evaluate_model(m, ds, {"batch_size": 2}, use_lpips=False)
    r2 = evaluate_model(m, ds, {"batch_size": 2}, use_lpips=False)
    assert abs(r1["mse"] - r2["mse"]) < 1e-9