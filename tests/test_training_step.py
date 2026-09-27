"""STAGE 10 tests - trainer internals with a tiny synthetic run."""

from __future__ import annotations

import torch

from models.mmg_convlstm import MMGConfig, MMGConvLSTM
from training.trainer import Trainer


class _TinySet(torch.utils.data.Dataset):
    """Synthetic [20,1,64,64] sequences in [0,1]."""
    def __init__(self, n=6):
        g = torch.Generator().manual_seed(0)
        self.data = torch.rand(n, 20, 1, 64, 64, generator=g)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]


def _make_trainer(tmp_path, epochs=2):
    torch.manual_seed(0)
    model = MMGConvLSTM(MMGConfig(in_channels=1))
    cfg = {"epochs": epochs, "batch_size": 2, "lr": 1e-3,
           "weight_decay": 1e-4, "scheduler": "none", "max_grad_norm": 1.0,
           "num_workers": 0, "early_stopping_patience": 5 ,"loss_kind": "mse"}
    return Trainer(model, _TinySet(), _TinySet(4), cfg, str(tmp_path))


def test_one_epoch_and_backward(tmp_path):
    tr = _make_trainer(tmp_path)
    stats = tr.train_one_epoch(0)
    assert stats["train_loss"] >= 0.0
    assert stats["grad_norm"] >= 0.0
    assert stats["epoch_time_s"] > 0.0


def test_validation_runs(tmp_path):
    tr = _make_trainer(tmp_path)
    v = tr.validate()
    assert v["val_loss"] >= 0.0
    assert v["val_mse"] >= 0.0
    # val_mse must equal val_loss under the base MSE protocol
    assert abs(v["val_loss"] - v["val_mse"]) < 1e-6

def test_fit_saves_checkpoints_and_history(tmp_path):
    tr = _make_trainer(tmp_path, epochs=2)
    hist = tr.fit()
    assert len(hist) == 2
    assert (tmp_path / "latest.pth").exists()
    assert (tmp_path / "best.pth").exists()


def test_resume_from_checkpoint(tmp_path):
    tr = _make_trainer(tmp_path, epochs=2)
    tr.fit()
    from training.checkpoint import load_checkpoint
    model2 = MMGConvLSTM(MMGConfig(in_channels=1))
    ckpt = load_checkpoint(tmp_path / "latest.pth", model2)
    assert ckpt["epoch"] == 2
    # compare on CPU (trainer model lives on CUDA; load is CPU-mapped)
    sd1 = {k: v.cpu() for k, v in tr.model.state_dict().items()}
    sd2 = {k: v.cpu() for k, v in model2.state_dict().items()}
    assert set(sd1.keys()) == set(sd2.keys())
    for name in sd1:
        assert torch.equal(sd1[name], sd2[name]), name

def test_nan_detection_stops(tmp_path):
    """A NaN loss must raise, not silently continue."""
    tr = _make_trainer(tmp_path)
    try:
        tr._nan_stop(float("nan"), "test")
        raised = False
    except FloatingPointError:
        raised = True
    assert raised