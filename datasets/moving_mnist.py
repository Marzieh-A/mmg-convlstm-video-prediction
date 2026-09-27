"""STAGE 1 - Moving MNIST dataset with the project tensor contract.

Contract: every dataset returns tensors of shape [N, T, C, H, W].
input  = frames[:, 0:10]
target = frames[:, 10:20]
Splits are deterministic and disjoint: train 8000 / val 1000 / test 1000.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

RAW_PATH_DEFAULT = Path("datasets/raw/moving_mnist/mnist_test_seq.npy")

EXPECTED_SHAPE = (20, 10000, 64, 64)  # [T, N, H, W] official layout
SEQ_LEN, HEIGHT, WIDTH = 20, 64, 64
TRAIN_N, VAL_N = 8000, 1000

SPLIT_OFFSETS = {"train": 0, "val": TRAIN_N, "test": TRAIN_N + VAL_N}
SPLIT_SIZES = {"train": TRAIN_N, "val": VAL_N, "test": 1000}


class MovingMNIST(Dataset):
    """Loads the official mnist_test_seq.npy once, validates it, and serves
    slices as [T, C, H, W] float32 in [0, 1] (C=1)."""

    def __init__(self, split: str = "train", raw_path: Path | str = RAW_PATH_DEFAULT):
        assert split in SPLIT_SIZES, f"unknown split: {split}"
        self.split = split
        raw_path = Path(raw_path)
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw Moving MNIST file not found: {raw_path}. "
                "Run the forensic inspection first; do not auto-download."
            )

        # load with allow_pickle=False for safety
        arr = np.load(raw_path, allow_pickle=False)  # [T, N, H, W]

        # ---- validation (no forced reshape) ----
        if arr.ndim != 4:
            raise ValueError(f"expected 4-D [T,N,H,W], got shape {arr.shape}")
        if tuple(arr.shape) != EXPECTED_SHAPE:
            raise ValueError(
                f"file shape {arr.shape} != expected {EXPECTED_SHAPE}; "
                "the raw file is incompatible -> STOP, do not reshape."
            )
        if not np.isfinite(arr).all():
            raise ValueError("dataset contains NaN/Inf values")

        # transpose [T, N, H, W] -> [N, T, H, W]; this is a layout change,
        # not a data fabrication (same 20*10000*64*64 elements)
        arr = arr.transpose(1, 0, 2, 3)

        off = SPLIT_OFFSETS[split]
        n = SPLIT_SIZES[split]
        self.data = arr[off: off + n]

        # normalize to [0, 1] once, in float32
        self.data = self.data.astype(np.float32) / 255.0

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = torch.from_numpy(self.data[idx])          # [T, H, W]
        seq = seq.unsqueeze(1)                          # [T, 1, H, W] -> C=1
        return seq.contiguous()                         # [T, C, H, W]


def get_splits(raw_path: Path | str = RAW_PATH_DEFAULT):
    """Return (train, val, test) datasets with disjoint index ranges."""
    return (MovingMNIST("train", raw_path),
            MovingMNIST("val", raw_path),
            MovingMNIST("test", raw_path))


def split_input_target(batch: torch.Tensor):
    """[B, 20, C, H, W] -> ([B,10,C,H,W] input, [B,10,C,H,W] target)."""
    assert batch.ndim == 5 and batch.shape[1] == SEQ_LEN, batch.shape
    return batch[:, 0:10], batch[:, 10:20]