"""STAGE 12 - UCF101 video dataset with video-level splitting.

Path is CONFIGURABLE (defaults point to the local raw copies). Rules honored:
  - split at the VIDEO level (official UCF101 splits preferred when
    available; otherwise deterministic hash-based split by video name)
  - no frames of one video in two different splits
  - temporal order preserved; resize 64x64; RGB; [0, 1]; 20 frames
  - a small smoke-test subset mode for fast validation

Sampling (rev2): consecutive 20-frame windows, NOT linspace over the whole
video. Rationale: next-frame prediction is only meaningful when consecutive
frames are temporally adjacent (UCF101 is 25 fps; linspace would produce
multi-second gaps). Train picks a random window per epoch (free temporal
augmentation); val/test use a deterministic centered window for
reproducibility.

cv2 is imported locally in each method (NOT stored as an attribute): the
Dataset must stay picklable for Windows spawn workers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SEQ_LEN, HEIGHT, WIDTH = 20, 64, 64
DEFAULT_ROOT = Path("datasets/raw/UCF-101")
DEFAULT_SPLIT_DIR = Path("datasets/raw/ucfTrainTestlist")


def _video_split(name: str, n_test: float = 0.2) -> str:
    """Deterministic video-level split by hash of the video filename."""
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) / (16 ** 32)
    return "test" if h < n_test else ("val" if h < n_test + 0.1 else "train")


def list_videos(root: Path, split: str, max_videos: int | None = None,
                split_dir: Path | None = None) -> list[Path]:
    """List .avi files assigned to `split`.

    Prefers official UCF101 split files if present; falls back to a
    deterministic hash split otherwise. Either way the split is at the
    video level - a video never appears in two splits.
    Official split01 has no dedicated validation set, so validation is
    carved out of the official train list with a deterministic hash -
    train and val remain disjoint subsets of the official train list.
    """
    all_videos = sorted(root.rglob("*.avi"))
    if not all_videos:
        raise FileNotFoundError(f"no .avi videos under {root}")

    if split_dir is not None and split_dir.exists():
        # 'val' is carved from the official TRAIN list; test uses its own list
        list_name = "testlist01.txt" if split == "test" else "trainlist01.txt"
        wanted = set()
        for line in (split_dir / list_name).read_text().splitlines():
            wanted.add(line.strip().split()[0].replace("/", "\\"))
        basenames = {w.split("\\")[-1] for w in wanted}
        vids = [v for v in all_videos if str(v.relative_to(root)) in wanted
                or v.name in basenames]

        if split == "val":
            vids = [v for v in vids if _video_split(v.name) == "val"]
        elif split == "train":
            vids = [v for v in vids if _video_split(v.name) != "val"]
    else:
        vids = [v for v in all_videos if _video_split(v.name) == split]

    return vids[:max_videos] if max_videos else vids


class UCF101(Dataset):
    """Each item: [20, 3, 64, 64] float32 in [0, 1] - a CONSECUTIVE
    20-frame window of one video."""

    def __init__(self, split: str = "train", root: Path | str = DEFAULT_ROOT,
                 split_dir: Path | str | None = DEFAULT_SPLIT_DIR,
                 max_videos: int | None = None,   # smoke-test subset
                 seq_len: int = SEQ_LEN):
        assert split in ("train", "val", "test")
        import cv2  # fail fast with a clear error; NOT stored (picklability)
        self.split = split
        self.root = Path(root)
        sd = Path(split_dir) if split_dir else None
        self.videos = list_videos(self.root, split, max_videos, sd)
        if not self.videos:
            raise FileNotFoundError(f"no videos for split '{split}'")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.videos)

    def _read_video(self, path: Path) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)                      # BGR, HxWx3, uint8
        cap.release()
        if len(frames) < 1:
            raise ValueError(f"unreadable video: {path}")
        return np.stack(frames)                       # [T, H, W, 3] BGR

    def _window_indices(self, n_total: int) -> list[int]:
        """Consecutive-window sampling (see module docstring).

        - shorter than needed: loop deterministically (documented)
        - train: random start (temporal augmentation)
        - val/test: deterministic centered start
        """
        T, L = n_total, self.seq_len
        if T < L:
            return [i % T for i in range(L)]
        if self.split == "train":
            start = int(np.random.randint(0, T - L + 1))
        else:
            start = max(0, (T - L) // 2)
        return list(range(start, start + L))

    def __getitem__(self, idx: int) -> torch.Tensor:
        import cv2
        vid = self.videos[idx]
        raw = self._read_video(vid)                   # [T, H, W, 3] BGR
        idxs = self._window_indices(len(raw))

        frames = []
        for i in idxs:
            f = cv2.resize(raw[i], (WIDTH, HEIGHT))       # 64x64
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)        # RGB
            frames.append(f)
        seq = np.stack(frames).astype(np.float32) / 255.0   # [20, 64, 64, 3]
        seq = np.transpose(seq, (0, 3, 1, 2))               # [20, 3, 64, 64]

        t = torch.from_numpy(np.ascontiguousarray(seq))
        assert t.shape == (self.seq_len, 3, HEIGHT, WIDTH)
        return t.contiguous()


def split_input_target(batch: torch.Tensor):
    assert batch.ndim == 5 and batch.shape[1] == SEQ_LEN
    return batch[:, 0:10], batch[:, 10:20]