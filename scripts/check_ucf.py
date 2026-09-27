"""Smoke-check the UCF101 dataset: shapes, ranges, split sizes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ucf101 import UCF101

for split in ("train", "val", "test"):
    d = UCF101(split, max_videos=4)
    s = d[0]
    print(f"{split}: len={len(d)} | shape={tuple(s.shape)} | "
          f"range=[{float(s.min()):.3f}, {float(s.max()):.3f}]")