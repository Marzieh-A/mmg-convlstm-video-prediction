"""STAGE 12 - Inspect the local UCF101 copy before training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.ucf101 import DEFAULT_ROOT, DEFAULT_SPLIT_DIR, list_videos


def main():
    print(f"root      : {DEFAULT_ROOT}  exists={DEFAULT_ROOT.exists()}")
    print(f"split_dir : {DEFAULT_SPLIT_DIR}  exists={DEFAULT_SPLIT_DIR.exists()}")
    if not DEFAULT_ROOT.exists():
        print("UCF101 root NOT FOUND - fix the path in configs/ucf101.yaml")
        return 1
    for split in ("train", "val", "test"):
        vids = list_videos(DEFAULT_ROOT, split, None, DEFAULT_SPLIT_DIR)
        print(f"{split:5s}: {len(vids)} videos | first: "
              f"{vids[0].name if vids else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())