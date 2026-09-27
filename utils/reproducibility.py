"""Reproducibility: unified seeding and run-record persistence."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch


def build_reproducibility_record(config: dict, seed: int) -> dict:
    """Attach hardware/software metadata to a run config dict."""
    rec = dict(config)
    rec["seed"] = seed
    rec["python_version"] = sys.version.split()[0]
    rec["torch_version"] = torch.__version__
    rec["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        rec["cuda_version"] = torch.version.cuda
        rec["cudnn_version"] = torch.backends.cudnn.version()
        rec["gpu_name"] = torch.cuda.get_device_properties(0).name
    rec["platform"] = platform.platform()
    return rec


def save_run_record(config: dict, seed: int, out_dir: str | Path) -> Path:
    """Write config + environment metadata as JSON next to the results."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = build_reproducibility_record(config, seed)
    path = out / "run_record.json"
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return path