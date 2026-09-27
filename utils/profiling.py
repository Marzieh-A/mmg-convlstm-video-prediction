"""STAGE 17 - Computational profiling with MEASURED values only.

Reports: total/trainable parameters, inference time (GPU-synchronized),
peak GPU memory, throughput. FLOPs are reported ONLY if the optional
thop/ptflops profiler is actually installed and succeeds - never invented.

Usage:
    python -m utils.profiling --channels 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.device import get_device


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


@torch.no_grad()
def measure_inference(model: torch.nn.Module, in_shape: tuple,
                      n_warmup: int = 3, n_runs: int = 10) -> dict:
    """Synchronized GPU timing: warmup, then mean/std over n_runs."""
    device = get_device()
    model = model.to(device).eval()
    x = torch.rand(*in_shape, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    mean_t = sum(times) / n_runs
    std_t = (sum((t - mean_t) ** 2 for t in times) / n_runs) ** 0.5
    B = in_shape[0]
    frames_per_seq = in_shape[1]
    return {
        "inference_time_mean_s": mean_t,
        "inference_time_std_s": std_t,
        "peak_gpu_memory_mb": peak_mem_mb,
        "throughput_sequences_per_s": B / mean_t,
        "throughput_frames_per_s": B * frames_per_seq / mean_t,
        "n_warmup": n_warmup, "n_runs": n_runs,
    }


def measure_flops(model: torch.nn.Module, in_shape: tuple) -> dict | None:
    """FLOPs ONLY via a real profiler if installed. Returns None otherwise -
    we never report estimated or invented FLOPs."""
    try:
        import thop
    except ImportError:
        return None
    try:
        model_c = model.cpu().eval()
        x = torch.rand(*in_shape)
        macs, _ = thop.profile(model_c, inputs=(x,), verbose=False)
        return {"macs": int(macs), "flops_estimated": int(2 * macs),
                "tool": "thop"}
    except Exception:
        return None


def profile_model(model: torch.nn.Module, in_shape: tuple,
                  measure_flops_flag: bool = False) -> dict:
    rep = {"input_shape": list(in_shape),
           "device": str(get_device())}
    rep.update(count_parameters(model))
    rep.update(measure_inference(model, in_shape))
    if measure_flops_flag:
        fl = measure_flops(model, in_shape)
        rep["flops"] = fl if fl else "not available (install thop)"
    if torch.cuda.is_available():
        rep["gpu_name"] = torch.cuda.get_device_properties(0).name
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--variant", default="D")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--flops", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from models.ablations import AblationConfig, AblationModel
    from baselines.convlstm_baseline import BaselineConfig, ConvLSTMBaseline

    if args.variant == "baseline":
        model = ConvLSTMBaseline(BaselineConfig(in_channels=args.channels))
    else:
        model = AblationModel(AblationConfig(in_channels=args.channels,
                                             variant=args.variant))

    shape = (args.batch, 10, args.channels, 64, 64)
    rep = profile_model(model, shape, measure_flops_flag=args.flops)

    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()