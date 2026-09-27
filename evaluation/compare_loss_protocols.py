"""STAGE 16e - Metric-curve comparison: MSE protocol vs warmup+composite.

Reads two eval JSONs (same format as evaluation.evaluate output) and draws
SSIM / LPIPS / MSE / MAE horizon curves, two lines per figure.

Usage (from project root):
    python -m evaluation.compare_loss_protocols \
        --mse outputs/moving_mnist_mse/eval_D.json \
        --composite outputs/moving_mnist_gdl/eval_D.json \
        --out outputs/compare_loss
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 200
COLORS = {"mse": "#888888", "composite": "#d62728"}
LABELS = {"mse": "D: MSE (base protocol)",
          "composite": "D: warmup + MSE+L1+GDL (composite)"}
STYLES = {"mse": "--", "composite": "-"}
METRICS = ["ssim", "lpips", "mse", "mae"]


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mse", required=True)
    ap.add_argument("--composite", required=True)
    ap.add_argument("--out", default="outputs/compare_loss")
    args = ap.parse_args()

    data = {"mse": _load(args.mse), "composite": _load(args.composite)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for key in ("mse", "composite"):
            vals = data[key][f"horizon_{metric}"]
            ax.plot(range(1, len(vals) + 1), vals, marker="o", markersize=4,
                    color=COLORS[key], linestyle=STYLES[key], label=LABELS[key])
            ax.annotate(f"{vals[-1]:.4f}", (len(vals), vals[-1]),
                        textcoords="offset points", xytext=(5, -3),
                        fontsize=8, color=COLORS[key])
        better = "higher is better" if metric == "ssim" else "lower is better"
        ax.set_xlabel("prediction horizon")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by horizon ({better})")
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        p = out / f"loss_compare_{metric}.png"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {p}")

    # overall summary bars
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, metric in zip(axes, METRICS):
        xs, hs, cs = [], [], []
        for i, key in enumerate(("mse", "composite")):
            xs.append(i); hs.append(data[key][metric]); cs.append(COLORS[key])
        ax.bar(xs, hs, color=cs)
        ax.set_xticks(xs); ax.set_xticklabels(["MSE", "composite"])
        ax.set_title(f"{metric.upper()} (overall)")
        ax.grid(alpha=0.3, axis="y")
        for x, h in zip(xs, hs):
            ax.text(x, h, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Overall test metrics: MSE protocol vs warmup+composite")
    fig.tight_layout()
    p = out / "loss_compare_overall_bars.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {p}")


if __name__ == "__main__":
    main()
