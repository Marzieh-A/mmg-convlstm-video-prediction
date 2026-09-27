"""STAGE 16f - Horizon metric curves for a SINGLE evaluation JSON.

Usage (from project root):
    python -m evaluation.plot_single_eval --json outputs/ucf101/eval_D.json \
        --label "UCF101: D (warmup+composite)" --out outputs/ucf101/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 200
COLOR = "#d62728"
METRICS = ["ssim", "lpips", "mse", "mae"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", default="outputs/single_eval_plots")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        vals = data[f"horizon_{metric}"]
        better = "higher is better" if metric == "ssim" else "lower is better"
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(range(1, len(vals) + 1), vals, marker="o", markersize=5,
                color=COLOR, linewidth=1.8, label=args.label)
        for i, v in enumerate(vals, start=1):
            if i in (1, 5, 10):
                ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xlabel("prediction horizon")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{args.label}\n{metric.upper()} by horizon ({better})")
        ax.set_xticks(range(1, len(vals) + 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.savefig(out / f"ucf_{metric}_horizon.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out / f'ucf_{metric}_horizon.png'}")

    # overall summary bars
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = range(len(METRICS))
    hs = [data[m] for m in METRICS]
    ax.bar(xs, hs, color=COLOR)
    ax.set_xticks(xs); ax.set_xticklabels([m.upper() for m in METRICS])
    ax.set_title(f"{args.label} - overall test metrics")
    ax.grid(alpha=0.3, axis="y")
    for x, h in zip(xs, hs):
        ax.text(x, h, f"{h:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "ucf_overall_bars.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out / 'ucf_overall_bars.png'}")


if __name__ == "__main__":
    main()