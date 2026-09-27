"""STAGE 16b - Multi-variant comparison figures.

Reads eval_{A,B,C,D}.json produced by evaluation.evaluate and draws
one figure per metric with all four variant curves on the same axes,
plus a summary bar chart of overall values and a markdown table.

Usage:
    python -m evaluation.compare_variants --dir outputs/moving_mnist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 200
VARIANTS = ["A", "B", "C", "D"]
LABELS = {
    "A": "A: ConvLSTM (baseline)",
    "B": "B: + Multi-Scale Encoder",
    "C": "C: + Motion Residual",
    "D": "D: + Motion Gate (MMG, full)",
}
COLORS = {"A": "#888888", "B": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}
STYLES = {"A": "--", "B": "-.", "C": "-", "D": "-"}


def _load(eval_dir: Path, variant: str) -> dict | None:
    p = eval_dir / f"eval_{variant}.json"
    if not p.exists():
        print(f"missing: {p} - skipping variant {variant}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def plot_metric_curves(eval_dir: Path, metric: str, out_dir: Path) -> Path | None:
    """One figure per metric: all variant curves by horizon."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    found = False
    for v in VARIANTS:
        data = _load(eval_dir, v)
        if data is None or f"horizon_{metric}" not in data:
            continue
        vals = data[f"horizon_{metric}"]
        ax.plot(range(1, len(vals) + 1), vals, marker="o", markersize=4,
                color=COLORS[v], linestyle=STYLES[v], label=LABELS[v])
        found = True
    if not found:
        plt.close(fig)
        return None

    better = "higher is better" if metric in ("ssim",) else "lower is better"
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by horizon ({better}) - Moving MNIST")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"compare_{metric}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overall_bars(eval_dir: Path, out_dir: Path) -> Path | None:
    """Grouped bar chart of overall values for all metrics x variants."""
    metrics = ["mse", "mae", "ssim", "lpips"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, metric in zip(axes, metrics):
        xs, heights, labels = [], [], []
        for i, v in enumerate(VARIANTS):
            data = _load(eval_dir, v)
            if data is None:
                continue
            xs.append(i)
            heights.append(data[metric])
            labels.append(v)
        if not heights:
            ax.axis("off")
            continue
        ax.bar(xs, heights, color=[COLORS[v] for v in labels])
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(f"{metric.upper()}")
        ax.grid(alpha=0.3, axis="y")
        for x, h in zip(xs, heights):           # value labels on bars
            ax.text(x, h, f"{h:.4f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Overall test metrics by variant (Moving MNIST)")
    fig.tight_layout()
    path = out_dir / "compare_overall_bars.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def write_comparison_table(eval_dir: Path, out_path: Path) -> Path:
    """Markdown table: variant rows x (overall + t+1/t+5/t+10) columns."""
    metrics = ["mse", "mae", "ssim", "lpips"]
    cols = ["overall"] + [f"t+{k}" for k in (1, 5, 10)]
    lines = ["| Variant | " + " | ".join(
        f"{m.upper()} {c}" for m in metrics for c in cols) + " |",
        "|" + "---|" * (len(metrics) * len(cols) + 1)]
    for v in VARIANTS:
        data = _load(eval_dir, v)
        if data is None:
            continue
        row = [v]
        for m in metrics:
            row.append(f"{data[m]:.4f}")
            for k in (1, 5, 10):
                row.append(f"{data[f'horizon_{m}'][k - 1]:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder containing eval_*.json")
    ap.add_argument("--dataset", default="Moving MNIST")
    args = ap.parse_args()

    d = Path(args.dir)
    for metric in ("mse", "mae", "ssim", "lpips"):
        p = plot_metric_curves(d, metric, d)
        print(f"saved: {p}" if p else f"no data for {metric}")
    p = plot_overall_bars(d, d)
    print(f"saved: {p}" if p else "no overall bar data")
    p = write_comparison_table(d, d / "comparison_table.md")
    print(f"saved: {p}")


if __name__ == "__main__":
    main()