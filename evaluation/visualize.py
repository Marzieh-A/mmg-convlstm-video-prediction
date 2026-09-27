"""STAGE 16 - High-resolution saved visualizations.

Produces (all saved under out_dir, 200 DPI, no interactive display):
  1. frame_grid.png       - ground truth vs prediction grids
  2. sequence.png         - prediction sequence strip
  3. error_maps.png       - per-horizon absolute error maps
  4. loss_curves.png      - train/val loss from training history
  5. horizon_mse.png      - MSE by horizon
  6. horizon_mae.png      - MAE by horizon
  7. horizon_ssim.png     - SSIM by horizon
  8. horizon_lpips.png    - LPIPS by horizon (if present in results)

Usage (programmatic):
    from evaluation.visualize import plot_all_horizons, plot_frame_grid, ...
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless - everything is saved, nothing shown
import matplotlib.pyplot as plt
import torch

DPI = 200


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------- 1-3: frame-level visualizations ----------------

def plot_frame_grid(gt_seq, pred_seq, out_dir, sample: int = 0,
                    max_frames: int = 10, tag: str = "sample0"):
    """gt_seq/pred_seq: [T, C, H, W] in [0, 1] for ONE sample.
    Two rows: top = ground truth, bottom = prediction."""
    T = min(gt_seq.shape[0], max_frames)
    fig, axes = plt.subplots(2, T, figsize=(1.4 * T, 3.2))
    for t in range(T):
        for row, seq in enumerate((gt_seq, pred_seq)):
            ax = axes[row, t]
            img = seq[t].detach().cpu()
            if img.shape[0] == 1:
                ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t+{t + 1}", fontsize=8)
    axes[0, 0].set_ylabel("GT", fontsize=9)
    axes[1, 0].set_ylabel("Pred", fontsize=9)
    fig.suptitle("Ground truth vs prediction")
    return _save(fig, Path(out_dir), f"frame_grid_{tag}.png")


def plot_sequence(pred_seq, out_dir, tag: str = "pred_seq"):
    """Single strip of the predicted frames."""
    T = pred_seq.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(1.4 * T, 1.8))
    for t in range(T):
        ax = axes[t]
        img = pred_seq[t].detach().cpu()
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t+{t + 1}", fontsize=7)
    fig.suptitle("Prediction sequence")
    return _save(fig, Path(out_dir), f"{tag}.png")


def plot_error_maps(gt_seq, pred_seq, out_dir, max_frames: int = 10,
                    tag: str = "errors"):
    """Absolute error |pred - gt| per horizon (grayscale heat)."""
    T = min(gt_seq.shape[0], max_frames)
    err = (pred_seq[:T] - gt_seq[:T]).abs().mean(dim=1).detach().cpu()
    vmax = float(err.max()) if float(err.max()) > 0 else 1.0
    fig, axes = plt.subplots(1, T, figsize=(1.4 * T, 1.8))
    for t in range(T):
        im = axes[t].imshow(err[t], cmap="inferno", vmin=0, vmax=vmax)
        axes[t].set_xticks([]); axes[t].set_yticks([])
        axes[t].set_title(f"t+{t + 1}", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle("Absolute error maps |pred - gt|")
    return _save(fig, Path(out_dir), f"{tag}.png")


# ---------------- 4: training curves ----------------

def plot_loss_curves(history: list, out_dir, tag: str = "loss"):
    """history: list of dicts with keys train_loss/val_loss/epoch (Trainer)."""
    epochs = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    va = [h["val_loss"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, tr, label="train loss")
    ax.plot(epochs, va, label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    ax.set_title("Training / validation loss")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, Path(out_dir), f"{tag}.png")


# ---------------- 5-8: horizon plots ----------------

def plot_all_horizons(eval_json_path: str | Path, out_dir):
    """Reads an eval_*.json produced by evaluation.evaluate and plots
    every available metric by horizon. The eval JSON contains
    horizon_<metric>: list of T values."""
    data = json.loads(Path(eval_json_path).read_text(encoding="utf-8"))
    paths = []
    for metric, better, color in (("mse", "lower is better", "tab:red"),
                                  ("mae", "lower is better", "tab:orange"),
                                  ("ssim", "higher is better", "tab:green"),
                                  ("lpips", "lower is better", "tab:blue")):
        key = f"horizon_{metric}"
        if key not in data:
            continue
        vals = data[key]
        T = len(vals)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, T + 1), vals, marker="o", color=color)
        ax.set_xlabel("prediction horizon")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by horizon ({better})")
        ax.grid(alpha=0.3)
        ax.set_xticks(range(1, T + 1, max(1, T // 10)))
        paths.append(_save(fig, Path(out_dir), f"horizon_{metric}.png"))
    return paths


def make_horizon_table_markdown(eval_json_path: str | Path,
                                out_path: str | Path) -> Path:
    """Save the Horizon | MSE | MAE | SSIM | LPIPS table as markdown."""
    data = json.loads(Path(eval_json_path).read_text(encoding="utf-8"))
    metrics = [m for m in ("mse", "mae", "ssim", "lpips")
               if f"horizon_{m}" in data]
    T = len(data[f"horizon_{metrics[0]}"])
    lines = ["| Horizon | " + " | ".join(m.upper() for m in metrics) + " |",
             "|" + "---|" * (len(metrics) + 1)]
    for t in range(T):
        row = [f"t+{t + 1}"] + [f"{data[f'horizon_{m}'][t]:.6f}" for m in metrics]
        lines.append("| " + " | ".join(row) + " |")
    overall = ["Overall"] + [f"{data.get('overall_' + m, float('nan')):.6f}"
                             for m in metrics]
    lines.append("| " + " | ".join(overall) + " |")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path

