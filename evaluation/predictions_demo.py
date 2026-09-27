"""
STAGE 16c - Visualize sample predictions of all variants side by side.

For a few test samples, runs each loaded variant and saves:
  - predictions_<tag>.png : rows = GT, A, B, C, D predictions over 10 horizons
  - errors_<tag>.png      : |pred - GT| error maps per variant
  - input_frames_<tag>.png: the 10 observed input frames (context)

Important:
    Image enhancement is applied ONLY for visualization.
    It does NOT modify the model predictions or evaluation metrics.

Usage:
    python -m evaluation.predictions_demo \
        --config configs/moving_mnist.yaml \
        --variant-dir outputs/moving_mnist \
        --samples 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

import torch
import yaml


# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from datasets.moving_mnist import get_splits
from evaluation.evaluate import build_model, load_weights
from utils.device import get_device
from utils.seed import set_seed


# ---------------------------------------------------------------------
# Visualization settings
# ---------------------------------------------------------------------

DPI = 300

VARIANTS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------
# Visualization enhancement
# ---------------------------------------------------------------------

def _enhance_grayscale_for_display(
    img: np.ndarray,
    contrast: float = 1.15,
    sharpness: float = 1.30,
    unsharp_percent: int = 100,
    unsharp_radius: float = 1.0,
    unsharp_threshold: int = 2,
) -> np.ndarray:
    """
    Enhance a grayscale image ONLY for visualization.

    IMPORTANT:
        This function is never used for metric calculation.
        Therefore, MSE / MAE / SSIM / LPIPS remain unchanged.

    Parameters
    ----------
    img:
        Grayscale image in [0, 1].

    contrast:
        Mild contrast enhancement.

    sharpness:
        Mild sharpness enhancement.

    unsharp_percent:
        Strength of edge enhancement.

    unsharp_radius:
        Radius used by UnsharpMask.

    unsharp_threshold:
        Threshold below which sharpening is not applied.

    Returns
    -------
    np.ndarray
        Enhanced grayscale image in [0, 1].
    """

    # Safety: keep original image in valid range
    img = np.clip(img, 0.0, 1.0)

    # Convert to uint8 for PIL processing
    img_uint8 = (
        img * 255.0
    ).round().astype(np.uint8)

    pil_img = Image.fromarray(
        img_uint8,
        mode="L"
    )

    # -------------------------------------------------------------
    # Mild contrast enhancement
    # -------------------------------------------------------------

    pil_img = ImageEnhance.Contrast(
        pil_img
    ).enhance(contrast)

    # -------------------------------------------------------------
    # Mild sharpness enhancement
    # -------------------------------------------------------------

    pil_img = ImageEnhance.Sharpness(
        pil_img
    ).enhance(sharpness)

    # -------------------------------------------------------------
    # Mild edge enhancement
    # -------------------------------------------------------------

    pil_img = pil_img.filter(
        ImageFilter.UnsharpMask(
            radius=unsharp_radius,
            percent=unsharp_percent,
            threshold=unsharp_threshold,
        )
    )

    # Back to [0, 1]
    enhanced = (
        np.asarray(pil_img)
        .astype(np.float32)
        / 255.0
    )

    return np.clip(
        enhanced,
        0.0,
        1.0
    )


# ---------------------------------------------------------------------
# Image display
# ---------------------------------------------------------------------

def _imshow(
    ax,
    frame,
    enhance: bool = True,
):
    """
    Display a frame.

    Enhancement is ONLY for visualization.
    The original tensor remains untouched.
    """

    img = frame.detach().cpu()

    # -------------------------------------------------------------
    # Grayscale image
    # -------------------------------------------------------------

    if img.shape[0] == 1:

        arr = img[0].numpy()

        if enhance:
            arr = _enhance_grayscale_for_display(
                arr
            )

        ax.imshow(
            arr,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

    # -------------------------------------------------------------
    # RGB image
    # -------------------------------------------------------------

    else:

        arr = (
            img
            .permute(1, 2, 0)
            .clamp(0, 1)
            .numpy()
        )

        ax.imshow(
            arr,
            interpolation="nearest",
        )

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------
# Main visualization routine
# ---------------------------------------------------------------------

def run_demo(
    config_path: str,
    ckpt_dir: str,
    n_samples: int = 3,
    device_str: str | None = None,
):

    # -----------------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------------

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        cfg = yaml.safe_load(f)

    # Reproducibility
    set_seed(
        cfg["training"]["seed"],
        deterministic=True
    )

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = (
        torch.device(device_str)
        if device_str
        else get_device()
    )

    print(f"device: {device}")

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------

    if cfg.get("dataset") == "ucf101":

        from datasets.ucf101 import UCF101

        mv = cfg.get(
            "max_videos_per_split"
        )

        test_set = UCF101(
            "test",
            root=cfg["root"],
            split_dir=cfg["split_dir"],
            max_videos=mv,
        )

        print(
            f"ucf101 test videos: "
            f"{len(test_set)}"
        )

    else:

        _, _, test_set = get_splits(
            cfg["raw_path"]
        )

        print(
            f"moving-mnist test samples: "
            f"{len(test_set)}"
        )

    # -----------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------

    out_dir = Path(ckpt_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------------------
    # Load all variants
    # -----------------------------------------------------------------

    models = {}

    for v in VARIANTS:

        ckpt = (
            out_dir
            / f"ablation_{v}"
            / "best.pth"
        )

        if not ckpt.exists():

            print(
                f"checkpoint missing for "
                f"{v}: {ckpt} - skipping"
            )

            continue

        # Build model
        model = build_model(
            cfg,
            v
        )

        # Load checkpoint
        model, epoch = load_weights(
            model,
            str(ckpt)
        )

        # Evaluation mode
        model = (
            model
            .to(device)
            .eval()
        )

        models[v] = model

        print(
            f"loaded {v} "
            f"(epoch {epoch})"
        )

    if not models:

        raise FileNotFoundError(
            "no variant checkpoints found"
        )

    # -----------------------------------------------------------------
    # Process samples
    # -----------------------------------------------------------------

    for s in range(n_samples):

        print(
            f"\nprocessing sample {s}..."
        )

        # -------------------------------------------------------------
        # Load sequence
        # -------------------------------------------------------------

        seq = (
            test_set[s]
            .unsqueeze(0)
            .to(device)
        )

        # [1, 20, C, H, W]
        inp = seq[:, 0:10]
        tgt = seq[:, 10:20]

        # =============================================================
        # FIGURE 1
        # Input context
        # =============================================================

        fig, axes = plt.subplots(
            1,
            10,
            figsize=(16, 2.0)
        )

        axes = np.atleast_1d(axes)

        for t in range(10):

            _imshow(
                axes[t],
                inp[0, t],
                enhance=True,
            )

            axes[t].set_title(
                f"t{t + 1}",
                fontsize=9,
                pad=4,
            )

        fig.suptitle(
            f"Sample {s}: observed input frames",
            fontsize=13,
            y=1.02,
        )

        plt.subplots_adjust(
            left=0.02,
            right=0.995,
            top=0.82,
            bottom=0.02,
            wspace=0.08,
        )

        fig.savefig(
            out_dir
            / f"input_frames_sample{s}.png",
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.08,
        )

        plt.close(fig)

        # =============================================================
        # Generate predictions ONCE
        # =============================================================

        predictions = {}

        for v, model in models.items():

            with torch.no_grad():

                pred = model(inp)[0]

            # [10, C, H, W]
            predictions[v] = pred.detach()

        # =============================================================
        # FIGURE 2
        # Ground truth + predictions
        # =============================================================

        n_rows = 1 + len(models)

        fig, axes = plt.subplots(
            n_rows,
            10,
            figsize=(16, 7.5),
        )

        axes = np.atleast_2d(
            axes
        )

        # -------------------------------------------------------------
        # Ground Truth row
        # -------------------------------------------------------------

        for t in range(10):

            _imshow(
                axes[0, t],
                tgt[0, t],
                enhance=True,
            )

            axes[0, t].set_title(
                f"t+{t + 1}",
                fontsize=9,
                pad=5,
            )

        axes[0, 0].set_ylabel(
            "GT",
            fontsize=11,
            rotation=90,
            labelpad=8,
        )

        # -------------------------------------------------------------
        # Variant rows
        # -------------------------------------------------------------

        for r, v in enumerate(
            models.keys(),
            start=1
        ):

            pred = predictions[v]

            for t in range(10):

                _imshow(
                    axes[r, t],
                    pred[t],
                    enhance=True,
                )

            axes[r, 0].set_ylabel(
                v,
                fontsize=11,
                rotation=90,
                labelpad=8,
            )

        # -------------------------------------------------------------
        # Figure title
        # -------------------------------------------------------------

        fig.suptitle(
            f"Sample {s}: ground truth vs variant predictions",
            fontsize=14,
            y=0.985,
        )

        # -------------------------------------------------------------
        # Layout
        # -------------------------------------------------------------

        plt.subplots_adjust(
            left=0.045,
            right=0.995,
            top=0.91,
            bottom=0.025,
            wspace=0.08,
            hspace=0.18,
        )

        # -------------------------------------------------------------
        # Save
        # -------------------------------------------------------------

        output_path = (
            out_dir
            / f"predictions_sample{s}.png"
        )

        fig.savefig(
            output_path,
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.08,
        )

        plt.close(fig)

        # =============================================================
        # FIGURE 3
        # Error maps
        # =============================================================

        fig, axes = plt.subplots(
            len(models),
            10,
            figsize=(
                16,
                1.9 * len(models)
            ),
        )

        axes = np.atleast_2d(
            axes
        )

        im = None

        for r, v in enumerate(
            models.keys()
        ):

            pred = predictions[v]

            # Absolute error
            err = (
                pred - tgt[0]
            ).abs().mean(dim=1).cpu()

            # Same scale across all horizons
            vmax = float(
                err.max()
            )

            if vmax <= 0:
                vmax = 1.0

            for t in range(10):

                im = axes[r, t].imshow(
                    err[t],
                    cmap="inferno",
                    vmin=0,
                    vmax=vmax,
                    interpolation="nearest",
                )

                axes[r, t].set_xticks([])
                axes[r, t].set_yticks([])

            axes[r, 0].set_ylabel(
                v,
                fontsize=10,
                rotation=90,
                labelpad=8,
            )

        if im is not None:

            fig.colorbar(
                im,
                ax=axes,
                fraction=0.02,
                pad=0.015,
            )

        fig.suptitle(
            f"Sample {s}: |pred - GT| error maps",
            fontsize=13,
            y=0.99,
        )

        plt.subplots_adjust(
            left=0.045,
            right=0.94,
            top=0.88,
            bottom=0.03,
            wspace=0.08,
            hspace=0.18,
        )

        fig.savefig(
            out_dir
            / f"errors_sample{s}.png",
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.08,
        )

        plt.close(fig)

        print(
            f"sample {s}: saved"
        )

    print("\ndone.")


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )

    ap.add_argument(
        "--variant-dir",
        required=True,
        help="Directory containing ablation_A/B/C/D.",
    )

    ap.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of test samples to visualize.",
    )

    ap.add_argument(
        "--device",
        default=None,
        help="Optional device, e.g. cuda or cpu.",
    )

    args = ap.parse_args()

    run_demo(
        args.config,
        args.variant_dir,
        args.samples,
        args.device,
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()