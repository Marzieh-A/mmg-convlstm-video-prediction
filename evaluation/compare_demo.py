"""
STAGE 16d - Side-by-side prediction comparison:
MSE model vs warmup+composite model on the same test samples.

Important:
    Image enhancement is applied ONLY for visualization.
    The original model predictions are not modified.
    Therefore, MSE / MAE / SSIM / LPIPS remain unchanged.

Usage (from project root):

    python -m evaluation.compare_demo \
        --config configs/moving_mnist.yaml \
        --ckpt-mse outputs/moving_mnist_mse/ablation_D/best.pth \
        --ckpt-composite outputs/moving_mnist_warmup_run1/latest.pth \
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

from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter

import torch
import yaml


# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)


from datasets.moving_mnist import get_splits
from evaluation.evaluate import build_model
from utils.device import get_device
from utils.seed import set_seed


# ---------------------------------------------------------------------
# Visualization settings
# ---------------------------------------------------------------------

DPI = 300


# ---------------------------------------------------------------------
# Image enhancement
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
    Enhance a grayscale frame ONLY for visualization.

    This function does NOT modify the model output used for
    quantitative evaluation.
    """

    # Keep image in [0, 1]
    img = np.clip(
        img,
        0.0,
        1.0
    )

    # Convert to 8-bit
    img_uint8 = (
        img * 255.0
    ).round().astype(np.uint8)

    # PIL image
    pil_img = Image.fromarray(
        img_uint8,
        mode="L"
    )

    # -------------------------------------------------------------
    # Contrast
    # -------------------------------------------------------------

    pil_img = ImageEnhance.Contrast(
        pil_img
    ).enhance(contrast)

    # -------------------------------------------------------------
    # Sharpness
    # -------------------------------------------------------------

    pil_img = ImageEnhance.Sharpness(
        pil_img
    ).enhance(sharpness)

    # -------------------------------------------------------------
    # Edge enhancement
    # -------------------------------------------------------------

    pil_img = pil_img.filter(
        ImageFilter.UnsharpMask(
            radius=unsharp_radius,
            percent=unsharp_percent,
            threshold=unsharp_threshold
        )
    )

    # Back to numpy [0, 1]
    result = (
        np.asarray(pil_img)
        .astype(np.float32)
        / 255.0
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ---------------------------------------------------------------------
# Display one frame
# ---------------------------------------------------------------------

def _imshow(
    ax,
    frame,
    enhance: bool = True
):
    """
    Display a frame with optional visualization enhancement.

    IMPORTANT:
        Enhancement is applied only to the image shown on the plot.
        The original tensor remains unchanged.
    """

    img = frame.detach().cpu()

    # -------------------------------------------------------------
    # Grayscale
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
            interpolation="nearest"
        )

    # -------------------------------------------------------------
    # RGB
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
            interpolation="nearest"
        )

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    # =============================================================
    # Arguments
    # =============================================================

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True
    )

    ap.add_argument(
        "--ckpt-mse",
        required=True
    )

    ap.add_argument(
        "--ckpt-composite",
        required=True
    )

    ap.add_argument(
        "--samples",
        type=int,
        default=3
    )

    args = ap.parse_args()

    # =============================================================
    # Configuration
    # =============================================================

    with open(
        args.config,
        encoding="utf-8"
    ) as f:

        cfg = yaml.safe_load(f)

    # Reproducibility
    set_seed(
        cfg["training"]["seed"],
        deterministic=True
    )

    # Device
    device = get_device()

    print(
        f"device: {device}"
    )

    # =============================================================
    # Load MSE model
    # =============================================================

    print(
        "\nLoading MSE model..."
    )

    m_mse = build_model(
        cfg,
        "D"
    )

    ck = torch.load(
        args.ckpt_mse,
        map_location="cpu",
        weights_only=False
    )

    m_mse.load_state_dict(
        ck["model_state"],
        strict=True
    )

    m_mse = (
        m_mse
        .to(device)
        .eval()
    )

    print(
        "MSE model loaded."
    )

    # =============================================================
    # Load composite model
    # =============================================================

    print(
        "Loading warmup+composite model..."
    )

    m_comp = build_model(
        cfg,
        "D"
    )

    ck = torch.load(
        args.ckpt_composite,
        map_location="cpu",
        weights_only=False
    )

    m_comp.load_state_dict(
        ck["model_state"],
        strict=True
    )

    m_comp = (
        m_comp
        .to(device)
        .eval()
    )

    print(
        "Warmup+composite model loaded."
    )

    # =============================================================
    # Dataset
    # =============================================================

    _, _, test_set = get_splits(
        cfg["raw_path"]
    )

    print(
        f"test samples: {len(test_set)}"
    )

    # =============================================================
    # Output directory
    # =============================================================

    out_dir = Path(
        "outputs/compare_demo"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =============================================================
    # Process samples
    # =============================================================

    for s in range(args.samples):

        print(
            f"\nProcessing sample {s}..."
        )

        # ---------------------------------------------------------
        # Load sequence
        # ---------------------------------------------------------

        seq = (
            test_set[s]
            .unsqueeze(0)
            .to(device)
        )

        # [1, 10, C, H, W]
        inp = seq[:, 0:10]

        # [1, 10, C, H, W]
        tgt = seq[:, 10:20]

        # ---------------------------------------------------------
        # Generate predictions
        # ---------------------------------------------------------

        with torch.no_grad():

            p1 = m_mse(inp)[0]

            p2 = m_comp(inp)[0]

        # =========================================================
        # Create figure
        # =========================================================

        fig, axes = plt.subplots(
            3,
            10,
            figsize=(16, 5.4)
        )

        # Make sure axes is 2D
        axes = np.asarray(
            axes
        )

        # =========================================================
        # Ground Truth
        # =========================================================

        for t in range(10):

            _imshow(
                axes[0, t],
                tgt[0, t],
                enhance=True
            )

            axes[0, t].set_title(
                f"$t+{t + 1}$",
                fontsize=10,
                pad=6
            )

        # =========================================================
        # D - MSE
        # =========================================================

        for t in range(10):

            _imshow(
                axes[1, t],
                p1[t],
                enhance=True
            )

        # =========================================================
        # D - Warmup + Composite
        # =========================================================

        for t in range(10):

            _imshow(
                axes[2, t],
                p2[t],
                enhance=True
            )

        # =========================================================
        # Row labels
        # =========================================================

        axes[0, 0].set_ylabel(
            "GT",
            fontsize=11,
            rotation=90,
            labelpad=10
        )

        axes[1, 0].set_ylabel(
            "D (MSE)",
            fontsize=11,
            rotation=90,
            labelpad=10
        )

        axes[2, 0].set_ylabel(
            "D (Warmup + Composite)",
            fontsize=11,
            rotation=90,
            labelpad=10
        )

        # =========================================================
        # Main title
        # =========================================================

        fig.suptitle(
            f"Sample {s}: comparison of loss functions",
            fontsize=14,
            y=0.985
        )

        # =========================================================
        # Layout
        # =========================================================

        plt.subplots_adjust(
            left=0.075,
            right=0.995,
            top=0.90,
            bottom=0.035,
            wspace=0.08,
            hspace=0.20
        )

        # =========================================================
        # Save high-resolution image
        # =========================================================

        output_path = (
            out_dir
            / f"loss_compare_sample{s}.png"
        )

        fig.savefig(
            output_path,
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.08
        )

        plt.close(fig)

        print(
            f"sample {s}: saved -> {output_path}"
        )

    print(
        "\nDone."
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()