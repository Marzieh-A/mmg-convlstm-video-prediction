"""STAGE 16c test - demo pipeline on a tiny synthetic setup (no checkpoints)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.ablations import AblationConfig, AblationModel


def test_grid_rendering(tmp_path):
    """The rendering part of the demo (without checkpoints) works headless."""
    torch.manual_seed(0)
    m = AblationModel(AblationConfig(variant="A")).eval()
    x = torch.rand(1, 10, 1, 32, 32)
    with torch.no_grad():
        pred = m(x)[0]
    fig, axes = plt.subplots(2, 10, figsize=(14, 3))
    for t in range(10):
        axes[0, t].imshow(pred[t, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, t].imshow(pred[t, 0], cmap="inferno")
        axes[0, t].axis("off"); axes[1, t].axis("off")
    p = tmp_path / "grid.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    assert p.exists() and p.stat().st_size > 1000