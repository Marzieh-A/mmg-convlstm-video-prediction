"""STAGE 16 tests - all figures are produced, saved, non-empty, headless."""

from __future__ import annotations

import json

import torch

from evaluation.visualize import (
    make_horizon_table_markdown, plot_all_horizons, plot_error_maps,
    plot_frame_grid, plot_loss_curves, plot_sequence,
)


def _seq(T=10, C=1):
    return torch.rand(T, C, 64, 64)


def test_frame_grid_and_sequence_and_errors(tmp_path):
    gt, pr = _seq(), _seq()
    p1 = plot_frame_grid(gt, pr, tmp_path)
    p2 = plot_sequence(pr, tmp_path)
    p3 = plot_error_maps(gt, pr, tmp_path)
    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 1000


def test_rgb_frames(tmp_path):
    gt, pr = _seq(C=3), _seq(C=3)
    p = plot_frame_grid(gt, pr, tmp_path)
    assert p.exists()


def test_loss_curves(tmp_path):
    hist = [{"epoch": e, "train_loss": 0.1 / (e + 1), "val_loss": 0.12 / (e + 1)}
            for e in range(5)]
    p = plot_loss_curves(hist, tmp_path)
    assert p.exists() and p.stat().st_size > 1000


def test_horizon_plots_and_table(tmp_path):
    data = {
        "horizon_mse": [0.1 - 0.001 * i for i in range(10)],
        "horizon_mae": [0.05 + 0.001 * i for i in range(10)],
        "horizon_ssim": [0.8 - 0.01 * i for i in range(10)],
        "horizon_lpips": [0.2 + 0.02 * i for i in range(10)],
        "overall_mse": 0.0955, "overall_mae": 0.0545,
        "overall_ssim": 0.755, "overall_lpips": 0.29,
    }
    jp = tmp_path / "eval.json"
    jp.write_text(json.dumps(data))
    paths = plot_all_horizons(jp, tmp_path)
    assert len(paths) == 4
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
    tbl = make_horizon_table_markdown(jp, tmp_path / "table.md")
    text = tbl.read_text()
    assert "t+1" in text and "Overall" in text and "LPIPS" in text