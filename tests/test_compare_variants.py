"""STAGE 16b tests - comparison figures and table."""

from __future__ import annotations

import json

from evaluation.compare_variants import (
    plot_metric_curves, plot_overall_bars, write_comparison_table,
)


def _make_eval(tmp_path, v):
    data = {
        "mse": 0.027, "mae": 0.061, "ssim": 0.64, "lpips": 0.38,
        "horizon_mse": [0.01 + 0.003 * i for i in range(10)],
        "horizon_mae": [0.03 + 0.005 * i for i in range(10)],
        "horizon_ssim": [0.88 - 0.04 * i for i in range(10)],
        "horizon_lpips": [0.10 + 0.04 * i for i in range(10)],
    }
    (tmp_path / f"eval_{v}.json").write_text(json.dumps(data))


def test_curves_and_bars_and_table(tmp_path):
    for v in "ABCD":
        _make_eval(tmp_path, v)
    for m in ("mse", "mae", "ssim", "lpips"):
        p = plot_metric_curves(tmp_path, m, tmp_path)
        assert p is not None and p.exists() and p.stat().st_size > 1000
    p = plot_overall_bars(tmp_path, tmp_path)
    assert p.exists() and p.stat().st_size > 1000
    tbl = write_comparison_table(tmp_path, tmp_path / "comparison_table.md")
    text = tbl.read_text()
    assert "| A |" in text and "| D |" in text
    assert text.count("|") > 40   # full grid present


def test_missing_variant_is_skipped(tmp_path):
    for v in "ABC":
        _make_eval(tmp_path, v)
    p = plot_metric_curves(tmp_path, "ssim", tmp_path)   # D missing
    assert p is not None and p.exists()