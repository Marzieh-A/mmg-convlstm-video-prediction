"""STAGE 17 tests - profiler produces finite measured values."""

from __future__ import annotations

import math

import torch

from models.ablations import AblationConfig, AblationModel
from utils.profiling import count_parameters, measure_inference


def test_parameter_count():
    torch.manual_seed(0)
    m = AblationModel(AblationConfig(variant="D"))
    c = count_parameters(m)
    assert c["total_parameters"] > 0
    assert c["trainable_parameters"] <= c["total_parameters"]


def test_inference_measurement_finite():
    torch.manual_seed(0)
    m = AblationModel(AblationConfig(variant="A"))
    rep = measure_inference(m, (1, 10, 1, 32, 32), n_warmup=1, n_runs=2)
    assert math.isfinite(rep["inference_time_mean_s"])
    assert rep["inference_time_mean_s"] > 0
    assert rep["throughput_sequences_per_s"] > 0
    assert rep["n_runs"] == 2