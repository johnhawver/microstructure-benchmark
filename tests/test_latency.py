"""Tests for latency benchmarks."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from src.latency import _percentile_us, benchmark_xgb_inference
from src.models.xgb_model import DEFAULT_PARAMS


def test_percentile_us():
    times = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    stats = _percentile_us(times)
    assert stats["p50_us"] == 3000.0
    assert stats["p95_us"] >= stats["p50_us"]
    assert stats["p99_us"] >= stats["p95_us"]


def test_benchmark_xgb_inference_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 12))
    y = rng.integers(0, 3, size=100)
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(
        DEFAULT_PARAMS,
        dtrain,
        num_boost_round=5,
        verbose_eval=False,
    )
    stats = benchmark_xgb_inference(booster, X, n_warmup=5, n_iters=20)
    assert stats["stage"] == "xgb_inference"
    assert stats["p50_us"] > 0
    assert stats["p99_us"] >= stats["p50_us"]
