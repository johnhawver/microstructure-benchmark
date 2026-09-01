"""Per-stage latency benchmarks (features, XGBoost, LSTM inference)."""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from src.config import BAR_MS
from src.data_io import add_mid_and_spread, load_mbp1
from src.features import FEATURES, build_features_from_mbp

# vol_200 needs 200 bars @ 100 ms = 20 s of history before a window is valid.
_LOOKBACK_BARS = 200
_BARS_PER_SECOND = 1000 // BAR_MS


def _percentile_us(times_s: np.ndarray) -> dict[str, float]:
    """Return p50/p95/p99 latency in microseconds from wall times in seconds."""
    if times_s.size == 0:
        nan = float("nan")
        return {"p50_us": nan, "p95_us": nan, "p99_us": nan}
    us = times_s * 1e6
    return {
        "p50_us": float(np.percentile(us, 50)),
        "p95_us": float(np.percentile(us, 95)),
        "p99_us": float(np.percentile(us, 99)),
    }


def benchmark_feature_compute(
    df: pl.DataFrame,
    n_warmup: int = 100,
    n_iters: int = 1000,
    bar_ms: int = BAR_MS,
) -> dict[str, Any]:
    """Time feature pipeline on rolling 1-second MBP windows.

    ``df`` must be event-level MBP-1 with ``ts_event`` (quotes added via
    ``add_mid_and_spread`` before calling). Reports per-bar latency in the
    last 1 s of each window (microseconds).
    """
    if "bid" not in df.columns or "ask" not in df.columns:
        raise ValueError("df must include bid/ask — run add_mid_and_spread first")

    sorted_df = df.sort("ts_event")
    if sorted_df.height < 10_000:
        raise ValueError("df too small for rolling benchmark — need ~10k+ events")

    # Anchor grid from a full pass on a 30-minute slice (enough for warmup + iters).
    need_bars = n_warmup + n_iters + _LOOKBACK_BARS + _BARS_PER_SECOND
    need_seconds = need_bars * bar_ms / 1000.0
    t0 = sorted_df["ts_event"].min()
    t_end = t0 + timedelta(seconds=need_seconds + 30)
    slice_df = sorted_df.filter(pl.col("ts_event") <= t_end)
    mbp_full = add_mid_and_spread(slice_df.lazy()) if "mid" not in slice_df.columns else slice_df.lazy()

    ref = build_features_from_mbp(mbp_full, bar_ms=bar_ms)
    if ref.height < need_bars:
        raise ValueError(
            f"Not enough bars ({ref.height}) for {need_bars} required — widen time slice"
        )

    bar_ts = ref["ts_event"].to_list()
    lookback_td = timedelta(milliseconds=_LOOKBACK_BARS * bar_ms)
    one_sec_td = timedelta(milliseconds=_BARS_PER_SECOND * bar_ms)

    times_per_bar: list[float] = []
    for i in range(n_warmup, n_warmup + n_iters):
        win_end = bar_ts[i + _LOOKBACK_BARS]
        win_start = win_end - lookback_td - one_sec_td
        mbp_win = mbp_full.filter(
            (pl.col("ts_event") >= win_start) & (pl.col("ts_event") <= win_end)
        )
        t_start = time.perf_counter()
        out = build_features_from_mbp(mbp_win, bar_ms=bar_ms)
        elapsed = time.perf_counter() - t_start

        tail = out.filter(pl.col("ts_event") > win_end - one_sec_td)
        n_bars = max(int(tail.height), 1)
        times_per_bar.append(elapsed / n_bars)

    stats = _percentile_us(np.asarray(times_per_bar))
    stats["n_iters"] = int(n_iters)
    stats["stage"] = "feature_compute"
    return stats


def benchmark_xgb_inference(
    booster: xgb.Booster,
    X: np.ndarray,
    n_warmup: int = 100,
    n_iters: int = 10_000,
) -> dict[str, Any]:
    """Single-row ``booster.predict`` latency (microseconds)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(FEATURES):
        raise ValueError(f"X must be [N, {len(FEATURES)}], got {X.shape}")
    if X.shape[0] < 1:
        raise ValueError("X must have at least one row")

    row = X[0:1]
    kwargs: dict[str, Any] = {}
    if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
        kwargs["iteration_range"] = (0, booster.best_iteration + 1)

    times: list[float] = []
    for _ in range(n_warmup + n_iters):
        t0 = time.perf_counter()
        booster.predict(xgb.DMatrix(row), **kwargs)
        times.append(time.perf_counter() - t0)

    stats = _percentile_us(np.asarray(times[n_warmup:]))
    stats["n_iters"] = int(n_iters)
    stats["stage"] = "xgb_inference"
    return stats


def benchmark_lstm_inference(
    model: Any,
    X: Any,
    n_warmup: int = 100,
    n_iters: int = 10_000,
) -> dict[str, Any]:
    """Single-sequence LSTM forward pass latency (microseconds).

    ``X`` shape ``[1, seq_len, n_features]`` (batch_first).
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)
    X = X.to(device)
    if X.ndim != 3 or X.shape[0] != 1:
        raise ValueError(f"X must be [1, seq_len, n_features], got {tuple(X.shape)}")

    times: list[float] = []
    with torch.inference_mode():
        for _ in range(n_warmup + n_iters):
            t0 = time.perf_counter()
            model(X)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    stats = _percentile_us(np.asarray(times[n_warmup:]))
    stats["n_iters"] = int(n_iters)
    stats["stage"] = "lstm_inference"
    return stats


def load_mbp_benchmark_slice(day: str = "2025-09-15", minutes: int = 45) -> pl.DataFrame:
    """Load a contiguous MBP-1 slice for feature benchmarking."""
    from datetime import date, datetime, timedelta, timezone

    d1 = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    start = f"{day}T14:00:00"
    end_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) + timedelta(
        minutes=minutes
    )
    end = end_dt.isoformat()
    return (
        add_mid_and_spread(load_mbp1(day, d1))
        .filter(
            (pl.col("ts_event") >= pl.lit(start).str.to_datetime(time_zone="UTC"))
            & (pl.col("ts_event") < pl.lit(end).str.to_datetime(time_zone="UTC"))
        )
        .collect()
    )
