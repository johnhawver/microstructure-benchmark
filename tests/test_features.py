"""Tests for bar resampling and features."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from src.config import BAR_MS
from src.data_io import add_mid_and_spread, load_mbp1
from src.features import (
    add_microprice_tilt,
    add_returns,
    add_rolling_vol,
    resample_to_bars,
)


def test_resample_to_bars_one_hour():
    """One hour of 100ms bars should be ~36k rows with non-null mid."""
    lf = add_mid_and_spread(load_mbp1("2025-09-15", "2025-09-16")).filter(
        (pl.col("ts_event") >= pl.lit("2025-09-15T14:00:00").str.to_datetime(time_zone="UTC"))
        & (pl.col("ts_event") < pl.lit("2025-09-15T15:00:00").str.to_datetime(time_zone="UTC"))
    )
    bars = resample_to_bars(lf, BAR_MS)
    n = bars.select(pl.len()).collect().item()
    null_mid = bars.select(pl.col("mid").null_count()).collect().item()

    assert 30_000 <= n <= 42_000
    assert null_mid == 0


def test_vol_and_tilt_on_synthetic_random_walk():
    """vol_50 ≈ injected sigma; microprice_tilt stays in [-1, 1]."""
    rng = np.random.default_rng(0)
    n = 5_000
    sigma = 1e-3
    shocks = rng.normal(0.0, sigma, size=n)
    mid = 100.0 * np.exp(np.cumsum(shocks))
    spread = np.full(n, 0.25)
    w = rng.uniform(0.0, 1.0, size=n)
    bid = mid - spread / 2.0
    ask = mid + spread / 2.0
    microprice = w * ask + (1.0 - w) * bid

    start = datetime(2025, 9, 15, 14, 0, tzinfo=timezone.utc)
    ts = [start + timedelta(milliseconds=100 * i) for i in range(n)]

    lf = pl.DataFrame(
        {
            "ts_event": ts,
            "mid": mid,
            "spread": spread,
            "bid": bid,
            "ask": ask,
            "microprice": microprice,
        }
    ).lazy()

    out = add_microprice_tilt(add_rolling_vol(add_returns(lf), [50])).collect()

    vol_mean = out.select(pl.col("vol_50").drop_nulls().mean()).item()
    tilt_min = out.select(pl.col("microprice_tilt").min()).item()
    tilt_max = out.select(pl.col("microprice_tilt").max()).item()

    assert abs(vol_mean - sigma) / sigma < 0.15
    assert -1.0 <= tilt_min <= tilt_max <= 1.0