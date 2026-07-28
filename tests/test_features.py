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


from src.features import (
    add_microprice_tilt,
    add_returns,
    add_rolling_vol,
    compute_ofi_events,
    resample_to_bars,
)
def test_ofi_events_hand_rolled():
    """5-row book fixture: OFI matches Cont-Kukanov-Stoikov by hand."""
    # Row 0 = baseline (dropped). Then:
    # 1) bid size 10→15, ask unchanged → e_bid=+5, e_ask=0 → ofi=5
    # 2) bid price up, sz=8; ask size 10→12 → e_bid=+8, e_ask=+2 → ofi=6
    # 3) ask price down, sz=5; bid unchanged → e_bid=0, e_ask=+5 → ofi=-5
    # 4) bid price down (prev sz 8); ask unchanged → e_bid=-8, e_ask=0 → ofi=-8
    start = datetime(2025, 9, 15, 14, 0, tzinfo=timezone.utc)
    lf = pl.DataFrame(
        {
            "ts_event": [start + timedelta(milliseconds=100 * i) for i in range(5)],
            "bid": [100.0, 100.0, 100.25, 100.25, 100.0],
            "ask": [101.0, 101.0, 101.0, 100.75, 100.75],
            "bid_sz": [10, 15, 8, 8, 3],
            "ask_sz": [10, 10, 12, 5, 5],
        }
    ).lazy()

    out = compute_ofi_events(lf).collect()
    assert out["ofi"].to_list() == [5, 6, -5, -8]