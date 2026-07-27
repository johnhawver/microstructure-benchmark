"""Tests for bar resampling and features."""

from __future__ import annotations

import polars as pl

from src.config import BAR_MS
from src.data_io import add_mid_and_spread, load_mbp1
from src.features import resample_to_bars


def test_resample_to_bars_one_hour():
    """One hour of 100ms bars should be ~36k rows with non-null mid."""
    lf = add_mid_and_spread(load_mbp1("2025-09-15", "2025-09-16")).filter(
        (pl.col("ts_event") >= pl.lit("2025-09-15T14:00:00").str.to_datetime(time_zone="UTC"))
        & (pl.col("ts_event") < pl.lit("2025-09-15T15:00:00").str.to_datetime(time_zone="UTC"))
    )
    bars = resample_to_bars(lf, BAR_MS)
    n = bars.select(pl.len()).collect().item()
    null_mid = bars.select(pl.col("mid").null_count()).collect().item()

    assert 30_000 <= n <= 42_000  # ~36_000; allow gaps / sparse buckets
    assert null_mid == 0