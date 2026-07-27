"""Tests for MBP-1 loader and mid/spread helpers."""

from __future__ import annotations

import polars as pl

from src.data_io import add_mid_and_spread, load_mbp1


def test_mid_and_spread_sanity():
    """One day of cleaned data should have positive mid and sane spreads."""
    lf = add_mid_and_spread(load_mbp1("2025-09-15", "2025-09-16"))
    stats = lf.select(
        pl.len().alias("n"),
        pl.col("mid").min().alias("mid_min"),
        pl.col("spread").min().alias("spread_min"),
        pl.col("spread").max().alias("spread_max"),
    ).collect()

    assert stats["n"][0] > 0
    assert stats["mid_min"][0] > 0
    assert stats["spread_min"][0] >= 0
    assert stats["spread_max"][0] < 50


def test_filter_rth():
    """RTH on an EDT day should keep only [13:30, 20:00) UTC."""
    from datetime import datetime, timezone

    from src.data_io import filter_rth

    lo = datetime(2025, 9, 15, 13, 30, tzinfo=timezone.utc)
    hi = datetime(2025, 9, 15, 20, 0, tzinfo=timezone.utc)

    lf = filter_rth(load_mbp1("2025-09-15", "2025-09-16"))
    outside = (
        lf.filter((pl.col("ts_event") < lo) | (pl.col("ts_event") >= hi))
        .select(pl.len())
        .collect()
        .item()
    )

    assert outside == 0
    assert lf.select(pl.len()).collect().item() > 0


