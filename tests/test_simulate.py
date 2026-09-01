"""Tests for trading simulator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from src.simulate import Simulator, summary_stats


def test_simulator_long_then_flat_pays_spread():
    """Enter long on bar 0, flat on bar 2; spread cost reduces PnL."""
    start = datetime(2025, 9, 15, 14, 0, tzinfo=timezone.utc)
    mid = 100.0
    spread = 0.25
    n = 4
    df = pl.DataFrame(
        {
            "ts_event": [start + timedelta(milliseconds=100 * i) for i in range(n)],
            "mid": [mid, mid + 1.0, mid + 1.0, mid + 1.0],
            "bid": [mid - spread / 2] * n,
            "ask": [mid + spread / 2] * n,
            "signed_score": [0.5, 0.5, 0.0, 0.0],
        }
    )
    sim = Simulator(
        tick_size=0.25,
        tick_value=0.50,
        entry_threshold=0.2,
        fill_delay_bars=0,
        commission_per_trade=0.0,
    )
    out = sim.run(df)

    # Entry at bar 0 (no delay); exit at bar 2 when score goes flat.
    assert out["position"].to_list() == [1, 1, 0, 0]
    assert out["cum_pnl"][-1] == out["trade_pnl"].sum()
    assert out["cum_pnl"][-1] < 2.0  # spread costs bite


def test_summary_stats_keys():
    start = datetime(2025, 9, 15, 14, 0, tzinfo=timezone.utc)
    df = pl.DataFrame(
        {
            "ts_event": [start + timedelta(days=i) for i in range(3)],
            "mid": [100.0, 101.0, 100.5],
            "bid": [99.875, 100.875, 100.375],
            "ask": [100.125, 101.125, 100.625],
            "signed_score": [0.0, 0.0, 0.0],
        }
    )
    out = Simulator(fill_delay_bars=0, commission_per_trade=0.0).run(df)
    stats = summary_stats(out)
    assert {"total_pnl", "sharpe", "max_drawdown", "hit_rate", "avg_trade_duration_bars", "trade_count"} <= stats.keys()


def test_fill_delay_defers_entry():
    """With delay=1, decision at bar 0 fills at bar 1."""
    start = datetime(2025, 9, 15, 14, 0, tzinfo=timezone.utc)
    mid = 100.0
    spread = 0.25
    n = 3
    df = pl.DataFrame(
        {
            "ts_event": [start + timedelta(milliseconds=100 * i) for i in range(n)],
            "mid": [mid, mid, mid],
            "bid": [mid - spread / 2] * n,
            "ask": [mid + spread / 2] * n,
            "signed_score": [0.5, 0.5, 0.5],
        }
    )
    out = Simulator(
        entry_threshold=0.2,
        fill_delay_bars=1,
        commission_per_trade=0.0,
    ).run(df)
    assert out["position"].to_list() == [0, 1, 1]
