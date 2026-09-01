"""Tests for walk-forward splitter + embargo."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from src.config import BAR_MS
from src.splits import WalkForwardSplitter


def _synth_bars(
    n_days: int = 15,
    bars_per_day: int = 200,
    start: date | None = None,
) -> pl.DataFrame:
    """Contiguous 100ms bars spanning ``n_days`` calendar sessions (weekdays only)."""
    start = start or date(2025, 9, 15)
    rows_ts: list[datetime] = []
    d = start
    days_made = 0
    while days_made < n_days:
        if d.weekday() < 5:  # Mon–Fri
            t0 = datetime(d.year, d.month, d.day, 14, 0, 0, tzinfo=timezone.utc)
            for k in range(bars_per_day):
                rows_ts.append(t0 + timedelta(milliseconds=BAR_MS * k))
            days_made += 1
        d += timedelta(days=1)

    return pl.DataFrame(
        {
            "ts_event": rows_ts,
            "mid": np.linspace(100.0, 101.0, len(rows_ts)),
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", "UTC")))


@pytest.fixture
def splitter() -> WalkForwardSplitter:
    return WalkForwardSplitter(
        n_train_days=10,
        n_test_days=1,
        embargo_bars=100,
        step_days=1,
    )


def test_train_test_disjoint(splitter: WalkForwardSplitter):
    df = _synth_bars(n_days=15, bars_per_day=200)
    for train_idx, test_idx in splitter.split(df):
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert np.intersect1d(train_idx, test_idx).size == 0


def test_embargo_respected(splitter: WalkForwardSplitter):
    """max(train_ts) + embargo_seconds <= min(test_ts) for every fold."""
    df = _synth_bars(n_days=15, bars_per_day=200)
    sorted_df = df.sort("ts_event")
    ts = sorted_df["ts_event"].to_numpy()

    for train_idx, embargo_idx, test_idx in splitter.split_with_embargo(df):
        assert embargo_idx.size == splitter.embargo_bars
        assert np.intersect1d(train_idx, embargo_idx).size == 0
        assert np.intersect1d(embargo_idx, test_idx).size == 0

        train_ts = ts[train_idx]
        test_ts = ts[test_idx]
        gap = (test_ts.min() - train_ts.max()) / np.timedelta64(1, "s")
        assert gap >= splitter.embargo_seconds - 1e-9


def test_test_folds_cover_non_warmup_exactly_once(splitter: WalkForwardSplitter):
    df = _synth_bars(n_days=15, bars_per_day=50)
    sorted_df = df.sort("ts_event")
    day_col = sorted_df.select(pl.col("ts_event").dt.date().alias("d"))["d"]

    non_warmup = splitter.testable_days(df)
    assert len(non_warmup) == 5  # 15 - 10

    seen: list = []
    for _train_idx, test_idx in splitter.split(df):
        test_days = sorted({day_col[int(i)] for i in test_idx})
        assert len(test_days) == splitter.n_test_days
        seen.extend(test_days)

    assert seen == non_warmup
    assert len(seen) == len(set(seen))


def test_embargo_bars_stripped_from_train(splitter: WalkForwardSplitter):
    """Last embargo_bars of the raw train window are not in train_idx."""
    bars_per_day = 200
    df = _synth_bars(n_days=12, bars_per_day=bars_per_day)
    train_idx, embargo_idx, test_idx = next(splitter.split_with_embargo(df))

    # Fold 0: 10 train days × 200 bars, then drop 100 → 1900 train rows.
    assert train_idx.size == 10 * bars_per_day - splitter.embargo_bars
    assert embargo_idx.size == splitter.embargo_bars
    assert test_idx.size == bars_per_day
    # Embargo sits immediately after train in index space (sorted frame).
    assert embargo_idx.min() == train_idx.max() + 1
    assert test_idx.min() > embargo_idx.max()


def test_n_folds_default_window(splitter: WalkForwardSplitter):
    df = _synth_bars(n_days=15, bars_per_day=20)
    assert splitter.n_folds(df) == 5
    assert sum(1 for _ in splitter.split(df)) == 5
