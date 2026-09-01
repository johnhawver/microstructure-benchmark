"""Systematic leakage audit — any failure blocks the project until fixed."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from src.config import BAR_MS, LABEL_HORIZON_BARS, PARQUET_DIR
from src.data_io import add_mid_and_spread, load_mbp1
from src.features import (
    FEATURES,
    add_autocorr,
    add_kyle_lambda,
    add_microprice_tilt,
    add_returns,
    add_rolling_vol,
    add_spread_features,
    add_trade_imbalance,
    aggregate_ofi_to_bars,
    compute_ofi_events,
    resample_to_bars,
)
from src.labels import forward_return_sign, triple_barrier
from src.models.lstm_model import FeatureScaler
from src.splits import WalkForwardSplitter
from src.walkforward import run_walkforward_lstm


def _bar_features(bars_with_ofi100: pl.LazyFrame) -> pl.DataFrame:
    """Same bar-level feature stack as build_feature_frame (after OFI join)."""
    lf = bars_with_ofi100.with_columns(pl.col("ofi_sum_100ms").fill_null(0)).with_columns(
        pl.col("ofi_sum_100ms").rolling_sum(10).alias("ofi_sum_1000ms"),
        pl.col("ofi_sum_100ms").rolling_sum(50).alias("ofi_sum_5000ms"),
    )
    lf = add_returns(lf)
    lf = add_rolling_vol(lf, [10, 50, 200])
    lf = add_autocorr(lf, lag=1, window=50)
    lf = add_microprice_tilt(lf)
    lf = add_spread_features(lf)
    lf = add_trade_imbalance(lf, [10, 50, 200])
    lf = add_kyle_lambda(lf, 50)
    return lf.select(["ts_event", *FEATURES]).collect()


def _sample_bars() -> pl.DataFrame:
    """Ten minutes of real RTH bars with OFI joined (enough for rolling windows)."""
    mbp = add_mid_and_spread(load_mbp1("2025-09-15", "2025-09-16")).filter(
        (pl.col("ts_event") >= pl.lit("2025-09-15T14:00:00").str.to_datetime(time_zone="UTC"))
        & (pl.col("ts_event") < pl.lit("2025-09-15T14:10:00").str.to_datetime(time_zone="UTC"))
    )
    return (
        resample_to_bars(mbp, BAR_MS)
        .join(aggregate_ofi_to_bars(compute_ofi_events(mbp), 100), on="ts_event", how="left")
        .collect()
    )


def test_feature_causality():
    """Each feature at t must not change when all rows after t are dropped."""
    bars = _sample_bars()
    n = min(2_000, bars.height // 2)
    assert n > 300

    full = _bar_features(bars.lazy())
    trunc = _bar_features(bars.head(n).lazy())

    matched = trunc.join(full, on="ts_event", how="inner", suffix="_full")
    assert matched.height == trunc.height

    for col in FEATURES:
        assert matched[col].equals(matched[f"{col}_full"]), (
            f"{col} changed when future bars were removed (lookahead)"
        )


def test_label_horizon():
    """Perturbing mid beyond t+horizon must not change labels at t."""
    paths = sorted(PARQUET_DIR.glob("mnq_features_*.parquet"))
    if paths:
        df = pl.read_parquet(paths[0]).head(5_000)
    else:
        rng = np.random.default_rng(42)
        n = 2_000
        mid = 20_000.0 + np.cumsum(rng.normal(0, 0.5, n))
        sigma = np.abs(rng.normal(0.001, 0.0002, n)) + 1e-6
        start = datetime(2025, 9, 15, 14, 0, tzinfo=timezone.utc)
        ts = [start + timedelta(milliseconds=100 * i) for i in range(n)]
        df = pl.DataFrame(
            {
                "ts_event": ts,
                "mid": mid,
                "vol_50": sigma,
            }
        )

    mid = df["mid"]
    sigma = df["vol_50"]
    horizon = LABEL_HORIZON_BARS

    tb_full = triple_barrier(mid, sigma, horizon, k=1.0).to_numpy()
    fr_full = forward_return_sign(mid, horizon, threshold_bps=1.0).to_numpy()

    # Perturb only rows strictly after t0 + horizon (per-bar labeling window).
    t0 = min(500, len(mid) - horizon - 2)
    mid_arr = mid.to_numpy().copy()
    mid_arr[t0 + horizon + 1 :] = mid_arr[t0] * 10.0
    perturbed = df.with_columns(pl.Series("mid", mid_arr))

    tb_pert = triple_barrier(
        perturbed["mid"], perturbed["vol_50"], horizon, k=1.0
    ).to_numpy()
    fr_pert = forward_return_sign(
        perturbed["mid"], horizon, threshold_bps=1.0
    ).to_numpy()

    assert tb_full[t0] == tb_pert[t0], (
        f"label_tb[{t0}] changed after perturbing mid>{t0 + horizon}"
    )
    assert fr_full[t0] == fr_pert[t0], (
        f"label_fr[{t0}] changed after perturbing mid>{t0 + horizon}"
    )

    # Spot-check a few earlier bars (unaffected by perturbation starting at t0+horizon+1).
    for t in range(min(10, t0)):
        assert tb_full[t] == tb_pert[t]
        assert fr_full[t] == fr_pert[t]


def test_embargo_respected():
    """max(train_ts) + embargo_seconds <= min(test_ts) on real feature data."""
    paths = sorted(PARQUET_DIR.glob("mnq_features_*.parquet"))
    if len(paths) < 12:
        pytest.skip("need >= 12 days of mnq_features_*.parquet")

    frames = [pl.read_parquet(p) for p in paths[:15]]
    df = pl.concat(frames, how="vertical_relaxed").sort("ts_event")

    splitter = WalkForwardSplitter(
        n_train_days=10, n_test_days=1, embargo_bars=100, step_days=1
    )
    sorted_df = df.sort("ts_event")
    ts = sorted_df["ts_event"].to_numpy()

    for train_idx, embargo_idx, test_idx in splitter.split_with_embargo(df):
        train_ts = ts[train_idx]
        test_ts = ts[test_idx]
        gap = (test_ts.min() - train_ts.max()) / np.timedelta64(1, "s")
        assert gap >= splitter.embargo_seconds - 1e-9
        assert embargo_idx.size == splitter.embargo_bars


def test_scaler_train_only():
    """FeatureScaler.fit must never see rows from the fold's test index."""
    bars_per_day = 120
    n_days = 12
    start = date(2025, 9, 15)
    rows_ts: list[datetime] = []
    feat0: list[float] = []
    d = start
    days_made = 0
    row_id = 0
    while days_made < n_days:
        if d.weekday() < 5:
            t0 = datetime(d.year, d.month, d.day, 14, 0, 0, tzinfo=timezone.utc)
            for _ in range(bars_per_day):
                rows_ts.append(t0)
                feat0.append(float(row_id))
                row_id += 1
                t0 += timedelta(milliseconds=BAR_MS)
            days_made += 1
        d += timedelta(days=1)

    n_rows = len(rows_ts)
    df = pl.DataFrame(
        {
            "ts_event": rows_ts,
            "mid": np.linspace(100.0, 101.0, n_rows),
            "label_tb": np.random.default_rng(0).integers(-1, 2, size=n_rows),
            **{c: np.ones(n_rows, dtype=np.float64) for c in FEATURES},
        }
    ).with_columns(
        pl.col("ts_event").cast(pl.Datetime("ns", "UTC")),
        pl.Series("microprice_tilt", feat0),  # row fingerprint in one column
    )

    splitter = WalkForwardSplitter(
        n_train_days=10, n_test_days=1, embargo_bars=100, step_days=1
    )
    sorted_df = df.sort("ts_event")
    train_idx, test_idx = next(splitter.split(sorted_df))
    forbidden = set(int(i) for i in test_idx)

    original_fit = FeatureScaler.fit

    def guarded_fit(self, X, *args, **kwargs):
        ids = np.asarray(X)[:, FEATURES.index("microprice_tilt")]
        overlap = {int(v) for v in ids if int(v) in forbidden}
        if overlap:
            raise AssertionError(
                f"FeatureScaler.fit called with {len(overlap)} test-row fingerprints"
            )
        return original_fit(self, X, *args, **kwargs)

    class _OneFoldSplitter:
        """Run only the first walk-forward fold."""

        def __init__(self, inner: WalkForwardSplitter) -> None:
            self.inner = inner

        def split(self, frame: pl.DataFrame):
            yield next(self.inner.split(frame))

    one_fold = _OneFoldSplitter(splitter)

    with patch.object(FeatureScaler, "fit", guarded_fit):
        run_walkforward_lstm(
            splitter=one_fold,
            df=df,
            features=FEATURES,
            label_col="label_tb",
            epochs=1,
            hidden_size=8,
            batch_size=64,
            patience=1,
        )


def test_no_shuffle_in_walkforward():
    """Every train index must precede every test index (time-ordered folds)."""
    paths = sorted(PARQUET_DIR.glob("mnq_features_*.parquet"))
    if len(paths) < 12:
        pytest.skip("need >= 12 days of mnq_features_*.parquet")

    frames = [pl.read_parquet(p) for p in paths[:15]]
    df = pl.concat(frames, how="vertical_relaxed")

    splitter = WalkForwardSplitter(
        n_train_days=10, n_test_days=1, embargo_bars=100, step_days=1
    )
    for train_idx, test_idx in splitter.split(df):
        assert train_idx.max() < test_idx.min(), (
            "test index precedes train index — fold is not strictly forward in time"
        )


def test_manual_embargo_sniff_one_fold():
    """Print train_end / test_start gap for fold 0 (manual sanity check)."""
    paths = sorted(PARQUET_DIR.glob("mnq_features_*.parquet"))
    if len(paths) < 12:
        pytest.skip("need >= 12 days of mnq_features_*.parquet")

    frames = [pl.read_parquet(p) for p in paths[:15]]
    df = pl.concat(frames, how="vertical_relaxed").sort("ts_event")
    splitter = WalkForwardSplitter(
        n_train_days=10, n_test_days=1, embargo_bars=100, step_days=1
    )
    train_idx, test_idx = next(splitter.split(df))
    ts = df.sort("ts_event")["ts_event"]
    train_end = ts[int(train_idx.max())]
    test_start = ts[int(test_idx.min())]
    gap_s = (test_start - train_end).total_seconds()
    print(f"fold 0: train_end={train_end} test_start={test_start} gap_s={gap_s:.3f}")
    assert gap_s >= splitter.embargo_seconds - 1e-3
