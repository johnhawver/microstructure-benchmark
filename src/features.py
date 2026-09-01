"""Feature engineering: resampling and derived signals."""

from __future__ import annotations

import polars as pl

from datetime import date, timedelta
from src.config import BAR_MS
from src.data_io import add_mid_and_spread, load_mbp1

# Final model feature columns (mid is kept separately for labeling).
FEATURES = [
    "microprice_tilt",
    "spread_bps",
    "rel_spread",
    "vol_10",
    "vol_50",
    "vol_200",
    "autocorr_lag1_w50",
    "ofi_sum_100ms",
    "ofi_sum_1000ms",
    "ofi_sum_5000ms",
    "trade_imb_50",
    "kyle_lambda_50",
]


def resample_to_bars(lf: pl.LazyFrame, bar_ms: int) -> pl.LazyFrame:
    """Resample event-time MBP-1 to fixed bars (last quotes + trade aggregates)."""
    return (
        lf.sort("ts_event")
        .group_by_dynamic(
            "ts_event",
            every=f"{bar_ms}ms",
            closed="left",
            label="left",
        )
        .agg(
            pl.col("bid").last(),
            pl.col("ask").last(),
            pl.col("bid_sz").last(),
            pl.col("ask_sz").last(),
            pl.col("mid").last(),
            pl.col("microprice").last(),
            pl.col("spread").last(),
            pl.col("size")
            .filter((pl.col("action") == "T") & (pl.col("side") == "B"))
            .sum()
            .fill_null(0)
            .alias("buy_vol"),
            pl.col("size")
            .filter((pl.col("action") == "T") & (pl.col("side") == "A"))
            .sum()
            .fill_null(0)
            .alias("sell_vol"),
            pl.col("action")
            .filter(pl.col("action") == "T")
            .len()
            .alias("trade_count"),
        )
        .with_columns(pl.col("ts_event").dt.date().alias("_date"))
        .with_columns(
            pl.col("bid").fill_null(strategy="forward").over("_date"),
            pl.col("ask").fill_null(strategy="forward").over("_date"),
            pl.col("bid_sz").fill_null(strategy="forward").over("_date"),
            pl.col("ask_sz").fill_null(strategy="forward").over("_date"),
            pl.col("mid").fill_null(strategy="forward").over("_date"),
            pl.col("microprice").fill_null(strategy="forward").over("_date"),
            pl.col("spread").fill_null(strategy="forward").over("_date"),
        )
        .drop("_date")
        .filter(pl.col("mid").is_not_null())
    )


def add_returns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append log_ret = ln(mid / mid_prev); drop the first bar of each day."""
    return (
        lf.with_columns(
            (pl.col("mid") / pl.col("mid").shift(1).over(pl.col("ts_event").dt.date()))
            .log()
            .alias("log_ret")
        )
        .filter(pl.col("log_ret").is_not_null())
    )

def add_rolling_vol(lf: pl.LazyFrame, windows: list[int]) -> pl.LazyFrame:
    """Append rolling std of log_ret for each window length (in bars)."""
    cols = [
        pl.col("log_ret").rolling_std(N).alias(f"vol_{N}")
        for N in windows
    ]
    return lf.with_columns(cols)


def _rolling_cov(a: pl.Expr, b: pl.Expr, window: int) -> pl.Expr:
    """E[ab] - E[a]E[b] over a rolling window."""
    return (a * b).rolling_mean(window) - a.rolling_mean(window) * b.rolling_mean(window)


def _rolling_var(a: pl.Expr, window: int) -> pl.Expr:
    """E[a^2] - E[a]^2 over a rolling window."""
    return (a * a).rolling_mean(window) - a.rolling_mean(window).pow(2)


def add_autocorr(lf: pl.LazyFrame, lag: int, window: int) -> pl.LazyFrame:
    """Append rolling correlation of log_ret with log_ret shifted by lag."""
    x = pl.col("log_ret")
    y = pl.col("log_ret").shift(lag)
    return lf.with_columns(
        (
            _rolling_cov(x, y, window)
            / (_rolling_var(x, window).sqrt() * _rolling_var(y, window).sqrt())
        ).alias(f"autocorr_lag{lag}_w{window}")
    )


def add_microprice_tilt(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append microprice_tilt = (microprice - mid) / spread (fraction of spread)."""
    return lf.with_columns(
        ((pl.col("microprice") - pl.col("mid")) / pl.col("spread")).alias("microprice_tilt")
    )


def compute_ofi_events(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Compute per-event Cont-Kukanov-Stoikov OFI on raw MBP-1 rows."""
    lf = lf.with_columns(
        pl.col("bid_sz").cast(pl.Int64),
        pl.col("ask_sz").cast(pl.Int64),
    )
    bid_prev = pl.col("bid").shift(1).over(pl.col("ts_event").dt.date())
    ask_prev = pl.col("ask").shift(1).over(pl.col("ts_event").dt.date())
    bid_sz_prev = pl.col("bid_sz").shift(1).over(pl.col("ts_event").dt.date())
    ask_sz_prev = pl.col("ask_sz").shift(1).over(pl.col("ts_event").dt.date())

    e_bid = (
        pl.when(pl.col("bid") > bid_prev)
        .then(pl.col("bid_sz"))
        .when(pl.col("bid") == bid_prev)
        .then(pl.col("bid_sz") - bid_sz_prev)
        .otherwise(-bid_sz_prev)
    )
    e_ask = (
        pl.when(pl.col("ask") < ask_prev)
        .then(pl.col("ask_sz"))
        .when(pl.col("ask") == ask_prev)
        .then(pl.col("ask_sz") - ask_sz_prev)
        .otherwise(-ask_sz_prev)
    )

    return (
        lf.sort("ts_event")
        .with_columns((e_bid - e_ask).alias("ofi"))
        .filter(pl.col("ofi").is_not_null())
    )


def aggregate_ofi_to_bars(lf_events: pl.LazyFrame, bar_ms: int) -> pl.LazyFrame:
    """Sum per-event OFI into fixed bars; column name ofi_sum_{bar_ms}ms."""
    col = f"ofi_sum_{bar_ms}ms"
    return (
        lf_events.sort("ts_event")
        .group_by_dynamic(
            "ts_event",
            every=f"{bar_ms}ms",
            closed="left",
            label="left",
        )
        .agg(pl.col("ofi").sum().alias(col))
    )


def add_trade_imbalance(lf: pl.LazyFrame, windows: list[int]) -> pl.LazyFrame:
    """Append rolling trade imbalance for each window length (in bars)."""
    cols = [
        (
            (pl.col("buy_vol").rolling_sum(N) - pl.col("sell_vol").rolling_sum(N))
            / (pl.col("buy_vol").rolling_sum(N) + pl.col("sell_vol").rolling_sum(N) + 1e-9)
        ).alias(f"trade_imb_{N}")
        for N in windows
    ]
    return lf.with_columns(cols)


def add_kyle_lambda(lf: pl.LazyFrame, window: int) -> pl.LazyFrame:
    """Append rolling Kyle lambda: cov(log_ret, signed_vol) / var(signed_vol)."""
    signed_vol = pl.col("buy_vol") - pl.col("sell_vol")
    return lf.with_columns(
        (
            _rolling_cov(pl.col("log_ret"), signed_vol, window)
            / _rolling_var(signed_vol, window)
        ).alias(f"kyle_lambda_{window}")
    )


def add_spread_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append spread_bps and rel_spread (vs 200-bar mean)."""
    return lf.with_columns(
        (pl.col("spread") / pl.col("mid") * 1e4).alias("spread_bps"),
        (pl.col("spread") / pl.col("spread").rolling_mean(200)).alias("rel_spread"),
    )


def build_features_from_mbp(mbp: pl.LazyFrame, bar_ms: int = BAR_MS) -> pl.DataFrame:
    """Run the full feature pipeline on pre-loaded MBP-1 (mid/spread columns)."""
    bars = resample_to_bars(mbp, bar_ms)
    ofi_100 = aggregate_ofi_to_bars(compute_ofi_events(mbp), 100)
    lf = (
        bars.join(ofi_100, on="ts_event", how="left")
        .with_columns(pl.col("ofi_sum_100ms").fill_null(0))
        .with_columns(
            pl.col("ofi_sum_100ms").rolling_sum(10).alias("ofi_sum_1000ms"),
            pl.col("ofi_sum_100ms").rolling_sum(50).alias("ofi_sum_5000ms"),
        )
    )
    lf = add_returns(lf)
    lf = add_rolling_vol(lf, [10, 50, 200])
    lf = add_autocorr(lf, lag=1, window=50)
    lf = add_microprice_tilt(lf)
    lf = add_spread_features(lf)
    lf = add_trade_imbalance(lf, [10, 50, 200])
    lf = add_kyle_lambda(lf, 50)
    return lf.select(["ts_event", "mid", *FEATURES]).collect()


def build_feature_frame(day: str) -> pl.DataFrame:
    """Run the full one-day feature pipeline; return ts_event + FEATURES + mid."""
    d1 = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    mbp = add_mid_and_spread(load_mbp1(day, d1))
    return build_features_from_mbp(mbp)