"""Feature engineering: resampling and derived signals."""

from __future__ import annotations

import polars as pl


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

def add_autocorr(lf: pl.LazyFrame, lag: int, window: int) -> pl.LazyFrame:
    """Append rolling correlation of log_ret with log_ret shifted by lag."""
    return lf.with_columns(
        pl.col("log_ret")
        .rolling_corr(pl.col("log_ret").shift(lag), window_size=window)
        .alias(f"autocorr_lag{lag}_w{window}")
    )


def add_microprice_tilt(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append microprice_tilt = (microprice - mid) / spread (fraction of spread)."""
    return lf.with_columns(
        ((pl.col("microprice") - pl.col("mid")) / pl.col("spread")).alias("microprice_tilt")
    )