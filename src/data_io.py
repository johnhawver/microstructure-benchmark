"""Parquet I/O helper functions"""

from __future__ import annotations

import re
from datetime import time
from pathlib import Path

import polars as pl

from src.config import PARQUET_DIR, RTH_END_ET, RTH_START_ET

_DATE_RE = re.compile(r"mnq_mbp1_(\d{4}-\d{2}-\d{2})\.parquet$")


def write_parquet(df: pl.DataFrame, path: str | Path) -> None:
    """Write a Polars DataFrame to a Parquet file at the given path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", statistics=True)


def scan_parquet(path: str | Path) -> pl.LazyFrame:
    """Scan a Parquet file at the given path and return a LazyFrame."""
    return pl.scan_parquet(str(path))


def load_mbp1(start: str, end: str) -> pl.LazyFrame:
    """Load cleaned MNQ MBP-1 data for dates in [start, end) as a LazyFrame."""
    paths: list[str] = []
    for path in sorted(PARQUET_DIR.glob("mnq_mbp1_*.parquet")):
        match = _DATE_RE.match(path.name)
        if match is None:
            continue
        date = match.group(1)
        if start <= date < end:
            paths.append(str(path))

    if not paths:
        raise FileNotFoundError(
            f"No mnq_mbp1_YYYY-MM-DD.parquet files in {PARQUET_DIR} for [{start}, {end})"
        )

    return (
        pl.scan_parquet(paths)
        .filter(
            (pl.col("ts_event") >= pl.lit(start).str.to_datetime(time_zone="UTC"))
            & (pl.col("ts_event") < pl.lit(end).str.to_datetime(time_zone="UTC"))
        )
        # Prices are already real floats (Databento to_df() scales the 1e-9 ints).
        .with_columns(
            pl.col("bid_px_00").cast(pl.Float64).alias("bid"),
            pl.col("ask_px_00").cast(pl.Float64).alias("ask"),
            pl.col("price").cast(pl.Float64),
            pl.col("bid_sz_00").alias("bid_sz"),
            pl.col("ask_sz_00").alias("ask_sz"),
        )
        .drop(["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"])
        .filter((pl.col("bid") > 0) & (pl.col("ask") > 0) & (pl.col("ask") >= pl.col("bid")))
    )


def add_mid_and_spread(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Append mid, spread, and microprice columns to an MBP-1 LazyFrame."""
    return lf.with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        (
            (pl.col("bid_sz") * pl.col("ask") + pl.col("ask_sz") * pl.col("bid"))
            / (pl.col("bid_sz") + pl.col("ask_sz"))
        ).alias("microprice"),
    )


def is_rth(ts_utc: pl.Expr) -> pl.Expr:
    """Return True when a UTC timestamp falls in US Eastern RTH (09:30-16:00)."""
    t = ts_utc.dt.convert_time_zone("America/New_York").dt.time()
    return (t >= time.fromisoformat(RTH_START_ET)) & (t < time.fromisoformat(RTH_END_ET))

def filter_rth(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Filter a LazyFrame to only include US Eastern RTH (09:30-16:00)."""
    return lf.filter(is_rth(pl.col("ts_event")))