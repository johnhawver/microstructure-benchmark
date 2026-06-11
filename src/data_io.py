"""Parquet I/O helper functions"""

from __future__ import annotations
import polars as pl
from pathlib import Path

def write_parquet(df: pl.DataFrame, path: str | Path) -> None:
    """Write a Polars DataFrame to a Parquet file at the given path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", statistics=True)

def scan_parquet(path: str | Path) -> pl.LazyFrame:
    """Scan a Parquet file at the given path and return a LazyFrame."""
    return pl.scan_parquet(str(path))