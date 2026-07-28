"""Build 100ms bar parquet files from daily MNQ MBP-1 data."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.config import BAR_MS, DATA_END, DATA_START, PARQUET_DIR
from src.data_io import add_mid_and_spread, load_mbp1, write_parquet
from src.features import (
    aggregate_ofi_to_bars,
    compute_ofi_events,
    resample_to_bars,
)


def iter_dates(start: str, end: str) -> list[str]:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while d0 < d1:
        out.append(d0.isoformat())
        d0 += timedelta(days=1)
    return out


def _has_ofi_columns(path: Path) -> bool:
    schema = pl.scan_parquet(path).collect_schema().names()
    needed = {"ofi_sum_100ms", "ofi_sum_1000ms", "ofi_sum_5000ms"}
    return needed.issubset(schema)


def build_day(day: str) -> Path | None:
    src = PARQUET_DIR / f"mnq_mbp1_{day}.parquet"
    if not src.exists():
        print(f"{day}: skip (no mbp1 file)")
        return None

    out = PARQUET_DIR / f"mnq_bars100ms_{day}.parquet"
    if out.exists() and _has_ofi_columns(out):
        print(f"{day}: skip (already exists with OFI)")
        return out

    d1 = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    mbp = add_mid_and_spread(load_mbp1(day, d1))
    bars = resample_to_bars(mbp, BAR_MS)
    ofi_100 = aggregate_ofi_to_bars(compute_ofi_events(mbp), 100)

    # ofi_sum_1000/5000ms = rolling sums of 100ms OFI (10 bars=1s, 50=5s)
    df = (
        bars.join(ofi_100, on="ts_event", how="left")
        .with_columns(pl.col("ofi_sum_100ms").fill_null(0))
        .with_columns(
            pl.col("ofi_sum_100ms").rolling_sum(10).alias("ofi_sum_1000ms"),
            pl.col("ofi_sum_100ms").rolling_sum(50).alias("ofi_sum_5000ms"),
        )
        .collect()
    )
    write_parquet(df, out)
    print(f"{day}: wrote {out.name} rows={df.height:,}")
    return out


def main() -> None:
    for day in iter_dates(DATA_START, DATA_END):
        build_day(day)


if __name__ == "__main__":
    main()