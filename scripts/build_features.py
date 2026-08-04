"""Build daily feature+label parquet files from MNQ MBP-1 data."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.config import DATA_END, DATA_START, PARQUET_DIR
from src.data_io import write_parquet
from src.features import build_feature_frame
from src.labels import build_labels


def iter_dates(start: str, end: str) -> list[str]:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while d0 < d1:
        out.append(d0.isoformat())
        d0 += timedelta(days=1)
    return out


def _has_labels(path: Path) -> bool:
    names = pl.scan_parquet(path).collect_schema().names()
    return {"label_tb", "label_fr"}.issubset(names)


def build_day(day: str) -> Path | None:
    src = PARQUET_DIR / f"mnq_mbp1_{day}.parquet"
    if not src.exists():
        print(f"{day}: skip (no mbp1 file)")
        return None

    out = PARQUET_DIR / f"mnq_features_{day}.parquet"
    if out.exists() and _has_labels(out):
        print(f"{day}: skip (features+labels already exist)")
        return out

    df = build_labels(build_feature_frame(day))
    write_parquet(df, out)
    print(f"{day}: wrote {out.name} rows={df.height:,} cols={df.width}")
    return out


def main() -> None:
    for day in iter_dates(DATA_START, DATA_END):
        build_day(day)


if __name__ == "__main__":
    main()