"""Build 100ms bar parquet files from daily MNQ MBP-1 data."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from src.config import BAR_MS, DATA_END, DATA_START, PARQUET_DIR
from src.data_io import add_mid_and_spread, load_mbp1, write_parquet
from src.features import resample_to_bars


def iter_dates(start: str, end: str) -> list[str]:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while d0 < d1:
        out.append(d0.isoformat())
        d0 += timedelta(days=1)
    return out


def build_day(day: str) -> Path | None:
    src = PARQUET_DIR / f"mnq_mbp1_{day}.parquet"
    if not src.exists():
        print(f"{day}: skip (no mbp1 file)")
        return None

    out = PARQUET_DIR / f"mnq_bars100ms_{day}.parquet"
    if out.exists():
        print(f"{day}: skip (already exists)")
        return out

    d1 = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    df = resample_to_bars(add_mid_and_spread(load_mbp1(day, d1)), BAR_MS).collect()
    write_parquet(df, out)
    print(f"{day}: wrote {out.name} rows={df.height:,}")
    return out


def main() -> None:
    for day in iter_dates(DATA_START, DATA_END):
        build_day(day)


if __name__ == "__main__":
    main()