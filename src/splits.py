"""Walk-forward train/test splits with an embargo gap (no look-ahead)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import polars as pl

from src.config import BAR_MS


class WalkForwardSplitter:
    """Sliding day-aligned walk-forward with a trailing embargo on train.

    For each fold:
      train days → (drop last ``embargo_bars`` train rows) → embargo → test days

    Indices are positions into ``df`` **after** sorting by ``ts_event``.
    Embargo rows are discarded (not used for train or test).
    """

    def __init__(
        self,
        n_train_days: int = 10,
        n_test_days: int = 1,
        embargo_bars: int = 100,
        step_days: int = 1,
        bar_ms: int = BAR_MS,
    ) -> None:
        if n_train_days < 1 or n_test_days < 1:
            raise ValueError("n_train_days and n_test_days must be >= 1")
        if embargo_bars < 0:
            raise ValueError("embargo_bars must be >= 0")
        if step_days < 1:
            raise ValueError("step_days must be >= 1")
        self.n_train_days = int(n_train_days)
        self.n_test_days = int(n_test_days)
        self.embargo_bars = int(embargo_bars)
        self.step_days = int(step_days)
        self.bar_ms = int(bar_ms)

    @property
    def embargo_seconds(self) -> float:
        """Embargo length in seconds (``embargo_bars`` × bar duration)."""
        return self.embargo_bars * self.bar_ms / 1000.0

    def unique_days(self, df: pl.DataFrame) -> list:
        """Calendar dates present in ``df``, in time order."""
        if "ts_event" not in df.columns:
            raise ValueError("DataFrame must have a ts_event column")
        sorted_df = df.sort("ts_event")
        return (
            sorted_df.select(pl.col("ts_event").dt.date().alias("_day"))
            .unique(maintain_order=True)["_day"]
            .to_list()
        )

    def split(self, df: pl.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` for each fold.

        ``train_idx`` / ``test_idx`` are integer positions into the
        ``ts_event``-sorted frame (same order as ``df.sort("ts_event")``).
        """
        for train_idx, _embargo_idx, test_idx in self.split_with_embargo(df):
            yield train_idx, test_idx

    def split_with_embargo(
        self, df: pl.DataFrame
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, embargo_idx, test_idx)`` for each fold."""
        if "ts_event" not in df.columns:
            raise ValueError("DataFrame must have a ts_event column")

        sorted_df = df.sort("ts_event")
        if sorted_df.height == 0:
            return

        days = (
            sorted_df.select(pl.col("ts_event").dt.date().alias("_day"))["_day"].to_list()
        )
        unique: list = []
        seen: set = set()
        for d in days:
            if d not in seen:
                seen.add(d)
                unique.append(d)

        day_to_indices: dict = {d: [] for d in unique}
        for i, d in enumerate(days):
            day_to_indices[d].append(i)

        need = self.n_train_days + self.n_test_days
        start = 0
        while start + need <= len(unique):
            train_days = unique[start : start + self.n_train_days]
            test_days = unique[
                start + self.n_train_days : start + self.n_train_days + self.n_test_days
            ]

            train_all = np.asarray(
                [i for d in train_days for i in day_to_indices[d]], dtype=np.int64
            )
            test_idx = np.asarray(
                [i for d in test_days for i in day_to_indices[d]], dtype=np.int64
            )

            # Trailing embargo: last embargo_bars of the train window (by time).
            if self.embargo_bars > 0 and train_all.size > 0:
                n_drop = min(self.embargo_bars, int(train_all.size))
                embargo_idx = train_all[-n_drop:].copy()
                train_idx = train_all[:-n_drop].copy()
            else:
                embargo_idx = np.array([], dtype=np.int64)
                train_idx = train_all

            yield train_idx, embargo_idx, test_idx
            start += self.step_days

    def n_folds(self, df: pl.DataFrame) -> int:
        """How many folds ``split`` would produce on ``df``."""
        n_days = len(self.unique_days(df))
        need = self.n_train_days + self.n_test_days
        if n_days < need:
            return 0
        return 1 + (n_days - need) // self.step_days

    def warmup_days(self, df: pl.DataFrame) -> list:
        """First ``n_train_days`` calendar days (never used as a pure test day)."""
        days = self.unique_days(df)
        return days[: self.n_train_days]

    def testable_days(self, df: pl.DataFrame) -> list:
        """Non-warmup days that can appear as test under this config."""
        days = self.unique_days(df)
        return days[self.n_train_days :]
