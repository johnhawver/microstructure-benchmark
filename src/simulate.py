"""Bar-by-bar trading simulator from model signed scores."""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit


@njit
def _score_to_target(
    score: float, entry_threshold: float, max_position: int
) -> int:
    if score > entry_threshold:
        return max_position
    if score < -entry_threshold:
        return -max_position
    return 0


@njit
def _simulate_nb(
    signed_score: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    entry_threshold: float,
    max_position: int,
    tick_size: float,
    tick_value: float,
    fill_delay_bars: int,
    commission_per_trade: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba core: positions, per-bar PnL, cumulative PnL.

    A decision at bar ``t`` fills at bar ``t + fill_delay_bars`` using that
  bar's quotes. Commission is charged per contract per side on each fill.
    """
    n = signed_score.shape[0]
    position = np.zeros(n, dtype=np.int8)
    trade_pnl = np.zeros(n, dtype=np.float64)
    cum_pnl = np.zeros(n, dtype=np.float64)

    pos = 0
    total = 0.0
    price_to_dollars = tick_value / tick_size

    for t in range(n):
        if t >= fill_delay_bars:
            dec_t = t - fill_delay_bars
            target = _score_to_target(
                signed_score[dec_t], entry_threshold, max_position
            )
            if target != pos:
                half_spread = (ask[t] - bid[t]) / 2.0
                delta = abs(target - pos)
                spread_cost = delta * half_spread * price_to_dollars
                comm_cost = delta * commission_per_trade
                trade_pnl[t] -= spread_cost + comm_cost
                total -= spread_cost + comm_cost
                pos = target

        position[t] = pos

        if t < n - 1:
            mtm = pos * (mid[t + 1] - mid[t]) * price_to_dollars
            trade_pnl[t] += mtm
            total += mtm

        cum_pnl[t] = total

    return position, trade_pnl, cum_pnl


class Simulator:
    """Convert signed scores into positions and dollar PnL.

    Score rule: ``+max_position`` if ``s > entry_threshold``,
    ``-max_position`` if ``s < -entry_threshold``, else flat.

    Decisions at bar ``t`` fill at ``t + fill_delay_bars`` (paying that bar's
    bid/ask). Each filled contract pays half-spread plus ``commission_per_trade``.
    Mark-to-market uses ``position[t] * (mid[t+1] - mid[t])`` in dollars.
    """

    def __init__(
        self,
        tick_size: float = 0.25,
        tick_value: float = 0.50,
        entry_threshold: float = 0.2,
        max_position: int = 1,
        fill_delay_bars: int = 1,
        commission_per_trade: float = 0.35,
    ) -> None:
        if tick_size <= 0 or tick_value <= 0:
            raise ValueError("tick_size and tick_value must be positive")
        if max_position < 1:
            raise ValueError("max_position must be >= 1")
        if fill_delay_bars < 0:
            raise ValueError("fill_delay_bars must be >= 0")
        if commission_per_trade < 0:
            raise ValueError("commission_per_trade must be >= 0")
        self.tick_size = float(tick_size)
        self.tick_value = float(tick_value)
        self.entry_threshold = float(entry_threshold)
        self.max_position = int(max_position)
        self.fill_delay_bars = int(fill_delay_bars)
        self.commission_per_trade = float(commission_per_trade)

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        """Simulate on rows with ``ts_event, mid, bid, ask, signed_score``."""
        needed = ["ts_event", "mid", "bid", "ask", "signed_score"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        sorted_df = df.sort("ts_event")
        mid = sorted_df["mid"].to_numpy().astype(np.float64)
        bid = sorted_df["bid"].to_numpy().astype(np.float64)
        ask = sorted_df["ask"].to_numpy().astype(np.float64)
        score = sorted_df["signed_score"].to_numpy().astype(np.float64)

        if np.any(ask < bid):
            raise ValueError("crossed book: ask < bid")
        if np.any(mid <= 0):
            raise ValueError("mid must be positive")

        position, trade_pnl, cum_pnl = _simulate_nb(
            score,
            mid,
            bid,
            ask,
            self.entry_threshold,
            self.max_position,
            self.tick_size,
            self.tick_value,
            self.fill_delay_bars,
            self.commission_per_trade,
        )

        return sorted_df.with_columns(
            pl.Series("position", position),
            pl.Series("trade_pnl", trade_pnl),
            pl.Series("cum_pnl", cum_pnl),
        )


def summary_stats(sim_df: pl.DataFrame) -> dict[str, float]:
    """Return total PnL, Sharpe, max drawdown, hit rate, avg duration, trade count."""
    if sim_df.height == 0:
        return {
            "total_pnl": 0.0,
            "sharpe": float("nan"),
            "max_drawdown": 0.0,
            "hit_rate": float("nan"),
            "avg_trade_duration_bars": float("nan"),
            "trade_count": 0.0,
        }

    df = sim_df.sort("ts_event")
    total_pnl = float(df["trade_pnl"].sum())

    daily = (
        df.with_columns(pl.col("ts_event").dt.date().alias("_day"))
        .group_by("_day")
        .agg(pl.col("trade_pnl").sum().alias("daily_pnl"))
        .sort("_day")
    )
    daily_pnl = daily["daily_pnl"].to_numpy()
    if daily_pnl.size >= 2 and np.std(daily_pnl) > 0:
        sharpe = float(np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252))
    else:
        sharpe = float("nan")

    cum = df["cum_pnl"].to_numpy()
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak
    max_drawdown = float(drawdown.min())

    pos = df["position"].to_numpy()
    trade_count = 0
    winners = 0
    durations: list[float] = []
    spell_pnl = 0.0
    spell_len = 0
    prev = 0

    for i in range(df.height):
        p = int(pos[i])
        spell_pnl += float(df["trade_pnl"][i])
        if p != 0:
            spell_len += 1
        if p != prev:
            if prev != 0:
                trade_count += 1
                if spell_pnl > 0:
                    winners += 1
                durations.append(float(spell_len))
                spell_pnl = 0.0
                spell_len = 0 if p == 0 else 1
            elif p != 0:
                spell_pnl = float(df["trade_pnl"][i])
                spell_len = 1
            prev = p

    if prev != 0:
        trade_count += 1
        if spell_pnl > 0:
            winners += 1
        durations.append(float(spell_len))

    hit_rate = float(winners / trade_count) if trade_count > 0 else float("nan")
    avg_dur = float(np.mean(durations)) if durations else float("nan")

    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "avg_trade_duration_bars": avg_dur,
        "trade_count": float(trade_count),
    }
