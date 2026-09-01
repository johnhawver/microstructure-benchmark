"""Run trading simulation on walk-forward predictions.

Default: latency/threshold sweep (Day 27).
Quick mode: single run per model (Day 26 style).

  .venv/bin/python scripts/run_simulation.py --no-wandb
  .venv/bin/python scripts/run_simulation.py --quick --no-wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import wandb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PARQUET_DIR
from src.simulate import Simulator, summary_stats

RESULTS_DIR = ROOT / "data" / "results"
XGB_PREDS = RESULTS_DIR / "xgb_wf.parquet"
LSTM_PREDS = RESULTS_DIR / "lstm_wf.parquet"
XGB_SIM_OUT = RESULTS_DIR / "xgb_sim.parquet"
LSTM_SIM_OUT = RESULTS_DIR / "lstm_sim.parquet"
SWEEP_OUT = RESULTS_DIR / "sim_sweep.parquet"

DELAYS = (0, 1, 5, 10)
THRESHOLDS = (0.1, 0.2, 0.3)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Single run per model (fill_delay=1, threshold=0.2).",
    )
    p.add_argument("--entry-threshold", type=float, default=0.2)
    p.add_argument("--fill-delay-bars", type=int, default=1)
    return p.parse_args()


def _load_bar_quotes() -> pl.DataFrame:
    """Bid/ask/mid for all bar timestamps (join key for predictions)."""
    paths = sorted(PARQUET_DIR.glob("mnq_bars100ms_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No mnq_bars100ms_*.parquet under {PARQUET_DIR}. "
            "Run scripts/build_features.py first."
        )
    lf = pl.concat([pl.scan_parquet(p) for p in paths], how="vertical_relaxed")
    return (
        lf.select(["ts_event", "mid", "bid", "ask"])
        .collect()
        .unique(subset=["ts_event"], keep="last")
        .sort("ts_event")
    )


def prepare_sim_frame(preds: pd.DataFrame, quotes: pl.DataFrame) -> pl.DataFrame:
    """Attach signed_score and L1 quotes for simulation."""
    frame = (
        pl.from_pandas(preds)
        .with_columns(
            (pl.col("prob_up") - pl.col("prob_down")).alias("signed_score"),
            pl.col("ts_event").dt.replace_time_zone("UTC"),
        )
    )
    quotes = quotes.with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
    joined = frame.join(quotes, on="ts_event", how="inner")
    n_before = frame.height
    n_after = joined.height
    if n_after < n_before:
        print(f"  dropped {n_before - n_after:,} rows without bar quotes")
    if n_after == 0:
        raise ValueError("no rows after joining predictions with bar quotes")
    return joined


def run_one(
    name: str,
    preds_path: Path,
    out_path: Path | None,
    sim_in: pl.DataFrame,
    simulator: Simulator,
    use_wandb: bool,
    wandb_group: str = "trading-sim",
) -> dict[str, float]:
    sim_out = simulator.run(sim_in)

    if out_path is not None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        sim_out.write_parquet(out_path)
        print(f"wrote {out_path}")

    stats = summary_stats(sim_out)
    print(
        f"  {name} delay={simulator.fill_delay_bars} thr={simulator.entry_threshold}: "
        f"pnl={stats['total_pnl']:.1f} sharpe={stats['sharpe']:.2f} "
        f"trades={stats['trade_count']:.0f}"
    )

    if use_wandb:
        run = wandb.init(
            project="mnq-microstructure",
            name=(
                f"{name}-d{simulator.fill_delay_bars}-t{simulator.entry_threshold}"
            ),
            group=wandb_group,
            job_type="simulate",
            config={
                "model": name,
                "entry_threshold": simulator.entry_threshold,
                "fill_delay_bars": simulator.fill_delay_bars,
                "commission_per_trade": simulator.commission_per_trade,
                "tick_size": simulator.tick_size,
                "tick_value": simulator.tick_value,
                "max_position": simulator.max_position,
            },
            settings=wandb.Settings(init_timeout=300),
        )
        wandb.log(stats)
        equity = sim_out.select(["ts_event", "cum_pnl"]).to_pandas()
        wandb.log({"equity_curve": wandb.Table(dataframe=equity)})
        run.finish()

    return stats


def run_sweep(use_wandb: bool) -> pd.DataFrame:
    quotes = _load_bar_quotes()
    print(f"quotes: {quotes.height:,} bars")

    models = {
        "xgb": (XGB_PREDS, prepare_sim_frame(pd.read_parquet(XGB_PREDS), quotes)),
        "lstm": (LSTM_PREDS, prepare_sim_frame(pd.read_parquet(LSTM_PREDS), quotes)),
    }

    rows: list[dict] = []
    for name, (_path, sim_in) in models.items():
        print(f"\n=== {name} sweep ({len(DELAYS)} delays × {len(THRESHOLDS)} thresholds) ===")
        for delay in DELAYS:
            for thr in THRESHOLDS:
                sim = Simulator(
                    entry_threshold=thr,
                    fill_delay_bars=delay,
                    commission_per_trade=0.35,
                )
                stats = run_one(
                    name,
                    _path,
                    out_path=None,
                    sim_in=sim_in,
                    simulator=sim,
                    use_wandb=use_wandb,
                    wandb_group="trading-sim-sweep",
                )
                rows.append(
                    {
                        "model": name,
                        "fill_delay_bars": delay,
                        "entry_threshold": thr,
                        **stats,
                    }
                )

    sweep = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_parquet(SWEEP_OUT, index=False)
    print(f"\nWrote {SWEEP_OUT} ({len(sweep)} rows)")
    return sweep


def main() -> None:
    args = _parse_args()
    use_wandb = not args.no_wandb

    if args.quick:
        print("Loading bar quotes...")
        quotes = _load_bar_quotes()
        simulator = Simulator(
            entry_threshold=args.entry_threshold,
            fill_delay_bars=args.fill_delay_bars,
        )
        for name, path, out in [
            ("xgb", XGB_PREDS, XGB_SIM_OUT),
            ("lstm", LSTM_PREDS, LSTM_SIM_OUT),
        ]:
            preds = pd.read_parquet(path)
            print(f"\n=== {name} ===")
            sim_in = prepare_sim_frame(preds, quotes)
            run_one(name, path, out, sim_in, simulator, use_wandb)
        return

    run_sweep(use_wandb)


if __name__ == "__main__":
    main()
