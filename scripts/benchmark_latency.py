"""Benchmark per-stage pipeline latency (features, XGB, LSTM).

  .venv/bin/python scripts/benchmark_latency.py
  .venv/bin/python scripts/benchmark_latency.py --quick --no-wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import wandb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PARQUET_DIR
from src.features import FEATURES
from src.latency import (
    benchmark_feature_compute,
    benchmark_lstm_inference,
    benchmark_xgb_inference,
    load_mbp_benchmark_slice,
)
from src.models.lstm_model import MNQLSTM, SEQ_LEN, FeatureScaler
from src.models.xgb_model import DEFAULT_PARAMS, train_xgb

RESULTS_DIR = ROOT / "data" / "results"
LATENCY_CSV = RESULTS_DIR / "latency.csv"
MODELS_DIR = ROOT / "data" / "models"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fewer iterations for a fast smoke run.",
    )
    p.add_argument("--day", default="2025-09-15", help="MBP day for feature bench.")
    return p.parse_args()


def _load_xgb_booster(quick: bool) -> tuple[object, np.ndarray]:
    paths = sorted(PARQUET_DIR.glob("mnq_features_*.parquet"))[:3]
    if not paths:
        raise FileNotFoundError("Need feature parquets — run build_features.py")
    df = pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")
    df = df.with_columns(
        [
            pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in FEATURES
        ]
    ).drop_nulls(subset=FEATURES + ["label_tb"])
    n = 50_000 if quick else 200_000
    df = df.head(n)
    X = df.select(FEATURES).to_numpy().astype(np.float64)
    y = df["label_tb"].to_numpy()
    ok = np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    split = int(len(y) * 0.8)
    booster = train_xgb(
        X[:split],
        y[:split],
        X[split : split + max(1000, split // 5)],
        y[split : split + max(1000, split // 5)],
        params=DEFAULT_PARAMS,
        num_boost_round=50 if quick else 200,
        early_stopping_rounds=10,
    )
    return booster, X[:1]


def _load_lstm_model(quick: bool) -> tuple[torch.nn.Module, torch.Tensor]:
    ckpt = MODELS_DIR / "lstm_fold0.pt"
    scaler_path = MODELS_DIR / "lstm_fold0_scaler.npz"
    if not ckpt.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Need {ckpt} and scaler — run walk-forward LSTM first."
        )

    hidden = 32
    model = MNQLSTM(hidden_size=hidden, num_layers=2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    # One realistic input: scaled random features shaped [1, SEQ_LEN, F].
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(SEQ_LEN, len(FEATURES))).astype(np.float64)
    scaler = FeatureScaler.load(scaler_path)
    scaled = scaler.transform(raw)
    X = torch.from_numpy(scaled).unsqueeze(0)
    return model, X


def main() -> None:
    args = _parse_args()
    n_feat_iters = 50 if args.quick else 200
    n_inf_iters = 500 if args.quick else 10_000

    print("=== Feature compute ===")
    mbp = load_mbp_benchmark_slice(day=args.day, minutes=45 if not args.quick else 30)
    print(f"MBP events: {mbp.height:,}")
    feat_stats = benchmark_feature_compute(
        mbp, n_warmup=20 if args.quick else 100, n_iters=n_feat_iters
    )
    print(feat_stats)

    print("\n=== XGBoost inference ===")
    booster, x_row = _load_xgb_booster(args.quick)
    xgb_stats = benchmark_xgb_inference(
        booster, x_row, n_warmup=20 if args.quick else 100, n_iters=n_inf_iters
    )
    print(xgb_stats)

    print("\n=== LSTM inference ===")
    lstm_model, x_seq = _load_lstm_model(args.quick)
    lstm_stats = benchmark_lstm_inference(
        lstm_model, x_seq, n_warmup=20 if args.quick else 100, n_iters=n_inf_iters
    )
    print(lstm_stats)

    rows = [feat_stats, xgb_stats, lstm_stats]
    df = pd.DataFrame(rows)[["stage", "p50_us", "p95_us", "p99_us", "n_iters"]]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LATENCY_CSV, index=False)
    print(f"\nWrote {LATENCY_CSV}")
    print(df.to_string(index=False))

    xgb_p99 = xgb_stats["p99_us"]
    lstm_p99 = lstm_stats["p99_us"]
    feat_p99 = feat_stats["p99_us"]
    ratio = lstm_p99 / xgb_p99 if xgb_p99 > 0 else float("nan")
    print(f"\nLSTM / XGB p99 ratio: {ratio:.1f}x")
    bar_us = 100_000  # 100 ms bar
    print(
        f"Feature p99 vs 100ms bar budget: {feat_p99/1000:.2f} ms "
        f"({100*feat_p99/bar_us:.1f}% of bar)"
    )

    if not args.no_wandb:
        run = wandb.init(
            project="mnq-microstructure",
            name="latency-benchmark",
            group="latency",
            job_type="benchmark",
            settings=wandb.Settings(init_timeout=300),
        )
        table = wandb.Table(dataframe=df)
        wandb.log({"latency_table": table})
        wandb.log(
            {
                "latency_p99_us": wandb.plot.bar(
                    table,
                    "stage",
                    "p99_us",
                    title="p99 latency by stage (µs)",
                )
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
