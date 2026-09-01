"""Random-label control: shuffled train labels should not beat the majority baseline.

If accuracy stays well above always-predict-zero after label shuffle, suspect
feature-side leakage.

  .venv/bin/python scripts/leakage_sanity.py
  .venv/bin/python scripts/leakage_sanity.py --max-folds 2 --no-wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PARQUET_DIR
from src.features import FEATURES
from src.models.xgb_model import DEFAULT_PARAMS
from src.splits import WalkForwardSplitter
from src.walkforward import run_walkforward

# Reuse walk-forward driver helpers.
from scripts.run_walkforward import (  # noqa: E402
    LABEL_COL,
    RESULTS_DIR,
    XGBWalkForwardModel,
    _cap_splitter,
    _fold_metrics,
    _load_all_features,
    _signed_to_cls,
)

OUT_PATH = RESULTS_DIR / "xgb_wf_random_labels.parquet"


def _majority_baseline_accuracy(y_cls: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class."""
    if y_cls.size == 0:
        return float("nan")
    counts = np.bincount(y_cls, minlength=3)
    return float(counts.max() / y_cls.size)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-folds", type=int, default=None, help="Cap folds (debug).")
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = _load_all_features()
    splitter = WalkForwardSplitter(
        n_train_days=10,
        n_test_days=1,
        embargo_bars=100,
        step_days=1,
    )
    active = _cap_splitter(splitter, args.max_folds)

    # Manual sniff test: fold 0 train/test gap.
    train_idx, test_idx = next(splitter.split(df))
    ts = df.sort("ts_event")["ts_event"]
    train_end = ts[int(train_idx.max())]
    test_start = ts[int(test_idx.min())]
    gap_s = (test_start - train_end).total_seconds()
    print(
        f"fold 0 sniff: train_end={train_end} test_start={test_start} "
        f"gap_s={gap_s:.3f} (embargo={splitter.embargo_seconds}s)"
    )

    print("Running XGB walk-forward with shuffled train labels per fold...")
    preds = run_walkforward(
        model_factory=lambda: XGBWalkForwardModel(params=DEFAULT_PARAMS),
        splitter=active,
        df=df,
        features=FEATURES,
        label_col=LABEL_COL,
        shuffle_train_labels=True,
        shuffle_seed=args.shuffle_seed,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(preds):,} rows)")

    per_fold: list[dict[str, float]] = []
    print("\n=== Random-label control (per fold) ===")
    print(f"{'fold':>4} {'acc':>8} {'majority':>10} {'delta':>8}")
    for fold, g in preds.groupby("fold", sort=True):
        m = _fold_metrics(g)
        y_cls = _signed_to_cls(g["y_true"].to_numpy())
        maj = _majority_baseline_accuracy(y_cls)
        delta = m["accuracy"] - maj
        per_fold.append({**m, "majority_acc": maj, "acc_minus_majority": delta})
        print(
            f"{int(fold):4d} {m['accuracy']:8.4f} {maj:10.4f} {delta:+8.4f}"
        )

    accs = np.array([m["accuracy"] for m in per_fold])
    majs = np.array([m["majority_acc"] for m in per_fold])
    mean_acc = float(accs.mean())
    mean_maj = float(majs.mean())
    mean_delta = mean_acc - mean_maj

    print("\n=== Summary ===")
    print(f"  mean_accuracy:        {mean_acc:.4f}")
    print(f"  mean_majority_baseline: {mean_maj:.4f}")
    print(f"  mean_acc - majority:  {mean_delta:+.4f}")
    if mean_delta > 0.02:
        print(
            "  WARNING: shuffled labels still beat majority by >2pp — "
            "investigate feature leakage."
        )
    else:
        print(
            "  OK: shuffled-label accuracy near majority baseline — "
            "no strong feature-side leakage signal."
        )


if __name__ == "__main__":
    main()
