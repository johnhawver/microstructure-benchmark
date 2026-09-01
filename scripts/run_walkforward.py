"""Walk-forward XGBoost or LSTM over embargoed day-aligned folds.

  .venv/bin/python scripts/run_walkforward.py --model xgb
  .venv/bin/python scripts/run_walkforward.py --model lstm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import wandb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PARQUET_DIR
from src.features import FEATURES
from src.models.xgb_model import DEFAULT_PARAMS, predict_xgb, train_xgb
from src.splits import WalkForwardSplitter
from src.walkforward import run_walkforward, run_walkforward_lstm

LABEL_COL = "label_tb"
RESULTS_DIR = ROOT / "data" / "results"
MODELS_DIR = ROOT / "data" / "models"
XGB_OUT = RESULTS_DIR / "xgb_wf.parquet"
LSTM_OUT = RESULTS_DIR / "lstm_wf.parquet"


class XGBWalkForwardModel:
    """Fresh XGBoost booster per fold (fit on train, early-stop on val)."""

    def __init__(self, params: dict | None = None) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.booster = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.booster = train_xgb(
            X_train, y_train, X_val, y_val, params=self.params
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model not fit")
        return predict_xgb(self.booster, X)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=("xgb", "lstm"),
        default="xgb",
        help="Which model family to walk-forward.",
    )
    p.add_argument("--no-wandb", action="store_true", help="Skip W&B logging.")
    p.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional cap on number of folds (debug).",
    )
    p.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="LSTM hidden size (default 32 for CPU; use 64 if you have a GPU).",
    )
    p.add_argument("--epochs", type=int, default=20, help="LSTM max epochs.")
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Load existing results parquet and only print/log metrics.",
    )
    return p.parse_args()


def _load_all_features() -> pl.DataFrame:
    paths = sorted(PARQUET_DIR.glob("mnq_features_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No mnq_features_*.parquet under {PARQUET_DIR}. "
            "Run scripts/build_features.py first."
        )
    frames = [pl.read_parquet(p) for p in paths]
    df = pl.concat(frames, how="vertical_relaxed")
    needed = ["ts_event", LABEL_COL, *FEATURES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    n_before = df.height
    df = df.with_columns(
        [
            pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in FEATURES
        ]
    ).drop_nulls(subset=needed)
    n_dropped = n_before - df.height
    if n_dropped:
        print(f"Dropped {n_dropped} rows with null/inf features")
    return df.sort("ts_event")


def _signed_to_cls(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(np.int64)
    out = np.empty_like(y)
    out[y == -1] = 0
    out[y == 0] = 1
    out[y == 1] = 2
    return out


def _fold_metrics(g: pd.DataFrame) -> dict[str, float]:
    y_cls = _signed_to_cls(g["y_true"].to_numpy())
    probs = np.column_stack(
        [
            g["prob_down"].to_numpy(),
            g["prob_zero"].to_numpy(),
            g["prob_up"].to_numpy(),
        ]
    )
    y_pred = probs.argmax(axis=1)
    acc = float(accuracy_score(y_cls, y_pred))
    ll = float(log_loss(y_cls, probs, labels=[0, 1, 2]))
    f1 = float(
        f1_score(y_cls, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    )
    mask = y_cls != 1
    if mask.sum() >= 2 and len(np.unique(y_cls[mask])) == 2:
        auc = float(roc_auc_score((y_cls[mask] == 2).astype(int), probs[mask, 2]))
    else:
        auc = float("nan")
    return {
        "accuracy": acc,
        "log_loss": ll,
        "macro_f1": f1,
        "auc_up_vs_down": auc,
        "n": int(len(g)),
    }


def _aggregate_metrics(per_fold: list[dict[str, float]]) -> dict[str, float]:
    keys = ["accuracy", "log_loss", "macro_f1", "auc_up_vs_down"]
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([m[k] for m in per_fold], dtype=np.float64)
        out[f"mean_{k}"] = float(np.nanmean(vals))
        out[f"std_{k}"] = float(np.nanstd(vals))
    out["n_folds"] = float(len(per_fold))
    return out


def _cap_splitter(splitter: WalkForwardSplitter, max_folds: int | None):
    if max_folds is None:
        return splitter

    class _CappedSplitter:
        def __init__(self, inner: WalkForwardSplitter, n: int) -> None:
            self.inner = inner
            self.n = n

        def split(self, frame: pl.DataFrame):
            for i, pair in enumerate(self.inner.split(frame)):
                if i >= self.n:
                    break
                yield pair

    return _CappedSplitter(splitter, max_folds)


def _log_and_summarize(
    preds: pd.DataFrame,
    *,
    model_name: str,
    wandb_group: str,
    config: dict,
    use_wandb: bool,
) -> None:
    per_fold: list[dict[str, float]] = []
    print("\n=== Per-fold metrics ===")
    print(f"{'fold':>4} {'acc':>8} {'logloss':>10} {'macroF1':>10} {'auc':>8} {'n':>10}")
    for fold, g in preds.groupby("fold", sort=True):
        m = _fold_metrics(g)
        per_fold.append(m)
        print(
            f"{int(fold):4d} {m['accuracy']:8.4f} {m['log_loss']:10.4f} "
            f"{m['macro_f1']:10.4f} {m['auc_up_vs_down']:8.4f} {m['n']:10d}"
        )
        if use_wandb:
            run = wandb.init(
                project="mnq-microstructure",
                name=f"{model_name}-wf-fold{int(fold)}",
                group=wandb_group,
                job_type="fold",
                config={**config, "fold": int(fold)},
                settings=wandb.Settings(init_timeout=300),
            )
            wandb.log(m)
            run.finish()

    agg = _aggregate_metrics(per_fold)
    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")

    if use_wandb:
        run = wandb.init(
            project="mnq-microstructure",
            name=f"{model_name}-wf-aggregate",
            group=wandb_group,
            job_type="aggregate",
            config=config,
            settings=wandb.Settings(init_timeout=300),
        )
        wandb.log(agg)
        table = wandb.Table(
            columns=["fold", "accuracy", "log_loss", "macro_f1", "auc_up_vs_down", "n"],
            data=[
                [
                    i,
                    m["accuracy"],
                    m["log_loss"],
                    m["macro_f1"],
                    m["auc_up_vs_down"],
                    m["n"],
                ]
                for i, m in enumerate(per_fold)
            ],
        )
        wandb.log({"per_fold_metrics": table})
        run.finish()


def main() -> None:
    args = _parse_args()

    if args.model == "xgb":
        out_path = XGB_OUT
        wandb_group = "walkforward-xgb"
        config: dict = {**DEFAULT_PARAMS}
    else:
        out_path = LSTM_OUT
        wandb_group = "walkforward-lstm"
        config = {
            "hidden_size": args.hidden_size,
            "epochs": args.epochs,
            "seq_len": 50,
        }

    if args.summarize_only:
        if not out_path.exists():
            raise FileNotFoundError(
                f"No results at {out_path}. Run walk-forward training first."
            )
        preds = pd.read_parquet(out_path)
        print(f"Loaded {out_path} ({len(preds):,} rows)")
        _log_and_summarize(
            preds,
            model_name=args.model,
            wandb_group=wandb_group,
            config=config,
            use_wandb=not args.no_wandb,
        )
        return

    df = _load_all_features()
    splitter = WalkForwardSplitter(
        n_train_days=10,
        n_test_days=1,
        embargo_bars=100,
        step_days=1,
    )
    active = _cap_splitter(splitter, args.max_folds)
    print(f"Loaded {df.height:,} rows across {len(splitter.unique_days(df))} days")
    print(f"Walk-forward folds: {splitter.n_folds(df)}"
          + (f" (capped to {args.max_folds})" if args.max_folds else ""))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "xgb":
        def xgb_factory() -> XGBWalkForwardModel:
            return XGBWalkForwardModel(params=DEFAULT_PARAMS)

        preds = run_walkforward(
            model_factory=xgb_factory,
            splitter=active,
            df=df,
            features=FEATURES,
            label_col=LABEL_COL,
        )
    else:
        preds = run_walkforward_lstm(
            splitter=active,
            df=df,
            features=FEATURES,
            label_col=LABEL_COL,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            models_dir=MODELS_DIR,
        )

    preds.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(preds):,} rows)")
    _log_and_summarize(
        preds,
        model_name=args.model,
        wandb_group=wandb_group,
        config=config,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
