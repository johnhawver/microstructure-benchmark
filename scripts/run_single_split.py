"""Train XGBoost on days 1-10, evaluate on day 11, log to W&B."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import wandb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PARQUET_DIR
from src.features import FEATURES
from src.models.xgb_model import DEFAULT_PARAMS, predict_xgb, train_xgb

LABEL_COL = "label_tb"
N_TRAIN_DAYS = 10
N_TEST_DAYS = 1


def _feature_paths() -> list[Path]:
    paths = sorted(PARQUET_DIR.glob("mnq_features_*.parquet"))
    if len(paths) < N_TRAIN_DAYS + N_TEST_DAYS:
        raise FileNotFoundError(
            f"Need at least {N_TRAIN_DAYS + N_TEST_DAYS} feature files in {PARQUET_DIR}, "
            f"found {len(paths)}. Run scripts/build_features.py first."
        )
    return paths


def _load_days(paths: list[Path]) -> pl.DataFrame:
    frames = [pl.read_parquet(p) for p in paths]
    df = pl.concat(frames, how="vertical_relaxed")
    needed = FEATURES + [LABEL_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in feature parquet: {missing}")
    # Feature math can produce ±inf (e.g. /spread=0); XGBoost rejects those.
    n_before = df.height
    df = df.with_columns(
        [
            pl.when(pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in FEATURES
        ]
    ).drop_nulls(subset=needed)
    n_dropped = n_before - df.height
    if n_dropped:
        print(f"Dropped {n_dropped} rows with null/inf features")
    return df


def _xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(FEATURES).to_numpy().astype(np.float64, copy=False)
    y = df.select(LABEL_COL).to_numpy().reshape(-1)
    ok = np.isfinite(X).all(axis=1)
    if not ok.all():
        X, y = X[ok], y[ok]
    return X, y


def _time_split_train_val(
    df: pl.DataFrame, val_frac: float = 0.2
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Last val_frac of rows (by time order) = validation for early stopping."""
    n = df.height
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    return df.head(n_train), df.tail(n_val)


def _metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_cls = np.where(y_true == -1, 0, np.where(y_true == 0, 1, 2))
    y_pred = probs.argmax(axis=1)

    acc = float(accuracy_score(y_cls, y_pred))
    ll = float(log_loss(y_cls, probs, labels=[0, 1, 2]))

    mask = y_true != 0
    if mask.sum() >= 2 and len(np.unique(y_true[mask])) == 2:
        auc = float(roc_auc_score((y_true[mask] == 1).astype(int), probs[mask, 2]))
    else:
        auc = float("nan")

    return {"accuracy": acc, "log_loss": ll, "auc_up_vs_down": auc}


def main() -> None:
    paths = _feature_paths()
    train_paths = paths[:N_TRAIN_DAYS]
    test_paths = paths[N_TRAIN_DAYS : N_TRAIN_DAYS + N_TEST_DAYS]

    print("Train days:", [p.name for p in train_paths])
    print("Test day:  ", [p.name for p in test_paths])

    train_all = _load_days(train_paths)
    test_df = _load_days(test_paths)
    train_df, val_df = _time_split_train_val(train_all, val_frac=0.2)

    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)

    print(f"Rows — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    print("Train label counts:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Test  label counts:", dict(zip(*np.unique(y_test, return_counts=True))))

    run = wandb.init(
        project="mnq-microstructure",
        name="xgb-baseline",
        config=DEFAULT_PARAMS,
    )

    booster = train_xgb(X_train, y_train, X_val, y_val, params=DEFAULT_PARAMS)
    probs = predict_xgb(booster, X_test)
    metrics = _metrics(y_test, probs)

    y_cls = np.where(y_test == -1, 0, np.where(y_test == 0, 1, 2))
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_cls, y_pred, labels=[0, 1, 2])

    print("Test metrics:", metrics)
    print("Confusion matrix (rows=true [-1,0,+1], cols=pred):")
    print(cm)

    wandb.log(
        {
            **metrics,
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_cls.tolist(),
                preds=y_pred.tolist(),
                class_names=["-1", "0", "+1"],
            ),
            "best_iteration": getattr(booster, "best_iteration", None),
        }
    )
    run.finish()


if __name__ == "__main__":
    main()