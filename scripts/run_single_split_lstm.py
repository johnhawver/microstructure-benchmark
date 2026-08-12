"""Train LSTM on days 1-10, evaluate on day 11, log to W&B.

Sanity (Day 18 step 6):
  .venv/bin/python scripts/run_single_split_lstm.py --sanity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PARQUET_DIR
from src.features import FEATURES
from src.models.lstm_model import (
    HIDDEN_SIZE,
    N_CLASSES,
    N_FEATURES,
    NUM_LAYERS,
    SEQ_LEN,
    SequenceDataset,
    train_lstm,
)

LABEL_COL = "label_tb"
N_TRAIN_DAYS = 10
N_TEST_DAYS = 1
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 512


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sanity",
        action="store_true",
        help="Day 18 step 6: one day only, 5 epochs (check train loss drops).",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip Weights & Biases (offline sanity).",
    )
    return p.parse_args()


def _feature_paths(n_needed: int) -> list[Path]:
    paths = sorted(PARQUET_DIR.glob("mnq_features_*.parquet"))
    if len(paths) < n_needed:
        raise FileNotFoundError(
            f"Need at least {n_needed} feature files in {PARQUET_DIR}, "
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
    # Feature math can produce ±inf; SequenceDataset also skips non-finite windows.
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


def _time_split_train_val(
    df: pl.DataFrame, val_frac: float = 0.2
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Last val_frac of rows (by time order) = validation for early stopping."""
    n = df.height
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    return df.head(n_train), df.tail(n_val)


def _predict_probs(
    model, ds: SequenceDataset, batch_size: int = BATCH_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true class ids [N], probs [N, 3]) for a SequenceDataset."""
    device = next(model.parameters()).device
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = F.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            ys.append(y.numpy())
    if not ys:
        return np.array([], dtype=np.int64), np.zeros((0, N_CLASSES), dtype=np.float64)
    return np.concatenate(ys), np.concatenate(probs, axis=0)


def _metrics(y_cls: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """y_cls in {0,1,2} = {-1,0,+1}; probs shape [N, 3]."""
    y_pred = probs.argmax(axis=1)
    acc = float(accuracy_score(y_cls, y_pred))
    ll = float(log_loss(y_cls, probs, labels=[0, 1, 2]))

    # AUC of p(+1) vs p(-1) on non-flat labels only (same idea as XGB script).
    mask = y_cls != 1
    if mask.sum() >= 2 and len(np.unique(y_cls[mask])) == 2:
        auc = float(roc_auc_score((y_cls[mask] == 2).astype(int), probs[mask, 2]))
    else:
        auc = float("nan")

    return {"accuracy": acc, "log_loss": ll, "auc_up_vs_down": auc}


def main() -> None:
    args = _parse_args()
    sanity = args.sanity
    n_train_days = 1 if sanity else N_TRAIN_DAYS
    n_test_days = 0 if sanity else N_TEST_DAYS
    epochs = args.epochs if args.epochs is not None else (5 if sanity else EPOCHS)
    # Sanity: don't early-stop before all 5 epochs finish.
    patience = epochs if sanity else 3

    n_needed = n_train_days + max(n_test_days, 0)
    paths = _feature_paths(n_needed)
    train_paths = paths[:n_train_days]
    test_paths = paths[n_train_days : n_train_days + n_test_days]

    print("Mode:      ", "sanity (1 day, 5 epochs)" if sanity else "full single split")
    print("Train days:", [p.name for p in train_paths])
    print("Test day:  ", [p.name for p in test_paths] if test_paths else "(none — eval on val)")

    train_all = _load_days(train_paths)
    train_df, val_df = _time_split_train_val(train_all, val_frac=0.2)
    test_df = _load_days(test_paths) if test_paths else val_df

    train_ds = SequenceDataset(train_df, label_col=LABEL_COL)
    val_ds = SequenceDataset(val_df, label_col=LABEL_COL)
    test_ds = SequenceDataset(test_df, label_col=LABEL_COL)

    print(
        f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )
    print(
        "Train label counts:",
        dict(zip(*np.unique(train_ds.y[train_ds.valid_indices], return_counts=True))),
    )
    print(
        "Test  label counts:",
        dict(zip(*np.unique(test_ds.y[test_ds.valid_indices], return_counts=True))),
    )

    run = None
    if not args.no_wandb:
        run = wandb.init(
            project="mnq-microstructure",
            name="lstm-sanity" if sanity else "lstm-baseline",
            config={
                "sanity": sanity,
                "seq_len": SEQ_LEN,
                "n_features": N_FEATURES,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "n_classes": N_CLASSES,
                "epochs": epochs,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "n_train_days": n_train_days,
                "n_test_days": n_test_days,
            },
        )

    model = train_lstm(
        train_ds,
        val_ds,
        epochs=epochs,
        lr=LR,
        batch_size=BATCH_SIZE,
        patience=patience,
    )
    y_cls, probs = _predict_probs(model, test_ds)
    metrics = _metrics(y_cls, probs)

    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_cls, y_pred, labels=[0, 1, 2])

    print("Test metrics:", metrics)
    print("Confusion matrix (rows=true [-1,0,+1], cols=pred):")
    print(cm)

    if run is not None:
        wandb.log(
            {
                **metrics,
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_cls.tolist(),
                    preds=y_pred.tolist(),
                    class_names=["-1", "0", "+1"],
                ),
                "n_train_samples": len(train_ds),
                "n_val_samples": len(val_ds),
                "n_test_samples": len(test_ds),
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
