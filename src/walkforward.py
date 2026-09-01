"""Walk-forward training / evaluation loop."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import polars as pl

from src.splits import WalkForwardSplitter

# Signed labels {-1, 0, +1} → class ids for metrics / argmax alignment.
_LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
_CLASS_TO_SIGNED = {0: -1, 1: 0, 2: 1}


class FoldModel(Protocol):
    """Minimal interface for a tabular walk-forward model."""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities shape ``[N, 3]`` for classes {0,1,2} = {-1,0,+1}."""
        ...


def _map_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(np.int64)
    out = np.empty_like(y)
    for raw, cls in _LABEL_TO_CLASS.items():
        out[y == raw] = cls
    return out


def _time_split_train_val(
    idx: np.ndarray, val_frac: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Last ``val_frac`` of ``idx`` (already time-ordered) → validation."""
    n = int(idx.size)
    if n < 2:
        return idx, idx
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    if n_train < 1:
        n_train, n_val = 1, n - 1
    return idx[:n_train], idx[n_train:]


def run_walkforward(
    model_factory: Callable[[], FoldModel],
    splitter: WalkForwardSplitter,
    df: pl.DataFrame,
    features: Sequence[str],
    label_col: str = "label_tb",
    val_frac: float = 0.2,
) -> pd.DataFrame:
    """Train a fresh model per fold; return long-form out-of-sample predictions.

    Columns: ``fold, ts_event, y_true, y_pred, prob_up, prob_down, prob_zero``.
    ``y_true`` / ``y_pred`` are signed labels in ``{-1, 0, +1}``.
    Indices from ``splitter`` refer to ``df.sort("ts_event")``.
    """
    needed = ["ts_event", label_col, *features]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    sorted_df = df.sort("ts_event")
    X_all = sorted_df.select(list(features)).to_numpy().astype(np.float64, copy=False)
    y_all = sorted_df.select(label_col).to_numpy().reshape(-1)
    ts_all = sorted_df["ts_event"].to_numpy()

    rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(sorted_df)):
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        # Drop non-finite feature rows inside this fold's train/test.
        train_ok = np.isfinite(X_all[train_idx]).all(axis=1)
        test_ok = np.isfinite(X_all[test_idx]).all(axis=1)
        train_idx = train_idx[train_ok]
        test_idx = test_idx[test_ok]
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        tr_idx, va_idx = _time_split_train_val(train_idx, val_frac=val_frac)
        model = model_factory()
        model.fit(
            X_all[tr_idx],
            y_all[tr_idx],
            X_all[va_idx],
            y_all[va_idx],
        )
        probs = np.asarray(model.predict_proba(X_all[test_idx]), dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] != 3:
            raise ValueError(f"predict_proba must return [N, 3], got {probs.shape}")
        # XGBoost softprob can drift off the simplex by float noise.
        probs = np.clip(probs, 1e-15, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)

        y_true = y_all[test_idx].astype(np.int64)
        y_pred_cls = probs.argmax(axis=1)
        y_pred = np.vectorize(_CLASS_TO_SIGNED.get)(y_pred_cls).astype(np.int64)

        for i in range(test_idx.size):
            rows.append(
                {
                    "fold": fold,
                    "ts_event": ts_all[test_idx[i]],
                    "y_true": int(y_true[i]),
                    "y_pred": int(y_pred[i]),
                    "prob_down": float(probs[i, 0]),
                    "prob_zero": float(probs[i, 1]),
                    "prob_up": float(probs[i, 2]),
                }
            )
        print(
            f"fold {fold}: train={tr_idx.size} val={va_idx.size} "
            f"test={test_idx.size}"
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "fold",
                "ts_event",
                "y_true",
                "y_pred",
                "prob_up",
                "prob_down",
                "prob_zero",
            ]
        )
    return pd.DataFrame(rows)


def _scale_frame(
    df: pl.DataFrame, features: Sequence[str], scaler: Any
) -> pl.DataFrame:
    X_scaled = scaler.transform(df.select(list(features)).to_numpy())
    return df.with_columns(
        [
            pl.Series(name=features[i], values=X_scaled[:, i])
            for i in range(len(features))
        ]
    )


def _predict_sequence_probs(
    model: Any,
    ds: Any,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (local_end_indices, y_cls, probs) for a SequenceDataset."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    ys: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            ys.append(y.numpy())
    if not ys:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.zeros((0, 3), dtype=np.float64)
    return (
        ds.valid_indices.copy(),
        np.concatenate(ys),
        np.concatenate(probs, axis=0),
    )


def run_walkforward_lstm(
    splitter: WalkForwardSplitter,
    df: pl.DataFrame,
    features: Sequence[str],
    label_col: str = "label_tb",
    val_frac: float = 0.2,
    seq_len: int = 50,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
    patience: int = 3,
    hidden_size: int = 32,
    models_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Walk-forward LSTM with per-fold train-only scaling and a fresh model.

    For each fold:
      1. Fit ``FeatureScaler`` on train rows only.
      2. Train ``MNQLSTM`` on scaled train (val = last ``val_frac``).
      3. Predict test ends (with ``seq_len-1`` bars of pre-test context for
         windowing — context is input only, never used as a label target).
      4. Optionally save ``lstm_fold{k}.pt`` + scaler under ``models_dir``.

    Returns the same long-form schema as ``run_walkforward``.
    """
    from pathlib import Path as _Path

    import torch

    from src.models.lstm_model import (
        FeatureScaler,
        SequenceDataset,
        train_lstm,
    )

    needed = ["ts_event", label_col, *features]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    sorted_df = df.sort("ts_event")
    y_all = sorted_df.select(label_col).to_numpy().reshape(-1)
    ts_all = sorted_df["ts_event"].to_numpy()
    X_raw = sorted_df.select(list(features)).to_numpy().astype(np.float64, copy=False)

    out_dir = Path(models_dir) if models_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    test_set_cache: set[int] = set()

    for fold, (train_idx, test_idx) in enumerate(splitter.split(sorted_df)):
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        train_ok = np.isfinite(X_raw[train_idx]).all(axis=1)
        test_ok = np.isfinite(X_raw[test_idx]).all(axis=1)
        train_idx = train_idx[train_ok]
        test_idx = test_idx[test_ok]
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        # Train-only scaler (never fit on test).
        scaler = FeatureScaler().fit(X_raw[train_idx])

        tr_idx, va_idx = _time_split_train_val(train_idx, val_frac=val_frac)
        train_frame = _scale_frame(sorted_df[tr_idx.tolist()], features, scaler)
        val_frame = _scale_frame(sorted_df[va_idx.tolist()], features, scaler)

        train_ds = SequenceDataset(
            train_frame, feature_cols=list(features), label_col=label_col, seq_len=seq_len
        )
        val_ds = SequenceDataset(
            val_frame, feature_cols=list(features), label_col=label_col, seq_len=seq_len
        )
        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"fold {fold}: skipping — empty sequence dataset")
            continue

        print(
            f"fold {fold}: train_seq={len(train_ds)} val_seq={len(val_ds)} "
            f"test_rows={test_idx.size} hidden={hidden_size}",
            flush=True,
        )
        model = train_lstm(
            train_ds,
            val_ds,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            num_workers=0,
            hidden_size=hidden_size,
        )

        if out_dir is not None:
            torch.save(model.state_dict(), out_dir / f"lstm_fold{fold}.pt")
            scaler.save(out_dir / f"lstm_fold{fold}_scaler.npz")

        # Prepend seq_len-1 bars before test so early test ends have a full window.
        context_start = max(0, int(test_idx.min()) - (seq_len - 1))
        slice_end = int(test_idx.max()) + 1
        slice_idx = np.arange(context_start, slice_end, dtype=np.int64)
        test_frame = _scale_frame(sorted_df[slice_idx.tolist()], features, scaler)
        test_ds = SequenceDataset(
            test_frame, feature_cols=list(features), label_col=label_col, seq_len=seq_len
        )

        local_ends, y_cls, probs = _predict_sequence_probs(
            model, test_ds, batch_size=batch_size
        )
        probs = np.clip(probs, 1e-15, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)

        test_set_cache = set(int(i) for i in test_idx)
        y_pred_cls = probs.argmax(axis=1)
        kept = 0
        for j, local_t in enumerate(local_ends):
            global_t = int(context_start + local_t)
            if global_t not in test_set_cache:
                continue
            kept += 1
            rows.append(
                {
                    "fold": fold,
                    "ts_event": ts_all[global_t],
                    "y_true": int(y_all[global_t]),
                    "y_pred": int(_CLASS_TO_SIGNED[int(y_pred_cls[j])]),
                    "prob_down": float(probs[j, 0]),
                    "prob_zero": float(probs[j, 1]),
                    "prob_up": float(probs[j, 2]),
                }
            )
        print(f"fold {fold}: wrote {kept} test predictions")

    if not rows:
        return pd.DataFrame(
            columns=[
                "fold",
                "ts_event",
                "y_true",
                "y_pred",
                "prob_up",
                "prob_down",
                "prob_zero",
            ]
        )
    return pd.DataFrame(rows)
