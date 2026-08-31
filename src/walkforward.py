"""Walk-forward training / evaluation loop."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
