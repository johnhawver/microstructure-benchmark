"""XGBoost vs LSTM on the same single split (days 1-10 train / day 11 test).

Trains XGB fresh; loads the saved LSTM + train-only scaler by default
(use --retrain-lstm to train from scratch — ~1h on CPU).

  .venv/bin/python scripts/compare_models.py
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
    f1_score,
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
    FeatureScaler,
    MNQLSTM,
    SequenceDataset,
    train_lstm,
)
from src.models.xgb_model import DEFAULT_PARAMS, predict_xgb, train_xgb

LABEL_COL = "label_tb"
N_TRAIN_DAYS = 10
N_TEST_DAYS = 1
LSTM_BATCH = 512
LSTM_EPOCHS = 20
LSTM_LR = 1e-3
MODELS_DIR = ROOT / "data" / "models"
MODEL_PATH = MODELS_DIR / "lstm_split0.pt"
SCALER_PATH = MODELS_DIR / "lstm_split0_scaler.npz"
RESULTS_DIR = ROOT / "data" / "results"
PRED_PATH = RESULTS_DIR / "single_split_compare.parquet"
WANDB_GROUP = "single-split-compare"
WANDB_TABLE_MAX_ROWS = 5_000


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--retrain-lstm",
        action="store_true",
        help="Train LSTM from scratch instead of loading the saved checkpoint.",
    )
    p.add_argument("--no-wandb", action="store_true", help="Skip W&B logging.")
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
    needed = [LABEL_COL, *FEATURES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in feature parquet: {missing}")
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
    return df


def _time_split_train_val(
    df: pl.DataFrame, val_frac: float = 0.2
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n = df.height
    n_val = max(1, int(n * val_frac))
    return df.head(n - n_val), df.tail(n_val)


def _xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(FEATURES).to_numpy().astype(np.float64, copy=False)
    y = df.select(LABEL_COL).to_numpy().reshape(-1)
    ok = np.isfinite(X).all(axis=1)
    if not ok.all():
        X, y = X[ok], y[ok]
    return X, y


def _label_to_class(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(np.int64)
    out = np.empty_like(y)
    out[y == -1] = 0
    out[y == 0] = 1
    out[y == 1] = 2
    return out


def _metrics(y_cls: np.ndarray, probs: np.ndarray) -> dict[str, float]:
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
    }


def _scale_features(df: pl.DataFrame, scaler: FeatureScaler) -> pl.DataFrame:
    X_scaled = scaler.transform(df.select(FEATURES).to_numpy())
    return df.with_columns(
        [pl.Series(name=FEATURES[i], values=X_scaled[:, i]) for i in range(len(FEATURES))]
    )


def _lstm_predict_probs(
    model: MNQLSTM, ds: SequenceDataset, batch_size: int = LSTM_BATCH
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row ends t, y_cls, probs) aligned to SequenceDataset.valid_indices."""
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
        return empty, empty, np.zeros((0, N_CLASSES), dtype=np.float64)
    return ds.valid_indices.copy(), np.concatenate(ys), np.concatenate(probs, axis=0)


def _run_xgb(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    *,
    use_wandb: bool,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)

    run = None
    if use_wandb:
        run = wandb.init(
            project="mnq-microstructure",
            name="xgb-single-split",
            group=WANDB_GROUP,
            job_type="xgb",
            config=DEFAULT_PARAMS,
            reinit=True,
        )

    booster = train_xgb(X_train, y_train, X_val, y_val, params=DEFAULT_PARAMS)
    probs = predict_xgb(booster, X_test)
    y_cls = _label_to_class(y_test)
    metrics = _metrics(y_cls, probs)
    print("XGB test metrics (all bars):", metrics)

    if run is not None:
        wandb.log(metrics)
        run.finish()
    return metrics, y_cls, probs


def _get_lstm_model_and_scaler(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    *,
    retrain: bool,
) -> tuple[MNQLSTM, FeatureScaler]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    have_ckpt = MODEL_PATH.exists() and SCALER_PATH.exists()

    if retrain or not have_ckpt:
        if not retrain:
            print(f"Missing checkpoint; training LSTM → {MODEL_PATH}")
        scaler = FeatureScaler().fit(train_df.select(FEATURES).to_numpy())
        train_ds = SequenceDataset(_scale_features(train_df, scaler), label_col=LABEL_COL)
        val_ds = SequenceDataset(_scale_features(val_df, scaler), label_col=LABEL_COL)
        model = train_lstm(
            train_ds,
            val_ds,
            epochs=LSTM_EPOCHS,
            lr=LSTM_LR,
            batch_size=LSTM_BATCH,
            patience=3,
        )
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        scaler.save(SCALER_PATH)
        return model, scaler

    print(f"Loading LSTM from {MODEL_PATH}")
    scaler = FeatureScaler.load(SCALER_PATH)
    model = MNQLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model, scaler


def _run_lstm(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    *,
    retrain: bool,
    use_wandb: bool,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    run = None
    if use_wandb:
        run = wandb.init(
            project="mnq-microstructure",
            name="lstm-single-split",
            group=WANDB_GROUP,
            job_type="lstm",
            config={
                "seq_len": SEQ_LEN,
                "n_features": N_FEATURES,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "retrain": retrain,
            },
            reinit=True,
        )

    model, scaler = _get_lstm_model_and_scaler(train_df, val_df, retrain=retrain)
    test_ds = SequenceDataset(
        _scale_features(test_df, scaler), label_col=LABEL_COL
    )
    idx, y_cls, probs = _lstm_predict_probs(model, test_ds)
    metrics = _metrics(y_cls, probs)
    print("LSTM test metrics:", metrics)

    if run is not None:
        wandb.log(metrics)
        run.finish()
    return metrics, idx, y_cls, probs


def _log_wandb_tables(
    ts: np.ndarray,
    y_cls: np.ndarray,
    xgb_probs: np.ndarray,
    lstm_probs: np.ndarray,
    xgb_metrics: dict[str, float],
    lstm_metrics: dict[str, float],
) -> None:
    n = len(y_cls)
    step = max(1, n // WANDB_TABLE_MAX_ROWS)
    sel = np.arange(0, n, step)[:WANDB_TABLE_MAX_ROWS]
    pred_table = wandb.Table(
        columns=[
            "ts_event",
            "y_cls",
            "xgb_p_down",
            "xgb_p_zero",
            "xgb_p_up",
            "lstm_p_down",
            "lstm_p_zero",
            "lstm_p_up",
        ],
        data=[
            [
                str(ts[i]),
                int(y_cls[i]),
                float(xgb_probs[i, 0]),
                float(xgb_probs[i, 1]),
                float(xgb_probs[i, 2]),
                float(lstm_probs[i, 0]),
                float(lstm_probs[i, 1]),
                float(lstm_probs[i, 2]),
            ]
            for i in sel
        ],
    )
    compare_table = wandb.Table(
        columns=["model", "accuracy", "log_loss", "macro_f1", "auc_up_vs_down"],
        data=[
            [
                name,
                m["accuracy"],
                m["log_loss"],
                m["macro_f1"],
                m["auc_up_vs_down"],
            ]
            for name, m in (("xgb", xgb_metrics), ("lstm", lstm_metrics))
        ],
    )
    run = wandb.init(
        project="mnq-microstructure",
        name="single-split-summary",
        group=WANDB_GROUP,
        job_type="compare",
        reinit=True,
    )
    wandb.log({"predictions": pred_table, "compare_table": compare_table})
    run.finish()


def main() -> None:
    args = _parse_args()
    paths = _feature_paths(N_TRAIN_DAYS + N_TEST_DAYS)
    train_paths = paths[:N_TRAIN_DAYS]
    test_paths = paths[N_TRAIN_DAYS : N_TRAIN_DAYS + N_TEST_DAYS]
    print("Train days:", [p.name for p in train_paths])
    print("Test day:  ", [p.name for p in test_paths])

    train_all = _load_days(train_paths)
    test_df = _load_days(test_paths)
    train_df, val_df = _time_split_train_val(train_all, val_frac=0.2)

    _, y_xgb, xgb_probs_full = _run_xgb(
        train_df, val_df, test_df, use_wandb=not args.no_wandb
    )
    lstm_metrics, lstm_idx, y_cls, lstm_probs = _run_lstm(
        train_df,
        val_df,
        test_df,
        retrain=args.retrain_lstm,
        use_wandb=not args.no_wandb,
    )

    xgb_probs = xgb_probs_full[lstm_idx]
    if not np.array_equal(y_xgb[lstm_idx], y_cls):
        raise RuntimeError("XGB/LSTM label alignment mismatch on test indices")
    xgb_aligned = _metrics(y_cls, xgb_probs)
    print("XGB metrics on LSTM-aligned bars:", xgb_aligned)

    ts = (
        test_df["ts_event"].to_numpy()[lstm_idx]
        if "ts_event" in test_df.columns
        else np.arange(len(lstm_idx))
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "ts_event": ts,
            "y_cls": y_cls,
            "xgb_p_down": xgb_probs[:, 0],
            "xgb_p_zero": xgb_probs[:, 1],
            "xgb_p_up": xgb_probs[:, 2],
            "lstm_p_down": lstm_probs[:, 0],
            "lstm_p_zero": lstm_probs[:, 1],
            "lstm_p_up": lstm_probs[:, 2],
        }
    ).write_parquet(PRED_PATH)
    print(f"Wrote {PRED_PATH}")

    print("\n=== Single-split comparison (aligned bars) ===")
    print(f"{'model':<8} {'acc':>8} {'logloss':>10} {'macroF1':>10} {'auc':>8}")
    for name, m in (("xgb", xgb_aligned), ("lstm", lstm_metrics)):
        print(
            f"{name:<8} {m['accuracy']:8.4f} {m['log_loss']:10.4f} "
            f"{m['macro_f1']:10.4f} {m['auc_up_vs_down']:8.4f}"
        )

    if not args.no_wandb:
        _log_wandb_tables(ts, y_cls, xgb_probs, lstm_probs, xgb_aligned, lstm_metrics)


if __name__ == "__main__":
    main()
