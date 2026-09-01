"""Aggregate walk-forward metrics into summary.csv + wf_metrics.png.

  .venv/bin/python scripts/summarize_walkforward.py
  .venv/bin/python scripts/summarize_walkforward.py --no-wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "data" / "results"
FIG_DIR = ROOT / "notebooks" / "figures"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
FIG_PATH = FIG_DIR / "wf_metrics.png"

XGB_PATH = RESULTS_DIR / "xgb_wf.parquet"
LSTM_PATH = RESULTS_DIR / "lstm_wf.parquet"
RANDOM_PATH = RESULTS_DIR / "xgb_wf_random_labels.parquet"


def _signed_to_cls(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(np.int64)
    out = np.empty_like(y)
    out[y == -1] = 0
    out[y == 0] = 1
    out[y == 1] = 2
    return out


def _fold_metrics_from_probs(g: pd.DataFrame) -> dict[str, float]:
    y_cls = _signed_to_cls(g["y_true"].to_numpy())
    probs = np.column_stack(
        [g["prob_down"].to_numpy(), g["prob_zero"].to_numpy(), g["prob_up"].to_numpy()]
    )
    y_pred = probs.argmax(axis=1)
    mask = y_cls != 1
    if mask.sum() >= 2 and len(np.unique(y_cls[mask])) == 2:
        auc = float(roc_auc_score((y_cls[mask] == 2).astype(int), probs[mask, 2]))
    else:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_cls, y_pred)),
        "log_loss": float(log_loss(y_cls, probs, labels=[0, 1, 2])),
        "macro_f1": float(
            f1_score(y_cls, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
        ),
        "auc_up_vs_down": auc,
    }


def _always_zero_fold_metrics(g: pd.DataFrame) -> dict[str, float]:
    y_cls = _signed_to_cls(g["y_true"].to_numpy())
    y_pred = np.ones_like(y_cls)
    probs = np.zeros((len(y_cls), 3), dtype=np.float64)
    probs[:, 1] = 1.0
    mask = y_cls != 1
    if mask.sum() >= 2 and len(np.unique(y_cls[mask])) == 2:
        auc = float(roc_auc_score((y_cls[mask] == 2).astype(int), probs[mask, 2]))
    else:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_cls, y_pred)),
        "log_loss": float(log_loss(y_cls, probs, labels=[0, 1, 2])),
        "macro_f1": float(
            f1_score(y_cls, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
        ),
        "auc_up_vs_down": auc,
    }


def _per_fold_table(
    preds: pd.DataFrame, metric_fn=_fold_metrics_from_probs
) -> pd.DataFrame:
    rows = []
    for fold, g in preds.groupby("fold", sort=True):
        m = metric_fn(g)
        m["fold"] = int(fold)
        m["test_start"] = pd.Timestamp(g["ts_event"].min())
        rows.append(m)
    return pd.DataFrame(rows)


def _aggregate_row(per_fold: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_accuracy": float(per_fold["accuracy"].mean()),
        "mean_macro_f1": float(per_fold["macro_f1"].mean()),
        "mean_log_loss": float(per_fold["log_loss"].mean()),
        "mean_auc": float(per_fold["auc_up_vs_down"].mean()),
        "std_accuracy": float(per_fold["accuracy"].std()),
    }


def build_summary_table(
    preds_xgb: pd.DataFrame,
    preds_lstm: pd.DataFrame,
    preds_random: pd.DataFrame,
) -> pd.DataFrame:
    """Rows: XGBoost, LSTM, Always-zero, Random-label control."""
    rows = []
    for name, preds, fn in [
        ("XGBoost", preds_xgb, _fold_metrics_from_probs),
        ("LSTM", preds_lstm, _fold_metrics_from_probs),
        ("Always-zero baseline", preds_xgb, _always_zero_fold_metrics),
        ("Random-label control (XGB)", preds_random, _fold_metrics_from_probs),
    ]:
        per_fold = _per_fold_table(preds, fn)
        agg = _aggregate_row(per_fold)
        rows.append({"model": name, **agg})
    return pd.DataFrame(rows)


def plot_walkforward_metrics(
    metrics_xgb: pd.DataFrame,
    metrics_lstm: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(metrics_xgb["test_start"], metrics_xgb["accuracy"], marker="o", label="XGBoost")
    ax.plot(metrics_lstm["test_start"], metrics_lstm["accuracy"], marker="s", label="LSTM")
    ax.axhline(summary.loc[summary["model"] == "Always-zero baseline", "mean_accuracy"].iloc[0],
               color="gray", linestyle="--", linewidth=0.8, label="always-zero")
    ax.set_title("Per-fold accuracy")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 1]
    ax.plot(metrics_xgb["test_start"], metrics_xgb["auc_up_vs_down"], marker="o", label="XGBoost")
    ax.plot(metrics_lstm["test_start"], metrics_lstm["auc_up_vs_down"], marker="s", label="LSTM")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_title("Per-fold AUC (up vs down)")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 0]
    models = summary["model"].tolist()
    x = np.arange(len(models))
    ax.bar(x, summary["mean_accuracy"], yerr=summary["std_accuracy"], capsize=4, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_title("Mean accuracy ± std (per-fold)")
    ax.set_ylabel("accuracy")

    ax = axes[1, 1]
    width = 0.35
    ax.bar(x - width / 2, summary["mean_macro_f1"], width, label="macro-F1")
    ax.bar(x + width / 2, summary["mean_auc"], width, label="AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_title("Mean macro-F1 and AUC")
    ax.legend(fontsize=8)

    fig.suptitle("Walk-forward summary — XGBoost vs LSTM vs baselines", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    for path in (XGB_PATH, LSTM_PATH, RANDOM_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

    preds_xgb = pd.read_parquet(XGB_PATH)
    preds_lstm = pd.read_parquet(LSTM_PATH)
    preds_random = pd.read_parquet(RANDOM_PATH)

    summary = build_summary_table(preds_xgb, preds_lstm, preds_random)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    metrics_xgb = _per_fold_table(preds_xgb)
    metrics_lstm = _per_fold_table(preds_lstm)
    plot_walkforward_metrics(metrics_xgb, metrics_lstm, summary, FIG_PATH)

    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {FIG_PATH}")
    print("\n=== Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not args.no_wandb:
        run = wandb.init(
            project="mnq-microstructure",
            name="wf-summary",
            group="walkforward-summary",
            job_type="aggregate",
            settings=wandb.Settings(init_timeout=300),
        )
        wandb.log({"walkforward_summary": wandb.Table(dataframe=summary)})
        wandb.log({"wf_metrics_png": wandb.Image(str(FIG_PATH))})
        run.finish()


if __name__ == "__main__":
    main()
