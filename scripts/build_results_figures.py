"""Build results figures for notebooks/03_results.ipynb.

  .venv/bin/python scripts/build_results_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
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
from src.simulate import Simulator, summary_stats

FIG_DIR = ROOT / "notebooks" / "figures"
RESULTS_DIR = ROOT / "data" / "results"


def _signed_to_cls(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(np.int64)
    out = np.empty_like(y)
    out[y == -1] = 0
    out[y == 0] = 1
    out[y == 1] = 2
    return out


def fold_metrics(g: pd.DataFrame) -> dict:
    y_cls = _signed_to_cls(g["y_true"].to_numpy())
    probs = np.column_stack(
        [g["prob_down"].to_numpy(), g["prob_zero"].to_numpy(), g["prob_up"].to_numpy()]
    ).astype(np.float64)
    # Model outputs are float32, so rows sum to 1 only to ~1e-7. Renormalize in
    # float64 so log_loss does not warn about probabilities not summing to one.
    probs /= probs.sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)
    mask = y_cls != 1
    if mask.sum() >= 2 and len(np.unique(y_cls[mask])) == 2:
        auc = float(roc_auc_score((y_cls[mask] == 2).astype(int), probs[mask, 2]))
    else:
        auc = float("nan")
    return {
        "fold": int(g["fold"].iloc[0]),
        "accuracy": float(accuracy_score(y_cls, y_pred)),
        "log_loss": float(log_loss(y_cls, probs, labels=[0, 1, 2])),
        "macro_f1": float(
            f1_score(y_cls, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
        ),
        "auc_up_vs_down": auc,
        "n": len(g),
        "test_start": pd.Timestamp(g["ts_event"].min()),
    }


def fig01_class_balance(out: Path = FIG_DIR / "fig01_class_balance.png") -> None:
    paths = sorted(PARQUET_DIR.glob("mnq_features_*.parquet"))
    df = pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")
    counts = (
        df.group_by("label_tb")
        .len()
        .sort("label_tb")
        .rename({"len": "count"})
    )
    labels = { -1: "down (-1)", 0: "neutral (0)", 1: "up (+1)" }
    names = [labels[int(r["label_tb"])] for r in counts.iter_rows(named=True)]
    vals = counts["count"].to_list()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, color=["#c44e52", "#999999", "#55a868"])
    ax.set_title("Triple-barrier label distribution (all feature days)")
    ax.set_ylabel("count")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig02_feature_corr(
    day: str = "2025-09-15", out: Path = FIG_DIR / "fig02_feature_corr.png"
) -> None:
    path = PARQUET_DIR / f"mnq_features_{day}.parquet"
    df = pl.read_parquet(path).select(FEATURES).drop_nulls()
    corr = df.to_pandas().corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True)
    ax.set_title(f"Feature correlation — {day}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig03_wf_metrics(
    out: Path = FIG_DIR / "fig03_wf_metrics.png",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds_xgb = pd.read_parquet(RESULTS_DIR / "xgb_wf.parquet")
    preds_lstm = pd.read_parquet(RESULTS_DIR / "lstm_wf.parquet")
    metrics_xgb = pd.DataFrame([fold_metrics(g) for _, g in preds_xgb.groupby("fold")])
    metrics_lstm = pd.DataFrame([fold_metrics(g) for _, g in preds_lstm.groupby("fold")])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    axes[0].plot(
        metrics_xgb["test_start"], metrics_xgb["accuracy"], marker="o", label="XGBoost"
    )
    axes[0].plot(
        metrics_lstm["test_start"], metrics_lstm["accuracy"], marker="s", label="LSTM"
    )
    axes[1].plot(
        metrics_xgb["test_start"],
        metrics_xgb["auc_up_vs_down"],
        marker="o",
        label="XGBoost",
    )
    axes[1].plot(
        metrics_lstm["test_start"],
        metrics_lstm["auc_up_vs_down"],
        marker="s",
        label="LSTM",
    )
    axes[0].set_title("Per-fold accuracy")
    axes[0].set_ylabel("accuracy")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("Per-fold AUC (up vs down)")
    axes[1].set_ylabel("AUC")
    for ax in axes:
        ax.set_xlabel("test-day start")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Walk-forward — XGBoost vs LSTM", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return metrics_xgb, metrics_lstm


def _load_quotes() -> pl.DataFrame:
    paths = sorted(PARQUET_DIR.glob("mnq_bars100ms_*.parquet"))
    lf = pl.concat([pl.scan_parquet(p) for p in paths], how="vertical_relaxed")
    return (
        lf.select(["ts_event", "mid", "bid", "ask"])
        .collect()
        .unique(subset=["ts_event"], keep="last")
        .with_columns(pl.col("ts_event").dt.replace_time_zone("UTC"))
    )


def _prepare_sim_frame(preds: pd.DataFrame, quotes: pl.DataFrame) -> pl.DataFrame:
    frame = (
        pl.from_pandas(preds)
        .with_columns(
            (pl.col("prob_up") - pl.col("prob_down")).alias("signed_score"),
            pl.col("ts_event").dt.replace_time_zone("UTC"),
        )
    )
    return frame.join(quotes, on="ts_event", how="inner")


def fig04_equity_curves(out: Path = FIG_DIR / "fig04_equity_curves.png") -> None:
    quotes = _load_quotes()
    sim = Simulator(
        entry_threshold=0.2,
        fill_delay_bars=1,
        commission_per_trade=0.35,
    )
    curves: dict[str, pl.DataFrame] = {}
    for name, path in [("XGBoost", "xgb_wf.parquet"), ("LSTM", "lstm_wf.parquet")]:
        preds = pd.read_parquet(RESULTS_DIR / path)
        sim_in = _prepare_sim_frame(preds, quotes)
        curves[name] = sim.run(sim_in)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, sim_out in curves.items():
        pdf = sim_out.select(["ts_event", "cum_pnl"]).to_pandas()
        ax.plot(pdf["ts_event"], pdf["cum_pnl"], label=name, linewidth=0.8, alpha=0.9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(
        "Equity curves — threshold=0.2, 1-bar fill delay, $0.35/side commission"
    )
    ax.set_xlabel("ts_event")
    ax.set_ylabel("cumulative PnL ($)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig05_latency_heatmap(out: Path = FIG_DIR / "fig05_latency_heatmap.png") -> None:
    sweep = pd.read_parquet(RESULTS_DIR / "sim_sweep.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    im = None
    for ax, model in zip(axes, ["xgb", "lstm"]):
        sub = sweep[sweep["model"] == model]
        pivot = sub.pivot(
            index="fill_delay_bars", columns="entry_threshold", values="sharpe"
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-25, vmax=5)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("entry_threshold")
        ax.set_ylabel("fill_delay_bars")
        ax.set_title(f"{model.upper()} — Sharpe")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(
                    j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=8
                )
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Sharpe")
    fig.suptitle("Sharpe vs fill latency and entry threshold", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig06_inference_p99(out: Path = FIG_DIR / "fig06_inference_p99.png") -> None:
    lat = pd.read_csv(RESULTS_DIR / "latency.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(lat))
    w = 0.25
    ax.bar(x - w, lat["p50_us"], width=w, label="p50")
    ax.bar(x, lat["p95_us"], width=w, label="p95")
    ax.bar(x + w, lat["p99_us"], width=w, label="p99")
    ax.set_xticks(x)
    ax.set_xticklabels(lat["stage"], rotation=15, ha="right")
    ax.set_ylabel("latency (µs, log scale)")
    ax.set_title("Per-stage latency vs 100 ms bar budget")
    # Log scale: stages run ~1e2-1e3 us against a 1e5 us budget, so a linear
    # axis flattens every bar to the baseline and hides the p50/p95/p99 spread.
    ax.set_yscale("log")
    ax.set_ylim(100, 200_000)
    ax.axhline(100_000, color="crimson", linestyle="--", linewidth=1.0)
    ax.text(
        len(lat) - 0.5,
        100_000,
        " 100 ms bar budget",
        color="crimson",
        va="bottom",
        ha="right",
        fontsize=9,
    )
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_summary_table() -> pd.DataFrame:
    from scripts.summarize_walkforward import build_summary_table

    preds_xgb = pd.read_parquet(RESULTS_DIR / "xgb_wf.parquet")
    preds_lstm = pd.read_parquet(RESULTS_DIR / "lstm_wf.parquet")
    preds_random = pd.read_parquet(RESULTS_DIR / "xgb_wf_random_labels.parquet")
    return build_summary_table(preds_xgb, preds_lstm, preds_random)


def build_all_figures() -> None:
    print("fig01 class balance...")
    fig01_class_balance()
    print("fig02 feature corr...")
    fig02_feature_corr()
    print("fig03 walk-forward metrics...")
    fig03_wf_metrics()
    print("fig04 equity curves (realistic costs)...")
    fig04_equity_curves()
    print("fig05 latency heatmap...")
    fig05_latency_heatmap()
    print("fig06 inference latency...")
    fig06_inference_p99()
    print(f"All figures saved under {FIG_DIR}/")


def main() -> None:
    os.chdir(ROOT)
    build_all_figures()
    summary = build_summary_table()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
