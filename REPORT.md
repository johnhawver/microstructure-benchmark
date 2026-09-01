# MNQ Intraday Microstructure Benchmark — Report

## Project

This project builds a leakage-free benchmark for intraday MNQ (Micro E-mini Nasdaq-100) futures microstructure. Starting from Databento MBP-1 tick data, it engineers 12 L1 features on 100 ms bars, labels bars with a volatility-scaled triple-barrier method, and compares XGBoost vs a 2-layer LSTM under embargoed walk-forward validation. Results are evaluated both as classification metrics and as simulated PnL after spread, commission, and fill-latency costs, with per-stage latency profiling. The goal is not to claim alpha, but to demonstrate a production-style quant research workflow: causal features, rigorous validation, honest negative results, and deployability checks.

## Data

| Field | Value |
|---|---|
| Source | [Databento](https://databento.com) historical API |
| Dataset | `GLBX.MDP3` (CME Globex MDP 3.0) |
| Schema | MBP-1 (top-of-book quotes + trades) |
| Symbol | `MNQ.c.0` (continuous MNQ) |
| Date range | 2025-09-15 → 2025-10-04 (17 calendar parquet days; ~15 usable sessions) |
| Total feature rows | 5,957,250 (100 ms bars, after cleaning) |
| Budget | Hard cap $30 USD (`BUDGET_USD` in `src/config.py`); actual spend well under cap |

Raw and derived parquet live under `data/` (gitignored). Re-download with `scripts/download_data.py` after setting `DATABENTO_API_KEY` in `.env`.

## Features

All features are computed causally on 100 ms bars (no future peek). `mid` is kept for labeling only.

| Feature | Description |
|---|---|
| `microprice_tilt` | `(microprice − mid) / spread` — depth imbalance in spread units |
| `spread_bps` | `spread / mid × 10⁴` |
| `rel_spread` | `spread / 200-bar rolling mean spread` |
| `vol_10` | Rolling std of log returns (1 s window) |
| `vol_50` | Rolling std of log returns (5 s window) |
| `vol_200` | Rolling std of log returns (20 s window) |
| `autocorr_lag1_w50` | Lag-1 return autocorrelation over 50 bars |
| `ofi_sum_100ms` | Order-flow imbalance summed per bar (Cont–Kukanov–Stoikov) |
| `ofi_sum_1000ms` | 10-bar rolling sum of `ofi_sum_100ms` |
| `ofi_sum_5000ms` | 50-bar rolling sum of `ofi_sum_100ms` |
| `trade_imb_50` | `(buy_vol − sell_vol) / (buy_vol + sell_vol)` over 50 bars |
| `kyle_lambda_50` | Rolling `cov(return, signed_vol) / var(signed_vol)` |

## Labels

**Triple-barrier** (`label_tb`): horizon = 50 bars (5 s), barriers at `mid ± k × vol_50 × mid` with `k = 5.0`. Label is +1 if upper barrier hit first, −1 if lower, 0 if horizon expires.

**Class balance** (all days): down −1 **35.7%**, neutral 0 **28.5%**, up +1 **35.8%**.

Forward-return labels (`label_fr`) are also computed but models train on `label_tb`.

## Models

| Hyperparameter | XGBoost | LSTM |
|---|---|---|
| Task | 3-class softmax | 3-class cross-entropy |
| Input | 12 features (tabular) | 50 × 12 sequence (`SEQ_LEN=50`) |
| Key params | `max_depth=5`, `eta=0.05`, `subsample=0.8`, `colsample_bytree=0.8` | `hidden=32` (WF) / 64 (single-split), `layers=2`, `dropout=0.2` |
| Optimizer | Histogram trees, early stopping | Adam `lr=1e-3`, early stop patience=3 |
| Scaling | None | Per-fold `StandardScaler` on train only |
| Training | Up to 500 rounds, 30-round early stop | Up to 20 epochs |

Signed trading score: `p(+1) − p(−1)`.

## Validation

Walk-forward splitter (`WalkForwardSplitter`):

| Parameter | Value |
|---|---|
| Train window | 10 calendar days |
| Test window | 1 day |
| Step | 1 day |
| Embargo | 100 bars (10 s at 100 ms) |
| Folds | 7 test days |

Fresh model (and LSTM scaler) per fold. Last 20% of train used for early stopping.

## Leakage audit

Tests in `tests/test_leakage.py` (all pass):

- Feature causality (no future-row dependence)
- Label horizon (perturbing post-horizon mids does not change labels)
- Embargo gap on real data (≥ 10 s between train end and test start)
- Scaler fit never sees test rows
- No backward time ordering in fold indices

**Random-label control:** XGB walk-forward with shuffled train labels → mean accuracy **0.357** vs real XGB **0.410** and majority baseline **0.360**. No feature-side leakage signal.

## Predictive results

### Per-fold accuracy / AUC (up vs down)

| Fold | XGB acc | XGB AUC | LSTM acc | LSTM AUC |
|---:|---:|---:|---:|---:|
| 0 | 0.408 | 0.514 | 0.410 | 0.515 |
| 1 | 0.411 | 0.498 | 0.400 | 0.490 |
| 2 | 0.410 | 0.509 | 0.412 | 0.510 |
| 3 | 0.411 | 0.510 | 0.409 | 0.511 |
| 4 | 0.415 | 0.500 | 0.415 | 0.501 |
| 5 | 0.408 | 0.510 | 0.417 | 0.518 |
| 6 | 0.408 | 0.519 | 0.409 | 0.519 |

### Aggregate (7 folds)

| Model | Mean acc | Std acc | Macro-F1 | Log-loss | AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.410 | 0.003 | 0.408 | 1.059 | 0.509 |
| LSTM | 0.410 | 0.005 | 0.409 | 1.058 | 0.509 |
| Always-zero | 0.288 | 0.011 | 0.149 | 25.68 | 0.500 |
| Random-label XGB | 0.357 | 0.005 | 0.274 | 1.094 | 0.503 |

**Conclusion:** Models learn label structure vs naive baselines but have **no meaningful directional edge** (AUC ≈ 0.51).

## Trading results

Simulator: `signed_score = p(+1) − p(−1)`, `entry_threshold = 0.2`, `fill_delay_bars = 1`, `commission = $0.35`/side, half-spread on fills.

| Model | Total PnL | Sharpe | Trades | Hit rate |
|---|---:|---:|---:|---:|
| XGBoost | −$38,622 | −8.13 | 26,523 | 3.1% |
| LSTM | −$661 | −13.77 | 431 | 3.7% |

XGB loses more in dollars because it overtrades; LSTM is quieter but still unprofitable.

### Sharpe by fill delay × threshold (XGB)

| delay \\ threshold | 0.1 | 0.2 | 0.3 |
|---:|---:|---:|---:|
| 0 | −20.7 | −8.1 | −6.9 |
| 1 | −21.0 | −8.1 | −6.9 |
| 5 | −21.0 | −8.1 | −6.9 |
| 10 | −21.0 | −8.1 | −6.9 |

Extra latency barely changes Sharpe at fixed threshold (same trade count); low thresholds destroy performance via churn.

See `notebooks/figures/fig04_equity_curves.png` and `fig05_latency_heatmap.png`.

## Latency

Measured on project hardware (CPU). Bar budget = **100 ms** = 100,000 µs.

| Stage | p50 (µs) | p95 (µs) | p99 (µs) |
|---|---:|---:|---:|
| Feature compute | 1,065 | 2,237 | **3,897** |
| XGB inference | 156 | 661 | 2,564 |
| LSTM inference | 497 | 2,270 | 5,141 |

Feature compute is the largest stage at p99 (~3.9% of bar budget). LSTM inference is ~2× slower than XGB at p99. All stages are deployable at 100 ms; **alpha, not compute, is the binding constraint**.

## What I'd do next

- **Raise the decision bar:** Higher entry thresholds and meta-labeling (bet only when \|score\| is large) to cut XGB churn before tuning architecture.
- **Richer event-time features:** Queue-reactive OFI, trade clustering, and session/regime indicators; keep the same leakage audit harness.
- **Proper backtest:** Queue-position model, partial fills, and live-latency distribution instead of fixed 1-bar delay.

---

*Figures: `notebooks/03_results.ipynb` / `notebooks/03_results.html`. Tests: `pytest -q` (22 passing).*
