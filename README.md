# MNQ Intraday Microstructure Benchmark

[![Tests](https://img.shields.io/badge/tests-22%20passing-brightgreen)](.)
[![Python](https://img.shields.io/badge/python-3.11-blue)](.)

Leakage-free intraday futures microstructure benchmark on **MNQ** (Micro E-mini Nasdaq-100): 12 L1 features on 100 ms bars, **XGBoost vs LSTM** under embargoed walk-forward validation, trading simulation with realistic costs, and per-stage latency profiling.

**Full write-up:** [REPORT.md](REPORT.md) · **Results notebook:** [notebooks/03_results.html](notebooks/03_results.html)

## Highlights

- Walk-forward validation with 10 s embargo (no look-ahead)
- Systematic leakage audit + random-label control
- Honest negative results: AUC ~0.51, unprofitable after spread + commission
- All pipeline stages under 5 ms p99 (fits 100 ms bar budget)

## Reproduce

Requires **Python 3.11**, a [Databento](https://databento.com) API key, and ~10 GB disk for parquet. Budget guardrail: $30 max (`src/config.py`).

```bash
# 1. Clone and enter repo
git clone https://github.com/johnhawver/microstructure-benchmark.git
cd microstructure-benchmark

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# 3. API key
cp .env.example .env
# Edit .env: DATABENTO_API_KEY=db-...

# 4. Download MBP-1 data (2025-09-15 → 2025-10-04)
python scripts/download_data.py

# 5. Build 100 ms feature + label parquets
python scripts/build_features.py

# 6. Walk-forward models (XGB ~30 min; LSTM ~1–3 h CPU)
python scripts/run_walkforward.py --model xgb --no-wandb
python scripts/run_walkforward.py --model lstm --no-wandb

# 7. Leakage audit + aggregate metrics
pytest tests/test_leakage.py -q
python scripts/leakage_sanity.py --no-wandb
python scripts/summarize_walkforward.py --no-wandb

# 8. Trading simulation (latency/threshold sweep)
python scripts/run_simulation.py --no-wandb

# 9. Latency benchmarks
python scripts/benchmark_latency.py --no-wandb

# 10. Final figures + notebook
python scripts/build_results_figures.py
jupyter nbconvert --to html notebooks/03_results.ipynb

# 11. Full test suite
pytest -q
```

### Quick path (already have data + walk-forward results)

```bash
source .venv/bin/activate
pytest -q
python scripts/build_results_figures.py
python scripts/run_simulation.py --no-wandb
python scripts/benchmark_latency.py --quick --no-wandb
```

### Optional: Weights & Biases

```bash
wandb login
# Omit --no-wandb on run_walkforward.py, run_simulation.py, benchmark_latency.py
```

## Project layout

```
src/           # features, labels, models, splits, simulate, latency
scripts/       # CLI entry points
tests/         # pytest (leakage, features, splits, simulate)
notebooks/     # EDA, features, results (03_results.html)
data/          # gitignored: raw DBN, parquet, models, results
REPORT.md      # 3-page final report
```

## License

MIT (see repository). Market data © Databento / exchange — for research use per your Databento agreement.
