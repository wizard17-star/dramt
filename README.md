# DRAM-T: Dynamic and Risk-Aware Multimodal Transformer

MSc thesis project implementation: a multimodal Transformer that fuses OHLCV market
data, mixed-frequency macroeconomic indicators, and FinBERT-scored GDELT news
sentiment to jointly produce multi-horizon (1-10 trading day) return forecasts and
explicit risk outputs (conditional volatility, cross-asset correlation, portfolio VaR)
for a 5-stock tech portfolio (AAPL, GOOGL, MSFT, AMZN, META; NVDA and ^GSPC tracked
as extra reference series).

## Compute note

This project was built and run on a machine with **no CUDA-capable GPU** (Intel Iris
Xe integrated graphics only). All training code auto-detects CUDA and will use it if
available (`src/utils/seed.py::get_device`), but the default `config.yaml` sweep
grids, model width/depth, and epoch budgets are sized for **CPU-only** training. See
the `notes.compute_assumption` field in `config.yaml` for what to restore if a CUDA
GPU becomes available later.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python check_env.py         # confirms Python/torch/CUDA status
pytest tests/                # smoke tests
```

## Reproduce

```bash
python download_data.py
python -m src.train --config experiments/full.yaml
```

## Repo structure

See `PROJECT_BUILD_PROMPT.md` (in the parent thesis folder) for the full spec this
repo implements. Layout:

```
config.yaml            hyperparameters, paths, seeds, sweep grids, portfolio weights
download_data.py        fetch equities + macro + GDELT news
src/data/                loaders, indicators, macro_midas, sentiment_gdelt_finbert, windows, splits
src/models/              dramt.py, attention.py, heads.py, baselines/
src/train.py, evaluate.py, ablation.py, stats_tests.py
src/utils/                metrics.py, var_es.py, plotting.py, seed.py, config.py
experiments/, runs/, results/ (tables_*.csv, tables_*.tex, figures/, results.md)
tests/
```

## Status

- [x] M0 - repo scaffold, config, seeding, env check
- [x] M1 - data download (equities + macro done; GDELT news pull runs in background, heavily rate-limited)
- [x] M2 - features / sentiment / MIDAS / windowing / splits (sentiment features pending GDELT data)
- [x] M3 - DRAM-T model
- [x] M4 - training loop (fold 0 plumbing run passed on CPU)
- [ ] M5 - baselines
- [ ] M6 - full walk-forward + sweep
- [ ] M7 - ablations + significance tests
- [ ] M8 - tables + figures
- [ ] M9 - thesis handoff
