# DRAM-T: Dynamic and Risk-Aware Multimodal Transformer

Code and results for the MSc thesis *Dynamic and Risk-Aware Multimodal Transformer
Models for Financial Forecasting Using External Data Streams* (Serhat Aslan, s34090,
Polish-Japanese Academy of Information Technology).

DRAM-T is a multimodal Transformer that fuses OHLCV market data, mixed-frequency
macroeconomic indicators and FinBERT-scored GDELT news sentiment to jointly produce
multi-horizon (1-10 trading day) return forecasts together with explicit risk outputs:
conditional volatility, cross-asset correlation and portfolio VaR. The portfolio is
five large-cap technology stocks (AAPL, GOOGL, MSFT, AMZN, META), with NVDA and ^GSPC
tracked as reference series.

## What the results say

`results/results.md` is the full record. In short: the model produces well-calibrated
risk output but no reliable conditional-mean signal at any horizon. The volatility head
reaches parity with GARCH(1,1) rather than beating it, the correlation output lands
within 1.3% of DCC-GARCH, and every ablation effect is smaller than its own seed
spread. Those negative point-accuracy findings are reported as findings, not hidden.

## Compute

Final results were produced on an NVIDIA RTX 5070 (12 GB, CUDA 12.8): 88 model runs
under 4-fold expanding walk-forward validation. Training code auto-detects CUDA
(`src/utils/seed.py::get_device`) and falls back to CPU, though the sweep grids in
`config.yaml` are sized for the GPU configuration described in
`notes.compute_assumption`.

An earlier generation of results was produced CPU-only on different data, before the
FinBERT cache finished scoring the retrieved corpus. Those numbers are **not**
comparable to the current ones and are kept under `results_cpu_era/` for provenance.
`results/results.md` explains exactly what changed.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python check_env.py         # confirms Python / torch / CUDA status
pytest tests/               # unit tests, including look-ahead checks
```

## Reproduce

```bash
python download_data.py                              # equities + macro + GDELT news
python -m src.data.build_features                    # requires the FinBERT cache
python -m src.sweep --folds all --shard k --nshards 4
python -m scripts.gpu_chain --parallel 4             # all runs + analyses
python -m scripts.summarize_results
```

Seeds are fixed. Standardizers, sigma calibration and the GARCH/HAR parameters are
fitted on train and validation splits only. The test suite includes explicit
look-ahead tests for the rolling calibration, the regime signals and the permutation
ablation.

## Repo structure

```
config.yaml              hyperparameters, paths, seeds, sweep grids, portfolio weights
download_data.py         fetch equities + macro + GDELT news
DATA_CARD.md             data sources, coverage and known limitations
src/data/                loaders, indicators, macro_midas, sentiment_gdelt_finbert,
                         windows, splits
src/models/              dramt.py, attention.py, heads.py, baselines/
src/                     train.py, evaluate.py, ablation.py, stats_tests.py,
                         horizon_analysis.py, sweep.py
src/utils/               metrics.py, var_es.py, plotting.py, seed.py, config.py
scripts/                 gpu_chain.py, summarize_results.py, drivers
experiments/             per-run configs
results/                 results.md plus tables_*.csv, tables_*.tex, eval_*.json
results_cpu_era/         superseded CPU-era metrics, kept for provenance
tests/
```

## What is not in this repository

Raw and processed data are excluded by `.gitignore`, along with training checkpoints:

- `data/raw/gdelt/` and `data/processed/finbert_cache/` hold a rate-limited GDELT pull
  and 168,411 FinBERT-scored headlines. They are large and slow to rebuild, so they
  stay local. `download_data.py` and `DATA_CARD.md` document how to regenerate them.
- `runs/` and `runs_cpu_era/` hold model checkpoints and are not tracked. The metrics
  derived from them are, under `results/` and `results_cpu_era/`.

The thesis document itself is kept separately and is not part of this repository.

## License

MIT, see [LICENSE](LICENSE).
