# DRAM-T: Dynamic and Risk-Aware Multimodal Transformer

Code for the MSc thesis *Dynamic and Risk-Aware Multimodal Transformer Models for
Financial Forecasting Using External Data Streams*
(Serhat Aslan, s34090, Polish-Japanese Academy of Information Technology).

DRAM-T reads three kinds of data at once and predicts both returns and risk.

| | |
|---|---|
| **Inputs** | prices and technical indicators, macroeconomic series, news sentiment (FinBERT over GDELT headlines) |
| **Outputs** | returns for 1 to 10 trading days, plus volatility, correlation and portfolio VaR |
| **Portfolio** | AAPL, GOOGL, MSFT, AMZN, META |
| **Validation** | 4-fold expanding walk-forward |

## What the results say

| Question | Answer |
|---|---|
| Does it predict returns? | No. No reliable signal at any horizon. |
| Is the risk output calibrated? | Yes. |
| Does it beat GARCH on volatility? | No. It matches it. |
| How close is the correlation head to DCC-GARCH? | Within 1.3%. |
| Do the ablations show a clear effect? | No. Every effect is smaller than the seed spread. |

The negative findings are reported as findings. Full record: `results/results.md`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python check_env.py
pytest tests/
```

## Reproduce

```bash
python download_data.py
python -m src.data.build_features
python -m src.sweep --folds all --shard k --nshards 4
python -m scripts.gpu_chain --parallel 4
python -m scripts.summarize_results
```

Final results came from 88 model runs on an RTX 5070 (CUDA 12.8). Seeds are fixed.
Scalers and all baseline parameters are fitted on training data only. The test suite
includes look-ahead checks.

## Layout

| Path | Contents |
|---|---|
| `config.yaml` | settings, seeds, sweep grid |
| `src/data/` | loading, indicators, MIDAS, sentiment, windows, splits |
| `src/models/` | the model, attention, output heads, baselines |
| `src/` | train, evaluate, ablation, stats_tests, sweep |
| `scripts/` | run chains |
| `results/` | `results.md`, tables, metrics |
| `results_cpu_era/` | superseded metrics from an earlier CPU-only run |
| `tests/` | unit tests |
| `DATA_CARD.md` | data sources and their limits |

## Not in this repository

| Excluded | Reason |
|---|---|
| `data/` | the GDELT pull and 168,411 FinBERT-scored headlines. Large and slow to rebuild. `download_data.py` and `DATA_CARD.md` explain how. |
| `runs/` | model checkpoints. The metrics computed from them are in `results/`. |
| the thesis | kept separately |

`results_cpu_era/` is not comparable to the current numbers. It predates the finished
sentiment cache, which grew from 54,236 to 168,411 scored headlines. `results/results.md`
explains what changed.

## License

MIT, see [LICENSE](LICENSE).
