# DRAM-T: Dynamic and Risk-Aware Multimodal Transformer

Reference implementation for the MSc thesis *Dynamic and Risk-Aware Multimodal
Transformer Models for Financial Forecasting Using External Data Streams*
(Serhat Aslan, s34090, Polish-Japanese Academy of Information Technology).

DRAM-T is a multimodal Transformer that fuses market, macroeconomic and textual
information into a single joint predictive distribution. Rather than emitting a point
forecast alone, it produces the conditional mean, the conditional volatility and the
cross-asset correlation structure together, so that portfolio Value-at-Risk follows
from the model's own output instead of a downstream assumption. The study evaluates
that design under a walk-forward protocol built to survive the two failure modes that
make most published gains in this area unreproducible: look-ahead leakage and
multiplicity in significance testing.

---

## Architecture

| Component | Design |
|---|---|
| **Modality encoders** | Separate encoders for price and technical indicators, mixed-frequency macroeconomic series (MIDAS-weighted) and FinBERT sentiment derived from GDELT headlines |
| **Fusion** | Inter-modal cross-attention with a regime-conditioned softmax gate over modalities, so modality weighting varies with market state rather than being fixed. A per-timestep variant of the gate is evaluated as a separate configuration |
| **Backbone** | Transformer encoder over the fused sequence, followed by a horizon-query decoder for 1 to 10 trading days |
| **Mean head** | Student-t likelihood with a learnable per-horizon degrees-of-freedom parameter |
| **Volatility head** | GARCH-hybrid, combining a learned component with a parametric conditional-variance path |
| **Correlation head** | Cholesky-parameterized, guaranteeing a positive-definite 5×5 correlation matrix |
| **Uncertainty** | MC-dropout, decomposed into aleatoric and epistemic parts by the law of total variance |

Portfolio: AAPL, GOOGL, MSFT, AMZN and META, with NVDA and the S&P 500 as reference
series.

## Evaluation protocol

The evaluation is the substance of the study, so it is specified before any run and
applied identically to every model.

| Element | Choice |
|---|---|
| **Cross-validation** | 4-fold expanding walk-forward, purge gap of `T + h_max` samples between train and test |
| **Test set** | 460 anchors, shared identically by all 88 runs and verified programmatically |
| **Model selection** | 1,920-trial sweep over `T × d_model × n_layers × λ1 × λ2 × lr`, scored on validation folds only |
| **Point significance** | Diebold–Mariano with Newey–West HAC standard errors, Holm and Benjamini–Hochberg multiplicity control |
| **Joint significance** | Model Confidence Set (Hansen, Lunde & Nason) with a circular block bootstrap, block length matched to the maximum horizon |
| **Risk backtesting** | Kupiec POF, Christoffersen independence, Acerbi–Székely ES Test 2 |
| **Ablations** | Permutation ablations that destroy information while holding architecture and parameter count fixed |
| **Leakage control** | Scalers, sigma calibration and all baseline parameters fitted on training data only, enforced by dedicated look-ahead unit tests |

Selected configuration: `T=40, d_model=256, n_layers=2, λ1=0.5, λ2=0.1, lr=5e-4`.
Final results come from 88 model runs on an NVIDIA RTX 5070 (CUDA 12.8).

## Findings

**The conditional mean is not predictable at these horizons. The study demonstrates
this rather than assuming it.** The result is reported as a finding
because the testing methodology is what establishes it:

| Test applied to the reference model against 87 others | Significant at 5% |
|---|---|
| Wilcoxon, assuming independent loss differentials | 41 / 87 |
| Diebold–Mariano with Newey–West HAC (lag 9) | 22 / 87 |
| Diebold–Mariano with Holm correction | 1 / 87 |

Multi-horizon forecasts overlap, so loss differentials are strongly autocorrelated and
Wilcoxon's independence assumption fails. Once that autocorrelation and the
multiplicity of testing one model against 87 others are both handled, no baseline and
no ablation is separable from the proposed model on point accuracy. The single
surviving comparison is the seed ensemble against one of its own members, which
demonstrates that seed averaging helps rather than constituting an independent result.

The Model Confidence Set states the same conclusion positively: at the 90% level it
retains all 88 models, meaning the data cannot identify a best one.

**The risk output is where the model delivers.** Predictive distributions are well
calibrated, the volatility head reaches parity with GARCH(1,1) instead of surpassing
it and the correlation head lands within 1.3% of DCC-GARCH. Every ablation effect is
smaller than the spread across random seeds, which is itself reported rather than
absorbed into a favourable comparison.

Full record with all tables: [`results/results.md`](results/results.md).

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python check_env.py
pytest tests/
```

```bash
python download_data.py
python -m src.data.build_features
python -m src.sweep --folds all --shard k --nshards 4
python -m scripts.gpu_chain --parallel 4
python -m scripts.summarize_results
```

Seeds are fixed throughout. The suite carries over 100 unit tests, including explicit
look-ahead tests for the rolling calibration, the regime signals and the permutation
ablation.

## Repository layout

| Path | Contents |
|---|---|
| `src/models/` | `dramt.py`, attention, output heads, baseline models |
| `src/data/` | loaders, indicators, MIDAS macro alignment, GDELT/FinBERT sentiment, windowing, splits |
| `src/` | `train.py`, `evaluate.py`, `ablation.py`, `stats_tests.py`, `sweep.py`, `horizon_analysis.py` |
| `src/utils/` | metrics, VaR and ES, plotting, seeding, config |
| `scripts/` | end-to-end run chains and result summarization |
| `experiments/` | per-run configurations |
| `results/` | `results.md`, result tables and per-run metrics |
| `results_cpu_era/` | superseded metrics, retained for provenance |
| `tests/` | unit tests |
| `config.yaml` | hyperparameters, seeds, sweep grid, paths |
| `DATA_CARD.md` | data sources, coverage and known limitations |

## Data availability

Raw and processed data are excluded from version control. The GDELT corpus is a
rate-limited pull scored into a cache of 168,411 FinBERT-labelled headlines, which is
slow to rebuild but fully reproducible through `download_data.py`. `DATA_CARD.md`
documents every source together with its coverage limits, including the reduced
sentiment coverage over the evaluation window, which the thesis treats as a stated
limitation rather than a footnote.

Model checkpoints under `runs/` are likewise excluded. Every metric derived from them
is included under `results/`.

## License

Released under the MIT License, see [LICENSE](LICENSE).
