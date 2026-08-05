# Results Summary — DRAM-T: Dynamic and Risk-Aware Multimodal Transformer

**GPU study, 2026-08-05.** All numbers below come from real runs on an NVIDIA
RTX 5070 (CUDA 12.8). 88 model runs, 4-fold expanding walk-forward validation
(purge gap = T + h_max samples), 5-stock tech portfolio (AAPL, GOOGL, MSFT,
AMZN, META), horizons 1–10 trading days, 460 test anchors shared identically
by every run (verified programmatically). Full tables: `tables_*.csv/.tex`;
significance: `stats_wilcoxon.csv`, `stats_mcs.csv`; analyses: `analysis_*.csv`.

**Selected configuration** (1,920-trial sweep over T × d_model × n_layers ×
λ1 × λ2 × lr, scored on validation across all four folds):
`T=40, d_model=256, n_layers=2, λ1=0.5, λ2=0.1, lr=5e-4`.

> **Not comparable to the CPU-era numbers in the thesis draft.** The FinBERT
> cache grew from 54,236 to 168,411 scored headlines — the earlier machine had
> never finished scoring the retrieved corpus (no GDELT was re-fetched). The
> horizon set also went from 6 to 10. Every model here was re-run on the new
> dataset; the CPU-era artifacts are preserved under `runs_cpu_era/` and
> `results_cpu_era/`.

---

## Headline point accuracy (4-fold mean)

| Model | MAE | RMSE | DirAcc | 10-day VaR breach |
|---|---|---|---|---|
| ARIMA | **0.03280** | **0.04487** | 0.551 | – |
| GARCH(1,1) | 0.03287 | 0.04502 | 0.536 | 4.3% |
| DRAM-T (10-seed ensemble, Gaussian) | 0.03299 | 0.04521 | 0.549 | 10.0% |
| **DRAM-T (10-seed ensemble, Student-t + GARCH-hybrid)** | 0.03301 | 0.04524 | **0.552** | **4.6%** |
| **Martingale (zero forecast)** | **0.03308** | 0.04502 | – | 5.0% |
| DRAM-T (Gaussian, single seed) | 0.03353 | 0.04585 | 0.543 | 9.6% |
| Temporal Fusion Transformer | 0.03576 | 0.04848 | 0.513 | – |

The last two rows of the DRAM-T family are the study's proposed model. The
**Student-t + GARCH-hybrid ensemble is the configuration that carries every
contribution at once**: its point accuracy is indistinguishable from the
Gaussian ensemble (0.00002 MAE apart, against a 0.00095 seed standard
deviation) and its directional accuracy is slightly higher, while its 10-day
VaR breach rate falls from 10.0% to 4.6% and the Expected-Shortfall test goes
from rejected to accepted. It is the model the thesis should present.

The martingale row is the point of the table. A **constant zero forecast**
scores 0.03308; the best model in the study beats it by 0.00028, or 0.8%.
DRAM-T's ensemble beats it by 0.00009 (0.3%). At daily-to-two-week horizons
the conditional mean of these returns is, for practical purposes, not
forecastable, and the ranking above is a ranking of how gracefully each model
declines to predict.

The TFT result is worth stating plainly: the strongest published
general-purpose forecasting transformer is the **worst** model here.

### Significance

Reference model = the Student-t + GARCH-hybrid ensemble, against all 87 others:

| Test | Significant at 5% |
|---|---|
| Raw Wilcoxon (as in the earlier draft) | **41 / 87** |
| Diebold–Mariano, Newey–West HAC (lag 9) | 22 / 87 |
| Diebold–Mariano + Holm | **1 / 87** |

These are overlapping multi-horizon forecasts — an anchor's 10-day return
shares 9 of its 10 days with the next anchor's — so the loss differentials are
strongly autocorrelated and Wilcoxon's independence assumption is violated.
Correcting for that, and for testing one model against 87 others, **essentially
no pair of models in this study is statistically distinguishable on point
accuracy.**

The single surviving comparison is the ensemble against one of its own member
seeds (`dramt_tg_seed44`, p = 0.0015). That is a demonstration that averaging
over seeds helps, not an independent finding: the ensemble contains the seed
it is being compared against. **No baseline and no ablation is separable from
the proposed model.**

### Model Confidence Set

Pairwise tests can only say that no individual comparison survives correction.
The Model Confidence Set (Hansen, Lunde & Nason 2011) states the same result
positively: it returns the set of models that contains the best one with
probability 1−α. Computed on per-anchor absolute errors with a circular
**block** bootstrap (block length 10 = the maximum horizon, because
overlapping forecasts make an i.i.d. bootstrap overstate precision):

| Confidence level | Models retained (of 88) | First exclusions |
|---|---|---|
| 90% (α = 0.10) | **88 / 88** | — |
| 75% (α = 0.25) | 87 / 88 | one individual seed |
| 50% (α = 0.50) | 86 / 88 | + one ablation seed |
| 25% (α = 0.75) | 80 / 88 | + TFT, more seeds |

**At the 90% level the confidence set contains every model in the study** —
the proposed architecture, all thirteen baselines, all forty ablation runs,
and the constant-zero martingale. Not until the level is relaxed to 25%, where
the set would exclude the true best model three times in four, does a genuine
baseline (TFT) drop out.

This is the cleanest statement of the study's central negative result: **on
point accuracy the data cannot separate a multimodal transformer from
predicting nothing at all.** It is a property of daily equity returns at these
horizons, not a defect of the implementation, and it is consistent with the
linear-model findings of Zeng et al. (2023) on general time-series
benchmarks.

---

## RQ1 — does fusing macro + sentiment improve point accuracy?

**No, and the earlier positive result does not survive.**

Ablations, 5 seeds each, mean ± seed standard deviation:

| Configuration | MAE | seed sd | Δ vs full | clears noise? |
|---|---|---|---|---|
| full | 0.03385 | 0.00076 | — | — |
| − sentiment | 0.03384 | 0.00087 | −0.00001 | no |
| sentiment **permuted** | 0.03406 | 0.00046 | +0.00020 | no |
| − macro | 0.03367 | 0.00021 | −0.00018 | no |
| macro **permuted** | 0.03407 | 0.00203 | +0.00022 | no |
| − dynamic weighting | 0.03360 | 0.00065 | −0.00025 | no |
| − risk head | 0.03395 | 0.00054 | +0.00009 | no |
| numerical only | 0.03398 | 0.00160 | +0.00012 | no |

**Every effect is smaller than its own seed spread.** The CPU-era table
reported "removing sentiment significantly degrades accuracy (p < 1e-5)" from
a single seed; at full capacity with five seeds the effect is −0.00001 MAE,
i.e. nothing.

The **permutation** rows are the methodological point. Deleting a modality
also deletes its encoder and ~272k parameters, so a deletion ablation cannot
separate "the information mattered" from "the model got smaller". The
permutation ablation keeps the architecture byte-identical and destroys only
the modality's correspondence with the target. Both are within noise here, so
neither information nor capacity has a demonstrable effect.

### Where sentiment actually exists

News coverage is thin: the share of trading days carrying any news is NVDA
71%, AAPL 55%, MSFT 33%, AMZN 14%, META 12%, GOOGL 9%. Splitting the same
predictions by whether the anchor's own input window contained news:

| Removing sentiment costs | on **news** days | on **no-news** days |
|---|---|---|
| by deletion | −0.00003 | +0.00161 |
| by permutation | +0.00020 | +0.00126 |

If sentiment carried signal, the news column would have to degrade more. It is
the reverse, and this reproduces the same pattern found on the CPU-era data.
Sentiment provides no measurable benefit **even on the days when news exists**.

---

## RQ2 — does the risk-aware head yield better-calibrated uncertainty?

**This is the study's positive result.** 10-day portfolio VaR at 95%,
breach rate per fold (nominal 5%):

| Model | fold 0 | fold 1 | fold 2 | fold 3 | mean |
|---|---|---|---|---|---|
| DRAM-T ensemble, **Gaussian** | 0.9% | 13.0% | 16.5% | 9.6% | 10.0% |
| DRAM-T, Gaussian (single seed) | 0.0% | 12.2% | 18.3% | 7.8% | 9.6% |
| **DRAM-T ensemble, Student-t + GARCH-hybrid** | 0.0% | 7.0% | 8.7% | 2.6% | **4.6%** |
| DRAM-T, Student-t + GARCH-hybrid (single seed) | 0.0% | 7.0% | 5.2% | 6.1% | 4.6% |
| DRAM-T, Student-t | 0.0% | 7.8% | 5.2% | 0.9% | 3.5% |
| GARCH(1,1) | 0.0% | 4.3% | 7.8% | 5.2% | 4.3% |
| HAR-RV | 0.0% | 3.5% | 5.2% | 6.1% | 3.7% |
| Martingale + historical vol | 0.0% | 3.5% | 7.8% | 8.7% | 5.0% |

The Gaussian head reproduces the draft's failure (9.6% against a nominal 5%,
with 18.3% in fold 2). **Switching the likelihood to Student-t with a
GARCH-hybrid volatility head brings DRAM-T to 4.6% — indistinguishable from
GARCH's own 4.3%.** The deep model's tail risk is now as well calibrated as
the dedicated econometric model's, which was the main open weakness.

Expected Shortfall (Acerbi–Székely Test 2, simulated null):

| Model | Z₂ | p | verdict |
|---|---|---|---|
| DRAM-T ensemble, Gaussian | −1.09 | 0.000 | **rejected** — tail losses worse than predicted |
| DRAM-T, Gaussian (single seed) | −1.02 | 0.000 | **rejected** |
| **DRAM-T ensemble, Student-t + GARCH** | **+0.33** | **0.20** | not rejected |
| DRAM-T, Student-t + GARCH (single seed) | +0.32 | 0.45 | not rejected |
| DRAM-T, Student-t | +0.44 | 0.30 | not rejected |
| GARCH(1,1) | +0.14 | 0.06 | not rejected |

**Which of the three changes did the work.** Rolling calibration alone moved
the Gaussian model from 11.3% to 10.0% — real but small. The Student-t
likelihood is what closed the gap. The heavy tail is doing the work, not the
recalibration.

**Overcorrection, stated honestly.** The most aggressive variants overshoot:
`riskonly` breaches at 1.3% and `rank_only` at 1.1%, with 95% coverage at
0.987 against a nominal 0.95. These are now too conservative. The
best-calibrated configuration is Student-t + GARCH-hybrid, not the most
heavily risk-weighted one.

**A caveat on Kupiec.** `Kupiec_p_min` is near zero for *every* model
including the martingale null, because fold 0 (2023-03 to 2023-09) was a
strong low-volatility rally — mean 10-day portfolio return +1.83%, worst loss
−6.07%, only 25% of windows negative — so no model breaches at all there, and
zero breaches in 115 observations is itself improbable under a 5% rate. The
per-fold table above is the honest presentation; the pooled minimum is not.

### The correlation head — a second output that works

The model's third output is a 5×5 correlation matrix, trained through the
Frobenius term of the composite loss. Scored on the strict upper triangle (the
diagonal is 1 by construction), against a null of a constant correlation
matrix estimated on training data only:

| Model | Corr RMSE | vs static null |
|---|---|---|
| DCC-GARCH | **0.3431** | −9.3% |
| DRAM-T, Student-t + GARCH-hybrid | 0.3471 | −8.2% |
| **DRAM-T ensemble, Student-t + GARCH-hybrid** | 0.3475 | −8.1% |
| DRAM-T, Student-t | 0.3508 | −7.2% |
| *static train correlation (null)* | *0.3782* | — |

**62 of the 76 runs carrying a learned correlation head beat the static
null**, by 6–9%. DRAM-T lands within 1.3% of DCC-GARCH, a model included in
the study specifically because forecasting a correlation matrix is what it is
for.

#### Correlation forecasts *are* statistically separable — unlike point forecasts

Running the identical machinery (Diebold–Mariano with HAC variance, Holm
correction, and a block-bootstrap Model Confidence Set) on per-anchor
correlation errors gives a different answer from the point-forecast case:

| | point forecasts | correlation forecasts |
|---|---|---|
| DM + Holm significant | 1 / 87 | **10 / 75** |
| MCS retained (α = 0.10) | 88 / 88 | **69 / 76** |

The seven models the correlation MCS eliminates are **exactly the ones whose
correlation head was not trained**: all five `point_only` seeds (risk heads
detached from the loss, so the Frobenius term is absent) plus two marginal
ablation seeds. Every one of them is *worse than the static null*:

| | mean Corr RMSE |
|---|---|
| `point_only` (risk head removed) | **0.5981** |
| static train correlation (null) | 0.3782 |
| all models with a trained risk head | **0.3633** |

**This is the only statistically significant ablation effect in the entire
study** (p ≈ 1e-11 against the reference). It is a direct, quantitative
demonstration that the composite loss's risk terms do real work: with them the
correlation head beats a constant matrix, without them it is markedly worse
than one. The result the CPU-era draft asserted qualitatively for the
risk-aware head now has significance behind it — on correlation, not on the
mean.

This is the second output that demonstrably works. Against the martingale
result for the mean, the picture across the three heads is: the **mean** is
not forecastable, the **volatility** is (and is now as well calibrated as
GARCH), and the **correlation** is, beating its own naive baseline and
matching a dedicated multivariate GARCH.

### Epistemic vs aleatoric uncertainty (MC-dropout)

50 stochastic forward passes per prediction, decomposing the predictive
variance into model uncertainty and market uncertainty:

| Model | epistemic σ | aleatoric σ | epistemic share of variance |
|---|---|---|---|
| DRAM-T, Gaussian | 0.00248 | 0.03822 | 0.61% |
| DRAM-T, Student-t | 0.00154 | 0.02665 | 0.38% |
| DRAM-T, Student-t + GARCH-hybrid | 0.00175 | 0.03110 | 0.35% |

**Model uncertainty is under 1% of total predictive variance.** Folding it
into the predictive distribution moves the 10-day VaR breach rate by at most
0.2 percentage points (4.57% → 4.35%). The uncertainty in these forecasts is
almost entirely irreducible market noise, not uncertainty about the model —
which is consistent with everything else here, and means MC-dropout is not a
route to better calibration in this setting.

### Economic value of the volatility forecast

Exposure scaled to a 10% annualised volatility target from each model's own
1-day sigma, applied to the next day's return:

| Model | realized vol | Sharpe | max drawdown | mean leverage |
|---|---|---|---|---|
| HAR-RV | 0.098 | **0.77** | −0.16 | 0.44 |
| DRAM-T, Student-t + GARCH | 0.097 | 0.71 | −0.17 | 0.43 |
| DRAM-T, Gaussian + GARCH | 0.095 | 0.70 | −0.17 | 0.42 |
| DRAM-T, Student-t | 0.098 | 0.63 | −0.16 | 0.41 |
| DRAM-T, λ1=2 | **0.100** | 0.19 | −0.19 | 0.38 |

Every model lands close to the target, so the differences are in risk-adjusted
return rather than in volatility control. HAR-RV — a three-term linear
regression — gives the best Sharpe. Sharpe differences over 460 anchors are
not tested for significance and should not be read as an investment claim.

---

## RQ3 — does dynamic weighting beat static fusion?

**Still not supported.**

| Variant | MAE | DirAcc |
|---|---|---|
| per-timestep gate | 0.03284 | 0.558 |
| extended + per-timestep gate | 0.03309 | 0.551 |
| extended gate signals | 0.03375 | 0.541 |
| reference (per-window gate) | 0.03353 | 0.543 |
| **static fusion ablation** | **0.03360** | – |

The per-timestep gate is nominally the best DRAM-T variant on both MAE and
DirAcc, but the margin over the reference (0.00069) is below the 10-seed
standard deviation (0.00095), and the static-fusion ablation remains
nominally *better* than the full dynamic model (−0.00025). Enriching the gate
with VIX level and slope, trailing average correlation and drawdown did not
change this. Gating remains interpretable but not demonstrably useful.

---

## Fold sensitivity — does any conclusion rest on one test block?

Fold 0 (2023-03 – 2023-09) is anomalous: a low-volatility rally in which the
10-day portfolio return averaged +1.83%, the worst loss was −6.07%, and only
25% of windows were negative. No model breaches its VaR there, so every model
including the martingale null fails Kupiec on that fold, and any
minimum-over-folds statistic is dominated by it. Each headline metric was
therefore recomputed with each fold left out in turn
(`analysis_fold_sensitivity.csv`).

**The negative point-accuracy result gets stronger, not weaker:**

| Folds used | Models beating the martingale null (of 88) |
|---|---|
| all four | 20 / 88 |
| **excluding fold 0** | **7 / 88** |
| excluding fold 1 | 28 / 88 |
| excluding fold 2 | 26 / 88 |
| excluding fold 3 | 29 / 88 |

Whatever edge the models hold over a constant zero forecast comes
disproportionately from the calm fold. Across the three more volatile blocks,
**only 7 of 88 models beat predicting nothing.**

(HAR-RV and the martingale have identical MAE in every column, as they must:
HAR-RV models variance and leaves the mean at zero. It is a useful check that
the pipeline is doing what it claims.)

**The RQ2 result is robust to fold choice:**

| Model | drop f0 | drop f1 | drop f2 | drop f3 | all |
|---|---|---|---|---|---|
| DRAM-T ensemble, Gaussian | 13.0% | 9.0% | 7.8% | 10.1% | 10.0% |
| **DRAM-T ensemble, Student-t + GARCH** | 6.1% | 3.8% | 3.2% | 5.2% | **4.6%** |
| GARCH(1,1) | 5.8% | 4.4% | 3.2% | 4.1% | 4.3% |

The Student-t + GARCH ensemble sits near the nominal 5% in every variant and
the Gaussian one fails in every variant. Neither conclusion depends on which
fold is included.

---

## RQ4 — robustness across regimes

Fold difficulty dominates every model choice. Realized 10-day portfolio
return volatility rises monotonically across folds (0.030 → 0.040 → 0.054 →
0.054) and mean return falls (+1.83% → +0.96% → +0.17% → −0.35%). In the
sweep, the spread of validation scores between folds (2.43–4.35) was roughly
ten times any hyperparameter effect.

Seed robustness, 10 seeds per family:

| Family | MAE range | seed sd | ensemble MAE | ensemble VaR breach |
|---|---|---|---|---|
| Gaussian head | 0.03293–0.03566 | 0.00095 | 0.03299 | 10.0% |
| Student-t + GARCH-hybrid head | see `tables_seeds.csv` | – | 0.03301 | **4.6%** |

In both families the ensemble lands near the best individual seed, so
averaging recovers most of the seed penalty. Crucially the two ensembles are
**indistinguishable on point accuracy** (0.00002 MAE apart, two orders of
magnitude below the seed standard deviation) while differing by a factor of
two in VaR breach rate: the risk-calibration changes are free in accuracy
terms.

---

## What the GPU changed, and what it did not

The hardware upgrade allowed the full specified grid: 1,920 sweep trials
against 32 previously, d_model up to 256, n_layers to 4, T to 40, 100-epoch
budgets, 10 seeds, and a TFT baseline. It did not improve point accuracy,
and the sweep shows why:

- **Depth has no effect.** Mean validation scores for 2 / 3 / 4 layers differ
  in the third decimal, and the best layer count differs by fold.
- **Width and window length help only at the ceiling.** d_model 256 gave the
  best score in all four folds, but by 0.6–1.2%; the *average* configuration
  was no better than at 128.
- **Training stops almost immediately.** Median best epoch across 1,920
  trials was 3, maximum 67 — the 100-epoch budget is never approached.
- **Fold-to-fold variation dwarfs all of it.**

The bottleneck is signal, not compute. Realized volatility has lag-1
autocorrelation 0.92 and is clearly learnable; the conditional mean of returns
at these horizons is not.

---

## Data limitations (full detail in DATA_CARD.md)

- 168,411 unique headlines FinBERT-scored across 2017-02-15 – 2026-06-30.
- GDELT DOC 2.0 rejects pre-2017 queries; sentiment for 2016 is unavailable.
- Rate limiting forced monthly query resolution outside the
  2022-12 – 2023-03 case window, where resolution is daily (coverage
  74.7–98.8% of trading days there, versus 9–71% over the full span).
- MAPE on returns is unstable near zero and is reported but not relied upon.

## Reproduce

```
python -m src.data.build_features                    # FinBERT cached
python -m src.sweep --folds all --shard k --nshards 4
python -m scripts.gpu_chain --parallel 4             # all 77 runs + analyses
python -m scripts.summarize_results
```
Deterministic seeds; standardizers, sigma calibration and GARCH/HAR
parameters fitted on train/validation only. 107 unit tests (`pytest tests/`),
including look-ahead tests for the rolling calibration, the extended regime
signals and the permutation ablation.
