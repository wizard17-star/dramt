# Results Summary — DRAM-T: Dynamic and Risk-Aware Multimodal Transformer

All numbers are from real runs: 4-fold expanding walk-forward validation
(purge gap = T + h_max samples), 5-stock tech portfolio (AAPL, GOOGL, MSFT,
AMZN, META), horizons {1,2,3,5,8,10} trading days, test span 2023-03 to
2026-06 across folds. Point metrics: mean over all stocks x horizons x test
anchors, mean ± std across folds. Full tables: `tables_*.csv/.tex`;
significance: `stats_wilcoxon.csv` (paired two-sided Wilcoxon on per-sample
absolute errors), `stats_bootstrap.csv` (1000-resample CIs).

**Compute caveat (applies to every deep result):** no CUDA GPU was available;
model width/depth, window sweep, and epoch budgets were reduced accordingly
(config.yaml `notes.compute_assumption`). TFT was excluded for the same
reason. Results should be read as "DRAM-T under a small-model CPU budget",
not the architecture's ceiling.

## Headline comparison (4-fold mean)

| Model | MAE | RMSE | DirAcc | VolRMSE | CRPS | Cov95 | VaR breach (nom 5%) |
|---|---|---|---|---|---|---|---|
| ARIMA | **0.0301** | **0.0421** | **0.545** | – | – | – | – |
| GARCH-MIDAS | 0.0301 | 0.0423 | 0.529 | 0.0131 | 0.0222 | 0.941 | 4.7% |
| GARCH(1,1) | 0.0302 | 0.0424 | 0.529 | **0.0125** | **0.0221** | 0.940 | **4.3%** |
| DCC-GARCH | 0.0302 | 0.0424 | 0.529 | 0.0125 | 0.0221 | **0.940** | 5.4% |
| Sentiment-LSTM | 0.0306 | 0.0429 | 0.536 | – | – | – | – |
| LSTM / GRU / CNN-BiLSTM(±attn) / Transformer | 0.0305–0.0313 | 0.0428–0.0437 | 0.498–0.516 | – | – | – | – |
| **DRAM-T (seed 42)** | 0.0307 | 0.0428 | 0.511 | 0.0159 | 0.0231 | 0.929 | 11.9% |
| DRAM-T (seed 44, best) | 0.0303 | 0.0421 | 0.545 | 0.0159 | 0.0228 | 0.927 | 11.9% |

Seed robustness (seeds 42/43/44): MAE 0.0303–0.0308, DirAcc 0.511–0.545 —
non-trivial seed variance; the seed-42 vs seed-44 point difference is NOT
significant (Wilcoxon p=0.19).

## RQ1 — Does fusing macro + sentiment improve point accuracy?

**Within the architecture: yes for sentiment, inconclusive for macro.**
Removing the sentiment modality significantly degrades point accuracy
(ablation `-sentiment`: MAE 0.0315 vs full 0.0307, Wilcoxon p < 1e-5);
`numerical-only` is also significantly worse than full (p = 0.020).
Removing macro is directionally worse but not significant (p = 0.117) —
consistent with the macro modality's information already reaching the model
through slower channels (rates ffilled daily; monthly series move rarely).

**Against external baselines: partial.** DRAM-T significantly beats GRU
(p = 0.003) and is at par with the remaining deep baselines, but ARIMA
remains significantly better on pooled point error (p < 0.001) and the GARCH
family marginally so (p ≈ 0.01–0.03). Honest reading: at daily-to-two-week
horizons, mean returns are barely predictable and simple mean models are
extremely strong; the fusion advantage shows up *within* the deep model
class, not over econometric mean baselines.

## RQ2 — Does the risk-aware head yield better-calibrated uncertainty?

**Versus a point-only deep model: clearly yes.** The `point-only` ablation
(risk heads detached; intervals from the same post-hoc val calibration)
gives 95% coverage 0.891, CRPS 0.0240, and a 23.7% 10-day VaR breach rate;
the full risk-aware model reaches coverage 0.929, CRPS 0.0231, breach 11.9%.
The composite NLL+Frobenius loss demonstrably teaches usable uncertainty.

**Versus dedicated econometric risk models: no.** Once both families are
evaluated in consistent units, GARCH(1,1)/DCC deliver near-nominal 10-day VaR
(4.3–5.4% breaches), higher coverage (0.94) and lower CRPS (0.0221) and
VolRMSE (0.0125). DRAM-T's 10-day VaR under-estimates risk (11.9–13.2%
breaches; Kupiec rejects), driven by folds 2–3 where per-fold calibration
drifted (fold-0 breach 0%, fold-2 20.7% despite val-fitted scaling) — the
learned volatility does not yet track regime shifts across a fold the way a
recursive GARCH filter does. All models, including GARCH, fail the
Christoffersen independence test in at least one fold (breach clustering).

## RQ3 — Does dynamic weighting beat static fusion?

**Not in this configuration.** The `static-fusion` ablation is statistically
indistinguishable from the full model on point error (p = 0.26) and even
nominally better (MAE 0.0303 vs 0.0307, best VolRMSE 0.0136 among DRAM-T
variants). The gating network produces interpretable, regime-varying weights
(figure `modality_weights`: numerical weight rises in the volatile early-2025
period), but interpretability is currently its main contribution — the
small-model CPU budget and limited sentiment history are plausible reasons
the input-dependent gate could not exploit its extra capacity.

## RQ4 — Robustness across regimes?

Fold 3 (test ≈ 2025-09 – 2026-06) degrades EVERY model's point accuracy by
~15–17% MAE relative to earlier folds (e.g. ARIMA MAE 0.024→0.035, DirAcc
0.62→0.47; DRAM-T similar). DRAM-T's fold-to-fold std (±0.0058 RMSE) matches
the baselines — no extra fragility, but also no special robustness. Its VaR
calibration, however, is clearly regime-sensitive (RQ2), whereas GARCH
adapts recursively. Conclusion: the framework is as robust as the baselines
on point accuracy, and less robust on tail-risk calibration.

## Data limitations (full detail in DATA_CARD.md)

- GDELT DOC 2.0 rejects pre-2017 queries → no sentiment for 2016 windows.
- Aggressive API rate-limiting forced 3-day query chunks (daily resolution
  retained via per-article timestamps) inside the Dec-2022–Mar-2023 case
  window and monthly resolution elsewhere; 54,236 headlines FinBERT-scored.
- Case-window daily news coverage: 92–99% of trading days per portfolio
  ticker, except GOOGL 72% (250-record cap truncates the very-high-volume
  "Google" keyword). NVDA (context feature only) partially covered.
- MAPE on returns is reported but unstable near zero returns (entries with
  |y| < 1e-4 excluded); prefer MAE/RMSE/DirAcc.

## Reproduce

```
python download_data.py                 # equities + macro + GDELT (slow, cached, resumable)
python -m src.data.build_features       # FinBERT scoring + windowed tensors
python -m src.sweep                     # hyperparameter selection (fold 0)
python -m src.train --config experiments/final.yaml
python -m src.baselines --all --suffix ""
python -m src.ablation --suffix ""
python -m src.evaluate && python -m src.stats_tests --reference dramt_final
python -m src.tables && python -m src.utils.plotting --run dramt_final
```
Fixed seed 42 (variants 43/44), deterministic flags on, standardizers and
calibration fitted on train/validation only. 40 unit tests (`pytest tests/`).
