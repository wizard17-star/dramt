# Exploratory runs — deliberately outside the main run set

Runs here are NOT part of the reported 88-run study and must not be mixed into
`runs/`. They are kept because `results/results.md` cites their numbers.

## dramt_risksel

Trained to test whether the sweep's selection criterion mattered. The sweep
score is 101.6%-driven by the point-forecast component, which this study finds
unlearnable, so the recorded trials were re-scored with a risk-weighted,
fold-standardised criterion (`src/sweep.py --select-by risk`). That selects
T=20, d_model=256, n_layers=4, lambda1=0.5, lambda2=0.5, lr=5e-5 — notably a
different lambda2 and a different T.

Result: it lands inside the 10-seed range of the selected configuration on
MAE, VolRMSE, CorrRMSE and VaR breach alike. Reported in results.md as a null
result.

**Why it is not in `runs/`:** it uses T=20 while the main study uses T=40, so
it has 116 test anchors per fold instead of 115. Keeping it alongside the main
runs makes `scripts/gpu_chain.py --only 8` report an anchor-set mismatch and
would cause `src/stats_tests.py` to silently skip it in every paired test.
Its metrics are preserved in `results/eval_summary_cmp.csv`.
