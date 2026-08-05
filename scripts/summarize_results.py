"""Read every results artifact and print a digest organised by research question.

Why this exists: results.md must be written from numbers that were actually
produced, and hand-copying dozens of figures out of seven CSVs is exactly how
a wrong number ends up in a thesis. This reads the artifacts and prints the
quantities each RQ turns on, so the write-up is transcription-free. It invents
nothing: every line is read from a file, and anything missing is reported as
missing rather than filled in.

Usage:
    python -m scripts.summarize_results            # after the chain completes
    python -m scripts.summarize_results --md       # markdown-ready fragments
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("results")


def _load(name: str) -> pd.DataFrame | None:
    p = Path(RES) / name
    if not p.exists():
        print(f"  [missing] {p}")
        return None
    return pd.read_csv(p)


def _fmt(x, d=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def _row(df: pd.DataFrame, run: str):
    m = df[df["run"] == run]
    return m.iloc[0] if len(m) else None


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results",
                    help="read artifacts from here (used to test the digest "
                         "against a synthetic schema without touching results/)")
    RES = Path(ap.parse_args().results_dir)

    section("INPUTS")
    ev = _load("eval_summary.csv")
    ev_g = _load("eval_summary_global.csv")
    ev_r = _load("eval_summary_rolling.csv")
    stats = _load("stats_wilcoxon.csv")
    strata = _load("analysis_news_strata.csv")
    econ = _load("analysis_economic.csv")
    if ev is None:
        raise SystemExit("results/eval_summary.csv missing - run the chain first")
    print(f"  runs evaluated: {len(ev)}")

    # ---------------------------------------------------------------- RQ1
    section("RQ1 - does multimodal fusion improve POINT accuracy?")
    print("Reference point: the martingale (zero-forecast) null.\n")
    cols = ["run", "MAE", "MAE_std", "RMSE", "DirAcc"]
    key = ["baseline_martingale", "baseline_arima", "baseline_har_rv",
           "baseline_tft", "dramt_gpu", "dramt_t", "dramt_t_garch",
           "dramt_ensemble", "dramt_tg_ensemble"]
    print(ev[ev["run"].isin(key)][cols].to_string(index=False))
    null = _row(ev, "baseline_martingale")
    if null is not None:
        print(f"\n  Models BEATING the zero-forecast null (MAE < {_fmt(null['MAE'], 5)}):")
        better = ev[ev["MAE"] < null["MAE"]].sort_values("MAE")
        print("   ", list(better["run"]) if len(better) else "NONE")

    if stats is not None:
        need = {"wilcoxon_p", "dm_p", "dm_p_holm"}
        if not need <= set(stats.columns):
            # a legacy stats file from the pre-Diebold-Mariano protocol
            print(f"\n  [stale] stats_wilcoxon.csv lacks {sorted(need - set(stats.columns))}"
                  f" - it predates the DM/Holm protocol; rerun src.stats_tests")
        else:
            print("\n  Significance vs reference (Diebold-Mariano, Holm-adjusted):")
            sig_raw = int((stats["wilcoxon_p"] < 0.05).sum())
            sig_dm = int((stats["dm_p"] < 0.05).sum())
            sig_holm = int((stats["dm_p_holm"] < 0.05).sum())
            print(f"    {sig_raw}/{len(stats)} raw Wilcoxon | {sig_dm}/{len(stats)} DM | "
                  f"{sig_holm}/{len(stats)} DM+Holm")
            if sig_holm:
                print(stats[stats["dm_p_holm"] < 0.05][
                    ["model_B", "mean_diff_A_minus_B", "dm_p_holm"]].to_string(index=False))

    # ------------------------------------------------------------ ablation
    section("RQ1/RQ3 - ablations (mean +/- SEED std) and permutation controls")
    abl = ev[ev["run"].str.startswith("ablation_")].copy()
    if len(abl):
        abl["config"] = abl["run"].str.replace(r"_s\d+$", "", regex=True)
        g = abl.groupby("config")["MAE"].agg(["mean", "std", "count"])
        full = g.loc["ablation_full", "mean"] if "ablation_full" in g.index else np.nan
        print(f"{'config':34s} {'MAE':>9s} {'seed sd':>9s} {'delta':>9s} {'n':>3s}  clears noise?")
        for cfg, r in g.iterrows():
            delta = r["mean"] - full
            clears = "yes" if abs(delta) > (r["std"] or 0) else "NO"
            if cfg == "ablation_full":
                clears = "-"
            print(f"{cfg:34s} {r['mean']:9.5f} {r['std']:9.5f} {delta:+9.5f} "
                  f"{int(r['count']):3d}  {clears}")
        print("\n  Key contrast - information vs capacity:")
        for a, b in (("ablation_no_sentiment", "ablation_perm_sentiment"),
                     ("ablation_no_macro", "ablation_perm_macro")):
            if a in g.index and b in g.index:
                print(f"    {a.replace('ablation_',''):18s} delta={g.loc[a,'mean']-full:+.5f}"
                      f"   vs {b.replace('ablation_',''):18s} delta={g.loc[b,'mean']-full:+.5f}")
                print("      -> deletion removes capacity too; permutation removes "
                      "ONLY information")

    # ---------------------------------------------------------------- RQ2
    section("RQ2 - risk calibration (10-day portfolio VaR, nominal 5%)")
    rcols = [c for c in ["run", "VolRMSE", "CRPS", "Coverage95", "VaRBreach",
                         "Kupiec_p_min", "ES_Z2", "ES_Z2_p_min", "df_h10"]
             if c in ev.columns]
    risk = ev.dropna(subset=["VaRBreach"]) if "VaRBreach" in ev.columns else ev.iloc[0:0]
    if len(risk):
        print(risk[rcols].sort_values("VaRBreach").to_string(index=False))
    if ev_g is not None and ev_r is not None:
        print("\n  Effect of ROLLING calibration (same predictions, different "
              "post-hoc scaling):")
        gi, ri = ev_g.set_index("run"), ev_r.set_index("run")
        common = [r for r in gi.index if r in ri.index
                  and "VaRBreach" in gi.columns and not pd.isna(gi.loc[r].get("VaRBreach"))]
        print(f"    {'run':26s} {'breach global':>14s} {'breach rolling':>15s} "
              f"{'Kupiec g':>9s} {'Kupiec r':>9s}")
        for r in common[:18]:
            print(f"    {r:26s} {100*gi.loc[r]['VaRBreach']:13.2f}% "
                  f"{100*ri.loc[r]['VaRBreach']:14.2f}% "
                  f"{gi.loc[r].get('Kupiec_p_min', float('nan')):9.4f} "
                  f"{ri.loc[r].get('Kupiec_p_min', float('nan')):9.4f}")

    # --------------------------------------------------------------- econ
    section("RQ2 (economic) - volatility targeting")
    if econ is not None:
        print(econ[["run", "ann_vol", "vol_target_error", "sharpe",
                    "max_drawdown", "mean_leverage"]].head(12).to_string(index=False))

    # ------------------------------------------------------------- strata
    section("RQ1 (sentiment) - accuracy split by news availability")
    if strata is not None:
        s = strata.copy()
        # collapse per-seed ablation runs so the contrast is legible
        s["config"] = s["run"].str.replace(r"_s\d+$", "", regex=True)
        piv = s.groupby(["config", "stratum"])["MAE"].mean().unstack()
        focus = ["ablation_full", "ablation_no_sentiment", "ablation_perm_sentiment",
                 "ablation_no_macro", "ablation_perm_macro", "dramt_ensemble"]
        rows = [f for f in focus if f in piv.index]
        if rows:
            print(piv.loc[rows].to_string())
            base = piv.loc["ablation_full"] if "ablation_full" in piv.index else None
            if base is not None:
                print("\n  cost of removing sentiment, by stratum "
                      "(if sentiment carries signal, the NEWS column must be worse):")
                for cfg in ("ablation_no_sentiment", "ablation_perm_sentiment"):
                    if cfg in piv.index:
                        d_news = piv.loc[cfg, "news"] - base["news"]
                        d_no = piv.loc[cfg, "no_news"] - base["no_news"]
                        print(f"    {cfg.replace('ablation_',''):18s} "
                              f"news {d_news:+.5f}   no_news {d_no:+.5f}")
        else:
            print(piv.head(12).to_string())

    # ---------------------------------------------------------------- RQ3
    section("RQ3 - dynamic weighting variants")
    g3 = ev[ev["run"].str.startswith("dramt_gate")]
    base3 = _row(ev, "dramt_gpu")
    if len(g3):
        print(g3[["run", "MAE", "DirAcc"] +
                 ([c for c in ["VolRMSE", "VaRBreach"] if c in ev.columns])
                 ].to_string(index=False))
    if base3 is not None:
        print(f"\n  reference dramt_gpu MAE={_fmt(base3['MAE'], 5)}")

    # ------------------------------------------------------------ objective
    section("Objective variants (loss rebalancing / ranking)")
    obj = ev[ev["run"].str.startswith(("dramt_lam1", "dramt_risk", "dramt_rank"))]
    if len(obj):
        print(obj[["run", "MAE", "DirAcc"] +
                  ([c for c in ["VolRMSE", "CRPS", "VaRBreach"] if c in ev.columns])
                  ].to_string(index=False))

    # ---------------------------------------------------------------- seeds
    section("Seed robustness, by ensemble family")
    for prefix, ens_name, label in (
            ("dramt_seed", "dramt_ensemble", "Gaussian head"),
            ("dramt_tg_seed", "dramt_tg_ensemble", "Student-t + GARCH-hybrid head")):
        sd = ev[ev["run"].str.match(rf"^{prefix}\d+$")]
        if not len(sd):
            continue
        print(f"  {label} ({len(sd)} seeds)")
        print(f"    MAE {sd['MAE'].min():.5f}-{sd['MAE'].max():.5f} "
              f"(sd {sd['MAE'].std():.5f})   DirAcc "
              f"{sd['DirAcc'].min():.4f}-{sd['DirAcc'].max():.4f}")
        if "VaRBreach" in sd.columns and sd["VaRBreach"].notna().any():
            print(f"    VaR breach {100*sd['VaRBreach'].min():.1f}%-"
                  f"{100*sd['VaRBreach'].max():.1f}%")
        ens = _row(ev, ens_name)
        if ens is not None:
            vb = _v(ens, "VaRBreach")
            print(f"    ENSEMBLE MAE={_fmt(ens['MAE'], 5)} "
                  f"DirAcc={_fmt(ens['DirAcc'], 4)} "
                  f"VaRbreach={'n/a' if _isnan(vb) else f'{100*vb:.2f}%'} "
                  f"ES_Z2={_fmt(_v(ens, 'ES_Z2'), 3)}")

    print("\nDone. Every number above was read from results/; nothing inferred.")


if __name__ == "__main__":
    main()
