"""Generate thesis tables (CSV + LaTeX) from results/eval_summary*.csv.

Outputs to results/:
  tables_point.csv/.tex        - point accuracy per model (mean +/- std)
  tables_risk.csv/.tex         - risk metrics incl. Kupiec / Christoffersen /
                                 Acerbi-Szekely ES backtest
  tables_ablation.csv/.tex     - the 6 ablation configs
  tables_comparison.csv/.tex   - proposed vs baselines headline table
  tables_calibration.csv/.tex  - RQ2: global vs rolling sigma calibration
  tables_objective.csv/.tex    - loss-rebalancing / ranking variants
  tables_seeds.csv/.tex        - seed robustness + ensemble

Usage: python -m src.tables
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_LABELS = {
    # --- nulls and econometric baselines ---
    "baseline_martingale": "Martingale (zero forecast)",
    "baseline_arima": "ARIMA",
    "baseline_garch": "GARCH(1,1)",
    "baseline_garch_midas": "GARCH-MIDAS",
    "baseline_dcc_garch": "DCC-GARCH",
    "baseline_har_rv": "HAR-RV",
    # --- deep baselines ---
    "baseline_lstm": "LSTM",
    "baseline_gru": "GRU",
    "baseline_cnn_bilstm": "CNN-BiLSTM",
    "baseline_cnn_bilstm_attn": "CNN-BiLSTM-Attn",
    "baseline_transformer": "Transformer (OHLCV)",
    "baseline_sentiment_lstm": "Sentiment-LSTM",
    "baseline_tft": "Temporal Fusion Transformer",
    # --- DRAM-T risk variants ---
    "dramt_gpu": "DRAM-T (Gaussian)",
    "dramt_t": "DRAM-T (Student-$t$)",
    "dramt_garch": "DRAM-T (Gaussian, GARCH-hybrid vol)",
    "dramt_t_garch": "DRAM-T (Student-$t$, GARCH-hybrid vol)",
    "dramt_t_har": "DRAM-T (Student-$t$, HAR-hybrid vol)",
    # --- objective variants ---
    "dramt_lam1_2": r"DRAM-T ($\lambda_1=2$)",
    "dramt_lam1_5": r"DRAM-T ($\lambda_1=5$)",
    "dramt_riskonly": "DRAM-T (risk-only loss)",
    "dramt_rank": "DRAM-T (+ ranking loss)",
    "dramt_rank_only": "DRAM-T (ranking replaces mean)",
    # --- RQ3 gating variants ---
    "dramt_gate_ext": "DRAM-T (extended gate signals)",
    "dramt_gate_time": "DRAM-T (per-timestep gate)",
    "dramt_gate_ext_time": "DRAM-T (extended + per-timestep gate)",
    # --- ensembles ---
    "dramt_ensemble": "DRAM-T (10-seed ensemble, Gaussian)",
    "dramt_tg_ensemble": "DRAM-T (10-seed ensemble, Student-$t$ + GARCH-hybrid)",
    # --- ablations ---
    "ablation_full": "Full model",
    "ablation_no_sentiment": "-- sentiment",
    "ablation_no_macro": "-- macro",
    "ablation_static_fusion": "-- dynamic weighting",
    "ablation_point_only": "-- risk-aware head",
    "ablation_numerical_only": "numerical only",
    "ablation_perm_sentiment": "sentiment permuted (same architecture)",
    "ablation_perm_macro": "macro permuted (same architecture)",
    # --- legacy CPU-era names, kept so archived summaries still label ---
    "dramt_final": "DRAM-T (CPU-era)",
    "dramt_full": "DRAM-T (CPU-era, no sentiment)",
}

# Individual ensemble-member runs, of any ensemble family. They are excluded
# from the headline tables (10-20 near-identical rows would swamp the
# comparison) and summarised in the seed-robustness table instead.
SEED_RE = re.compile(r"^dramt_(?:tg_)?seed\d+$")
OBJECTIVE_RUNS = ["dramt_lam1_2", "dramt_lam1_5", "dramt_riskonly",
                  "dramt_rank", "dramt_rank_only"]
GATE_RUNS = ["dramt_gate_ext", "dramt_gate_time", "dramt_gate_ext_time"]


def _v(row, col, default=np.nan):
    """Column value that tolerates the column being absent entirely (a chain
    step may not have produced it)."""
    try:
        val = row[col]
    except (KeyError, IndexError):
        return default
    return default if val is None else val


def _isnan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return x is None


def _fmt(mean, std=None, digits: int = 4) -> str:
    if _isnan(mean):
        return "--"
    if std is None or _isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _write(df: pd.DataFrame, results_dir: Path, name: str,
           caption: str, label: str) -> None:
    if df.empty:
        logger.info("skip %s (no rows)", name)
        return
    df.to_csv(results_dir / f"{name}.csv", index=False)
    tex = df.to_latex(index=False, escape=False,
                      column_format="l" + "c" * (len(df.columns) - 1),
                      caption=caption, label=label)
    (results_dir / f"{name}.tex").write_text(tex, encoding="utf-8")
    logger.info("wrote %s.{csv,tex} (%d rows)", results_dir / name, len(df))


def _load(results_dir: Path, suffix: str = "") -> pd.DataFrame | None:
    path = results_dir / f"eval_summary{suffix}.csv"
    if not path.exists():
        logger.warning("%s missing", path)
        return None
    df = pd.read_csv(path)
    df["label"] = df["run"].map(MODEL_LABELS).fillna(df["run"])
    return df


def build_tables() -> None:
    base = load_config("config.yaml")
    results_dir = Path(base["paths"]["results_dir"])
    summary = _load(results_dir)
    if summary is None:
        raise SystemExit("results/eval_summary.csv missing - run src.evaluate first")

    is_ablation = summary["run"].str.startswith("ablation_")
    # SEED_RE is anchored to ^dramt_seed\d+$, so per-seed ablation runs
    # (ablation_<cfg>_s<seed>) are caught by is_ablation and never reach the
    # seed-robustness table.
    is_seed = summary["run"].str.match(SEED_RE)
    is_objective = summary["run"].isin(OBJECTIVE_RUNS)

    # Individual seed runs are excluded from the headline tables: they are 10
    # near-identical rows that would swamp the comparison. They get their own
    # robustness table, and the ensemble represents them in the main tables.
    models = summary[~is_ablation & ~is_seed].copy()
    ablations = summary[is_ablation].copy()

    # ---- point accuracy -------------------------------------------------
    dfp = pd.DataFrame([{
        "Model": r["label"],
        "MAE": _fmt(_v(r, "MAE"), _v(r, "MAE_std")),
        "RMSE": _fmt(_v(r, "RMSE"), _v(r, "RMSE_std")),
        "DirAcc": _fmt(_v(r, "DirAcc"), _v(r, "DirAcc_std"), 3),
    } for _, r in models.iterrows()])
    if not dfp.empty:
        dfp = dfp.sort_values("MAE")
    _write(dfp, results_dir, "tables_point",
           "Point-forecast accuracy across walk-forward folds (mean $\\pm$ std). "
           "The martingale row is the zero-forecast null.", "tab:point")

    # ---- risk -----------------------------------------------------------
    risk_rows = []
    for _, r in models.iterrows():
        if _isnan(_v(r, "VolRMSE")):
            continue
        row = {
            "Model": r["label"],
            "VaR breach (\\%)": (f"{100 * _v(r, 'VaRBreach'):.2f}"
                                 if not _isnan(_v(r, "VaRBreach")) else "--"),
            "Vol RMSE": _fmt(_v(r, "VolRMSE"), _v(r, "VolRMSE_std")),
            "Coverage 95\\%": _fmt(_v(r, "Coverage95"), _v(r, "Coverage95_std"), 3),
            "CRPS": _fmt(_v(r, "CRPS"), _v(r, "CRPS_std")),
            "Kupiec $p_{\\min}$": _fmt(_v(r, "Kupiec_p_min"), None),
            "Christof. $p_{\\min}$": _fmt(_v(r, "Christof_p_min"), None),
        }
        # Acerbi-Szekely ES backtest: Z2 near 0 = calibrated, Z2 < 0 = the
        # model understates tail losses. Unlike Kupiec this scores exceedance
        # SEVERITY, so it separates models with equal breach counts.
        if not _isnan(_v(r, "ES_Z2")):
            row["ES $Z_2$"] = _fmt(_v(r, "ES_Z2"), None, 3)
            row["ES $p_{\\min}$"] = _fmt(_v(r, "ES_Z2_p_min"), None)
        if not _isnan(_v(r, "df_h10")):
            row["$\\nu$ (h=10)"] = _fmt(_v(r, "df_h10"), None, 2)
        risk_rows.append(row)
    _write(pd.DataFrame(risk_rows), results_dir, "tables_risk",
           "Risk-forecast metrics at the 10-day horizon: VaR backtest calibration, "
           "volatility accuracy, interval coverage, CRPS, and the Acerbi-Szekely "
           "Expected Shortfall test (nominal breach rate 5\\%).", "tab:risk")

    # ---- ablations ------------------------------------------------------
    # Grouped over seeds. The +/- is the SEED spread, not the fold spread:
    # single-seed ablation deltas here are smaller than the seed noise floor,
    # so an effect is only credible if it exceeds this column.
    if len(ablations):
        ablations = ablations.copy()
        ablations["config"] = ablations["run"].str.replace(r"_s\d+$", "", regex=True)
        # permutation variants sit next to the deletion they disambiguate
        order = ["ablation_full",
                 "ablation_no_sentiment", "ablation_perm_sentiment",
                 "ablation_no_macro", "ablation_perm_macro",
                 "ablation_static_fusion", "ablation_point_only",
                 "ablation_numerical_only"]
        rank = {n: i for i, n in enumerate(order)}

        full_mae = np.nan
        grp = ablations.groupby("config")
        if "ablation_full" in grp.groups:
            full_mae = float(grp.get_group("ablation_full")["MAE"].mean())

        rows = []
        for cfg, g in grp:
            n_seeds = len(g)
            mae_mean = float(g["MAE"].mean())
            mae_sd = float(g["MAE"].std(ddof=1)) if n_seeds > 1 else np.nan
            delta = mae_mean - full_mae if np.isfinite(full_mae) else np.nan
            row = {
                "__o": rank.get(cfg, 99),
                "Configuration": MODEL_LABELS.get(cfg, cfg),
                "seeds": n_seeds,
                "MAE": _fmt(mae_mean, mae_sd),
                "$\\Delta$MAE vs full": ("--" if cfg == "ablation_full" or _isnan(delta)
                                         else f"{delta:+.5f}"),
                "DirAcc": _fmt(float(g["DirAcc"].mean()),
                               float(g["DirAcc"].std(ddof=1)) if n_seeds > 1 else None, 3),
            }
            if "VolRMSE" in g.columns and g["VolRMSE"].notna().any():
                row["Vol RMSE"] = _fmt(float(g["VolRMSE"].mean()),
                                       float(g["VolRMSE"].std(ddof=1)) if n_seeds > 1 else None)
                row["VaR breach (\\%)"] = (f"{100 * float(g['VaRBreach'].mean()):.2f}"
                                           if g["VaRBreach"].notna().any() else "--")
            else:
                row["Vol RMSE"] = row["VaR breach (\\%)"] = "--"
            rows.append(row)
        dfa = pd.DataFrame(rows).sort_values("__o").drop(columns="__o")
        _write(dfa, results_dir, "tables_ablation",
               "Ablation study: each component removed in turn. Values are mean "
               "$\\pm$ standard deviation ACROSS SEEDS (folds averaged within a "
               "seed). An effect is only interpretable if $\\Delta$MAE exceeds "
               "the seed spread.", "tab:ablation")

    # ---- headline comparison -------------------------------------------
    dfc = pd.DataFrame([{
        "Model": r["label"],
        "MAE": _fmt(_v(r, "MAE"), _v(r, "MAE_std")),
        "RMSE": _fmt(_v(r, "RMSE"), _v(r, "RMSE_std")),
        "DirAcc": _fmt(_v(r, "DirAcc"), _v(r, "DirAcc_std"), 3),
        "CRPS": _fmt(_v(r, "CRPS"), _v(r, "CRPS_std")),
        "Cov. 95\\%": _fmt(_v(r, "Coverage95"), None, 3),
        "VaR breach (\\%)": (f"{100 * _v(r, 'VaRBreach'):.2f}"
                             if not _isnan(_v(r, "VaRBreach")) else "--"),
    } for _, r in models.iterrows()])
    if not dfc.empty:
        dfc = dfc.sort_values("RMSE")
    _write(dfc, results_dir, "tables_comparison",
           "Comparative analysis: proposed model versus baselines "
           "(point accuracy and risk calibration).", "tab:comparison")

    # ---- RQ2: sigma calibration ----------------------------------------
    # The single most important table for RQ2: the same trained models scored
    # under the original per-fold constant calibration versus the regime
    # adaptive rolling one. Same predictions, different post-hoc calibration,
    # so any difference is attributable to the calibration alone.
    g, roll = _load(results_dir, "_global"), _load(results_dir, "_rolling")
    if g is not None and roll is not None:
        gi = g.set_index("run")
        ri = roll.set_index("run")
        cal_rows = []
        for run in gi.index:
            if run not in ri.index or _isnan(_v(gi.loc[run], "VaRBreach")):
                continue
            if SEED_RE.match(run):
                continue
            a, b = gi.loc[run], ri.loc[run]
            cal_rows.append({
                "Model": MODEL_LABELS.get(run, run),
                "Breach global (\\%)": f"{100 * _v(a, 'VaRBreach'):.2f}",
                "Breach rolling (\\%)": f"{100 * _v(b, 'VaRBreach'):.2f}",
                "Kupiec $p$ global": _fmt(_v(a, "Kupiec_p_min"), None),
                "Kupiec $p$ rolling": _fmt(_v(b, "Kupiec_p_min"), None),
                "ES $Z_2$ global": _fmt(_v(a, "ES_Z2"), None, 3),
                "ES $Z_2$ rolling": _fmt(_v(b, "ES_Z2"), None, 3),
            })
        _write(pd.DataFrame(cal_rows), results_dir, "tables_calibration",
               "RQ2: effect of regime-adaptive rolling volatility calibration on "
               "10-day portfolio VaR (nominal breach rate 5\\%). Identical "
               "predictions under both columns; only the post-hoc calibration differs.",
               "tab:calibration")

    # ---- objective variants --------------------------------------------
    obj = summary[is_objective]
    if len(obj):
        dfo = pd.DataFrame([{
            "Objective": r["label"],
            "MAE": _fmt(_v(r, "MAE"), _v(r, "MAE_std")),
            "DirAcc": _fmt(_v(r, "DirAcc"), _v(r, "DirAcc_std"), 3),
            "Vol RMSE": _fmt(_v(r, "VolRMSE"), _v(r, "VolRMSE_std")),
            "CRPS": _fmt(_v(r, "CRPS"), _v(r, "CRPS_std")),
            "VaR breach (\\%)": (f"{100 * _v(r, 'VaRBreach'):.2f}"
                                 if not _isnan(_v(r, "VaRBreach")) else "--"),
        } for _, r in obj.iterrows()])
        _write(dfo, results_dir, "tables_objective",
               "Loss-balance and ranking-objective variants. Motivation: a "
               "constant-zero forecast is competitive on point error, so the "
               "mean term carries little signal while the volatility term is "
               "down-weighted by $\\lambda_1$.", "tab:objective")

    # ---- seed robustness -----------------------------------------------
    # One block per ensemble family, so a Gaussian member is never averaged
    # together with a Student-t + GARCH-hybrid one.
    families = [("dramt_seed", "dramt_ensemble", "Gaussian head"),
                ("dramt_tg_seed", "dramt_tg_ensemble",
                 "Student-$t$ + GARCH-hybrid head")]
    rows = []
    for prefix, ens_name, label in families:
        fam = summary[summary["run"].str.match(rf"^{prefix}\d+$")]
        if not len(fam):
            continue
        rows.append({"Run": f"\\textit{{{label}}} ({len(fam)} seeds)",
                     "MAE": "", "RMSE": "", "DirAcc": "", "VaR breach (\\%)": ""})
        rows.append({
            "Run": "  range (min - max)",
            "MAE": f"{fam['MAE'].min():.5f} - {fam['MAE'].max():.5f}",
            "RMSE": f"{fam['RMSE'].min():.5f} - {fam['RMSE'].max():.5f}",
            "DirAcc": f"{fam['DirAcc'].min():.3f} - {fam['DirAcc'].max():.3f}",
            "VaR breach (\\%)": (f"{100*fam['VaRBreach'].min():.1f} - "
                                 f"{100*fam['VaRBreach'].max():.1f}"
                                 if "VaRBreach" in fam and fam["VaRBreach"].notna().any()
                                 else "--"),
        })
        rows.append({
            "Run": "  seed std",
            "MAE": _fmt(float(fam["MAE"].std(ddof=1)), None, 5),
            "RMSE": _fmt(float(fam["RMSE"].std(ddof=1)), None, 5),
            "DirAcc": _fmt(float(fam["DirAcc"].std(ddof=1)), None, 3),
            "VaR breach (\\%)": "",
        })
        ens = summary[summary["run"] == ens_name]
        if len(ens):
            e = ens.iloc[0]
            rows.append({
                "Run": "  \\textbf{ensemble}",
                "MAE": _fmt(_v(e, "MAE"), None, 5),
                "RMSE": _fmt(_v(e, "RMSE"), None, 5),
                "DirAcc": _fmt(_v(e, "DirAcc"), None, 3),
                "VaR breach (\\%)": (f"{100 * _v(e, 'VaRBreach'):.2f}"
                                     if not _isnan(_v(e, "VaRBreach")) else "--"),
            })
    if rows:
        _write(pd.DataFrame(rows), results_dir, "tables_seeds",
               "Seed robustness by ensemble family. The Gaussian family is the "
               "originally selected configuration; the Student-$t$ + "
               "GARCH-hybrid family additionally carries the RQ2 calibration "
               "changes, so its ensemble is the model that embodies every "
               "contribution at once.", "tab:seeds")


if __name__ == "__main__":
    build_tables()
