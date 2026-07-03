"""Generate thesis tables (CSV + LaTeX) from results/eval_summary.csv.

Outputs to results/:
  tables_point.csv/.tex      - point accuracy per model (mean +/- std)
  tables_risk.csv/.tex       - risk metrics per model
  tables_ablation.csv/.tex   - 6 ablation configs (when ablation runs exist)
  tables_comparison.csv/.tex - proposed vs baselines combined headline table

Usage: python -m src.tables
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_LABELS = {
    "dramt_final": "DRAM-T (proposed)",
    "dramt_full": "DRAM-T (interim, no sentiment)",
    "baseline_arima": "ARIMA",
    "baseline_garch": "GARCH(1,1)",
    "baseline_garch_midas": "GARCH-MIDAS",
    "baseline_dcc_garch": "DCC-GARCH",
    "baseline_lstm": "LSTM",
    "baseline_gru": "GRU",
    "baseline_cnn_bilstm": "CNN-BiLSTM",
    "baseline_cnn_bilstm_attn": "CNN-BiLSTM-Attn",
    "baseline_transformer": "Transformer (OHLCV)",
    "baseline_sentiment_lstm": "Sentiment-LSTM",
    "ablation_full": "Full model",
    "ablation_no_sentiment": "-- sentiment",
    "ablation_no_macro": "-- macro",
    "ablation_static_fusion": "-- dynamic weighting",
    "ablation_point_only": "-- risk-aware head",
    "ablation_numerical_only": "numerical only",
}


def _fmt(mean: float, std: float | None, digits: int = 4) -> str:
    if std is None or np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _write_tex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    tex = df.to_latex(index=False, escape=False, column_format="l" + "c" * (len(df.columns) - 1),
                      caption=caption, label=label)
    path.write_text(tex, encoding="utf-8")
    logger.info("wrote %s", path)


def build_tables() -> None:
    base = load_config("config.yaml")
    results_dir = Path(base["paths"]["results_dir"])
    summary = pd.read_csv(results_dir / "eval_summary.csv")
    summary["label"] = summary["run"].map(MODEL_LABELS).fillna(summary["run"])

    is_ablation = summary["run"].str.startswith("ablation_")
    models = summary[~is_ablation].copy()
    ablations = summary[is_ablation].copy()

    # ---- point-accuracy table ----
    point_rows = []
    for _, r in models.iterrows():
        point_rows.append({
            "Model": r["label"],
            "MAE": _fmt(r["MAE"], r.get("MAE_std")),
            "MSE": _fmt(r["MSE"], r.get("MSE_std"), 6),
            "RMSE": _fmt(r["RMSE"], r.get("RMSE_std")),
            "MAPE (\\%)": _fmt(r["MAPE"], r.get("MAPE_std"), 1),
            "DirAcc": _fmt(r["DirAcc"], r.get("DirAcc_std"), 3),
        })
    dfp = pd.DataFrame(point_rows).sort_values("MAE")
    dfp.to_csv(results_dir / "tables_point.csv", index=False)
    _write_tex(dfp, results_dir / "tables_point.tex",
               "Point-forecast accuracy across walk-forward folds (mean $\\pm$ std).",
               "tab:point")

    # ---- risk table (models exposing sigma) ----
    if "VolRMSE" in models.columns:
        risk = models.dropna(subset=["VolRMSE"])
        risk_rows = []
        for _, r in risk.iterrows():
            risk_rows.append({
                "Model": r["label"],
                "VaR breach (\\%)": f"{100 * r['VaRBreach']:.2f} (nom. 5)",
                "Vol RMSE": _fmt(r["VolRMSE"], r.get("VolRMSE_std")),
                "Coverage 95\\%": _fmt(r["Coverage95"], r.get("Coverage95_std"), 3),
                "CRPS": _fmt(r["CRPS"], r.get("CRPS_std")),
                "Kupiec $p_{min}$": f"{r['Kupiec_p_min']:.4f}",
                "Christof. $p_{min}$": f"{r['Christof_p_min']:.4f}",
            })
        dfr = pd.DataFrame(risk_rows)
        dfr.to_csv(results_dir / "tables_risk.csv", index=False)
        _write_tex(dfr, results_dir / "tables_risk.tex",
                   "Risk-forecast metrics: VaR backtest calibration, volatility accuracy, "
                   "interval coverage and CRPS (mean across folds).", "tab:risk")

    # ---- ablation table ----
    if len(ablations):
        order = ["ablation_full", "ablation_no_sentiment", "ablation_no_macro",
                 "ablation_static_fusion", "ablation_point_only", "ablation_numerical_only"]
        ablations["__o"] = ablations["run"].map({n: i for i, n in enumerate(order)})
        ablations = ablations.sort_values("__o")
        ab_rows = []
        for _, r in ablations.iterrows():
            row = {
                "Configuration": r["label"],
                "MAE": _fmt(r["MAE"], r.get("MAE_std")),
                "RMSE": _fmt(r["RMSE"], r.get("RMSE_std")),
                "DirAcc": _fmt(r["DirAcc"], r.get("DirAcc_std"), 3),
            }
            if not np.isnan(r.get("VolRMSE", np.nan)):
                row["Vol RMSE"] = _fmt(r["VolRMSE"], r.get("VolRMSE_std"))
                row["CRPS"] = _fmt(r["CRPS"], r.get("CRPS_std"))
                row["VaR breach (\\%)"] = f"{100 * r['VaRBreach']:.2f}"
            else:
                row["Vol RMSE"] = row["CRPS"] = row["VaR breach (\\%)"] = "--"
            ab_rows.append(row)
        dfa = pd.DataFrame(ab_rows)
        dfa.to_csv(results_dir / "tables_ablation.csv", index=False)
        _write_tex(dfa, results_dir / "tables_ablation.tex",
                   "Ablation study: each component removed in turn (mean across folds).",
                   "tab:ablation")

    # ---- headline comparison table ----
    comp_rows = []
    for _, r in models.iterrows():
        comp_rows.append({
            "Model": r["label"],
            "RMSE": _fmt(r["RMSE"], r.get("RMSE_std")),
            "DirAcc": _fmt(r["DirAcc"], r.get("DirAcc_std"), 3),
            "CRPS": _fmt(r["CRPS"], r.get("CRPS_std")) if not np.isnan(r.get("CRPS", np.nan)) else "--",
            "Coverage 95\\%": _fmt(r["Coverage95"], None, 3) if not np.isnan(r.get("Coverage95", np.nan)) else "--",
            "VaR breach (\\%)": f"{100 * r['VaRBreach']:.2f}" if not np.isnan(r.get("VaRBreach", np.nan)) else "--",
        })
    dfc = pd.DataFrame(comp_rows).sort_values("RMSE")
    dfc.to_csv(results_dir / "tables_comparison.csv", index=False)
    _write_tex(dfc, results_dir / "tables_comparison.tex",
               "Comparative analysis: proposed model vs baselines (accuracy + risk).",
               "tab:comparison")


if __name__ == "__main__":
    build_tables()
