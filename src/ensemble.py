"""Seed ensemble: combine N independently-seeded runs into one evaluable model.

Motivation: the CPU-era study found non-trivial seed variance (DirAcc
0.511-0.545 across 3 seeds), which makes any single-seed comparison fragile.
Averaging over seeds is the standard remedy and is itself a legitimate model,
so the ensemble is written out in the same on-disk format as any other run and
evaluated by the same pipeline -- no special-casing in evaluate.py.

Combination rules
-----------------
mu      arithmetic mean over seeds.

sigma   LAW OF TOTAL VARIANCE, not a mean of sigmas:
            sigma_ens^2 = mean_s(sigma_s^2) + var_s(mu_s)
        The first term is the average within-model (aleatoric) variance; the
        second is the disagreement between seeds. Averaging sigma directly
        would throw the disagreement term away and leave the ensemble
        systematically over-confident -- precisely the failure mode RQ2 is
        trying to fix. Under dist="student_t" sigma is a SCALE, so the
        variance identity is applied to the implied variance
        (scale^2 * nu/(nu-2)) and converted back to a scale afterwards.

corr    mean of the per-seed correlation matrices, then renormalised to unit
        diagonal. A convex combination of correlation matrices is PSD, so the
        result is a valid correlation matrix.

df      mean over seeds (each seed learns its own per-horizon nu).

Usage:
    python -m src.ensemble --runs dramt_s42 dramt_s43 ... --out dramt_ensemble
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.utils.config import load_config
from src.utils.metrics import point_metrics
from src.utils.var_es import student_t_std_factor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _combine(members: list[dict], key_mu: str = "mu", key_sigma: str = "sigma") -> dict:
    mu = np.stack([m[key_mu] for m in members])                 # (S_seeds, n, S, H)
    mu_ens = mu.mean(axis=0)
    mu_var = mu.var(axis=0, ddof=1) if len(members) > 1 else np.zeros_like(mu_ens)

    out = {"mu": mu_ens}

    sigmas = [m[key_sigma] for m in members if m[key_sigma].size]
    if sigmas:
        sig = np.stack(sigmas)
        dfs = [m["df"] for m in members if "df" in m and np.size(m["df"])]
        if dfs:
            df_ens = np.stack(dfs).mean(axis=0)                  # (H,)
            # scale -> variance -> combine -> back to scale
            f2 = student_t_std_factor(np.stack(dfs))[:, None, None, :] ** 2
            var_within = (sig ** 2 * f2).mean(axis=0)
            var_total = var_within + mu_var
            out["sigma"] = np.sqrt(var_total) / student_t_std_factor(df_ens)[None, None, :]
            out["df"] = df_ens
        else:
            out["sigma"] = np.sqrt((sig ** 2).mean(axis=0) + mu_var)
            out["df"] = np.array([])
    else:
        out["sigma"] = np.array([])
        out["df"] = np.array([])
    return out


def _renormalise_corr(corr: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diagonal(corr, axis1=1, axis2=2), 1e-12, None))
    return corr / (d[:, :, None] * d[:, None, :])


def build_ensemble(runs_dir: Path, member_names: list[str], out_name: str,
                   n_folds: int) -> None:
    out_dir = runs_dir / out_name
    results = []
    for k in range(n_folds):
        paths = [runs_dir / r / f"fold{k}" / "test_predictions.npz" for r in member_names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"fold {k}: missing member predictions: {missing}")
        members = [dict(np.load(p)) for p in paths]

        ref = members[0]
        for m in members[1:]:
            if not np.array_equal(m["anchor_dates"], ref["anchor_dates"]):
                raise ValueError(
                    f"fold {k}: ensemble members do not share test anchors - "
                    "they were not trained on identical folds"
                )

        comb = _combine(members)
        corr = np.stack([m["corr"] for m in members]).mean(axis=0)
        corr = _renormalise_corr(corr)
        weights = np.stack([m["weights"] for m in members]).mean(axis=0)

        fold_dir = out_dir / f"fold{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fold_dir / "test_predictions.npz",
            mu=comb["mu"], sigma=comb["sigma"], df=comb["df"], corr=corr,
            weights=weights,
            mu_mc=np.array([]), sigma_epistemic=np.array([]),
            sigma_aleatoric_mc=np.array([]),
            y_ret=ref["y_ret"], y_vol=ref["y_vol"], y_corr=ref["y_corr"],
            test_idx=ref["test_idx"], anchor_dates=ref["anchor_dates"],
        )

        # the ensemble also needs validation predictions so the post-hoc sigma
        # calibration in evaluate.py can be fitted for it like any other run
        vpaths = [runs_dir / r / f"fold{k}" / "val_predictions.npz" for r in member_names]
        if all(p.exists() for p in vpaths):
            vmembers = [dict(np.load(p)) for p in vpaths]
            vcomb = _combine(vmembers)
            np.savez_compressed(
                fold_dir / "val_predictions.npz",
                mu=vcomb["mu"], sigma=vcomb["sigma"], df=vcomb["df"],
                y_ret=vmembers[0]["y_ret"],
                anchor_dates=vmembers[0].get("anchor_dates", np.array([])),
            )

        pm = point_metrics(ref["y_ret"], comb["mu"])
        results.append({"fold": k, "test_point_metrics": pm,
                        "n_members": len(member_names)})
        logger.info("fold %d ensemble of %d seeds: %s", k, len(member_names),
                    {kk: round(v, 6) for kk, v in pm.items()})

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "members.json").write_text(json.dumps(member_names, indent=2))
    logger.info("wrote %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="member run names")
    parser.add_argument("--out", default="dramt_ensemble")
    args = parser.parse_args()

    base = load_config("config.yaml")
    build_ensemble(Path(base["paths"]["runs_dir"]), args.runs, args.out,
                   base["splits"]["n_folds"])


if __name__ == "__main__":
    main()
