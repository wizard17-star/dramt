"""Composite point + risk loss for DRAM-T.

L = L_point + lambda1 * L_vol + lambda2 * L_corr
- L_point: Huber (delta=1.0 in percent units) on multi-horizon returns
- L_vol:   negative log-likelihood of the predictive distribution, mu detached
           so the vol head learns calibration without dragging the mean head
- L_corr:  Frobenius ||R_hat - R_realized||_F^2 (Cholesky-parameterized R_hat
           is always a valid correlation matrix, so no constraint penalty needed)
lambda3 (L2) is applied as optimizer weight_decay, not in this function.

Predictive distribution (`dist`):
- "gaussian":   L_vol = 0.5*(log sigma^2 + (y-mu)^2/sigma^2), sigma = std.
- "student_t":  L_vol = Student-t NLL with a LEARNABLE degrees-of-freedom
                parameter nu (per horizon, supplied by the model as out["df"]).
                Here sigma is the t SCALE, not the standard deviation; the two
                differ by sqrt(nu/(nu-2)) and every downstream risk metric must
                apply that factor (src/utils/var_es.py:student_t_std_factor).

Motivation for student_t: under the Gaussian head the 10-day portfolio VaR
breached at 11.9% against a nominal 5% (Kupiec rejects). A Gaussian predictive
density cannot simultaneously fit the centre and the tails of daily equity
returns, so the fitted sigma is pulled toward the bulk and the tail quantile
comes out too small. A heavier-tailed likelihood lets the model widen the tail
without inflating the central dispersion.

All target tensors are expected PRE-SCALED to percent units (target_scale).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# 0.5*log(2*pi): the additive constant that gaussian_nll OMITS.
#
# gaussian_nll is the original M3 form and drops this constant. It is a
# constant, so it changes no gradient and no optimization outcome, and every
# Gaussian run (including the whole hyperparameter sweep) is internally
# consistent without it. It is kept as-is so previously recorded l_vol values
# and sweep scores stay comparable.
#
# But student_t_nll below IS a complete negative log-likelihood. So l_vol is
# NOT directly comparable between dist="gaussian" and dist="student_t" runs:
# add GAUSSIAN_NLL_CONST to a Gaussian l_vol before comparing the two.
GAUSSIAN_NLL_CONST = 0.5 * math.log(2.0 * math.pi)


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL up to the additive constant GAUSSIAN_NLL_CONST."""
    sigma2 = sigma ** 2
    return 0.5 * (torch.log(sigma2) + (y - mu) ** 2 / sigma2)


def student_t_nll(y: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor,
                  nu: torch.Tensor) -> torch.Tensor:
    """Elementwise negative log-likelihood of y ~ t_nu(mu, scale).

    -log p = -lgamma((nu+1)/2) + lgamma(nu/2) + 0.5*log(nu*pi) + log(scale)
             + ((nu+1)/2) * log1p(z^2/nu),      z = (y-mu)/scale
    """
    z = (y - mu) / scale
    return (
        -torch.lgamma((nu + 1.0) / 2.0)
        + torch.lgamma(nu / 2.0)
        + 0.5 * torch.log(nu * math.pi)
        + torch.log(scale)
        + (nu + 1.0) / 2.0 * torch.log1p(z ** 2 / nu)
    )


class CompositeLoss(nn.Module):
    def __init__(self, lambda1: float = 0.1, lambda2: float = 0.1,
                 point_loss: str = "huber", risk_head: bool = True,
                 dist: str = "gaussian") -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.risk_head = risk_head
        if dist not in ("gaussian", "student_t"):
            raise ValueError(f"unknown dist {dist!r}")
        self.dist = dist
        if point_loss == "huber":
            self.point_fn = nn.HuberLoss(delta=1.0)
        elif point_loss == "mse":
            self.point_fn = nn.MSELoss()
        else:
            raise ValueError(f"unknown point_loss {point_loss!r}")

    def forward(
        self,
        out: dict[str, torch.Tensor],
        y_ret: torch.Tensor,    # (B,S,H) percent units
        y_corr: torch.Tensor,   # (B,S,S)
    ) -> dict[str, torch.Tensor]:
        l_point = self.point_fn(out["mu"], y_ret)
        if self.risk_head:
            mu_det = out["mu"].detach()
            if self.dist == "student_t":
                nu = out["df"]                      # (1,1,H) broadcast over B,S
                l_vol = student_t_nll(y_ret, mu_det, out["sigma"], nu).mean()
            else:
                l_vol = gaussian_nll(y_ret, mu_det, out["sigma"]).mean()
            l_corr = ((out["corr"] - y_corr) ** 2).sum(dim=(1, 2)).mean()
            total = l_point + self.lambda1 * l_vol + self.lambda2 * l_corr
        else:
            l_vol = torch.zeros((), device=y_ret.device)
            l_corr = torch.zeros((), device=y_ret.device)
            total = l_point
        return {"loss": total, "l_point": l_point, "l_vol": l_vol, "l_corr": l_corr}
