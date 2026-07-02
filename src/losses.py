"""Composite point + risk loss for DRAM-T.

L = L_point + lambda1 * L_vol + lambda2 * L_corr
- L_point: Huber (delta=1.0 in percent units) on multi-horizon returns
- L_vol:   Gaussian NLL 0.5*(log sigma^2 + (y-mu)^2/sigma^2), mu detached so
           the vol head learns calibration without dragging the mean head
- L_corr:  Frobenius ||R_hat - R_realized||_F^2 (Cholesky-parameterized R_hat
           is always a valid correlation matrix, so no constraint penalty needed)
lambda3 (L2) is applied as optimizer weight_decay, not in this function.

All target tensors are expected PRE-SCALED to percent units (target_scale).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    def __init__(self, lambda1: float = 0.1, lambda2: float = 0.1,
                 point_loss: str = "huber", risk_head: bool = True) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.risk_head = risk_head
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
            sigma2 = out["sigma"] ** 2
            l_vol = (0.5 * (torch.log(sigma2) + (y_ret - out["mu"].detach()) ** 2 / sigma2)).mean()
            l_corr = ((out["corr"] - y_corr) ** 2).sum(dim=(1, 2)).mean()
            total = l_point + self.lambda1 * l_vol + self.lambda2 * l_corr
        else:
            l_vol = torch.zeros((), device=y_ret.device)
            l_corr = torch.zeros((), device=y_ret.device)
            total = l_point
        return {"loss": total, "l_point": l_point, "l_vol": l_vol, "l_corr": l_corr}
