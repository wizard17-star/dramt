"""Multitask risk-aware output heads for DRAM-T.

- MeanHead:  mu_hat    (B, S, H)
- VolHead:   sigma_hat (B, S, H), strictly positive via softplus + eps
- CorrHead:  valid correlation matrix via Cholesky factor L: Sigma = L L^T,
             then normalized to a correlation matrix R (unit diagonal, PSD).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MeanHead(nn.Module):
    def __init__(self, d_model: int, n_stocks: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_stocks)

    def forward(self, horizon_repr: torch.Tensor) -> torch.Tensor:  # (B,H,d) -> (B,S,H)
        return self.proj(horizon_repr).transpose(1, 2)


class VolHead(nn.Module):
    """sigma_hat (B,S,H), strictly positive.

    mode="learned"       sigma = softplus(Wh)                (original)
    mode="garch_hybrid"  sigma = softplus(Wh + b0) * sigma_garch
    mode="har_hybrid"    same, with a HAR-RV filter instead of GARCH(1,1)

    In hybrid mode the recursive GARCH(1,1) filter supplies the volatility
    LEVEL and the network only learns a multiplicative correction. The bias is
    initialised so softplus(b0) == 1, i.e. the head starts as the identity on
    the GARCH forecast and has to earn any deviation from it. This directly
    targets the diagnosed failure: a windowed transformer has no recursion and
    so cannot track a volatility regime shift inside a fold, while a GARCH
    filter updates from every new observation.
    """

    # softplus(0.5413) == 1.0
    _UNIT_SOFTPLUS_BIAS = 0.5413248546129181

    def __init__(self, d_model: int, n_stocks: int, eps: float = 1e-6,
                 mode: str = "learned") -> None:
        super().__init__()
        if mode not in ("learned", "garch_hybrid", "har_hybrid"):
            raise ValueError(f"unknown vol mode {mode!r}")
        self.mode = mode
        self.hybrid = mode in ("garch_hybrid", "har_hybrid")
        self.proj = nn.Linear(d_model, n_stocks)
        self.eps = eps
        if self.hybrid:
            nn.init.zeros_(self.proj.weight)
            nn.init.constant_(self.proj.bias, self._UNIT_SOFTPLUS_BIAS)

    def forward(self, horizon_repr: torch.Tensor,
                garch_vol: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.proj(horizon_repr).transpose(1, 2)          # (B,S,H)
        if self.hybrid:
            if garch_vol is None:
                raise ValueError(f"vol_mode={self.mode!r} requires garch_vol")
            return nn.functional.softplus(raw) * garch_vol.clamp_min(self.eps) + self.eps
        return nn.functional.softplus(raw) + self.eps


class CorrHead(nn.Module):
    """Predict S*(S+1)/2 Cholesky parameters from a pooled representation.

    Diagonal entries pass through softplus + eps (positive definiteness);
    Sigma = L L^T is then normalized to correlation:
    R = D^{-1/2} Sigma D^{-1/2}, D = diag(Sigma). R is symmetric PSD with
    unit diagonal by construction.
    """

    def __init__(self, d_model: int, n_stocks: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.S = n_stocks
        self.eps = eps
        n_tril = n_stocks * (n_stocks + 1) // 2
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_tril),
        )
        idx = torch.tril_indices(n_stocks, n_stocks)
        self.register_buffer("tril_idx", idx, persistent=False)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pooled: (B, d) -> (R, L) with R: (B,S,S) correlation, L: (B,S,S).

        The Cholesky construction and the D^{-1/2} normalization are forced to
        float32 even under autocast: reduced precision here would perturb the
        rsqrt of near-zero diagonals and can push R off the unit diagonal.
        """
        B = pooled.shape[0]
        params = self.proj(pooled).float()                            # (B, n_tril)
        L = params.new_zeros(B, self.S, self.S)
        L[:, self.tril_idx[0], self.tril_idx[1]] = params
        diag = torch.diagonal(L, dim1=1, dim2=2)
        diag_pos = nn.functional.softplus(diag) + self.eps
        L = L - torch.diag_embed(diag) + torch.diag_embed(diag_pos)
        sigma = L @ L.transpose(1, 2)                                 # (B,S,S) PSD
        d = torch.diagonal(sigma, dim1=1, dim2=2).clamp_min(self.eps)
        inv_sqrt = torch.rsqrt(d)
        R = sigma * inv_sqrt.unsqueeze(1) * inv_sqrt.unsqueeze(2)
        return R, L
