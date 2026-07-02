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
    def __init__(self, d_model: int, n_stocks: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_stocks)
        self.eps = eps

    def forward(self, horizon_repr: torch.Tensor) -> torch.Tensor:  # (B,H,d) -> (B,S,H)
        raw = self.proj(horizon_repr).transpose(1, 2)
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
        """pooled: (B, d) -> (R, L) with R: (B,S,S) correlation, L: (B,S,S)."""
        B = pooled.shape[0]
        params = self.proj(pooled)                                    # (B, n_tril)
        L = pooled.new_zeros(B, self.S, self.S)
        L[:, self.tril_idx[0], self.tril_idx[1]] = params
        diag = torch.diagonal(L, dim1=1, dim2=2)
        diag_pos = nn.functional.softplus(diag) + self.eps
        L = L - torch.diag_embed(diag) + torch.diag_embed(diag_pos)
        sigma = L @ L.transpose(1, 2)                                 # (B,S,S) PSD
        d = torch.diagonal(sigma, dim1=1, dim2=2).clamp_min(self.eps)
        inv_sqrt = torch.rsqrt(d)
        R = sigma * inv_sqrt.unsqueeze(1) * inv_sqrt.unsqueeze(2)
        return R, L
