"""DRAM-T: Dynamic and Risk-Aware Multimodal Transformer.

Forward pipeline (numbers = architecture spec items):
1. per-modality Linear projection to d_model
2. Time2Vec time embedding added per modality
3. feature-level attention on raw features (before projection)
4. intra-modal encoder (self-attention) per modality
5. inter-modal cross-attention (each modality attends to the others)
6. dynamic market-state gating -> input-dependent softmax weights over modalities
7. N-layer Transformer backbone over fused tokens + horizon-query decoder
8. multitask heads: mean, volatility (softplus), correlation (Cholesky)

Ablation flags:
- use_sentiment / use_macro: drop a modality entirely (inputs ignored).
- dynamic_weighting=False: static fusion (simple mean of modality tokens ==
  fixed equal coefficients, the "static concatenation" ablation).
- risk_head=False: point-only model (vol/corr outputs still produced for API
  compatibility but detached from any gradient via the loss, handled in train).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention import (
    CrossModalAttention,
    DecoderLayer,
    EncoderLayer,
    FeatureLevelAttention,
    Time2Vec,
)
from src.models.heads import CorrHead, MeanHead, VolHead


class ModalityEncoder(nn.Module):
    """Items 1-4 for one modality: feature attention -> projection -> time
    embedding -> intra-modal self-attention encoder."""

    def __init__(self, n_features: int, d_model: int, n_heads: int,
                 ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.feat_attn = FeatureLevelAttention(n_features)
        self.proj = nn.Linear(n_features, d_model)
        self.time2vec = Time2Vec(d_model)
        self.encoder = EncoderLayer(d_model, n_heads, ffn_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,F) -> (B,T,d)
        x = self.feat_attn(x)
        h = self.proj(x)
        h = h + self.time2vec(h.shape[1], h.device).unsqueeze(0)
        return self.encoder(self.drop(h))


class GatingMLP(nn.Module):
    """Item 6: regime-conditioned softmax weights over active modalities.

    Regime signal (B, n_regime) is computed by the data pipeline from recent
    realized volatility, return sign, and macro level (see train.py); the MLP
    maps it to input-dependent modality weights.
    """

    def __init__(self, n_regime: int, n_modalities: int, hidden: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_regime, hidden), nn.GELU(), nn.Linear(hidden, n_modalities),
        )

    def forward(self, regime: torch.Tensor) -> torch.Tensor:  # (B, M) softmax weights
        return torch.softmax(self.mlp(regime), dim=-1)


class DRAMT(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        n_macro_features: int,
        n_sent_features: int,
        n_regime_features: int,
        n_stocks: int,
        n_horizons: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_mult: int = 2,
        dropout: float = 0.2,
        use_sentiment: bool = True,
        use_macro: bool = True,
        dynamic_weighting: bool = True,
        risk_head: bool = True,
        dist: str = "gaussian",
    ) -> None:
        super().__init__()
        self.use_sentiment = use_sentiment
        self.use_macro = use_macro
        self.dynamic_weighting = dynamic_weighting
        self.risk_head = risk_head
        self.dist = dist

        if dist == "student_t":
            # One learnable degrees-of-freedom per HORIZON, shared across
            # stocks. Sharing across stocks is deliberate: with a common nu at
            # a given horizon the 5-stock vector is multivariate-t, and then
            # w'r is exactly t_nu(w'mu, w'(D R D)w) -- so the portfolio VaR
            # aggregation stays exact instead of being an approximation.
            # Per-horizon (rather than global) because tail heaviness of
            # cumulative returns shrinks as the horizon lengthens.
            # nu = 2 + softplus(raw) keeps nu > 2, i.e. finite variance, which
            # student_t_std_factor() needs to convert scale -> std.
            self.df_raw = nn.Parameter(torch.full((n_horizons,), 1.6))  # nu ~ 3.9

        self.enc_num = ModalityEncoder(n_num_features, d_model, n_heads, ffn_mult, dropout)
        if use_macro:
            self.enc_macro = ModalityEncoder(n_macro_features, d_model, n_heads, ffn_mult, dropout)
        if use_sentiment:
            self.enc_sent = ModalityEncoder(n_sent_features, d_model, n_heads, ffn_mult, dropout)

        self.n_modalities = 1 + int(use_macro) + int(use_sentiment)
        if self.n_modalities > 1:
            self.cross = nn.ModuleList([
                CrossModalAttention(d_model, n_heads, ffn_mult, dropout)
                for _ in range(self.n_modalities)
            ])
            if dynamic_weighting:
                self.gate = GatingMLP(n_regime_features, self.n_modalities)

        self.backbone = nn.ModuleList([
            EncoderLayer(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)
        ])
        self.horizon_queries = nn.Parameter(torch.randn(n_horizons, d_model) * 0.02)
        self.decoder = DecoderLayer(d_model, n_heads, ffn_mult, dropout)

        self.mean_head = MeanHead(d_model, n_stocks)
        self.vol_head = VolHead(d_model, n_stocks)
        self.corr_head = CorrHead(d_model, n_stocks)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x_num: torch.Tensor,          # (B,T,F_num)
        x_macro: torch.Tensor,        # (B,T,F_macro)
        x_sent: torch.Tensor,         # (B,T,F_sent)
        regime: torch.Tensor,         # (B,n_regime)
    ) -> dict[str, torch.Tensor]:
        tokens = [self.enc_num(x_num)]
        if self.use_macro:
            tokens.append(self.enc_macro(x_macro))
        if self.use_sentiment:
            tokens.append(self.enc_sent(x_sent))

        # item 5: each modality attends to the concatenation of the others
        if self.n_modalities > 1:
            crossed = []
            for i, tok in enumerate(tokens):
                others = torch.cat([t for j, t in enumerate(tokens) if j != i], dim=1)
                crossed.append(self.cross[i](tok, others))
            tokens = crossed

        # item 6: fuse with input-dependent weights (or static mean)
        stacked = torch.stack(tokens, dim=1)                     # (B,M,T,d)
        if self.n_modalities > 1 and self.dynamic_weighting:
            w = self.gate(regime)                                 # (B,M)
            fused = (stacked * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        else:
            w = torch.full(
                (x_num.shape[0], self.n_modalities),
                1.0 / self.n_modalities, device=x_num.device,
            )
            fused = stacked.mean(dim=1)                           # (B,T,d)

        # item 7: backbone + horizon-query decoder
        for layer in self.backbone:
            fused = layer(fused)
        fused = self.final_norm(fused)
        queries = self.horizon_queries.unsqueeze(0).expand(fused.shape[0], -1, -1)
        horizon_repr = self.decoder(queries, fused)               # (B,H,d)

        # item 8: heads
        mu = self.mean_head(horizon_repr)                         # (B,S,H)
        sigma = self.vol_head(horizon_repr)                       # (B,S,H)
        pooled = horizon_repr.mean(dim=1)                         # (B,d)
        R, L = self.corr_head(pooled)                             # (B,S,S)

        out = {"mu": mu, "sigma": sigma, "corr": R, "chol": L, "modality_weights": w}
        if self.dist == "student_t":
            # (1,1,H) so it broadcasts against (B,S,H) targets
            out["df"] = (2.0 + nn.functional.softplus(self.df_raw)).view(1, 1, -1)
        return out
