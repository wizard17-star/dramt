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

    The regime signal (B, n_regime) comes from the data pipeline. With
    regime_signals='extended' it carries VIX level and slope, trailing average
    portfolio correlation and drawdown in addition to the original four
    (see src/data/regime.py) - the RQ3 hypothesis being that the original
    4-dim signal was too coarse to identify when a modality should dominate.

    per_timestep=False  ->  one weight vector per sample,   (B, M)
    per_timestep=True   ->  a weight vector per TIME STEP,  (B, T, M)

    Per-timestep gating conditions on the regime AND on the fused token at
    that step, so the model can shift modality emphasis WITHIN a window (e.g.
    lean on news around an earnings day inside an otherwise quiet month)
    rather than only across windows.
    """

    def __init__(self, n_regime: int, n_modalities: int, hidden: int = 32,
                 per_timestep: bool = False, d_model: int | None = None) -> None:
        super().__init__()
        self.per_timestep = per_timestep
        if per_timestep:
            if d_model is None:
                raise ValueError("per-timestep gating needs d_model")
            self.reg_proj = nn.Linear(n_regime, hidden)
            self.tok_proj = nn.Linear(d_model, hidden)
            self.out = nn.Sequential(nn.GELU(), nn.Linear(hidden, n_modalities))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_regime, hidden), nn.GELU(), nn.Linear(hidden, n_modalities),
            )

    def forward(self, regime: torch.Tensor,
                tokens: torch.Tensor | None = None) -> torch.Tensor:
        if not self.per_timestep:
            return torch.softmax(self.mlp(regime), dim=-1)             # (B,M)
        # tokens: (B,M,T,d) -> per-step summary (B,T,d)
        h = self.reg_proj(regime).unsqueeze(1) + self.tok_proj(tokens.mean(dim=1))
        return torch.softmax(self.out(h), dim=-1)                       # (B,T,M)


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
        vol_mode: str = "learned",
        gate_per_timestep: bool = False,
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
                self.gate = GatingMLP(n_regime_features, self.n_modalities,
                                      per_timestep=gate_per_timestep, d_model=d_model)

        self.backbone = nn.ModuleList([
            EncoderLayer(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)
        ])
        self.horizon_queries = nn.Parameter(torch.randn(n_horizons, d_model) * 0.02)
        self.decoder = DecoderLayer(d_model, n_heads, ffn_mult, dropout)

        self.mean_head = MeanHead(d_model, n_stocks)
        self.vol_head = VolHead(d_model, n_stocks, mode=vol_mode)
        self.corr_head = CorrHead(d_model, n_stocks)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x_num: torch.Tensor,          # (B,T,F_num)
        x_macro: torch.Tensor,        # (B,T,F_macro)
        x_sent: torch.Tensor,         # (B,T,F_sent)
        regime: torch.Tensor,         # (B,n_regime)
        garch_vol: torch.Tensor | None = None,   # (B,S,H), vol_mode='garch_hybrid'
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
            if self.gate.per_timestep:
                w_t = self.gate(regime, stacked)                  # (B,T,M)
                fused = (stacked * w_t.permute(0, 2, 1).unsqueeze(-1)).sum(dim=1)
                # report the window-average weights so the logged/plotted
                # modality_weights stay (B,M) across both gating modes
                w = w_t.mean(dim=1)                               # (B,M)
            else:
                w = self.gate(regime)                             # (B,M)
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
        sigma = self.vol_head(horizon_repr, garch_vol)            # (B,S,H)
        pooled = horizon_repr.mean(dim=1)                         # (B,d)
        R, L = self.corr_head(pooled)                             # (B,S,S)

        out = {"mu": mu, "sigma": sigma, "corr": R, "chol": L, "modality_weights": w}
        if self.dist == "student_t":
            # (1,1,H) so it broadcasts against (B,S,H) targets
            out["df"] = (2.0 + nn.functional.softplus(self.df_raw)).view(1, 1, -1)
        return out
