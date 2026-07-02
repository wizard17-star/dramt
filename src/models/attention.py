"""Attention & embedding building blocks for DRAM-T.

Modules (numbering follows the architecture spec):
2. Time2Vec time-aware embedding (+ sinusoidal alternative)
3. Feature-level attention: X_att = softmax(Conv1d(X)) * X (elementwise)
4. Intra-modal encoder layer: pre-norm MHA + position-wise FFN, residual, dropout
5. Inter-modal cross-attention: modality A queries the other modalities' tokens
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """t2v(t)[0] = w0*t + b0 (trend); t2v(t)[1:] = sin(w_i*t + b_i) (periodic).

    Applied to the within-window time index 0..T-1, output added to token embeddings.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1, 1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(1, d_model - 1) * 0.01)
        self.b = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, T: int, device: torch.device) -> torch.Tensor:  # (T, d_model)
        t = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)  # (T,1)
        trend = t @ self.w0 + self.b0                      # (T,1)
        periodic = torch.sin(t @ self.w + self.b)          # (T,d-1)
        return torch.cat([trend, periodic], dim=-1)


def sinusoidal_encoding(T: int, d_model: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(T, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: (d_model + 1) // 2])
    return pe


class FeatureLevelAttention(nn.Module):
    """Per-modality feature gating: A = softmax_over_features(Conv1d(X)); out = A * X.

    Conv1d runs along time with kernel 3 (causal padding on the left only, so no
    future information crosses into position t), producing one logit per feature
    per timestep; softmax over the feature axis re-weights features per step.
    """

    def __init__(self, n_features: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(n_features, n_features, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,F)
        z = x.transpose(1, 2)                                   # (B,F,T)
        z = nn.functional.pad(z, (self.kernel_size - 1, 0))     # causal left pad
        logits = self.conv(z).transpose(1, 2)                   # (B,T,F)
        attn = torch.softmax(logits, dim=-1)
        return attn * x


class EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer (MHA + FFN, residual, dropout)."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,d)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class CrossModalAttention(nn.Module):
    """Modality A (query) attends to the concatenated tokens of the other
    modalities (key/value); pre-norm, residual, followed by FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, query_tokens: torch.Tensor, other_tokens: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.norm_q(query_tokens), self.norm_kv(other_tokens),
                         self.norm_kv(other_tokens), need_weights=False)
        x = query_tokens + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """Horizon-query decoder layer: self-attention over horizon queries +
    cross-attention to the fused encoder memory."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self.norm1(queries)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        q = queries + self.drop(a)
        a, _ = self.cross_attn(self.norm2(q), memory, memory, need_weights=False)
        q = q + self.drop(a)
        q = q + self.drop(self.ffn(self.norm3(q)))
        return q
