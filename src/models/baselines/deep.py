"""Deep baselines. All take (B, T, F) input and output point forecasts (B, S, H).

Unimodal models receive the numerical modality only (OHLCV+indicators);
SentimentLSTM receives numerical + sentiment concatenated (static fusion).
None produce risk outputs — they are the point-only comparison set (RQ2).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention import EncoderLayer, sinusoidal_encoding


class LSTMBaseline(nn.Module):
    def __init__(self, n_features: int, n_stocks: int, n_horizons: int,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_stocks * n_horizons)
        self.S, self.H = n_stocks, n_horizons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        y = self.head(out[:, -1])
        return y.view(-1, self.S, self.H)


class GRUBaseline(nn.Module):
    def __init__(self, n_features: int, n_stocks: int, n_horizons: int,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_stocks * n_horizons)
        self.S, self.H = n_stocks, n_horizons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        y = self.head(out[:, -1])
        return y.view(-1, self.S, self.H)


class CNNBiLSTM(nn.Module):
    """Conv1d feature extractor (causal) -> BiLSTM -> last-step head."""

    def __init__(self, n_features: int, n_stocks: int, n_horizons: int,
                 conv_channels: int = 64, hidden: int = 64, dropout: float = 0.2,
                 use_attention: bool = False) -> None:
        super().__init__()
        self.kernel = 3
        self.conv = nn.Conv1d(n_features, conv_channels, self.kernel)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(conv_channels, hidden, 1, batch_first=True, bidirectional=True)
        self.use_attention = use_attention
        if use_attention:
            self.attn_vec = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, n_stocks * n_horizons)
        self.S, self.H = n_stocks, n_horizons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = nn.functional.pad(z, (self.kernel - 1, 0))          # causal
        z = self.relu(self.conv(z)).transpose(1, 2)             # (B,T,C)
        out, _ = self.bilstm(z)                                  # (B,T,2h)
        if self.use_attention:
            w = torch.softmax(self.attn_vec(out), dim=1)         # (B,T,1)
            rep = (w * out).sum(dim=1)
        else:
            rep = out[:, -1]
        return self.head(self.drop(rep)).view(-1, self.S, self.H)


class VanillaTransformer(nn.Module):
    """Plain Transformer encoder over the numerical window; mean-pool + head."""

    def __init__(self, n_features: int, n_stocks: int, n_horizons: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 ffn_mult: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_stocks * n_horizons)
        self.S, self.H = n_stocks, n_horizons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = h + sinusoidal_encoding(h.shape[1], h.shape[2], h.device).unsqueeze(0)
        for layer in self.layers:
            h = layer(h)
        rep = self.norm(h).mean(dim=1)
        return self.head(rep).view(-1, self.S, self.H)


class SentimentLSTM(nn.Module):
    """Multimodal baseline: price + sentiment CONCATENATED (static fusion),
    single LSTM — the classic sentiment-augmented architecture."""

    def __init__(self, n_num_features: int, n_sent_features: int,
                 n_stocks: int, n_horizons: int,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_num_features + n_sent_features, hidden, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_stocks * n_horizons)
        self.S, self.H = n_stocks, n_horizons

    def forward(self, x_num: torch.Tensor, x_sent: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(torch.cat([x_num, x_sent], dim=-1))
        y = self.head(out[:, -1])
        return y.view(-1, self.S, self.H)


def make_deep_baseline(name: str, n_num: int, n_sent: int, n_stocks: int, n_horizons: int) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return LSTMBaseline(n_num, n_stocks, n_horizons)
    if name == "gru":
        return GRUBaseline(n_num, n_stocks, n_horizons)
    if name == "cnn_bilstm":
        return CNNBiLSTM(n_num, n_stocks, n_horizons, use_attention=False)
    if name == "cnn_bilstm_attn":
        return CNNBiLSTM(n_num, n_stocks, n_horizons, use_attention=True)
    if name == "transformer":
        return VanillaTransformer(n_num, n_stocks, n_horizons)
    if name == "sentiment_lstm":
        return SentimentLSTM(n_num, n_sent, n_stocks, n_horizons)
    raise ValueError(f"unknown deep baseline {name!r}")
