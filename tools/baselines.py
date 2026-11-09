"""External forecasting baselines for the benchmark comparison (not part of the grouped_rope package).

Compact re-implementations following the published architectures and default
configurations; they are reference points, not the authors' exact code:

- statistical: Naive (repeat last value), SeasonalNaive (repeat last period),
  WindowMean (mean of the lookback window). No training.
- machine learning: ridge regression (closed form, see run_baselines.py) and
  LightGBM direct multi-step models (see run_baselines.py).
- deep learning: DLinear (Zeng et al., AAAI 2023), LSTM encoder + linear head.
- Transformers: PatchTST (Nie et al., ICLR 2023; channel-independent patches,
  RevIN, BatchNorm encoder) and iTransformer (Liu et al., ICLR 2024; one token
  per variate, multivariate input).

Every model maps [batch, lookback, channels] -> [batch, horizon, channels].
All except iTransformer are channel-independent (channels = 1 in the runner).
"""
import torch
from torch import nn


# ---------------------------------------------------------------- statistical
class Naive(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        return x[:, -1:].expand(-1, self.horizon, -1)


class SeasonalNaive(nn.Module):
    """y[t] = x[L - period + (t mod period)]: repeat the last observed period."""

    def __init__(self, horizon, period):
        super().__init__()
        self.horizon, self.period = horizon, period

    def forward(self, x):
        idx = x.shape[1] - self.period + torch.arange(self.horizon, device=x.device) % self.period
        return x[:, idx]


class WindowMean(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        return x.mean(1, keepdim=True).expand(-1, self.horizon, -1)


# ---------------------------------------------------------------- deep learning
class DLinear(nn.Module):
    """Moving-average trend/seasonal decomposition, one shared linear map per component."""

    def __init__(self, lookback, horizon, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(lookback, horizon)
        self.seasonal = nn.Linear(lookback, horizon)

    def forward(self, x):
        pad = (self.kernel - 1) // 2
        padded = torch.cat([x[:, :1].expand(-1, pad, -1), x, x[:, -1:].expand(-1, self.kernel - 1 - pad, -1)], 1)
        trend = padded.unfold(1, self.kernel, 1).mean(-1)  # [B, L, C]
        seasonal = x - trend
        out = self.trend(trend.transpose(1, 2)) + self.seasonal(seasonal.transpose(1, 2))
        return out.transpose(1, 2)


class LSTMForecaster(nn.Module):
    """LSTM encoder over the history; the last hidden state is mapped to all horizon steps."""

    def __init__(self, horizon, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])[..., None]


# ---------------------------------------------------------------- Transformers
class RevIN(nn.Module):
    """Reversible instance normalization with a learnable per-channel affine map."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def norm(self, x):
        self.mean = x.mean(1, keepdim=True).detach()
        self.std = (x.var(1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        return (x - self.mean) / self.std * self.weight + self.bias

    def denorm(self, y):
        return (y - self.bias) / (self.weight + self.eps * self.eps) * self.std + self.mean


class BatchNormTokens(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):  # [B, N, D]
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    """Post-norm Transformer encoder layer with BatchNorm (PatchTST) or LayerNorm (iTransformer)."""

    def __init__(self, d_model, n_heads, d_ff, dropout, norm="layer", activation=nn.GELU):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), activation(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        make = BatchNormTokens if norm == "batch" else nn.LayerNorm
        self.norm1, self.norm2 = make(d_model), make(d_model)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop1(self.attn(x, x, x, need_weights=False)[0]))
        return self.norm2(x + self.drop2(self.ff(x)))


class PatchTST(nn.Module):
    """PatchTST/supervised: RevIN, overlapping patches (end padding), learned positions, flatten head."""

    def __init__(self, lookback, horizon, patch=16, stride=8, d_model=128, n_heads=16, layers=3, d_ff=256,
                 dropout=0.2, head_dropout=0.0):
        super().__init__()
        self.patch, self.stride = patch, stride
        n_patches = (lookback - patch) // stride + 2
        self.revin = RevIN(1)
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.empty(n_patches, d_model).uniform_(-0.02, 0.02))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, dropout, norm="batch") for _ in range(layers))
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(head_dropout), nn.Linear(n_patches * d_model, horizon))

    def forward(self, x):  # [B, L, 1]
        z = self.revin.norm(x)[..., 0]
        z = torch.cat([z, z[:, -1:].expand(-1, self.stride)], 1)  # ReplicationPad1d((0, stride))
        tokens = self.drop(self.embed(z.unfold(1, self.patch, self.stride)) + self.pos)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.revin.denorm(self.head(tokens)[..., None])


class ITransformer(nn.Module):
    """iTransformer: each variate's whole history is one token; attention runs across variates."""

    def __init__(self, lookback, horizon, d_model=512, n_heads=8, layers=2, d_ff=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(lookback, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):  # [B, L, C]
        mean = x.mean(1, keepdim=True).detach()
        std = (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        tokens = self.drop(self.embed(((x - mean) / std).transpose(1, 2)))  # [B, C, D]
        for layer in self.layers:
            tokens = layer(tokens)
        return self.head(self.norm(tokens)).transpose(1, 2) * std + mean


# ---------------------------------------------------------------- configurations
SEASON = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96, "weather": 144, "electricity": 24, "traffic": 24,
          "exchange_rate": None}

# PatchTST/42 paper settings: small model on small datasets, default model elsewhere.
PATCHTST = {name: (dict(d_model=16, n_heads=4, d_ff=128, dropout=0.3) if name in ("ETTh1", "ETTh2", "exchange_rate")
                   else dict(d_model=128, n_heads=16, d_ff=256, dropout=0.2)) for name in SEASON}

# iTransformer official long-term forecasting scripts (layers, d_model = d_ff, batch size).
ITRANSFORMER = {"ETTh1": (2, 256, 32), "ETTh2": (2, 128, 32), "ETTm1": (2, 128, 32), "ETTm2": (2, 128, 32),
                "weather": (3, 512, 32), "exchange_rate": (2, 128, 32), "electricity": (3, 512, 16),
                "traffic": (4, 512, 16)}


def build(name, dataset, lookback, horizon):
    """Construct a trainable deep baseline. Returns (model, multivariate)."""
    if name == "dlinear":
        return DLinear(lookback, horizon), False
    if name == "lstm":
        return LSTMForecaster(horizon), False
    if name == "patchtst":
        return PatchTST(lookback, horizon, **PATCHTST[dataset]), False
    if name == "itransformer":
        layers, d, _ = ITRANSFORMER[dataset]
        return ITransformer(lookback, horizon, d_model=d, layers=layers, d_ff=d), True
    raise ValueError(name)


def statistical(name, dataset, horizon):
    if name == "naive":
        return Naive(horizon)
    if name == "seasonal_naive":
        return None if SEASON[dataset] is None else SeasonalNaive(horizon, SEASON[dataset])
    if name == "window_mean":
        return WindowMean(horizon)
    raise ValueError(name)


def count(model):
    return sum(p.numel() for p in model.parameters())


__all__ = ["build", "statistical", "count", "SEASON", "ITRANSFORMER"]
