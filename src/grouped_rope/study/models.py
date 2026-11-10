"""Baselines and the shared channel-independent wrapper used by every learned model.

All learned models see one channel at a time ([batch, lookback, 1]) with shared
weights across channels (channel independence, as in DLinear/PatchTST), and the
same per-window instance normalisation (RevIN without affine parameters): each
input window is standardised by its own mean/std and the forecast is mapped back.
The wrapper is identical for the Transformer variants, the linear model and the
LSTM, so it is part of the shared backbone, not of the proposed method.
"""

import torch
from torch import Tensor, nn

from ..model import GroupedRoPEForecaster


class InstanceNorm(nn.Module):
    """Per-window normalisation around any univariate core [B, L, 1] -> [B, H, 1]."""

    def __init__(self, core: nn.Module, eps: float = 1e-5):
        super().__init__()
        self.core, self.eps = core, eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        return self.core((x - mean) / std) * std + mean


class LinearForecaster(nn.Module):
    """One linear map from the lookback window to the horizon (RLinear when normalised)."""

    def __init__(self, lookback: int, horizon: int):
        super().__init__()
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x.squeeze(-1)).unsqueeze(-1)


class LSTMForecaster(nn.Module):
    def __init__(self, horizon: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).unsqueeze(-1)


class PatchForecaster(nn.Module):
    """PatchTST-style tokens: overlapping patches of the window are the Transformer tokens.

    The series is right-padded with its last value by `stride` (as in PatchTST) and cut into
    patches of length `patch_len` with step `stride`; RoPE then acts on the patch index.
    """

    def __init__(self, horizon: int, patch_len: int = 16, stride: int = 8, **transformer_kwargs):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.core = GroupedRoPEForecaster(input_dim=patch_len, output_dim=1, horizon=horizon, forecast_head="pooled",
                                          **transformer_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        z = x.squeeze(-1)
        z = torch.cat((z, z[:, -1:].expand(-1, self.stride)), dim=1)
        return self.core(z.unfold(1, self.patch_len, self.stride))


class NaiveForecaster(nn.Module):
    """Repeat the last observed value (random-walk forecast)."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = horizon

    def forward(self, x: Tensor) -> Tensor:
        return x[:, -1:].expand(-1, self.horizon, -1)


class SeasonalNaiveForecaster(nn.Module):
    """Repeat the last observed seasonal cycle of length `period`."""

    def __init__(self, horizon: int, period: int):
        super().__init__()
        self.horizon, self.period = horizon, period

    def forward(self, x: Tensor) -> Tensor:
        if self.period > x.shape[1]:
            raise ValueError("period must not exceed the lookback")
        idx = x.shape[1] - self.period + torch.arange(self.horizon, device=x.device) % self.period
        return x[:, idx]


def build_model(kind: str, lookback: int, horizon: int, period: int | None = None,
                patch_len: int = 16, stride: int = 8, **transformer_kwargs) -> nn.Module:
    """kind: transformer | patch_transformer | linear | lstm | naive | seasonal_naive (univariate model)."""
    if kind == "transformer":
        core = GroupedRoPEForecaster(input_dim=1, horizon=horizon, forecast_head="pooled", **transformer_kwargs)
        return InstanceNorm(core)
    if kind == "patch_transformer":
        return InstanceNorm(PatchForecaster(horizon, patch_len, stride, **transformer_kwargs))
    if kind == "linear":
        return InstanceNorm(LinearForecaster(lookback, horizon))
    if kind == "lstm":
        return InstanceNorm(LSTMForecaster(horizon))
    if kind == "naive":
        return NaiveForecaster(horizon)
    if kind == "seasonal_naive":
        if period is None:
            raise ValueError("seasonal_naive needs a period")
        return SeasonalNaiveForecaster(horizon, period)
    raise ValueError(f"Unknown model kind {kind}")


def rope_core(model: nn.Module) -> GroupedRoPEForecaster | None:
    """The GroupedRoPEForecaster inside a wrapped model, if any."""
    return next((m for m in model.modules() if isinstance(m, GroupedRoPEForecaster)), None)


def forecast_channels(model: nn.Module, x: Tensor) -> Tensor:
    """Apply a univariate model to every channel: [B, L, C] -> [B, H, C]."""
    batch, length, channels = x.shape
    flat = x.permute(0, 2, 1).reshape(batch * channels, length, 1)
    out = model(flat)
    return out.reshape(batch, channels, -1).permute(0, 2, 1)
