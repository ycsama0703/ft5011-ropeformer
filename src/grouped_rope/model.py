"""General time-series forecasting with grouped, learnable RoPE frequencies."""

from dataclasses import dataclass
from typing import Literal
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


RoPEMode = Literal["none", "standard", "fixed_head", "head", "shared_group", "grouped"]
ROPE_MODES = ("none", "standard", "fixed_head", "head", "shared_group", "grouped")
ForecastHead = Literal["horizon_tokens", "pooled"]


@dataclass(frozen=True)
class ForecastConfig:
    input_dim: int = 1
    output_dim: int | None = None
    horizon: int = 24
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    rope_base: float = 10000.0
    rope_mode: RoPEMode = "grouped"
    n_freq_groups: int = 2
    head_scale_init: tuple[float, ...] | None = None
    forecast_head: ForecastHead = "horizon_tokens"
    pooling: Literal["last", "mean"] | None = None  # pooled head only; None means "mean"
    causal: bool = False

    def __post_init__(self):
        for name in ("input_dim", "horizon", "d_model", "n_heads", "num_layers", "d_ff", "n_freq_groups"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.output_dim is not None and (type(self.output_dim) is not int or self.output_dim <= 0):
            raise ValueError("output_dim must be a positive integer or None")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model must be divisible by n_heads and head dimension must be even")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError("rope_base must be finite and greater than 1")
        if self.rope_mode not in ROPE_MODES:
            raise ValueError(f"Unknown rope_mode: {self.rope_mode}")
        if self.rope_mode in ("shared_group", "grouped") and self.n_freq_groups > self.d_model // self.n_heads // 2:
            raise ValueError("n_freq_groups cannot exceed the number of rotary coordinate pairs")
        if self.forecast_head not in ("horizon_tokens", "pooled"):
            raise ValueError("forecast_head must be 'horizon_tokens' or 'pooled'")
        if self.pooling not in (None, "last", "mean"):
            raise ValueError("pooling must be 'last', 'mean' or None")
        if self.pooling is not None and self.forecast_head != "pooled":
            raise ValueError("pooling is only used by forecast_head='pooled'")
        if self.head_scale_init is not None:
            if self.rope_mode not in ("head", "fixed_head"):
                raise ValueError("head_scale_init is only supported for head and fixed_head modes")
            if len(self.head_scale_init) != self.n_heads or any(
                not math.isfinite(s) or s <= 0 for s in self.head_scale_init
            ):
                raise ValueError("head_scale_init must contain one finite positive scale per head")


def rotate_half(x: Tensor) -> Tensor:
    """Pair channel j with channel j + d/2 (the half-split RoPE convention)."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + rotate_half(x) * sin


class GroupedRotaryEmbedding(nn.Module):
    """Per-layer frequency parameters. Groups follow descending base frequency.

    The first group contains the highest frequencies. Uneven groups differ by
    at most one coordinate pair. No FFT, input-dependent scales, or filtering
    is performed. All trainable log-scales start at zero unless explicitly set.
    """

    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.mode = cfg.rope_mode
        pairs = self.head_dim // 2
        frequencies = cfg.rope_base ** (-torch.arange(pairs, dtype=torch.float32) / pairs)
        self.register_buffer("base_frequencies", frequencies)
        groups = cfg.n_freq_groups if cfg.rope_mode in ("shared_group", "grouped") else 1
        self.n_groups = groups
        sizes = [len(part) for part in torch.tensor_split(torch.arange(pairs), groups)]
        group_index = torch.repeat_interleave(torch.arange(groups), torch.tensor(sizes))
        self.register_buffer("group_index", group_index)
        rows = 1 if cfg.rope_mode in ("none", "standard", "shared_group") else cfg.n_heads
        scales = torch.zeros(rows, groups)
        if cfg.head_scale_init is not None:
            scales[:, 0] = torch.tensor(cfg.head_scale_init).log()
        if cfg.rope_mode in ("none", "standard", "fixed_head"):
            self.register_buffer("log_scales", scales)
        else:
            self.log_scales = nn.Parameter(scales)

    def effective_frequencies(self) -> Tensor:
        """Return [heads, coordinate_pairs] angular frequencies in radians/step.

        The diagnostic mode "none" (NoPE) has zero frequency: every rotation is the identity.
        """
        if self.mode == "none":
            return self.base_frequencies.float().new_zeros(self.n_heads, self.base_frequencies.numel())
        scales = self.log_scales.float().exp()[:, self.group_index]
        return (self.base_frequencies.float()[None, :] * scales).expand(self.n_heads, -1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        # q/k: [batch, heads, time, head_dim]. Evaluate the actual rotation;
        # interpolation of cached trigonometric tables is deliberately avoided.
        if q.shape != k.shape or q.ndim != 4 or q.shape[1] != self.n_heads or q.shape[-1] != self.head_dim:
            raise ValueError("q and k must have matching [batch, heads, time, head_dim] shapes")
        if self.mode == "none":
            return q, k
        with torch.autocast(device_type=q.device.type, enabled=False):
            positions = torch.arange(q.shape[2], device=q.device, dtype=torch.float32)
            angles = self.effective_frequencies()[:, None, :] * positions[None, :, None]
            angles = torch.cat((angles, angles), dim=-1)
            cos, sin = angles.cos()[None].to(q.dtype), angles.sin()[None].to(q.dtype)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.rotary = GroupedRotaryEmbedding(cfg)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.out_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x: Tensor, mask: Tensor | None = None, return_attention: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        # mask: optional boolean [time, time]; True means the query may attend to the key.
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rotary(q, k)
        if return_attention:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            attention = scores.softmax(dim=-1)
            y = self.attn_dropout(attention) @ v
            attention = attention.detach()
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
            attention = None
        y = y.transpose(1, 2).reshape(batch, length, width)
        return self.out_dropout(self.out_proj(y)), attention


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.attention = MultiHeadAttentionRoPE(cfg)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout),
        )

    def forward(
        self, x: Tensor, mask: Tensor | None = None, return_attention: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        update, attention = self.attention(self.norm1(x), mask, return_attention)
        x = x + update
        return x + self.ff(self.norm2(x)), attention


def horizon_token_mask(history: int, horizon: int, causal: bool, device=None) -> Tensor:
    """Boolean [history+horizon]^2 mask; True means the query may attend to the key.

    History queries see only history (lower-triangular if causal), so history
    states do not depend on the horizon. Placeholder queries see all history
    plus themselves only, so step h's forecast does not depend on the total horizon.
    """
    total = history + horizon
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    past = torch.ones(history, history, dtype=torch.bool, device=device)
    mask[:history, :history] = past.tril() if causal else past
    mask[history:, :history] = True
    future = torch.arange(history, total, device=device)
    mask[future, future] = True
    return mask


class GroupedRoPEForecaster(nn.Module):
    """Direct multi-step forecasting. forward(x) -> [batch, horizon, output_dim].

    forecast_head="horizon_tokens" (default) appends one shared learnable
    placeholder token per future step at positions history..history+horizon-1;
    RoPE is then the only thing distinguishing future steps, and each step is
    read out by a shared per-token linear layer. forecast_head="pooled" keeps
    the earlier mean/last pooling + flat linear head as an ablation.

    Keyword arguments use exactly the ForecastConfig field names. Legacy oil
    interfaces and silent argument aliases are intentionally not supported.
    """

    def __init__(self, cfg: ForecastConfig | None = None, **kwargs):
        super().__init__()
        if cfg is not None and kwargs:
            raise TypeError("Pass either cfg or configuration keywords, not both")
        self.cfg = cfg if cfg is not None else ForecastConfig(**kwargs)
        cfg = self.cfg
        self.output_dim = cfg.input_dim if cfg.output_dim is None else cfg.output_dim
        self.input_projection = nn.Linear(cfg.input_dim, cfg.d_model)
        if cfg.forecast_head == "horizon_tokens":
            self.horizon_token = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.output_norm = nn.LayerNorm(cfg.d_model)
        width = self.output_dim if cfg.forecast_head == "horizon_tokens" else cfg.horizon * self.output_dim
        self.forecast_head = nn.Linear(cfg.d_model, width)

    def forward(self, x: Tensor, *, horizon: int | None = None, return_attention: bool = False):
        """horizon overrides cfg.horizon at call time (horizon_tokens head only)."""
        if x.ndim != 3 or x.shape[-1] != self.cfg.input_dim or x.shape[1] == 0:
            raise ValueError(f"Expected [batch, nonempty time, {self.cfg.input_dim}] input")
        if not x.is_floating_point():
            raise TypeError("Input must be a floating-point tensor")
        tokens = self.cfg.forecast_head == "horizon_tokens"
        if horizon is not None and not tokens:
            raise ValueError("horizon can only be overridden with forecast_head='horizon_tokens'")
        if horizon is not None and (type(horizon) is not int or horizon <= 0):
            raise ValueError("horizon must be a positive integer")
        horizon = self.cfg.horizon if horizon is None else horizon
        batch, history = x.shape[:2]
        h = self.input_projection(x)
        if tokens:
            placeholders = self.horizon_token.to(h.dtype).expand(batch, horizon, -1)
            h = torch.cat((h, placeholders), dim=1)
            mask = horizon_token_mask(history, horizon, self.cfg.causal, x.device)
        else:
            mask = torch.ones(history, history, dtype=torch.bool, device=x.device).tril() if self.cfg.causal else None
        maps = []
        for block in self.blocks:
            h, attention = block(h, mask, return_attention)
            if return_attention:
                maps.append(attention)
        h = self.output_norm(h)
        if tokens:
            forecast = self.forecast_head(h[:, history:])
        else:
            pooled = h[:, -1] if self.cfg.pooling == "last" else h.mean(dim=1)
            forecast = self.forecast_head(pooled).reshape(batch, horizon, self.output_dim)
        return (forecast, maps) if return_attention else forecast

    def rope_diagnostics(self) -> list[dict[str, Tensor]]:
        """Detached snapshots; periods describe rotations, not learned signal periods.

        In the diagnostic mode "none", frequencies are zero and periods are infinite.
        """
        result = []
        for block in self.blocks:
            rotary = block.attention.rotary
            frequency = rotary.effective_frequencies().detach().clone()
            result.append({
                "scales": rotary.log_scales.detach().exp().clone(),
                "group_index": rotary.group_index.detach().clone(),
                "frequencies": frequency,
                "periods": 2 * math.pi / frequency,
            })
        return result
