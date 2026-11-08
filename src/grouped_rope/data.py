"""Benchmark CSV loading, standard chronological splits and window indexing.

Nothing here downloads data; use tools/download_data.py. Conventions follow the
long-horizon forecasting literature (Informer/Autoformer code):

- ETTh*: 12/4/4 months -> rows [0, 8640), [8640, 11520), [11520, 14400); the
  remaining rows are unused. ETTm* uses the same months at 4 rows per hour.
- other datasets: chronological 70/10/20 with int() rounding of train and test.
- validation/test histories may start up to ``lookback`` rows before the split
  border (observations available before the forecast origin), but every target
  step lies inside its own split.
- z-score statistics are fitted on the training rows only.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

SPLITS = ("train", "val", "test")


def load_csv(path: str | Path) -> tuple[list[str], torch.Tensor]:
    """Read a benchmark CSV with a leading date column. Returns (column names, [time, channels] float64)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row[1:]] for row in reader]
    values = torch.tensor(rows, dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError(f"{path} contains missing or non-finite values")
    return header[1:], values


def split_borders(name: str, length: int, lookback: int) -> dict[str, tuple[int, int, int]]:
    """Per split: (first history row, first target row, end row exclusive)."""
    if name.startswith("ETTh") or name.startswith("ETTm"):
        per_month = 30 * 24 * (4 if name.startswith("ETTm") else 1)
        ends = (12 * per_month, 16 * per_month, 20 * per_month)
        if ends[-1] > length:
            raise ValueError(f"{name} needs {ends[-1]} rows, found {length}")
    else:
        n_train, n_test = int(length * 0.7), int(length * 0.2)
        ends = (n_train, length - n_test, length)
    starts = (0, ends[0], ends[1])
    return {split: (max(0, start - lookback), start, end) for split, start, end in zip(SPLITS, starts, ends)}


@dataclass
class WindowedSeries:
    """A normalized multivariate series with channel-independent window indexing.

    A sample is (origin, channel): history = series[origin-L:origin, channel],
    target = series[origin:origin+H, channel], both inside one split.
    """

    series: torch.Tensor  # [time, channels], normalized with training statistics
    mean: torch.Tensor
    std: torch.Tensor
    borders: dict[str, tuple[int, int, int]]
    lookback: int

    @classmethod
    def from_csv(cls, path: str | Path, name: str, lookback: int, dtype=torch.float32) -> "WindowedSeries":
        _, values = load_csv(path)
        borders = split_borders(name, values.shape[0], lookback)
        train = values[: borders["train"][2]]
        mean, std = train.mean(0), train.std(0, unbiased=False)  # population std, as sklearn StandardScaler
        std = torch.where(std > 0, std, torch.ones_like(std))
        return cls(((values - mean) / std).to(dtype), mean, std, borders, lookback)

    @property
    def channels(self) -> int:
        return self.series.shape[1]

    def origins(self, split: str, horizon: int) -> torch.Tensor:
        """Forecast origins whose history fits in the data and whose targets stay inside the split."""
        first_history, first_target, end = self.borders[split]
        first = max(first_target, first_history + self.lookback)
        return torch.arange(first, end - horizon + 1)

    def samples(self, split: str, horizon: int) -> torch.Tensor:
        """All (origin, channel) pairs as a [n, 2] long tensor."""
        origins = self.origins(split, horizon)
        channels = torch.arange(self.channels)
        return torch.cartesian_prod(origins, channels)

    def to(self, device) -> "WindowedSeries":
        self.series = self.series.to(device)
        return self

    def batch(self, samples: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather histories [B, L, 1] and targets [B, H, 1] for (origin, channel) samples."""
        samples = samples.to(self.series.device)
        offsets = torch.arange(-self.lookback, horizon, device=self.series.device)
        window = self.series[samples[:, :1] + offsets, samples[:, 1:]]
        return window[:, : self.lookback, None], window[:, self.lookback :, None]
