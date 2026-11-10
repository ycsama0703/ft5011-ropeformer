"""Benchmark data: standard chronological splits, train-only scaling, sliding windows.

Splits follow the long-term forecasting convention (Autoformer / DLinear / PatchTST):
ETT-hour uses 12/4/4 months, ETT-minute the same months at 15-minute resolution,
PEMS 60/20/20, and every other CSV a 70/10/20 chronological split. Validation and test histories
may start before their split boundary (they only use observations known at the
forecast origin); target windows never cross into a later split.

Any CSV whose first column is a date and whose other columns are numeric can be
used. Unknown names fall back to the 70/10/20 split, so the code runs on new data.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# period = dominant seasonal period in steps (for the seasonal-naive baseline);
# None means no meaningful seasonality (seasonal naive is skipped).
DATASETS = {
    "ETTh1": dict(file="ETTh1.csv", split="ett_hour", period=24),
    "ETTh2": dict(file="ETTh2.csv", split="ett_hour", period=24),
    "ETTm1": dict(file="ETTm1.csv", split="ett_minute", period=96),
    "ETTm2": dict(file="ETTm2.csv", split="ett_minute", period=96),
    "weather": dict(file="weather.csv", split="ratio", period=144),
    "electricity": dict(file="electricity.csv", split="ratio", period=24),
    "traffic": dict(file="traffic.csv", split="ratio", period=24),
    "exchange_rate": dict(file="exchange_rate.csv", split="ratio", period=None),
    "crypto_hourly": dict(file="crypto_hourly.csv", split="ratio", period=24),
    "solar": dict(file="solar.csv", split="ratio", period=144),
    "PEMS04": dict(file="PEMS04.csv", split="ratio622", period=288),
    "PEMS08": dict(file="PEMS08.csv", split="ratio622", period=288),
    "synthetic_multiscale": dict(file="synthetic_multiscale.csv", split="ratio", period=24),
}

SPLITS = ("train", "val", "test")


def split_borders(n_rows: int, lookback: int, scheme: str) -> tuple[list[int], list[int]]:
    """Return (starts, ends) row indices for train/val/test (history included)."""
    if scheme in ("ett_hour", "ett_minute"):
        unit = 30 * 24 * (4 if scheme == "ett_minute" else 1)
        ends = [12 * unit, 16 * unit, 20 * unit]
    elif scheme == "ratio":
        n_train, n_test = int(n_rows * 0.7), int(n_rows * 0.2)
        ends = [n_train, n_rows - n_test, n_rows]
    elif scheme == "ratio622":  # PEMS convention (Time-Series-Library): 60/20/20
        n_train, n_val = int(n_rows * 0.6), int(n_rows * 0.2)
        ends = [n_train, n_train + n_val, n_rows]
    else:
        raise ValueError(f"Unknown split scheme {scheme}")
    if ends[-1] > n_rows:
        raise ValueError(f"Split needs {ends[-1]} rows but the file has {n_rows}")
    starts = [0, ends[0] - lookback, ends[1] - lookback]
    return starts, ends


@dataclass
class SeriesData:
    """Standardised series on a device plus per-split window bookkeeping."""

    name: str
    series: torch.Tensor  # [time, channels], standardised with train statistics
    starts: list[int]
    ends: list[int]
    lookback: int
    horizon: int
    period: int | None
    mean: np.ndarray
    std: np.ndarray
    columns: list[str]

    @property
    def n_channels(self) -> int:
        return self.series.shape[1]

    def n_windows(self, split: str) -> int:
        i = SPLITS.index(split)
        return self.ends[i] - self.starts[i] - self.lookback - self.horizon + 1

    def windows(self, split: str, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return x [n, lookback, C] and y [n, horizon, C] for window indices of a split."""
        offset = self.starts[SPLITS.index(split)]
        t = torch.arange(self.lookback + self.horizon, device=self.series.device)
        rows = (index.to(self.series.device)[:, None] + offset) + t[None, :]
        block = self.series[rows]  # [n, L+H, C]
        return block[:, : self.lookback], block[:, self.lookback :]

    def pairs(self, split: str, index: torch.Tensor, channel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Univariate samples: x [n, lookback, 1], y [n, horizon, 1] for (window, channel) pairs."""
        offset = self.starts[SPLITS.index(split)]
        t = torch.arange(self.lookback + self.horizon, device=self.series.device)
        rows = (index[:, None] + offset) + t[None, :]
        block = self.series[rows, channel[:, None]].unsqueeze(-1)  # [n, L+H, 1]
        return block[:, : self.lookback], block[:, self.lookback :]


def load_dataset(
    name: str,
    data_dir: str | Path,
    lookback: int,
    horizon: int,
    device: str | torch.device = "cpu",
    file: str | None = None,
) -> SeriesData:
    spec = DATASETS.get(name, dict(file=f"{name}.csv", split="ratio", period=None))
    path = Path(data_dir) / (file or spec["file"])
    frame = pd.read_csv(path)
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)  # first column is the date
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains missing or non-finite values")
    starts, ends = split_borders(len(values), lookback, spec["split"])
    values = values[: ends[-1]]
    train = values[: ends[0]]
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # constant training columns stay finite
    scaled = (values - mean) / std
    data = SeriesData(
        name=name,
        series=torch.tensor(scaled, dtype=torch.float32, device=device),
        starts=starts,
        ends=ends,
        lookback=lookback,
        horizon=horizon,
        period=spec["period"],
        mean=mean,
        std=std,
        columns=list(frame.columns[1:]),
    )
    for split in SPLITS:
        if data.n_windows(split) < 1:
            raise ValueError(f"{name}: no {split} windows for lookback={lookback}, horizon={horizon}")
    return data
