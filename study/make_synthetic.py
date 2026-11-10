"""Generate the synthetic multi-scale mechanism dataset (deterministic).

Each channel mixes a non-sinusoidal fast cycle (period 24, three harmonics), a slow
cycle (period 168, two harmonics) and AR(1) noise. Both periods fit inside a
336-step lookback, and their ratio (7) differs from the spacing of the RoPE
frequency ladder, which is the situation where independent group scales should
matter. Written in the same CSV format as the real benchmarks.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def harmonic_cycle(t, period, n_harmonics, rng):
    out = np.zeros_like(t, dtype=float)
    for k in range(1, n_harmonics + 1):
        amp = rng.uniform(0.3, 1.0) / k
        out += amp * np.sin(2 * np.pi * k * t / period + rng.uniform(0, 2 * np.pi))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/synthetic_multiscale.csv"))
    p.add_argument("--length", type=int, default=20000)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    t = np.arange(args.length, dtype=float)
    columns = {}
    for c in range(args.channels):
        fast = rng.uniform(0.5, 1.5) * harmonic_cycle(t, 24, 3, rng)
        slow = rng.uniform(0.5, 1.5) * harmonic_cycle(t, 168, 2, rng)
        noise = np.zeros(args.length)
        eps = rng.normal(0, 0.3, args.length)
        for i in range(1, args.length):
            noise[i] = 0.8 * noise[i - 1] + eps[i]
        columns[f"c{c}" if c < args.channels - 1 else "OT"] = fast + slow + noise
    frame = pd.DataFrame(columns)
    frame.insert(0, "date", pd.date_range("2020-01-01", periods=args.length, freq="h").strftime("%Y-%m-%d %H:%M:%S"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, float_format="%.6f")
    print(f"wrote {args.out}: {frame.shape}")


if __name__ == "__main__":
    main()
