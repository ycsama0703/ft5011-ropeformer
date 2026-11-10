"""Daily seasonal strength per dataset and channel (training rows only) -> results/periodicity.json.

Seasonal strength F = max(0, 1 - Var(R)/Var(S+R)) (Wang, Smith & Hyndman 2006), computed after removing a centred
moving average of 7 periods; S is the mean profile by phase (t mod P), R the remainder. P = one day of samples
(5 trading days for exchange_rate).
"""
import json, sys
sys.path.insert(0, "src")
import torch
from grouped_rope.data import load_csv, split_borders

def detrend(x, w):
    k = torch.ones(1, 1, w, dtype=x.dtype) / w
    ma = torch.nn.functional.conv1d(x.T[:, None], k, padding=w // 2)[:, 0, : x.shape[0]].T
    return (x - ma)[w: -w]

def strength(x, P):
    phase = torch.arange(x.shape[0]) % P
    S = torch.stack([x[phase == p].mean(0) for p in range(P)])[phase]
    return (1 - (x - S).var(0) / x.var(0)).clamp(min=0)

out = {}
for name, day in (("electricity", 24), ("ETTh1", 24), ("ETTm1", 96), ("weather", 144), ("btc_hourly", 24),
                  ("epf_np", 24), ("epf_pjm", 24), ("exchange_rate", 5)):
    try:
        names, v = load_csv(f"data/{name}.csv")
    except FileNotFoundError:
        continue
    x = detrend(v[:split_borders(name, v.shape[0], 336)["train"][2]], day * 7)
    s = strength(x, day)
    out[name] = {"period_steps": day, "channels": names, "per_channel": s.tolist(), "median": s.median().item()}
json.dump(out, open("results/periodicity.json", "w"), indent=1)
print({k: round(v["median"], 3) for k, v in out.items()})
