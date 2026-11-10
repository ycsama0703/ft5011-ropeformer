"""Per channel: daily periodicity (training rows) vs extrapolation error of fixed vs learnable RoPE (saved train@96 models)."""
import argparse, statistics as st, sys
import torch
sys.path.insert(0, "tools"); sys.path.insert(0, "src")
import run_benchmark as rb
from grouped_rope import GroupedRoPEForecaster
from grouped_rope.data import WindowedSeries, load_csv, split_borders

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True); p.add_argument("--period", type=int, required=True)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); p.add_argument("--bases", type=float, nargs="+", default=[1000.0, 10000.0])
p.add_argument("--horizon", type=int, default=720); p.add_argument("--json", default=None)
a = p.parse_args()
args = argparse.Namespace(instance_norm=False, amp="bf16", eval_chunk=1024)
dev = torch.device("cuda")
names, raw = load_csv(f"data/{a.dataset}.csv")
x = raw[:split_borders(a.dataset, raw.shape[0], 336)["train"][2]]
def acf(x, k):
    x = x - x.mean(0); return (x[:-k] * x[k:]).sum(0) / (x.square().sum(0) + 1e-12)
period = (acf(x, a.period) - acf(x, a.period // 2)).tolist()
data = WindowedSeries.from_csv(f"data/{a.dataset}.csv", a.dataset, 336).to(dev)
test = data.samples("test", a.horizon)
cfg = dict(input_dim=1, d_model=64, n_heads=4, num_layers=2, d_ff=128, dropout=0.1, n_freq_groups=2,
           forecast_head="horizon_tokens", patch_len=16, horizon=96)

@torch.no_grad()
def per_channel(model):
    model.eval(); C = data.channels
    tot = torch.zeros(C, dtype=torch.float64, device=dev); cnt = torch.zeros(C, dtype=torch.float64, device=dev)
    for i in range(0, test.shape[0], 1024):
        s = test[i:i + 1024]
        xb, yb = data.batch(s, a.horizon)
        e = (rb.predict(model, xb, args, horizon=a.horizon) - yb).double()[:, 96:, 0].square().mean(1)  # steps 97..H
        ch = s[:, 1].to(dev)
        tot.index_add_(0, ch, e); cnt.index_add_(0, ch, torch.ones_like(e))
    return (tot / cnt).cpu()

res = {}
for mode in ("standard", "head"):
    vals = []
    for base in a.bases:
        for seed in a.seeds:
            m = GroupedRoPEForecaster(rope_mode=mode, rope_base=base, **cfg).to(dev)
            m.load_state_dict(torch.load(f"results/ckpt/{a.dataset}_p16_{mode}_b{base:g}_s{seed}.pt", map_location=dev))
            vals.append(per_channel(m))
    res[mode] = torch.stack(vals).mean(0)
ratio = res["standard"] / res["head"]
order = sorted(range(len(names)), key=lambda c: -period[c])
print(f"{a.dataset}: per channel, steps 97..{a.horizon}; ratio = fixed-RoPE MSE / learnable MSE (>1: learnable helps)")
for c in order:
    print(f"  {names[c][:16]:16} periodicity {period[c]:+.2f}   fixed {res['standard'][c]:.3f}  learnable {res['head'][c]:.3f}  ratio {ratio[c]:.2f}")
pr = torch.tensor(period); rr = ratio.log()
pc = ((pr - pr.mean()) * (rr - rr.mean())).sum() / (pr.std() * rr.std() * (len(pr) - 1))
print(f"Pearson corr(periodicity, log ratio) = {pc:.2f} over {len(pr)} channels")

if a.json:
    import json
    json.dump({"dataset": a.dataset, "channels": names, "periodicity": period, "fixed": res["standard"].tolist(),
               "learnable": res["head"].tolist()}, open(a.json, "w"), indent=1)
