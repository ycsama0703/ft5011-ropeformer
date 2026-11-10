"""Diebold-Mariano tests on saved train@96 checkpoints: model A vs model B on the extrapolated steps.

For each seed and base: per forecast origin t, loss L_t = squared error averaged over channels and over
steps (--from-step, --horizon]; d_t = L_A - L_B. DM = mean(d) / sqrt(LRV/T), LRV = Newey-West long-run
variance with Bartlett kernel (autocovariances / T) and lag = --horizon - 1 (overlapping multi-step windows). Two-sided normal p-value.
"""
import argparse, json, math, sys
from pathlib import Path
import torch
sys.path.insert(0, "tools")
import run_benchmark as rb
from grouped_rope import GroupedRoPEForecaster
from grouped_rope.data import WindowedSeries

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True); p.add_argument("--ckpt-dir", default="results/ckpt")
p.add_argument("--a", default="standard"); p.add_argument("--b", default="head")
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); p.add_argument("--bases", type=float, nargs="+", default=[1000.0, 10000.0])
p.add_argument("--horizon", type=int, default=720); p.add_argument("--from-step", type=int, default=96)
p.add_argument("--instance-norm", action="store_true"); p.add_argument("--out", required=True)
a = p.parse_args()
args = argparse.Namespace(instance_norm=a.instance_norm, amp="bf16", eval_chunk=1024)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = WindowedSeries.from_csv(f"data/{a.dataset}.csv", a.dataset, 336).to(dev)
test = data.samples("test", a.horizon)
origins = torch.unique(test[:, 0])
cfg = dict(input_dim=1, d_model=64, n_heads=4, num_layers=2, d_ff=128, dropout=0.1, n_freq_groups=2,
           forecast_head="horizon_tokens", patch_len=16, horizon=96)


def build(spec, base):
    mode, _, init = spec.partition(":")
    extra = {"head_scale_init": tuple(float(v) for v in init.split(","))} if init else {}
    return GroupedRoPEForecaster(rope_mode=mode, rope_base=base, **extra, **cfg).to(dev)


@torch.no_grad()
def per_origin_loss(spec, base, seed):
    m = build(spec, base)
    m.load_state_dict(torch.load(f"{a.ckpt_dir}/{a.dataset}_p16_{spec.replace(':', '-')}_b{base:g}_s{seed}.pt", map_location=dev))
    m.eval()
    tot = torch.zeros(origins.numel(), dtype=torch.float64, device=dev)
    cnt = torch.zeros_like(tot)
    index = {int(o): i for i, o in enumerate(origins.tolist())}
    pos = torch.tensor([index[int(o)] for o in test[:, 0].tolist()], device=dev)
    for i in range(0, test.shape[0], 1024):
        s = test[i:i + 1024]
        x, y = data.batch(s, a.horizon)
        e = (rb.predict(m, x, args, horizon=a.horizon) - y).double()[:, a.from_step:a.horizon, 0].square().mean(1)
        tot.index_add_(0, pos[i:i + 1024], e); cnt.index_add_(0, pos[i:i + 1024], torch.ones_like(e))
    return (tot / cnt).cpu()


def dm(d, lag):
    # Newey-West with Bartlett weights and autocovariances divided by T (guarantees a non-negative estimate)
    T = d.numel(); dc = d - d.mean()
    lrv = (dc * dc).sum() / T
    for k in range(1, min(lag, T - 1) + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] * dc[:-k]).sum() / T
    stat = d.mean() / math.sqrt(max(lrv.item(), 1e-300) / T)
    pval = math.erfc(abs(stat) / math.sqrt(2))
    return stat.item() if torch.is_tensor(stat) else stat, pval


rows = []
for base in a.bases:
    for seed in a.seeds:
        la, lb = per_origin_loss(a.a, base, seed), per_origin_loss(a.b, base, seed)
        d = la - lb
        stat, pval = dm(d, a.horizon - 1)
        rows.append({"base": base, "seed": seed, "mse_a": la.mean().item(), "mse_b": lb.mean().item(),
                     "mean_diff": d.mean().item(), "dm": stat, "p": pval, "T": d.numel()})
        print(f"{a.dataset:14} b{base:<6g} s{seed}  {a.a} {la.mean():.3f} vs {a.b} {lb.mean():.3f}  DM={stat:+.2f}  p={pval:.2g}  (T={d.numel()})", flush=True)
Path(a.out).write_text(json.dumps({"dataset": a.dataset, "a": a.a, "b": a.b, "steps": [a.from_step + 1, a.horizon],
                                   "nw_lag": a.horizon - 1, "rows": rows}, indent=1))
