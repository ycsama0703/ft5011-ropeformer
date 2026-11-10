"""Train / val / test MSE at the training horizon for saved checkpoints (models trained at H=96).

train: fixed random subset (seed 0) of training windows, capped at --train-samples;
val: all validation windows; test: all test windows for H=96 (standard benchmark windows).
"""
import argparse, glob, json, re, statistics as st, sys
from pathlib import Path
import torch
sys.path.insert(0, "tools")
import run_benchmark as rb
from grouped_rope import GroupedRoPEForecaster
from grouped_rope.data import WindowedSeries

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--ckpt-dir", default="results/ckpt")
p.add_argument("--patch-len", type=int, default=16)
p.add_argument("--instance-norm", action="store_true")
p.add_argument("--train-samples", type=int, default=200000)
p.add_argument("--out", required=True)
a = p.parse_args()
args = argparse.Namespace(instance_norm=a.instance_norm, amp="bf16", eval_chunk=1024)
dev = torch.device("cuda")
data = WindowedSeries.from_csv(f"data/{a.dataset}.csv", a.dataset, 336).to(dev)
H = 96
sets = {}
tr = data.samples("train", H)
if a.train_samples < tr.shape[0]:
    tr = tr[torch.randperm(tr.shape[0], generator=torch.Generator().manual_seed(0))[:a.train_samples].sort().values]
sets["train"], sets["val"], sets["test"] = tr, data.samples("val", H), data.samples("test", H)
cfg = dict(input_dim=1, d_model=64, n_heads=4, num_layers=2, d_ff=128, dropout=0.1, n_freq_groups=2,
           forecast_head="horizon_tokens", patch_len=a.patch_len, horizon=H)
pat = re.compile(rf"{re.escape(a.dataset)}_p{a.patch_len}_(.+)_b([0-9.e+]+)_s(\d+)\.pt$")
rows = []
for f in sorted(glob.glob(f"{a.ckpt_dir}/{a.dataset}_p{a.patch_len}_*.pt")):
    m = pat.search(Path(f).name)
    if not m:
        continue
    spec, base, seed = m.group(1).replace("-", ":", 1) if m.group(1).startswith("fixed_head-") else m.group(1), float(m.group(2)), int(m.group(3))
    mode, _, init = spec.partition(":")
    extra = {"head_scale_init": tuple(float(v) for v in init.split(","))} if init else {}
    model = GroupedRoPEForecaster(rope_mode=mode, rope_base=base, **extra, **cfg).to(dev)
    model.load_state_dict(torch.load(f, map_location=dev))
    row = {"model": spec, "base": base, "seed": seed}
    for k, s in sets.items():
        row[k] = rb.evaluate(model, data, s, H, args)[0]
    rows.append(row)
Path(a.out).write_text(json.dumps({"dataset": a.dataset, "instance_norm": a.instance_norm,
                                   "train_windows_used": int(tr.shape[0]), "rows": rows}, indent=1))
print(f"{a.dataset} (instance_norm={a.instance_norm}, train windows used={tr.shape[0]})  MSE at H={H}, mean±std over seeds")
for spec in dict.fromkeys(r["model"] for r in rows):
    for base in sorted({r["base"] for r in rows if r["model"] == spec}):
        rs = [r for r in rows if r["model"] == spec and r["base"] == base]
        cell = lambda k: f"{st.mean(r[k] for r in rs):.3f}±{(st.stdev([r[k] for r in rs]) if len(rs) > 1 else 0):.3f}"
        print(f"  {spec:34} b{base:<6g} n={len(rs)}  train {cell('train')}  val {cell('val')}  test {cell('test')}")
