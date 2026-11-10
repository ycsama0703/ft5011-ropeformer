"""Linear reference at H=96 with train/val/test MSE on the standard H=96 windows (same protocol as run_benchmark)."""
import argparse, json, statistics as st, sys
from pathlib import Path
import torch
sys.path.insert(0, "tools")
import run_benchmark as rb
from grouped_rope.data import WindowedSeries

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True); p.add_argument("--steps", type=int, required=True)
p.add_argument("--eval-every", type=int, default=200); p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--instance-norm", action="store_true"); p.add_argument("--out", required=True)
a = p.parse_args()
args = argparse.Namespace(instance_norm=a.instance_norm, amp="none", eval_chunk=4096, horizon=96, lr=1e-3, scale_lr=None,
                          warmup_steps=300, steps=a.steps, weight_decay=0.01, batch_size=256, eval_every=a.eval_every)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = WindowedSeries.from_csv(f"data/{a.dataset}.csv", a.dataset, 336).to(dev)
tr, va, te = data.samples("train", 96), data.samples("val", 96), data.samples("test", 96)
if va.shape[0] > 32768:
    va = va[torch.randperm(va.shape[0], generator=torch.Generator().manual_seed(0))[:32768].sort().values]
tr_eval = tr if tr.shape[0] <= 200000 else tr[torch.randperm(tr.shape[0], generator=torch.Generator().manual_seed(0))[:200000].sort().values]
rows = []
for seed in a.seeds:
    torch.manual_seed(seed)
    m = rb.LinearForecaster(336, 96).to(dev)
    r = rb.train(m, data, (tr, va, te, None), args, seed)
    rows.append({"seed": seed, "train": rb.evaluate(m, data, tr_eval, 96, args)[0],
                 "val": rb.evaluate(m, data, data.samples("val", 96), 96, args)[0], "test": r["test_mse"], "test_mae": r["test_mae"]})
Path(a.out).write_text(json.dumps({"dataset": a.dataset, "instance_norm": a.instance_norm, "rows": rows}, indent=1))
f = lambda k: f"{st.mean(r[k] for r in rows):.3f}±{st.stdev([r[k] for r in rows]):.3f}"
print(f"{a.dataset:14} norm={a.instance_norm}  linear H96  train {f('train')}  val {f('val')}  test {f('test')}")
