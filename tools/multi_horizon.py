"""One model for every horizon: each training batch uses a horizon drawn from --train-horizons.

Same data, splits, optimizer, schedule, clipping and paired seeds as run_benchmark.train; only the
per-batch horizon differs. Training windows are those that fit the longest training horizon, so all
batches come from the same window set. Model selection uses validation MSE averaged over steps
1..max(--train-horizons). Test: per-step MSE on the windows that fit --test-horizon (the same windows
extrap_study.py uses with max(--eval-horizons) = --test-horizon), so results are directly comparable
with its extrapolated and directly trained models.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmark as rb  # noqa: E402
from grouped_rope import GroupedRoPEForecaster  # noqa: E402
from grouped_rope.data import WindowedSeries  # noqa: E402


def train_multi(model, data, train_samples, val_samples, horizons, args, seed):
    scale_params = [p for n, p in model.named_parameters() if n.endswith("log_scales")]
    other_params = [p for n, p in model.named_parameters() if not n.endswith("log_scales")]
    groups = [{"params": other_params, "weight_decay": args.weight_decay}]
    if scale_params:
        groups.append({"params": scale_params, "weight_decay": 0.0, "lr": args.scale_lr or args.lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    warmup = min(args.warmup_steps, args.steps)
    decay = max(1, args.steps - warmup)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: (i + 1) / warmup if i < warmup else 0.5 * (1 + math.cos(math.pi * (i - warmup) / decay)))
    batches = torch.Generator().manual_seed(seed + 1000)
    pick_h = torch.Generator().manual_seed(seed + 2000)
    longest = max(horizons)
    best, curve, trajectory = (float("inf"), None, -1), [], []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        h = horizons[torch.randint(len(horizons), (1,), generator=pick_h).item()]
        idx = torch.randint(train_samples.shape[0], (args.batch_size,), generator=batches)
        x, y = data.batch(train_samples[idx], h)
        loss = (rb.predict(model, x, args, horizon=h) - y).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            if scale_params:
                trajectory.append({"step": step, "log_scales": [p.detach().cpu().tolist() for p in scale_params]})
            sq, _ = rb.step_errors(model, data, val_samples, longest, args, call_horizon=longest)
            val = sq.mean().item()
            curve.append({"step": step, "train_loss": loss.item(), "val_mse": val})
            if val < best[0]:
                best = (val, {k: v.detach().clone() for k, v in model.state_dict().items()}, step)
    model.load_state_dict(best[1])
    out = {"val_mse": best[0], "best_step": best[2], "train_seconds": round(time.perf_counter() - start, 2),
           "curve": curve}
    if trajectory:
        out["scale_trajectory"] = trajectory
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--train-horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    p.add_argument("--test-horizon", type=int, default=720)
    p.add_argument("--report-horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    p.add_argument("--modes", nargs="+", default=["standard", "head"])
    p.add_argument("--bases", type=float, nargs="+", default=[1000.0, 10000.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--groups", type=int, default=2)
    p.add_argument("--patch-len", type=int, default=16)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scale-lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--val-samples", type=int, default=32768)
    p.add_argument("--eval-chunk", type=int, default=1024)
    p.add_argument("--instance-norm", action="store_true")
    p.add_argument("--amp", choices=["none", "bf16"], default="bf16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt-dir", type=Path, default=Path("results/ckpt_mh"))
    p.add_argument("--train-eval-samples", type=int, default=200000,
                   help="Fixed random subset of training windows for the reported training MSE")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = WindowedSeries.from_csv(args.data_dir / f"{args.dataset}.csv", args.dataset, args.lookback).to(device)
    longest = max(args.train_horizons)
    train_samples = data.samples("train", longest)
    val_all = data.samples("val", longest)
    if args.val_samples < val_all.shape[0]:
        pick = torch.randperm(val_all.shape[0], generator=torch.Generator().manual_seed(0))[:args.val_samples]
        val_all = val_all[pick.sort().values]
    test = data.samples("test", args.test_horizon)
    train_eval = train_samples
    if args.train_eval_samples < train_eval.shape[0]:
        pick = torch.randperm(train_eval.shape[0], generator=torch.Generator().manual_seed(0))[:args.train_eval_samples]
        train_eval = train_eval[pick.sort().values]
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(input_dim=1, d_model=args.d_model, n_heads=args.n_heads, num_layers=args.layers, d_ff=args.d_ff,
               dropout=args.dropout, n_freq_groups=args.groups, forecast_head="horizon_tokens",
               patch_len=args.patch_len, horizon=min(args.train_horizons))
    runs = []
    for seed in args.seeds:
        for mode in args.modes:
            for base in args.bases:
                torch.manual_seed(seed)
                model = GroupedRoPEForecaster(rope_mode=mode, rope_base=base, **cfg).to(device)
                r = train_multi(model, data, train_samples, val_all, args.train_horizons, args, seed)
                per_step, _ = rb.step_errors(model, data, test, args.test_horizon, args, call_horizon=args.test_horizon)
                per_step = per_step.cpu().tolist()
                r["per_step"] = per_step
                r["test_mse"] = {str(h): sum(per_step[:h]) / h for h in args.report_horizons}
                tr, _ = rb.step_errors(model, data, train_eval, longest, args, call_horizon=longest)
                tr = tr.cpu().tolist()
                r["train_mse"] = {str(h): sum(tr[:h]) / h for h in args.report_horizons if h <= longest}
                torch.save(model.state_dict(), args.ckpt_dir / f"{args.dataset}_p{args.patch_len}_{mode}_b{base:g}_s{seed}.pt")
                runs.append({"model": mode, "base": base, "seed": seed, **r})
                print(f"seed={seed} {mode:12} base={base:<7g} best@{r['best_step']:<5} "
                      + " ".join(f"H{h}={v:.4f}" for h, v in r["test_mse"].items()), flush=True)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps({"args": {k: (str(v) if isinstance(v, Path) else v)
                                                         for k, v in vars(args).items()}, "runs": runs}, indent=1))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
