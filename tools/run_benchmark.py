"""Shared benchmark runner: same backbone, budget and data for every RoPE mode.

Channel-independent setting: each variable is forecast as its own univariate
series (input_dim=1), so a sample is an (origin, channel) pair. Data come from
data/<dataset>.csv (see tools/download_data.py) with the standard chronological
split and z-score statistics fitted on training rows only. Metrics are MSE/MAE
on the normalized scale, averaged over every test window and channel
(no drop_last).

Protocol per seed: torch.manual_seed(seed) before building each model (paired
initialization), identical batch order from a generator seeded with seed+1000,
AdamW with log_scales in a no-weight-decay group (own lr via --scale-lr), warmup + cosine decay,
best state chosen on validation MSE. Each mode picks rope_base by mean
validation MSE over seeds; test is reported only for that choice. Validation
uses a fixed random subset of windows when --val-samples is smaller than the
validation set (recorded in the output).

With --eval-horizons, the horizon_tokens model trained at --horizon is also
scored on longer horizons (test windows that fit the longest horizon).

Shared options applied identically to every model, including the linear
reference, and recorded in the output:
--instance-norm  per-window z-score of each history (no learnable affine); the
                 forecast is mapped back with the same statistics, so metrics
                 stay on the dataset's training-fitted scale.
--amp bf16       bfloat16 autocast for model forward passes. RoPE angles and
                 sin/cos are still computed in float32 inside the model; the
                 loss and metrics are computed in float32/float64.
"""
import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grouped_rope import GroupedRoPEForecaster
from grouped_rope.data import WindowedSeries
from grouped_rope.model import ROPE_MODES

MODES = ("standard", "head", "shared_group", "grouped")


class LinearForecaster(nn.Module):
    """Channel-independent lookback -> horizon linear map (reference baseline)."""

    def __init__(self, lookback, horizon):
        super().__init__()
        self.proj = nn.Linear(lookback, horizon)

    def forward(self, x):
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


def predict(model, x, args, **kwargs):
    """Forward pass with the shared instance-norm and autocast options."""
    if args.instance_norm:
        mean = x.mean(1, keepdim=True)
        std = (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        x = (x - mean) / std
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=args.amp == "bf16"):
        out = model(x, **kwargs)
    out = out.float()
    return out * std + mean if args.instance_norm else out


@torch.no_grad()
def step_errors(model, data, samples, horizon, args, call_horizon=None):
    """Per-step MSE and MAE over all given samples."""
    model.eval()
    kwargs = {} if call_horizon is None else {"horizon": call_horizon}
    sq = torch.zeros(horizon, dtype=torch.float64, device=data.series.device)
    ab = torch.zeros_like(sq)
    for i in range(0, samples.shape[0], args.eval_chunk):
        x, y = data.batch(samples[i:i + args.eval_chunk], horizon)
        error = (predict(model, x, args, **kwargs) - y).double()
        sq += error.square().mean(dim=2).sum(0)  # channel mean: identical for [B, H, 1] samples
        ab += error.abs().mean(dim=2).sum(0)
    return sq / samples.shape[0], ab / samples.shape[0]


def evaluate(model, data, samples, horizon, args):
    sq, ab = step_errors(model, data, samples, horizon, args)
    return sq.mean().item(), ab.mean().item()


def train(model, data, splits, args, seed):
    train_samples, val_samples, test_samples, extra_samples = splits
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
    batches = torch.Generator().manual_seed(seed + 1000)  # identical batch order across modes
    trajectory, curve = [], []
    best = (float("inf"), None, -1)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(train_samples.shape[0], (args.batch_size,), generator=batches)
        x, y = data.batch(train_samples[idx], args.horizon)
        loss = (predict(model, x, args) - y).square().mean()
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
            val_mse, _ = evaluate(model, data, val_samples, args.horizon, args)
            curve.append({"step": step, "train_loss": loss.item(), "val_mse": val_mse})
            if val_mse < best[0]:
                best = (val_mse, {k: v.detach().clone() for k, v in model.state_dict().items()}, step)
    seconds = time.perf_counter() - start
    model.load_state_dict(best[1])
    test_mse, test_mae = evaluate(model, data, test_samples, args.horizon, args)
    result = {"val_mse": best[0], "best_step": best[2], "test_mse": test_mse, "test_mae": test_mae,
              "train_seconds": round(seconds, 2), "curve": curve,
              "parameters": sum(p.numel() for p in model.parameters())}
    if extra_samples is not None and isinstance(model, GroupedRoPEForecaster):
        longest = max(args.eval_horizons)
        per_step, _ = step_errors(model, data, extra_samples, longest, args, call_horizon=longest)
        result["extrapolation_per_step_mse"] = per_step.cpu().tolist()
        result["extrapolation_mse"] = {str(h): per_step[:h].mean().item() for h in [args.horizon, *args.eval_horizons]}
    if trajectory:
        result["scale_trajectory"] = trajectory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # keep the cached GPU memory of parallel processes bounded
    if isinstance(model, GroupedRoPEForecaster):
        result["scales"] = [info["scales"].cpu().tolist() for info in model.rope_diagnostics()]
    return result


def mean_std(values):
    t = torch.tensor(values, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if len(values) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Name of data/<dataset>.csv, e.g. ETTh1 or electricity")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--lookback", type=int, default=336)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--eval-horizons", type=int, nargs="*", default=[],
                        help="Longer test horizons for the trained horizon_tokens model")
    parser.add_argument("--modes", nargs="+", choices=ROPE_MODES, default=list(MODES))
    parser.add_argument("--head", choices=["horizon_tokens", "pooled"], default="horizon_tokens")
    parser.add_argument("--bases", type=float, nargs="+", default=[1000.0, 10000.0])
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--patch-len", type=int, default=None,
                        help="Non-overlapping patch tokens of this many steps for every RoPE mode (default: point tokens)")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scale-lr", type=float, default=None,
                        help="Learning rate of the RoPE log-scales (every learnable mode; default: --lr)")
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-samples", type=int, default=32768,
                        help="Fixed random subset of validation windows used for model selection")
    parser.add_argument("--eval-chunk", type=int, default=1024,
                        help="Windows per evaluation forward; larger chunks need much more GPU memory")
    parser.add_argument("--instance-norm", action="store_true",
                        help="Per-window z-score of histories for every model (no learnable parameters)")
    parser.add_argument("--amp", choices=["none", "bf16"], default="none")
    parser.add_argument("--no-linear", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.eval_horizons and args.head != "horizon_tokens":
        parser.error("--eval-horizons requires --head horizon_tokens")
    out = args.out or Path("results/benchmark") / f"{args.dataset}_L{args.lookback}_H{args.horizon}.json"
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    path = args.data_dir / f"{args.dataset}.csv"
    data = WindowedSeries.from_csv(path, args.dataset, args.lookback).to(device)
    train_samples = data.samples("train", args.horizon)
    val_all = data.samples("val", args.horizon)
    if args.val_samples < val_all.shape[0]:
        pick = torch.randperm(val_all.shape[0], generator=torch.Generator().manual_seed(0))[:args.val_samples]
        val_samples = val_all[pick.sort().values]
    else:
        val_samples = val_all
    test_samples = data.samples("test", args.horizon)
    extra_samples = data.samples("test", max(args.eval_horizons)) if args.eval_horizons else None
    counts = {"rows": data.series.shape[0], "channels": data.channels, "borders": data.borders,
              "train_samples": train_samples.shape[0], "val_samples_used": val_samples.shape[0],
              "val_samples_total": val_all.shape[0], "test_samples": test_samples.shape[0],
              "extrapolation_test_samples": None if extra_samples is None else extra_samples.shape[0]}
    print(f"{args.dataset}: {counts}", flush=True)
    splits = (train_samples, val_samples, test_samples, extra_samples)
    model_cfg = dict(input_dim=1, horizon=args.horizon, d_model=args.d_model, n_heads=args.n_heads,
                     num_layers=args.layers, d_ff=args.d_ff, dropout=args.dropout, n_freq_groups=args.groups,
                     forecast_head=args.head, patch_len=args.patch_len)

    runs = []
    for seed in args.seeds:
        if not args.no_linear:
            torch.manual_seed(seed)
            result = train(LinearForecaster(args.lookback, args.horizon).to(device), data, splits, args, seed)
            runs.append({"model": "linear", "base": None, "seed": seed, **result})
            print(f"seed={seed} linear                     val={result['val_mse']:.4f} test={result['test_mse']:.4f} "
                  f"mae={result['test_mae']:.4f}", flush=True)
        for mode in args.modes:
            for base in args.bases:
                torch.manual_seed(seed)  # identical backbone init across modes (log-scales start at 0)
                model = GroupedRoPEForecaster(rope_mode=mode, rope_base=base, **model_cfg).to(device)
                result = train(model, data, splits, args, seed)
                runs.append({"model": mode, "base": base, "seed": seed, **result})
                print(f"seed={seed} {mode:12} base={base:<7g}  val={result['val_mse']:.4f} "
                      f"test={result['test_mse']:.4f} mae={result['test_mae']:.4f} best@{result['best_step']} "
                      f"({result['train_seconds']}s)", flush=True)

    summary = []
    names = (["linear"] if not args.no_linear else []) + list(args.modes)
    print(f"\n{args.dataset} L={args.lookback} H={args.horizon} head={args.head} patch_len={args.patch_len} "
          f"instance_norm={args.instance_norm} amp={args.amp}. "
          "Base chosen per mode by mean validation MSE; test reported only for that base")
    print(f"{'model':12} {'base':>7} {'val MSE':>16} {'test MSE':>16} {'test MAE':>16}")
    for name in names:
        candidates = []
        for base in ([None] if name == "linear" else args.bases):
            rows = [r for r in runs if r["model"] == name and r["base"] == base]
            candidates.append((mean_std([r["val_mse"] for r in rows])[0], base, rows))
        _, base, rows = min(candidates, key=lambda c: c[0])
        entry = {"model": name, "chosen_base": base}
        for key in ("val_mse", "test_mse", "test_mae"):
            entry[key] = mean_std([r[key] for r in rows])
        if all("extrapolation_mse" in r for r in rows):
            entry["extrapolation_mse"] = {h: mean_std([r["extrapolation_mse"][h] for r in rows])
                                          for h in rows[0]["extrapolation_mse"]}
        summary.append(entry)
        cells = " ".join(f"{m:.4f}±{s:.4f}".rjust(16) for m, s in (entry["val_mse"], entry["test_mse"], entry["test_mae"]))
        extra = "".join(f"  H={h}: {m:.4f}±{s:.4f}" for h, (m, s) in entry.get("extrapolation_mse", {}).items())
        print(f"{name:12} {str(base if base is None else f'{base:g}'):>7} {cells}{extra}")

    environment = {"python": platform.python_version(), "torch": torch.__version__, "device": str(device),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                   "tf32": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "model_config": model_cfg, "data": counts, "environment": environment,
               "summary": summary, "runs": runs}
    out.write_text(json.dumps(payload, indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
