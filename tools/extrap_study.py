"""Horizon-extrapolation study built on run_benchmark.train (same data, protocol and budget).

For each (mode, base, seed): train at --horizon, keep the validation-best state, then score
the test windows that fit max(--eval-horizons) at every step with
  learned  - the trained model as is
  reset    - the same weights with every learnable log-scale set to 0 (standard RoPE frequencies)
Learnable modes therefore get a paired "were the learned scales responsible?" ablation.
--direct-horizons additionally trains linear and Transformer models directly at those horizons
(reference for what extrapolation should be compared against). Checkpoints go to --ckpt-dir.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmark as rb  # noqa: E402
from grouped_rope import GroupedRoPEForecaster  # noqa: E402
from grouped_rope.data import WindowedSeries  # noqa: E402


def zero_scales(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("log_scales"):
                p.zero_()


def build(spec, base, horizon, cfg):
    """spec is a rope_mode, or 'fixed_head:s1,s2,...' for frozen per-head scales."""
    mode, _, init = spec.partition(":")
    extra = {"head_scale_init": tuple(float(v) for v in init.split(","))} if init else {}
    return GroupedRoPEForecaster(rope_mode=mode, rope_base=base, horizon=horizon, **extra, **cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--eval-horizons", type=int, nargs="+", default=[192, 336])
    p.add_argument("--direct-horizons", type=int, nargs="*", default=[])
    p.add_argument("--direct-modes", nargs="*", default=["standard", "head"])
    p.add_argument("--modes", nargs="+", default=["standard", "head", "grouped"])
    p.add_argument("--bases", type=float, nargs="+", default=[1000.0, 10000.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--groups", type=int, default=2)
    p.add_argument("--patch-len", type=int, default=None)
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
    p.add_argument("--ckpt-dir", type=Path, default=Path("results/ckpt"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_p{args.patch_len or 1}"

    data = WindowedSeries.from_csv(args.data_dir / f"{args.dataset}.csv", args.dataset, args.lookback).to(device)
    longest = max(args.eval_horizons)

    def splits_for(horizon):
        val_all = data.samples("val", horizon)
        if args.val_samples < val_all.shape[0]:
            pick = torch.randperm(val_all.shape[0], generator=torch.Generator().manual_seed(0))[:args.val_samples]
            val_all = val_all[pick.sort().values]
        # every model is tested on the same windows: those that fit the longest horizon
        return data.samples("train", horizon), val_all, data.samples("test", longest), None

    cfg = dict(input_dim=1, d_model=args.d_model, n_heads=args.n_heads, num_layers=args.layers, d_ff=args.d_ff,
               dropout=args.dropout, n_freq_groups=args.groups, forecast_head="horizon_tokens",
               patch_len=args.patch_len)
    runs = []
    out_payload = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "runs": runs}

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_payload, indent=1))

    # 1) train at --horizon, extrapolate, paired scale-reset ablation
    train_args = argparse.Namespace(**{**vars(args), "eval_horizons": []})
    splits = splits_for(args.horizon)
    for seed in args.seeds:
        for mode in args.modes:
            for base in args.bases:
                torch.manual_seed(seed)
                model = build(mode, base, args.horizon, cfg).to(device)
                r = rb.train(model, data, splits, train_args, seed)
                r.pop("test_mse"), r.pop("test_mae")  # recomputed below on the common long-horizon windows
                learned, _ = rb.step_errors(model, data, splits[2], longest, args, call_horizon=longest)
                row = {"kind": "extrapolate", "model": mode, "base": base, "seed": seed, **r,
                       "per_step_learned": learned.cpu().tolist()}
                torch.save(model.state_dict(), args.ckpt_dir / f"{tag}_{mode.replace(':', '-')}_b{base:g}_s{seed}.pt")
                if any(n.endswith("log_scales") for n, _ in model.named_parameters()):
                    zero_scales(model)
                    reset, _ = rb.step_errors(model, data, splits[2], longest, args, call_horizon=longest)
                    row["per_step_reset"] = reset.cpu().tolist()
                runs.append(row)
                msg = " ".join(f"H{h}={learned[:h].mean().item():.4f}" for h in [args.horizon, *args.eval_horizons])
                if "per_step_reset" in row:
                    msg += f" | reset H{longest}={sum(row['per_step_reset']) / longest:.4f}"
                print(f"seed={seed} {mode:12} base={base:<7g} best@{r['best_step']:<5} {msg}", flush=True)
                save()

    # 2) direct-training references at longer horizons (same test windows)
    for horizon in args.direct_horizons:
        d_splits = splits_for(horizon)
        d_args = argparse.Namespace(**{**vars(args), "horizon": horizon, "eval_horizons": []})
        for seed in args.seeds:
            torch.manual_seed(seed)
            lin = rb.LinearForecaster(args.lookback, horizon).to(device)
            r = rb.train(lin, data, d_splits, d_args, seed)
            runs.append({"kind": "direct", "model": "linear", "base": None, "seed": seed, "horizon": horizon,
                         "test_mse": r["test_mse"], "test_mae": r["test_mae"], "best_step": r["best_step"]})
            print(f"direct H={horizon} seed={seed} linear                    test={r['test_mse']:.4f}", flush=True)
            for mode in args.direct_modes:
                for base in args.bases:
                    torch.manual_seed(seed)
                    model = build(mode, base, horizon, cfg).to(device)
                    r = rb.train(model, data, d_splits, d_args, seed)
                    runs.append({"kind": "direct", "model": mode, "base": base, "seed": seed, "horizon": horizon,
                                 "test_mse": r["test_mse"], "test_mae": r["test_mae"], "best_step": r["best_step"],
                                 "val_mse": r["val_mse"]})
                    print(f"direct H={horizon} seed={seed} {mode:12} base={base:<7g} test={r['test_mse']:.4f}", flush=True)
                    save()
    save()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
