"""Train and evaluate one model on one benchmark; writes result.json, losses and a checkpoint.

Selection uses the validation split only (early stopping on validation MSE). The test
split is evaluated once, with the best validation checkpoint. Metrics are on the
train-standardised scale, the convention of the long-term forecasting literature.
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cuBLAS
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grouped_rope.study.models import build_model, forecast_channels, rope_core  # noqa: E402
from grouped_rope.study.data import load_dataset  # noqa: E402

LEARNED = ("transformer", "patch_transformer", "linear", "lstm")


def subset(n_windows: int, n_channels: int, cap_sequences: int, device) -> torch.Tensor:
    """Evenly spaced window indices so that windows x channels <= cap (deterministic)."""
    keep = max(1, min(n_windows, cap_sequences // n_channels))
    return torch.linspace(0, n_windows - 1, keep, device=device).round().long().unique()


def autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, data, split, index, max_sequences=8192, amp=False):
    """Per-window MSE and MAE (averaged over horizon and channels)."""
    model.eval()
    chunk = max(1, max_sequences // data.n_channels)
    mse, mae = [], []
    for start in range(0, len(index), chunk):
        x, y = data.windows(split, index[start : start + chunk])
        with autocast(x.device, amp):
            pred = forecast_channels(model, x)
        err = pred.float() - y
        mse.append(err.square().mean(dim=(1, 2)))
        mae.append(err.abs().mean(dim=(1, 2)))
    return torch.cat(mse).double().cpu().numpy(), torch.cat(mae).double().cpu().numpy()


def scale_snapshot(model):
    core = rope_core(model)
    if core is None:
        return None
    return [b.attention.rotary.log_scales.detach().float().cpu().tolist() for b in core.blocks]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--file", default=None, help="CSV file name if not the default for --dataset")
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--model", choices=["transformer", "patch_transformer", "linear", "lstm", "naive", "seasonal_naive"],
                   default="transformer")
    p.add_argument("--patch-len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--rope-mode", default="grouped")
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--groups", type=int, default=2)
    p.add_argument("--pooling", choices=["last", "mean"], default="last")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scale-lr", type=float, default=1e-2, help="learning rate of RoPE log-scales (all learnable modes)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--patience", type=int, default=4, help="evaluations without validation improvement")
    p.add_argument("--val-cap", type=int, default=25_000, help="max validation sequences used for early stopping")
    p.add_argument("--train-eval-cap", type=int, default=50_000, help="max training sequences used to report train metrics")
    p.add_argument("--amp", type=int, default=1, help="bf16 autocast on CUDA (1) or full fp32 (0)")
    p.add_argument("--deterministic", type=int, default=1,
                   help="1 = deterministic GPU kernels (bit-reproducible retraining, slower)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--from-checkpoint", type=Path, default=None,
                   help="skip training; load this model.pt and only evaluate (replication of reported numbers)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()
    data = load_dataset(args.dataset, args.data_dir, args.lookback, args.horizon, device, args.file)

    tf_kwargs = dict(
        d_model=args.d_model, n_heads=args.n_heads, num_layers=args.layers, d_ff=args.d_ff,
        dropout=args.dropout, rope_base=args.rope_base, rope_mode=args.rope_mode,
        n_freq_groups=args.groups, pooling=args.pooling,
    ) if args.model in ("transformer", "patch_transformer") else {}
    if args.model == "patch_transformer":
        tf_kwargs.update(patch_len=args.patch_len, stride=args.stride)
    if args.model == "seasonal_naive" and data.period is None:
        raise SystemExit(f"{args.dataset} has no seasonal period")
    model = build_model(args.model, args.lookback, args.horizon, data.period, **tf_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_scale = sum(p.numel() for n, p in model.named_parameters() if n.endswith("log_scales"))

    val_index = subset(data.n_windows("val"), data.n_channels, args.val_cap, device)
    history, scales, best = [], [], dict(val_mse=math.inf, step=0)
    best_state = None
    if args.from_checkpoint is not None:
        model.load_state_dict(torch.load(args.from_checkpoint, map_location=device, weights_only=True)["state_dict"])
    elif args.model in LEARNED:
        scale_params = [p for n, p in model.named_parameters() if n.endswith("log_scales")]
        other = [p for n, p in model.named_parameters() if not n.endswith("log_scales")]
        groups = [dict(params=other, lr=args.lr, weight_decay=args.weight_decay)]
        if scale_params:
            groups.append(dict(params=scale_params, lr=args.scale_lr, weight_decay=0.0))
        optimizer = torch.optim.AdamW(groups)
        base_lrs = [g["lr"] for g in optimizer.param_groups]
        gen = torch.Generator(device=device).manual_seed(args.seed)
        n_train, stale = data.n_windows("train"), 0
        scales.append(dict(step=0, log_scales=scale_snapshot(model)))
        for step in range(1, args.steps + 1):
            factor = min(1.0, step / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
            for g, lr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = lr * factor
            model.train()
            win = torch.randint(n_train, (args.batch,), device=device, generator=gen)
            ch = torch.randint(data.n_channels, (args.batch,), device=device, generator=gen)
            x, y = data.pairs("train", win, ch)
            with autocast(device, args.amp):
                pred = model(x)
            loss = (pred.float() - y).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % args.eval_every == 0 or step == args.steps:
                val_mse = float(evaluate(model, data, "val", val_index, amp=args.amp)[0].mean())
                history.append(dict(step=step, train_loss=loss.item(), val_mse=val_mse))
                scales.append(dict(step=step, log_scales=scale_snapshot(model)))
                if val_mse < best["val_mse"]:
                    best, stale = dict(val_mse=val_mse, step=step), 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
        model.load_state_dict(best_state)

    train_index = subset(data.n_windows("train"), data.n_channels, args.train_eval_cap, device)
    test_index = torch.arange(data.n_windows("test"), device=device)
    full_val = torch.arange(data.n_windows("val"), device=device)
    metrics = {}
    for split, index in (("train", train_index), ("val", full_val), ("test", test_index)):
        mse, mae = evaluate(model, data, split, index, amp=args.amp)
        metrics[split] = dict(mse=float(mse.mean()), mae=float(mae.mean()), windows=len(index))
        if split == "test":
            np.save(args.out / "test_mse.npy", mse.astype(np.float32))
            np.save(args.out / "test_mae.npy", mae.astype(np.float32))

    core = rope_core(model)
    result = dict(
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        model_config=asdict(core.cfg) if core is not None else None,
        metrics=metrics,
        best=best,
        history=history,
        scales=scales,
        final_rope=[
            dict(scales=d["scales"].tolist(), periods=d["periods"].tolist())
            for d in core.rope_diagnostics()
        ] if core is not None else None,
        n_params=n_params,
        n_scale_params=n_scale,
        n_channels=data.n_channels,
        windows={s: data.n_windows(s) for s in ("train", "val", "test")},
        seconds=time.time() - t0,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        torch=torch.__version__,
    )
    (args.out / "result.json").write_text(json.dumps(result, indent=1))
    if args.model in LEARNED and not args.no_checkpoint and args.from_checkpoint is None:
        torch.save(dict(args=result["args"], state_dict=model.state_dict()), args.out / "model.pt")
    label = (f"{args.model}/{args.rope_mode}/G{args.groups}/b{args.rope_base:g}"
             if args.model in ("transformer", "patch_transformer") else args.model)
    print(f"{args.dataset} H={args.horizon} {label}/s{args.seed}: "
          f"test MSE={metrics['test']['mse']:.4f} MAE={metrics['test']['mae']:.4f} "
          f"(val {metrics['val']['mse']:.4f}, train {metrics['train']['mse']:.4f}, best step {best['step']}, {result['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
