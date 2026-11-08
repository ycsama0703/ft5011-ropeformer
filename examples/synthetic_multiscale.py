"""Controlled two-period synthetic check of RoPE frequency parameterizations.

Tasks range from exactly periodic (fixed) to drifting, per-sequence and
switching cycles that a fixed linear filter cannot extrapolate exactly.
Question: does grouped scaling still help once every mode gets the same
rope_base search? Each mode picks its base by mean validation MSE; test MSE is
reported only for that choice. A linear lookback->horizon model is included
as a backbone sanity reference, not as a RoPE variant.

With --head horizon_tokens and --eval-horizons, test windows are also scored
beyond the training horizon (the model is called with a longer horizon); this
measures horizon extrapolation. Log-scale trajectories are recorded at every
validation check so it is visible whether the frequency scales actually move.

Mechanism experiment only; synthetic results are not benchmark evidence.
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
from grouped_rope.model import ROPE_MODES

MODES = ("standard", "head", "shared_group", "grouped")


TASKS = ("fixed", "random_period", "quasi_periodic", "regime_switch")


def uniform(shape, low, high, generator):
    return low + (high - low) * torch.rand(shape, generator=generator)


def make_data(task, count, lookback, horizon, periods, noise, generator):
    """Fast + slow components built from per-step frequencies and amplitudes.

    fixed:          both periods fixed (exactly linearly predictable).
    random_period:  each sequence draws its own fast/slow periods around the nominal ones.
    quasi_periodic: log-frequency and log-amplitude follow slow random walks (cycle timing drifts).
    regime_switch:  the fast period jumps to a new value at a random time in the recent history.
    """
    length = lookback + horizon
    fast, slow = periods
    shape = (count, length, 2)
    nominal = torch.tensor([fast, slow], dtype=torch.float32)
    log_amp = torch.log(uniform((count, 1, 2), 0.5, 1.5, generator)).expand(shape).clone()
    if task == "fixed":
        period = nominal.expand(shape).clone()
    elif task == "random_period":
        period = nominal * uniform((count, 1, 2), 2 / 3, 4 / 3, generator)
        period = period.expand(shape).clone()
    elif task == "quasi_periodic":
        drift = (0.01 * torch.randn(shape, generator=generator)).cumsum(1)
        period = nominal * torch.exp(drift - drift[:, lookback - 1:lookback])
        log_amp = log_amp + (0.02 * torch.randn(shape, generator=generator)).cumsum(1)
    elif task == "regime_switch":
        period = nominal.expand(shape).clone()
        before = uniform((count,), fast / 2, fast * 1.5, generator)
        after = uniform((count,), fast / 2, fast * 1.5, generator)
        after = torch.where((after - before).abs() < fast / 4, before + fast / 2 * torch.sign(after - before + 1e-6), after)
        switch = torch.randint(lookback - 4 * int(fast), lookback - int(fast), (count,), generator=generator)
        steps = torch.arange(length)[None, :]
        period[..., 0] = torch.where(steps < switch[:, None], before[:, None], after[:, None])
    else:
        raise ValueError(f"Unknown task: {task}")
    phase = uniform((count, 1, 2), 0, 2 * math.pi, generator) + (2 * math.pi / period).cumsum(1)
    signal = (log_amp.exp() * torch.sin(phase)).sum(-1, keepdim=True)
    signal = signal + noise * torch.randn(count, length, 1, generator=generator)
    return signal[:, :lookback], signal[:, lookback:]


class LinearForecaster(nn.Module):
    def __init__(self, lookback, horizon):
        super().__init__()
        self.proj = nn.Linear(lookback, horizon)

    def forward(self, x):
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


@torch.no_grad()
def step_errors(model, x, y, horizon=None, chunk=256):
    """Per-step mean squared and absolute errors over all windows (no drop_last)."""
    model.eval()
    kwargs = {} if horizon is None else {"horizon": horizon}
    sq, ab = 0.0, 0.0
    for i in range(0, x.shape[0], chunk):
        error = model(x[i:i + chunk], **kwargs) - y[i:i + chunk]
        sq = sq + error.square().sum(dim=(0, 2))
        ab = ab + error.abs().sum(dim=(0, 2))
    return sq / (y.shape[0] * y.shape[2]), ab / (y.shape[0] * y.shape[2])


def evaluate(model, x, y):
    sq, ab = step_errors(model, x, y)
    return sq.mean().item(), ab.mean().item()


def train(model, data, args, seed, device):
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = data
    scale_params = [p for n, p in model.named_parameters() if n.endswith("log_scales")]
    other_params = [p for n, p in model.named_parameters() if not n.endswith("log_scales")]
    groups = [{"params": other_params, "weight_decay": args.weight_decay}]
    if scale_params:
        groups.append({"params": scale_params, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    warmup = min(args.warmup_steps, args.steps)
    decay = max(1, args.steps - warmup)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: (i + 1) / warmup if i < warmup else 0.5 * (1 + math.cos(math.pi * (i - warmup) / decay)))
    trajectory = []
    batches = torch.Generator().manual_seed(seed + 1000)  # identical batch order across modes
    best = (float("inf"), None, -1)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(train_x.shape[0], (args.batch_size,), generator=batches).to(device)
        loss = (model(train_x[idx]) - train_y[idx]).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            if scale_params:
                trajectory.append({"step": step, "log_scales": [p.detach().cpu().tolist() for p in scale_params]})
            val_mse, _ = evaluate(model, val_x, val_y)
            if val_mse < best[0]:
                best = (val_mse, {k: v.detach().clone() for k, v in model.state_dict().items()}, step)
    seconds = time.perf_counter() - start
    model.load_state_dict(best[1])
    horizon = val_y.shape[1]
    test_mse, test_mae = evaluate(model, test_x, test_y[:, :horizon])
    result = {"val_mse": best[0], "best_step": best[2], "test_mse": test_mse, "test_mae": test_mae,
              "seconds": round(seconds, 2)}
    if isinstance(model, GroupedRoPEForecaster) and model.cfg.forecast_head == "horizon_tokens":
        per_step, _ = step_errors(model, test_x, test_y, horizon=test_y.shape[1])
        result["test_per_step_mse"] = per_step.cpu().tolist()
        result["extrapolation_mse"] = {str(h): per_step[:h].mean().item() for h in args.eval_horizons}
    if trajectory:
        result["scale_trajectory"] = trajectory
    if isinstance(model, GroupedRoPEForecaster):
        result["periods"] = [info["periods"].cpu().tolist() for info in model.rope_diagnostics()]
        result["scales"] = [info["scales"].cpu().tolist() for info in model.rope_diagnostics()]
    return result


def mean_std(values):
    t = torch.tensor(values, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if len(values) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, default="fixed")
    parser.add_argument("--periods", type=float, nargs=2, default=[24.0, 168.0], metavar=("FAST", "SLOW"))
    parser.add_argument("--lookback", type=int, default=336)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--modes", nargs="+", choices=ROPE_MODES, default=list(MODES),
                        help="'none' (NoPE) is a diagnostic, not part of the main comparison")
    parser.add_argument("--head", choices=["horizon_tokens", "pooled"], default="horizon_tokens")
    parser.add_argument("--eval-horizons", type=int, nargs="*", default=[],
                        help="Extra test horizons (horizon_tokens only); may exceed --horizon")
    parser.add_argument("--bases", type=float, nargs="+", default=[1000.0, 10000.0])
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--data-seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup, then cosine decay to 0")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--no-linear", action="store_true", help="Skip the linear reference model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("results/synthetic_multiscale.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.eval_horizons and args.head != "horizon_tokens":
        parser.error("--eval-horizons requires --head horizon_tokens")
    splits = ((args.n_train, args.horizon), (args.n_val, args.horizon),
              (args.n_test, max([args.horizon, *args.eval_horizons])))
    # One generator per split, so extra test horizons do not change train/validation data.
    data = tuple(
        tuple(t.to(device) for t in make_data(args.task, n, args.lookback, h, args.periods, args.noise,
                                              torch.Generator().manual_seed(args.data_seed + offset)))
        for offset, (n, h) in enumerate(splits)
    )
    model_cfg = dict(input_dim=1, horizon=args.horizon, d_model=64, n_heads=4, num_layers=2,
                     d_ff=128, dropout=0.0, n_freq_groups=args.groups, forecast_head=args.head)

    runs = []
    for seed in args.seeds:
        if not args.no_linear:
            torch.manual_seed(seed)
            result = train(LinearForecaster(args.lookback, args.horizon).to(device), data, args, seed, device)
            runs.append({"model": "linear", "base": None, "seed": seed, **result})
            print(f"seed={seed} linear                     val={result['val_mse']:.5f} test={result['test_mse']:.5f}")
        for mode in args.modes:
            for base in args.bases:
                torch.manual_seed(seed)  # identical backbone init across modes (log-scales start at 0)
                model = GroupedRoPEForecaster(rope_mode=mode, rope_base=base, **model_cfg).to(device)
                result = train(model, data, args, seed, device)
                runs.append({"model": mode, "base": base, "seed": seed, **result})
                print(f"seed={seed} {mode:12} base={base:<7g}  val={result['val_mse']:.5f} "
                      f"test={result['test_mse']:.5f} ({result['seconds']}s)", flush=True)

    summary = []
    names = (["linear"] if not args.no_linear else []) + list(args.modes)
    print(f"\ntask={args.task}, head={args.head}. Per mode: base chosen by mean validation MSE; test reported only for that base")
    print(f"{'model':12} {'base':>7} {'val MSE':>18} {'test MSE':>18} {'test MAE':>18}")
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
        cells = " ".join(f"{m:.5f}±{s:.5f}".rjust(18) for m, s in (entry["val_mse"], entry["test_mse"], entry["test_mae"]))
        extra = "".join(f"  H={h}: {m:.5f}±{s:.5f}" for h, (m, s) in entry.get("extrapolation_mse", {}).items())
        print(f"{name:12} {str(base if base is None else f'{base:g}'):>7} {cells}{extra}")
    print("Synthetic mechanism check only; not evidence about real forecasting benchmarks.")

    environment = {"python": platform.python_version(), "torch": torch.__version__, "device": str(device),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "model_config": model_cfg, "environment": environment, "summary": summary, "runs": runs}
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
