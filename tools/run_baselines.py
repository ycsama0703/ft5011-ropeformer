"""Run external baselines under the same data, split, metrics and training protocol as run_benchmark.py.

Categories (see tools/baselines.py):
  statistical  naive, seasonal_naive, window_mean          (no training)
  ml           ridge (closed form on all training windows; penalty chosen on validation)
               lightgbm (direct strategy: one model per horizon step, CPU, subsampled training windows)
  deep         dlinear, lstm
  transformer  patchtst, itransformer

Trainable deep models use the run_benchmark.py training loop unchanged (AdamW,
warmup + cosine, the dataset's step budget from benchmark_suite.py, best state
on validation, bf16 autocast by default). Each picks its learning rate from
--lrs by validation MSE, analogous to the rope_base choice of the RoPE models.
Channel-independent models see (origin, channel) windows; iTransformer sees all
channels of an origin at once ([B, L, C]). Metrics average over every test
(origin, channel) pair in both cases, so the numbers are directly comparable.

    python tools/run_baselines.py --dataset ETTh1 --horizon 96
    python tools/run_baselines.py --dataset traffic --horizon 96 --baselines lightgbm --out results/baselines/x.json
"""
import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as bl  # noqa: E402
import run_benchmark as rb  # noqa: E402
from benchmark_suite import SUITE  # noqa: E402
from grouped_rope.data import WindowedSeries  # noqa: E402

STATISTICAL = ("naive", "seasonal_naive", "window_mean")
ML = ("ridge", "lightgbm")
DEEP = ("dlinear", "lstm", "patchtst", "itransformer")
CATEGORY = {**{n: "statistical" for n in STATISTICAL}, **{n: "ml" for n in ML}, "dlinear": "deep", "lstm": "deep",
            "patchtst": "transformer", "itransformer": "transformer"}


class MultivariateView:
    """Origin-indexed windows with all channels: x [B, L, C], y [B, H, C]."""

    def __init__(self, data):
        self.series, self.lookback = data.series, data.lookback

    def batch(self, origins, horizon):
        origins = origins.to(self.series.device)
        window = self.series[origins[:, None] + torch.arange(-self.lookback, horizon, device=origins.device)]
        return window[:, : self.lookback], window[:, self.lookback:]


class RidgeModel(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, x):
        z = x[..., 0].double()
        return (z @ self.weight[:-1] + self.weight[-1]).float()[..., None]


def run_ridge(data, splits, args):
    train_s, val_s, test_s = splits
    d = args.lookback + 1
    xtx = torch.zeros(d, d, dtype=torch.float64, device=data.series.device)
    xty = torch.zeros(d, args.horizon, dtype=torch.float64, device=data.series.device)
    for i in range(0, train_s.shape[0], 65536):
        x, y = data.batch(train_s[i:i + 65536], args.horizon)
        features = torch.cat([x[..., 0].double(), torch.ones_like(x[:, :1, 0], dtype=torch.float64)], 1)
        xtx += features.T @ features
        xty += features.T @ y[..., 0].double()
    scale = torch.diagonal(xtx)[:-1].mean()
    best = None
    for lam in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        penalty = torch.full((d,), lam * scale.item(), dtype=torch.float64, device=xtx.device)
        penalty[-1] = 0.0  # no penalty on the intercept
        model = RidgeModel(torch.linalg.solve(xtx + torch.diag(penalty), xty))
        val_mse, _ = rb.evaluate(model, data, val_s, args.horizon, args)
        if best is None or val_mse < best[0]:
            best = (val_mse, lam, model)
    test_mse, test_mae = rb.evaluate(best[2], data, test_s, args.horizon, args)
    return {"val_mse": best[0], "test_mse": test_mse, "test_mae": test_mae, "hyper": {"lambda": best[1]},
            "parameters": d * args.horizon}


def lgb_features(x, lags):
    """Per-window features relative to the last value (tree models cannot extrapolate levels)."""
    z = x[..., 0].double()
    last = z[:, -1:]
    stats = torch.stack([z.mean(1) - last[:, 0], z.std(1), z.min(1).values - last[:, 0], z.max(1).values - last[:, 0]], 1)
    return torch.cat([z[:, -lags:] - last, stats], 1).float().cpu().numpy(), last


def run_lightgbm(data, splits, args, seed, train_windows=50000, val_windows=10000, lags=96):
    import lightgbm as lgb
    train_s, val_s, test_s = splits
    g = torch.Generator().manual_seed(seed)
    tr = train_s[torch.randperm(train_s.shape[0], generator=g)[:train_windows]]
    va = val_s[torch.randperm(val_s.shape[0], generator=g)[:val_windows]]
    (xtr, ytr), (xva, yva) = data.batch(tr, args.horizon), data.batch(va, args.horizon)
    ftr, ltr = lgb_features(xtr, lags)
    fva, lva = lgb_features(xva, lags)
    rtr = (ytr[..., 0].double() - ltr).cpu().numpy()
    rva = (yva[..., 0].double() - lva).cpu().numpy()
    models, val_sq = [], 0.0
    for h in range(args.horizon):
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=args.lgb_jobs)
        m.fit(ftr, rtr[:, h], eval_set=[(fva, rva[:, h])], callbacks=[lgb.early_stopping(20, verbose=False)])
        models.append(m.booster_)
        val_sq += ((m.predict(fva) - rva[:, h]) ** 2).mean()
    sq = ab = 0.0
    for i in range(0, test_s.shape[0], 200000):
        x, y = data.batch(test_s[i:i + 200000], args.horizon)
        f, last = lgb_features(x, lags)
        pred = torch.stack([torch.from_numpy(m.predict(f, num_threads=args.lgb_jobs)) for m in models], 1)
        error = pred.to(last.device) + last - y[..., 0].double()
        sq += error.square().mean(1).sum().item()
        ab += error.abs().mean(1).sum().item()
    n = test_s.shape[0]
    return {"val_mse": val_sq / args.horizon, "test_mse": sq / n, "test_mae": ab / n,
            "hyper": {"train_windows": tr.shape[0], "val_windows": va.shape[0], "lags": lags,
                      "mean_trees": sum(m.num_trees() for m in models) / len(models)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(SUITE))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--lookback", type=int, default=336)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--baselines", nargs="+", choices=[*STATISTICAL, *ML, *DEEP],
                        default=[*STATISTICAL, "ridge", *DEEP])
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="Default: the dataset's benchmark_suite.py budget")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-samples", type=int, default=32768)
    parser.add_argument("--eval-chunk", type=int, default=1024)
    parser.add_argument("--amp", choices=["none", "bf16"], default="bf16")
    parser.add_argument("--lgb-jobs", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    steps, every = SUITE[args.dataset]
    args.steps = args.steps or steps
    args.eval_every = args.eval_every or every
    args.instance_norm, args.eval_horizons = False, []  # shared run_benchmark.py options, off for baselines
    out = args.out or Path("results/baselines") / f"{args.dataset}_L{args.lookback}_H{args.horizon}.json"
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = WindowedSeries.from_csv(args.data_dir / f"{args.dataset}.csv", args.dataset, args.lookback).to(device)
    train_s, test_s = data.samples("train", args.horizon), data.samples("test", args.horizon)
    val_all = data.samples("val", args.horizon)
    pick = torch.randperm(val_all.shape[0], generator=torch.Generator().manual_seed(0))[:args.val_samples]
    val_s = val_all[pick.sort().values]  # same subset rule as run_benchmark.py
    ci_splits = (train_s, val_s, test_s)
    origins = {s: data.origins(s, args.horizon) for s in ("train", "val", "test")}
    n_val_origins = min(origins["val"].shape[0], max(512, args.val_samples // data.channels))
    pick = torch.randperm(origins["val"].shape[0], generator=torch.Generator().manual_seed(0))[:n_val_origins]
    mv_splits = (origins["train"], origins["val"][pick.sort().values], origins["test"])
    print(f"{args.dataset} H={args.horizon}: {test_s.shape[0]} test windows, {data.channels} channels", flush=True)

    exact = copy.copy(args)
    exact.amp = "none"  # statistical and ridge predictions in float32/float64
    runs = []
    for name in args.baselines:
        start = time.perf_counter()
        if name in STATISTICAL:
            model = bl.statistical(name, args.dataset, args.horizon)
            if model is None:
                print(f"{name}: not defined for {args.dataset}", flush=True)
                continue
            val_mse, _ = rb.evaluate(model, data, val_s, args.horizon, exact)
            test_mse, test_mae = rb.evaluate(model, data, test_s, args.horizon, exact)
            candidates = [{"val_mse": val_mse, "test_mse": test_mse, "test_mae": test_mae, "parameters": 0}]
        elif name == "ridge":
            candidates = [run_ridge(data, ci_splits, exact)]
        elif name == "lightgbm":
            candidates = [run_lightgbm(data, ci_splits, args, args.seed)]
        else:
            candidates = []
            for lr in args.lrs:
                torch.manual_seed(args.seed)
                model, multivariate = bl.build(name, args.dataset, args.lookback, args.horizon)
                run_args = copy.copy(args)
                run_args.lr = lr
                if multivariate:
                    run_args.batch_size = bl.ITRANSFORMER[args.dataset][2]
                    run_args.eval_chunk = max(4, 16384 // data.channels)
                    view, splits = MultivariateView(data), mv_splits
                else:
                    view, splits = data, ci_splits
                result = rb.train(model.to(device), view, (*splits, None), run_args, args.seed)
                result["hyper"] = {"lr": lr, "batch_size": run_args.batch_size}
                candidates.append(result)
                print(f"  {name} lr={lr:g}: val={result['val_mse']:.4f} test={result['test_mse']:.4f} "
                      f"best@{result['best_step']} ({result['train_seconds']}s)", flush=True)
        best = min(candidates, key=lambda r: r["val_mse"])
        seconds = round(time.perf_counter() - start, 1)
        runs.append({"model": name, "category": CATEGORY[name], "seed": args.seed, "seconds": seconds,
                     "chosen": best, "candidates": candidates})
        print(f"{name:15} val={best['val_mse']:.4f} test={best['test_mse']:.4f} mae={best['test_mae']:.4f} "
              f"{best.get('hyper', '')} ({seconds}s)", flush=True)

    environment = {"python": platform.python_version(), "torch": torch.__version__, "device": str(device),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()}
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "data": {"channels": data.channels, "test_windows": test_s.shape[0], "borders": data.borders},
               "environment": environment, "runs": runs}
    out.write_text(json.dumps(payload, indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
