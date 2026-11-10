"""Run the experiment grid with several processes per GPU. Finished runs are skipped.

Example:
    python study/sweep.py --datasets ETTh1 ETTh2 --gpus 0 --workers 8
"""

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grouped_rope.study.data import DATASETS  # noqa: E402
MODES = ("standard", "head", "shared_group", "grouped")


def run_specs(args):
    specs = []  # seed-major order: every dataset gets seed 0 before any dataset gets seed 1
    for seed, (dataset, horizon) in itertools.product(args.seeds, itertools.product(args.datasets, args.horizons)):
        base_dir = args.runs / dataset / f"L{args.lookback}_H{horizon}"
        common = ["--dataset", dataset, "--data-dir", str(args.data_dir), "--lookback", str(args.lookback),
                  "--horizon", str(horizon), "--steps", str(args.steps)]
        for base in args.bases:
            for mode in args.modes:
                # "grouped4"/"grouped8" = grouped with that many frequency groups
                groups = int(mode[len("grouped"):]) if mode.startswith("grouped") and mode != "grouped" else 2
                rope_mode = "grouped" if mode.startswith("grouped") else mode
                prefix = "ptf" if args.model == "patch_transformer" else "tf"
                tag = (f"{prefix}-{mode}-b{base:g}-{args.pooling}" + ("" if args.scale_lr == 1e-2 else f"-slr{args.scale_lr:g}")
                       + ("" if args.steps == 3000 else f"-st{args.steps}"))
                specs.append((base_dir / tag / f"s{seed}", common + [
                    "--model", args.model, "--rope-mode", rope_mode, "--groups", str(groups),
                    "--rope-base", str(base), "--pooling", args.pooling, "--seed", str(seed),
                    "--scale-lr", str(args.scale_lr)]))
        for kind in args.baselines:
            if kind in ("naive", "seasonal_naive") and seed != args.seeds[0]:
                continue  # deterministic baselines need one run
            if kind == "seasonal_naive" and DATASETS.get(dataset, {}).get("period") is None:
                continue  # no seasonal period (e.g. exchange rates)
            kdir = kind + ("" if args.steps == 3000 or kind in ("naive", "seasonal_naive") else f"-st{args.steps}")
            specs.append((base_dir / kdir / f"s{seed}", common + ["--model", kind, "--seed", str(seed)]))
    return specs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--horizons", nargs="+", type=int, default=[96, 336])
    p.add_argument("--lookback", type=int, default=336)
    p.add_argument("--modes", nargs="+", default=list(MODES), help="standard head shared_group grouped grouped4")
    p.add_argument("--bases", nargs="+", type=float, default=[100.0, 10000.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--baselines", nargs="*", default=["naive", "seasonal_naive", "linear", "lstm"])
    p.add_argument("--pooling", default="last")
    p.add_argument("--scale-lr", type=float, default=1e-2, help="learning rate of the RoPE log-scales")
    p.add_argument("--model", choices=["transformer", "patch_transformer"], default="transformer")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    todo = [(d, a) for d, a in run_specs(args) if not (d / "result.json").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    if args.dry_run:
        for d, a in todo:
            print(d)
        return

    def launch(job):
        i, (out, extra) = job
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus[i % len(args.gpus)], OMP_NUM_THREADS="2")
        cmd = [sys.executable, str(ROOT / "study" / "train.py"), *extra, "--out", str(out)]
        with open(out / "log.txt", "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        tail = (out / "log.txt").read_text().strip().splitlines()[-1:] or [""]
        print(("OK   " if code == 0 else f"FAIL({code}) ") + tail[0], flush=True)
        return code

    with ThreadPoolExecutor(args.workers) as pool:
        codes = list(pool.map(launch, enumerate(todo)))
    print(f"done: {codes.count(0)} ok, {len(codes) - codes.count(0)} failed", flush=True)


if __name__ == "__main__":
    main()
