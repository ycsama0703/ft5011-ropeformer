"""Run the benchmark suite: one run_benchmark.py call per (dataset, horizon), sequentially.

Training budgets are fixed per dataset and identical for every RoPE mode and the
linear reference; they are recorded in each output file. Finished outputs are
skipped, so an interrupted suite can be resumed by re-running the same command.

    python tools/benchmark_suite.py --datasets ETTh1 electricity --amp bf16
    python tools/benchmark_suite.py --list
"""
import argparse
import subprocess
import sys
from pathlib import Path

# dataset: (training steps, validation interval). Batch size 256 everywhere.
SUITE = {
    "ETTh1": (2000, 200),
    "ETTh2": (2000, 200),
    "ETTm1": (4000, 250),
    "ETTm2": (4000, 250),
    "weather": (6000, 500),
    "exchange_rate": (2000, 200),
    "electricity": (8000, 500),
    "traffic": (8000, 500),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=sorted(SUITE), default=list(SUITE))
    parser.add_argument("--horizons", type=int, nargs="+", default=[96])
    parser.add_argument("--lookback", type=int, default=336)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--bases", type=float, nargs="+", default=[1000.0, 10000.0])
    parser.add_argument("--amp", choices=["none", "bf16"], default="bf16")
    parser.add_argument("--instance-norm", action="store_true")
    parser.add_argument("--patch-len", type=int, default=None)
    parser.add_argument("--scale-lr", type=float, default=None, help="Passed to run_benchmark.py")
    parser.add_argument("--tag", default="", help="Suffix for output file names, e.g. _b10-1000_slr1e-2")
    parser.add_argument("--out-dir", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--list", action="store_true", help="Print the planned commands and exit")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.amp}{'_norm' if args.instance_norm else ''}{f'_p{args.patch_len}' if args.patch_len else ''}{args.tag}"
    jobs = []
    for dataset in args.datasets:
        steps, every = SUITE[dataset]
        for horizon in args.horizons:
            out = args.out_dir / f"{dataset}_L{args.lookback}_H{horizon}_{tag}.json"
            cmd = [sys.executable, "tools/run_benchmark.py", "--dataset", dataset, "--lookback", str(args.lookback),
                   "--horizon", str(horizon), "--steps", str(steps), "--eval-every", str(every),
                   "--seeds", *map(str, args.seeds), "--bases", *map(str, args.bases), "--amp", args.amp,
                   "--out", str(out)]
            if args.instance_norm:
                cmd.append("--instance-norm")
            if args.patch_len:
                cmd += ["--patch-len", str(args.patch_len)]
            if args.scale_lr:
                cmd += ["--scale-lr", str(args.scale_lr)]
            jobs.append((out, cmd))
    for out, cmd in jobs:
        if args.list:
            print(("[done] " if out.exists() else "") + " ".join(cmd))
            continue
        if out.exists():
            print(f"skip {out} (exists)", flush=True)
            continue
        print(f"run {out}", flush=True)
        with open(out.with_suffix(".log"), "w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"{'ok' if code == 0 else f'FAILED ({code})'} {out}", flush=True)


if __name__ == "__main__":
    main()
