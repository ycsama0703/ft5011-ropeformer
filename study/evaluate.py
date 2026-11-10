"""Re-evaluate saved checkpoints and compare with the numbers stored in result.json.

    python study/evaluate.py runs/traffic/L336_H96/tf-grouped-b10000-last/s0 [more run dirs]

Uses the exact arguments recorded in each run's result.json, loads model.pt, skips
training, recomputes train/val/test metrics and reports the absolute differences.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--data-dir", type=Path, default=None, help="override the data directory recorded in the run")
    args = p.parse_args()
    worst = 0.0
    for run in args.runs:
        stored = json.loads((run / "result.json").read_text())
        a = stored["args"]
        cmd = [sys.executable, str(ROOT / "study" / "train.py")]
        for key, value in a.items():
            if key in ("out", "from_checkpoint", "no_checkpoint", "device") or value is None or value is False:
                continue
            if key == "data_dir" and args.data_dir is not None:
                value = str(args.data_dir)
            cmd += [f"--{key.replace('_', '-')}"] + ([] if value is True else [str(value)])
        with tempfile.TemporaryDirectory() as tmp:
            if (run / "model.pt").exists():
                cmd += ["--from-checkpoint", str(run / "model.pt")]
            subprocess.run(cmd + ["--out", tmp], check=True, stdout=subprocess.DEVNULL)
            new = json.loads((Path(tmp) / "result.json").read_text())["metrics"]
        for split in ("train", "val", "test"):
            for metric in ("mse", "mae"):
                diff = abs(new[split][metric] - stored["metrics"][split][metric])
                worst = max(worst, diff)
        print(f"{run}: test MSE stored {stored['metrics']['test']['mse']:.5f} "
              f"recomputed {new['test']['mse']:.5f}")
    print(f"max absolute difference over all metrics: {worst:.2e}")


if __name__ == "__main__":
    main()
