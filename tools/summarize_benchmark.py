"""Merge run_benchmark.py outputs (e.g. one file per seed) and redo per-mode base selection.

Base is chosen per mode by mean validation MSE over all merged seeds; test is
reported only for that base. Also prints paired per-seed differences against
a reference mode and the range of learned frequency scales.

    python tools/summarize_benchmark.py results/benchmark/electricity_L336_H96_s*.json
"""
import argparse
import json
import math
from pathlib import Path


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return mean, std


def flatten(values):
    if isinstance(values, list):
        for v in values:
            yield from flatten(v)
    else:
        yield values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--reference", default="standard", help="Mode for paired per-seed differences")
    args = parser.parse_args()
    payloads = [json.loads(f.read_text()) for f in args.files]
    keys = {(p["args"]["dataset"], p["args"]["lookback"], p["args"]["horizon"], p["args"]["head"],
             p["args"]["steps"], p["args"]["batch_size"], p["args"]["lr"],
             p["args"].get("instance_norm", False), p["args"].get("amp", "none"),
             p["args"].get("patch_len"), p["args"].get("scale_lr")) for p in payloads}
    if len(keys) != 1:
        raise SystemExit(f"Files use different settings: {keys}")
    runs = [r for p in payloads for r in p["runs"]]
    seen = set()
    for r in runs:
        key = (r["model"], r["base"], r["seed"])
        if key in seen:
            raise SystemExit(f"Duplicate run {key}")
        seen.add(key)
    a = payloads[0]["args"]
    names = list(dict.fromkeys(r["model"] for r in runs))
    print(f"{a['dataset']} L={a['lookback']} H={a['horizon']} head={a['head']} steps={a['steps']}; "
          f"seeds={sorted({r['seed'] for r in runs})}")
    print(f"{'model':12} {'base':>7} {'val MSE':>16} {'test MSE':>16} {'test MAE':>16}  extrapolation MSE")
    chosen = {}
    for name in names:
        candidates = []
        for base in sorted({r["base"] for r in runs if r["model"] == name}, key=lambda b: -1 if b is None else b):
            rows = sorted([r for r in runs if r["model"] == name and r["base"] == base], key=lambda r: r["seed"])
            candidates.append((mean_std([r["val_mse"] for r in rows])[0], base, rows))
        _, base, rows = min(candidates, key=lambda c: c[0])
        chosen[name] = rows
        cells = " ".join(f"{m:.4f}±{s:.4f}".rjust(16)
                         for m, s in (mean_std([r[k] for r in rows]) for k in ("val_mse", "test_mse", "test_mae")))
        extra = ""
        if all("extrapolation_mse" in r for r in rows):
            extra = "  ".join(f"H={h}: {mean_std([r['extrapolation_mse'][h] for r in rows])[0]:.4f}"
                              for h in rows[0]["extrapolation_mse"])
        print(f"{name:12} {'-' if base is None else f'{base:g}':>7} {cells}  {extra}")

    if args.reference in chosen:
        print(f"\nPaired test MSE difference vs {args.reference} (negative = better), per seed:")
        ref = {r["seed"]: r["test_mse"] for r in chosen[args.reference]}
        for name, rows in chosen.items():
            if name == args.reference:
                continue
            diffs = [r["test_mse"] - ref[r["seed"]] for r in rows if r["seed"] in ref]
            wins = sum(d < 0 for d in diffs)
            print(f"  {name:12} " + " ".join(f"{d:+.4f}" for d in diffs) + f"   wins {wins}/{len(diffs)}")

    print("\nLearned frequency scales exp(log_scale), chosen base, all seeds:")
    for name, rows in chosen.items():
        values = [math.exp(v) for r in rows for v in flatten([t for t in r.get("scale_trajectory", [{}])[-1:]
                                                              for t in [t.get("log_scales", [])]])]
        if values:
            print(f"  {name:12} min {min(values):.2f}  max {max(values):.2f}")
        best = [r["best_step"] for r in rows]
        print(f"  {name:12} best_step {best}")


if __name__ == "__main__":
    main()
