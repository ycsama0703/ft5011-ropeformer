"""Merge results from the repository's tools/ pipeline with this study into one leaderboard.

Both pipelines use the same CSVs, chronological splits, train-only standardisation, all test
(origin, channel) windows, lookback 336 and horizon 96, and report MSE on the standardised
scale. This was verified directly: tools/run_baselines.py and study/train.py give the same
number of test windows and identical naive-forecast test MSE (to 1e-6) on ETTh1, electricity
and exchange_rate. Training protocols (budgets, heads, tokenisation) differ and are listed in
the caption, so the table compares models, while RoPE-variant comparisons stay within one backbone.

    python study/merge_external.py --runs runs \
        --benchmark results/benchmark/*.json --baselines results/baselines/*.json \
        --out results/tables/leaderboard.tex
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "study"))
from aggregate import collect, label, select_bases  # noqa: E402

COMMON = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange_rate", "electricity", "traffic"]
TOOLS_MODE = {"standard": "A", "head": "B", "shared_group": "C", "grouped": "D", "none": "NoPE", "linear": "Linear"}


def ours(runs: Path) -> dict[str, dict[str, float]]:
    """Test MSE (mean over seeds 0-2) of this study's reference models, validation-selected bases."""
    df = collect(runs)
    df = df[df.seed.isin([0, 1, 2]) & (df.scale_lr.isna() | np.isclose(df.scale_lr, 1e-2))]
    short, long = select_bases(df[df.steps == 3000]), select_bases(df[df.steps == 12000])
    columns = {
        "Naive": (short, "Naive"), "RLinear": (short, "RLinear"), "LSTM": (short, "LSTM"),
        "pt-D": (short, "D"), "pt-F": (short, "F"), "patch-D 12k": (long, "PD"),
    }
    table = defaultdict(dict)
    for col, (frame, name) in columns.items():
        means = frame[frame.name == name].groupby("dataset").test_mse.mean()
        for d, v in means.items():
            table[d][col] = float(v)
    return table


def tools_benchmark(files: list[Path]) -> dict[str, dict[str, float]]:
    """tools/run_benchmark.py outputs: per (head, patch, instance-norm, steps) and mode, select base on val."""
    groups = defaultdict(list)
    for f in files:
        p = json.loads(f.read_text())
        a = p["args"]
        if a.get("lookback") != 336 or a.get("horizon") != 96:
            continue
        tag = ("HT" if a.get("head") == "horizon_tokens" else "pooled") + (f"-p{a['patch_len']}" if a.get("patch_len") else "") \
            + ("-IN" if a.get("instance_norm") else "") + f" {a.get('steps')}"
        for r in p["runs"]:
            groups[(a["dataset"], tag, r["model"])].append(r)
    table = defaultdict(dict)
    for (dataset, tag, model), rows in groups.items():
        bases = defaultdict(list)
        for r in rows:
            bases[r.get("base")].append(r)
        best = min(bases.values(), key=lambda rs: np.mean([r["val_mse"] for r in rs]))
        table[dataset][f"{tag} {TOOLS_MODE.get(model, model)}"] = float(np.mean([r["test_mse"] for r in best]))
    return table


def tools_baselines(files: list[Path], check: dict[str, int]) -> dict[str, dict[str, float]]:
    """tools/run_baselines.py outputs (one seed, hyper-parameters chosen on validation)."""
    table = defaultdict(dict)
    for f in files:
        p = json.loads(f.read_text())
        a = p["args"]
        if a.get("lookback") != 336 or a.get("horizon") != 96:
            continue
        d = a["dataset"]
        if d in check and p.get("data", {}).get("test_windows") not in (None, check[d]):
            print(f"WARNING {f}: {p['data']['test_windows']} test windows, this study has {check[d]}")
        for r in p["runs"]:
            table[d][r["model"]] = float(r["chosen"]["test_mse"])
    return table


def test_windows(runs: Path) -> dict[str, int]:
    out = {}
    for f in runs.glob("*/L336_H96/naive/s0/result.json"):
        r = json.loads(f.read_text())
        out[r["args"]["dataset"]] = r["windows"]["test"] * r["n_channels"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    p.add_argument("--benchmark", nargs="*", type=Path, default=[])
    p.add_argument("--baselines", nargs="*", type=Path, default=[])
    p.add_argument("--datasets", nargs="+", default=COMMON)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "tables" / "leaderboard.tex")
    args = p.parse_args()

    merged = defaultdict(dict)
    for part in (ours(args.runs), tools_benchmark(args.benchmark), tools_baselines(args.baselines, test_windows(args.runs))):
        for d, cols in part.items():
            merged[d].update(cols)
    frame = pd.DataFrame(merged).T.reindex(args.datasets)
    frame = frame.dropna(axis=1, how="all")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out.with_suffix(".csv"))
    cols = list(frame.columns)
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Dataset & " + " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for d, row in frame.iterrows():
        best = np.nanmin(row.values.astype(float))
        cells = ["--" if np.isnan(v) else (f"$\\mathbf{{{v:.3f}}}$" if np.isclose(v, best) else f"{v:.3f}") for v in row.values]
        lines.append(label(d) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.out.write_text("\n".join(lines) + "\n")
    print(frame.round(4).to_string())
    print(f"wrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
