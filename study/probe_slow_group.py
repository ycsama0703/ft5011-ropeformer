"""Inference-time probe: stop the rotation of the slow rotary pairs in trained checkpoints.

For each trained Transformer, test MSE is computed twice on the same evenly spaced test
windows: as trained, and with the effective frequencies of the slower half of the pairs
(j >= m/2, i.e. group 1 of G=2) set to zero, which turns those pairs into position-free
channels (as p-RoPE does by construction, Barbero et al. 2025). A large increase means the
model uses slow-pair positional information; no change means the slow pairs act as content
channels. No retraining; the checkpoints themselves are not modified.

    python study/probe_slow_group.py --runs runs --out results/probe_slow_group.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "study"))
from grouped_rope.study.models import build_model  # noqa: E402
from grouped_rope.study.data import load_dataset  # noqa: E402
from grouped_rope.model import GroupedRotaryEmbedding  # noqa: E402
from train import evaluate, subset  # noqa: E402

NAMES = {"standard": "A", "head": "B", "shared_group": "C", "grouped": "D", "partial": "S"}


def freeze_slow_pairs(model, first_frozen):
    """Patch every rotary module so pairs j >= first_frozen have zero frequency (no rotation)."""
    for module in model.modules():
        if isinstance(module, GroupedRotaryEmbedding):
            pairs = module.base_frequencies.numel()
            keep = (torch.arange(pairs, device=module.base_frequencies.device) < first_frozen).float()
            original = module.effective_frequencies
            module.effective_frequencies = lambda original=original, keep=keep: original() * keep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "probe_slow_group.csv")
    p.add_argument("--data-dir", type=Path, default=None, help="override the data directory stored in the runs")
    p.add_argument("--test-cap", type=int, default=100_000, help="max test sequences (evenly spaced windows)")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, cache = [], {}
    for f in sorted(args.runs.glob("*/L*_H*/*/s*/result.json")):
        r = json.loads(f.read_text())
        a = r["args"]
        if a["model"] not in ("transformer", "patch_transformer") or a.get("groups", 2) != 2:
            continue
        if not (f.parent / "model.pt").exists() or abs(float(a.get("scale_lr", 1e-2)) - 1e-2) > 1e-12:
            continue
        data_dir = str(args.data_dir or a["data_dir"])
        key = (a["dataset"], a["lookback"], a["horizon"])
        if key not in cache:
            cache = {key: load_dataset(a["dataset"], data_dir, a["lookback"], a["horizon"], device, a.get("file"))}
        data = cache[key]
        tf = dict(d_model=a["d_model"], n_heads=a["n_heads"], num_layers=a["layers"], d_ff=a["d_ff"],
                  dropout=a["dropout"], rope_base=a["rope_base"], rope_mode=a["rope_mode"],
                  n_freq_groups=a["groups"], pooling=a["pooling"])
        if a["model"] == "patch_transformer":
            tf.update(patch_len=a["patch_len"], stride=a["stride"])
        model = build_model(a["model"], a["lookback"], a["horizon"], data.period, **tf).to(device)
        model.load_state_dict(torch.load(f.parent / "model.pt", map_location=device, weights_only=True)["state_dict"])
        index = subset(data.n_windows("test"), data.n_channels, args.test_cap, device)
        amp = bool(a.get("amp", 1))
        base_mse = float(evaluate(model, data, "test", index, amp=amp)[0].mean())
        m = a["d_model"] // a["n_heads"] // 2
        freeze_slow_pairs(model, 3 * m // 4)  # slowest quarter of the pairs (j = 6, 7 for m = 8)
        quarter_mse = float(evaluate(model, data, "test", index, amp=amp)[0].mean())
        model = build_model(a["model"], a["lookback"], a["horizon"], data.period, **tf).to(device)
        model.load_state_dict(torch.load(f.parent / "model.pt", map_location=device, weights_only=True)["state_dict"])
        freeze_slow_pairs(model, m // 2)  # whole slow group (j = 4..7)
        frozen_mse = float(evaluate(model, data, "test", index, amp=amp)[0].mean())
        name = ("P" if a["model"] == "patch_transformer" else "") + NAMES.get(a["rope_mode"], a["rope_mode"])
        rows.append(dict(dataset=a["dataset"], name=name, base=float(a["rope_base"]), seed=a["seed"],
                         test_mse=base_mse, frozen_mse=frozen_mse, quarter_mse=quarter_mse,
                         change_pct=100 * (frozen_mse - base_mse) / base_mse,
                         change_quarter_pct=100 * (quarter_mse - base_mse) / base_mse, windows=len(index)))
        print(f"{a['dataset']:22s} {name:3s} b{a['rope_base']:<7g} s{a['seed']}: "
              f"{base_mse:.4f} -> half {frozen_mse:.4f} ({rows[-1]['change_pct']:+.2f}%), "
              f"quarter {quarter_mse:.4f} ({rows[-1]['change_quarter_pct']:+.2f}%)", flush=True)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    if not df.empty:
        summary = df.groupby(["name", "base"])[["change_pct", "change_quarter_pct"]].agg(["mean", "median", "count"])
        print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
