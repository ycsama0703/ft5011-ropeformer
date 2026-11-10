"""Mechanism figures from the released ETTm1 checkpoints and results (CPU is enough; no training).

Setting: ETTm1, patch 16 (one day = 6 tokens), 21 history tokens, trained at H=96 (6 placeholders),
queried at H=720 (45 placeholders), base 1e4, seeds 0-2.

fig7_scales_and_reset   (a) learned per-head scales during training vs the day-aligned value 2*pi/6;
                        (b) per-step test MSE: fixed RoPE, learnable, and the same learnable weights with
                            every scale reset to 1 at test time (from r2_ETTm1.json).
fig8_attention_offset   attention of each placeholder split by the time-of-day offset of the history token
                        it attends to, against forecast lead; the line is the offset that the RoPE arithmetic
                        of phase_figures.py predicts the model will confuse with "same time of day".
Attention is the last layer, mean over heads and seeds, as in the report.
fig4_attention_repro    re-computation of the report's Figure 4 from the checkpoints.

    python3 report/mechanism_figures.py --results results --data-dir data
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_rope import GroupedRoPEForecaster  # noqa: E402
from grouped_rope.data import WindowedSeries  # noqa: E402
from phase_figures import (AQUA, BLUE, GRID, INK, INK2, ORANGE, nearest_seen, save)  # noqa: E402

PER_DAY, HISTORY, TRAIN_P, TEST_P, PATCH = 6, 21, 6, 45, 16
ALIGNED = 2 * math.pi / PER_DAY
MODELS = [  # (key, label, directory, file mode, head_scale_init, color, linestyle)
    ("fixed", "Fixed RoPE (1.000)", "ckpt", "standard", None, ORANGE, "-"),
    ("mis", "Fixed 0.955", "ckpt_r3", "fixed_head-0.955,0.955,0.955,0.955", (0.955,) * 4, ORANGE, ":"),
    ("learned", "Learned", "ckpt", "head", None, BLUE, "-"),
    ("aligned", "Fixed 1.047 (= 1 day)", "ckpt_r3", "fixed_head-1.0472,1.0472,1.0472,1.0472", (1.0472,) * 4, AQUA, "-"),
]
SPEC = {"fixed": ("standard", None), "mis": ("fixed_head", (0.955,) * 4), "learned": ("head", None),
        "aligned": ("fixed_head", (1.0472,) * 4)}


def load_model(results, key, seed, base=10000.0):
    _, _, folder, filemode, *_ = next(m for m in MODELS if m[0] == key)
    mode, init = SPEC[key]
    model = GroupedRoPEForecaster(input_dim=1, horizon=96, d_model=64, n_heads=4, num_layers=2, d_ff=128,
                                  dropout=0.1, n_freq_groups=2, forecast_head="horizon_tokens", patch_len=PATCH,
                                  rope_mode=mode, rope_base=base,
                                  **({"head_scale_init": init} if init else {}))
    state = torch.load(results / folder / f"ETTm1_p{PATCH}_{filemode}_b{base:g}_s{seed}.pt", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def placeholder_attention(model, x, chunk=512):
    """Mean attention [layers, heads, placeholders, history] of the 45 placeholders over the given windows."""
    total = None
    for i in range(0, x.shape[0], chunk):
        _, maps = model(x[i:i + chunk], horizon=TEST_P * PATCH, return_attention=True)
        block = torch.stack([m[:, :, HISTORY:, :HISTORY].sum(0) for m in maps])
        total = block if total is None else total + block
    return (total / x.shape[0]).numpy()


def offsets():
    """Time-of-day offset (in tokens, 0..5) between every placeholder and every history token."""
    p = HISTORY + np.arange(TEST_P)[:, None]
    u = np.arange(HISTORY)[None, :]
    return (p - u) % PER_DAY


def fig_scales_and_reset(study):
    runs = [r for r in study["runs"] if r["kind"] == "extrapolate" and r["base"] == 10000.0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.3), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.3})
    for r in [r for r in runs if r["model"] == "head"]:
        steps = [0] + [t["step"] for t in r["scale_trajectory"]]
        for layer in range(2):
            for head in range(4):
                values = [1.0] + [math.exp(t["log_scales"][layer][head][0]) for t in r["scale_trajectory"]]
                a1.plot(steps, values, color=BLUE, lw=0.7, alpha=0.45)
    a1.axhline(ALIGNED, color=AQUA, lw=1.4, ls="--")
    a1.axhline(1.0, color=ORANGE, lw=1.0)
    a1.text(steps[-1], ALIGNED + 0.004, "2π/6 = 1.047 (one day)", ha="right", va="bottom", fontsize=7, color=INK)
    a1.text(steps[-1], 1.0 - 0.004, "standard RoPE", ha="right", va="top", fontsize=7, color=INK)
    a1.set_xlabel("Training step")
    a1.set_ylabel("Learned scale exp(a)")
    a1.set_title("(a) 8 heads × 3 seeds, base 1e4", fontsize=9, color=INK)

    def curve(model, key):
        rows = [r[key] for r in runs if r["model"] == model]
        v = np.mean(rows, 0)
        return np.convolve(v, np.ones(24) / 24, mode="valid")  # one-day moving average (96 steps = 1 day; 24 = 6 h)

    lead = (np.arange(len(curve("standard", "per_step_learned"))) + 12) / 96
    a2.axvspan(0, 1, color=GRID, alpha=0.45, lw=0)
    a2.text(0.5, 0.97, "training\nhorizon", transform=a2.get_xaxis_transform(), ha="center", va="top", fontsize=7,
            color=INK2)
    for (model, key, label, color, ls) in (("standard", "per_step_learned", "Fixed RoPE", ORANGE, "-"),
                                           ("head", "per_step_reset", "Learned, scales reset to 1", ORANGE, "--"),
                                           ("head", "per_step_learned", "Learned", BLUE, "-")):
        a2.plot(lead, curve(model, key), color=color, ls=ls, lw=1.5, label=label)
    a2.set_xlabel("Forecast lead (days)")
    a2.set_ylabel("Test MSE per step")
    a2.set_title("(b) Remove the learned scales at test time", fontsize=9, color=INK)
    a2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.2, 1.0))
    save(fig, "fig7_scales_and_reset")
    final = [math.exp(v[0]) for r in runs if r["model"] == "head" for layer in r["scale_trajectory"][-1]["log_scales"]
             for v in layer]
    return float(np.mean(final)), float(np.min(final)), float(np.max(final))


def fig_attention_offset(attn):
    """attn: key -> [layers, heads, 45, 21] averaged over seeds."""
    off = offsets()
    lead = (np.arange(TEST_P) + 1) / PER_DAY
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.9), sharey=True, gridspec_kw={"wspace": 0.12})
    # theory: offset whose position code best matches the trained "same time of day" code (phase_figures.py)
    for ax, (key, label, *_rest) in zip(axes, MODELS):
        a = attn[key][-1].mean(0)  # [45, 21]
        share = np.stack([(a * (off == r)).sum(1) for r in range(PER_DAY)], 1)  # [45, 6]
        share = np.roll(share, 2, axis=1)  # rows -8h .. +12h, so 0 h sits in the middle
        im = ax.imshow(share.T, aspect="auto", origin="lower", cmap="cividis", vmin=0, vmax=0.6,
                       extent=(lead[0] - 1 / 12, lead[-1] + 1 / 12, -0.5, PER_DAY - 0.5))
        scale = {"fixed": 1.0, "mis": 0.955, "learned": None, "aligned": ALIGNED}[key]
        if scale is not None:
            lags, _, _, hours = nearest_seen(scale)
            # same-time-of-day lags: placeholder at lead k sees history at lag k + j*6; use lag of the most recent
            # same-time token (k rounded up to a multiple of 6) as the representative distance
            rep = [((HISTORY + k) // PER_DAY) * PER_DAY for k in range(TEST_P)]
            pred = np.array([hours[d - 1] / 4 for d in rep])
            ax.plot(lead, pred + 2, color="white", lw=1.0, ls="--")
        ax.axvline(1.0, color="white", lw=0.6, alpha=0.6)
        ax.set_title(label.replace(" (= 1 day)", ""), fontsize=8, color=INK)
        ax.set_xlabel("Lead (days)")
        ax.grid(False)
    axes[0].set_yticks(range(PER_DAY), ["−8 h", "−4 h", "0 h", "+4 h", "+8 h", "±12 h"])
    axes[0].set_ylabel("Time-of-day offset\nof attended history")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.ax.set_title("share", fontsize=7, color=INK, pad=4)
    cbar.ax.tick_params(labelsize=7)
    save(fig, "fig8_attention_offset")


def fig4_repro(attn):
    off = offsets()
    lead = (np.arange(TEST_P) + 1) / PER_DAY
    fig = plt.figure(figsize=(6.6, 2.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1], wspace=0.35)
    ax = fig.add_subplot(grid[0, 0])
    for key, label, _, _, _, color, ls in MODELS:
        share = (attn[key][-1].mean(0) * (off == 0)).sum(1)
        ax.plot(lead, share, color=color, ls=ls, lw=1.5, label=label)
    ax.axhline(1 / PER_DAY, color=INK2, lw=0.6)
    ax.text(lead[-1], 1 / PER_DAY + 0.01, "chance", ha="right", fontsize=7, color=INK2)
    ax.axvline(1.0, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Forecast lead (days)")
    ax.set_ylabel("Attention on same\ntime of day")
    ax.set_title("(a) Share of placeholder attention", fontsize=9, color=INK)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    for col, (key, title) in enumerate((("fixed", "(b) Fixed RoPE"), ("aligned", "(c) Fixed 1.047")), start=1):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(attn[key][-1].mean(0), aspect="auto", cmap="cividis", interpolation="nearest",
                  extent=(-HISTORY - 0.5, -0.5, TEST_P + 0.5, 0.5))
        ax.axhline(TRAIN_P + 0.5, color="white", ls="--", lw=0.8)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("History patch")
        ax.set_ylabel("Future patch" if col == 1 else "")
        ax.grid(False)
    save(fig, "fig4_attention_repro")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--windows", type=int, default=4096, help="Random test (origin, channel) windows")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    mean_scale, lo, hi = fig_scales_and_reset(json.loads((args.results / "r2_ETTm1.json").read_text()))
    print(f"learned per-head scales at the end of training: mean {mean_scale:.3f} (range {lo:.3f}-{hi:.3f})")

    data = WindowedSeries.from_csv(args.data_dir / "ETTm1.csv", "ETTm1", HISTORY * PATCH)
    samples = data.samples("test", TEST_P * PATCH)
    pick = torch.randperm(samples.shape[0], generator=torch.Generator().manual_seed(0))[:args.windows]
    x, _ = data.batch(samples[pick.sort().values], TEST_P * PATCH)
    attn = {}
    for key, *_ in MODELS:
        attn[key] = np.mean([placeholder_attention(load_model(args.results, key, s), x) for s in args.seeds], 0)
        share = (attn[key][-1].mean(0) * (offsets() == 0)).sum(1)
        print(f"{key:8} same-time share: lead 1 day {share[5]:.2f}, 2 days {share[11]:.2f}, "
              f"4 days {share[23]:.2f}, 7.5 days {share[44]:.2f}")
    np.savez_compressed(Path(__file__).resolve().parent / "figures" / "attention_ETTm1.npz", **attn)
    fig_attention_offset(attn)
    fig4_repro(attn)
    print("Saved fig7_scales_and_reset, fig8_attention_offset, fig4_attention_repro")


if __name__ == "__main__":
    main()
