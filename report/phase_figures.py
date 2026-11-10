"""Phase-drift and position-code figures (pure RoPE arithmetic, no trained model needed).

Setting of the ETTm1 mechanism study: 15-minute data, patches of 16 steps, so one day = 6 tokens.
History 336 steps = 21 tokens (positions 0..20); training horizon 96 = 6 placeholders (21..26);
queried at 720 = 45 placeholders (21..65). RoPE attention depends on the lag d = t - u between a
placeholder t and a history token u, through e^{i s w_j d}. The fastest pair has w_0 = 1 for every
base, so its angle is s*d. Lags seen in training: 1..26; lags used at H=720: 1..65.

fig5_phase_drift: angle of the fastest pair at "same time of day" lags d = 6k. Aligned (s = 2pi/6)
  it is identical every day; otherwise it drifts by (6s - 2pi) per day.
fig6_code_similarity: for each lag used at test time, the most similar lag seen in training
  (cosine of the fastest-pair angle difference) and the time-of-day it therefore "looks like".

    python3 report/phase_figures.py
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

# Same palette and style as make_assets.py: misaligned scales orange, aligned scales blue/aqua (as in fig4).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300})

PER_DAY = 6                    # tokens per day (96 steps / 16)
HISTORY, TRAIN_H, TEST_H = 21, 6, 45
SEEN = HISTORY + TRAIN_H - 1   # largest lag seen in training (26)
MAX_LAG = HISTORY + TEST_H - 1  # largest lag at H=720 (65)
ALIGNED = 2 * math.pi / PER_DAY

SCALES = [  # (scale, label, color, linestyle)
    (1.000, "Fixed RoPE (1.000)", ORANGE, "-"),
    (0.955, "Fixed 0.955", ORANGE, ":"),
    (1.100, "Fixed 1.10", ORANGE, "--"),
    (1.040, "Learned (1.04)", BLUE, "-"),
    (ALIGNED, "Fixed 1.047 (= 1 day)", AQUA, "-"),
]


def wrap(angle):
    """Wrap radians to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig5: phase drift
def phase_drift():
    fig = plt.figure(figsize=(6.6, 2.3))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.0], wspace=0.55)
    days = np.arange(0, MAX_LAG // PER_DAY + 1)  # same time of day, k days back
    seen = days * PER_DAY <= SEEN
    cmap = plt.get_cmap("Blues")
    shades = cmap(np.linspace(0.35, 1.0, len(days)))
    for col, (scale, title, color) in enumerate(((1.0, "(a) Fixed RoPE (1.000)", ORANGE),
                                                 (ALIGNED, "(b) Fixed 1.047", AQUA))):
        ax = fig.add_subplot(grid[0, col], projection="polar")
        angle = scale * days * PER_DAY
        ax.plot(np.linspace(0, 2 * np.pi, 200), np.ones(200), color=GRID, lw=0.8)
        for k in range(len(days))[::-1]:  # day 0 drawn last, on top
            ax.scatter(angle[k], 1.0, s=34, zorder=3, linewidths=1.0,
                       facecolor=shades[k] if seen[k] else "white", edgecolor=shades[k])
        if scale == 1.0:
            ax.annotate("day 0", (angle[0], 1.0), xytext=(angle[0] + 0.15, 1.45), fontsize=7, color=INK,
                        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
            ax.annotate(f"day {days[-1]}", (angle[-1], 1.0), xytext=(angle[-1] - 0.35, 1.5), fontsize=7,
                        color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
        else:
            ax.annotate(f"days 0-{days[-1]}\n(all overlap)", (angle[0], 1.0), xytext=(2.6, 0.55), fontsize=7,
                        color=INK, ha="center", arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
        ax.set_ylim(0, 1.6)
        ax.set_yticks([])
        ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False), [])
        ax.grid(color=GRID, lw=0.5)
        ax.spines["polar"].set_visible(False)
        ax.set_title(title, fontsize=9, color=INK, pad=10)
    ax = fig.add_subplot(grid[0, 2])
    lag_days = np.linspace(0, MAX_LAG / PER_DAY, 300)
    ax.axvspan(0, SEEN / PER_DAY, color=GRID, alpha=0.45, lw=0)
    ax.text(SEEN / PER_DAY / 2, 0.97, "lags seen\nin training", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7, color=INK2)
    for scale, label, color, ls in SCALES:
        drift = np.degrees((PER_DAY * scale - 2 * math.pi) * lag_days)
        ax.plot(lag_days, drift, color=color, ls=ls, lw=1.6)
        nudge = {1.040: -5, ALIGNED: 4}.get(scale, 0)
        ax.annotate(label, (lag_days[-1], drift[-1]), xytext=(4, nudge), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlim(0, MAX_LAG / PER_DAY)
    ax.set_xlabel("Lag (days back, same time of day)")
    ax.set_ylabel("Phase drift of fastest pair (°)")
    ax.set_title("(c) Drift accumulates with lag", fontsize=9, color=INK)
    save(fig, "fig5_phase_drift")


# ---------------------------------------------------------------- fig6: which training lag does each test lag look like
def nearest_seen(scale):
    """For every lag 1..MAX_LAG: similarity to every seen lag 1..SEEN, and the time-of-day error of the best match."""
    lags = np.arange(1, MAX_LAG + 1)
    seen = np.arange(1, SEEN + 1)
    sim = np.cos(scale * (lags[:, None] - seen[None, :]))
    best = seen[sim.argmax(1)]
    offset = (lags - best) % PER_DAY                     # tokens; 0 = same time of day
    hours = np.where(offset > PER_DAY / 2, offset - PER_DAY, offset) * 4.0  # one token = 4 hours, wrap to (-12, 12]
    return lags, seen, sim, hours


def code_similarity():
    fig = plt.figure(figsize=(6.6, 2.6))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.06, 0.32, 1.55], wspace=0.3)
    for col, (scale, title) in enumerate(((1.0, "(a) Fixed RoPE (1.000)"), (ALIGNED, "(b) Fixed 1.047"))):
        ax = fig.add_subplot(grid[0, col])
        lags, seen, sim, hours = nearest_seen(scale)
        im = ax.imshow(sim, aspect="auto", cmap="cividis", vmin=-1, vmax=1, interpolation="nearest",
                       extent=(0.5, SEEN + 0.5, MAX_LAG + 0.5, 0.5))
        ax.axhline(SEEN + 0.5, color="white", ls="--", lw=0.8)
        best = seen[sim.argmax(1)]
        wrong = hours != 0
        ax.scatter(best[wrong], lags[wrong], s=5, color=ORANGE, zorder=3, lw=0)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("Lag seen in training")
        ax.set_ylabel("Lag at test time" if col == 0 else "")
        ax.grid(False)
        ax.set_xticks([1, 6, 12, 18, 24])
    cbar = fig.colorbar(im, cax=fig.add_subplot(grid[0, 2]))
    cbar.ax.set_title("cos Δθ", fontsize=7, color=INK, pad=4)
    cbar.ax.tick_params(labelsize=7)
    ax = fig.add_subplot(grid[0, 4])  # column 3 is an empty spacer
    ax.axvspan(0, SEEN / PER_DAY, color=GRID, alpha=0.45, lw=0)
    ax.text(SEEN / PER_DAY / 2, 0.97, "seen in\ntraining", transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=7, color=INK2)
    for scale, label, color, ls in SCALES:
        lags, _, _, hours = nearest_seen(scale)
        on_top = scale == 1.040  # learned overlaps the aligned line; draw it dashed on top
        ax.step(lags / PER_DAY, hours, where="mid", color=color, ls=(0, (3, 2)) if on_top else ls,
                lw=1.2 if on_top else 1.8, label=label, zorder=4 if on_top else 3)
    ax.set_yticks([-12, -8, -4, 0, 4, 8, 12])
    ax.set_xlim(0, MAX_LAG / PER_DAY)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Hour error of best match (h)")
    ax.set_title("(c) Which hour the code looks like", fontsize=9, color=INK)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.8, ncol=1)
    save(fig, "fig6_code_similarity")


if __name__ == "__main__":
    phase_drift()
    code_similarity()
    for scale, label, *_ in SCALES:
        drift = math.degrees(PER_DAY * scale - 2 * math.pi)
        _, _, _, hours = nearest_seen(scale)
        first = next((l for l, h in zip(range(1, MAX_LAG + 1), hours) if h != 0), None)
        print(f"{label:24} drift {drift:+6.1f} deg/day; first lag mistaken for another hour: "
              f"{'never' if first is None else f'{first} tokens = {first / PER_DAY:.1f} days'}")
    print(f"Saved {FIG / 'fig5_phase_drift.pdf'} and {FIG / 'fig6_code_similarity.pdf'}")
