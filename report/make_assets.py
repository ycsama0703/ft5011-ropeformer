"""Build every figure and LaTeX table of the report from report/results/*.json.

Re-run after new results arrive:  python3 report/make_assets.py [results_dir]
Base selection follows the repository protocol: for each model, rope_base is chosen by mean validation MSE
over seeds; test numbers are reported only for that base.
"""
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
import sys
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"  # e.g. python report/make_assets.py results
FIG, TAB = ROOT / "figures", ROOT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

# reference palette (light): blue, orange, aqua; neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300})

NAMES = {"ETTh1": "ETTh1", "ETTm1": "ETTm1", "weather": "Weather", "electricity": "Electricity",
         "epf_np": "EPF-NP", "epf_pjm": "EPF-PJM", "exchange_rate": "Exchange", "exchange_rate_norm": "Exchange",
         "btc_hourly": "BTC"}


def load(name):
    f = RES / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def ms(v, d=3):
    if not v:
        return "--"
    s = st.stdev(v) if len(v) > 1 else 0.0
    return f"{st.mean(v):.{d}f}$\\pm${s:.{d}f}"


def mean(v):
    return st.mean(v) if v else float("nan")


def pick_base(runs, model, key="val_mse"):
    bases = sorted({r["base"] for r in runs if r["model"] == model})
    if not bases:
        return None
    return min(bases, key=lambda b: mean([r[key] for r in runs if r["model"] == model and r["base"] == b]))


# ---------------------------------------------------------------- extrapolation study files per dataset
STUDY = {"ETTh1": "r2_ETTh1_720", "ETTm1": "r2_ETTm1", "weather": "r2_weather", "electricity": "r2_electricity",
         "epf_np": "fin_epf_np", "epf_pjm": "fin_epf_pjm", "exchange_rate_norm": "r2_exchange_rate_norm",
         "btc_hourly": "fin_btc_hourly_norm"}
MULTI = {"ETTh1": "mh_ETTh1", "ETTm1": "mh_ETTm1", "weather": "mh_weather", "electricity": "mh_electricity",
         "epf_np": "mhfin_epf_np", "epf_pjm": "mhfin_epf_pjm", "exchange_rate_norm": "mh_exchange_rate_norm",
         "btc_hourly": "mhfin_btc_hourly_norm"}
TRAINVAL = {"ETTh1": "trainval_ETTh1", "ETTm1": "trainval_ETTm1", "weather": "trainval_weather",
            "electricity": "trainval_electricity", "epf_np": "trainval_epf_np", "epf_pjm": "trainval_epf_pjm",
            "exchange_rate_norm": "trainval_exchange_rate_norm"}
LIN96 = {"ETTh1": "lin96_ETTh1", "ETTm1": "lin96_ETTm1", "weather": "lin96_weather", "electricity": "lin96_electricity",
         "epf_np": "lin96_epf_np", "epf_pjm": "lin96_epf_pjm", "exchange_rate_norm": "lin96_exchange_rate_norm"}
DM = {"ETTh1": "dm_ETTh1", "ETTm1": "dm_ETTm1", "weather": "dm_weather", "electricity": "dm_electricity",
      "epf_np": "dm_epf_np", "epf_pjm": "dm_epf_pjm", "exchange_rate_norm": "dm_exchange_rate_norm"}
ORDER = ["electricity", "ETTh1", "ETTm1", "weather", "epf_np", "epf_pjm", "exchange_rate_norm"]  # BTC results are in the release but not in the report


def extrap(study, model, H=720, lo=0):
    """Per-seed MSE over steps (lo, H] for the validation-chosen base of a train@96 model."""
    ex = [r for r in study["runs"] if r["kind"] == "extrapolate"]
    b = pick_base(ex, model)
    return [st.mean(r["per_step_learned"][lo:H]) for r in ex if r["model"] == model and r["base"] == b], b


def direct(study, model, H):
    di = [r for r in study["runs"] if r["kind"] == "direct" and r["horizon"] == H]
    if model == "linear":
        return [r["test_mse"] for r in di if r["model"] == "linear"]
    b = pick_base([r for r in di if r["model"] == model], model)
    return [r["test_mse"] for r in di if r["model"] == model and r["base"] == b]


def multi(mh, model, H):
    if mh is None:
        return []
    b = pick_base(mh["runs"], model)
    return [r["test_mse"][str(H)] for r in mh["runs"] if r["model"] == model and r["base"] == b]


# ---------------------------------------------------------------- Table 1: in-horizon (H=96) train/val/test
lines = []
for ds in ORDER:
    tv, lin = load(TRAINVAL.get(ds, "")), load(LIN96.get(ds, ""))
    if tv is None:
        continue
    cells = [NAMES[ds]]
    if lin:
        cells += [ms([r["train"] for r in lin["rows"]]), ms([r["test"] for r in lin["rows"]])]
    else:
        cells += ["--", "--"]
    for model in ("standard", "head"):
        rows = [r for r in tv["rows"] if r["model"] == model]
        b = min(sorted({r["base"] for r in rows}), key=lambda b: mean([r["val"] for r in rows if r["base"] == b]))
        sel = [r for r in rows if r["base"] == b]
        cells += [ms([r["train"] for r in sel]), ms([r["test"] for r in sel])]
    lines.append(" & ".join(cells) + r" \\")
(TAB / "t1_inhorizon.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table 2: extrapolation to H=720
lines = []
for ds in ORDER:
    s, mh, dm = load(STUDY.get(ds, "")), load(MULTI.get(ds, "")), load(DM.get(ds, ""))
    if s is None:
        continue
    fx, _ = extrap(s, "standard")
    ln, _ = extrap(s, "head")
    sig = "--"
    if dm:
        rows = dm["rows"]
        better = sum(1 for r in rows if r["mean_diff"] > 0 and r["p"] < 0.05)
        worse = sum(1 for r in rows if r["mean_diff"] < 0 and r["p"] < 0.05)
        sig = f"{better}\\,/\\,{worse}"
    cells = [NAMES[ds], ms(fx), ms(ln), sig, ms(multi(mh, "head", 720)), ms(direct(s, "head", 720)),
             ms(direct(s, "linear", 720))]
    lines.append(" & ".join(cells) + r" \\")
(TAB / "t2_extrap.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Table 3: mechanism / ablations
lines = []
c = load("r2_ETTh1_controls")
s16 = load("study_ETTh1_p16")
if c and s16:
    def h336(study, model):
        rs = [r for r in study["runs"] if r["kind"] == "extrapolate" and r["model"] == model]
        return [st.mean(r["per_step_learned"][:336]) for r in rs]
    spec = [("Fixed RoPE (scale 1)", s16, "standard"), ("Learnable, per head", s16, "head"),
            ("Learnable, grouped", s16, "grouped"), ("Learnable, shared by heads", c, "shared_group"),
            ("Fixed, mild spread (0.9--1.1)", c, "fixed_head:0.9,0.97,1.03,1.1"),
            ("Fixed, wide spread (0.5--2.0)", c, "fixed_head:0.5,0.8,1.25,2.0")]
    lines.append(r"\multicolumn{2}{l}{\emph{ETTh1, steps 1--336, 5 seeds}}\\")
    for label, study, model in spec:
        lines.append(f"{label} & {ms(h336(study, model))} \\\\")
a = load("r3_ETTm1_align")
e = load("r2_ETTm1")
if a and e:
    def h720(study, model):
        rs = [r for r in study["runs"] if r["kind"] == "extrapolate" and r["model"] == model]
        return [st.mean(r["per_step_learned"][:720]) for r in rs]
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\emph{ETTm1, steps 1--720, 3 seeds $\times$ 2 bases}}\\")
    for label, study, model in [("Fixed RoPE (scale 1)", e, "standard"), ("Learnable, per head", e, "head"),
                                ("Fixed 1.047 (= 1 day)", a, "fixed_head:1.0472,1.0472,1.0472,1.0472"),
                                ("Fixed 0.955 (misaligned)", a, "fixed_head:0.955,0.955,0.955,0.955"),
                                ("Fixed 1.10 (misaligned)", a, "fixed_head:1.10,1.10,1.10,1.10")]:
        lines.append(f"{label} & {ms(h720(study, model))} \\\\")
(TAB / "t3_mechanism.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- Figure 1: per-step error curves


def smooth(v, w=24):
    out, acc = [], 0.0
    for i, x in enumerate(v):
        acc += x
        if i >= w:
            acc -= v[i - w]
        out.append(acc / min(i + 1, w))
    return out


fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.95))
for ax, ds in zip(axes, ["electricity", "ETTm1"]):
    s, mh = load(STUDY[ds]), load(MULTI[ds])
    base = 10000.0
    for model, color, label in [("standard", ORANGE, "Fixed RoPE"),
                                ("head", BLUE, "Learnable scale")]:
        rs = [r["per_step_learned"] for r in s["runs"] if r["kind"] == "extrapolate" and r["model"] == model and r["base"] == base]
        curve = [st.mean(x) for x in zip(*rs)]
        ax.plot(range(1, len(curve) + 1), smooth(curve), color=color, lw=1.6, label=label)
    if mh:
        rs = [r["per_step"] for r in mh["runs"] if r["model"] == "head" and r["base"] == base]
        if rs:
            curve = [st.mean(x) for x in zip(*rs)]
            ax.plot(range(1, len(curve) + 1), smooth(curve), color=AQUA, lw=1.6, label="Learnable scale + multi-horizon training")
    ax.axvline(96, color=INK2, lw=0.8, ls="--")
    ax.text(110, ax.get_ylim()[1] * 0.97, "training\nhorizon", va="top", ha="left", fontsize=8, color=INK2)
    ax.set_title(NAMES[ds], fontsize=9, color=INK)
    ax.set_xlabel("Forecast step")
    ax.set_xlim(1, 720)
axes[0].set_ylabel("Test MSE")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.09), fontsize=8)
fig.savefig(FIG / "fig1_perstep.pdf")
fig.savefig(FIG / "fig1_perstep.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: alignment bars (ETTm1, H=720)
def learned_mean(study):
    v = []
    for r in study["runs"]:
        if r["kind"] == "extrapolate" and r["model"] == "head":
            best = [p for p in r["scale_trajectory"] if p["step"] == r["best_step"]][0]["log_scales"]
            v += [math.exp(x[0]) for layer in best for x in layer]
    return st.mean(v)


if a and e:
    items = [("0.955", a, "fixed_head:0.955,0.955,0.955,0.955", ORANGE),
             ("1.000 (standard)", e, "standard", ORANGE),
             ("1.100", a, "fixed_head:1.10,1.10,1.10,1.10", ORANGE),
             ("1.047 (= 1 day)", a, "fixed_head:1.0472,1.0472,1.0472,1.0472", BLUE),
             (f"learned ({learned_mean(e):.2f})", e, "head", BLUE)]
    fig, ax = plt.subplots(figsize=(2.6, 1.8))
    for i, (label, study, model, color) in enumerate(items):
        v = [st.mean(r["per_step_learned"][:720]) for r in study["runs"] if r["kind"] == "extrapolate" and r["model"] == model]
        m, sd = st.mean(v), st.stdev(v)
        ax.barh(i, m, color=color, height=0.6, xerr=sd, error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
        ax.text(m + sd + 0.02, i, f"{m:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(items)), [x[0] for x in items])
    ax.invert_yaxis()
    ax.set_xlabel("ETTm1 test MSE, steps 1--720")
    ax.set_ylabel("Frequency scale")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.15)
    fig.savefig(FIG / "fig2_alignment.pdf")
    fig.savefig(FIG / "fig2_alignment.png")
    plt.close(fig)

# ---------------------------------------------------------------- Figure 3: periodicity vs benefit
per = load("periodicity")
if per:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.3))
    for ds in ORDER:
        s = load(STUDY.get(ds, ""))
        key = "exchange_rate" if ds == "exchange_rate_norm" else ds
        if s is None or key not in per:
            continue
        ex = [r for r in s["runs"] if r["kind"] == "extrapolate"]
        fx = mean([st.mean(r["per_step_learned"][96:720]) for r in ex if r["model"] == "standard"])
        ln = mean([st.mean(r["per_step_learned"][96:720]) for r in ex if r["model"] == "head"])
        x, y = per[key]["median"], fx / ln
        fin = ds in ("epf_np", "epf_pjm", "exchange_rate_norm", "btc_hourly")
        a1.scatter(x, y, s=28, color=ORANGE if fin else BLUE, edgecolor="white", linewidth=1, zorder=3)
        off = {"weather": (7, -11), "ETTm1": (7, -3), "epf_pjm": (-40, 4), "epf_np": (7, -3), "electricity": (-52, -3)}
        a1.annotate(NAMES[ds], (x, y), textcoords="offset points", xytext=off.get(ds, (4, 3)), fontsize=8, color=INK)
    a1.axhline(1, color=INK2, lw=0.8, ls="--")
    a1.set_xlabel("Daily seasonal strength (median)")
    a1.set_ylabel("MSE ratio fixed / learnable")
    a1.set_title("(a) Datasets (orange: markets)", fontsize=9, color=INK)
    for ds, color in (("weather", AQUA), ("ETTm1", BLUE)):
        pc = load(f"perchan_{ds}")
        if pc is None:
            continue
        xs = per[ds]["per_channel"]
        ys = [f / l for f, l in zip(pc["fixed"], pc["learnable"])]
        a2.scatter(xs, ys, s=22, color=color, edgecolor="white", linewidth=1, label=NAMES[ds], zorder=3)
    a2.axhline(1, color=INK2, lw=0.8, ls="--")
    a2.set_xlabel("Daily seasonal strength")
    a2.set_title("(b) Channels within a dataset", fontsize=9, color=INK)
    a2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_periodicity.pdf")
    fig.savefig(FIG / "fig3_periodicity.png")
    plt.close(fig)

print("tables:", *sorted(p.name for p in TAB.iterdir()))
print("figures:", *sorted(p.name for p in FIG.iterdir()))
