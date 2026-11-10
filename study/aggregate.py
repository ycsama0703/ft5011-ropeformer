"""Aggregate finished runs into tables, significance tests and figures.

Base selection: for every Transformer variant the RoPE base is chosen by mean
validation MSE over seeds (same candidate set for every variant). Significance:
Diebold-Mariano test on seed-averaged per-window test MSE, Newey-West (Bartlett)
long-run variance with lag = horizon (targets of consecutive windows overlap),
Harvey-Leybourne-Newbold correction, t(n-1) reference, Holm across datasets.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["synthetic_multiscale", "ETTm1", "ETTm2", "ETTh1", "ETTh2", "electricity", "traffic",
         "PEMS04", "PEMS08", "solar", "weather", "crypto_hourly", "exchange_rate"]
LABEL = {"synthetic_multiscale": "Synthetic", "electricity": "Electricity", "traffic": "Traffic",
         "solar": "Solar", "weather": "Weather", "crypto_hourly": "Crypto (fin.)",
         "exchange_rate": "Exchange (fin.)"}
VARIANT = {"standard": "A", "head": "B", "shared_group": "C", "grouped": "D", "grouped4": "E", "grouped8": "F",
           "partial": "S"}  # S = p-RoPE: slow group not rotated
TRANSFORMERS = set(VARIANT.values()) | {"P" + v for v in VARIANT.values()}
BASELINE = {"naive": "Naive", "seasonal_naive": "SNaive", "linear": "RLinear", "lstm": "LSTM"}
COLUMNS = ["Naive", "SNaive", "RLinear", "LSTM", "A", "B", "C", "D"]


def label(d):
    return LABEL.get(d, d)


def collect(runs: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs.glob("*/L*_H*/*/s*/result.json")):
        r = json.loads(f.read_text())
        a = r["args"]
        if a.get("from_checkpoint"):
            continue  # replication re-evaluations are not separate runs
        if a["model"] in ("transformer", "patch_transformer"):
            mode = a["rope_mode"] + (str(a["groups"]) if a["rope_mode"] == "grouped" and a["groups"] != 2 else "")
            name, base = VARIANT.get(mode, mode), float(a["rope_base"])
            if a["model"] == "patch_transformer":
                name = "P" + name
            scale_lr = float(a.get("scale_lr", 1e-2))
        else:
            name, base, scale_lr = BASELINE[a["model"]], np.nan, np.nan
        m = r["metrics"]
        rows.append(dict(
            dataset=a["dataset"], H=a["horizon"], L=a["lookback"], name=name, base=base, seed=a["seed"],
            scale_lr=scale_lr, steps=a.get("steps", 3000),
            path=str(f.parent), train_mse=m["train"]["mse"], train_mae=m["train"]["mae"],
            val_mse=m["val"]["mse"], test_mse=m["test"]["mse"], test_mae=m["test"]["mae"],
            best_step=r["best"]["step"], seconds=r["seconds"], n_params=r["n_params"],
        ))
    return pd.DataFrame(rows)


def select_bases(df: pd.DataFrame) -> pd.DataFrame:
    """Keep, for each Transformer variant, the base with the lowest mean validation MSE."""
    tf = df[df.name.isin(TRANSFORMERS)]
    mean_val = tf.groupby(["dataset", "L", "H", "name", "base"]).val_mse.mean().reset_index()
    best = mean_val.loc[mean_val.groupby(["dataset", "L", "H", "name"]).val_mse.idxmin()]
    keep = tf.merge(best[["dataset", "L", "H", "name", "base"]], on=["dataset", "L", "H", "name", "base"])
    return pd.concat([df[~df.name.isin(TRANSFORMERS)], keep], ignore_index=True)


def window_losses(df, dataset, H, name, base=None):
    sel = df[(df.dataset == dataset) & (df.H == H) & (df.name == name)]
    if base is not None:
        sel = sel[sel.base == base]
    if sel.empty:
        return None
    arrays = [np.load(Path(p) / "test_mse.npy").astype(np.float64) for p in sel.path]
    return np.mean(arrays, axis=0)


def dm_test(loss_a, loss_b, h):
    """DM test of equal accuracy; positive statistic means b has lower loss than a."""
    d = loss_a - loss_b
    n = len(d)
    dc = d - d.mean()
    lag = min(h, n - 1)
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    lrv = max(lrv, 1e-300)
    stat = d.mean() / np.sqrt(lrv / n)
    stat *= np.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1.0) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def fmt(mean, sd, bold=False):
    sd_text = f"{sd:.3f}"
    sd_text = sd_text[1:] if sd_text.startswith("0.") else sd_text
    body = f"{mean:.3f}" + (f"_{{\\pm{sd_text}}}" if np.isfinite(sd) and sd > 0 else "")
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def fmt_p(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def main_table(sel, datasets, H, out):
    lines = [r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}", r"\toprule",
             "Dataset & " + " & ".join(COLUMNS) + r" \\", r"\midrule"]
    for block, metric in (("Test MSE", "test_mse"), ("Train MSE", "train_mse")):
        lines.append(rf"\multicolumn{{{len(COLUMNS) + 1}}}{{l}}{{\emph{{{block}}}}} \\")
        for d in datasets:
            g = sel[(sel.dataset == d) & (sel.H == H)].groupby("name")[metric].agg(["mean", "std"])
            shown = g.loc[[c for c in COLUMNS if c in g.index]]
            best = shown["mean"].min() if metric == "test_mse" and not shown.empty else None
            cells = [fmt(g.loc[c, "mean"], g.loc[c, "std"], best is not None and np.isclose(g.loc[c, "mean"], best))
                     if c in g.index else "--" for c in COLUMNS]
            lines.append(label(d) + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if metric == "test_mse" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    out.write_text("\n".join(lines) + "\n")


def significance(df, sel, datasets, H):
    rows = []
    for d in datasets:
        loss = {n: window_losses(sel, d, H, n) for n in ("A", "B", "C", "D", "RLinear")}
        if any(loss[n] is None for n in ("A", "B", "D")):
            continue
        row = dict(dataset=d)
        for ref in ("B", "A", "C"):
            if loss[ref] is None:
                continue
            stat, p = dm_test(loss[ref], loss["D"], H)
            row[f"gain_vs_{ref}"] = 100 * (loss[ref].mean() - loss["D"].mean()) / loss[ref].mean()
            row[f"dm_vs_{ref}"], row[f"p_vs_{ref}"] = stat, p
        stat, p = dm_test(loss["A"], loss["B"], H)
        row["gain_B_vs_A"], row["p_B_vs_A"] = 100 * (loss["A"].mean() - loss["B"].mean()) / loss["A"].mean(), p
        if loss["RLinear"] is not None:
            row["gain_D_vs_RLinear"] = 100 * (loss["RLinear"].mean() - loss["D"].mean()) / loss["RLinear"].mean()
        # per-seed paired direction (same seed => same init and batches)
        s = sel[(sel.dataset == d) & (sel.H == H)]
        b = s[s.name == "B"].set_index("seed").test_mse
        dd = s[s.name == "D"].set_index("seed").test_mse
        common = b.index.intersection(dd.index)
        row["seeds_D_better_B"] = f"{int((dd[common] < b[common]).sum())}/{len(common)}"
        rows.append(row)
    sig = pd.DataFrame(rows)
    if not sig.empty:
        sig["holm_vs_B"] = holm(sig.p_vs_B)
        sig["holm_vs_A"] = holm(sig.p_vs_A)
    return sig


def significance_table(sig, out):
    lines = [r"\begin{tabular}{lrrrrcrrr}", r"\toprule",
             r" & \multicolumn{5}{c}{D vs B (per-head scales)} & \multicolumn{3}{c}{D vs A (standard RoPE)} \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-9}",
             r"Dataset & $\Delta$MSE\% & DM & $p$ & Holm $p$ & seeds & $\Delta$MSE\% & $p$ & Holm $p$ \\", r"\midrule"]
    for _, r in sig.iterrows():
        lines.append(f"{label(r.dataset)} & {r.gain_vs_B:+.2f} & {r.dm_vs_B:+.2f} & {fmt_p(r.p_vs_B)} & "
                     f"{fmt_p(r.holm_vs_B)} & {r.seeds_D_better_B} & {r.gain_vs_A:+.2f} & {fmt_p(r.p_vs_A)} & "
                     f"{fmt_p(r.holm_vs_A)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def base_table(df, datasets, H, out):
    """Per-base comparison (tests P1 and the default-base setting). Both bases are reported."""
    rows = []
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{tuned-range base 100} & \multicolumn{4}{c}{default base $10^4$} & \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}",
             r"Dataset & D vs B & $p$ & D vs B & $p$ & D vs A & $p$ & D$_{10^4}$ vs A$_{100}$ \\", r"\midrule"]
    for d in datasets:
        L = {(n, b): window_losses(df, d, H, n, b) for n in "ABD" for b in (100.0, 10000.0)}
        if any(v is None for v in L.values()):
            continue
        g = lambda ref, new: 100 * (ref.mean() - new.mean()) / ref.mean()
        row = dict(dataset=d)
        for b in (100.0, 10000.0):
            for n in "ABD":
                row[f"{n}_{b:g}"] = float(L[(n, b)].mean())
            row[f"gain_DvsB_{b:g}"] = g(L[("B", b)], L[("D", b)])
            row[f"p_DvsB_{b:g}"] = dm_test(L[("B", b)], L[("D", b)], H)[1]
            row[f"gain_DvsA_{b:g}"] = g(L[("A", b)], L[("D", b)])
            row[f"p_DvsA_{b:g}"] = dm_test(L[("A", b)], L[("D", b)], H)[1]
            row[f"gain_BvsA_{b:g}"] = g(L[("A", b)], L[("B", b)])
        row["gain_D1e4_vs_A100"] = g(L[("A", 100.0)], L[("D", 10000.0)])
        rows.append(row)
        lines.append(f"{label(d)} & {row['gain_DvsB_100']:+.2f} & {fmt_p(row['p_DvsB_100'])} & "
                     f"{row['gain_DvsB_10000']:+.2f} & {fmt_p(row['p_DvsB_10000'])} & "
                     f"{row['gain_DvsA_10000']:+.2f} & {fmt_p(row['p_DvsA_10000'])} & "
                     f"{row['gain_D1e4_vs_A100']:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    return pd.DataFrame(rows)


def groups_table(sel, datasets, H, out):
    """Ablation over the number of groups: D (G=2) vs E (G=4), each at its validation-selected base."""
    rows, lines = [], [r"\begin{tabular}{lcccrr}", r"\toprule",
                       r"Dataset & B ($G{=}1$) & D ($G{=}2$) & E ($G{=}4$) & $\Delta$\% E vs D & $p$ \\", r"\midrule"]
    for d in datasets:
        e = window_losses(sel, d, H, "E")
        if e is None:
            continue
        b, dd = window_losses(sel, d, H, "B"), window_losses(sel, d, H, "D")
        gain = 100 * (dd.mean() - e.mean()) / dd.mean()
        p = dm_test(dd, e, H)[1]
        rows.append(dict(dataset=d, B=b.mean(), D=dd.mean(), E=e.mean(), gain_E_vs_D=gain, p=p))
        lines.append(f"{label(d)} & {b.mean():.3f} & {dd.mean():.3f} & {e.mean():.3f} & {gain:+.2f} & {fmt_p(p)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if rows:
        out.write_text("\n".join(lines) + "\n")
    return pd.DataFrame(rows)


def seed_level(df, sel, datasets, H):
    """Paired across (dataset, seed): relative test-MSE reduction and Wilcoxon signed-rank test.

    This captures training noise (each seed is one full training run), unlike the window-level DM test.
    """
    rows = []
    settings = [("selected base", sel, None), ("base 100", df, 100.0), ("base 10000", df, 10000.0)]
    for setting, frame, base in settings:
        for ref, new in (("A", "B"), ("A", "D"), ("B", "D"), ("D", "C")):
            diffs = []
            for d in datasets:
                f = frame[(frame.dataset == d) & (frame.H == H)]
                if base is not None:
                    f = f[f.base == base]
                r = f[f.name == ref].set_index("seed").test_mse
                n = f[f.name == new].set_index("seed").test_mse
                for seed in r.index.intersection(n.index):
                    diffs.append(100 * (r[seed] - n[seed]) / r[seed])
            if len(diffs) < 3:
                continue
            diffs = np.array(diffs)
            p = stats.wilcoxon(diffs).pvalue if np.any(diffs != 0) else 1.0
            rows.append(dict(setting=setting, comparison=f"{new} vs {ref}", pairs=len(diffs),
                             mean_gain=diffs.mean(), median_gain=float(np.median(diffs)),
                             wins=int((diffs > 0).sum()), p_wilcoxon=float(p)))
    return pd.DataFrame(rows)


def seed_level_table(sl, out):
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Setting & Comparison & pairs & mean $\Delta$\% & wins & Wilcoxon $p$ \\", r"\midrule"]
    for _, r in sl.iterrows():
        lines.append(f"{r.setting} & {r.comparison} & {r.pairs} & {r.mean_gain:+.2f} & {r.wins}/{r.pairs} & {fmt_p(r.p_wilcoxon)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def scale_lr_table(full, datasets, H, out):
    """Ablation: RoPE log-scale learning rate 1e-2 (main) vs larger, for B and D at each base."""
    other = sorted(set(full.scale_lr.dropna()) - {1e-2})
    if not other:
        return pd.DataFrame()
    rows = []
    for slr in [1e-2] + other:
        for base in (100.0, 10000.0):
            diffs_db = []
            for d in datasets:
                f = full[(full.dataset == d) & (full.H == H) & (full.base == base) & np.isclose(full.scale_lr, slr)]
                b = f[f.name == "B"].set_index("seed").test_mse
                dd = f[f.name == "D"].set_index("seed").test_mse
                a = full[(full.dataset == d) & (full.H == H) & (full.base == base) & (full.name == "A")].set_index("seed").test_mse
                for seed in b.index.intersection(dd.index).intersection(a.index):
                    diffs_db.append((d, seed, 100 * (b[seed] - dd[seed]) / b[seed], 100 * (a[seed] - dd[seed]) / a[seed],
                                     100 * (a[seed] - b[seed]) / a[seed]))
            if not diffs_db:
                continue
            arr = np.array([x[2:] for x in diffs_db])
            rows.append(dict(scale_lr=slr, base=base, pairs=len(arr),
                             D_vs_B=arr[:, 0].mean(), p_D_vs_B=float(stats.wilcoxon(arr[:, 0]).pvalue),
                             D_vs_A=arr[:, 1].mean(), p_D_vs_A=float(stats.wilcoxon(arr[:, 1]).pvalue),
                             B_vs_A=arr[:, 2].mean(), p_B_vs_A=float(stats.wilcoxon(arr[:, 2]).pvalue)))
    t = pd.DataFrame(rows)
    lines = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
             r"scale lr & base & pairs & D vs B & $p$ & D vs A & $p$ & B vs A & $p$ \\", r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.scale_lr:g} & {r.base:g} & {r.pairs} & {r.D_vs_B:+.2f} & {fmt_p(r.p_D_vs_B)} & "
                     f"{r.D_vs_A:+.2f} & {fmt_p(r.p_D_vs_A)} & {r.B_vs_A:+.2f} & {fmt_p(r.p_B_vs_A)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    return t


def scale_summary(df, datasets, H, base=10000.0):
    """Mean final log-scale per group for D (and B) at the given base, over layers, heads, seeds."""
    rows = []
    for d in datasets:
        for name in ("D", "B"):
            sel = df[(df.dataset == d) & (df.H == H) & (df.name == name) & (df.base == base)]
            finals = []
            for p in sel.path:
                r = json.loads((Path(p) / "result.json").read_text())
                best_step = r["best"]["step"]
                snap = next((s for s in r["scales"] if s["step"] == best_step), r["scales"][-1])
                finals.append(np.array(snap["log_scales"]))  # [layers, rows, groups]
            if finals:
                arr = np.stack(finals)
                row = dict(dataset=d, name=name)
                for g in range(arr.shape[-1]):
                    row[f"log_scale_g{g}"] = float(arr[..., g].mean())
                    row[f"scale_g{g}"] = float(np.exp(arr[..., g]).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def gain_figure(sig_like, sel, datasets, H, out):
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    names, colors = ("B", "C", "D"), ("#9aa5b1", "#6c8ebf", "#d0563b")
    present = [d for d in datasets if not sel[(sel.dataset == d) & (sel.H == H) & (sel.name == "A")].empty]
    x = np.arange(len(present))
    for k, (n, c) in enumerate(zip(names, colors)):
        vals = []
        for d in present:
            a = sel[(sel.dataset == d) & (sel.H == H) & (sel.name == "A")].test_mse.mean()
            v = sel[(sel.dataset == d) & (sel.H == H) & (sel.name == n)].test_mse.mean()
            vals.append(100 * (a - v) / a if np.isfinite(v) else np.nan)
        if np.all(np.isnan(vals)):
            continue  # variant not run (yet)
        ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=c, label={"B": "B per-head", "C": "C shared groups",
                                                                 "D": "D grouped (ours)"}[n])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [label(d).replace(" (fin.)", "") for d in present], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("test-MSE reduction vs A (%)", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def trajectory_figure(df, datasets, H, out, base=10000.0):
    present = [d for d in datasets if not df[(df.dataset == d) & (df.H == H) & (df.name == "D") & (df.base == base)].empty]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(1.9 * len(present), 3.0), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], present):
        for name, style in (("D", "-"), ("B", ":")):
            sel = df[(df.dataset == d) & (df.H == H) & (df.name == name) & (df.base == base) & (df.seed == 0)]
            if sel.empty:
                continue
            r = json.loads((Path(sel.path.iloc[0]) / "result.json").read_text())
            steps = [s["step"] for s in r["scales"]]
            arr = np.array([s["log_scales"] for s in r["scales"]])  # [T, layers, rows, groups]
            for layer, color in zip(range(arr.shape[1]), ("#d0563b", "#3b6fd0")):
                for g in range(arr.shape[3]):
                    curve = np.exp(arr[:, layer, :, g]).mean(axis=1)
                    lbl = None
                    if name == "D":
                        lbl = f"D layer {layer + 1}, {'fast' if g == 0 else 'slow'} group"
                    elif g == 0:
                        lbl = f"B layer {layer + 1}"
                    ax.plot(steps, curve, style, color=color, alpha=1.0 if g == 1 or name == "B" else 0.45, lw=1.2, label=lbl)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
        ax.set_title(label(d).replace(" (fin.)", ""), fontsize=10)
        ax.set_xlabel("step", fontsize=9)
        ax.tick_params(which="both", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("learned scale exp(a)", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="restrict to these seeds (default: all)")
    p.add_argument("--trajectory-datasets", nargs="+", default=["synthetic_multiscale", "traffic", "crypto_hourly", "exchange_rate"])
    args = p.parse_args()
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    full = collect(args.runs)
    if full.empty:
        raise SystemExit("no finished runs")
    if args.seeds is not None:
        full = full[full.seed.isin(args.seeds)]
    full.to_csv(args.out / "all_runs.csv", index=False)
    # main protocol: RoPE log-scale lr 1e-2; other scale learning rates are a separate ablation
    df = full[(full.scale_lr.isna() | np.isclose(full.scale_lr, 1e-2)) & (full.steps == 3000)].copy()
    datasets = [d for d in ORDER if d in set(df.dataset)] + sorted(set(df.dataset) - set(ORDER))
    H = args.horizon
    sel = select_bases(df)
    sel.to_csv(args.out / "selected_runs.csv", index=False)

    main_table(sel, datasets, H, args.out / "tables" / "main.tex")
    sig = significance(df, sel, datasets, H)
    sig.to_csv(args.out / "significance.csv", index=False)
    if not sig.empty:
        significance_table(sig, args.out / "tables" / "significance.tex")
    base = base_table(df, datasets, H, args.out / "tables" / "base.tex")
    base.to_csv(args.out / "base_ablation.csv", index=False)
    sl = seed_level(df, sel, datasets, H)
    sl.to_csv(args.out / "seed_level.csv", index=False)
    if not sl.empty:
        seed_level_table(sl, args.out / "tables" / "seed_level.tex")
    slr = scale_lr_table(full, datasets, H, args.out / "tables" / "scale_lr.tex")
    slr.to_csv(args.out / "scale_lr_ablation.csv", index=False)
    groups = groups_table(sel, datasets, H, args.out / "tables" / "groups.tex")
    groups.to_csv(args.out / "groups_ablation.csv", index=False)
    scales = scale_summary(df, datasets, H)
    scales.to_csv(args.out / "scale_summary.csv", index=False)
    gain_figure(sig, sel, datasets, H, args.out / "figures" / "gain_by_dataset.pdf")
    trajectory_figure(df, [d for d in args.trajectory_datasets if d in datasets], H,
                      args.out / "figures" / "scale_trajectories.pdf")

    chosen = sel[sel.name.isin(VARIANT.values())].groupby(["dataset", "name"]).base.first().unstack()
    summary = dict(
        runs=len(df), datasets=datasets,
        selected_base=chosen.to_dict(orient="index"),
        mean_test_mse=sel.groupby(["dataset", "name"]).test_mse.mean().unstack().round(4).to_dict(orient="index"),
        mean_train_mse=sel.groupby(["dataset", "name"]).train_mse.mean().unstack().round(4).to_dict(orient="index"),
        significance=sig.round(4).to_dict(orient="records"),
        base_ablation=base.round(4).to_dict(orient="records"),
        scales=scales.round(4).to_dict(orient="records"),
        groups_ablation=groups.round(4).to_dict(orient="records"),
        seed_level=sl.round(4).to_dict(orient="records"),
        scale_lr_ablation=slr.round(4).to_dict(orient="records"),
        gpu_seconds=float(df.seconds.sum()),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    pd.set_option("display.width", 200)
    print(sel.groupby(["dataset", "name"]).test_mse.mean().unstack().reindex(columns=[c for c in COLUMNS if c in set(sel.name)]).round(4))
    if not sig.empty:
        print(sig[["dataset", "gain_vs_B", "p_vs_B", "holm_vs_B", "seeds_D_better_B", "gain_vs_A", "p_vs_A"]].round(4))
    print(base.round(4))
    print(scales.round(3))
    print(sl.round(4).to_string())
    if not slr.empty:
        print(slr.round(4).to_string())


if __name__ == "__main__":
    main()
