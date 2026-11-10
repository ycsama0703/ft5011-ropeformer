# Findings of the RoPE frequency-parameterisation study

Everything here is computed from the 1462 deterministic runs in `outputs/all_runs.csv`; `outputs/digest.md` holds every number with its test. Section numbers such as "digest §3" refer to that file.

## 中文速览

- **主结论 1：可学习的 RoPE 频率倍数稳定优于固定 RoPE。** 默认 base（1e4）下，逐点 token 版本 D 对 A 提升 1.68%，39 对里赢 34 对，p<0.001；patch 版本提升 0.79%。换成 5 个种子结论不变，base 100 下也显著。
- **主结论 2：每个注意力头要有自己的倍数。** 各头共用倍数的 C 比 D 差 1.65%（p<0.001）。
- **主结论 3：分两组（G=2，原提案 D）相对"每头一个倍数"（B）没有显著收益，P1 不成立；但每对频率单独一个倍数（G=8，F）显著优于 B**，336 步窗口上约 +0.8%，两个 base 下都是 p<0.05。42 个 token 的 patch 骨架上，分几组都没差别。
- **机制：**
  - 在 1e4 下，第 6、7 对频率在 336 步窗口里几乎不转，模型完全没有用到它们。推理时冻结它们，误差只变 0.1%；冻结第 4–7 对则误差 +71%。
  - 一个共用的慢组倍数没法同时照顾"用到的第 4、5 对"和"没用到的第 6、7 对"，所以 G=2 不够，G=8 才有效。
  - 倍数只学到约 0.9–1.5 倍。把倍数的学习率调大 5 倍，结果不变。
- **训练预算：** 把 patch 骨架的训练加到 4 倍（12000 步）后，RoPE 各版本的差别消失。同时 Transformer 在 13 个数据集里的 8 个上超过 RLinear，PEMS 上好 30–45%，加密货币上好 3.8%。
- **统计教训：** 起初用非确定性算子跑的 ETTh1 上，D 对 B 是 +9.0%（DM 检验 p<0.001）；改用确定性算子重跑后是 −0.01%。所以统计推断以种子级 Wilcoxon 为准，DM 检验只作描述。
- **和队长 `tools/` 流程可比：** 测试集逐位相同。Naive 在 3 个数据集上的 MSE，两边一致到 1e-6。可以用 `study/merge_external.py` 把两边结果合成一张表。

## 1. Question and predictions

RoPE rotates query/key pair $j$ by $t\,\omega_j$ with $\omega_j=b^{-j/m}$ ($d_{model}=64$, 4 heads, $m=8$ pairs per head). The variants are:

| Variant | Formula | Scales per layer |
|---|---|---|
| A, standard | $\hat\omega=\omega_j$ | 0 |
| B, per-head | $\omega_j e^{a_{\ell,h}}$ | $H$ |
| C, groups shared across heads | $\omega_j e^{a_{\ell,g(j)}}$ | $G$ |
| D, grouped (original proposal) | $\omega_j e^{a_{\ell,h,g(j)}}$, $G=2$ contiguous groups | $HG$ |
| E | $G=4$ | |
| F | $G=8$, one scale per pair | |
| S, p-RoPE diagnostic | slow half not rotated | 0 |

All scales start at 1, so every variant starts as standard RoPE.

**Geometry.** With $L=336$ and $b=10^4$:
- the fast group has periods of 6.3–199 steps;
- the slow group's periods are 628–19,869 steps, so across the window it rotates by at most 3.4, 1.1, 0.34 and 0.11 rad.

Per-head scaling cannot speed up the slow pairs without pushing pair 0 beyond the Nyquist angle π: a factor of about 6 gives about 6 rad/step. The ladder fits the window at $b^*=(L/2\pi)^{m/(m-1)}$, which is about 94 for $L=336$ and about 9 for 42 patch tokens.

Predictions, stated before the experiments:
- **P1:** the gain of D over B is larger at $b=10^4$ than at the window-matched base.
- **P2:** gains appear on multi-periodic data, not on a random-walk control.
- **P3:** the slow-group scales move further from 1 than the fast-group scales.

## 2. Setup (identical for every variant)

**Datasets (13).**
- Eleven public benchmarks:
  - ETTh1/h2 and ETTm1/m2, 7 channels;
  - electricity, 321 channels;
  - traffic, 862 channels;
  - weather, 21 channels;
  - solar, 137 channels;
  - exchange_rate, 8 channels: the random-walk control;
  - PEMS04 and PEMS08, 307 and 170 channels.
- A synthetic mechanism set: 16 channels, period-24 and period-168 cycles plus AR(1) noise.
- `crypto_hourly`, built from Binance public spot klines for 8 USDT pairs, 2021-08-13 to 2026-10-31. It has 16 channels: per pair, log Parkinson volatility and log quote volume.
- Provenance is in `docs/data_sources.md`.

**Splits.** ETT 12/4/4 months, PEMS 60/20/20, others 70/10/20. Standardisation uses the training split only. Every test window and channel is evaluated; metrics are MSE/MAE on the standardised scale.

**Backbones.** All use channel independence plus per-window instance normalisation (RevIN without affine).
- *Point tokens*: $L=336$ tokens; `GroupedRoPEForecaster`, 2 layers, `forecast_head="pooled"`, last-token pooling.
- *Patch tokens*: overlapping patches of 16 with stride 8, giving 42 tokens; same core.

**Training.**
- AdamW at lr 1e-3, weight decay 0.01; RoPE log-scales at lr 1e-2 with no decay.
- Batch 256, 3000 steps (12,000 in the long-budget patch runs), warm-up then cosine decay to 10%, gradient clip 1.0.
- bf16 autocast, deterministic kernels.
- Early stopping on validation (every 250 steps, patience 4).

**Selection and seeds.** The base is chosen per variant on validation: {100, 1e4} for point tokens and {10, 1e4} for patch tokens. Seeds 0–2 everywhere; seeds 3–4 were added on 9 datasets.

**Statistics.** The confirmatory test is a Wilcoxon signed-rank test over (dataset, seed) pairs of relative test-MSE change. Per-dataset Diebold–Mariano tests (Newey–West lag 96, HLN, Holm) are descriptive only.

**Baselines.** Naive, seasonal naive, RLinear (instance norm + linear) and LSTM.

## 3. Results

### 3.1 Overall, point tokens at 3k steps (`outputs/tables/main.tex`; train and test MSE)

- RLinear is best on 8 of 13 datasets: synthetic, ETTm1/m2, ETTh1/h2, electricity, traffic and crypto (crypto by 0.5%).
- The Transformers are best on PEMS04, PEMS08, solar and weather.
- Naive is best on exchange_rate: 0.081 against ≥0.084 for every learned model.
- Train–test gaps are shared by RLinear and the Transformers; weather even has a higher train than test MSE. The gaps therefore reflect shifts between splits, not over-fitting.
- The Transformer deficit on electricity and traffic is under-training: 3k steps is only 0.13 and 0.075 epochs (section 5).

### 3.2 Learnable vs fixed frequencies (`outputs/tables/tests.tex`; digest §3, §8)

Wilcoxon over (dataset, seed) pairs; mean relative test-MSE reduction:

| Comparison | point, b=1e4 | point, b=100 | point, selected b | patch, b=1e4 | patch, b=10 | patch, 12k (selected) |
|---|---|---|---|---|---|---|
| B vs A | **+1.14%** (27/39, p=0.0015) | +0.34% (n.s.) | +0.45% (n.s.) | **+0.68%** (p=0.014) | +0.28% (n.s.) | −0.41% (n.s.) |
| D vs A | **+1.68%** (34/39, p<1e-4) | +0.48% (n.s.) | **+0.94%** (p=0.021) | **+0.79%** (p=0.005) | **+0.90%** (p=0.016) | −0.08% (n.s.) |
| D vs B | +0.52% (22/39, p=0.12) | +0.13% (n.s.) | +0.47% (n.s.) | +0.10% (n.s.) | +0.52% (n.s.) | +0.26% (n.s.) |
| C vs D | **−1.65%** (6/39, p<1e-4) | −0.59% (n.s.) | **−1.21%** (p=0.0004) | – | – | – |

With five seeds on nine datasets (45 pairs), learnable scales are significant at both bases:
- D vs A: +1.63% at 1e4 and +0.89% at b=100;
- B vs A: +1.52% and +0.72%;
- all p<0.01.

D vs B stays n.s. (+0.10%, +0.16%).

### 3.3 Number of groups (digest §3)

| Comparison | Point, b=100 | Point, b=1e4 | Point, selected b |
|---|---|---|---|
| **F (G=8) vs B** | **+0.86%** (21/27, p=0.012) | **+0.76%** (18/27, p=0.023) | +0.83% (p=0.019) |
| F vs D | +0.64% (p=0.012) | +0.70% (p=0.09) | |
| F vs A | +1.56% (p<0.001) | +2.42% (p<0.001) | |

- The F rows use 9 datasets × 3 seeds = 27 pairs.
- E (G=4) vs D is n.s. at every base (4 datasets, 12 pairs; `outputs/tables/groups.tex`).
- On patch tokens no group count matters: F vs D is −0.45% at 1e4 and +0.02% at b=10, both n.s.

The answer to the research question: relaxing within-head frequency ratios helps on the long point-token window, but only when it is fine-grained. Two groups are too coarse; per-pair scales work. This is consistent with LeRoPE (per-frequency scales; see docs/related_work.md).

### 3.4 Mechanism (digest §6, §7, §9)

**Learned scales.** In D at b=1e4 the slow-group scale ends at 0.92–1.54× (above 1 on 12 of 13 datasets) and the fast group at 0.93–1.23×. The slow group moves further from 1 on 11 of 13. This is far below the ×6 the geometry asks for.
- Raising the scale learning rate to 5e-2 changes nothing: D −0.14% at b=100 and +0.60% at 1e4 relative to the default, 27 pairs, n.s.

**Inference probe** (`probe_slow_group.py`, 912 checkpoints). Median test-MSE change when pairs stop rotating at inference:

| Backbone | Freeze pairs 6–7 | Freeze pairs 4–7 |
|---|---|---|
| A, b=1e4 | +0.1% | +71% |
| B, b=1e4 | +0.1% | +31% |
| D, b=1e4 | +0.2% | +40% |
| A, b=100 | +129% | +257% |
| D, b=100 | +95% | +199% |
| patch A/B/D, b=10 | +12–16% | +49–55% |
| patch, b=1e4 | 0.0% | 0.0–0.1% |

What matters is how far a pair turns across the window. Pairs 4–5 turn 3.4 and 1.1 rad over 336 steps and are used. Pairs 6–7 (≤0.34 rad) and the whole slow group on 42 tokens (≤0.42 rad) are not. One slow-group scale cannot serve both kinds of pair; per-pair scales can, which is why G=8 helps on point tokens only.

**p-RoPE (S: slow group frozen during training).**
- Point tokens: −2.16% vs A at b=100 (7/27, p<0.001) and −0.57% at 1e4 (n.s.).
- Patch tokens: harmless, +0.05% and +0.03%, even though rotating patch models rely on the slow group (+49–55% when frozen).
- So slow rotations are needed only on the long window; otherwise the slow pairs act as content channels (Barbero et al., ICLR 2025).

**P1** is not supported for G=2; the ordering is even reversed on patch tokens. **P3** holds in direction, not in size. **P2** holds partly: gains do not track periodicity strength, and the exchange-rate control is weak because its runs stop near initialisation (best step 250 in 34 of 38 point-token Transformer runs).

### 3.5 Training budget, patch tokens at 12k steps (`outputs/tables/patch.tex`)

At 12k steps all RoPE variants converge (D vs A −0.08%, B vs A −0.41%, D vs B +0.26%; all n.s.). Learnable scales therefore mainly speed up training under a limited budget.

The patch Transformer (D) beats RLinear on 8 of 13 datasets:

| Dataset | PD (12k) vs RLinear |
|---|---|
| PEMS08 | +45% |
| PEMS04 | +30% |
| weather | +15% |
| solar | +14% |
| traffic | +3.8% |
| crypto | +3.8% |
| exchange | +3.6% |
| synthetic | +0.8% |

RLinear keeps ETTh1 (−8%), ETTh2 (−15%), ETTm1 (−1.2%), ETTm2 (−8%) and electricity (−2.8%).

### 3.6 Finance

- **crypto_hourly** (intraday volatility and volume, daily and weekly seasonality):
  - the 12k patch Transformer reaches MSE 0.528, against 0.549 for RLinear (3.8% better) and 1.107 for naive (about 52% better);
  - on point tokens at 3k, F (G=8) reaches 0.5425 against RLinear's 0.5446;
  - learnable scales help: D vs A is +2.2% at the default base (descriptive DM p<0.001).
- **Exchange rates:** the naive forecast beats every learned model, consistent with weak-form efficiency and Meese–Rogoff.
- **Uses:** intraday VaR and volatility targeting, VWAP volume curves, market-making quote widths.

### 3.7 Statistics lesson

The first, non-deterministic run gave D vs B on ETTh1 at 1e4 as +9.0%, with window-level DM p<0.001. Deterministic reruns give −0.01%. Same-seed Transformer retraining with non-deterministic kernels moved the ETTh1 test MSE by up to 9%.

Window-level DM tests treat two fixed forecast sequences as given and ignore training noise. They are therefore not valid for comparing training procedures; the (dataset, seed) Wilcoxon test is.

## 4. Comparability with the repository's `tools/` pipeline

`tools/run_baselines.py` and `study/train.py` produce identical test sets (same number of test windows, naive test MSE equal to 1e-6):

| Dataset | Test samples | Naive MSE |
|---|---|---|
| ETTh1 | 19,495 | 1.294371 |
| electricity | 1,657,965 | 1.587837 |
| exchange_rate | 11,376 | 0.081126 |

Results of the two pipelines can therefore share a leaderboard on the 8 common datasets (ETTh1/2, ETTm1/2, weather, exchange_rate, electricity, traffic). Training protocols differ (budgets, heads, tokens), so RoPE-variant comparisons should stay within one backbone.

```bash
python study/merge_external.py --runs runs --benchmark results/benchmark/*.json \
    --baselines results/baselines/*.json --out results/tables/leaderboard.tex
```

## 5. Training epochs within the budget (256 sequences per step)

| Dataset | Epochs at 3k steps | Epochs at 12k steps |
|---|---|---|
| synthetic | 3.54 | 14.2 |
| ETTh1/2 | 13.4 | 53.5 |
| ETTm1/2 | 3.22 | 12.9 |
| electricity | 0.133 | 0.532 |
| traffic | 0.075 | 0.301 |
| solar | 0.154 | 0.617 |
| weather | 1.00 | 4.01 |
| crypto | 1.52 | 6.08 |
| exchange | 19.7 | 78.7 |
| PEMS04 | 0.256 | 1.03 |
| PEMS08 | 0.439 | 1.76 |

## 6. Limitations

- One small backbone (d_model 64, 2 layers), one lookback/horizon (336/96), two bases per backbone.
- Mostly 3 seeds; F and S cover 9 datasets, E covers 4.
- Point tokens were not run at 12k steps.
- The crypto feature has 2 floored hours per pair (Binance halt, 2023-03-24) in the training split.
- (dataset, seed) pairs share test sets within a dataset, so Wilcoxon p-values are somewhat optimistic.

## 7. Files in `outputs/`

| File | Content |
|---|---|
| `all_runs.csv` | Every run (1462): dataset, variant (`name`), backbone, base, seed, steps, scale lr, train/val/test MSE and MAE, best step, wall time, parameters, run path |
| `selected_runs.csv` | Seeds 0–2, validation-selected bases (basis of tables 1–3) |
| `digest.md` | Every number above, with the test behind it |
| `seed_level.csv` | Wilcoxon tests, point tokens (A/B/C/D) |
| `significance.csv`, `base_ablation.csv` | Per-dataset DM tests (descriptive) at the selected base and per base |
| `scale_summary.csv` | Learned fast/slow-group scales of B and D at b=1e4 |
| `scale_lr_ablation.csv` | Scale lr 5e-2 vs 1e-2 |
| `groups_ablation.csv` | G=4 vs G=2 |
| `probe_slow_group.csv` | Inference probe for every checkpoint |
| `summary.json` | Machine-readable summary from `aggregate.py` |
| `tables/*.tex` | LaTeX tables: `main` (train and test), `tests`, `patch`, `base`, `significance`, `groups`, `seed_level`, `scale_lr` |
| `figures/*.pdf` | Learned-scale trajectories; relative gain over A per dataset |

Run directories (`result.json`, per-window test errors, `model.pt`, ≈0.5 GB) and the data (≈0.7 GB) are not in Git. They can be regenerated with `study/run_all.sh`, or obtained from the study's author on request.
