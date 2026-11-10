# RoPE frequency-parameterisation study (`study/`)

This directory holds a complete benchmark study of RoPE frequency parameterisations: code, results and documentation. **Start with [FINDINGS.md](FINDINGS.md)** (all conclusions and numbers); `outputs/` holds every table, test and per-run result. It reuses the package's
`GroupedRoPEForecaster` with `forecast_head="pooled", pooling="last"` (the earlier pooled
backbone, kept in `model.py` as the head ablation). Additional pieces live in two places:
- the self-contained benchmark pipeline in `src/grouped_rope/study/`:
  - `data.py`: standard splits, train-only scaling, windows;
  - `models.py`: the channel-independent + instance-norm wrapper, overlapping-patch tokens (16/8), RLinear, LSTM, naive, seasonal naive;
- one extra diagnostic RoPE mode in `model.py`: `rope_mode="partial"` (p-RoPE: the slow half of the pairs is not rotated).

Nothing else in the package is changed. The study does not use `tools/` or the horizon-token head.

## Verified replication
Run on an A100 with PyTorch 2.15.0+cu132.
- Loading the study's saved `model.pt` files with this code reproduces the stored train/val/test MSE exactly (`max |diff| = 0`). This was checked for every variant (A/B/C/D/F/S, patch, 12k steps, RLinear, LSTM).
- Retraining reproduces them bit for bit (`--deterministic 1`, the default).

## Commands (from the repository root)
```bash
python study/build_data.py --out data                    # download + verify + build 12 public/crypto CSVs
python study/make_synthetic.py --out data/synthetic_multiscale.csv
GPUS=0 WORKERS=6 bash study/run_all.sh                   # every experiment phase, then tables/tests/probe/digest
python study/evaluate.py runs/<dataset>/L336_H96/<run>/s0   # recompute one run's metrics from its model.pt
```

The scripts, in the order `run_all.sh` uses them:

| Script | Role |
| --- | --- |
| `train.py` | One run: early stopping on validation, single test evaluation. Writes `result.json`, per-window test errors and `model.pt` |
| `sweep.py` | Grid runner over datasets × variants × bases × seeds. Skips finished runs; works with several processes per GPU |
| `aggregate.py` | Validation-selected base, main table, seed-level Wilcoxon, descriptive per-dataset DM tests, figures |
| `probe_slow_group.py` | Inference probe: stops the rotation of slow pairs in trained checkpoints |
| `digest.py` | Writes `results/digest.md` (every number in FINDINGS.md) and the tests/patch tables |
| `merge_external.py` | Leaderboard combining this study with `tools/run_benchmark.py` / `tools/run_baselines.py` outputs (identical test sets, verified) |

Variants map to `--rope-mode` as follows:
- `standard` (A), `head` (B), `shared_group` (C), `grouped` (D);
- `grouped4` / `grouped8` give G=4/8 (E/F);
- `partial` is S.
- The patch backbone is `--model patch_transformer`.

Data provenance (URLs, sha256, processing) is in `docs/data_sources.md`.

## Layout

```text
study/
  FINDINGS.md            conclusions, key numbers, caveats (start here)
  README.md              this file: how to reproduce
  outputs/               shipped results (tables, tests, per-run CSV, probe, figures, digest)
  docs/related_work.md   verified related work (RoPE frequency scaling, p-RoPE, TS Transformers)
  *.py, run_all.sh       pipeline (data build -> sweeps -> aggregate -> probe -> digest -> merge)
src/grouped_rope/study/  data pipeline and wrappers/baselines used by the scripts
docs/data_sources.md     dataset URLs, sha256 and processing steps
```

`study/run_all.sh` writes to `runs/` and `results/` (ignored by Git); the committed copies of the
results are in `study/outputs/`.
