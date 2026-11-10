# Learning the Rhythm: RoPE beyond the training horizon

FT5011 Deep Learning in Finance, final project (NUS, 2026).
Liu Yuncong, Zhang Yihang, Ma Xiaoyu.

**Report:** [`report/main.pdf`](report/main.pdf) (5 pages + replication appendix).

A Transformer forecaster that represents every future step by a placeholder token can be trained at one horizon
and queried at any other. With standard rotary position encoding (RoPE) it degrades sharply beyond the training
horizon, because RoPE's fastest rotation is out of step with the daily cycle. Learnable per-head frequency scales
(4 parameters per layer) realign the rotation and cut the 720-step error by 38-66% on seasonal benchmarks.

## Repository layout

```text
src/grouped_rope/     model (GroupedRoPEForecaster, RoPE scales, placeholder head) and data pipeline
tests/                unit tests: RoPE invariants, horizon independence, splits and normalisation
tools/                training and evaluation
  download_data.py      benchmark CSVs from the Time-Series-Library dataset
  run_benchmark.py      shared training loop (AdamW, warm-up + cosine, validation-based checkpointing)
  extrap_study.py       train at H=96, evaluate up to H=720, direct-training references
  multi_horizon.py      one model trained on mixed horizons
  smoke_test.py         forward/backward check of every RoPE variant
experiments/          financial data builder, train/test MSE, linear reference, Diebold-Mariano tests,
                      per-channel analysis, seasonal strength (commands in experiments/README.md)
report/               report source (main.tex), PDF, tables, figures and the scripts that make them
```

## Setup

```bash
pip install -e .
pip install -r requirements.txt
python -m unittest discover -s tests
```

Python 3.10+, PyTorch 2.0+. All commands run from the repository root.

## Reproduce the report without training (CPU is enough)

1. Download `ft5011-horizon-extrapolation-results.zip` (71 MB) from the release
   <https://github.com/ycsama0703/ft5011-ropeformer/releases/tag/final-report-2026-11-10>
   and unzip it at the repository root. It contains every result file (`results/*.json`) and the 257 trained
   checkpoints used in the report (`results/ckpt*/`, described in `results/RESULTS_README.md`).
2. Download the data (step 0 of [`experiments/README.md`](experiments/README.md)); ETTm1 is enough for the figures.
3. Regenerate the tables and figures:

```bash
python report/make_assets.py results                                  # Tables 1-3, Figs. 2-3
python report/phase_figures.py                                        # RoPE phase arithmetic
python report/mechanism_figures.py --results results --data-dir data  # Figs. 4-5 from the checkpoints (~10 s)
```

The regenerated tables are identical to `report/tables/`, and re-evaluating the checkpoints reproduces the
reported test errors to within 0.01%.

## Reproduce the report by retraining (GPU recommended)

Every training command, with its step budget, is listed in [`experiments/README.md`](experiments/README.md)
and in Appendix A of the report. One run takes 20 s to 3 min on a single GPU. RoPE base and checkpoints are
chosen on the validation split only; the test split is never used for training or tuning.

GPU kernels are not deterministic, so single runs differ slightly between machines. Retraining ETTh1
(3 seeds x 2 bases) on an RTX 3090 gave a 720-step MSE of 0.982 (fixed RoPE) and 0.463 (learnable scales)
against 0.987 and 0.472 in the report; single runs were within 3%.

## Data

- ETTh1, ETTm1, Weather, Electricity, exchange rates: Time-Series-Library (CC BY 4.0),
  <https://huggingface.co/datasets/thuml/Time-Series-Library>, via `python tools/download_data.py`.
- Day-ahead electricity prices (Nord Pool, PJM; hourly 2013-2018): <https://zenodo.org/records/4624805>,
  built with `python -I experiments/build_fin.py raw data`.

Data, results and checkpoints are not stored in Git (`data/`, `results/` are ignored).

If `tools/download_data.py` fails with an SSL certificate error (common with the python.org macOS build), run
`Install Certificates.command` from your Python folder, or download the CSVs with `curl` from the URLs in the script.
