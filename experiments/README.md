# Horizon-extrapolation experiments (FT5011 final report)

Run everything from the repository root. Scripts in `experiments/` expect a CUDA GPU; `capped.py` is an optional
wrapper that limits the process to 8% of GPU memory on a shared machine (`python experiments/capped.py <script> <args>`).
All results land in `results/` (Git-ignored). Python 3.10+, PyTorch >= 2.0.

## Trained models and results (no retraining needed)

Download `ft5011-horizon-extrapolation-results.zip` from the release
https://github.com/ycsama0703/ft5011-ropeformer/releases/tag/final-report-2026-11-10
and unzip it at the repository root. It contains every result file (`results/*.json`) and the 257 checkpoints
used in the report (`results/ckpt*/`, described in `results/RESULTS_README.md`). With it, step 4 below and
`python report/make_assets.py results` run without training anything.

**Expected variation when retraining.** GPU kernels are not deterministic, so single runs differ slightly
between machines. Retraining ETTh1 (step 1, 3 seeds x 2 bases) on an RTX 3090 gave a 720-step MSE of 0.982 (fixed RoPE)
and 0.463 (learnable) against 0.987 and 0.472 in the report (single runs within 3%). Re-evaluating the released
checkpoints reproduces the reported numbers to within 0.01%, also on CPU.

## 0. Setup and data
```bash
pip install -e .
python -m unittest discover -s tests        # 29 tests
python tools/download_data.py ETTh1 ETTm1 weather electricity exchange_rate
# financial data (EPF prices, Bitcoin): download into raw/, then build CSVs into data/
mkdir -p raw/epf raw/btc
curl -L -o raw/epf/NP.csv  "https://zenodo.org/records/4624805/files/NP.csv?download=1"
curl -L -o raw/epf/PJM.csv "https://zenodo.org/records/4624805/files/PJM.csv?download=1"
for y in 2019 2020 2021 2022 2023 2024; do for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  curl -sSf -o raw/btc/b.zip https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-$y-$m.zip && unzip -o -q raw/btc/b.zip -d raw/btc; done; done
python -I experiments/build_fin.py raw data
```

Common flags below: `C="--patch-len 16 --amp bf16 --seeds 0 1 2 --bases 1000 10000"`.
Step budgets S: ETTh1, epf_np, epf_pjm, exchange_rate 2000 (eval every 200); ETTm1 4000 (250); weather 6000 (500); electricity 8000 (500).

## 1. Train at H=96, extrapolate to 720, direct-training references (Table 2, Fig. 1)
```bash
python tools/extrap_study.py --dataset ETTm1 --steps 4000 --eval-every 250 $C --modes standard head grouped \
  --eval-horizons 192 336 720 --direct-horizons 336 720 --direct-modes head --out results/r2_ETTm1.json
# same for weather (6000/500), electricity (8000/500) -> results/r2_<dataset>.json
python tools/extrap_study.py --dataset ETTh1 --steps 2000 --eval-every 200 $C --modes standard head \
  --eval-horizons 192 336 720 --direct-horizons 336 720 --direct-modes head --ckpt-dir results/ckpt_mh --out results/r2_ETTh1_720.json
python tools/extrap_study.py --dataset epf_np --steps 2000 --eval-every 200 $C --modes standard head \
  --eval-horizons 192 336 720 --direct-horizons 96 336 720 --direct-modes head --ckpt-dir results/ckpt_fin --out results/fin_epf_np.json
# same for epf_pjm -> results/fin_epf_pjm.json
python tools/extrap_study.py --dataset exchange_rate --instance-norm --steps 2000 --eval-every 200 $C --modes standard head \
  --eval-horizons 192 336 720 --direct-horizons 96 336 720 --direct-modes head --ckpt-dir results/ckpt_fxn --out results/r2_exchange_rate_norm.json
```

## 2. Multi-horizon training (Table 2, Fig. 1)
```bash
python tools/multi_horizon.py --dataset ETTm1 --steps 4000 --eval-every 250 $C --modes standard head \
  --train-horizons 96 192 336 720 --test-horizon 720 --out results/mh_ETTm1.json
# ETTh1/weather/epf_np/epf_pjm (mhfin_<d>.json) analogously; electricity used --bases 10000 only;
# exchange_rate with --instance-norm -> results/mh_exchange_rate_norm.json
```

## 3. Mechanism ablations (Table 3, Fig. 2)
```bash
python tools/extrap_study.py --dataset ETTh1 --steps 2000 --eval-every 200 --patch-len 16 --amp bf16 --seeds 0 1 2 3 4 \
  --bases 1000 10000 --modes standard head grouped --eval-horizons 192 336 --direct-horizons 336 --direct-modes standard head \
  --out results/study_ETTh1_p16.json
python tools/extrap_study.py --dataset ETTh1 --steps 2000 --eval-every 200 --patch-len 16 --amp bf16 --seeds 0 1 2 3 4 \
  --bases 1000 10000 --modes fixed_head:0.9,0.97,1.03,1.1 fixed_head:0.5,0.8,1.25,2.0 shared_group --out results/r2_ETTh1_controls.json
python tools/extrap_study.py --dataset ETTm1 --steps 4000 --eval-every 250 $C --eval-horizons 192 336 720 \
  --modes fixed_head:1.0472,1.0472,1.0472,1.0472 fixed_head:0.955,0.955,0.955,0.955 fixed_head:1.10,1.10,1.10,1.10 \
  --ckpt-dir results/ckpt_r3 --out results/r3_ETTm1_align.json
```

## 4. Analysis (Table 1, Fig. 3, significance)
```bash
python experiments/linear96.py --dataset ETTm1 --steps 4000 --out results/lin96_ETTm1.json          # each dataset, its budget
python experiments/train_eval.py --dataset ETTm1 --out results/trainval_ETTm1.json                # --ckpt-dir for EPF/exchange
python experiments/dm_test.py --dataset ETTm1 --out results/dm_ETTm1.json                          # Diebold-Mariano, fixed vs learnable
python experiments/dm_test.py --dataset ETTm1 --ckpt-dir results/ckpt_r3 --a fixed_head:0.955,0.955,0.955,0.955 \
  --b fixed_head:1.0472,1.0472,1.0472,1.0472 --out results/dm_ETTm1_align.json
python experiments/per_channel.py --dataset weather --period 144 --json results/perchan_weather.json
python experiments/per_channel.py --dataset ETTm1 --period 96 --json results/perchan_ETTm1.json
python experiments/periodicity.py
python report/make_assets.py results       # writes report/figures and report/tables
```
Checkpoint directories: `results/ckpt` (ETTh1 ablations, ETTm1, weather, electricity), `results/ckpt_mh` (ETTh1, 720),
`results/ckpt_fin` (EPF), `results/ckpt_fxn` (exchange, RevIN), `results/ckpt_r3` (alignment).
