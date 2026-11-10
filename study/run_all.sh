#!/usr/bin/env bash
# Full replication: data -> all experiment phases -> tables, tests, figures, probe, digest.
# Run from the repository root. GPU strongly recommended (one A100: roughly 10-15 hours in total;
# phases are independent and can be split across GPUs with GPUS="0 1"). Finished runs are skipped.
set -euo pipefail
GPUS=${GPUS:-0}
WORKERS=${WORKERS:-6}
DATA=${DATA:-data}
P1="synthetic_multiscale ETTm1 crypto_hourly weather exchange_rate ETTh1 solar electricity traffic"
ALL="$P1 ETTh2 ETTm2 PEMS04 PEMS08"
SWEEP="python study/sweep.py --horizons 96 --data-dir $DATA --runs runs --gpus $GPUS --workers $WORKERS"

# 1. Data: download + verify (checksums, shapes) + build the CSVs; then the synthetic set.
python study/build_data.py --out "$DATA"
python study/make_synthetic.py --out "$DATA/synthetic_multiscale.csv"

# 2. Point-token backbone (L=336 tokens), 3000 steps, bases 100 and 1e4, seeds 0-2.
$SWEEP --datasets $P1 --modes standard head grouped                     # A, B, D + all baselines
$SWEEP --datasets $P1 --modes shared_group --baselines                  # C
$SWEEP --datasets ETTh2 ETTm2 PEMS04 PEMS08                              # A, B, C, D + baselines
$SWEEP --datasets synthetic_multiscale ETTm1 crypto_hourly weather --modes grouped4 --baselines   # E (G=4)
$SWEEP --datasets $P1 --modes grouped8 --baselines                      # F (G=8)
$SWEEP --datasets $P1 --modes partial --baselines                       # S (p-RoPE: slow group frozen)

# 3. Robustness on the point-token backbone.
$SWEEP --datasets $P1 --modes head grouped --baselines --scale-lr 0.05  # RoPE scale lr 5e-2
$SWEEP --datasets $P1 --modes standard head grouped --seeds 3 4 --baselines linear lstm   # seeds 3-4

# 4. Patch-token backbone (patch 16, stride 8 -> 42 tokens), bases 10 and 1e4, seeds 0-2.
$SWEEP --model patch_transformer --datasets $ALL --modes standard head grouped grouped8 --bases 10 10000 --baselines
$SWEEP --model patch_transformer --datasets $ALL --modes partial --bases 10 10000 --baselines
$SWEEP --model patch_transformer --datasets $ALL --modes standard head grouped --bases 10 10000 \
  --baselines linear --steps 12000                                       # 4x training budget

# 5. Tables, significance tests, figures (seeds 0-2), inference probe, and the results digest.
python study/aggregate.py --runs runs --out results --seeds 0 1 2
python study/probe_slow_group.py --runs runs --out results/probe_slow_group.csv --test-cap 50000
python study/digest.py --runs runs --probe results/probe_slow_group.csv --out results/digest.md
