# AI collaboration guide

Read README.md and docs/experiments.md before changing the model. This is a
Deep Learning group project about frequency parameterization in RoPE for
continuous time-series forecasting. It is not an oil/finance application.

## Scientific scope

- Main hypothesis: independently scaling fixed groups of positional frequencies
  can be more useful than scaling the whole frequency ladder per head.
- Default main variant: grouped, G=2. All log-scales start at zero.
- Position frequencies are not input signal frequency estimates or band-pass filters.
- Do not claim head specialization, true-period recovery, novelty, or accuracy
  improvements without suitable evidence. Synthetic pipeline output is not a benchmark.
- Keep discussions of proposed extensions distinct from implemented features.
- Do not use earlier projects' experiment results as evidence for this project.

## Preserve the comparison

- standard/head/shared_group/grouped share the same backbone and output head.
- Every layer has its own scale parameters. shared_group shares within a layer,
  across heads; it does not share across layers.
- Groups are contiguous coordinate-pair groups in descending base-frequency
  order; group 0 is initially highest frequency. Use half-split coordinate pairs.
- Effective frequencies are base_frequency * exp(log_scale).
- Directly evaluate sin/cos at scaled positions. Never interpolate a finite cache.
- Keep the initial standard-RoPE equivalence, relative-position identity and
  norm preservation. Tying group scales must recover per-head scaling.
- Do not add regularizers, dynamic scales, FFT preprocessing, RevIN, attention
  gates, or architecture changes to only the proposed model without a separate,
  explicitly requested experiment. Do not silently tune only one variant.

## Code and data contracts

- Public package: grouped_rope; main class: GroupedRoPEForecaster.
- Input [batch, history, input_dim], output [batch, horizon, output_dim].
- Config arguments are explicit; reject unknown legacy keywords.
- No external dataset download at import time or model construction.
- Fit preprocessing only on training data. Target windows must not cross splits.
  Validation/test histories may use observations already available before origin.
- Do not change a benchmark split or normalize using the full series silently.
- Random seeds and matched budgets must be recorded for real comparisons.
- No import-time training, file writes, or global random seed changes in the package.
- Keep dependencies small; tests use standard-library unittest plus torch.

## Validation

Run from repository root:

```bash
python -m unittest discover -s tests -v
python tools/smoke_test.py
python examples/train_synthetic.py --mode grouped --steps 100
```

After a mathematical change, add a targeted invariant or reference calculation.
After an API change, update README, tools and examples together. Report exactly
which platforms were tested; do not imply CUDA/CI verification from CPU tests.

## State and workflow

- Model, tests, diagnostics and synthetic demo exist. Default forecast head is
  horizon_tokens (placeholder tokens + structured mask, see
  docs/plan_horizon_tokens.md); pooled is kept only as a head ablation and
  rope_mode="none" only as a NoPE diagnostic. Keep the horizon-independence
  and history-independence invariants tested.
- Benchmark data (grouped_rope.data, tools/download_data.py), the shared runner
  tools/run_benchmark.py (channel-independent, linear reference) and
  tools/summarize_benchmark.py exist for ETTh1/Electricity. External baselines (tools/baselines.py, tools/run_baselines.py) are compact
  re-implementations; tools/benchmark_suite.py runs the 8-dataset suite and
  tools/vast_job.py runs it on rented vast.ai GPUs (costs money; destroy after fetch).
  patch_len is a backbone option shared by all RoPE modes. Benchmark results live in results/ and are not committed.
- Keep checkpoints/data/results out of Git. Never commit tokens or .env files.
- Do not publish to GitHub, push, or choose a project license without user direction.
- Avoid changing the earlier D:/NUS/RopeFormer repository as part of this project.
