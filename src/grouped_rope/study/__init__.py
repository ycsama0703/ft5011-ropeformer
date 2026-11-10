"""Benchmark study of RoPE frequency parameterisations (point and overlapping-patch tokens).

Self-contained data pipeline (`study.data`) and wrappers/baselines (`study.models`) used by
the scripts in the top-level `study/` directory. The Transformer is the package's
GroupedRoPEForecaster with forecast_head="pooled" (last-token pooling); see study/README.md.
"""
