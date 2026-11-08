"""Forward/backward check for all position-encoding variants."""
import itertools
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grouped_rope import GroupedRoPEForecaster
from grouped_rope.model import ROPE_MODES


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    x, target = torch.randn(4, 96, 3), torch.randn(4, 24, 3)
    for head, mode in itertools.product(("horizon_tokens", "pooled"), ROPE_MODES):
        torch.manual_seed(42)
        model = GroupedRoPEForecaster(input_dim=3, horizon=24, rope_mode=mode, forecast_head=head)
        out = model(x)
        assert out.shape == target.shape and torch.isfinite(out).all()
        loss = (out - target).square().mean()
        loss.backward()
        scales = [(n, p) for n, p in model.named_parameters() if n.endswith("log_scales")]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in scales)
        count = sum(p.numel() for _, p in scales)
        print(f"{head:14} {mode:12} output={tuple(out.shape)} trainable_scales={count} loss={loss.item():.4f}")
    print("OK: forward and backward")


if __name__ == "__main__":
    main()
