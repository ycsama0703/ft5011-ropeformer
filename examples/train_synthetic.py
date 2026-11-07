"""Train on independent synthetic sequences; a pipeline demo, not a benchmark."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grouped_rope import GroupedRoPEForecaster


def make_data(count, lookback, horizon, seed):
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(lookback + horizon)[None, :, None]
    periods = torch.tensor([12.0, 32.0])[None, None, :]
    phases = torch.rand(count, 1, 2, generator=generator) * (2 * torch.pi)
    amplitudes = 0.5 + torch.rand(count, 1, 2, generator=generator)
    signal = (amplitudes * torch.sin(2 * torch.pi * t / periods + phases)).sum(-1, keepdim=True)
    signal = signal + 0.05 * torch.randn(count, lookback + horizon, 1, generator=generator)
    return signal[:, :lookback], signal[:, lookback:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["standard", "fixed_head", "head", "shared_group", "grouped"], default="grouped")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    train_x, train_y = make_data(256, 64, 12, args.seed + 1)
    val_x, val_y = make_data(64, 64, 12, args.seed + 2)
    test_x, test_y = make_data(64, 64, 12, args.seed + 3)
    model = GroupedRoPEForecaster(input_dim=1, horizon=12, d_model=32, n_heads=4, num_layers=2, d_ff=64, dropout=0.0, rope_mode=args.mode)
    scale_params = [p for n, p in model.named_parameters() if n.endswith("log_scales")]
    other_params = [p for n, p in model.named_parameters() if not n.endswith("log_scales")]
    optimizer = torch.optim.AdamW([
        {"params": other_params, "weight_decay": 0.01},
        {"params": scale_params, "weight_decay": 0.0},
    ], lr=1e-3)
    with torch.no_grad():
        initial = (model(val_x) - val_y).square().mean().item()
    best_loss, best_state = float("inf"), None
    for step in range(args.steps):
        model.train()
        idx = torch.randint(train_x.shape[0], (32,))
        loss = (model(train_x[idx]) - train_y[idx]).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = (model(val_x) - val_y).square().mean().item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        error = model(test_x) - test_y
    print(f"mode={args.mode}, steps={args.steps}")
    print(f"validation MSE: initial={initial:.6f}, best={best_loss:.6f}")
    print(f"held-out synthetic test: MSE={error.square().mean():.6f}, MAE={error.abs().mean():.6f}")
    print("First-layer scales:", model.rope_diagnostics()[0]["scales"].tolist())
    print("Pipeline check only; these numbers are not evidence of improvement over a baseline.")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, args.checkpoint)
        print(f"Saved {args.checkpoint}")


if __name__ == "__main__":
    main()
