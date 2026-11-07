"""Inspect effective position frequencies; this is not an input FFT."""
import argparse
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grouped_rope import GroupedRoPEForecaster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["standard", "fixed_head", "head", "shared_group", "grouped"], default="grouped")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--base", type=float, default=10000.0)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint containing config and state_dict")
    args = parser.parse_args()
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = GroupedRoPEForecaster(**payload["config"])
        model.load_state_dict(payload["state_dict"])
    else:
        model = GroupedRoPEForecaster(rope_mode=args.mode, n_freq_groups=args.groups, rope_base=args.base)
    for layer, info in enumerate(model.rope_diagnostics()):
        print(f"Layer {layer}: scales={info['scales'].tolist()}")
        print("Groups (high to low base frequency):", info["group_index"].tolist())
        print("Angular frequencies (rad/step):", info["frequencies"].tolist())
        print("Rotary periods (steps):", info["periods"].tolist())
    print("Rotary periods are not estimates of the signal's true periods.")


if __name__ == "__main__":
    main()
