"""Mathematical invariants and forecasting API checks; no benchmark claims."""
import io
import math
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grouped_rope import GroupedRotaryEmbedding, GroupedRoPEForecaster, ForecastConfig


def cfg(**kwargs):
    values = dict(d_model=24, n_heads=3, num_layers=2, dropout=0.0)
    values.update(kwargs)
    return ForecastConfig(**values)


class RotaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)

    def test_identity_at_initialization(self):
        q, k = torch.randn(2, 3, 17, 8), torch.randn(2, 3, 17, 8)
        reference = GroupedRotaryEmbedding(cfg(rope_mode="standard"))(q, k)
        for mode in ("fixed_head", "head", "shared_group", "grouped"):
            result = GroupedRotaryEmbedding(cfg(rope_mode=mode))(q, k)
            for actual, expected in zip(result, reference):
                torch.testing.assert_close(actual, expected)

    def test_exact_rotation_for_large_scales(self):
        rotary = GroupedRotaryEmbedding(cfg())
        with torch.no_grad():
            rotary.log_scales.fill_(math.log(7.5))
        q = torch.randn(2, 3, 19, 8)
        actual, _ = rotary(q, q)
        angle = torch.arange(19)[None, :, None] * rotary.effective_frequencies()[:, None]
        a, b = q.chunk(2, dim=-1)
        expected = torch.cat((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.square().sum(-1), q.square().sum(-1))

    def test_relative_position_property(self):
        rotary = GroupedRotaryEmbedding(cfg())
        with torch.no_grad():
            rotary.log_scales.copy_(torch.randn_like(rotary.log_scales) * 0.3)
        q = torch.randn(2, 3, 1, 8).expand(-1, -1, 23, -1)
        k = torch.randn_like(q[:, :, :1]).expand_as(q)
        rq, rk = rotary(q, k)
        a = (rq[:, :, 4] * rk[:, :, 11]).sum(-1)
        b = (rq[:, :, 9] * rk[:, :, 16]).sum(-1)
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)

    def test_tied_groups_recover_original_head_scaling(self):
        head = GroupedRotaryEmbedding(cfg(rope_mode="head"))
        grouped = GroupedRotaryEmbedding(cfg())
        with torch.no_grad():
            head.log_scales.copy_(torch.tensor([[-0.5], [0.2], [0.8]]))
            grouped.log_scales.copy_(head.log_scales.expand(-1, 2))
        torch.testing.assert_close(head.effective_frequencies(), grouped.effective_frequencies())
        self.assertEqual(grouped.group_index.tolist(), [0, 0, 1, 1])

    def test_scale_gradients(self):
        rotary = GroupedRotaryEmbedding(cfg())
        q, k = torch.randn(2, 3, 13, 8), torch.randn(2, 3, 13, 8)
        rq, rk = rotary(q, k)
        (rq[:, :, -1] * rk[:, :, 0]).sum().backward()
        grad = rotary.log_scales.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue((grad.abs() > 0).all())

    def test_uneven_groups_and_counts(self):
        rotary = GroupedRotaryEmbedding(cfg(n_freq_groups=3))
        sizes = torch.bincount(rotary.group_index)
        self.assertEqual(int(sizes.max() - sizes.min()), 1)
        for mode, count in [("standard", 0), ("fixed_head", 0), ("head", 6), ("shared_group", 4), ("grouped", 12)]:
            model = GroupedRoPEForecaster(cfg(rope_mode=mode))
            self.assertEqual(sum(p.numel() for n, p in model.named_parameters() if n.endswith("log_scales")), count)


class ModelTests(unittest.TestCase):
    def test_output_backward_and_attention_parity(self):
        for causal in (False, True):
            model = GroupedRoPEForecaster(cfg(input_dim=3, output_dim=2, horizon=7, causal=causal)).eval()
            x = torch.randn(2, 19, 3)
            plain = model(x)
            traced, maps = model(x, return_attention=True)
            self.assertEqual(tuple(plain.shape), (2, 7, 2))
            torch.testing.assert_close(plain, traced, atol=1e-6, rtol=1e-5)
            self.assertEqual(len(maps), 2)
            self.assertFalse(maps[0].requires_grad)
            torch.testing.assert_close(maps[0].sum(-1), torch.ones(2, 3, 19))
            if causal:
                self.assertTrue((maps[0].triu(1) == 0).all())
            plain.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_save_load(self):
        model = GroupedRoPEForecaster(cfg(input_dim=2, pooling="last")).eval()
        with torch.no_grad():
            model.blocks[0].attention.rotary.log_scales.fill_(0.3)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = GroupedRoPEForecaster(model.cfg).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        x = torch.randn(2, 11, 2)
        torch.testing.assert_close(model(x), restored(x))

    def test_validation(self):
        for kwargs in ({"d_model": 25}, {"n_freq_groups": 10}, {"rope_base": 1}, {"dropout": 1}, {"input_dim": 0}, {"rope_mode": "unknown"}):
            with self.assertRaises(ValueError):
                cfg(**kwargs) if "d_model" not in kwargs else ForecastConfig(**kwargs)
        with self.assertRaises(TypeError):
            GroupedRoPEForecaster(ff_mult=2)
        with self.assertRaises(TypeError):
            GroupedRoPEForecaster(cfg(), horizon=3)
        with self.assertRaises(ValueError):
            GroupedRoPEForecaster(cfg())(torch.randn(2, 0, 1))

    def test_cpu_autocast(self):
        model = GroupedRoPEForecaster(cfg(input_dim=2))
        with torch.backends.mkldnn.flags(enabled=False):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                forecast = model(torch.randn(2, 12, 2))
                loss = forecast.float().square().mean()
            loss.backward()
        self.assertTrue(torch.isfinite(forecast).all())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
