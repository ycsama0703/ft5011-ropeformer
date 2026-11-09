"""Mathematical invariants and forecasting API checks; no benchmark claims."""
import io
import math
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grouped_rope import GroupedRotaryEmbedding, GroupedRoPEForecaster, ForecastConfig, horizon_token_mask


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
        for mode, count in [("none", 0), ("standard", 0), ("fixed_head", 0), ("head", 6), ("shared_group", 4), ("grouped", 12)]:
            model = GroupedRoPEForecaster(cfg(rope_mode=mode))
            self.assertEqual(sum(p.numel() for n, p in model.named_parameters() if n.endswith("log_scales")), count)


class ModelTests(unittest.TestCase):
    def test_output_backward_and_attention_parity(self):
        for head in ("horizon_tokens", "pooled"):
            for causal in (False, True):
                model = GroupedRoPEForecaster(cfg(input_dim=3, output_dim=2, horizon=7, causal=causal, forecast_head=head)).eval()
                x = torch.randn(2, 19, 3)
                plain = model(x)
                traced, maps = model(x, return_attention=True)
                self.assertEqual(tuple(plain.shape), (2, 7, 2))
                torch.testing.assert_close(plain, traced, atol=1e-6, rtol=1e-5)
                self.assertEqual(len(maps), 2)
                self.assertFalse(maps[0].requires_grad)
                length = 19 + 7 if head == "horizon_tokens" else 19
                torch.testing.assert_close(maps[0].sum(-1), torch.ones(2, 3, length))
                if causal:
                    self.assertTrue((maps[0].triu(1) == 0).all())
                plain.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_save_load(self):
        model = GroupedRoPEForecaster(cfg(input_dim=2, forecast_head="pooled", pooling="last")).eval()
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
        for kwargs in ({"pooling": "mean"}, {"forecast_head": "decoder"}):
            with self.assertRaises(ValueError):
                cfg(**kwargs)
        with self.assertRaises(ValueError):
            GroupedRoPEForecaster(cfg(forecast_head="pooled"))(torch.randn(2, 5, 1), horizon=3)
        for horizon in (0, 2.0):
            with self.assertRaises(ValueError):
                GroupedRoPEForecaster(cfg())(torch.randn(2, 5, 1), horizon=horizon)
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


class HorizonTokenTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def trained_like(self, **kwargs):
        model = GroupedRoPEForecaster(cfg(input_dim=2, horizon=6, **kwargs)).eval()
        for name, param in model.named_parameters():
            if name.endswith("log_scales"):
                with torch.no_grad():
                    param.copy_(torch.randn_like(param) * 0.3)
        return model

    def test_mask_structure(self):
        mask = horizon_token_mask(4, 3, causal=False)
        expected = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 1],
        ], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected))
        causal = horizon_token_mask(4, 3, causal=True)
        self.assertTrue(torch.equal(causal[:4, :4], torch.ones(4, 4, dtype=torch.bool).tril()))
        self.assertTrue(torch.equal(causal[4:], expected[4:]))

    def test_forecast_prefix_independent_of_horizon(self):
        for causal in (False, True):
            model = self.trained_like(causal=causal)
            x = torch.randn(3, 17, 2)
            short, long = model(x, horizon=4), model(x, horizon=11)
            torch.testing.assert_close(short, long[:, :4])
            self.assertEqual(tuple(long.shape), (3, 11, 2))

    def test_history_attention_independent_of_horizon(self):
        model = self.trained_like()
        x = torch.randn(2, 13, 2)
        _, short = model(x, horizon=3, return_attention=True)
        _, long = model(x, horizon=9, return_attention=True)
        for a, b in zip(short, long):
            torch.testing.assert_close(a[..., :13, :13], b[..., :13, :13])
            self.assertTrue((b[..., :13, 13:] == 0).all())
            future = b[..., 13:, 13:]
            self.assertTrue((future - torch.diag_embed(future.diagonal(dim1=-2, dim2=-1)) == 0).all())

    def test_nope_gives_constant_forecast_over_horizon(self):
        model = self.trained_like(rope_mode="none")
        forecast = model(torch.randn(2, 15, 2), horizon=5)
        torch.testing.assert_close(forecast, forecast[:, :1].expand_as(forecast))

    def test_rope_distinguishes_future_steps(self):
        forecast = self.trained_like()(torch.randn(2, 15, 2), horizon=5)
        self.assertGreater((forecast - forecast[:, :1]).abs().max().item(), 1e-4)

    def test_parameter_count_independent_of_horizon(self):
        count = lambda h: sum(p.numel() for p in GroupedRoPEForecaster(cfg(horizon=h)).parameters())
        self.assertEqual(count(3), count(50))

    def test_modes_differ_only_in_log_scales(self):
        shapes = {}
        for mode in ("none", "standard", "head", "shared_group", "grouped"):
            torch.manual_seed(0)
            model = GroupedRoPEForecaster(cfg(rope_mode=mode))
            shapes[mode] = {n: p.detach().clone() for n, p in model.named_parameters() if not n.endswith("log_scales")}
        for mode, params in shapes.items():
            self.assertEqual(params.keys(), shapes["standard"].keys())
            for name, value in params.items():
                torch.testing.assert_close(value, shapes["standard"][name], msg=f"{mode}: {name}")


class PatchTokenTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def model(self, patch_len=4, **kwargs):
        model = GroupedRoPEForecaster(cfg(input_dim=2, horizon=6, patch_len=patch_len, **kwargs)).eval()
        for name, param in model.named_parameters():
            if name.endswith("log_scales"):
                with torch.no_grad():
                    param.copy_(torch.randn_like(param) * 0.3)
        return model

    def test_patch_len_one_matches_point_tokens(self):
        torch.manual_seed(0)
        point = GroupedRoPEForecaster(cfg(input_dim=2, horizon=6)).eval()
        torch.manual_seed(0)
        patched = GroupedRoPEForecaster(cfg(input_dim=2, horizon=6, patch_len=1)).eval()
        x = torch.randn(3, 12, 2)
        torch.testing.assert_close(point(x), patched(x))

    def test_shapes_and_partial_last_patch(self):
        model = self.model()
        for horizon in (1, 4, 6, 9):
            self.assertEqual(tuple(model(torch.randn(2, 16, 2), horizon=horizon).shape), (2, horizon, 2))

    def test_prefix_independent_of_horizon(self):
        model = self.model()
        x = torch.randn(3, 16, 2)
        torch.testing.assert_close(model(x, horizon=5), model(x, horizon=13)[:, :5])

    def test_oldest_remainder_rows_are_dropped(self):
        model = self.model()
        x = torch.randn(2, 16, 2)
        padded = torch.cat([torch.randn(2, 3, 2), x], 1)  # 19 steps -> the same 4 patches
        torch.testing.assert_close(model(x), model(padded))

    def test_patch_reshape_is_time_major(self):
        model = self.model(patch_len=2, rope_mode="standard")
        x = torch.randn(1, 4, 2)
        expected = model.input_projection(torch.stack([x[0, :2].reshape(-1), x[0, 2:].reshape(-1)]))
        with torch.no_grad():
            got = model.input_projection(x.reshape(1, 2, 4))[0]
        torch.testing.assert_close(got, expected)

    def test_history_shorter_than_patch_rejected(self):
        with self.assertRaises(ValueError):
            self.model()(torch.randn(1, 3, 2))
        with self.assertRaises(ValueError):
            ForecastConfig(patch_len=0)

    def test_modes_differ_only_in_log_scales(self):
        params = {}
        for mode in ("standard", "head", "shared_group", "grouped"):
            torch.manual_seed(0)
            model = GroupedRoPEForecaster(cfg(patch_len=4, rope_mode=mode))
            params[mode] = {n: p.detach() for n, p in model.named_parameters() if not n.endswith("log_scales")}
        for mode, values in params.items():
            for name, value in values.items():
                torch.testing.assert_close(value, params["standard"][name], msg=f"{mode}: {name}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
