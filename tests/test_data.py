"""Benchmark data contract: standard splits, training-only statistics, aligned windows."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grouped_rope.data import WindowedSeries, split_borders


def write_csv(path, values):
    lines = ["date," + ",".join(f"c{j}" for j in range(values.shape[1]))]
    lines += [f"t{i}," + ",".join(repr(float(v)) for v in row) for i, row in enumerate(values.tolist())]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class SplitTests(unittest.TestCase):
    def test_ett_hourly_borders(self):
        borders = split_borders("ETTh1", 17420, lookback=336)
        self.assertEqual(borders["train"], (0, 0, 8640))
        self.assertEqual(borders["val"], (8640 - 336, 8640, 11520))
        self.assertEqual(borders["test"], (11520 - 336, 11520, 14400))

    def test_custom_70_10_20(self):
        borders = split_borders("electricity", 26304, lookback=96)
        n_train, n_test = int(26304 * 0.7), int(26304 * 0.2)
        self.assertEqual(borders["train"][2], n_train)
        self.assertEqual(borders["val"], (n_train - 96, n_train, 26304 - n_test))
        self.assertEqual(borders["test"], (26304 - n_test - 96, 26304 - n_test, 26304))


class WindowTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.raw = torch.randn(200, 3, generator=generator, dtype=torch.float64) * torch.tensor([1.0, 5.0, 0.1]) + 7
        self.raw[150:] += 100.0  # a shift only in val/test must not affect the statistics
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "toy.csv"
        write_csv(self.path, self.raw)
        self.data = WindowedSeries.from_csv(self.path, "toy", lookback=12, dtype=torch.float64)

    def tearDown(self):
        self.tmp.cleanup()

    def test_normalization_uses_training_rows_only(self):
        train_end = self.data.borders["train"][2]
        self.assertEqual(train_end, 140)
        train = self.raw[:train_end]
        torch.testing.assert_close(self.data.mean, train.mean(0))
        torch.testing.assert_close(self.data.std, train.std(0, unbiased=False))
        torch.testing.assert_close(self.data.series[:train_end].mean(0), torch.zeros(3, dtype=torch.float64))

    def test_targets_stay_inside_split_and_history_is_available(self):
        horizon = 5
        for split in ("train", "val", "test"):
            first_history, first_target, end = self.data.borders[split]
            origins = self.data.origins(split, horizon)
            self.assertGreaterEqual(origins.min().item(), first_target)
            self.assertLessEqual(origins.max().item() + horizon, end)
            self.assertGreaterEqual(origins.min().item() - 12, first_history)
            self.assertEqual(origins.max().item() + horizon, end)  # last target row is used

    def test_batch_matches_raw_rows(self):
        horizon = 4
        samples = self.data.samples("val", horizon)
        self.assertEqual(samples.shape[0], self.data.origins("val", horizon).numel() * 3)
        x, y = self.data.batch(samples[[0, 7, -1]], horizon)
        self.assertEqual(x.shape, (3, 12, 1))
        self.assertEqual(y.shape, (3, horizon, 1))
        for row, (origin, channel) in enumerate(samples[[0, 7, -1]].tolist()):
            expected = (self.raw[origin - 12:origin + horizon, channel] - self.data.mean[channel]) / self.data.std[channel]
            torch.testing.assert_close(torch.cat([x[row, :, 0], y[row, :, 0]]), expected)


if __name__ == "__main__":
    unittest.main()
