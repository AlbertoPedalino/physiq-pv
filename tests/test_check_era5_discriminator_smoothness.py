"""The discriminator must be read on the observed field, the reconstruction and their smoothed copies."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan.full_grid import FullGridSTGAN
from physiq_pv.anomaly_detection.stgan.grid import build_spatial_grid
from scripts import check_era5_discriminator_smoothness as script

HEIGHT, WIDTH, FEATURES, CONTEXT, STEPS = 6, 7, 3, 4, 30
LATITUDES = 45 - np.arange(HEIGHT) * .5
LONGITUDES = 7 + np.arange(WIDTH) * .5


class SmoothingTests(unittest.TestCase):
    def test_smoothing_keeps_a_constant_field_and_lowers_the_roughness(self):
        mask = torch.ones(1, 1, HEIGHT, WIDTH)
        constant = torch.full((1, FEATURES, HEIGHT, WIDTH), 2.5)
        torch.testing.assert_close(script.smoothed(constant, mask), constant)
        self.assertEqual(float(script.roughness(constant, mask)[0]), 0.0)
        noisy = torch.randn(1, FEATURES, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(0))
        self.assertLess(float(script.roughness(script.smoothed(noisy, mask), mask)[0]),
                        float(script.roughness(noisy, mask)[0]))

    def test_cells_off_the_lattice_are_left_out(self):
        mask = torch.ones(1, 1, HEIGHT, WIDTH)
        mask[..., 0, 0] = 0
        field = torch.ones(1, FEATURES, HEIGHT, WIDTH)
        field[..., 0, 0] = 100.0  # an absent cell: neither smoothed into its neighbours nor counted
        smooth = script.smoothed(field * mask, mask)
        self.assertEqual(float(smooth[0, 0, 0, 0]), 0.0)
        torch.testing.assert_close(smooth[0, :, 1, 1], torch.ones(FEATURES))
        self.assertEqual(float(script.roughness(field * mask, mask)[0]), 0.0)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        latitudes, longitudes = (axis.ravel() for axis in np.meshgrid(LATITUDES, LONGITUDES, indexing="ij"))
        grid = build_spatial_grid(latitudes, longitudes, patch_size=3, grid_crs="EPSG:4326")
        torch.manual_seed(0)
        config = dict(n_features=FEATURES, hidden_size=8, n_layers=1, cnn_channels=4, cnn_layers=1, patch_size=3,
                      time_feature_size=4, kernel_size=3, dropout_enabled=True, dropout_p=0.2,
                      discriminator_input_dropout=False, reconstruction_feature_weights=None,
                      discriminator_dropout_p=0.2)
        model = FullGridSTGAN(grid=grid, **config)
        self.checkpoint = root / "model_epoch_1.pt"
        torch.save({"format_version": 2, "model_class": "STGAN_CONVGRU", "cnn_training_mode": "full_grid",
                    "grid": {"node_indices": grid.node_indices, "valid_mask": grid.valid_mask,
                             "row_indices": grid.row_indices, "column_indices": grid.column_indices},
                    "model_config": config, "model_state_dict": model.state_dict(),
                    "window_config": {"recent_steps": 2, "trend_steps": CONTEXT}, "time_encoding": "cyclic",
                    "normalization": {"minimum": [-3.0] * FEATURES, "scale": [6.0] * FEATURES}}, self.checkpoint)
        cache = root / "cache"
        cache.mkdir()
        rng = np.random.default_rng(1)
        np.save(cache / "test.npy", rng.normal(size=(CONTEXT + STEPS, HEIGHT * WIDTH, FEATURES)).astype(np.float32))
        self.run_dir = root / "run"
        self.run_dir.mkdir()
        times = pd.date_range("2005-01-01", periods=STEPS, freq="3h")
        np.save(self.run_dir / "test_timestamps.npy", times.as_unit("ns").asi8)
        (self.run_dir / "metadata.json").write_text(json.dumps({
            "status": "complete", "scored_period": "test",
            "backend": {"seasonal_normalization": {"mean_file": str(cache / "mean.npy")}},
            "grid": {"latitudes": LATITUDES.tolist(), "longitudes": LONGITUDES.tolist(),
                     "rows": np.repeat(np.arange(HEIGHT), WIDTH).tolist(),
                     "cols": np.tile(np.arange(WIDTH), HEIGHT).tolist()}}), encoding="utf-8")

    def test_every_sampled_timestamp_of_the_period_gets_a_row(self):
        csv = Path(self.temp.name) / "rows.csv"
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(script.main(["--run-dir", str(self.run_dir), "--checkpoint", str(self.checkpoint),
                                          "--samples", "5", "--threads", "1", "--csv", str(csv)]), 0)
        table = pd.read_csv(csv)
        self.assertEqual(len(table), 5)
        # 30 // 5 = 6, made odd: one every 7 of the 30 targets of the period, never a timestamp of the context.
        self.assertEqual(table["target"].tolist(), [CONTEXT + k for k in range(0, 30, 7)][:5])
        self.assertEqual(table["timestamp"].iloc[0], "2005-01-01 00:00:00")
        numbers = table.drop(columns=["timestamp"]).to_numpy(dtype=float)
        self.assertTrue(np.isfinite(numbers).all())
        self.assertTrue((table["roughness smoothed observed"] < table["roughness observed"]).all())
        for name in script.FIELDS:
            self.assertTrue(table[f"D {name}"].between(0, 1).all())
        self.assertIn("reconstructed / observed", out.getvalue())


if __name__ == "__main__":
    unittest.main()
