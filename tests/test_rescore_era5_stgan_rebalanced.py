"""The rescoring must give the rebalanced score of every draw, with the scale of the saved errors."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import physiq_pv.anomaly_detection.stgan as stgan
from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan, load_stgan_checkpoint
from physiq_pv.era5.data import ERA5Cubes
from scripts import rebalance_era5_stgan_scores as rebalance
from scripts import rescore_era5_stgan_rebalanced as script
from scripts import run_era5_stgan as cli
from test_stgan_mmd import COLS, NAMES, ROWS, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

# The model of this branch that scores a whole timestamp per batch.
WHOLE_GRID = (dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4,
                   gat_recurrence="gated", recent_steps=2)
              if hasattr(stgan, "STGANGraphDataset") else dict(cnn_training_mode="full_grid"))
GRID = {"latitudes": [45.0, 44.5, 44.0], "longitudes": [7.0, 7.5, 8.0], "rows": ROWS.tolist(), "cols": COLS.tolist(),
        "spacing": 0.5}


class TransformTests(unittest.TestCase):
    def test_tensor_transform_is_the_numpy_one(self):
        rng = np.random.default_rng(0)
        sample = rng.gamma(0.5, 1.0, (20000, 3)).astype(np.float32)
        sample[: 12000, 2] = 0  # many equal errors
        # Values inside the sample, equal to its extremes and beyond them.
        errors = np.concatenate((rng.gamma(0.5, 1.0, (4, 50, 3)), np.zeros((4, 1, 3)), np.full((4, 1, 3), 1e3),
                                 np.broadcast_to(sample.max(axis=0), (4, 1, 3))), axis=1).astype(np.float32)
        for scale in ("mean", "rarity", "none"):
            fitted = rebalance.fit(sample, scale)
            expected = rebalance.transform(errors, scale, fitted)
            result = script.transform(torch.as_tensor(errors), scale, fitted)
            self.assertEqual(result.dtype, torch.float64)
            np.testing.assert_allclose(result.numpy(), expected, rtol=2e-5, atol=1e-6)
        # A variable whose errors are all equal has one point in its table.
        constant = rebalance.fit(np.ones((100, 1), dtype=np.float32), "rarity")
        np.testing.assert_allclose(script.transform(torch.ones(3, 1), "rarity", constant).numpy(),
                                   rebalance.transform(np.ones((3, 1), dtype=np.float32), "rarity", constant), rtol=1e-6)


class BandTests(unittest.TestCase):
    def test_bands_on_known_values_and_an_empty_band(self):
        rng = np.random.default_rng(1)
        mean = rng.random((300, 40)) + 0.01
        rows, correlation = script.band_table(mean, 0.1 * mean)
        self.assertEqual([row[0] for row in rows], ["0-50", "50-90", "90-95", "95-99", "99-99.9", "99.9-100"])
        np.testing.assert_allclose([row[3] for row in rows], 0.1, rtol=1e-6)
        self.assertTrue((np.diff([row[1] for row in rows]) > 0).all())
        self.assertAlmostEqual(correlation, 1.0, places=6)
        # Forty-five points: no value falls between the 99th and the 99.9th percentile.
        few = rng.random((5, 9))
        self.assertTrue(np.isnan(script.band_table(few, 0.1 * few)[0][4][1]))


class RunTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields()
        build_pca(self.root / "pca", self.values)
        build(self.root / "reference", self.values, self.root / "pca")
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=self.values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=2, batch_size=1, score_batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, device="cpu", grid_crs="EPSG:4326", timestep_hours=3, angular_grid_spacing=.5,
            grid_audit_knn=False, score_mode="paper", cache_normalized=False, mc_samples=2, monitoring_timestamps=4,
            monitoring_feature_mmd_samples=16, checkpoint_path=self.root / "run" / "model.pt",
            score_dir=self.root / "run" / "scores", pca_reference=self.root / "pca",
            mmd_reference=self.root / "reference", score_storage="memmap", **WHOLE_GRID)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options):
            pass
        self.run_dir = self.root / "run"
        # What the ERA5 command adds to the directory of a run.
        (self.run_dir / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")
        np.save(self.run_dir / "test_timestamps.npy", TIMES[TEST].asi8)

    def cubes(self):
        return ERA5Cubes(train=self.values[TRAIN], validation=self.values[VALIDATION], test=self.values[TEST],
                         train_timestamps=TIMES[TRAIN], validation_timestamps=TIMES[VALIDATION],
                         test_timestamps=TIMES[TEST], **NAMES)

    def main(self, *arguments):
        output = io.StringIO()
        with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), redirect_stdout(output):
            self.assertEqual(script.main(["--prepared-dir", "unused", "--run-dir", str(self.run_dir), "--epoch", "2",
                                          "--device", "cpu", "--scale-stride", "1", *arguments]), 0)
        return output.getvalue()

    def test_rescore_is_the_rebalanced_score_of_every_draw(self):
        model, payload = load_stgan_checkpoint(self.run_dir / "model_epoch_2.pt", device="cpu")
        names = list(NAMES["feature_names"])
        cubes = self.cubes()
        try:
            dataset = script.checkpoints.test_dataset(cubes, payload)
            positions = np.arange(len(dataset))
            for scale in ("mean", "rarity"):
                columns, fitted = script.fitted_scales(self.run_dir, scale, None, 1, names)
                torch.manual_seed(0)
                mean, spread, of_mean_errors = script.rescore(model, dataset, positions, scale=scale, columns=columns,
                                                              fitted=fitted, mc_samples=4)
                # The same draws, taken directly from the model and rebalanced with numpy.
                torch.manual_seed(0)
                expected, expected_of_mean = [], []
                with script.scoring_mode(model, True), torch.no_grad():
                    for position in positions:
                        draws = model.score_draws(*dataset.fetch_batch([int(position)])[:5], 4).float().numpy()[:, :, 2:]
                        expected.append(rebalance.transform(draws, scale, fitted).mean(axis=2, dtype=np.float64))
                        expected_of_mean.append(rebalance.transform(draws.mean(axis=0), scale, fitted).mean(axis=1))
                expected = np.stack(expected, axis=1)
                np.testing.assert_allclose(mean.ravel(), expected.mean(axis=0).ravel(), rtol=2e-4, atol=1e-6)
                np.testing.assert_allclose(spread.ravel(), expected.std(axis=0).ravel(), rtol=2e-3, atol=1e-6)
                np.testing.assert_allclose(of_mean_errors.ravel(), np.concatenate(expected_of_mean), rtol=2e-4, atol=1e-6)
                self.assertEqual(mean.shape, (len(dataset), dataset.n_locations))
                self.assertGreater(spread.mean(), 0)
                if scale == "mean":
                    # A linear scale: the mean over the draws is the score of the mean errors.
                    np.testing.assert_allclose(mean, of_mean_errors, rtol=2e-4, atol=1e-6)
        finally:
            cubes.close()
        # The model is left in evaluation mode, and unknown variables are refused.
        self.assertFalse(any(module.training for module in model.modules()))
        with self.assertRaisesRegex(ValueError, "Unknown variables"):
            script.fitted_scales(self.run_dir, "mean", ["wind"], 1, names)
        self.assertEqual(script.fitted_scales(self.run_dir, "mean", ["b"], 1, names)[0], [1])

    def test_sample_run_prints_the_spread_and_writes_nothing(self):
        before = sorted(path.name for path in self.root.iterdir())
        text = self.main("--timestamps", "5", "--mc-samples", "3")
        for expected in ("scale=rarity variables=2 epoch=2 timestamps=5 locations=9 mc_samples=3",
                         "h for all the 8 timestamps", "=== Monte Carlo spread of the rebalanced score by score band ===",
                         "99.9-100", "rank correlation between score and spread",
                         "=== mean over the draws against the score built on the mean errors ==="):
            self.assertIn(expected, text)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), before)
        with self.assertRaisesRegex(ValueError, "at least two draws"):
            self.main("--mc-samples", "1")

    def test_output_is_a_run_with_mean_and_spread(self):
        text = self.main("--output-dir", str(self.root / "rescored"), "--mc-samples", "3", "--scale", "mean",
                         "--timestamps", "2")
        self.assertIn("timestamps=8", text)  # with an output every timestamp is scored
        mean = np.load(self.root / "rescored" / "scores" / "anomaly_mean.npy")
        spread = np.load(self.root / "rescored" / "scores" / "anomaly_std.npy")
        self.assertEqual((mean.shape, spread.shape), ((8, 9), (8, 9)))
        self.assertTrue((spread >= 0).all() and spread.max() > 0)
        np.testing.assert_array_equal(np.load(self.root / "rescored" / "test_timestamps.npy"), TIMES[TEST].asi8)
        manifest = json.loads((self.root / "rescored" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["scale"], manifest["features"], manifest["mc_samples"], manifest["epoch"]),
                         ("mean", ["a", "b"], 3, 2))
        # With the mean as scale the result is close to the score rebalanced from the saved errors
        # (other draws), and `events` reads it with its spread.
        rebalance.rebalance(self.run_dir, self.root / "balanced", scale="mean", scale_stride=1)
        saved = np.load(self.root / "balanced" / "scores" / "anomaly_mean.npy")
        self.assertGreater(np.corrcoef(saved.ravel(), mean.ravel())[0, 1], 0.9)
        with redirect_stdout(io.StringIO()):
            cli.main(["events", "--run-dir", str(self.root / "rescored"), "--output-dir", str(self.root / "events"),
                      "--top-percent", "50", "--opening-iterations", "0", "--closing-iterations", "0"])
        self.assertEqual(json.loads((self.root / "events" / "metadata.json").read_text(encoding="utf-8"))["status"], "complete")
        with self.assertRaisesRegex(ValueError, "new/empty"):
            self.main("--output-dir", str(self.root / "rescored"))


if __name__ == "__main__":
    unittest.main()
