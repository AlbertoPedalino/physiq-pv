"""The band check must measure coverage, width and spread of the dropout draws of the reconstruction."""
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
from scripts import check_era5_stgan_reconstruction_band as script
from test_stgan_mmd import COLS, NAMES, ROWS, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

# The model of this branch that scores a whole timestamp per batch.
WHOLE_GRID = (dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4,
                   gat_recurrence="gated", recent_steps=2)
              if hasattr(stgan, "STGANGraphDataset") else dict(cnn_training_mode="full_grid"))
GRID = {"latitudes": [45.0, 44.5, 44.0], "longitudes": [7.0, 7.5, 8.0], "rows": ROWS.tolist(), "cols": COLS.tolist(),
        "spacing": 0.5}


class TableTests(unittest.TestCase):
    def setUp(self):
        """Two variables. The first has a band that fits its deviations; the second a band ten times too narrow."""
        rng = np.random.default_rng(0)
        points = 40000
        self.spread = np.column_stack((np.full(points, 1.0), np.full(points, 0.1)))
        self.deviation = np.abs(rng.normal(size=(points, 2)))
        self.width = 2 * 1.645 * self.spread
        self.inside = self.deviation <= 1.645 * self.spread
        self.score = self.deviation.sum(axis=1)

    def test_by_variable(self):
        table = script.by_variable(self.inside, self.width, self.deviation, self.spread)
        np.testing.assert_allclose(table[:, 0], [0.90, 0.131], atol=0.01)  # coverage: nominal, and far below it
        np.testing.assert_allclose(table[:, 1], [1.645, 0.1645], rtol=1e-6)  # width over the range of 2
        np.testing.assert_allclose(table[:, 2], np.sqrt(2 / np.pi), rtol=0.02)
        np.testing.assert_allclose(table[:, 4], [1 / np.sqrt(2 / np.pi), 0.1 / np.sqrt(2 / np.pi)], rtol=0.02)

    def test_by_score_band_the_highest_scores_are_the_least_covered(self):
        table = script.by_score_band(self.inside, self.width, self.score)
        self.assertEqual(table.shape, (len(script.BANDS) - 1, 3))
        self.assertEqual(int(table[:, 0].sum()), len(self.score))
        self.assertTrue((np.diff(table[:, 1]) < 0).all())
        np.testing.assert_allclose(table[:, 2], (1.645 + 0.1645) / 2, rtol=1e-6)

    def test_by_deviation_band_a_constant_spread_is_narrower_in_proportion_where_the_deviation_is_larger(self):
        table = script.by_deviation_band(self.inside, self.deviation, self.spread)
        self.assertTrue((np.diff(table[:, 0]) > 0).all())  # the deviation grows from band to band
        np.testing.assert_allclose(table[:, 1], 0.55, rtol=1e-6)  # the mean of the two constant spreads
        self.assertTrue((np.diff(table[:, 2]) < 0).all())  # spread over deviation falls
        self.assertTrue((np.diff(table[:, 3]) <= 1e-12).all())  # and so does the coverage
        self.assertEqual(table[-1, 3], 0.0)  # the largest deviations are outside both bands
        # The bands are those of the values themselves.
        np.testing.assert_array_equal(np.bincount(script.band_of(np.arange(1000.0))), [500, 400, 50, 40, 10])


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
            epochs=1, batch_size=1, score_batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, device="cpu", grid_crs="EPSG:4326", timestep_hours=3, angular_grid_spacing=.5,
            grid_audit_knn=False, score_mode="paper", cache_normalized=False, mc_samples=2, monitoring_timestamps=4,
            monitoring_feature_mmd_samples=16, checkpoint_path=self.root / "run" / "model.pt",
            score_dir=self.root / "run" / "scores", pca_reference=self.root / "pca",
            mmd_reference=self.root / "reference", score_storage="memmap", **WHOLE_GRID)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options):
            pass
        self.run_dir = self.root / "run"
        (self.run_dir / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")

    def cubes(self):
        return ERA5Cubes(train=self.values[TRAIN], validation=self.values[VALIDATION], test=self.values[TEST],
                         train_timestamps=TIMES[TRAIN], validation_timestamps=TIMES[VALIDATION],
                         test_timestamps=TIMES[TEST], **NAMES)

    def test_draw_statistics_are_those_of_the_draws_of_the_model(self):
        model, payload = load_stgan_checkpoint(self.run_dir / "model_epoch_1.pt", device="cpu")
        names = list(NAMES["feature_names"])
        cubes = self.cubes()
        try:
            dataset = script.checkpoints.test_dataset(cubes, payload)
            positions = np.arange(len(dataset))
            columns, fitted = script.rescore.fitted_scales(self.run_dir, "rarity", None, 1, names)
            torch.manual_seed(0)
            inside, width, deviation, spread, score = script.draw_statistics(
                model, dataset, positions, scale="rarity", columns=columns, fitted=fitted, mc_samples=6)
            # The same draws, taken directly from the model.
            torch.manual_seed(0)
            expected = {"width": [], "deviation": [], "spread": [], "inside": []}
            with script.scoring_mode(model, True), torch.no_grad():
                for position in positions:
                    batch = dataset.fetch_batch([int(position)])
                    draws = []
                    for _ in range(6):
                        cells, generated = model.reconstructed_cells(*batch[:5])
                        draws.append(generated.float().numpy())
                    draws, cells = np.stack(draws), cells.numpy()
                    lower, upper = np.quantile(draws, [0.05, 0.95], axis=0)
                    expected["width"].append(upper - lower)
                    expected["deviation"].append(np.abs(cells - draws.mean(axis=0)))
                    expected["spread"].append(draws.std(axis=0))
                    expected["inside"].append((cells >= lower) & (cells <= upper))
        finally:
            cubes.close()
        points = len(dataset) * dataset.n_locations
        self.assertEqual((inside.shape, width.shape, score.shape), ((points, len(names)), (points, len(names)), (points,)))
        np.testing.assert_allclose(width, np.concatenate(expected["width"]), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(deviation, np.concatenate(expected["deviation"]), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(spread, np.concatenate(expected["spread"]), rtol=1e-4, atol=1e-6)
        self.assertGreater((inside == np.concatenate(expected["inside"])).mean(), 0.99)
        self.assertGreater(spread.mean(), 0)
        self.assertTrue((score > 0).all())
        self.assertFalse(any(module.training for module in model.modules()))

    def test_a_seasonal_run_reads_its_cache(self):
        model, payload = load_stgan_checkpoint(self.run_dir / "model_epoch_1.pt", device="cpu")
        cubes = self.cubes()
        try:
            plain = script.checkpoints.test_dataset(cubes, payload)
            seasonal = {**payload, "normalization": {"kind": "training_only_seasonal_standardisation_then_feature_minmax"}}
            with self.assertRaisesRegex(ValueError, "need their standardised cache"):
                script.checkpoints.test_dataset(cubes, seasonal)
            # A cache that holds the raw values with the run's min-max gives the same inputs as the plain run.
            cache = self.root / "cache"
            cache.mkdir()
            context = payload["window_config"]["trend_steps"]
            np.save(cache / "test.npy", np.concatenate((self.values[VALIDATION][-context:], self.values[TEST])))
            (cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {
                "minimum": list(map(float, payload["normalization"]["minimum"])),
                "scale": list(map(float, payload["normalization"]["scale"]))}}), encoding="utf-8")
            read = script.checkpoints.test_dataset(cubes, seasonal, cache_dir=cache)
            self.assertEqual(len(read), len(plain))
            self.assertTrue(read.target_timestamps.equals(plain.target_timestamps))
            for position in (0, len(plain) - 1):
                for left, right in zip(read.fetch_batch([position]), plain.fetch_batch([position])):
                    np.testing.assert_allclose(left.numpy(), right.numpy(), rtol=1e-6, atol=1e-6)
        finally:
            cubes.close()

    def test_main_splits_the_points_by_group_of_the_scores(self):
        rng = np.random.default_rng(0)
        paths = []
        for name in ("run_model", "run_other"):
            path = self.root / name
            (path / "scores").mkdir(parents=True)
            np.save(path / "scores" / "anomaly_mean.npy", rng.random((len(TIMES[TEST]), 9)).astype(np.float32))
            np.save(path / "test_timestamps.npy", TIMES[TEST].as_unit("ns").asi8)
            (path / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")
            paths.append(path)
        output = io.StringIO()
        with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), redirect_stdout(output):
            self.assertEqual(script.main(["--prepared-dir", "unused", "--run-dir", str(self.run_dir), "--epoch", "1",
                                          "--device", "cpu", "--timestamps", "5", "--mc-samples", "4", "--scale-stride",
                                          "1", "--score-dir", str(paths[0]), "--by-dirs", str(paths[1]),
                                          "--top-percent", "30"]), 0)
        text = output.getvalue()
        self.assertIn("=== by group of the scores: flagged in the highest 30% by model and by other", text)
        rows = [line for line in text.splitlines() if line.startswith(("score alone", "with another", "others alone",
                                                                        "none", "all points"))]
        values = [int(line[20:30]) for line in rows]
        self.assertEqual(len(values), 5)
        self.assertEqual(sum(values[:4]), values[4])
        self.assertEqual(values[4], 45)
        with self.assertRaisesRegex(ValueError, "together"):
            script.main(["--prepared-dir", "unused", "--run-dir", str(self.run_dir), "--epoch", "1",
                         "--score-dir", str(paths[0])])

    def test_main_prints_the_three_tables(self):
        output = io.StringIO()
        with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), redirect_stdout(output):
            self.assertEqual(script.main(["--prepared-dir", "unused", "--run-dir", str(self.run_dir), "--epoch", "1",
                                          "--device", "cpu", "--timestamps", "5", "--mc-samples", "4",
                                          "--scale-stride", "1"]), 0)
        text = output.getvalue()
        for expected in ("[band] epoch=1 timestamps=5 locations=9 mc_samples=4 points=45 variables=2",
                         "=== band of the reconstruction by variable", "nominal coverage 90%", "all variables",
                         "=== by band of the rebalanced score, all variables ===",
                         "=== the band against the deviation", "spread/deviation"):
            self.assertIn(expected, text)
        for name in NAMES["feature_names"]:
            self.assertRegex(text, rf"(?m)^{name} +\d+\.\d%")
        with self.assertRaisesRegex(ValueError, "at least two draws"), \
                patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})):
            script.main(["--prepared-dir", "unused", "--run-dir", str(self.run_dir), "--epoch", "1", "--mc-samples", "1"])


if __name__ == "__main__":
    unittest.main()
