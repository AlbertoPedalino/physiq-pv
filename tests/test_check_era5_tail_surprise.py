"""The tail check must read the observation, the forecast and the step before at the same points, and group them."""
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
from scripts import check_era5_tail_surprise as script
from test_stgan_mmd import COLS, NAMES, ROWS, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

WHOLE_GRID = (dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4,
                   gat_recurrence="gated", recent_steps=2)
              if hasattr(stgan, "STGANGraphDataset") else dict(cnn_training_mode="full_grid"))
GRID = {"latitudes": [45.0, 44.5, 44.0], "longitudes": [7.0, 7.5, 8.0], "rows": ROWS.tolist(), "cols": COLS.tolist(),
        "spacing": 0.5}
SEASONAL = "training_only_seasonal_standardisation_then_feature_minmax"


class MeasureTests(unittest.TestCase):
    def test_the_end_of_an_event_forecast_as_going_on(self):
        previous, observed = np.array([2.0, 0.2, -2.0, 3.5]), np.array([0.2, 2.0, -0.1, 2.5])
        forecast = np.array([1.8, 0.3, -0.2, 3.2])
        marks = script.measures(previous, observed, forecast)
        # Back to normal, forecast still unusual and near the step before: all three, and over for the observation.
        np.testing.assert_array_equal(marks[0], [True, True, True, True, False, False, True])
        # An event arriving, forecast near the step before: persistent only; the observation is unusual.
        np.testing.assert_array_equal(marks[1], [False, False, True, False, True, False, False])
        # Back to normal and the forecast followed it: ending, beyond by little, not persistent.
        np.testing.assert_array_equal(marks[2], [True, True, False, False, False, False, False])
        # Weakening but still far from normal: all three, and still an event.
        np.testing.assert_array_equal(marks[3], [True, True, True, True, True, True, False])


class AucTests(unittest.TestCase):
    def test_area_under_the_roc_curve(self):
        self.assertEqual(script.roc_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])), 1.0)
        self.assertEqual(script.roc_auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])), 0.0)
        self.assertEqual(script.roc_auc(np.array([1.0, 3.0]), np.array([2.0, np.nan])), 0.5)
        self.assertEqual(script.roc_auc(np.array([2.0]), np.array([2.0])), 0.5)  # a tie counts half
        self.assertTrue(np.isnan(script.roc_auc(np.array([]), np.array([1.0]))))


class ClassTests(unittest.TestCase):
    def test_groups_kinds_and_distance(self):
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        model, other = np.zeros((3, 9), dtype=bool), np.zeros((3, 9), dtype=bool)
        other[1, 0] = True  # the others flag cell 0 at the centre
        model[1, [1, 2, 0]] = True  # cell 1 next to it, cell 2 two cells away, cell 0 inside
        other[0, 1] = True  # the others flagged cell 1 before: a trail
        other[2, 3] = True  # and will flag cell 3 after: a front, no score on it now
        group, kind, near = script.classes(model, other, 1, 1, rows, cols, (3, 3), 1)
        self.assertEqual((group[1], kind[1], near[1]), (0, 1, True))  # model only, trail, distance 1
        self.assertEqual((group[2], near[2]), (0, False))  # distance 2
        self.assertEqual(group[0], 2)  # inside
        self.assertEqual((group[3], kind[3], near[3]), (1, 0, True))  # neither, front
        self.assertEqual(kind[8], 3)
        self.assertTrue(script.classes(model, other, 1, 1, rows, cols, (3, 3), 2)[2][2])


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
        self.model, payload = load_stgan_checkpoint(self.run_dir / "model_epoch_1.pt", device="cpu")
        # A seasonal run whose cache holds the raw values with the run's min-max: the data are in raw units.
        self.payload = {**payload, "normalization": {"kind": SEASONAL}}
        self.cache = self.root / "cache"
        self.cache.mkdir()
        context = payload["window_config"]["trend_steps"]
        np.save(self.cache / "test.npy", np.concatenate((self.values[VALIDATION][-context:], self.values[TEST])))
        (self.cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {
            "minimum": list(map(float, payload["normalization"]["minimum"])),
            "scale": list(map(float, payload["normalization"]["scale"]))}}), encoding="utf-8")
        (self.run_dir / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID, "backend": {
            "seasonal_normalization": {"mean_file": str(self.cache / "mean.npy")}}}), encoding="utf-8")

    def cubes(self):
        return ERA5Cubes(train=self.values[TRAIN], validation=self.values[VALIDATION], test=self.values[TEST],
                         train_timestamps=TIMES[TRAIN], validation_timestamps=TIMES[VALIDATION],
                         test_timestamps=TIMES[TEST], **NAMES)

    def test_forecasts_are_the_model_output_at_the_points_of_the_data(self):
        cubes = self.cubes()
        try:
            dataset = script.checkpoints.test_dataset(cubes, self.payload, cache_dir=self.cache)
            positions = np.arange(len(dataset))
            previous, observed, forecast, errors, samples, places = script.forecasts(self.model, dataset, positions)
            expected = []
            with script.scoring_mode(self.model, False), torch.no_grad():
                for position in positions:
                    generated = self.model.reconstructed_cells(*dataset.fetch_batch([int(position)])[:5])[1]
                    expected.append(generated.float().numpy())
        finally:
            cubes.close()
        expected = np.concatenate(expected)
        data = np.load(self.cache / "test.npy")
        targets = dataset.targets[samples]
        self.assertEqual(observed.shape, (len(dataset) * 9, 2))
        np.testing.assert_allclose(observed, data[targets, places])
        np.testing.assert_allclose(previous, data[targets - 1, places])
        np.testing.assert_allclose(forecast, (expected + 1) / 2 * dataset.scale + dataset.minimum, rtol=1e-5, atol=1e-5)
        weights = self.model.reconstruction_feature_weights
        squared = (expected - dataset._normalise(observed)) ** 2
        squared = squared if weights is None else squared * weights.numpy()
        np.testing.assert_allclose(errors, squared, rtol=1e-4, atol=1e-6)
        self.assertFalse(any(module.training for module in self.model.modules()))

    def test_main_groups_the_points(self):
        rng = np.random.default_rng(0)
        paths = []
        for name in ("run_model", "run_other"):
            path = self.root / name
            (path / "scores").mkdir(parents=True)
            np.save(path / "scores" / "anomaly_mean.npy", rng.random((len(TIMES[TEST]), 9)).astype(np.float32))
            np.save(path / "test_timestamps.npy", TIMES[TEST].as_unit("ns").asi8)
            (path / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")
            paths.append(path)
        arguments = ["--prepared-dir", "unused", "--run-dir", str(self.run_dir), "--epoch", "1", "--score-dir",
                     str(paths[0]), "--by-dir", str(paths[1]), "--top-percent", "40", "--max-distance", "2"]
        output = io.StringIO()
        with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), \
                patch.object(script.stgan, "load_stgan_checkpoint", return_value=(self.model, self.payload)), \
                redirect_stdout(output):
            rows = script.main(arguments)
        text = output.getvalue()
        # The first and the last timestamps have no whole window: 6 of the 8 are read.
        self.assertIn("[tail-surprise] model=model_epoch_1.pt timestamps=6 points=54 | flagged: the highest 40% of model, "
                      "40% of other", text)
        self.assertEqual(len(rows), len(script.GROUPS) * len(script.KINDS) + 1)
        self.assertLessEqual(sum(row[0] for row in rows.values()), 54)
        self.assertGreater(sum(row[0] for row in rows.values()), 0)
        self.assertIn("main variables of the model only", text)
        # The spread of the dropout draws of a run scored over the same timestamps, read at the same points.
        components = self.root / "components"
        (components / "scores").mkdir(parents=True)
        shape = (len(TIMES[TEST]), 9)
        for name, value in (("generator_mean", 2.0), ("generator_std", 0.2), ("discriminator_std", 0.05)):
            np.save(components / "scores" / f"{name}.npy", np.full(shape, value, dtype=np.float32))
        np.save(components / "test_timestamps.npy", TIMES[TEST].as_unit("ns").asi8)
        with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), \
                patch.object(script.stgan, "load_stgan_checkpoint", return_value=(self.model, self.payload)), \
                redirect_stdout(io.StringIO()):
            spread = script.main([*arguments, "--components-dir", str(components)])["spread"]
        medians = {key: values for key, values in spread.items() if not key.startswith("AUC")}
        for values in medians.values():
            finite = [value for value in values if np.isfinite(value)]
            self.assertTrue(all(abs(value - 0.1) < 1e-6 or abs(value - 0.05) < 1e-6 for value in finite))
        self.assertTrue(any(np.isfinite(values[0]) for values in medians.values()))
        # The same spread everywhere tells nothing apart: every area under the ROC curve is a half, or none.
        for key in ("AUC, model only, all kinds", "AUC, inside (model and other)"):
            for value in spread[key][2:]:
                self.assertTrue(np.isnan(value) or abs(value - 0.5) < 1e-9)
        # And so is the error of G, and every spread at equal error.
        for key in ("AUC at equal error, model only, all kinds", "AUC at equal error, inside (model and other)"):
            for value in spread[key]:
                self.assertTrue(np.isnan(value) or abs(value - 0.5) < 1e-9)
        plain = {**self.payload, "normalization": {"kind": "minmax"}}
        with patch.object(script.stgan, "load_stgan_checkpoint", return_value=(self.model, plain)), \
                self.assertRaisesRegex(ValueError, "seasonal normalization"):
            script.main(arguments)


if __name__ == "__main__":
    unittest.main()
