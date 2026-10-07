"""ERA5 protocol: train 1980-2003, validation 2004, test from 2005.

The validation is read by the per-epoch monitoring and by the validation objective only: the
score of the test, its min-max factors and the trained weights do not depend on it.
The same file serves the ConvGRU and the GAT branch.
"""
from contextlib import redirect_stdout
import gc
import inspect
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection import stgan
from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan.data import ContextArray
from physiq_pv.era5.cube import CubeGrid
from physiq_pv.era5.data import ERA5Cubes, load_prepared, prepare_era5
from physiq_pv.era5.features import FEATURE_NAMES
from scripts import run_era5_stgan

HAS_GRAPH = hasattr(stgan, "STGANGAT")
TIMES = pd.date_range("2002-01-01", "2005-01-02 21:00", freq="3h").as_unit("ns")
VALUES = np.random.default_rng(12).normal(size=(len(TIMES), 2, len(FEATURE_NAMES))).astype(np.float32)
YEAR = TIMES.year.to_numpy()


def write_cache(root, *, older):
    """A prepared cache of the same data, in the current layout or in the one that preceded the validation."""
    root.mkdir(parents=True)
    grid = CubeGrid.from_locations(np.array([45., 45.]), np.array([7., 7.5]))
    if older:  # First partition through 2002, second partition 2003-2004.
        parts = {"train": YEAR <= 2002, "calibration": (YEAR >= 2003) & (YEAR <= 2004), "test": YEAR >= 2005}
        years = {"train_end_year": 2002, "calibration_start_year": 2003, "calibration_end_year": 2004}
    else:
        parts = {"train": YEAR <= 2003, "validation": YEAR == 2004, "test": YEAR >= 2005}
        years = {"train_end_year": 2003, "validation_start_year": 2004, "validation_end_year": 2004}
    for name, rows in parts.items():
        np.save(root / f"{name}.npy", VALUES[rows])
        np.save(root / f"{name}_timestamps.npy", TIMES[rows].asi8)
    metadata = {"status": "complete", "grid": grid.to_dict(), "features": list(FEATURE_NAMES), "start_year": 2002,
                "score_end_year": 2005, "test_start_year": 2005, "timestep_hours": 3, **years}
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(gc.collect)
        self.root = Path(self.temp.name)

    def test_prepare_writes_train_validation_and_test(self):
        defaults = inspect.signature(prepare_era5).parameters
        self.assertEqual((defaults["start_year"].default, defaults["train_end_year"].default,
                          defaults["validation_end_year"].default), (1980, 2003, 2004))
        self.assertNotIn("calibration_end_year", defaults)
        args = run_era5_stgan.parser().parse_args(["prepare", "--output-dir", "cache"])
        self.assertEqual((args.train_end_year, args.validation_end_year), (2003, 2004))
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            run_era5_stgan.parser().parse_args(["prepare", "--output-dir", "cache", "--calibration-end-year", "2004"])

        def blocks(root, year, month, **kwargs):
            times = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month * 8,
                                  freq="3h").as_unit("ns")
            yield times, np.full((len(times), 2, 2, 15), float(year), np.float32)
        with patch("physiq_pv.era5.data.month_files", return_value=[]), \
                patch("physiq_pv.era5.data.monthly_blocks", side_effect=blocks):
            cubes, _, metadata = prepare_era5(self.root, self.root / "cache", start_year=1980, train_end_year=1980,
                                              validation_end_year=1981, score_end_year=1982, area=(30.5, 0, 30, .5))
        try:
            self.assertIsInstance(cubes, ERA5Cubes)
            self.assertEqual(sorted(path.name for path in (self.root / "cache").glob("*.npy")),
                             ["test.npy", "test_timestamps.npy", "train.npy", "train_timestamps.npy",
                              "validation.npy", "validation_timestamps.npy"])
            self.assertEqual([set(times.year) for times in (cubes.train_timestamps, cubes.validation_timestamps,
                                                           cubes.test_timestamps)], [{1980}, {1981}, {1982}])
            self.assertEqual((float(cubes.train[0, 0, 0]), float(cubes.validation[-1, 0, 0]), float(cubes.test[0, 0, 0])),
                             (1980., 1981., 1982.))
            self.assertEqual((metadata["train_end_year"], metadata["validation_start_year"],
                              metadata["validation_end_year"], metadata["test_start_year"]), (1980, 1981, 1981, 1982))
            self.assertNotIn("calibrat", json.dumps(metadata).lower())
        finally:
            cubes.close()

    def test_older_cache_is_read_as_train_through_2003_and_validation_2004(self):
        loaded = {name: load_prepared(write_cache(self.root / name, older=name == "older"))
                  for name in ("current", "older")}
        try:
            for name, (cubes, _, metadata) in loaded.items():
                with self.subTest(cache=name):
                    self.assertEqual((cubes.train_timestamps[0], cubes.train_timestamps[-1]),
                                     (pd.Timestamp("2002-01-01"), pd.Timestamp("2003-12-31 21:00")))
                    self.assertEqual((cubes.validation_timestamps[0], cubes.validation_timestamps[-1],
                                      len(cubes.validation_timestamps)),
                                     (pd.Timestamp("2004-01-01"), pd.Timestamp("2004-12-31 21:00"), 366 * 8))
                    self.assertEqual(cubes.test_timestamps[0], pd.Timestamp("2005-01-01"))
                    self.assertEqual((len(cubes.train), len(cubes.validation), len(cubes.test)),
                                     (730 * 8, 366 * 8, 16))
                    np.testing.assert_array_equal(cubes.train[0:len(cubes.train)], VALUES[YEAR <= 2003])
                    np.testing.assert_array_equal(cubes.validation, VALUES[YEAR == 2004])
                    np.testing.assert_array_equal(cubes.test, VALUES[YEAR >= 2005])
                    self.assertEqual((metadata["train_end_year"], metadata["validation_start_year"],
                                      metadata["validation_end_year"], metadata["test_start_year"]),
                                     (2003, 2004, 2004, 2005))
                    self.assertNotIn("calibrat", json.dumps(metadata).lower())
                    self.assertFalse(hasattr(cubes, "calibration"))
            self.assertIsInstance(loaded["older"][0].train, ContextArray)
            # Training through 2004 (no validation) joins the two parts again, whatever the cache.
            reference = VALUES[YEAR <= 2004]
            times, nodes = np.array([[0, 2919], [2920, 5839], [5840, 8767]]), np.array([[0, 1]])
            for name, (cubes, _, _) in loaded.items():
                merged = ContextArray(cubes.train, cubes.validation)
                self.assertEqual((len(merged), merged.shape), (len(reference), reference.shape))
                for window in (slice(0, 10), slice(2900, 2940), slice(5800, 5900), slice(2900, 5900), slice(-56, None)):
                    np.testing.assert_array_equal(merged[window], reference[window], err_msg=f"{name} {window}")
                for row in (0, 2919, 2920, 5839, 5840, len(reference) - 1, -1):
                    np.testing.assert_array_equal(merged[row], reference[row])
                np.testing.assert_array_equal(merged[times, nodes, :], reference[times, nodes, :])
                copied = pickle.loads(pickle.dumps(merged))  # What a loader worker receives.
                np.testing.assert_array_equal(copied[2900:5900], reference[2900:5900])
                del copied
        finally:
            for cubes, _, _ in loaded.values():
                cubes.close()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(gc.collect)
        self.root = Path(self.temp.name)

    def run_training(self, cache, name, holdout):
        seen = {}
        result = SimpleNamespace(test_timestamps=TIMES[YEAR >= 2005], metadata={})

        class Fit:
            def __init__(self, train, test, **kwargs):
                # Copies: the cache mappings are closed when the run ends.
                seen.update(kwargs, train=np.array(train[0:len(train)]), test=np.array(test))
                if "validation" in kwargs:
                    seen["validation"] = np.array(kwargs["validation"])

            def __enter__(self):
                return result

            def __exit__(self, *args):
                return False
        with patch.object(run_era5_stgan, "fit_and_score_stgan", Fit), redirect_stdout(io.StringIO()):
            metadata = run_era5_stgan.run_training(prepared_dir=cache, output_dir=self.root / name,
                config=STGANCNNConfig(validation_holdout=holdout), device="cpu")
        return seen, metadata

    def test_split_reaches_the_training_from_either_cache(self):
        for layout in ("current", "older"):
            cache = write_cache(self.root / f"cache_{layout}", older=layout == "older")
            for holdout in (True, False):
                with self.subTest(cache=layout, holdout=holdout):
                    seen, metadata = self.run_training(cache, f"run_{layout}_{holdout}", holdout)
                    last_train = 2003 if holdout else 2004
                    np.testing.assert_array_equal(seen["train"], VALUES[YEAR <= last_train])
                    self.assertTrue(seen["train_timestamps"].equals(TIMES[YEAR <= last_train]))
                    np.testing.assert_array_equal(seen["test"], VALUES[YEAR >= 2005])
                    self.assertTrue(seen["test_timestamps"].equals(TIMES[YEAR >= 2005]))
                    self.assertEqual(seen["score_mode"], "paper")  # The test score never involves the validation.
                    self.assertNotIn("calibration", seen)
                    self.assertNotIn("calibration_timestamps", seen)
                    self.assertEqual("validation" in seen, holdout)
                    if holdout:  # Monitoring and objective see the year 2004, all of it, and nothing else.
                        np.testing.assert_array_equal(seen["validation"], VALUES[YEAR == 2004])
                        self.assertTrue(seen["validation_timestamps"].equals(TIMES[YEAR == 2004]))
                    self.assertEqual((metadata["effective_train_end_year"], metadata["validation_years"]),
                                     (last_train, [2004] if holdout else None))
                    self.assertEqual((metadata["preparation"]["train_end_year"],
                                      metadata["preparation"]["validation_end_year"]), (2003, 2004))
                    saved = (self.root / f"run_{layout}_{holdout}" / "metadata.json").read_text()
                    self.assertNotIn("calibrat", saved.lower())

    def test_other_splits_are_rejected(self):
        cache = write_cache(self.root / "cache", older=False)
        path = cache / "metadata.json"
        original = json.loads(path.read_text())
        for change in ({"train_end_year": 2002}, {"validation_end_year": 2003}, {"test_start_year": 2006}):
            path.write_text(json.dumps({**original, **change}))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "train through 2003, validation 2004"):
                self.run_training(cache, "rejected", True)
            gc.collect()


class PipelineTests(unittest.TestCase):
    """A year-long validation between the training tail of 2003 and the first test days of 2005."""
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.times = pd.date_range("2003-12-29", "2005-01-02 21:00", freq="3h")
        self.year = self.times.year.to_numpy()
        self.values = np.random.default_rng(5).normal(size=(len(self.times), 9, 2)).astype(np.float32)

    def encoders(self):
        yield "convgru", {}
        if HAS_GRAPH:
            yield "gat", dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4096)

    def fit(self, name, validation, **overrides):
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        train, test = self.year == 2003, self.year == 2005
        options = dict(train_timestamps=self.times[train], test_timestamps=self.times[test],
            validation=validation, validation_timestamps=self.times[self.year == 2004],
            location_names=tuple(map(str, range(9))), feature_names=("a", "b"),
            latitudes=45 - rows * .5, longitudes=7 + cols * .5,
            epochs=2, batch_size=4, score_batch_size=512, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=6, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, mc_samples=2, checkpoint_path=self.root / name / "model.pt")
        options.update(overrides)
        with redirect_stdout(io.StringIO()), \
                fit_and_score_stgan(self.values[train], self.values[test], **options) as result:
            return SimpleNamespace(metadata=result.metadata, scores=np.array(result.test_scores),
                std=np.array(result.anomaly_std), history=pd.read_csv(self.root / name / "training_history.csv"),
                weights=torch.load(self.root / name / "model.pt", weights_only=False)["model_state_dict"])

    def test_monitoring_and_objective_read_2004_and_the_test_score_does_not(self):
        validation = self.values[self.year == 2004]
        other = validation.copy()
        other[:-2] *= 7.  # Another 2004, except the two steps that are the input history of the first test steps.
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                first, second = (self.fit(f"{encoder}_{index}", data, **extra)
                                 for index, data in enumerate((validation, other)))
                metadata = first.metadata
                self.assertEqual(list(metadata["splits"]), ["train", "validation", "test"])
                self.assertEqual((metadata["splits"]["validation"]["start"], metadata["splits"]["validation"]["end"],
                                  metadata["splits"]["validation"]["timestamps"]),
                                 ("2004-01-01 00:00:00", "2004-12-31 21:00:00", 366 * 8))
                self.assertEqual(metadata["splits"]["train"]["end"], "2003-12-31 21:00:00")
                self.assertNotIn("calibrat", json.dumps(metadata, default=str).lower())
                # The fixed monitoring timestamps: 32, spread over the whole of 2004 and nowhere else.
                subset = pd.DatetimeIndex(metadata["monitoring"]["subset"]["timestamps"])
                self.assertEqual((len(subset), set(subset.year), subset[0], subset[-1]),
                                 (32, {2004}, pd.Timestamp("2004-01-01"), pd.Timestamp("2004-12-31 21:00")))
                self.assertEqual(sorted(set(subset.month)), list(range(1, 13)))
                self.assertEqual(subset.tolist(), pd.DatetimeIndex(second.metadata["monitoring"]["subset"]["timestamps"]).tolist())
                objective = metadata["validation_objective"]
                self.assertEqual((objective["start"], objective["end"], objective["timestamps"],
                                  objective["candidate_points"], objective["excluded_points"]),
                                 ("2004-01-01 00:00:00", "2004-12-31 21:00:00", 366 * 8, 366 * 8 * 9, 0))
                # Another validation changes what is measured on it, and nothing of the model or of the test.
                self.assertNotEqual(objective["median_plus_p95"], second.metadata["validation_objective"]["median_plus_p95"])
                self.assertFalse(np.allclose(first.history.validation_discriminator_feature_discrepancy,
                                             second.history.validation_discriminator_feature_discrepancy))
                for key, value in first.weights.items():
                    self.assertTrue(torch.equal(value, second.weights[key]), key)
                np.testing.assert_array_equal(first.scores, second.scores)
                np.testing.assert_array_equal(first.std, second.std)
                for key in ("score_normalization", "score_statistics"):
                    self.assertEqual(first.metadata[key], second.metadata[key])
                self.assertEqual(metadata["score_normalization"]["fit_period"], "complete_test_mc_mean_components")
                np.testing.assert_array_equal(first.history.generator_loss, second.history.generator_loss)


if __name__ == "__main__":
    unittest.main()
