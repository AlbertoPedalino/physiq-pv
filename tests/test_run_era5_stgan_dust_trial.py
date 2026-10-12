"""The dust trial must add the CAMS dust as a 16th variable, split by time and write a run the checks read."""
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

from physiq_pv.anomaly_detection.stgan import STGANCNNConfig
from physiq_pv.era5.data import load_prepared
from physiq_pv.era5.features import FEATURE_NAMES
from scripts import rebalance_era5_stgan_scores as rebalance
from scripts import run_era5_stgan_dust_trial as script

ROWS, COLS = np.indices((3, 3)).reshape(2, -1)
LATITUDES, LONGITUDES = np.array([45.0, 44.5, 44.0]), np.array([7.0, 7.5, 8.0])
TEST_TIMES = pd.date_range("2005-01-01", "2005-02-09 21:00", freq="3h")
DUST_DAYS = pd.date_range("2005-01-01", "2005-02-09", freq="D")
DUST_LATITUDES, DUST_LONGITUDES = np.array([46.5, 45.75, 45.0, 44.25, 43.5]), np.array([6.0, 6.75, 7.5, 8.25, 9.0])


def dust_field(day, latitude, longitude):
    """Linear in the day and in the coordinates: the interpolations must give it back exactly."""
    return 0.01 * day + 0.1 * latitude + 0.2 * longitude


class Fixture(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        prepared = self.root / "prepared"
        prepared.mkdir()
        periods = {"train": pd.date_range("2003-12-31", periods=8, freq="3h"),
                   "validation": pd.date_range("2004-12-31", periods=8, freq="3h"), "test": TEST_TIMES}
        self.test = rng.normal(size=(len(TEST_TIMES), 9, len(FEATURE_NAMES))).astype(np.float32)
        for name, times in periods.items():
            values = self.test if name == "test" else rng.normal(size=(len(times), 9, len(FEATURE_NAMES))).astype(np.float32)
            np.save(prepared / f"{name}.npy", values)
            np.save(prepared / f"{name}_timestamps.npy", times.as_unit("ns").asi8)
        (prepared / "metadata.json").write_text(json.dumps({
            "status": "complete", "features": list(FEATURE_NAMES), "start_year": 2003, "train_end_year": 2003,
            "validation_start_year": 2004, "validation_end_year": 2004, "test_start_year": 2005, "score_end_year": 2005,
            "grid": {"latitudes": LATITUDES.tolist(), "longitudes": LONGITUDES.tolist(), "rows": ROWS.tolist(),
                     "cols": COLS.tolist(), "spacing": 0.5}}), encoding="utf-8")
        (self.root / "dust").mkdir()
        self.dust = np.array([dust_field(day, DUST_LATITUDES[:, None], DUST_LONGITUDES[None, :])
                              for day in range(len(DUST_DAYS))], dtype=np.float32)
        np.savez(self.root / "dust" / "dust_2005.npz", dust=self.dust, days=DUST_DAYS.values.astype("datetime64[D]"),
                 latitudes=DUST_LATITUDES, longitudes=DUST_LONGITUDES)
        self.cell_latitudes, self.cell_longitudes = LATITUDES[ROWS], LONGITUDES[COLS]


class InterpolationTests(Fixture):
    def test_dust_on_the_cells_is_bilinear_and_clamped_at_the_edge(self):
        daily = script.dust_on_cells(self.dust, DUST_LATITUDES, DUST_LONGITUDES, self.cell_latitudes, self.cell_longitudes)
        self.assertEqual(daily.shape, (len(DUST_DAYS), 9))
        expected = np.array([dust_field(day, self.cell_latitudes, self.cell_longitudes) for day in range(len(DUST_DAYS))])
        np.testing.assert_allclose(daily, expected, rtol=1e-5)
        # A cell outside the dust grid takes the value of the nearest edge.
        outside = script.dust_on_cells(self.dust[:1], DUST_LATITUDES, DUST_LONGITUDES, np.array([50.0]), np.array([20.0]))
        np.testing.assert_allclose(outside, [[dust_field(0, 46.5, 9.0)]], rtol=1e-5)

    def test_dust_at_the_timestamps_is_linear_between_the_daily_means_at_noon(self):
        daily = np.arange(4, dtype=np.float32)[:, None] * np.array([1.0, 10.0], dtype=np.float32)
        days = pd.date_range("2005-01-01", periods=4, freq="D")
        times = pd.DatetimeIndex(["2005-01-01 00:00", "2005-01-01 12:00", "2005-01-02 00:00", "2005-01-02 12:00",
                                  "2005-01-02 18:00", "2005-01-04 12:00", "2005-01-04 21:00"])
        np.testing.assert_allclose(script.dust_at(daily, days, times)[:, 0], [0, 0, 0.5, 1, 1.25, 3, 3], atol=1e-6)
        np.testing.assert_allclose(script.dust_at(daily, days, times)[:, 1], [0, 0, 5, 10, 12.5, 30, 30], atol=1e-5)


class ArrayTests(Fixture):
    def test_arrays_hold_the_cache_and_the_dust_and_split_at_the_boundary(self):
        cubes = load_prepared(self.root / "prepared")[0]
        try:
            arrays = script.build_arrays(cubes, self.dust, DUST_DAYS, DUST_LATITUDES, DUST_LONGITUDES,
                                         self.root / "data", train_end=pd.Timestamp("2005-01-31"))
            (train, train_times), (test, test_times) = arrays["train"], arrays["test"]
            self.assertEqual((train.shape, test.shape), ((240, 9, 16), (80, 9, 16)))
            self.assertEqual((train_times[0], train_times[-1], test_times[0], test_times[-1]),
                             (TEST_TIMES[0], pd.Timestamp("2005-01-30 21:00"), pd.Timestamp("2005-01-31"), TEST_TIMES[-1]))
            np.testing.assert_array_equal(train[:, :, :15], self.test[:240])
            np.testing.assert_array_equal(test[:, :, :15], self.test[240:])
            # The dust at 12 UTC of a day is the daily field of that day on the cell.
            noon = int(np.flatnonzero(test_times == pd.Timestamp("2005-02-03 12:00"))[0])
            day = int(np.flatnonzero(DUST_DAYS == pd.Timestamp("2005-02-03"))[0])
            np.testing.assert_allclose(test[noon, :, 15], dust_field(day, self.cell_latitudes, self.cell_longitudes), rtol=1e-5)
            self.assertTrue(np.isfinite(train).all() and np.isfinite(test).all())
            for array, _ in arrays.values():
                array._mmap.close()
            for boundary in ("2004-01-01", "2006-01-01"):
                with self.assertRaisesRegex(ValueError, "no training or no test"):
                    script.build_arrays(cubes, self.dust, DUST_DAYS, DUST_LATITUDES, DUST_LONGITUDES,
                                        self.root / "other", train_end=pd.Timestamp(boundary))
            with self.assertRaisesRegex(ValueError, "missing days"):
                script.build_arrays(cubes, self.dust[::2], DUST_DAYS[::2], DUST_LATITUDES, DUST_LONGITUDES,
                                    self.root / "gaps", train_end=pd.Timestamp("2005-01-31"))
        finally:
            cubes.close()


class RunTests(Fixture):
    def test_trial_run_is_a_run_with_sixteen_variables(self):
        config = STGANCNNConfig(epochs=1, batch_size=1, score_batch_size=1, cnn_training_mode="full_grid", hidden_size=4,
                                n_layers=1, cnn_channels=2, cnn_layers=1, recent_steps=2, trend_steps=2,
                                time_encoding="cyclic", grid_crs="EPSG:4326", num_workers=0, mc_samples=2,
                                score_storage="memmap", discriminator_input_dropout=False)
        output = io.StringIO()
        with redirect_stdout(output):
            manifest = script.run_trial(prepared_dir=self.root / "prepared", dust_dir=self.root / "dust",
                                        output_dir=self.root / "run", config=config, train_end="2005-01-31", device="cpu")
        names = [*FEATURE_NAMES, "dust_aerosol_optical_depth_550nm"]
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["preparation"]["features"], names)
        self.assertEqual(manifest["splits"], {"train": ["2005-01-01 00:00:00", "2005-01-30 21:00:00"],
                                              "test": ["2005-01-31 00:00:00", "2005-02-09 21:00:00"]})
        self.assertIn("variables 16", output.getvalue())
        saved = json.loads((self.root / "run" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["backend"]["discriminator_input_dropout"], False)
        self.assertEqual(saved["grid"]["rows"], ROWS.tolist())
        scores = self.root / "run" / "scores"
        self.assertEqual(np.load(scores / "feature_scores.npy", mmap_mode="r").shape, (80, 9, 16))
        self.assertEqual(np.load(scores / "anomaly_mean.npy", mmap_mode="r").shape, (80, 9))
        self.assertEqual(len(np.load(self.root / "run" / "test_timestamps.npy")), 80)
        # The rebalancing reads the run and gives the dust the weight of any other variable.
        balanced = rebalance.rebalance(self.root / "run", self.root / "balanced", scale="rarity", scale_stride=1)
        self.assertEqual(balanced["features"], names)
        self.assertAlmostEqual(balanced["share_of_the_new_score"]["dust_aerosol_optical_depth_550nm"], 1 / 16, delta=0.02)
        with self.assertRaisesRegex(ValueError, "new/empty"):
            script.run_trial(prepared_dir=self.root / "prepared", dust_dir=self.root / "dust",
                             output_dir=self.root / "run", config=config, train_end="2005-01-31", device="cpu")

    def test_command_line_defaults_are_the_last_full_grid_runs(self):
        args = script.parser().parse_args(["--prepared-dir", "p", "--dust-dir", "d", "--output-dir", "o"])
        config = script.trial_config(args)
        self.assertEqual((args.train_years, args.train_end, config.epochs, config.cnn_training_mode, config.recent_steps,
                          config.time_encoding, config.precision, config.discriminator_input_dropout,
                          config.reconstruction_weights, config.normalization, config.validation_holdout),
                         (15, None, 1, "full_grid", 16, "cyclic", "bf16", False, "uniform", "minmax", False))
        weighted = script.trial_config(script.parser().parse_args(
            ["--prepared-dir", "p", "--dust-dir", "d", "--output-dir", "o", "--reconstruction-weights", "daily_change",
             "--epochs", "3", "--normalization", "seasonal"]))
        self.assertEqual((weighted.reconstruction_weights, weighted.epochs, weighted.normalization),
                         ("daily_change", 3, "seasonal"))

    def test_trial_run_with_the_seasonal_normalization_and_the_weights(self):
        config = STGANCNNConfig(epochs=1, batch_size=1, score_batch_size=1, cnn_training_mode="full_grid", hidden_size=4,
                                n_layers=1, cnn_channels=2, cnn_layers=1, recent_steps=2, trend_steps=2,
                                time_encoding="cyclic", grid_crs="EPSG:4326", num_workers=0, mc_samples=2,
                                score_storage="memmap", discriminator_input_dropout=False, normalization="seasonal",
                                seasonal_window_days=183, reconstruction_weights="daily_change")  # 30 days of training
        output = io.StringIO()
        with redirect_stdout(output):
            manifest = script.run_trial(prepared_dir=self.root / "prepared", dust_dir=self.root / "dust",
                                        output_dir=self.root / "run", config=config, train_end="2005-01-31", device="cpu")
        self.assertEqual(manifest["config"]["normalization"], "seasonal")
        self.assertIsNotNone(manifest["backend"]["seasonal_normalization"])
        self.assertIn("dust_aerosol_optical_depth_550nm=", output.getvalue())
        scores = np.load(self.root / "run" / "scores" / "feature_scores.npy", mmap_mode="r")
        self.assertEqual(scores.shape, (80, 9, 16))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
