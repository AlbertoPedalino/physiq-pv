"""Cyclic time encoding and seasonal input normalization for the patch ConvGRU."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan import (
    STGAN, STGANCNNConfig, STGANWindowDataset, build_spatial_grid,
    fit_and_score_stgan, load_stgan_checkpoint)


def fixture(height=3, width=4, *, trend=4, steps=12, **options):
    rows, cols = np.indices((height, width)).reshape(2, -1)
    grid = build_spatial_grid(45-rows*.5, 7+cols*.5, grid_crs="EPSG:4326",
                              angular_spacing=.5, audit_knn=False)
    data = np.random.default_rng(7).normal(size=(steps, len(rows), 15)).astype(np.float32)
    times = pd.date_range("2004-12-30", periods=steps, freq="3h").as_unit("ns")
    return STGANWindowDataset(data, times, grid, feature_minimum=np.zeros(15),
        feature_scale=np.ones(15), recent_steps=1, trend_steps=trend, stride=1, **options)


class TimeEncodingAndSeasonalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_cyclic_time_encoding_is_local_per_location_and_annual_shared(self):
        from physiq_pv.anomaly_detection.stgan.data import annual_phase_features, cyclic_time_features
        from scripts.run_era5_stgan import parser
        times = pd.DatetimeIndex(["2005-01-01 00:00", "2005-01-01 12:00", "2005-07-02 12:00"])
        hours = np.array([0., 12., 12.])
        # Rows: timestamps; columns: longitudes 0, 90E, 15W (0h, +6h, -1h).
        features = cyclic_time_features(hours[:, None], annual_phase_features(times)[:, None],
                                        np.array([0., 90., -15.]) / 15)
        self.assertEqual(features.shape, (3, 3, 4))
        np.testing.assert_allclose(features[1, :, :2], [[0, -1], [-1, 0], [np.sin(np.pi*11/12), np.cos(np.pi*11/12)]], atol=1e-6)
        np.testing.assert_allclose(features[0, 0], [0, 1, 0, 1], atol=1e-6)
        np.testing.assert_array_equal(features[:, 0, 2:], features[:, 2, 2:])
        np.testing.assert_allclose(features[2, 0, 2:], [0, -1], atol=1e-2)
        onehot = fixture()
        longitudes = 7 + onehot.grid.column_indices * 15.  # One hour of solar time per column.
        cyclic = fixture(time_encoding="cyclic", longitudes=longitudes)
        self.assertEqual((cyclic.time_feature_size, onehot.time_feature_size), (4, 31))
        # Samples are ordered by target time, then location: all 12 cells of the second target.
        calendar = cyclic.fetch_batch(np.arange(12, 24))[3]
        self.assertEqual(tuple(calendar.shape), (12, 4))
        self.assertFalse(torch.allclose(calendar[0, :2], calendar[1, :2]))
        torch.testing.assert_close(calendar[0], calendar[4])  # Same column, same local time.
        torch.testing.assert_close(calendar[:, 2:], calendar[:1, 2:].expand(12, -1))
        torch.testing.assert_close(cyclic[17][3], calendar[5])
        # The one-hot calendar is unchanged: weekday + hour, identical for every location.
        legacy = onehot.fetch_batch(np.arange(12, 24))[3]
        self.assertEqual(tuple(legacy.shape), (12, 31))
        self.assertTrue(torch.equal(legacy, legacy[:1].expand(12, -1)))
        self.assertEqual(legacy.sum(dim=1).tolist(), [2.] * 12)
        model = STGAN(n_features=15, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1,
                      time_feature_size=4, dropout_enabled=False).eval()
        recent, trend, mask, calendar, *_ = cyclic.fetch_batch([12, 17])
        changed = calendar.clone()
        changed[1] += 1
        with torch.no_grad():
            delta = (model.generator(recent, trend, mask, changed)
                     - model.generator(recent, trend, mask, calendar)).abs().sum(dim=(1, 2, 3))
        self.assertGreater(float(delta[1]), 0)
        self.assertEqual(float(delta[0]), 0)
        with self.assertRaises(ValueError):
            STGANCNNConfig(time_encoding="weekly")
        with self.assertRaises(ValueError):
            fixture(time_encoding="cyclic", longitudes=None)
        with self.assertRaises(TypeError):  # The former boolean switch is gone.
            STGANCNNConfig(annual_cycle=True)
        base = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        parse = parser().parse_args
        self.assertEqual((parse(base).time_encoding, parse(base + ["--time-encoding", "cyclic"]).time_encoding),
                         ("onehot", "cyclic"))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse(base + ["--annual-cycle"])

    def test_time_encoding_pipeline_checkpoint_and_resume(self):
        dataset = fixture(3, 3, steps=14)
        data = dataset.data
        options = dict(train_timestamps=dataset.timestamps[:11], test_timestamps=dataset.timestamps[11:],
            location_names=tuple(map(str, range(9))), feature_names=tuple(f"f{i}" for i in range(15)),
            latitudes=45-dataset.grid.row_indices*.5, longitudes=7+dataset.grid.column_indices*.5,
            epochs=1, batch_size=8, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1, trend_steps=4,
            grid_crs="EPSG:4326", angular_grid_spacing=.5, grid_audit_knn=False, timestep_hours=3,
            device="cpu", mc_samples=2, score_mode="paper")
        def run(name, **extra):
            with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(
                    data[:11], data[11:], checkpoint_path=self.root/name/"model.pt",
                    **{**options, **extra}) as result:
                self.assertEqual(result.test_scores.shape, (3, 9))
                self.assertTrue(np.isfinite(result.test_scores).all())
                return result.metadata
        for encoding, size in (("onehot", 31), ("cyclic", 4)):
            metadata = run(encoding, time_encoding=encoding)
            self.assertEqual((metadata["time_encoding"], metadata["time_feature_size"]), (encoding, size))
            self.assertNotIn("annual_cycle", metadata)
            self.assertFalse(metadata["paper_alignment"]["reference_hyperparameters"])
            self.assertEqual("cyclic_local_solar_time_and_annual_phase_instead_of_weekday_hour_onehot"
                             in metadata["paper_alignment"]["domain_adaptations"], encoding == "cyclic")
            model, payload = load_stgan_checkpoint(self.root/encoding/"model.pt")
            self.assertEqual((payload["model_class"], payload["time_encoding"],
                              payload["model_config"]["time_feature_size"]), ("STGAN_CONVGRU", encoding, size))
            self.assertEqual(model.generator.time_projection[0].in_features, size)
        # A one-hot run cannot continue from a cyclic epoch checkpoint, nor the reverse.
        for encoding, other in (("cyclic", "onehot"), ("onehot", "cyclic")):
            with self.assertRaisesRegex(ValueError, "time_encoding"):
                run(encoding, time_encoding=other, epochs=2, resume_from=self.root/encoding/"model_epoch_1.pt")
        # Checkpoints written before time_encoding: annual_cycle=False is the one-hot encoding.
        checkpoint = self.root/"onehot/model_epoch_1.pt"
        legacy = torch.load(checkpoint, weights_only=False)
        del legacy["time_encoding"]
        for annual_cycle, accepted in ((False, True), (True, False)):
            torch.save({**legacy, "annual_cycle": annual_cycle}, checkpoint)
            if accepted:
                self.assertEqual(run("onehot", epochs=2, resume_from=checkpoint)["resume"]["completed_epochs"], 1)
                with self.assertRaisesRegex(ValueError, "time_encoding"):
                    run("onehot", time_encoding="cyclic", epochs=2, resume_from=checkpoint)
            else:
                with self.assertRaisesRegex(ValueError, "time_encoding"):
                    run("onehot", epochs=2, resume_from=checkpoint)
        # Unfused components remain available with the new encoding.
        metadata = run("components", time_encoding="cyclic", score_mode="components")
        self.assertEqual((metadata["score_mode"], metadata["time_feature_size"]), ("components", 4))

    def test_seasonal_climatology_matches_direct_statistics_and_removes_cycles(self):
        from physiq_pv.anomaly_detection.stgan.seasonal import fit_seasonal_climatology, seasonal_memmap
        times = pd.date_range("2001-01-01", "2004-12-31 21:00", freq="3h").as_unit("ns")
        rng = np.random.default_rng(5)
        year = 2 * np.pi * (times.dayofyear.to_numpy() - 1) / 365.25
        hour = 2 * np.pi * times.hour.to_numpy() / 24
        # Location-specific annual and diurnal cycles plus noise; feature 2 is constant.
        amplitude = np.array([1., 2., 3.])[None, :, None]
        data = (1e5 + amplitude * (3 * np.sin(year) + 2 * np.cos(hour))[:, None, None]
                + rng.normal(size=(len(times), 3, 3)) * np.array([1., 2., 0.]))
        data[..., 2] = 7.
        data = data.astype(np.float64)
        climatology = fit_seasonal_climatology(data, times, self.root/"cache", window_days=15)
        try:
            self.assertEqual(climatology.mean.shape, (8, 366, 3, 3))
            self.assertEqual(sorted(p.name for p in (self.root/"cache").iterdir()),
                             ["seasonal_mean.npy", "seasonal_std.npy"])
            # 40th day of year, 09:00, with a circular +-15 day window.
            for day in (40, 3, 360):
                offsets = (times.dayofyear.to_numpy() - 1 - day + 183) % 366 - 183
                window = (np.abs(offsets) <= 15) & (times.hour == 9)
                chosen = data[window]
                np.testing.assert_allclose(climatology.mean[3, day], chosen.mean(axis=0), rtol=1e-6)
                # Spread is measured around each sample's own smoothed daily mean.
                residual = chosen - climatology.mean[3][times.dayofyear.to_numpy()[window] - 1]
                np.testing.assert_allclose(climatology.std[3, day, :, :2],
                                           np.sqrt((residual**2).mean(axis=0))[:, :2], rtol=1e-3)
            self.assertEqual(climatology.metadata["window_days_each_side"], 15)
            self.assertGreaterEqual(climatology.metadata["min_samples_per_bin"], 31)
            standardised = np.array(seasonal_memmap(data, times, climatology, self.root/"cache/train.npy"))
            self.assertEqual(standardised.dtype, np.float32)
            self.assertTrue(np.isfinite(standardised).all())
            np.testing.assert_array_equal(standardised[..., 2], 0)
            # Cycles are gone: every location has ~zero mean and ~unit spread in every month and hour.
            for month, slot in ((1, 0), (7, 4), (10, 7)):
                chosen = np.asarray(standardised[(times.month == month) & (times.hour == slot * 3)])[..., :2]
                np.testing.assert_allclose(chosen.mean(axis=0), 0, atol=.35)
                np.testing.assert_allclose(chosen.std(axis=0), 1, atol=.35)
            raw_spread = data[..., 0].std(axis=0)
            self.assertGreater(raw_spread[2] / raw_spread[0], 2)
            # Unseen period: same bins, statistics untouched; a wrong time of day is rejected.
            later = pd.date_range("2005-03-01", periods=8, freq="3h").as_unit("ns")
            shifted = np.array(seasonal_memmap(data[:8] + 50, later, climatology, self.root/"cache/test.npy"))
            self.assertGreater(float(np.asarray(shifted)[..., 0].min()), 5)
            with self.assertRaisesRegex(ValueError, "time of day"):
                seasonal_memmap(data[:8], later + pd.Timedelta(hours=1), climatology, self.root/"cache/bad.npy")
        finally:
            climatology.close()
        with self.assertRaisesRegex(ValueError, "Too few"):
            fit_seasonal_climatology(data[:16], times[:16], self.root/"short", window_days=0)

    def test_a_line_fitted_on_the_training_years_removes_the_drift(self):
        from physiq_pv.anomaly_detection.stgan.seasonal import (
            fit_seasonal_climatology, fit_seasonal_trend, seasonal_memmap)
        from scripts.run_era5_stgan import parser
        times = pd.date_range("1990-01-01", "2005-12-31 21:00", freq="3h").as_unit("ns")
        rng = np.random.default_rng(3)
        cycle = 3 * np.sin(2 * np.pi * (times.dayofyear.to_numpy() - 1) / 365.25) + 2 * np.cos(2 * np.pi * times.hour.to_numpy() / 24)
        years = times.year.to_numpy() + (times.dayofyear.to_numpy() - 1) / 365.25
        # Feature 0 warms by 0.1 of its noise a year, at every location; feature 1 has no trend.
        data = cycle[:, None, None] + rng.normal(size=(len(times), 4, 2))
        data[..., 0] += 0.1 * (years - 1995)[:, None]
        train = times.year <= 1999
        climatology = fit_seasonal_climatology(data[train], times[train], self.root / "cache", window_days=15)
        try:
            trend = fit_seasonal_trend(data[train], times[train], climatology)
            self.assertEqual((trend["fit_years"], trend["zero_at_year"], trend["timestamp_stride"]), ([1990, 1999], 1995.0, 9))
            # In standard deviations of the training data, which hold a little of the trend itself.
            self.assertAlmostEqual(trend["slope_per_year"][0], 0.1, delta=0.02)
            self.assertAlmostEqual(trend["slope_per_year"][1], 0, delta=0.01)
            later = ~train
            kept = np.asarray(seasonal_memmap(data[later], times[later], climatology, self.root / "kept.npy"))
            removed = np.asarray(seasonal_memmap(data[later], times[later], climatology, self.root / "removed.npy",
                                                 trend=trend))
            # Six years after the training ones: without the line every value is far from the normal.
            self.assertGreater(kept[..., 0].mean(), 0.6)
            self.assertLess(abs(removed[..., 0].mean()), 0.1)
            self.assertLess(abs(removed[-2920:, :, 0].mean()), 0.15)  # the last year too
            np.testing.assert_allclose(removed[..., 1], kept[..., 1], atol=0.1)
            # The training data themselves lose their trend and keep a zero mean.
            fitted = np.asarray(seasonal_memmap(data[train], times[train], climatology, self.root / "train.npy", trend=trend))
            early, late = fitted[:8 * 365 * 3, :, 0].mean(), fitted[-8 * 365 * 3:, :, 0].mean()
            self.assertLess(abs(early - late), 0.1)
            self.assertLess(abs(fitted[..., 0].mean()), 0.02)
            with self.assertRaisesRegex(ValueError, "one slope per feature"):
                seasonal_memmap(data[later], times[later], climatology, self.root / "bad.npy",
                                trend={**trend, "slope_per_year": [0.0]})
            with self.assertRaisesRegex(ValueError, "odd"):
                fit_seasonal_trend(data[train], times[train], climatology, stride=8)
            with self.assertRaisesRegex(ValueError, "two training years"):
                fit_seasonal_trend(data[:2920], times[:2920], climatology)
        finally:
            climatology.close()
        self.assertEqual(STGANCNNConfig().seasonal_trend, "none")
        self.assertEqual(STGANCNNConfig(normalization="seasonal", seasonal_trend="linear").seasonal_trend, "linear")
        with self.assertRaisesRegex(ValueError, "needs normalization=seasonal"):
            STGANCNNConfig(seasonal_trend="linear")
        with self.assertRaisesRegex(ValueError, "none or linear"):
            STGANCNNConfig(normalization="seasonal", seasonal_trend="quadratic")
        base = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        self.assertEqual(parser().parse_args(base).seasonal_trend, "none")
        self.assertEqual(parser().parse_args(base + ["--seasonal-trend", "linear"]).seasonal_trend, "linear")

    def test_a_floor_of_the_standard_deviation_for_some_features(self):
        from physiq_pv.anomaly_detection.stgan.seasonal import fit_seasonal_climatology
        from scripts import run_era5_stgan as cli
        times = pd.date_range("2001-01-01", periods=8 * 365 * 3 + 40, freq="3h").as_unit("ns")
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        rng = np.random.default_rng(6)
        data = rng.normal(size=(len(times), 9, 15)).astype(np.float32)
        # Features 4 and 7 barely vary in summer, where they are large elsewhere.
        quiet = np.isin(times.month, (6, 7, 8))
        for feature in (4, 7):
            data[:, :, feature] *= np.where(quiet, 0.001, 5.0)[:, None]
        split = 8 * 365 * 3
        one = fit_seasonal_climatology(data[:split], times[:split], self.root / "one", std_floor=0.01)
        each = fit_seasonal_climatology(data[:split], times[:split], self.root / "each",
                                        std_floor=[0.3 if feature == 4 else 0.01 for feature in range(15)])
        try:
            self.assertEqual(one.metadata["std_floor_fraction_of_feature_std"], 0.01)
            self.assertEqual(each.metadata["std_floor_fraction_of_feature_std"], [0.01] * 4 + [0.3] + [0.01] * 10)
            np.testing.assert_array_equal(np.delete(np.asarray(each.std), 4, axis=3), np.delete(np.asarray(one.std), 4, axis=3))
            np.testing.assert_array_equal(np.asarray(each.mean), np.asarray(one.mean))
            # In the middle of the summer feature 4 has its higher floor and feature 7 the lower one.
            july = pd.Timestamp("2001-07-15").dayofyear - 1
            self.assertGreater(float(np.asarray(each.std)[0, july, 0, 4]), 20 * float(np.asarray(one.std)[0, july, 0, 4]))
            self.assertEqual(float(np.asarray(each.std)[0, july, 0, 7]), float(np.asarray(one.std)[0, july, 0, 7]))
        finally:
            one.close()
            each.close()
        for wrong in ([0.1, 0.2], [0.1] * 14 + [0.0], float("nan")):
            with self.assertRaisesRegex(ValueError, "std_floor"):
                fit_seasonal_climatology(data[:split], times[:split], self.root / "wrong", std_floor=wrong)
        options = dict(train_timestamps=times[:split], test_timestamps=times[split:],
            location_names=tuple(map(str, range(9))), feature_names=tuple(f"f{i}" for i in range(15)),
            latitudes=45-rows*.5, longitudes=7+cols*.5, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1,
            trend_steps=4, grid_crs="EPSG:4326", angular_grid_spacing=.5, grid_audit_knn=False,
            timestep_hours=3, device="cpu", mc_samples=2, score_mode="paper", train_samples_per_epoch=6,
            time_encoding="cyclic", normalization="seasonal", epochs=1)
        with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(
                data[:split], data[split:], checkpoint_path=self.root / "run" / "model.pt",
                seasonal_std_floors={"f4": 0.3}, **options) as result:
            floors = result.metadata["seasonal_normalization"]["std_floor_fraction_of_feature_std"]
        self.assertEqual(floors, [0.01] * 4 + [0.3] + [0.01] * 10)
        saved = np.load(self.root / "run" / "normalized_cache" / "seasonal_std.npy", mmap_mode="r")
        reference = np.load(self.root / "each" / "seasonal_std.npy", mmap_mode="r")
        np.testing.assert_array_equal(np.asarray(saved), np.asarray(reference))
        del saved, reference
        with self.assertRaisesRegex(ValueError, "not among the features"), contextlib.redirect_stdout(io.StringIO()):
            fit_and_score_stgan(data[:split], data[split:], checkpoint_path=self.root / "bad" / "model.pt",
                                seasonal_std_floors={"rain": 0.3}, **options)
        for wrong in ((("f4", 0),), (("f4", 2),), ((4, 0.1),), ()):
            with self.assertRaisesRegex(ValueError, "seasonal_std_floors"):
                STGANCNNConfig(normalization="seasonal", seasonal_std_floors=wrong)
        with self.assertRaisesRegex(ValueError, "seasonal_std_floors"):
            STGANCNNConfig(seasonal_std_floors=(("f4", 0.1),))
        self.assertIsNone(STGANCNNConfig().seasonal_std_floors)
        base = ["train", "--prepared-dir", "p", "--output-dir", "o"]
        self.assertIsNone(cli.floors_by_variable(cli.parser().parse_args(base).seasonal_std_floors))
        args = cli.parser().parse_args(base + ["--seasonal-std-floors", "total_precipitation=0.1", "soil=0.3"])
        self.assertEqual(cli.floors_by_variable(args.seasonal_std_floors), (("total_precipitation", 0.1), ("soil", 0.3)))
        with self.assertRaisesRegex(ValueError, "VARIABLE=FLOOR"):
            cli.floors_by_variable(["total_precipitation"])

    def test_a_feature_is_standardised_on_its_fourth_root(self):
        from physiq_pv.anomaly_detection.stgan.seasonal import FourthRoot, fit_seasonal_climatology, seasonal_memmap
        from scripts.run_era5_stgan import parser
        values = np.arange(24, dtype=np.float32).reshape(4, 2, 3) - 2
        read = FourthRoot(values, [1])
        self.assertEqual((len(read), read.shape, read.ndim), (4, (4, 2, 3), 3))
        block = read[1:3]
        np.testing.assert_array_equal(block[..., [0, 2]], values[1:3][..., [0, 2]])
        np.testing.assert_allclose(block[..., 1], np.maximum(values[1:3, :, 1], 0) ** 0.25)
        self.assertEqual(read[0:1][0, 0, 1], 0)  # a value below 0
        np.testing.assert_array_equal(values[0, 0], [-2, -1, 0])  # the data are not changed
        # In the pipeline: the rain, 0 most of the time, is standardised as its fourth root would be.
        times = pd.date_range("2001-01-01", periods=8 * 365 * 3 + 40, freq="3h").as_unit("ns")
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        rng = np.random.default_rng(4)
        data = rng.normal(size=(len(times), 9, 15)).astype(np.float32)
        data[..., 4] = rng.exponential(size=(len(times), 9)) * (rng.random((len(times), 9)) < 0.3)
        split = 8 * 365 * 3
        options = dict(train_timestamps=times[:split], test_timestamps=times[split:],
            location_names=tuple(map(str, range(9))), feature_names=tuple(f"f{i}" for i in range(15)),
            latitudes=45-rows*.5, longitudes=7+cols*.5, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1,
            trend_steps=4, grid_crs="EPSG:4326", angular_grid_spacing=.5, grid_audit_knn=False,
            timestep_hours=3, device="cpu", mc_samples=2, score_mode="paper", train_samples_per_epoch=6,
            time_encoding="cyclic", normalization="seasonal", epochs=1)
        caches, details = {}, {}
        for name, features in (("plain", None), ("root", ("f4",))):
            with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(
                    data[:split], data[split:], checkpoint_path=self.root / name / "model.pt",
                    seasonal_fourth_root_features=features, **options) as result:
                details[name] = result.metadata
            caches[name] = np.array(np.load(self.root / name / "normalized_cache" / "train.npy", mmap_mode="r"))
        self.assertEqual(details["root"]["seasonal_normalization"]["fourth_root_features"], ["f4"])
        self.assertIsNone(details["plain"]["seasonal_normalization"]["fourth_root_features"])
        self.assertIn("fourth_root_of_some_inputs_before_their_standardisation",
                      details["root"]["paper_alignment"]["domain_adaptations"])
        np.testing.assert_array_equal(np.delete(caches["root"], 4, axis=2), np.delete(caches["plain"], 4, axis=2))
        rooted = data[:split].astype(np.float64)
        rooted[..., 4] = rooted[..., 4] ** 0.25
        climatology = fit_seasonal_climatology(rooted, times[:split], self.root / "direct")
        try:
            expected = np.array(seasonal_memmap(rooted, times[:split], climatology, self.root / "direct" / "train.npy"))
        finally:
            climatology.close()
        np.testing.assert_allclose(caches["root"][..., 4], expected[..., 4], atol=1e-5)
        # Less of the fourth root is far from the rest than of the rain itself.
        self.assertLess(np.abs(caches["root"][..., 4]).max(), np.abs(caches["plain"][..., 4]).max())
        with self.assertRaisesRegex(ValueError, "not among the features"), contextlib.redirect_stdout(io.StringIO()):
            fit_and_score_stgan(data[:split], data[split:], checkpoint_path=self.root / "wrong" / "model.pt",
                                seasonal_fourth_root_features=("rain",), **options)
        with self.assertRaisesRegex(ValueError, "needs normalization=seasonal"):
            STGANCNNConfig(seasonal_fourth_root_features=("f4",))
        self.assertIsNone(STGANCNNConfig().seasonal_fourth_root_features)
        base = ["train", "--prepared-dir", "p", "--output-dir", "o"]
        self.assertIsNone(parser().parse_args(base).seasonal_fourth_root_features)
        self.assertEqual(parser().parse_args(base + ["--seasonal-fourth-root-features", "total_precipitation"]
                                             ).seasonal_fourth_root_features, ["total_precipitation"])

    def test_the_pipeline_removes_the_line_from_every_period(self):
        times = pd.date_range("2001-01-01", periods=8 * 365 * 4 + 40, freq="3h").as_unit("ns")
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        phase = 2 * np.pi * (times.dayofyear.to_numpy() / 365 + times.hour.to_numpy() / 24)
        years = times.year.to_numpy() + (times.dayofyear.to_numpy() - 1) / 365.25
        data = (np.sin(phase)[:, None, None] + np.random.default_rng(9).normal(size=(len(times), 9, 15)) * .1)
        data[..., 0] += 0.05 * (years - 2003)[:, None]  # half the noise a year
        data = data.astype(np.float32)
        split = 8 * 365 * 4
        options = dict(train_timestamps=times[:split], test_timestamps=times[split:],
            location_names=tuple(map(str, range(9))), feature_names=tuple(f"f{i}" for i in range(15)),
            latitudes=45-rows*.5, longitudes=7+cols*.5, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1,
            trend_steps=4, grid_crs="EPSG:4326", angular_grid_spacing=.5, grid_audit_knn=False,
            timestep_hours=3, device="cpu", mc_samples=2, score_mode="paper", train_samples_per_epoch=6,
            time_encoding="cyclic", normalization="seasonal", epochs=1)
        caches, details = {}, {}
        for name in ("none", "linear"):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), fit_and_score_stgan(
                    data[:split], data[split:], checkpoint_path=self.root / name / "model.pt",
                    seasonal_trend=name, **options) as result:
                details[name] = result.metadata
            caches[name] = {part: np.array(np.load(self.root / name / "normalized_cache" / f"{part}.npy", mmap_mode="r"))
                            for part in ("train", "test")}
            self.assertEqual("line in time removed" in output.getvalue(), name == "linear")
        trend = details["linear"]["seasonal_normalization"]["trend"]
        self.assertIsNone(details["none"]["seasonal_normalization"]["trend"])
        self.assertEqual(trend["fit_years"], [2001, 2004])
        self.assertGreater(trend["slope_per_year"][0], 0.2)
        self.assertLess(max(map(abs, trend["slope_per_year"][1:])), 0.05)
        self.assertIn("line_in_time_of_the_training_years_removed_from_the_standardised_inputs",
                      details["linear"]["paper_alignment"]["domain_adaptations"])
        self.assertNotIn("line_in_time_of_the_training_years_removed_from_the_standardised_inputs",
                         details["none"]["paper_alignment"]["domain_adaptations"])
        # The test days follow the training years: with the line they are back at the normal.
        self.assertGreater(caches["none"]["test"][..., 0].mean(), 0.5)
        self.assertLess(abs(caches["linear"]["test"][..., 0].mean()), 0.3)
        np.testing.assert_allclose(caches["linear"]["test"][..., 1:], caches["none"]["test"][..., 1:], atol=0.15)
        _, payload = load_stgan_checkpoint(self.root / "linear" / "model.pt")
        self.assertEqual(payload["normalization"]["seasonal"]["trend"]["slope_per_year"], trend["slope_per_year"])
        # A higher floor of the standard deviation is another normalization, named in the run.
        with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(
                data[:split], data[split:], checkpoint_path=self.root / "floor" / "model.pt", seasonal_std_floor=0.5,
                **options) as result:
            floored = result.metadata["seasonal_normalization"]["std_floor_fraction_of_feature_std"]
        self.assertEqual((floored, details["none"]["seasonal_normalization"]["std_floor_fraction_of_feature_std"]),
                         (0.5, 0.01))
        low = np.load(self.root / "none" / "normalized_cache" / "seasonal_std.npy", mmap_mode="r")
        high = np.load(self.root / "floor" / "normalized_cache" / "seasonal_std.npy", mmap_mode="r")
        self.assertTrue((np.asarray(high) >= np.asarray(low)).all() and (np.asarray(high) > np.asarray(low)).any())
        del low, high
        self.assertEqual(STGANCNNConfig().seasonal_std_floor, 0.01)
        for wrong in (0, 1.5, "0.1"):
            with self.assertRaisesRegex(ValueError, "seasonal_std_floor"):
                STGANCNNConfig(normalization="seasonal", seasonal_std_floor=wrong)
        from scripts.run_era5_stgan import parser as cli
        base = ["train", "--prepared-dir", "p", "--output-dir", "o"]
        self.assertEqual((cli().parse_args(base).seasonal_std_floor,
                          cli().parse_args(base + ["--seasonal-std-floor", "0.1"]).seasonal_std_floor), (0.01, 0.1))
        # The line of some features only: the others are standardised as without a trend.
        with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(
                data[:split], data[split:], checkpoint_path=self.root / "some" / "model.pt", seasonal_trend="linear",
                seasonal_trend_features=("f0", "f3"), **options) as result:
            some = result.metadata["seasonal_normalization"]["trend"]
        self.assertEqual(some["features"], ["f0", "f3"])
        self.assertEqual(some["slope_per_year"][0], trend["slope_per_year"][0])
        self.assertEqual(some["slope_per_year"][3], trend["slope_per_year"][3])
        self.assertEqual([slope for index, slope in enumerate(some["slope_per_year"]) if index not in (0, 3)], [0.0] * 13)
        cached = np.array(np.load(self.root / "some" / "normalized_cache" / "test.npy", mmap_mode="r"))
        np.testing.assert_array_equal(cached[..., 0], caches["linear"]["test"][..., 0])
        np.testing.assert_array_equal(cached[..., 1], caches["none"]["test"][..., 1])
        with self.assertRaisesRegex(ValueError, "not among the features"), contextlib.redirect_stdout(io.StringIO()):
            fit_and_score_stgan(data[:split], data[split:], checkpoint_path=self.root / "wrong" / "model.pt",
                                seasonal_trend="linear", seasonal_trend_features=("sea_ice",), **options)
        with self.assertRaisesRegex(ValueError, "seasonal_trend=linear"):
            STGANCNNConfig(normalization="seasonal", seasonal_trend_features=("f0",))
        from scripts.run_era5_stgan import parser
        args = parser().parse_args(["train", "--prepared-dir", "p", "--output-dir", "o", "--normalization", "seasonal",
                                    "--seasonal-trend", "linear", "--seasonal-trend-features", "2m_temperature", "temperature"])
        self.assertEqual(args.seasonal_trend_features, ["2m_temperature", "temperature"])
        self.assertIsNone(parser().parse_args(["train", "--prepared-dir", "p", "--output-dir", "o"]).seasonal_trend_features)
        # A run cannot go on from a checkpoint of the other normalization.
        with self.assertRaisesRegex(ValueError, "normalization"), contextlib.redirect_stdout(io.StringIO()):
            fit_and_score_stgan(data[:split], data[split:], checkpoint_path=self.root / "linear" / "model.pt",
                                resume_from=self.root / "linear" / "model_epoch_1.pt", seasonal_trend="none",
                                **{**options, "epochs": 2})

    def test_seasonal_normalization_pipeline_resume_and_cli(self):
        from physiq_pv.anomaly_detection.stgan.seasonal import fit_seasonal_climatology
        from scripts.run_era5_stgan import parser
        times = pd.date_range("2003-01-01", periods=8 * 365 + 20, freq="3h").as_unit("ns")
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        phase = 2 * np.pi * (times.dayofyear.to_numpy() / 365 + times.hour.to_numpy() / 24)
        data = (np.sin(phase)[:, None, None] * (1 + rows)[None, :, None]
                + np.random.default_rng(9).normal(size=(len(times), 9, 15)) * .1).astype(np.float32)
        split = 8 * 365
        options = dict(train_timestamps=times[:split], test_timestamps=times[split:],
            location_names=tuple(map(str, range(9))), feature_names=tuple(f"f{i}" for i in range(15)),
            latitudes=45-rows*.5, longitudes=7+cols*.5,
            hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1,
            trend_steps=4, grid_crs="EPSG:4326", angular_grid_spacing=.5, grid_audit_knn=False,
            timestep_hours=3, device="cpu", mc_samples=2, score_mode="paper",
            train_samples_per_epoch=6, time_encoding="cyclic", normalization="seasonal")
        root = self.root/"seasonal"
        with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(data[:split], data[split:],
                checkpoint_path=root/"model.pt", epochs=1, **options) as result:
            self.assertEqual(result.test_scores.shape, (20, 9))
            self.assertTrue(np.isfinite(result.test_scores).all())
            metadata = result.metadata
        self.assertEqual(metadata["normalization"], "training_only_seasonal_standardisation_then_feature_minmax")
        self.assertEqual(metadata["seasonal_normalization"]["window_days_each_side"], 15)
        # Fitted on the training period only: the test split never reaches the statistics.
        self.assertEqual(metadata["seasonal_normalization"]["fit_period"],
                         {"start": str(times[0]), "end": str(times[split-1]), "timestamps": split})
        self.assertFalse(metadata["runtime"]["normalized_disk_cache"])
        self.assertFalse(metadata["paper_alignment"]["reference_hyperparameters"])
        self.assertIn("inputs_standardised_per_location_day_of_year_and_time_of_day",
                      metadata["paper_alignment"]["domain_adaptations"])
        self.assertTrue((root/"normalized_cache/seasonal_mean.npy").is_file())
        reference = fit_seasonal_climatology(data[:split], times[:split], self.root/"reference")
        try:
            np.testing.assert_array_equal(np.load(root/"normalized_cache/seasonal_mean.npy"), reference.mean)
            np.testing.assert_array_equal(np.load(root/"normalized_cache/seasonal_std.npy"), reference.std)
        finally:
            reference.close()
        cached = np.load(root/"normalized_cache/train.npy", mmap_mode="r")
        self.assertLess(abs(float(np.asarray(cached).mean())), .05)
        del cached
        _, payload = load_stgan_checkpoint(root/"model.pt")
        self.assertTrue(payload["normalization"]["kind"].startswith("training_only_seasonal"))
        self.assertEqual(payload["normalization"]["seasonal"]["day_bins"], 366)
        with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(data[:split], data[split:],
                checkpoint_path=root/"model.pt", epochs=2, resume_from=root/"model_epoch_1.pt", **options) as result:
            self.assertEqual(result.metadata["resume"]["completed_epochs"], 1)
        with self.assertRaisesRegex(ValueError, "normalization"), contextlib.redirect_stdout(io.StringIO()):
            fit_and_score_stgan(data[:split], data[split:], checkpoint_path=root/"model.pt", epochs=2,
                resume_from=root/"model_epoch_1.pt", **{**options, "normalization": "minmax"})
        with self.assertRaisesRegex(ValueError, "normalization"), contextlib.redirect_stdout(io.StringIO()):
            fit_and_score_stgan(data[:split], data[split:], checkpoint_path=root/"model.pt", epochs=2,
                resume_from=root/"model_epoch_1.pt", **{**options, "seasonal_window_days": 5})
        with self.assertRaisesRegex(ValueError, "checkpoint_path"), contextlib.redirect_stdout(io.StringIO()):
            fit_and_score_stgan(data[:split], data[split:], epochs=1, **options)
        # The default stays the train-only min-max, with no seasonal files.
        with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(data[:split], data[split:],
                checkpoint_path=self.root/"minmax/model.pt", epochs=1,
                **{key: value for key, value in options.items() if key != "normalization"}) as result:
            self.assertEqual((result.metadata["normalization"], result.metadata["seasonal_normalization"]),
                             ("training_only_feature_minmax", None))
        self.assertFalse((self.root/"minmax/normalized_cache/seasonal_mean.npy").exists())
        self.assertEqual((STGANCNNConfig().normalization, STGANCNNConfig().time_encoding), ("minmax", "onehot"))
        with self.assertRaises(ValueError):
            STGANCNNConfig(normalization="zscore")
        base = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        parse = parser().parse_args
        args = parse(base + ["--normalization", "seasonal", "--seasonal-window-days", "10"])
        self.assertEqual((parse(base).normalization, args.normalization, args.seasonal_window_days),
                         ("minmax", "seasonal", 10))


if __name__ == "__main__":
    unittest.main()
