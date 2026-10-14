"""A validation of normal weather: the years after an earlier end of training, the values flagged with no
model, the PCA and the MMD references on fields of normal weather, and a run that leaves the flagged
values out of its validation and replaces them in the history of its windows.

The whole path of the commands runs here on a small prepared cache, as on the server.
"""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan.full_grid import STGANFullGridDataset
from physiq_pv.anomaly_detection.stgan.grid import build_spatial_grid
from physiq_pv.anomaly_detection.stgan.normal_masks import load_normal_masks, normal_rows
from physiq_pv.anomaly_detection.stgan.objective import reconstruction_statistics
from physiq_pv.era5.cube import CubeGrid
from physiq_pv.era5.data import load_prepared
from physiq_pv.era5.features import FEATURE_NAMES
from physiq_pv.experiments.stgan_wandb import default_config
from scripts import run_era5_stgan

LATITUDES, LONGITUDES = np.array([45.0, 44.5, 44.0]), np.array([7.0, 7.5, 8.0])
ROWS, COLS = (axis.ravel() for axis in np.indices((3, 3)))
TIMES = pd.date_range("2002-01-01", "2005-01-10 21:00", freq="3h").as_unit("ns")
SPLIT = {"train": (2002, 2003), "validation": (2004, 2004), "test": (2005, 2005)}
# Days of unusual weather in 2003, a validation year once the training ends in 2002: one cell far from normal.
UNUSUAL = [pd.Timestamp("2003-03-10 12:00"), pd.Timestamp("2003-07-20 06:00"), pd.Timestamp("2004-10-05 15:00")]
UNUSUAL_CELL = 4


def weather():
    """[time, 9 cells, 15 variables]: a yearly and a daily cycle, noise, and the unusual values."""
    rng = np.random.default_rng(0)
    year = 2 * np.pi * TIMES.dayofyear.to_numpy() / 365.25
    day = 2 * np.pi * TIMES.hour.to_numpy() / 24
    amplitude = rng.uniform(.5, 2, size=(1, 9, len(FEATURE_NAMES)))
    values = (amplitude * (np.sin(year)[:, None, None] + .5 * np.cos(day)[:, None, None])
              + .3 * rng.normal(size=(len(TIMES), 9, len(FEATURE_NAMES))))
    for moment in UNUSUAL:
        values[TIMES.get_loc(moment), UNUSUAL_CELL] += 6.0
    return values.astype(np.float32)


def prepare(directory):
    """A prepared cache as `prepare_era5` writes it: training 2002-2003, validation 2004, test from 2005."""
    directory.mkdir()
    values = weather()
    for name, (first, last) in SPLIT.items():
        chosen = (TIMES.year >= first) & (TIMES.year <= last)
        np.save(directory / f"{name}.npy", values[chosen])
        np.save(directory / f"{name}_timestamps.npy", TIMES[chosen].asi8)
    grid = CubeGrid(LATITUDES, LONGITUDES, ROWS, COLS)
    (directory / "metadata.json").write_text(json.dumps({
        "status": "complete", "grid": grid.to_dict(), "features": list(FEATURE_NAMES), "start_year": 2002,
        "train_end_year": 2003, "validation_start_year": 2004, "validation_end_year": 2004,
        "test_start_year": 2005, "score_end_year": 2005}), encoding="utf-8")
    return values


def quiet(function, *args, **options):
    with redirect_stdout(io.StringIO()):
        return function(*args, **options)


class LoadTests(unittest.TestCase):
    def test_an_earlier_end_of_training_moves_the_years_after_it_to_the_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            values = prepare(Path(temp) / "prepared")
            cubes, _, metadata = load_prepared(Path(temp) / "prepared", train_end_year=2002)
            try:
                self.assertEqual((cubes.train_timestamps[-1], cubes.validation_timestamps[0], cubes.validation_timestamps[-1]),
                                 (pd.Timestamp("2002-12-31 21:00"), pd.Timestamp("2003-01-01"), pd.Timestamp("2004-12-31 21:00")))
                self.assertEqual((metadata["train_end_year"], metadata["validation_start_year"]), (2002, 2003))
                chosen = (TIMES.year >= 2003) & (TIMES.year <= 2004)
                np.testing.assert_array_equal(np.asarray(cubes.validation[:]), values[chosen])
                np.testing.assert_array_equal(np.asarray(cubes.train), values[TIMES.year == 2002])
                # Read across the two files, one row and a slice.
                last = int((TIMES.year == 2003).sum())
                np.testing.assert_array_equal(cubes.validation[last], values[chosen][last])
                np.testing.assert_array_equal(cubes.validation[last - 2:last + 2], values[chosen][last - 2:last + 2])
            finally:
                cubes.close()
            same = load_prepared(Path(temp) / "prepared", train_end_year=2003)
            self.assertEqual(len(same[0].validation_timestamps), int((TIMES.year == 2004).sum()))
            same[0].close()
            later = load_prepared(Path(temp) / "prepared", train_end_year=2004)  # a later end is not applied
            self.assertEqual(later[2]["train_end_year"], 2003)
            later[0].close()
            with self.assertRaisesRegex(ValueError, "before the first year"):
                load_prepared(Path(temp) / "prepared", train_end_year=2001)


class PieceTests(unittest.TestCase):
    def test_normal_rows_are_as_many_in_every_month_and_among_the_most_normal(self):
        times = pd.date_range("2001-01-01", "2002-12-31 21:00", freq="3h")
        share = np.random.default_rng(1).random(len(times))
        rows = normal_rows(times, share, 120, share_quantile=.5, seed=3)
        self.assertEqual(len(np.unique(rows)), 120)
        self.assertTrue((np.bincount(times[rows].month, minlength=13)[1:] == 10).all())
        for month in range(1, 13):
            inside = times.month == month
            self.assertTrue((share[rows][times[rows].month == month] <= np.quantile(share[inside], .5)).all())
        np.testing.assert_array_equal(rows, normal_rows(times, share, 120, share_quantile=.5, seed=3))
        with self.assertRaisesRegex(ValueError, "needed"):
            normal_rows(times, share, 12000, share_quantile=.01)

    def test_the_history_comes_from_the_inputs_and_the_target_from_the_data(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(12, 9, 2)).astype(np.float32)
        inputs = data.copy()
        inputs[5:8] = 0
        times = pd.date_range("2003-01-01", periods=12, freq="3h")
        grid = build_spatial_grid(45 - np.repeat([0, .5, 1], 3), np.tile([7, 7.5, 8], 3), patch_size=3,
                                  grid_crs="EPSG:4326", angular_spacing=.5, audit_knn=False)
        options = dict(feature_minimum=np.zeros(2, np.float32), feature_scale=np.full(2, 2, np.float32),
                       recent_steps=2, trend_steps=3, stride=1, normalized=False, time_encoding="cyclic",
                       longitudes=np.tile([7, 7.5, 8], 3))
        plain = STGANFullGridDataset(data, times, grid, **options)
        cleaned = STGANFullGridDataset(data, times, grid, input_data=inputs, **options)
        target = int(np.nonzero(plain.targets == 8)[0][0])  # history 5..7 replaced, target 8 as it is
        first, second = plain.fetch_batch([target]), cleaned.fetch_batch([target])
        torch.testing.assert_close(first[4], second[4])  # the observed target
        self.assertTrue(torch.equal(second[1], torch.full_like(second[1], -1.0)))  # zero, normalised
        self.assertFalse(torch.equal(first[1], second[1]))
        copied = pickle.loads(pickle.dumps(cleaned))
        for one, other in zip(copied.fetch_batch([target]), second):
            self.assertTrue(torch.equal(one, other))
        with self.assertRaisesRegex(ValueError, "shape"):
            STGANFullGridDataset(data, times, grid, input_data=inputs[:5], **options)

    def test_flagged_points_are_left_out_and_counted(self):
        values = np.arange(12, dtype=np.float64).reshape(3, 4)
        valid = np.ones((3, 4), dtype=bool)
        result = reconstruction_statistics(values, valid & ~np.eye(3, 4, dtype=bool))
        self.assertEqual((result["valid_points"], result["excluded_points"]), (9, 3))


class CommandTests(unittest.TestCase):
    """cache, normal masks, PCA and MMD references, then a run: the order of the commands on the server."""

    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = prepare(self.root / "prepared")
        self.common = ["--prepared-dir", str(self.root / "prepared"), "--train-end-year", "2002"]

    def test_the_commands_in_order(self):
        quiet(run_era5_stgan.main, [
            "train", *self.common, "--output-dir", str(self.root / "cache_run"), "--validation-holdout",
            "--normalization", "seasonal", "--cnn-training-mode", "full_grid", "--reconstruction-weights",
            "daily_change", "--shared-cache-dir", str(self.root / "cache"), "--device", "cpu", "--cache-only"])
        built = json.loads((self.root / "cache_run" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(built["status"], "cache_only")
        cache = Path(built["backend"]["shared_cache"])
        self.assertEqual([path.name for path in (self.root / "cache").iterdir()], [cache.name])
        self.assertFalse((self.root / "cache_run" / "model_epoch_1.pt").exists())

        quiet(run_era5_stgan.main, ["normal-masks", *self.common, "--shared-cache-dir", str(cache),
                                    "--output-dir", str(self.root / "masks")])
        masks = load_normal_masks(self.root / "masks")
        validation_times = TIMES[(TIMES.year >= 2003) & (TIMES.year <= 2004)]
        for moment in UNUSUAL:
            self.assertTrue(masks.validation_mask[validation_times.get_loc(moment), UNUSUAL_CELL], moment)
        # About 5% + 5% of the training values, less their overlap. With one training year the seasonal normal
        # fits that year a little too well, so the validation flags more; with twenty years far less so.
        self.assertTrue(.05 < masks.metadata["train"]["flagged_share"] < .1)
        self.assertLess(masks.validation_mask.mean(), .3)
        self.assertEqual(masks.validation_mask.shape, (len(validation_times), 9))
        standardised = np.load(cache / "validation.npy")
        inputs = np.load(masks.inputs_path)
        context = len(standardised) - len(validation_times)
        np.testing.assert_array_equal(inputs[:context], standardised[:context])
        np.testing.assert_array_equal(inputs[context:][masks.validation_mask], 0)
        np.testing.assert_array_equal(inputs[context:][~masks.validation_mask], standardised[context:][~masks.validation_mask])
        self.assertTrue(np.isnan(masks.train_share[0]))  # no timestamp before the first one

        quiet(run_era5_stgan.main, ["pca-reference", *self.common, "--shared-cache-dir", str(cache),
                                    "--normal-masks-dir", str(self.root / "masks"), "--output-dir",
                                    str(self.root / "pca"), "--pca-samples", "48", "--pca-components", "6"])
        pca = json.loads((self.root / "pca" / "metadata.json").read_text(encoding="utf-8"))
        self.assertIn("seasonal", pca["normalization"]["kind"])
        self.assertEqual(pca["normalization"]["minimum"], json.loads((cache / "cache.json").read_text())["built"]["minimum"])
        fitted = pd.DatetimeIndex(np.load(self.root / "pca" / "pca_fit_timestamps.npy"))
        self.assertTrue((fitted.year == 2002).all())
        self.assertTrue((np.bincount(fitted.month, minlength=13)[1:] == 4).all())

        quiet(run_era5_stgan.main, ["mmd-reference", *self.common, "--shared-cache-dir", str(cache),
                                    "--normal-masks-dir", str(self.root / "masks"), "--pca-reference-dir",
                                    str(self.root / "pca"), "--output-dir", str(self.root / "mmd"),
                                    "--subsets", "3", "--subset-size", "8", "--pca-components", "4"])
        chosen = pd.DatetimeIndex(np.load(self.root / "mmd" / "subset_timestamps.npy").ravel())
        self.assertTrue(((chosen.year >= 2003) & (chosen.year <= 2004)).all())
        self.assertEqual(len(chosen.unique()), 24)
        share = masks.validation_share[validation_times.get_indexer(chosen)]
        self.assertLessEqual(share.max(), np.quantile(masks.validation_share, .9))

        config = replace(default_config("era5"), precision="fp32", epochs=2, num_workers=0, hidden_size=4,
            n_layers=1, cnn_channels=4, cnn_layers=1, trend_steps=56, train_samples_per_epoch=4, mc_samples=2,
            cnn_training_mode="full_grid", batch_size=4, score_batch_size=256, normalization="seasonal",
            reconstruction_weights="daily_change", train_end_year=2002, monitoring_timestamps=4,
            monitoring_feature_mmd_samples=16, mmd_objective_window=2, time_encoding="cyclic")
        runs = {}
        for name, anomalies in (("kept", "kept"), ("replaced", "excluded_and_replaced")):
            runs[name] = quiet(run_era5_stgan.run_training, prepared_dir=self.root / "prepared",
                output_dir=self.root / name, config=replace(config, validation_anomalies=anomalies), device="cpu",
                pca_reference_dir=self.root / "pca", mmd_reference_dir=self.root / "mmd",
                shared_cache_dir=self.root / "cache", skip_final_scoring=True,
                normal_masks_dir=None if anomalies == "kept" else self.root / "masks")
        kept, replaced = runs["kept"], runs["replaced"]
        self.assertEqual((replaced["effective_train_end_year"], replaced["validation_years"]), (2002, [2003, 2004]))
        self.assertEqual(replaced["backend"]["normal_masks"]["validation_anomalies"], "excluded_and_replaced")
        objective = replaced["backend"]["validation_objective"]
        flagged = int(masks.validation_mask.sum())
        self.assertEqual(objective["flagged_points_left_out"], flagged)
        self.assertEqual(objective["valid_points"], masks.validation_mask.size - flagged)
        self.assertEqual(kept["backend"]["validation_objective"]["valid_points"], masks.validation_mask.size)
        errors = np.load(self.root / "replaced" / "validation_generator_mean.npy")
        self.assertEqual(errors.shape, masks.validation_mask.shape)
        np.testing.assert_array_equal(np.load(self.root / "replaced" / "validation_timestamps.npy"), validation_times.asi8)
        history = pd.read_csv(self.root / "replaced" / "training_history.csv")
        self.assertTrue(history["validation_pca_mmd_rolling_mean"].notna().all())
        # Same weights after training: the two runs differ in their validation only.
        self.assertNotEqual(kept["backend"]["validation_objective"]["median"], objective["median"])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "validation_holdout"):
            replace(default_config("era5"), validation_holdout=False, train_end_year=2002)
        with self.assertRaisesRegex(ValueError, "validation_anomalies"):
            replace(default_config("era5"), validation_anomalies="dropped")
        config = replace(default_config("era5"), precision="fp32", num_workers=0, normalization="seasonal",
                         train_end_year=2002, validation_anomalies="excluded")
        with self.assertRaisesRegex(ValueError, "normal_masks is needed"):
            quiet(run_era5_stgan.run_training, prepared_dir=self.root / "prepared", output_dir=self.root / "x",
                  config=config, device="cpu", shared_cache_dir=self.root / "cache")


if __name__ == "__main__":
    unittest.main()
