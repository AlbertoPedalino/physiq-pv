"""The training command must carry every seasonal option to the run, as its arguments give them."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.era5.cube import CubeGrid
from physiq_pv.era5.data import ERA5Cubes
from physiq_pv.experiments.stgan_wandb import default_config
from scripts import run_era5_stgan as cli

NAMES = ("warming", "rain", "steady")
PERIODS = {"train": pd.date_range("2002-01-01", "2003-12-31 21:00", freq="3h"),
           "validation": pd.date_range("2004-01-01", "2004-12-31 21:00", freq="3h"),
           "test": pd.date_range("2005-01-01", "2005-01-10 21:00", freq="3h")}
LAT, LON = np.array([45, 45, 44.5, 44.5]), np.array([7, 7.5, 7, 7.5])


class SeasonalRunOptionsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(2)
        self.arrays = {}
        for name, times in PERIODS.items():
            values = rng.normal(size=(len(times), 4, len(NAMES))).astype(np.float32)
            values[..., 0] += (0.5 * (times.year + (times.dayofyear - 1) / 365.25 - 2003.5)).to_numpy()[:, None]
            values[..., 1] = rng.exponential(size=(len(times), 4)) * (rng.random((len(times), 4)) < 0.3)
            np.save(self.root / f"{name}.npy", values)
            self.arrays[name] = np.load(self.root / f"{name}.npy", mmap_mode="r")
        self.addCleanup(lambda: [array._mmap.close() for array in self.arrays.values()])

    def run_command(self, name, *arguments, holdout=False):
        """The run of the training command with these arguments, on a small model: the seasonal options go
        from the arguments to the configuration as `main` passes them."""
        args = cli.parser().parse_args(["train", "--prepared-dir", "p", "--output-dir", "o", "--normalization", "seasonal",
                                        *arguments])
        config = replace(
            default_config("era5"), precision="fp32", epochs=1, num_workers=0, hidden_size=4, n_layers=1, cnn_channels=4,
            cnn_layers=1, trend_steps=2, train_samples_per_epoch=2, mc_samples=2, validation_holdout=holdout,
            normalization=args.normalization, seasonal_window_days=args.seasonal_window_days,
            seasonal_trend=args.seasonal_trend, seasonal_std_floor=args.seasonal_std_floor,
            seasonal_std_floors=cli.floors_by_variable(args.seasonal_std_floors),
            seasonal_fourth_root_features=(None if args.seasonal_fourth_root_features is None
                                           else tuple(args.seasonal_fourth_root_features)),
            seasonal_trend_features=(None if args.seasonal_trend_features is None
                                     else tuple(args.seasonal_trend_features)),
            reconstruction_weights="daily_change")
        cubes = ERA5Cubes(
            train=self.arrays["train"], validation=self.arrays["validation"], test=self.arrays["test"],
            train_timestamps=PERIODS["train"], validation_timestamps=PERIODS["validation"],
            test_timestamps=PERIODS["test"], location_names=("0", "1", "2", "3"), feature_names=NAMES,
            latitudes=LAT, longitudes=LON)
        preparation = {"train_end_year": 2003, "validation_end_year": 2004, "test_start_year": 2005}
        output = io.StringIO()
        with patch("scripts.run_era5_stgan.load_prepared", return_value=(
                cubes, CubeGrid.from_locations(LAT, LON), preparation)), patch.object(ERA5Cubes, "close", lambda self: None), \
                redirect_stdout(output):
            metadata = cli.run_training(prepared_dir=self.root, output_dir=self.root / name, config=config, device="cpu",
                                        shared_cache_dir=self.root / "shared")
        return metadata, output.getvalue()

    def test_every_seasonal_option_reaches_the_run(self):
        options = ("--seasonal-std-floors", "rain=0.3", "--seasonal-fourth-root-features", "rain",
                   "--seasonal-trend", "linear", "--seasonal-trend-features", "warming")
        for name, holdout in (("through_2004", False), ("through_2003", True)):
            with self.subTest(name):
                metadata, text = self.run_command(name, *options, holdout=holdout)
                self.assertEqual(metadata["status"], "complete")
                seasonal = metadata["backend"]["seasonal_normalization"]
                self.assertEqual(seasonal["std_floor_fraction_of_feature_std"], [0.01, 0.3, 0.01])
                self.assertEqual(seasonal["fourth_root_features"], ["rain"])
                self.assertEqual(seasonal["trend"]["features"], ["warming"])
                self.assertEqual(seasonal["trend"]["fit_years"], [2002, 2003 if holdout else 2004])
                self.assertGreater(seasonal["trend"]["slope_per_year"][0], 0.2)
                self.assertEqual(seasonal["trend"]["slope_per_year"][1:], [0.0, 0.0])
                self.assertEqual(seasonal["fit_period"]["end"], str(PERIODS["train" if holdout else "validation"][-1]))
                self.assertIn("line in time removed", text)
                self.assertIn("shared cache built in", text)
                self.assertEqual(metadata["effective_train_end_year"], 2003 if holdout else 2004)
                saved = json.loads((self.root / name / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["config"]["seasonal_std_floors"], [["rain", 0.3]])
                self.assertTrue((self.root / name / "scores" / "anomaly_mean.npy").is_file())
        # A second run with the same options reads the cache; other options are another cache.
        self.assertIn("shared cache read from", self.run_command("again", *options)[1])
        self.assertIn("shared cache built in", self.run_command("plain")[1])
        with self.assertRaisesRegex(ValueError, "not among the features"):
            self.run_command("wrong", "--seasonal-std-floors", "snow=0.3")


if __name__ == "__main__":
    unittest.main()
