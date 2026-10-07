"""The sweep samples the generator rate and the D/G rate ratio; D's rate is their product.

The same file serves the ConvGRU and the GAT branch.
"""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan import STGANCNNConfig
from physiq_pv.experiments import stgan_wandb
from physiq_pv.experiments.stgan_wandb import default_config, resolve_config, run_tracked, validate_sweep
from scripts.run_stgan_wandb import parse_args

DRAFT = Path(__file__).resolve().parents[1] / "sweeps/stgan_bayes.draft.yaml"
RATE = {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3}
RATIO = {"distribution": "log_uniform_values", "min": .25, "max": 4.}


def draft():
    import yaml
    return yaml.safe_load(DRAFT.read_text())


class FakeRun:
    id, url, entity, project = "run", "url", "entity", "project"

    def __init__(self, config, sweep_id):
        self.config, self.sweep_id, self.summary, self.logs, self.finished = config, sweep_id, {}, [], []

    def __enter__(self):
        return self

    def __exit__(self, kind, *args):
        self.finished.append(1 if kind else 0)
        return False

    def finish(self, exit_code=None):
        raise AssertionError("a run was ended before its training")

    def define_metric(self, name, **kwargs):
        pass

    def log(self, values):
        self.logs.append(dict(values))


class SweepSpaceTests(unittest.TestCase):
    def test_sweep_varies_the_generator_rate_the_ratio_the_weight_and_the_updates_only(self):
        parameters = validate_sweep(draft())["parameters"]
        self.assertEqual({key: spec for key, spec in parameters.items() if "value" not in spec}, {
            "generator_learning_rate": RATE, "discriminator_lr_ratio": RATIO,
            "generator_reconstruction_weight": {"distribution": "log_uniform_values", "min": 50., "max": 2000.},
            "discriminator_generator_update_ratio": {"values": ["1:1", "2:1", "1:2"]}})
        # The rate of D is derived, never sampled, and the legacy generator name is not there either.
        self.assertFalse({"discriminator_learning_rate", "learning_rate"} & set(parameters))
        # Log-uniform around 1: a ratio and its inverse are equally likely.
        self.assertAlmostEqual(math.log(RATIO["min"]), -math.log(RATIO["max"]))
        self.assertAlmostEqual(math.log(.5) - math.log(RATIO["min"]), math.log(RATIO["max"]) - math.log(2.))
        # What stays fixed.
        fixed = {key: spec["value"] for key, spec in parameters.items() if "value" in spec}
        self.assertLessEqual({"precision": "bf16", "recent_steps": 1, "validation_holdout": True,
                              "monitoring_timestamps": 32, "monitoring_feature_mmd_every_n_epochs": 1,
                              "monitoring_feature_mmd_samples": 1024, "mmd_objective_window": 5,
                              "epochs": 6, "seed": 20}.items(), fixed.items())
        self.assertEqual(draft()["metric"], {"name": "validation/pca_mmd_rolling_mean", "goal": "minimize"})

    def test_discriminator_rate_is_the_generator_rate_times_the_ratio(self):
        for ratio, expected in ((.5, 5e-5), (1., 1e-4), (2., 2e-4), (4., 4e-4), (.25, 2.5e-5)):
            config, _ = resolve_config(default_config("era5"), {"generator_learning_rate": 1e-4,
                                                               "discriminator_lr_ratio": ratio}, 20)
            self.assertEqual(config.effective_generator_learning_rate, 1e-4)
            self.assertAlmostEqual(config.effective_discriminator_learning_rate, expected, places=15)
            self.assertEqual(config.effective_discriminator_learning_rate, 1e-4 * ratio)
            self.assertIsNone(config.discriminator_learning_rate)  # Derived, not a second way to give it.
        # Every corner of the space: the ratio of the effective rates is the sampled one, within [0.25, 4].
        for generator in (RATE["min"], 1e-4, RATE["max"]):
            for ratio in (RATIO["min"], 1., RATIO["max"]):
                config, _ = resolve_config(default_config("era5"), {"generator_learning_rate": generator,
                                                                   "discriminator_lr_ratio": ratio}, 20)
                self.assertAlmostEqual(config.effective_discriminator_learning_rate
                                       / config.effective_generator_learning_rate, ratio, places=12)

    def test_each_rate_is_given_in_one_way(self):
        both = {"generator_learning_rate": RATE, "discriminator_lr_ratio": RATIO, "discriminator_learning_rate": RATE}
        with self.assertRaisesRegex(ValueError, "discriminator_learning_rate or discriminator_lr_ratio"):
            validate_sweep({**draft(), "parameters": both})
        with self.assertRaisesRegex(ValueError, "discriminator rate once"):
            resolve_config(default_config("era5"), {"generator_learning_rate": 1e-4, "discriminator_lr_ratio": 2.,
                                                    "discriminator_learning_rate": 2e-4}, 20)
        # Outside the sweep the rate of D can still be given directly, instead of the ratio.
        config, _ = resolve_config(default_config("era5"), {"generator_learning_rate": 1e-4,
                                                           "discriminator_learning_rate": 3e-3}, 20)
        self.assertEqual((config.effective_discriminator_learning_rate, config.discriminator_lr_ratio), (3e-3, 1.))


class RunTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "prepared").mkdir()

    def member(self, name, sampled, *, sweep_id="sweep"):
        """One run with the values an agent would put in wandb.config; training itself is replaced."""
        args = parse_args(["--backend", "era5", "--prepared-dir", str(self.root / "prepared"), "--output-root",
                           str(self.root / name), "--device", "cpu", "--wandb-mode", "offline",
                           "--pca-reference-dir", str(self.root / "pca"), "--mmd-reference-dir", str(self.root / "mmd")])
        runs = []

        def initialize(**kwargs):
            runs.append(FakeRun({**kwargs["config"], **sampled}, sweep_id))
            return runs[-1]

        def train(args, config, seed, output, on_epoch):
            on_epoch({"epoch": 1, "seconds": 1., "generator_loss": 2., "validation_pca_mmd_rolling_mean": .3})
            return {"precision": config.precision, "performance": {}}
        with patch("wandb.init", side_effect=initialize), redirect_stdout(io.StringIO()), \
                patch.object(stgan_wandb, "execute_training", side_effect=train) as training:
            directory = run_tracked(args)
        return runs[0], training, json.loads((directory / "wandb_run.json").read_text())

    def test_every_sweep_member_trains_and_logs_its_effective_rates(self):
        cases = [(1e-4, .5), (1e-4, 1.), (1e-4, 2.), (1e-4, 4.)]
        cases += [(generator, ratio) for generator in (RATE["min"], RATE["max"]) for ratio in (RATIO["min"], RATIO["max"])]
        for index, (generator, ratio) in enumerate(cases):
            with self.subTest(generator=generator, ratio=ratio):
                run, training, saved = self.member(f"member_{index}", {
                    "generator_learning_rate": generator, "discriminator_lr_ratio": ratio,
                    "discriminator_generator_update_ratio": "2:1"})
                training.assert_called_once()
                config = training.call_args.args[1]
                self.assertEqual((config.effective_generator_learning_rate, config.effective_discriminator_learning_rate),
                                 (generator, generator * ratio))
                # The summary names the three quantities, although D's rate is derived.
                self.assertEqual((run.summary["generator_learning_rate"], run.summary["discriminator_learning_rate"]),
                                 (generator, generator * ratio))
                self.assertAlmostEqual(run.summary["discriminator_to_generator_lr_ratio"], ratio, places=12)
                self.assertEqual((run.summary["status"], saved["status"], run.finished), ("complete", "complete", [0]))
                self.assertEqual(run.logs[0]["validation/pca_mmd_rolling_mean"], .3)
                self.assertEqual((saved["config"]["discriminator_lr_ratio"], saved["config"]["discriminator_learning_rate"]),
                                 (ratio, None))
        # No run is refused because of its rates any more: the former check is gone.
        for name in ("sweep_lr_ratio_allowed", "SWEEP_LR_RATIO_BOUNDS", "INVALID_CONFIGURATION"):
            self.assertFalse(hasattr(stgan_wandb, name), name)
        self.assertNotIn("invalid_configuration", Path(stgan_wandb.__file__).read_text(encoding="utf-8"))

    def test_runs_outside_a_sweep_keep_both_forms(self):
        # The rate of D given directly, with any ratio to G's: trained, in a sweep or not.
        for sweep_id in (None, "sweep"):
            run, training, saved = self.member(f"direct_{sweep_id}", {
                "generator_learning_rate": 2e-5, "discriminator_learning_rate": 2e-5 * 33.8334}, sweep_id=sweep_id)
            training.assert_called_once()
            self.assertAlmostEqual(run.summary["discriminator_to_generator_lr_ratio"], 33.8334)
            self.assertEqual((saved["status"], run.finished), ("complete", [0]))
        run, training, _ = self.member("legacy", {"learning_rate": 2e-4, "discriminator_lr_ratio": 2.}, sweep_id=None)
        self.assertEqual((run.summary["generator_learning_rate"], run.summary["discriminator_learning_rate"]),
                         (2e-4, 4e-4))

    def test_training_uses_the_derived_rate(self):
        from physiq_pv.era5.cube import CubeGrid
        from physiq_pv.era5.data import ERA5Cubes
        from scripts.run_era5_stgan import run_training
        values = np.random.default_rng(9).normal(size=(16, 4, 2)).astype(np.float32)
        times = pd.date_range("2004-12-30 12:00", periods=16, freq="3h")
        lat, lon = np.array([45, 45, 44.5, 44.5]), np.array([7, 7.5, 7, 7.5])
        cubes = ERA5Cubes(train=values[:8], validation=values[8:12], test=values[12:],
            train_timestamps=times[:8], validation_timestamps=times[8:12], test_timestamps=times[12:],
            location_names=("0", "1", "2", "3"), feature_names=("a", "b"), latitudes=lat, longitudes=lon)
        base = replace(default_config("era5"), precision="fp32", epochs=1, num_workers=0,
            hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1, trend_steps=2,
            train_samples_per_epoch=1, mc_samples=2, cache_normalized=False)
        config, _ = resolve_config(base, {"generator_learning_rate": 1e-4, "discriminator_lr_ratio": 2.}, 20)
        self.assertIsInstance(config, STGANCNNConfig)
        preparation = {"train_end_year": 2003, "validation_end_year": 2004, "test_start_year": 2005}
        with patch("scripts.run_era5_stgan.load_prepared", return_value=(
                cubes, CubeGrid.from_locations(lat, lon), preparation)), redirect_stdout(io.StringIO()):
            metadata = run_training(prepared_dir=self.root, output_dir=self.root / "era5", config=config, device="cpu")
        self.assertEqual((metadata["backend"]["generator_learning_rate"], metadata["backend"]["discriminator_learning_rate"]),
                         (1e-4, 2e-4))
        payload = torch.load(self.root / "era5/model_epoch_1.pt", weights_only=False)
        self.assertEqual((payload["generator_optimizer_state_dict"]["param_groups"][0]["lr"],
                          payload["discriminator_optimizer_state_dict"]["param_groups"][0]["lr"]), (1e-4, 2e-4))


if __name__ == "__main__":
    unittest.main()
