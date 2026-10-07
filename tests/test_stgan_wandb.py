"""W&B sweep injection, failure lifecycle and offline STGAN integration."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
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

from physiq_pv.experiments.stgan_wandb import (
    WANDB_ENTITY, WANDB_PROJECT, default_config, resolve_config, run_tracked, validate_sweep)
from scripts.run_stgan_wandb import parse_args

HAS_GAT = "spatial_encoder" in asdict(default_config("pvgis"))


class FakeRun:
    id = "test-run"
    url = "https://wandb.ai/example/test-run"
    sweep_id = "test-sweep"
    entity = WANDB_ENTITY
    project = WANDB_PROJECT

    def __init__(self, config):
        self.config, self.summary, self.logs, self.definitions = config, {}, [], []
        self.exception_type = None

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.exception_type = kind

    def log(self, values):
        self.logs.append(dict(values))

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))


class WandbTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "manifest.csv"
        self.source.touch()
        self.args = parse_args(["--backend", "pvgis", "--manifest", str(self.source),
            "--output-root", str(self.root / "runs"), "--device", "cpu", "--wandb-mode", "offline"])

    def test_defaults_dry_run_and_reject_unknown_config(self):
        self.assertEqual((self.args.wandb_entity, self.args.wandb_project), (WANDB_ENTITY, WANDB_PROJECT))
        self.assertEqual(default_config("pvgis").precision, "bf16")
        self.args.dry_run = True
        with patch("wandb.init") as init, redirect_stdout(io.StringIO()) as stream:
            run_tracked(self.args)
        init.assert_not_called()
        self.assertEqual(json.loads(stream.getvalue())["config"]["precision"], "bf16")
        # score_mode is not a sweepable field: the ERA5 entrypoint always scores in paper mode.
        for bad in ({"learnig_rate": .1}, {"score_mode": "calibrated"}, {"seed": True}, {"seed": -1}):
            with self.assertRaises(ValueError):
                resolve_config(default_config("pvgis"), bad, 20)

    def test_agent_config_controls_training_and_logs_live_epochs(self):
        seen = {}
        def initialize(**kwargs):
            self.assertEqual((kwargs["entity"], kwargs["project"]), (WANDB_ENTITY, WANDB_PROJECT))
            seen["run"] = FakeRun({**kwargs["config"], "learning_rate": .004, "discriminator_lr_ratio": .5,
                                   "hidden_size": 8, "seed": 41, "precision": "fp32"})
            return seen["run"]
        def train(args, config, seed, output, on_epoch):
            self.assertEqual((config.learning_rate, config.hidden_size, seed), (.004, 8, 41))
            self.assertEqual(output.name, "results")
            on_epoch(dict(epoch=1, generator_loss=3., discriminator_loss=.6, seconds=2., samples=4,
                          discriminator_updates=4, generator_updates=4))
            self.assertEqual(seen["run"].logs[0]["train/generator_loss"], 3.)
            return {"precision": config.precision, "performance": {"training_seconds": 2.,
                    "peak_cuda_memory_bytes": None, "precision": "fp32"}, "parameter_counts": {"generator": 5}}
        with patch("wandb.init", side_effect=initialize), \
             patch("physiq_pv.experiments.stgan_wandb.execute_training", side_effect=train):
            output = run_tracked(self.args)
        saved = json.loads((output / "wandb_run.json").read_text())
        self.assertEqual((saved["sweep_id"], saved["status"], saved["seed"]), ("test-sweep", "complete", 41))
        self.assertEqual(saved["config"]["learning_rate"], .004)
        self.assertEqual(saved["config"]["discriminator_lr_ratio"], .5)
        self.assertEqual(seen["run"].summary["discriminator_learning_rate"], .002)
        self.assertEqual(seen["run"].summary["status"], "complete")
        self.assertEqual(seen["run"].logs[-1], {"performance/training_seconds": 2.})

    def test_failed_member_remains_failed_and_preserves_identity(self):
        run = FakeRun({**asdict(default_config("pvgis")), "seed": 20})
        with patch("wandb.init", return_value=run), \
             patch("physiq_pv.experiments.stgan_wandb.execute_training", side_effect=RuntimeError("training failed")):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                run_tracked(self.args)
        self.assertIs(run.exception_type, RuntimeError)
        saved = json.loads((self.root / "runs/pvgis/test-run/wandb_run.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(run.summary["status"], "failed")

    def test_draft_cannot_create_a_sweep_without_research_choices(self):
        import yaml
        from scripts.create_stgan_sweep import main
        path = Path(__file__).resolve().parents[1] / "sweeps/stgan_bayes.draft.yaml"
        draft = yaml.safe_load(path.read_text())
        self.assertEqual((draft["method"], draft["entity"], draft["project"]), ("bayes", WANDB_ENTITY, WANDB_PROJECT))
        self.assertEqual(draft["metric"], {"name": "validation/pca_mmd_rolling_mean", "goal": "minimize"})
        self.assertEqual(validate_sweep(draft)["method"], "bayes")
        # Without a complete metric nothing can be registered.
        for metric in ({"name": None, "goal": None}, {"name": "validation/pca_mmd_rolling_mean", "goal": None}):
            with self.assertRaisesRegex(ValueError, "metric"):
                validate_sweep({**draft, "metric": metric})
        incomplete = self.root / "incomplete.yaml"
        incomplete.write_text(yaml.safe_dump({**draft, "metric": {"name": None, "goal": None}}), encoding="utf-8")
        with patch("wandb.sweep") as create, patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                main([str(incomplete), "--create"])
        create.assert_not_called()
        parameters = draft["parameters"]
        self.assertEqual(parameters["discriminator_lr_ratio"],
                         {"distribution": "log_uniform_values", "min": .25, "max": 4.})
        self.assertNotIn("discriminator_learning_rate", parameters)  # One way to set each rate.
        sampled = {key: spec.get("value", spec.get("min", spec.get("values", [None])[0]))
                   for key, spec in parameters.items()}
        config, _ = resolve_config(default_config("era5"), sampled, 20)
        self.assertEqual((config.recent_steps, config.precision), (1, "bf16"))
        draft["parameters"] = {}
        with self.assertRaisesRegex(ValueError, "hyperparameters"):
            validate_sweep(draft)
        draft["parameters"] = {"learning_rate": {"values": [.001, .002]}}
        self.assertEqual(validate_sweep(draft)["method"], "bayes")

    def write_tiny_manifest(self):
        from pyproj import Transformer
        x, y = np.meshgrid(400000. + np.arange(2)*5000, 5000000. - np.arange(2)*5000)
        lon, lat = Transformer.from_crs(32632, 4326, always_xy=True).transform(x.ravel(), y.ravel())
        times = pd.date_range("2018-12-31 12:00", periods=16, freq="h")
        values = np.random.default_rng(7).normal(size=(16, 4, 3))
        rows = []
        for site in range(4):
            row = dict(location=str(site), site_key=f"site_{site}", latitude=lat[site], longitude=lon[site])
            splits = {"train": (0, 8 if HAS_GAT else 12), "test": (12, 16)}
            if HAS_GAT:
                splits["validation"] = (8, 12)
            for split, (start, end) in splits.items():
                path = self.root / f"site_{site}_{split}.csv"
                pd.DataFrame({"timestamp": times[start:end], "solar_irradiance_poa": 300+values[start:end, site, 0],
                    "temperature_2m": 10+values[start:end, site, 1], "wind_speed_10m": 3+values[start:end, site, 2],
                    "is_daytime": True}).to_csv(path, index=False)
                row[f"{split}_csv"] = str(path)
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.source, index=False)

    def test_real_sdk_offline_training_and_checkpoint(self):
        import wandb
        self.addCleanup(wandb.teardown)  # Release the SDK service log before Windows temp cleanup.
        self.write_tiny_manifest()
        config = dict(precision="fp32", epochs=1, batch_size=4, hidden_size=4,
            learning_rate=.0002, discriminator_lr_ratio=2.,
            n_layers=1, cnn_channels=4, cnn_layers=1, trend_steps=2, train_samples_per_epoch=4,
            cache_normalized=False, score_storage="memory")
        if HAS_GAT:
            config["mc_samples"] = 2
        self.args.model_config = self.root / "model.json"
        self.args.model_config.write_text(json.dumps(config))
        environment = {f"WANDB_{name}_DIR": str(self.root / name.lower()) for name in ("CONFIG", "CACHE", "DATA")}
        with patch.dict("os.environ", environment), redirect_stdout(io.StringIO()):
            output = run_tracked(self.args)
        saved = json.loads((output / "wandb_run.json").read_text())
        self.assertEqual((saved["status"], saved["config"]["precision"]), ("complete", "fp32"))
        self.assertIsNone(saved["sweep_id"])
        self.assertTrue((output / "results/seed_20/checkpoint.pt").is_file())
        payload = torch.load(output / "results/seed_20/checkpoint.pt", weights_only=False)
        self.assertEqual(payload["training"]["discriminator_learning_rate"], .0004)
        self.assertTrue(list((self.root / "runs/wandb").glob("offline-run-*/*.wandb")))

    @unittest.skipUnless(HAS_GAT, "ERA5 exists in the GAT branch")
    def test_era5_shared_entrypoint_trains_and_calls_epoch_callback(self):
        from scripts.run_era5_stgan import run_training
        from physiq_pv.era5.data import ERA5Cubes
        from physiq_pv.era5.cube import CubeGrid
        values = np.random.default_rng(9).normal(size=(16, 4, 2)).astype(np.float32)
        times = pd.date_range("2004-12-30 12:00", periods=16, freq="3h")
        lat, lon = np.array([45, 45, 44.5, 44.5]), np.array([7, 7.5, 7, 7.5])
        cubes = ERA5Cubes(train=values[:8], validation=values[8:12], test=values[12:],
            train_timestamps=times[:8], validation_timestamps=times[8:12], test_timestamps=times[12:],
            location_names=("0", "1", "2", "3"), feature_names=("a", "b"), latitudes=lat, longitudes=lon)
        config = replace(default_config("era5"), precision="fp32", epochs=1, num_workers=0,
            learning_rate=.0002, discriminator_lr_ratio=.5,
            hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1, trend_steps=2,
            train_samples_per_epoch=1, mc_samples=2, cache_normalized=False)
        records = []
        def callback(record):
            records.append(dict(record))
            record["epoch"] = -1  # Callback receives a copy, not the training history.
        with patch("scripts.run_era5_stgan.load_prepared", return_value=(cubes,
                CubeGrid.from_locations(lat, lon), {"train_end_year": 2003, "validation_end_year": 2004, "test_start_year": 2005})), \
                redirect_stdout(io.StringIO()):
            metadata = run_training(prepared_dir=self.root, output_dir=self.root / "era5",
                config=config, device="cpu", on_epoch=callback)
        self.assertEqual([r["epoch"] for r in records], [1])
        history = pd.read_csv(self.root / "era5/training_history.csv")
        self.assertEqual(history["epoch"].tolist(), [1])
        self.assertEqual(metadata["backend"]["precision"], "fp32")
        self.assertEqual((metadata["backend"]["score_mode"], metadata["backend"]["paper_alignment"]["score_equation"]),
                         ("paper", True))
        payload = torch.load(self.root / "era5/model_epoch_1.pt", weights_only=False)
        self.assertEqual(payload["generator_optimizer_state_dict"]["param_groups"][0]["lr"], .0002)
        self.assertEqual(payload["discriminator_optimizer_state_dict"]["param_groups"][0]["lr"], .0001)


if __name__ == "__main__":
    unittest.main()
