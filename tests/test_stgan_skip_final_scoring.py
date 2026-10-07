"""--skip-final-scoring: training and validation monitoring without scoring the test.

The same file serves the ConvGRU and the GAT branch; graph cases run where the GAT exists.
"""
from contextlib import redirect_stdout
from dataclasses import replace
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan import pipeline as pipeline_module
from physiq_pv.era5.cube import CubeGrid
from physiq_pv.era5.data import ERA5Cubes
from physiq_pv.experiments.stgan_wandb import RunLogger, default_config, execute_training
from scripts import run_era5_stgan
from scripts.run_stgan_wandb import parse_args
from test_stgan_mmd import HAS_GRAPH, METRICS, NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

SCORED = AssertionError("the test was scored")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields()
        build_pca(self.root / "pca", self.values)
        build(self.root / "reference", self.values, self.root / "pca")

    def encoders(self):
        yield "convgru", {}
        if HAS_GRAPH:
            yield "gat", dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4)

    def fit(self, name, **overrides):
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=self.values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=2, batch_size=4, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=6, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, mc_samples=2, monitoring_timestamps=4, monitoring_feature_mmd_samples=16,
            checkpoint_path=self.root / name / "model.pt", score_dir=self.root / name / "scores",
            pca_reference=self.root / "pca", mmd_reference=self.root / "reference")
        options.update(overrides)
        stream = io.StringIO()
        with redirect_stdout(stream), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options) as result:
            return SimpleNamespace(result=result, metadata=result.metadata, printed=stream.getvalue(),
                scores=None if result.test_scores is None else np.array(result.test_scores),
                files=sorted(path.relative_to(self.root / name).as_posix() for path in (self.root / name).rglob("*")),
                history=pd.read_csv(self.root / name / "training_history.csv"),
                epoch=lambda n: torch.load(self.root / name / f"model_epoch_{n}.pt", weights_only=False))

    def test_skipping_ends_after_training_and_validation_without_scoring_the_test(self):
        self.assertIs(inspect.signature(fit_and_score_stgan).parameters["skip_final_scoring"].default, False)
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                normal = self.fit(f"{encoder}_normal", **extra)
                with patch.object(pipeline_module, "score_components", side_effect=SCORED), \
                        patch.object(pipeline_module, "normalize_paper_mc_scores", side_effect=SCORED):
                    skipped = self.fit(f"{encoder}_skipped", skip_final_scoring=True, **extra)
                # No score of the test, and nothing written for it.
                result = skipped.result
                self.assertEqual((result.test_scores, result.anomaly_std, result.test_feature_scores,
                                  result.test_generator_scores, result.test_discriminator_scores), (None,) * 5)
                self.assertEqual(skipped.files, ["model_epoch_1.pt", "model_epoch_2.pt", "training_history.csv"])
                self.assertIn("model.pt", normal.files)
                self.assertIn("scores", normal.files)  # The score directory of a complete run.
                self.assertIn("final scoring skipped", skipped.printed)
                self.assertEqual((skipped.metadata["final_scoring"], skipped.metadata["score_normalization"],
                                  skipped.metadata["score_statistics"]), ("skipped", None, None))
                self.assertNotIn("scoring_seconds", skipped.metadata["performance"])
                # The default is the run as it was: scored, with nothing about a skipped scoring.
                self.assertNotIn("final_scoring", normal.metadata)
                self.assertIsNotNone(normal.scores)
                self.assertIn("scoring_seconds", normal.metadata["performance"])
                # Training and every validation metric are those of the complete run.
                for epoch in (1, 2):
                    for key, value in normal.epoch(epoch)["model_state_dict"].items():
                        self.assertTrue(torch.equal(value, skipped.epoch(epoch)["model_state_dict"][key]), key)
                    self.assertEqual(skipped.epoch(epoch)["mmd_reference"], normal.epoch(epoch)["mmd_reference"])
                self.assertEqual(list(skipped.history.columns), list(normal.history.columns))
                for column in normal.history.columns:
                    if not column.endswith("seconds"):
                        np.testing.assert_array_equal(skipped.history[column], normal.history[column], err_msg=column)
                for column in METRICS + ("validation_pca_mmd_seconds", "validation_discriminator_feature_mmd"):
                    self.assertTrue(skipped.history[column].notna().all(), column)
                for key in ("pca_mmd", "pca_reference", "monitoring", "splits", "parameter_counts"):
                    self.assertEqual(skipped.metadata[key], normal.metadata[key], key)
                expected, found = ({name: value for name, value in run.metadata["validation_objective"].items()
                                    if name != "seconds"} for run in (normal, skipped))
                self.assertEqual(expected, found)
                self.assertGreater(skipped.metadata["performance"]["training_seconds"], 0)
        with self.assertRaisesRegex(ValueError, "skip_final_scoring"):
            self.fit("bad", skip_final_scoring=1)


class CommandTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_era5(self, name, **options):
        values = np.random.default_rng(9).normal(size=(16, 4, 2)).astype(np.float32)
        times = pd.date_range("2004-12-30 12:00", periods=16, freq="3h")
        lat, lon = np.array([45, 45, 44.5, 44.5]), np.array([7, 7.5, 7, 7.5])
        cubes = ERA5Cubes(train=values[:8], validation=values[8:12], test=values[12:],
            train_timestamps=times[:8], validation_timestamps=times[8:12], test_timestamps=times[12:],
            location_names=("0", "1", "2", "3"), feature_names=("a", "b"), latitudes=lat, longitudes=lon)
        config = replace(default_config("era5"), precision="fp32", epochs=1, num_workers=0,
            hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1, trend_steps=2,
            train_samples_per_epoch=1, mc_samples=2, cache_normalized=False)
        preparation = {"train_end_year": 2003, "validation_end_year": 2004, "test_start_year": 2005}
        with patch("scripts.run_era5_stgan.load_prepared", return_value=(
                cubes, CubeGrid.from_locations(lat, lon), preparation)), redirect_stdout(io.StringIO()):
            metadata = run_era5_stgan.run_training(prepared_dir=self.root, output_dir=self.root / name,
                                                   config=config, device="cpu", **options)
        return metadata, sorted(path.relative_to(self.root / name).as_posix() for path in (self.root / name).rglob("*"))

    def test_training_command_writes_a_training_only_run(self):
        normal, normal_files = self.run_era5("normal")
        with patch.object(pipeline_module, "score_components", side_effect=SCORED):
            skipped, files = self.run_era5("skipped", skip_final_scoring=True)
        self.assertEqual(files, ["metadata.json", "model_epoch_1.pt", "training_history.csv"])
        self.assertEqual((skipped["status"], skipped["final_scoring"], skipped["backend"]["final_scoring"]),
                         ("training_only", "skipped", "skipped"))
        self.assertFalse([key for key in skipped if key.endswith("_file")])
        self.assertEqual(json.loads((self.root / "skipped" / "metadata.json").read_text()), skipped)
        self.assertEqual((skipped["config"], skipped["validation_years"]), (normal["config"], [2004]))
        self.assertIsNotNone(skipped["backend"]["validation_objective"])
        # The default run is unchanged: complete, with its score maps.
        self.assertEqual((normal["status"], normal["scores_file"]), ("complete", "scores/anomaly_mean.npy"))
        self.assertNotIn("final_scoring", normal)
        self.assertLessEqual({"model.pt", "test_timestamps.npy", "scores/anomaly_mean.npy"}, set(normal_files))

    def test_flag_is_off_by_default_and_reaches_the_training(self):
        era5 = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        with patch.object(run_era5_stgan, "run_training", return_value={}) as train, redirect_stdout(io.StringIO()):
            run_era5_stgan.main(era5)
            self.assertIs(train.call_args.kwargs["skip_final_scoring"], False)
            run_era5_stgan.main(era5 + ["--skip-final-scoring"])
        self.assertIs(train.call_args.kwargs["skip_final_scoring"], True)
        self.assertIs(inspect.signature(run_era5_stgan.run_training).parameters["skip_final_scoring"].default, False)
        # The W&B runner: the same flag, for ERA5 only; a training-only result is logged as it is.
        base = ["--backend", "era5", "--prepared-dir", str(self.root), "--output-root", str(self.root / "runs"),
                "--device", "cpu", "--wandb-mode", "offline"]
        self.assertEqual((parse_args(base).skip_final_scoring, parse_args(base + ["--skip-final-scoring"]).skip_final_scoring),
                         (False, True))
        config = default_config("era5")
        for arguments, expected in ((base, False), (base + ["--skip-final-scoring"], True)):
            with patch("scripts.run_era5_stgan.run_training", return_value={"backend": {"precision": "fp32"}}) as train:
                execute_training(parse_args(arguments), config, 20, self.root / "out", None)
            self.assertIs(train.call_args.kwargs["skip_final_scoring"], expected)
        pvgis = parse_args(["--backend", "pvgis", "--manifest", str(self.root / "m.csv"), "--skip-final-scoring"])
        with patch("scripts.run_pvgis_stgan.run_stgan") as run_stgan, self.assertRaisesRegex(ValueError, "ERA5 backend only"):
            execute_training(pvgis, default_config("pvgis"), 20, self.root / "out", None)
        run_stgan.assert_not_called()

        class Run:
            def __init__(self):
                self.logs, self.summary = [], {}

            def define_metric(self, name, **kwargs):
                pass

            def log(self, values):
                self.logs.append(dict(values))

        run = Run()
        RunLogger(run).result({"final_scoring": "skipped", "precision": "bf16", "score_normalization": None,
                               "score_statistics": None, "validation_objective": None,
                               "pca_mmd": {"fingerprint": "abc"}, "performance": {"training_seconds": 2.}})
        self.assertEqual(run.logs, [{"performance/training_seconds": 2.}])
        self.assertEqual((run.summary["mmd_reference"], run.summary["precision"]), ({"fingerprint": "abc"}, "bf16"))


if __name__ == "__main__":
    unittest.main()
