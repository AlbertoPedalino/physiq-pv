"""adam_beta1: first-moment decay of both Adam optimizers; 0.9 keeps every earlier run unchanged.

The same file serves the ConvGRU and the GAT branch.
"""
from contextlib import redirect_stdout
import inspect
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan
from scripts import run_era5_stgan
from test_stgan_mmd import NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields


class ConfigTests(unittest.TestCase):
    def test_default_keeps_the_pytorch_value(self):
        self.assertEqual(STGANCNNConfig().adam_beta1, 0.9)
        self.assertEqual(inspect.signature(fit_and_score_stgan).parameters["adam_beta1"].default, 0.9)
        self.assertEqual(torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]).defaults["betas"], (0.9, 0.999))

    def test_values_outside_the_unit_interval_are_rejected(self):
        self.assertEqual(STGANCNNConfig(adam_beta1=.5).adam_beta1, .5)
        self.assertEqual(STGANCNNConfig(adam_beta1=0).adam_beta1, 0)
        for value in (1., -.1, float("nan"), float("inf"), "0.5", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                STGANCNNConfig(adam_beta1=value)

    def test_command_line_option(self):
        parser = run_era5_stgan.parser()
        base = ["train", "--prepared-dir", "p", "--output-dir", "o"]
        self.assertEqual(parser.parse_args(base).adam_beta1, 0.9)
        self.assertEqual(parser.parse_args(base + ["--adam-beta1", "0.5"]).adam_beta1, 0.5)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields()
        build_pca(self.root / "pca", self.values)
        build(self.root / "reference", self.values, self.root / "pca")

    def fit(self, name, **overrides):
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=self.values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=1, batch_size=4, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=6, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, mc_samples=2, monitoring_timestamps=4, monitoring_feature_mmd_samples=16,
            skip_final_scoring=True, spatial_encoder="convgru",
            checkpoint_path=self.root / name / "model.pt", score_dir=self.root / name / "scores",
            pca_reference=self.root / "pca", mmd_reference=self.root / "reference")
        if "spatial_encoder" not in inspect.signature(fit_and_score_stgan).parameters:
            options.pop("spatial_encoder")  # The ConvGRU branch has no other encoder.
        options.update(overrides)
        stream = io.StringIO()
        with redirect_stdout(stream), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options) as result:
            return SimpleNamespace(metadata=result.metadata, printed=stream.getvalue(),
                epoch=lambda n: torch.load(self.root / name / f"model_epoch_{n}.pt", weights_only=False))

    def betas(self, payload):
        return {name: {tuple(group["betas"]) for group in payload[f"{name}_optimizer_state_dict"]["param_groups"]}
                for name in ("generator", "discriminator")}

    def test_both_optimizers_use_the_value_and_the_run_records_it(self):
        for name, value in (("default", None), ("dcgan", .5)):
            with self.subTest(run=name):
                run = self.fit(name, **({} if value is None else {"adam_beta1": value}))
                expected = .9 if value is None else value
                payload = run.epoch(1)
                self.assertEqual(self.betas(payload), {"generator": {(expected, .999)},
                                                       "discriminator": {(expected, .999)}})
                self.assertEqual(payload["learning_rates"]["adam_beta1"], expected)
                self.assertIn(f"beta1={expected:g}", run.printed)

    def test_the_two_values_train_different_weights(self):
        first = self.fit("default").epoch(1)["model_state_dict"]
        second = self.fit("dcgan", adam_beta1=.5).epoch(1)["model_state_dict"]
        self.assertTrue(any(not torch.equal(first[key], second[key]) for key in first))

    def test_resume_rejects_another_value_and_accepts_the_same(self):
        self.fit("dcgan", adam_beta1=.5)
        checkpoint = self.root / "dcgan" / "model_epoch_1.pt"
        with self.assertRaisesRegex(ValueError, "adam_beta1"):
            self.fit("dcgan", epochs=2, resume_from=checkpoint)
        resumed = self.fit("dcgan", epochs=2, adam_beta1=.5, resume_from=checkpoint)
        self.assertEqual(self.betas(resumed.epoch(2))["generator"], {(.5, .999)})

    def test_a_checkpoint_without_the_option_resumes_with_the_default(self):
        self.fit("old")
        checkpoint = self.root / "old" / "model_epoch_1.pt"
        payload = torch.load(checkpoint, weights_only=False)
        del payload["learning_rates"]["adam_beta1"]  # As written before the option existed.
        torch.save(payload, checkpoint)
        resumed = self.fit("old", epochs=2, resume_from=checkpoint)
        self.assertEqual(self.betas(resumed.epoch(2))["discriminator"], {(.9, .999)})


if __name__ == "__main__":
    unittest.main()
