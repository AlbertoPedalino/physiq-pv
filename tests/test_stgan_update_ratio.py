"""D:G optimizer update ratio: effective optimizer steps, 1:1 equivalence, resume, CLI and W&B.

The same file serves the ConvGRU and the GAT branch; graph cases run where the GAT exists.
"""
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import copy
import io
import json
from pathlib import Path
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
from physiq_pv.anomaly_detection.stgan.training import gan_train_step
from test_stgan_mc_dropout import small_model
from test_stgan_performance import fixture

HAS_GRAPH = hasattr(stgan, "STGANGAT")
RATIOS = {"1:1": (1, 1), "2:1": (2, 1), "1:2": (1, 2)}
NAME = "discriminator_generator_update_ratio"


def training_paths():
    """(name, model, batch) per training step of this branch; dropout off for exact comparisons."""
    yield "convgru", small_model(dropout_enabled=False).train(), fixture().fetch_batch(list(range(7)))[:5]
    if HAS_GRAPH:
        from test_stgan_gat import fixture as graph_fixture
        dataset, model = graph_fixture()  # Several discriminator chunks per global batch.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        yield "gat", model.train(), dataset.fetch_batch([0, 3])[:5]


def adam_steps(optimizer):
    return max((int(state["step"]) for state in optimizer.state.values()), default=0)


def run_step(model, batch, **options):
    """One training step on a copy, counting real optimizer.step() calls and timed scopes."""
    model = copy.deepcopy(model)
    d_optimizer = torch.optim.Adam(model.discriminator.parameters())
    g_optimizer = torch.optim.Adam(model.generator.parameters())
    scopes, losses = Counter(), {"D_backward": [], "G_backward": []}
    @contextmanager
    def measure(name):
        scopes[name] += 1
        yield
    def observe(stage, values):
        if stage in losses:
            losses[stage].append(values["loss"].detach().clone())
    with patch.object(d_optimizer, "step", wraps=d_optimizer.step) as d_step, \
         patch.object(g_optimizer, "step", wraps=g_optimizer.step) as g_step:
        g_loss, d_loss = gan_train_step(model, batch, g_optimizer, d_optimizer,
                                        measure=measure, observe=observe, **options)
    return SimpleNamespace(model=model, calls=(d_step.call_count, g_step.call_count),
        adam=(adam_steps(d_optimizer), adam_steps(g_optimizer)), scopes=scopes,
        d_loss=d_loss, g_loss=g_loss, d_losses=losses["D_backward"], g_losses=losses["G_backward"])


def same_weights(a, b):
    return all(torch.equal(value, b.state_dict()[name]) for name, value in a.state_dict().items())


class UpdateRatioTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_ratio_is_a_count_of_updates_with_one_to_one_default(self):
        default = STGANCNNConfig()
        self.assertEqual((getattr(default, NAME), default.discriminator_updates_per_batch,
                          default.generator_updates_per_batch), ("1:1", 1, 1))
        for ratio, steps in RATIOS.items():
            config = STGANCNNConfig(**{NAME: ratio}, learning_rate=.0002, discriminator_lr_ratio=4.)
            self.assertEqual((config.discriminator_updates_per_batch, config.generator_updates_per_batch), steps)
            # Independent of the learning-rate ratio.
            self.assertEqual(config.effective_discriminator_learning_rate, .0008)
        # 61 is how YAML reads an unquoted 1:1.
        for bad in ("2:2", "0:1", "1:0", "1", "1:1:1", " 1:1", "2/1", 61, 2, 1., None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, NAME):
                STGANCNNConfig(**{NAME: bad})

    def test_each_ratio_runs_exactly_that_many_optimizer_steps(self):
        for name, model, batch in training_paths():
            chunks = run_step(model, batch).scopes["D_backward"]
            # The GAT accumulates several discriminator chunks into one update.
            self.assertTrue(chunks > 1 if name == "gat" else chunks == 1, (name, chunks))
            for ratio, (d, g) in RATIOS.items():
                with self.subTest(path=name, ratio=ratio):
                    result = run_step(model, batch, discriminator_steps=d, generator_steps=g)
                    self.assertEqual(result.calls, (d, g))
                    self.assertEqual(result.adam, (d, g))
                    self.assertEqual((result.scopes["D_optimizer"], result.scopes["G_optimizer"]), (d, g))
                    # Chunk backwards scale with the updates; they are never counted as updates.
                    self.assertEqual(result.scopes["D_backward"], chunks * d)
                    self.assertEqual((len(result.d_losses), len(result.g_losses)), (d, g))
                    torch.testing.assert_close(result.d_loss, torch.stack(result.d_losses).mean())
                    torch.testing.assert_close(result.g_loss, torch.stack(result.g_losses).mean())
            for bad in (dict(discriminator_steps=0), dict(generator_steps=0)):
                with self.assertRaises(ValueError):
                    run_step(model, batch, **bad)

    def test_one_to_one_is_the_single_step_and_extra_steps_touch_one_network(self):
        for name, model, batch in training_paths():
            with self.subTest(path=name):
                default = run_step(model, batch)
                explicit = run_step(model, batch, discriminator_steps=1, generator_steps=1)
                self.assertTrue(same_weights(default.model, explicit.model))
                self.assertTrue(torch.equal(default.d_loss, explicit.d_loss)
                                and torch.equal(default.g_loss, explicit.g_loss))
                self.assertFalse(same_weights(default.model, model))
                more_d = run_step(model, batch, discriminator_steps=2, generator_steps=1)
                more_g = run_step(model, batch, discriminator_steps=1, generator_steps=2)
                # The first update of each network is the 1:1 update.
                self.assertTrue(torch.equal(more_d.d_losses[0], default.d_loss))
                self.assertTrue(torch.equal(more_g.g_losses[0], default.g_loss))
                self.assertFalse(same_weights(more_d.model.discriminator, default.model.discriminator))
                # G steps follow all D steps: a second G step leaves D exactly as after 1:1.
                self.assertTrue(same_weights(more_g.model.discriminator, default.model.discriminator))
                self.assertFalse(same_weights(more_g.model.generator, default.model.generator))
                # The second step on the same batch starts from the updated network.
                self.assertFalse(torch.equal(more_d.d_losses[1], more_d.d_losses[0]))
                self.assertFalse(torch.equal(more_g.g_losses[1], more_g.g_losses[0]))

    def fit(self, name, **overrides):
        values = np.random.default_rng(3).normal(size=(12, 4, 2)).astype(np.float32)
        times = pd.date_range("2004-12-30", periods=12, freq="3h")
        options = dict(train_timestamps=times[:8], test_timestamps=times[8:],
            location_names=("0", "1", "2", "3"), feature_names=("a", "b"),
            latitudes=np.array([45., 45., 44.5, 44.5]), longitudes=np.array([7., 7.5, 7., 7.5]),
            epochs=1, batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=3, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, dropout_enabled=False, mc_dropout_enabled=False,
            checkpoint_path=self.root/name/"model.pt")
        options.update(overrides)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(values[:8], values[8:], **options) as result:
            return result.metadata

    def encoders(self):
        yield "convgru", {}
        if HAS_GRAPH:
            # Chunk size 1: four discriminator chunks per global batch.
            yield "gat", dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=1)

    def test_pipeline_counts_effective_updates_and_exports_the_ratio(self):
        for encoder, extra in self.encoders():
            for ratio, (d, g) in RATIOS.items():
                with self.subTest(encoder=encoder, ratio=ratio):
                    name = f"{encoder}_{ratio.replace(':', 'to')}"
                    metadata = self.fit(name, **{NAME: ratio}, **extra)
                    self.assertEqual((metadata[NAME], metadata["discriminator_updates_per_batch"],
                                      metadata["generator_updates_per_batch"]), (ratio, d, g))
                    self.assertEqual(metadata["paper_alignment"]["reference_hyperparameters"], False)
                    history = pd.read_csv(self.root/name/"training_history.csv")
                    # Three batches per epoch, whatever the number of discriminator chunks.
                    self.assertEqual((history.discriminator_updates.tolist(), history.generator_updates.tolist()),
                                     ([3 * d], [3 * g]))
                    epoch = torch.load(self.root/name/"model_epoch_1.pt", weights_only=False)
                    final = torch.load(self.root/name/"model.pt", weights_only=False)
                    self.assertEqual((epoch[NAME], final["training"][NAME]), (ratio, ratio))
                    self.assertEqual((final["training"]["discriminator_updates_per_batch"],
                                      final["training"]["generator_updates_per_batch"]), (d, g))
                    for network, steps in (("discriminator", 3 * d), ("generator", 3 * g)):
                        state = epoch[f"{network}_optimizer_state_dict"]["state"]
                        self.assertEqual({int(s["step"]) for s in state.values()}, {steps})

    def test_resume_requires_the_checkpoint_ratio(self):
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                name = f"resume_{encoder}"
                self.fit(name, **{NAME: "2:1"}, **extra)
                checkpoint = self.root/name/"model_epoch_1.pt"
                metadata = self.fit(name, epochs=2, resume_from=checkpoint, **{NAME: "2:1"}, **extra)
                self.assertEqual(metadata["resume"]["completed_epochs"], 1)
                history = pd.read_csv(self.root/name/"training_history.csv")
                self.assertEqual((history.discriminator_updates.tolist(), history.generator_updates.tolist()),
                                 ([6, 6], [3, 3]))
                for other in ("1:1", "1:2"):
                    with self.assertRaisesRegex(ValueError, NAME):
                        self.fit(name, epochs=2, resume_from=checkpoint, **{NAME: other}, **extra)
                # Checkpoints written before the option always used one D and one G step.
                self.fit(f"legacy_{encoder}", **extra)
                checkpoint = self.root/f"legacy_{encoder}"/"model_epoch_1.pt"
                payload = torch.load(checkpoint, weights_only=False)
                self.assertEqual(payload.pop(NAME), "1:1")
                torch.save(payload, checkpoint)
                self.assertEqual(self.fit(f"legacy_{encoder}", epochs=2, resume_from=checkpoint,
                                          **extra)["resume"]["completed_epochs"], 1)
                with self.assertRaisesRegex(ValueError, NAME):
                    self.fit(f"legacy_{encoder}", epochs=2, resume_from=checkpoint, **{NAME: "1:2"}, **extra)

    def test_cli_passes_the_ratio_to_both_runners(self):
        from scripts import run_era5_stgan, run_pvgis_stgan
        era5 = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        pvgis = ["--manifest", "unused", "--out-dir", "unused"]
        flag = "--discriminator-generator-update-ratio"
        for ratio, steps in RATIOS.items():
            options = [] if ratio == "1:1" else [flag, ratio]  # 1:1 is the default.
            with patch.object(run_era5_stgan, "run_training", return_value={}) as train, redirect_stdout(io.StringIO()):
                run_era5_stgan.main([*era5, *options])
            configs = [train.call_args.kwargs["config"]]
            with patch.object(run_pvgis_stgan, "run_stgan") as train:
                run_pvgis_stgan.main([*pvgis, *options])
            configs.append(train.call_args.kwargs["config"])
            for config in configs:
                self.assertEqual((getattr(config, NAME), config.discriminator_updates_per_batch,
                                  config.generator_updates_per_batch), (ratio, *steps))
        with self.assertRaisesRegex(ValueError, NAME), redirect_stdout(io.StringIO()):
            run_era5_stgan.main([*era5, flag, "2:2"])

    def test_wandb_sweep_samples_logs_and_validates_the_ratio(self):
        import yaml
        from physiq_pv.experiments.stgan_wandb import default_config, resolve_config, run_tracked
        from scripts.run_stgan_wandb import parse_args
        from test_stgan_wandb import FakeRun
        path = Path(__file__).resolve().parents[1] / "sweeps/stgan_bayes.draft.yaml"
        draft = yaml.safe_load(path.read_text())
        parameters = draft["parameters"]
        self.assertEqual(parameters[NAME], {"values": list(RATIOS)})
        self.assertEqual({key for key, spec in parameters.items() if "value" not in spec},
                         {"generator_learning_rate", "discriminator_learning_rate", "generator_reconstruction_weight", NAME})
        self.assertEqual(draft["metric"], {"name": "validation/pca_mmd_rolling_mean", "goal": "minimize"})
        fixed = {key: spec["value"] for key, spec in parameters.items() if "value" in spec}
        for ratio, steps in RATIOS.items():
            config, _ = resolve_config(default_config("era5"), {**fixed, NAME: ratio}, 20)
            self.assertEqual((config.discriminator_updates_per_batch, config.generator_updates_per_batch), steps)
        with self.assertRaisesRegex(ValueError, "quote it in YAML"):
            resolve_config(default_config("era5"), {NAME: yaml.safe_load("1:1")}, 20)

        source = self.root / "prepared"
        source.mkdir()
        args = parse_args(["--backend", "era5", "--prepared-dir", str(source), "--output-root",
                           str(self.root / "runs"), "--device", "cpu", "--wandb-mode", "offline",
                           "--pca-reference-dir", str(self.root / "pca_reference"),
                           "--mmd-reference-dir", str(self.root / "mmd_reference")])
        seen = {}
        def initialize(**kwargs):
            self.assertEqual(kwargs["config"][NAME], "1:1")
            seen["run"] = FakeRun({**kwargs["config"], NAME: "2:1"})  # The agent's sampled value.
            return seen["run"]
        def train(args, config, seed, output, on_epoch):
            self.assertEqual((config.discriminator_updates_per_batch, config.generator_updates_per_batch), (2, 1))
            on_epoch(dict(epoch=1, generator_loss=3., discriminator_loss=.6, seconds=2., samples=4,
                          discriminator_updates=8, generator_updates=4))
            return {"precision": config.precision, "performance": {}}
        with patch("wandb.init", side_effect=initialize), \
             patch("physiq_pv.experiments.stgan_wandb.execute_training", side_effect=train):
            output = run_tracked(args)
        run = seen["run"]
        self.assertEqual((run.summary[NAME], run.summary["discriminator_updates_per_batch"],
                          run.summary["generator_updates_per_batch"]), ("2:1", 2, 1))
        self.assertEqual((run.logs[0]["train/discriminator_updates"], run.logs[0]["train/generator_updates"]), (8, 4))
        self.assertEqual(json.loads((output / "wandb_run.json").read_text())["config"][NAME], "2:1")


if __name__ == "__main__":
    unittest.main()
