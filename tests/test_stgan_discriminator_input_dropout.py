"""`discriminator_input_dropout=False`: D judges the reconstruction without the generator's dropout,
in training and in the score, while the reconstruction loss and the Monte Carlo error keep it."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from physiq_pv.anomaly_detection.stgan import (
    STGAN, STGANCNNConfig, STGANGAT, build_spatial_grid, fit_and_score_stgan, grid_edge_index, load_stgan_checkpoint,
)
from physiq_pv.anomaly_detection.stgan.model import dropout_disabled, masked_cell_mean
from physiq_pv.anomaly_detection.stgan.scoring import scoring_mode
from physiq_pv.anomaly_detection.stgan.training import gan_train_step
from scripts import run_era5_stgan
from test_stgan_mmd import NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

FEATURES, RECENT, TREND = 3, 2, 5


def models(*, dropout=True):
    """The patch ConvGRU model and the two graph models, each with a batch of its own shape."""
    generator = torch.Generator().manual_seed(0)

    def random(*shape):
        return torch.randn(*shape, generator=generator)
    torch.manual_seed(0)
    patch = STGAN(n_features=FEATURES, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=2, dropout_enabled=dropout)
    cases = [("patch", patch, (random(4, RECENT, FEATURES, 3, 3), random(4, TREND, FEATURES), torch.ones(4, 1, 3, 3),
                               random(4, 31), random(4, FEATURES, 3, 3)))]
    rows, cols = np.indices((3, 4)).reshape(2, -1)
    grid = build_spatial_grid(45 - rows * .5, 7 + cols * .5, grid_crs="EPSG:4326", angular_spacing=.5, audit_knn=False)
    nodes = len(rows)
    for recurrence in ("pointwise", "gated"):
        graph = STGANGAT(n_features=FEATURES, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=2,
                         edge_index=grid_edge_index(grid.row_indices, grid.column_indices),
                         node_indices=grid.node_indices, recent_steps=RECENT, gat_hidden_dim=3, gat_heads=2,
                         discriminator_chunk_size=5, gat_recurrence=recurrence, dropout_enabled=dropout)
        cases.append((recurrence, graph, (random(2, RECENT, nodes, FEATURES), random(2, nodes, TREND, FEATURES),
                                          torch.ones(2, nodes, 1), random(2, 31), random(2, nodes, FEATURES))))
    return cases


def step(model, batch, **options):
    """One training step; returns the losses, the terms and what the step showed to D."""
    seen = {}
    optimizers = [torch.optim.Adam(part.parameters(), lr=1e-3) for part in (model.generator, model.discriminator)]
    model.train()
    result = gan_train_step(model, batch, *optimizers, return_terms=True,
                            observe=lambda name, values: seen.setdefault(name, dict(values)), **options)
    return result, seen


def reconstruction_error(model, generated, batch):
    """The reconstruction term of the step for one reconstruction: the patch mean of every sample or center."""
    recent, _, mask, _, observed = batch
    if not getattr(model, "global_graph", False):
        return masked_cell_mean(torch.where(mask.bool(), generated - observed, 0.0).square(), mask).mean()
    count, total = recent.shape[0] * recent.shape[2], 0.0
    for ids, times, safe, valid, _, real_patch in model.patch_inputs(recent, observed):
        errors = (model.gather_patch(generated, times, safe, valid) - real_patch).square()
        total = total + masked_cell_mean(errors, valid).mean() * (len(ids) / count)
    return total


class TrainingStepTests(unittest.TestCase):
    def test_by_default_the_discriminator_judges_the_output_of_the_generator(self):
        for name, model, batch in models():
            with self.subTest(model=name):
                self.assertTrue(model.discriminator_input_dropout)
                torch.manual_seed(0)
                _, seen = step(model, batch)
                self.assertNotIn("judged", seen["G_forward"])

    def test_without_the_option_the_discriminator_judges_the_reconstruction_without_dropout(self):
        for share_history in (True, False):
            for name, model, batch in models():
                with self.subTest(model=name, share_history=share_history):
                    model.discriminator_input_dropout = False
                    recent, trend, mask, calendar, observed = batch
                    model.eval()
                    with torch.no_grad():
                        clean = model.generator(recent, trend, mask, calendar)
                    before = [parameter.detach().clone() for parameter in model.generator.parameters()]
                    torch.manual_seed(0)
                    (generator_loss, _, terms), seen = step(model, batch, share_history=share_history)
                    # D step: the reconstruction has no dropout noise.
                    torch.testing.assert_close(seen["D_forward"]["generated"], clean, rtol=1e-5, atol=1e-6)
                    # G step: D judges the same clean reconstruction; the loss uses the one with dropout.
                    generated, judged = seen["G_forward"]["generated"], seen["G_forward"]["judged"]
                    torch.testing.assert_close(judged, clean, rtol=1e-5, atol=1e-6)
                    self.assertFalse(torch.allclose(generated, clean))
                    with torch.no_grad():
                        expected = reconstruction_error(model, generated, batch)
                        other = reconstruction_error(model, judged, batch)
                    torch.testing.assert_close(terms["generator_reconstruction_loss"].double(), expected.double(),
                                               rtol=1e-5, atol=1e-7)
                    self.assertFalse(torch.allclose(expected, other))
                    torch.testing.assert_close(
                        generator_loss.double(),
                        (terms["generator_reconstruction_weighted_loss"] + terms["generator_adversarial_loss"]).double(),
                        rtol=1e-5, atol=1e-6)
                    self.assertTrue(judged.requires_grad and generated.requires_grad)
                    self.assertTrue(any(not torch.equal(old, new.detach())
                                        for old, new in zip(before, model.generator.parameters())))
                    self.assertTrue(all(module.training for module in model.modules()))

    def test_the_adversarial_term_alone_updates_the_generator_through_the_clean_reconstruction(self):
        for name, model, batch in models():
            with self.subTest(model=name):
                model.discriminator_input_dropout = False
                before = [parameter.detach().clone() for parameter in model.generator.parameters()]
                torch.manual_seed(0)
                (_, _, terms), _ = step(model, batch, reconstruction_weight=0.0)
                self.assertEqual(float(terms["generator_reconstruction_weighted_loss"]), 0.0)
                changed = [not torch.equal(old, new.detach()) for old, new in zip(before, model.generator.parameters())]
                # Every part of G, the encoders included, receives the adversarial gradient.
                self.assertGreater(sum(changed), len(changed) // 2)

    def test_without_dropout_layers_the_option_changes_nothing(self):
        for name, model, batch in models(dropout=False):
            with self.subTest(model=name):
                state = {key: value.clone() for key, value in model.state_dict().items()}
                results = []
                for option in (True, False):
                    model.load_state_dict(state)
                    model.discriminator_input_dropout = option
                    torch.manual_seed(0)
                    (generator_loss, discriminator_loss, _), _ = step(model, batch)
                    results.append((generator_loss, discriminator_loss, *model.state_dict().values()))
                for a, b in zip(*results):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_dropout_disabled_restores_every_layer(self):
        _, model, _ = models()[2]
        model.train()
        model.generator.fusion_dropout.eval()
        with dropout_disabled(model.generator):
            self.assertFalse(any(layer.training for layer in model.generator.modules()
                                 if isinstance(layer, torch.nn.Dropout)))
        self.assertTrue(model.generator.spatial_dropout.training and model.generator.temporal_dropout.training)
        self.assertFalse(model.generator.fusion_dropout.training)


class ScoreDrawTests(unittest.TestCase):
    def test_the_discriminator_component_is_the_same_in_every_draw(self):
        for share_history in (True, False):
            for name, model, batch in models():
                with self.subTest(model=name, share_history=share_history):
                    with scoring_mode(model, True), torch.no_grad():
                        torch.manual_seed(0)
                        noisy = model.score_draws(*batch, 4, share_history=share_history)
                        model.discriminator_input_dropout = False
                        torch.manual_seed(0)
                        draws = model.score_draws(*batch, 4, share_history=share_history)
                    model.eval()
                    with torch.no_grad():
                        # One pass with dropout off: D(observation) - D(reconstruction without dropout).
                        expected = model.monitoring_outputs(*batch)[1]
                    # With the option the component varies between draws; without it, it does not.
                    self.assertGreater(float(noisy[:, :, 1].std(dim=0).max()), 0)
                    for draw in draws:
                        torch.testing.assert_close(draw[:, 1], expected, rtol=1e-4, atol=1e-5)
                    # The reconstruction error and the feature errors keep the draws.
                    self.assertGreater(float(draws[:, :, 0].std(dim=0).max()), 0)
                    torch.testing.assert_close(draws[:, :, 0], noisy[:, :, 0])
                    torch.testing.assert_close(draws[:, :, 2:], noisy[:, :, 2:])


class RunTests(unittest.TestCase):
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
            epochs=1, batch_size=1, score_batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, recent_steps=2, device="cpu", grid_crs="EPSG:4326", timestep_hours=3,
            angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper", cache_normalized=False, mc_samples=3,
            monitoring_timestamps=4, monitoring_feature_mmd_samples=16, score_storage="memmap",
            spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4, gat_recurrence="gated",
            checkpoint_path=self.root / name / "model.pt", score_dir=self.root / name / "scores",
            pca_reference=self.root / "pca", mmd_reference=self.root / "reference")
        options.update(overrides)
        printed = io.StringIO()
        with redirect_stdout(printed), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options) as result:
            return result.metadata, printed.getvalue()

    def test_the_run_records_the_option_and_scores_with_it(self):
        metadata, printed = self.fit("clean", discriminator_input_dropout=False)
        self.assertIs(metadata["discriminator_input_dropout"], False)
        self.assertIn("[stgan] discriminator input: reconstruction without the generator's dropout", printed)
        self.assertIn("discriminator_judges_the_reconstruction_without_generator_dropout", str(metadata))
        model, payload = load_stgan_checkpoint(self.root / "clean" / "model.pt")
        self.assertIs(payload["model_config"]["discriminator_input_dropout"], False)
        self.assertIs(model.discriminator_input_dropout, False)
        scores = self.root / "clean" / "scores"
        # D's component has no Monte Carlo spread; the reconstruction error keeps it.
        self.assertEqual(float(np.abs(np.load(scores / "discriminator_std.npy")).max()), 0.0)
        self.assertGreater(float(np.load(scores / "generator_std.npy").max()), 0.0)
        # The default run says nothing and keeps the spread of both components.
        metadata, printed = self.fit("default")
        self.assertIs(metadata["discriminator_input_dropout"], True)
        self.assertNotIn("discriminator input", printed)
        self.assertGreater(float(np.load(self.root / "default" / "scores" / "discriminator_std.npy").max()), 0.0)
        self.assertIs(load_stgan_checkpoint(self.root / "default" / "model.pt")[0].discriminator_input_dropout, True)

    def test_a_resume_with_the_other_setting_is_refused(self):
        self.fit("clean", discriminator_input_dropout=False)
        with self.assertRaisesRegex(ValueError, "discriminator_input_dropout"):
            self.fit("resumed", epochs=2, resume_from=self.root / "clean" / "model_epoch_1.pt")
        # The same setting resumes.
        metadata, printed = self.fit("continued", epochs=2, discriminator_input_dropout=False,
                                     resume_from=self.root / "clean" / "model_epoch_1.pt")
        self.assertIn("completed_epochs=1/2", printed)
        self.assertIs(metadata["discriminator_input_dropout"], False)

    def test_the_patch_convgru_of_this_branch_takes_the_option_too(self):
        metadata, _ = self.fit("patch", spatial_encoder="convgru", batch_size=4, score_batch_size=4,
                               discriminator_input_dropout=False)
        self.assertIs(metadata["discriminator_input_dropout"], False)
        self.assertEqual(float(np.abs(np.load(self.root / "patch" / "scores" / "discriminator_std.npy")).max()), 0.0)


class OptionTests(unittest.TestCase):
    def test_default_and_validation(self):
        self.assertIs(STGANCNNConfig().discriminator_input_dropout, True)
        with self.assertRaisesRegex(ValueError, "discriminator_input_dropout must be a boolean"):
            STGANCNNConfig(discriminator_input_dropout="no")

    def test_command_line(self):
        base = ["train", "--prepared-dir", "prepared", "--output-dir", "run"]
        self.assertIs(run_era5_stgan.parser().parse_args(base).discriminator_input_dropout, True)
        self.assertIs(
            run_era5_stgan.parser().parse_args(base + ["--no-discriminator-input-dropout"]).discriminator_input_dropout,
            False)


if __name__ == "__main__":
    unittest.main()
