"""Full-grid CNN training mode: complete fields as samples, the local patch discriminator on every cell."""
from contextlib import redirect_stdout
import io
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
from torch.nn.functional import binary_cross_entropy

from physiq_pv.anomaly_detection.stgan import (
    STGAN, FullGridSTGAN, STGANCNNConfig, STGANFullGridDataset, STGANWindowDataset, build_spatial_grid,
    fit_and_score_stgan, load_stgan_checkpoint, masked_cell_mean)
from physiq_pv.anomaly_detection.stgan.mmd import MMDMonitor, load_mmd_reference
from physiq_pv.anomaly_detection.stgan.pca_reference import load_pca_reference
from physiq_pv.anomaly_detection.stgan.training import gan_train_step
from scripts import run_era5_stgan
from test_stgan_mmd import METRICS, NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

HEIGHT, WIDTH, FEATURES, STEPS = 5, 7, 3, 40
ROWS, COLS = np.indices((HEIGHT, WIDTH)).reshape(2, -1)
PRESENT = np.ones(HEIGHT * WIDTH, dtype=bool)
PRESENT[[9, 17]] = False  # Two cells without a location: (1, 2) and (2, 3).
LAT, LON = (45 - ROWS * .5)[PRESENT], (7 + COLS * .5)[PRESENT]
N = int(PRESENT.sum())
STAMPS = pd.date_range("2001-01-01", periods=STEPS, freq="3h")
POSITIONS = np.array([0, 5, 11])
CASES = [(recent, encoding) for recent in (1, 16) for encoding in ("onehot", "cyclic")]
# Cells in the interior, on the four borders, in two corners and next to the missing cells.
CELLS = [(2, 5), (3, 1), (0, 3), (4, 4), (2, 0), (1, 6), (0, 0), (4, 6), (1, 1), (1, 3), (2, 2), (3, 3)]


def case(recent_steps=1, encoding="onehot", *, dropout=False, seed=0):
    """A lattice with two holes, its patch and full-grid datasets, and the two models with the same weights."""
    grid = build_spatial_grid(LAT, LON, grid_crs="EPSG:4326", angular_spacing=.5, audit_knn=False)
    data = np.random.default_rng(seed).normal(size=(STEPS, N, FEATURES)).astype(np.float32)
    options = dict(feature_minimum=data.min(axis=(0, 1)), feature_scale=np.ptp(data, axis=(0, 1)),
                   recent_steps=recent_steps, trend_steps=max(recent_steps, 6), stride=1,
                   time_encoding=encoding, longitudes=LON)
    patches, fields_ = STGANWindowDataset(data, STAMPS, grid, **options), STGANFullGridDataset(data, STAMPS, grid, **options)
    config = dict(n_features=FEATURES, hidden_size=8, n_layers=1, cnn_channels=4, cnn_layers=2,
                  time_feature_size=patches.time_feature_size, dropout_enabled=dropout)
    torch.manual_seed(seed)
    patch_model = STGAN(**config)
    model = FullGridSTGAN(grid=grid, **config)
    model.load_state_dict(patch_model.state_dict())
    return SimpleNamespace(grid=grid, patches=patches, fields=fields_, patch_model=patch_model, model=model,
                           field=fields_.fetch_batch(POSITIONS),
                           patch=patches.fetch_batch((POSITIONS[:, None] * N + np.arange(N)).ravel()))


def extract(field, row, column):
    """The 3x3 neighbourhood of one cell of [...,H,W], zero outside the lattice: no gather involved."""
    padded = torch.nn.functional.pad(field, (1, 1, 1, 1))
    return padded[..., row:row + 3, column:column + 3]


def location_of(grid, row, column):
    return int(np.flatnonzero((grid.row_indices == row) & (grid.column_indices == column))[0])


class LayoutTests(unittest.TestCase):
    def test_a_field_holds_the_windows_of_the_patch_dataset(self):
        for recent_steps, encoding in CASES:
            with self.subTest(recent_steps=recent_steps, encoding=encoding):
                c = case(recent_steps, encoding)
                recent, trend, mask, calendar, observed, position, location = c.field
                batch, steps = len(POSITIONS), max(recent_steps, 6)
                self.assertEqual(len(c.fields), len(c.fields.targets))  # One sample per timestamp.
                self.assertEqual(len(c.patches), len(c.fields) * N)
                self.assertEqual((recent.shape, trend.shape, mask.shape, observed.shape),
                                 ((batch, recent_steps, FEATURES, HEIGHT, WIDTH), (batch, steps, N, FEATURES),
                                  (batch, 1, HEIGHT, WIDTH), (batch, FEATURES, HEIGHT, WIDTH)))
                self.assertEqual(calendar.shape, (batch, 31) if encoding == "onehot" else (batch, N, 4))
                layout = c.model.layout
                # Every cell's patch, mask and trend are those of the patch dataset, bit for bit.
                self.assertTrue(torch.equal(layout.patches(recent), c.patch[0]))
                self.assertTrue(torch.equal(layout.patches(observed), c.patch[4]))
                self.assertTrue(torch.equal(layout.patch_masks(batch), c.patch[2]))
                self.assertTrue(torch.equal(trend.transpose(1, 2).flatten(0, 1), c.patch[1]))
                per_cell = calendar[:, None].expand(-1, N, -1) if encoding == "onehot" else calendar
                self.assertTrue(torch.equal(per_cell.flatten(0, 1), c.patch[3]))
                self.assertTrue(torch.equal(position, c.patch[5]) and torch.equal(location, c.patch[6]))
                # The lattice: locations where they are, zero and mask 0 at the two missing cells.
                self.assertEqual(int(mask[0].sum()), N)
                self.assertTrue(torch.equal(mask[0, 0].flatten(), torch.from_numpy(PRESENT.astype(np.float32))))
                self.assertFalse(recent[:, :, :, ~mask[0, 0].bool()].any() or observed[:, :, ~mask[0, 0].bool()].any())
                self.assertTrue(torch.equal(layout.cells(observed), c.patches.fetch_batch(
                    (POSITIONS[:, None] * N + np.arange(N)).ravel())[4][:, :, 1, 1]))
                self.assertTrue(torch.equal(layout.cells(layout.field(trend[:, -1])), trend[:, -1].flatten(0, 1)))

    def test_single_sample_and_bad_indices(self):
        c = case(2, "cyclic")
        single, batch = c.fields[5], c.fields.fetch_batch([5])
        for index in range(5):
            self.assertTrue(torch.equal(single[index], batch[index][0]))
        self.assertTrue(torch.equal(single[5], batch[5]) and torch.equal(single[6], batch[6]))
        for bad in ([], [-1], [len(c.fields)]):
            with self.assertRaises(IndexError):
                c.fields.fetch_batch(bad)


class DiscriminatorTests(unittest.TestCase):
    def test_local_score_of_every_cell_is_the_patch_discriminator_on_its_patch(self):
        for recent_steps, encoding in CASES:
            with self.subTest(recent_steps=recent_steps, encoding=encoding):
                c = case(recent_steps, encoding)
                recent, trend, mask, calendar, observed = c.field[:5]
                d, patch_d, layout = c.model.discriminator, c.patch_model.discriminator, c.model.layout
                with torch.no_grad():
                    generated = c.model.generator(recent, trend, mask, calendar)
                    real, fake = d.score_pair(recent, observed, generated, mask)
                    logits = d(torch.cat((recent, generated[:, None]), dim=1), mask, return_logits=True)
                    history = d.encode_history(recent, mask)
                    penultimate = d.penultimate(history, generated, mask)
                    # The patch discriminator on the batch of all patches: the same weights and masks.
                    expected_real, expected_fake = patch_d.score_pair(c.patch[0], c.patch[4], layout.patches(generated), c.patch[2])
                    expected_logits = patch_d(torch.cat((c.patch[0], layout.patches(generated)[:, None]), dim=1),
                                              c.patch[2], return_logits=True)
                # One local score per cell: as many as the patches the patch model scores for these timestamps.
                self.assertEqual((real.shape, fake.shape, penultimate.shape),
                                 ((len(POSITIONS) * N, 1), (len(POSITIONS) * N, 1), (len(POSITIONS) * N, 8)))
                self.assertEqual(len(real), len(c.patch[0]))
                for found, expected in ((real, expected_real), (fake, expected_fake), (logits, expected_logits)):
                    self.assertTrue(torch.allclose(found, expected, rtol=0, atol=1e-6))
                self.assertTrue(torch.allclose(fake, torch.sigmoid(logits), atol=1e-6))
                self.assertTrue(torch.allclose(d.score_from_penultimate(penultimate), fake, atol=1e-6))
                self.assertGreater(float(real.std()), 0)
                # Cell by cell, on patches cut out of the fields by hand: interior, borders, corners, holes.
                occupied = mask[0]
                for timestamp in range(len(POSITIONS)):
                    for row, column in CELLS:
                        row_index = timestamp * N + location_of(c.grid, row, column)
                        with torch.no_grad():
                            one_real, one_fake = patch_d.score_pair(
                                extract(recent[timestamp], row, column)[None], extract(observed[timestamp], row, column)[None],
                                extract(generated[timestamp], row, column)[None], extract(occupied, row, column)[None])
                        self.assertAlmostEqual(float(real[row_index]), float(one_real), places=6)
                        self.assertAlmostEqual(float(fake[row_index]), float(one_fake), places=6)
                self.assertLess(int(extract(occupied, 0, 0).sum()), 9)  # A corner patch is incomplete.
                self.assertLess(int(extract(occupied, 1, 3).sum()), 9)  # So is one next to a missing cell.

    def test_discriminator_is_local_and_the_generator_is_not(self):
        c = case(1)
        recent, trend, mask, calendar, observed = c.field[:5]
        cell = location_of(c.grid, 2, 5)
        with torch.no_grad():
            history = c.model.discriminator.encode_history(recent, mask)
            reference = c.model.discriminator.score_current(history, observed, mask)
            generated = c.model.generator(recent, trend, mask, calendar)
            far = observed.clone()
            far[0, :, 2, 2] += 5.  # Three columns from (2, 5): outside its 3x3 patch, inside that of (2, 1).
            moved = c.model.discriminator.score_current(history, far, mask)
            far_recent = recent.clone()
            far_recent[0, :, :, 3, 3] += 5.  # Two columns from (2, 5): outside its patch.
            regenerated = c.model.generator(far_recent, trend, mask, calendar)
        self.assertEqual(float(moved[cell]), float(reference[cell]))
        self.assertNotEqual(float(moved[location_of(c.grid, 2, 1)]), float(reference[location_of(c.grid, 2, 1)]))
        # G convolves the whole field: two ConvGRU layers reach two cells away, beyond the patch.
        self.assertGreater(float((regenerated - generated)[0, :, 2, 5].abs().max()), 0)


class GeneratorTests(unittest.TestCase):
    def test_shapes_through_the_generator(self):
        for recent_steps, encoding in CASES:
            with self.subTest(recent_steps=recent_steps, encoding=encoding):
                c = case(recent_steps, encoding)
                recent, trend, mask, calendar, observed = c.field[:5]
                batch, generator, seen = len(POSITIONS), c.model.generator, {}
                hooks = [module.register_forward_hook(lambda _, inputs, output, name=name: seen.__setitem__(
                    name, (tuple(inputs[0].shape), tuple((output[0] if isinstance(output, tuple) else output).shape))))
                    for name, module in (("convgru", generator.recent_encoder), ("trend", generator.trend_encoder),
                                         ("time", generator.time_projection), ("output", generator.output_projection))]
                with torch.no_grad():
                    generated = generator(recent, trend, mask, calendar)
                for hook in hooks:
                    hook.remove()
                self.assertEqual(seen["convgru"], ((batch, recent_steps, FEATURES, HEIGHT, WIDTH), (batch, 4, HEIGHT, WIDTH)))
                self.assertEqual(seen["trend"], ((batch * N, max(recent_steps, 6), FEATURES), (batch * N, max(recent_steps, 6), 8)))
                self.assertEqual(seen["time"][1], (batch, 8) if encoding == "onehot" else (batch, N, 8))
                self.assertEqual(seen["output"], ((batch, 4 + 2 * 8, HEIGHT, WIDTH), (batch, FEATURES, HEIGHT, WIDTH)))
                self.assertEqual(generated.shape, observed.shape)
                self.assertTrue(torch.isfinite(generated).all())
                # Missing cells stay zero in the reconstruction, as in the observations.
                self.assertFalse(generated[:, :, ~mask[0, 0].bool()].any())
                self.assertTrue(generated[:, :, mask[0, 0].bool()].abs().gt(0).all())

    def test_trend_and_cyclic_time_are_those_of_each_cell(self):
        c = case(4, "cyclic")
        recent, trend, mask, calendar, observed = c.field[:5]
        generator = c.model.generator
        with torch.no_grad():
            spatial, temporal, projected = generator.encode(recent, trend, mask, calendar)
            expected, _ = c.patch_model.generator.trend_encoder(c.patch[1])
            expected_time = c.patch_model.generator.time_projection(c.patch[3])
        self.assertTrue(torch.allclose(temporal.flatten(0, 1), expected[:, -1], atol=1e-6))
        self.assertTrue(torch.allclose(projected.flatten(0, 1), expected_time, atol=1e-6))
        self.assertGreater(float(calendar[0, :, 0].std()), 0)  # Local solar time differs with the longitude.

    def test_same_weights_as_the_patch_model_and_dropout_stays_in_the_generator(self):
        c = case(1, dropout=True)
        self.assertEqual(list(c.model.state_dict()), list(c.patch_model.state_dict()))
        self.assertEqual(c.model.parameter_counts(), c.patch_model.parameter_counts())
        c.model.train()
        recent, trend, mask, calendar, observed = c.field[:5]
        with torch.no_grad():
            first, second = (c.model.generator(recent, trend, mask, calendar) for _ in range(2))
        self.assertFalse(torch.equal(first, second))
        c.model.eval()
        with torch.no_grad():
            first, second = (c.model.generator(recent, trend, mask, calendar) for _ in range(2))
        self.assertTrue(torch.equal(first, second))


class TrainingTests(unittest.TestCase):
    def test_losses_are_cell_means_finite_and_update_both_networks(self):
        for recent_steps, encoding in CASES:
            with self.subTest(recent_steps=recent_steps, encoding=encoding):
                c = case(recent_steps, encoding)
                model, batch = c.model, c.field[:5]
                recent, trend, mask, calendar, observed = batch
                with torch.no_grad():
                    generated = model.generator(recent, trend, mask, calendar)
                    real, fake = c.patch_model.discriminator.score_pair(
                        c.patch[0], c.patch[4], model.layout.patches(generated), c.patch[2])
                # Rates of zero leave the weights as they are: the terms are those of the initial model.
                frozen = [torch.optim.Adam(part.parameters(), lr=0.) for part in (model.generator, model.discriminator)]
                generator_loss, discriminator_loss, terms = gan_train_step(model, batch, *frozen, return_terms=True)
                valid = mask.bool().expand_as(observed)
                # Reconstruction: mean over the cells with a location and the features, then over timestamps.
                expected = torch.stack([(generated[b] - observed[b])[valid[b]].square().mean() for b in range(len(recent))]).mean()
                self.assertAlmostEqual(float(terms["generator_reconstruction_loss"]), float(expected), places=6)
                self.assertAlmostEqual(float(terms["generator_reconstruction_weighted_loss"]), 500 * float(expected), places=3)
                # Adversarial terms: the mean of the BCE over the local outputs, one per cell.
                self.assertAlmostEqual(float(terms["discriminator_real_loss"]),
                                       .5 * float(binary_cross_entropy(real, torch.zeros_like(real))), places=6)
                self.assertAlmostEqual(float(terms["discriminator_fake_loss"]),
                                       .5 * float(binary_cross_entropy(fake, torch.ones_like(fake))), places=6)
                self.assertAlmostEqual(float(terms["generator_adversarial_loss"]),
                                       float(binary_cross_entropy(fake, torch.zeros_like(fake))), places=6)
                self.assertAlmostEqual(float(terms["discriminator_real_mean"]), float(real.mean()), places=6)
                self.assertTrue(torch.isfinite(generator_loss) and torch.isfinite(discriminator_loss))
                # A real step: finite gradients in both networks, and both move.
                before = {name: value.clone() for name, value in model.state_dict().items()}
                optimizers = [torch.optim.Adam(part.parameters(), lr=1e-3) for part in (model.generator, model.discriminator)]
                losses = gan_train_step(model, batch, *optimizers)
                self.assertTrue(all(torch.isfinite(loss) for loss in losses))
                for part in ("generator", "discriminator"):
                    changed = [name for name, value in model.state_dict().items()
                               if name.startswith(part) and not torch.equal(value, before[name])]
                    self.assertTrue(changed, part)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.generator.parameters()))
                self.assertTrue(all(p.requires_grad for p in model.discriminator.parameters()))

    def test_adversarial_gradient_reaches_the_generator_through_the_local_discriminator(self):
        c = case(2)
        before = [p.detach().clone() for p in c.model.generator.parameters()]
        optimizers = [torch.optim.Adam(part.parameters(), lr=1e-3) for part in (c.model.generator, c.model.discriminator)]
        gan_train_step(c.model, c.field[:5], *optimizers, reconstruction_weight=0.)
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(c.model.generator.parameters(), before)))

    def test_missing_cells_do_not_enter_the_losses(self):
        c = case(2)
        recent, trend, mask, calendar, observed = c.field[:5]
        altered = observed.clone()
        altered[:, :, ~mask[0, 0].bool()] = 9.
        results = []
        for target in (observed, altered):
            copy = FullGridSTGAN(grid=c.grid, n_features=FEATURES, hidden_size=8, n_layers=1, cnn_channels=4,
                                 cnn_layers=2, dropout_enabled=False)
            copy.load_state_dict(c.model.state_dict())
            frozen = [torch.optim.Adam(part.parameters(), lr=0.) for part in (copy.generator, copy.discriminator)]
            results.append(gan_train_step(copy, (recent, trend, mask, calendar, target), *frozen, return_terms=True)[2])
        for name, value in results[0].items():
            self.assertEqual(float(value), float(results[1][name]), name)

    def test_scores_keep_the_patch_definitions(self):
        c = case(2, "cyclic", dropout=True)
        recent, trend, mask, calendar, observed = c.field[:5]
        model, layout = c.model, c.model.layout
        model.eval()
        with torch.no_grad():
            draws = model.score_draws(recent, trend, mask, calendar, observed, 3)
            generated = model.generator(recent, trend, mask, calendar)
            errors = (generated - observed).square()
            real, fake = c.patch_model.discriminator.score_pair(c.patch[0], c.patch[4], layout.patches(generated), c.patch[2])
            g, d, real_features, fake_features = model.monitoring_outputs(recent, trend, mask, calendar, observed)
            cells = model.reconstructed_cells(recent, trend, mask, calendar, observed)
        self.assertEqual(draws.shape, (3, len(POSITIONS) * N, 2 + FEATURES))
        # Reconstruction component: mean squared error over the cell's patch, as in the patch model.
        expected = masked_cell_mean(layout.patches(errors), c.patch[2])
        self.assertTrue(torch.allclose(draws[0, :, 0], expected, atol=1e-6))
        self.assertTrue(torch.allclose(draws[0, :, 1:2], real - fake, atol=1e-6))
        self.assertTrue(torch.allclose(draws[0, :, 2:], layout.cells(errors), atol=1e-6))
        self.assertTrue(torch.equal(draws[0], draws[1]))  # Dropout off: the draws coincide.
        self.assertTrue(torch.allclose(g, expected, atol=1e-6) and torch.allclose(d, (real - fake).squeeze(1), atol=1e-6))
        self.assertEqual((real_features.shape, fake_features.shape), ((len(POSITIONS) * N, 8),) * 2)
        self.assertTrue(torch.equal(cells[0], c.patch[4][:, :, 1, 1]))
        self.assertTrue(torch.allclose(cells[1], layout.cells(generated), atol=1e-6))
        self.assertTrue(model.reconstruction_valid(mask).all() and len(model.reconstruction_valid(mask)) == len(POSITIONS) * N)
        # MC dropout: only G's dropouts are stochastic, and the draws differ.
        for module in model.generator.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
        with torch.no_grad():
            stochastic = model.score_draws(recent, trend, mask, calendar, observed, 3)
        self.assertFalse(torch.equal(stochastic[0], stochastic[1]))
        self.assertTrue(torch.isfinite(stochastic).all())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields()
        build_pca(self.root / "pca", self.values)
        build(self.root / "reference", self.values, self.root / "pca")

    def fit(self, name, *, references=True, **overrides):
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=self.values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=2, batch_size=1, score_batch_size=2, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, device="cpu", grid_crs="EPSG:4326", cnn_training_mode="full_grid",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, mc_samples=2, monitoring_timestamps=4, monitoring_feature_mmd_samples=16,
            checkpoint_path=self.root / name / "model.pt", score_dir=self.root / name / "scores",
            pca_reference=self.root / "pca" if references else None,
            mmd_reference=self.root / "reference" if references else None)
        options.update(overrides)
        if options["cnn_training_mode"] is None:  # The pipeline's own default.
            del options["cnn_training_mode"]
        stream = io.StringIO()
        with redirect_stdout(stream), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options) as result:
            return SimpleNamespace(metadata=result.metadata, printed=stream.getvalue(),
                scores=np.array(result.test_scores), std=np.array(result.anomaly_std),
                features=np.array(result.test_feature_scores),
                history=pd.read_csv(self.root / name / "training_history.csv"),
                epoch=lambda n: torch.load(self.root / name / f"model_epoch_{n}.pt", weights_only=False))

    def test_a_small_training_run_steps_once_per_timestamp(self):
        for recent_steps, encoding in CASES:
            with self.subTest(recent_steps=recent_steps, encoding=encoding):
                name = f"{recent_steps}_{encoding}"
                trend_steps = max(recent_steps, 2)
                run = self.fit(name, recent_steps=recent_steps, trend_steps=trend_steps, time_encoding=encoding)
                timestamps = 20 - trend_steps  # Training timestamps with a complete history.
                audit = run.metadata["training_audit"]
                self.assertEqual(audit, {"cnn_training_mode": "full_grid", "batch_unit": "global_timestamps",
                    "timestamps": timestamps, "locations": 9, "samples_per_epoch": timestamps, "batch_size": 1,
                    "batches_per_epoch": timestamps, "cells_per_batch": 9, "discriminator_patches_per_batch": 9})
                self.assertIn("'batch_unit': 'global_timestamps'", run.printed)
                self.assertIn(f"batch={timestamps}/{timestamps}", run.printed)
                # One D step and one G step per timestamp: 9 times fewer than one per cell.
                for column in ("discriminator_updates", "generator_updates", "samples"):
                    self.assertEqual(run.history[column].tolist(), [timestamps] * 2, column)
                for column in ("generator_loss", "discriminator_loss", "generator_reconstruction_loss") + METRICS:
                    self.assertTrue(np.isfinite(run.history[column]).all(), column)
                self.assertEqual((run.scores.shape, run.std.shape, run.features.shape), ((8, 9), (8, 9), (8, 9, 2)))
                self.assertTrue(np.isfinite(run.scores).all() and np.isfinite(run.features).all())
                self.assertGreater(run.scores.std(), 0)
                self.assertEqual((run.metadata["cnn_training_mode"], run.metadata["training_sampling"],
                                  run.metadata["reconstruction_reduction"]),
                                 ("full_grid", "complete_shuffled_global_timestamps",
                                  "mean_valid_cells_and_features_per_field_then_mean_timestamps"))
                self.assertFalse(run.metadata["paper_alignment"]["reference_hyperparameters"])
                self.assertIn("local_patch_discriminator_scored_on_every_cell_of_the_field",
                              run.metadata["paper_alignment"]["domain_adaptations"])
                self.assertEqual(run.metadata["validation_objective"]["valid_points"], 16 * 9)
                # The checkpoint is rebuilt as a full-grid model, with the patch model's parameter names.
                restored, payload = load_stgan_checkpoint(self.root / name / "model.pt")
                self.assertIsInstance(restored, FullGridSTGAN)
                self.assertEqual((payload["cnn_training_mode"], run.epoch(1)["cnn_training_mode"]), ("full_grid",) * 2)
                self.assertEqual(list(restored.state_dict()), list(STGAN(**payload["model_config"]).state_dict()))

    def test_monitoring_and_pca_mmd_leave_no_trace_on_training(self):
        monitored = self.fit("monitored")
        plain = self.fit("plain", references=False, monitoring_timestamps=0)
        self.assertNotIn("validation_pca_mmd_mean", plain.history.columns)
        for column in METRICS + ("validation_discriminator_feature_discrepancy", "validation_discriminator_feature_mmd"):
            self.assertTrue(monitored.history[column].notna().all(), column)
        # Same weights after every epoch and the same test scores: the monitors only read the model.
        for epoch in (1, 2):
            for key, value in plain.epoch(epoch)["model_state_dict"].items():
                self.assertTrue(torch.equal(value, monitored.epoch(epoch)["model_state_dict"][key]), key)
        np.testing.assert_array_equal(monitored.scores, plain.scores)
        for column in ("generator_loss", "discriminator_loss"):
            np.testing.assert_array_equal(monitored.history[column], plain.history[column])
        # Directly: no gradient, no change of mode, of weights or of the random streams.
        model, _ = load_stgan_checkpoint(self.root / "monitored" / "model.pt")
        model.train()
        grid = build_spatial_grid(NAMES["latitudes"], NAMES["longitudes"], grid_crs="EPSG:4326",
                                  angular_spacing=.5, audit_knn=False)
        pca = load_pca_reference(self.root / "pca")
        # The validation year with its two timestamps of training history: the reference's targets.
        data, times = np.concatenate((self.values[18:20], self.values[VALIDATION])), TIMES[18:36]
        dataset = STGANFullGridDataset(data, times, grid, feature_minimum=pca.minimum, feature_scale=pca.scale,
                                       recent_steps=1, trend_steps=2, stride=1, longitudes=NAMES["longitudes"])
        monitor = MMDMonitor(dataset, load_mmd_reference(self.root / "reference"), pca, batch_size=2, device="cpu")
        before = {name: value.clone() for name, value in model.state_dict().items()}
        state = torch.get_rng_state()
        result = monitor.evaluate(model)
        self.assertTrue(np.isfinite(list(result.values())).all())
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertTrue(all(module.training for module in model.modules()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()))

    def test_patch_mode_is_the_default_and_keeps_its_unit(self):
        self.assertEqual(STGANCNNConfig().cnn_training_mode, "patch")
        for bad in (dict(cnn_training_mode="grid"), dict(cnn_training_mode="full_grid", execution_mode="legacy")):
            with self.assertRaisesRegex(ValueError, "cnn_training_mode"):
                STGANCNNConfig(**bad)
        patch_run = self.fit("patch", cnn_training_mode="patch", batch_size=4, score_batch_size=None)
        default = self.fit("default", cnn_training_mode=None, batch_size=4, score_batch_size=None)
        np.testing.assert_array_equal(patch_run.scores, default.scores)
        self.assertEqual(default.metadata["cnn_training_mode"], "patch")
        audit = patch_run.metadata["training_audit"]
        self.assertEqual(audit, {"cnn_training_mode": "patch", "batch_unit": "time_location_patches",
            "timestamps": 18, "locations": 9, "samples_per_epoch": 162, "batch_size": 4, "batches_per_epoch": 41,
            "cells_per_batch": 4, "discriminator_patches_per_batch": 4})
        self.assertEqual(patch_run.history.discriminator_updates.tolist(), [41, 41])
        self.assertEqual((patch_run.metadata["training_sampling"], patch_run.metadata["reconstruction_reduction"]),
                         ("complete_shuffled_time_location_product",
                          "mean_valid_cells_and_features_per_sample_then_mean_samples"))
        restored, payload = load_stgan_checkpoint(self.root / "patch" / "model.pt")
        self.assertIs(type(restored), STGAN)
        self.assertNotIn("row_indices", payload["grid"])
        # The two units are different runs: a checkpoint of one does not resume the other.
        self.fit("field")
        for source, mode, batch_size in (("patch", "full_grid", 1), ("field", "patch", 4)):
            with self.assertRaisesRegex(ValueError, "cnn_training_mode"):
                self.fit(f"resumed_as_{mode}", cnn_training_mode=mode, epochs=3, batch_size=batch_size,
                         resume_from=self.root / source / "model_epoch_2.pt")
        # An epoch checkpoint written before the option is a patch one.
        old = patch_run.epoch(2)
        old.pop("cnn_training_mode")
        torch.save(old, self.root / "old.pt")
        self.assertIs(type(load_stgan_checkpoint(self.root / "old.pt")[0]), STGAN)

    def test_command_selects_the_mode_and_its_batch_defaults(self):
        base = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        cases = (([], ("patch", 256, 1024)), (["--cnn-training-mode", "patch"], ("patch", 256, 1024)),
                 (["--cnn-training-mode", "full_grid"], ("full_grid", 1, 1)),
                 (["--cnn-training-mode", "full_grid", "--batch-size", "2", "--score-batch-size", "4"], ("full_grid", 2, 4)),
                 (["--batch-size", "128"], ("patch", 128, 1024)))
        for arguments, expected in cases:
            with patch.object(run_era5_stgan, "run_training", return_value={}) as train, redirect_stdout(io.StringIO()):
                run_era5_stgan.main(base + arguments)
            config = train.call_args.kwargs["config"]
            self.assertEqual((config.cnn_training_mode, config.batch_size, config.score_batch_size), expected)
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            run_era5_stgan.main(base + ["--cnn-training-mode", "global"])


if __name__ == "__main__":
    unittest.main()
