"""Validation MMD in the fixed PCA space: reference artefact, per-epoch metrics, rolling objective.

The same file serves the ConvGRU and the GAT branch; graph cases run where the GAT exists.
"""
from contextlib import redirect_stdout
import copy
import hashlib
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
from scipy.spatial.distance import cdist, pdist
import torch

from physiq_pv.anomaly_detection import stgan
from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan import mmd as mmd_module
from physiq_pv.anomaly_detection.stgan import pca_reference as pca_module
from physiq_pv.anomaly_detection.stgan.mmd import (
    ARRAYS, SCALES, MMDMonitor, build_mmd_reference, choose_subsets, load_mmd_reference, median_sigma,
    multiscale_mmd2)
from physiq_pv.anomaly_detection.stgan.pca_reference import build_pca_reference, load_pca_reference
from physiq_pv.era5.data import ERA5Cubes
from physiq_pv.experiments.stgan_wandb import RunLogger, default_config, run_tracked, validate_sweep
from scripts import run_era5_stgan

HAS_GRAPH = hasattr(stgan, "STGANGAT")
TIMES = pd.date_range("2003-12-28 12:00", periods=44, freq="3h").as_unit("ns")
TRAIN, VALIDATION, TEST = slice(0, 20), slice(20, 36), slice(36, 44)
ROWS, COLS = np.indices((3, 3)).reshape(2, -1)
NAMES = dict(feature_names=("a", "b"), location_names=tuple(map(str, range(9))),
             latitudes=45 - ROWS * .5, longitudes=7 + COLS * .5)
COMPONENTS = 3  # Leading components of the small PCA reference that the MMD uses.
METRICS = ("validation_pca_mmd_mean", "validation_pca_mmd_variance", "validation_pca_mmd_rolling_mean")
OBJECTIVE = "validation/pca_mmd_rolling_mean"


def fields(seed=3, steps=len(TIMES)):
    """Fields [time, 9 locations, 2 variables] with a few dominant spatial patterns plus noise."""
    rng = np.random.default_rng(seed)
    patterns = rng.normal(size=(3, 9, 2)) * np.array([4., 2., 1.])[:, None, None]
    return (np.einsum("tk,knf->tnf", rng.normal(size=(steps, 3)), patterns)
            + .05 * rng.normal(size=(steps, 9, 2))).astype(np.float32)


def build_pca(directory, values):
    with redirect_stdout(io.StringIO()):
        return build_pca_reference(values[TRAIN], TIMES[TRAIN], output_dir=directory, pca_samples=16, **NAMES)


def build(directory, values, pca, *, validation=None, **options):
    options = {"n_subsets": 5, "subset_size": 3, "n_components": COMPONENTS, **{**NAMES, **options}}
    with redirect_stdout(io.StringIO()):
        return build_mmd_reference(values[VALIDATION] if validation is None else validation, TIMES[VALIDATION],
                                   load_pca_reference(pca), output_dir=directory, **options)


def explicit_mmd2(x, y, bandwidths):
    """The definition, term by term, with the mean of the RBF kernels of the given bandwidths."""
    def kernel(a, b):
        squared = cdist(a, b, "sqeuclidean")
        return np.mean([np.exp(-squared / (2 * bandwidth ** 2)) for bandwidth in bandwidths], axis=0)
    kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
    n, m = len(x), len(y)
    off_diagonal = lambda k: sum(k[i, j] for i in range(len(k)) for j in range(len(k)) if i != j)
    return off_diagonal(kxx) / (n * (n - 1)) + off_diagonal(kyy) / (m * (m - 1)) - 2 * kxy.sum() / (n * m)


def file_hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(directory).iterdir())}


class Unread:
    """A partition that fails on any access to its values."""
    def __getitem__(self, key):
        raise AssertionError("a partition other than the validation one was read")
    __array__ = __len__ = __iter__ = __getitem__


class MathTests(unittest.TestCase):
    def test_unbiased_multiscale_mmd_matches_its_definition(self):
        rng = np.random.default_rng(2)
        x, y = rng.normal(size=(9, 4)), rng.normal(size=(7, 4)) + .8
        sigma = 1.7
        value = multiscale_mmd2(x, y, sigma)
        self.assertAlmostEqual(value, explicit_mmd2(x, y, (sigma / 2, sigma, 2 * sigma)), places=12)
        self.assertEqual(SCALES, (.5, 1., 2.))
        # The kernel is the mean of the three, not one of them and not another set of bandwidths.
        singles = [multiscale_mmd2(x, y, sigma, scales=(scale,)) for scale in SCALES]
        self.assertAlmostEqual(value, np.mean(singles), places=12)
        for index, bandwidth in enumerate((sigma / 2, sigma, 2 * sigma)):
            self.assertAlmostEqual(singles[index], explicit_mmd2(x, y, (bandwidth,)), places=12)
        self.assertGreater(max(abs(value - single) for single in singles), 1e-3)
        self.assertNotAlmostEqual(value, explicit_mmd2(x, y, (sigma / 4, sigma, 4 * sigma)), places=4)
        # Unbiased: the diagonal terms are left out, so equal samples do not give a positive value.
        self.assertLess(multiscale_mmd2(x, x, sigma), 0)
        same = rng.normal(size=(400, 4))
        self.assertLess(abs(multiscale_mmd2(same[:200], same[200:], 2.)), .02)
        self.assertGreater(multiscale_mmd2(same[:200], same[200:] + 1.5, 2.), .2)
        self.assertAlmostEqual(multiscale_mmd2(x, y, sigma, chunk_size=2), value, places=12)
        for bad in (dict(sigma=0.), dict(sigma=float("nan"))):
            with self.assertRaises(ValueError):
                multiscale_mmd2(x, y, **bad)
        with self.assertRaises(ValueError):
            multiscale_mmd2(x[:1], y, sigma)

    def test_median_bandwidth_and_fixed_subsets(self):
        points = np.random.default_rng(3).normal(size=(40, 5))
        self.assertAlmostEqual(median_sigma(points), np.median(pdist(points)), places=12)
        self.assertAlmostEqual(median_sigma(points, chunk_size=7), np.median(pdist(points)), places=12)
        with self.assertRaises(ValueError):
            median_sigma(np.ones((5, 3)))
        # The validation year: 2928 three-hourly timestamps, 5 subsets of 512 without overlap.
        positions, disjoint = choose_subsets(2928, 5, 512)
        self.assertEqual((positions.shape, disjoint, len(np.unique(positions))), ((5, 512), True, 2560))
        self.assertTrue((np.diff(positions, axis=1) > 0).all() and positions.min() >= 0 and positions.max() < 2928)
        # Spread over the whole validation, and the same whatever happened to the global random state.
        self.assertTrue(all(row.min() < 100 and row.max() > 2800 for row in positions))
        np.random.seed(123)
        torch.manual_seed(123)
        np.testing.assert_array_equal(choose_subsets(2928, 5, 512)[0], positions)
        self.assertFalse(np.array_equal(choose_subsets(2928, 5, 512, seed=1)[0], positions))
        overlapping, disjoint = choose_subsets(10, 5, 4)  # Too few samples for disjoint subsets.
        self.assertEqual((overlapping.shape, disjoint), ((5, 4), False))
        self.assertTrue(all(len(set(row)) == 4 for row in overlapping))
        with self.assertRaises(ValueError):
            choose_subsets(3, 2, 4)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields()
        self.pca = build_pca(self.root / "pca", self.values)

    def test_reference_uses_the_saved_pca_and_validation_observations_only(self):
        before = file_hashes(self.root / "pca")
        with patch.object(pca_module, "fit_pca", side_effect=AssertionError("the PCA was refitted")):
            reference = build(self.root / "a", self.values, self.root / "pca")
        self.assertEqual(file_hashes(self.root / "pca"), before)
        metadata = reference.metadata
        # Subsets: fixed validation timestamps, without overlap; a sample is one complete field.
        self.assertEqual((reference.positions.shape, metadata["subsets"]["disjoint"], len(np.unique(reference.positions))),
                         ((5, 3), True, 15))
        np.testing.assert_array_equal(reference.positions, choose_subsets(16, 5, 3)[0])
        np.testing.assert_array_equal(reference.timestamps, TIMES[VALIDATION].asi8[reference.positions])
        self.assertEqual((metadata["sample"], metadata["subsets"]["from"], metadata["subsets"]["seed"]),
                         ("one_complete_field_per_validation_timestamp", "validation", 0))
        # Real features: the saved normalization and the leading saved components, nothing fitted here.
        real = self.pca.transform_raw(self.values[VALIDATION][reference.positions.ravel()], COMPONENTS)
        np.testing.assert_allclose(reference.real_features.reshape(15, COMPONENTS), real, rtol=1e-12)
        self.assertEqual((reference.pca_fingerprint, reference.n_components, metadata["pca_reference"]["refitted"]),
                         (self.pca.fingerprint, COMPONENTS, False))
        # Bandwidth: the median distance between those real features, saved once.
        self.assertAlmostEqual(reference.sigma, np.median(pdist(real)), places=12)
        self.assertEqual((metadata["bandwidth"]["sigma"], metadata["bandwidth"]["fitted_on"], metadata["bandwidth"]["scales"]),
                         (reference.sigma, "observations_only", [.5, 1., 2.]))
        self.assertEqual(metadata["bandwidth"]["bandwidths"], [reference.sigma / 2, reference.sigma, 2 * reference.sigma])
        self.assertEqual((metadata["training_data_used"], metadata["test_data_used"], metadata["labels_used"],
                          metadata["model_used"]), (False,) * 4)
        self.assertEqual(metadata["data"]["validation"],
                         {"start": str(TIMES[20]), "end": str(TIMES[35]), "timestamps": 16})
        summary = reference.summary()
        self.assertEqual((summary["fingerprint"], summary["pca_reference"], summary["n_components"], summary["n_subsets"],
                          summary["subset_size"], summary["subsets_disjoint"], summary["sigma"]),
                         (reference.fingerprint, self.pca.fingerprint, COMPONENTS, 5, 3, True, reference.sigma))
        # Saved, reloadable, checksummed; an existing reference is never overwritten.
        self.assertEqual(sorted(path.name for path in (self.root / "a").iterdir()),
                         sorted([name + ".npy" for name in ARRAYS] + ["metadata.json"]))
        reloaded = load_mmd_reference(self.root / "a")
        self.assertEqual((reloaded.fingerprint, reloaded.sigma, len(reference.fingerprint)),
                         (reference.fingerprint, reference.sigma, 64))
        for name in ("positions", "timestamps", "real_features"):
            np.testing.assert_array_equal(getattr(reloaded, name), getattr(reference, name))
        with self.assertRaisesRegex(ValueError, "new/empty"):
            build(self.root / "a", self.values, self.root / "pca")

    def test_same_inputs_give_the_same_reference(self):
        first = build(self.root / "a", self.values, self.root / "pca")
        again = build(self.root / "b", self.values, self.root / "pca")
        self.assertEqual(first.fingerprint, again.fingerprint)
        self.assertEqual(file_hashes(self.root / "a"), file_hashes(self.root / "b"))
        # Another validation: other real features and bandwidth, the same subsets.
        other_validation = build(self.root / "c", self.values, self.root / "pca",
                                 validation=fields(seed=9)[VALIDATION] * 50)
        changed = {name for name, digest in file_hashes(self.root / "c").items()
                   if digest != file_hashes(self.root / "a")[name]}
        self.assertEqual(changed, {"real_features.npy", "metadata.json"})
        self.assertNotEqual((other_validation.sigma, other_validation.fingerprint), (first.sigma, first.fingerprint))
        np.testing.assert_array_equal(other_validation.positions, first.positions)
        # Another PCA reference: another MMD reference, even with the same subsets.
        other_train = self.values.copy()
        other_train[TRAIN] = fields(seed=5)[TRAIN]
        other_pca = build_pca(self.root / "other_pca", other_train)
        on_other = build(self.root / "d", self.values, self.root / "other_pca")
        self.assertEqual(on_other.pca_fingerprint, other_pca.fingerprint)
        self.assertNotEqual(on_other.fingerprint, first.fingerprint)
        # The choices that are left open.
        chosen = build(self.root / "e", self.values, self.root / "pca", n_components=2, n_subsets=2, subset_size=4,
                       subset_seed=7)
        self.assertEqual((chosen.n_components, chosen.real_features.shape, chosen.metadata["subsets"]["seed"]),
                         (2, (2, 4, 2), 7))
        np.testing.assert_array_equal(chosen.real_features, self.pca.transform_raw(
            self.values[VALIDATION][chosen.positions.ravel()], 2).reshape(2, 4, 2))
        # More components than the PCA reference stores, or another grid: refused.
        with self.assertRaisesRegex(ValueError, "components stored"):
            build(self.root / "f", self.values, self.root / "pca", n_components=len(self.pca.components) + 1)
        with self.assertRaisesRegex(ValueError, "other variables or locations"):
            build(self.root / "g", self.values, self.root / "pca", feature_names=("a", "c"))
        with self.assertRaisesRegex(ValueError, "other variables or locations"):
            build(self.root / "h", self.values, self.root / "pca", latitudes=NAMES["latitudes"] + 1)
        self.assertFalse(any((self.root / name).exists() for name in "fgh"))

    def test_altered_files_are_rejected(self):
        build(self.root / "a", self.values, self.root / "pca")
        for name in ("subset_positions.npy", "subset_timestamps.npy", "real_features.npy"):
            with self.subTest(file=name):
                path = self.root / "a" / name
                original = path.read_bytes()
                array = np.load(path)
                array.reshape(-1)[0] += 1
                np.save(path, array)
                with self.assertRaisesRegex(ValueError, "checksum"):
                    load_mmd_reference(self.root / "a")
                path.write_bytes(original)
        load_mmd_reference(self.root / "a")
        metadata_path = self.root / "a" / "metadata.json"
        saved = metadata_path.read_text()
        for change in (lambda m: m["bandwidth"].update(sigma=m["bandwidth"]["sigma"] * 2),
                       lambda m: m["pca_reference"].update(fingerprint="0" * 64),
                       lambda m: m["pca_reference"].update(n_components=2)):
            metadata = json.loads(saved)
            change(metadata)
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_mmd_reference(self.root / "a")
        metadata_path.write_text(saved)
        load_mmd_reference(self.root / "a")
        with self.assertRaisesRegex(FileNotFoundError, "mmd-reference"):
            load_mmd_reference(self.root / "missing")

    def test_command_reads_the_validation_partition_only(self):
        cubes = ERA5Cubes(train=Unread(), validation=self.values[VALIDATION], test=Unread(),
                          train_timestamps=TIMES[TRAIN], validation_timestamps=TIMES[VALIDATION],
                          test_timestamps=TIMES[TEST], **NAMES)
        preparation = {"train_end_year": 2003, "validation_end_year": 2004, "test_start_year": 2005}
        command = ["mmd-reference", "--prepared-dir", "unused", "--pca-reference-dir", str(self.root / "pca"),
                   "--output-dir", str(self.root / "cli"), "--subset-size", "3", "--pca-components", str(COMPONENTS)]
        with patch.object(run_era5_stgan, "load_prepared", return_value=(cubes, None, preparation)), \
                patch.object(pca_module, "fit_pca", side_effect=AssertionError("the PCA was refitted")), \
                redirect_stdout(io.StringIO()) as stream:
            run_era5_stgan.main(command)
        printed = json.loads(stream.getvalue()[stream.getvalue().index("\n{\n") + 1:])
        reference = load_mmd_reference(self.root / "cli")
        self.assertEqual((printed["fingerprint"], printed["sigma"], printed["pca_reference"]),
                         (reference.fingerprint, reference.sigma, self.pca.fingerprint))
        self.assertEqual(reference.fingerprint, build(self.root / "direct", self.values, self.root / "pca").fingerprint)
        defaults = run_era5_stgan.parser().parse_args(
            ["mmd-reference", "--prepared-dir", "p", "--pca-reference-dir", "q", "--output-dir", "o"])
        self.assertEqual((defaults.pca_components, defaults.subsets, defaults.subset_size, defaults.subset_seed),
                         (100, 5, 512, 0))
        with patch.object(run_era5_stgan, "load_prepared", return_value=(
                cubes, None, {**preparation, "train_end_year": 2002})), self.assertRaisesRegex(ValueError, "2003"):
            run_era5_stgan.main(command[:6] + [str(self.root / "other")])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields()
        self.pca = build_pca(self.root / "pca", self.values)
        self.reference = build(self.root / "reference", self.values, self.root / "pca")

    def encoders(self):
        yield "convgru", {}
        if HAS_GRAPH:
            yield "gat", dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4)

    def fit(self, name, *, values=None, reference="reference", pca="pca", **overrides):
        values = self.values if values is None else values
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=3, batch_size=4, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=6, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, mc_samples=2, monitoring_timestamps=4, monitoring_feature_mmd_samples=16,
            checkpoint_path=self.root / name / "model.pt",
            pca_reference=None if pca is None else self.root / pca,
            mmd_reference=None if reference is None else self.root / reference)
        options.update(overrides)
        monitors, create, evaluate = [], MMDMonitor.__init__, MMDMonitor.evaluate

        def capture(monitor, *args, **kwargs):
            create(monitor, *args, **kwargs)
            monitors.append(monitor)

        def keep_model(monitor, model):
            monitor.last_model = copy.deepcopy(model)  # The weights this epoch was measured on.
            return evaluate(monitor, model)
        with redirect_stdout(io.StringIO()), patch.object(MMDMonitor, "__init__", capture), \
                patch.object(MMDMonitor, "evaluate", keep_model), \
                fit_and_score_stgan(values[TRAIN], values[TEST], **options) as result:
            return SimpleNamespace(metadata=result.metadata, scores=np.array(result.test_scores),
                std=np.array(result.anomaly_std), history=pd.read_csv(self.root / name / "training_history.csv"),
                monitor=monitors[0] if monitors else None,
                final=torch.load(self.root / name / "model.pt", weights_only=False),
                epoch=lambda n: torch.load(self.root / name / f"model_epoch_{n}.pt", weights_only=False))

    def reconstruction_features(self, model, monitor):
        """PCA features of G's reconstruction of every reference timestamp, recomputed field by field."""
        dataset, reference = monitor.dataset, monitor.reference
        per_target = len(dataset) // len(dataset.targets)
        targets = pd.DatetimeIndex(dataset.target_timestamps).as_unit("ns").asi8
        features = np.empty(reference.real_features.shape)
        model.eval()
        with torch.no_grad():
            for subset, sample in np.ndindex(reference.timestamps.shape):
                target = int(np.flatnonzero(targets == reference.timestamps[subset, sample])[0])
                batch = dataset.fetch_batch(target * per_target + np.arange(per_target))
                generated = model.generator(*batch[:4]).float().numpy()
                if generated.ndim == 4:  # ConvGRU: one patch per location; its centre is the location itself.
                    centre = generated.shape[-1] // 2
                    generated = generated[:, :, centre, centre]
                # The complete field [location, variable], flattened and projected by the PCA reference.
                features[subset, sample] = self.pca.transform(generated.reshape(dataset.n_locations, -1), COMPONENTS)
        return features

    def test_epoch_metrics_are_the_subset_mmd_of_the_reconstructions(self):
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                run = self.fit(encoder, **extra)
                monitor, reference, history = run.monitor, self.reference, run.history
                for column in METRICS + ("validation_pca_mmd_seconds",):
                    self.assertTrue(history[column].notna().all(), column)
                # The old diagnostic is still there, under its own name.
                self.assertIn("validation_discriminator_feature_mmd", history.columns)
                # The last epoch, recomputed from the final weights with the frozen reference.
                model, values = monitor.last_model, []

                def spy(x, y, sigma, **options):
                    values.append((np.array(x), np.array(y), sigma))
                    return multiscale_mmd2(x, y, sigma, **options)
                with patch.object(mmd_module, "multiscale_mmd2", side_effect=spy), \
                        patch.object(mmd_module, "median_sigma", side_effect=AssertionError("sigma was recomputed")), \
                        patch.object(pca_module, "fit_pca", side_effect=AssertionError("the PCA was refitted")):
                    result = monitor.evaluate(model)
                fake = self.reconstruction_features(model, monitor)
                self.assertEqual(len(values), 5)
                for subset, (x, y, sigma) in enumerate(values):
                    np.testing.assert_array_equal(x, reference.real_features[subset])  # The saved real features.
                    np.testing.assert_allclose(y, fake[subset], rtol=1e-6, atol=1e-8)
                    self.assertEqual((sigma, x.shape, y.shape), (reference.sigma, (3, COMPONENTS), (3, COMPONENTS)))
                per_subset = [explicit_mmd2(reference.real_features[subset], fake[subset],
                                            (reference.sigma / 2, reference.sigma, 2 * reference.sigma))
                              for subset in range(5)]
                np.testing.assert_allclose(monitor.subset_values, per_subset, rtol=1e-6, atol=1e-10)
                self.assertAlmostEqual(result["validation_pca_mmd_mean"], np.mean(per_subset), places=9)
                self.assertAlmostEqual(result["validation_pca_mmd_variance"], np.var(per_subset, ddof=1), places=9)
                self.assertGreater(result["validation_pca_mmd_variance"], 0)
                self.assertEqual(set(result), set(METRICS[:2]))
                self.assertAlmostEqual(history.validation_pca_mmd_mean.iloc[-1], result["validation_pca_mmd_mean"], places=9)
                self.assertAlmostEqual(history.validation_pca_mmd_variance.iloc[-1],
                                       result["validation_pca_mmd_variance"], places=9)
                # What the run records about the reference.
                recorded = run.metadata["pca_mmd"]
                self.assertEqual((recorded["fingerprint"], recorded["pca_reference"], recorded["sigma"],
                                  recorded["n_components"], recorded["n_subsets"], recorded["subset_size"],
                                  recorded["objective_window"]),
                                 (reference.fingerprint, self.pca.fingerprint, reference.sigma, COMPONENTS, 5, 3, 5))
                self.assertEqual(recorded["kernel_bandwidths"], [reference.sigma / 2, reference.sigma, 2 * reference.sigma])
                self.assertAlmostEqual(recorded["validation_pca_mmd_rolling_mean"],
                                       history.validation_pca_mmd_rolling_mean.iloc[-1], places=12)
                self.assertEqual(run.metadata["pca_reference"]["fingerprint"], self.pca.fingerprint)
                self.assertEqual((run.epoch(3)["mmd_reference"], run.epoch(3)["pca_reference"]),
                                 (reference.fingerprint, self.pca.fingerprint))
                # Other reconstructions: another MMD, with the same bandwidth and the same real features.
                noisy = copy.deepcopy(model)
                with torch.no_grad():
                    for parameter in noisy.generator.parameters():
                        parameter.add_(.3 * torch.randn_like(parameter))
                before = file_hashes(self.root / "reference"), file_hashes(self.root / "pca")
                values.clear()
                with patch.object(mmd_module, "multiscale_mmd2", side_effect=spy):
                    other = monitor.evaluate(noisy)
                self.assertNotAlmostEqual(other["validation_pca_mmd_mean"], result["validation_pca_mmd_mean"], places=6)
                self.assertEqual({sigma for _, _, sigma in values}, {reference.sigma})
                for subset, (x, _, _) in enumerate(values):
                    np.testing.assert_array_equal(x, reference.real_features[subset])
                self.assertEqual((monitor.reference.sigma, file_hashes(self.root / "reference"), file_hashes(self.root / "pca")),
                                 (reference.sigma, *before))

    def test_rolling_mean_uses_the_last_epochs_available(self):
        encoder, extra = next(self.encoders())
        for window, name, epochs in ((5, "default", 7), (2, "two", 3), (1, "one", 3)):
            with self.subTest(window=window):
                overrides = {} if name == "default" else {"mmd_objective_window": window}
                history = self.fit(f"rolling_{name}", epochs=epochs, **overrides, **extra).history
                means = history.validation_pca_mmd_mean.to_numpy()
                self.assertEqual(len(means), epochs)
                # Fewer epochs than the window: all of them; afterwards the last `window` only.
                expected = [means[max(0, index + 1 - window):index + 1].mean() for index in range(len(means))]
                np.testing.assert_allclose(history.validation_pca_mmd_rolling_mean, expected, rtol=1e-12)
        self.assertAlmostEqual(history.validation_pca_mmd_rolling_mean.iloc[0], history.validation_pca_mmd_mean.iloc[0])
        self.assertEqual(STGANCNNConfig().mmd_objective_window, 5)
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                STGANCNNConfig(mmd_objective_window=bad)

    def test_reference_is_shared_by_runs_and_never_depends_on_them(self):
        before = file_hashes(self.root / "reference"), file_hashes(self.root / "pca")
        runs = {}
        for encoder, extra in self.encoders():
            for seed, rate in ((20, 1e-3), (7, 3e-4)):
                runs[encoder, seed] = self.fit(f"{encoder}_{seed}", seed=seed, generator_learning_rate=rate, **extra)
        first = next(iter(runs.values()))
        for key, run in runs.items():
            with self.subTest(run=key):
                self.assertEqual(run.metadata["pca_mmd"], {**first.metadata["pca_mmd"],
                                 "validation_pca_mmd_rolling_mean": run.metadata["pca_mmd"]["validation_pca_mmd_rolling_mean"]})
                self.assertEqual(run.metadata["pca_mmd"]["fingerprint"], self.reference.fingerprint)
                self.assertEqual(run.metadata["pca_reference"], first.metadata["pca_reference"])
                # The same validation timestamps, the same real features, the same PCA and bandwidth.
                np.testing.assert_array_equal(run.monitor.reference.timestamps, self.reference.timestamps)
                np.testing.assert_array_equal(run.monitor.reference.real_features, self.reference.real_features)
                np.testing.assert_array_equal(run.monitor.pca.components, self.pca.components)
                np.testing.assert_array_equal(run.monitor.dataset.target_timestamps[run.monitor.targets].asi8,
                                              np.unique(self.reference.timestamps))
                self.assertTrue(run.monitor.real_checked)  # Its own observations reproduced the saved features.
                self.assertEqual(run.monitor.reference.sigma, self.reference.sigma)
        # Different runs, different reconstructions and different values; the reference files did not change.
        values = {round(run.metadata["pca_mmd"]["validation_pca_mmd_rolling_mean"], 12) for run in runs.values()}
        self.assertEqual(len(values), len(runs))
        self.assertEqual((file_hashes(self.root / "reference"), file_hashes(self.root / "pca")), before)

    def test_mmd_leaves_training_and_test_scores_unchanged_and_ignores_the_test_set(self):
        other_test = self.values.copy()
        other_test[TEST] = fields(seed=11)[TEST]
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                with_mmd = self.fit(f"{encoder}_mmd", **extra)
                without = self.fit(f"{encoder}_plain", reference=None, **extra)
                bare = self.fit(f"{encoder}_bare", reference=None, pca=None, monitoring_timestamps=0, **extra)
                self.assertIsNone(without.metadata["pca_mmd"])
                self.assertFalse([column for column in without.history.columns if "pca_mmd" in column])
                for other in (without, bare):
                    for checkpoint in (lambda run: run.final, lambda run: run.epoch(1), lambda run: run.epoch(3)):
                        for key, value in checkpoint(with_mmd)["model_state_dict"].items():
                            self.assertTrue(torch.equal(value, checkpoint(other)["model_state_dict"][key]), key)
                    np.testing.assert_array_equal(with_mmd.scores, other.scores)  # MC draws of the test included.
                    np.testing.assert_array_equal(with_mmd.std, other.std)
                    for key in ("score_normalization", "score_statistics", "validation_objective"):
                        expected, found = with_mmd.metadata[key], other.metadata[key]
                        if key == "validation_objective":
                            expected, found = ({k: v for k, v in item.items() if k != "seconds"} for item in (expected, found))
                        self.assertEqual(expected, found, key)
                    # Every training column: losses, their terms, optimizer steps.
                    for column in other.history.columns:
                        if not column.startswith("validation_") and not column.endswith("seconds"):
                            np.testing.assert_array_equal(with_mmd.history[column], other.history[column], err_msg=column)
                for column in without.history.columns:  # The other validation diagnostics too.
                    if not column.endswith("seconds"):
                        np.testing.assert_array_equal(with_mmd.history[column], without.history[column], err_msg=column)
                self.assertEqual(without.epoch(3)["mmd_reference"], None)
                # Another test set: other scores, the same MMD at every epoch.
                changed = self.fit(f"{encoder}_test", values=other_test, **extra)
                self.assertFalse(np.array_equal(changed.scores, with_mmd.scores))
                for column in METRICS:
                    np.testing.assert_array_equal(changed.history[column], with_mmd.history[column])

    def test_incompatible_references_and_resume(self):
        encoder, extra = next(self.encoders())
        # An MMD reference built on another PCA reference, or one whose observations are not this run's.
        other = self.values.copy()
        other[VALIDATION] = fields(seed=5)[VALIDATION]
        build_pca(self.root / "pca_again", self.values)  # The same PCA: accepted, whatever its directory.
        self.fit("same_pca", pca="pca_again", epochs=1, **extra)
        other_train = self.values.copy()
        other_train[TRAIN] = fields(seed=5)[TRAIN]
        build_pca(self.root / "other_pca", other_train)
        build(self.root / "on_other_pca", self.values, self.root / "other_pca")
        with self.assertRaisesRegex(ValueError, "built on another one"):
            self.fit("other_pca", reference="on_other_pca", **extra)
        build(self.root / "other_validation", other, self.root / "pca")
        with self.assertRaisesRegex(ValueError, "do not reproduce the MMD reference features"):
            self.fit("other_validation", reference="other_validation", **extra)
        with self.assertRaisesRegex(FileNotFoundError, "never built by a run"):
            self.fit("missing", reference="nowhere", **extra)
        with self.assertRaisesRegex(FileNotFoundError, "never fitted by a run"):
            self.fit("missing_pca", pca="nowhere", **extra)
        with self.assertRaisesRegex(ValueError, "needs the PCA reference"):
            self.fit("no_pca", pca=None, **extra)
        with self.assertRaisesRegex(ValueError, "validation period"):
            self.fit("no_validation", validation=None, validation_timestamps=None, **extra)
        with self.assertRaisesRegex(ValueError, "minmax"):
            self.fit("seasonal", normalization="seasonal", seasonal_window_days=1, **extra)
        shifted = build(self.root / "shifted", self.values, self.root / "pca", subset_seed=4)
        self.assertNotEqual(shifted.fingerprint, self.reference.fingerprint)
        # Resume: the rolling mean continues, on the same reference only.
        options = dict(dropout_enabled=False, mc_dropout_enabled=False, train_samples_per_epoch=None, **extra)
        full = self.fit("full", **options)
        self.fit("cut", epochs=2, **options)
        checkpoint = self.root / "cut" / "model_epoch_2.pt"
        resumed = self.fit("cut", resume_from=checkpoint, **options)
        for column in METRICS:  # Earlier epochs are read back from the history file: equal up to its rounding.
            np.testing.assert_allclose(resumed.history[column], full.history[column], rtol=1e-12, atol=0)
        self.assertEqual(len(resumed.history), 3)
        for reference in ("shifted", None):
            with self.assertRaisesRegex(ValueError, "mmd_reference"):
                self.fit("cut", resume_from=checkpoint, reference=reference, **options)


class InterfaceTests(unittest.TestCase):
    def test_sweep_objective_and_wandb_names(self):
        import yaml
        draft = yaml.safe_load((Path(__file__).resolve().parents[1] / "sweeps/stgan_bayes.draft.yaml").read_text())
        self.assertEqual(draft["metric"], {"name": OBJECTIVE, "goal": "minimize"})
        self.assertIs(validate_sweep(draft), draft)
        self.assertEqual(draft["parameters"]["mmd_objective_window"], {"value": 5})
        self.assertEqual({key for key, spec in draft["parameters"].items() if "value" not in spec},
                         {"generator_learning_rate", "discriminator_lr_ratio", "generator_reconstruction_weight",
                          "discriminator_generator_update_ratio"})

        class Run:
            def __init__(self):
                self.logs, self.summary = [], {}

            def define_metric(self, name, **kwargs):
                pass

            def log(self, values):
                self.logs.append(dict(values))

        run = Run()
        logger = RunLogger(run)
        record = {"epoch": 2, "seconds": 1., "generator_loss": 3., "validation_pca_mmd_mean": .25,
                  "validation_pca_mmd_variance": .01, "validation_pca_mmd_rolling_mean": .3,
                  "validation_pca_mmd_seconds": 2., "validation_discriminator_feature_mmd": .5}
        logger.epoch(record)
        logged = run.logs[0]
        self.assertEqual((logged["validation/pca_mmd_mean"], logged["validation/pca_mmd_variance"],
                          logged["validation/pca_mmd_rolling_mean"], logged["validation/discriminator_feature_mmd"]),
                         (.25, .01, .3, .5))
        self.assertIn(draft["metric"]["name"], logged)
        logger.result({"precision": "fp32", "pca_mmd": {"fingerprint": "abc", "n_components": 100, "sigma": 1.5},
                       "validation_objective": {"candidate_points": 4, "valid_points": 4, "excluded_points": 0,
                                                "mean": 1., "std": 1., "median": 1., "p95": 2., "median_plus_p95": 3.,
                                                "seconds": 1.}})
        self.assertEqual(run.summary["mmd_reference"], {"fingerprint": "abc", "n_components": 100, "sigma": 1.5})
        # The reconstruction statistic is still logged, as a diagnostic: it is not the objective.
        self.assertEqual(run.logs[-1]["validation/reconstruction_raw_median_plus_p95"], 3.)
        self.assertNotEqual(draft["metric"]["name"], "validation/reconstruction_raw_median_plus_p95")
        self.assertNotEqual(draft["metric"]["name"], "validation/discriminator_feature_mmd")

    def test_wandb_runner_passes_the_references_and_a_sweep_requires_both(self):
        from scripts.run_stgan_wandb import parse_args

        class FakeRun:
            id, url, entity, project = "run", "url", "entity", "project"

            def __init__(self, config, sweep_id):
                self.config, self.sweep_id, self.summary, self.logs = config, sweep_id, {}, []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def define_metric(self, name, **kwargs):
                pass

            def log(self, values):
                self.logs.append(values)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prepared").mkdir()
            base = ["--backend", "era5", "--prepared-dir", str(root / "prepared"), "--output-root", str(root / "runs"),
                    "--device", "cpu", "--wandb-mode", "offline"]
            pca, mmd = ["--pca-reference-dir", str(root / "pca")], ["--mmd-reference-dir", str(root / "mmd")]
            with patch.dict("os.environ", {"STGAN_MMD_REFERENCE_DIR": str(root / "from_environment")}):
                self.assertEqual(parse_args(base).mmd_reference_dir, root / "from_environment")
            with patch.dict("os.environ", {}, clear=False) as environment:
                environment.pop("STGAN_MMD_REFERENCE_DIR", None)
                environment.pop("STGAN_PCA_REFERENCE_DIR", None)
                self.assertIsNone(parse_args(base).mmd_reference_dir)
                only_pca, only_mmd, both, neither = (parse_args(base + extra) for extra in (pca, mmd, pca + mmd, []))
            self.assertTrue(default_config("era5").validation_holdout)
            cases = ((only_pca, "sweep", "STGAN_MMD_REFERENCE_DIR"), (only_mmd, "sweep", "STGAN_PCA_REFERENCE_DIR"),
                     (neither, "sweep", "REFERENCE_DIR"), (neither, None, None), (both, "sweep", None))
            for index, (args, sweep_id, error) in enumerate(cases):
                args.output_root = root / f"runs_{index}"
                with self.subTest(case=index), \
                        patch("wandb.init", side_effect=lambda **kwargs: FakeRun(kwargs["config"], sweep_id)), \
                        patch("scripts.run_era5_stgan.run_training",
                              return_value={"backend": {"precision": "fp32"}}) as train:
                    if error is None:
                        run_tracked(args)
                        self.assertEqual((train.call_args.kwargs["mmd_reference_dir"],
                                          train.call_args.kwargs["pca_reference_dir"]),
                                         (args.mmd_reference_dir, args.pca_reference_dir))
                        self.assertEqual(train.call_args.kwargs["config"].mmd_objective_window, 5)
                    else:
                        with self.assertRaisesRegex(ValueError, error):
                            run_tracked(args)
                        train.assert_not_called()

    def test_training_command_passes_the_reference(self):
        era5 = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        with patch.object(run_era5_stgan, "run_training", return_value={}) as train, redirect_stdout(io.StringIO()):
            run_era5_stgan.main(era5)
            self.assertEqual((train.call_args.kwargs["mmd_reference_dir"],
                              train.call_args.kwargs["config"].mmd_objective_window), (None, 5))
            run_era5_stgan.main(era5 + ["--pca-reference-dir", "pca", "--mmd-reference-dir", "reference",
                                        "--mmd-objective-window", "3"])
        self.assertEqual((train.call_args.kwargs["pca_reference_dir"], train.call_args.kwargs["mmd_reference_dir"],
                          train.call_args.kwargs["config"].mmd_objective_window), (Path("pca"), Path("reference"), 3))


if __name__ == "__main__":
    unittest.main()
