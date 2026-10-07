"""Fixed PCA feature space of complete fields: fitted once on training data, saved, loaded by every run.

The same file serves the ConvGRU and the GAT branch; graph cases run where the GAT exists.
"""
from contextlib import redirect_stdout
import hashlib
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
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection import stgan
from physiq_pv.anomaly_detection.stgan import STGAN, STGANCNNConfig, STGANWindowDataset, build_spatial_grid, fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan import pca_reference as module
from physiq_pv.anomaly_detection.stgan.pca_reference import (
    ARRAYS, MARKS, PLOT, THRESHOLDS, build_pca_reference, components_for_variance, fit_pca, load_pca_reference,
    model_features, normalise, plot_cumulative_variance)
from physiq_pv.anomaly_detection.stgan.pipeline import _feature_minmax
from physiq_pv.era5.data import ERA5Cubes
from physiq_pv.experiments.stgan_wandb import default_config, run_tracked
from scripts import run_era5_stgan

HAS_GRAPH = hasattr(stgan, "STGANGAT")


def layout(height, width):
    rows, cols = np.indices((height, width)).reshape(2, -1)
    return dict(feature_names=("a", "b"), location_names=tuple(f"r{r}_c{c}" for r, c in zip(rows, cols)),
                latitudes=45 - rows * .5, longitudes=7 + cols * .5)


def fields(steps, locations, seed=3):
    """Fields [time, location, 2 variables]: a few dominant spatial patterns plus noise of full rank."""
    rng = np.random.default_rng(seed)
    patterns = rng.normal(size=(3, locations, 2)) * np.array([4., 2., 1.])[:, None, None]
    return (np.einsum("tk,knf->tnf", rng.normal(size=(steps, 3)), patterns)
            + .3 * rng.normal(size=(steps, locations, 2))).astype(np.float32)


def slow_fields(steps, locations, seed=11):
    """Fields [time, location, 2 variables] whose variance decays slowly over many directions."""
    rng = np.random.default_rng(seed)
    weights = np.arange(1, locations * 2 + 1, dtype=np.float64) ** -.5
    return (rng.normal(size=(steps, locations * 2)) * rng.permutation(weights)).reshape(steps, locations, 2).astype(np.float32)


def smallest_reaching(cumulative, threshold):
    """The smallest number of leading components whose cumulative value reaches `threshold`, by plain search."""
    return next((number for number, value in enumerate(cumulative, start=1) if value >= threshold), None)


WIDE, SMALL = layout(6, 10), layout(3, 3)  # 120 and 18 values per flattened field.
LARGE = layout(18, 25)  # 900 values per flattened field: room for 300 components.
WIDE_TIMES = pd.date_range("2003-01-01", periods=140, freq="3h").as_unit("ns")
LARGE_TIMES = pd.date_range("2003-01-01", periods=900, freq="3h").as_unit("ns")
TIMES = pd.date_range("2003-12-28 12:00", periods=44, freq="3h").as_unit("ns")
TRAIN, VALIDATION, TEST = slice(0, 20), slice(20, 36), slice(36, 44)


def build(directory, values, times, names, **options):
    with redirect_stdout(io.StringIO()):
        return build_pca_reference(values, times, output_dir=directory, **names, **options)


def file_hashes(directory):
    return {name + ".npy": hashlib.sha256((Path(directory) / (name + ".npy")).read_bytes()).hexdigest()
            for name in ARRAYS}


class Unread:
    """A partition that fails on any access to its values."""
    def __getitem__(self, key):
        raise AssertionError("a partition other than the training one was read")
    __array__ = __len__ = __iter__ = __getitem__


class FitTests(unittest.TestCase):
    def test_exact_pca_keeps_the_leading_components_without_a_variance_cutoff(self):
        samples = fields(130, 60).reshape(130, 120)
        centred = samples.astype(np.float64) - samples.astype(np.float64).mean(axis=0)
        _, singular, right = np.linalg.svd(centred, full_matrices=False)
        ratio = singular ** 2 / (singular ** 2).sum()
        pca = fit_pca(samples.copy())
        # Three patterns already explain most of the variance: 100 components are kept all the same.
        self.assertGreater(ratio[:3].sum(), .8)
        self.assertEqual((len(pca["components"]), len(pca["explained_variance_ratio"])), (100, 100))
        np.testing.assert_allclose(pca["explained_variance_ratio"], ratio[:100], rtol=1e-5)
        np.testing.assert_allclose(pca["explained_variance"], singular[:100] ** 2 / 129, rtol=1e-5)
        np.testing.assert_allclose(pca["mean"], samples.mean(axis=0), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(pca["components"] @ pca["components"].T, np.eye(100), atol=1e-4)
        np.testing.assert_allclose(np.abs(pca["components"][:20] @ right[:20].T), np.eye(20), atol=1e-3)
        cumulative = np.cumsum(pca["explained_variance_ratio"])
        self.assertTrue((np.diff(cumulative) > 0).all() and cumulative[-1] <= 1 + 1e-12)
        self.assertTrue((np.diff(pca["explained_variance_ratio"]) <= 1e-12).all())
        self.assertEqual(len(fit_pca(samples.copy(), n_components=110)["components"]), 110)
        np.testing.assert_array_equal(fit_pca(samples.copy(), n_components=7)["components"], pca["components"][:7])
        np.testing.assert_allclose(fit_pca(samples.copy(), block=17)["components"], pca["components"], atol=1e-5)
        # Fewer components only when the rank of the data is lower.
        rank_five = (np.random.default_rng(0).normal(size=(40, 5)) @ np.random.default_rng(1).normal(size=(5, 120)))
        low = fit_pca(rank_five.astype(np.float32))
        self.assertEqual((len(low["components"]), low["rank"]), (5, 5))
        self.assertAlmostEqual(float(low["explained_variance_ratio"].sum()), 1., places=6)

    def test_three_hundred_components_extend_the_same_space(self):
        samples = slow_fields(340, 200).reshape(340, 400)
        centred = samples.astype(np.float64) - samples.astype(np.float64).mean(axis=0)
        singular = np.linalg.svd(centred, compute_uv=False)
        ratio = singular ** 2 / (singular ** 2).sum()
        pca, hundred = fit_pca(samples.copy(), n_components=300), fit_pca(samples.copy())
        self.assertEqual((len(pca["components"]), len(pca["explained_variance_ratio"]), pca["rank"]), (300, 300, 339))
        np.testing.assert_allclose(pca["explained_variance_ratio"], ratio[:300], rtol=1e-5)
        np.testing.assert_allclose(np.cumsum(pca["explained_variance_ratio"]), np.cumsum(ratio)[:300], rtol=1e-6)
        np.testing.assert_allclose(pca["explained_variance"], singular[:300] ** 2 / 339, rtol=1e-5)
        np.testing.assert_allclose(pca["components"] @ pca["components"].T, np.eye(300), atol=1e-4)
        # Asking for more components changes nothing of the first 100: same mean, directions and variance.
        np.testing.assert_array_equal(pca["mean"], hundred["mean"])
        np.testing.assert_array_equal(pca["explained_variance_ratio"][:100], hundred["explained_variance_ratio"])
        np.testing.assert_allclose(pca["components"][:100], hundred["components"], rtol=0, atol=1e-6)
        # No variance cutoff: 300 are kept although fewer already explain 95%, and more than 300 on request.
        self.assertLess(components_for_variance(pca["explained_variance_ratio"])["0.95"], 300)
        self.assertEqual(len(fit_pca(samples.copy(), n_components=320)["components"]), 320)

    def test_smallest_number_of_components_reaching_each_threshold(self):
        self.assertEqual(THRESHOLDS, (.80, .85, .90, .95))
        ratio = np.array([.60, .21, .05, .05, .02, .03])  # Cumulative: .60 .81 .86 .91 .93 .96
        self.assertEqual(components_for_variance(ratio), {"0.80": 2, "0.85": 3, "0.90": 4, "0.95": 6})
        # A threshold that the given components do not reach is reported as such, not replaced by the last one.
        self.assertEqual(components_for_variance(ratio[:5]), {"0.80": 2, "0.85": 3, "0.90": 4, "0.95": None})
        self.assertEqual(components_for_variance(ratio[:1]), {"0.80": None, "0.85": None, "0.90": None, "0.95": None})
        self.assertEqual(components_for_variance([.97, .01]), {"0.80": 1, "0.85": 1, "0.90": 1, "0.95": 1})
        self.assertEqual(components_for_variance(ratio, thresholds=(.5, .99)), {"0.50": 1, "0.99": None})
        slow = fit_pca(slow_fields(340, 200).reshape(340, 400), n_components=300)["explained_variance_ratio"]
        for kept in (300, 100, 40):
            cumulative = np.cumsum(slow[:kept])
            self.assertEqual(components_for_variance(slow[:kept]),
                             {f"{threshold:.2f}": smallest_reaching(cumulative, threshold) for threshold in THRESHOLDS})


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields(140, 60)

    def test_reference_is_fitted_on_training_fields_only_and_documents_itself(self):
        parameters = inspect.signature(build_pca_reference).parameters
        self.assertFalse([name for name in parameters if "valid" in name or "test" in name or "model" in name])
        self.assertEqual((parameters["pca_samples"].default, parameters["n_components"].default), (4096, 100))
        reference = build(self.root / "a", self.values, WIDE_TIMES, WIDE, pca_samples=130)
        metadata, pca = reference.metadata, reference.metadata["pca"]
        minimum, scale = _feature_minmax(self.values)
        np.testing.assert_array_equal(reference.minimum, minimum)
        np.testing.assert_array_equal(reference.scale, scale)
        self.assertEqual(metadata["normalization"]["minimum"], minimum.tolist())
        # 130 training fields, evenly spaced over the whole period, each flattened to 120 values.
        rows = np.linspace(0, 139, 130).round().astype(int)
        self.assertEqual((rows[0], rows[-1], len(set(rows))), (0, 139, 130))
        np.testing.assert_array_equal(np.load(self.root / "a" / "pca_fit_timestamps.npy"), WIDE_TIMES.asi8[rows])
        expected = fit_pca(normalise(self.values[rows], minimum, scale).reshape(130, 120))
        np.testing.assert_array_equal(reference.components.reshape(100, 120), expected["components"])
        np.testing.assert_array_equal(reference.mean.reshape(-1), expected["mean"])
        self.assertEqual((metadata["data"]["field_shape_height_width_channels"], metadata["data"]["complete_grid"],
                          metadata["data"]["flattened_dimension"], metadata["data"]["flatten_order"]),
                         ([6, 10, 2], True, 120, "location_then_variable"))
        self.assertEqual((pca["fit_on"], pca["fit_samples"], pca["requested_fit_samples"], pca["dimension"],
                          pca["stored_components"], pca["chosen_components"]), ("train", 130, 130, 120, 100, None))
        self.assertAlmostEqual(pca["fit_fraction_of_training"], 130 / 140)
        self.assertIn("evenly_spaced", pca["fit_selection"])
        self.assertEqual((metadata["validation_data_used"], metadata["test_data_used"], metadata["labels_used"],
                          metadata["model_used"]), (False,) * 4)
        # Explained variance: per component, cumulative and monotone, and reported at the marks that are stored.
        ratio = np.array(pca["explained_variance_ratio"])
        cumulative = np.array(pca["cumulative_explained_variance"])
        np.testing.assert_array_equal(ratio, reference.explained_variance_ratio)
        np.testing.assert_allclose(cumulative, np.cumsum(ratio), rtol=1e-12)
        np.testing.assert_allclose(reference.cumulative_explained_variance, cumulative, rtol=1e-12)
        self.assertEqual((len(ratio), bool((ratio > 0).all()), bool((np.diff(ratio) <= 0).all())), (100, True, True))
        self.assertTrue((np.diff(cumulative) > 0).all() and 0 < cumulative[0] < cumulative[-1] <= 1)
        self.assertEqual(MARKS, (50, 75, 100, 150, 200, 250, 300))
        marked = pca["cumulative_explained_variance_at"]
        self.assertEqual(marked, {str(mark): (float(np.cumsum(ratio)[mark - 1]) if mark <= 100 else None)
                                  for mark in MARKS})
        self.assertTrue(marked["50"] < marked["75"] < marked["100"])
        self.assertEqual(reference.summary()["cumulative_explained_variance_at"], marked)
        reached = pca["components_for_cumulative_explained_variance"]
        self.assertEqual(reached, {f"{threshold:.2f}": smallest_reaching(cumulative, threshold)
                                   for threshold in THRESHOLDS})
        self.assertEqual(reference.summary()["components_for_cumulative_explained_variance"], reached)
        self.assertIsNone(reference.summary()["chosen_components"])
        # The saved artefact and its plot.
        self.assertEqual(sorted(path.name for path in (self.root / "a").iterdir()),
                         sorted([name + ".npy" for name in ARRAYS] + ["metadata.json", PLOT]))
        self.assertGreater((self.root / "a" / PLOT).stat().st_size, 10000)
        self.assertEqual(plot_cumulative_variance(ratio, self.root / "again.png"),
                         {mark: marked[str(mark)] for mark in MARKS if mark <= 100})
        self.assertEqual(plot_cumulative_variance(ratio[:60], self.root / "short.png"), {50: marked["50"]})

    def test_three_hundred_components_go_to_a_new_reference_and_nothing_is_chosen(self):
        values = slow_fields(900, 450)
        hundred = build(self.root / "hundred", values, LARGE_TIMES, LARGE, pca_samples=880)
        before = file_hashes(self.root / "hundred"), (self.root / "hundred" / "metadata.json").read_bytes()
        # The existing reference is never overwritten: more components need a directory of their own.
        with self.assertRaisesRegex(ValueError, "new/empty"):
            build(self.root / "hundred", values, LARGE_TIMES, LARGE, pca_samples=880, n_components=300)
        stream = io.StringIO()
        with redirect_stdout(stream):
            reference = build_pca_reference(values, LARGE_TIMES, output_dir=self.root / "three_hundred", **LARGE,
                                            pca_samples=880, n_components=300)
        self.assertEqual((file_hashes(self.root / "hundred"), (self.root / "hundred" / "metadata.json").read_bytes()),
                         before)
        pca = reference.metadata["pca"]
        self.assertEqual((len(reference.components), pca["stored_components"], pca["requested_components"],
                          pca["chosen_components"], reference.summary()["chosen_components"]), (300, 300, 300, None, None))
        # Same training data, normalization, fit samples and method: the first 100 components are the old space.
        for key in ("data", "normalization", "sample"):
            self.assertEqual(reference.metadata[key], hundred.metadata[key])
        for key in ("method", "fit_on", "fit_samples", "requested_fit_samples", "fit_selection",
                    "fit_fraction_of_training", "dimension", "rank"):
            self.assertEqual(pca[key], hundred.metadata["pca"][key], key)
        for name in ("scaler_minimum", "scaler_scale", "pca_mean", "pca_fit_timestamps"):
            self.assertEqual(reference.metadata["files"][name + ".npy"], hundred.metadata["files"][name + ".npy"])
        np.testing.assert_array_equal(reference.explained_variance_ratio[:100], hundred.explained_variance_ratio)
        np.testing.assert_allclose(reference.components[:100], hundred.components, rtol=0, atol=1e-6)
        self.assertNotEqual(reference.fingerprint, hundred.fingerprint)
        raw = slow_fields(4, 450, seed=5)
        np.testing.assert_allclose(reference.transform_raw(raw, 100), hundred.transform_raw(raw), rtol=0, atol=1e-4)
        # Cumulative explained variance at every mark up to 300.
        cumulative = np.cumsum(np.load(self.root / "three_hundred" / "pca_explained_variance_ratio.npy"))
        np.testing.assert_allclose(pca["cumulative_explained_variance"], cumulative, rtol=1e-12)
        self.assertEqual(len(cumulative), 300)
        marked = pca["cumulative_explained_variance_at"]
        self.assertEqual(marked, {str(mark): float(cumulative[mark - 1]) for mark in MARKS})
        self.assertEqual(list(marked.values()), sorted(marked.values()))
        self.assertEqual({key: marked[key] for key in ("50", "75", "100")},
                         {key: hundred.metadata["pca"]["cumulative_explained_variance_at"][key] for key in ("50", "75", "100")})
        self.assertEqual(reference.summary()["cumulative_explained_variance_at"], marked)
        # Thresholds: the smallest number of components reaching each, and an explicit "not reached".
        reached = pca["components_for_cumulative_explained_variance"]
        self.assertEqual(reached, {f"{threshold:.2f}": smallest_reaching(cumulative, threshold) for threshold in THRESHOLDS})
        self.assertTrue(100 < reached["0.80"] < reached["0.85"] < reached["0.90"] <= 300)
        self.assertIsNone(reached["0.95"])
        self.assertLess(cumulative[reached["0.90"] - 2], .90)
        self.assertEqual(reference.summary()["components_for_cumulative_explained_variance"], reached)
        self.assertEqual(hundred.metadata["pca"]["components_for_cumulative_explained_variance"],
                         dict.fromkeys(reached))  # 100 components reach none of them here.
        printed = stream.getvalue()
        self.assertIn(f"cumulative explained variance >= 0.90: {reached['0.90']} components", printed)
        self.assertIn("cumulative explained variance >= 0.95: not reached within the 300 stored", printed)
        # The artefact: the same files, verified on load, and a plot that reaches 300 components.
        self.assertEqual(sorted(path.name for path in (self.root / "three_hundred").iterdir()),
                         sorted([name + ".npy" for name in ARRAYS] + ["metadata.json", PLOT]))
        self.assertEqual((reference.metadata["files"], len(reference.fingerprint)),
                         (file_hashes(self.root / "three_hundred"), 64))
        self.assertGreater((self.root / "three_hundred" / PLOT).stat().st_size, 10000)
        self.assertEqual(plot_cumulative_variance(reference.explained_variance_ratio, self.root / "again.png"),
                         {mark: marked[str(mark)] for mark in MARKS})
        np.testing.assert_array_equal(load_pca_reference(self.root / "three_hundred").components, reference.components)

    def test_hash_is_stable_and_an_existing_reference_is_loaded_not_refitted(self):
        first = build(self.root / "a", self.values, WIDE_TIMES, WIDE, pca_samples=130)
        again = build(self.root / "b", self.values, WIDE_TIMES, WIDE, pca_samples=130)
        self.assertEqual((first.fingerprint, file_hashes(self.root / "a")), (again.fingerprint, file_hashes(self.root / "b")))
        self.assertEqual((len(first.fingerprint), first.metadata["files"]), (64, file_hashes(self.root / "a")))
        with patch.object(module, "fit_pca", side_effect=AssertionError("refitted")):
            reloaded = load_pca_reference(self.root / "a")
            with self.assertRaisesRegex(ValueError, "never refitted"):
                build(self.root / "a", self.values, WIDE_TIMES, WIDE, pca_samples=130)
        self.assertEqual(reloaded.fingerprint, first.fingerprint)
        for name in ("minimum", "scale", "mean", "components", "explained_variance_ratio"):
            np.testing.assert_array_equal(getattr(reloaded, name), getattr(first, name))
        # Other training data, another number of fit samples: another reference.
        self.assertNotEqual(build(self.root / "c", fields(140, 60, seed=8), WIDE_TIMES, WIDE, pca_samples=130).fingerprint,
                            first.fingerprint)
        self.assertNotEqual(build(self.root / "d", self.values, WIDE_TIMES, WIDE, pca_samples=110).fingerprint,
                            first.fingerprint)
        # Altered files are rejected; a missing reference is an error that names the command.
        path = self.root / "a" / "pca_components.npy"
        original = path.read_bytes()
        changed = np.load(path)
        changed.reshape(-1)[0] += 1
        np.save(path, changed)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_pca_reference(self.root / "a")
        path.write_bytes(original)
        load_pca_reference(self.root / "a")
        with self.assertRaisesRegex(FileNotFoundError, "pca-reference"):
            load_pca_reference(self.root / "missing")

    def test_command_reads_the_training_partition_only(self):
        cubes = ERA5Cubes(train=self.values, validation=Unread(), test=Unread(), train_timestamps=WIDE_TIMES,
                          validation_timestamps=WIDE_TIMES[:0], test_timestamps=WIDE_TIMES[:0], **WIDE)
        preparation = {"train_end_year": 2003, "validation_end_year": 2004, "test_start_year": 2005}
        command = ["pca-reference", "--prepared-dir", "unused", "--output-dir", str(self.root / "cli"),
                   "--pca-samples", "130"]
        with patch.object(run_era5_stgan, "load_prepared", return_value=(cubes, None, preparation)), \
                redirect_stdout(io.StringIO()) as stream:
            run_era5_stgan.main(command)
        printed = json.loads(stream.getvalue()[stream.getvalue().index("\n{\n") + 1:])
        reference = load_pca_reference(self.root / "cli")
        self.assertEqual(printed["fingerprint"], reference.fingerprint)
        self.assertEqual(Path(printed["plot"]), reference.directory / PLOT)
        self.assertEqual(printed["cumulative_explained_variance_at"], reference.metadata["pca"]["cumulative_explained_variance_at"])
        self.assertEqual(printed["components_for_cumulative_explained_variance"],
                         reference.metadata["pca"]["components_for_cumulative_explained_variance"])
        self.assertIsNone(printed["chosen_components"])
        self.assertEqual(reference.fingerprint, build(self.root / "direct", self.values, WIDE_TIMES, WIDE, pca_samples=130).fingerprint)
        defaults = run_era5_stgan.parser().parse_args(["pca-reference", "--prepared-dir", "p", "--output-dir", "o"])
        self.assertEqual((defaults.pca_samples, defaults.pca_components), (4096, 100))
        more = run_era5_stgan.parser().parse_args(["pca-reference", "--prepared-dir", "p", "--output-dir", "o",
                                                   "--pca-components", "300"])
        self.assertEqual((more.pca_samples, more.pca_components), (4096, 300))
        with patch.object(run_era5_stgan, "load_prepared", return_value=(
                cubes, None, {**preparation, "train_end_year": 2002})), self.assertRaisesRegex(ValueError, "2003"):
            run_era5_stgan.main(command[:4] + [str(self.root / "other")])

    def test_one_transformation_for_any_field(self):
        reference = build(self.root / "a", self.values, WIDE_TIMES, WIDE, pca_samples=130)
        raw = fields(5, 60, seed=21)
        normalised = normalise(raw, reference.minimum, reference.scale)
        expected = ((normalised.reshape(5, 120).astype(np.float64) - reference.mean.reshape(-1))
                    @ reference.components.reshape(100, 120).astype(np.float64).T)
        np.testing.assert_allclose(reference.transform(normalised), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(reference.transform_raw(raw), reference.transform(normalised))
        np.testing.assert_allclose(reference.transform(normalised[0]), reference.transform(normalised)[0],
                                   rtol=1e-12, atol=1e-14)
        for kept in (50, 75, 100):  # The same leading features, whatever number is chosen later.
            np.testing.assert_allclose(reference.transform(normalised, kept), reference.transform(normalised)[:, :kept],
                                       rtol=1e-12, atol=1e-14)
        for bad in (dict(fields=normalised[:, :59]), dict(fields=normalised, n_components=101),
                    dict(fields=normalised, n_components=0)):
            with self.assertRaises(ValueError):
                reference.transform(**bad)


class Identity(torch.nn.Module):
    """A generator that returns the observation: its reconstruction is the observation itself."""
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Dropout(.5)

    def reconstructed_cells(self, recent, trend, mask, calendar, observed):
        if observed.ndim == 4:
            centre = observed.shape[-1] // 2
            cells = observed[:, :, centre, centre]
        else:
            cells = observed.reshape(-1, observed.shape[-1])
        return cells, cells.clone()


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.values = fields(44, 9)
        self.reference = build(Path(self.temp.name) / "reference", self.values[TRAIN], TIMES[TRAIN], SMALL, pca_samples=16)
        self.grid = build_spatial_grid(SMALL["latitudes"], SMALL["longitudes"], grid_crs="EPSG:4326", angular_spacing=.5,
                                       audit_knn=False)
        self.options = dict(feature_minimum=self.reference.minimum, feature_scale=self.reference.scale, recent_steps=1,
                            trend_steps=2, stride=1)
        self.targets = np.array([1, 4, 5, 9])

    def datasets(self):
        yield "window", STGANWindowDataset(self.values[VALIDATION], TIMES[VALIDATION], self.grid, **self.options)
        if HAS_GRAPH:
            yield "graph", stgan.STGANGraphDataset(self.values[VALIDATION], TIMES[VALIDATION], self.grid, **self.options)

    def models(self):
        sizes = dict(n_features=2, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1)
        yield "convgru", STGAN(**sizes), STGANWindowDataset(self.values[VALIDATION], TIMES[VALIDATION], self.grid, **self.options)
        if HAS_GRAPH:
            model = stgan.STGANGAT(**sizes, edge_index=stgan.grid_edge_index(self.grid.row_indices, self.grid.column_indices),
                                   node_indices=self.grid.node_indices, recent_steps=1, gat_hidden_dim=2, gat_heads=2,
                                   discriminator_chunk_size=4)
            yield "gat", model, stgan.STGANGraphDataset(self.values[VALIDATION], TIMES[VALIDATION], self.grid, **self.options)

    def test_same_input_gives_the_same_features_in_either_batch_layout(self):
        results = {}
        for name, dataset in self.datasets():
            with self.subTest(layout=name):
                raw = self.values[VALIDATION][dataset.targets[self.targets]]
                real, reconstructed = model_features(Identity(), dataset, self.reference, self.targets, batch_size=5)
                # The observation and an identical reconstruction: the very same features.
                np.testing.assert_array_equal(real, reconstructed)
                np.testing.assert_allclose(real, self.reference.transform_raw(raw), rtol=1e-9, atol=1e-9)
                self.assertEqual(real.shape, (4, len(self.reference.components)))
                few = model_features(Identity(), dataset, self.reference, self.targets, batch_size=64, n_components=3)[0]
                np.testing.assert_allclose(few, real[:, :3], rtol=1e-12, atol=1e-12)
                results[name] = real
        for real in results.values():
            np.testing.assert_allclose(real, results["window"], rtol=1e-12, atol=1e-12)

    def test_observations_and_reconstructions_of_a_model_use_the_same_transformation(self):
        for name, model, dataset in self.models():
            with self.subTest(model=name):
                model.train()
                rng = torch.get_rng_state()
                real, reconstructed = model_features(model, dataset, self.reference, self.targets, batch_size=7)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(all(part.training for part in model.modules()))
                per_target = len(dataset) // len(dataset.targets)
                generated = []
                model.eval()
                with torch.no_grad():
                    for target in self.targets:  # The complete reconstructed field of one timestamp.
                        batch = dataset.fetch_batch(target * per_target + np.arange(per_target))
                        output = model.generator(*batch[:4]).float().numpy()
                        if output.ndim == 4:
                            output = output[:, :, output.shape[-1] // 2, output.shape[-1] // 2]
                        generated.append(output.reshape(9, 2))
                raw = self.values[VALIDATION][dataset.targets[self.targets]]
                np.testing.assert_allclose(real, self.reference.transform_raw(raw), rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(reconstructed, self.reference.transform(np.stack(generated)), rtol=1e-6, atol=1e-7)
                self.assertFalse(np.allclose(real, reconstructed))
        with self.assertRaises(ValueError):
            model_features(Identity(), dataset, self.reference, np.array([1, 1]), batch_size=4)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields(44, 9)
        self.reference = build(self.root / "reference", self.values[TRAIN], TIMES[TRAIN], SMALL, pca_samples=16)

    def encoders(self):
        yield "convgru", {}
        if HAS_GRAPH:
            yield "gat", dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4)

    def fit(self, name, *, values=None, reference="reference", **overrides):
        values = self.values if values is None else values
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **SMALL,
            epochs=2, batch_size=4, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=6, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, mc_samples=2, monitoring_timestamps=4, monitoring_feature_mmd_samples=16,
            checkpoint_path=self.root / name / "model.pt",
            pca_reference=None if reference is None else self.root / reference)
        options.update(overrides)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(values[TRAIN], values[TEST], **options) as result:
            return SimpleNamespace(metadata=result.metadata, scores=np.array(result.test_scores),
                std=np.array(result.anomaly_std), history=pd.read_csv(self.root / name / "training_history.csv"),
                final=torch.load(self.root / name / "model.pt", weights_only=False),
                epoch=lambda n: torch.load(self.root / name / f"model_epoch_{n}.pt", weights_only=False))

    def test_every_run_loads_the_same_reference_and_nothing_else_changes(self):
        before = file_hashes(self.root / "reference")
        recorded = []
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                with patch.object(module, "fit_pca", side_effect=AssertionError("the PCA was refitted by a run")):
                    loaded = self.fit(f"{encoder}_pca", **extra)
                    other_seed = self.fit(f"{encoder}_seed", seed=7, generator_learning_rate=3e-4, **extra)
                plain = self.fit(f"{encoder}_plain", reference=None, **extra)
                summary = loaded.metadata["pca_reference"]
                self.assertEqual((summary["fingerprint"], summary["stored_components"], summary["chosen_components"]),
                                 (self.reference.fingerprint, len(self.reference.components), None))
                self.assertEqual(other_seed.metadata["pca_reference"], summary)
                self.assertEqual(loaded.epoch(2)["pca_reference"], self.reference.fingerprint)
                self.assertEqual((plain.metadata["pca_reference"], plain.epoch(2)["pca_reference"]), (None, None))
                recorded.append(summary)
                # Loading the reference changes nothing of the training, of the score or of the evaluation.
                for checkpoint in (lambda run: run.final, lambda run: run.epoch(1)):
                    for key, value in checkpoint(loaded)["model_state_dict"].items():
                        self.assertTrue(torch.equal(value, checkpoint(plain)["model_state_dict"][key]), key)
                np.testing.assert_array_equal(loaded.scores, plain.scores)
                np.testing.assert_array_equal(loaded.std, plain.std)
                self.assertEqual(list(loaded.history.columns), list(plain.history.columns))
                for column in loaded.history.columns:
                    if not column.endswith("seconds"):
                        np.testing.assert_array_equal(loaded.history[column], plain.history[column], err_msg=column)
                for key in ("score_normalization", "score_statistics", "splits"):
                    self.assertEqual(loaded.metadata[key], plain.metadata[key])
                # No MMD objective yet: no such metric, no such option.
                self.assertFalse([column for column in loaded.history.columns
                                  if "mmd" in column and "discriminator_feature" not in column])
        self.assertTrue(all(summary == recorded[0] for summary in recorded))  # ConvGRU and GAT: the same artefact.
        self.assertEqual(file_hashes(self.root / "reference"), before)
        self.assertFalse(hasattr(STGANCNNConfig(), "mmd_objective_window"))

    def test_missing_or_incompatible_reference_stops_the_run(self):
        encoder, extra = next(self.encoders())
        with self.assertRaisesRegex(FileNotFoundError, "never fitted by a run"):
            self.fit("missing", reference="nowhere", **extra)
        other = self.values.copy()
        other[TRAIN] = fields(44, 9, seed=5)[TRAIN]
        build(self.root / "other", other[TRAIN], TIMES[TRAIN], SMALL, pca_samples=16)
        with self.assertRaisesRegex(ValueError, "normalization"):
            self.fit("mismatch", reference="other", **extra)
        with self.assertRaisesRegex(ValueError, "minmax"):
            self.fit("seasonal", normalization="seasonal", seasonal_window_days=1, **extra)


class InterfaceTests(unittest.TestCase):
    def test_runners_pass_the_reference_and_a_sweep_member_requires_it(self):
        from scripts.run_stgan_wandb import parse_args
        import yaml
        draft = yaml.safe_load((Path(__file__).resolve().parents[1] / "sweeps/stgan_bayes.draft.yaml").read_text())
        self.assertEqual(draft["metric"], {"name": None, "goal": None})  # No objective is selected here.
        self.assertFalse([name for name in draft["parameters"] if "pca" in name or name.startswith("mmd")])
        era5 = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        with patch.object(run_era5_stgan, "run_training", return_value={}) as train, redirect_stdout(io.StringIO()):
            run_era5_stgan.main(era5)
            self.assertIsNone(train.call_args.kwargs["pca_reference_dir"])
            run_era5_stgan.main(era5 + ["--pca-reference-dir", "reference"])
        self.assertEqual(train.call_args.kwargs["pca_reference_dir"], Path("reference"))

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
            with patch.dict("os.environ", {"STGAN_PCA_REFERENCE_DIR": str(root / "from_environment")}):
                self.assertEqual(parse_args(base).pca_reference_dir, root / "from_environment")
            with patch.dict("os.environ", {}, clear=False) as environment:
                environment.pop("STGAN_PCA_REFERENCE_DIR", None)
                missing, given = parse_args(base), parse_args(base + ["--pca-reference-dir", str(root / "reference")])
            self.assertIsNone(missing.pca_reference_dir)
            self.assertTrue(default_config("era5").validation_holdout)
            result = {"backend": {"precision": "fp32", "pca_reference": {"fingerprint": "abc"}}}
            for index, (args, sweep_id, accepted) in enumerate(((missing, "sweep", False), (missing, None, True),
                                                                (given, "sweep", True))):
                args.output_root = root / f"runs_{index}"
                runs = []

                def start(sweep_id=sweep_id, **kwargs):
                    runs.append(FakeRun(kwargs["config"], sweep_id))
                    return runs[-1]
                with self.subTest(case=index), patch("wandb.init", side_effect=start), \
                        patch("scripts.run_era5_stgan.run_training", return_value=result) as train:
                    if accepted:
                        run_tracked(args)
                        self.assertEqual(train.call_args.kwargs["pca_reference_dir"], args.pca_reference_dir)
                        self.assertEqual(runs[0].summary["pca_reference"], {"fingerprint": "abc"})
                    else:
                        with self.assertRaisesRegex(ValueError, "STGAN_PCA_REFERENCE_DIR"):
                            run_tracked(args)
                        train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
