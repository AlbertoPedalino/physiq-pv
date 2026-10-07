"""Validation objective: mask-aware statistics of the MC-mean reconstruction error.

The same file serves the ConvGRU and the GAT branch; graph cases run where the GAT exists.
"""
from contextlib import redirect_stdout
import copy
import io
import math
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
from physiq_pv.anomaly_detection.stgan import STGANWindowDataset, build_spatial_grid, fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan import objective as objective_module
from physiq_pv.anomaly_detection.stgan.loading import make_loader
from physiq_pv.anomaly_detection.stgan.objective import COUNTS, STATISTICS, reconstruction_statistics, validation_objective
from physiq_pv.anomaly_detection.stgan.scoring import scoring_mode
from physiq_pv.experiments.stgan_wandb import RunLogger
from test_stgan_mc_dropout import small_model
from test_stgan_performance import fixture

HAS_GRAPH = hasattr(stgan, "STGANGAT")


def expected(values, valid):
    """The statistics by their definition, on the valid values only."""
    included = np.asarray(values)[valid].astype(np.float64)
    median, p95 = np.quantile(included, [.5, .95])
    return {"candidate_points": valid.size, "valid_points": included.size,
            "excluded_points": valid.size - included.size, "mean": included.mean(), "std": included.std(),
            "median": median, "p95": p95, "median_plus_p95": median + p95}


def tampered(dataset, change):
    """The same dataset, with `change` applied to every batch it serves."""
    class Tampered(type(dataset)):
        def fetch_batch(self, indices):
            return change(*super().fetch_batch(indices))
    clone = copy.copy(dataset)
    clone.__class__ = Tampered
    return clone


def huge_in_absent_cells(recent, trend, mask, calendar, observed, *rest):
    absent = mask == 0
    return (torch.where(absent[:, None], 1e6, recent), trend, mask, calendar,
            torch.where(absent, 1e6, observed), *rest)


def without_location(index):
    def change(recent, trend, mask, calendar, observed, positions, locations):
        mask = mask.clone()
        mask[locations == index] = 0  # Not one cell left in the mask of these samples.
        return recent, trend, mask, calendar, observed, positions, locations
    return change


def objective(model, dataset, *, mc=True, **options):
    with redirect_stdout(io.StringIO()):
        result = validation_objective(model, dataset, batch_size=5, device="cpu", mc_dropout_enabled=mc,
                                      mc_samples=3, **options)
    result.pop("seconds")
    return result


def draws_by_hand(model, dataset, *, mc=True):
    """Reconstruction error of every draw [draw,time,location] and validity [time,location]."""
    samples = 3 if mc else 1
    shape = (len(dataset.targets), dataset.n_locations)
    errors, valid = np.empty((samples, *shape)), np.empty(shape, dtype=bool)
    with scoring_mode(model, mc), torch.no_grad():
        for batch in make_loader(dataset, batch_size=5, shuffle=False, device="cpu"):
            time, location = batch[-2].numpy().reshape(-1), batch[-1].numpy().reshape(-1)
            errors[:, time, location] = model.score_draws(*batch[:5], samples)[:, :, 0].numpy()
            valid[time, location] = model.reconstruction_valid(batch[2]).numpy()
    return errors, valid


class StatisticsTests(unittest.TestCase):
    def check(self, result, reference, rtol=1e-12):
        for name in COUNTS:
            self.assertEqual(result[name], reference[name], name)
        for name in STATISTICS:
            np.testing.assert_allclose(result[name], reference[name], rtol=rtol, atol=0, err_msg=name)

    def test_only_valid_points_enter_and_what_sits_in_invalid_slots_is_irrelevant(self):
        rng = np.random.default_rng(0)
        values = rng.lognormal(size=(40, 9)).astype(np.float32)
        valid = rng.random(values.shape) > .2
        valid[:, 4] = False  # One cell is never valid.
        base = reconstruction_statistics(values, valid)
        self.check(base, expected(values, valid))
        self.assertEqual((base["candidate_points"], base["valid_points"] + base["excluded_points"]), (360, 360))
        self.assertGreater(base["excluded_points"], 40)
        self.assertEqual(base["median_plus_p95"], base["median"] + base["p95"])
        self.assertEqual(tuple(base), COUNTS + STATISTICS)
        for filler in (0., 1e30, -7., np.nan, np.inf):
            other = values.copy()
            other[~valid] = filler
            self.assertEqual(reconstruction_statistics(other, valid), base, filler)
        self.check(reconstruction_statistics(values, valid, chunk_size=7), base)  # Chunks only bound memory.
        changed = values.copy()
        changed[np.argwhere(valid)[0][0], np.argwhere(valid)[0][1]] += 1e3  # A valid value does count.
        self.assertNotEqual(reconstruction_statistics(changed, valid)["mean"], base["mean"])

    def test_a_fully_invalid_point_is_excluded_and_not_a_zero(self):
        values = np.arange(1., 21., dtype=np.float32).reshape(4, 5)
        valid = np.ones_like(values, dtype=bool)
        valid[:, 0] = False
        values[:, 0] = 0.  # What a mask without cells gives to the reconstruction error.
        result = reconstruction_statistics(values, valid)
        self.check(result, expected(values, valid))
        self.assertEqual((result["candidate_points"], result["valid_points"], result["excluded_points"]), (20, 16, 4))
        as_zeros = expected(values, np.ones_like(valid))  # The zeros would pull every statistic down.
        for name in STATISTICS:
            self.assertNotAlmostEqual(result[name], as_zeros[name], places=3, msg=name)
        self.assertGreater(result["median"], as_zeros["median"])

    def test_included_values_must_be_finite_and_exist(self):
        values, valid = np.ones((3, 4), np.float32), np.ones((3, 4), bool)
        for bad in (np.nan, np.inf, -np.inf):
            broken = values.copy()
            broken[1, 2] = bad
            with self.assertRaises(FloatingPointError):
                reconstruction_statistics(broken, valid)
        with self.assertRaisesRegex(ValueError, "No valid"):
            reconstruction_statistics(values, np.zeros_like(valid))
        for wrong in (valid[:2], valid.astype(np.float32)):
            with self.assertRaisesRegex(ValueError, "boolean"):
                reconstruction_statistics(values, wrong)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
    check = StatisticsTests.check

    def test_mc_mean_of_the_draws_with_one_mask_and_no_trace_left(self):
        dataset, model = fixture(), small_model(dropout_enabled=True).train()
        rng = torch.get_rng_state()
        result = objective(model, dataset)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))  # The draws that follow are untouched.
        self.assertTrue(all(module.training for module in model.modules()))
        self.assertEqual((result["timestamps"], result["locations"], result["effective_mc_samples"]),
                         (len(dataset.targets), 9, 3))
        self.assertEqual((result["start"], result["end"]),
                         (str(dataset.target_timestamps[0]), str(dataset.target_timestamps[-1])))
        errors, valid = draws_by_hand(model, dataset)  # From the same random state: the same draws.
        self.assertTrue(valid.all())
        self.assertFalse(np.array_equal(errors[0], errors[1]))
        self.check(result, expected(errors.mean(axis=0).astype(np.float32), valid), rtol=1e-9)
        single = expected(errors[0].astype(np.float32), valid)  # One draw is another distribution.
        self.assertNotAlmostEqual(result["median_plus_p95"], single["median_plus_p95"], places=6)
        # Without MC Dropout: one deterministic pass.
        plain = objective(model, dataset, mc=False)
        errors, valid = draws_by_hand(model, dataset, mc=False)
        self.assertEqual((plain["effective_mc_samples"], errors.shape[0]), (1, 1))
        self.check(plain, expected(errors[0].astype(np.float32), valid), rtol=1e-9)

    def test_values_in_masked_cells_do_not_change_the_statistics(self):
        dataset, model = fixture(), small_model(dropout_enabled=True)
        mask = dataset.fetch_batch(np.arange(len(dataset)))[2]
        self.assertTrue(bool((mask == 0).any()) and bool(mask.flatten(1).any(dim=1).all()))
        base = objective(model, dataset)
        loud = tampered(dataset, huge_in_absent_cells)
        served = loud.fetch_batch(np.arange(len(dataset)))
        self.assertEqual(float(served[0].max()), 1e6)
        self.assertEqual(float(served[4].max()), 1e6)
        self.assertEqual(objective(model, loud), base)
        self.assertEqual(base["excluded_points"], 0)

    def test_a_point_without_valid_cells_is_excluded_from_the_model_statistics(self):
        dataset, model = fixture(), small_model(dropout_enabled=False)
        base = objective(model, dataset, mc=False)
        holed = tampered(dataset, without_location(4))
        result = objective(model, holed, mc=False)
        errors, valid = draws_by_hand(model, holed, mc=False)
        self.assertEqual((~valid).sum(axis=0).tolist(), [0, 0, 0, 0, len(dataset.targets), 0, 0, 0, 0])
        self.assertTrue((errors[0][~valid] == 0).all())  # The value that must not enter the distribution.
        self.check(result, expected(errors[0].astype(np.float32), valid), rtol=1e-9)
        self.assertEqual((result["candidate_points"], result["excluded_points"]),
                         (base["candidate_points"], len(dataset.targets)))
        as_zeros = expected(errors[0].astype(np.float32), np.ones_like(valid))
        self.assertGreater(result["median"], as_zeros["median"])
        self.assertNotAlmostEqual(result["mean"], as_zeros["mean"], places=6)

    @unittest.skipUnless(HAS_GRAPH, "The GAT exists in the GAT branch")
    def test_graph_model_follows_the_same_rule(self):
        from test_stgan_gat import fixture as graph_fixture
        dataset, model = graph_fixture()
        result = objective(model, dataset)
        errors, valid = draws_by_hand(model, dataset)
        self.assertTrue(valid.all())
        self.check(result, expected(errors.mean(axis=0).astype(np.float32), valid), rtol=1e-9)
        self.assertEqual((result["candidate_points"], result["excluded_points"]), (len(dataset.targets) * 12, 0))
        # A node whose patch has no cell: its error is the zero of an empty mask, and it is left out.
        model.node_indices[5] = -1
        result = objective(model, dataset, mc=False)
        errors, valid = draws_by_hand(model, dataset, mc=False)
        self.assertEqual(np.flatnonzero((~valid).all(axis=0)).tolist(), [5])
        self.assertTrue((errors[0][~valid] == 0).all())
        self.check(result, expected(errors[0].astype(np.float32), valid), rtol=1e-9)
        self.assertEqual(result["excluded_points"], len(dataset.targets))


class Stub(torch.nn.Module):
    """A model whose error and validity depend on the observed target cell only, in either batch layout."""
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Dropout(.5)

    def score_draws(self, recent, trend, mask, calendar, observed, samples, *, share_history=True):
        if observed.ndim == 4:  # ConvGRU batches: [sample, feature, patch row, patch column]
            centre = observed.shape[-1] // 2
            cells = observed[:, :, centre, centre]
        else:  # Graph batches: [timestamp, node, feature]
            cells = observed.reshape(-1, observed.shape[-1])
        self.valid = cells[:, 1] > -.5
        error = torch.where(self.valid, cells[:, 0].square(), torch.nan)
        return torch.stack([error * (draw + 1) for draw in range(samples)])[..., None]

    def reconstruction_valid(self, mask):
        return self.valid


class LayoutTests(unittest.TestCase):
    def test_window_and_graph_batches_give_the_same_statistics_on_the_same_valid_data(self):
        rows, cols = np.indices((3, 4)).reshape(2, -1)
        grid = build_spatial_grid(45 - rows * .5, 7 + cols * .5, grid_crs="EPSG:4326", angular_spacing=.5,
                                  audit_knn=False)
        data = np.random.default_rng(4).normal(size=(14, 12, 2)).astype(np.float32)
        data[:, 5, 1] = -9.  # One cell is invalid at every timestamp.
        times = pd.date_range("2003-01-01", periods=14, freq="3h").as_unit("ns")
        options = dict(feature_minimum=np.zeros(2), feature_scale=np.ones(2), recent_steps=1, trend_steps=4, stride=1)
        datasets = {"window": STGANWindowDataset(data, times, grid, **options)}
        if HAS_GRAPH:
            datasets["graph"] = stgan.STGANGraphDataset(data, times, grid, **options)
        results = {name: objective(Stub(), dataset) for name, dataset in datasets.items()}
        window = datasets["window"]
        cells = window._normalise(data[window.targets])
        valid = cells[..., 1] > -.5
        reference = expected((np.square(cells[..., 0]) * 2).astype(np.float32), valid)  # Mean of 1, 2, 3 draws.
        self.assertEqual((reference["candidate_points"], int((~valid).all(axis=0).sum())), (len(window.targets) * 12, 1))
        self.assertGreater(reference["excluded_points"], len(window.targets))
        for name, result in results.items():
            with self.subTest(layout=name):
                StatisticsTests.check(self, result, reference, rtol=1e-6)
                self.assertEqual((result["timestamps"], result["locations"]), (len(window.targets), 12))
        for result in results.values():
            self.assertEqual(result, results["window"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.times = pd.date_range("2002-12-29", periods=24, freq="3h")

    def encoders(self):
        yield "convgru", {}
        if HAS_GRAPH:
            yield "gat", dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4)

    def fit(self, name, *, holdout=True, **overrides):
        values = np.random.default_rng(3).normal(size=(24, 9, 2)).astype(np.float32)
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        splits = dict(validation=values[10:18], validation_timestamps=self.times[10:18]) if holdout else {}
        train = slice(0, 10) if holdout else slice(0, 18)
        options = dict(train_timestamps=self.times[train], test_timestamps=self.times[18:],
            location_names=tuple(map(str, range(9))), feature_names=("a", "b"),
            latitudes=45 - rows * .5, longitudes=7 + cols * .5,
            epochs=1, batch_size=4, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=6, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, mc_samples=3, monitoring_timestamps=4, monitoring_feature_mmd_samples=16,
            checkpoint_path=self.root / name / "model.pt", **splits)
        options.update(overrides)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(values[train], values[18:], **options) as result:
            return SimpleNamespace(metadata=result.metadata, scores=np.array(result.test_scores),
                std=np.array(result.anomaly_std), generator=np.array(result.test_generator_scores),
                weights=torch.load(self.root / name / "model.pt", weights_only=False)["model_state_dict"])

    def test_complete_validation_is_measured_once_after_training(self):
        reported = {}
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                calls = []
                def record(model, dataset, **options):
                    calls.append((dataset, options))
                    return validation_objective(model, dataset, **options)
                with patch("physiq_pv.anomaly_detection.stgan.pipeline.validation_objective", side_effect=record):
                    run = self.fit(encoder, **extra)
                (dataset, options), = calls
                # Every validation timestamp, not the monitoring subset, with the MC protocol of the scoring.
                self.assertTrue(dataset.target_timestamps.equals(self.times[10:18]))
                self.assertEqual((options["mc_dropout_enabled"], options["mc_samples"]), (True, 3))
                result = run.metadata["validation_objective"]
                self.assertEqual((result["split"], result["start"], result["end"], result["timestamps"],
                                  result["locations"], result["effective_mc_samples"]),
                                 ("validation", str(self.times[10]), str(self.times[17]), 8, 9, 3))
                self.assertEqual((result["candidate_points"], result["valid_points"], result["excluded_points"]),
                                 (72, 72, 0))
                self.assertEqual(result["median_plus_p95"], result["median"] + result["p95"])
                self.assertTrue(all(math.isfinite(result[name]) and result[name] > 0 for name in STATISTICS))
                self.assertGreaterEqual(result["p95"], result["median"])
                self.assertEqual(run.metadata["monitoring"]["subset"]["n_timestamps"], 4)
                reported[encoder] = {name: result[name] for name in
                                     (*COUNTS, "split", "start", "end", "timestamps", "locations", "validity")}
                plain = self.fit(encoder + "_plain", dropout_enabled=False, mc_dropout_enabled=False, **extra)
                self.assertEqual(plain.metadata["validation_objective"]["effective_mc_samples"], 1)
                with patch("physiq_pv.anomaly_detection.stgan.pipeline.validation_objective") as unused:
                    merged = self.fit(encoder + "_merged", holdout=False, **extra)
                unused.assert_not_called()
                self.assertIsNone(merged.metadata["validation_objective"])
        for points in reported.values():  # The same validation points and the same rule in every architecture.
            self.assertEqual(points, reported["convgru"])

    def test_objective_leaves_weights_and_test_scores_unchanged(self):
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                measured = self.fit(encoder + "_measured", **extra)
                with patch("physiq_pv.anomaly_detection.stgan.pipeline.validation_objective", return_value=None):
                    skipped = self.fit(encoder + "_skipped", **extra)
                self.assertIsNotNone(measured.metadata["validation_objective"])
                self.assertIsNone(skipped.metadata["validation_objective"])
                for name in ("scores", "std", "generator"):  # MC draws of the test included.
                    np.testing.assert_array_equal(getattr(measured, name), getattr(skipped, name), err_msg=name)
                self.assertGreater(float(measured.std.max()), 0)
                for key, value in measured.weights.items():
                    self.assertTrue(torch.equal(value, skipped.weights[key]), key)
                for key in ("score_normalization", "score_statistics"):
                    self.assertEqual(measured.metadata[key], skipped.metadata[key])


class InterfaceTests(unittest.TestCase):
    def test_wandb_names_and_unset_sweep_metric(self):
        class Run:
            def __init__(self):
                self.logs, self.summary = [], {}

            def define_metric(self, name, **kwargs):
                pass

            def log(self, values):
                self.logs.append(dict(values))

        result = {"candidate_points": 72, "valid_points": 70, "excluded_points": 2, "mean": .3, "std": .1,
                  "median": .25, "p95": .5, "median_plus_p95": .75, "seconds": 4.}
        names = {"validation/reconstruction_raw_mean": .3, "validation/reconstruction_raw_std": .1,
                 "validation/reconstruction_raw_median": .25, "validation/reconstruction_raw_p95": .5,
                 "validation/reconstruction_raw_median_plus_p95": .75,
                 "validation/objective_candidate_points": 72, "validation/objective_valid_points": 70,
                 "validation/objective_excluded_points": 2, "validation/objective_seconds": 4.}
        run = Run()
        RunLogger(run).result({"precision": "fp32", "validation_objective": result})
        self.assertEqual(run.logs, [names])
        run = Run()
        logger = RunLogger(run)
        logger.epoch({"epoch": 6, "seconds": 1., "generator_loss": 2.})
        logger.result({"precision": "fp32", "validation_objective": result})
        self.assertEqual(run.logs[-1], {**names, "epoch": 6})
        run = Run()
        RunLogger(run).result({"precision": "fp32", "validation_objective": None})
        self.assertEqual(run.logs, [])
        import yaml
        draft = yaml.safe_load((Path(__file__).resolve().parents[1] / "sweeps/stgan_bayes.draft.yaml").read_text())
        self.assertEqual(draft["metric"], {"name": None, "goal": None})  # The objective is logged, not selected.
        self.assertTrue(hasattr(objective_module, "preserved_rng"))


if __name__ == "__main__":
    unittest.main()
