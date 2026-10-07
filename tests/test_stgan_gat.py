"""Synthetic global GAT contracts; no real ERA5 training."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import binary_cross_entropy as bce
from torch.utils._python_dispatch import TorchDispatchMode

from physiq_pv.anomaly_detection.stgan import (
    STGANGAT, STGANGATConfig, STGANGraphDataset, STGANWindowDataset,
    build_spatial_grid, grid_edge_index, TwoLayerGAT, ConvGRU,
    fit_and_score_stgan, load_stgan_checkpoint, masked_cell_mean)
from physiq_pv.anomaly_detection.stgan.scoring import (
    scoring_mode, score_components, normalize_mc_scores, fit_calibration_ranges)
from physiq_pv.anomaly_detection.stgan.training import gan_train_step
from physiq_pv.era5.cube import CubeGrid, score_cube_chunks
from physiq_pv.era5.events import EventConfig, process_events


def fixture(height=3, width=4, *, recent=1, trend=4, indices=None, steps=12):
    rows, cols = np.indices((height, width)).reshape(2, -1)
    if indices is not None:
        rows, cols = rows[indices], cols[indices]
    grid = build_spatial_grid(45-rows*.5, 7+cols*.5, grid_crs="EPSG:4326",
                              angular_spacing=.5, audit_knn=False)
    data = np.random.default_rng(7).normal(size=(steps, len(rows), 15)).astype(np.float32)
    times = pd.date_range("2004-12-30", periods=steps, freq="3h").as_unit("ns")
    dataset = STGANGraphDataset(data, times, grid, feature_minimum=np.zeros(15),
        feature_scale=np.ones(15), recent_steps=recent, trend_steps=trend, stride=1)
    model = STGANGAT(n_features=15, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=2,
        edge_index=grid_edge_index(grid.row_indices, grid.column_indices), node_indices=grid.node_indices,
        recent_steps=recent, gat_hidden_dim=3, gat_heads=2, discriminator_chunk_size=5)
    return dataset, model


class GATTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_edges_corners_borders_holes_permutation_no_wrap(self):
        rows, cols = np.indices((3, 4)).reshape(2, -1)
        edge = grid_edge_index(rows, cols).numpy()
        self.assertEqual(edge.shape, (2, (3*3-2)*(3*4-2)))
        self.assertEqual(len(set(map(tuple, edge.T))), edge.shape[1])
        degrees = np.bincount(edge[1])
        self.assertEqual(degrees[0], 4)
        self.assertEqual(degrees[1], 6)
        self.assertEqual(degrees[5], 9)
        self.assertNotIn((3, 4), set(map(tuple, edge.T)))
        chosen = np.array([11, 0, 1, 4, 8, 6, 10])
        r, c = rows[chosen], cols[chosen]
        actual = set(map(tuple, grid_edge_index(r, c).numpy().T))
        expected = {(i, j) for i in range(len(r)) for j in range(len(r))
                    if max(abs(r[i]-r[j]), abs(c[i]-c[j])) <= 1}
        self.assertEqual(actual, expected)
        self.assertTrue(all((i, i) in actual for i in range(len(r))))
        self.assertEqual(grid_edge_index([0, 5], [0, 5]).tolist(), [[0, 1], [0, 1]])

    def test_cyclic_time_encoding_is_local_per_node_and_annual_shared(self):
        from physiq_pv.anomaly_detection.stgan.data import annual_phase_features, cyclic_time_features
        from scripts.run_era5_stgan import parse_args
        times = pd.DatetimeIndex(["2005-01-01 00:00", "2005-01-01 12:00", "2005-07-02 12:00"])
        hours = np.array([0., 12., 12.])
        # Rows: timestamps; columns: longitudes 0, 90E, 15W (0h, +6h, -1h).
        features = cyclic_time_features(hours[:, None], annual_phase_features(times)[:, None],
                                        np.array([0., 90., -15.]) / 15)
        self.assertEqual(features.shape, (3, 3, 4))
        np.testing.assert_allclose(features[1, :, :2], [[0, -1], [-1, 0], [np.sin(np.pi*11/12), np.cos(np.pi*11/12)]], atol=1e-6)
        np.testing.assert_allclose(features[0, 0], [0, 1, 0, 1], atol=1e-6)
        np.testing.assert_array_equal(features[:, 0, 2:], features[:, 2, 2:])
        np.testing.assert_allclose(features[2, 0, 2:], [0, -1], atol=1e-2)
        dataset, _ = fixture()
        longitudes = 7 + dataset.grid.column_indices * 15.  # One hour of solar time per column.
        options = dict(feature_minimum=np.zeros(15), feature_scale=np.ones(15), recent_steps=1,
                       trend_steps=4, stride=1, time_encoding="cyclic", longitudes=longitudes)
        graph = STGANGraphDataset(dataset.data, dataset.timestamps, dataset.grid, **options)
        patches = STGANWindowDataset(dataset.data, dataset.timestamps, dataset.grid, **options)
        calendar = graph.fetch_batch([0, 3])[3]
        self.assertEqual(tuple(calendar.shape), (2, 12, 4))
        self.assertFalse(torch.allclose(calendar[:, 0, :2], calendar[:, 1, :2]))
        torch.testing.assert_close(calendar[:, 0], calendar[:, 4])  # Same column, same local time.
        torch.testing.assert_close(calendar[..., 2:], calendar[:, :1, 2:].expand(-1, 12, -1))
        torch.testing.assert_close(patches.fetch_batch(np.arange(12, 24))[3], graph.fetch_batch([1])[3][0])
        torch.testing.assert_close(patches[17][3], graph.fetch_batch([1])[3][0, 5])
        self.assertEqual((graph.time_feature_size, dataset.time_feature_size), (4, 31))
        grid = dataset.grid
        model = STGANGAT(n_features=15, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=2,
            edge_index=grid_edge_index(grid.row_indices, grid.column_indices), node_indices=grid.node_indices,
            gat_hidden_dim=3, gat_heads=2, time_feature_size=4).eval()
        recent, trend, mask, calendar, *_ = graph.fetch_batch([0, 3])
        changed = calendar.clone()
        changed[:, 5] += 1
        with torch.no_grad():
            delta = (model.generator(recent, trend, mask, changed)
                     - model.generator(recent, trend, mask, calendar)).abs().sum(dim=(0, 2))
        self.assertGreater(float(delta[5]), 0)
        self.assertEqual(float(delta.sum() - delta[5]), 0)
        with self.assertRaises(ValueError):
            STGANGATConfig(time_encoding="weekly")
        with self.assertRaises(ValueError):
            STGANGraphDataset(dataset.data, dataset.timestamps, grid, **{**options, "longitudes": None})
        base = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        self.assertEqual((parse_args(base).time_encoding, parse_args(base + ["--time-encoding", "cyclic"]).time_encoding),
                         ("onehot", "cyclic"))

    def test_cyclic_time_encoding_pipeline_for_gat_and_convgru(self):
        dataset, _ = fixture(3, 3, steps=14)
        data = dataset.data
        options = dict(train_timestamps=dataset.timestamps[:11], test_timestamps=dataset.timestamps[11:],
            location_names=tuple(map(str, range(9))), feature_names=tuple(f"f{i}" for i in range(15)),
            latitudes=45-dataset.grid.row_indices*.5, longitudes=7+dataset.grid.column_indices*.5,
            epochs=1, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1, trend_steps=4,
            grid_crs="EPSG:4326", angular_grid_spacing=.5, grid_audit_knn=False, timestep_hours=3,
            device="cpu", mc_samples=2, score_mode="paper", time_encoding="cyclic")
        for encoder, extra in (("gat", dict(gat_hidden_dim=3, gat_heads=2)), ("convgru", dict(batch_size=8))):
            root = self.root/encoder
            with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(data[:11], data[11:],
                    checkpoint_path=root/"model.pt", spatial_encoder=encoder, **options, **extra) as result:
                self.assertEqual(result.test_scores.shape, (3, 9))
                self.assertTrue(np.isfinite(result.test_scores).all())
                self.assertEqual(result.metadata["time_encoding"], "cyclic")
                self.assertEqual(result.metadata["time_feature_size"], 4)
                self.assertFalse(result.metadata["paper_alignment"]["reference_hyperparameters"])
            _, payload = load_stgan_checkpoint(root/"model.pt")
            self.assertEqual((payload["time_encoding"], payload["model_config"]["time_feature_size"]), ("cyclic", 4))
            # A one-hot run cannot continue from a cyclic epoch checkpoint.
            with self.assertRaisesRegex(ValueError, "time_encoding"), contextlib.redirect_stdout(io.StringIO()):
                fit_and_score_stgan(data[:11], data[11:], checkpoint_path=root/"model.pt", spatial_encoder=encoder,
                    resume_from=root/"model_epoch_1.pt", **{**options, "time_encoding": "onehot"}, **extra)
        # Checkpoints written before time_encoding: annual_cycle=False is the one-hot encoding.
        def onehot(**extra):
            with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(data[:11], data[11:],
                    checkpoint_path=self.root/"legacy/model.pt", spatial_encoder="gat", gat_hidden_dim=3,
                    gat_heads=2, **{**options, "time_encoding": "onehot", **extra}) as result:
                return result.metadata["resume"]
        onehot()
        checkpoint = self.root/"legacy/model_epoch_1.pt"
        legacy = torch.load(checkpoint, weights_only=False)
        del legacy["time_encoding"]
        torch.save({**legacy, "annual_cycle": False}, checkpoint)
        self.assertEqual(onehot(epochs=2, resume_from=checkpoint)["completed_epochs"], 1)
        with self.assertRaisesRegex(ValueError, "time_encoding"):
            onehot(epochs=2, resume_from=checkpoint, time_encoding="cyclic")
        torch.save({**legacy, "annual_cycle": True}, checkpoint)
        with self.assertRaisesRegex(ValueError, "time_encoding"):
            onehot(epochs=2, resume_from=checkpoint)

    def test_seasonal_climatology_matches_direct_statistics_and_removes_cycles(self):
        from physiq_pv.anomaly_detection.stgan.seasonal import fit_seasonal_climatology, seasonal_memmap
        times = pd.date_range("2001-01-01", "2004-12-31 21:00", freq="3h").as_unit("ns")
        rng = np.random.default_rng(5)
        year = 2 * np.pi * (times.dayofyear.to_numpy() - 1) / 365.25
        hour = 2 * np.pi * times.hour.to_numpy() / 24
        # Location-specific annual and diurnal cycles plus noise; feature 2 is constant.
        amplitude = np.array([1., 2., 3.])[None, :, None]
        data = (1e5 + amplitude * (3 * np.sin(year) + 2 * np.cos(hour))[:, None, None]
                + rng.normal(size=(len(times), 3, 3)) * np.array([1., 2., 0.]))
        data[..., 2] = 7.
        data = data.astype(np.float64)
        climatology = fit_seasonal_climatology(data, times, self.root/"cache", window_days=15)
        try:
            self.assertEqual(climatology.mean.shape, (8, 366, 3, 3))
            self.assertEqual(sorted(p.name for p in (self.root/"cache").iterdir()),
                             ["seasonal_mean.npy", "seasonal_std.npy"])
            # 40th day of year, 09:00, with a circular +-15 day window.
            for day in (40, 3, 360):
                offsets = (times.dayofyear.to_numpy() - 1 - day + 183) % 366 - 183
                window = (np.abs(offsets) <= 15) & (times.hour == 9)
                chosen = data[window]
                np.testing.assert_allclose(climatology.mean[3, day], chosen.mean(axis=0), rtol=1e-6)
                # Spread is measured around each sample's own smoothed daily mean.
                residual = chosen - climatology.mean[3][times.dayofyear.to_numpy()[window] - 1]
                np.testing.assert_allclose(climatology.std[3, day, :, :2],
                                           np.sqrt((residual**2).mean(axis=0))[:, :2], rtol=1e-3)
            self.assertEqual(climatology.metadata["window_days_each_side"], 15)
            self.assertGreaterEqual(climatology.metadata["min_samples_per_bin"], 31)
            standardised = np.array(seasonal_memmap(data, times, climatology, self.root/"cache/train.npy"))
            self.assertEqual(standardised.dtype, np.float32)
            self.assertTrue(np.isfinite(standardised).all())
            np.testing.assert_array_equal(standardised[..., 2], 0)
            # Cycles are gone: every location has ~zero mean and ~unit spread in every month and hour.
            for month, slot in ((1, 0), (7, 4), (10, 7)):
                chosen = np.asarray(standardised[(times.month == month) & (times.hour == slot * 3)])[..., :2]
                np.testing.assert_allclose(chosen.mean(axis=0), 0, atol=.35)
                np.testing.assert_allclose(chosen.std(axis=0), 1, atol=.35)
            raw_spread = data[..., 0].std(axis=0)
            self.assertGreater(raw_spread[2] / raw_spread[0], 2)
            # Unseen period: same bins, statistics untouched; a wrong time of day is rejected.
            later = pd.date_range("2005-03-01", periods=8, freq="3h").as_unit("ns")
            shifted = np.array(seasonal_memmap(data[:8] + 50, later, climatology, self.root/"cache/test.npy"))
            self.assertGreater(float(np.asarray(shifted)[..., 0].min()), 5)
            with self.assertRaisesRegex(ValueError, "time of day"):
                seasonal_memmap(data[:8], later + pd.Timedelta(hours=1), climatology, self.root/"cache/bad.npy")
        finally:
            climatology.close()
        with self.assertRaisesRegex(ValueError, "Too few"):
            fit_seasonal_climatology(data[:16], times[:16], self.root/"short", window_days=0)

    def test_seasonal_normalization_pipeline_resume_and_cli(self):
        from scripts.run_era5_stgan import parse_args
        times = pd.date_range("2003-01-01", periods=8 * 365 + 20, freq="3h").as_unit("ns")
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        phase = 2 * np.pi * (times.dayofyear.to_numpy() / 365 + times.hour.to_numpy() / 24)
        data = (np.sin(phase)[:, None, None] * (1 + rows)[None, :, None]
                + np.random.default_rng(9).normal(size=(len(times), 9, 15)) * .1).astype(np.float32)
        split = 8 * 365
        options = dict(train_timestamps=times[:split], test_timestamps=times[split:],
            location_names=tuple(map(str, range(9))), feature_names=tuple(f"f{i}" for i in range(15)),
            latitudes=45-rows*.5, longitudes=7+cols*.5, spatial_encoder="gat",
            hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1, gat_hidden_dim=3, gat_heads=2,
            trend_steps=4, grid_crs="EPSG:4326", angular_grid_spacing=.5, grid_audit_knn=False,
            timestep_hours=3, device="cpu", mc_samples=2, score_mode="paper",
            train_samples_per_epoch=6, time_encoding="cyclic", normalization="seasonal")
        root = self.root/"seasonal"
        with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(data[:split], data[split:],
                checkpoint_path=root/"model.pt", epochs=1, **options) as result:
            self.assertEqual(result.test_scores.shape, (20, 9))
            self.assertTrue(np.isfinite(result.test_scores).all())
            metadata = result.metadata
        self.assertEqual(metadata["normalization"], "training_only_seasonal_standardisation_then_feature_minmax")
        self.assertEqual(metadata["seasonal_normalization"]["window_days_each_side"], 15)
        self.assertEqual(metadata["seasonal_normalization"]["fit_period"]["end"], str(times[split-1]))
        self.assertFalse(metadata["runtime"]["normalized_disk_cache"])
        self.assertFalse(metadata["paper_alignment"]["reference_hyperparameters"])
        self.assertTrue((root/"normalized_cache/seasonal_mean.npy").is_file())
        cached = np.load(root/"normalized_cache/train.npy", mmap_mode="r")
        self.assertLess(abs(float(np.asarray(cached).mean())), .05)
        del cached
        _, payload = load_stgan_checkpoint(root/"model.pt")
        self.assertTrue(payload["normalization"]["kind"].startswith("training_only_seasonal"))
        self.assertEqual(payload["normalization"]["seasonal"]["day_bins"], 366)
        with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(data[:split], data[split:],
                checkpoint_path=root/"model.pt", epochs=2, resume_from=root/"model_epoch_1.pt", **options) as result:
            self.assertEqual(result.metadata["resume"]["completed_epochs"], 1)
        with self.assertRaisesRegex(ValueError, "normalization"), contextlib.redirect_stdout(io.StringIO()):
            fit_and_score_stgan(data[:split], data[split:], checkpoint_path=root/"model.pt", epochs=2,
                resume_from=root/"model_epoch_1.pt", **{**options, "normalization": "minmax"})
        with self.assertRaisesRegex(ValueError, "checkpoint_path"), contextlib.redirect_stdout(io.StringIO()):
            fit_and_score_stgan(data[:split], data[split:], epochs=1, **options)
        with self.assertRaises(ValueError):
            STGANGATConfig(normalization="zscore")
        base = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        args = parse_args(base + ["--normalization", "seasonal", "--seasonal-window-days", "10"])
        self.assertEqual((parse_args(base).normalization, args.normalization, args.seasonal_window_days),
                         ("minmax", "seasonal", 10))

    def test_config_cli_and_two_layers(self):
        from scripts.run_era5_stgan import parse_args
        args = parse_args(["train", "--prepared-dir", "unused", "--output-dir", "unused"])
        self.assertEqual((args.spatial_encoder, args.batch_size, args.score_batch_size), ("gat", 1, 1))
        config = STGANGATConfig()
        self.assertEqual((config.gat_layers, config.dropout_p, config.mc_samples, config.save_raw_mc), (2, .2, 20, False))
        for kwargs in ({"gat_layers": 1}, {"gat_layers": 3}, {"gat_layers": True},
                       {"gat_heads": 0}, {"gat_hidden_dim": -1}, {"discriminator_chunk_size": 0}):
            with self.assertRaises(ValueError):
                STGANGATConfig(**kwargs)

    def test_cli_backend_batch_defaults_and_explicit_overrides(self):
        from scripts.run_era5_stgan import parser, parse_args
        required = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        raw = parser().parse_args(required)
        self.assertIsNone(raw.batch_size)
        self.assertIsNone(raw.score_batch_size)
        for backend, defaults in (("gat", (1, 1)), ("convgru", (256, 1024))):
            for overrides, expected in (
                    ([], defaults),
                    (["--batch-size", "7"], (7, defaults[1])),
                    (["--score-batch-size", "11"], (defaults[0], 11)),
                    (["--batch-size", "7", "--score-batch-size", "11"], (7, 11))):
                with self.subTest(backend=backend, overrides=overrides):
                    args = parse_args(required + ["--spatial-encoder", backend] + overrides)
                    self.assertEqual((args.batch_size, args.score_batch_size), expected)

    def test_trend_last_owns_only_last_state_storage(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                _, model = fixture(trend=56, steps=58)
                generator = model.generator.to(device)
                values = torch.randn(5, 56, 15, device=device)
                sequences = []
                hook = generator.trend_encoder.register_forward_hook(
                    lambda module, args, output: sequences.append(output[0]))
                try:
                    with torch.no_grad():
                        last = generator._trend_last(values)
                finally:
                    hook.remove()
                sequence = sequences[0]
                self.assertIsNone(last._base)
                self.assertNotEqual(last.untyped_storage().data_ptr(), sequence.untyped_storage().data_ptr())
                self.assertEqual(last.untyped_storage().nbytes(), last.numel() * last.element_size())
                self.assertLess(last.untyped_storage().nbytes(), sequence.untyped_storage().nbytes())
                torch.testing.assert_close(last, sequence[:, -1], rtol=0, atol=0)

    def test_trend_copy_and_chunking_preserve_outputs_and_gradients(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for chunk_size in (2, 256):  # Includes a partial last chunk and the unchunked path.
                with self.subTest(device=device, chunk_size=chunk_size):
                    _, model = fixture(trend=56, steps=58)
                    generator = model.generator.to(device)
                    generator.trend_chunk_size = chunk_size
                    reference = copy.deepcopy(generator.trend_encoder)
                    values = torch.randn(5, 56, 15, device=device, requires_grad=True)
                    reference_values = values.detach().clone().requires_grad_(True)
                    actual = generator.encode_trend(values)
                    expected = reference(reference_values)[0][:, -1]
                    torch.testing.assert_close(actual, expected)
                    weights = torch.randn_like(actual)
                    (actual * weights).sum().backward()
                    (expected * weights).sum().backward()
                    torch.testing.assert_close(values.grad, reference_values.grad, atol=1e-6, rtol=1e-4)
                    for (name, parameter), other in zip(generator.trend_encoder.named_parameters(), reference.parameters()):
                        with self.subTest(parameter=name):
                            self.assertIsNotNone(parameter.grad)
                            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=1e-4)

    def test_global_shapes_temporal_forward_and_gradients(self):
        for recent in (1, 3):
            dataset, model = fixture(recent=recent)
            batch = dataset.fetch_batch([0, 2])
            self.assertEqual(batch[0].shape, (2, recent, 12, 15))
            self.assertEqual(batch[1].shape, (2, 12, 4, 15))
            self.assertEqual(len(model.generator.recent_encoder.layers), 2)
            self.assertIsInstance(model.generator.recent_temporal, ConvGRU)
            self.assertTrue(all(m.kernel_size == (1, 1) for m in model.generator.recent_temporal.modules()
                                if isinstance(m, torch.nn.Conv2d)))
            predicted = model.generator(*batch[:4])
            self.assertEqual(predicted.shape, (2, 12, 15))
            predicted.square().mean().backward()
            for name, parameter in model.generator.named_parameters():
                with self.subTest(recent=recent, parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    gradient = float(parameter.grad.abs().sum())
                    if recent == 1 and name.startswith("recent_temporal.") and ".reset." in name:
                        # The reset gate multiplies the zero initial state.
                        self.assertEqual(gradient, 0)
                    else:
                        self.assertGreater(gradient, 0)

    def test_exact_two_hop_receptive_field(self):
        rows, cols = np.indices((7, 7)).reshape(2, -1)
        encoder = TwoLayerGAT(15, 4, 3, 2, grid_edge_index(rows, cols))
        values = torch.randn(1, 49, 15, requires_grad=True)
        encoder(values)[0, 24].sum().backward()
        influence = values.grad.abs().sum(-1)[0]
        distance = np.maximum(abs(rows-3), abs(cols-3))
        self.assertTrue(torch.all(influence[distance <= 2] > 0))
        self.assertTrue(torch.all(influence[distance > 2] == 0))

    def test_sparse_allocation_forward_and_backward(self):
        nodes = 143
        class NoDenseAdjacency(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                output = func(*args, **(kwargs or {}))
                def check(value):
                    if isinstance(value, torch.Tensor):
                        self_test.assertFalse(any(a == b == nodes for a, b in zip(value.shape, value.shape[1:])),
                                              f"Dense adjacency in {func}: {value.shape}")
                    elif isinstance(value, (tuple, list)):
                        for item in value:
                            check(item)
                check(output)
                return output
        self_test = self
        with NoDenseAdjacency():
            dataset, model = fixture(11, 13)
            self.assertLessEqual(model.generator.recent_encoder.edge_index.shape[1], 9 * nodes)
            model.generator(*dataset.fetch_batch([0])[:4]).square().mean().backward()

    def test_discriminator_patch_adapter_matches_existing_dataset(self):
        graph, model = fixture(indices=np.array([11, 0, 1, 4, 8, 6, 10]))
        legacy = STGANWindowDataset(graph.data, graph.timestamps, graph.grid,
            feature_minimum=graph.minimum, feature_scale=graph.scale,
            recent_steps=1, trend_steps=4, stride=1)
        recent, _, _, _, observed, *_ = graph.fetch_batch([0, 2])
        reference = legacy.fetch_batch([p*graph.n_locations+n for p in (0, 2) for n in range(graph.n_locations)])
        patches = list(model.patch_batches(recent, observed, observed))
        for column, expected in ((1, reference[0]), (2, reference[4]), (4, reference[2])):
            torch.testing.assert_close(torch.cat([p[column] for p in patches]), expected, check_dtype=False)
        for scalar, batched in zip(graph[2], graph.fetch_batch([2])):
            torch.testing.assert_close(scalar, batched[0])

    def test_chunked_adam_update_matches_unchunked_reference_and_two_g_calls(self):
        dataset, chunked = fixture(recent=2)
        chunked.generator.spatial_dropout.p = 0
        chunked.generator.temporal_dropout.p = 0
        chunked.generator.fusion_dropout.p = 0
        reference = copy.deepcopy(chunked)
        reference.discriminator_chunk_size = 10000
        chunked.generator.trend_chunk_size = 5
        batch = dataset.fetch_batch([0, 1])[:5]
        recent, trend, mask, calendar, observed = batch
        go = torch.optim.Adam(chunked.generator.parameters(), lr=.001)
        do = torch.optim.Adam(chunked.discriminator.parameters(), lr=.001)
        seen = []
        def observe(stage, values):
            if stage == "D_backward":
                self.assertTrue(all(p.grad is None for p in chunked.generator.parameters()))
            seen.append(stage)
        with patch.object(chunked.generator, "forward", wraps=chunked.generator.forward) as call:
            losses = gan_train_step(chunked, batch, go, do, observe=observe)
        self.assertEqual(call.call_count, 2)
        rg = torch.optim.Adam(reference.generator.parameters(), lr=.001)
        rd = torch.optim.Adam(reference.discriminator.parameters(), lr=.001)
        with torch.no_grad():
            predicted = reference.generator(recent, trend, mask, calendar)
        _, history, real_patch, fake_patch, valid = next(reference.patch_batches(recent, observed, predicted))
        real, fake = reference.discriminator.score_pair(history, real_patch, fake_patch, valid)
        dloss = .5 * (bce(real, torch.zeros_like(real)) + bce(fake, torch.ones_like(fake)))
        dloss.backward()
        rd.step()
        for p in reference.discriminator.parameters():
            p.requires_grad_(False)
        predicted = reference.generator(recent, trend, mask, calendar)
        _, history, real_patch, fake_patch, valid = next(reference.patch_batches(recent, observed, predicted))
        fake = reference.discriminator.score_current(reference.discriminator.encode_history(history, valid), fake_patch, valid)
        gloss = 500 * masked_cell_mean((fake_patch-real_patch).square(), valid).mean() + bce(fake, torch.zeros_like(fake))
        gloss.backward()
        rg.step()
        torch.testing.assert_close(losses[0], gloss.detach())
        torch.testing.assert_close(losses[1], dloss.detach())
        for (name, actual), expected in zip(chunked.named_parameters(), reference.parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(actual.grad, expected.grad, atol=1e-5, rtol=2e-4)

    def test_mc_stochastic_mode_restore_and_memmap(self):
        dataset, model = fixture(steps=7)
        initial = [m.training for m in model.modules()]
        with scoring_mode(model, True):
            batch = dataset.fetch_batch([0])
            first, second = model.generator(*batch[:4]), model.generator(*batch[:4])
            self.assertFalse(torch.equal(first, second))
            self.assertFalse(model.discriminator.training)
            self.assertFalse(model.generator.trend_encoder.training)
        self.assertEqual(initial, [m.training for m in model.modules()])
        outputs = []
        for backend in ("memory", "memmap"):
            torch.manual_seed(8)
            with patch.object(model.generator, "encode", wraps=model.generator.encode) as encode, \
                 patch.object(model.generator, "decode", wraps=model.generator.decode) as decode:
                g, d, f, store = score_components(model, dataset, batch_size=2, device="cpu", n_features=15,
                    mc_dropout_enabled=True, mc_samples=20, storage=backend, output_dir=self.root/backend)
            try:
                # Two batches: the encoders run once per batch, the dropouts once per draw.
                self.assertEqual((encode.call_count, decode.call_count), (2, 40))
                self.assertEqual(g.shape, (20, 3, 12))
                self.assertEqual(f.shape, (3, 12, 15))
                mean, std, *_ = normalize_mc_scores(g, d, store, normalization=fit_calibration_ranges(g, d), chunk_size=7)
                self.assertTrue((std > 0).all())
                outputs.append((mean.copy(), std.copy()))
                store.discard_temporary_raw()
                if backend == "memmap":
                    self.assertFalse(list((self.root/backend).glob(".mc_raw_*")))
            finally:
                store.close()
        for a, b in zip(*outputs):
            np.testing.assert_array_equal(a, b)

    def test_shared_mc_scoring_equals_repeated_full_draws(self):
        for recent in (1, 3):
            dataset, model = fixture(recent=recent, steps=9)
            batch = dataset.fetch_batch([0, 2])[:5]
            with scoring_mode(model, True), torch.no_grad():
                # encode + decode is forward, dropout masks included.
                torch.manual_seed(5)
                reference = model.generator(*batch[:4])
                torch.manual_seed(5)
                split = model.generator.decode(*model.generator.encode(*batch[:4]), batch[2])
                self.assertTrue(torch.equal(reference, split))
                for share_history in (True, False):
                    torch.manual_seed(11)
                    repeated = torch.stack([model.score_draw(*batch, share_history=share_history)
                                            for _ in range(6)])
                    after = torch.rand(3)
                    torch.manual_seed(11)
                    shared = model.score_draws(*batch, 6, share_history=share_history)
                    self.assertEqual(shared.shape, (6, 24, 17))
                    self.assertTrue(torch.equal(shared, repeated), (recent, share_history))
                    self.assertFalse(torch.equal(shared[0], shared[1]))
                    # The RNG is left exactly where six full draws leave it.
                    self.assertTrue(torch.equal(after, torch.rand(3)))
            # The whole scoring pass: same raw draws and features as the per-draw loop.
            outputs = []
            for shared_work in (True, False):
                def per_draw(*args, share_history=True):
                    return torch.stack([model.score_draw(*args[:5], share_history=share_history)
                                        for _ in range(args[5])])
                torch.manual_seed(3)
                with contextlib.redirect_stdout(io.StringIO()), \
                     patch.object(model, "score_draws", model.score_draws if shared_work else per_draw):
                    g, d, f, store = score_components(model, dataset, batch_size=2, device="cpu",
                        n_features=15, mc_dropout_enabled=True, mc_samples=5, storage="memory")
                outputs.append((g.copy(), d.copy(), f.copy()))
                store.close()
            for a, b in zip(*outputs):
                np.testing.assert_array_equal(a, b)

    def test_components_and_scoring_match_patch_formula(self):
        dataset, model = fixture(recent=2)
        batch = dataset.fetch_batch([0, 1])[:5]
        model.eval()
        with torch.no_grad():
            predicted, real, fake, errors = model.components(*batch)
            packed = model.score_draw(*batch)
            other = model.score_draw(*batch, share_history=False)
        self.assertEqual(predicted.shape, (2, 12, 15))
        self.assertEqual(real.shape, (2, 12, 1))
        torch.testing.assert_close(packed, other)
        torch.testing.assert_close(packed[:, 1], (real-fake).flatten())
        torch.testing.assert_close(packed[:, 2:], errors.flatten(0, 1))
        reference = []
        for _, _, actual, prediction, valid in model.patch_batches(batch[0], batch[4], predicted):
            reference.append(masked_cell_mean((prediction-actual).square(), valid))
        torch.testing.assert_close(packed[:, 0], torch.cat(reference))

    def test_mc_disabled_or_dropout_zero_is_deterministic(self):
        dataset, model = fixture(steps=5)
        for dropout_p, mc in ((0, True), (.2, False)):
            for module in model.generator.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = dropout_p
            g, d, _, store = score_components(model, dataset, batch_size=1, device="cpu", n_features=15,
                mc_dropout_enabled=mc, mc_samples=3)
            try:
                _, std, *_ = normalize_mc_scores(g, d, store, normalization=fit_calibration_ranges(g, d))
                np.testing.assert_array_equal(std, np.zeros((1, 12)))
            finally:
                store.close()

    def test_memmap_spawn_loading_preserves_global_windows(self):
        from physiq_pv.anomaly_detection.stgan.loading import make_loader, close_loader
        dataset, _ = fixture(steps=6)
        path = self.root/"data.npy"
        np.save(path, dataset.data)
        dataset.data = np.load(path, mmap_mode="r")
        loader = make_loader(dataset, batch_size=2, num_workers=1, persistent_workers=True)
        try:
            actual = next(iter(loader))
            expected = dataset.fetch_batch([0, 1])
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
        finally:
            close_loader(loader)
            dataset.data._mmap.close()

    def test_pipeline_calibration_checkpoint_cubes_events(self):
        dataset, _ = fixture(3, 3, steps=14)
        data = dataset.data
        options = dict(calibration=data[8:11], calibration_timestamps=dataset.timestamps[8:11],
            train_timestamps=dataset.timestamps[:8], test_timestamps=dataset.timestamps[11:],
            location_names=tuple(map(str, range(9))), feature_names=tuple(f"f{i}" for i in range(15)),
            latitudes=45-dataset.grid.row_indices*.5, longitudes=7+dataset.grid.column_indices*.5,
            spatial_encoder="gat", epochs=1, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1,
            gat_hidden_dim=3, gat_heads=2, discriminator_chunk_size=4, trend_steps=4,
            grid_crs="EPSG:4326", angular_grid_spacing=.5, grid_audit_knn=False, timestep_hours=3,
            device="cpu", mc_samples=3, score_storage="memmap", score_chunk_size=5)
        ranges = []
        for run, test in enumerate((data[11:], data[11:]+100)):
            root = self.root/str(run)
            with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(data[:8], test,
                    checkpoint_path=root/"model.pt", **options) as result:
                ranges.append(result.metadata["score_normalization"])
                self.assertEqual(result.test_scores.shape, (3, 9))
                self.assertEqual(result.metadata["runtime"]["train_batch_size"], 1)
                self.assertEqual(ranges[-1]["fit_period"], "calibration_only")
                self.assertFalse(ranges[-1]["clipping"])
                if run:
                    self.assertGreater(float(result.test_scores.max()), 2)
                scores, std = result.test_scores.copy(), result.anomaly_std.copy()
                layout = CubeGrid(45-np.arange(3)*.5, 7+np.arange(3)*.5,
                                  dataset.grid.row_indices, dataset.grid.column_indices)
                direct = np.concatenate([b for _, b in score_cube_chunks(scores, layout, 2)])
                config = EventConfig(absolute_threshold=float(np.median(scores)), opening_iterations=0,
                                    closing_iterations=0, chunk_size=2)
                process_events(scores, result.test_timestamps, layout, root/"events", config, uncertainty=std)
                np.testing.assert_array_equal(np.load(root/"events/anomaly_mean_cube.npy"), direct)
                np.testing.assert_array_equal(np.load(root/"events/uncertainty_cube.npy")[:, layout.rows, layout.cols], std)
                self.assertTrue((root/"events/events.sqlite").is_file())
                model, payload = load_stgan_checkpoint(root/"model.pt")
                self.assertEqual(payload["model_class"], "STGAN_GAT")
                self.assertEqual(payload["time_encoding"], "onehot")
                self.assertEqual(payload["model_config"]["time_feature_size"], 31)
                self.assertEqual(model.parameter_counts(), payload["parameter_counts"])
                self.assertEqual(payload["score_normalization"], ranges[-1])
                self.assertEqual(payload["window_config"], {"recent_steps": 1, "trend_steps": 4})
                torch.testing.assert_close(model.generator.recent_encoder.edge_index,
                                           grid_edge_index(dataset.grid.row_indices, dataset.grid.column_indices))
        self.assertEqual(ranges[0], ranges[1])


if __name__ == "__main__":
    unittest.main()
