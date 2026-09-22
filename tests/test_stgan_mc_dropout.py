"""Synthetic MC-dropout contracts; no ERA5 observations or production training."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan import STGAN, STGANCNNConfig, fit_and_score_stgan, load_stgan_checkpoint
from physiq_pv.anomaly_detection.stgan.scoring import fit_calibration_ranges, ScoreStore, normalize_mc_scores, score_components, scoring_mode
from physiq_pv.anomaly_detection.stgan.training import gan_train_step
from physiq_pv.era5.cube import CubeGrid
from physiq_pv.era5.events import EventConfig, process_events
from test_stgan_performance import fixture


def small_model(**kwargs):
    return STGAN(n_features=3, hidden_size=8, n_layers=1, cnn_channels=4, cnn_layers=1, **kwargs)


def reference_scores(g, d):
    ar, br = float(g.max()-g.min()), float(d.max()-d.min())
    return ((g-g.min())/(ar if ar >= 1e-8 else 1) +
            (d-d.min())/(br if br >= 1e-8 else 1))


class MCDropoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_configuration_cli_and_invalid_values(self):
        from scripts.run_era5_stgan import parser
        config = STGANCNNConfig()
        self.assertEqual((config.dropout_enabled, config.dropout_p, config.mc_dropout_enabled, config.mc_samples),
                         (True, .2, True, 20))
        args = parser().parse_args(['train', '--prepared-dir', 'unused', '--output-dir', 'unused',
            '--no-dropout-enabled', '--dropout-p', '.4', '--no-mc-dropout-enabled', '--mc-samples', '3'])
        self.assertEqual((args.dropout_enabled, args.dropout_p, args.mc_dropout_enabled, args.mc_samples),
                         (False, .4, False, 3))
        for kwargs in (dict(dropout_p=-.1), dict(dropout_p=1), dict(dropout_p=float('nan')),
                       dict(mc_samples=0), dict(mc_samples=True), dict(mc_samples=1.5),
                       dict(dropout_enabled=1), dict(mc_dropout_enabled='true')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                STGANCNNConfig(**kwargs)

    def test_each_dropout_is_stochastic_and_output_remains_bounded_and_masked(self):
        model = small_model()
        batch = fixture().fetch_batch(list(range(12)))[:4]
        drops = [(name, module) for name, module in model.generator.named_modules()
                 if isinstance(module, torch.nn.Dropout)]
        self.assertEqual([name for name, _ in drops], ['spatial_dropout', 'temporal_dropout', 'fusion_dropout'])
        self.assertFalse(any(isinstance(m, torch.nn.modules.dropout._DropoutNd)
                             for m in model.discriminator.modules()))
        self.assertIsInstance(model.generator.output_projection[-1], torch.nn.Tanh)
        for name, module in drops:
            with self.subTest(name=name):
                model.eval()
                module.train()
                first, second = model.generator(*batch), model.generator(*batch)
                self.assertFalse(torch.equal(first, second))
                self.assertLessEqual(float(first.detach().abs().max()), 1)
                self.assertTrue(torch.equal(first.masked_select(~batch[2].bool()),
                                            torch.zeros_like(first.masked_select(~batch[2].bool()))))
        model.eval()
        self.assertTrue(torch.equal(model.generator(*batch), model.generator(*batch)))
        for options in (dict(dropout_enabled=False), dict(dropout_p=0)):
            deterministic = small_model(**options).train()
            self.assertTrue(torch.equal(deterministic.generator(*batch), deterministic.generator(*batch)))

    def test_training_always_uses_two_independent_forwards_and_no_g_grad_in_d_step(self):
        for reuse in (False, True):
            model = small_model().train()
            batch = fixture().fetch_batch(list(range(7)))[:5]
            go = torch.optim.Adam(model.generator.parameters())
            do = torch.optim.Adam(model.discriminator.parameters())
            generated = {}
            def observe(stage, values):
                if stage in ('D_forward', 'G_forward'):
                    generated[stage] = values['generated'].detach().clone()
                if stage == 'D_backward':
                    self.assertTrue(all(p.grad is None for p in model.generator.parameters()))
            with patch.object(model.generator, 'forward', wraps=model.generator.forward) as forward:
                gan_train_step(model, batch, go, do, reuse_generator=reuse, observe=observe)
            self.assertEqual(forward.call_count, 2)
            self.assertFalse(torch.equal(generated['D_forward'], generated['G_forward']))

    def test_mc_forward_count_modes_and_raw_components(self):
        dataset = fixture(data=fixture().data[:9], trend=6)
        model = small_model().train()
        # An eval-sensitive module verifies that MC never enables all of G.
        bn = torch.nn.BatchNorm1d(8)
        model.generator.time_projection.append(bn)
        initial = [m.training for m in model.modules()]
        running = bn.running_mean.clone()
        seen = []
        def check_modes(module, inputs, outputs):
            for child in model.modules():
                self.assertEqual(child.training, isinstance(child, torch.nn.Dropout))
            seen.append(outputs.detach().clone())
        hook = model.generator.register_forward_hook(check_modes)
        try:
            g, d, features, store = score_components(model, dataset, batch_size=10, device='cpu',
                n_features=3, mc_dropout_enabled=True, mc_samples=20)
        finally:
            hook.remove()
        try:
            self.assertEqual(len(seen), 3*20)  # Includes the final partial batch.
            self.assertEqual(g.shape, (20, 3, 9))
            self.assertEqual(features.shape, (3, 9, 3))
            self.assertFalse(torch.equal(seen[0], seen[1]))
            self.assertEqual([m.training for m in model.modules()], initial)
            self.assertTrue(torch.equal(bn.running_mean, running))
            expected = reference_scores(g, d).astype(np.float64)
            mean, std, gm, dm, *_ = normalize_mc_scores(g, d, store, normalization=fit_calibration_ranges(g, d), chunk_size=7)
            np.testing.assert_allclose(mean, expected.mean(0), atol=1e-7)
            np.testing.assert_allclose(std, expected.std(0), atol=1e-7)
            np.testing.assert_allclose(gm, g.mean(0, dtype=np.float64), rtol=1e-7)
            np.testing.assert_allclose(dm, d.mean(0, dtype=np.float64), atol=1e-8)
            self.assertTrue((std > 0).any())
        finally:
            store.close()

    def test_mc_disabled_single_sample_and_disabled_dropout(self):
        dataset = fixture(data=fixture().data[:8], trend=6)
        for enabled, count, dropout, probability in ((False, 20, True, .2), (True, 1, True, .2),
                                                     (True, 4, False, .2), (True, 4, True, 0)):
            model = small_model(dropout_enabled=dropout, dropout_p=probability)
            with patch.object(model.generator, 'forward', wraps=model.generator.forward) as forward:
                g, d, _, store = score_components(model, dataset, batch_size=30, device='cpu', n_features=3,
                    mc_dropout_enabled=enabled, mc_samples=count)
            try:
                self.assertEqual(forward.call_count, count if enabled else 1)
                _, std, *_ = normalize_mc_scores(g, d, store, normalization=fit_calibration_ranges(g, d))
                np.testing.assert_array_equal(std, np.zeros((2, 9)))
            finally:
                store.close()

    def test_mode_and_memmap_cleanup_on_failure(self):
        model = small_model().train()
        model.generator.trend_encoder.eval()  # Mixed original states must survive.
        states = [m.training for m in model.modules()]
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with scoring_mode(model, True):
                raise RuntimeError('failure')
        self.assertEqual([m.training for m in model.modules()], states)
        with patch.object(model, 'components', side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                score_components(model, fixture(), batch_size=10, device='cpu', n_features=3,
                    mc_dropout_enabled=True, storage='memmap', output_dir=self.root)
        self.assertEqual([m.training for m in model.modules()], states)
        for path in self.root.glob('*.npy'):
            path.unlink()  # Windows catches any leaked mapping.

    def test_known_mean_std_normalization_before_aggregation_and_storage_equivalence(self):
        g = np.array([[[0, 1, 4], [2, 1, 3]], [[9, 5, 2], [2, 6, 8]],
                      [[3, 3, 3], [3, 3, 3]]], np.float32)
        d = np.array([[[1, -1, 0], [3, 0, 2]], [[0, 4, 1], [-3, 2, 1]],
                      [[0, 0, 0], [0, 0, 0]]], np.float32)
        expected = reference_scores(g, d).astype(np.float64)
        wrong = reference_scores(g.mean(0)[None], d.mean(0)[None])[0]
        self.assertFalse(np.allclose(expected.mean(0), wrong))
        for backend, chunk in (('memory', 1), ('memory', 99), ('memmap', 2)):
            store = ScoreStore(root=self.root, backend=backend)
            try:
                mean, std, *_ = normalize_mc_scores(g, d, store, normalization=fit_calibration_ranges(g, d), chunk_size=chunk)
                np.testing.assert_allclose(mean, expected.mean(0), atol=1e-7)
                np.testing.assert_allclose(std, expected.std(0, ddof=0), atol=1e-7)
            finally:
                store.close()

    def test_synthetic_fit_checkpoint_and_score_files(self):
        from pyproj import Transformer
        x, y = np.meshgrid(400000.+np.arange(3)*5000, 5000000.-np.arange(3)*5000)
        lon, lat = Transformer.from_crs(32632, 4326, always_xy=True).transform(x.ravel(), y.ravel())
        values = fixture().data[:16]
        times = pd.date_range('2000', periods=16, freq='h')
        checkpoint = self.root/'model.pt'
        with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(values[:12], values[12:],
            train_timestamps=times[:12]-pd.Timedelta(hours=4), test_timestamps=times[12:], calibration=values[8:12], calibration_timestamps=times[8:12], location_names=tuple(map(str, range(9))),
            feature_names=('a', 'b', 'c'), latitudes=lat, longitudes=lon, epochs=1, batch_size=17,
            score_batch_size=23, hidden_size=8, n_layers=1, cnn_channels=4, cnn_layers=1,
            trend_steps=4, device='cpu', mc_samples=3, score_storage='memmap',
            score_dir=self.root/'scores', checkpoint_path=checkpoint) as result:
            self.assertIs(result.anomaly_mean, result.test_scores)
            self.assertEqual(result.anomaly_mean.shape, (4, 9))
            self.assertEqual(result.anomaly_std.shape, (4, 9))
            self.assertTrue((result.anomaly_std > 0).any())
            np.testing.assert_array_equal(np.load(self.root/'scores/anomaly_mean.npy'), result.anomaly_mean)
            np.testing.assert_array_equal(np.load(self.root/'scores/anomaly_std.npy'), result.anomaly_std)
            restored, payload = load_stgan_checkpoint(checkpoint)
            self.assertTrue(payload['model_config']['dropout_enabled'])
            self.assertEqual(payload['mc_config']['mc_samples'], 3)
            self.assertEqual(restored.generator.fusion_dropout.p, .2)

    def test_cubes_uncertainty_event_split_merge_and_cli(self):
        from scripts.run_era5_stgan import main
        rows, cols = np.indices((7, 11))
        # Reorder locations and omit a cell to test both cube mappings.
        keep = np.random.default_rng(4).permutation(np.arange(1, 77))
        grid = CubeGrid(45-np.arange(7)*.5, 7+np.arange(11)*.5, rows.ravel()[keep], cols.ravel()[keep])
        frames = np.zeros((3, 7, 11), np.float32)
        frames[0, 2:5, 1:4] = 2
        frames[0, 2:5, 7:10] = 2
        frames[1, 2:5, 1:10] = 2  # Merge, followed by split.
        frames[2] = frames[0]
        scores = frames.reshape(3, -1)[:, keep]
        std = np.arange(scores.size, dtype=np.float32).reshape(scores.shape)/100
        times = pd.date_range('2005', periods=3, freq='3h')
        config = EventConfig(absolute_threshold=1, opening_iterations=0, closing_iterations=0,
                             link_policy='overlap', chunk_size=1)
        output = self.root/'events'
        metadata = process_events(scores, times, grid, output, config, uncertainty=std)
        self.assertEqual(metadata['n_events'], 1)
        for filename, expected in (('anomaly_mean_cube.npy', scores), ('uncertainty_cube.npy', std)):
            cube = np.load(output/filename)
            self.assertEqual(cube.shape, (3, 7, 11))
            np.testing.assert_array_equal(cube[:, grid.rows, grid.cols], expected)
            self.assertTrue(np.isnan(cube[:, 0, 0]).all())
        with contextlib.closing(sqlite3.connect(output/'events.sqlite')) as db:
            mean, maximum = db.execute('SELECT mean_uncertainty,max_uncertainty FROM events').fetchone()
            chosen = std[scores > 1]
            self.assertAlmostEqual(mean, chosen.mean(dtype=np.float64), places=7)
            self.assertEqual(maximum, chosen.max())
            self.assertEqual({r[0] for r in db.execute('SELECT relation FROM links')}, {'merge', 'split'})
            clusters = db.execute('SELECT time_index,cells,mean_uncertainty,max_uncertainty FROM clusters').fetchall()
        self.assertEqual(len(clusters), 5)
        labels = np.load(output/'cluster_labels.npy')
        ucube = np.load(output/'uncertainty_cube.npy')
        for cluster_id, (time, cells, mean, maximum) in enumerate(clusters, 1):
            selected = ucube[time][labels[time] == cluster_id]
            self.assertEqual(len(selected), cells)
            self.assertAlmostEqual(mean, selected.mean(dtype=np.float64))
            self.assertEqual(maximum, selected.max())
        # Changing uncertainty must never change detection or temporal links.
        process_events(scores, times, grid, self.root/'other', config, uncertainty=std*100)
        np.testing.assert_array_equal(labels, np.load(self.root/'other/cluster_labels.npy'))
        run = self.root/'run'
        run.mkdir()
        np.save(run/'mean.npy', scores)
        np.save(run/'std.npy', std)
        np.save(run/'test_timestamps.npy', times.as_unit('ns').asi8)
        (run/'metadata.json').write_text(json.dumps(dict(status='complete', grid=grid.to_dict(),
            scores_file='mean.npy', anomaly_mean_file='mean.npy', anomaly_std_file='std.npy')))
        with contextlib.redirect_stdout(io.StringIO()):
            main(['events', '--run-dir', str(run), '--output-dir', str(self.root/'cli'),
                  '--absolute-threshold', '1', '--opening-iterations', '0', '--closing-iterations', '0'])
        np.testing.assert_array_equal(np.load(self.root/'cli/uncertainty_cube.npy'), ucube)
        self.assertIn('mean_uncertainty', pd.read_csv(self.root/'cli/events.csv'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
