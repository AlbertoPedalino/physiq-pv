"""One shared MC normalizer: formula, bounded storage and saved ranges."""
import contextlib
import inspect
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

from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan, load_stgan_checkpoint
from physiq_pv.anomaly_detection.stgan.scoring import fit_calibration_ranges, ScoreStore, component_range, normalize_mc_scores, score_components
from test_stgan_mc_dropout import reference_scores, small_model
from test_stgan_performance import fixture


def shared_reference(g, d):
    scores = reference_scores(g, d).astype(np.float64)
    return scores.mean(0).astype(np.float32), scores.std(0, ddof=0).astype(np.float32)


class MCNormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.g = np.array([[[0, 1, 4], [2, 1, 3]], [[9, 5, 2], [2, 6, 8]],
                           [[3, 3, 3], [3, 3, 3]]], np.float32)
        self.d = np.array([[[1, -1, 0], [3, 0, 2]], [[0, 4, 1], [-3, 2, 1]],
                           [[0, 0, 0], [0, 0, 0]]], np.float32)

    def test_config_api_cli_have_no_normalization_selector(self):
        from scripts.run_era5_stgan import parser
        from scripts.run_pvgis_stgan import parse_args
        self.assertNotIn('mc_normalization', STGANCNNConfig().to_dict())
        for function in (fit_and_score_stgan, normalize_mc_scores):
            self.assertNotIn('mc_normalization', inspect.signature(function).parameters)
        for parse, required in ((parser().parse_args, ['train', '--prepared-dir', 'unused', '--output-dir', 'unused']),
                                (parse_args, ['--manifest', 'unused', '--out-dir', 'unused'])):
            self.assertNotIn('mc_normalization', vars(parse(required)))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse(required+['--mc-normalization', 'shared'])

    def test_shared_formula_memory_memmap_chunking_and_allocation(self):
        expected = shared_reference(self.g, self.d)
        for backend in ('memory', 'memmap'):
            for chunk in (1, 5, 1000):
                store = ScoreStore(root=self.root/f'{backend}_{chunk}', backend=backend)
                try:
                    g = store.allocate_raw('generator_scores', self.g.shape)
                    d = store.allocate_raw('discriminator_scores', self.d.shape)
                    g[:], d[:] = self.g, self.d
                    calibration = fit_calibration_ranges(g, d, chunk_size=chunk)
                    with patch('physiq_pv.anomaly_detection.stgan.scoring.component_range', side_effect=AssertionError('test must not fit ranges')):
                        mean, std, gm, dm, gr, dr = normalize_mc_scores(g, d, store, normalization=calibration, chunk_size=chunk)
                    self.assertEqual(gr, (0., 9.))
                    self.assertEqual(dr, (-3., 7.))
                    self.assertEqual(mean.shape, (2, 3))
                    self.assertEqual(std.shape, (2, 3))
                    np.testing.assert_array_equal(mean, expected[0])
                    np.testing.assert_array_equal(std, expected[1])
                    np.testing.assert_array_equal(gm, self.g.mean(0, dtype=np.float64).astype(np.float32))
                    np.testing.assert_array_equal(dm, self.d.mean(0, dtype=np.float64).astype(np.float32))
                    self.assertEqual(len(store.arrays), 4)
                    self.assertEqual(sum(a.nbytes for a in store.arrays), 4*2*3*4)
                finally:
                    store.close()

    def test_constant_components_and_identical_draws(self):
        for g, d in ((np.ones((3, 2, 3), np.float32), np.full((3, 2, 3), -.2, np.float32)),
                     (np.repeat(self.g[:1], 3, axis=0), np.repeat(self.d[:1], 3, axis=0))):
            store = ScoreStore()
            try:
                mean, std, *_ = normalize_mc_scores(g, d, store, normalization=fit_calibration_ranges(g, d))
                np.testing.assert_array_equal(mean, shared_reference(g, d)[0])
                np.testing.assert_array_equal(std, np.zeros((2, 3)))
            finally:
                store.close()

    def test_one_draw_and_2d_non_mc_inputs_agree(self):
        expected = reference_scores(self.g[:1], self.d[:1])[0]
        for g, d in ((self.g[:1], self.d[:1]), (self.g[0], self.d[0])):
            store = ScoreStore()
            try:
                mean, std, *_ = normalize_mc_scores(g, d, store, normalization=fit_calibration_ranges(g, d), chunk_size=2)
                np.testing.assert_array_equal(mean, expected)
                np.testing.assert_array_equal(std, np.zeros_like(mean))
            finally:
                store.close()

    def test_scale_and_offset_dispersion_is_preserved(self):
        base = np.linspace(.1, 1, 12, dtype=np.float32).reshape(3, 4)
        g = np.stack((base, 2*base+.5))
        d = np.zeros_like(g)
        store = ScoreStore()
        try:
            mean, std, *_ = normalize_mc_scores(g, d, store, normalization=fit_calibration_ranges(g, d), chunk_size=3)
            expected = shared_reference(g, d)
            np.testing.assert_allclose(mean, expected[0], rtol=2e-7, atol=1e-7)
            np.testing.assert_allclose(std, expected[1], rtol=2e-7, atol=1e-7)
            self.assertTrue((std > .1).all())
        finally:
            store.close()

    def test_remote_extreme_does_not_inject_variance_into_fixed_cells(self):
        g = np.repeat(self.g[:1], 3, axis=0)
        g[:, 0, 0] = [10, 20, 30]
        d = np.zeros_like(g)
        store = ScoreStore()
        try:
            _, std, *_ = normalize_mc_scores(g, d, store, normalization=fit_calibration_ranges(g, d), chunk_size=2)
            self.assertGreater(std[0, 0], 0)
            np.testing.assert_array_equal(std.ravel()[1:], np.zeros(5))
        finally:
            store.close()

    def test_disabled_dropout_p0_and_mc_off_have_zero_std(self):
        dataset = fixture(data=fixture().data[:8], trend=6)
        for options, mc in ((dict(dropout_enabled=False), True), (dict(dropout_p=0), True), ({}, False)):
            model = small_model(**options)
            g, d, _, store = score_components(model, dataset, batch_size=7, device='cpu', n_features=3,
                mc_dropout_enabled=mc, mc_samples=3, storage='memmap', output_dir=self.root)
            try:
                _, std, *_ = normalize_mc_scores(g, d, store, normalization=fit_calibration_ranges(g, d))
                np.testing.assert_array_equal(std, np.zeros((2, 9)))
            finally:
                store.close()

    def test_checkpoint_and_metadata_save_actual_ranges_without_mode(self):
        from pyproj import Transformer
        x, y = np.meshgrid(400000.+np.arange(3)*5000, 5000000.-np.arange(3)*5000)
        lon, lat = Transformer.from_crs(32632, 4326, always_xy=True).transform(x.ravel(), y.ravel())
        data = fixture().data[:10]
        times = pd.date_range('2005', periods=10, freq='h')
        with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(data[:8], data[8:],
            train_timestamps=times[:8]-pd.Timedelta(hours=4), test_timestamps=times[8:], calibration=data[4:8], calibration_timestamps=times[4:8], location_names=tuple(map(str, range(9))),
            feature_names=('a', 'b', 'c'), latitudes=lat, longitudes=lon, epochs=1, batch_size=16,
            hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1, trend_steps=3, device='cpu',
            mc_samples=3, score_storage='memmap', score_dir=self.root/'scores',
            checkpoint_path=self.root/'model.pt', save_raw_mc=True) as result:
            g = np.load(self.root/'scores/generator_scores.npy')
            d = np.load(self.root/'scores/discriminator_scores.npy')
            normalizer = result.metadata['score_normalization']
            cg = np.load(self.root/'scores/calibration/generator_scores.npy')
            cd = np.load(self.root/'scores/calibration/discriminator_scores.npy')
            self.assertEqual((normalizer['r_min'], normalizer['r_max']), component_range(cg))
            self.assertEqual((normalizer['d_min'], normalizer['d_max']), component_range(cd))
            self.assertEqual(normalizer['fit_period'], 'calibration_only')
            self.assertTrue(normalizer['frozen_during_test'])
            self.assertFalse(normalizer['clipping'])
            self.assertEqual(normalizer['fit_axes'], ['M', 'T', 'N'])
            self.assertEqual(result.anomaly_mean.shape, (2, 9))
            self.assertEqual(result.anomaly_std.shape, (2, 9))
            scores = ((g-normalizer['r_min'])/(normalizer['r_max']-normalizer['r_min']) +
                      (d-normalizer['d_min'])/(normalizer['d_max']-normalizer['d_min'])).astype(np.float64)
            expected = scores.mean(0), scores.std(0, ddof=0)
            np.testing.assert_allclose(result.anomaly_mean, expected[0], rtol=2e-7, atol=1e-7)
            np.testing.assert_allclose(result.anomaly_std, expected[1], rtol=2e-7, atol=1e-7)
            _, payload = load_stgan_checkpoint(self.root/'model.pt')
            _, epoch = load_stgan_checkpoint(self.root/'model_epoch_1.pt')
            self.assertEqual(payload['score_normalization'], normalizer)
            self.assertEqual(payload['splits'], result.metadata['splits'])
            self.assertLess(payload['splits']['train']['end'], payload['splits']['calibration']['start'])
            self.assertLess(payload['splits']['calibration']['end'], payload['splits']['test']['start'])
            store = ScoreStore()
            try:
                with patch('physiq_pv.anomaly_detection.stgan.scoring.component_range', side_effect=AssertionError('no refit')):
                    mean, std, *_ = normalize_mc_scores(g, d, store, normalization=payload['score_normalization'], chunk_size=1)
                np.testing.assert_array_equal(mean, result.anomaly_mean)
                np.testing.assert_array_equal(std, result.anomaly_std)
            finally:
                store.close()
            self.assertEqual(payload['mc_config'], dict(mc_dropout_enabled=True, mc_samples=3))
            self.assertEqual(epoch['mc_config'], payload['mc_config'])
            for metadata in (result.metadata, normalizer, payload['mc_config'], epoch['mc_config']):
                self.assertNotIn('mc_normalization', metadata)
                self.assertNotIn('mc_policy', metadata)
            self.assertEqual(torch.load(self.root/'model.pt', weights_only=False)['score_normalization'], normalizer)

    def test_frozen_calibration_no_clipping_outliers_and_chunking(self):
        calibration_g = np.array([[[1, 3]], [[2, 5]]], np.float32)
        calibration_d = np.array([[[-2, 0]], [[0, 2]]], np.float32)
        ranges = fit_calibration_ranges(calibration_g, calibration_d, chunk_size=1)
        self.assertEqual(ranges, dict(r_min=1., r_max=5., d_min=-2., d_max=2.))
        g = np.array([[[-3, 9, 13]], [[-7, 13, 17]]], np.float32)
        d = np.array([[[-6, 6, 2]], [[-2, 10, 6]]], np.float32)
        expected = ((g-1)/4+(d+2)/4).astype(np.float64)
        for backend in ('memory', 'memmap'):
            for chunk in (1, 2, 100):
                store = ScoreStore(root=self.root/f'outliers_{backend}_{chunk}', backend=backend)
                try:
                    raw_g = store.allocate_raw('g', g.shape)
                    raw_d = store.allocate_raw('d', d.shape)
                    raw_g[:], raw_d[:] = g, d
                    before = dict(ranges)
                    with patch('physiq_pv.anomaly_detection.stgan.scoring.component_range', side_effect=AssertionError('test leakage')):
                        mean, std, *_ = normalize_mc_scores(raw_g, raw_d, store, normalization=ranges, chunk_size=chunk)
                    np.testing.assert_array_equal(mean, expected.mean(0))
                    np.testing.assert_array_equal(std, expected.std(0, ddof=0))
                    self.assertLess(mean[0, 0], 0)
                    self.assertTrue((mean[0, 1:] > 1).all())
                    self.assertEqual(ranges, before)
                finally:
                    store.close()
                self.assertEqual(list((self.root/f'outliers_{backend}_{chunk}').glob('.mc_raw_*')), [])

    def test_calibration_is_independent_of_test_data_and_scratch_is_released(self):
        from pyproj import Transformer
        x, y = np.meshgrid(400000.+np.arange(3)*5000, 5000000.-np.arange(3)*5000)
        lon, lat = Transformer.from_crs(32632, 4326, always_xy=True).transform(x.ravel(), y.ravel())
        data = fixture().data[:14]
        times = pd.date_range('2005', periods=14, freq='h')
        baseline = None
        for extreme in (False, True):
            directory = self.root/str(extreme)
            with patch('physiq_pv.anomaly_detection.stgan.pipeline.fit_calibration_ranges', wraps=fit_calibration_ranges) as fit, contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(
                data[:8], data[12:]*100 if extreme else data[12:],
                calibration=data[8:12], calibration_timestamps=times[8:12],
                train_timestamps=times[:8], test_timestamps=times[12:], location_names=tuple(map(str, range(9))),
                feature_names=('a', 'b', 'c'), latitudes=lat, longitudes=lon, epochs=1, batch_size=16,
                hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1, trend_steps=3, device='cpu',
                mc_samples=3, score_storage='memmap', score_dir=directory, score_chunk_size=7,
                score_stride=2) as result:
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(fit.call_args.args[0].shape, (3, 4, 9))
                self.assertEqual(result.anomaly_mean.shape, (1, 9))  # Calibration still uses every point.
                ranges = result.metadata['score_normalization']
                if baseline is None:
                    baseline = ranges.copy()
                else:
                    self.assertEqual(ranges, baseline)
                    self.assertTrue((result.anomaly_mean > 1).any())
                self.assertEqual(list(directory.glob('.mc_*')), [])

    def test_missing_calibration_cannot_fall_back_to_test(self):
        store = ScoreStore()
        with self.assertRaises(TypeError):
            normalize_mc_scores(self.g, self.d, store)
        self.assertEqual(store.arrays, [])

    def test_calibration_failure_cleans_scratch_before_test(self):
        from pyproj import Transformer
        x, y = np.meshgrid(400000.+np.arange(3)*5000, 5000000.-np.arange(3)*5000)
        lon, lat = Transformer.from_crs(32632, 4326, always_xy=True).transform(x.ravel(), y.ravel())
        data = fixture().data[:14]
        times = pd.date_range('2005', periods=14, freq='h')
        directory = self.root/'failure'
        with patch('physiq_pv.anomaly_detection.stgan.pipeline.fit_calibration_ranges', side_effect=ValueError('synthetic calibration failure')), \
             patch('physiq_pv.anomaly_detection.stgan.pipeline.score_components', wraps=score_components) as score, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'synthetic calibration failure'):
            fit_and_score_stgan(data[:8], data[12:], calibration=data[8:12], calibration_timestamps=times[8:12],
                train_timestamps=times[:8], test_timestamps=times[12:], location_names=tuple(map(str, range(9))),
                feature_names=('a', 'b', 'c'), latitudes=lat, longitudes=lon, epochs=1, batch_size=16,
                hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1, trend_steps=3, device='cpu',
                mc_samples=3, score_storage='memmap', score_dir=directory)
        self.assertEqual(score.call_count, 1)  # Test inference was never reached.
        self.assertEqual(list(directory.iterdir()), [])

    def test_era5_cli_propagates_calibration_split(self):
        from unittest.mock import Mock
        from scripts.run_era5_stgan import main, parser
        from physiq_pv.era5.data import prepare_era5
        args = parser().parse_args(['prepare', '--output-dir', str(self.root)])
        self.assertEqual((args.train_end_year, args.calibration_end_year), (2002, 2004))
        signature = inspect.signature(prepare_era5)
        self.assertEqual(signature.parameters['train_end_year'].default, 2002)
        self.assertEqual(signature.parameters['calibration_end_year'].default, 2004)
        cubes = Mock()
        with patch('scripts.run_era5_stgan.prepare_era5', return_value=(cubes, None, {})) as prepare, contextlib.redirect_stdout(io.StringIO()):
            main(['prepare', '--output-dir', str(self.root), '--train-end-year', '2000',
                  '--calibration-end-year', '2004', '--end-year', '2006'])
        self.assertEqual(prepare.call_args.kwargs['train_end_year'], 2000)
        self.assertEqual(prepare.call_args.kwargs['calibration_end_year'], 2004)
        self.assertEqual(prepare.call_args.kwargs['score_end_year'], '2006')
        cubes.close.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
