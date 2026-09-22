"""Synthetic normalization audit and opt-in raw MC persistence contracts."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan.scoring import fit_calibration_ranges, ScoreStore, normalize_mc_scores, score_components
from scripts.compare_stgan_mc_normalization import storage_estimate
from test_stgan_mc_dropout import small_model
from test_stgan_performance import fixture


SUMMARY_FILES = {'anomaly_mean.npy', 'anomaly_std.npy', 'generator_mean.npy',
                 'discriminator_mean.npy', 'feature_scores.npy'}
RAW_FILES = {'generator_scores.npy', 'discriminator_scores.npy'}


class MCReviewTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_config_and_both_cli_defaults_and_debug_opt_in(self):
        from scripts.run_era5_stgan import parser
        from scripts.run_pvgis_stgan import parse_args
        self.assertFalse(STGANCNNConfig().save_raw_mc)
        with self.assertRaisesRegex(ValueError, 'save_raw_mc'):
            STGANCNNConfig(save_raw_mc=1)
        for parse, required in ((parser().parse_args, ['train', '--prepared-dir', 'unused', '--output-dir', 'unused']),
                                (parse_args, ['--manifest', 'unused', '--out-dir', 'unused'])):
            self.assertFalse(parse(required).save_raw_mc)
            self.assertTrue(parse(required+['--save-raw-mc']).save_raw_mc)
            self.assertFalse(parse(required+['--save-raw-mc', '--no-save-raw-mc']).save_raw_mc)

    def test_storage_estimate_counts_grid_timestamps_and_float32_bytes(self):
        estimate = storage_estimate()
        self.assertEqual(estimate['locations'], 81*131)
        self.assertEqual(estimate['timestamps'], 61360)
        self.assertEqual(estimate['both_raw_components']['bytes'], 104174553600)
        self.assertEqual(estimate['retained_default']['bytes'], 49482912960)
        self.assertEqual(estimate['peak_array_payload']['bytes'], 153657466560)
        hourly = storage_estimate(timestep_hours=1)
        masked = storage_estimate(n_locations=5000)
        self.assertEqual(hourly['both_raw_components']['bytes'], 3*estimate['both_raw_components']['bytes'])
        self.assertEqual(masked['both_raw_components']['bytes'], 20*61360*5000*4*2)
        # A real small .npy allocation confirms dtype/shape payload, plus its header.
        store = ScoreStore(root=self.root, backend='memmap')
        try:
            raw = store.allocate_raw('generator_scores', (20, 10, 9), save=True)
            self.assertEqual(raw.nbytes, 20*10*9*4)
            self.assertEqual((self.root/'generator_scores.npy').stat().st_size, raw.nbytes+raw.offset)
        finally:
            store.close()

    def test_storage_flag_preserves_draws_rng_results_and_forward_count(self):
        dataset = fixture(data=fixture().data[:8], trend=6)
        model = small_model()
        baseline = None
        baseline_rng = None
        for backend, keep in (('memory', True), ('memory', False), ('memmap', True), ('memmap', False)):
            output = self.root/f'{backend}_{keep}'
            torch.manual_seed(32)
            with patch.object(model.generator, 'forward', wraps=model.generator.forward) as forward:
                g, d, f, store = score_components(model, dataset, batch_size=7, device='cpu', n_features=3,
                    mc_dropout_enabled=True, mc_samples=3, storage=backend, output_dir=output, save_raw_mc=keep)
            try:
                self.assertEqual(forward.call_count, 9)  # 3 draws * 3 batches (partial last batch).
                summaries = normalize_mc_scores(g, d, store, normalization=fit_calibration_ranges(g, d), chunk_size=5)
                arrays = [np.array(a) for a in (*summaries[:4], f)]
                rng = torch.get_rng_state().clone()
                if baseline is None:
                    baseline, baseline_rng = arrays, rng
                else:
                    for a, b in zip(baseline, arrays):
                        np.testing.assert_array_equal(a, b)
                    self.assertTrue(torch.equal(rng, baseline_rng))
                if backend == 'memmap' and not keep:
                    scratch = list(output.glob('.mc_raw_*'))
                    self.assertEqual(len(scratch), 1)
                    self.assertEqual({p.name for p in scratch[0].iterdir()}, RAW_FILES)
                store.discard_temporary_raw()
                if backend == 'memmap':
                    self.assertEqual({p.name for p in output.iterdir()}, SUMMARY_FILES | (RAW_FILES if keep else set()))
                    if keep:
                        for name, expected in (('generator_scores.npy', g), ('discriminator_scores.npy', d)):
                            np.testing.assert_array_equal(np.load(output/name), expected)
            finally:
                store.close()
            if backend == 'memmap':
                self.assertEqual({p.name for p in output.iterdir()}, SUMMARY_FILES | (RAW_FILES if keep else set()))

    def test_temporary_raw_cleanup_after_scoring_and_normalization_failures(self):
        model, dataset = small_model(), fixture(data=fixture().data[:8], trend=6)
        with patch.object(model, 'components', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                score_components(model, dataset, batch_size=7, device='cpu', n_features=3,
                    mc_dropout_enabled=True, storage='memmap', output_dir=self.root)
        self.assertEqual(list(self.root.glob('.mc_raw_*')), [])
        self.assertFalse(any((self.root/name).exists() for name in RAW_FILES))
        store = ScoreStore(root=self.root, backend='memmap')
        raw = store.allocate_raw('raw', (2, 2, 2))
        raw[:] = np.nan
        try:
            with self.assertRaisesRegex(ValueError, 'finite'):
                normalize_mc_scores(raw, raw, store, normalization=fit_calibration_ranges(raw, raw))
        finally:
            store.close()
        self.assertEqual(list(self.root.glob('.mc_raw_*')), [])
        # Anonymous output + scratch directories are both owned and removed.
        store = ScoreStore(backend='memmap')
        raw = store.allocate_raw('raw', (1, 2, 2))
        root = store.root
        store.close()
        self.assertFalse(root.exists())

    def test_pipeline_exports_only_five_summaries_unless_debug_enabled(self):
        from pyproj import Transformer
        import pandas as pd
        x, y = np.meshgrid(400000.+np.arange(3)*5000, 5000000.-np.arange(3)*5000)
        lon, lat = Transformer.from_crs(32632, 4326, always_xy=True).transform(x.ravel(), y.ravel())
        values = fixture().data[:10]
        times = pd.date_range('2005', periods=10, freq='h')
        baseline = None
        for keep in (False, True):
            directory = self.root/str(keep)
            with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(values[:8], values[8:],
                train_timestamps=times[:8]-pd.Timedelta(hours=4), test_timestamps=times[8:], calibration=values[4:8], calibration_timestamps=times[4:8], location_names=tuple(map(str, range(9))),
                feature_names=('a', 'b', 'c'), latitudes=lat, longitudes=lon, epochs=1, batch_size=16,
                hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1, trend_steps=3, device='cpu',
                mc_samples=3, score_storage='memmap', score_dir=directory, save_raw_mc=keep) as result:
                self.assertEqual(result.metadata['save_raw_mc'], keep)
                self.assertEqual({p.name for p in directory.iterdir()}, SUMMARY_FILES | (RAW_FILES | {'calibration'} if keep else set()))
                current = [np.array(a) for a in (result.anomaly_mean, result.anomaly_std,
                    result.test_generator_scores, result.test_discriminator_scores, result.test_feature_scores)]
                if baseline is None:
                    baseline = current
                else:
                    for a, b in zip(baseline, current):
                        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main(verbosity=2)
