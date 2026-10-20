"""The forecast anomaly of the generator, set at the timestamp it is known at, and the observed anomaly."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import physiq_pv.anomaly_detection.stgan as stgan
from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan, load_stgan_checkpoint
from physiq_pv.era5.data import ERA5Cubes
from scripts import score_era5_forecast_anomaly as script
from test_stgan_mmd import COLS, NAMES, ROWS, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

WHOLE_GRID = (dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4,
                   gat_recurrence="gated", recent_steps=2)
              if hasattr(stgan, "STGANGraphDataset") else dict(cnn_training_mode="full_grid"))
GRID = {"latitudes": [45.0, 44.5, 44.0], "longitudes": [7.0, 7.5, 8.0], "rows": ROWS.tolist(), "cols": COLS.tolist(),
        "spacing": 0.5}


class ForecastAnomalyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields()
        build_pca(self.root / "pca", self.values)
        build(self.root / "reference", self.values, self.root / "pca")
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=self.values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=1, batch_size=1, score_batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, device="cpu", grid_crs="EPSG:4326", timestep_hours=3, angular_grid_spacing=.5,
            grid_audit_knn=False, score_mode="paper", cache_normalized=False, mc_samples=2, monitoring_timestamps=4,
            monitoring_feature_mmd_samples=16, checkpoint_path=self.root / "run" / "model.pt",
            score_dir=self.root / "run" / "scores", pca_reference=self.root / "pca",
            mmd_reference=self.root / "reference", score_storage="memmap", **WHOLE_GRID)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options):
            pass
        self.run_dir = self.root / "run"
        self.model, payload = load_stgan_checkpoint(self.run_dir / "model_epoch_1.pt", device="cpu")
        self.payload = {**payload, "normalization": {"kind": "training_only_seasonal_standardisation_then_feature_minmax"}}
        # A cache of the raw values with the run's min-max: the seasonal normal is 0 of the raw values here.
        self.cache = self.root / "cache"
        self.cache.mkdir()
        context = payload["window_config"]["trend_steps"]
        np.save(self.cache / "test.npy", np.concatenate((self.values[VALIDATION][-context:], self.values[TEST])))
        np.save(self.cache / "validation.npy", np.concatenate((self.values[TRAIN][-context:], self.values[VALIDATION])))
        (self.cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {
            "minimum": list(map(float, payload["normalization"]["minimum"])),
            "scale": list(map(float, payload["normalization"]["scale"]))}}), encoding="utf-8")
        (self.run_dir / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID, "backend": {
            "seasonal_normalization": {"mean_file": str(self.cache / "mean.npy")}}}), encoding="utf-8")

    def cubes(self):
        return ERA5Cubes(train=self.values[TRAIN], validation=self.values[VALIDATION], test=self.values[TEST],
                         train_timestamps=TIMES[TRAIN], validation_timestamps=TIMES[VALIDATION],
                         test_timestamps=TIMES[TEST], **NAMES)

    def run_period(self, period):
        with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), \
                patch.object(script.stgan, "load_stgan_checkpoint", return_value=(self.model, self.payload)), \
                redirect_stdout(io.StringIO()):
            script.main(["--run-dir", str(self.run_dir), "--checkpoint", "unused", "--prepared-dir", "unused",
                         "--period", period, "--output-dir", str(self.root / f"{period}_forecast"),
                         "--persistence-dir", str(self.root / f"{period}_observed"), "--device", "cpu"])
        return (np.load(self.root / f"{period}_forecast" / "scores" / "anomaly_mean.npy"),
                np.load(self.root / f"{period}_observed" / "scores" / "anomaly_mean.npy"))

    def test_the_forecast_of_the_next_timestamp_is_set_at_this_one(self):
        ahead, observed = self.run_period("test")
        cubes = self.cubes()
        try:
            dataset = script.checkpoints.test_dataset(cubes, self.payload, cache_dir=self.cache)
            weights = self.model.reconstruction_feature_weights
            weights = np.ones(2) if weights is None else weights.numpy()
            normal = -np.asarray(dataset.minimum) / np.asarray(dataset.scale) * 2 - 1
            expected_forecast, expected_observed = [], []
            with script.scoring_mode(self.model, False), torch.no_grad():
                for position in range(len(dataset)):
                    seen, generated = self.model.reconstructed_cells(*dataset.fetch_batch([position])[:5])
                    expected_forecast.append(((generated.numpy() - normal) ** 2 * weights).mean(axis=1))
                    expected_observed.append(((seen.numpy() - normal) ** 2 * weights).mean(axis=1))
        finally:
            cubes.close()
        self.assertEqual(ahead.shape, (len(TIMES[TEST]), 9))
        np.testing.assert_allclose(observed, np.stack(expected_observed), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(ahead[:-1], np.stack(expected_forecast)[1:], rtol=1e-4, atol=1e-6)
        self.assertTrue(np.isnan(ahead[-1]).all())
        metadata = json.loads((self.root / "test_forecast" / "metadata.json").read_text())
        self.assertEqual((metadata["status"], metadata["scored_period"]), ("complete", "test"))
        np.testing.assert_array_equal(np.load(self.root / "test_forecast" / "test_timestamps.npy"),
                                      TIMES[TEST].as_unit("ns").asi8)

    def test_the_validation_years_with_their_real_history(self):
        ahead, observed = self.run_period("validation")
        self.assertEqual(observed.shape, (len(TIMES[VALIDATION]), 9))
        np.testing.assert_array_equal(np.load(self.root / "validation_observed" / "test_timestamps.npy"),
                                      TIMES[VALIDATION].as_unit("ns").asi8)
        self.assertTrue(np.isfinite(ahead[:-1]).all() and np.isnan(ahead[-1]).all())
        self.assertTrue((observed >= 0).all())


if __name__ == "__main__":
    unittest.main()
