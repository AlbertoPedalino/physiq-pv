"""The run report must read the history, the log, the Monte Carlo spread and the events of a real run."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch

import physiq_pv.anomaly_detection.stgan as stgan
from physiq_pv.anomaly_detection.stgan import build_spatial_grid, fit_and_score_stgan
from physiq_pv.era5.cube import CubeGrid
from physiq_pv.era5.events import EventConfig, process_events
from scripts import report_era5_stgan_run as script
from test_stgan_mmd import NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

# The model of this branch that scores a whole timestamp per batch.
WHOLE_GRID = (dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4,
                   gat_recurrence="gated", recent_steps=2)
              if hasattr(stgan, "STGANGraphDataset") else dict(cnn_training_mode="full_grid"))

LOG = """[stgan] Adam lr_G=0.001
[stgan] precision=bf16 epoch=1/2 batch=100/250 D_mean=0.60 G_mean=9.0 interval=1min epoch_eta=0.1h
[stgan] precision=bf16 epoch=1/2 batch=200/250 D_mean=0.58 G_mean=7.0 interval=1min epoch_eta=0.1h
[stgan] precision=bf16 epoch=1/2 batch=250/250 D_mean=0.57 G_mean=6.9
[stgan] epoch=2/2 batch=100/250 D_mean=0.57 G_mean=6.8
[stgan] epoch=2/2 batch=200/250 D_mean=0.20 G_mean=19.5
[stgan] epoch=2/2 batch=212/250 D_mean=0.90 G_mean=1.0
"""


class ReportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_log_summary_gives_ranges_and_the_largest_jump_without_the_short_last_line(self):
        (self.root / "train.log").write_text(LOG, encoding="utf-8")
        summary = script.log_summary(self.root / "train.log").set_index("epoch")
        # Epoch 1: its last line covers 50 batches, half an interval: kept. Epoch 2: 12 batches: left out.
        self.assertEqual(summary.loc[1, "lines"], 3)
        self.assertEqual(summary.loc[2, "lines"], 2)
        self.assertEqual((summary.loc[1, "D_min"], summary.loc[1, "D_max"]), (0.57, 0.60))
        self.assertAlmostEqual(summary.loc[1, "largest_G_jump"], 2.0)
        self.assertEqual(summary.loc[1, "at_batch"], 200)
        self.assertAlmostEqual(summary.loc[2, "largest_G_jump"], 12.7)
        self.assertEqual((summary.loc[2, "G_max"], summary.loc[2, "D_min"]), (19.5, 0.20))

    def test_spread_by_band_on_known_values(self):
        (self.root / "scores").mkdir()
        rng = np.random.default_rng(0)
        mean = rng.random((200, 50)).astype(np.float32) + 0.01
        np.save(self.root / "scores" / "anomaly_mean.npy", mean)
        np.save(self.root / "scores" / "anomaly_std.npy", (0.1 * mean).astype(np.float32))
        np.save(self.root / "scores" / "generator_std.npy", np.full((200, 50), 0.02, np.float32))
        np.save(self.root / "scores" / "discriminator_std.npy", np.zeros((200, 50), np.float32))
        table, correlation = script.spread_by_band(self.root, stride=1)
        self.assertEqual(list(table.score_percentiles), ["0-50", "50-90", "90-95", "95-99", "99-99.9", "99.9-100"])
        np.testing.assert_allclose(table.spread_over_score, 0.1, rtol=1e-5)
        self.assertTrue((np.diff(table.mean_score) > 0).all())
        self.assertAlmostEqual(correlation, 1.0, places=6)
        components = script.component_spread(self.root, stride=1)
        self.assertAlmostEqual(components["generator"][0], 0.02, places=6)
        self.assertEqual(components["discriminator"], (0.0, 0.0))

    def test_main_on_a_real_run_and_its_events(self):
        values = fields()
        build_pca(self.root / "pca", values)
        build(self.root / "reference", values, self.root / "pca")
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(
                values[TRAIN], values[TEST], train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
                validation=values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
                epochs=2, batch_size=1, score_batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
                trend_steps=2, device="cpu", grid_crs="EPSG:4326", timestep_hours=3, angular_grid_spacing=.5,
                grid_audit_knn=False, score_mode="paper", cache_normalized=False, mc_samples=3, monitoring_timestamps=4,
                monitoring_feature_mmd_samples=16, score_storage="memmap", checkpoint_path=self.root / "run" / "model.pt",
                score_dir=self.root / "run" / "scores", pca_reference=self.root / "pca",
                mmd_reference=self.root / "reference", **WHOLE_GRID) as result:
            scores = np.load(self.root / "run" / "scores" / "anomaly_mean.npy")
            spread = np.load(self.root / "run" / "scores" / "anomaly_std.npy")
            latitudes, longitudes = np.asarray(NAMES["latitudes"]), np.asarray(NAMES["longitudes"])
            grid = build_spatial_grid(latitudes, longitudes, grid_crs="EPSG:4326", angular_spacing=.5, audit_knn=False)
            cube = CubeGrid(latitudes=np.unique(latitudes)[::-1], longitudes=np.unique(longitudes),
                            rows=grid.row_indices, cols=grid.column_indices)
            with redirect_stdout(io.StringIO()):
                process_events(scores, result.test_timestamps, cube, self.root / "events",
                               EventConfig(top_percent=50, opening_iterations=0, closing_iterations=0),
                               uncertainty=spread)
        (self.root / "train.log").write_text(LOG, encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--run-dir", str(self.root / "run"), "--log", str(self.root / "train.log"),
                                          "--events-dir", str(self.root / "events")]), 0)
        text = output.getvalue()
        for expected in ("=== training history ===", "generator_reconstruction_loss", "validation_pca_mmd_rolling_mean",
                         "=== progress lines per epoch ===", "largest_G_jump",
                         "=== Monte Carlo spread of the score by score band ===", "rank correlation between score and spread",
                         "generator component: mean spread", "=== events:", "status=complete", "LARGEST (cells x timesteps)",
                         "MOST INTENSE (highest score, at least 24 h)", "share of flagged cells by month"):
            self.assertIn(expected, text)
        history = pd.read_csv(self.root / "run" / "training_history.csv")
        self.assertEqual(len(script.history_table(self.root / "run")), len(history))
        self.assertEqual(json.loads((self.root / "events" / "metadata.json").read_text())["status"], "complete")
        self.assertIn("spread/score", text)
        self.assertRegex(text, r"\d{4}-\d{2}-\d{2}T\d{2} -> \d{4}-\d{2}-\d{2}T\d{2} \|")


if __name__ == "__main__":
    unittest.main()
