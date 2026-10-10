"""The training progress line also reports the time of its interval and the time left in the epoch.

The same file serves the ConvGRU and the GAT branch.
"""
from contextlib import redirect_stdout
import inspect
import io
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan
from test_stgan_mmd import NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

# The expression of notebooks/era5_stgan_training_losses.ipynb: it must keep matching.
NOTEBOOK = re.compile(r'\[stgan\] (?:precision=\S+ )?epoch=(\d+)/(\d+) batch=(\d+)/(\d+) '
                      r'D_mean=([\d.eE+-]+) G_mean=([\d.eE+-]+)')
TIMING = re.compile(r'G_mean=\S+ interval=(\d+\.\d)min epoch_eta=(\d+\.\d\d)h$')


class ProgressTimingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields()
        build_pca(self.root / "pca", self.values)
        build(self.root / "reference", self.values, self.root / "pca")

    def test_every_progress_line_ends_with_interval_and_remaining_time(self):
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=self.values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=2, batch_size=4, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=12, log_interval=1, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, mc_samples=2, monitoring_timestamps=4, monitoring_feature_mmd_samples=16,
            skip_final_scoring=True, checkpoint_path=self.root / "run" / "model.pt",
            score_dir=self.root / "run" / "scores", pca_reference=self.root / "pca",
            mmd_reference=self.root / "reference")
        if "spatial_encoder" in inspect.signature(fit_and_score_stgan).parameters:
            options["spatial_encoder"] = "convgru"
        stream = io.StringIO()
        with redirect_stdout(stream), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options):
            pass
        lines = [line for line in stream.getvalue().splitlines() if "D_mean=" in line]
        self.assertEqual(len(lines), 6)  # Three batches per epoch, two epochs.
        for line in lines:
            self.assertIsNotNone(NOTEBOOK.search(line), line)
            self.assertIsNotNone(TIMING.search(line), line)
        last_of_epoch = [line for line in lines if NOTEBOOK.search(line).group(3) == NOTEBOOK.search(line).group(4)]
        self.assertEqual(len(last_of_epoch), 2)
        for line in last_of_epoch:
            self.assertEqual(TIMING.search(line).group(2), "0.00")  # Nothing is left in the epoch.


if __name__ == "__main__":
    unittest.main()
