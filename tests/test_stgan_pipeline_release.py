"""A run that releases the file pages it has read must train and score exactly as one that keeps them."""
from contextlib import redirect_stdout
import io
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
from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan, loading, pipeline
from test_stgan_mmd import NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

# The model of this branch that scores a whole timestamp per batch.
WHOLE_GRID = (dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4,
                   gat_recurrence="gated", recent_steps=2)
              if hasattr(stgan, "STGANGraphDataset") else dict(cnn_training_mode="full_grid"))


class PipelineReleaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        values = fields()
        build_pca(self.root / "pca", values)
        build(self.root / "reference", values, self.root / "pca")
        # The data read from a file, as the runs read the prepared cube: the release then acts on a real map.
        np.save(self.root / "values.npy", values)
        self.values = np.load(self.root / "values.npy", mmap_mode="r")
        self.addCleanup(lambda: self.values._mmap.close())

    def run_once(self, name):
        torch.manual_seed(0)
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=self.values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=2, batch_size=1, score_batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, device="cpu", grid_crs="EPSG:4326", timestep_hours=3, angular_grid_spacing=.5,
            grid_audit_knn=False, score_mode="paper", cache_normalized=False, mc_samples=2, monitoring_timestamps=4,
            monitoring_feature_mmd_samples=16, checkpoint_path=self.root / name / "model.pt",
            score_dir=self.root / name / "scores", pca_reference=self.root / "pca",
            mmd_reference=self.root / "reference", score_storage="memmap", seed=7, **WHOLE_GRID)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options):
            pass
        return self.root / name

    def test_training_and_scores_are_the_same_with_and_without_the_release(self):
        calls = []
        real = loading.release_pages

        def counted(*arrays):
            calls.append(len(arrays))
            return real(*arrays)

        with patch.object(pipeline, "release_pages", counted), patch.object(loading, "RELEASE_EVERY", 1):
            released = self.run_once("released")
        with patch.object(pipeline, "release_pages", lambda *arrays: None), \
                patch.object(loading, "release_pages", lambda *arrays: None):
            kept = self.run_once("kept")
        # In the main process: once per period read by the checks (train, validation, test), once before the
        # training and once after each of the two epochs.
        self.assertEqual(len(calls), 3 + 1 + 2)
        for epoch in (1, 2):
            first = torch.load(released / f"model_epoch_{epoch}.pt", map_location="cpu", weights_only=False)
            second = torch.load(kept / f"model_epoch_{epoch}.pt", map_location="cpu", weights_only=False)
            for key, value in first["model_state_dict"].items():
                self.assertTrue(torch.equal(value, second["model_state_dict"][key]), (epoch, key))
        files = sorted(path.name for path in (released / "scores").glob("*.npy"))
        self.assertTrue(files)
        for name in files:
            np.testing.assert_array_equal(np.load(released / "scores" / name), np.load(kept / "scores" / name), name)


if __name__ == "__main__":
    unittest.main()
