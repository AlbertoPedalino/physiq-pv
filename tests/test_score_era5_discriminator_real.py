"""The script that adds discriminator_real.npy to a finished run must give what scoring itself saves."""
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

from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan, load_stgan_checkpoint
from physiq_pv.era5.data import ERA5Cubes
from scripts import score_era5_discriminator_real as script
from test_stgan_mmd import NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields


class ScriptTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields()
        build_pca(self.root / "pca", self.values)
        build(self.root / "reference", self.values, self.root / "pca")

    def cubes(self):
        return ERA5Cubes(train=self.values[TRAIN], validation=self.values[VALIDATION], test=self.values[TEST],
                         train_timestamps=TIMES[TRAIN], validation_timestamps=TIMES[VALIDATION],
                         test_timestamps=TIMES[TEST], **NAMES)

    def fit(self, name, **overrides):
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=self.values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=1, batch_size=1, score_batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, recent_steps=2, device="cpu", grid_crs="EPSG:4326", timestep_hours=3,
            angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper", cache_normalized=False,
            mc_samples=3, monitoring_timestamps=4, monitoring_feature_mmd_samples=16, score_storage="memmap",
            spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4,
            checkpoint_path=self.root / name / "model.pt", score_dir=self.root / name / "scores",
            pca_reference=self.root / "pca", mmd_reference=self.root / "reference")
        train = self.values[TRAIN]
        if not overrides.pop("holdout", True):
            # As the runs that precede the validation split: 2004 is part of the training data.
            train = np.concatenate((train, self.values[VALIDATION]))
            options.update(train_timestamps=TIMES[TRAIN].append(TIMES[VALIDATION]), validation=None,
                           validation_timestamps=None, pca_reference=None, mmd_reference=None)
        options.update(overrides)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(train, self.values[TEST], **options):
            pass
        return self.root / name

    def test_script_reproduces_the_array_saved_by_scoring(self):
        for name, extra in (("gat", {}), ("gat_gru", {"gat_recurrence": "gated"}),
                            ("no_holdout", {"holdout": False}),
                            ("patch", {"spatial_encoder": "convgru", "batch_size": 4, "score_batch_size": 4})):
            with self.subTest(run=name):
                run = self.fit(name, **extra)
                saved = np.load(run / "scores" / "discriminator_real.npy")
                (run / "scores" / "discriminator_real.npy").rename(run / "scores" / "from_scoring.npy")
                with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), \
                        redirect_stdout(io.StringIO()) as printed:
                    script.main(["--prepared-dir", "unused", "--run-dir", str(run), "--device", "cpu",
                                 "--num-workers", "0", "--discriminator-chunk-size", "7",
                                 "--batch-size", "1" if "patch" not in name else "8"])
                rebuilt = np.load(run / "scores" / "discriminator_real.npy")
                self.assertEqual(rebuilt.shape, saved.shape)
                np.testing.assert_allclose(rebuilt, saved, rtol=1e-5, atol=1e-6)
                self.assertIn("share outside [0,1]=0.00000", printed.getvalue())

    def test_existing_file_is_kept_unless_overwrite(self):
        run = self.fit("gat")
        before = (run / "scores" / "discriminator_real.npy").read_bytes()
        with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), \
                self.assertRaises(SystemExit):
            script.main(["--prepared-dir", "unused", "--run-dir", str(run), "--device", "cpu"])
        self.assertEqual((run / "scores" / "discriminator_real.npy").read_bytes(), before)

    def test_a_directory_without_a_finished_run_is_refused(self):
        (self.root / "empty").mkdir()
        with self.assertRaises(SystemExit):
            script.main(["--prepared-dir", "unused", "--run-dir", str(self.root / "empty"), "--device", "cpu"])

    def test_a_failure_leaves_no_partial_file(self):
        run = self.fit("gat")
        target = run / "scores" / "discriminator_real.npy"
        target.unlink()
        model, _ = load_stgan_checkpoint(run / "model.pt")
        with patch.object(type(model), "observation_scores", side_effect=RuntimeError("stop")), \
                patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), \
                redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            script.main(["--prepared-dir", "unused", "--run-dir", str(run), "--device", "cpu", "--num-workers", "0"])
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
