"""The checkpoint analysis must rebuild the run's test dataset and read the components correctly."""
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
from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan, load_stgan_checkpoint
from physiq_pv.era5.data import ERA5Cubes
from scripts import analyze_era5_stgan_checkpoints as script
from test_stgan_mmd import NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

# The model of this branch that scores a whole timestamp per batch.
WHOLE_GRID = (dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4,
                   gat_recurrence="gated", recent_steps=2)
              if hasattr(stgan, "STGANGraphDataset") else dict(cnn_training_mode="full_grid"))


class RunTests(unittest.TestCase):
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
            epochs=2, batch_size=1, score_batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, device="cpu", grid_crs="EPSG:4326", timestep_hours=3, angular_grid_spacing=.5,
            grid_audit_knn=False, score_mode="paper", cache_normalized=False, mc_samples=2, monitoring_timestamps=4,
            monitoring_feature_mmd_samples=16, checkpoint_path=self.root / "run" / "model.pt",
            score_dir=self.root / "run" / "scores", pca_reference=self.root / "pca",
            mmd_reference=self.root / "reference", score_storage="memmap", **WHOLE_GRID)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options):
            pass
        self.run_dir = self.root / "run"

    def cubes(self):
        return ERA5Cubes(train=self.values[TRAIN], validation=self.values[VALIDATION], test=self.values[TEST],
                         train_timestamps=TIMES[TRAIN], validation_timestamps=TIMES[VALIDATION],
                         test_timestamps=TIMES[TEST], **NAMES)

    def test_last_checkpoint_gives_the_observation_score_saved_by_the_run(self):
        saved = np.load(self.run_dir / "scores" / "discriminator_real.npy")
        model, payload = load_stgan_checkpoint(self.run_dir / "model_epoch_2.pt", device="cpu")
        cubes = self.cubes()
        try:
            dataset = script.test_dataset(cubes, payload)
            outputs = script.checkpoint_outputs(model, dataset, np.arange(len(dataset)))
        finally:
            cubes.close()
        self.assertEqual(len(dataset), len(saved))
        np.testing.assert_allclose(outputs["real"].reshape(saved.shape), saved, rtol=1e-5, atol=1e-6)
        # D(reconstruction) is a probability, and the error a mean of squares.
        self.assertTrue(((outputs["generated"] > -1e-5) & (outputs["generated"] < 1 + 1e-5)).all())
        self.assertTrue((outputs["error"] >= 0).all())
        # The first checkpoint is a different model.
        earlier = script.checkpoint_outputs(load_stgan_checkpoint(self.run_dir / "model_epoch_1.pt", device="cpu")[0],
                                            dataset, np.arange(len(dataset)))
        self.assertFalse(np.allclose(earlier["real"], outputs["real"]))

    def test_draws_change_the_reconstruction_terms_and_not_the_observation_score(self):
        model, payload = load_stgan_checkpoint(self.run_dir / "model_epoch_2.pt", device="cpu")
        cubes = self.cubes()
        try:
            dataset = script.test_dataset(cubes, payload)
            positions = np.arange(3)
            deterministic = script.checkpoint_outputs(model, dataset, positions)
            torch.manual_seed(0)
            drawn = script.checkpoint_outputs(model, dataset, positions, mc_samples=4)
            # The same draws, taken directly from the model.
            torch.manual_seed(0)
            expected = []
            with script.scoring_mode(model, True), torch.no_grad():
                for position in positions:
                    batch = dataset.fetch_batch([int(position)])
                    expected.append(model.score_draws(*batch[:5], 4).float().mean(0))
            expected = torch.cat(expected).double().numpy()
        finally:
            cubes.close()
        np.testing.assert_array_equal(drawn["real"], deterministic["real"])
        np.testing.assert_allclose(drawn["error"], expected[:, 0], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(drawn["real"] - drawn["generated"], expected[:, 1], rtol=1e-6, atol=1e-7)
        self.assertFalse(np.allclose(drawn["generated"], deterministic["generated"]))
        self.assertFalse(np.allclose(drawn["error"], deterministic["error"]))
        # The model is left in evaluation mode.
        self.assertFalse(any(module.training for module in model.modules()))

    def test_main_prints_every_epoch_and_the_stability_between_them(self):
        output = io.StringIO()
        with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), redirect_stdout(output):
            self.assertEqual(script.main(["--prepared-dir", "unused", "--run-dir", str(self.run_dir), "--device", "cpu",
                                          "--timestamps", "5", "--mc-samples", "3"]), 0)
        text = output.getvalue()
        self.assertIn("mc_samples=3", text)
        for expected in ("=== epoch 1 ===", "=== epoch 2 ===", "top 1% of the score: D(observation)",
                         "top 0.1% of the component: D(observation)", "combined score, epoch 1 -> 2: of the top 1%"):
            self.assertIn(expected, text)
        only_second = io.StringIO()
        with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), redirect_stdout(only_second):
            script.main(["--prepared-dir", "unused", "--run-dir", str(self.run_dir), "--device", "cpu",
                         "--timestamps", "5", "--epochs", "2", "--mc-samples", "0"])
        self.assertIn("mc_samples=0", only_second.getvalue())
        self.assertNotIn("=== epoch 1 ===", only_second.getvalue())
        self.assertNotIn("stability", only_second.getvalue())

    def test_patch_checkpoints_are_refused(self):
        with self.assertRaisesRegex(ValueError, "whole timestamp per batch"):
            script.test_dataset(None, {"model_class": "STGAN_CONVGRU"})


class SummaryTests(unittest.TestCase):
    def outputs(self, *, discriminator_sees_the_observation):
        rng = np.random.default_rng(0)
        size = 20000
        unusual = np.zeros(size, dtype=bool)
        unusual[:200] = True
        error = rng.random(size) * 0.01 + unusual * 0.05
        real = rng.random(size) * 0.01
        generated = 0.95 + rng.random(size) * 0.01
        if discriminator_sees_the_observation:
            real = real + unusual * 0.8          # D finds the unusual observations unusual
        else:
            generated = generated - unusual * 0.8  # D only stops telling their reconstruction apart
        return {"error": error, "real": real, "generated": generated}

    def test_top_scores_say_which_term_of_the_discriminator_moved(self):
        seen = script.summary(self.outputs(discriminator_sees_the_observation=True))
        blind = script.summary(self.outputs(discriminator_sees_the_observation=False))
        for result, real, generated, moved in ((seen, 0.8, 0.955, 1.0), (blind, 0.005, 0.155, 0.0)):
            top_real, top_generated, top_error, observation_moved = result["top"]["score", 1]
            self.assertAlmostEqual(top_real, real, delta=0.01)
            self.assertAlmostEqual(top_generated, generated, delta=0.01)
            self.assertAlmostEqual(top_error, 0.055, delta=0.002)
            self.assertEqual(observation_moved, moved)
        self.assertAlmostEqual(seen["real_above_half"], 0.01)
        self.assertEqual(blind["real_above_half"], 0.0)
        # Both give the same component, so the same combined score: only the split differs.
        np.testing.assert_allclose(script.combined_score(self.outputs(discriminator_sees_the_observation=True)),
                                   script.combined_score(self.outputs(discriminator_sees_the_observation=False)))


if __name__ == "__main__":
    unittest.main()
