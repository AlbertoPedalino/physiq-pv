"""A run trained without its final scoring, scored afterwards from its last checkpoint, must give the scores
of the same run with its final scoring."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from physiq_pv.experiments.stgan_wandb import default_config
from scripts import run_era5_stgan
from scripts import score_era5_stgan_run as script
from test_era5_normal_validation import prepare, quiet


class ScoreRunTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        prepare(self.root / "prepared")
        common = ["--prepared-dir", str(self.root / "prepared"), "--train-end-year", "2002"]
        quiet(run_era5_stgan.main, [
            "train", *common, "--output-dir", str(self.root / "cache_run"), "--validation-holdout",
            "--normalization", "seasonal", "--cnn-training-mode", "full_grid", "--reconstruction-weights",
            "daily_change", "--shared-cache-dir", str(self.root / "cache"), "--device", "cpu", "--cache-only"])
        cache = json.loads((self.root / "cache_run" / "metadata.json").read_text(encoding="utf-8"))["backend"]["shared_cache"]
        quiet(run_era5_stgan.main, ["normal-masks", *common, "--shared-cache-dir", cache, "--output-dir",
                                    str(self.root / "masks")])
        quiet(run_era5_stgan.main, ["pca-reference", *common, "--shared-cache-dir", cache, "--normal-masks-dir",
                                    str(self.root / "masks"), "--output-dir", str(self.root / "pca"),
                                    "--pca-samples", "48", "--pca-components", "6"])
        quiet(run_era5_stgan.main, ["mmd-reference", *common, "--shared-cache-dir", cache, "--normal-masks-dir",
                                    str(self.root / "masks"), "--pca-reference-dir", str(self.root / "pca"),
                                    "--output-dir", str(self.root / "mmd"), "--subsets", "3", "--subset-size", "8",
                                    "--pca-components", "4"])
        self.config = replace(default_config("era5"), precision="fp32", epochs=2, num_workers=0, hidden_size=4,
            n_layers=1, cnn_channels=4, cnn_layers=1, trend_steps=56, train_samples_per_epoch=4, mc_samples=2,
            cnn_training_mode="full_grid", batch_size=4, score_batch_size=256, normalization="seasonal",
            reconstruction_weights="daily_change", train_end_year=2002, monitoring_timestamps=4,
            monitoring_feature_mmd_samples=16, mmd_objective_window=2, time_encoding="cyclic",
            validation_anomalies="excluded_and_replaced", generator_learning_rate=5e-4, discriminator_lr_ratio=0.9,
            generator_reconstruction_weight=150.0, discriminator_dropout_p=0.2,
            # Without the dropout draws the scores of the same weights are the same numbers. With them, a scoring
            # after training and one after a resume draw different masks, since the random state differs.
            mc_dropout_enabled=False)
        self.references = dict(pca_reference_dir=self.root / "pca", mmd_reference_dir=self.root / "mmd",
                               shared_cache_dir=self.root / "cache", normal_masks_dir=self.root / "masks")

    def train(self, name, skip):
        return quiet(run_era5_stgan.run_training, prepared_dir=self.root / "prepared", output_dir=self.root / name,
                     config=self.config, device="cpu", seed=7, skip_final_scoring=skip, **self.references)

    def test_the_scores_are_those_of_the_run_with_its_final_scoring(self):
        full = self.train("full", skip=False)
        trained = self.train("sweep", skip=True)
        self.assertEqual(trained["status"], "training_only")
        inputs, built = script.scoring_inputs(self.root / "sweep")
        self.assertEqual((inputs["seed"], inputs["resume_from"].name), (7, "model_epoch_2.pt"))
        self.assertEqual(inputs["config"], self.config)
        self.assertEqual(Path(inputs["normal_masks_dir"]), self.root / "masks")
        self.assertEqual(Path(inputs["mmd_reference_dir"]), self.root / "mmd")
        self.assertEqual(built.parent, self.root / "cache")
        scored = quiet(script.main, ["--run-dir", str(self.root / "sweep"), "--prepared-dir",
                                     str(self.root / "prepared"), "--output-dir", str(self.root / "scored"),
                                     "--device", "cpu"])
        self.assertEqual(scored["status"], "complete")
        self.assertEqual(scored["backend"]["resume"]["completed_epochs"], 2)
        weights, resumed = (torch.load(self.root / name / "model_epoch_2.pt", weights_only=False)["model_state_dict"]
                            for name in ("full", "sweep"))
        self.assertTrue(all(torch.equal(weights[key], resumed[key]) for key in weights))
        for name in ("anomaly_mean", "generator_mean", "discriminator_mean"):
            np.testing.assert_allclose(np.load(self.root / "scored" / "scores" / f"{name}.npy"),
                                       np.load(self.root / "full" / "scores" / f"{name}.npy"), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(np.load(self.root / "scored" / "test_timestamps.npy"),
                                      np.load(self.root / "full" / "test_timestamps.npy"))
        self.assertEqual(scored["config"], full["config"])
        # The run directory is left as it was; a second scoring needs a new directory.
        self.assertEqual(json.loads((self.root / "sweep" / "metadata.json").read_text())["status"], "training_only")
        with self.assertRaisesRegex(ValueError, "new/empty"):
            quiet(script.main, ["--run-dir", str(self.root / "sweep"), "--prepared-dir", str(self.root / "prepared"),
                                "--output-dir", str(self.root / "scored"), "--device", "cpu"])
        with self.assertRaisesRegex(ValueError, "without its final scoring"):
            script.scoring_inputs(self.root / "scored")

        # The climatology of the generator's error fitted on the validation years, without the flagged values.
        quiet(run_era5_stgan.main, ["climatology", "--run-dir", str(self.root / "scored"), "--output-dir",
                                    str(self.root / "gen_clim"), "--score-component", "generator", "--reference",
                                    "validation", "--min-samples", "2"])
        clim = json.loads((self.root / "gen_clim" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual((clim["reference"], clim["source_score_component"]), ("validation", "generator"))
        errors = np.load(self.root / "scored" / "validation_generator_mean.npy")
        mask = np.load(self.root / "masks" / "validation_mask.npy")
        self.assertEqual(clim["normalization"]["reference_values_left_out"], int((mask & np.isfinite(errors)).sum()))
        self.assertGreater(clim["normalization"]["reference_values_left_out"], 0)
        self.assertEqual(Path(clim["normalization"]["normal_masks"]).resolve(), (self.root / "masks").resolve())
        self.assertGreater(np.isfinite(np.load(self.root / "gen_clim" / "scores" / "anomaly_mean.npy")).mean(), .5)
        with self.assertRaisesRegex(ValueError, "generator only"):
            quiet(run_era5_stgan.main, ["climatology", "--run-dir", str(self.root / "scored"), "--output-dir",
                                        str(self.root / "combined_clim"), "--reference", "validation"])
        self.assertFalse((self.root / "combined_clim").exists())


if __name__ == "__main__":
    unittest.main()
