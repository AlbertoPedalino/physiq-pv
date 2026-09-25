"""ERA5 train-through-2004 protocol and paper-style MC score fusion."""
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan.data import ContextArray
from physiq_pv.anomaly_detection.stgan.pipeline import fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan.scoring import (
    ScoreStore, summarize_raw_mc_components, normalize_paper_mc_scores)


class ERA5ComponentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_component_moments_and_covariance(self):
        generator = np.array([[[1., 2.]], [[3., 4.]], [[5., 8.]]], np.float32)
        discriminator = np.array([[[5., 3.]], [[3., 4.]], [[1., 7.]]], np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            store = ScoreStore(root=tmp, backend="memmap")
            try:
                gm, gs, dm, ds, cov = summarize_raw_mc_components(
                    generator, discriminator, store, chunk_size=1)
                np.testing.assert_allclose(gm, generator.mean(0))
                np.testing.assert_allclose(gs, generator.std(0, ddof=0))
                np.testing.assert_allclose(dm, discriminator.mean(0))
                np.testing.assert_allclose(ds, discriminator.std(0, ddof=0))
                expected_cov = ((generator-generator.mean(0)) *
                                (discriminator-discriminator.mean(0))).mean(0)
                np.testing.assert_allclose(cov, expected_cov, atol=1e-6)
                for name in ("generator_mean", "generator_std", "discriminator_mean",
                             "discriminator_std", "component_covariance"):
                    self.assertTrue((Path(tmp)/(name+".npy")).exists())
            finally:
                store.close()

    def test_paper_score_uses_global_test_ranges_and_mc_draw_uncertainty(self):
        generator = np.array([[[1., 2.]], [[3., 4.]], [[5., 8.]]], np.float32)
        discriminator = np.array([[[5., 3.]], [[3., 4.]], [[1., 7.]]], np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            store = ScoreStore(root=tmp, backend="memmap")
            try:
                gm, gs, dm, ds, cov = summarize_raw_mc_components(
                    generator, discriminator, store, chunk_size=1)
                score, uncertainty, normalization = normalize_paper_mc_scores(
                    gm, gs, dm, ds, cov, store, chunk_size=1,
                    raw_generator=generator, raw_discriminator=discriminator)
                expected = ((generator-normalization["r_min"])/
                            (normalization["r_max"]-normalization["r_min"]) +
                            (discriminator-normalization["d_min"])/
                            (normalization["d_max"]-normalization["d_min"]))
                np.testing.assert_allclose(score, expected.mean(axis=0), atol=1e-6)
                np.testing.assert_allclose(uncertainty, expected.std(axis=0), atol=1e-6)
                self.assertEqual(normalization["fit_period"], "complete_test_mc_mean_components")
                self.assertTrue((Path(tmp)/"anomaly_mean.npy").exists())
            finally:
                store.close()

    def test_paper_score_constant_component_has_zero_contribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ScoreStore(root=tmp, backend="memory")
            try:
                gm = np.array([[1., 1.]], np.float32)
                dm = np.array([[1., 3.]], np.float32)
                zeros = np.zeros_like(gm)
                scores, uncertainty, _ = normalize_paper_mc_scores(
                    gm, zeros, dm, zeros, zeros, store,
                    raw_generator=gm, raw_discriminator=dm)
                np.testing.assert_array_equal(scores, [[0., 1.]])
                np.testing.assert_array_equal(uncertainty, [[0., 0.]])
            finally:
                store.close()

    def test_pipeline_scores_components_without_calibration(self):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(16, 9, 2)).astype(np.float32)
        train = ContextArray(data[:8], data[8:12])
        times = pd.date_range("2004-12-30 12:00", periods=16, freq="3h")
        lat, lon = np.meshgrid(50-np.arange(3)*.5, 5+np.arange(3)*.5,
                               indexing="ij")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with redirect_stdout(io.StringIO()), fit_and_score_stgan(
                train, data[12:], train_timestamps=times[:12], test_timestamps=times[12:],
                location_names=tuple(map(str, range(9))), feature_names=("a", "b"),
                latitudes=lat.ravel(), longitudes=lon.ravel(), score_mode="components",
                epochs=1, batch_size=17, score_batch_size=23, hidden_size=8,
                n_layers=1, cnn_channels=4, cnn_layers=1, trend_steps=4,
                grid_crs="EPSG:4326", angular_grid_spacing=.5, grid_audit_knn=False,
                device="cpu", mc_samples=3, score_storage="memmap", timestep_hours=3,
                score_dir=root/"scores", checkpoint_path=root/"model.pt",
            ) as result:
                self.assertEqual(result.metadata["score_mode"], "components")
                self.assertIsNone(result.metadata["score_normalization"])
                self.assertEqual(result.metadata["splits"]["train"]["end"], str(times[11]))
                self.assertEqual(result.metadata["splits"]["test"]["start"], str(times[12]))
                self.assertNotIn("calibration", result.metadata["splits"])
                np.testing.assert_array_equal(result.test_scores, result.test_generator_scores)
                np.testing.assert_array_equal(result.anomaly_std,
                                              np.load(root/"scores/generator_std.npy"))
                self.assertTrue((root/"scores/discriminator_std.npy").exists())
                self.assertTrue((root/"scores/component_covariance.npy").exists())
                self.assertFalse((root/"scores/anomaly_mean.npy").exists())

    def test_pipeline_paper_score_without_calibration(self):
        rng = np.random.default_rng(8)
        data = rng.normal(size=(16, 9, 2)).astype(np.float32)
        times = pd.date_range("2004-12-30 12:00", periods=16, freq="3h")
        lat, lon = np.meshgrid(50-np.arange(3)*.5, 5+np.arange(3)*.5,
                               indexing="ij")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with redirect_stdout(io.StringIO()), fit_and_score_stgan(
                ContextArray(data[:8], data[8:12]), data[12:],
                train_timestamps=times[:12], test_timestamps=times[12:],
                location_names=tuple(map(str, range(9))), feature_names=("a", "b"),
                latitudes=lat.ravel(), longitudes=lon.ravel(), score_mode="paper",
                epochs=1, batch_size=17, score_batch_size=23, hidden_size=8,
                n_layers=1, cnn_channels=4, cnn_layers=1, trend_steps=4,
                grid_crs="EPSG:4326", angular_grid_spacing=.5, grid_audit_knn=False,
                device="cpu", mc_samples=3, score_storage="memmap", timestep_hours=3,
                score_dir=root/"scores", checkpoint_path=root/"model.pt",
            ) as result:
                self.assertEqual(result.metadata["score_mode"], "paper")
                self.assertEqual(result.metadata["score_normalization"]["fit_period"],
                                 "complete_test_mc_mean_components")
                self.assertTrue(result.metadata["paper_alignment"]["score_equation"])
                self.assertNotIn("calibration", result.metadata["splits"])
                self.assertTrue((root/"scores/anomaly_mean.npy").exists())
                self.assertTrue((root/"scores/anomaly_std.npy").exists())
                self.assertTrue(np.isfinite(result.test_scores).all())

    def test_era5_runner_reuses_prepared_2003_2004_as_training(self):
        from scripts.run_era5_stgan import main

        cubes = SimpleNamespace(
            train=np.zeros((2, 1, 1), np.float32),
            calibration=np.ones((2, 1, 1), np.float32),
            test=np.ones((1, 1, 1), np.float32),
            train_timestamps=pd.date_range("2002-12-31 18:00", periods=2, freq="3h"),
            calibration_timestamps=pd.date_range("2003-01-01", periods=2, freq="3h"),
            test_timestamps=pd.date_range("2005-01-01", periods=1, freq="3h"),
            location_names=("one",), feature_names=("a",),
            latitudes=np.array([50.]), longitudes=np.array([5.]), close=lambda: None)
        grid = SimpleNamespace(to_dict=lambda: {"grid": "stub"})
        result = SimpleNamespace(test_timestamps=cubes.test_timestamps, metadata={})

        @contextmanager
        def fake_fit(train, test, **kwargs):
            self.assertIsInstance(train, ContextArray)
            self.assertEqual(len(train), 4)
            self.assertEqual(kwargs["score_mode"], "paper")
            self.assertNotIn("calibration", kwargs)
            yield result

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"run"
            with patch("scripts.run_era5_stgan.load_prepared", return_value=(
                    cubes, grid, {"calibration_end_year": 2004, "test_start_year": 2005})), \
                 patch("scripts.run_era5_stgan.fit_and_score_stgan", side_effect=fake_fit), \
                 redirect_stdout(io.StringIO()):
                main(["train", "--prepared-dir", "unused", "--output-dir", str(output)])
            metadata = json.loads((output/"metadata.json").read_text())
            self.assertEqual(metadata["effective_train_end_year"], 2004)
            self.assertEqual(metadata["scores_file"], "scores/anomaly_mean.npy")

    def test_event_cli_selects_matching_component_uncertainty(self):
        from scripts.run_era5_stgan import main

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/"run"
            scores = run/"scores"
            scores.mkdir(parents=True)
            for name, value in (("generator_mean", 1), ("generator_std", 2),
                                ("discriminator_mean", 3), ("discriminator_std", 4)):
                np.save(scores/(name+".npy"), np.full((1, 1), value, np.float32))
            np.save(run/"test_timestamps.npy", pd.date_range("2005", periods=1).asi8)
            metadata = {"status": "complete", "grid": {},
                        **{name+"_file": "scores/"+name+".npy" for name in (
                            "generator_mean", "generator_std", "discriminator_mean", "discriminator_std")}}
            (run/"metadata.json").write_text(json.dumps(metadata))
            output = Path(tmp)/"events"

            def fake_events(selected, timestamps, grid, destination, config, *, uncertainty):
                self.assertEqual(float(selected[0, 0]), 3.)
                self.assertEqual(float(uncertainty[0, 0]), 4.)
                destination.mkdir()
                return {"status": "complete"}

            with patch("scripts.run_era5_stgan.CubeGrid", return_value=object()), \
                 patch("scripts.run_era5_stgan.process_events", side_effect=fake_events), \
                 redirect_stdout(io.StringIO()):
                main(["events", "--run-dir", str(run), "--output-dir", str(output),
                      "--score-component", "discriminator"])
            saved = json.loads((output/"metadata.json").read_text())
            self.assertEqual(saved["score_component"], "discriminator")

    def test_event_cli_defaults_to_combined_score(self):
        from scripts.run_era5_stgan import main

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/"run"
            scores = run/"scores"
            scores.mkdir(parents=True)
            np.save(scores/"anomaly_mean.npy", np.array([[7.]], np.float32))
            np.save(scores/"anomaly_std.npy", np.array([[.5]], np.float32))
            np.save(run/"test_timestamps.npy", pd.date_range("2005", periods=1).asi8)
            (run/"metadata.json").write_text(json.dumps({
                "status": "complete", "grid": {},
                "anomaly_mean_file": "scores/anomaly_mean.npy",
                "anomaly_std_file": "scores/anomaly_std.npy"}))
            output = Path(tmp)/"events"

            def fake_events(selected, timestamps, grid, destination, config, *, uncertainty):
                self.assertEqual(float(selected[0, 0]), 7.)
                self.assertEqual(float(uncertainty[0, 0]), .5)
                destination.mkdir()
                return {"status": "complete"}

            with patch("scripts.run_era5_stgan.CubeGrid", return_value=object()), \
                 patch("scripts.run_era5_stgan.process_events", side_effect=fake_events), \
                 redirect_stdout(io.StringIO()):
                main(["events", "--run-dir", str(run), "--output-dir", str(output)])
            saved = json.loads((output/"metadata.json").read_text())
            self.assertEqual(saved["score_component"], "combined")


if __name__ == "__main__":
    unittest.main()
