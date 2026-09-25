"""Paper-style ERA5 split and fused MC scores for the global GAT branch."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_stgan_gat import fixture
from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan.data import ContextArray
from physiq_pv.anomaly_detection.stgan.scoring import (
    ScoreStore, summarize_raw_mc_components, normalize_paper_mc_scores)


class GATPaperTests(unittest.TestCase):
    def test_fusion_uses_test_wide_ranges_and_draw_uncertainty(self):
        generator = np.array([[[1., 2.]], [[3., 4.]], [[5., 8.]]], np.float32)
        discriminator = np.array([[[5., 3.]], [[3., 4.]], [[1., 7.]]], np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            store = ScoreStore(root=tmp, backend="memmap")
            try:
                gm, gs, dm, ds, cov = summarize_raw_mc_components(generator, discriminator, store)
                score, std, ranges = normalize_paper_mc_scores(
                    gm, gs, dm, ds, cov, store, raw_generator=generator,
                    raw_discriminator=discriminator)
                draws = ((generator-ranges["r_min"])/(ranges["r_max"]-ranges["r_min"]) +
                         (discriminator-ranges["d_min"])/(ranges["d_max"]-ranges["d_min"]))
                np.testing.assert_allclose(score, draws.mean(0), atol=1e-6)
                np.testing.assert_allclose(std, draws.std(0), atol=1e-6)
            finally:
                store.close()

    def test_gat_pipeline_scores_paper_without_calibration(self):
        dataset, _ = fixture(3, 3, steps=14)
        data = dataset.data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()), fit_and_score_stgan(
                ContextArray(data[:8], data[8:11]), data[11:],
                train_timestamps=dataset.timestamps[:11], test_timestamps=dataset.timestamps[11:],
                location_names=tuple(map(str, range(9))),
                feature_names=tuple(f"f{i}" for i in range(15)),
                latitudes=45-dataset.grid.row_indices*.5,
                longitudes=7+dataset.grid.column_indices*.5,
                spatial_encoder="gat", score_mode="paper", epochs=1,
                hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1,
                gat_hidden_dim=3, gat_heads=2, discriminator_chunk_size=4,
                trend_steps=4, grid_crs="EPSG:4326", angular_grid_spacing=.5,
                grid_audit_knn=False, timestep_hours=3, device="cpu", mc_samples=3,
                score_storage="memmap", score_chunk_size=5, checkpoint_path=root/"model.pt",
                score_dir=root/"scores",
            ) as result:
                self.assertEqual(result.metadata["score_mode"], "paper")
                self.assertNotIn("calibration", result.metadata["splits"])
                self.assertEqual(result.metadata["score_normalization"]["fit_period"],
                                 "complete_test_mc_mean_components")
                self.assertTrue(result.metadata["paper_alignment"]["score_equation"])
                self.assertTrue(np.isfinite(result.test_scores).all())
                self.assertTrue((root/"scores/anomaly_mean.npy").is_file())
                self.assertTrue((root/"scores/discriminator_mean.npy").is_file())

    def test_runner_uses_both_pre_2005_partitions(self):
        from scripts.run_era5_stgan import main
        dataset, _ = fixture(3, 3, steps=14)
        data = dataset.data
        cubes = SimpleNamespace(
            train=data[:8], calibration=data[8:11], test=data[11:],
            train_timestamps=dataset.timestamps[:8],
            calibration_timestamps=dataset.timestamps[8:11],
            test_timestamps=dataset.timestamps[11:],
            location_names=tuple(map(str, range(9))),
            feature_names=tuple(f"f{i}" for i in range(15)),
            latitudes=45-dataset.grid.row_indices*.5,
            longitudes=7+dataset.grid.column_indices*.5,
            close=lambda: None)
        grid = SimpleNamespace(to_dict=lambda: {"grid": "stub"})
        result = SimpleNamespace(test_timestamps=cubes.test_timestamps, metadata={})

        @contextlib.contextmanager
        def fake_fit(train, test, **kwargs):
            self.assertIsInstance(train, ContextArray)
            self.assertEqual(len(train), 11)
            self.assertEqual(kwargs["score_mode"], "paper")
            self.assertNotIn("calibration", kwargs)
            yield result

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"run"
            with patch("scripts.run_era5_stgan.load_prepared", return_value=(
                    cubes, grid, {"calibration_end_year": 2004, "test_start_year": 2005})), \
                 patch("scripts.run_era5_stgan.fit_and_score_stgan", side_effect=fake_fit), \
                 contextlib.redirect_stdout(io.StringIO()):
                main(["train", "--prepared-dir", "unused", "--output-dir", str(output)])
            metadata = json.loads((output/"metadata.json").read_text())
            self.assertEqual(metadata["effective_train_end_year"], 2004)
            self.assertEqual(metadata["scores_file"], "scores/anomaly_mean.npy")

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
                 contextlib.redirect_stdout(io.StringIO()):
                main(["events", "--run-dir", str(run), "--output-dir", str(output)])
            saved = json.loads((output/"metadata.json").read_text())
            self.assertEqual(saved["score_component"], "combined")


if __name__ == "__main__":
    unittest.main()
