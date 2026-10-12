"""The exploration of the deviation-over-spread score must compute the three scores on the same draws."""
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
import pandas as pd
import torch

import physiq_pv.anomaly_detection.stgan as stgan
from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan
from physiq_pv.era5.data import ERA5Cubes
from scripts import try_era5_stgan_uncertainty_score as script
from test_stgan_mmd import COLS, NAMES, ROWS, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

# The model of this branch that scores a whole timestamp per batch.
WHOLE_GRID = (dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4,
                   gat_recurrence="gated", recent_steps=2)
              if hasattr(stgan, "STGANGraphDataset") else dict(cnn_training_mode="full_grid"))
GRID = {"latitudes": [45.0, 44.5, 44.0], "longitudes": [7.0, 7.5, 8.0], "rows": ROWS.tolist(), "cols": COLS.tolist(),
        "spacing": 0.5}
# Six whole days of test data after the validation of the fixture.
LONG_TEST = pd.date_range(TIMES[VALIDATION][-1] + pd.Timedelta(hours=3), periods=48, freq="3h").as_unit("ns")


class PieceTests(unittest.TestCase):
    def test_standardised_is_the_squared_deviation_over_the_spread_with_a_floor(self):
        draws = torch.tensor([[[1.0, 10.0]], [[3.0, 10.0]]])  # mean 2 and 10; spread 1 and 0
        cells = torch.tensor([[5.0, 11.0]])
        np.testing.assert_allclose(script.standardised(cells, draws)[:, 0].numpy(), [9.0])
        self.assertTrue(torch.isinf(script.standardised(cells, draws)[0, 1]))
        floored = script.standardised(cells, draws, torch.tensor([2.0, 0.5]))
        np.testing.assert_allclose(floored.numpy(), [[(3 / 2) ** 2, (1 / 0.5) ** 2]])

    def test_ordinary_days_come_in_pairs_and_avoid_the_target_days(self):
        days = pd.date_range("2005-01-01", periods=40, freq="D")
        targets = pd.DatetimeIndex(["2005-01-10", "2005-01-20"])
        anchors, ordinary = script.choose_days(days, targets, 10)
        self.assertEqual(len(anchors), 5)
        self.assertFalse(anchors.isin(targets).any() or ordinary.isin(targets).any())
        self.assertTrue((anchors - script.DAY).isin(days).all())
        self.assertTrue(anchors.isin(ordinary).all())
        self.assertGreater(len(ordinary), len(anchors))
        with self.assertRaisesRegex(ValueError, "no ordinary day"):
            script.choose_days(days[:1], targets, 10)
        self.assertEqual(script.percentile_among(3, [1, 2, 3, 4]), 50.0)


class RunTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields(steps=len(TIMES) + len(LONG_TEST))
        build_pca(self.root / "pca", self.values[:len(TIMES)])
        build(self.root / "reference", self.values[:len(TIMES)], self.root / "pca")
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
        (self.run_dir / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")

    def cubes(self):
        """The partitions of the fixture, with six days of test data instead of one."""
        return ERA5Cubes(train=self.values[TRAIN], validation=self.values[VALIDATION], test=self.values[len(TIMES):],
                         train_timestamps=TIMES[TRAIN], validation_timestamps=TIMES[VALIDATION],
                         test_timestamps=LONG_TEST, **NAMES)

    def test_three_scores_on_the_same_draws(self):
        model, payload = stgan.load_stgan_checkpoint(self.run_dir / "model_epoch_1.pt", device="cpu")
        names = list(NAMES["feature_names"])
        cubes = self.cubes()
        try:
            dataset = script.checkpoints.test_dataset(cubes, payload)
            columns, fitted = script.rescore.fitted_scales(self.run_dir, "rarity", None, 1, names)
            floor, calibration, means, shares = script.fit_pilot(model, dataset, np.arange(8), mc_samples=6,
                                                                 device="cpu", precision="fp32")
            with script.scoring_mode(model, True), torch.no_grad():
                torch.manual_seed(0)
                cells, draws = script.draws_of(model, dataset.fetch_batch([10]), 6, "cpu", "fp32")
                scores = script.three_scores(cells, draws, columns=columns, fitted=fitted, floor=floor,
                                             calibration=calibration)
        finally:
            cubes.close()
        self.assertEqual((draws.shape, cells.shape, scores.shape), ((6, 9, 2), (9, 2), (3, 9)))
        squared = script.standardised(cells, draws, floor).double()
        torch.testing.assert_close(scores[1], squared.mean(dim=1))
        torch.testing.assert_close(scores[2], (squared / calibration).mean(dim=1))
        errors = (draws - cells).square().mean(dim=0)
        torch.testing.assert_close(scores[0], script.rescore.transform(errors, "rarity", fitted).mean(dim=1))
        # The calibration is the mean on the pilot: every variable then weighs the same there.
        self.assertEqual((floor.shape, means.shape), ((2,), (2,)))
        self.assertTrue((floor > 0).all() and (means > 0).all())
        for share in shares:
            self.assertAlmostEqual(float(share.sum()), 1.0, places=6)
        self.assertFalse(any(module.training for module in model.modules()))

    def test_main_reads_the_documented_days_against_the_ordinary_ones(self):
        output = io.StringIO()
        with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), \
                patch.object(script, "KNOWN_DAYS", ("2004-01-05", "2019-06-26")), \
                patch.object(script, "KNOWN_ZONES", {"Piedmont": (44, 45, 7, 8)}), redirect_stdout(output):
            self.assertEqual(script.main(["--prepared-dir", "unused", "--run-dir", str(self.run_dir), "--epoch", "1",
                                          "--device", "cpu", "--mc-samples", "4", "--scale-stride", "1",
                                          "--ordinary-days", "4"]), 0)
        text = output.getvalue()
        # One documented day is in the period; the other is not and is left out.
        self.assertIn("documented days of 2019=1 dust arrivals=0", text)
        for expected in ("=== (deviation / spread) squared by variable, on ordinary timestamps ===", "top 1%, calibrated",
                         "=== documented days of 2019: percentile of the day among the ordinary days ===", "Piedmont",
                         "2004-01-05", "days and places at or above 95, of 1: rarity"):
            self.assertIn(expected, text)
        self.assertNotIn("dust arrivals:", text)
        row = next(line for line in text.splitlines() if line.startswith("2004-01-05")).split()
        self.assertEqual(len(row), 4)
        self.assertTrue(all(0 <= float(value) <= 100 for value in row[1:]))

    def test_main_reads_a_dust_arrival_and_the_place_upwind(self):
        days = LONG_TEST.normalize().unique()
        everywhere = {name: (44, 45, 7, 8) for name in script.events.BOXES}
        output = io.StringIO()
        with patch.object(script, "load_prepared", return_value=(self.cubes(), None, {})), \
                patch.object(script, "KNOWN_DAYS", ()), patch.object(script.events, "BOXES", everywhere), \
                patch.object(script, "dust_arrivals", return_value=[(script.events.ITALY, days[2])]) as arrivals, \
                redirect_stdout(output):
            self.assertEqual(script.main(["--prepared-dir", "unused", "--run-dir", str(self.run_dir), "--epoch", "1",
                                          "--device", "cpu", "--mc-samples", "4", "--scale-stride", "1",
                                          "--ordinary-days", "4", "--dust-dir", "dust", "--dust-percentile", "90"]), 0)
        self.assertEqual(arrivals.call_args.args, (Path("dust"), days[0], days[-1], 90.0))
        text = output.getvalue()
        self.assertIn("documented days of 2019=0 dust arrivals=1", text)
        self.assertNotIn("=== documented days of 2019", text)
        row = next(line for line in text.splitlines() if line.startswith(f"{script.events.ITALY} {days[2].date()}"))
        self.assertTrue(all(0 <= float(value) <= 100 for value in row.split()[-3:]))
        self.assertIn("arrivals at or above 95, of 1: rarity", text)
        self.assertRegex(text, r"on an ordinary day the same reading is that high: rarity \d+%")

    def test_dust_arrivals_are_those_of_the_documented_events_inside_the_period(self):
        days = pd.date_range("2018-01-01", "2019-12-31", freq="D")
        dust = {region: pd.Series(50.0, index=days) for region in script.events.SOURCES}
        dust[script.events.CENTRAL]["2019-04-23"] = 99.0
        with patch.object(script, "load_dust", return_value=(None,) * 4), \
                patch.object(script.events, "dust_percentiles", return_value=dust):
            found = script.dust_arrivals("dust", days[0], days[-1], 95)
            self.assertEqual(found, [(script.events.CENTRAL, pd.Timestamp("2019-04-23"))])
            # An event whose days before the start are not in the scored period is left out.
            self.assertEqual(script.dust_arrivals("dust", pd.Timestamp("2019-04-21"), days[-1], 95), [])


if __name__ == "__main__":
    unittest.main()
