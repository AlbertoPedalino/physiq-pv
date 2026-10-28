"""The correlation check must say where and when the scores of G, of D and the combined one go together."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_score_correlation as script

TIMES = pd.date_range("2005-01-01", "2005-03-31 21:00", freq="3h")
GRID = {"latitudes": [45.0, 44.5, 44.0], "longitudes": [10.0, 10.5, 11.0, 11.5], "rows": np.repeat(np.arange(3), 4).tolist(),
        "cols": np.tile(np.arange(4), 3).tolist(), "spacing": 0.5}
CELLS = len(GRID["rows"])


class ScoreCorrelationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        # D is noise; G is noise too, except that in the first half of the cells it follows D in time; the
        # combined score is D plus a little noise, so that it goes with D everywhere.
        self.discriminator = rng.random((len(TIMES), CELLS)).astype(np.float32)
        self.generator = rng.random((len(TIMES), CELLS)).astype(np.float32)
        self.generator[:, :CELLS // 2] = self.discriminator[:, :CELLS // 2] ** 3
        self.combined = self.discriminator + 0.01 * rng.random((len(TIMES), CELLS)).astype(np.float32)

    def write_run(self):
        run = self.root / "run"
        (run / "scores").mkdir(parents=True)
        for name, values in (("anomaly_mean", self.combined), ("generator_mean", self.generator),
                             ("discriminator_mean", self.discriminator)):
            np.save(run / "scores" / f"{name}.npy", values)
        np.save(run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "anomaly_mean_file": "scores/anomaly_mean.npy",
             "generator_mean_file": "scores/generator_mean.npy",
             "discriminator_mean_file": "scores/discriminator_mean.npy"}), encoding="utf-8")
        return run

    def write_climatology(self, name, values, times=TIMES):
        """A run with one score, as the climatology step writes it."""
        run = self.root / name
        (run / "scores").mkdir(parents=True)
        np.save(run / "scores" / "anomaly_mean.npy", values)
        np.save(run / "test_timestamps.npy", times.as_unit("ns").asi8)
        (run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "anomaly_mean_file": "scores/anomaly_mean.npy"}), encoding="utf-8")
        return run

    def scores(self):
        """G and D of the run, and G and D after a climatology: per cell, a shift and a scale of the same values."""
        run = self.write_run()
        shift = np.random.default_rng(3).random(CELLS).astype(np.float32)
        return run, ["G={}@generator".format(run), "D={}@discriminator".format(run),
                     "G+clim={}".format(self.write_climatology("gclim", (self.generator - shift) / (1 + shift))),
                     "D+clim={}".format(self.write_climatology("dclim", (self.discriminator - shift) / (1 + shift)))]

    def run_script(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main([*map(str, arguments)]), 0)
        return output.getvalue()

    def test_correlation_matches_numpy_and_skips_missing_values(self):
        rng = np.random.default_rng(1)
        x, y = rng.random((5, 40)), rng.random((5, 40))
        y[:, :20] += x[:, :20]
        expected = [np.corrcoef(a, b)[0, 1] for a, b in zip(x, y)]
        np.testing.assert_allclose(script.correlation(x, y, axis=1), expected)
        x[0, 3] = np.nan
        self.assertAlmostEqual(script.correlation(x, y, axis=1)[0], np.corrcoef(np.delete(x[0], 3), np.delete(y[0], 3))[0, 1])
        self.assertTrue(np.isnan(script.correlation(np.ones((1, 5)), y[:1, :5], axis=1)[0]))

    def test_rank_correlation_is_one_for_a_monotone_relation(self):
        x = np.random.default_rng(2).random((3, 50))
        result = script.pair_correlations({"G": x, "D": x ** 5, "C": -x}, axis=1)
        np.testing.assert_allclose(result["G - D spearman"], 1)
        np.testing.assert_allclose(result["D - C spearman"], -1)
        self.assertTrue((result["G - D pearson"] < 1).all())
        self.assertEqual(len(result), 2 * 3)

    def test_parse_score(self):
        self.assertEqual(script.parse_score("G=out/run@generator"), ("G", Path("out/run"), "generator"))
        self.assertEqual(script.parse_score("G+clim=out/clim"), ("G+clim", Path("out/clim"), "combined"))
        for wrong in ("out/run", "G=", "G=out@nothing"):
            with self.assertRaises(ValueError):
                script.parse_score(wrong)

    def test_spatial_and_temporal_correlations_of_four_scores(self):
        output = self.root / "out"
        _, scores = self.scores()
        text = self.run_script("--scores", *scores, "--output-dir", output, "--map-times", "2005-02-01T12:00")
        spatial = pd.read_csv(output / "spatial_correlation.csv", index_col=0, parse_dates=True)
        self.assertEqual(len(spatial), len(TIMES))
        self.assertEqual(len(spatial.columns), 6 * 2)  # six pairs, two correlations
        temporal = np.load(output / "temporal_correlation.npz")
        # In time, at every cell: the climatology is a shift and a scale of the cell, so the rank correlation of
        # a score with its own climatology is 1, and G follows D in the first half of the cells only.
        np.testing.assert_allclose(temporal["G_-_G+clim_spearman"], 1)
        np.testing.assert_allclose(temporal["G_-_D_spearman"][:CELLS // 2], 1)
        np.testing.assert_allclose(temporal["G+clim_-_D+clim_spearman"][:CELLS // 2], 1)
        self.assertLess(np.abs(temporal["G_-_D_spearman"][CELLS // 2:]).max(), 0.2)
        # In space the climatology changes the order of the cells: below 1.
        self.assertLess(spatial["G - G+clim spearman"].median(), 0.99)
        for name in ("spatial_correlation.png", "temporal_correlation_pearson.png", "temporal_correlation_spearman.png",
                     "scores_20050201_1200.png"):
            self.assertTrue((output / name).is_file(), name)
        self.assertIn("G+clim - D+clim spearman", text)
        self.assertIn("Jun-Aug", text)

    def test_scores_must_share_the_period_and_be_two_to_four(self):
        run, scores = self.scores()
        with self.assertRaises(ValueError):
            self.run_script("--scores", scores[0], "--output-dir", self.root / "one")
        with self.assertRaises(ValueError):
            self.run_script("--scores", *scores, "C={}".format(run), "--output-dir", self.root / "five")
        shorter = self.write_climatology("short", self.generator[:-8], TIMES[:-8])
        with self.assertRaises(ValueError):
            self.run_script("--scores", scores[0], "S={}".format(shorter), "--output-dir", self.root / "short")

    def test_time_stride_reads_every_time_of_day(self):
        output = self.root / "out"
        _, scores = self.scores()
        self.run_script("--scores", *scores[:2], "--output-dir", output, "--time-stride", "3")
        spatial = pd.read_csv(output / "spatial_correlation.csv", index_col=0, parse_dates=True)
        self.assertEqual(len(spatial), len(TIMES[::3]))
        self.assertEqual(sorted(set(spatial.index.hour)), list(range(0, 24, 3)))
        with self.assertRaises(ValueError):
            self.run_script("--scores", *scores[:2], "--output-dir", output, "--time-stride", "2")


if __name__ == "__main__":
    unittest.main()
