"""The check against unseen variables must tell a score that flags unusual weather from one that does not."""
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
import xarray as xr

from scripts import check_era5_score_against_unseen as script

TIMES = pd.date_range("2005-01-01", "2006-12-31 21:00", freq="3h")
LATITUDES, LONGITUDES = [45.0, 44.5], [10.0, 10.5, 11.0]
GRID = {"latitudes": LATITUDES, "longitudes": LONGITUDES, "rows": [0, 0, 0, 1, 1, 1], "cols": [0, 1, 2, 0, 1, 2],
        "spacing": 0.5}
# One day a month the gust is strong everywhere and the first score is high; another day the second score is high.
STORMY, OTHER = TIMES.day == 10, TIMES.day == 20
# A day a month of two years is 2 of the 60 values of a cell, month and time of day, and 3.3% of the values.
TOP, UNUSUAL = "3.2", "5"


class ScoreAgainstUnseenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        gust = rng.random((len(TIMES), 6)).astype(np.float32)
        gust[STORMY] += 10
        self.still = rng.random((len(TIMES), 6)).astype(np.float32)  # a variable that follows no score
        self.unseen = self.root / "unseen"
        self.write_files(gust)
        first, second = (rng.random((len(TIMES), 6)).astype(np.float32) * 0.1 for _ in range(2))
        first[STORMY] += 1
        second[OTHER] += 1
        self.paths = [self.write_run("run_model", first), self.write_run("run_other", second)]

    def write_files(self, gust, *, skip=None):
        for year in (2005, 2006):
            for month in range(1, 13):
                if (year, month) == skip:
                    continue
                chosen = (TIMES.year == year) & (TIMES.month == month)
                (self.unseen / str(year)).mkdir(parents=True, exist_ok=True)
                # The files hold the latitudes from the north and one cell more than the score in longitude.
                fields = {name: (("valid_time", "latitude", "longitude"),
                                 np.concatenate((values[chosen].reshape(-1, 2, 3), np.zeros((chosen.sum(), 2, 1), np.float32)),
                                                axis=2))
                          for name, values in (("i10fg", gust), ("cape", self.still))}
                xr.Dataset(fields, coords={"valid_time": TIMES[chosen], "latitude": LATITUDES,
                                           "longitude": [*LONGITUDES, 11.5]}).to_netcdf(
                    self.unseen / str(year) / f"era5_independent_{year}_{month:02d}.nc")

    def write_run(self, name, scores, *, times=TIMES):
        (self.root / name / "scores").mkdir(parents=True)
        np.save(self.root / name / "scores" / "anomaly_mean.npy", scores)
        np.save(self.root / name / "test_timestamps.npy", times.as_unit("ns").asi8)
        (self.root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")
        return self.root / name

    def run_script(self, *arguments, paths=None):
        paths = paths or self.paths
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--unseen-dir", str(self.unseen), "--score-dir", str(paths[0]), "--by-dirs",
                                          *map(str, paths[1:]), "--top-percent", TOP, "--unusual-percent", UNUSUAL,
                                          *map(str, arguments)]), 0)
        text = output.getvalue()
        tables = [{line[:script.NAME].strip(): [float("nan") if value == "-" else float(value)
                                                for value in line[script.NAME:].split()]
                   for line in part.splitlines()[2:]} for part in text.split("\n\n")[1:]]
        return text, tables

    def test_the_score_that_flags_unusual_weather(self):
        text, (groups, flagged) = self.run_script()
        self.assertIn("score: model, combined | others: other", text)
        self.assertEqual(text.split("\n\n")[1].splitlines()[1].split(), ["values", "cape", "i10fg"])
        # What the first score flags alone is the stormy days: the gust is unusual in all of them.
        alone = groups["score alone, none of the others"]
        self.assertAlmostEqual(alone[0], STORMY.sum() * 6, delta=STORMY.sum() * 6 * 0.1)
        self.assertGreater(alone[2], 85)
        # What the second score flags is not: the gust is unusual there no more than anywhere.
        self.assertLess(groups["others, not the score"][2], 5)
        np.testing.assert_allclose(groups["all values"][1:], [5, 5], atol=1.5)
        # The variable that follows no score is unusual in every group as often as anywhere.
        self.assertLess(alone[1], 10)
        # The two scores hardly ever flag the same value here.
        self.assertLess(groups["score and at least one other"][0], alone[0] * 0.02)
        # Of the values with an unusual gust the first score flags the stormy ones, two in three; the second few.
        self.assertGreater(flagged["model"][1], 45)
        self.assertLess(flagged["other"][1], 8)
        self.assertLess(flagged["model"][0], 8)

    def test_the_low_values_of_a_variable_of_the_run(self):
        # The run kept its standardised data: the radiation is low on the stormy days, and never changes at night.
        rng = np.random.default_rng(1)
        context = 4
        standardised = rng.normal(size=(context + len(TIMES), 6, 2)).astype(np.float32)
        standardised[context:][STORMY, :, 1] -= 10
        night = np.concatenate((np.zeros(context, dtype=bool), TIMES.hour < 6))
        standardised[night, :, 1] = 0
        cache = self.root / "cache"
        cache.mkdir()
        np.save(cache / "test.npy", standardised)
        (cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {"scale": [1, 1]}}), encoding="utf-8")
        run = self.root / "run"
        run.mkdir()
        (run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": ["wind", "radiation"]},
             "backend": {"seasonal_normalization": {"mean_file": str(cache / "seasonal_mean.npy")}}}), encoding="utf-8")
        output = io.StringIO()
        arguments = ["--score-dir", str(self.paths[0]), "--by-dirs", str(self.paths[1]), "--top-percent", TOP,
                     "--unusual-percent", UNUSUAL]
        with redirect_stdout(output):
            self.assertEqual(script.main([*arguments, "--run-dir", str(run), "--variables", "radiation", "--tail", "low"]), 0)
        text = output.getvalue()
        self.assertIn("unusual: the lowest 5% of a variable of the run", text)
        rows = {line[:script.NAME].strip(): line[script.NAME:].split() for line in text.splitlines()}
        # The score flags the stormy days: by day the radiation is low in all of them; the nights do not count.
        self.assertGreater(float(rows["score alone, none of the others"][1]), 85)
        self.assertLess(float(rows["others, not the score"][1]), 5)
        self.assertAlmostEqual(float(rows["all values"][1]), 5, delta=1.5)
        self.assertGreater(float(rows["model"][0]), 45)
        for refused in (["--variables", "radiation"], ["--run-dir", str(run)],
                        ["--run-dir", str(run), "--variables", "radiation", "--unseen-dir", str(self.unseen)]):
            with self.assertRaisesRegex(ValueError, "Give --unseen-dir, or --run-dir with --variables"):
                script.main([*arguments, *refused])
        with self.assertRaisesRegex(ValueError, "Unknown variables"):
            script.main([*arguments, "--run-dir", str(run), "--variables", "rain"])

    def test_only_the_values_near_the_normal(self):
        # A run whose variables are near their normal on the stormy days and far from it on the days of the
        # other score: only the first are counted.
        rng = np.random.default_rng(2)
        context = 4
        standardised = rng.normal(scale=0.5, size=(context + len(TIMES), 6, 2)).astype(np.float32)
        standardised[context:][OTHER, :, 0] = 4
        cache = self.root / "near"
        cache.mkdir()
        np.save(cache / "test.npy", standardised)
        (cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {"scale": [1, 1]}}), encoding="utf-8")
        run = self.root / "run_near"
        run.mkdir()
        (run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": ["wind", "rain"]},
             "backend": {"seasonal_normalization": {"mean_file": str(cache / "seasonal_mean.npy")}}}), encoding="utf-8")
        text, (groups, flagged) = self.run_script("--near-normal-run-dir", run, "--far", 3)
        self.assertIn("only the values where no variable of the run is more than 3 standard deviations", text)
        # The days of the other score are left out, those of the first score stay and the gust is unusual in them.
        self.assertLess(groups["others, not the score"][0], OTHER.sum() * 6 * 0.1)
        self.assertGreater(groups["score alone, none of the others"][0], STORMY.sum() * 6 * 0.9)
        self.assertGreater(groups["score alone, none of the others"][2], 85)
        # The reference in the last column: what the score flags of all the values counted.
        self.assertAlmostEqual(flagged["model"][2], 100 * groups["score alone, none of the others"][0] / groups["all values"][0],
                               delta=0.3)
        self.assertGreater(flagged["model"][1], 10 * flagged["model"][2])

    def test_the_flags_split_by_the_spread_of_the_draws(self):
        # The draws of the first score agree on the stormy days and disagree elsewhere.
        rng = np.random.default_rng(3)
        scores = np.load(self.paths[0] / "scores" / "anomaly_mean.npy")
        spread = np.where(STORMY[:, None], 0.001, 10.0).astype(np.float32) * np.ones_like(scores)
        np.save(self.paths[0] / "scores" / "anomaly_std.npy", spread)
        manifest = json.loads((self.paths[0] / "metadata.json").read_text(encoding="utf-8"))
        (self.paths[0] / "metadata.json").write_text(json.dumps({**manifest, "anomaly_std_file": "scores/anomaly_std.npy"}),
                                                     encoding="utf-8")
        text, (groups, _) = self.run_script("--split-by-spread")
        solid, limit = groups["score alone, solid"], groups["score alone, at the limit"]
        self.assertEqual(solid[0] + limit[0], groups["score alone, none of the others"][0])
        # The solid flags are the stormy days, where the gust is unusual; those at the limit are not.
        self.assertGreater(solid[2], 85)
        self.assertLess(limit[2] if limit[0] else 0, 10)
        self.assertIn("with another, at the limit", text)
        with self.assertRaisesRegex(ValueError, "kept no spread"):
            self.run_script("--split-by-spread", paths=[self.paths[1], self.paths[0]])

    def test_at_equal_score_the_spread_alone(self):
        # Among the flags of the first score, those with a small spread are the stormy days, whatever the score:
        # within each band of the score the lower spread has the unusual gust and the higher one has not.
        scores = np.load(self.paths[0] / "scores" / "anomaly_mean.npy")
        noisy = (TIMES.day == 11)[:, None] & np.ones_like(scores, dtype=bool)
        scores[noisy] += 1  # another day of high score, with no unusual gust and a large spread
        np.save(self.paths[0] / "scores" / "anomaly_mean.npy", scores)
        spread = np.where(noisy, 5.0, 0.001).astype(np.float32)
        np.save(self.paths[0] / "scores" / "anomaly_std.npy", spread)
        manifest = json.loads((self.paths[0] / "metadata.json").read_text(encoding="utf-8"))
        (self.paths[0] / "metadata.json").write_text(json.dumps({**manifest, "anomaly_std_file": "scores/anomaly_std.npy"}),
                                                     encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--unseen-dir", str(self.unseen), "--score-dir", str(self.paths[0]),
                                          "--by-dirs", str(self.paths[1]), "--top-percent", "6.5", "--unusual-percent",
                                          UNUSUAL, "--at-equal-score"]), 0)
        text = output.getvalue()
        self.assertIn("at equal score, all the flags of the score", text)
        self.assertIn("at equal score, the flags of the score alone", text)
        block = text.split("at equal score, all the flags of the score")[1].split("\n\n")[0]
        rows = {line[:script.NAME].strip(): [float("nan" if value == "-" else value) for value in line[script.NAME:].split()]
                for line in block.splitlines() if line.startswith("quarter")}
        self.assertEqual(len(rows), 2 * script.QUARTERS)
        total = sum(values[0] for values in rows.values())
        self.assertGreater(total, 0)
        lower = sum(values[0] * values[2] for name, values in rows.items() if name.endswith("lower spread") and values[0])
        higher = sum(values[0] * values[2] for name, values in rows.items() if name.endswith("higher spread") and values[0])
        self.assertGreater(lower, 3 * higher)

    def test_refusals(self):
        shorter = self.write_run("shorter", np.zeros((len(TIMES) - 8, 6), dtype=np.float32), times=TIMES[:-8])
        with self.assertRaisesRegex(ValueError, "same timestamps"):
            self.run_script(paths=[self.paths[0], shorter])
        (self.unseen / "2006" / "era5_independent_2006_03.nc").unlink()
        with self.assertRaisesRegex(ValueError, "No file of the unseen variables for 2006-03"):
            self.run_script()
        # The years downloaded so far can be read alone: a day a month is then 1 of the 31 values of its kind.
        text, (groups, _) = self.run_script("--years", 2005, 2005)
        self.assertIn("years 2005-2005", text)
        self.assertAlmostEqual(groups["all values"][0], (TIMES.year == 2005).sum() * 6)
        self.assertGreater(groups["score alone, none of the others"][2], 85)
        with self.assertRaisesRegex(ValueError, "no timestamp in 1990-1991"):
            self.run_script("--years", 1990, 1991)


if __name__ == "__main__":
    unittest.main()
