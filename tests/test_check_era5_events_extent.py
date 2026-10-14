"""The extent check must tell the part of an event only its score flags from the weather around the event."""
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

from scripts import check_era5_events_extent as script

TIMES = pd.date_range("2005-01-01", "2005-12-31 21:00", freq="3h")
LATITUDES, LONGITUDES = [45.0, 44.5, 44.0], [7.0, 7.5, 8.0, 8.5]
ROWS, COLS = np.indices((3, 4)).reshape(2, -1)
GRID = {"latitudes": LATITUDES, "longitudes": LONGITUDES, "rows": ROWS.tolist(), "cols": COLS.tolist(), "spacing": 0.5}
PART_DAYS, ALONE_DAYS = TIMES.day == 10, TIMES.day == 20
BLOCK = [0, 1, 4, 5]  # the cells (0,0), (0,1), (1,0), (1,1) of the event the others flag in part
CORNER = 11  # the cell (2,3) of the event of the score alone


class EventsExtentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        model, other = (rng.random((len(TIMES), 12)).astype(np.float32) * 0.1 for _ in range(2))
        model[np.ix_(PART_DAYS, BLOCK)] += 1
        model[ALONE_DAYS, CORNER] += 1
        other[PART_DAYS, 0] += 1  # the others flag one cell of the four
        self.runs = [self.write_run("run_model", model), self.write_run("run_other", other)]
        labels = np.zeros((len(TIMES), 3, 4), dtype=np.int64)
        labels[np.ix_(PART_DAYS, [0, 1], [0, 1])] = 1
        labels[ALONE_DAYS, 2, 3] = 2
        self.events = self.root / "events"
        self.events.mkdir()
        np.save(self.events / "cluster_labels.npy", labels)
        np.save(self.events / "timestamps.npy", TIMES.as_unit("ns").asi8)
        pd.DataFrame({"cluster_id": [1, 2], "event_id": [1, 2]}).to_csv(self.events / "clusters.csv", index=False)
        # The unseen variable is unusual over both events, on their days, and nowhere else.
        cape = rng.random((len(TIMES), 3, 4)).astype(np.float32)
        cape[np.ix_(PART_DAYS, [0, 1], [0, 1])] += 10
        cape[ALONE_DAYS, 2, 3] += 10
        self.unseen = self.root / "unseen"
        (self.unseen / "2005").mkdir(parents=True)
        for month in range(1, 13):
            chosen = TIMES.month == month
            xr.Dataset({"cape": (("valid_time", "latitude", "longitude"), cape[chosen])},
                       coords={"valid_time": TIMES[chosen], "latitude": LATITUDES, "longitude": LONGITUDES}
                       ).to_netcdf(self.unseen / "2005" / f"era5_independent_2005_{month:02d}.nc")

    def write_run(self, name, scores):
        run = self.root / name
        (run / "scores").mkdir(parents=True)
        np.save(run / "scores" / "anomaly_mean.npy", scores)
        np.save(run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (run / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")
        return run

    def test_the_part_only_the_score_flags_is_the_weather_of_its_event(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--events-dir", str(self.events), "--score-dir", str(self.runs[0]),
                                          "--by-dirs", str(self.runs[1]), "--unseen-dir", str(self.unseen),
                                          "--top-percent", "2", "--ring", "1"]), 0)
        text = output.getvalue()
        self.assertIn("events in part: 1, of the score alone: 1", text)
        rows = {line[:script.NAME].strip(): [float("nan" if value == "-" else value) for value in line[script.NAME:].split()]
                for line in text.splitlines()[3:] if line[:script.NAME].strip()}
        self.assertGreater(rows["events in part: the others flag"][1], 90)
        self.assertGreater(rows["events in part: score only"][1], 90)
        self.assertLess(rows["events in part: ring around"][1], 10)
        self.assertGreater(rows["events of the score alone"][1], 90)
        self.assertLess(rows["events of the score alone: ring"][1], 10)
        self.assertAlmostEqual(rows["all values"][1], 5, delta=2)

    def test_the_part_the_others_flag_later(self):
        # The event lasts the first half of its day; the others flag the cells only the score flags in the second
        # half, when the event is over: the score sees them first.
        first, second = PART_DAYS & (TIMES.hour < 12), PART_DAYS & (TIMES.hour >= 12)
        rng = np.random.default_rng(1)
        model, other = (rng.random((len(TIMES), 12)).astype(np.float32) * 0.1 for _ in range(2))
        model[np.ix_(first, BLOCK)] += 1
        model[ALONE_DAYS, CORNER] += 1
        other[first, 0] += 1
        other[np.ix_(second, [1, 4, 5])] += 1
        np.save(self.runs[0] / "scores" / "anomaly_mean.npy", model)
        np.save(self.runs[1] / "scores" / "anomaly_mean.npy", other)
        labels = np.load(self.events / "cluster_labels.npy")
        labels[np.ix_(second, [0, 1], [0, 1])] = 0
        np.save(self.events / "cluster_labels.npy", labels)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--events-dir", str(self.events), "--score-dir", str(self.runs[0]),
                                          "--by-dirs", str(self.runs[1]), "--top-percent", "3", "--ring", "1",
                                          "--wake-hours", "12"]), 0)
        text = output.getvalue()
        self.assertNotIn("unseen variable", text)
        block = text.split("within 12 h before and after")[1]
        rows = {line[:script.NAME].strip(): [float("nan" if value == "-" else value) for value in line[script.NAME:].split()]
                for line in block.splitlines()[2:] if line[:script.NAME].strip()}
        only = rows["events in part: score only"]
        self.assertGreater(only[2], 90)  # flagged by the others after
        self.assertGreater(only[2], only[1] + 10)  # more after than before
        self.assertLess(rows["events in part: ring around"][2], 20)
        with self.assertRaisesRegex(ValueError, "Give --unseen-dir, --wake-hours"):
            script.main(["--events-dir", str(self.events), "--score-dir", str(self.runs[0]), "--by-dirs", str(self.runs[1])])

    def test_the_low_values_of_a_variable_of_the_run(self):
        # The run kept its standardised data: the radiation is low over both events, on their days.
        context = 4
        rng = np.random.default_rng(2)
        standardised = rng.normal(size=(context + len(TIMES), 12, 2)).astype(np.float32)
        standardised[context:][np.ix_(PART_DAYS, BLOCK, [1])] -= 10
        standardised[context:][ALONE_DAYS, CORNER, 1] -= 10
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
        arguments = ["--events-dir", str(self.events), "--score-dir", str(self.runs[0]), "--by-dirs", str(self.runs[1]),
                     "--top-percent", "2", "--ring", "1"]
        with redirect_stdout(output):
            self.assertEqual(script.main([*arguments, "--run-dir", str(run), "--variables", "radiation", "--tail", "low"]), 0)
        text = output.getvalue()
        self.assertIn("is in its lowest 5%", text)
        rows = {line[:script.NAME].strip(): [float("nan" if value == "-" else value) for value in line[script.NAME:].split()]
                for line in text.splitlines()[3:] if line[:script.NAME].strip()}
        self.assertGreater(rows["events in part: score only"][1], 90)
        self.assertLess(rows["events in part: ring around"][1], 10)
        with self.assertRaisesRegex(ValueError, "not both"):
            script.main([*arguments, "--run-dir", str(run), "--variables", "radiation", "--unseen-dir", str(self.unseen)])
        with self.assertRaisesRegex(ValueError, "with --variables"):
            script.main([*arguments, "--run-dir", str(run)])

    def test_how_fast_the_weather_changes_in_the_parts(self):
        # The data jump at every step on the cell of the event of the score alone, on its days, and are still around.
        context = 4
        rng = np.random.default_rng(3)
        standardised = rng.normal(scale=0.1, size=(context + len(TIMES), 12, 2)).astype(np.float32)
        jumps = np.zeros(len(TIMES), dtype=bool)
        jumps[ALONE_DAYS] = True
        standardised[context:][jumps, CORNER, :] += np.where(np.arange(jumps.sum()) % 2, 5.0, -5.0)[:, None]
        cache = self.root / "fast"
        cache.mkdir()
        np.save(cache / "test.npy", standardised)
        (cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {"scale": [1, 1]}}), encoding="utf-8")
        run = self.root / "run_fast"
        run.mkdir()
        (run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": ["wind", "rain"]},
             "backend": {"seasonal_normalization": {"mean_file": str(cache / "seasonal_mean.npy")}}}), encoding="utf-8")
        arguments = ["--events-dir", str(self.events), "--score-dir", str(self.runs[0]), "--by-dirs", str(self.runs[1]),
                     "--top-percent", "2", "--ring", "1", "--run-dir", str(run)]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main([*arguments, "--change-hours", "3"]), 0)
        text = output.getvalue()
        self.assertIn("change in 3 h", text)
        rows = {line[:script.NAME].strip(): [float("nan" if value == "-" else value) for value in line[script.NAME:].split()]
                for line in text.splitlines()[3:] if line[:script.NAME].strip()}
        self.assertGreater(rows["events of the score alone"][1], 90)
        self.assertLess(rows["events of the score alone: ring"][1], 10)
        for wrong in (["--change-hours", "4"], ["--change-hours", "3", "--variables", "rain"]):
            with self.assertRaisesRegex(ValueError, "change-hours needs"):
                script.main([*arguments, *wrong])


if __name__ == "__main__":
    unittest.main()
