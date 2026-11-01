"""The cases must follow the rule fixed in advance, and carry what is needed to look at them."""
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

from scripts import check_era5_event_cases as script

HEIGHT = WIDTH = 12
CELLS = HEIGHT * WIDTH
CONTEXT = 4
TIMES = pd.date_range("2005-01-01", periods=60, freq="3h")
GRID = {"latitudes": (45 - np.arange(HEIGHT) * .5).tolist(), "longitudes": (7 + np.arange(WIDTH) * .5).tolist(),
        "rows": np.repeat(np.arange(HEIGHT), WIDTH).tolist(), "cols": np.tile(np.arange(WIDTH), HEIGHT).tolist()}
NAMES = ["temperature", "rain", "wind"]
cell = lambda row, column: row * WIDTH + column
LARGEST = ([10, 11, 12, 13], [cell(r, c) for r in (1, 2, 3) for c in (1, 2, 3)])  # 36 values of rain
SMALLER = ([30, 31, 32, 33], [cell(r, c) for r in (8, 9) for c in (8, 9, 10)])  # 24 values of rain
WIND = ([45, 46, 47], [cell(r, c) for r in (8, 9) for c in (1, 2, 3, 4)])  # 24 values of wind
SINGLE = ([50], [cell(0, 11)])  # one value of temperature


class ChooseCasesTests(unittest.TestCase):
    def test_the_rule(self):
        sizes = np.array([100, 30, 40, 5, 500, 25, 60])
        kinds = np.array([1, 1, 1, 1, 0, 0, 2])
        # Types by values in the large events: 0 (525), 1 (170); 2 (60) is left out with two types.
        self.assertEqual(script.choose_cases(sizes, kinds, 2, 20), [(0, 4), (1, 0)])

    def test_a_type_with_no_large_event_has_no_case(self):
        self.assertEqual(script.choose_cases(np.array([30, 3]), np.array([0, 1]), 3, 20), [(0, 0)])

    def test_peak_is_the_earliest_of_the_fullest(self):
        self.assertEqual(script.peak_of(np.array([4, 4, 7, 7, 9])), 4)


class EventCasesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        self.inputs = (0.3 * rng.normal(size=(CONTEXT + len(TIMES), CELLS, 3))).astype(np.float32)
        self.events = np.zeros((len(TIMES), CELLS), dtype=np.float32)
        self.other = rng.random((len(TIMES), CELLS)).astype(np.float32)
        for (steps, cells), variable in ((LARGEST, 1), (SMALLER, 1), (WIND, 2), (SINGLE, 0)):
            for step in steps:
                self.events[step, cells] = 10
                self.inputs[CONTEXT + step, cells, variable] = 5
        cache = self.root / "cache"
        cache.mkdir()
        np.save(cache / "test.npy", self.inputs)
        self.inputs_run = self.root / "model"
        self.inputs_run.mkdir()
        (self.inputs_run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": NAMES},
             "backend": {"seasonal_normalization": {"mean_file": str(cache / "seasonal_mean.npy")}}}), encoding="utf-8")

    def write_run(self, name, values):
        run = self.root / name
        (run / "scores").mkdir(parents=True)
        np.save(run / "scores" / "anomaly_mean.npy", values)
        np.save(run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "anomaly_mean_file": "scores/anomaly_mean.npy"}), encoding="utf-8")
        return run

    def test_groups_and_cases(self):
        events = self.write_run("events", self.events)
        other = self.write_run("other", self.other)
        output = self.root / "cases"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(script.main(["--events", f"E={events}", "--scores", f"E={events}", f"O={other}",
                                          "--inputs-run", str(self.inputs_run), "--output-dir", str(output),
                                          "--far", "2", "--around", "2"]), 0)
        groups = pd.read_csv(output / "event_groups.csv", dtype={"size": str}).set_index(["type", "size"])
        rain = groups.loc[("rain", "20-99")]
        self.assertEqual((rain["events"], rain["values"]), (2, 36 + 24))
        self.assertEqual(groups.loc[("rain", "all"), "values"], 60)
        self.assertEqual(groups.loc[("temperature", "1"), "events"], 1)
        self.assertEqual(groups.loc[("all", "all"), "values"], 36 + 24 + 24 + 1)

        cases = pd.read_csv(output / "cases.csv")
        self.assertEqual(cases["type"].tolist(), ["rain", "wind"])
        self.assertEqual(cases["values"].tolist(), [36, 24])
        self.assertEqual(cases["timestamps"].tolist(), [4, 3])
        self.assertEqual(cases["timestamp"].tolist(), [str(TIMES[10]), str(TIMES[45])])

        with np.load(output / "case_00.npz") as case:
            self.assertEqual(json.loads(str(case["case"]))["type"], "rain")
            self.assertEqual(set(np.nonzero(case["core"])[0]), set(LARGEST[1]))
            self.assertEqual(case["edge"].sum(), 16)
            self.assertFalse(case["far"][cell(4, 4)])
            self.assertTrue(case["far"][cell(10, 10)])
            self.assertEqual(case["input"][cell(2, 2)], 5)
            np.testing.assert_array_equal(case["score_O"], self.other[10])
            self.assertEqual(case["event_cells"].tolist(), [9, 9, 9, 9])
            self.assertEqual(len(case["event_times"]), 4)
            self.assertEqual(len(case["series_times"]), 5)
            self.assertEqual(case["series_core"].tolist(), [False, False, True, True, True])
            self.assertEqual(len(case["series_O"]), 5)
            self.assertEqual(len(case["quantiles_O"]), len(script.PERCENTILES))


if __name__ == "__main__":
    unittest.main()
