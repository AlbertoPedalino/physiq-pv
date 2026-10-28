"""The events of a score must be split by type and size, and the other scores read on each kind."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_event_types_sizes as script
from scripts.check_era5_edge_event_types import SIGNALS, TYPES

HEIGHT = WIDTH = 8
CELLS = HEIGHT * WIDTH
TIMES = pd.date_range("2005-01-01", periods=480, freq="3h")  # January to March
GRID = {"latitudes": (45 - np.arange(HEIGHT) * .5).tolist(), "longitudes": (7 + np.arange(WIDTH) * .5).tolist(),
        "rows": np.repeat(np.arange(HEIGHT), WIDTH).tolist(), "cols": np.tile(np.arange(WIDTH), HEIGHT).tolist()}
cell = lambda row, column: row * WIDTH + column
# Four events, far apart: (timestamps, cells, signals present on all their values).
EVENTS = {
    "wind, 1": ([10], [cell(0, 0)], ["wind"]),
    "heat, 20+": ([100, 101, 102], [cell(r, c) for r in (3, 4, 5) for c in (3, 4, 5)], ["heat"]),
    "mixed, 2-4": ([300], [cell(0, 6), cell(0, 7), cell(1, 6), cell(1, 7)], ["wind", "rain"]),
    "other, 2-4": ([400, 401, 402], [cell(7, 0)], []),
}


class EventTypesSizesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        self.present = np.zeros((len(TIMES), CELLS, len(SIGNALS)), dtype=bool)
        self.present[rng.random(self.present.shape) < 0.01] = True  # rare signals everywhere
        self.events_score = np.zeros((len(TIMES), CELLS), dtype=np.float32)
        self.finder = rng.random((len(TIMES), CELLS)).astype(np.float32) * 0.5
        for name, (steps, cells, present) in EVENTS.items():
            for step in steps:
                self.events_score[step, cells] = 10
                self.present[step, cells] = False
                for signal in present:
                    self.present[step, cells, SIGNALS.index(signal)] = True
                # The other score flags the wind and heat events, and none of the other two.
                self.finder[step, cells] = 10 if name.startswith(("wind", "heat")) else 0

    def write_run(self, name, values):
        run = self.root / name
        (run / "scores").mkdir(parents=True)
        np.save(run / "scores" / "anomaly_mean.npy", values)
        np.save(run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "anomaly_mean_file": "scores/anomaly_mean.npy"}), encoding="utf-8")
        return run

    def signals_of(self, positions):
        return self.present[positions]

    def test_types_and_sizes_of_the_events(self):
        flagged = self.events_score > 1
        layout = (np.asarray(GRID["rows"]), np.asarray(GRID["cols"]), (HEIGHT, WIDTH))
        described = script.describe(flagged, TIMES, layout, self.signals_of, {}, ratio=2.0)
        self.assertEqual(sorted(described["sizes"].tolist()), [1, 3, 4, 27])
        kinds = {int(size): TYPES[kind] for size, kind in zip(described["sizes"], described["types"])}
        self.assertEqual(kinds, {1: "wind", 27: "heat", 4: "mixed", 3: "other"})
        self.assertEqual(script.size_bands(np.array([1, 2, 4, 5, 19, 20, 500])).tolist(), [0, 1, 1, 2, 2, 3, 3])

    def test_the_other_scores_on_each_kind(self):
        events = self.write_run("events", self.events_score)
        finder = self.write_run("finder", self.finder)
        csv = self.root / "table.csv"
        output = io.StringIO()
        with patch.object(script, "prepared_signals", lambda *args: (self.signals_of, None)), redirect_stdout(output):
            self.assertEqual(script.main(["--events", f"E={events}", "--others", f"F={finder}", "--top-percent", "5",
                                          "--prepared-dir", str(self.root), "--unseen-dir", str(self.root),
                                          "--csv", str(csv)]), 0)
        table = pd.read_csv(csv)
        by_type = table[table["size"] == "all"].set_index("type")
        self.assertEqual(by_type.loc["all", "events"], 4)
        self.assertEqual(by_type.loc["wind", "F found %"], 100)
        self.assertEqual(by_type.loc["heat", "F found %"], 100)
        self.assertEqual(by_type.loc["mixed", "F found %"], 0)
        self.assertEqual(by_type.loc["other", "F found %"], 0)
        self.assertGreater(by_type.loc["heat", "F percentile"], 99)
        self.assertLess(by_type.loc["other", "F percentile"], 1)
        self.assertAlmostEqual(by_type.loc["all", "values %"], 100)
        self.assertAlmostEqual(by_type.loc["heat", "values %"], 100 * 27 / 35)
        by_size = table[table["type"] == "all"].set_index("size")
        self.assertEqual(by_size.loc["2-4", "events"], 2)
        self.assertIn("4 events", output.getvalue())


if __name__ == "__main__":
    unittest.main()
