"""The scores must be read in the core, at the edge and away from the events, by type and size of event."""
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

from scripts import check_era5_event_zones as script

HEIGHT = WIDTH = 12
CELLS = HEIGHT * WIDTH
CONTEXT = 4
TIMES = pd.date_range("2005-01-01", periods=60, freq="3h")
GRID = {"latitudes": (45 - np.arange(HEIGHT) * .5).tolist(), "longitudes": (7 + np.arange(WIDTH) * .5).tolist(),
        "rows": np.repeat(np.arange(HEIGHT), WIDTH).tolist(), "cols": np.tile(np.arange(WIDTH), HEIGHT).tolist()}
NAMES = ["temperature", "rain", "wind"]
cell = lambda row, column: row * WIDTH + column
BIG = ([10, 11, 12, 13], [cell(r, c) for r in (2, 3, 4) for c in (2, 3, 4)])  # 36 values: a rain event
SMALL = ([30], [cell(9, 9)])  # one value: a temperature event


def ring(cells):
    """The cells next to `cells` that are not among them."""
    rows, cols = np.divmod(np.asarray(cells), WIDTH)
    around = {(r + dr, c + dc) for r, c in zip(rows, cols) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    inside = set(zip(rows, cols))
    return [cell(r, c) for r, c in around - inside if 0 <= r < HEIGHT and 0 <= c < WIDTH]


class EventZonesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        self.inputs = (0.3 * rng.normal(size=(CONTEXT + len(TIMES), CELLS, 3))).astype(np.float32)
        self.events = np.zeros((len(TIMES), CELLS), dtype=np.float32)
        # A score high in the cores, lower at their edges, low elsewhere.
        self.score = rng.random((len(TIMES), CELLS)).astype(np.float32)
        for (steps, cells), variable in ((BIG, 1), (SMALL, 0)):
            for step in steps:
                self.events[step, cells] = 10
                self.inputs[CONTEXT + step, cells, variable] = 5
                self.score[step, ring(cells)] = 50
                self.score[step, cells] = 100
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

    def test_zones_by_type_and_size(self):
        events, score = self.write_run("events", self.events), self.write_run("score", self.score)
        csv = self.root / "zones.csv"
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--events", f"E={events}", "--scores", f"S={score}", "--inputs-run",
                                          str(self.inputs_run), "--far", "2", "--csv", str(csv)]), 0)
        table = pd.read_csv(csv, dtype={"size": str}).set_index(["type", "size"])
        largest, single = script.band_label(*script.SIZES[-1]), script.band_label(*script.SIZES[0])
        rain, temperature = table.loc[("rain", largest)], table.loc[("temperature", single)]
        self.assertEqual((rain["events"], temperature["events"]), (1, 1))
        self.assertEqual(rain["S core values"], 36)
        self.assertEqual(rain["S edge values"], 4 * 16)  # the ring of a 3 x 3 block, at 4 timestamps
        self.assertEqual(temperature["S edge values"], 8)
        self.assertGreater(rain["S core percentile"], 99)
        self.assertGreater(rain["S edge percentile"], 95)
        self.assertEqual(rain["S core flagged %"], 100)
        far = pd.read_csv(csv).set_index("type").loc["far from every event"]
        self.assertAlmostEqual(far["S core percentile"], 50, delta=8)
        # The highest 5% of S is mostly taken by the cores and edges (its threshold is read on 1 timestamp in 10,
        # two of which hold an event): far from them S flags less than 5%.
        self.assertTrue(0 < far["S core flagged %"] < 5)
        # Far: no cell within 2 cells of a core at that timestamp, i.e. outside 7 x 7 cells around the block at its
        # 4 timestamps and outside the 5 x 5 cells around (9, 9), cut by the border, at its timestamp.
        self.assertEqual(far["S core values"], len(TIMES) * CELLS - 4 * 49 - 25)
        self.assertIn("events of E", output.getvalue())

    def test_zones_at_one_timestamp(self):
        rows, cols = np.asarray(GRID["rows"]), np.asarray(GRID["cols"])
        labels = np.zeros(CELLS, dtype=np.int64)
        labels[[cell(5, 5), cell(5, 6)]] = 700  # one event, with a large global label
        labels[cell(0, 0)] = 3
        edge, away = script.zones_at(labels, rows, cols, (HEIGHT, WIDTH), far=1)
        self.assertEqual(set(np.nonzero(edge == 700)[0]), set(ring([cell(5, 5), cell(5, 6)])))
        self.assertEqual(set(np.nonzero(edge == 3)[0]), set(ring([cell(0, 0)])))
        self.assertFalse(away[cell(5, 7)])
        self.assertTrue(away[cell(5, 8)])

    def test_type_is_the_input_farthest_from_its_normal(self):
        steps, cells = np.array([0, 0, 1]), np.array([3, 4, 3])
        events = np.array([0, 0, 1])
        inputs = np.zeros((CONTEXT + 2, CELLS, 3))
        inputs[CONTEXT, [3, 4], 2] = -4  # event 0: the wind far below its normal
        inputs[CONTEXT + 1, 3, 0] = 3  # event 1: the temperature
        self.assertEqual(script.event_types(steps, cells, events, 2, inputs, CONTEXT).tolist(), [2, 0])


if __name__ == "__main__":
    unittest.main()
