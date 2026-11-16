"""The profile of a score across its events: position from the border of the core, by type."""
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

from scripts import check_era5_event_profile as script

HEIGHT = WIDTH = 20
CELLS = HEIGHT * WIDTH
CONTEXT = 4
TIMES = pd.date_range("2005-01-01", periods=40, freq="3h")
GRID = {"latitudes": (45 - np.arange(HEIGHT) * .5).tolist(), "longitudes": (7 + np.arange(WIDTH) * .5).tolist(),
        "rows": np.repeat(np.arange(HEIGHT), WIDTH).tolist(), "cols": np.tile(np.arange(WIDTH), HEIGHT).tolist()}
NAMES = ["temperature", "rain", "wind"]
cell = lambda row, column: row * WIDTH + column
BLOCK = [cell(r, c) for r in range(5, 10) for c in range(5, 10)]  # 5 x 5: its centre is 3 cells deep


class PositionsTests(unittest.TestCase):
    def test_positions_at_one_timestamp(self):
        labels = np.zeros((9, 9), dtype=np.int64)
        labels[3:6, 3:6] = 7
        position, event = script.positions_at(labels, limit=3)
        self.assertEqual(position[4, 4], -2)  # the centre of a 3 x 3 block
        self.assertEqual(position[3, 3], -1)
        self.assertEqual(position[2, 2], 1)  # diagonal neighbour
        self.assertEqual(position[1, 4], 2)
        self.assertEqual(position[0, 0], 3)  # 3 cells away
        self.assertTrue((event == 7).all())

    def test_nothing_flagged(self):
        self.assertIsNone(script.positions_at(np.zeros((4, 4), dtype=np.int64), 3))

    def test_a_cell_outside_belongs_to_the_nearest_event(self):
        labels = np.zeros((1, 9), dtype=np.int64)
        labels[0, 0], labels[0, 8] = 1, 2
        _, event = script.positions_at(labels, 8)
        self.assertEqual(event[0, 1], 1)
        self.assertEqual(event[0, 7], 2)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        self.inputs = (0.1 * rng.normal(size=(CONTEXT + len(TIMES), CELLS, 3))).astype(np.float32)
        self.events = np.zeros((len(TIMES), CELLS), dtype=np.float32)
        self.score = rng.random((len(TIMES), CELLS)).astype(np.float32)
        for step in (10, 11):
            self.events[step, BLOCK] = 10
            self.inputs[CONTEXT + step, BLOCK, 1] = 5
            self.score[step, BLOCK] = 100
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

    def test_profile(self):
        events, score = self.write_run("events", self.events), self.write_run("score", self.score)
        csv = self.root / "profile.csv"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(script.main(["--events", f"E={events}", "--scores", f"S={score}", "--inputs-run",
                                          str(self.inputs_run), "--limit", "4", "--csv", str(csv)]), 0)
        table = pd.read_csv(csv).set_index(["type", "position"])
        rain = table.loc["rain"]
        # At the two timestamps of the event: a 5 x 5 block has 16, 8 and 1 cells at depths 1, 2, 3.
        self.assertEqual(rain.loc[-1, "values"], 2 * 16)
        self.assertEqual(rain.loc[-2, "values"], 2 * 8)
        self.assertEqual(rain.loc[-3, "values"], 2 * 1)
        self.assertEqual(rain.loc[1, "values"], 2 * 24)  # the ring around it
        self.assertEqual(rain.loc[-1, "variable distance"], 5.025)
        self.assertLess(rain.loc[1, "variable distance"], 0.5)
        self.assertGreaterEqual(rain.loc[-1, "S percentile"], 99)
        self.assertLess(rain.loc[4, "S percentile"], 70)
        self.assertEqual(table.loc[("all", -1), "values"], 2 * 16)
        self.assertNotIn("temperature", table.index.get_level_values(0))

    def test_profile_around_the_concordant_clusters(self):
        # A cube of votes: a short cluster (the 5 x 5 block at two timestamps, 50 values), a long one (a 5 x 5 block
        # kept 30 timestamps, 87 h) and a small one (one cell at one timestamp).
        votes = np.zeros((len(TIMES), CELLS), dtype=np.float32)
        votes[10:12, BLOCK] = 6
        far_block = [cell(r, c) for r in range(13, 18) for c in range(13, 18)]
        votes[5:35, far_block] = 5
        votes[2, cell(0, 19)] = 9
        votes[3, cell(0, 0)] = 4  # below the votes needed: not concordant
        cube = self.root / "cube"
        run = self.write_run("cube/votes", votes)
        score = self.write_run("score", self.score)
        (cube / "summary.json").write_text(json.dumps(
            {"vote": {"votes_needed": 5, "detectors": {"S": f"{score}@combined"}},
             "selection": {"min_values": 20, "max_hours": 72}}), encoding="utf-8")
        self.assertTrue(run.exists())
        csv = self.root / "profile_cube.csv"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(script.main(["--cube", str(cube), "--inputs-run", str(self.inputs_run), "--limit", "4",
                                          "--csv", str(csv)]), 0)
        table = pd.read_csv(csv).set_index(["type", "position"])
        self.assertEqual(table.loc[("short", -1), "values"], 2 * 16)
        self.assertEqual(table.loc[("long", -1), "values"], 30 * 16)
        self.assertEqual(table.loc[("small", -1), "values"], 1)
        self.assertGreaterEqual(table.loc[("short", -1), "S percentile"], 99)
        self.assertIn("S q25", table.columns)

    def test_a_cube_closed_after_the_vote_is_refused(self):
        cube = self.root / "cube_after"
        self.write_run("cube_after/votes", np.zeros((len(TIMES), CELLS), dtype=np.float32))
        (cube / "summary.json").write_text(json.dumps(
            {"vote": {"votes_needed": 5, "detectors": {}, "closure": {"where": "after"}},
             "selection": {"min_values": 20, "max_hours": 72}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            script.main(["--cube", str(cube), "--inputs-run", str(self.inputs_run), "--csv", str(self.root / "y.csv")])

    def test_events_or_cube(self):
        with self.assertRaises(ValueError):
            script.main(["--inputs-run", str(self.inputs_run), "--csv", str(self.root / "x.csv")])


if __name__ == "__main__":
    unittest.main()
