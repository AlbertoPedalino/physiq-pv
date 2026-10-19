"""The spatial lead check must say whether a score flags the cells around a place before an episode starts there."""
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

from scripts import check_era5_spatial_lead as script

TIMES = pd.date_range("2005-01-01", periods=300, freq="3h")
HEIGHT, WIDTH = 12, 30


class RingTests(unittest.TestCase):
    def test_rings_by_brute_force(self):
        maps = np.random.default_rng(0).random((3, 9, 11)) > 0.93
        result = script.ring_flags(maps, script.RINGS)
        for r, (low, high) in enumerate(script.RINGS):
            for t in range(3):
                for y in range(9):
                    for x in range(11):
                        expected = any(maps[t, y + dy, x + dx]
                                       for dy in range(-high, high + 1) for dx in range(-high, high + 1)
                                       if low <= max(abs(dy), abs(dx)) and 0 <= y + dy < 9 and 0 <= x + dx < 11)
                        self.assertEqual(bool(result[r, t, y, x]), expected, (r, t, y, x))

    def test_starts_need_quiet_steps(self):
        first = np.zeros((30, 2), dtype=bool)
        first[10:12, 0] = True
        first[15, 0] = True  # only 3 steps after the last flag: not a start
        first[20, 1] = True
        positions = script.starts(first, quiet=8)
        self.assertEqual([(step, list(where)) for step, where in enumerate(positions) if len(where)],
                         [(10, [0]), (20, [1])])


class SpatialLeadTests(unittest.TestCase):
    def test_a_front_seen_upwind(self):
        # A front moves one column east every 3 hours; the model flags it 3 columns ahead of the normal,
        # that is 9 hours before the normal reaches a cell. Every 40 steps a new front crosses the grid.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
            model, normal = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
            for begin in range(10, len(TIMES) - WIDTH, 40):
                for step in range(WIDTH + 3):
                    for score, column in ((model, step), (normal, step - 3)):
                        if 0 <= column < WIDTH and begin + step < len(TIMES):
                            score[begin + step, cols == column] = 5
            grid = {"latitudes": list(60 - 0.5 * np.arange(HEIGHT)), "longitudes": list(0.5 * np.arange(WIDTH)),
                    "rows": rows.tolist(), "cols": cols.tolist(), "spacing": 0.5}
            runs = []
            for name, scores in (("run_clim", model), ("run_normal", normal)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", scores)
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}),
                                                           encoding="utf-8")
                runs.append(root / name)
            csv = root / "lead.csv"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]),
                                              "--top-percent", "10", "--csv", str(csv)]), 0)
            table = pd.read_csv(csv)
            pick = lambda starts_of, precursor, low, lag: table[
                (table.starts_of == starts_of) & (table.precursor == precursor)
                & (table.ring_low == low) & (table.lag_hours == lag)].iloc[0]
            # Starts of the normal at column c: the model flags column c+3 at the start (ring 3-5), column c+2 three
            # hours before and c+1 six hours before (ring 1-2); the last 3 columns have nothing east of them.
            for low, lag in ((3, 0), (1, 3), (1, 6)):
                row = pick("normal", "clim", low, lag)
                self.assertGreater(row.share, 85, (low, lag))
                self.assertLess(row.chance, 40, (low, lag))
            self.assertGreater(pick("normal", "clim not normal", 3, 0).share, 85)
            # The cell itself is flagged by the model only 9 h before, a lag that is not read.
            self.assertLess(pick("normal", "clim", 0, 6).share, 1)
            self.assertEqual(pick("normal", "normal (itself)", 0, 3).share, 0)
            # The other way round: before a start of the model the normal has not reached the cell yet.
            self.assertLess(pick("clim", "normal", 0, 3).share, 1)
            self.assertGreater(pick("clim", "normal", 3, 3).share, 85)


if __name__ == "__main__":
    unittest.main()
