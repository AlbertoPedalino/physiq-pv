"""The event check must say which score flags an event first, by how much and how far, treating both alike."""
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

from scripts import check_era5_event_first as script

TIMES = pd.date_range("2005-01-01", periods=200, freq="3h")
HEIGHT, WIDTH = 12, 30


class EventFirstTests(unittest.TestCase):
    def test_who_is_first(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
            model, normal = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
            near = lambda column, chosen=(2, 3): np.isin(rows, chosen) & (cols == column)
            # A: the model flags column 5 from step 20, the other column 6 from step 21: model first by 3 h, 1 cell.
            model[20:22, near(5)] = 5
            normal[21:23, near(6)] = 5
            # B: the other flags column 15 from step 50, the model column 16 at step 52: other first by 6 h.
            normal[50:53, near(15)] = 5
            model[52, near(16)] = 5
            # C: both flag the same cell at step 80. D: only the model, at step 120.
            model[80, near(25, (8,))] = normal[80, near(25, (8,))] = 5
            model[120, near(10, (8,))] = 5
            latitudes, longitudes = list(60 - 0.5 * np.arange(HEIGHT)), list(0.5 * np.arange(WIDTH))
            grid = {"latitudes": latitudes, "longitudes": longitudes, "rows": rows.tolist(), "cols": cols.tolist(),
                    "spacing": 0.5}
            runs = []
            for name, scores in (("run_clim", model), ("run_normal", normal)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", scores)
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}),
                                                           encoding="utf-8")
                runs.append(root / name)
            csv = root / "first.csv"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]), "--top-percent", "5",
                                              "--csv", str(csv)]), 0)
            self.assertIn("events: 4; flagged by both 3, only by clim 1, only by normal 0", output.getvalue())
            table = pd.read_csv(csv, parse_dates=["first_time"]).sort_values("first_time")
            self.assertEqual(table.lead_hours.tolist(), [3, -6, 0])
            self.assertEqual(table.distance_cells.tolist(), [1, 1, 0])
            self.assertEqual(table["values"].tolist(), [8, 8, 1])


if __name__ == "__main__":
    unittest.main()
