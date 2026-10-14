"""The lead check must say which score starts an episode first at a cell, and by how many hours."""
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

from scripts import check_era5_lead as script

TIMES = pd.date_range("2005-01-01", periods=400, freq="3h")


class LeadTests(unittest.TestCase):
    def test_groups_by_hand(self):
        first = np.zeros((40, 5), dtype=bool)
        second = np.zeros((40, 5), dtype=bool)
        first[20:24] = True
        second[18:24, 0] = True          # cell 0: the second already flags, 2 steps before
        second[16, 1] = True             # cell 1: it flagged 4 steps before, then stopped
        second[20:22, 2] = True          # cell 2: together
        second[23, 3] = True             # cell 3: 3 steps later
        before, last, after, count = script.starts_and_other(first, second, quiet=8, window=8)
        self.assertEqual(count, 5)
        result = script.table(before, last, after, 8, 3)
        self.assertEqual((result["already"], result["gap"], result["together"], result["later"], result["none"]),
                         (20.0, 20.0, 20.0, 20.0, 20.0))
        self.assertEqual((result["lead_median_h"], result["lag_median_h"]), (6.0, 9.0))
        self.assertEqual(result["bands"][1], 20.0)  # 6 h
        # A flag after fewer quiet steps than required is not a start.
        first[26] = True
        self.assertEqual(script.starts_and_other(first, second, quiet=8, window=8)[3], 5)

    def test_the_second_score_anticipates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Zero outside the episodes, so that the highest 30% flags the episodes and nothing else.
            model, normal = (np.zeros((len(TIMES), 30), dtype=np.float32) for _ in range(2))
            for start in range(30, 380, 40):  # episodes: the model rises 6 hours before the normal
                model[start - 2:start + 4] = 5
                normal[start:start + 4] = 5
            runs = []
            for name, scores in (("run_clim", model), ("run_normal", normal)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", scores)
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (root / name / "metadata.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
                runs.append(root / name)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]),
                                              "--top-percent", "30"]), 0)
            rows = [line for line in output.getvalue().splitlines() if line.startswith("starts of")]
            values = [[float(value) for value in line[script.NAME:].split()] for line in rows]
            # Starts of the normal: the model is already flagging, by 6 hours. Starts of the model: the normal later.
            self.assertGreater(values[1][1], 90)
            self.assertEqual(values[1][7], 6.0)
            self.assertLess(values[0][1], 10)
            self.assertGreater(values[0][10], 90)


if __name__ == "__main__":
    unittest.main()
