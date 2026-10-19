"""The direction check must tell an edge on the side an event moves to from one on the side it leaves."""
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

from scripts import check_era5_edge_direction as script

TIMES = pd.date_range("2005-01-01", periods=120, freq="3h")
HEIGHT, WIDTH = 12, 30


def run(root, offset):
    """A 3 x 3 block flagged by the other score moves one column east every step; the model flags the column
    `offset` columns from its centre. Returns the table of the check."""
    rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
    model, normal = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
    for step in range(len(TIMES)):
        centre = 3 + step % 24
        normal[step, np.isin(rows, (4, 5, 6)) & (np.abs(cols - centre) <= 1)] = 5
        model[step, np.isin(rows, (4, 5, 6)) & (cols == centre + offset)] = 5
    latitudes, longitudes = list(60 - 0.5 * np.arange(HEIGHT)), list(0.5 * np.arange(WIDTH))
    grid = {"latitudes": latitudes, "longitudes": longitudes, "rows": rows.tolist(), "cols": cols.tolist(), "spacing": 0.5}
    runs = []
    for name, scores in ((f"run_clim_{offset}", model), (f"run_normal_{offset}", normal)):
        (root / name / "scores").mkdir(parents=True)
        np.save(root / name / "scores" / "anomaly_mean.npy", scores)
        np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}), encoding="utf-8")
        runs.append(root / name)
    csv = root / f"direction_{offset}.csv"
    with redirect_stdout(io.StringIO()):
        script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]), "--top-percent", "5", "--csv", str(csv)])
    return pd.read_csv(csv)


class EdgeDirectionTests(unittest.TestCase):
    def test_front_and_back(self):
        with tempfile.TemporaryDirectory() as temp:
            pick = lambda table, edge, target, hours, column: table[
                (table.edge == edge) & (table.target == target) & (table.hours == hours)].iloc[0][column]
            front, back = run(Path(temp), 2), run(Path(temp), -2)
            for target in ("shift", "growth"):
                self.assertEqual(pick(front, "first only", target, 3, "same_side"), 100)
                self.assertGreater(pick(front, "first only", target, 3, "mean_cosine"), 0.99)
                self.assertEqual(pick(back, "first only", target, 3, "same_side"), 0)
            # The ring only has a direction where the border of the grid cuts it.
            self.assertLess(pick(front, "ring", "shift", 3, "events"), pick(front, "first only", "shift", 3, "events"))


if __name__ == "__main__":
    unittest.main()
