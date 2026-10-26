"""The unreached check must find what only the model flags far in time from the others, and the weather after it."""
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

from scripts import check_era5_unreached as script

TIMES = pd.date_range("2005-01-01", periods=200, freq="3h")
HEIGHT, WIDTH = 12, 30
ROWS, COLS = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
LATITUDES, LONGITUDES = list(60 - 0.5 * np.arange(HEIGHT)), list(0.5 * np.arange(WIDTH))
BLOCK = np.isin(ROWS, (3, 4)) & (COLS == 10)  # the unreached event: two cells, two timestamps (50, 51)
BESIDE = (ROWS == 8) & (COLS == 5)  # flagged by the model at 100, reached by the other score 6 h later


def write_runs(root, metadata=None):
    """The model's run and the other score's run; `metadata` is added to the model's metadata.json."""
    model, normal = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
    model[50:52, BLOCK] = 5  # unreached: the other score never comes near in time
    model[100, BESIDE] = 5
    normal[102, BESIDE] = 5  # reached 6 h later: not unreached
    normal[:, (ROWS == 10) & (COLS == 25)] = 5  # the other score always flags a far cell
    grid = {"latitudes": LATITUDES, "longitudes": LONGITUDES, "rows": ROWS.tolist(), "cols": COLS.tolist(),
            "spacing": 0.5}
    runs = []
    for name, scores, extra in (("run_clim", model, metadata or {}), ("run_normal", normal, {})):
        (root / name / "scores").mkdir(parents=True)
        np.save(root / name / "scores" / "anomaly_mean.npy", scores)
        np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid, **extra}),
                                                   encoding="utf-8")
        runs.append(root / name)
    return runs, model.shape


def run(argv):
    output = io.StringIO()
    with redirect_stdout(output):
        assert script.main(argv) == 0
    return output.getvalue()


class UnreachedTests(unittest.TestCase):
    def test_one_event_before_a_gust(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs, shape = write_runs(root)
            # The gust is calm at the event and very strong 6 h after its start, at its cells.
            rng = np.random.default_rng(0)
            gust = 0.5 + 0.4 * rng.random(shape).astype(np.float32)
            gust[50:52, BLOCK] = 0
            gust[52, BLOCK] = 10
            (root / "unseen" / "2005").mkdir(parents=True)
            xr.Dataset({"i10fg": (("valid_time", "latitude", "longitude"), gust.reshape(-1, HEIGHT, WIDTH))},
                       coords={"valid_time": TIMES, "latitude": LATITUDES, "longitude": LONGITUDES}
                       ).to_netcdf(root / "unseen" / "2005" / "era5_independent_2005_01.nc")
            csv = root / "unreached.csv"
            text = run(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]), "--top-percent", "5",
                        "--unseen-dir", str(root / "unseen"), "--csv", str(csv)])
            self.assertIn("unreached values: 4 (80.0% of the 5 values clim flags)", text)
            self.assertIn("events: 1,", text)
            table = pd.read_csv(csv)
            self.assertEqual((table.loc[0, "values"], table.loc[0, "duration_hours"]), (4, 6))
            line = next(line for line in text.splitlines() if line.startswith("event starts, i10fg"))
            shares = [float(value) for value in line.split()[-6:]]  # -6, -3, 0, 3, 6, 12 h
            self.assertEqual(shares[2], 0)
            self.assertEqual(shares[4], 100)

    def test_what_the_error_of_the_unreached_values_is_made_of(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Weighted, b is the main variable of the unreached event (1.2 > 1), a of the reached value.
            runs, shape = write_runs(root, {"preparation": {"features": ["a", "b"]},
                                            "backend": {"reconstruction_feature_weights": {"a": 1.0, "b": 2.0}}})
            errors = np.full((*shape, 2), 0.01, dtype=np.float32)
            errors[50:52, BLOCK] = (1.0, 0.6)
            errors[100, BESIDE] = (3.0, 0.1)
            np.save(runs[0] / "scores" / "feature_scores.npy", errors)
            text = run(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]), "--top-percent", "5",
                        "--feature-errors-dir", str(runs[0])])
            self.assertIn("with the weights of the run", text)
            rows = {line.split()[0]: [float(value) for value in line.split()[1:]] for line in text.splitlines()
                    if line.split() and line.split()[0] in ("a", "b")}
            # unreached share, all-flags share, ratio, main unreached, main all flags, ratio
            self.assertAlmostEqual(rows["b"][0], 100 * 4.8 / 8.8, places=1)
            self.assertAlmostEqual(rows["b"][1], 100 * 5.0 / 12.0, places=1)
            self.assertEqual((rows["b"][3], rows["b"][4]), (100.0, 80.0))
            self.assertEqual((rows["a"][3], rows["a"][4]), (0.0, 20.0))
            self.assertIn("events of two values or more (1): all values with the same main variable 100.0%", text)


if __name__ == "__main__":
    unittest.main()
