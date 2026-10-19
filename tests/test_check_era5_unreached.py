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


class UnreachedTests(unittest.TestCase):
    def test_one_event_before_a_gust(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
            model, normal = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
            block = np.isin(rows, (3, 4)) & (cols == 10)
            model[50:52, block] = 5  # unreached: the other score never comes near in time
            beside = (rows == 8) & (cols == 5)
            model[100, beside] = 5
            normal[102, beside] = 5  # reached 6 h later: not unreached
            normal[:, (rows == 10) & (cols == 25)] = 5  # the other score always flags a far cell
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
            # The gust is calm at the event and very strong 6 h after its start, at its cells.
            rng = np.random.default_rng(0)
            gust = 0.5 + 0.4 * rng.random(model.shape).astype(np.float32)
            gust[50:52, block] = 0
            gust[52, block] = 10
            (root / "unseen" / "2005").mkdir(parents=True)
            xr.Dataset({"i10fg": (("valid_time", "latitude", "longitude"), gust.reshape(-1, HEIGHT, WIDTH))},
                       coords={"valid_time": TIMES, "latitude": latitudes, "longitude": longitudes}
                       ).to_netcdf(root / "unseen" / "2005" / "era5_independent_2005_01.nc")
            csv = root / "unreached.csv"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]), "--top-percent", "5",
                                              "--unseen-dir", str(root / "unseen"), "--csv", str(csv)]), 0)
            text = output.getvalue()
            self.assertIn("unreached values: 4 (80.0% of the 5 values clim flags)", text)
            self.assertIn("events: 1,", text)
            table = pd.read_csv(csv)
            self.assertEqual((table.loc[0, "values"], table.loc[0, "duration_hours"]), (4, 6))
            line = next(line for line in text.splitlines() if line.startswith("event starts, i10fg"))
            shares = [float(value) for value in line.split()[-6:]]  # -6, -3, 0, 3, 6, 12 h
            self.assertEqual(shares[2], 0)
            self.assertEqual(shares[4], 100)


if __name__ == "__main__":
    unittest.main()
