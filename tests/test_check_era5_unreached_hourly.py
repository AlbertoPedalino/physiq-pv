"""The hourly check must read how long and how abruptly the weather of the unreached values changes."""
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

from scripts import check_era5_unreached_hourly as script

TIMES = pd.date_range("2005-01-01", periods=200, freq="3h")
HOURS = pd.date_range("2005-01-01", "2005-01-31 23:00", freq="1h")
HEIGHT, WIDTH = 12, 30


class UnreachedHourlyTests(unittest.TestCase):
    def test_a_one_hour_gust_at_the_unreached_event(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
            model, normal = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
            block = np.isin(rows, (3, 4)) & (cols == 10)
            model[50:52, block] = 5  # unreached: the other score never comes near in time
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
            # The gust is ordinary everywhere and very strong for one hour at the event's first timestamp only.
            gust = 0.5 + 0.4 * np.random.default_rng(0).random((len(HOURS), HEIGHT * WIDTH)).astype(np.float32)
            hour = HOURS.get_loc(TIMES[50])
            gust[hour - 3:hour + 7, block] = 0.6
            gust[hour, block] = 10
            (root / "hourly" / "2005").mkdir(parents=True)
            xr.Dataset({"i10fg": (("valid_time", "latitude", "longitude"), gust.reshape(-1, HEIGHT, WIDTH))},
                       coords={"valid_time": HOURS, "latitude": latitudes, "longitude": longitudes}
                       ).to_netcdf(root / "hourly" / "2005" / "era5_hourly_check_2005_01.nc")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]),
                                              "--hourly-dir", str(root / "hourly")]), 0)
            lines = output.getvalue().splitlines()
            self.assertIn("unreached values 4", lines[1])

            def row(title, after):
                start = next(i for i, line in enumerate(lines) if line.startswith(title))
                line = next(line for line in lines[start:] if line.startswith(after))
                return [float(value) for value in line[len(after):].split()]

            unusual = row("% of the values with the variable in its highest", "unreached values, i10fg")
            # Two of the four values are the gust's hour; the other two see it 3 hours before them.
            self.assertEqual(unusual, [50, 0, 0, 50, 0, 0, 0])
            lengths = row("hours in a row", "unreached values, i10fg")
            self.assertEqual(lengths[:2], [100, 0])  # one hour only
            ratio, near, count = row("abrupt change", "unreached values, i10fg")
            self.assertAlmostEqual(ratio, 0.5, places=3)  # up in one hour, down in the next
            self.assertEqual((near, count), (100, 2))


if __name__ == "__main__":
    unittest.main()
