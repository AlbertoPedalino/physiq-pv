"""The lead check against unseen variables must say whether a flag that comes first is on unusual weather."""
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

from scripts import check_era5_lead_unseen as script

TIMES = pd.date_range("2005-01-01", "2006-12-31 21:00", freq="3h")
LATITUDES, LONGITUDES = [45.0, 44.5], [10.0, 10.5, 11.0]
GRID = {"latitudes": LATITUDES, "longitudes": LONGITUDES, "rows": [0, 0, 0, 1, 1, 1], "cols": [0, 1, 2, 0, 1, 2],
        "spacing": 0.5}
# On day 10 of every month the gust is strong at 06 and 09 UTC; the model flags both, the other score only 09:
# the model sees it 3 h first. On day 20 at 12 UTC the other score flags weather that is not unusual.
# Two days a month of two years are 2 of the 62 values of a cell, month and time of day: unusual at 5%.
STORM = (TIMES.day == 10) & np.isin(TIMES.hour, (6, 9))
LATE = (TIMES.day == 10) & (TIMES.hour == 9)
CALM = (TIMES.day == 20) & (TIMES.hour == 12)


class LeadUnseenTests(unittest.TestCase):
    def test_a_flag_that_comes_first_on_a_gust(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(0)
            gust = rng.random((len(TIMES), 6)).astype(np.float32)
            gust[STORM] += 10
            cape = rng.random((len(TIMES), 6)).astype(np.float32)  # follows no score
            for year in (2005, 2006):
                for month in range(1, 13):
                    chosen = (TIMES.year == year) & (TIMES.month == month)
                    (root / "unseen" / str(year)).mkdir(parents=True, exist_ok=True)
                    xr.Dataset({name: (("valid_time", "latitude", "longitude"), values[chosen].reshape(-1, 2, 3))
                                for name, values in (("i10fg", gust), ("cape", cape))},
                               coords={"valid_time": TIMES[chosen], "latitude": LATITUDES, "longitude": LONGITUDES}
                               ).to_netcdf(root / "unseen" / str(year) / f"era5_independent_{year}_{month:02d}.nc")
            runs = []
            for name, flagged in (("run_clim", STORM), ("run_normal", LATE | CALM)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", np.repeat(flagged[:, None], 6, axis=1).astype(np.float32))
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}),
                                                           encoding="utf-8")
                runs.append(root / name)
            csv = root / "lead_unseen.csv"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(script.main(["--unseen-dir", str(root / "unseen"), "--score-dir", str(runs[0]),
                                              "--by-dir", str(runs[1]), "--top-percent", "2", "--csv", str(csv)]), 0)
            table = pd.read_csv(csv)
            pick = lambda starts_of, group, variable, hours: table[
                (table.starts_of == starts_of) & ~table.chance & (table.group == group)
                & (table.variable == variable) & (table.hours_before == hours)].iloc[0]
            # Starts of the other score: day 10 at 09, the model already on; day 20, the model not there.
            already, calm = pick("normal", "already on", "i10fg", 3), pick("normal", "not at the start", "i10fg", 3)
            self.assertEqual((already.starts, calm.starts), (24 * 6, 24 * 6))
            self.assertEqual(already.share, 100)
            self.assertEqual(pick("normal", "already on", "i10fg", 0).share, 100)
            self.assertLess(calm.share, 20)
            self.assertLess(pick("normal", "already on", "cape", 3).share, 20)
            # Starts of the model: day 10 at 06, the other score not there yet; the gust is unusual at the start.
            model = pick("clim", "not at the start", "i10fg", 0)
            self.assertEqual((model.starts, model.share), (24 * 6, 100))
            self.assertEqual(pick("clim", "already on", "i10fg", 0).starts, 0)


if __name__ == "__main__":
    unittest.main()
