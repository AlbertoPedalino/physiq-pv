"""The event agreement must tell an event the other run finds from one it nearly flags and one it misses."""
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

from scripts import check_era5_event_agreement as script

TIMES = pd.date_range("2005-01-01", periods=200, freq="3h")
HEIGHT, WIDTH = 12, 30
ROWS, COLS = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
FOUND = np.isin(ROWS, (3, 4)) & (COLS == 10)  # two cells at timestamps 50 and 51: four values
NEAR = (ROWS == 8) & (COLS == 5)  # one value at timestamp 100
MISSED = (ROWS == 1) & (COLS == 25)  # one value at timestamp 150


class EventAgreementTests(unittest.TestCase):
    def test_found_near_and_missed_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32)
            first[50:52, FOUND] = 5
            first[100, NEAR] = 5
            first[150, MISSED] = 5
            # The other run is ordinary noise in [0, 1): its highest 5% is above ~0.95, its highest 20% above ~0.8.
            other = np.random.default_rng(0).random(first.shape).astype(np.float32)
            other[50, np.nonzero(FOUND)[0][0]] = 5  # one value of the first event only
            other[100, NEAR] = 0.9
            other[150, MISSED] = 0.1
            grid = {"latitudes": list(60 - 0.5 * np.arange(HEIGHT)), "longitudes": list(0.5 * np.arange(WIDTH)),
                    "rows": ROWS.tolist(), "cols": COLS.tolist(), "spacing": 0.5}
            runs = []
            for name, scores in (("first", first), ("other", other)):
                run = root / name / "val_gen_clim_day"
                (run / "scores").mkdir(parents=True)
                np.save(run / "scores" / "anomaly_mean.npy", scores)
                np.save(run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (run / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}), encoding="utf-8")
                runs.append(run)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(script.main(["--score-dir", str(runs[0]), "--other-dir", str(runs[1])]), 0)
            lines = output.getvalue().splitlines()
            start = next(i for i, line in enumerate(lines) if line.startswith("events of first/val_gen_clim_day"))
            every = next(line for line in lines[start:] if line.startswith("all"))
            events, found, near, missed, values_missed = (float(value) for value in every.split()[1:])
            self.assertEqual(events, 3)
            self.assertAlmostEqual(found, 33.3, places=1)
            self.assertAlmostEqual(near, 33.3, places=1)
            self.assertAlmostEqual(missed, 33.3, places=1)
            self.assertAlmostEqual(values_missed, 100 / 6, places=1)  # one value of six
            listed = lines[next(i for i, line in enumerate(lines[start:], start) if line.startswith("largest")) + 2]
            self.assertTrue(listed.startswith(str(TIMES[150])))
            self.assertEqual(listed.split()[2:], ["3", "1", "59.5", "12.5"])


if __name__ == "__main__":
    unittest.main()
