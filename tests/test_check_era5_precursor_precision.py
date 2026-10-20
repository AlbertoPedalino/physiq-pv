"""The precursor check must count, at calm cells, how often each case of alarms is followed by an event."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_precursor_precision as script

TIMES = pd.date_range("2005-01-01", periods=200, freq="3h")
HEIGHT, WIDTH = 3, 10
CELLS = HEIGHT * WIDTH


class PrecursorPrecisionTests(unittest.TestCase):
    def test_precision_of_each_case(self):
        model, event, early = (np.zeros((len(TIMES), CELLS), dtype=bool) for _ in range(3))
        for step in (50, 100, 150, 180):
            event[step, 5 if step < 180 else 15] = True
        model[49, 5] = early[49, 5] = True  # both before the event at 50
        model[99, 5] = True  # the model alone before the event at 100
        early[149, 5] = True  # the simple score alone before the event at 150
        model[20, 25] = True  # an alarm of the model with no event after
        early[30, 28] = True  # and one of the simple score
        rows, cols = np.divmod(np.arange(CELLS), WIDTH)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "metadata.json").write_text(json.dumps({"grid": {
                "latitudes": [45.0, 44.5, 44.0], "longitudes": list(7 + 0.5 * np.arange(WIDTH)),
                "rows": rows.tolist(), "cols": cols.tolist()}}), encoding="utf-8")
            levels = {5: model, 1: event, 5.5: early}
            fake = lambda path, percent: (levels[percent], TIMES, 0.0)
            with patch.object(script, "flags", side_effect=fake), redirect_stdout(io.StringIO()):
                result = script.main(["--score-dir", str(run), "--by-dir", str(run), "--top-percent", "5",
                                      "--by-top-percent", "1", "--early-percent", "5.5", "--radius", "0",
                                      "--lead-hours", "3"])
        self.assertEqual(result[(0, 3, "both")][:2], (1, 100.0))
        self.assertEqual(result[(0, 3, "first only")][:2], (2, 50.0))
        self.assertEqual(result[(0, 3, "second early only")][:2], (2, 50.0))
        cells, precision, lift = result[(0, 3, "neither")]
        self.assertGreater(precision, 0)  # the event at 180 came with no alarm
        self.assertAlmostEqual(result[(0, 3, "both")][2], 100.0 / precision)
        # A cell is calm only without an event flag around it in the 24 h before: the cell of the event at 50 is
        # not calm from 51 to 58, so the alarms there would not count.
        self.assertLess(cells, (len(TIMES) - 8 - 1) * CELLS)
        # With every cell: all timestamps but the last, less the cells where the event is there now.
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "metadata.json").write_text(json.dumps({"grid": {
                "latitudes": [45.0, 44.5, 44.0], "longitudes": list(7 + 0.5 * np.arange(WIDTH)),
                "rows": rows.tolist(), "cols": cols.tolist()}}), encoding="utf-8")
            with patch.object(script, "flags", side_effect=fake), redirect_stdout(io.StringIO()):
                every = script.main(["--score-dir", str(run), "--by-dir", str(run), "--top-percent", "5",
                                     "--by-top-percent", "1", "--early-percent", "5.5", "--radius", "0",
                                     "--lead-hours", "3", "--all-cells"])
        self.assertEqual(sum(every[(0, 3, case)][0] for case in script.CASES), (len(TIMES) - 1) * CELLS - 4)
        self.assertEqual(every[(0, 3, "both")][:2], (1, 100.0))

        # The CAPE as the referee: unusual after the alarms of both and of the model, the gust after the simple score.
        weather = np.zeros((len(TIMES), CELLS, 2), dtype=bool)
        weather[[50, 100], 5, 0] = True  # CAPE
        weather[150, 5, 1] = True  # gust
        from scripts import nowcast_era5_anomalies as nowcast
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "metadata.json").write_text(json.dumps({"grid": {
                "latitudes": [45.0, 44.5, 44.0], "longitudes": list(7 + 0.5 * np.arange(WIDTH)),
                "rows": rows.tolist(), "cols": cols.tolist()}}), encoding="utf-8")
            judged = {}
            for target in ("cape", "gust"):
                with patch.object(script, "flags", side_effect=fake), \
                        patch.object(nowcast, "unseen_flags", return_value=weather), redirect_stdout(io.StringIO()):
                    judged[target] = script.main(["--score-dir", str(run), "--by-dir", str(run), "--top-percent", "5",
                                                  "--by-top-percent", "1", "--early-percent", "5.5", "--radius", "0",
                                                  "--lead-hours", "3", "--target", target, "--unseen-dir", str(run)])
        self.assertEqual(judged["cape"][(0, 3, "both")][1], 100.0)
        self.assertEqual(judged["cape"][(0, 3, "first only")][1], 50.0)
        self.assertEqual(judged["cape"][(0, 3, "second early only")][1], 0.0)
        self.assertEqual(judged["gust"][(0, 3, "second early only")][1], 50.0)
        self.assertEqual(judged["gust"][(0, 3, "both")][1], 0.0)


if __name__ == "__main__":
    unittest.main()
