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
            with patch.object(script, "flags", side_effect=fake), \
                    patch.object(nowcast, "unseen_flags", return_value=weather), redirect_stdout(io.StringIO()):
                every = script.main(["--score-dir", str(run), "--by-dir", str(run), "--top-percent", "5",
                                     "--by-top-percent", "1", "--early-percent", "5.5", "--radius", "0",
                                     "--lead-hours", "3", "--target", "cape", "--unseen-dir", str(run), "--all-cells"])
        # Every cell but where the CAPE is unusual now (the events of the simple score count too).
        self.assertEqual(sum(every[(0, 3, case)][0] for case in script.CASES), (len(TIMES) - 1) * CELLS - 2)
        self.assertEqual(judged["cape"][(0, 3, "both")][1], 100.0)
        self.assertEqual(judged["cape"][(0, 3, "first only")][1], 50.0)
        self.assertEqual(judged["cape"][(0, 3, "second early only")][1], 0.0)
        self.assertEqual(judged["gust"][(0, 3, "second early only")][1], 50.0)
        self.assertEqual(judged["gust"][(0, 3, "both")][1], 0.0)


    def test_flags_carried_by_the_motion_and_by_the_wind(self):
        # One flagged cell on the middle row moves one column east every step.
        early = np.zeros((len(TIMES), CELLS), dtype=bool)
        for step in range(len(TIMES)):
            early[step, WIDTH + step % (WIDTH - 2)] = True
        rows, cols = np.divmod(np.arange(CELLS), WIDTH)
        latitudes = [45.0, 44.5, 44.0]
        grid = {"latitudes": latitudes, "longitudes": list(7 + 0.5 * np.arange(WIDTH)), "rows": rows.tolist(),
                "cols": cols.tolist()}
        from scripts import check_era5_edge_direction as direction
        from scripts import nowcast_era5_anomalies as nowcast
        # A west wind that moves one column in a step of 3 hours.
        east = nowcast.METERS_PER_DEGREE * np.cos(np.radians(np.asarray(latitudes)))[rows] * 0.5 / (3 * 3600)
        wind = lambda wind_dir, timestamps, positions, *_: np.stack(
            [np.broadcast_to(east, (len(positions), CELLS)), np.zeros((len(positions), CELLS))], axis=2)
        with patch.object(direction, "wind_of_month", side_effect=wind):
            by_wind = script.carried(early, TIMES, grid, "wind", Path("."))
        by_motion = script.carried(early, TIMES, grid, "motion", None)
        for moved in (by_wind, by_motion):
            step = 5
            here = WIDTH + step
            self.assertTrue(moved[step, here + 1])  # where it goes
            self.assertFalse(moved[step, here])  # flagged now
            self.assertFalse(moved[step, here - 2])  # behind
        self.assertFalse(by_motion[0].any())  # no step before


if __name__ == "__main__":
    unittest.main()
