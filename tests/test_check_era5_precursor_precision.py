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
            # The dropout draws: G disagrees more at the alarm of the model followed by an event.
            (run / "scores").mkdir()
            spread = np.ones((len(TIMES), CELLS))
            spread[99, 5] = 5
            for name, values in (("generator_mean", np.full((len(TIMES), CELLS), 2.0)), ("generator_std", spread),
                                 ("discriminator_mean", np.zeros((len(TIMES), CELLS))),
                                 ("discriminator_std", np.ones((len(TIMES), CELLS)))):
                np.save(run / "scores" / f"{name}.npy", values.astype(np.float32))
            np.save(run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
            with patch.object(script, "flags", side_effect=fake), redirect_stdout(io.StringIO()):
                result = script.main(["--score-dir", str(run), "--by-dir", str(run), "--top-percent", "5",
                                      "--by-top-percent", "1", "--early-percent", "5.5", "--radius", "0",
                                      "--lead-hours", "3", "--components-dir", str(run), "--components-stride", "1"])
                # Over the usual spreads (here the same at every hour): the same order of the cells.
                from scripts import build_era5_spread_normals as build
                with redirect_stdout(io.StringIO()):
                    build.main(["--components-dir", str(run), "--output", str(run / "normals.npz")])
                usual = script.main(["--score-dir", str(run), "--by-dir", str(run), "--top-percent", "5",
                                     "--by-top-percent", "1", "--early-percent", "5.5", "--radius", "0",
                                     "--lead-hours", "3", "--components-dir", str(run), "--components-stride", "1",
                                     "--spread-normals", str(run / "normals.npz")])
        self.assertEqual({row["spread"]: row for row in usual["spreads"]["first only"]}["D spread"]["AUC"], 0.5)
        alone = {row["spread"]: row for row in result.pop("spreads")["first only"]}
        self.assertEqual(alone["G spread"]["AUC"], 1.0)  # followed at 100, not after the alarm at 20
        self.assertEqual(alone["D spread"]["AUC"], 0.5)
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


    def test_what_the_error_is_made_of_by_case_and_outcome(self):
        model, event, early = (np.zeros((len(TIMES), CELLS), dtype=bool) for _ in range(3))
        event[100, 5] = True
        model[99, 5] = True  # the model alone, followed by the event
        model[20, 25] = True  # the model alone, nothing after
        rows, cols = np.divmod(np.arange(CELLS), WIDTH)
        errors = np.ones((len(TIMES), CELLS, 2), dtype=np.float32)
        errors[99, 5] = (9, 1)  # the first variable makes the error of the alarm followed by the event
        errors[20, 25] = (1, 3)  # the second that of the alarm with nothing after
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "metadata.json").write_text(json.dumps({
                "grid": {"latitudes": [45.0, 44.5, 44.0], "longitudes": list(7 + 0.5 * np.arange(WIDTH)),
                         "rows": rows.tolist(), "cols": cols.tolist()},
                "preparation": {"features": ["rain", "wind"]}}), encoding="utf-8")
            (run / "scores").mkdir()
            np.save(run / "scores" / "feature_scores.npy", errors)
            levels = {5: model, 1: event, 5.5: early}
            with patch.object(script, "flags", side_effect=lambda path, percent: (levels[percent], TIMES, 0.0)), \
                    redirect_stdout(io.StringIO()):
                result = script.main(["--score-dir", str(run), "--by-dir", str(run), "--early-percent", "5.5",
                                      "--radius", "0", "--lead-hours", "3", "--feature-errors-dir", str(run),
                                      "--neither-share", "0"])
        read, share, main = result["makeup"][("first only", 1)]
        self.assertEqual(read, 1)
        self.assertAlmostEqual(share["rain"], 0.9)
        self.assertEqual(main, {"rain": 1.0, "wind": 0.0})
        read, share, main = result["makeup"][("first only", 0)]
        self.assertEqual(read, 1)
        self.assertAlmostEqual(share["wind"], 0.75)
        self.assertEqual(main["wind"], 1.0)
        self.assertEqual(result["makeup"][("neither", 0)][0], 0)  # none of its cells read

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
