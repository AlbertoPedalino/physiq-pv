"""The edge event check must type the events the other scores flag and tell which of them the model extends."""
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

from physiq_pv.era5.cube import CubeGrid
from physiq_pv.era5.features import FEATURE_NAMES
from scripts import check_era5_edge_event_types as script

TIMES = pd.date_range("2005-01-01", periods=300, freq="3h")
HEIGHT, WIDTH = 12, 30
EVENT_STEPS = (50, 150, 250)  # at different times of day, so that each cell-hour has one event value at most
WIND = (slice(3, 9), slice(2, 8))  # 6 x 6 cells of the windy event
HEAT = (slice(3, 9), slice(20, 26))  # and of the hot one


class EdgeEventTypeTests(unittest.TestCase):
    def test_the_model_extends_the_windy_events_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
            latitudes, longitudes = 60 - 0.5 * np.arange(HEIGHT), 0.5 * np.arange(WIDTH)
            in_box = lambda box: ((rows >= box[0].start) & (rows < box[0].stop) & (cols >= box[1].start)
                                  & (cols < box[1].stop))
            wind, heat = in_box(WIND), in_box(HEAT)
            ring = in_box((slice(2, 10), slice(1, 9))) & ~wind  # the cells next to the windy event
            model, others = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
            gust = np.ones((len(TIMES), HEIGHT * WIDTH), dtype=np.float32)
            temperature = np.full((len(TIMES), HEIGHT * WIDTH), 280.0, dtype=np.float32)
            for step in EVENT_STEPS:
                others[step, wind | heat] = 5
                model[step, ring] = 5
                gust[step, wind] = 30
                temperature[step, heat] = 300
            grid = {"latitudes": list(latitudes), "longitudes": list(longitudes), "rows": rows.tolist(),
                    "cols": cols.tolist(), "spacing": 0.5}
            runs = []
            for name, scores in (("run_model", model), ("run_others", others)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", scores)
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}),
                                                           encoding="utf-8")
                runs.append(root / name)
            # The prepared data: dry, with the temperature above; a few earlier timestamps for training and validation.
            prepared = root / "prepared"
            prepared.mkdir()
            data = np.zeros((len(TIMES), HEIGHT * WIDTH, len(FEATURE_NAMES)), dtype=np.float32)
            data[:, :, FEATURE_NAMES.index("2m_temperature")] = temperature
            for part, times, values in (("train", pd.date_range("2003-01-01", periods=8, freq="3h"), data[:8]),
                                        ("validation", pd.date_range("2004-01-01", periods=8, freq="3h"), data[:8]),
                                        ("test", TIMES, data)):
                np.save(prepared / f"{part}.npy", values)
                np.save(prepared / f"{part}_timestamps.npy", times.as_unit("ns").asi8)
            (prepared / "metadata.json").write_text(json.dumps({
                "status": "complete", "grid": CubeGrid(latitudes, longitudes, rows, cols).to_dict(),
                "features": list(FEATURE_NAMES), "start_year": 2003, "train_end_year": 2003,
                "validation_start_year": 2004, "validation_end_year": 2004, "test_start_year": 2005,
                "score_end_year": 2005}), encoding="utf-8")
            cape = np.full(gust.shape, 100.0, dtype=np.float32)
            (root / "unseen" / "2005").mkdir(parents=True)
            for month in (1, 2):
                chosen = TIMES.month == month
                xr.Dataset({name: (("valid_time", "latitude", "longitude"), values[chosen].reshape(-1, HEIGHT, WIDTH))
                            for name, values in (("i10fg", gust), ("cape", cape))},
                           coords={"valid_time": TIMES[chosen], "latitude": latitudes, "longitude": longitudes}
                           ).to_netcdf(root / "unseen" / "2005" / f"era5_independent_2005_{month:02d}.nc")
            output = io.StringIO()
            with redirect_stdout(output):
                table = script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]), "--top-percent", "5",
                                     "--prepared-dir", str(prepared), "--unseen-dir", str(root / "unseen")])
            overall = {row["population"]: row for row in table if row["selection"].startswith("% of the cells")}
            events, edge = overall["event cells of the others"], overall["edge, model alone"]
            self.assertEqual(events["cells"], 2 * 36 * len(EVENT_STEPS))
            self.assertEqual(edge["cells"], 28 * len(EVENT_STEPS))
            self.assertAlmostEqual(events["wind"], 50)
            self.assertAlmostEqual(events["heat"], 50)
            self.assertAlmostEqual(edge["wind"], 100)
            self.assertEqual(edge["heat"], 0)
            self.assertAlmostEqual(edge["signal_wind"], 100)  # every cell of the windy event has the strong gust
            self.assertEqual(events["rain"] + events["convective"] + events["other"] + events["mixed"], 0)
            # The edge no score flags: the ring of the hot event only, the ring of the windy one is the model's.
            nobody = overall["edge, no score"]
            self.assertEqual(nobody["cells"], 28 * len(EVENT_STEPS))
            self.assertAlmostEqual(nobody["heat"], 100)
            self.assertEqual(nobody["wind"], 0)
            winter = {row["population"]: row for row in table if row["selection"] == "season DJF"}
            self.assertEqual(winter["edge, model alone"]["cells"], edge["cells"])
            self.assertIn("a signal is in an event at 2 times its share", output.getvalue())

    def test_beyond_reads_each_time_of_day_and_cell_apart(self):
        values = np.tile(np.arange(10.0)[:, None], (2, 3))  # 20 timestamps, two times of day, 3 cells
        hours = np.repeat([0, 3], 10)
        high = script.beyond(values, hours, 90)
        self.assertEqual(high.sum(), 2 * 3)  # the value 9 of each time of day and cell
        among = values >= 5
        self.assertTrue(script.beyond(values, hours, 50, among=among)[values == 9].all())
        self.assertFalse(script.beyond(values, hours, 50, among=np.zeros_like(among)).any())


if __name__ == "__main__":
    unittest.main()
