"""A known event is read over the cells of its box and the timestamps of its days; a striking day matches an event
when it falls in its days and its patch has a cell in its box."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_known_events as script


def events_file(folder):
    path = Path(folder) / "eventi.csv"
    pd.DataFrame({"id": ["a", "b"], "tipo": ["vento", "caldo"], "inizio": ["2010-01-10", "2010-01-20"],
                  "fine": ["2010-01-10", "2010-01-21"], "lat_min": [40.5, 39.5], "lat_max": [41.0, 39.5],
                  "lon_min": [10.0, 12.0], "lon_max": [11.0, 12.0]}).to_csv(path, index=False)
    return path


class KnownEventsTests(unittest.TestCase):
    def test_share_of_the_box_and_days(self):
        times = pd.date_range("2010-01-01", "2010-01-31 21:00", freq="3h")
        latitudes, longitudes = [41.0, 40.5, 40.0, 39.5], [10.0, 10.5, 11.0, 11.5, 12.0]
        rows = [row for row in range(4) for _ in range(5)]
        cols = [col for _ in range(4) for col in range(5)]
        scores = np.zeros((len(times), 20), dtype=np.float32)
        day = np.flatnonzero(times.normalize() == pd.Timestamp("2010-01-10"))
        scores[day[2], [0, 1, 2]] = 5.0  # 3 of the 6 cells of box a, at one of its 8 timestamps
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            (run / "scores").mkdir(parents=True)
            np.save(run / "scores" / "anomaly_mean.npy", scores)
            np.save(run / "test_timestamps.npy", times.as_unit("ns").asi8)
            reference = Path(folder) / "reference"
            (reference / "scores").mkdir(parents=True)
            np.save(reference / "scores" / "anomaly_mean.npy", np.linspace(0, 1, 1000, dtype=np.float32).reshape(50, 20))
            (run / "metadata.json").write_text(json.dumps(
                {"status": "complete", "grid": {"latitudes": latitudes, "longitudes": longitudes, "rows": rows,
                                                "cols": cols}, "anomaly_mean_file": "scores/anomaly_mean.npy",
                 "threshold_reference": str(reference)}), encoding="utf-8")
            output = Path(folder) / "out"
            script.main(["--events", str(events_file(folder)), "--score", f"G={run}", "--output-dir", str(output)])
            table = pd.read_csv(output / "eventi_noti.csv").set_index("id")
            self.assertEqual((table.loc["a", "celle"], table.loc["a", "istanti"]), (6, 8))
            self.assertAlmostEqual(table.loc["a", "segnalati %"], 100 * 3 / 48, places=3)
            self.assertAlmostEqual(table.loc["a", "segnalati nell'istante più segnalato %"], 50.0)
            self.assertEqual(table.loc["a", "istante più segnalato"], str(times[day[2]]))
            self.assertEqual((table.loc["b", "celle"], table.loc["b", "istanti"]), (1, 16))
            self.assertEqual(table.loc["b", "segnalati %"], 0)

    def test_striking_day_inside_an_event(self):
        with tempfile.TemporaryDirectory() as folder:
            striking = Path(folder) / "giorni"
            striking.mkdir()
            pd.DataFrame({"rango": [1, 2, 3], "giorno": ["2010-01-10", "2010-01-21", "2010-01-25"]}).to_csv(
                striking / "casi.csv", index=False)
            inside, outside = np.zeros((4, 5), bool), np.zeros((4, 5), bool)
            inside[0, 1] = True  # 41N 10.5E, in box a
            outside[0, 0] = True  # 41N 10E, not in box b
            np.savez(striking / "casi.npz", latitudes=np.array([41.0, 40.5, 40.0, 39.5]),
                     longitudes=np.array([10.0, 10.5, 11.0, 11.5, 12.0]), macchia_01=inside, macchia_02=outside,
                     macchia_03=inside)
            output = Path(folder) / "out"
            script.main(["--events", str(events_file(folder)), "--striking-dir", str(striking),
                         "--output-dir", str(output)])
            table = pd.read_csv(output / "giorni_ed_eventi.csv")
            self.assertEqual(list(table["id"]), ["a"])
            self.assertEqual(int(table.loc[0, "rango"]), 1)


if __name__ == "__main__":
    unittest.main()
