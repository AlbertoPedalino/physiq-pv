"""The onset of a known event is measured on the raw values with the rules of anticipo.md; a score that flags the
ring before the box, and the box before the onset, has a lead in space and in time against random dates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from physiq_pv.era5.features import FEATURE_NAMES
from scripts import check_era5_known_events_lead as script

LATITUDES = [45.0 - 0.5 * index for index in range(10)]
LONGITUDES = [10.0 + 0.5 * index for index in range(10)]
ROWS = [row for row in range(10) for _ in range(10)]
COLS = [col for _ in range(10) for col in range(10)]
GRID = {"latitudes": LATITUDES, "longitudes": LONGITUDES, "rows": ROWS, "cols": COLS}
T2M, U10 = FEATURE_NAMES.index("2m_temperature"), FEATURE_NAMES.index("10m_u_component_of_wind")


def cube(times, rng):
    values = np.zeros((len(times), 100, len(FEATURE_NAMES)), dtype=np.float32)
    values[:, :, T2M] = 280 + rng.normal(0, 1, (len(times), 100))
    values[:, :, U10] = rng.normal(0, 1, (len(times), 100))
    values[:, :, U10 + 1] = rng.normal(0, 1, (len(times), 100))
    return values


def prepared(folder, rng):
    folder.mkdir(parents=True)
    parts = {"train": pd.date_range("2000-01-01", "2001-12-31 21:00", freq="3h"),
             "validation": pd.date_range("2002-01-01", "2002-12-31 21:00", freq="3h"),
             "test": pd.date_range("2003-01-01", "2004-12-31 21:00", freq="3h")}
    arrays = {}
    for name, times in parts.items():
        arrays[name] = cube(times, rng)
        np.save(folder / f"{name}_timestamps.npy", times.values)
    test = parts["test"]
    heat = (test >= "2004-07-09") & (test < "2004-07-15")
    arrays["test"][heat, :, T2M] += 10  # the whole grid, from one day before the first day of the source
    wind = (test >= "2003-01-20 06:00") & (test <= "2003-01-21 21:00")
    arrays["test"][wind, :, U10] = 30
    for name, values in arrays.items():
        np.save(folder / f"{name}.npy", values)
    (folder / "metadata.json").write_text(json.dumps(
        {"status": "complete", "features": list(FEATURE_NAMES), "train_end_year": 2001, "validation_end_year": 2002,
         "grid": GRID}), encoding="utf-8")
    return test


def score_run(folder, times, values):
    (folder / "scores").mkdir(parents=True)
    np.save(folder / "scores" / "anomaly_mean.npy", values)
    np.save(folder / "test_timestamps.npy", times.as_unit("ns").asi8)
    (folder / "metadata.json").write_text(json.dumps(
        {"status": "complete", "grid": GRID, "anomaly_mean_file": "scores/anomaly_mean.npy"}), encoding="utf-8")


class HelperTests(unittest.TestCase):
    def test_first_run(self):
        self.assertEqual(script.first_run([True, False, True, True, True, True], 3), 2)
        self.assertIsNone(script.first_run([True, True, False, True], 3))

    def test_calendar_day_puts_29_february_on_1_march(self):
        days = pd.DatetimeIndex(["2001-03-01", "2004-02-29", "2004-03-01", "2004-12-31"])
        self.assertEqual(list(script.calendar_day(days)), [59, 59, 59, 364])

    def test_ring_is_outside_the_box_and_near_it(self):
        layout = script.Layout(GRID)
        event = {"lat_min": 42.0, "lat_max": 43.0, "lon_min": 12.0, "lon_max": 13.0}
        ring = script.ring_cells(event, layout, 60)
        inside = script.in_box(event, layout.latitudes, layout.longitudes)
        self.assertFalse((ring & inside).any())
        self.assertTrue(ring[(layout.latitudes == 43.5) & (layout.longitudes == 12.5)].all())  # 55.6 km north
        self.assertFalse(ring[(layout.latitudes == 44.5) & (layout.longitudes == 12.5)].any())  # 167 km north


class LeadTests(unittest.TestCase):
    def test_onsets_and_lead_in_time_and_space(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            test = prepared(folder / "prepared", rng)
            events = folder / "eventi.csv"
            pd.DataFrame({"id": ["h", "w", "d"], "tipo": ["caldo", "vento", "polvere"],
                          "inizio": ["2004-07-10", "2003-01-20", "2004-03-05"],
                          "fine": ["2004-07-14", "2003-01-21", "2004-03-06"],
                          "lat_min": [42.0, 42.0, 42.0], "lat_max": [43.5, 43.5, 43.5],
                          "lon_min": [11.5, 11.5, 11.5], "lon_max": [13.0, 13.0, 13.0]}).to_csv(events, index=False)
            layout = script.Layout(GRID)
            box = script.in_box({"lat_min": 42.0, "lat_max": 43.5, "lon_min": 11.5, "lon_max": 13.0},
                                layout.latitudes, layout.longitudes)
            model = rng.uniform(0, 1, (len(test), 100)).astype(np.float32)
            outside = np.flatnonzero((test >= "2004-07-07 12:00") & (test < "2004-07-15"))
            inside = np.flatnonzero((test >= "2004-07-08") & (test < "2004-07-15"))
            model[np.ix_(outside, np.flatnonzero(~box))] = 10  # the ring from 36 h before the onset
            model[np.ix_(inside, np.flatnonzero(box))] = 10  # the box from 24 h before the onset
            score_run(folder / "model", test, model)
            score_run(folder / "flat", test, rng.uniform(0, 1, (len(test), 100)).astype(np.float32))
            output = folder / "out"
            script.main(["--events", str(events), "--prepared-dir", str(folder / "prepared"), "--train-end-year", "2001",
                         "--model", f"G={folder / 'model'}", "--simple", f"flat={folder / 'flat'}", "--random", "20",
                         "--rings", "100", "--output-dir", str(output)])

            starts = pd.read_csv(output / "inizi.csv", parse_dates=["inizio"])
            onset = {(row["id"], row["variante"]): (row["inizio"], row["origine"]) for _, row in starts.iterrows()}
            self.assertEqual(onset[("h", "principale")], (pd.Timestamp("2004-07-09"), "misurato"))
            self.assertEqual(onset[("w", "principale")], (pd.Timestamp("2003-01-20 06:00"), "misurato"))
            self.assertEqual(onset[("w", "meta area")], (pd.Timestamp("2003-01-20 06:00"), "misurato"))
            self.assertEqual(onset[("d", "principale")], (pd.Timestamp("2004-03-05"), "fonte"))

            table = pd.read_csv(output / "per_evento.csv")
            row = table[(table["id"] == "h") & (table["metodo"] == "G") & (table["finestra h"] == 72)
                        & (table["segnale percentile"] == 95)].iloc[0]
            self.assertTrue(row["rilevato"])
            self.assertEqual(row["anticipo h"], 24)
            self.assertTrue(row["anello 100 km: prima del riquadro"])
            self.assertEqual(row["anello 100 km: ore prima del riquadro"], 12)
            self.assertEqual(row["famiglia"], "modello")
            for name in ("curve.csv", "riepilogo.csv", "composito.csv", "metadata.json"):
                self.assertTrue((output / name).is_file(), name)
            # Pure noise: one threshold over the whole window, so a random date passes it about 5% of the times,
            # not at almost every window as with a threshold per lag.
            flat = table[table["metodo"] == "flat"]
            self.assertLessEqual(flat["casuali con segnale prima %"].max(), 10)
            self.assertLessEqual(flat["casuali rilevati %"].max(), 10)
            summary = pd.read_csv(output / "riepilogo.csv")
            self.assertEqual(set(summary["famiglia"]), {"modello", "semplici"})


if __name__ == "__main__":
    unittest.main()
