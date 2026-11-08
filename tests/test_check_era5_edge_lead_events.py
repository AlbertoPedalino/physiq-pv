"""On the known events only, the edge lead check must find the lead of a score that flags the cells a front reaches
next, with an interval from resampling the events."""
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

from scripts import check_era5_edge_lead as script

TIMES = pd.date_range("2005-01-01", periods=300, freq="3h")
HEIGHT, WIDTH = 12, 30


def runs(root, ahead):
    """A front moves one column east every 3 hours, a new one every 40 steps; the second score flags its column,
    the model the column `ahead` columns east of it (negative: behind)."""
    rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
    model, normal = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
    for begin in range(10, len(TIMES), 40):
        for step in range(WIDTH + 4):
            for score, column in ((model, step + ahead), (normal, step)):
                if 0 <= column < WIDTH and begin + step < len(TIMES):
                    score[begin + step, cols == column] = 5
    grid = {"latitudes": list(60 - 0.5 * np.arange(HEIGHT)), "longitudes": list(0.5 * np.arange(WIDTH)),
            "rows": rows.tolist(), "cols": cols.tolist(), "spacing": 0.5}
    paths = []
    for name, scores in ((f"run_model_{ahead}", model), (f"run_normal_{ahead}", normal)):
        (root / name / "scores").mkdir(parents=True)
        np.save(root / name / "scores" / "anomaly_mean.npy", scores)
        np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}), encoding="utf-8")
        paths.append(root / name)
    return paths


class EdgeLeadEventsTests(unittest.TestCase):
    def check(self, ahead):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, normal = runs(root, ahead)
            events = root / "eventi.csv"
            days = [str(day.date()) for day in TIMES.normalize().unique()[2:30:4]]
            pd.DataFrame({"id": [f"e{index}" for index in range(len(days))], "tipo": ["vento"] * len(days),
                          "inizio": days, "fine": days, "lat_min": 56.0, "lat_max": 58.0, "lon_min": 4.0,
                          "lon_max": 10.0}).to_csv(events, index=False)
            # The dropout run: error 1 everywhere, spread 0.5 where the model flags and 0.1 elsewhere.
            flagged = np.load(model / "scores" / "anomaly_mean.npy") > 0
            dropout = root / "dropout"
            (dropout / "scores").mkdir(parents=True)
            np.save(dropout / "scores" / "generator_mean.npy", np.ones(flagged.shape, dtype=np.float32))
            np.save(dropout / "scores" / "generator_std.npy", np.where(flagged, 0.5, 0.1).astype(np.float32))
            np.save(dropout / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
            (dropout / "metadata.json").write_text(json.dumps(
                {"generator_mean_file": "scores/generator_mean.npy", "generator_std_file": "scores/generator_std.npy"}),
                encoding="utf-8")
            out = root / "eventi_bordo.csv"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(script.main(["--score-dir", str(model), "--by-dir", str(normal), "--top-percent", "10",
                                              "--events", str(events), "--ring-km", "100", "--bootstrap", "200",
                                              "--events-csv", str(out), "--ratio-run", str(dropout)]), 0)
            table = pd.read_csv(out)
            self.per_event = pd.read_csv(root / "eventi_bordo_per_evento.csv")
            spread = pd.read_csv(root / "eventi_bordo_dropout.csv")
            self.spread = spread.set_index(["distance", "group", "kind"])["median relative spread"]
            return table[(table["type"] == "all") & (table["distance"].astype(str) == "2")
                         & (table["window_hours"] == 6)].iloc[0]

    def test_a_score_ahead_has_a_positive_excess(self):
        row = self.check(2)
        self.assertGreater(row["first front"], 95)
        self.assertGreater(row["excess 2.5%"], 0)

    def test_one_cell_ahead_every_event_and_the_dropout(self):
        self.check(1)
        # Every event with values of the model alone, one cell ahead, has a positive excess within 3 h.
        enough = self.per_event[self.per_event["values"] > 0]
        self.assertGreater(len(enough), 0)
        self.assertTrue((enough["excess 3 h"] > 0).all())
        # The relative spread is read where the values are: 0.5 on the cells of the model, 0.1 on the others.
        self.assertAlmostEqual(self.spread[("1", "first only", "front")], 0.5, delta=0.02)
        self.assertAlmostEqual(self.spread[("1", "neither", "trail")], 0.1, delta=0.005)

    def test_a_score_behind_has_a_negative_excess(self):
        row = self.check(-2)
        self.assertGreater(row["first trail"], 95)
        self.assertLess(row["excess 97.5%"], 0)


if __name__ == "__main__":
    unittest.main()
