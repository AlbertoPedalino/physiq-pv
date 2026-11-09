"""The cells only the model flags just ahead of a front that brings rain get the rain in the next hours."""
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

from physiq_pv.era5.features import FEATURE_NAMES
from scripts import check_era5_alone_future as script

TIMES = pd.date_range("2005-01-01", periods=300, freq="3h")
HEIGHT, WIDTH = 12, 30
RAIN = FEATURE_NAMES.index("total_precipitation")


class AloneFutureTests(unittest.TestCase):
    def test_rain_comes_to_the_cells_ahead(self):
        rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
        grid = {"latitudes": list(60 - 0.5 * np.arange(HEIGHT)), "longitudes": list(0.5 * np.arange(WIDTH)),
                "rows": rows.tolist(), "cols": cols.tolist(), "spacing": 0.5}
        model, normal = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
        test = np.zeros((len(TIMES), HEIGHT * WIDTH, len(FEATURE_NAMES)), dtype=np.float32)
        # A front moves one column east every 3 hours, a new one every 40 steps: the second score flags its column,
        # where it rains, and the model the column just ahead, where the rain arrives 3 hours later.
        for begin in range(10, len(TIMES), 40):
            for step in range(WIDTH + 1):
                if begin + step >= len(TIMES):
                    break
                if step < WIDTH:
                    normal[begin + step, cols == step] = 5
                    test[begin + step, cols == step, RAIN] = 0.002
                if step + 1 < WIDTH:
                    model[begin + step, cols == step + 1] = 5
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = {}
            for name, scores in (("run_model", model), ("run_normal", normal)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", scores)
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}),
                                                           encoding="utf-8")
                runs[name] = root / name
            prepared = root / "prepared"
            prepared.mkdir()
            parts = {"train": pd.date_range("2000-01-01", periods=16, freq="3h"),
                     "validation": pd.date_range("2001-01-01", periods=16, freq="3h"), "test": TIMES}
            for name, times in parts.items():
                values = test if name == "test" else np.zeros((len(times), HEIGHT * WIDTH, len(FEATURE_NAMES)),
                                                              dtype=np.float32)
                np.save(prepared / f"{name}.npy", values)
                np.save(prepared / f"{name}_timestamps.npy", times.values)
            (prepared / "metadata.json").write_text(json.dumps(
                {"status": "complete", "features": list(FEATURE_NAMES), "train_end_year": 2000,
                 "validation_end_year": 2001, "grid": grid}), encoding="utf-8")
            csv = root / "future.csv"
            with redirect_stdout(io.StringIO()):
                script.main(["--score-dir", str(runs["run_model"]), "--by-dir", str(runs["run_normal"]),
                             "--prepared-dir", str(prepared), "--top-percent", "10", "--stride", "1",
                             "--neither-share", "1", "--second-share", "1", "--csv", str(csv)])
            table = pd.read_csv(csv).set_index("group")
            alone = table.loc["model only, all"]
            self.assertGreater(alone["cells"], 0)
            self.assertEqual(alone["raining now %"], 0)  # ahead of the front: not yet
            self.assertGreater(alone["raining +3h %"], 95)  # the front arrives in 3 hours
            self.assertGreater(alone["starts raining +3h %"], 95)
            self.assertLess(table.loc["neither (reference)", "raining +3h %"], 5)  # behind the front: it left
            self.assertGreater(table.loc["normal only", "raining now %"], 95)
            # By size: the area of the second is a column of 12 cells; the model's cells are fronts, the others trails.
            sized = pd.read_csv(root / "future_dimensione.csv")
            row = sized[(sized["group"] == "model only, all")].iloc[0]
            self.assertEqual(row["area cells"], "10-99")
            self.assertGreater(row["front %"], 95)
            self.assertGreater(row["excess pp"], 150)


class SidesTests(unittest.TestCase):
    def test_above_and_below_normal(self):
        table = pd.DataFrame({"rain now": [0.0, 0.5], "rain +3h": [0.2, 0.0],
                              "cloud above now": [True, False], "cloud below now": [False, True],
                              "cloud above +3h": [True, True], "cloud below +3h": [False, False]})
        row = script.summarise(table, [3], 0.1, "g")
        self.assertEqual((row["cloud above now %"], row["cloud below now %"], row["cloud away now %"]), (50, 50, 100))
        self.assertEqual((row["cloud above +3h %"], row["cloud below +3h %"]), (100, 0))
        self.assertEqual(row["starts raining +3h %"], 100)  # the one cell dry now rains 3 hours later


if __name__ == "__main__":
    unittest.main()
