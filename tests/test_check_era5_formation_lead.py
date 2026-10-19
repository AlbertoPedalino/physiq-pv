"""The formation check must tell a score that flags new weather before it forms from one that does not."""
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

from scripts import check_era5_formation_lead as script

TIMES = pd.date_range("2005-01-01", periods=300, freq="3h")
HEIGHT, WIDTH = 12, 30
BEGINS = range(20, 280, 40)  # every 40 steps new weather forms on the same block of four cells


class FormationLeadTests(unittest.TestCase):
    def test_new_weather_seen_two_steps_before(self):
        # The other score flags the block at its highest 1% from each begin, for 6 steps every other time
        # (it lasts) and for one step otherwise (it ends). One step before, the block is high enough for its
        # 5% only. Two steps before, the model flags the column east of the block.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
            block = np.isin(rows, (5, 6)) & np.isin(cols, (15, 16))
            model, normal = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
            # 2% of the cells always at 7, far from the block: the 1% level flags above 7, the 5% level above 0.
            normal[:, (rows < 2) & (cols < 4)] = 7
            for number, begin in enumerate(BEGINS):
                normal[begin:begin + (6 if number % 2 == 0 else 1), block] = 10
                normal[begin - 1, block] = 5
                model[begin - 2, np.isin(rows, (5, 6)) & (cols == 17)] = 5
            latitudes, longitudes = list(60 - 0.5 * np.arange(HEIGHT)), list(0.5 * np.arange(WIDTH))
            grid = {"latitudes": latitudes, "longitudes": longitudes, "rows": rows.tolist(), "cols": cols.tolist(),
                    "spacing": 0.5}
            runs = []
            for name, scores in (("run_clim", model), ("run_normal", normal)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", scores)
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}),
                                                           encoding="utf-8")
                runs.append(root / name)
            csv = root / "formation.csv"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]), "--top-percent", "5",
                                              "--by-top-percent", "1", "--early-percent", "5", "--csv", str(csv)]), 0)
            self.assertIn(f"births: {4 * len(BEGINS)} cells in {len(BEGINS)} groups", output.getvalue())
            self.assertIn("that last 16, that end 12", output.getvalue())
            table = pd.read_csv(csv)
            pick = lambda watcher, radius, group, when: table[
                (table.watcher == watcher) & (table.radius == radius) & (table.group == group)
                & (table.when == when)].iloc[0].share
            # The model is one or two cells away, 6 h before: seen within 2 cells and 6 h, not within 3 h, not at the cell.
            self.assertEqual(pick("clim", 2, "all births", "6h"), 100)
            self.assertEqual(pick("clim", 2, "all births", "3h"), 0)
            self.assertEqual(pick("clim", 0, "all births", "24h"), 0)
            self.assertEqual(pick("clim", 2, "births that end", "6h"), 100)
            self.assertLess(pick("clim", 2, "chance", "6h"), 5)
            # The other score at 5% flags the block itself 3 h before; the model adds the 6 h on its own.
            self.assertEqual(pick("normal at 5%", 0, "all births", "3h"), 100)
            self.assertEqual(pick("clim without normal at 5%", 2, "all births", "6h"), 100)
            # Who flags the square around the birth: within 3 h only the other score at 5%, within 6 h both.
            case = lambda radius, when, name: table[
                (table.watcher == "who flags") & (table.radius == radius) & (table.group == "all births")
                & (table.when == when) & (table.case == name)].iloc[0].share
            self.assertEqual(case(2, "3h", "second early only"), 100)
            self.assertEqual(case(2, "6h", "both"), 100)
            self.assertEqual(case(0, "6h", "second early only"), 100)
            self.assertEqual(case(2, "3h", "first only"), 0)


if __name__ == "__main__":
    unittest.main()
