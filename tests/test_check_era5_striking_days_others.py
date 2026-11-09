"""The check must say how much of a striking patch of the model another score flags, then and within 3 h."""
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

from scripts import check_era5_striking_days_others as script

TIMES = pd.date_range("2005-01-01", periods=40, freq="3h")
HEIGHT, WIDTH = 6, 10


class StrikingDaysOthersTests(unittest.TestCase):
    def test_overlap_now_and_around(self):
        rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
        grid = {"latitudes": list(60 - 0.5 * np.arange(HEIGHT)), "longitudes": list(0.5 * np.arange(WIDTH)),
                "rows": rows.tolist(), "cols": cols.tolist(), "spacing": 0.5}
        patch = np.zeros((HEIGHT, WIDTH), dtype=bool)
        patch[:, :4] = True  # the patch of the model on the day: columns 0-3, at step 20
        early, late = np.zeros((len(TIMES), HEIGHT * WIDTH), np.float32), np.zeros((len(TIMES), HEIGHT * WIDTH), np.float32)
        early[20, cols < 2] = 5  # flags half of the patch at the same step
        late[21, cols < 4] = 5  # flags all of it, one step later
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = {}
            for name, scores in (("early", early), ("late", late)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", scores)
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}),
                                                           encoding="utf-8")
                runs[name] = root / name
            cases = root / "cases"
            cases.mkdir()
            pd.DataFrame({"rango": [1], "giorno": [str(TIMES[20].date())], "istante": [str(TIMES[20])]}).to_csv(
                cases / "casi.csv", index=False)
            np.savez(cases / "casi.npz", macchia_01=patch)
            csv = root / "out.csv"
            with redirect_stdout(io.StringIO()):
                script.main(["--cases-dir", str(cases), "--by", f"early={runs['early']}", f"late={runs['late']}",
                             "--top-percent", "10", "--csv", str(csv)])
            table = pd.read_csv(csv).set_index("score")
            self.assertEqual(table.loc["early", "patch flagged %"], 50)
            self.assertEqual(table.loc["late", "patch flagged %"], 0)
            self.assertEqual(table.loc["late", "patch flagged within 3 h %"], 100)
            self.assertEqual(table.loc["any", "patch flagged within 3 h %"], 100)
            self.assertAlmostEqual(table.loc["early", "own largest patch % of domain"], 100 * 12 / 60)
            self.assertEqual(table.loc["early", "patch inside its own largest %"], 50)


if __name__ == "__main__":
    unittest.main()
