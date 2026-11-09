"""The best of several scores on the known events: highest share, which score, median, most days seen."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from scripts import summarize_era5_known_events_best as script


class BestTests(unittest.TestCase):
    def test_best_share_and_days(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # a sees event x more, b sees event y more; days over 10%: a 2 on x, b 1 on x and 2 on y
            for name, shares, daily in (("a", (30, 2), ((20, 15, 3), (1, 2, None))),
                                        ("b", (10, 8), ((12, 5, 1), (11, 30, None)))):
                (root / name).mkdir()
                pd.DataFrame({"id": ["x", "y"], "segnalati %": shares}).to_csv(root / name / "eventi_noti.csv",
                                                                                 index=False)
                pd.DataFrame(daily, index=pd.Index(["x", "y"], name="id"), columns=["0", "1", "2"]).to_csv(
                    root / name / "eventi_per_giorno.csv")
            csv = root / "best.csv"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(script.main(["--runs", f"a={root / 'a'}", f"b={root / 'b'}", "--csv", str(csv)]), 0)
            table = pd.read_csv(csv).set_index("id")
            self.assertEqual(list(table["metodo"]), ["a", "b"])
            self.assertEqual(list(table["migliore %"]), [30, 8])
            self.assertEqual(list(table["mediana %"]), [20, 5])
            self.assertEqual(list(table["giorni visti migliore"]), [2, 2])


if __name__ == "__main__":
    unittest.main()
