import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.set_era5_cube_min_values import kind_counts, main, relabel


class RelabelTest(unittest.TestCase):
    def test_kinds(self):
        table = pd.DataFrame({"values": [5, 30, 30, 100], "hours": [0, 24, 96, 72]})
        kinds = relabel(table, 20, 72)["kind"].tolist()
        self.assertEqual(kinds, ["small", "analysed", "long", "analysed"])
        kinds = relabel(table, 50, 72)["kind"].tolist()
        self.assertEqual(kinds, ["small", "small", "small", "analysed"])

    def test_counts(self):
        table = relabel(pd.DataFrame({"values": [5, 15], "hours": [0, 0]}), 10, 72)
        counts = kind_counts(table)
        self.assertEqual(counts["small"]["clusters"], 1)
        self.assertAlmostEqual(counts["analysed"]["values_percent"], 75.0)

    @unittest.skipIf(os.name == "nt", "symbolic links need privileges on Windows")
    def test_writes_a_cube(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube, out = Path(tmp) / "cube", Path(tmp) / "out"
            (cube / "votes").mkdir(parents=True)
            (cube / "flags").mkdir()
            pd.DataFrame({"cluster": [1, 2], "values": [5, 40], "hours": [0, 96], "kind": ["small", "long"]}).to_csv(
                cube / "clusters.csv", index=False)
            (cube / "summary.json").write_text(json.dumps({"selection": {"min_values": 20, "max_hours": 72}}))
            main(["--cube", str(cube), "--output", str(out), "--min-values", "50"])
            self.assertEqual(pd.read_csv(out / "clusters.csv")["kind"].tolist(), ["small", "small"])
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["selection"]["min_values"], 50)
            self.assertTrue((out / "votes").is_symlink() and (out / "flags").is_symlink())


if __name__ == "__main__":
    unittest.main()
