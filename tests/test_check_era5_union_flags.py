"""Several second scores are joined on the timestamps they all have; the wind at 850 hPa can come from the cache."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_edge_direction as direction
from scripts import check_era5_lead as lead


def run(folder, times, values):
    (folder / "scores").mkdir(parents=True)
    np.save(folder / "scores" / "anomaly_mean.npy", values)
    np.save(folder / "test_timestamps.npy", times.as_unit("ns").asi8)
    (folder / "metadata.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    return folder


class UnionFlagsTests(unittest.TestCase):
    def test_union_on_the_common_timestamps(self):
        times = pd.date_range("2005-01-01", periods=40, freq="3h")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a_values, b_values = np.zeros((40, 3), np.float32), np.zeros((38, 3), np.float32)
            a_values[::4, 0] = 1  # 10 of 120 values: above the highest 10% by itself
            b_values[1::4, 2] = 1
            a, b = run(root / "a", times, a_values), run(root / "b", times[2:], b_values)
            common = lead.common_timestamps([a, b])
            self.assertTrue(common.equals(times[2:]))
            union = lead.union_flags([a, b], 10, common)
            self.assertEqual(union.shape, (38, 3))
            self.assertTrue(union[2, 0] and union[1, 2] and not union[0, 1])  # a at 2005-01-01 12:00, b at its 2nd
            self.assertEqual(lead.short_names(root / "run_model", [root / "run_a", root / "run_b"]),
                             ["model", "a or b"])

    def test_wind_of_the_cache(self):
        times = pd.date_range("2005-01-01", periods=6, freq="3h")
        cube = np.arange(6 * 2 * 4, dtype=np.float32).reshape(6, 2, 4)
        wind = direction.wind_of_cache((cube, times, [1, 3]), times[[4, 1]])
        self.assertTrue(np.array_equal(wind, cube[[4, 1]][:, :, [1, 3]]))


if __name__ == "__main__":
    unittest.main()
