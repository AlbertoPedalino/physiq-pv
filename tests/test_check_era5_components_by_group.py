"""The components check must read G, D and their dropout spread in the groups of two scores."""
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

from scripts import check_era5_components_by_group as script

TIMES = pd.date_range("2005-01-01", periods=400, freq="3h")
CELLS = 50
GRID = {"latitudes": [45.0], "longitudes": list(np.arange(CELLS) * 0.5), "rows": [0] * CELLS,
        "cols": list(range(CELLS)), "spacing": 0.5}


def run(directory, files):
    (directory / "scores").mkdir(parents=True)
    for name, values in files.items():
        np.save(directory / "scores" / f"{name}.npy", values.astype(np.float32))
    np.save(directory / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
    (directory / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID,
                                                         "anomaly_mean_file": "scores/anomaly_mean.npy"}), encoding="utf-8")
    return directory


class ComponentsByGroupTests(unittest.TestCase):
    def test_spread_and_discriminator_where_the_first_score_flags_alone(self):
        rng = np.random.default_rng(0)
        shape = (len(TIMES), CELLS)
        first, other = rng.random(shape), rng.random(shape)
        first[:, :5] += 10  # cells 0-4: the first score alone
        first[:, 5:10] += 10  # cells 5-9: both
        other[:, 5:15] += 10  # cells 10-14: the other alone
        g_mean = np.ones(shape)
        g_std = np.full(shape, 0.1)
        g_std[:, :5] = 0.5  # the draws of G disagree where the first score flags alone
        d_mean = rng.random(shape)
        d_mean[:, :10] += 10  # and D sees those values as unusual, as those both flag
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            score = run(root / "first", {"anomaly_mean": first})
            others = run(root / "other", {"anomaly_mean": other})
            components = run(root / "components", {"anomaly_mean": first, "generator_mean": g_mean, "generator_std": g_std,
                                                   "discriminator_mean": d_mean, "discriminator_std": np.full(shape, 0.2)})
            with redirect_stdout(io.StringIO()):
                result = script.main(["--score-dir", str(score), "--by-dir", str(others), "--components-dir", str(components),
                                      "--top-percent", "20", "--stride", "1"])
        alone, none = result["first alone"], result["none"]
        self.assertEqual(alone["values"], 5 * len(TIMES))
        self.assertEqual(result["both"]["values"], 5 * len(TIMES))
        self.assertEqual(result["others alone"]["values"], 5 * len(TIMES))
        self.assertAlmostEqual(alone["G spr/mean"], 0.5, places=5)
        self.assertAlmostEqual(none["G spr/mean"], 0.1, places=5)
        self.assertAlmostEqual(alone["D flags %"], 100)
        self.assertEqual(none["D flags %"], 0)
        self.assertAlmostEqual(none["D spread"], 0.2, places=5)


    def test_spreads_that_tell_the_hits(self):
        rng = np.random.default_rng(0)
        hit = np.arange(1000) < 300
        values = np.column_stack([np.where(hit, 2.0, 1.0) + 0.1 * rng.random(1000), rng.random(1000) + 1,
                                  rng.random(1000)])
        rows = {row["spread"]: row for row in script.compare_spreads(hit, values)}
        self.assertEqual(rows["G spread"]["AUC"], 1.0)
        self.assertAlmostEqual(rows["D spread"]["AUC"], 0.5, delta=0.1)
        self.assertGreater(rows[script.COMBINED]["AUC"], 0.8)  # G carries D along
        self.assertEqual(rows["G spread"]["hits"], 300)
        with redirect_stdout(io.StringIO()) as printed:
            script.print_spreads("title", [("group", list(rows.values()))])
        self.assertIn("mean rank", printed.getvalue())
        # G up where the hits are, D down there: only the opposed combination sees both.
        values[:, 2] = np.where(hit, 0.0, 1.0) + 0.1 * rng.random(1000)
        rows = {row["spread"]: row for row in script.compare_spreads(hit, values)}
        self.assertEqual(rows[script.OPPOSED]["AUC"], 1.0)
        self.assertLess(rows[script.COMBINED]["AUC"], rows[script.OPPOSED]["AUC"])

    def test_spreads_over_their_usual_value(self):
        # The spread of G is twice as large at 12 UTC as at the other hours, at every cell.
        hours = np.asarray(TIMES.hour)
        spread = np.where(hours == 12, 2.0, 1.0)[:, None] * np.ones((1, CELLS))
        components = {"generator_mean": np.full((len(TIMES), CELLS), 4.0), "generator_std": spread,
                      "discriminator_std": np.full((len(TIMES), CELLS), 0.5)}
        normals = script.build_spread_normals(components, TIMES)
        noon = int(np.nonzero(hours == 12)[0][0])
        self.assertTrue(np.allclose(script.spread_normal(normals, TIMES[noon])[:, 0], 2.0))
        self.assertTrue(np.allclose(script.spreads_at(components, noon, script.spread_normal(normals, TIMES[noon])), 1.0))
        self.assertIsNone(script.spread_normal(None, TIMES[0]))
        with tempfile.TemporaryDirectory() as temp:
            from scripts import build_era5_spread_normals as build
            run(Path(temp) / "run", {name: values for name, values in components.items()}
                | {"discriminator_mean": np.zeros((len(TIMES), CELLS))})
            with redirect_stdout(io.StringIO()):
                build.main(["--components-dir", str(Path(temp) / "run"), "--output", str(Path(temp) / "n.npz")])
            loaded = script.load_spread_normals(Path(temp) / "n.npz")
        self.assertTrue(np.allclose(loaded["values"], normals["values"]))


if __name__ == "__main__":
    unittest.main()
