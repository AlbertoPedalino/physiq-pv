"""Who flags a concordant cluster first: the matched groups, the leads in hours, the paired comparison, a detector
that does not vote."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import build_era5_concordant_cube as cube
from scripts import check_era5_concordant_lead as script
from tests.test_build_era5_concordant_cube import write_run


class MatchTests(unittest.TestCase):
    def test_only_the_groups_touching_the_cluster(self):
        flags = np.zeros((6, 1, 6), dtype=bool)
        own = np.zeros_like(flags)
        own[3:5, 0, 1] = True
        flags[1:5, 0, 1] = True  # starts two timestamps before the cluster, at the same cell
        flags[0, 0, 4] = True  # a separate flag earlier: not the same group
        self.assertEqual(script.match_span(flags, own), (1, 4))
        self.assertIsNone(script.match_span(np.zeros_like(flags), own))


class TrackTests(unittest.TestCase):
    def test_along_track_is_positive_ahead_of_the_motion(self):
        own = np.zeros((3, 1, 8), dtype=bool)
        for t in range(3):
            own[t, 0, t + 1] = True  # moving east one cell a timestamp
        ahead = np.roll(own, 1, axis=2)  # one cell east of the cluster at every timestamp
        x = np.arange(8, dtype=float)[None, :] * 50.0  # 50 km a cell
        y = np.zeros((1, 8))
        along, cross, distance = script.track_errors(own, ahead, x, y, 0, 2)
        self.assertAlmostEqual(along, 50.0)
        self.assertAlmostEqual(cross, 0.0)
        self.assertAlmostEqual(distance, 100.0)
        self.assertAlmostEqual(script.track_errors(own, np.roll(own, -1, axis=2), x, y, 0, 2)[0], -50.0)
        self.assertEqual(script.speed_class(100.0, 6), "0.5-1")  # 100 km in 6 h: 0.9 cells every 3 h


class StateTests(unittest.TestCase):
    def test_moderate_flags_ahead_and_far_flags_on_the_cluster(self):
        height, width, times = 6, 16, 10
        rows, cols = np.divmod(np.arange(height * width), width)
        grid = {"rows": rows.tolist(), "cols": cols.tolist(), "latitudes": np.linspace(50, 47.5, height).tolist(),
                "longitudes": np.linspace(5, 12.5, width).tolist()}
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        moving = np.zeros((times, height, width))
        ahead = np.zeros_like(moving)
        for t in range(times):  # a block moving east; d0 also flags the column it enters next
            moving[t, 2:4, t:t + 2] = 10
            ahead[t, 2:4, t + 2:t + 3] = 10
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            runs = []
            for k in range(4):
                signal = moving + ahead if k == 0 else moving if k < 3 else -moving - ahead
                scores = 0.01 * rng.random((times, height * width)) + signal.reshape(times, -1)
                write_run(folder / f"d{k}", scores, timestamps, grid)
                runs.append(f"d{k}={folder / f'd{k}'}")
            inputs = np.zeros((times, height * width, 2), dtype=np.float32)  # far on the block, moderate ahead
            inputs[:, :, 0] = (3 * (moving > 0) + 1 * (ahead > 0)).reshape(times, -1)
            (folder / "cache").mkdir()
            np.save(folder / "cache" / "test.npy", inputs)
            (folder / "model").mkdir()
            (folder / "model" / "metadata.json").write_text(json.dumps(
                {"backend": {"seasonal_normalization": {"mean_file": str(folder / "cache" / "mean.npy")}},
                 "scored_period": "test", "preparation": {"features": ["a", "b"]}}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                cube.main(["--runs", *runs, "--output", str(folder / "out"), "--top-percent", "12", "--votes", "3",
                           "--closure", "none", "--min-values", "5", "--max-hours", "100"])
                script.main(["--cube", str(folder / "out"), "--first", "d0", "--kind", "analysed",
                             "--inputs-run", str(folder / "model"), "--csv", str(folder / "s.csv")])
            table = pd.read_csv(folder / "s.csv")
            big = table.loc[table["values"].idxmax(), "cluster"] if "values" in table else None
            d0 = table[(table["detector"] == "d0") & (table["cluster"] == big)].iloc[0]
            self.assertGreater(d0["along_moderate_km"], 20)  # the column ahead, moderate
            self.assertLess(abs(d0["along_far_km"]), 10)  # the block itself, far, sits on the cluster


class LeadTests(unittest.TestCase):
    def test_leads_and_the_paired_comparison(self):
        height, width, times = 8, 8, 16
        rows, cols = np.divmod(np.arange(height * width), width)
        grid = {"rows": rows.tolist(), "cols": cols.tolist(), "latitudes": np.linspace(50, 46.5, height).tolist(),
                "longitudes": np.linspace(5, 8.5, width).tolist()}
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        block = np.zeros((times, height, width))
        block[5:8, 2:4, 2:4] = 10  # the event, seen by d0, d1, d2
        early = block.copy()
        early[4, 2, 2] = 10  # d0 sees its first cell one timestamp before
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            runs = []
            for k in range(4):
                signal = early if k == 0 else block if k < 3 else -block  # d3 never flags the event, only noise
                scores = 0.01 * rng.random((times, height * width)) + signal.reshape(times, -1)
                write_run(folder / f"d{k}", scores, timestamps, grid)
                runs.append(f"d{k}={folder / f'd{k}'}")
            with contextlib.redirect_stdout(io.StringIO()):
                cube.main(["--runs", *runs, "--output", str(folder / "out"), "--top-percent", "0.5", "--votes", "3",
                           "--closure", "none", "--pad-cells", "2", "--pad-hours", "6", "--min-values", "5"])
                script.main(["--cube", str(folder / "out"), "--first", "d0", "--extra", f"p={folder / 'd0'}",
                             "--csv", str(folder / "lead.csv")])
            table = pd.read_csv(folder / "out" / "clusters.csv")
            big = table.loc[table["values"].idxmax(), "cluster"]
            lead = pd.read_csv(folder / "lead.csv")
            lead = lead[lead["cluster"] == big].set_index("detector")
            self.assertEqual(lead.loc["d0", "start_lead_h"], -3.0)
            self.assertEqual(lead.loc["d1", "start_lead_h"], 0.0)
            self.assertEqual(lead.loc["d1", "end_lead_h"], 0.0)
            self.assertFalse(bool(lead.loc["d3", "matched"]))
            # The detector that does not vote, read from the run of d0, gives what d0 gives.
            self.assertEqual(lead.loc["p", "start_lead_h"], lead.loc["d0", "start_lead_h"])
            frame = pd.read_csv(folder / "lead.csv")
            result = script.paired(frame[frame["matched"]], "d0", "d1", "start_lead_h", "smaller")
            self.assertGreater(result["first ahead %"], 0)

            # The probability of the draws: a spread far smaller than the margin over the threshold on the event keeps it
            # at p >= 0.9; a spread as large as the score everywhere else drops the noise.
            spread = np.where(early.reshape(times, -1) > 0, 0.01, 10.0).astype(np.float32)
            np.save(folder / "d0" / "scores" / "anomaly_std.npy", spread)
            manifest = json.loads((folder / "d0" / "metadata.json").read_text(encoding="utf-8"))
            manifest["anomaly_std_file"] = "scores/anomaly_std.npy"
            (folder / "d0" / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                script.main(["--cube", str(folder / "out"), "--first", "d0", "--probability", "d0", "--levels", "0.9",
                             "--csv", str(folder / "lead_p.csv")])
            lead = pd.read_csv(folder / "lead_p.csv")
            lead = lead[lead["cluster"] == big].set_index("detector")
            self.assertEqual(lead.loc["d0_p0.9", "start_lead_h"], -3.0)
            self.assertIn("curve over the probability", printed.getvalue())
            self.assertIn("d0_rand0.9", lead.index)  # the control at equal number
            # d0 starts one timestamp before the cluster, with a tiny spread: that flag is stable.
            self.assertEqual(lead.loc["d0", "before_values"], 1)
            self.assertEqual(lead.loc["d0", "before_p90"], 1)


if __name__ == "__main__":
    unittest.main()
