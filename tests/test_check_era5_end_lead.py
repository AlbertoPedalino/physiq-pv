"""The end lead check must tell a score that stops where the other stops next from one that does not."""
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
import xarray as xr

from scripts import check_era5_end_lead as script

TIMES = pd.date_range("2005-01-01", periods=400, freq="3h")
HEIGHT, WIDTH = 6, 10


class EndLeadTests(unittest.TestCase):
    def test_the_first_stops_one_step_before_the_second(self):
        # Episodes of 8 steps every 20 steps over every cell: the other score flags all 8 steps, the model the first 6
        # only in half the cells (it stops 2 steps before the other there), all 8 in the other half. The gust is strong
        # during the 8 steps, so where the model stopped the gust is still strong one step later, then gone.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cells = HEIGHT * WIDTH
            rows, cols = np.divmod(np.arange(cells), WIDTH)
            model, other = (np.zeros((len(TIMES), cells), dtype=np.float32) for _ in range(2))
            early = np.arange(cells) % 2 == 0
            for begin in range(5, len(TIMES) - 10, 20):
                other[begin:begin + 8] = 5
                model[begin:begin + 6] = 5
                model[begin + 6:begin + 8, ~early] = 5
            latitudes, longitudes = list(60 - 0.5 * np.arange(HEIGHT)), list(0.5 * np.arange(WIDTH))
            grid = {"latitudes": latitudes, "longitudes": longitudes, "rows": rows.tolist(), "cols": cols.tolist(),
                    "spacing": 0.5}
            runs = []
            for name, scores in (("run_model", model), ("run_other", other)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", scores)
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                metadata = {"status": "complete", "grid": grid}
                if name == "run_model":  # the dropout draws: low spread where the model stops early
                    spread = np.where(early, 0.01, 0.5)[None, :] * np.ones((len(TIMES), 1), dtype=np.float32)
                    np.save(root / name / "scores" / "generator_std.npy", spread.astype(np.float32))
                    np.save(root / name / "scores" / "generator_mean.npy", np.ones_like(spread, dtype=np.float32))
                    metadata.update(generator_std_file="scores/generator_std.npy",
                                    generator_mean_file="scores/generator_mean.npy")
                (root / name / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                runs.append(root / name)
            rng = np.random.default_rng(0)
            gust = rng.random(other.shape).astype(np.float32) + 10 * (other > 0)
            (root / "unseen" / "2005").mkdir(parents=True)
            for month in (1, 2):
                chosen = TIMES.month == month
                xr.Dataset({"i10fg": (("valid_time", "latitude", "longitude"), gust[chosen].reshape(-1, HEIGHT, WIDTH))},
                           coords={"valid_time": TIMES[chosen], "latitude": latitudes, "longitude": longitudes}
                           ).to_netcdf(root / "unseen" / "2005" / f"era5_independent_2005_{month:02d}.nc")
            # Standardised inputs: three variables far from normal while the other score flags, none otherwise.
            cache = root / "cache"
            cache.mkdir()
            standardised = np.zeros((len(TIMES) + 2, cells, 15), dtype=np.float32)
            standardised[2:, :, :3] = 3.0 * (other > 0)[:, :, None]
            np.save(cache / "test.npy", standardised)
            np.save(cache / "seasonal_mean.npy", np.zeros(1, dtype=np.float32))
            (root / "inputs").mkdir()
            np.save(root / "inputs" / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
            (root / "inputs" / "metadata.json").write_text(json.dumps({
                "status": "complete", "grid": grid, "scored_period": "test",
                "backend": {"seasonal_normalization": {"mean_file": str(cache / "seasonal_mean.npy")}},
                "preparation": {"features": [f"v{i}" for i in range(15)]}}), encoding="utf-8")
            csv = root / "end.csv"
            output = io.StringIO()
            with redirect_stdout(output):
                script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]), "--top-percent", "50",
                             "--unseen-dir", str(root / "unseen"), "--ratio-run", str(runs[0]),
                             "--inputs-run", str(root / "inputs"), "--csv", str(csv)])
            table = pd.read_csv(csv, index_col=0)
            stopped = table["second stopped within 3 h %"]
            # Where the model stopped (two steps before the other), the other has not stopped yet within 3 h but has
            # within 6 h; where the model goes on, the other stops with it at the end of the episode.
            self.assertAlmostEqual(stopped["first stopped"], 0.0)
            self.assertAlmostEqual(table["second stopped within 6 h %"]["first stopped"], 100.0)
            self.assertGreater(stopped["first goes on"], 0.0)
            thirds = pd.read_csv(root / "end_terzi.csv", index_col=0)
            self.assertEqual(int(thirds["cells"].sum()), int(table["cells"]["first stopped"]))
            self.assertIn("first stopped, by third", output.getvalue())
            # At equal strength (three variables far at t in both groups) the result is the same as overall.
            strength = pd.read_csv(root / "end_forza.csv")
            row = strength[(strength["variables beyond sd at t"] == "2-3") & (strength["group"] == "first stopped")]
            self.assertEqual(int(row["cells"].iloc[0]), int(table["cells"]["first stopped"]))
            self.assertAlmostEqual(float(row["second stopped within 6 h %"].iloc[0]), 100.0)
            # The thirds at equal strength: every cell where the model stopped is in one third, compared with the
            # cells where it goes on weighted to the same strengths.
            equal = pd.read_csv(root / "end_terzi_forza.csv", index_col=0)
            self.assertEqual(int(equal["cells"].sum()), int(table["cells"]["first stopped"]))
            self.assertIn("  same, goes on weighted (3 h)", equal.columns)
            # Six hours later the episode is over where the model stopped: no variable is far any more.
            self.assertAlmostEqual(table["no variable beyond 2 sd +6h %"]["first stopped"], 100.0)
            self.assertAlmostEqual(table["no variable beyond 2 sd +3h %"]["first stopped"], 0.0)


if __name__ == "__main__":
    unittest.main()
