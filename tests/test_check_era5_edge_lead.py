"""The edge lead check must tell a score that flags the cells a front reaches next from one that does not."""
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

from scripts import check_era5_edge_lead as script

TIMES = pd.date_range("2005-01-01", periods=300, freq="3h")
HEIGHT, WIDTH = 12, 30


class EdgeLeadTests(unittest.TestCase):
    def test_a_front_seen_two_columns_ahead(self):
        # A front moves one column east every 3 hours and every 40 steps a new one crosses the grid. The other
        # score flags the column of the front, the model the column two ahead: the model flags a cell 6 h before.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
            model, normal = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
            for begin in range(10, len(TIMES), 40):
                for step in range(WIDTH + 2):
                    for score, column in ((model, step + 2), (normal, step)):
                        if column < WIDTH and begin + step < len(TIMES):
                            score[begin + step, cols == column] = 5
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
            # The gust is strong where the model flags: the weather is there before the other score sees it.
            rng = np.random.default_rng(0)
            gust = rng.random(model.shape).astype(np.float32) + 10 * (model > 0)
            for month in (1, 2):
                chosen = TIMES.month == month
                (root / "unseen" / "2005").mkdir(parents=True, exist_ok=True)
                xr.Dataset({"i10fg": (("valid_time", "latitude", "longitude"), gust[chosen].reshape(-1, HEIGHT, WIDTH))},
                           coords={"valid_time": TIMES[chosen], "latitude": latitudes, "longitude": longitudes}
                           ).to_netcdf(root / "unseen" / "2005" / f"era5_independent_2005_{month:02d}.nc")
            csv = root / "edge.csv"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]), "--top-percent", "10",
                                              "--unseen-dir", str(root / "unseen"), "--csv", str(csv)]), 0)
            table = pd.read_csv(csv)
            pick = lambda group, distance, hours, kind: table[
                (table.group == group) & (table.distance == distance) & (table.window_hours == hours)
                & (table.kind == kind)].iloc[0].share
            # Two cells from the front: the model's cells are reached within 6 h, not within 3; the others are behind.
            self.assertGreater(pick("first only", "2", 6, "front"), 95)
            self.assertLess(pick("first only", "2", 3, "front"), 5)
            self.assertGreater(pick("neither", "2", 6, "trail"), 95)
            self.assertLess(pick("neither", "2", 6, "front"), 5)
            text = output.getvalue()
            arrival = next(line for line in text.split("hours until")[1].splitlines() if line.startswith("2, clim only"))
            self.assertEqual(arrival.split()[-1], "6")
            # The gust is strong on the fronts of the model (every pass of a front at a cell, more than 5% of the
            # values of a cell and hour: not all unusual), far more than on the cells it does not flag.
            lines = text.split("unseen variable")[1].splitlines()
            share = lambda start, column: float(next(line for line in lines if line.startswith(start)).split()[column])
            self.assertGreater(share("2, clim only, i10fg", -4), 3 * share("2, neither, i10fg", -3))
            # With several second scores a value is flagged when any of them flags it: adding one that never
            # flags changes nothing.
            (root / "run_quiet" / "scores").mkdir(parents=True)
            np.save(root / "run_quiet" / "scores" / "anomaly_mean.npy", np.zeros_like(normal))
            np.save(root / "run_quiet" / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
            (root / "run_quiet" / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}),
                                                              encoding="utf-8")
            union = root / "edge_union.csv"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]),
                                              str(root / "run_quiet"), "--top-percent", "10", "--csv", str(union)]), 0)
            pd.testing.assert_series_equal(pd.read_csv(union).share, table.share)


if __name__ == "__main__":
    unittest.main()
