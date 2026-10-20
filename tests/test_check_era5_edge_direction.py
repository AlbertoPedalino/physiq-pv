"""The direction check must tell an edge on the side an event moves to from one on the side it leaves."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xarray as xr

from scripts import check_era5_edge_direction as script
from scripts import nowcast_era5_anomalies as nowcast

TIMES = pd.date_range("2005-01-01", periods=120, freq="3h")
HEIGHT, WIDTH = 12, 30
LATITUDES, LONGITUDES = list(60 - 0.5 * np.arange(HEIGHT)), list(0.5 * np.arange(WIDTH))


def run(root, offset, target="simple"):
    """A 3 x 3 block flagged by the other score moves one column east every step, carried by a west wind; the
    model flags the column `offset` columns from its centre. With `target` cape, the CAPE is unusual on the block.
    Returns the table of the check."""
    rows, cols = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
    model, normal = (np.zeros((len(TIMES), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
    for step in range(len(TIMES)):
        centre = 3 + step % 24
        normal[step, np.isin(rows, (4, 5, 6)) & (np.abs(cols - centre) <= 1)] = 5
        model[step, np.isin(rows, (4, 5, 6)) & (cols == centre + offset)] = 5
    grid = {"latitudes": LATITUDES, "longitudes": LONGITUDES, "rows": rows.tolist(), "cols": cols.tolist(), "spacing": 0.5}
    runs = []
    for name, scores in ((f"run_clim_{offset}_{target}", model), (f"run_normal_{offset}_{target}", normal)):
        (root / name / "scores").mkdir(parents=True)
        np.save(root / name / "scores" / "anomaly_mean.npy", scores)
        np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}), encoding="utf-8")
        runs.append(root / name)
    wind = root / "wind" / "2005"
    if not wind.exists():
        wind.mkdir(parents=True)
        shape = (len(TIMES), HEIGHT, WIDTH)
        xr.Dataset({"u": (("valid_time", "latitude", "longitude"), np.full(shape, 10.0, dtype=np.float32)),
                    "v": (("valid_time", "latitude", "longitude"), np.full(shape, 0.5, dtype=np.float32)),
                    "w": (("valid_time", "latitude", "longitude"), np.zeros(shape, dtype=np.float32))},
                   coords={"valid_time": TIMES, "latitude": LATITUDES, "longitude": LONGITUDES}
                   ).to_netcdf(wind / "era5_850_2005_01.nc")
    csv = root / f"direction_{offset}_{target}.csv"
    weather = np.zeros((len(TIMES), HEIGHT * WIDTH, 2), dtype=bool)
    weather[:, :, 0] = normal > 0
    with redirect_stdout(io.StringIO()), patch.object(nowcast, "unseen_flags", return_value=weather):
        script.main(["--score-dir", str(runs[0]), "--by-dir", str(runs[1]), "--top-percent", "5",
                     "--wind-dir", str(root / "wind"), "--csv", str(csv), "--target", target]
                    + ([] if target == "simple" else ["--unseen-dir", str(root)]))
    return pd.read_csv(csv)


class EdgeDirectionTests(unittest.TestCase):
    def test_front_and_back(self):
        with tempfile.TemporaryDirectory() as temp:
            pick = lambda table, kind, direction, target, column: table[
                (table.table == kind) & (table.direction == direction) & (table.target == target)
                & (table.hours == 3)].iloc[0][column]
            front, back = run(Path(temp), 2), run(Path(temp), -2)
            for target in ("shift", "growth"):
                self.assertEqual(pick(front, "own events", "first only", target, "same_side"), 100)
                self.assertGreater(pick(front, "own events", "first only", target, "mean_cosine"), 0.99)
                self.assertEqual(pick(back, "own events", "first only", target, "same_side"), 0)
                # The block goes on as it came and where the wind carries it.
                for reference in ("past motion", "wind at 850 hPa"):
                    self.assertEqual(pick(front, "same events", reference, target, "same_side"), 100)
            # The ring of a square block only has a direction where the border of the grid cuts it.
            self.assertLess(pick(front, "own events", "ring", "shift", "events"),
                            pick(front, "own events", "first only", "shift", "events"))
            case = lambda table, reference, name: table[
                (table.table == "cases") & (table.direction == reference) & (table.target == "shift")
                & (table.hours == 3) & (table.case == name)].iloc[0].share
            self.assertEqual(case(front, "wind at 850 hPa", "both right"), 100)
            self.assertEqual(case(back, "wind at 850 hPa", "reference only"), 100)
            self.assertEqual(case(back, "past motion", "reference only"), 100)
            # The edge ahead agrees with the wind and both are right; the edge behind never agrees, and then the
            # wind is right and the edge is not.
            agreement = lambda table, reference, column: table[
                (table.table == "agreement") & (table.direction == reference) & (table.target == "shift")
                & (table.hours == 3)].iloc[0][column]
            for reference in ("past motion", "wind at 850 hPa"):
                self.assertEqual(agreement(front, reference, "agree"), 100)
                self.assertEqual(agreement(front, reference, "right if agree"), 100)
                self.assertEqual(agreement(front, reference, "combined right"), 100)
                self.assertEqual(agreement(back, reference, "agree"), 0)
                self.assertEqual(agreement(back, reference, "ref right if not"), 100)
                self.assertEqual(agreement(back, reference, "edge right if not"), 0)
                self.assertTrue(np.isnan(agreement(back, reference, "right if agree")))
            # Judged by the CAPE, unusual where the block arrives: the edge ahead points there, the one behind not.
            front, back = run(Path(temp), 2, "cape"), run(Path(temp), -2, "cape")
            for target in ("shift", "growth"):
                self.assertEqual(pick(front, "own events", "first only", target, "same_side"), 100)
                self.assertEqual(pick(back, "own events", "first only", target, "same_side"), 0)
                self.assertEqual(pick(front, "same events", "wind at 850 hPa", target, "same_side"), 100)
            # With the dropout draws of a run: the spreads over the edge are read (all right here, so no AUC).
            model_dir, other_dir = Path(temp) / "run_clim_2_simple", Path(temp) / "run_normal_2_simple"
            for name in ("generator_mean", "generator_std", "discriminator_mean", "discriminator_std"):
                np.save(model_dir / "scores" / f"{name}.npy", np.ones((len(TIMES), HEIGHT * WIDTH), dtype=np.float32))
            with redirect_stdout(io.StringIO()) as printed:
                script.main(["--score-dir", str(model_dir), "--by-dir", str(other_dir), "--top-percent", "5",
                             "--components-dir", str(model_dir)])
            self.assertIn("dropout spreads over the edge", printed.getvalue())
        with self.assertRaisesRegex(ValueError, "needs --unseen-dir"):
            script.main(["--score-dir", "x", "--by-dir", "x", "--target", "cape"])


if __name__ == "__main__":
    unittest.main()
