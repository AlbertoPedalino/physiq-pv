"""The nowcast of new anomalies must learn the wind for the moving ones and the model for those that form."""
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

from scripts import nowcast_era5_anomalies as script

HEIGHT, WIDTH = 12, 24
LATITUDES, LONGITUDES = 5.0 - 0.5 * np.arange(HEIGHT), 0.5 * np.arange(WIDTH)
ROWS, COLS = np.divmod(np.arange(HEIGHT * WIDTH), WIDTH)
# A wind that carries a field one column east in 3 hours.
EAST = script.METERS_PER_DEGREE * np.cos(np.radians(LATITUDES))[:, None] * 0.5 / (3 * 3600)


def scores(times, seed):
    """The simple score (events at 10, else 0) and the model (5 the step before a formation, and false alarms)."""
    rng = np.random.default_rng(seed)
    simple, model = (np.zeros((len(times), HEIGHT * WIDTH), dtype=np.float32) for _ in range(2))
    for start in range(10, len(times) - WIDTH, 40):  # a 2 x 2 block crossing the top rows eastwards
        for step in range(WIDTH - 1):
            simple[start + step, (ROWS < 2) & ((COLS == step) | (COLS == step + 1))] = 10
    for step in range(12, len(times) - 1, 6):  # single cells forming in the calm bottom rows
        cell = (HEIGHT - 2) * WIDTH + rng.integers(WIDTH)
        simple[step + 1, cell] = 10
        model[step, cell] = 5
        model[step, (HEIGHT - 1) * WIDTH + rng.integers(WIDTH)] = 5  # an alarm with nothing after
    return simple, model


def write_run(directory, times, values):
    (directory / "scores").mkdir(parents=True)
    np.save(directory / "scores" / "anomaly_mean.npy", values)
    np.save(directory / "test_timestamps.npy", times.as_unit("ns").asi8)
    (directory / "metadata.json").write_text(json.dumps({"status": "complete", "grid": {
        "latitudes": list(LATITUDES), "longitudes": list(LONGITUDES), "rows": ROWS.tolist(), "cols": COLS.tolist()}}),
        encoding="utf-8")
    return directory


class NowcastTests(unittest.TestCase):
    def test_wind_for_the_moving_anomalies_and_the_model_for_those_that_form(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = {}
            for name, start, seed in (("fit", "2000-01-01", 0), ("read", "2005-01-01", 1)):
                times = pd.date_range(start, periods=440, freq="3h")
                simple, model = scores(times, seed)
                runs[name] = (write_run(root / f"{name}_model", times, model), write_run(root / f"{name}_simple", times, simple))
                for month in sorted(set(times.month)):
                    chosen = times.month == month
                    folder = root / "wind" / str(times.year[0])
                    folder.mkdir(parents=True, exist_ok=True)
                    u = np.broadcast_to(EAST, (int(chosen.sum()), HEIGHT, WIDTH)).astype(np.float32)
                    xr.Dataset({"u": (("valid_time", "latitude", "longitude"), u),
                                "v": (("valid_time", "latitude", "longitude"), np.zeros_like(u))},
                               coords={"valid_time": times[chosen], "latitude": LATITUDES, "longitude": LONGITUDES}
                               ).to_netcdf(folder / f"era5_850_{times.year[0]}_{month:02d}.nc")
            with redirect_stdout(io.StringIO()):
                rows = script.main(["--fit-model-dir", str(runs["fit"][0]), "--fit-dir", str(runs["fit"][1]),
                                    "--score-model-dir", str(runs["read"][0]), "--score-dir", str(runs["read"][1]),
                                    "--wind-dir", str(root / "wind"), "--event-percent", "10", "--stride", "1",
                                    "--negative-share", "0.3", "--alarm-percent", "5"])
        table = pd.DataFrame(rows).set_index(["kind", "score"])
        auc = lambda kind, score: table.loc[(kind, score), "ROC AUC"]
        self.assertGreater(auc("movement", "downwind alone"), 0.8)
        self.assertGreater(auc("movement", "A, with the model"), 0.8)
        self.assertGreater(auc("formation", "A, with the model"), auc("formation", "A, without the model") + 0.2)
        self.assertGreater(auc("formation", "B, with the model"), auc("formation", "B, without the model") + 0.2)
        self.assertGreater(table.loc[("formation", "B, with the model"), "caught %"], 50)


if __name__ == "__main__":
    unittest.main()
