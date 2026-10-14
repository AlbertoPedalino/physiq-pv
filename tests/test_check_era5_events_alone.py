"""The events check must tell the events a score sees alone from those it shares, and the convinced ones."""
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

from scripts import check_era5_events_alone as script

TIMES = pd.date_range("2005-01-01", periods=400, freq="3h")
GRID = {"latitudes": [45.0, 44.5], "longitudes": [7.0, 7.5, 8.0], "rows": [0, 0, 0, 1, 1, 1], "cols": [0, 1, 2, 0, 1, 2],
        "spacing": 0.5}


class EventsAloneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        model, normal = (rng.random((len(TIMES), 6)).astype(np.float32) * 0.1 for _ in range(2))
        spread = np.full(model.shape, 0.001, dtype=np.float32)
        # Event 1, two steps on cells 0 and 1, the model alone and sure; event 3 on cells 3 and 4, shared;
        # event 4 on cell 5, the model alone and unsure.
        model[10:12, :2] = 1
        normal[10:12, :2] = 0
        model[20, 3:5] = normal[20, 3:5] = 1
        model[30, 5] = 1
        normal[30, 5] = 0
        spread[30, 5] = 5
        self.model = self.write_run("run_model", model, spread)
        self.normal = self.write_run("run_normal", normal)
        labels = np.zeros((len(TIMES), 2, 3), dtype=np.int64)
        labels[10, 0, :2], labels[11, 0, :2], labels[20, 1, :2], labels[30, 1, 2] = 1, 2, 3, 4
        self.events = self.root / "events"
        self.events.mkdir()
        np.save(self.events / "cluster_labels.npy", labels)
        np.save(self.events / "timestamps.npy", TIMES.as_unit("ns").asi8)
        pd.DataFrame({"cluster_id": [1, 2, 3, 4], "event_id": [1, 1, 3, 4]}).to_csv(self.events / "clusters.csv", index=False)
        pd.DataFrame({"event_id": [1, 3, 4], "duration_hours": [6.0, 3.0, 3.0]}).to_csv(self.events / "events.csv",
                                                                                         index=False)

    def write_run(self, name, scores, spread=None):
        run = self.root / name
        (run / "scores").mkdir(parents=True)
        np.save(run / "scores" / "anomaly_mean.npy", scores)
        manifest = {"status": "complete", "grid": GRID, "anomaly_mean_file": "scores/anomaly_mean.npy"}
        if spread is not None:
            np.save(run / "scores" / "anomaly_std.npy", spread)
            manifest["anomaly_std_file"] = "scores/anomaly_std.npy"
        np.save(run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (run / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
        return run

    def run_script(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--events-dir", str(self.events), "--score-dir", str(self.model), "--by-dirs",
                                          str(self.normal), "--top-percent", "3", *map(str, arguments)]), 0)
        text = output.getvalue()
        return text, {line[:script.NAME].strip(): line[script.NAME:].split() for line in text.splitlines()
                      if line[:script.NAME].strip() and not line.startswith(("[", "class"))}

    def test_the_classes_and_the_convinced_events(self):
        text, rows = self.run_script()
        self.assertIn(": 3 events", text)
        self.assertEqual(rows["all events"][0], "3")
        self.assertEqual(rows["seen by the score alone"][:4], ["2", "66.7", "5", "50.0"])
        self.assertEqual(rows["shared with the other scores"][:4], ["1", "33.3", "2", "100.0"])
        self.assertEqual(rows["partly"][0], "0")
        # By duration, among the events seen alone: the long one is convinced, the short one is not.
        self.assertEqual(rows["one timestep"][:4], ["1", "50.0", "1", "0.0"])
        self.assertEqual(rows["up to 24 h"][:4], ["1", "50.0", "4", "100.0"])
        self.assertEqual(rows["more than 24 h"][0], "0")

    def test_refusals(self):
        manifest = json.loads((self.model / "metadata.json").read_text(encoding="utf-8"))
        del manifest["anomaly_std_file"]
        (self.model / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "kept no spread"):
            self.run_script()


if __name__ == "__main__":
    unittest.main()
