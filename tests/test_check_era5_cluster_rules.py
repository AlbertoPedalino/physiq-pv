"""The first rule of the cluster check must give the clusters of check_era5_event_agreement.events_of."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts import check_era5_cluster_rules as script
from scripts.check_era5_event_agreement import events_of


class ClusterRulesTests(unittest.TestCase):
    def test_the_thesis_rule_matches_events_of(self):
        rng = np.random.default_rng(1)
        height, width, times = 9, 11, 40
        rows, cols = np.divmod(np.arange(height * width), width)
        flagged = rng.random((times, height * width)) < 0.15
        ids, patch_times, count = script.patches(flagged, rows, cols, (height, width), 8)
        of_patch = script.clusters(ids, count, 0.0)
        steps, cells, events, n = events_of(flagged, rows, cols, (height, width))
        ours = of_patch[ids[steps, cells]]
        # Same partition of the flagged values: a one-to-one map between the two labellings.
        pairs = set(zip(ours.tolist(), events.tolist()))
        self.assertEqual(len(pairs), len(set(ours.tolist())))
        self.assertEqual(len(pairs), n)

    def test_no_link_in_time_gives_one_cluster_per_patch(self):
        flagged = np.zeros((3, 4), dtype=bool)
        flagged[:, 0] = True  # the same cell at three timestamps
        rows, cols = np.zeros(4, dtype=np.int64), np.arange(4)
        ids, times, count = script.patches(flagged, rows, cols, (1, 4), 8)
        self.assertEqual(len(set(script.clusters(ids, count, None).tolist())), 3)
        self.assertEqual(len(set(script.clusters(ids, count, 0.0).tolist())), 1)


if __name__ == "__main__":
    unittest.main()
