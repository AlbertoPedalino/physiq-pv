"""Matching of the groups of a detector to the concordant clusters (Bourdin et al. 2022): near within the radius at the
same timestamp, groups of one cluster merged, the criterion in time of Crawford et al. (2021) on the leads and why a
match fails it, the border of the period, the sensitivities (Davis, cut), POD and FAR, the groups near small clusters
only and those without counterpart with their time from the nearest target; checked on a case made by hand, against
brute force on real latitudes, and through the whole script (also against the cube without the detector's vote)."""
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
from scripts import check_era5_concordant_match as script
from scripts.era5_km_zone import haversine_km, row_offsets


def hand_case():
    """30 timestamps, 3 rows at 45N (0.5 deg = 39 km along a row), radius 100 km: up to 2 cells along a row."""
    times, height, width = 30, 3, 20
    offsets = row_offsets(np.full(height, 45.0), 0.5, 100.0)
    clusters = np.zeros((times, height, width), dtype=np.int32)
    clusters[10:14, 1, 2:4] = 1  # analysed, timestamps 10-13
    clusters[5:26, 1, 15:17] = 2  # long, timestamps 5-25
    clusters[2, 1, 9] = 3  # small
    table = pd.DataFrame({"cluster": [1, 2, 3], "kind": ["analysed", "long", "small"],
                          "t0": [10, 5, 2], "t1": [13, 25, 2], "values": [8, 42, 1]})
    groups = np.zeros_like(clusters)
    groups[8:14, 1, 4:6] = 1  # next to cluster 1, starts 2 timestamps earlier
    groups[20:30, 0, 14] = 2  # next to cluster 2, mostly after it, up to the last timestamp of the period
    groups[2, 1, 10] = 3  # next to the small cluster only
    groups[3:6, 1, 7] = 4  # 4 cells (157 km) from cluster 1, far from 2; cluster 2 exists meanwhile
    groups[1:5, 1, 4] = 5  # where cluster 1 will be, ending 6 timestamps before it starts
    groups[11:13, 1, 1] = 6  # also next to cluster 1: merged with group 1
    groups[16:18, 1, 4] = 7  # where cluster 1 was, starting 3 timestamps after it ends
    return times, offsets, clusters, table, groups


class MatchingTests(unittest.TestCase):
    def test_hand_made_case(self):
        times, offsets, clusters, table, groups = hand_case()
        result, lonely, summary = script.evaluate("d", clusters, table, groups, 7, offsets, 3.0, 8, 0.6, (0.5, 2.0),
                                                  times)
        one = result.set_index("cluster").loc[1]
        self.assertTrue(one["caught"])
        self.assertEqual(one["groups"], 2)
        self.assertEqual((one["start"], one["end"], one["n1"], one["n2"], one["nm"]), (8, 13, 6, 4, 4))
        self.assertAlmostEqual(one["crawford"], 0.8)
        self.assertTrue(one["passes_crawford"])
        self.assertTrue(one["passes_davis"])  # 4 / 6
        self.assertFalse(one["border"])
        self.assertEqual((one["start_lead_h"], one["end_lead_h"]), (-6.0, 0.0))
        self.assertEqual(one["start_cut_h"], -6.0)  # within the cut of 8 timestamps
        two = result.set_index("cluster").loc[2]
        self.assertTrue(two["caught"])
        self.assertAlmostEqual(two["crawford"], 2 * 6 / (10 + 21))
        self.assertFalse(two["passes_crawford"])
        self.assertEqual(two["reason"], "shorter")
        self.assertFalse(two["passes_davis"])  # 21 / 10 > 2
        self.assertTrue(two["border"])  # the match reaches the last timestamp of the period
        self.assertEqual(two["end_lead_h"], 12.0)
        self.assertNotIn(3, result["cluster"].tolist())  # small clusters are no targets
        self.assertEqual((summary["matched"], summary["near_small_only"], summary["without_counterpart"]), (3, 1, 3))
        alone = lonely.set_index("group")
        self.assertEqual(alone.loc[5, "gap_h"], -18.0)  # ends at 4, cluster 1 starts at 10, within 100 km
        self.assertEqual(alone.loc[7, "gap_h"], 9.0)  # starts at 16, cluster 1 ended at 13
        self.assertTrue(np.isnan(alone.loc[4, "gap_h"]))
        self.assertEqual(alone.loc[4, "category"], "coexists, far")  # cluster 2 exists at timestamps 5
        self.assertEqual(alone.loc[5, "category"], "near a target in time")

        # The cut is active with a cut of one timestamp: the start lead of cluster 1 becomes -3 h.
        cut, _, _ = script.evaluate("d", clusters, table, groups, 7, offsets, 3.0, 1, 0.6, (0.5, 2.0), times)
        self.assertEqual(cut.set_index("cluster").loc[1, "start_cut_h"], -3.0)

        # POD and FAR (Bourdin eqs. 3-4): H = 2, M = 0; FA = 3 without counterpart, 4 with the one near a small cluster.
        rates = script.scores(result, [summary]).set_index("kind")
        self.assertEqual(rates.loc["analysed", "POD"], 1.0)
        self.assertAlmostEqual(rates.loc["analysed", "FAR (without counterpart)"], 3 / 5)
        self.assertAlmostEqual(rates.loc["analysed", "FAR (+ near small only)"], 4 / 6)
        self.assertEqual(rates.loc["long", "failing: longer / shorter / shifted"], "0/1/0")

    def test_a_longer_group_fails_crawford(self):
        times, offsets, clusters, table, _ = hand_case()
        groups = np.zeros_like(clusters)
        groups[0:29, 1, 4] = 1  # next to cluster 1 for 29 timestamps: n1 = 29, n2 = 4, nm = 4
        result, _, _ = script.evaluate("d", clusters, table, groups, 1, offsets, 3.0, 8, 0.6, (0.5, 2.0), times)
        one = result.set_index("cluster").loc[1]
        self.assertAlmostEqual(one["crawford"], 8 / 33)
        self.assertEqual(one["reason"], "longer")
        self.assertTrue(one["border"])  # starts at the first timestamp of the period

    def test_the_cluster_with_most_associated_values_wins_and_ties_go_low(self):
        times, height, width = 5, 1, 12
        offsets = row_offsets(np.full(height, 45.0), 0.5, 100.0)
        clusters = np.zeros((times, height, width), dtype=np.int32)
        clusters[1:3, 0, 2] = 1  # two values near the group
        clusters[1:3, 0, 6:9] = 2  # six values, four of them near the group
        table = pd.DataFrame({"cluster": [1, 2], "kind": ["analysed", "analysed"], "t0": [1, 1], "t1": [2, 2],
                              "values": [2, 6]})
        groups = np.zeros_like(clusters)
        groups[1:3, 0, 4:6] = 1
        result, _, _ = script.evaluate("d", clusters, table, groups, 1, offsets, 3.0, 8, 0.6, (0.5, 2.0), times)
        caught = result.set_index("cluster")["caught"]
        self.assertFalse(caught.loc[1])  # touched, but the group went to cluster 2
        self.assertTrue(caught.loc[2])
        self.assertTrue(result.set_index("cluster").loc[1, "touched"])
        # A tie: one value of each cluster near the group, at the same distance: the lower cluster wins.
        clusters[:] = 0
        clusters[1, 0, 3] = 2
        clusters[1, 0, 7] = 1
        groups[:] = 0
        groups[1, 0, 5] = 1
        table = pd.DataFrame({"cluster": [1, 2], "kind": ["analysed", "analysed"], "t0": [1, 1], "t1": [1, 1],
                              "values": [1, 1]})
        result, _, _ = script.evaluate("d", clusters, table, groups, 1, offsets, 3.0, 8, 0.6, (0.5, 2.0), times)
        self.assertEqual(result.set_index("cluster")["caught"].to_dict(), {1: True, 2: False})

    def test_time_gaps_ties(self):
        times, height, width = 20, 1, 12
        offsets = row_offsets(np.full(height, 45.0), 0.5, 100.0)
        clusters = np.zeros((times, height, width), dtype=np.int32)
        clusters[8:10, 0, 2] = 2  # starts 3 timestamps after the end of group 1, next to it
        clusters[8:10, 0, 3] = 1  # the same start, also next to it: the lower cluster wins
        clusters[2:4, 0, 9] = 3  # ends 3 timestamps before group 2 starts; cluster 4 starts 3 after it ends
        clusters[13:15, 0, 10] = 4
        table = pd.DataFrame({"cluster": [1, 2, 3, 4], "kind": ["analysed"] * 4, "t0": [8, 8, 2, 13],
                              "t1": [9, 9, 3, 14], "values": [2, 2, 2, 2]})
        groups = np.zeros_like(clusters)
        groups[3:6, 0, 4] = 1  # ends at 5
        groups[6:11, 0, 9] = 2  # 6-10: after cluster 3 by 3 (+9 h), before cluster 4 by 3 (-9 h): tie to cluster 3
        _, lonely, _ = script.evaluate("d", clusters, table, groups, 2, offsets, 3.0, 8, 0.6, (0.5, 2.0), times)
        alone = lonely.set_index("group")
        self.assertEqual((alone.loc[1, "gap_h"], alone.loc[1, "gap_cluster"]), (-9.0, 1))
        self.assertEqual((alone.loc[2, "gap_h"], alone.loc[2, "gap_cluster"]), (9.0, 3))

    def test_keep_large(self):
        lattice = np.array([[[1, 1, 2, 3, 0]]], dtype=np.int32)
        kept, count = script.keep_large(lattice.copy(), np.array([3, 10, 1]), 5)
        self.assertEqual(count, 1)
        self.assertEqual(kept.tolist(), [[[0, 0, 1, 0, 0]]])

    def test_against_brute_force_on_real_latitudes(self):
        rng = np.random.default_rng(11)
        for latitudes in (60 - 0.5 * np.arange(12), 25.5 - 0.5 * np.arange(12)):
            height, width, times, radius = len(latitudes), 15, 3, 120.0
            longitudes = 0.5 * np.arange(width)
            offsets = row_offsets(latitudes, 0.5, radius)
            clusters = (rng.random((times, height, width)) < 0.08) * rng.integers(1, 6, (times, height, width))
            groups = (rng.random((times, height, width)) < 0.08) * rng.integers(1, 5, (times, height, width))
            clusters, groups = clusters.astype(np.int32), groups.astype(np.int32)
            got = {(int(k), int(g)): int(n) for k, g, n in script.near_counts(clusters, groups, offsets)}
            expected = {}
            for t in range(times):
                gcells = np.argwhere(groups[t] > 0)
                for r, c in np.argwhere(clusters[t] > 0):
                    d = haversine_km(latitudes[r], longitudes[c], latitudes[gcells[:, 0]], longitudes[gcells[:, 1]])
                    for g in set(groups[t][gcells[d < radius, 0], gcells[d < radius, 1]].tolist()):
                        key = (int(clusters[t, r, c]), int(g))
                        expected[key] = expected.get(key, 0) + 1
            self.assertEqual(got, expected)
            # disc: the cells closer than the radius to a set of cells.
            cells = np.array([3 * width + 4, 9 * width + 12])
            near = set(script.disc(cells, offsets, height, width).tolist())
            brute = {r * width + c for r in range(height) for c in range(width)
                     if min(haversine_km(latitudes[r], longitudes[c], latitudes[s // width], longitudes[s % width])
                            for s in cells) < radius}
            self.assertEqual(near, brute)

    def test_whole_script_and_the_deciding_vote(self):
        height, width, times = 6, 9, 24
        rows, cols = np.divmod(np.arange(height * width), width)
        grid = {"rows": rows.tolist(), "cols": cols.tolist(), "latitudes": np.linspace(50, 47.5, height).tolist(),
                "longitudes": np.linspace(5, 9, width).tolist()}
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        block = np.zeros((times, height, width))
        block[4:12, 1:4, 1:4] = 10  # seen by d0, d1 and d2: three votes, as many as needed
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            runs = []
            for k in range(4):
                scores = 0.01 * rng.random((times, height * width)) + (0 if k == 3 else block.reshape(times, -1))
                (folder / f"d{k}" / "scores").mkdir(parents=True)
                np.save(folder / f"d{k}" / "scores" / "anomaly_mean.npy", scores.astype(np.float32))
                np.save(folder / f"d{k}" / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
                (folder / f"d{k}" / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}),
                                                                encoding="utf-8")
                runs.append(f"d{k}={folder / f'd{k}'}")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                cube.main(["--runs", *runs, "--output", str(folder / "out"), "--top-percent", "15", "--votes", "3",
                           "--closure", "none", "--min-values", "5"])
                script.main(["--cube", str(folder / "out"), "--radius-km", "80", "--without-own", "--save-lonely",
                             "--csv", str(folder / "m.csv")])
            self.assertIn("values of the domain less than the radius from the cube", printed.getvalue())
            with_vote = pd.read_csv(folder / "m.csv")
            without = pd.read_csv(folder / "m_without_own.csv")
            self.assertTrue(with_vote[with_vote["detector"] == "d0"]["caught"].all())
            # d0 is the deciding vote of the block: against the cube without it the block is no cluster any more.
            self.assertEqual(len(without[without["detector"] == "d0"]), 0)
            # d3 does not flag the block: its vote decides nothing, the block stays a target against the cube without
            # it (whether a random group of d3 falls near it is chance: the base rate printed by the script).
            d3 = without[without["detector"] == "d3"]
            self.assertGreater(len(d3), 0)
            # --save-lonely: the values of the groups without counterpart, one file per detector, as many groups as
            # listed in the table of the groups without counterpart.
            lonely = pd.read_csv(folder / "m_lonely.csv")
            for k in range(4):
                info = json.loads((folder / "out" / "lonely" / f"d{k}.json").read_text(encoding="utf-8"))
                packed = np.load(folder / "out" / "lonely" / f"d{k}.npy")
                self.assertEqual(info["groups"], int((lonely["detector"] == f"d{k}").sum()))
                self.assertEqual(info["values"], int(np.unpackbits(packed, axis=1, count=height * width).sum()))
                self.assertEqual(info["values"] > 0, info["groups"] > 0)
            # The saved values are exactly those of the groups without counterpart, all far from every cluster:
            # rebuilt here from the flags of d0, its groups and the zone of the cube.
            from scripts.check_era5_concordant_boxes import label_clusters
            from scripts.era5_km_zone import owner_lattice
            out = folder / "out"
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            votes = np.load(out / "votes" / "scores" / "anomaly_mean.npy")
            concordant = np.nan_to_num(votes) >= summary["vote"]["votes_needed"]
            labels, _ = label_clusters(concordant, rows, cols, (height, width), summary["overlap"])
            owner, _, _ = owner_lattice(labels, grid, 80.0)
            for k in range(4):
                flags = np.unpackbits(np.load(out / "flags" / f"d{k}.npy"), axis=1, count=height * width).astype(bool)
                groups, sizes = label_clusters(flags, rows, cols, (height, width), summary["overlap"])
                groups, _ = script.keep_large(groups, sizes, summary["selection"]["min_values"])
                ids = lonely.loc[lonely["detector"] == f"d{k}", "group"].to_numpy()
                expected = np.isin(groups, ids)[:, rows, cols]
                saved = np.unpackbits(np.load(out / "lonely" / f"d{k}.npy"), axis=1, count=height * width).astype(bool)
                self.assertTrue(np.array_equal(saved, expected))
                self.assertFalse((saved & (owner[:, rows, cols] > 0)).any())


if __name__ == "__main__":
    unittest.main()
