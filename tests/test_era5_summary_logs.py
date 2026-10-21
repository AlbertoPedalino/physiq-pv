"""The summary readers must give back the numbers the checks printed."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import era5_summary_logs as logs

EOBS = """######## eobs val, soglia 5%, pressione_bassa
[eobs] 2000-01-01 to 2004-12-31 | flagged: the highest 5% of each score, at one timestamp of the UTC day at least | first: gen_clim | others: no_model_bal_clim, previous_bal_clim | unusual pressure: lowest 5% for the cell and the month | cells with stations: 2961 of 10611

Ratios of the shares, with the 95% interval of 2000 draws of whole years (5 years)
                                                                same day                    next day                      either
score alone / all cell-days                                         1.51 (1.37-1.67)            1.50 (1.35-1.68)    1.47 (1.34-1.63)
score alone / ring around score alone                               1.20 (1.10-1.30)            1.10 (1.00-1.20)    1.15 (1.05-1.25)
score and another / all cell-days                                   3.15 (3.05-3.23)            2.22 (2.14-2.33)    2.58 (2.50-2.66)
[eobs] 2000-01-01 to 2004-12-31 | flagged: the highest 5% of each score, at one timestamp of the UTC day at least | first: gen_clim | others: x | unusual change of the pressure over 1 day(s): lowest 5% for the cell and the month | cells with stations: 2961 of 10611
score alone / all cell-days                                         1.27 (1.17-1.37)            0.93 (0.89-0.99)    1.10 (1.05-1.17)
"""

EOBS_SHORT = """######## score: outputs/good2v/val_gen_clim
=== rain high change_days=0
% of the cell-days with unusual rain measured by the stations
score alone / all cell-days                                         1.26 (1.15-1.36)            1.49 (1.40-1.59)      1.39 (1.29-1.49)
=== pressure low change_days=1
score alone / all cell-days                                         1.48 (1.39-1.58)            0.96 (0.92-1.00)         1.23 (1.18-1.27)
######## score: outputs/stgan_good_sweep_2_validation
=== temperature low change_days=0
score alone / all cell-days                                         0.56 (0.50-0.60)            0.60 (0.55-0.65)         0.58 (0.52-0.63)
"""

PRECURSOR = """######## test, soglia 5%, stagionalita, giudice cape 5%
radius, lead, case                   calm cells    % of calmevent after %         lift
0 cell(s), 3h, both                    15229083         2.45         8.28          4.3
0 cell(s), 3h, first only              15042421         2.42         6.89          3.6
0 cell(s), 3h, second early only       14311689         2.31         4.38          2.3
0 cell(s), 3h, neither                576214422        92.82         1.93          1.0
0 cell(s), 3h, both                           0         0.00          nan          nan
"""

DIRECTION = """cosine between the direction and where the event goes, on the events where each has a direction: % ...
direction, where the event goes, hours           events  same side  within 45   mean cos
gen_clim only, shift, 3h                         166287       52.8       29.0      0.048
cosine between the direction and where the event goes, on the same events, where the edge, past motion and wind have a direction: % ...
direction, where the event goes, hours           events  same side  within 45   mean cos
gen_clim only, shift, 3h                         146776       52.9       29.2      0.051
wind at 850 hPa, shift, 3h                       146776       55.8       30.4     -0.091
on the same events: is the edge of gen_clim on the right side, and the reference? (% of the events)
reference, where the event goes, hours           both right      edge only reference only     both wrong  edge right when ref wrong
wind at 850 hPa, shift, 3h                             32.0           21.0           23.8           23.2       47.4

on the same events: when the edge of gen_clim and the reference point the same side (an angle under 90 degrees between them), how often are they right, and when they do not, which one is? (% of the events)
reference, where the event goes, hours                   agree    right if agree  ref right if not edge right if notcombined right         ref right
wind at 850 hPa, shift, 3h                                56.9              57.2              54.0              49.1          55.4              55.8
"""

TYPES = """[edge-types] 2005-01-01 to 2025-12-31 | flagged: ...

% of the cells by type of their event (types), and mean share of the event cells with each signal (s:)
                                   cells      wind      heat      cold      rain convective     mixed     other    s:wind    s:heat
event cells of the others       12345678       3.8      10.5       6.9       3.0       2.3      43.5      30.0       7.9      16.2
edge, gen_clim alone             1234567       4.5      11.5       3.4       3.0       2.9      56.3      18.5      11.1      21.8
edge, no score                   2345678       3.4      12.0       8.5       2.6       3.1      24.3      46.1       4.8      14.1

season DJF
                                   cells      wind      heat      cold      rain convective     mixed     other    s:wind    s:heat
edge, no score                         1       1.0       1.0       1.0       1.0       1.0       1.0       1.0       1.0       1.0
"""

CONFIDENCE = """classes of confidence of the flags of the model, by the spread of G at its error (1: ...)
flags, class                    points      obs >1 %   fcst only %
model only, 1                    14384          72.2          13.9
inside (model and other), 5         35703          71.8          21.5
all flags of the model, 3         45187          73.5          19.1
"""


class SummaryLogTests(unittest.TestCase):
    def test_blocks(self):
        parts = logs.blocks("head\n######## a, b\none\n######## c\ntwo\nthree")
        self.assertEqual(parts, {"": "head", "a, b": "one", "c": "two\nthree"})

    def test_eobs(self):
        runs = logs.eobs(EOBS)
        self.assertEqual([run["indicator"] for run in runs], ["pressure low", "pressure drop"])
        self.assertEqual(runs[0]["first"], "gen_clim")
        self.assertEqual(runs[0]["ratios"]["alone"]["either"], (1.47, 1.34, 1.63))
        self.assertEqual(runs[0]["ratios"]["shared"]["same day"], (3.15, 3.05, 3.23))
        self.assertEqual(runs[1]["ratios"]["alone"]["next day"], (0.93, 0.89, 0.99))
        self.assertNotIn("shared", runs[1]["ratios"])
        # The logs that kept only the tables: the run from "######## score:", the indicator from "===".
        runs = logs.eobs(EOBS_SHORT)
        self.assertEqual([(run["first"], run["indicator"]) for run in runs],
                         [("outputs/good2v/val_gen_clim", "rain"), ("outputs/good2v/val_gen_clim", "pressure drop"),
                          ("outputs/stgan_good_sweep_2_validation", "cold")])
        self.assertEqual(runs[0]["ratios"]["alone"]["either"], (1.39, 1.29, 1.49))
        # A title line that names the indicator but not the score.
        runs = logs.eobs("######## E-OBS test, pressure low change_days=1\n"
                         "score alone / all cell-days        1.42 (1.37-1.47)     0.93 (0.90-0.96)     1.17 (1.15-1.20)\n")
        self.assertEqual((runs[0]["first"], runs[0]["indicator"]), (None, "pressure drop"))
        self.assertEqual(runs[0]["ratios"]["alone"]["either"], (1.17, 1.15, 1.20))

    def test_precursor(self):
        rows = logs.precursor(logs.blocks(PRECURSOR)["test, soglia 5%, stagionalita, giudice cape 5%"])
        self.assertEqual(rows[(0, 3, "first only")], (15042421, 2.42, 6.89, 3.6))
        self.assertEqual(rows[(0, 3, "neither")][2], 1.93)
        self.assertNotEqual(rows[(0, 3, "both")][2], rows[(0, 3, "both")][2])  # the last row: nan

    def test_direction(self):
        tables = logs.direction(DIRECTION)
        self.assertEqual(tables["own events"][("gen_clim only", "shift", 3)], (166287, 52.8, 29.0, 0.048))
        self.assertEqual(tables["same events"][("gen_clim only", "shift", 3)][1], 52.9)
        self.assertEqual(tables["same events"][("wind at 850 hPa", "shift", 3)][3], -0.091)
        self.assertEqual(tables["cases"][("wind at 850 hPa", "shift", 3)], (32.0, 21.0, 23.8, 23.2, 47.4))
        self.assertEqual(tables["agreement"][("wind at 850 hPa", "shift", 3)][:2], (56.9, 57.2))

    def test_event_types(self):
        rows = logs.event_types(TYPES)
        self.assertEqual(set(rows), {"event cells of the others", "edge, gen_clim alone", "edge, no score"})
        self.assertEqual(rows["edge, gen_clim alone"]["mixed"], 56.3)
        self.assertEqual(rows["edge, no score"]["s:wind"], 4.8)
        self.assertEqual(rows["edge, no score"]["cells"], 2345678)

    def test_confidence(self):
        rows = logs.confidence(CONFIDENCE)
        self.assertEqual(rows[("model only", 1)], (14384, 72.2, 13.9))
        self.assertEqual(rows[("inside (model and other)", 5)][2], 21.5)
        self.assertEqual(rows[("all flags of the model", 3)][0], 45187)


if __name__ == "__main__":
    unittest.main()
