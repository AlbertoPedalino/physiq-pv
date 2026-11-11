"""When the first score stops flagging a cell while the second still flags it, does the second stop soon after, and
does the weather go back to normal?

No reference outside the data. Each score flags its highest `--top-percent` (threshold of the validation years when
the run names it); with several `--by-dir` the second score flags a value when any of them flags it. At every
timestamp t, the cells where both scores flagged at t-1 and the second still flags at t are split in two groups:
- first stopped: the first does not flag the cell at t;
- first goes on: the first still flags it at t (the reference: the same history, the first did not stop).
For each group, over the next timestamps (3 and 6 hours with data every 3 hours):
- the second has stopped: it no longer flags the cell at t+1 (within 3 h), at t+1 or t+2 (within 6 h);
- the first flags the cell again at t+1 (it stopped only for a step);
- with `--inputs-run` (standardised inputs): the share of cells with no input variable beyond `--far-sd` standard
  deviations from its seasonal normal, and the mean number of such variables, at t, t+1 and t+2;
- with `--unseen-dir` (CAPE and gust, which no score reads): the share of cells where each is in its highest
  `--unusual-percent` for the cell, the calendar month and the time of day, at t, t+1 and t+2.
The cells where the first stopped are also split by what the second does next: it stops within 6 h, or it still
flags the cell at t+1 and t+2. In the second part the weather tells who is right: if it goes back to normal there
too, the second is only late; if it stays as unusual as where the first goes on, the first stopped too early.
Where the first stops the anomaly may already be weaker, and a weak anomaly ends sooner anyway. With `--inputs-run`
the two groups are therefore also compared at equal strength: cut by the number of input variables beyond `--far-sd`
standard deviations at t (STRATA), and with the cells where the first goes on weighted to the strengths of the cells
where it stopped. If the differences hold at equal strength, they do not come from a weaker anomaly.
If the second stops sooner where the first stopped, and the weather there is closer to normal in the next hours than
where the first goes on, the first saw the end coming; if the second stops sooner but the weather stays as unusual,
the first only stopped early. The share of the second stopping within 3 h is also given year by year, with a sign
test, to tell a difference that holds every year from one made by a few.

With `--ratio-run` (the scored run of the model with the Monte Carlo dropout: its generator mean and spread) the
cells where the first stopped are also cut in thirds of the relative spread of the draws at the timestamp it stopped
(spread over mean of the error; cuts from those cells): does the first stop more rightly when its draws agree?

    python scripts/check_era5_end_lead.py --score-dir outputs/optuna27/test_generator_clim_day \\
        --by-dir outputs/confronti_good2/climatology/test/{normal,change_3h,pca,cblof,mcd,hbos,copod,iforest} \\
        --inputs-run outputs/stgan_optuna27_scored --unseen-dir data/era5_independent/independent --csv outputs/fine.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path.cwd()))  # after the copy next to the script: a copy run from the checkout imports itself
import numpy as np
import pandas as pd

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from scripts.check_era5_lead import common_timestamps, flags_at, union_flags
from scripts.check_era5_score_against_unseen import read_month, unusual_values

GROUPS = ("first stopped", "first goes on", "first stopped, second stops by 6 h",
          "first stopped, second still on at 6 h")
STRATA = ((0, 1), (2, 3), (4, 5), (6, None))  # variables beyond --far-sd at t
RATIO_BINS, RATIO_LOW, RATIO_HIGH = 400, -3.0, 1.0  # log10 of the relative spread, counted in these bins
AHEAD = 2  # steps read after t (t+1, t+2)
CHUNK = 64


def sign_test(positive, negative):
    """Two-sided sign test p-value."""
    n = positive + negative
    if n == 0:
        return float("nan")
    k = min(positive, negative)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def groups_at(first, second, t):
    """[group, cell] boolean at timestamp index t: both flagged at t-1, the second still at t, the first not / yes."""
    base = first[t - 1] & second[t - 1] & second[t]
    return np.stack((base & ~first[t], base & first[t]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score")
    parser.add_argument("--by-dir", type=Path, nargs="+", required=True, help="Run(s) of the second score")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--inputs-run", type=Path, help="Scored run with the standardised inputs")
    parser.add_argument("--far-sd", type=float, default=2.0)
    parser.add_argument("--unseen-dir", type=Path, help="Monthly NetCDF files of the unseen variables")
    parser.add_argument("--unusual-percent", type=float, default=5)
    parser.add_argument("--ratio-run", type=Path,
                        help="Scored run with the dropout draws: thirds of the relative spread where the first stopped")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)

    timestamps = common_timestamps([args.score_dir, *args.by_dir])
    first = flags_at(args.score_dir, args.top_percent, timestamps)
    second = union_flags(args.by_dir, args.top_percent, timestamps)
    step = timestamps[1] - timestamps[0]
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    if len(rows) != first.shape[1]:
        raise ValueError("The grid of the run does not match the cells of the scores")
    latitudes = np.asarray(grid["latitudes"], dtype=float)[rows]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[cols]

    inputs = None
    if args.inputs_run is not None:
        from scripts.check_era5_event_zones import inputs_of
        run_times = pd.DatetimeIndex(np.load(args.inputs_run / "test_timestamps.npy"))
        inputs, offset, input_names = inputs_of(args.inputs_run, run_times)
        input_rows = run_times.get_indexer(timestamps)
        if (input_rows < 0).any():
            raise ValueError("The inputs run does not cover the timestamps of the scores")
        input_rows = input_rows + offset

    ratio_maps = None
    if args.ratio_run is not None:
        from scripts.check_era5_dust_lead import score_file
        manifest = json.loads((args.ratio_run / "metadata.json").read_text(encoding="utf-8"))
        ratio_maps = (np.load(args.ratio_run / manifest["generator_std_file"], mmap_mode="r"),
                      np.load(args.ratio_run / score_file(manifest, "generator"), mmap_mode="r"))
        ratio_rows = pd.DatetimeIndex(np.load(args.ratio_run / "test_timestamps.npy")).get_indexer(timestamps)
        if (ratio_rows < 0).any():
            raise ValueError("The dropout run does not cover the timestamps of the scores")

    # t needs t-1 and t+1, t+2 at the step of the data, all in the same calendar month (the unseen variables are read
    # a month at a time).
    index = np.arange(len(timestamps))
    usable = (index >= 1) & (index < len(timestamps) - AHEAD)
    for back, ahead in ((1, 0), (0, 1), (0, 2)):
        shifted_ok = np.zeros(len(timestamps), dtype=bool)
        lo, hi = index - back, index + ahead
        valid = (lo >= 0) & (hi < len(timestamps))
        shifted_ok[valid] = (timestamps[hi[valid]] - timestamps[lo[valid]]) == (back + ahead) * step
        usable &= shifted_ok | ((back, ahead) == (0, 0))
    ahead_index = np.minimum(index + AHEAD, len(timestamps) - 1)
    usable &= (timestamps.month[ahead_index] == timestamps.month) & (timestamps.year[ahead_index] == timestamps.year)

    years = np.unique(timestamps.year)
    counts = np.zeros(len(GROUPS))
    stopped_3, stopped_6, back_on = (np.zeros(len(GROUPS)) for _ in range(3))
    year_counts, year_stopped = np.zeros((len(years), len(GROUPS))), np.zeros((len(years), len(GROUPS)))
    none_far = np.zeros((len(GROUPS), AHEAD + 1))
    far_sum = np.zeros((len(GROUPS), AHEAD + 1))
    unseen_hits = unseen_defined = unseen_names = None
    # first stopped (0) and first goes on (1) by stratum of strength at t: counts and outcomes
    edges = [low for low, _ in STRATA[1:]]
    st_n, st_stop3, st_stop6 = (np.zeros((2, len(STRATA))) for _ in range(3))
    st_none_far = np.zeros((2, AHEAD + 1, len(STRATA)))
    st_unseen_hits = st_unseen_defined = None
    # the cells where the first stopped, by bin of the relative spread: counts and outcomes
    bin_n, bin_stop3, bin_stop6, bin_back = (np.zeros(RATIO_BINS) for _ in range(4))
    bin_none_far = np.zeros((AHEAD + 1, RATIO_BINS))
    bin_unseen_hits = bin_unseen_defined = None

    try:
        for month in range(1, 13):
            positions = np.nonzero(timestamps.month == month)[0]
            if not len(positions):
                continue
            unusual = None
            if args.unseen_dir is not None:
                month_years = sorted(set(timestamps.year[positions]))
                parts = [read_month(args.unseen_dir, year, month, latitudes, longitudes) for year in month_years]
                names = parts[0][2]
                if unseen_names is None:
                    unseen_names = names
                    unseen_hits = np.zeros((len(GROUPS), AHEAD + 1, len(names)))
                    unseen_defined = np.zeros((len(GROUPS), AHEAD + 1, len(names)))
                    bin_unseen_hits = np.zeros((AHEAD + 1, len(names), RATIO_BINS))
                    st_unseen_hits = np.zeros((2, AHEAD + 1, len(names), len(STRATA)))
                    st_unseen_defined = np.zeros((2, AHEAD + 1, len(names), len(STRATA)))
                    bin_unseen_defined = np.zeros((AHEAD + 1, len(names), RATIO_BINS))
                elif names != unseen_names:
                    raise ValueError(f"The files of month {month} do not hold the variables of the other months")
                times = parts[0][0].append([part[0] for part in parts[1:]])
                found = timestamps.get_indexer(times)
                kept = found >= 0
                values = np.concatenate([part[1] for part in parts])[kept]
                del parts
                order = np.argsort(found[kept])
                if not np.array_equal(found[kept][order], positions):
                    raise ValueError(f"The unseen variables of month {month} do not cover the timestamps of the scores")
                unusual, defined = unusual_values(values[order], timestamps.hour[positions].to_numpy(),
                                                  args.unusual_percent)
                del values
                row_of = np.full(len(timestamps), -1, dtype=np.int64)
                row_of[positions] = np.arange(len(positions))
            chosen_all = positions[usable[positions]]
            for start in range(0, len(chosen_all), CHUNK):
                chosen = chosen_all[start:start + CHUNK]
                block = None
                if inputs is not None:
                    wanted = np.concatenate([input_rows[chosen + k] for k in range(AHEAD + 1)])
                    block = np.asarray(inputs[wanted], dtype=np.float32)[:, :, :len(input_names)]
                    with np.errstate(invalid="ignore"):
                        far = (np.abs(block) > args.far_sd).sum(axis=-1).reshape(AHEAD + 1, len(chosen), -1)
                    del block
                    release_pages(inputs)
                spread_bins = None
                if ratio_maps is not None:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ratio = np.log10(np.asarray(ratio_maps[0][ratio_rows[chosen]], dtype=np.float64)
                                         / np.asarray(ratio_maps[1][ratio_rows[chosen]], dtype=np.float64))
                    spread_bins = np.full(ratio.shape, -1, dtype=np.int64)
                    finite = np.isfinite(ratio)
                    spread_bins[finite] = np.clip(((ratio[finite] - RATIO_LOW) / (RATIO_HIGH - RATIO_LOW)
                                                   * RATIO_BINS).astype(np.int64), 0, RATIO_BINS - 1)
                    release_pages(*ratio_maps)
                for i, t in enumerate(chosen):
                    cells = groups_at(first, second, t)
                    later_on = second[t + 1] & second[t + 2]
                    cells = np.concatenate([cells, (cells[0] & ~later_on)[None], (cells[0] & later_on)[None]])
                    if spread_bins is not None:
                        alone = cells[0] & (spread_bins[i] >= 0)
                        b = spread_bins[i][alone]
                        bin_n += np.bincount(b, minlength=RATIO_BINS)
                        bin_stop3 += np.bincount(b, weights=~second[t + 1][alone], minlength=RATIO_BINS)
                        bin_stop6 += np.bincount(b, weights=(~second[t + 1] | ~second[t + 2])[alone],
                                                 minlength=RATIO_BINS)
                        bin_back += np.bincount(b, weights=first[t + 1][alone], minlength=RATIO_BINS)
                        if inputs is not None:
                            for k in range(AHEAD + 1):
                                bin_none_far[k] += np.bincount(b, weights=far[k, i][alone] == 0, minlength=RATIO_BINS)
                        if unusual is not None:
                            for k in range(AHEAD + 1):
                                r = row_of[t + k]
                                for v in range(len(unseen_names)):
                                    bin_unseen_hits[k, v] += np.bincount(b, weights=unusual[r][alone, v],
                                                                         minlength=RATIO_BINS)
                                    bin_unseen_defined[k, v] += np.bincount(b, weights=defined[r][alone, v],
                                                                            minlength=RATIO_BINS)
                    year = np.searchsorted(years, timestamps.year[t])
                    if inputs is not None:
                        stratum = np.digitize(far[0, i], edges)
                        for g in range(2):
                            mask = cells[g]
                            if not mask.any():
                                continue
                            st = stratum[mask]
                            st_n[g] += np.bincount(st, minlength=len(STRATA))
                            st_stop3[g] += np.bincount(st, weights=~second[t + 1][mask], minlength=len(STRATA))
                            st_stop6[g] += np.bincount(st, weights=(~second[t + 1] | ~second[t + 2])[mask],
                                                       minlength=len(STRATA))
                            for k in range(AHEAD + 1):
                                st_none_far[g, k] += np.bincount(st, weights=far[k, i][mask] == 0,
                                                                 minlength=len(STRATA))
                            if unusual is not None:
                                for k in range(AHEAD + 1):
                                    r = row_of[t + k]
                                    for v in range(len(unseen_names)):
                                        st_unseen_hits[g, k, v] += np.bincount(st, weights=unusual[r][mask, v],
                                                                               minlength=len(STRATA))
                                        st_unseen_defined[g, k, v] += np.bincount(st, weights=defined[r][mask, v],
                                                                                  minlength=len(STRATA))
                    for g in range(len(GROUPS)):
                        mask = cells[g]
                        n = int(mask.sum())
                        if not n:
                            continue
                        counts[g] += n
                        off_1 = mask & ~second[t + 1]
                        off_2 = mask & (~second[t + 1] | ~second[t + 2])
                        stopped_3[g] += off_1.sum()
                        stopped_6[g] += off_2.sum()
                        back_on[g] += (mask & first[t + 1]).sum()
                        year_counts[year, g] += n
                        year_stopped[year, g] += off_1.sum()
                        if inputs is not None:
                            for k in range(AHEAD + 1):
                                none_far[g, k] += (far[k, i][mask] == 0).sum()
                                far_sum[g, k] += far[k, i][mask].sum()
                        if unusual is not None:
                            for k in range(AHEAD + 1):
                                r = row_of[t + k]
                                unseen_hits[g, k] += unusual[r][mask].sum(axis=0)
                                unseen_defined[g, k] += defined[r][mask].sum(axis=0)
            del unusual
            print(f"[end-lead] month {month} read", flush=True)
    finally:
        if inputs is not None:
            inputs._mmap.close()

    hours = int(step / pd.Timedelta(hours=1))
    print(f"[end-lead] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest {args.top_percent:g}% "
          f"of {args.score_dir.name} (first) and of {' or '.join(path.name for path in args.by_dir)} (second) | cells "
          f"where both flagged {hours} h before and the second still flags: the first stopped, or goes on")
    rows_out = []
    for g, name in enumerate(GROUPS):
        row = {"group": name, "cells": int(counts[g]),
               f"second stopped within {hours} h %": 100 * stopped_3[g] / max(counts[g], 1),
               f"second stopped within {2 * hours} h %": 100 * stopped_6[g] / max(counts[g], 1),
               f"first flags again at +{hours} h %": 100 * back_on[g] / max(counts[g], 1)}
        if inputs is not None:
            for k in range(AHEAD + 1):
                when = "now" if k == 0 else f"+{k * hours}h"
                row[f"no variable beyond {args.far_sd:g} sd {when} %"] = 100 * none_far[g, k] / max(counts[g], 1)
                row[f"variables beyond {args.far_sd:g} sd {when}"] = far_sum[g, k] / max(counts[g], 1)
        if unseen_names is not None:
            for v, variable in enumerate(unseen_names):
                for k in range(AHEAD + 1):
                    when = "now" if k == 0 else f"+{k * hours}h"
                    row[f"{variable} highest {args.unusual_percent:g}% {when} %"] = (
                        100 * unseen_hits[g, k, v] / max(unseen_defined[g, k, v], 1))
        rows_out.append(row)
    table = pd.DataFrame(rows_out).set_index("group")
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.2f}".format):
        print(table.T)

    share = 100 * year_stopped / np.maximum(year_counts, 1)
    positive = int((share[:, 0] > share[:, 1]).sum())
    negative = int((share[:, 0] < share[:, 1]).sum())
    print(f"\nsecond stopped within {hours} h, year by year (% of the cells of the group)")
    print(pd.DataFrame({"year": years, "first stopped": share[:, 0], "first goes on": share[:, 1]}).to_string(
        index=False, float_format="{:.1f}".format))
    print(f"  first stopped above first goes on in {positive} of {positive + negative} years "
          f"(sign test p {sign_test(positive, negative):.3g})")
    if inputs is not None and st_n.sum():
        labels = [f"{low}-{high}" if high is not None else f">={low}" for low, high in STRATA]
        print(f"\nat equal strength: cells by number of variables beyond {args.far_sd:g} sd at t; 'goes on, weighted' "
              f"gives the cells where the first goes on the strengths of those where it stopped")
        weights = st_n[0] / st_n[0].sum()

        def rate(numerator, denominator):
            return 100 * numerator / np.maximum(denominator, 1)

        rows_st = []
        for j, label in enumerate(labels):
            for g, name in ((0, "first stopped"), (1, "first goes on")):
                row = {"variables beyond sd at t": label, "group": name, "cells": int(st_n[g, j]),
                       f"second stopped within {hours} h %": rate(st_stop3[g, j], st_n[g, j]),
                       f"second stopped within {2 * hours} h %": rate(st_stop6[g, j], st_n[g, j]),
                       f"no variable beyond {args.far_sd:g} sd +{2 * hours}h %": rate(st_none_far[g, 2, j], st_n[g, j])}
                if st_unseen_hits is not None:
                    for v, variable in enumerate(unseen_names):
                        for k in (0, 2):
                            when = "now" if k == 0 else f"+{k * hours}h"
                            row[f"{variable} highest {args.unusual_percent:g}% {when} %"] = rate(
                                st_unseen_hits[g, k, v, j], st_unseen_defined[g, k, v, j])
                rows_st.append(row)
        for g, name in ((0, "first stopped"), (1, "first goes on, weighted")):
            share = lambda array: float(np.nansum(weights * array))
            row = {"variables beyond sd at t": "all", "group": name, "cells": int(st_n[g].sum()),
                   f"second stopped within {hours} h %": share(rate(st_stop3[g], st_n[g])),
                   f"second stopped within {2 * hours} h %": share(rate(st_stop6[g], st_n[g])),
                   f"no variable beyond {args.far_sd:g} sd +{2 * hours}h %": share(rate(st_none_far[g, 2], st_n[g]))}
            if st_unseen_hits is not None:
                for v, variable in enumerate(unseen_names):
                    for k in (0, 2):
                        when = "now" if k == 0 else f"+{k * hours}h"
                        row[f"{variable} highest {args.unusual_percent:g}% {when} %"] = share(
                            rate(st_unseen_hits[g, k, v], st_unseen_defined[g, k, v]))
            rows_st.append(row)
        strata = pd.DataFrame(rows_st)
        with pd.option_context("display.width", 250, "display.max_columns", 60, "display.float_format",
                               "{:.2f}".format):
            print(strata.to_string(index=False))
        if args.csv:
            strata.to_csv(args.csv.with_name(args.csv.stem + "_forza.csv"), index=False)
    if ratio_maps is not None and bin_n.sum():
        cumulative = np.cumsum(bin_n) / bin_n.sum()
        cuts = [int(np.searchsorted(cumulative, 1 / 3)) + 1, int(np.searchsorted(cumulative, 2 / 3)) + 1]
        edges = [0, *cuts, RATIO_BINS]
        width = (RATIO_HIGH - RATIO_LOW) / RATIO_BINS
        limits = [10 ** (RATIO_LOW + cut * width) for cut in cuts]
        names_of = (f"low (<{limits[0]:.3f})", f"middle ({limits[0]:.3f}-{limits[1]:.3f})", f"high (>{limits[1]:.3f})")
        third_rows = []
        for name, lo, hi in zip(names_of, edges[:-1], edges[1:]):
            n = bin_n[lo:hi].sum()
            row = {"third": name, "cells": int(n),
                   f"second stopped within {hours} h %": 100 * bin_stop3[lo:hi].sum() / max(n, 1),
                   f"second stopped within {2 * hours} h %": 100 * bin_stop6[lo:hi].sum() / max(n, 1),
                   f"first flags again at +{hours} h %": 100 * bin_back[lo:hi].sum() / max(n, 1)}
            if inputs is not None:
                for k in range(AHEAD + 1):
                    when = "now" if k == 0 else f"+{k * hours}h"
                    row[f"no variable beyond {args.far_sd:g} sd {when} %"] = 100 * bin_none_far[k, lo:hi].sum() / max(n, 1)
            if bin_unseen_hits is not None:
                for v, variable in enumerate(unseen_names):
                    for k in range(AHEAD + 1):
                        when = "now" if k == 0 else f"+{k * hours}h"
                        row[f"{variable} highest {args.unusual_percent:g}% {when} %"] = (
                            100 * bin_unseen_hits[k, v, lo:hi].sum() / max(bin_unseen_defined[k, v, lo:hi].sum(), 1))
            third_rows.append(row)
        thirds = pd.DataFrame(third_rows).set_index("third")
        print(f"\nfirst stopped, by third of the relative spread of its dropout draws at the timestamp it stopped "
              f"(spread / mean of the error; cuts from these cells); compare with 'first goes on' above")
        with pd.option_context("display.width", 250, "display.max_columns", 60, "display.float_format",
                               "{:.2f}".format):
            print(thirds.T)
        if args.csv:
            thirds.to_csv(args.csv.with_name(args.csv.stem + "_terzi.csv"))
    if args.csv:
        table.to_csv(args.csv)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
