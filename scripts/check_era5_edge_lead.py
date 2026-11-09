"""Where only one score flags a cell next to what the other flags, does the other reach that cell later?

A value is flagged by a score when it is in its highest `--top-percent`. At every timestamp the cells
the second score flags make its area; every other cell has a distance from it, in cells of the grid,
counted as the larger of the row and column offsets (BANDS). Two groups of cells outside that area:
- first only: the first score flags the cell, the second does not;
- neither: no score flags it, the reference.
For every group, distance band and window (WINDOWS_HOURS) a value is put in one of four kinds by what
the second score does at the same cell within the window:
- front: it flags the cell after, not before: the first score saw the weather coming;
- trail: it flagged the cell before, not after: the first score holds on to weather that is leaving;
- both: before and after, the cell is inside a longer episode;
- none: neither before nor after.
At equal distance from the area of the second score, a share of fronts larger for "first only" than
for "neither" is a lead of the first score, in time (the hours to the arrival) and in space (the
distance of the cell). A trail share larger by as much says it extends the weather both ways.

With `--unseen-dir`, for the same groups and kinds (window `--unseen-hours`, default UNSEEN_WINDOW_HOURS), the
share of the values
where a variable no score saw is in its highest `--unusual-percent` for the cell, the calendar month
and the time of day: whether the fronts are weather already.

With several `--by-dir` the second score is any of them: a value is flagged when at least one of
them flags it, each at its own highest `--by-top-percent`, and "first only" is what none of them
flags.

The whole scored period is read. With `--csv` the shares of the kinds go to a file.

With `--events` (a CSV of known events: `id`, `tipo`, `inizio`, `fine`, `lat_min`, `lat_max`, `lon_min`, `lon_max`),
the same kinds are also counted only on the values of the events: the cells of the box of every event and of a ring
of `--ring-km` around it, from `--before-hours` before its onset (the measured onset of `--onsets`, the inizi.csv of
check_era5_known_events_lead.py, main variant; otherwise the first day of the source) to 21 UTC of its last day.
The cells of one event are not independent, so the uncertainty comes from `--bootstrap` resamples of the events:
the 95% interval of "front x" and of the excess of the first score only, (front - trail) of the first only minus
(front - trail) of neither at the same distance, which is positive when the first score sees the cells the second
reaches next more often than chance and than the cells it leaves. Written to `--events-csv`. The excess is also
given event by event (`<events-csv>_per_evento.csv`), with the share of the events where it is positive and a sign
test, to say whether the lead is consistent across the events or comes from a few of them.

The shares of the unseen variables at distance 1 are also given year by year (first only and neither, every kind
together and the kind none), to tell a share that holds every year from one made by a few, and, with `--ratio-run`,
by third of the relative spread of the first only (cuts from the whole period, as below) and kind
(`<csv>_unseen_anni.csv`, `<csv>_unseen_terzi.csv`).

With `--ratio-run` (the scored run of the model with the Monte Carlo dropout, its generator mean and spread), every
value also has the relative spread of the draws, the spread of the reconstruction error over its mean: how much the
draws disagree for the size of the error. Its median is given by group, distance and kind within 3 h, over the whole
period and on the events, kinds within 3 h and within 24 h, for all the events and type by type; on the events the
difference front - trail of the first only has an interval from resamples of the events (of that type). The values
of the first only at distance 1 are also cut in thirds of the relative spread (cuts from the whole period), and the
excess is given for every third: if the third where the draws agree most has the larger excess, the spread tells the
values that come before the second score from those that stay after it (`<events-csv>_terzi.csv`). A low relative
spread may only mean a strong flag (a large error), so the same values are also cut in thirds of the score of the
first (its strength), and inside every third of strength in thirds of the relative spread (cuts from the whole
period within that third): if the spread still separates the excess at equal strength, it is the spread that tells
them apart (`<events-csv>_forza.csv`).

    python scripts/check_era5_edge_lead.py --score-dir outputs/run_gen_clim --by-dir outputs/run_simple_max \
        --unseen-dir data/era5_independent/independent --csv outputs/edge_lead_5.csv
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path.cwd()))  # after the copy next to the script: a copy run from the checkout imports itself
import numpy as np
import pandas as pd
from scipy import ndimage

from scripts.check_era5_lead import flags
from scripts.check_era5_score_against_unseen import read_month, unusual_values

CHUNK = 256  # timestamps read at a time
RATIO_BINS, RATIO_LOW, RATIO_HIGH = 400, -3.0, 1.0  # log10 of the relative spread, counted in these bins
MIN_EVENT_VALUES = 30  # an event enters the per-event share with at least this many values of the first only
RATIO_WINDOWS = (3, 24)  # hours of the windows of the kinds read with the relative spread
STRENGTH_BINS, STRENGTH_LOW, STRENGTH_HIGH = 60, 0.0, 3.0  # log10 of the score of the first, counted in these bins
BANDS = ((1, 1), (2, 2), (3, 5), (6, 10), (11, None))  # cells of 0.5 degrees from the area of the second score
WINDOWS_HOURS = (3, 6, 12, 24)
UNSEEN_WINDOW_HOURS = 6
GROUPS = ("first only", "neither")
KINDS = ("none", "front", "trail", "both")
NAME, WIDTH = 34, 10


def since_and_until(flagged, cap):
    """Steps since the last flag strictly before every timestamp, and to the next one strictly after, per cell,
    capped at `cap`: two [time, cell] arrays of int8."""
    times, cells = flagged.shape
    since, until = np.empty((times, cells), dtype=np.int8), np.empty((times, cells), dtype=np.int8)
    count = np.full(cells, cap, dtype=np.int8)
    for step in range(times):
        since[step] = count
        count = np.where(flagged[step], 1, np.minimum(count + 1, cap)).astype(np.int8)
    count[:] = cap
    for step in range(times - 1, -1, -1):
        until[step] = count
        count = np.where(flagged[step], 1, np.minimum(count + 1, cap)).astype(np.int8)
    return since, until


def band_of(area, rows, cols, shape):
    """Distance band (index in BANDS) of every cell from the flagged cells of `area` [time, cell]."""
    result = np.empty(area.shape, dtype=np.int8)
    edges = np.array([low for low, _ in BANDS[1:]])
    for index, frame in enumerate(area):
        if not frame.any():
            result[index] = len(BANDS) - 1  # nothing flagged: every cell is far
            continue
        lattice = np.ones(shape, dtype=bool)
        lattice[rows[frame], cols[frame]] = False
        distance = ndimage.distance_transform_cdt(lattice, metric="chessboard")[rows, cols]
        result[index] = np.searchsorted(edges, distance, side="right")
    return result


def event_windows(events, onsets, grid, ring_km, before_hours):
    """(id, type, cells of the box and of the ring, first and last timestamp) of every event."""
    # imported here: these modules import, through others, since_and_until of this one
    from scripts.check_era5_known_events import in_box
    from scripts.check_era5_known_events_lead import Layout, ring_cells
    layout = Layout(grid)
    out = []
    for _, event in events.iterrows():
        onset = onsets.get(event["id"], event["inizio"].normalize())
        cells = in_box(event, layout.latitudes, layout.longitudes) | ring_cells(event, layout, ring_km)
        out.append((event["id"], event.get("tipo", ""), np.asarray(cells),
                    onset - pd.Timedelta(hours=before_hours), event["fine"].normalize() + pd.Timedelta(hours=21)))
    return out


def measures(counts):
    """From counts [..., group, band, window, kind]: front and trail % of both groups, front x, and the excess
    (front - trail of the first only) - (front - trail of neither), all [..., band, window]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        share = 100 * counts / counts.sum(axis=-1, keepdims=True)
        front, trail = share[..., 1], share[..., 2]
        return {"first front": front[..., 0, :, :], "first trail": trail[..., 0, :, :],
                "neither front": front[..., 1, :, :], "neither trail": trail[..., 1, :, :],
                "front x": front[..., 0, :, :] / front[..., 1, :, :],
                "excess": (front[..., 0, :, :] - trail[..., 0, :, :]) - (front[..., 1, :, :] - trail[..., 1, :, :])}


def ratio_bins(std, mean):
    """Bin of log10(std / mean) of every value (-1 where it is not defined)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.log10(np.asarray(std, dtype=np.float64) / np.asarray(mean, dtype=np.float64))
    bins = np.clip(((ratio - RATIO_LOW) / (RATIO_HIGH - RATIO_LOW) * RATIO_BINS).astype(np.int64), 0, RATIO_BINS - 1)
    return np.where(np.isfinite(ratio), bins, -1)


def histogram_median(counts):
    """Median relative spread of histograms [..., RATIO_BINS] (NaN where empty)."""
    total = counts.sum(axis=-1)
    cumulative = np.cumsum(counts, axis=-1)
    index = (cumulative < (total / 2)[..., None]).sum(axis=-1)
    centre = RATIO_LOW + (np.minimum(index, RATIO_BINS - 1) + 0.5) * (RATIO_HIGH - RATIO_LOW) / RATIO_BINS
    return np.where(total > 0, 10 ** centre, np.nan)


def ratio_report(histograms, band_names, title):
    """Median relative spread by group, distance and kind, for every window of RATIO_WINDOWS; histograms
    [group, band, ratio window, kind, bin]."""
    medians = histogram_median(histograms)  # [group, band, ratio window, kind]
    rows = []
    for r, hours in enumerate(RATIO_WINDOWS):
        print(f"\n{title}: median relative spread of the dropout (spread / mean of the error), kinds within {hours} h")
        print(f"{'distance, group':<{NAME}}" + "".join(f"{kind:>{WIDTH}}" for kind in ("front", "trail", "both", "none")))
        for b, band in enumerate(band_names):
            for g, name in enumerate(("first only", "neither")):
                print(f"{f'{band}, {name}':<{NAME}}"
                      + "".join(f"{medians[g, b, r, k]:>{WIDTH}.3f}" for k in (1, 2, 3, 0)))
                rows += [{"window_hours": hours, "distance": band, "group": name, "kind": kind,
                          "median relative spread": medians[g, b, r, k], "values": int(histograms[g, b, r, k].sum())}
                         for k, kind in enumerate(KINDS)]
    return pd.DataFrame(rows)


def event_ratio_report(event_ratio, windows_of_events, band_names, draws, seed, path):
    """On the known events, for every window of RATIO_WINDOWS, all the events and type by type: median relative
    spread of front and trail, of the first only and of neither, at distances 1 and 2, and the difference front -
    trail of the first only with the 95% interval of resamples of the events of the group."""
    rng = np.random.default_rng(seed)
    groups = [("all", list(range(len(windows_of_events))))]
    for kind in sorted({kind for _, kind, *_ in windows_of_events}):
        groups.append((kind, [index for index, (_, other, *_) in enumerate(windows_of_events) if other == kind]))
    bands = range(min(2, len(band_names)))
    rows = []
    for r, hours in enumerate(RATIO_WINDOWS):
        print(f"\nknown events, kinds within {hours} h: median relative spread of the dropout, front and trail; "
              f"first only front - trail with 95% from {draws} resamples of the events of the group")
        print(f"{'group, distance':<22}{'events':>8}{'n first':>10}{'front':>9}{'trail':>9}{'n front':>9}{'n trail':>9}"
              f"{'front-trail':>13}{'  (95%)':<20}")
        for name, chosen in groups:
            part = event_ratio[chosen][:, :, :, r]  # [event, group, band, kind, bin]
            own = histogram_median(part.sum(axis=0))  # [group, band, kind]
            counts = np.stack([np.bincount(rng.integers(0, len(chosen), len(chosen)), minlength=len(chosen))
                               for _ in range(draws)])  # [draw, event]: times every event is drawn
            fronts_trails = part[:, 0, :, 1:3]  # first only, [event, band, front/trail, bin]
            resampled = (counts @ fronts_trails.reshape(len(chosen), -1)).reshape(draws, *fronts_trails.shape[1:])
            medians = histogram_median(resampled)  # [draw, band, front/trail]
            for b in bands:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    low, high = np.nanpercentile(medians[:, b, 0] - medians[:, b, 1], (2.5, 97.5))
                difference = own[0, b, 1] - own[0, b, 2]
                print(f"{f'{name}, {band_names[b]}':<22}{len(chosen):>8d}{int(part[:, 0, b].sum()):>10d}"
                      f"{own[0, b, 1]:>9.3f}{own[0, b, 2]:>9.3f}{own[1, b, 1]:>9.3f}{own[1, b, 2]:>9.3f}"
                      f"{difference:>+13.3f}{f'  ({low:+.3f}, {high:+.3f})':<20}")
                rows.append({"window_hours": hours, "type": name, "distance": band_names[b], "events": len(chosen),
                             "values first only": int(part[:, 0, b].sum()),
                             "first front": own[0, b, 1], "first trail": own[0, b, 2],
                             "neither front": own[1, b, 1], "neither trail": own[1, b, 2],
                             "first front - trail": difference, "2.5%": low, "97.5%": high})
    if path is not None:
        pd.DataFrame(rows).to_csv(path.with_name(path.stem + "_dropout_tipi.csv"), index=False)


def per_event(event_kinds, windows_of_events, band_names, path):
    """Excess of every event at distance 1, its share positive over the events with enough values, and a sign test."""
    from scipy.stats import binomtest
    rows = []
    for e, (identifier, kind, *_) in enumerate(windows_of_events):
        own = measures(event_kinds[e])
        rows.append({"id": identifier, "type": kind, "values": int(event_kinds[e, 0, 0, 0].sum()),
                     **{f"excess {hours} h": float(own["excess"][0, w]) for w, hours in enumerate(WINDOWS_HOURS)},
                     **{f"front x {hours} h": float(own["front x"][0, w]) for w, hours in enumerate(WINDOWS_HOURS)}})
    table = pd.DataFrame(rows)
    enough = table[table["values"] >= MIN_EVENT_VALUES]
    print(f"\nevent by event, distance {band_names[0]}: events with at least {MIN_EVENT_VALUES} values of the first only: "
          f"{len(enough)} of {len(table)}")
    print(f"{'window':<10}{'positive':>10}{'negative':>10}{'share +':>10}{'sign test p':>13}{'median pp':>11}")
    for hours in WINDOWS_HOURS:
        values = enough[f"excess {hours} h"].dropna()
        positive, negative = int((values > 0).sum()), int((values < 0).sum())
        p = binomtest(positive, positive + negative).pvalue if positive + negative else np.nan
        print(f"{f'{hours} h':<10}{positive:>10d}{negative:>10d}{100 * positive / max(positive + negative, 1):>10.1f}"
              f"{p:>13.3g}{values.median():>11.1f}")
    print("by type, 3 h: " + "; ".join(
        f"{kind} {int((group['excess 3 h'] > 0).sum())}/{len(group)}"
        for kind, group in enough.groupby("type")))
    if path is not None:
        table.to_csv(path.with_name(path.stem + "_per_evento.csv"), index=False)


def event_report(event_kinds, windows_of_events, band_names, draws, seed, short, path, event_ratio=None):
    """Print and write the kinds on the values of the known events, with bootstrap intervals over the events."""
    total = measures(event_kinds.sum(axis=0))
    rng = np.random.default_rng(seed)
    count = len(event_kinds)
    sample = rng.integers(0, count, size=(draws, count))
    boot = measures(event_kinds[sample].sum(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low = {key: np.nanpercentile(value, 2.5, axis=0) for key, value in boot.items()}
        high = {key: np.nanpercentile(value, 97.5, axis=0) for key, value in boot.items()}
    values = event_kinds[:, 0, :, 0, :].sum(axis=-1)  # [event, band]: values of the first only
    print(f"\n[edge-lead, known events] {count} events, box and ring; {short[0]} only vs neither at the same "
          f"distance; 95% intervals from {draws} resamples of the events")
    print(f"{'distance, window':<22}{'values':>9}{'events':>8}{'front':>8}{'trail':>8}{'n front':>9}{'n trail':>9}"
          f"{'front x':>9}{'  (95%)':<16}{'excess pp':>10}{'  (95%)':<16}")
    rows = []
    for b, band in enumerate(band_names):
        for w, hours in enumerate(WINDOWS_HOURS):
            pick = lambda table, key: table[key][b, w]
            interval = lambda key: f"  ({low[key][b, w]:.2f}, {high[key][b, w]:.2f})"
            print(f"{f'{band}, {hours} h':<22}{int(values[:, b].sum()):>9d}{int((values[:, b] > 0).sum()):>8d}"
                  f"{pick(total, 'first front'):>8.1f}{pick(total, 'first trail'):>8.1f}"
                  f"{pick(total, 'neither front'):>9.1f}{pick(total, 'neither trail'):>9.1f}"
                  f"{pick(total, 'front x'):>9.2f}{interval('front x'):<16}{pick(total, 'excess'):>10.1f}"
                  f"{interval('excess'):<16}")
            rows.append({"type": "all", "distance": band, "window_hours": hours, "values": int(values[:, b].sum()),
                         "events with values": int((values[:, b] > 0).sum()),
                         **{key: float(pick(total, key)) for key in total},
                         **{f"{key} 2.5%": float(low[key][b, w]) for key in ("front x", "excess")},
                         **{f"{key} 97.5%": float(high[key][b, w]) for key in ("front x", "excess")}})
    print("excess: (front - trail) of the first only minus (front - trail) of neither, percentage points; "
          "front x: front of the first only over front of neither")
    kinds = sorted({kind for _, kind, *_ in windows_of_events})
    print(f"\nby type, distance {band_names[0]}, 95% from {draws} resamples of the events of the type")
    print(f"{'type, window':<18}{'events':>8}{'values':>9}{'front':>8}{'trail':>8}{'n front':>9}{'n trail':>9}"
          f"{'excess pp':>10}{'  (95%)':<18}")
    for kind in kinds:
        chosen = np.array([index for index, (_, other, *_) in enumerate(windows_of_events) if other == kind])
        part = measures(event_kinds[chosen].sum(axis=0))
        draws_of_type = np.stack([np.bincount(rng.integers(0, len(chosen), len(chosen)), minlength=len(chosen))
                                  for _ in range(draws)])
        flat = event_kinds[chosen][:, :, 0].reshape(len(chosen), -1)  # distance 1: [event, group x window x kind]
        resampled = measures((draws_of_type @ flat).reshape(draws, len(GROUPS), 1, len(WINDOWS_HOURS), len(KINDS)))
        for w, hours in enumerate(WINDOWS_HOURS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                low_type, high_type = np.nanpercentile(resampled["excess"][:, 0, w], (2.5, 97.5))
            print(f"{f'{kind}, {hours} h':<18}{len(chosen):>8d}{int(values[chosen, 0].sum()):>9d}"
                  f"{part['first front'][0, w]:>8.1f}{part['first trail'][0, w]:>8.1f}"
                  f"{part['neither front'][0, w]:>9.1f}{part['neither trail'][0, w]:>9.1f}"
                  f"{part['excess'][0, w]:>10.1f}{f'  ({low_type:.1f}, {high_type:.1f})':<18}")
            rows.append({"type": kind, "distance": band_names[0], "window_hours": hours,
                         "values": int(values[chosen, 0].sum()), "events with values": len(chosen),
                         **{key: float(value[0, w]) for key, value in part.items()},
                         "excess 2.5%": float(low_type), "excess 97.5%": float(high_type)})
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"written {path}")
    per_event(event_kinds, windows_of_events, band_names, path)
    if event_ratio is not None:
        table = ratio_report(event_ratio.sum(axis=0), band_names, "known events")
        if path is not None:
            table.to_csv(path.with_name(path.stem + "_dropout.csv"), index=False)
        event_ratio_report(event_ratio, windows_of_events, band_names, draws, seed, path)


def tercile_report(period, event_ratio, windows_of_events, draws, seed, path):
    """Excess of the first only at distance 1 by third of the relative spread (cuts from the whole period), within
    the windows of RATIO_WINDOWS: whole period, and on the events for all of them and type by type, with intervals
    from resamples of the events. period [group, band, ratio window, kind, bin], event_ratio [event, ...]."""
    values = period[0, 0, 0].sum(axis=0)  # first only, distance 1: every value once, whatever its kind
    cumulative = np.cumsum(values) / max(values.sum(), 1)
    cuts = [0, int(np.searchsorted(cumulative, 1 / 3)) + 1, int(np.searchsorted(cumulative, 2 / 3)) + 1, RATIO_BINS]
    edge = lambda index: 10 ** (RATIO_LOW + index * (RATIO_HIGH - RATIO_LOW) / RATIO_BINS)
    names = (f"low (<{edge(cuts[1]):.3f})", f"middle ({edge(cuts[1]):.3f}-{edge(cuts[2]):.3f})",
             f"high (>{edge(cuts[2]):.3f})")

    def thirds(histogram):
        """[..., group, ratio window, kind, bin] at distance 1 -> excess by third [..., ratio window, third] (%)."""
        first = np.stack([histogram[..., 0, :, :, cuts[t]:cuts[t + 1]].sum(axis=-1) for t in range(3)], axis=-3)
        neither = histogram[..., 1, :, :, :].sum(axis=-1)  # [..., ratio window, kind]
        with np.errstate(invalid="ignore", divide="ignore"):
            share = 100 * first / first.sum(axis=-1, keepdims=True)  # [..., third, ratio window, kind]
            reference = 100 * neither / neither.sum(axis=-1, keepdims=True)
        lead = share[..., 1] - share[..., 2] - (reference[..., None, :, 1] - reference[..., None, :, 2])
        # shares of every kind [..., ratio window, third, kind] and of neither [..., ratio window, kind]
        return np.swapaxes(lead, -1, -2), np.swapaxes(share, -2, -3), reference

    rng = np.random.default_rng(seed)
    groups = [("whole period", None), ("events: all", np.arange(len(windows_of_events)))]
    for kind in sorted({kind for _, kind, *_ in windows_of_events}):
        groups.append((f"events: {kind}", np.array([index for index, (_, other, *_) in enumerate(windows_of_events)
                                                    if other == kind])))
    rows = []
    for r, hours in enumerate(RATIO_WINDOWS):
        print(f"\nthirds of the relative spread of the first only at distance 1, kinds within {hours} h: excess pp "
              f"(front - trail of the third minus that of neither); 95% from {draws} resamples of the events")
        print(f"{'group':<20}" + "".join(f"{name:>34}" for name in names))
        for label, chosen in groups:
            if chosen is None:
                excess, shares, reference = thirds(period[:, 0])
                low = high = np.full(excess.shape, np.nan)
            else:
                part = event_ratio[chosen][:, :, 0]  # [event, group, ratio window, kind, bin]
                excess, shares, reference = thirds(part.sum(axis=0))
                counts = np.stack([np.bincount(rng.integers(0, len(chosen), len(chosen)), minlength=len(chosen))
                                   for _ in range(draws)])
                resampled = (counts @ part.reshape(len(chosen), -1)).reshape(draws, *part.shape[1:])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    low, high = np.nanpercentile(thirds(resampled)[0], (2.5, 97.5), axis=0)
            cells = []
            for t in range(3):
                interval = "" if np.isnan(low[r, t]) else f" ({low[r, t]:+.1f}, {high[r, t]:+.1f})"
                cells.append(f"{excess[r, t]:+.1f}{interval}")
                rows.append({"window_hours": hours, "group": label, "third": names[t],
                             **{f"{kind} %": shares[r, t, k] for k, kind in enumerate(KINDS)},
                             **{f"neither {kind} %": reference[r, k] for k, kind in enumerate(KINDS)},
                             "excess": excess[r, t], "2.5%": low[r, t], "97.5%": high[r, t]})
            print(f"{label:<20}" + "".join(f"{cell:>34}" for cell in cells))
        print(f"kinds within {hours} h, % of the values of the third: {' / '.join(KINDS)}; last: neither")
        for label, _ in groups:
            mine = [row for row in rows if row["window_hours"] == hours and row["group"] == label]
            parts = [" / ".join(f"{row[f'{kind} %']:.1f}" for kind in KINDS) for row in mine]
            parts.append(" / ".join(f"{mine[0][f'neither {kind} %']:.1f}" for kind in KINDS))
            print(f"{label:<20}" + "".join(f"{part:>28}" for part in parts))
    if path is not None:
        pd.DataFrame(rows).to_csv(path.with_name(path.stem + "_terzi.csv"), index=False)


def strength_report(joint, event_joint, period_neither, event_neither, windows_of_events, draws, seed, path):
    """Excess of the first only at distance 1 by third of its score (strength), and by third of the relative spread
    inside every third of strength. joint [ratio window, kind, strength bin, ratio bin] over the whole period,
    event_joint [event, ...]; period_neither and event_neither [(event,) ratio window, kind]: neither at distance 1."""
    marginal = joint[0].sum(axis=(0, 2))  # strength, every value once
    cut = lambda counts: [0, int(np.searchsorted(np.cumsum(counts) / max(counts.sum(), 1), 1 / 3)) + 1,
                          int(np.searchsorted(np.cumsum(counts) / max(counts.sum(), 1), 2 / 3)) + 1, len(counts)]
    strength_cuts = cut(marginal)
    ratio_cuts = [cut(joint[0][:, strength_cuts[t]:strength_cuts[t + 1]].sum(axis=(0, 1))) for t in range(3)]

    def cells(histogram):
        """[..., ratio window, kind, strength, ratio] -> kinds by strength third [..., rw, 3, kind] and by strength
        and ratio thirds [..., rw, 3, 3, kind]."""
        by_strength = np.stack([histogram[..., strength_cuts[t]:strength_cuts[t + 1], :].sum(axis=(-2, -1))
                                for t in range(3)], axis=-2)
        both = np.stack([np.stack([histogram[..., strength_cuts[t]:strength_cuts[t + 1],
                                             ratio_cuts[t][u]:ratio_cuts[t][u + 1]].sum(axis=(-2, -1))
                                   for u in range(3)], axis=-2) for t in range(3)], axis=-3)
        return by_strength, both

    def excess(first, neither):
        """first [..., kind], neither [..., kind] broadcast: (front - trail) % minus that of neither."""
        with np.errstate(invalid="ignore", divide="ignore"):
            share = 100 * first / first.sum(axis=-1, keepdims=True)
            reference = 100 * neither / neither.sum(axis=-1, keepdims=True)
        return share[..., 1] - share[..., 2] - (reference[..., 1] - reference[..., 2])

    def table(histogram, neither):
        by_strength, both = cells(histogram)
        return excess(by_strength, neither[..., None, :]), excess(both, neither[..., None, None, :])

    rng = np.random.default_rng(seed)
    groups = [("whole period", None), ("events: all", np.arange(len(windows_of_events)))]
    for kind in sorted({kind for _, kind, *_ in windows_of_events}):
        groups.append((f"events: {kind}", np.array([index for index, (_, other, *_) in enumerate(windows_of_events)
                                                    if other == kind])))
    edge = lambda index, low, high, bins: 10 ** (low + index * (high - low) / bins)
    strength_names = [f"{edge(strength_cuts[t], STRENGTH_LOW, STRENGTH_HIGH, STRENGTH_BINS):.2f}-"
                      f"{edge(strength_cuts[t + 1], STRENGTH_LOW, STRENGTH_HIGH, STRENGTH_BINS):.2f}" for t in range(3)]
    print(f"\nstrength of the first only at distance 1 (its score), thirds: {', '.join(strength_names)}; inside every "
          f"third, thirds of the relative spread (low, middle, high)")
    rows = []
    for label, chosen in groups:
        if chosen is None:
            strength, both = table(joint, period_neither)
            intervals = None
        else:
            own = event_joint[chosen]
            strength, both = table(own.sum(axis=0), event_neither[chosen].sum(axis=0))
            by_strength, by_both = cells(own)  # [event, rw, 3, kind], [event, rw, 3, 3, kind]
            counts = np.stack([np.bincount(rng.integers(0, len(chosen), len(chosen)), minlength=len(chosen))
                               for _ in range(draws)]).astype(float)
            resample = lambda array: (counts @ array.reshape(len(chosen), -1)).reshape(draws, *array.shape[1:])
            neither = resample(event_neither[chosen])
            boot_strength = excess(resample(by_strength), neither[..., None, :])
            boot_both = excess(resample(by_both), neither[..., None, None, :])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                intervals = (np.nanpercentile(boot_strength, (2.5, 97.5), axis=0),
                             np.nanpercentile(boot_both, (2.5, 97.5), axis=0))
        for r, hours in enumerate(RATIO_WINDOWS):
            print(f"{label}, within {hours} h (excess pp" + (", 95% from resamples of the events)" if intervals
                                                             is not None else ")"))
            for t in range(3):
                pieces = []
                for u, spread in enumerate(("low", "middle", "high")):
                    value = both[r, t, u]
                    interval = ("" if intervals is None else
                                f" ({intervals[1][0][r, t, u]:+.1f}, {intervals[1][1][r, t, u]:+.1f})")
                    pieces.append(f"{spread} {value:+.1f}{interval}")
                    rows.append({"window_hours": hours, "group": label, "strength": strength_names[t],
                                 "spread": spread, "excess": value,
                                 "2.5%": np.nan if intervals is None else intervals[1][0][r, t, u],
                                 "97.5%": np.nan if intervals is None else intervals[1][1][r, t, u]})
                whole = strength[r, t]
                interval = ("" if intervals is None else
                            f" ({intervals[0][0][r, t]:+.1f}, {intervals[0][1][r, t]:+.1f})")
                rows.append({"window_hours": hours, "group": label, "strength": strength_names[t], "spread": "all",
                             "excess": whole, "2.5%": np.nan if intervals is None else intervals[0][0][r, t],
                             "97.5%": np.nan if intervals is None else intervals[0][1][r, t]})
                print(f"  strength {strength_names[t]:<14} all {whole:+.1f}{interval} | " + " | ".join(pieces))
    if path is not None:
        pd.DataFrame(rows).to_csv(path.with_name(path.stem + "_forza.csv"), index=False)


def unseen_breakdown(years, year_hits, year_defined, names, percent, hours, path, spread=None):
    """The shares of the unseen variables at distance 1 year by year, and by third of the relative spread."""
    from scipy.stats import binomtest
    with np.errstate(invalid="ignore", divide="ignore"):
        every = 100 * year_hits.sum(axis=2) / year_defined.sum(axis=2)  # [year, group, variable]
        none = 100 * year_hits[:, :, 0] / year_defined[:, :, 0]
    rows = []
    for v, variable in enumerate(names):
        print(f"\n{variable}: % of the values at distance 1 in its highest {percent:g}%, year by year (kinds within "
              f"{hours} h together, and the kind none)")
        print(f"{'year':<8}{'first all':>12}{'neither all':>13}{'ratio':>8}{'first none':>12}{'neither none':>14}")
        for y, year in enumerate(years):
            print(f"{year:<8}{every[y, 0, v]:>12.1f}{every[y, 1, v]:>13.1f}{every[y, 0, v] / every[y, 1, v]:>8.2f}"
                  f"{none[y, 0, v]:>12.1f}{none[y, 1, v]:>14.1f}")
            rows.append({"variable": variable, "year": int(year), "first all %": every[y, 0, v],
                         "neither all %": every[y, 1, v], "first none %": none[y, 0, v], "neither none %": none[y, 1, v]})
        for label, values in (("all", every[:, :, v]), ("none", none[:, :, v])):
            above, below = int((values[:, 0] > values[:, 1]).sum()), int((values[:, 0] < values[:, 1]).sum())
            p = binomtest(above, above + below).pvalue if above + below else np.nan
            print(f"  {label}: first above neither in {above} of {above + below} years (sign test p {p:.3g})")
    if path is not None:
        pd.DataFrame(rows).to_csv(path.with_name(path.stem + "_unseen_anni.csv"), index=False)
    if spread is None:
        return
    histograms, spread_hits, spread_defined = spread
    values = histograms[0, 0, 0].sum(axis=0)  # first only, distance 1, every value once
    cumulative = np.cumsum(values) / max(values.sum(), 1)
    cuts = [0, int(np.searchsorted(cumulative, 1 / 3)) + 1, int(np.searchsorted(cumulative, 2 / 3)) + 1, RATIO_BINS]
    rows = []
    for v, variable in enumerate(names):
        print(f"\n{variable}: % in its highest {percent:g}% of the first only at distance 1, by third of the relative "
              f"spread (cuts from the whole period) and kind within {hours} h")
        print(f"{'third':<10}" + "".join(f"{kind:>10}" for kind in ("front", "trail", "both", "none", "all")))
        for t, third in enumerate(("low", "middle", "high")):
            hit = spread_hits[:, v, cuts[t]:cuts[t + 1]].sum(axis=-1)
            known = spread_defined[:, v, cuts[t]:cuts[t + 1]].sum(axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                share = 100 * hit / known
                total = 100 * hit.sum() / known.sum()
            print(f"{third:<10}" + "".join(f"{share[k]:>10.1f}" for k in (1, 2, 3, 0)) + f"{total:>10.1f}")
            rows.append({"variable": variable, "third": third, **{kind: share[k] for k, kind in enumerate(KINDS)},
                         "all": total})
    if path is not None:
        pd.DataFrame(rows).to_csv(path.with_name(path.stem + "_unseen_terzi.csv"), index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, nargs="+", required=True,
                        help="Run with the second score; with several, a value is flagged when any of them flags it")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the second score; default --top-percent")
    parser.add_argument("--unseen-dir", type=Path, help="Monthly NetCDF files of the unseen variables")
    parser.add_argument("--unusual-percent", type=float, default=5, help="Unusual: this highest share of an unseen variable")
    parser.add_argument("--unseen-hours", type=int, default=UNSEEN_WINDOW_HOURS,
                        help="Kinds of the unseen variables within this many hours (24: 'none' = never reached in a day)")
    parser.add_argument("--csv", type=Path, help="Also write the shares of the kinds here")
    parser.add_argument("--events", type=Path, help="CSV of known events: also count on their boxes and days only")
    parser.add_argument("--onsets", type=Path, help="inizi.csv of check_era5_known_events_lead.py (measured onsets)")
    parser.add_argument("--ring-km", type=float, default=250)
    parser.add_argument("--before-hours", type=float, default=72)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--events-csv", type=Path, help="Write the kinds on the known events here")
    parser.add_argument("--ratio-run", type=Path,
                        help="Scored run of the model with the dropout: relative spread of the draws by kind")
    args = parser.parse_args(argv)
    first, timestamps, _ = flags(args.score_dir, args.top_percent)
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    second = None
    for path in args.by_dir:
        flagged, other_timestamps, _ = flags(path, by_top)
        if not timestamps.equals(other_timestamps):
            common = timestamps.intersection(other_timestamps)  # runs scored over slightly different spans
            if len(common) < 2:
                raise ValueError(f"The scores have no timestamps in common: {path}")
            print(f"[edge-lead] {path.name}: {len(common)} timestamps in common of {len(timestamps)} and "
                  f"{len(other_timestamps)}", flush=True)
            first = first[timestamps.get_indexer(common)]
            second = None if second is None else second[timestamps.get_indexer(common)]
            flagged = flagged[other_timestamps.get_indexer(common)]
            timestamps = common
        if first.shape != flagged.shape:
            raise ValueError(f"The scores are not over the same cells: {path}")
        second = flagged if second is None else second | flagged
        del flagged
    step_hours = int(round((timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)))
    if any(hours % step_hours for hours in (*WINDOWS_HOURS, args.unseen_hours)):
        raise ValueError(f"The windows must be multiples of {step_hours} hours")
    windows = [hours // step_hours for hours in WINDOWS_HOURS]
    unseen_window = args.unseen_hours // step_hours
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    if len(rows) != first.shape[1]:
        raise ValueError("The grid of the run does not match the cells of the scores")
    latitudes = np.asarray(grid["latitudes"], dtype=float)[rows]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[cols]
    longest = max(windows)
    since, until = since_and_until(second, longest + 1)
    sizes = (len(GROUPS), len(BANDS))
    kinds = np.zeros((*sizes, len(windows), len(KINDS)))
    arrival = np.zeros((*sizes, longest + 2))  # steps to the next flag of the second, for the fronts of the longest window
    hits = defined_counts = names = None
    windows_of_events = event_kinds = None
    ratio_maps = ratio_rows = ratio_histograms = event_ratio = None
    if args.ratio_run is not None:
        from scripts.check_era5_dust_lead import score_file
        manifest = json.loads((args.ratio_run / "metadata.json").read_text(encoding="utf-8"))
        ratio_maps = (np.load(args.ratio_run / manifest["generator_std_file"], mmap_mode="r"),
                      np.load(args.ratio_run / score_file(manifest, "generator"), mmap_mode="r"))
        ratio_rows = pd.DatetimeIndex(np.load(args.ratio_run / "test_timestamps.npy")).get_indexer(timestamps)
        if (ratio_rows < 0).any() or ratio_maps[0].shape[1] != first.shape[1]:
            raise ValueError("The dropout run does not cover the timestamps and cells of the scores")
        ratio_histograms = np.zeros((*sizes, len(RATIO_WINDOWS), len(KINDS), RATIO_BINS))
        ratio_steps = [windows[WINDOWS_HOURS.index(hours)] for hours in RATIO_WINDOWS]
        # the score of the first (its strength), for its values alone at distance 1
        model_manifest = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))
        strength_map = np.load(args.score_dir / score_file(model_manifest, "combined"), mmap_mode="r")
        strength_rows = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy")).get_indexer(timestamps)
        joint = np.zeros((len(RATIO_WINDOWS), len(KINDS), STRENGTH_BINS, RATIO_BINS))
        joint_size = joint.size
    if args.events is not None:
        onsets = {}
        if args.onsets is not None:
            table = pd.read_csv(args.onsets, parse_dates=["inizio"])
            table = table[table["variante"] == "principale"] if "variante" in table else table
            onsets = dict(zip(table["id"], table["inizio"]))
        from scripts.check_era5_known_events import read_events
        windows_of_events = event_windows(read_events(args.events), onsets, grid, args.ring_km, args.before_hours)
        event_kinds = np.zeros((len(windows_of_events), *sizes, len(windows), len(KINDS)))
        # fronts of the longest window: steps until the second flags the cell, event by event
        event_arrival = np.zeros((len(windows_of_events), *sizes, longest + 2))
        event_arrival_ratio = None
        if ratio_maps is not None:
            event_ratio = np.zeros((len(windows_of_events), *sizes, len(RATIO_WINDOWS), len(KINDS), RATIO_BINS))
            event_joint = np.zeros((len(windows_of_events), *joint.shape))
            # the same for the first only at distance 1, by bin of the relative spread
            event_arrival_ratio = np.zeros((len(windows_of_events), longest + 2, RATIO_BINS))
    read = 0
    for month in range(1, 13):
        positions = np.nonzero(timestamps.month == month)[0]
        if not len(positions):
            continue
        unusual = None
        if args.unseen_dir is not None:
            years = sorted(set(timestamps.year[positions]))
            parts = [read_month(args.unseen_dir, year, month, latitudes, longitudes) for year in years]
            names = parts[0][2] if names is None else names
            if any(part[2] != names for part in parts):
                raise ValueError(f"The files of month {month} do not hold the variables of the other months")
            times = parts[0][0].append([part[0] for part in parts[1:]])
            found = timestamps.get_indexer(times)
            kept = found >= 0  # a file may hold timestamps the score does not have
            values = np.concatenate([part[1] for part in parts])[kept]
            del parts
            order = np.argsort(found[kept])
            if not np.array_equal(found[kept][order], positions):
                raise ValueError(f"The unseen variables of month {month} do not cover the timestamps of the scores")
            unusual, defined = unusual_values(values[order], timestamps.hour[positions].to_numpy(), args.unusual_percent)
            del values
            if hits is None:
                hits, defined_counts = (np.zeros((*sizes, len(KINDS), len(names))) for _ in range(2))
                years = np.unique(timestamps.year)
                # distance 1 only: [year, group, kind, variable] and [kind, variable, ratio bin] of the first only
                year_hits, year_defined = (np.zeros((len(years), len(GROUPS), len(KINDS), len(names))) for _ in range(2))
                if ratio_maps is not None:
                    spread_hits, spread_defined = (np.zeros((len(KINDS), len(names), RATIO_BINS)) for _ in range(2))
        for start in range(0, len(positions), CHUNK):
            chosen = positions[start:start + CHUNK]
            model, other = first[chosen], second[chosen]
            group = np.full(model.shape, -1, dtype=np.int64)
            group[model & ~other] = 0
            group[~model & ~other] = 1
            band = band_of(other, rows, cols, shape).astype(np.int64)
            valid = group >= 0
            cell_base = group * len(BANDS) + band
            base = cell_base[valid]
            chunk_since, chunk_until = since[chosen], until[chosen]
            before, after = chunk_since[valid], chunk_until[valid]
            chunk_bins = None
            if ratio_maps is not None:
                rows_read = ratio_rows[chosen]
                chunk_bins = ratio_bins(ratio_maps[0][rows_read], ratio_maps[1][rows_read])
                ratio_kind = [(chunk_until <= steps).astype(np.int64) + 2 * (chunk_since <= steps) for steps in ratio_steps]
                keep = valid & (chunk_bins >= 0)
                for r, kind in enumerate(ratio_kind):
                    index = (((cell_base[keep] * len(RATIO_WINDOWS) + r) * len(KINDS) + kind[keep]) * RATIO_BINS
                             + chunk_bins[keep])
                    ratio_histograms += np.bincount(index, minlength=ratio_histograms.size).reshape(ratio_histograms.shape)
                with np.errstate(invalid="ignore", divide="ignore"):
                    strength = np.log10(np.asarray(strength_map[strength_rows[chosen]], dtype=np.float64))
                finite_strength = np.where(np.isfinite(strength), strength, STRENGTH_LOW)  # the others are left out
                strength_bins = np.clip(((finite_strength - STRENGTH_LOW) / (STRENGTH_HIGH - STRENGTH_LOW)
                                         * STRENGTH_BINS).astype(np.int64), 0, STRENGTH_BINS - 1)
                edge_alone = keep & (cell_base == 0) & np.isfinite(strength)  # first only, distance 1
                joint_index = [((r * len(KINDS) + kind) * STRENGTH_BINS + strength_bins) * RATIO_BINS + chunk_bins
                               for r, kind in enumerate(ratio_kind)]
                for index in joint_index:
                    joint += np.bincount(index[edge_alone], minlength=joint_size).reshape(joint.shape)
            if windows_of_events is not None:
                chunk_times = timestamps[chosen]
                for e, (_, _, cells, first_time, last_time) in enumerate(windows_of_events):
                    at = np.asarray((chunk_times >= first_time) & (chunk_times <= last_time))
                    if not at.any():
                        continue
                    mask = valid & at[:, None] & cells[None, :]
                    inside, since_e, until_e = cell_base[mask], chunk_since[mask], chunk_until[mask]
                    if chunk_bins is not None:
                        keep = mask & (chunk_bins >= 0)
                        for r, kind in enumerate(ratio_kind):
                            index = (((cell_base[keep] * len(RATIO_WINDOWS) + r) * len(KINDS) + kind[keep]) * RATIO_BINS
                                     + chunk_bins[keep])
                            event_ratio[e] += np.bincount(index, minlength=event_ratio[e].size).reshape(
                                event_ratio[e].shape)
                        for index in joint_index:
                            event_joint[e] += np.bincount(index[mask & edge_alone], minlength=joint_size).reshape(
                                joint.shape)
                    front_e = (until_e <= longest) & (since_e > longest)
                    event_arrival[e] += np.bincount(inside[front_e] * (longest + 2) + until_e[front_e],
                                                    minlength=event_arrival[e].size).reshape(event_arrival[e].shape)
                    if event_arrival_ratio is not None:
                        chosen_front = (mask & (cell_base == 0) & (chunk_bins >= 0) & (chunk_until <= longest)
                                        & (chunk_since > longest))
                        event_arrival_ratio[e] += np.bincount(
                            chunk_until[chosen_front].astype(np.int64) * RATIO_BINS + chunk_bins[chosen_front],
                            minlength=event_arrival_ratio[e].size).reshape(event_arrival_ratio[e].shape)
                    for w, window in enumerate(windows):
                        kind = (until_e <= window).astype(np.int64) + 2 * (since_e <= window)
                        event_kinds[e, ..., w, :] += np.bincount(
                            inside * len(KINDS) + kind, minlength=len(GROUPS) * len(BANDS) * len(KINDS)
                        ).reshape(*sizes, len(KINDS))
            for w, window in enumerate(windows):
                kind = (after <= window).astype(np.int64) + 2 * (before <= window)
                kinds[..., w, :] += np.bincount(base * len(KINDS) + kind, minlength=kinds[..., w, :].size
                                                ).reshape(*sizes, len(KINDS))
            front = (after <= longest) & (before > longest)
            arrival += np.bincount(base[front] * (longest + 2) + after[front], minlength=arrival.size).reshape(arrival.shape)
            if unusual is not None:
                kind = (after <= unseen_window).astype(np.int64) + 2 * (before <= unseen_window)
                index = base * len(KINDS) + kind
                block, known = unusual[start:start + CHUNK][valid], defined[start:start + CHUNK][valid]
                for v in range(len(names)):
                    hits[..., v] += np.bincount(index, weights=block[:, v], minlength=hits[..., v].size).reshape(
                        *sizes, len(KINDS))
                    defined_counts[..., v] += np.bincount(index, weights=known[:, v],
                                                          minlength=hits[..., v].size).reshape(*sizes, len(KINDS))
                near = base % len(BANDS) == 0  # distance 1
                year_of = np.searchsorted(years, np.broadcast_to(timestamps.year[chosen].to_numpy()[:, None],
                                                                 valid.shape)[valid])
                index = (year_of * len(GROUPS) + base // len(BANDS)) * len(KINDS) + kind
                for v in range(len(names)):
                    year_hits[..., v] += np.bincount(index[near], weights=block[near, v],
                                                     minlength=year_hits[..., v].size).reshape(year_hits.shape[:-1])
                    year_defined[..., v] += np.bincount(index[near], weights=known[near, v],
                                                        minlength=year_hits[..., v].size).reshape(year_hits.shape[:-1])
                if chunk_bins is not None:
                    alone = (base == 0) & (chunk_bins[valid] >= 0)  # first only, distance 1, with a spread
                    index = kind[alone] * RATIO_BINS + chunk_bins[valid][alone]
                    for v in range(len(names)):
                        spread_hits[:, v] += np.bincount(index, weights=block[alone, v],
                                                         minlength=len(KINDS) * RATIO_BINS).reshape(len(KINDS), -1)
                        spread_defined[:, v] += np.bincount(index, weights=known[alone, v],
                                                            minlength=len(KINDS) * RATIO_BINS).reshape(len(KINDS), -1)
        read += len(positions)
        del unusual

    names_of_runs = [path.name for path in (args.score_dir, *args.by_dir)]
    common = len(os.path.commonprefix(names_of_runs))
    short = [name[common:] or name for name in names_of_runs]
    short = [short[0], " or ".join(short[1:])]
    band_names = [f"{low}" if low == high else (f"{low}-{high}" if high else f">{low - 1}") for low, high in BANDS]
    print(f"[edge-lead] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest {args.top_percent:g}% "
          f"of {short[0]}, {by_top:g}% of {short[1]} | distance: cells of the grid from the area {short[1]} flags at the "
          f"same timestamp, the larger of the row and column offsets")
    counts = kinds[..., 0, :].sum(axis=-1)  # [group, band]
    print(f"\nvalues by distance from the area of {short[1]} (% of the group)")
    print(f"{'':<{NAME}}{'values':>{WIDTH + 4}}" + "".join(f"{name:>{WIDTH}}" for name in band_names))
    for g, name in enumerate((f"{short[0]} only", "neither")):
        total = counts[g].sum()
        print(f"{name:<{NAME}}{int(total):>{WIDTH + 4}d}"
              + "".join(f"{100 * value / max(total, 1):>{WIDTH}.1f}" for value in counts[g]))
    rows_out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for w, hours in enumerate(WINDOWS_HOURS):
            share = 100 * kinds[..., w, :] / kinds[..., w, :].sum(axis=-1, keepdims=True)
            print(f"\nwithin {hours} h at the same cell, % of the values of each group and distance: {short[1]} flags it "
                  f"after and not before (front), before and not after (trail), both, none")
            print(f"{'distance, group':<{NAME}}" + "".join(f"{kind:>{WIDTH}}" for kind in ("front", "trail", "both", "none"))
                  + f"{'front x':>{WIDTH}}{'trail x':>{WIDTH}}")
            for b, band in enumerate(band_names):
                for g, name in enumerate((f"{short[0]} only", "neither")):
                    ratios = ([share[0, b, 1] / share[1, b, 1], share[0, b, 2] / share[1, b, 2]] if g == 0
                              else [np.nan, np.nan])
                    print(f"{f'{band}, {name}'[:NAME - 1]:<{NAME}}"
                          + "".join(f"{share[g, b, k]:>{WIDTH}.1f}" for k in (1, 2, 3, 0))
                          + "".join(f"{'' if np.isnan(r) else format(r, '.2f'):>{WIDTH}}" for r in ratios))
                    for k, kind in enumerate(KINDS):
                        rows_out.append({"group": "first only" if g == 0 else "neither", "first": short[0],
                                         "second": short[1], "distance": band, "window_hours": hours, "kind": kind,
                                         "share": share[g, b, k], "values": int(counts[g, b]),
                                         "top_percent": args.top_percent, "by_top_percent": by_top})
        print(f"\nfronts within {WINDOWS_HOURS[-1]} h: hours until {short[1]} flags the cell (% of the fronts)")
        steps = range(1, longest + 1)
        print(f"{'distance, group':<{NAME}}{'fronts':>{WIDTH + 4}}" + "".join(f"{f'{s * step_hours}h':>{WIDTH}}" for s in steps)
              + f"{'median h':>{WIDTH}}")
        for b, band in enumerate(band_names):
            for g, name in enumerate((f"{short[0]} only", "neither")):
                counted = arrival[g, b, 1:longest + 1]
                total = counted.sum()
                median = (np.searchsorted(np.cumsum(counted), total / 2) + 1) * step_hours if total else np.nan
                print(f"{f'{band}, {name}'[:NAME - 1]:<{NAME}}{int(total):>{WIDTH + 4}d}"
                      + "".join(f"{100 * value / max(total, 1):>{WIDTH}.1f}" for value in counted)
                      + f"{'-' if np.isnan(median) else format(median, '.0f'):>{WIDTH}}")
        if hits is not None:
            print(f"\n% of the values where an unseen variable is in its highest {args.unusual_percent:g}% for the cell, "
                  f"the calendar month and the time of day, by kind within {args.unseen_hours} h")
            print(f"{'distance, group, variable':<{NAME}}" + "".join(f"{kind:>{WIDTH}}" for kind in ("front", "trail", "both", "none")))
            for b, band in enumerate(band_names):
                for g, name in enumerate((f"{short[0]} only", "neither")):
                    for v, variable in enumerate(names):
                        share = 100 * hits[g, b, :, v] / defined_counts[g, b, :, v]
                        print(f"{f'{band}, {name}, {variable}'[:NAME - 1]:<{NAME}}"
                              + "".join(f"{'-' if np.isnan(share[k]) else format(share[k], '.1f'):>{WIDTH}}" for k in (1, 2, 3, 0)))
    print("front: the second score flags the cell after and not before, within the window; trail: before and not after; "
          "front x, trail x: the share of the first score only over that of neither, at the same distance")
    if args.unseen_dir is not None and read != len(timestamps):
        raise ValueError(f"Read {read} of the {len(timestamps)} timestamps")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    if hits is not None:
        unseen_breakdown(years, year_hits, year_defined, names, args.unusual_percent, args.unseen_hours, args.csv,
                         None if ratio_histograms is None else (ratio_histograms, spread_hits, spread_defined))
    if ratio_histograms is not None:
        table = ratio_report(ratio_histograms, band_names, "whole period")
        if args.csv is not None:
            table.to_csv(args.csv.with_name(args.csv.stem + "_dropout.csv"), index=False)
    if windows_of_events is not None:
        event_report(event_kinds, windows_of_events, band_names, args.bootstrap, args.seed, short, args.events_csv,
                     event_ratio)
        if event_ratio is not None:
            tercile_report(ratio_histograms, event_ratio, windows_of_events, args.bootstrap, args.seed, args.events_csv)
            steps = [WINDOWS_HOURS.index(hours) for hours in RATIO_WINDOWS]
            strength_report(joint, event_joint, kinds[1, 0][steps], event_kinds[:, 1, 0][:, steps], windows_of_events,
                            args.bootstrap, args.seed, args.events_csv)
        lead_by_type_report(event_arrival, event_arrival_ratio, ratio_histograms, windows_of_events, step_hours, short,
                            args.events_csv)
    return 0


def lead_by_type_report(event_arrival, event_arrival_ratio, ratio_histograms, windows_of_events, step_hours, short, path):
    """At distance 1, for the fronts of the longest window (the second flags the cell after and not before): the hours
    until the second flags it, by type of event and for all the events (median, and the share within one step), for
    the first only, for neither and, with the dropout draws, for the first only in the lowest third of the relative
    spread (cuts from the whole period, as in tercile_report)."""
    kinds_of = [kind for _, kind, *_ in windows_of_events]
    groups = [("all", np.arange(len(kinds_of)))] + [(kind, np.array([e for e, other in enumerate(kinds_of) if other == kind]))
                                                     for kind in sorted(set(kinds_of))]
    low = None
    if event_arrival_ratio is not None:
        values = ratio_histograms[0, 0, 0].sum(axis=0)
        low = int(np.searchsorted(np.cumsum(values) / max(values.sum(), 1), 1 / 3)) + 1

    def summary(counts):
        counted = counts[1:]  # steps 1 .. longest
        total = counted.sum()
        if not total:
            return 0, np.nan, np.nan
        median = (np.searchsorted(np.cumsum(counted), total / 2) + 1) * step_hours
        return int(total), median, 100 * counted[0] / total

    rows = []
    print(f"\nknown events, distance 1: fronts (within {(event_arrival.shape[-1] - 2) * step_hours} h) by type: hours "
          f"until {short[1]} flags the cell, median, and % within {step_hours} h")
    print(f"{'type':<12}" + "".join(f"{name:>30}" for name in ("first only", "first only, low spread", "neither")))
    for label, chosen in groups:
        cells = []
        parts = [("first only", event_arrival[chosen, 0, 0].sum(axis=0))]
        if low is not None:
            parts.append(("first only, low spread", event_arrival_ratio[chosen][..., :low].sum(axis=(0, 2))))
        parts.append(("neither", event_arrival[chosen, 1, 0].sum(axis=0)))
        for name, counts in parts:
            fronts, median, share = summary(counts)
            rows.append({"type": label, "group": name, "fronts": fronts, "median hours": median,
                         f"within {step_hours} h %": share})
            cells.append(f"{fronts} | {median:.0f} h | {share:.0f}%" if fronts else "-")
        if low is None:
            cells.insert(1, "-")
        print(f"{label:<12}" + "".join(f"{cell:>30}" for cell in cells))
    if path is not None:
        pd.DataFrame(rows).to_csv(path.with_name(path.stem + "_anticipo_tipi.csv"), index=False)


if __name__ == "__main__":
    raise SystemExit(main())
