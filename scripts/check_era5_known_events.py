"""The known events of a list compiled without the model, read against a score and against the striking days.

A final check with external knowledge, by a rule fixed on 5 November 2026 before any number was counted:

- with `--score`, for every event of `--events` (a CSV with `id`, `inizio`, `fine` as days, `lat_min`, `lat_max`,
  `lon_min`, `lon_max`): the cells of its box and the timestamps of its days (UTC, first and last included), the
  share of those values the score flags (above the threshold of its highest `--top-percent`, from the validation years
  when the run names a threshold reference, see score_limit), to be read against `--top-percent`, the share
  expected by chance; and the share at the timestamp of the event where it is highest;
- with `--striking-dir` (the output of list_era5_striking_days.py: casi.csv and casi.npz), for every striking day the
  events whose days include it and whose box holds at least one cell of its patch.

Outputs in `--output-dir`: `eventi_noti.csv` and `giorni_ed_eventi.csv`.

    python scripts/check_era5_known_events.py --events docs/era5_casi_noti/lista_ridotta.csv \
        --score G+clim=outputs/optuna27/test_generator_clim_day \
        --striking-dir outputs/optuna27/giorni_eclatanti --output-dir outputs/optuna27/eventi_noti
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path.cwd()))  # after the copy next to the script: a copy run from the checkout imports itself
import numpy as np
import pandas as pd

from scripts.check_era5_dust_lead import score_limit
from scripts.check_era5_event_zones import open_score, release
from scripts.check_era5_score_correlation import parse_score


def read_events(path):
    events = pd.read_csv(path, parse_dates=["inizio", "fine"])
    missing = {"id", "inizio", "fine", "lat_min", "lat_max", "lon_min", "lon_max"} - set(events.columns)
    if missing:
        raise ValueError(f"The events have no {', '.join(sorted(missing))}")
    return events


def in_box(event, latitudes, longitudes):
    """Boolean of the points (latitudes, longitudes of the same length) inside the box of an event, borders included."""
    return ((latitudes >= event["lat_min"]) & (latitudes <= event["lat_max"])
            & (longitudes >= event["lon_min"]) & (longitudes <= event["lon_max"]))


def in_days(event, timestamps):
    """Boolean of the timestamps on the days of an event, the first and the last included."""
    days = pd.DatetimeIndex(timestamps).normalize()
    return np.asarray((days >= event["inizio"].normalize()) & (days <= event["fine"].normalize()))


def score_events(events, source, top_percent):
    manifest, scores, maps = open_score(source)
    grid = manifest["grid"]
    rows, cols = np.asarray(grid["rows"]), np.asarray(grid["cols"])
    latitudes, longitudes = np.asarray(grid["latitudes"])[rows], np.asarray(grid["longitudes"])[cols]
    timestamps = pd.DatetimeIndex(np.load(source[1] / "test_timestamps.npy"))
    records = []
    try:
        limit = score_limit(scores, top_percent)
        for _, event in events.iterrows():
            cells = np.flatnonzero(in_box(event, latitudes, longitudes))
            steps = np.flatnonzero(in_days(event, timestamps))
            record = {"id": event["id"], "celle": len(cells), "istanti": len(steps)}
            if len(cells) and len(steps):
                values = np.asarray(scores[steps[0]:steps[-1] + 1], dtype=np.float64)[steps - steps[0]][:, cells]
                release(scores)
                finite = np.isfinite(values)
                with np.errstate(invalid="ignore"):
                    flagged = values > limit
                share = 100 * flagged.sum(axis=1) / np.maximum(finite.sum(axis=1), 1)
                peak = int(np.argmax(share))
                record.update({"segnalati %": 100 * flagged.sum() / max(int(finite.sum()), 1),
                               "atteso %": top_percent, "istante più segnalato": str(timestamps[steps[peak]]),
                               "segnalati nell'istante più segnalato %": float(share[peak])})
            records.append(record)
    finally:
        for array in maps:
            array._mmap.close()
    return pd.DataFrame(records), float(limit)


def striking_and_events(events, folder):
    cases = pd.read_csv(folder / "casi.csv")
    rows = []
    with np.load(folder / "casi.npz") as arrays:
        latitudes, longitudes = arrays["latitudes"], arrays["longitudes"]
        for _, case in cases.iterrows():
            patch = arrays[f"macchia_{int(case['rango']):02d}"]
            at_rows, at_cols = np.nonzero(patch)
            day = pd.Timestamp(case["giorno"])
            for _, event in events.iterrows():
                if not event["inizio"].normalize() <= day <= event["fine"].normalize():
                    continue
                inside = int(in_box(event, latitudes[at_rows], longitudes[at_cols]).sum())
                if inside:
                    rows.append({"rango": int(case["rango"]), "giorno": case["giorno"], "id": event["id"],
                                 "celle della macchia nel riquadro": inside, "celle della macchia": int(patch.sum())})
    return pd.DataFrame(rows, columns=["rango", "giorno", "id", "celle della macchia nel riquadro", "celle della macchia"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", type=Path, required=True, help="CSV of the known events")
    parser.add_argument("--score", help="NAME=DIR[@component]: the score whose flags are read")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--striking-dir", type=Path, help="Output of list_era5_striking_days.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.score is None and args.striking_dir is None:
        parser.error("give --score, --striking-dir or both")
    events = read_events(args.events)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    show = lambda table: print(table.round(2).to_string(index=False))
    if args.score is not None:
        table, limit = score_events(events, parse_score(args.score), args.top_percent)
        if "tipo" in events:
            named = events[["id", "tipo", "inizio", "fine"]].assign(
                inizio=events["inizio"].dt.strftime("%Y-%m-%d"), fine=events["fine"].dt.strftime("%Y-%m-%d"))
            table = named.merge(table, on="id")
        table.to_csv(args.output_dir / "eventi_noti.csv", index=False, float_format="%.4g")
        print(f"[eventi noti] {args.score}, threshold {limit:.4g} (its highest {args.top_percent:g}%): share of the "
              f"values of every event flagged, against {args.top_percent:g}% by chance")
        show(table)
    if args.striking_dir is not None:
        table = striking_and_events(events, args.striking_dir)
        table.to_csv(args.output_dir / "giorni_ed_eventi.csv", index=False)
        print(f"\n[giorni ed eventi] striking days inside the days and the box of a known event: {len(table)}")
        show(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
