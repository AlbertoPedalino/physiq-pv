"""Does one score flag an episode before another, at the same cell?

A value is flagged by a score when it is in its highest `--top-percent`. An episode of a score
starts at a cell when the score flags it after at least `--quiet-hours` without a flag there. For
every start of an episode of the first score, the second score at the same cell is put in one of
five groups:
- already flagging, without a gap up to the start: it saw the episode first, and by how many hours
  (the length of its run of flags before the start);
- earlier with a gap: it flagged within `--window-hours` before, then stopped;
- at the start: both begin on the same timestamp;
- later: it begins within `--window-hours` after the start, and by how many hours;
- not within `--window-hours` on either side.
The same is done the other way round. If the second score anticipates the first, its "already
flagging" share is larger in the first table than the first score's is in the second.

What happens by chance: the same groups at cells and timestamps where the score of the row has been
quiet as long as before a start and does not start (one timestamp every CHANCE_STEP). A share of
the starts well above it is a link between the two scores, not the length of their flags.
`--by-top-percent` flags the second score at its own level, e.g. only its strongest 1%.

    python scripts/check_era5_lead.py --score-dir outputs/run_gen_clim --by-dir outputs/run_no_model_bal_clim
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd

from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, score_file, score_limit

CHUNK = 512  # timestamps read at a time
CHANCE_STEP = 29  # the chance row reads one timestamp every this many
LEAD_BANDS = ((3, 3), (6, 6), (9, 12), (15, 24), (27, None))  # hours already flagging before the start
NAME, WIDTH = 34, 11


def flags(path, top_percent):
    """Flags [time, cell] of the highest `top_percent` of the score of a run, its timestamps and limit."""
    manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    scores = np.load(path / score_file(manifest, "combined"), mmap_mode="r")
    try:
        limit = score_limit(scores, top_percent)
        flagged = np.zeros(scores.shape, dtype=np.bool_)
        with np.errstate(invalid="ignore"):
            for start in range(0, len(scores), CHUNK):
                flagged[start:start + CHUNK] = np.asarray(scores[start:start + CHUNK]) > limit  # NaN is not flagged
    finally:
        scores._mmap.close()
    return flagged, pd.DatetimeIndex(np.load(path / "test_timestamps.npy")), limit


def starts_and_other(first, second, quiet, window, chance=False):
    """For every start of an episode of `first` [time, cell]: the run of `second` flags ending just before it
    (steps), the steps since the last flag of `second` before it, and the steps to the next flag of `second`
    from it (0: at the start). Steps beyond `window` are reported as window + 1. With `chance`, the same at
    the quiet cells where `first` does not start, one timestamp every CHANCE_STEP."""
    times, cells = first.shape
    big = np.int32(window + 1)
    run = np.zeros(cells, dtype=np.int32)    # consecutive flags of the second, ending at the step before
    since = np.full(cells, big)              # steps since the last flag of the second, before this step
    silent = np.full(cells, np.int32(quiet))  # steps without a flag of the first, before this step
    positions, before, last = [], [], []
    for step in range(times):
        quiet_now = silent >= quiet
        start = (~first[step] if chance else first[step]) & quiet_now
        if step == 0 or (chance and step % CHANCE_STEP):
            start[:] = False  # what came before the first timestamp is unknown; chance: not a sampled step
        where = np.nonzero(start)[0]
        positions.append(where)
        before.append(run[where].copy())
        last.append(since[where].copy())
        now = second[step]
        run = np.where(now, run + 1, 0).astype(np.int32)
        since = np.where(now, 1, np.minimum(since + 1, big)).astype(np.int32)
        silent = np.where(first[step], 0, np.minimum(silent + 1, quiet)).astype(np.int32)
    # Backwards: steps to the next flag of the second, the step itself included.
    after = [None] * times
    until = np.full(cells, big)
    for step in range(times - 1, -1, -1):
        until = np.where(second[step], 0, np.minimum(until + 1, big)).astype(np.int32)
        after[step] = until[positions[step]].copy()
    return (np.concatenate(before), np.concatenate(last), np.concatenate(after),
            sum(len(where) for where in positions))


def table(before, last, after, window, step_hours):
    """Shares (%) of the starts in each group, the lead bands within 'already flagging', the median lag."""
    count = len(before)
    already = before > 0
    gap = ~already & (last <= window)
    together = ~already & ~gap & (after == 0)
    later = ~already & ~gap & (after > 0) & (after <= window)
    share = lambda mask: 100 * mask.sum() / max(count, 1)
    hours = before * step_hours
    bands = [share(already & (hours >= low) & ((hours <= high) if high is not None else True)) for low, high in LEAD_BANDS]
    lag = float(np.median(after[later]) * step_hours) if later.any() else float("nan")
    lead = float(np.median(hours[already])) if already.any() else float("nan")
    return {"starts": count, "already": share(already), "bands": bands, "lead_median_h": lead, "gap": share(gap),
            "together": share(together), "later": share(later), "lag_median_h": lag,
            "none": share(~already & ~gap & ~together & ~later)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, required=True, help="Run with the second score")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the second score; default --top-percent")
    parser.add_argument("--quiet-hours", type=int, default=24, help="Hours without a flag before a start")
    parser.add_argument("--window-hours", type=int, default=24, help="Hours on each side that are looked at")
    args = parser.parse_args(argv)
    first, timestamps, first_limit = flags(args.score_dir, args.top_percent)
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    second, other_timestamps, second_limit = flags(args.by_dir, by_top)
    if not timestamps.equals(other_timestamps) or first.shape != second.shape:
        raise ValueError("The two scores are not over the same timestamps and cells")
    step_hours = int(round((timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)))
    if args.quiet_hours % step_hours or args.window_hours % step_hours:
        raise ValueError(f"quiet-hours and window-hours must be multiples of {step_hours}")
    quiet, window = args.quiet_hours // step_hours, args.window_hours // step_hours
    names = [path.name for path in (args.score_dir, args.by_dir)]
    common = len(os.path.commonprefix(names))
    short = [name[common:] or name for name in names]
    results = {}
    for index, (one, other) in enumerate(((first, second), (second, first))):
        for chance in (False, True):
            before, last, after, _ = starts_and_other(one, other, quiet, window, chance=chance)
            results[index, chance] = table(before, last, after, window, step_hours)
    print(f"[lead] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest {args.top_percent:g}% of "
          f"{short[0]}, {by_top:g}% of {short[1]} | start of an episode: a flag after {args.quiet_hours} h without one "
          f"at the cell | looked at: {args.window_hours} h on each side, same cell")
    print(f"flagged share by chance: {short[0]} {100 * first.mean():.1f}%, {short[1]} {100 * second.mean():.1f}%")
    columns = ["starts", "already", *[f"{low}-{high}h" if high and high != low else (f"{low}h" if high else f">{low - 3}h")
                                      for low, high in LEAD_BANDS], "lead med h", "gap", "together", "later",
               "lag med h", "none"]
    print("\n% of the starts of the episodes of the score in the row, by what the other score does at the same cell")
    print(f"{'':<{NAME}}" + "".join(f"{name:>{WIDTH}}" for name in columns))
    for index in (0, 1):
        for chance in (False, True):
            row = results[index, chance]
            values = [row["starts"], row["already"], *row["bands"], row["lead_median_h"], row["gap"], row["together"],
                      row["later"], row["lag_median_h"], row["none"]]
            label = (f"  chance, quiet {short[index]}" if chance else f"starts of {short[index]}, other {short[1 - index]}")
            print(f"{label[:NAME - 1]:<{NAME}}" + f"{values[0]:>{WIDTH}d}"
                  + "".join(f"{value:>{WIDTH}.1f}" for value in values[1:]))
    print("already: the other flags without a gap up to the start (it saw it first; bands by how long); gap: it "
          "flagged within the window before, then stopped; together: it begins on the same timestamp; later: it "
          "begins within the window after; none: not within the window; chance: the same where the score of the "
          "row was quiet as long and did not start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
