"""How often a score that looks 3 h ahead (P_n) is right, verified as a yes/no forecast against the concordant reference.

P_n at its valid timestamp T (shift_era5_forecast_to_valid_time.py) was known at T - 3 h. It says "at T this value will be
unusual" when it is above its threshold. The event to forecast is the reference of the 8 baselines without P_n: the
values of T that at least `--votes` of them flag after their closing (the flags written by build_era5_concordant_cube.py
in CUBE/flags, the cube without the vote of P_n; the same reference of the control of G_n). Every forecast is known at
T - 3 h and uses the flags as they are, with no closing (a closing would read T):
- the forecast score: its flag at T (`--forecast NAME=DIR`, a run at its valid timestamps);
- persistence of the reference: "unusual at T where the baselines agree at T - 3 h" (at least `--votes` of their flags
  without closing at T - 3 h, read from the runs the cube was built from);
- persistence of another score: its flag at T - 3 h (`--persistence NAME=DIR`, e.g. G_n).
The baselines are the detectors of the cube (votes/metadata.json) other than the forecast score and the persistence
scores, read from the same runs and components and with the same thresholds as the cube; the forecast score must be in the
cube with the same run, so its threshold is that of the cube too. All are counted on the same values: T and T - 3 h
consecutive, every run scores both, the reference is defined at T.

For each forecast, the contingency table against the reference (hits, misses, false alarms, correct negatives) and its
measures: POD = hits / (hits + misses) and FAR = false alarms / (hits + false alarms), as Bourdin et al. 2022 (q_r and
q_s of the thesis); also CSI = hits / (hits + misses + false alarms) and the bias (hits + false alarms) / (hits + misses);
and the POD of the onsets, the values of the reference at T that are not in it at T - 3 h (persistence of the reference
cannot get them by construction).

Rule fixed on 17 November 2026, before it was run: the forecast score "forecasts" if its POD is higher AND its FAR lower
than those of the best persistence (the one with the highest CSI).

    python scripts/check_era5_forecast_verification.py --cube outputs/concordate_5pc_Pn \\
        --forecast Pn=outputs/optuna27/previsione_mc_valida/climatology/test/P \\
        --persistence Gn=outputs/optuna27/test_generator_clim_day
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path.cwd()))  # after the copy next to the script: a copy run from the checkout imports itself
import numpy as np
import pandas as pd

from scripts.check_era5_dust_lead import score_file, score_limit
from scripts.check_era5_score_correlation import parse_score

CHUNK = 256
FIELDS = ("hits", "misses", "false_alarms", "correct_negatives", "onsets", "onset_hits")


def tally(observed_now, observed_before, forecast, usable):
    """Counts of one forecast on the usable values of a block (all boolean arrays of the same shape)."""
    f, o = forecast & usable, observed_now & usable
    onset = o & ~observed_before
    return np.array([(f & o).sum(), (~f & o).sum(), (f & ~o & usable).sum(), (~f & ~o & usable).sum(),
                     onset.sum(), (f & onset).sum()], dtype=np.int64)


def measures(counts):
    hits, misses, false_alarms, negatives, onsets, onset_hits = (int(value) for value in counts)
    ratio = lambda a, b: a / b if b else float("nan")
    total = hits + misses + false_alarms + negatives
    return {"flagged %": 100 * ratio(hits + false_alarms, total), "POD": ratio(hits, hits + misses),
            "FAR": ratio(false_alarms, hits + false_alarms), "CSI": ratio(hits, hits + misses + false_alarms),
            "bias": ratio(hits + false_alarms, hits + misses), "POD onsets": ratio(onset_hits, onsets),
            "hits": hits, "misses": misses, "false alarms": false_alarms}


def verdict(table, forecast):
    """The rule fixed before the run: POD higher and FAR lower than the persistence with the highest CSI."""
    persistences = table.drop(index=forecast)
    best = persistences["CSI"].idxmax()
    better = table.loc[forecast, "POD"] > table.loc[best, "POD"] and table.loc[forecast, "FAR"] < table.loc[best, "FAR"]
    return best, bool(better)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cube", type=Path, required=True, help="Cube of build_era5_concordant_cube.py (its flags/)")
    parser.add_argument("--forecast", required=True, help="NAME=DIR of the score at its valid timestamps")
    parser.add_argument("--persistence", nargs="*", default=[], help="NAME=DIR of scores used as persistence")
    parser.add_argument("--votes", type=int, default=5, help="Baselines needed for the reference")
    args = parser.parse_args(argv)
    forecast_name, forecast_dir, forecast_component = parse_score(args.forecast)
    persistence = [parse_score(text) for text in args.persistence]
    vote = json.loads((args.cube / "votes" / "metadata.json").read_text(encoding="utf-8"))["vote"]
    detectors = {name: parse_score(f"{name}={source}") for name, source in vote["detectors"].items()}
    if forecast_name not in detectors:
        raise ValueError(f"{forecast_name} is not a detector of the cube")
    if (detectors[forecast_name][1], detectors[forecast_name][2]) != (forecast_dir, forecast_component):
        raise ValueError(f"{forecast_name} is not the run of the cube: {vote['detectors'][forecast_name]}")
    left_out = {forecast_name, *(name for name, _, _ in persistence)}
    baselines = [source for name, source in detectors.items() if name not in left_out]
    if not 1 <= args.votes <= len(baselines):
        raise ValueError(f"votes must be between 1 and {len(baselines)}")
    timestamps = pd.DatetimeIndex(np.load(args.cube / "votes" / "test_timestamps.npy"))
    times = len(timestamps)
    sources = [detectors[forecast_name], *persistence, *baselines]
    arrays, limits = {}, {}
    for name, directory, component in sources:
        manifest = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if not timestamps.equals(pd.DatetimeIndex(np.load(directory / "test_timestamps.npy"))):
            raise ValueError(f"{name} is not over the timestamps of the cube")
        arrays[name] = np.load(directory / score_file(manifest, component), mmap_mode="r")
        limits[name] = (float(vote["thresholds"][name]) if name in detectors
                        else score_limit(arrays[name], vote["top_percent"]))
    cube_values = np.load(args.cube / "votes" / "scores" / "anomaly_mean.npy", mmap_mode="r")
    cells = cube_values.shape[1]
    packed = {name: np.load(args.cube / "flags" / f"{name}.npy", mmap_mode="r") for name, _, _ in baselines}
    step = np.median(np.diff(timestamps.asi8))
    consecutive = np.zeros(times, dtype=bool)
    consecutive[1:] = np.diff(timestamps.asi8) == step  # T - 3 h is the timestamp before
    print(f"[verify] {times} timestamps, {cells} cells; reference: at least {args.votes} of "
          f"{', '.join(name for name, _, _ in baselines)} after their closing; forecast {forecast_name} at its valid "
          f"timestamp; thresholds " + ", ".join(f"{name} {value:.4g}" for name, value in limits.items()), flush=True)

    names = [forecast_name, "reference_persistence", *[f"{name}_persistence" for name, _, _ in persistence]]
    counts = {name: np.zeros(len(FIELDS), dtype=np.int64) for name in names}
    with np.errstate(invalid="ignore"):
        for start in range(1, times, CHUNK):
            stop = min(start + CHUNK, times)
            span = slice(start - 1, stop)  # one timestamp before the block, for T - 3 h
            raw = {name: np.asarray(arrays[name][span], dtype=np.float64) for name in arrays}
            flags = {name: raw[name] > limits[name] for name in raw}
            finite = np.isfinite(np.asarray(cube_values[span]))
            for name in raw:
                finite &= np.isfinite(raw[name])
            closed = sum(np.unpackbits(np.asarray(packed[name][span]), axis=1, count=cells).astype(np.int16)
                         for name, _, _ in baselines)
            reference = closed >= args.votes
            agree = sum(flags[name].astype(np.int16) for name, _, _ in baselines) >= args.votes
            usable = finite[1:] & finite[:-1] & consecutive[start:stop, None]
            now, before = reference[1:], reference[:-1]
            counts[forecast_name] += tally(now, before, flags[forecast_name][1:], usable)
            counts["reference_persistence"] += tally(now, before, agree[:-1], usable)
            for name, _, _ in persistence:
                counts[f"{name}_persistence"] += tally(now, before, flags[name][:-1], usable)
    for array in [*arrays.values(), cube_values, *packed.values()]:
        array._mmap.close()

    table = pd.DataFrame({name: measures(value) for name, value in counts.items()}).T
    print("\n[verify] each forecast against the reference at T, all known at T - 3 h")
    print(table.to_string(float_format=lambda value: f"{value:.3f}"))
    best, better = verdict(table, forecast_name)
    print(f"\n[verify] rule fixed before the run: {forecast_name} forecasts if its POD is higher and its FAR lower than "
          f"those of the best persistence ({best}, highest CSI) -> {'yes' if better else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
