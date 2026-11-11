"""Do the flags of a score fall on cloudy daylight more than chance, and more than those of the other scores?

No reference outside the data. For every score of `--scores` (NAME=DIR[@component], as in the other checks), its
flags are its values above the threshold of its highest `--top-percent` (from the validation years, `score_limit`).
Only daylight values are read: the sun at least `--min-elevation` degrees above the horizon at the timestamp, from
the latitude, the longitude and the day of the year (no equation of time: an error of a few minutes). On one timestamp
every `--time-stride` of the test years, the check counts, over all the daylight values and over the flagged ones:
- overcast: total cloud cover at least `--overcast` (the value itself, from the prepared cache);
- clouds above normal: the standardised cloud cover (the input of the detector) at least `--departure`;
- dim: the standardised solar radiation at the surface at most -`--departure`;
- dim and overcast: both of the first and the third.
The share of each among the flagged values, divided by its share among all the daylight values, is the lift: 1 when
the flags fall on it as by chance. The recall is the share of the values of each kind that are flagged, against the
share of daylight values flagged. Cloud cover and radiation are inputs of every score: the check compares the scores
with each other on the same data, it does not test any of them against an outside truth.

    python scripts/check_era5_cloud_flags.py --prepared-dir data/era5_prepared \\
        --inputs-run outputs/stgan_optuna27_scored \\
        --scores Gn=outputs/optuna27/test_generator_clim_day normale=outputs/confronti_good2/climatology/test/normal \\
            3ore=outputs/confronti_good2/climatology/test/change_3h ... --csv outputs/nubi_segnalazioni.csv
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

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from physiq_pv.era5.data import load_prepared
from scripts.check_era5_dust_lead import score_limit
from scripts.check_era5_event_zones import inputs_of, open_score
from scripts.check_era5_score_correlation import parse_score

CLOUD, RADIATION = "total_cloud_cover", "surface_solar_radiation_downwards"
KINDS = ("overcast", "clouds above normal", "dim", "dim and overcast")
BLOCK = 128  # sampled timestamps read at a time


def solar_elevation(timestamps, latitudes, longitudes):
    """Elevation of the sun in degrees [timestamps, cells]: declination from the day of the year, hour angle from the
    UTC time and the longitude (local mean solar time)."""
    timestamps = pd.DatetimeIndex(timestamps)
    day = timestamps.dayofyear.to_numpy(dtype=np.float64)
    hours = (timestamps.hour + timestamps.minute / 60).to_numpy(dtype=np.float64)
    declination = np.radians(23.44) * np.sin(2 * np.pi * (284 + day) / 365)
    solar_time = hours[:, None] + np.asarray(longitudes, dtype=np.float64)[None, :] / 15
    angle = np.radians(15 * (solar_time - 12))
    latitude = np.radians(np.asarray(latitudes, dtype=np.float64))[None, :]
    sine = (np.sin(latitude) * np.sin(declination)[:, None]
            + np.cos(latitude) * np.cos(declination)[:, None] * np.cos(angle))
    return np.degrees(np.arcsin(np.clip(sine, -1, 1)))


def kinds_of(cloud, cloud_departure, radiation_departure, overcast, departure):
    """[kinds, ...] boolean: overcast, clouds above normal, dim, dim and overcast."""
    is_overcast = cloud >= overcast
    dim = radiation_departure <= -departure
    return np.stack((is_overcast, cloud_departure >= departure, dim, dim & is_overcast))


def count(kinds, daylight, flagged):
    """Counts over the daylight values: all, flagged, of each kind, of each kind and flagged."""
    flagged = flagged & daylight
    inside = kinds & daylight[None]
    return (int(daylight.sum()), int(flagged.sum()), inside.reshape(len(kinds), -1).sum(axis=1),
            (inside & flagged[None]).reshape(len(kinds), -1).sum(axis=1))


def table(totals):
    """One row per score: share of daylight flagged, and for every kind its share overall, among the flags, the lift
    and the recall."""
    rows = []
    for name, (values, flags, of_kind, of_kind_flagged) in totals.items():
        row = {"score": name, "daylight values": values, "flagged %": 100 * flags / values}
        for index, kind in enumerate(KINDS):
            overall = of_kind[index] / values
            among = of_kind_flagged[index] / flags if flags else np.nan
            row[f"{kind} % all"] = 100 * overall
            row[f"{kind} % flagged"] = 100 * among
            row[f"{kind} lift"] = among / overall if overall else np.nan
            row[f"{kind} recall %"] = 100 * of_kind_flagged[index] / of_kind[index] if of_kind[index] else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("score")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True, help="Prepared ERA5 cache: the cloud cover itself")
    parser.add_argument("--inputs-run", type=Path, required=True,
                        help="Scored run of the detector: its shared cache gives the standardised inputs")
    parser.add_argument("--scores", nargs="+", required=True, help="NAME=DIR[@component]")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--time-stride", type=int, default=9, help="Odd, so that every time of day is read")
    parser.add_argument("--min-elevation", type=float, default=15.0, help="Degrees of the sun above the horizon")
    parser.add_argument("--overcast", type=float, default=0.8, help="Total cloud cover of an overcast sky (0-1)")
    parser.add_argument("--departure", type=float, default=1.0, help="Standardised departure from normal")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    if args.time_stride < 1 or args.time_stride % 2 == 0:
        raise ValueError("time-stride must be odd")
    sources = [parse_score(text) for text in args.scores]
    if len({name for name, _, _ in sources}) != len(sources):
        raise ValueError("Give every score a different name")

    opened, maps, limits = {}, [], {}
    grid, timestamps = None, None
    cubes = inputs = None
    try:
        for source in sources:
            manifest, array, more = open_score(source)
            maps.extend(more)
            times = pd.DatetimeIndex(np.load(source[1] / "test_timestamps.npy"))
            if grid is None:
                grid, timestamps = manifest["grid"], times
            elif manifest["grid"] != grid or not times.equals(timestamps):
                raise ValueError(f"{source[0]} is not over the grid and timestamps of {sources[0][0]}")
            opened[source[0]] = array
            limits[source[0]] = score_limit(array, args.top_percent)
            release_pages(*getattr(array, "parts", (array,)))
        rows, cols = np.asarray(grid["rows"]), np.asarray(grid["cols"])
        latitudes = np.asarray(grid["latitudes"], dtype=np.float64)[rows]
        longitudes = np.asarray(grid["longitudes"], dtype=np.float64)[cols]

        cubes = load_prepared(args.prepared_dir)[0]
        if cubes.test.shape[1] != len(rows):
            raise ValueError("The prepared cache is not over the cells of the scores")
        raw_rows = cubes.test_timestamps.get_indexer(timestamps)
        if (raw_rows < 0).any():
            raise ValueError("The prepared test years do not cover the timestamps of the scores")
        inputs, first, names = inputs_of(args.inputs_run, timestamps)
        if inputs.shape[1] != len(rows):
            raise ValueError("The inputs run is not over the cells of the scores")
        cloud_raw = list(cubes.feature_names).index(CLOUD)
        cloud_std, radiation_std = names.index(CLOUD), names.index(RADIATION)

        sampled = np.arange(0, len(timestamps), args.time_stride)
        totals = {name: [0, 0, np.zeros(len(KINDS), dtype=np.int64), np.zeros(len(KINDS), dtype=np.int64)]
                  for name in opened}
        for start in range(0, len(sampled), BLOCK):
            steps = sampled[start:start + BLOCK]
            daylight = solar_elevation(timestamps[steps], latitudes, longitudes) >= args.min_elevation
            cloud = np.asarray(cubes.test[raw_rows[steps], :, cloud_raw], dtype=np.float64)
            standardised = np.asarray(inputs[first + steps], dtype=np.float64)
            kinds = kinds_of(cloud, standardised[:, :, cloud_std], standardised[:, :, radiation_std],
                             args.overcast, args.departure)
            del standardised
            for name, array in opened.items():
                with np.errstate(invalid="ignore"):
                    values = np.asarray(array[steps], dtype=np.float64)
                    flagged = values > limits[name]
                valid = daylight & np.isfinite(values)
                part = count(kinds, valid, flagged)
                total = totals[name]
                total[0] += part[0]
                total[1] += part[1]
                total[2] += part[2]
                total[3] += part[3]
                release_pages(*getattr(array, "parts", (array,)))
            release_pages(inputs)
            print(f"[clouds] {min(start + BLOCK, len(sampled))}/{len(sampled)} timestamps", flush=True)
    finally:
        for array in maps:
            array._mmap.close()
        if inputs is not None:
            inputs._mmap.close()

    result = table(totals)
    print(f"[clouds] {timestamps[0].date()} to {timestamps[-1].date()}, one timestamp every {args.time_stride}, "
          f"daylight: sun at least {args.min_elevation:g} degrees up | flags: each score's highest "
          f"{args.top_percent:g}% (threshold from the validation years) | overcast: cloud cover >= {args.overcast:g}; "
          f"above normal / dim: standardised cloud cover >= {args.departure:g} / radiation <= -{args.departure:g} | "
          f"lift: share among the flags / share among all daylight values (1 = chance)")
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.2f}".format):
        print(result[["daylight values", "flagged %"]])
        for kind in KINDS:
            print(f"\n{kind}")
            print(result[[f"{kind} % all", f"{kind} % flagged", f"{kind} lift", f"{kind} recall %"]])
    if args.csv:
        result.to_csv(args.csv)
        args.csv.with_suffix(".json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
