"""Does the anomaly score rise before Saharan dust reaches Europe? A check on independent data.

ERA5 has no dust variable, so a detector trained on it can only react to the weather that carries
the dust. This script takes the dust episodes from the CAMS global reanalysis (EAC4, dust aerosol
optical depth at 550 nm) and compares the score before each of them with the score on other days.

The rule is fixed before looking at the data:
- episode: the regional mean dust of a day is in its highest 1%; the first such day is the arrival,
  and arrivals closer than 7 days to the previous dusty day belong to the same episode;
- statistic: the highest daily share of cells above the score threshold in the 3 days before the
  arrival;
- dust amount: episodes are also split into a dustier and a less dusty half by their highest
  regional dust value, to see whether more dust goes with a higher score;
- comparison: the same statistic on days of the same calendar month, at least 10 days away from
  every dusty day. The p-value is the share of random draws of such days, one per episode, whose
  mean statistic reaches the observed one.

`--window 0 0` reads the score on the arrival day instead of before it, and `--analysis-years`
keeps a part of the period: a rule suggested by some years can then be checked on the others.

`--score-component generator` or `discriminator` reads one of the two parts of the score of a
training run instead of their sum: the reconstruction error or the discriminator part.

With `--prepared-dir` the check also asks why some episodes are flagged and others are not: it
measures how fast the circulation changes on the same days, as the difference of sea level pressure
and 10 m wind from 24 hours before, and compares the episodes with a stronger and a weaker change.

With `--score-region source` the score is read where the dust comes from instead of where it
arrives: the weather system that lifts and carries the dust is over northern Africa in the days
before, so each receptor region is paired with the part of northern Africa upwind of it. The dust
episodes stay those of the receptor region.

Missing years are downloaded from the Atmosphere Data Store, one request per year; a rerun skips
the years already saved. Credentials: ~/.adsapirc with `url: https://ads.atmosphere.copernicus.eu/api`
and the same `key` as ~/.cdsapirc, and the EAC4 licence accepted on the ADS website.

    python scripts/check_era5_dust_lead.py \
        --run-dir outputs/stgan_gatgru_recent16_cyclic_bf16_clim --dust-dir data/cams_dust
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd

DATASET = "cams-global-reanalysis-eac4"
VARIABLE = "dust_aerosol_optical_depth_550nm"
# The ADS serves this dataset in one of these; the first accepted one is kept for the next years.
DATA_FORMATS = ("netcdf_zip", "netcdf")
# name, south, north, west, east: receptor regions that do not overlap.
REGIONS = (("Iberian Peninsula", 36, 44, -10, 3), ("France", 44.5, 49, -2, 6.5), ("Italy", 37, 46, 7, 19),
           ("Greece and Balkans", 35, 45, 19.5, 29), ("Central Europe", 47, 54, 7, 20))
# Northern Africa upwind of each receptor region: name, south, north, west, east.
NORTHWEST_AFRICA = ("Northwest Africa", 27, 36, -10, 3)
CENTRAL_NORTH_AFRICA = ("Central North Africa", 27, 37, 3.5, 17)
NORTHEAST_AFRICA = ("Northeast Africa", 25, 33, 17.5, 30)
SOURCES = {"Iberian Peninsula": NORTHWEST_AFRICA, "France": NORTHWEST_AFRICA, "Italy": CENTRAL_NORTH_AFRICA,
           "Greece and Balkans": NORTHEAST_AFRICA, "Central Europe": CENTRAL_NORTH_AFRICA}
LAGS = tuple(range(-5, 4))
THRESHOLD_STRIDE = 10  # the score threshold is a percentile of every tenth timestamp
COMPONENTS = ("combined", "generator", "discriminator")
# The surface circulation: the variables whose 24-hour change measures how fast the weather turns.
CIRCULATION_FEATURES = ("mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind")
CHANGE_CHUNK = 256  # timestamps of the prepared test cube read at a time


def score_file(manifest, component):
    """Path, relative to the run, of the mean score of `component` in a run manifest."""
    if component == "combined":
        return manifest.get("anomaly_mean_file", manifest.get("scores_file", "scores/anomaly_mean.npy"))
    file = manifest.get(f"{component}_mean_file")
    if file is None:
        raise ValueError(f"Run does not contain {component} component scores")
    return file


def daily_means(path):
    """Daily mean fields of one downloaded file: (values [days, lat, lon], days, latitudes, longitudes).

    The ADS delivers a NetCDF or a zip holding one, with the time either as a `time` dimension or
    as `valid_time` over (forecast_period, forecast_reference_time).
    """
    import xarray as xr
    with tempfile.TemporaryDirectory() as temp:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                path = archive.extract(next(name for name in archive.namelist() if name.endswith(".nc")), temp)
        with xr.open_dataset(path) as dataset:
            field = dataset[next(name for name in dataset.data_vars if dataset[name].ndim >= 3)]
            time_name = "valid_time" if "valid_time" in field.coords else "time"
            field = field.stack(instant=[name for name in field.dims if name not in ("latitude", "longitude")])
            field = field.transpose("instant", "latitude", "longitude")
            days = pd.DatetimeIndex(np.asarray(field[time_name].values).reshape(-1)).normalize()
            values = np.asarray(field.values, dtype=np.float64)
            latitudes = np.asarray(dataset["latitude"].values, dtype=float)
            longitudes = (np.asarray(dataset["longitude"].values, dtype=float) + 180) % 360 - 180
    usable = ~days.isna() & np.isfinite(values).all(axis=(1, 2))
    unique_days, position = np.unique(days[usable], return_inverse=True)
    total = np.zeros((len(unique_days), *values.shape[1:]))
    np.add.at(total, position, values[usable])
    means = total / np.bincount(position)[:, None, None]
    return means.astype(np.float32), unique_days.astype("datetime64[D]"), latitudes, longitudes


def create_client():
    # cdsapi reads url and key itself: the token is never read or printed here.
    os.environ.setdefault("CDSAPI_RC", str(Path.home() / ".adsapirc"))
    try:
        import cdsapi
        return cdsapi.Client()
    except Exception:
        # Do not echo the error: it can contain configuration text.
        raise RuntimeError("Cannot initialise the ADS client. Check url and key in ~/.adsapirc") from None


def download_dust(directory, years, area):
    """Save dust_<year>.npz (daily means) for every year not already in `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    client, formats = None, list(DATA_FORMATS)
    for year in years:
        target, raw = directory / f"dust_{year}.npz", directory / f"raw_{year}.tmp"
        if target.exists():
            print(f"[dust] year={year} already saved", flush=True)
            continue
        client = client or create_client()
        for data_format in list(formats):
            request = {"variable": [VARIABLE], "date": [f"{year}-01-01/{year}-12-31"],
                       "time": [f"{hour:02d}:00" for hour in range(0, 24, 3)], "data_format": data_format,
                       "area": list(area)}
            try:
                client.retrieve(DATASET, request, str(raw))
            except Exception as error:
                print(f"[dust] year={year} data_format={data_format} refused: {str(error)[:400]}", flush=True)
                continue
            formats = [data_format]
            break
        else:
            raise RuntimeError(f"Dust download failed for {year}")
        values, days, latitudes, longitudes = daily_means(raw)
        np.savez(target, dust=values, days=days, latitudes=latitudes, longitudes=longitudes)
        raw.unlink()
        print(f"[dust] year={year} saved: days={len(days)} grid={len(latitudes)}x{len(longitudes)} "
              f"mean_aod={values.mean():.4f}", flush=True)


def load_dust(directory):
    """All saved years as (values [days, lat, lon], days, latitudes, longitudes)."""
    files = sorted(Path(directory).glob("dust_*.npz"))
    if not files:
        raise ValueError(f"No dust_<year>.npz in {directory}")
    years = [np.load(file) for file in files]
    return (np.concatenate([year["dust"] for year in years]),
            pd.DatetimeIndex(np.concatenate([year["days"] for year in years])),
            years[0]["latitudes"], years[0]["longitudes"])


def daily_share_above(scores, days, cells, threshold):
    """Daily percentage of the finite scores of `cells` above `threshold`, indexed by day."""
    values = np.asarray(scores[:, cells], dtype=np.float64)
    finite = np.isfinite(values)
    counts = pd.DataFrame({"above": ((values > threshold) & finite).sum(axis=1), "finite": finite.sum(axis=1)},
                          index=days).groupby(level=0).sum()
    if (counts.finite == 0).any():
        raise ValueError(f"No finite score on {counts.index[counts.finite == 0][0].date()}")
    return 100 * counts.above / counts.finite


def circulation_change(prepared_dir, timestamps, cells_of):
    """Daily circulation change of every region, from the test cube of a prepared ERA5 directory.

    For each of CIRCULATION_FEATURES the absolute difference from the value 24 hours before is
    averaged over the cells of the region and divided by its mean over the whole period, so that
    the variables weigh the same. The result is the mean of the variables, averaged by day; the
    first day has no value 24 hours before and is NaN.
    """
    prepared_dir = Path(prepared_dir)
    features = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))["features"]
    columns = [features.index(name) for name in CIRCULATION_FEATURES]
    if not pd.DatetimeIndex(np.load(prepared_dir / "test_timestamps.npy")).equals(timestamps):
        raise ValueError("The prepared directory and the run do not share the test timestamps")
    steps = int(round(pd.Timedelta(days=1) / (timestamps[1] - timestamps[0])))
    values = np.load(prepared_dir / "test.npy", mmap_mode="r")
    try:
        change = {name: np.full((len(timestamps), len(columns)), np.nan) for name in cells_of}
        for start in range(steps, len(timestamps), CHANGE_CHUNK):
            stop = min(start + CHANGE_CHUNK, len(timestamps))
            block = np.asarray(values[start - steps:stop][:, :, columns], dtype=np.float64)
            difference = np.abs(block[steps:] - block[:-steps])
            for name, cells in cells_of.items():
                change[name][start:stop] = np.nanmean(difference[:, cells], axis=1)
    finally:
        values._mmap.close()
    days = timestamps.normalize()
    return {name: pd.Series((regional / np.nanmean(regional, axis=0)).mean(axis=1), index=days).groupby(level=0).mean()
            for name, regional in change.items()}


def lead_test(share, dust, months, *, dust_quantile=0.99, min_gap_days=7, lead_days=3, window=None,
              exclusion_days=10, draws=2000, rng=None):
    """Compare the score around the dust arrivals with the score around other days of the same month.

    `share`, `dust` and `months` are daily vectors on the same consecutive days. The statistic is
    the highest share over `window`, a (first, last) pair of days from the arrival; by default the
    `lead_days` days before it. Returns the arrival positions, the statistic of every episode, its
    percentile among the comparison days of its month, the p-value and the mean share at each lag
    for episodes and comparison draws.
    """
    rng = rng or np.random.default_rng(0)
    share, dust, months = np.asarray(share, dtype=float), np.asarray(dust, dtype=float), np.asarray(months)
    first, last = window or (-lead_days, -1)
    if first > last:
        raise ValueError("The window must go from an earlier to a later day")
    margin, end_margin = max(-first, -min(LAGS)), max(last, max(LAGS))
    dusty = np.nonzero(dust >= np.quantile(dust, dust_quantile))[0]
    arrivals = np.array([day for index, day in enumerate(dusty)
                         if (index == 0 or day - dusty[index - 1] > min_gap_days)
                         and margin <= day < len(share) - end_margin], dtype=int)
    if not len(arrivals):
        raise ValueError("No dust arrival in the period")
    inner = np.arange(margin, len(share) - end_margin)
    before = np.full(len(share), np.nan)
    before[inner] = np.max([share[inner + lag] for lag in range(first, last + 1)], axis=0)
    free = np.zeros(len(share), dtype=bool)
    free[inner] = True
    for day in dusty:
        free[max(0, day - exclusion_days):day + exclusion_days + 1] = False
    candidates = {month: np.nonzero(free & (months == month))[0] for month in np.unique(months[arrivals])}
    if any(not len(days) for days in candidates.values()):
        raise ValueError("A month with a dust arrival has no comparison day")
    drawn = np.stack([rng.choice(candidates[months[day]], draws) for day in arrivals], axis=1)
    percentiles = np.array([100 * (before[candidates[months[day]]] < before[day]).mean() for day in arrivals])
    observed, null = before[arrivals], before[drawn]
    # Dust amount of an episode: its highest daily value from the arrival to the end of the gap.
    peak = np.array([dust[day:day + min_gap_days + 1].max() for day in arrivals])
    return {"arrivals": arrivals, "observed": observed, "null": null, "percentiles": percentiles, "dust_peak": peak,
            "p_value": float((1 + (null.mean(axis=1) >= observed.mean()).sum()) / (draws + 1)),
            "episode_profile": [float(share[arrivals + lag].mean()) for lag in LAGS],
            "comparison_profile": [float(share[drawn + lag].mean()) for lag in LAGS]}


def parser():
    root = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    root.add_argument("--run-dir", type=Path, required=True, help="Run with scores/anomaly_mean.npy, e.g. a climatology run")
    root.add_argument("--score-component", choices=COMPONENTS, default="combined",
                      help="Score to read: the combined one, or one of its two parts in a training run")
    root.add_argument("--dust-dir", type=Path, required=True, help="Where the yearly dust files are kept")
    root.add_argument("--prepared-dir", type=Path,
                      help="Prepared ERA5 directory of the run: also compare the episodes by how fast the "
                           "circulation changes on the same days")
    root.add_argument("--first-year", type=int, default=2005)
    root.add_argument("--last-year", type=int, default=2024)
    root.add_argument("--score-region", choices=("arrival", "source"), default="arrival",
                      help="Read the score where the dust arrives, or in the part of northern Africa upwind of it")
    root.add_argument("--top-percent", type=float, default=5, help="Score threshold: this highest share of the values")
    root.add_argument("--dust-quantile", type=float, default=0.99)
    root.add_argument("--min-gap-days", type=int, default=7)
    root.add_argument("--lead-days", type=int, default=3)
    root.add_argument("--window", type=int, nargs=2, metavar=("FIRST", "LAST"),
                      help="Days from the arrival over which the score is read, e.g. 0 0 for the arrival day; "
                           "default: the --lead-days days before it")
    root.add_argument("--analysis-years", type=int, nargs=2, metavar=("FIRST", "LAST"),
                      help="Use only these years of the saved dust and scores; default: all the saved years")
    root.add_argument("--exclusion-days", type=int, default=10)
    root.add_argument("--draws", type=int, default=2000)
    root.add_argument("--seed", type=int, default=0)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    file = score_file(manifest, args.score_component)
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
    # North, west, south, east: the domain of the scores.
    download_dust(args.dust_dir, range(args.first_year, args.last_year + 1),
                  [float(latitudes.max()), float(longitudes.min()), float(latitudes.min()), float(longitudes.max())])
    dust, dust_days, dust_latitudes, dust_longitudes = load_dust(args.dust_dir)
    timestamps = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy"))
    days = timestamps.normalize()
    cells_of = {}
    for name, *box in REGIONS:
        south, north, west, east = SOURCES[name][1:] if args.score_region == "source" else box
        cells_of[name] = np.nonzero((latitudes >= south) & (latitudes <= north)
                                    & (longitudes >= west) & (longitudes <= east))[0]
    change = None if args.prepared_dir is None else circulation_change(args.prepared_dir, timestamps, cells_of)
    scores = np.load(args.run_dir / file, mmap_mode="r")
    try:
        threshold = float(np.nanpercentile(np.asarray(scores[::THRESHOLD_STRIDE]), 100 - args.top_percent))
        print(f"[dust-lead] score={args.score_component} ({file}) | score threshold={threshold:.4g} "
              f"(top {args.top_percent:g}%) | dust days={len(dust_days)} "
              f"{dust_days[0].date()} -> {dust_days[-1].date()}")
        first_lag, last_lag = args.window or (-args.lead_days, -1)
        years = "all saved" if args.analysis_years is None else "{}-{}".format(*args.analysis_years)
        print(f"[dust-lead] rule: highest daily share of cells above the threshold from day {first_lag:+d} to day "
              f"{last_lag:+d} from the arrival | analysis years: {years}")
        rng = np.random.default_rng(args.seed)
        observed, null, score_percentiles, change_percentiles = [], [], [], []
        if change is not None:
            print("[dust-lead] circulation change: difference of " + ", ".join(CIRCULATION_FEATURES)
                  + " from 24 hours before, read on the same days and cells as the score")
        for name, south, north, west, east in REGIONS:
            score_name = SOURCES[name][0] if args.score_region == "source" else name
            cells = cells_of[name]
            inside = (((dust_latitudes >= south) & (dust_latitudes <= north))[:, None]
                      & ((dust_longitudes >= west) & (dust_longitudes <= east))[None, :])
            if not len(cells) or not inside.any():
                raise ValueError(f"Region {name} has no cell in the score or dust grid")
            share = daily_share_above(scores, days, cells, threshold)
            regional_dust = pd.Series(dust[:, inside].mean(axis=1), index=dust_days)
            common = share.index.intersection(regional_dust.index)
            if args.analysis_years is not None:
                common = common[(common.year >= args.analysis_years[0]) & (common.year <= args.analysis_years[1])]
            if len(common) < 2 or not (np.diff(common.values) == np.timedelta64(1, "D")).all():
                raise ValueError("Scores and dust must share consecutive days")
            result = lead_test(share[common], regional_dust[common], common.month, dust_quantile=args.dust_quantile,
                               min_gap_days=args.min_gap_days, lead_days=args.lead_days, window=args.window,
                               exclusion_days=args.exclusion_days, draws=args.draws, rng=rng)
            observed.append(result["observed"])
            null.append(result["null"])
            print(f"\n{name} | score read in {score_name} | cells={len(cells)} | dust episodes={len(result['arrivals'])}")
            print(f"statistic: episodes {result['observed'].mean():.1f}% | comparison days {result['null'].mean():.1f}% | "
                  f"mean percentile {result['percentiles'].mean():.0f} (50 expected) | "
                  f"episodes above the 90th percentile {100 * (result['percentiles'] >= 90).mean():.0f}% (10% expected) | "
                  f"p = {result['p_value']:.4f}")
            print(" | ".join(f"episodes above the {level}th percentile {100 * (result['percentiles'] >= level).mean():.0f}% "
                             f"({100 - level}% expected)" for level in (95, 99)))
            if len(result["arrivals"]) >= 4:
                dustier = result["dust_peak"] >= np.median(result["dust_peak"])
                above = result["percentiles"] >= 90
                rank_correlation = pd.Series(result["dust_peak"]).corr(pd.Series(result["percentiles"]), method="spearman")
                print(f"dustier half of the episodes above the 90th percentile {100 * above[dustier].mean():.0f}% | "
                      f"other half {100 * above[~dustier].mean():.0f}% | "
                      f"rank correlation between dust amount and percentile {rank_correlation:+.2f}")
            if change is not None:
                # The same episodes and comparison days, with the circulation change in place of the score.
                measured = lead_test(change[name][common], regional_dust[common], common.month,
                                     dust_quantile=args.dust_quantile, min_gap_days=args.min_gap_days,
                                     lead_days=args.lead_days, window=args.window, exclusion_days=args.exclusion_days,
                                     draws=args.draws, rng=np.random.default_rng(args.seed))
                score_percentiles.append(result["percentiles"])
                change_percentiles.append(measured["percentiles"])
                every_day = pd.concat([share[common], change[name][common]], axis=1).dropna()
                print(f"circulation change on the same days: mean percentile {measured['percentiles'].mean():.0f} "
                      f"(50 expected) | episodes above the 90th percentile "
                      f"{100 * (measured['percentiles'] >= 90).mean():.0f}% (10% expected) | on all days, rank correlation "
                      f"between change and score {every_day.corr(method='spearman').iloc[0, 1]:+.2f}")
                if len(result["arrivals"]) >= 4:
                    stronger = measured["percentiles"] >= np.median(measured["percentiles"])
                    flagged = result["percentiles"] >= 90
                    rank_correlation = pd.Series(measured["percentiles"]).corr(pd.Series(result["percentiles"]),
                                                                              method="spearman")
                    print(f"episodes with the stronger change: score above the 90th percentile "
                          f"{100 * flagged[stronger].mean():.0f}% | other half {100 * flagged[~stronger].mean():.0f}% | "
                          f"rank correlation between change and score percentile {rank_correlation:+.2f}")
            print("days from arrival:    " + " ".join(f"{lag:+5d}" for lag in LAGS))
            print("episodes (%):         " + " ".join(f"{value:5.1f}" for value in result["episode_profile"]))
            print("comparison days (%):  " + " ".join(f"{value:5.1f}" for value in result["comparison_profile"]))
        observed, null = np.concatenate(observed), np.concatenate(null, axis=1)
        p_value = (1 + (null.mean(axis=1) >= observed.mean()).sum()) / (args.draws + 1)
        print(f"\nALL REGIONS | dust episodes={len(observed)} | statistic: episodes {observed.mean():.1f}% | "
              f"comparison days {null.mean():.1f}% | p = {p_value:.4f}")
        print("One dust outbreak can reach several regions, so the overall p-value is optimistic.")
        if change is not None:
            score_percentiles, change_percentiles = np.concatenate(score_percentiles), np.concatenate(change_percentiles)
            print("ALL REGIONS by circulation change of the episode: " + " | ".join(
                f"{label}: {inside.sum()} episodes, score above the 90th percentile "
                + (f"{100 * (score_percentiles[inside] >= 90).mean():.0f}%" if inside.any() else "-")
                for label, inside in (("change below its 50th percentile", change_percentiles < 50),
                                      ("50th to 90th", (change_percentiles >= 50) & (change_percentiles < 90)),
                                      ("above the 90th", change_percentiles >= 90))))
            print(f"rank correlation between change and score percentile over all the episodes "
                  f"{pd.Series(change_percentiles).corr(pd.Series(score_percentiles), method='spearman'):+.2f}")
    finally:
        scores._mmap.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
