"""Try a score that divides the deviation of each variable by the spread of the dropout draws.

An exploration, to decide whether a complete scoring is worth it. The rebalanced score compares the
error of a variable with all its errors (`rarity`). Here the deviation of the observation from the
mean reconstruction is divided by the spread of the draws in that cell at that time: how many
"bands" the observation is away, the same unit for every variable. Two versions are tried:

- raw: the mean over the variables of (deviation / spread) squared;
- calibrated: the same, with each variable divided by its mean on ordinary days, because the band
  of some variables is too narrow everywhere and would otherwise make most of the score.

Nothing is scored for the whole period. The scores are computed with an epoch checkpoint on

- a sample of ordinary days, evenly spaced over the period: the days to compare with;
- the documented days of 2019 over Piedmont and over France and northern Italy;
- the arrivals of the documented dust events (see `check_era5_dust_events.py`), in the arrival
  region on the arrival day and upwind on the day before and on that day.

For every such day and place the daily share of cells above the 95th percentile of the ordinary
days is turned into the share of ordinary days below it, for the three scores on the same draws.
The ordinary days are of every month, so the percentiles are not those of the complete scoring.

    python scripts/try_era5_stgan_uncertainty_score.py --prepared-dir data/era5_prepared \
        --run-dir outputs/run --epoch 6 --dust-dir data/cams_dust
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

import physiq_pv.anomaly_detection.stgan as stgan
from physiq_pv.anomaly_detection.stgan.precision import autocast_context
from physiq_pv.anomaly_detection.stgan.scoring import scoring_mode
from physiq_pv.era5.data import load_prepared
from scripts import analyze_era5_stgan_checkpoints as checkpoints
from scripts import check_era5_dust_events as events
from scripts import rescore_era5_stgan_rebalanced as rescore
from scripts.check_era5_dust_lead import load_dust

SCORES = ("rarity", "deviation / spread", "deviation / spread, calibrated")
KNOWN_ZONES = {"Piedmont": (44, 46.5, 6.5, 9.5), "France and northern Italy": (43, 49, 0, 12)}
KNOWN_DAYS = ("2019-01-30", "2019-01-31", "2019-02-01", "2019-02-02", "2019-02-03", "2019-04-22", "2019-04-23",
              "2019-04-24", "2019-04-25", "2019-06-25", "2019-06-26", "2019-06-27", "2019-06-28", "2019-06-29",
              "2019-06-30", "2019-07-15", "2019-07-25", "2019-10-19", "2019-10-20", "2019-10-21", "2019-10-22",
              "2019-10-23", "2019-11-15", "2019-11-22", "2019-11-23", "2019-11-24")
ALARM = 95
FLOOR = 0.1  # the spread of a variable is never taken below this share of its mean spread
PILOT_TIMESTAMPS = 80  # ordinary timestamps on which the floor and the calibration are fitted
DAY = pd.Timedelta(days=1)


def draws_of(model, batch, mc_samples, device, precision):
    """Observed values [cells, variables] and the dropout draws of their reconstruction [draws, cells, variables]."""
    recent, trend, mask, calendar, observed = (values.to(device) for values in batch[:5])
    draws = []
    for _ in range(mc_samples):
        with autocast_context(precision, device):
            cells, generated = model.reconstructed_cells(recent, trend, mask, calendar, observed)
        draws.append(generated.float())
    return cells.float(), torch.stack(draws)


def standardised(cells, draws, floor=None):
    """(deviation / spread) squared [cells, variables]; with `floor`, the spread is not taken below it."""
    spread = draws.std(dim=0, correction=0)
    if floor is not None:
        spread = torch.maximum(spread, floor)
    return ((cells - draws.mean(dim=0)) / spread).square()


def three_scores(cells, draws, *, columns, fitted, floor, calibration):
    """The three scores of SCORES for every cell, as [3, cells]."""
    errors = (draws - cells).square().mean(dim=0)[:, columns]
    squared = standardised(cells, draws, floor)
    return torch.stack((rescore.transform(errors, "rarity", fitted).mean(dim=1), squared.double().mean(dim=1),
                        (squared.double() / calibration).mean(dim=1)))


def fit_pilot(model, dataset, positions, *, mc_samples, device, precision):
    """Floor of the spread and mean of (deviation / spread) squared of every variable, on ordinary timestamps;
    and the share of every variable in the highest 1% of the raw and of the calibrated score."""
    spreads, kept = [], []
    with scoring_mode(model, True), torch.no_grad():
        for position in positions:
            cells, draws = draws_of(model, dataset.fetch_batch([int(position)]), mc_samples, device, precision)
            spreads.append(draws.std(dim=0, correction=0).mean(dim=0))
            kept.append((cells.cpu(), draws.cpu()))
    floor = FLOOR * torch.stack(spreads).mean(dim=0)
    squared = torch.cat([standardised(cells.to(device), draws.to(device), floor).cpu() for cells, draws in kept]).double()
    calibration = squared.mean(dim=0)
    shares = []
    for values in (squared, squared / calibration):
        score = values.mean(dim=1)
        top = values[score >= torch.quantile(score, 0.99)].sum(dim=0)
        shares.append((top / top.sum()).numpy())
    return floor, calibration.to(device), calibration.cpu().numpy(), shares


def choose_days(all_days, target_days, count):
    """Ordinary days, evenly spaced over the period and in pairs: about `count` / 2 days, none of them a
    target day, and the day before each of them, which the reading upwind of an arrival needs.

    Returns the days with their day before in the period, and all the ordinary days.
    """
    free = all_days[~all_days.isin(target_days) & (all_days - DAY).isin(all_days)]
    if not len(free):
        raise ValueError("The period has no ordinary day with a day before it")
    anchors = free[np.unique(np.linspace(0, len(free) - 1, min(max(count // 2, 1), len(free))).astype(int))]
    before = (anchors - DAY)
    return anchors, anchors.union(before[~before.isin(target_days)])


def dust_arrivals(dust_dir, first_day, last_day, level):
    """(region, arrival day) of the documented dust events with a dust day inside the scored period."""
    dust = events.dust_percentiles(*load_dust(dust_dir))
    last_dust_day = next(iter(dust.values())).index[-1]
    found = []
    for start, end, regions in events.EVENTS:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if not regions or end > min(last_day, last_dust_day) or start - events.LEAD_DAYS * DAY < first_day:
            continue
        for region in regions:
            day, _ = events.arrival(dust[region], start, end, level)
            if day is not None:
                found.append((region, day))
    return found


def percentile_among(value, ordinary):
    """Share of the ordinary days whose statistic is below `value`, in percent."""
    return 100 * float((np.asarray(ordinary) < value).mean())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run with model_epoch_<n>.pt and scores/feature_scores.npy")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--dust-dir", type=Path, help="Yearly dust files; without them only the days of 2019 are read")
    parser.add_argument("--dust-percentile", type=float, default=95)
    parser.add_argument("--ordinary-days", type=int, default=150)
    parser.add_argument("--scale-stride", type=int, default=10)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
    boxes = {**events.BOXES, **KNOWN_ZONES}
    cells_of = {name: np.nonzero((latitudes >= south) & (latitudes <= north) & (longitudes >= west) & (longitudes <= east))[0]
                for name, (south, north, west, east) in boxes.items()}
    cubes = load_prepared(args.prepared_dir)[0]
    try:
        names = list(cubes.feature_names)
        timestamps = cubes.test_timestamps
        all_days = timestamps.normalize().unique()
        known = [day for day in pd.DatetimeIndex(KNOWN_DAYS) if day in all_days]
        arrivals = [] if args.dust_dir is None else dust_arrivals(args.dust_dir, all_days[0], all_days[-1], args.dust_percentile)
        target_days = pd.DatetimeIndex(sorted({*known, *(day for _, day in arrivals), *(day - DAY for _, day in arrivals)}))
        anchors, ordinary = choose_days(all_days, target_days, args.ordinary_days)
        needed = ordinary.union(anchors - DAY).union(target_days)
        positions = np.flatnonzero(timestamps.normalize().isin(needed))
        columns, fitted = rescore.fitted_scales(args.run_dir, "rarity", None, args.scale_stride, names)
        model, payload = stgan.load_stgan_checkpoint(args.run_dir / f"model_epoch_{args.epoch}.pt", device=args.device)
        if hasattr(model, "discriminator_chunk_size"):
            model.discriminator_chunk_size = 10611  # GAT: memory and speed only
        dataset = checkpoints.test_dataset(cubes, payload)
        if len(dataset) != len(timestamps):
            raise ValueError("The test dataset does not have one sample per test timestamp")
        precision = payload.get("precision", "fp32")
        ordinary_positions = np.flatnonzero(timestamps.normalize().isin(ordinary))
        pilot = ordinary_positions[np.unique(np.linspace(0, len(ordinary_positions) - 1,
                                                         min(PILOT_TIMESTAMPS, len(ordinary_positions))).astype(int))]
        floor, calibration, means, shares = fit_pilot(model, dataset, pilot, mc_samples=args.mc_samples,
                                                      device=args.device, precision=precision)
        values = np.empty((len(SCORES), len(positions), dataset.n_locations), dtype=np.float32)
        with scoring_mode(model, True), torch.no_grad():
            for row, position in enumerate(positions):
                batch = dataset.fetch_batch([int(position)])
                cells, draws = draws_of(model, batch, args.mc_samples, args.device, precision)
                order = batch[-1].numpy().reshape(-1)
                values[:, row, order] = three_scores(cells, draws, columns=columns, fitted=fitted, floor=floor,
                                                     calibration=calibration).cpu().numpy()
        days = timestamps[positions].normalize()
    finally:
        cubes.close()
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite model output.")
    print(f"[try] epoch={args.epoch} ordinary days={len(ordinary)} documented days of 2019={len(known)} "
          f"dust arrivals={len(arrivals)} timestamps={len(positions)} mc_samples={args.mc_samples}")
    print("\n=== (deviation / spread) squared by variable, on ordinary timestamps ===")
    print(f"{'variable':<36}{'mean':>10}{'top 1%, raw':>13}{'top 1%, calibrated':>20}")
    for index in np.argsort(-means):
        print(f"{names[index]:<36}{means[index]:>10.1f}{100 * shares[0][index]:>12.1f}%{100 * shares[1][index]:>19.1f}%")
    is_ordinary = days.isin(ordinary)
    # Per score and place: the daily share of cells above the 95th percentile of the cells of the ordinary days.
    share = {}
    for index, name in enumerate(SCORES):
        threshold = np.percentile(values[index][is_ordinary][::4], ALARM)
        for place, cells in cells_of.items():
            share[name, place] = pd.Series((values[index][:, cells] > threshold).mean(axis=1), index=days).groupby(level=0).mean()
    reading = lambda name, place, day: percentile_among(share[name, place][day], share[name, place][ordinary])
    header = "".join(f"{name:>32}" for name in SCORES)
    counts = np.zeros(len(SCORES), dtype=int)
    if known:
        print("\n=== documented days of 2019: percentile of the day among the ordinary days ===")
        for place in KNOWN_ZONES:
            print(f"\n{place}\n{'day':<12}{header}")
            for day in known:
                row = [reading(name, place, day) for name in SCORES]
                counts += np.array(row) >= ALARM
                print(f"{str(day.date()):<12}" + "".join(f"{value:>32.0f}" for value in row))
        print(f"\ndays and places at or above {ALARM}, of {len(known) * len(KNOWN_ZONES)}: "
              + " | ".join(f"{name} {count}" for name, count in zip(SCORES, counts)))
    if arrivals:
        print("\n=== dust arrivals: the highest percentile among the arrival region on the arrival day and the "
              "place upwind on the day before and on that day ===")
        print(f"{'arrival':<34}{header}")
        counts = np.zeros(len(SCORES), dtype=int)
        for region, day in arrivals:
            source = events.SOURCES[region]
            row = [max(reading(name, region, day), reading(name, source, day), reading(name, source, day - DAY))
                   for name in SCORES]
            counts += np.array(row) >= ALARM
            print(f"{region + ' ' + str(day.date()):<34}" + "".join(f"{value:>32.0f}" for value in row))
        print(f"\narrivals at or above {ALARM}, of {len(arrivals)}: "
              + " | ".join(f"{name} {count}" for name, count in zip(SCORES, counts)))
        # The same three readings on every ordinary day: how often the highest of them is that high anyway.
        expected = []
        for name in SCORES:
            rates = []
            for region in events.SOURCES:
                source = events.SOURCES[region]
                rates.append(np.mean([max(reading(name, region, day), reading(name, source, day),
                                          reading(name, source, day - DAY)) >= ALARM for day in anchors]))
            expected.append(np.mean(rates))
        print("on an ordinary day the same reading is that high: "
              + " | ".join(f"{name} {100 * rate:.0f}%" for name, rate in zip(SCORES, expected)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
