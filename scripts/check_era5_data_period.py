"""The period of every ERA5 variable, from its autocorrelation, for the buffer in time around the concordant clusters.

Paparrizos et al. (PVLDB 2022, Sec. 4.1-4.2) set the buffer around each anomaly to half of the period of the time series
on each side, "the period is an intrinsic characteristic of the time series", found from its autocorrelation or Fourier
transform. Here the time series are the ERA5 variables as prepared (`--prepared-dir`, physical units, no transform:
the normalisation belongs to the detectors), over the years of `--partition` (the test years by default). On `--cells`
cells drawn at random (seed `--seed`), every cell's series of every variable is standardised and its autocorrelation
computed with the FFT; the mean over the cells is printed for the lags up to `--max-lag-hours`, with its first local
maximum after lag 0 (the period), the prominence of that maximum (its autocorrelation minus the lowest one before it:
near 0 for noise, large for a cycle) and the period of the highest peak of the mean periodogram between
`--min-period-hours` and `--max-period-hours`. Gaps in the timestamps are counted; the series are taken as they are.

    python scripts/check_era5_data_period.py --prepared-dir data/era5_prepared --csv outputs/data_period.csv
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

CHUNK = 256  # timestamps read at a time (all cells and variables of a timestamp are read together)
BATCH = 256  # cells transformed at a time


def read_cells(data, cells):
    """The cells `cells` of a [timestamp, cell, variable] array opened with mmap, as float32 [timestamp, cell, var]."""
    out = np.empty((data.shape[0], len(cells), data.shape[2]), dtype=np.float32)
    for start in range(0, data.shape[0], CHUNK):
        out[start:start + CHUNK] = np.asarray(data[start:start + CHUNK])[:, cells]
    return out


def mean_autocorrelation(series, max_lag):
    """Mean over the columns of `series` [time, column] of the autocorrelation at lags 0..max_lag, with the mean
    periodogram (frequencies in cycles per step). Each column is standardised; missing values count as its mean;
    constant columns are left out."""
    n = series.shape[0]
    size = 1 << int(np.ceil(np.log2(2 * n)))  # zero padding: linear, not circular, autocorrelation
    acf = np.zeros(max_lag + 1)
    power = np.zeros(n // 2 + 1)
    used = 0
    for start in range(0, series.shape[1], BATCH):
        block = np.asarray(series[:, start:start + BATCH], dtype=np.float64)
        spread = np.nanstd(block, axis=0)
        block = block[:, spread > 0]
        if not block.shape[1]:
            continue
        block = np.nan_to_num((block - np.nanmean(block, axis=0)) / np.nanstd(block, axis=0))
        lagged = np.fft.irfft(np.abs(np.fft.rfft(block, n=size, axis=0)) ** 2, n=size, axis=0)[:max_lag + 1]
        acf += (lagged / lagged[0]).sum(axis=1)
        power += (np.abs(np.fft.rfft(block, axis=0)) ** 2).sum(axis=1)
        used += block.shape[1]
    return acf / max(used, 1), np.fft.rfftfreq(n), power / max(used, 1), used


def first_peak(acf):
    """Lag of the first local maximum of `acf` after lag 0 (None if it only decreases)."""
    for lag in range(1, len(acf) - 1):
        if acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1]:
            return lag
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True, help="Output of prepare_era5 (physiq_pv.era5.data)")
    parser.add_argument("--partition", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--cells", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-lag-hours", type=float, default=72)
    parser.add_argument("--min-period-hours", type=float, default=6)
    parser.add_argument("--max-period-hours", type=float, default=240)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    metadata = json.loads((args.prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    names = list(metadata["features"])
    timestamps = pd.DatetimeIndex(np.load(args.prepared_dir / f"{args.partition}_timestamps.npy"))
    steps = np.diff(timestamps.asi8) / 3.6e12
    step = float(np.median(steps))
    gaps = int((steps != step).sum())
    data = np.load(args.prepared_dir / f"{args.partition}.npy", mmap_mode="r")
    if data.shape[2] != len(names) or data.shape[0] != len(timestamps):
        raise ValueError("The prepared array does not match its timestamps and features")
    cells = np.sort(np.random.default_rng(args.seed).choice(data.shape[1], min(args.cells, data.shape[1]), False))
    values = read_cells(data, cells)
    data._mmap.close()
    print(f"[period] {args.partition}: {len(timestamps)} timestamps every {step:g} h ({gaps} gaps), {timestamps[0]} to "
          f"{timestamps[-1]}; {len(cells)} of {data.shape[1]} cells; {len(names)} variables", flush=True)
    max_lag = int(round(args.max_lag_hours / step))
    rows = []
    for k, name in enumerate(names):
        acf, frequency, power, used = mean_autocorrelation(values[:, :, k], max_lag)
        peak = first_peak(acf)
        prominence = acf[peak] - acf[1:peak].min() if peak and peak > 1 else np.nan
        with np.errstate(divide="ignore"):
            period = np.where(frequency > 0, step / frequency, np.inf)
        band = (period >= args.min_period_hours) & (period <= args.max_period_hours)
        top = period[band][np.argmax(power[band])] if band.any() and used else np.nan
        print(f"[period] {name}: first peak of the autocorrelation at {peak * step if peak else 'none'} h (prominence "
              f"{prominence:.3f}); highest periodogram peak between {args.min_period_hours:g} and "
              f"{args.max_period_hours:g} h at {top:.1f} h; {used} cells not constant", flush=True)
        rows.append({"variable": name, "cells": used, "first_peak_hours": peak * step if peak else np.nan,
                     "first_peak_prominence": prominence, "periodogram_peak_hours": top,
                     **{f"acf_{lag * step:g}h": acf[lag] for lag in range(max_lag + 1)}})
    table = pd.DataFrame(rows).set_index("variable")
    lags = [c for c in table.columns if c.startswith("acf_")]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\n[period] first peak (hours), its prominence and the periodogram peak, by variable")
        print(table[["first_peak_hours", "first_peak_prominence", "periodogram_peak_hours"]].round(3).to_string())
        print("\n[period] mean autocorrelation of the standardised variable by lag")
        print(table[lags].round(3).to_string())
    if args.csv:
        table.to_csv(args.csv)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
