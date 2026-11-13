"""The period of the score of every detector, from its autocorrelation, for the buffer in time around the clusters.

Paparrizos et al. (PVLDB 2022, Sec. 4.1-4.2) set the buffer around each anomaly to half of the period of the time series
on each side, the period being found from the autocorrelation or the Fourier transform of the series. Here the series
are the scores of the detectors (`--runs NAME=DIR[@component]`, as for build_era5_concordant_cube.py), each a block
[timestamp, cell]. On `--cells` cells drawn at random among those every timestamp scores (seed `--seed`), every cell's
score is standardised and its autocorrelation computed with the FFT; the mean over the cells is printed for the lags up
to `--max-lag-hours`, with its first local maximum after lag 0 (the period of Paparrizos et al.), the prominence of that
maximum (its autocorrelation minus the lowest one before it: near 0 for noise, large for a cycle) and the period of the
highest peak of the mean periodogram between `--min-period-hours` and `--max-period-hours`. Gaps in the timestamps are
counted and printed; the series are taken as they are (one step between consecutive timestamps).

    python scripts/check_era5_score_period.py --runs Gn=outputs/optuna27/test_generator_clim_day \\
        pca=outputs/confronti_good2/climatology/test/pca --csv outputs/score_period.csv
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
from scripts.check_era5_dust_lead import score_file
from scripts.check_era5_score_correlation import parse_score

CHUNK = 2048  # timestamps read at a time
BATCH = 256  # cells transformed at a time


def read_cells(scores, cells):
    """The columns `cells` of a [timestamp, cell] array opened with mmap, as float64."""
    out = np.empty((scores.shape[0], len(cells)), dtype=np.float64)
    for start in range(0, scores.shape[0], CHUNK):
        out[start:start + CHUNK] = np.asarray(scores[start:start + CHUNK])[:, cells]
    return out


def mean_autocorrelation(series, max_lag):
    """Mean over the columns of `series` [time, column] of the autocorrelation at lags 0..max_lag, with the mean
    periodogram (frequencies in cycles per step). Each column is standardised; missing values count as its mean."""
    n = series.shape[0]
    size = 1 << int(np.ceil(np.log2(2 * n)))  # zero padding: linear, not circular, autocorrelation
    acf = np.zeros(max_lag + 1)
    power = np.zeros(n // 2 + 1)
    for start in range(0, series.shape[1], BATCH):
        block = series[:, start:start + BATCH]
        block = (block - np.nanmean(block, axis=0)) / np.nanstd(block, axis=0)
        block = np.nan_to_num(block)
        spectrum = np.fft.rfft(block, n=size, axis=0)
        lagged = np.fft.irfft(np.abs(spectrum) ** 2, n=size, axis=0)[:max_lag + 1]
        acf += (lagged / lagged[0]).sum(axis=1)
        power += (np.abs(np.fft.rfft(block, axis=0)) ** 2).sum(axis=1)
    return acf / series.shape[1], np.fft.rfftfreq(n), power / series.shape[1]


def first_peak(acf):
    """Lag of the first local maximum of `acf` after lag 0 (None if it only decreases)."""
    for lag in range(1, len(acf) - 1):
        if acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1]:
            return lag
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", required=True, help="NAME=DIR[@component] of the scores")
    parser.add_argument("--cells", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-lag-hours", type=float, default=72)
    parser.add_argument("--min-period-hours", type=float, default=6)
    parser.add_argument("--max-period-hours", type=float, default=240)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    rows = []
    for name, directory, component in (parse_score(text) for text in args.runs):
        manifest = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        timestamps = pd.DatetimeIndex(np.load(directory / "test_timestamps.npy"))
        steps = np.diff(timestamps.asi8) / 3.6e12
        step = float(np.median(steps))
        gaps = int((steps != step).sum())
        scores = np.load(directory / score_file(manifest, component), mmap_mode="r")
        # The cells the detector scores: finite over the first block of timestamps (later gaps count as the mean).
        candidates = np.flatnonzero(np.isfinite(np.asarray(scores[:CHUNK])).all(axis=0))
        cells = np.sort(np.random.default_rng(args.seed).choice(candidates, min(args.cells, len(candidates)), False))
        series = read_cells(scores, cells)
        release_pages(scores)
        scores._mmap.close()
        max_lag = int(round(args.max_lag_hours / step))
        acf, frequency, power = mean_autocorrelation(series, max_lag)
        peak = first_peak(acf)
        prominence = acf[peak] - acf[1:peak].min() if peak and peak > 1 else np.nan
        with np.errstate(divide="ignore"):
            period = np.where(frequency > 0, step / frequency, np.inf)
        band = (period >= args.min_period_hours) & (period <= args.max_period_hours)
        top = period[band][np.argmax(power[band])] if band.any() else np.nan
        print(f"[period] {name}: {len(timestamps)} timestamps every {step:g} h ({gaps} gaps), {len(cells)} cells; first "
              f"peak of the autocorrelation at {peak * step if peak else 'none'} h (prominence {prominence:.3f}); highest periodogram peak between "
              f"{args.min_period_hours:g} and {args.max_period_hours:g} h at {top:.1f} h", flush=True)
        rows.append({"detector": name, "step_hours": step, "gaps": gaps, "cells": len(cells),
                     "first_peak_hours": peak * step if peak else np.nan, "first_peak_prominence": prominence,
                     "periodogram_peak_hours": top,
                     **{f"acf_{lag * step:g}h": acf[lag] for lag in range(max_lag + 1)}})
    table = pd.DataFrame(rows).set_index("detector")
    lags = [c for c in table.columns if c.startswith("acf_")]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\n[period] mean autocorrelation of the standardised score by lag")
        print(table[lags].round(3).to_string())
    if args.csv:
        table.to_csv(args.csv)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
