"""Does the Monte Carlo dropout band of the reconstruction contain the observation?

The anomaly score has no true value to compare its band with. The reconstruction has one: the
observed field. This script scores a sample of test timestamps with an epoch checkpoint, keeps the
dropout draws of the reconstruction of every cell and variable, and measures

- the coverage of the band (PICP): how often the observation falls between the 5th and the 95th
  percentile of the draws, against the nominal 90%, with the width of that band as a share of the
  range of the variable (NMPIW);
- the same by band of the rebalanced score: a band that tells normal from anomalous contains the
  observation on the ordinary points and not on the highest scores;
- the band against the deviation: by band of the absolute difference between the observation and
  the mean reconstruction, the spread of the draws in absolute value and as a share of that
  difference. A larger deviation can have a wider band and still a narrower one in proportion.

All values are in the units the model works in (min-max to [-1, 1], range 2). The rebalanced score
is the one of `rebalance_era5_stgan_scores.py --scale rarity`, fitted on the errors saved by the
run. The model must score one whole timestamp per batch (GAT, or the full-grid ConvGRU).

    python scripts/check_era5_stgan_reconstruction_band.py --prepared-dir data/era5_prepared \
        --run-dir outputs/run --epoch 6
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

import physiq_pv.anomaly_detection.stgan as stgan
from physiq_pv.anomaly_detection.stgan.precision import autocast_context
from physiq_pv.anomaly_detection.stgan.scoring import scoring_mode
from physiq_pv.era5.data import load_prepared
from scripts import analyze_era5_stgan_checkpoints as checkpoints
from scripts import rescore_era5_stgan_rebalanced as rescore

NOMINAL = 0.90  # the band goes from the 5th to the 95th percentile of the draws
BANDS = (0, 50, 90, 95, 99, 100)  # percentiles, of the score or of the deviation
RANGE = 2.0  # min-max to [-1, 1]


def draw_statistics(model, dataset, positions, *, scale, columns, fitted, mc_samples=20, device="cpu", precision="fp32"):
    """Per point (timestamp, cell) and variable: whether the observation is inside the band, the width of
    the band, the deviation of the observation from the mean reconstruction and the spread of the draws;
    and per point the rebalanced score. Arrays [points, variables] and [points]."""
    tail = (1 - NOMINAL) / 2
    inside, width, deviation, spread, score = [], [], [], [], []
    with scoring_mode(model, True), torch.no_grad():
        for position in positions:
            batch = dataset.fetch_batch([int(position)])
            recent, trend, mask, calendar, observed = (values.to(device) for values in batch[:5])
            draws = []
            for _ in range(mc_samples):
                with autocast_context(precision, device):
                    cells, generated = model.reconstructed_cells(recent, trend, mask, calendar, observed)
                draws.append(generated.float())
            draws, cells = torch.stack(draws), cells.float()
            lower, upper = torch.quantile(draws, torch.tensor([tail, 1 - tail], device=draws.device), dim=0)
            inside.append(((cells >= lower) & (cells <= upper)).cpu().numpy())
            width.append((upper - lower).cpu().numpy())
            deviation.append((cells - draws.mean(dim=0)).abs().cpu().numpy())
            spread.append(draws.std(dim=0, correction=0).cpu().numpy())
            errors = (draws - cells).square().mean(dim=0)[:, columns]
            score.append(rescore.transform(errors, scale, fitted).mean(dim=1).cpu().numpy())
    arrays = [np.concatenate(values) for values in (inside, width, deviation, spread, score)]
    if not all(np.isfinite(values).all() for values in arrays[1:]):
        raise FloatingPointError("Non-finite model output.")
    return arrays


def band_of(values):
    """Index of the band of BANDS of every value, by the percentiles of `values` themselves."""
    edges = np.percentile(values, BANDS)
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(BANDS) - 2)


def by_variable(inside, width, deviation, spread):
    """Per variable: coverage, width over the range, mean deviation, mean spread and spread over deviation."""
    return np.column_stack((inside.mean(axis=0), width.mean(axis=0) / RANGE, deviation.mean(axis=0),
                            spread.mean(axis=0), spread.mean(axis=0) / deviation.mean(axis=0)))


def by_score_band(inside, width, score):
    """Per band of the score: points, coverage and width over the range, all variables together."""
    band = band_of(score)
    return np.array([[(band == index).sum(), inside[band == index].mean(), width[band == index].mean() / RANGE]
                     for index in range(len(BANDS) - 1)])


def by_deviation_band(inside, deviation, spread):
    """Per band of the deviation of each variable: mean deviation, mean spread, spread over deviation and
    coverage. Each variable is put in bands by its own deviations and the variables are then averaged."""
    rows = np.zeros((len(BANDS) - 1, 4))
    for variable in range(deviation.shape[1]):
        band = band_of(deviation[:, variable])
        for index in range(len(BANDS) - 1):
            chosen = band == index
            mean_deviation, mean_spread = deviation[chosen, variable].mean(), spread[chosen, variable].mean()
            rows[index] += (mean_deviation, mean_spread, mean_spread / mean_deviation, inside[chosen, variable].mean())
    return rows / deviation.shape[1]


def print_report(names, inside, width, deviation, spread, score):
    label = lambda index: f"{BANDS[index]:g}-{BANDS[index + 1]:g}"
    print(f"\n=== band of the reconstruction by variable (5th to 95th percentile of the draws, nominal coverage "
          f"{100 * NOMINAL:.0f}%) ===")
    print(f"{'variable':<36}{'PICP':>8}{'width/range':>13}{'deviation':>11}{'spread':>10}{'spread/deviation':>18}")
    table = by_variable(inside, width, deviation, spread)
    for index in np.argsort(table[:, 0]):
        coverage, share, mean_deviation, mean_spread, ratio = table[index]
        print(f"{names[index]:<36}{100 * coverage:>7.1f}%{100 * share:>12.2f}%{mean_deviation:>11.4f}{mean_spread:>10.4f}"
              f"{ratio:>18.3f}")
    print(f"{'all variables':<36}{100 * inside.mean():>7.1f}%{100 * width.mean() / RANGE:>12.2f}%{deviation.mean():>11.4f}"
          f"{spread.mean():>10.4f}{spread.mean() / deviation.mean():>18.3f}")
    print("\n=== by band of the rebalanced score, all variables ===")
    print(f"{'score percentiles':<20}{'points':>10}{'PICP':>8}{'width/range':>13}")
    for index, (points, coverage, share) in enumerate(by_score_band(inside, width, score)):
        print(f"{label(index):<20}{int(points):>10d}{100 * coverage:>7.1f}%{100 * share:>12.2f}%")
    print("\n=== the band against the deviation: by band of the deviation of each variable, mean of the variables ===")
    print(f"{'deviation percentiles':<24}{'deviation':>11}{'spread':>10}{'spread/deviation':>18}{'PICP':>8}")
    for index, (mean_deviation, mean_spread, ratio, coverage) in enumerate(by_deviation_band(inside, deviation, spread)):
        print(f"{label(index):<24}{mean_deviation:>11.4f}{mean_spread:>10.4f}{ratio:>18.3f}{100 * coverage:>7.1f}%")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run with model_epoch_<n>.pt and scores/feature_scores.npy")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--timestamps", type=int, default=200, help="Test timestamps, evenly spaced over the period")
    parser.add_argument("--scale-stride", type=int, default=10, help="The scales of the score are read on one timestamp every this many")
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.mc_samples < 2:
        raise ValueError("The band needs at least two draws")
    cubes = load_prepared(args.prepared_dir)[0]
    try:
        names = list(cubes.feature_names)
        columns, fitted = rescore.fitted_scales(args.run_dir, "rarity", None, args.scale_stride, names)
        model, payload = stgan.load_stgan_checkpoint(args.run_dir / f"model_epoch_{args.epoch}.pt", device=args.device)
        if hasattr(model, "discriminator_chunk_size"):
            model.discriminator_chunk_size = 10611  # GAT: memory and speed only
        dataset = checkpoints.test_dataset(cubes, payload)
        positions = np.linspace(0, len(dataset) - 1, min(args.timestamps, len(dataset))).astype(int)
        inside, width, deviation, spread, score = draw_statistics(
            model, dataset, positions, scale="rarity", columns=columns, fitted=fitted, mc_samples=args.mc_samples,
            device=args.device, precision=payload.get("precision", "fp32"))
    finally:
        cubes.close()
    print(f"[band] epoch={args.epoch} timestamps={len(positions)} locations={dataset.n_locations} "
          f"mc_samples={args.mc_samples} points={len(score)} variables={len(names)}")
    print_report(names, inside, width, deviation, spread, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
