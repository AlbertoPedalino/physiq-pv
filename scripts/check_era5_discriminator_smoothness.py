"""Does the discriminator tell the reconstructions of the generator from the observations by their smoothness?

No reference outside the data is used. On a sample of timestamps of the scored period the model of a run (its
last epoch checkpoint, in eval mode: no dropout) reconstructs every field, and the discriminator judges, after the
same history: the observed field, the reconstruction, the observed field smoothed with a 3 x 3 mean over its
neighbours, and the reconstruction smoothed in the same way. D's output near 1 means "generated". Two questions:

- are the reconstructions smoother than the observations? The roughness of a field is the mean over the
  variables and over the pairs of touching cells (row and column neighbours) of their squared difference;
- does D call a field more "generated" when only its smoothness changes? If D tells reconstructions by their
  smoothness, D(smoothed observation) is above D(observation), and the more so the rougher the observation.

The local roughness of a cell is the mean over the variables of the squared difference between its value and
the mean of its 3 x 3 patch. Every timestamp gives the Spearman correlation over the cells between the local
roughness of the observation and D(observation), and between it and the reconstruction error G (the weighted
patch mean, as in the score). The cells with the highest G of each timestamp (`--top-percent` of them) are
read apart, as where G flags.

    python scripts/check_era5_discriminator_smoothness.py --run-dir outputs/stgan_good_sweep_2_scored \
        --checkpoint outputs/stgan_wandb/era5/5qlj2lwg/results/model_epoch_6.pt --samples 240
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
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from physiq_pv.anomaly_detection.stgan.full_grid import STGANFullGridDataset
from physiq_pv.anomaly_detection.stgan.grid import SpatialGrid
from physiq_pv.anomaly_detection.stgan.pipeline import load_stgan_checkpoint

FIELDS = ("observed", "reconstructed", "smoothed observed", "smoothed reconstructed")


def smoothed(field, mask):
    """3 x 3 mean of every cell over the cells of its patch that are on the lattice; zero off the lattice."""
    total = F.avg_pool2d(field * mask, 3, stride=1, padding=1)
    count = F.avg_pool2d(mask, 3, stride=1, padding=1)
    return torch.where(mask.bool(), total / count.clamp_min(1e-6), 0.0)


def roughness(field, mask):
    """Per field [B]: mean squared difference of touching cells, over the variables and the pairs on the lattice."""
    sums, counts = 0.0, 0.0
    for difference, valid in ((field[..., 1:, :] - field[..., :-1, :], mask[..., 1:, :] * mask[..., :-1, :]),
                              (field[..., :, 1:] - field[..., :, :-1], mask[..., :, 1:] * mask[..., :, :-1])):
        sums = sums + (difference.square() * valid).sum(dim=(1, 2, 3))
        counts = counts + valid.sum(dim=(1, 2, 3)) * field.shape[1]
    return sums / counts


def local_roughness(field, mask):
    """Per cell [B,H,W]: mean over the variables of the squared distance from the 3 x 3 mean."""
    return (field - smoothed(field, mask)).square().mean(dim=1) * mask[:, 0]


def judge(model, recent, trend, mask, calendar, observed):
    """The reconstruction, G and D's outputs on the four fields, after one history: per cell [B*N]."""
    reconstruction = model.discriminator_input(recent, trend, mask, calendar)
    errors = torch.where(mask.bool(), reconstruction - observed, 0.0).square()
    generator = model.reconstruction_score(model.weighted_errors(errors), mask)
    historical = model.discriminator.encode_history(recent, mask)
    fields = (observed, reconstruction, smoothed(observed, mask), smoothed(reconstruction, mask))
    outputs = model.discriminator.score_currents(historical, fields, mask)
    return fields, generator, [values.squeeze(-1) for values in outputs]


def check(model, dataset, positions, *, batch_size=2, top_percent=5.0):
    """One row per sampled timestamp: roughness of the four fields, D's means, the shares and correlations."""
    cells = torch.as_tensor(dataset.cells)
    rows = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch = dataset.fetch_batch(positions[start:start + batch_size])
            recent, trend, mask, calendar, observed = batch[:5]
            fields, generator, outputs = judge(model, recent, trend, mask, calendar, observed)
            count = len(observed)
            shape = (count, dataset.n_locations)
            generator, outputs = generator.reshape(shape), [values.reshape(shape) for values in outputs]
            local = local_roughness(observed, mask).flatten(1)[:, cells]
            rough = [roughness(field, mask) for field in fields]
            for index in range(count):
                g, real, smooth = generator[index].numpy(), outputs[0][index].numpy(), outputs[2][index].numpy()
                top = g >= np.quantile(g, 1 - top_percent / 100)
                rise = smooth - real
                rows.append({
                    "target": int(dataset.targets[positions[start + index]]),
                    **{f"roughness {name}": float(values[index]) for name, values in zip(FIELDS, rough)},
                    **{f"D {name}": float(values[index].mean()) for name, values in zip(FIELDS, outputs)},
                    "D rises when smoothed": float((rise > 0).mean()),
                    "D rises when smoothed, top G": float((rise[top] > 0).mean()),
                    "D change when smoothed": float(rise.mean()),
                    "D change when smoothed, top G": float(rise[top].mean()),
                    "D observed, top G": float(real[top].mean()),
                    "spearman local roughness - D observed": float(spearmanr(local[index].numpy(), real)[0]),
                    "spearman local roughness - G": float(spearmanr(local[index].numpy(), g)[0]),
                    "spearman local roughness - D change": float(spearmanr(local[index].numpy(), rise)[0]),
                })
    return pd.DataFrame(rows)


def saved_grid(payload):
    """The lattice the checkpoint was trained on, as load_stgan_checkpoint rebuilds it."""
    saved = payload["grid"]
    return SpatialGrid(saved["node_indices"], saved["valid_mask"], saved["row_indices"], saved["column_indices"], {})


def open_period(run_dir):
    """The scored period of a run over its shared cache: data with their context, timestamps, longitudes."""
    manifest = json.loads((Path(run_dir) / "metadata.json").read_text(encoding="utf-8"))
    seasonal = manifest.get("backend", {}).get("seasonal_normalization")
    if not seasonal:
        raise ValueError("The run has no seasonal normalization")
    cache = Path(seasonal["mean_file"]).parent
    period = manifest.get("scored_period", "test")
    data = np.load(cache / f"{period}.npy", mmap_mode="r")
    timestamps = pd.DatetimeIndex(np.load(Path(run_dir) / "test_timestamps.npy"))
    context = len(data) - len(timestamps)  # the cache holds a few timestamps of context before the period
    if context < 0:
        raise ValueError("The cache and the run are not over the same period")
    step = timestamps[1] - timestamps[0]
    with_context = pd.DatetimeIndex([timestamps[0] - step * k for k in range(context, 0, -1)]).append(timestamps)
    grid = manifest["grid"]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
    return data, with_context, context, longitudes, period


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Scored run (its period, grid and shared cache)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Epoch checkpoint of the trained model")
    parser.add_argument("--samples", type=int, default=240, help="Timestamps read, evenly spaced over the period")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--top-percent", type=float, default=5, help="Cells with the highest G of each timestamp")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--csv", type=Path, help="Also write one row per timestamp here")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    model, payload = load_stgan_checkpoint(args.checkpoint, device="cpu")
    data, timestamps, context, longitudes, period = open_period(args.run_dir)
    window, normalization = payload["window_config"], payload["normalization"]
    try:
        # The normalization and the windows the model was trained with, from its checkpoint.
        dataset = STGANFullGridDataset(
            data, timestamps, saved_grid(payload),
            feature_minimum=np.asarray(normalization["minimum"], dtype=np.float32),
            feature_scale=np.asarray(normalization["scale"], dtype=np.float32),
            recent_steps=window["recent_steps"], trend_steps=window["trend_steps"], stride=1, normalized=False,
            time_encoding=payload.get("time_encoding", "onehot"), longitudes=longitudes)
        scored = np.flatnonzero(dataset.targets >= context)  # the targets of the scored period, not of its context
        # An odd spacing, so that every time of day is read.
        spacing = max(1, len(scored) // args.samples) | 1
        positions = scored[::spacing][:args.samples]
        table = check(model, dataset, positions, batch_size=args.batch_size, top_percent=args.top_percent)
    finally:
        data._mmap.close()
    table.insert(1, "timestamp", timestamps[table["target"]])
    if args.csv is not None:
        table.to_csv(args.csv, index=False)
    print(f"[smoothness] {period}: {len(table)} timestamps, one every {spacing}, from {table['timestamp'].iloc[0]} "
          f"to {table['timestamp'].iloc[-1]} | model in eval mode (no dropout) | D near 1 = judged generated")
    print("\nroughness (mean squared difference of touching cells), median over the timestamps")
    for name in FIELDS:
        print(f"  {name:<24}{table[f'roughness {name}'].median():>10.5f}")
    print(f"  reconstructed / observed: median {np.median(table['roughness reconstructed'] / table['roughness observed']):.3f}, "
          f"below 1 in {100 * (table['roughness reconstructed'] < table['roughness observed']).mean():.1f}% of the timestamps")
    print("\nD's mean output over the cells, median over the timestamps")
    for name in FIELDS:
        print(f"  D({name}){'':<{22 - len(name)}}{table[f'D {name}'].median():>10.4f}")
    print(f"  D(observed), cells with the top {args.top_percent:g}% of G  {table['D observed, top G'].median():.4f}")
    print("\nsmoothing the observed field (D(smoothed observed) - D(observed)), median over the timestamps")
    for label, key in (("all cells", ""), (f"top {args.top_percent:g}% of G", ", top G")):
        print(f"  {label:<18} share of cells where D rises {100 * table[f'D rises when smoothed{key}'].median():5.1f}% | "
              f"mean change {table[f'D change when smoothed{key}'].median():+.4f}")
    print("\nSpearman over the cells of each timestamp, median (share of timestamps below 0)")
    for name in ("D observed", "G", "D change"):
        values = table[f"spearman local roughness - {name}"]
        print(f"  local roughness of the observation - {name:<11}{values.median():+.3f}  ({100 * (values < 0).mean():.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
