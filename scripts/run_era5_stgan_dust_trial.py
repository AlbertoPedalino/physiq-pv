"""A trial run with the Saharan dust among the inputs. Not the protocol of the thesis.

ERA5 has no aerosol variable, so the detector cannot see the dust itself. This trial adds the dust
aerosol optical depth of the CAMS reanalysis as a 16th variable and trains on the first years for
which it is available, instead of 1980-2003:

- the 15 ERA5 variables come from the test partition of the prepared cache (2005 onwards);
- the dust comes from the yearly files written by `check_era5_dust_lead.py`: daily means on the
  CAMS grid, interpolated to the cells of the model (bilinear) and to its timestamps (linear in
  time between the daily means, taken at 12 UTC);
- the first `--train-years` years of the common period train the model and the others are scored.

There is no validation year and none of the fixed PCA/MMD references, which are built for the 15
ERA5 variables. The run directory has the layout of the other ERA5 runs, so the rebalancing and
the dust checks read it; the dust of those checks is then no longer independent of the model.

    python scripts/run_era5_stgan_dust_trial.py --prepared-dir data/era5_prepared \
        --dust-dir data/cams_dust --output-dir outputs/stgan_dust_trial
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan
from physiq_pv.era5.data import load_prepared
from scripts.check_era5_dust_lead import VARIABLE as DUST_VARIABLE, load_dust

CHUNK = 256  # timestamps written at a time
NOON = pd.Timedelta(hours=12)


def dust_on_cells(dust, dust_latitudes, dust_longitudes, latitudes, longitudes):
    """Daily dust [days, cells] on the cells of the model, bilinear on the dust grid.

    A cell outside that grid takes the value of its edge.
    """
    lat_order, lon_order = np.argsort(dust_latitudes), np.argsort(dust_longitudes)
    grid_latitudes, grid_longitudes = np.asarray(dust_latitudes)[lat_order], np.asarray(dust_longitudes)[lon_order]
    points = np.column_stack((np.clip(latitudes, grid_latitudes[0], grid_latitudes[-1]),
                              np.clip(longitudes, grid_longitudes[0], grid_longitudes[-1])))
    result = np.empty((len(dust), len(points)), dtype=np.float32)
    for day, field in enumerate(dust):
        result[day] = RegularGridInterpolator((grid_latitudes, grid_longitudes),
                                              np.asarray(field, dtype=np.float64)[np.ix_(lat_order, lon_order)])(points)
    return result


def dust_at(daily, days, timestamps):
    """Dust [timestamps, cells]: linear in time between the daily means, taken at 12 UTC.

    Before the first noon and after the last one the nearest daily mean is kept.
    """
    position = (timestamps - (days[0] + NOON)) / pd.Timedelta(days=1)
    position = np.clip(np.asarray(position, dtype=np.float64), 0, len(days) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(days) - 1)
    weight = (position - lower)[:, None].astype(np.float32)
    return daily[lower] * (1 - weight) + daily[upper] * weight


def build_arrays(cubes, dust, dust_days, dust_latitudes, dust_longitudes, directory, *, train_end):
    """Write train.npy and test.npy [time, cells, 16] in `directory`; return them with their timestamps.

    The period is the part of the test partition of the cache that the dust covers; `train_end` is
    its first timestamp that is scored instead of trained on.
    """
    timestamps = cubes.test_timestamps
    covered = (timestamps >= dust_days[0]) & (timestamps < dust_days[-1] + pd.Timedelta(days=1))
    positions = np.flatnonzero(covered)
    if not len(positions) or not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
        raise ValueError("The dust and the prepared cache share no continuous period")
    if not (np.diff(dust_days.values) == np.timedelta64(1, "D")).all():
        raise ValueError("The saved dust has missing days")
    timestamps = timestamps[positions]
    is_train = timestamps < train_end
    if not is_train.any() or is_train.all():
        raise ValueError("The split leaves no training or no test timestamps")
    daily = dust_on_cells(dust, dust_latitudes, dust_longitudes, cubes.latitudes, cubes.longitudes)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, part in (("train", positions[is_train]), ("test", positions[~is_train])):
        shape = (len(part), cubes.test.shape[1], cubes.test.shape[2] + 1)
        array = np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape)
        for start in range(0, len(part), CHUNK):
            chunk = part[start:start + CHUNK]
            array[start:start + len(chunk), :, :-1] = cubes.test[chunk[0]:chunk[-1] + 1]
            array[start:start + len(chunk), :, -1] = dust_at(daily, dust_days, cubes.test_timestamps[chunk])
        array.flush()
        result[name] = (array, cubes.test_timestamps[part])
    return result


def run_trial(*, prepared_dir, dust_dir, output_dir, config, train_years=15, train_end=None, device="cuda", seed=20):
    """Build the 16-variable arrays, train and score; write the run directory and return its manifest."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty run directory")
    cubes, grid, preparation = load_prepared(prepared_dir)
    arrays = {}
    try:
        dust, dust_days, dust_latitudes, dust_longitudes = load_dust(dust_dir)
        first = max(cubes.test_timestamps[0].normalize(), dust_days[0])
        boundary = pd.Timestamp(train_end) if train_end is not None else first + pd.DateOffset(years=train_years)
        arrays = build_arrays(cubes, dust, dust_days, dust_latitudes, dust_longitudes, output / "data",
                              train_end=boundary)
        (train, train_timestamps), (test, test_timestamps) = arrays["train"], arrays["test"]
        feature_names = (*cubes.feature_names, DUST_VARIABLE)
        print(f"[dust-trial] train {train_timestamps[0]} -> {train_timestamps[-1]} ({len(train)} timestamps) | "
              f"test {test_timestamps[0]} -> {test_timestamps[-1]} ({len(test)} timestamps) | variables {len(feature_names)}",
              flush=True)
        options = asdict(config)
        options["lr"] = options.pop("learning_rate")
        options.pop("validation_holdout")
        with fit_and_score_stgan(
                train, test, train_timestamps=train_timestamps, test_timestamps=test_timestamps,
                location_names=cubes.location_names, feature_names=feature_names, latitudes=cubes.latitudes,
                longitudes=cubes.longitudes, device=device, seed=seed, checkpoint_path=output / "model.pt",
                score_dir=output / "scores", timestep_hours=3, dataset_name="era5_with_cams_dust",
                angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper", **options) as result:
            np.save(output / "test_timestamps.npy", result.test_timestamps.as_unit("ns").asi8)
            manifest = {
                "status": "complete", "grid": grid.to_dict(),
                "preparation": {**preparation, "features": list(feature_names), "source_prepared_dir": str(prepared_dir),
                                "dust": {"dir": str(dust_dir), "variable": DUST_VARIABLE,
                                         "interpolation": "bilinear in space, linear in time between daily means at 12 UTC"}},
                "trial": "ERA5 with CAMS dust as a 16th variable; not the train/validation/test protocol of the thesis",
                "splits": {"train": [str(train_timestamps[0]), str(train_timestamps[-1])],
                           "test": [str(test_timestamps[0]), str(test_timestamps[-1])]},
                "backend": result.metadata, "config": asdict(config),
                "scores_file": "scores/anomaly_mean.npy", "anomaly_mean_file": "scores/anomaly_mean.npy",
                "anomaly_std_file": "scores/anomaly_std.npy", "generator_mean_file": "scores/generator_mean.npy",
                "generator_std_file": "scores/generator_std.npy",
                "discriminator_mean_file": "scores/discriminator_mean.npy",
                "discriminator_std_file": "scores/discriminator_std.npy",
                "component_covariance_file": "scores/component_covariance.npy"}
            (output / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        for array, _ in arrays.values():
            array._mmap.close()
        cubes.close()
    return manifest


def parser():
    root = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    root.add_argument("--prepared-dir", type=Path, required=True)
    root.add_argument("--dust-dir", type=Path, required=True, help="Yearly dust files of check_era5_dust_lead.py")
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--train-years", type=int, default=15, help="Years of the common period that train the model")
    root.add_argument("--train-end", help="First scored timestamp, instead of --train-years (e.g. 2014-01-01)")
    root.add_argument("--epochs", type=int, default=1)
    root.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    root.add_argument("--recent-steps", type=int, default=16)
    root.add_argument("--reconstruction-weights", choices=("uniform", "daily_change"), default="uniform")
    root.add_argument("--normalization", choices=("minmax", "seasonal"), default="minmax",
                      help="seasonal: first remove the mean annual and diurnal cycle of each cell, dust included")
    root.add_argument("--discriminator-input-dropout", action=argparse.BooleanOptionalAction, default=False,
                      help="Default: D judges the reconstruction without the generator's dropout, as in the last runs")
    root.add_argument("--mc-samples", type=int, default=20)
    root.add_argument("--num-workers", type=int, default=4)
    root.add_argument("--device", default="cuda")
    root.add_argument("--seed", type=int, default=20)
    return root


def trial_config(args):
    """The full-grid ConvGRU of the last ERA5 runs, without a validation year."""
    return STGANCNNConfig(
        epochs=args.epochs, batch_size=1, score_batch_size=1, precision=args.precision, cnn_training_mode="full_grid",
        recent_steps=args.recent_steps, trend_steps=56, time_encoding="cyclic", grid_crs="EPSG:4326",
        num_workers=args.num_workers, discriminator_input_dropout=args.discriminator_input_dropout,
        reconstruction_weights=args.reconstruction_weights, normalization=args.normalization,
        mc_samples=args.mc_samples, score_storage="memmap")


def main(argv=None):
    args = parser().parse_args(argv)
    manifest = run_trial(prepared_dir=args.prepared_dir, dust_dir=args.dust_dir, output_dir=args.output_dir,
                         config=trial_config(args), train_years=args.train_years, train_end=args.train_end,
                         device=args.device, seed=args.seed)
    print(json.dumps({key: manifest[key] for key in ("status", "trial", "splits")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
