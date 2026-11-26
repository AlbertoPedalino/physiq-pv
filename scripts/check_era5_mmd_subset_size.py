"""How much the validation MMD of a run depends on the size of its validation subsets.

The sweep scored every run with the mean, over 5 fixed and disjoint subsets of 512 complete validation fields of
normal weather, of the unbiased MMD^2 between the PCA features of the observed fields and of their reconstructions
(stgan/mmd.py), and with the mean of that value over the last epochs (the sweep objective). 512 is a fixed value,
not tuned. This check measures, on saved runs and with nothing retrained, what the same distance gives with
larger subsets:

- the 5 x 512 subsets of the run, recomputed (it must reproduce the value in the run's training_history.csv);
- one subset with every normal validation field (share of flagged values at most the median of its calendar
  month, the rule of normal_masks.normal_rows, without the draw of an equal number per month);
- for several subset sizes, `--draws` random subsets of that size drawn from those fields (fixed seed): the
  spread of the distance due to which fields are drawn.

Everything else is the run's own: the run is rebuilt through the same entry point as score_era5_stgan_run.py
(configuration, shared cache, normal masks with the cleaned validation history, PCA and MMD references, frozen
kernel bandwidth) and stopped when it is about to build its training loader, before any training or scoring.
The weights of every epoch of the objective window are then loaded from the run's checkpoints. Two checks make
the rebuild trustworthy: the observed fields must reproduce the features saved in the MMD reference, and the
recomputed 5 x 512 distance is printed next to the one in training_history.csv. Nothing of the runs is written
or changed. Paths recorded before a move of the home directory are mapped with --path-map OLD=NEW.

    python scripts/check_era5_mmd_subset_size.py --prepared-dir data/era5_prepared \
        --run-dir <results of run 1> --run-dir <results of run 2> --output-dir outputs/check_mmd_subset_size \
        --path-map /home/apedalino=/data2/homes/apedalino

Writes results.csv (one row per run, epoch and measure) and summary.json, and prints the per-run objectives.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import re
import sys
import tempfile
from time import perf_counter
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan import pipeline
from physiq_pv.anomaly_detection.stgan.config import STGANCNNConfig
from physiq_pv.anomaly_detection.stgan.mmd import SCALES, multiscale_mmd2
from physiq_pv.anomaly_detection.stgan.normal_masks import cache_identity, load_normal_masks
from physiq_pv.anomaly_detection.stgan.pca_reference import model_features
from physiq_pv.anomaly_detection.stgan.shared_cache import MANIFEST
from scripts.run_era5_stgan import run_training

SIZES = (512, 1024, 2048)  # Subset sizes drawn at random; half of the normal fields and all of them are added.


class _Ready(Exception):
    """Raised when the run is about to build its training loader: everything the MMD reads exists by then."""


def remap(path, path_map):
    """`path` itself when it exists, else with the first matching OLD prefix replaced by NEW (moved home)."""
    if path is None:
        return None
    path = Path(path)
    if path.exists():
        return path
    text = path.as_posix()
    for old, new in path_map:
        old = Path(old).as_posix()
        if text == old or text.startswith(old.rstrip("/") + "/"):
            return Path(Path(new).as_posix() + text[len(old):])
    return path


def run_inputs(run_dir, path_map):
    """What score_era5_stgan_run.scoring_inputs reads, with paths of a moved home mapped to the new one."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "training_only":
        raise ValueError(f"Not a run trained without its final scoring (status {metadata.get('status')!r}): {run_dir}")
    names = {field.name for field in fields(STGANCNNConfig)}
    config = STGANCNNConfig(**{name: value for name, value in metadata["config"].items() if name in names})
    payload = torch.load(run_dir / f"model_epoch_{config.epochs}.pt", map_location="cpu", weights_only=False)
    backend = metadata["backend"]
    directory = lambda key: None if backend.get(key) is None else remap(backend[key]["directory"], path_map)
    seasonal = payload["normalization"].get("seasonal")
    built = None if seasonal is None else remap(Path(seasonal["mean_file"]).parent, path_map)
    for name, path in (("shared cache", built), ("PCA reference", directory("pca_reference")),
                       ("MMD reference", directory("pca_mmd")), ("normal masks", directory("normal_masks"))):
        if path is None or not path.is_dir():
            raise ValueError(f"{run_dir}: {name} not found at {path} (add --path-map OLD=NEW for a moved home)")
    # No resume_from: the epochs' weights are loaded here, so the resume checks on recorded paths do not apply.
    return dict(config=config, seed=int(payload["seed"]), pca_reference_dir=directory("pca_reference"),
                mmd_reference_dir=directory("pca_mmd"), normal_masks_dir=directory("normal_masks"),
                shared_cache_dir=built.parent), built


def comparable(value, path_map):
    """Cache inputs without the times of last change, with moved paths mapped: what must not differ."""
    if isinstance(value, dict):
        return {key: comparable(item, path_map) for key, item in value.items() if key != "modified_ns"}
    if isinstance(value, list):
        return [comparable(item, path_map) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return remap(value, path_map).as_posix()
    return value


def run_cache(directory, path_map):
    """A stand-in for pipeline.shared_cache that returns the run's own cache, never builds one.

    The cache name hashes the absolute paths and the times of last change of the prepared data, which a move
    of the home changes. The run's cache is used only if everything else the run asks for is identical.
    """
    manifest = json.loads((Path(directory) / MANIFEST).read_text(encoding="utf-8"))

    def use_run_cache(root, inputs, build):
        if comparable(inputs, path_map) != comparable(manifest["inputs"], path_map):
            differing = sorted(key for key in set(inputs) | set(manifest["inputs"])
                               if comparable(inputs.get(key), path_map) != comparable(manifest["inputs"].get(key), path_map))
            raise ValueError(f"The run's cache {directory} was built from other inputs: {differing}")
        return Path(directory), manifest["built"], True
    return use_run_cache


def normal_positions(timestamps, share, share_quantile=0.5):
    """Positions of every normal field: share of flagged values at most the quantile of its calendar month."""
    timestamps, share = pd.DatetimeIndex(timestamps), np.asarray(share, dtype=np.float64)
    if len(share) != len(timestamps):
        raise ValueError("One share per timestamp is needed")
    chosen = []
    for month in range(1, 13):
        positions = np.nonzero((timestamps.month == month) & np.isfinite(share))[0]
        if not len(positions):
            raise ValueError(f"No timestamp in month {month}")
        chosen.append(positions[share[positions] <= np.quantile(share[positions], share_quantile)])
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def kernel_matrix(first, second, sigma, scales=SCALES, chunk_size=1024):
    """Mean of the RBF kernels of stgan/mmd.multiscale_mmd2, as a matrix [len(first), len(second)]."""
    exponents = [-1.0 / (2.0 * (scale * sigma) ** 2) for scale in scales]
    out = torch.empty((len(first), len(second)), dtype=torch.float64, device=first.device)
    for start in range(0, len(first), chunk_size):
        squared = torch.cdist(first[start:start + chunk_size], second,
                              compute_mode="donot_use_mm_for_euclid_dist").square()
        out[start:start + chunk_size] = sum(torch.exp(squared * exponent) for exponent in exponents) / len(exponents)
    return out


class PairedMMD:
    """Unbiased multiscale MMD^2 between observed fields and their reconstructions, on any subset of them.

    The kernel matrices are computed once on all the fields; a subset only sums their entries, with the
    estimator of multiscale_mmd2 (observations and reconstructions of the same timestamps).
    """

    def __init__(self, real, fake, sigma, device):
        x = torch.as_tensor(real, dtype=torch.float64, device=device)
        y = torch.as_tensor(fake, dtype=torch.float64, device=device)
        self.xx, self.yy, self.xy = kernel_matrix(x, x, sigma), kernel_matrix(y, y, sigma), kernel_matrix(x, y, sigma)

    def __call__(self, subset):
        index = torch.as_tensor(np.asarray(subset, dtype=np.int64), device=self.xx.device)
        n = len(index)
        if n < 2:
            raise ValueError("A subset needs at least two fields")
        pick = lambda matrix: float(matrix.index_select(0, index).index_select(1, index).sum())
        return (pick(self.xx) - n) / (n * (n - 1)) + (pick(self.yy) - n) / (n * (n - 1)) - 2 * pick(self.xy) / (n * n)


def rebuild(run_dir, prepared_dir, device, scratch, path_map):
    """The run's MMD monitor (dataset, PCA, references) and its model, before any training step."""
    inputs, cache_directory = run_inputs(run_dir, path_map)
    recorded = load_normal_masks(inputs["normal_masks_dir"]).metadata.get("cache")
    captured = {}

    class CapturingMonitor(pipeline.MMDMonitor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["monitor"] = self

    def capturing(model_class):
        class Capturing(model_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured["model"] = self
        Capturing.__name__ = Capturing.__qualname__ = model_class.__name__
        return Capturing

    def same_cache(directory):
        """The masks' cache when it is this one under a moved home: same inputs, same directory once mapped."""
        identity = cache_identity(directory)
        if (recorded is not None and recorded.get("inputs_sha256") == identity["inputs_sha256"]
                and str(remap(recorded["directory"], path_map)) == identity["directory"]):
            return recorded
        return identity

    def stop(*args, **kwargs):
        raise _Ready

    with mock.patch.object(pipeline, "MMDMonitor", CapturingMonitor), \
            mock.patch.object(pipeline, "FullGridSTGAN", capturing(pipeline.FullGridSTGAN)), \
            mock.patch.object(pipeline, "STGAN", capturing(pipeline.STGAN)), \
            mock.patch.object(pipeline, "cache_identity", same_cache), \
            mock.patch.object(pipeline, "shared_cache", run_cache(cache_directory, path_map)), \
            mock.patch.object(pipeline, "make_loader", stop):
        try:
            run_training(prepared_dir=prepared_dir, output_dir=scratch, device=device, skip_final_scoring=True,
                         scored_period="test", **inputs)
        except _Ready:
            pass
    if "monitor" not in captured or "model" not in captured:
        raise RuntimeError(f"{run_dir}: the run did not reach its MMD monitor and its model")
    return captured["monitor"], captured["model"], inputs


def epoch_checkpoints(run_dir, epochs, window):
    """{epoch: path} of the epochs of the objective window that the run still has."""
    found = {int(re.fullmatch(r"model_epoch_(\d+)\.pt", path.name).group(1)): path
             for path in Path(run_dir).glob("model_epoch_*.pt") if re.fullmatch(r"model_epoch_(\d+)\.pt", path.name)}
    wanted = range(max(1, epochs - window + 1), epochs + 1)
    return {epoch: found[epoch] for epoch in wanted if epoch in found}


def check_run(run_dir, prepared_dir, device, draws, seed, scratch_root, path_map=()):
    run_dir = Path(run_dir)
    scratch = Path(tempfile.mkdtemp(prefix="run_", dir=scratch_root))
    started = perf_counter()
    monitor, model, inputs = rebuild(run_dir, prepared_dir, device, scratch, path_map)
    config = inputs["config"]
    window = int(config.mmd_objective_window)
    reference, dataset, pca = monitor.reference, monitor.dataset, monitor.pca
    masks = load_normal_masks(inputs["normal_masks_dir"])
    if len(masks.validation_share) != len(dataset.targets):
        raise ValueError("The normal masks do not cover the validation targets of the run")
    normal = normal_positions(dataset.target_timestamps, masks.validation_share)
    # Every field of the run's 5 x 512 subsets is a normal field: one forward pass covers both.
    if not np.isin(monitor.targets, normal).all():
        raise ValueError("Some fields of the run's subsets are not normal fields by the median rule")
    original = np.searchsorted(normal, monitor.targets)[monitor.inverse]  # [subset, sample] -> rows of `normal`
    history_path = run_dir / "training_history.csv"
    history = pd.read_csv(history_path).set_index("epoch") if history_path.is_file() else None
    checkpoints = epoch_checkpoints(run_dir, int(config.epochs), window)
    sizes = sorted({*(size for size in SIZES if size < len(normal)), len(normal) // 2})
    print(f"[mmd-size] {run_dir} | {len(normal)} normal validation fields | subsets {reference.positions.shape} | "
          f"epochs {sorted(checkpoints)} | sigma {reference.sigma:.6g} | rebuilt in {perf_counter() - started:.0f} s",
          flush=True)
    rows = []
    for epoch, path in checkpoints.items():
        started = perf_counter()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["completed_epochs"]) != epoch:
            raise ValueError(f"{path} is not epoch {epoch}")
        model.load_state_dict(payload["model_state_dict"])
        real, fake = model_features(model, dataset, pca, normal, batch_size=monitor.batch_size, device=monitor.device,
                                    precision=monitor.precision, n_components=reference.n_components)
        if not np.allclose(real[original], reference.real_features, rtol=1e-5, atol=1e-6 * reference.sigma):
            raise ValueError("The observed fields do not reproduce the features saved in the MMD reference")
        distance = PairedMMD(real, fake, reference.sigma, monitor.device)
        # The run's own measure, with its saved observed features, as in MMDMonitor.evaluate.
        recomputed = [multiscale_mmd2(reference.real_features[k], fake[original[k]], reference.sigma)
                      for k in range(len(original))]
        same = [distance(original[k]) for k in range(len(original))]
        if not np.allclose(recomputed, same, rtol=1e-6, atol=1e-9):
            raise ValueError("The kernel-matrix MMD does not reproduce stgan/mmd.multiscale_mmd2")
        logged = (float(history.loc[epoch, "validation_pca_mmd_mean"])
                  if history is not None and epoch in history.index else np.nan)
        base = {"run_dir": str(run_dir), "epoch": epoch}
        rows.append({**base, "measure": "run_5x512", "size": int(original.shape[1]), "draws": len(original),
                     "mean": float(np.mean(recomputed)), "std": float(np.std(recomputed, ddof=1)),
                     "min": float(np.min(recomputed)), "max": float(np.max(recomputed)), "logged": logged})
        rows.append({**base, "measure": "all_normal", "size": len(normal), "draws": 1, "mean": distance(np.arange(len(normal))),
                     "std": np.nan, "min": np.nan, "max": np.nan, "logged": np.nan})
        generator = np.random.Generator(np.random.PCG64(seed))
        for size in sizes:
            values = [distance(generator.choice(len(normal), size, replace=False)) for _ in range(draws)]
            rows.append({**base, "measure": "random_subsets", "size": int(size), "draws": draws,
                         "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                         "min": float(np.min(values)), "max": float(np.max(values)), "logged": np.nan})
        if np.isfinite(logged) and abs(np.mean(recomputed) - logged) > 0.02 * abs(logged):
            print(f"[mmd-size] WARNING epoch {epoch}: the recomputed 5x512 MMD differs from the logged one by more "
                  "than 2%: the rebuild is not the run's own setup, do not use these numbers", flush=True)
        print(f"[mmd-size] epoch {epoch}: run 5x512 {np.mean(recomputed):.5f} (logged {logged:.5f}) | "
              f"all {len(normal)} {rows[-len(sizes) - 1]['mean']:.5f} | {perf_counter() - started:.0f} s", flush=True)
        del distance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model, monitor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, {"run_dir": str(run_dir), "normal_fields": int(len(normal)), "epochs": sorted(checkpoints),
                  "objective_window": window, "sigma": reference.sigma, "sizes": [int(s) for s in sizes]}


def objectives(table):
    """Per run and measure: the mean over the available epochs of the window (the sweep objective)."""
    keyed = table.assign(key=table["measure"] + "_" + table["size"].astype(str))
    grouped = keyed.groupby(["run_dir", "key"])
    return grouped.agg(objective=("mean", "mean"), epochs=("epoch", "nunique"),
                       mean_draw_std=("std", "mean"), logged=("logged", "mean")).reset_index()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, action="append", required=True,
                        help="Results directory of a run trained without final scoring; repeat for several runs")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the results")
    parser.add_argument("--draws", type=int, default=100, help="Random subsets per size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                        help="Prefix of a moved directory recorded in the runs, and where it is now; repeatable")
    args = parser.parse_args(argv)
    path_map = [tuple(entry.split("=", 1)) for entry in args.path_map]
    if any(len(pair) != 2 or not all(pair) for pair in path_map):
        raise ValueError("--path-map takes OLD=NEW")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a new/empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = args.output_dir / "rebuilt_runs"  # The empty run directories the rebuild needs; nothing is kept.
    scratch_root.mkdir()
    rows, runs = [], []
    for run_dir in args.run_dir:
        run_rows, summary = check_run(run_dir, args.prepared_dir, args.device, args.draws, args.seed, scratch_root,
                                      path_map)
        rows += run_rows
        runs.append(summary)
        pd.DataFrame(rows).to_csv(args.output_dir / "results.csv", index=False)  # Kept after every run.
    table = pd.DataFrame(rows)
    summary = objectives(table)
    summary.to_csv(args.output_dir / "objectives.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(
        {"runs": runs, "draws": args.draws, "seed": args.seed, "kernel_scales": list(SCALES),
         "normal_fields": "share of flagged values at most the median of the calendar month (normal_masks rule)",
         "objective": "mean over the available epochs of the run's objective window"}, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.pivot(index="key", columns="run_dir", values="objective"), flush=True)
    print(f"[mmd-size] results in {args.output_dir}", flush=True)
    return table


if __name__ == "__main__":
    main()
