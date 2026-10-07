"""Validation MMD between observations and their reconstructions, in the fixed PCA feature space.

The MMD reference (validation subsets, PCA features of their observations, kernel bandwidth) is
built once, before any run, on top of a saved PCA reference (stgan/pca_reference.py), and saved
with checksums. Every run and every epoch loads both, unchanged: nothing in them depends on a
model, a seed or a sweep member. The MMD is a monitoring and model-selection quantity only: it
never enters a loss, an optimizer step, the anomaly score or the evaluation on the test, and it
uses no anomaly label.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .pca_reference import _sha256, model_features

FORMAT = 1
SCALES = (0.5, 1.0, 2.0)  # Kernel bandwidths, as multiples of the frozen sigma.
ARRAYS = ("subset_positions", "subset_timestamps", "real_features")


def choose_subsets(available, n_subsets, size, seed=0):
    """Fixed subsets of range(available): disjoint whenever there are enough samples.

    The draw depends on `seed` alone, never on a run. Returns sorted positions [subset, sample]
    and whether the subsets are disjoint.
    """
    if n_subsets < 1 or size < 2 or size > available:
        raise ValueError("Require n_subsets >= 1 and 2 <= subset size <= available validation samples.")
    generator = np.random.Generator(np.random.PCG64(seed))
    if n_subsets * size <= available:
        positions = generator.permutation(available)[:n_subsets * size].reshape(n_subsets, size)
        return np.sort(positions, axis=1).astype(np.int64), True
    positions = np.stack([generator.choice(available, size, replace=False) for _ in range(n_subsets)])
    return np.sort(positions, axis=1).astype(np.int64), False


def median_sigma(points, chunk_size=1024):
    """Median heuristic: the median Euclidean distance between distinct points [n, k]."""
    points = torch.as_tensor(np.asarray(points), dtype=torch.float64)
    if points.ndim != 2 or len(points) < 2:
        raise ValueError("The bandwidth needs at least two points [n, k].")
    distances = []
    for start in range(0, len(points), chunk_size):
        block = torch.cdist(points[start:start + chunk_size], points, compute_mode="donot_use_mm_for_euclid_dist")
        rows = torch.arange(start, start + len(block))[:, None]
        distances.append(block[torch.arange(len(points))[None, :] > rows])
    sigma = float(np.median(torch.cat(distances).numpy()))
    if not sigma > 0:
        raise ValueError("The reference points coincide: no kernel bandwidth.")
    return sigma


def multiscale_mmd2(first, second, sigma, *, scales=SCALES, chunk_size=1024):
    """Unbiased squared MMD with the mean of RBF kernels of bandwidth scale * sigma.

    k(a, b) = mean_s exp(-||a - b||^2 / (2 (s sigma)^2))
    MMD^2 = mean_{i != j} k(x_i, x_j) + mean_{i != j} k(y_i, y_j) - 2 mean_{i, j} k(x_i, y_j)

    `sigma` is given, never estimated here. Kernel sums are accumulated in row chunks, so the
    memory is O(chunk_size * n). The unbiased estimate can be slightly negative.
    """
    x, y = (torch.as_tensor(np.asarray(values), dtype=torch.float64) for values in (first, second))
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or min(len(x), len(y)) < 2:
        raise ValueError("MMD needs two (n, k) samples with at least two rows each.")
    if not sigma > 0 or not scales:
        raise ValueError("MMD needs a positive bandwidth.")
    exponents = [-1.0 / (2.0 * (scale * sigma) ** 2) for scale in scales]

    def kernel_sum(a, b):
        total = 0.0
        for start in range(0, len(a), chunk_size):
            squared = torch.cdist(a[start:start + chunk_size], b, compute_mode="donot_use_mm_for_euclid_dist").square()
            total += sum(float(torch.exp(squared * exponent).sum()) for exponent in exponents) / len(exponents)
        return total

    n, m = len(x), len(y)
    return (kernel_sum(x, x) - n) / (n * (n - 1)) + (kernel_sum(y, y) - m) / (m * (m - 1)) - 2 * kernel_sum(x, y) / (n * m)


def _fingerprint(files, sigma, pca_fingerprint, n_components):
    content = json.dumps({"format": FORMAT, "files": files, "sigma": repr(float(sigma)),
                          "pca_reference": pca_fingerprint, "n_components": int(n_components)}, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _locations_sha256(latitudes, longitudes, location_names):
    """The checksum of the grid that the PCA reference records."""
    return hashlib.sha256(np.ascontiguousarray(np.stack((latitudes, longitudes)), dtype=np.float64).tobytes()
                          + "\n".join(location_names).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MMDReference:
    """The saved reference: what every run loads, unchanged."""
    directory: Path
    metadata: dict
    positions: np.ndarray      # [subset, sample]: rows of the validation period
    timestamps: np.ndarray     # [subset, sample]: int64 nanoseconds
    real_features: np.ndarray  # [subset, sample, component]: PCA features of the observations

    @property
    def sigma(self):
        return float(self.metadata["bandwidth"]["sigma"])

    @property
    def fingerprint(self):
        return self.metadata["fingerprint"]

    @property
    def pca_fingerprint(self):
        return self.metadata["pca_reference"]["fingerprint"]

    @property
    def n_components(self):
        return int(self.metadata["pca_reference"]["n_components"])

    def summary(self):
        """What a run records about the reference it used."""
        subsets = self.metadata["subsets"]
        return {"directory": str(self.directory), "fingerprint": self.fingerprint, "format": self.metadata["format"],
                "pca_reference": self.pca_fingerprint, "n_components": self.n_components,
                "n_subsets": subsets["n_subsets"], "subset_size": subsets["size"], "subsets_disjoint": subsets["disjoint"],
                "subset_seed": subsets["seed"], "sigma": self.sigma,
                "kernel_bandwidths": [scale * self.sigma for scale in SCALES]}


def load_mmd_reference(directory):
    """Load a reference and verify every file against the checksum recorded when it was built."""
    directory = Path(directory)
    path = directory / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"No MMD reference in {directory}. It is never built by a run: build it once with "
                                "`run_era5_stgan.py mmd-reference`.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format") != FORMAT or set(metadata.get("files", {})) != {name + ".npy" for name in ARRAYS}:
        raise ValueError(f"Unsupported MMD reference format in {directory}.")
    for name, expected in metadata["files"].items():
        if not (directory / name).is_file() or _sha256(directory / name) != expected:
            raise ValueError(f"MMD reference file {name} is missing or does not match its checksum.")
    if _fingerprint(metadata["files"], metadata["bandwidth"]["sigma"], metadata["pca_reference"]["fingerprint"],
                    metadata["pca_reference"]["n_components"]) != metadata["fingerprint"]:
        raise ValueError("MMD reference metadata do not match their fingerprint.")
    arrays = {name: np.load(directory / (name + ".npy")) for name in ARRAYS}
    return MMDReference(directory=directory.resolve(), metadata=metadata, positions=arrays["subset_positions"],
                        timestamps=arrays["subset_timestamps"], real_features=arrays["real_features"])


def build_mmd_reference(validation, validation_timestamps, pca, *, feature_names, location_names, latitudes,
                        longitudes, output_dir, n_subsets=5, subset_size=512, subset_seed=0, n_components=100,
                        block=64):
    """Build and save the reference from the validation array [time, location, variable] and a PCA reference.

    A sample is the complete field of one validation timestamp. The subsets are fixed validation
    timestamps; their observations go through the saved normalization and the first `n_components`
    saved components of `pca`, which is only read; the bandwidth is the median distance between
    those PCA features, all subsets pooled. No training or test data, no model and no anomaly
    label is involved, and the PCA is not fitted again.
    """
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty MMD reference directory: an existing reference is loaded, never rebuilt.")
    validation_timestamps = pd.DatetimeIndex(validation_timestamps)
    if validation.ndim != 3 or len(validation) != len(validation_timestamps):
        raise ValueError("Validation data must be [time, location, variable] with one timestamp per row.")
    locations, variables = validation.shape[1:]
    if (locations, variables) != (len(location_names), len(feature_names)):
        raise ValueError("Location and feature names do not match the data.")
    data = pca.metadata["data"]
    if (tuple(data["features"]) != tuple(feature_names) or data["n_locations"] != locations
            or data["locations_sha256"] != _locations_sha256(latitudes, longitudes, location_names)):
        raise ValueError(f"PCA reference {pca.directory} has other variables or locations than the validation data.")
    if not pd.Timestamp(data["train"]["end"]) < validation_timestamps[0]:
        raise ValueError("The validation must follow the training period of the PCA reference.")
    if type(n_components) is not int or not 1 <= n_components <= len(pca.components):
        raise ValueError(f"n_components must be an integer between 1 and {len(pca.components)}, "
                         "the components stored in the PCA reference.")
    positions, disjoint = choose_subsets(len(validation), n_subsets, subset_size, subset_seed)
    rows, inverse = np.unique(positions, return_inverse=True)
    print(f"[mmd-reference] {n_subsets}x{subset_size} validation fields of {locations}x{variables} values, "
          f"{n_components} PCA components of {pca.fingerprint[:16]}", flush=True)
    features = np.empty((len(rows), n_components), dtype=np.float64)
    for start in range(0, len(rows), block):
        values = np.stack([np.asarray(validation[int(row)], dtype=np.float32) for row in rows[start:start + block]])
        if not np.isfinite(values).all():
            raise ValueError("Non-finite values in the validation data.")
        features[start:start + block] = pca.transform_raw(values, n_components)
    real_features = features[inverse.reshape(positions.shape)]
    sigma = median_sigma(real_features.reshape(-1, n_components))

    output.mkdir(parents=True, exist_ok=True)
    arrays = {"subset_positions": positions, "subset_timestamps": validation_timestamps.as_unit("ns").asi8[positions],
              "real_features": real_features}
    files = {}
    for name in ARRAYS:
        np.save(output / (name + ".npy"), arrays[name])
        files[name + ".npy"] = _sha256(output / (name + ".npy"))
    metadata = {
        "format": FORMAT,
        "sample": "one_complete_field_per_validation_timestamp",
        "data": {"features": list(feature_names), "n_locations": int(locations),
                 "locations_sha256": data["locations_sha256"],
                 "validation": {"start": str(validation_timestamps[0]), "end": str(validation_timestamps[-1]),
                                "timestamps": int(len(validation_timestamps))}},
        "pca_reference": {"fingerprint": pca.fingerprint, "stored_components": int(len(pca.components)),
                          "n_components": int(n_components),
                          "cumulative_explained_variance": float(pca.cumulative_explained_variance[n_components - 1]),
                          "transformation": "saved_normalization_then_projection_on_the_leading_saved_components",
                          "refitted": False},
        "subsets": {"from": "validation", "n_subsets": int(n_subsets), "size": int(subset_size), "seed": int(subset_seed),
                    "disjoint": bool(disjoint), "available_timestamps": int(len(validation_timestamps)),
                    "selection": "fixed_seed_permutation_of_the_validation_timestamps_cut_into_subsets"},
        "bandwidth": {"sigma": sigma, "estimator": "median_euclidean_distance_between_the_pooled_real_pca_features_of_the_subsets",
                      "points": int(real_features.shape[0] * real_features.shape[1]), "fitted_on": "observations_only",
                      "kernel": "mean_of_rbf_kernels", "scales": list(SCALES),
                      "bandwidths": [scale * sigma for scale in SCALES]},
        "estimator": "unbiased_squared_mmd_per_subset_between_observations_and_their_reconstructions",
        "training_data_used": False, "test_data_used": False, "labels_used": False, "model_used": False,
        "files": files, "fingerprint": _fingerprint(files, sigma, pca.fingerprint, n_components)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    reference = load_mmd_reference(output)
    print(f"[mmd-reference] subsets={n_subsets}x{subset_size} disjoint={disjoint} components={n_components} "
          f"sigma={sigma:.6g} fingerprint={reference.fingerprint}", flush=True)
    return reference


class MMDMonitor:
    """MMD^2 between the observations of the fixed validation subsets and their reconstructions by G.

    One deterministic forward of G (dropout off) per sample and epoch. Observations and
    reconstructions are in the model's normalized space, which is the PCA reference's, and go
    through the one transformation of `pca_reference.model_features`. The features of the
    observations are checked once against the saved ones, which are then used: the bandwidth and
    the real features are those of the reference, never recomputed.
    """

    def __init__(self, dataset, reference, pca, *, batch_size, device, precision="fp32"):
        self.dataset, self.reference, self.pca, self.batch_size = dataset, reference, pca, batch_size
        self.device, self.precision = torch.device(device), precision
        problems = []
        if reference.pca_fingerprint != pca.fingerprint or reference.n_components > len(pca.components):
            problems.append("PCA reference (the MMD reference was built on another one)")
        targets = pd.DatetimeIndex(dataset.target_timestamps).as_unit("ns").asi8
        rows = np.searchsorted(targets, reference.timestamps)
        if (rows >= len(targets)).any() or not np.array_equal(targets[np.minimum(rows, len(targets) - 1)],
                                                                reference.timestamps):
            problems.append("validation timestamps")
        if problems:
            raise ValueError(f"MMD reference {reference.directory} does not match this run: {problems}")
        self.targets, self.inverse = np.unique(rows, return_inverse=True)
        self.inverse = self.inverse.reshape(rows.shape)
        self.real_checked, self.subset_values = False, None

    def evaluate(self, model):
        """Mean and variance (ddof=1) over the subsets of the unbiased multiscale MMD^2 of this model."""
        real, fake = model_features(model, self.dataset, self.pca, self.targets, batch_size=self.batch_size,
                                    device=self.device, precision=self.precision,
                                    n_components=self.reference.n_components)
        saved = self.reference.real_features
        if not self.real_checked:
            if not np.allclose(real[self.inverse], saved, rtol=1e-5, atol=1e-6 * self.reference.sigma):
                raise ValueError("Validation observations of this run do not reproduce the MMD reference features.")
            self.real_checked = True
        self.subset_values = [multiscale_mmd2(saved[subset], fake[self.inverse[subset]], self.reference.sigma)
                              for subset in range(len(saved))]
        return {"validation_pca_mmd_mean": float(np.mean(self.subset_values)),
                "validation_pca_mmd_variance": float(np.var(self.subset_values, ddof=1)) if len(saved) > 1 else 0.0}
