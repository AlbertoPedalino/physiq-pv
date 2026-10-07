"""A fixed PCA feature space of complete fields, fitted once on training data.

Each sample is one timestamp: the field H x W x C of every location and variable, normalized
per variable with the training-only factors of the pipeline and flattened. The PCA is fitted on
training fields alone and saved with checksums; runs load it and never refit it. Nothing here
depends on a model, a seed or a sweep member, and nothing enters a loss, the anomaly score or
the evaluation on the test. The number of components to use is not chosen here: the leading
components and their explained variance are stored so that it can be chosen afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .monitoring import evenly_spaced, undisturbed
from .precision import autocast_context

FORMAT = 1
MARKS = (50, 75, 100, 150, 200, 250, 300)  # Numbers of components whose cumulative explained variance is reported.
THRESHOLDS = (.80, .85, .90, .95)  # Cumulative explained variance whose smallest number of components is reported.
ARRAYS = ("scaler_minimum", "scaler_scale", "pca_mean", "pca_components", "pca_explained_variance",
          "pca_explained_variance_ratio", "pca_fit_timestamps")
PLOT = "pca_cumulative_explained_variance.png"


def normalise(values, minimum, scale):
    """Per-variable min-max to [-1, 1] with training-only factors: the model's own input normalization."""
    return ((np.asarray(values, dtype=np.float32) - minimum) / scale * 2.0 - 1.0).astype(np.float32, copy=False)


def fit_pca(samples, *, n_components=100, block=256):
    """Exact PCA of the rows of `samples` [M, D], M << D, from the eigen-decomposition of their Gram matrix.

    `samples` is centred in place. At most `n_components` leading components are kept, fewer when
    the rank of the data is lower. The explained-variance ratio is relative to the total variance
    of the samples. Returns mean [D], components [k, D] (orthonormal rows), explained variance and
    its ratio for the k components.
    """
    count, size = samples.shape
    if count < 3 or n_components < 1:
        raise ValueError("PCA needs at least three samples and one component.")
    mean = samples.mean(axis=0, dtype=np.float64).astype(np.float32)
    samples -= mean
    gram = np.empty((count, count), dtype=np.float64)
    for first in range(0, count, block):
        rows = samples[first:first + block].astype(np.float64)
        for second in range(0, first + block, block):
            product = rows @ samples[second:second + block].astype(np.float64).T
            gram[first:first + block, second:second + block] = product
            gram[second:second + block, first:first + block] = product.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    eigenvalues, vectors = np.clip(eigenvalues[::-1], 0.0, None), vectors[:, ::-1]
    if not eigenvalues[0] > 0:
        raise ValueError("PCA samples are constant.")
    rank = int(min(count - 1, size, (eigenvalues > eigenvalues[0] * 1e-10).sum()))
    kept = min(int(n_components), rank)
    components = np.zeros((kept, size), dtype=np.float64)
    for first in range(0, count, block):
        components += vectors[first:first + block, :kept].T @ samples[first:first + block].astype(np.float64)
    components /= np.sqrt(eigenvalues[:kept])[:, None]
    # A fixed sign: the entry of largest magnitude of every component is positive.
    largest = np.abs(components).argmax(axis=1)
    components *= np.sign(components[np.arange(kept), largest])[:, None]
    return {"mean": mean, "components": components.astype(np.float32), "rank": rank,
            "explained_variance": eigenvalues[:kept] / (count - 1),
            "explained_variance_ratio": eigenvalues[:kept] / eigenvalues.sum()}


def components_for_variance(explained_variance_ratio, thresholds=THRESHOLDS):
    """Smallest number of leading components whose cumulative explained variance reaches each threshold.

    None where the given components do not reach it. A report: no number of components is chosen.
    """
    cumulative = np.cumsum(np.asarray(explained_variance_ratio, dtype=np.float64))
    return {f"{threshold:.2f}": (int(index) + 1 if index < len(cumulative) else None)
            for threshold, index in zip(thresholds, np.searchsorted(cumulative, thresholds, side="left"))}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(files):
    return hashlib.sha256(json.dumps({"format": FORMAT, "files": files}, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PCAReference:
    """The saved feature space: what every run loads, unchanged."""
    directory: Path
    metadata: dict
    minimum: np.ndarray                   # [variable]
    scale: np.ndarray                     # [variable]
    mean: np.ndarray                      # [location, variable]
    components: np.ndarray                # [component, location, variable]
    explained_variance_ratio: np.ndarray  # [component]

    @property
    def fingerprint(self):
        return self.metadata["fingerprint"]

    @property
    def cumulative_explained_variance(self):
        return np.cumsum(self.explained_variance_ratio)

    def transform(self, fields, n_components=None):
        """PCA features [..., k] of normalized fields [..., location, variable].

        The one transformation of observations and of reconstructions alike: flatten as
        location x variable, subtract the saved mean, project on the first `n_components`
        saved components (all of them when omitted).
        """
        fields = np.asarray(fields, dtype=np.float64)
        if fields.shape[-2:] != self.mean.shape:
            raise ValueError(f"Fields must end in {self.mean.shape} (location, variable).")
        kept = len(self.components) if n_components is None else int(n_components)
        if not 1 <= kept <= len(self.components):
            raise ValueError(f"n_components must be between 1 and {len(self.components)}.")
        flat = (fields - self.mean).reshape(*fields.shape[:-2], -1)
        return flat @ self.components[:kept].reshape(kept, -1).astype(np.float64).T

    def transform_raw(self, fields, n_components=None):
        """PCA features of fields in physical units: the saved normalization first, then `transform`."""
        return self.transform(normalise(fields, self.minimum, self.scale), n_components)

    def check(self, *, minimum, scale, feature_names, n_locations):
        """Fail unless a run has the variables, locations and training-only normalization of the reference."""
        problems = []
        if tuple(self.metadata["data"]["features"]) != tuple(feature_names):
            problems.append("features")
        if self.metadata["data"]["n_locations"] != n_locations:
            problems.append("locations")
        if (np.shape(minimum) != self.minimum.shape or not np.array_equal(minimum, self.minimum)
                or not np.array_equal(scale, self.scale)):
            problems.append("normalization (training data or normalization kind)")
        if problems:
            raise ValueError(f"PCA reference {self.directory} does not match this run: {problems}")

    def summary(self):
        """What a run records about the reference it loaded."""
        pca = self.metadata["pca"]
        return {"directory": str(self.directory), "fingerprint": self.fingerprint, "format": self.metadata["format"],
                "stored_components": pca["stored_components"], "fit_samples": pca["fit_samples"],
                "dimension": pca["dimension"],
                "cumulative_explained_variance_at": pca["cumulative_explained_variance_at"],
                # None: not reached by the stored components.
                "components_for_cumulative_explained_variance": components_for_variance(self.explained_variance_ratio),
                "chosen_components": None}  # The number of components to use is not decided yet.


def load_pca_reference(directory):
    """Load a reference and verify every file against the checksum recorded when it was built."""
    directory = Path(directory)
    path = directory / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"No PCA reference in {directory}. It is never fitted by a run: build it once with "
                                "`run_era5_stgan.py pca-reference`.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format") != FORMAT or set(metadata.get("files", {})) != {name + ".npy" for name in ARRAYS}:
        raise ValueError(f"Unsupported PCA reference format in {directory}.")
    for name, expected in metadata["files"].items():
        if not (directory / name).is_file() or _sha256(directory / name) != expected:
            raise ValueError(f"PCA reference file {name} is missing or does not match its checksum.")
    if _fingerprint(metadata["files"]) != metadata["fingerprint"]:
        raise ValueError("PCA reference metadata do not match their fingerprint.")
    arrays = {name: np.load(directory / (name + ".npy")) for name in ARRAYS}
    locations, variables = metadata["data"]["n_locations"], len(metadata["data"]["features"])
    return PCAReference(
        directory=directory.resolve(), metadata=metadata, minimum=arrays["scaler_minimum"], scale=arrays["scaler_scale"],
        mean=arrays["pca_mean"].reshape(locations, variables),
        components=arrays["pca_components"].reshape(-1, locations, variables),
        explained_variance_ratio=arrays["pca_explained_variance_ratio"])


def plot_cumulative_variance(explained_variance_ratio, path, marks=MARKS):
    """Save the cumulative explained variance against the number of components; return the marked values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cumulative = np.cumsum(np.asarray(explained_variance_ratio, dtype=np.float64))
    numbers = np.arange(1, len(cumulative) + 1)
    marked = {int(mark): float(cumulative[mark - 1]) for mark in marks if mark <= len(cumulative)}
    figure, axis = plt.subplots(figsize=(8, 4.8), dpi=150)
    axis.plot(numbers, cumulative, color="#2a5d9f", linewidth=2)
    for mark, value in marked.items():
        axis.plot([mark, mark], [0, value], color="#8a8a8a", linewidth=.8, linestyle=":")
        axis.plot([mark], [value], marker="o", markersize=7, color="#2a5d9f", markeredgecolor="white", markeredgewidth=1.5)
        # The value alone, below the curve: the tick under the dotted line gives the number of components.
        axis.annotate(f"{value:.4f}", (mark, value), textcoords="offset points", xytext=(5, -13),
                      ha="left", fontsize=8, color="#222222")
    axis.set_xlabel("Numero componenti PCA")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_title("Varianza spiegata cumulata della PCA (fit sul solo training)", fontsize=11, loc="left")
    axis.set_xlim(1, max(len(cumulative), 2) + 1)
    axis.set_ylim(0, 1.02)
    axis.set_xticks([tick for tick in (1, 25, *MARKS) if tick <= len(cumulative)])
    axis.grid(color="#dddddd", linewidth=.6)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return marked


def build_pca_reference(train, train_timestamps, *, feature_names, location_names, latitudes, longitudes,
                        output_dir, pca_samples=4096, n_components=100):
    """Fit and save the reference from the training array [time, location, variable] alone.

    Normalization: the training-only per-variable min-max of every run. PCA: exact, on
    `pca_samples` complete training fields evenly spaced over the whole training period (first and
    last timestamp included); their timestamps are saved. No validation or test data, no model
    and no anomaly label is involved.
    """
    from .pipeline import _feature_minmax
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty PCA reference directory: an existing reference is loaded, never refitted.")
    train_timestamps = pd.DatetimeIndex(train_timestamps)
    if train.ndim != 3 or len(train) != len(train_timestamps):
        raise ValueError("Training data must be [time, location, variable] with one timestamp per row.")
    locations, variables = train.shape[1:]
    if (locations, variables) != (len(location_names), len(feature_names)):
        raise ValueError("Location and feature names do not match the data.")
    print(f"[pca-reference] normalization: min-max of {len(train)} training timestamps", flush=True)
    minimum, scale = _feature_minmax(train)
    rows = evenly_spaced(len(train), pca_samples)
    print(f"[pca-reference] PCA: {len(rows)} training fields of {locations}x{variables} values", flush=True)
    samples = np.empty((len(rows), locations * variables), dtype=np.float32)
    for index, row in enumerate(rows):
        values = np.asarray(train[int(row)], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite values in the training data.")
        samples[index] = normalise(values, minimum, scale).reshape(-1)
    pca = fit_pca(samples, n_components=n_components)
    del samples

    output.mkdir(parents=True, exist_ok=True)
    arrays = {"scaler_minimum": minimum, "scaler_scale": scale, "pca_mean": pca["mean"],
              "pca_components": pca["components"], "pca_explained_variance": pca["explained_variance"],
              "pca_explained_variance_ratio": pca["explained_variance_ratio"],
              "pca_fit_timestamps": train_timestamps.as_unit("ns").asi8[rows]}
    files = {}
    for name in ARRAYS:
        np.save(output / (name + ".npy"), arrays[name])
        files[name + ".npy"] = _sha256(output / (name + ".npy"))
    cumulative = np.cumsum(pca["explained_variance_ratio"])
    height, width = len(np.unique(latitudes)), len(np.unique(longitudes))
    coordinates = hashlib.sha256(np.ascontiguousarray(np.stack((latitudes, longitudes)), dtype=np.float64).tobytes()
                                 + "\n".join(location_names).encode("utf-8")).hexdigest()
    metadata = {
        "format": FORMAT,
        "sample": "one_complete_field_per_timestamp",
        "data": {"features": list(feature_names), "n_locations": int(locations), "locations_sha256": coordinates,
                 "field_shape_height_width_channels": [height, width, int(variables)],
                 "complete_grid": bool(height * width == locations),
                 "flatten_order": "location_then_variable", "flattened_dimension": int(locations * variables),
                 "train": {"start": str(train_timestamps[0]), "end": str(train_timestamps[-1]),
                           "timestamps": int(len(train_timestamps))}},
        "normalization": {"kind": "training_only_feature_minmax_to_minus_one_one", "fit_on": "train",
                          "fit_timestamps": int(len(train_timestamps)), "minimum": minimum.tolist(),
                          "scale": scale.tolist()},
        "pca": {"method": "exact_pca_by_eigendecomposition_of_the_gram_matrix_of_the_fit_samples",
                "fit_on": "train", "fit_samples": int(len(rows)), "requested_fit_samples": int(pca_samples),
                "fit_selection": "training_timestamps_evenly_spaced_over_the_whole_training_period_ends_included",
                "fit_fraction_of_training": float(len(rows) / len(train_timestamps)),
                "dimension": int(locations * variables), "rank": pca["rank"],
                "stored_components": int(len(pca["components"])), "requested_components": int(n_components),
                "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
                "cumulative_explained_variance": cumulative.tolist(),
                "cumulative_explained_variance_at": {str(mark): (float(cumulative[mark - 1]) if mark <= len(cumulative)
                                                                 else None) for mark in MARKS},
                "components_for_cumulative_explained_variance": components_for_variance(pca["explained_variance_ratio"]),
                "chosen_components": None},
        "validation_data_used": False, "test_data_used": False, "labels_used": False, "model_used": False,
        "files": files, "fingerprint": _fingerprint(files)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_cumulative_variance(pca["explained_variance_ratio"], output / PLOT)
    reference = load_pca_reference(output)
    print(f"[pca-reference] samples={len(rows)} dimension={locations * variables} "
          f"components={len(reference.components)} cumulative_explained_variance="
          f"{metadata['pca']['cumulative_explained_variance_at']} fingerprint={reference.fingerprint}", flush=True)
    for threshold, count in metadata["pca"]["components_for_cumulative_explained_variance"].items():
        print(f"[pca-reference] cumulative explained variance >= {threshold}: "
              + (f"{count} components" if count else f"not reached within the {len(reference.components)} stored"),
              flush=True)
    return reference


def model_features(model, dataset, reference, targets, *, batch_size, device="cpu", precision="fp32",
                   n_components=None):
    """PCA features [target, k] of the observations of `dataset` and of their reconstructions by G.

    `targets` are positions among the dataset's target timestamps. One deterministic forward of G
    (dropout off) per sample; random streams and module modes are restored. Observed and
    reconstructed cells go through the same projection, cell by cell: the result is
    `reference.transform` of the complete field of each timestamp, for either architecture.
    """
    device, targets = torch.device(device), np.asarray(targets, dtype=np.int64)
    kept = len(reference.components) if n_components is None else int(n_components)
    if not 1 <= kept <= len(reference.components) or len(np.unique(targets)) != len(targets):
        raise ValueError("Require distinct targets and a valid number of components.")
    per_target = len(dataset) // len(dataset.targets)  # 1 sample per timestamp, or one per location.
    samples = (targets[:, None] * per_target + np.arange(per_target)).ravel()
    row_of_target = np.full(len(dataset.targets), -1, dtype=np.int64)
    row_of_target[targets] = np.arange(len(targets))
    mean = torch.as_tensor(reference.mean, dtype=torch.float64, device=device)
    components = torch.as_tensor(reference.components[:kept], dtype=torch.float64, device=device)
    features = torch.zeros((2, len(targets), kept), dtype=torch.float64, device=device)
    cells = torch.zeros(len(targets), dtype=torch.int64, device=device)
    with undisturbed(model, device):
        for start in range(0, len(samples), batch_size):
            batch = dataset.fetch_batch(samples[start:start + batch_size])
            time = torch.as_tensor(row_of_target[batch[-2].numpy().reshape(-1)], device=device)
            location = batch[-1].reshape(-1).to(device)
            with autocast_context(precision, device):
                pair = model.reconstructed_cells(*(values.to(device) for values in batch[:5]))
            for index, values in enumerate(pair):  # Observed, then reconstructed: one projection for both.
                features[index].index_add_(0, time, torch.einsum(
                    "pf,kpf->pk", values.double() - mean[location], components[:, location]))
            cells.index_add_(0, time, torch.ones_like(time))
    if not bool((cells == dataset.n_locations).all()) or not bool(torch.isfinite(features).all()):
        raise FloatingPointError("Fields are incomplete or non-finite.")
    real, reconstructed = features.cpu().numpy()
    return real, reconstructed
