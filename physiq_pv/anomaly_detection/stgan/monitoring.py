"""Per-epoch GAN monitoring on a fixed validation subset.

Diagnostic only. Nothing here enters a loss, an optimizer step, the official
anomaly score or the final evaluation, and no anomaly label is used.
"""
from __future__ import annotations

from contextlib import contextmanager
import random

import numpy as np
from scipy.stats import rankdata
import torch

from .precision import autocast_context

FEATURE_SPACE = "discriminator_penultimate_activations"


@contextmanager
def preserved_rng(device):
    """Give back the Python, NumPy and torch (CPU and this CUDA device) random streams as they were."""
    device = torch.device(device)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


@contextmanager
def undisturbed(model, device):
    """Evaluate without leaving a trace: RNG streams and module modes are restored.

    The forward below is deterministic (dropout off), but the states are saved
    anyway so that monitoring can never shift the training trajectory.
    """
    modes = [(module, module.training) for module in model.modules()]
    with preserved_rng(device):
        try:
            model.eval()
            with torch.no_grad():
                yield
        finally:
            for module, training in modes:
                module.training = training


def evenly_spaced(total, count):
    """At most `count` distinct positions spread over range(total), ends included."""
    if total < 1 or count < 1:
        raise ValueError("Need at least one position.")
    return np.unique(np.linspace(0, total - 1, min(count, total)).round().astype(np.int64))


def feature_discrepancy(real, fake):
    """Mean absolute difference between paired feature vectors: mean_i,k |real[i,k] - fake[i,k]|."""
    return float((real.double() - fake.double()).abs().mean())


def spearman(first, second):
    """Spearman rank correlation with average ranks for ties; 1 for identical rankings."""
    a, b = rankdata(np.asarray(first).ravel()), rankdata(np.asarray(second).ravel())
    if a.shape != b.shape or a.size < 2:
        raise ValueError("Spearman needs two equally long sequences of at least two values.")
    if np.array_equal(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    scale = np.sqrt(np.square(a).sum() * np.square(b).sum())
    return float((a * b).sum() / scale) if scale > 0 else float("nan")


def score_stability(previous, current):
    """How much the same cells, in the same order, changed score between two epochs."""
    previous, current = np.asarray(previous, dtype=np.float64), np.asarray(current, dtype=np.float64)
    if previous.shape != current.shape or previous.ndim != 1:
        raise ValueError("Score stability compares two aligned 1D score vectors.")
    change = np.abs(current - previous)
    return {"score_delta_mean": float(change.mean()), "score_delta_median": float(np.median(change)),
            "score_spearman": spearman(previous, current)}


def rbf_mmd2(first, second, *, bandwidth_points=2048, chunk_size=1024):
    """Unbiased squared MMD between two samples with an RBF kernel.

    k(a, b) = exp(-||a - b||^2 / (2 sigma^2)), with sigma^2 the median squared
    distance between distinct points of the pooled sample (median heuristic,
    on at most `bandwidth_points` evenly spaced pooled points).

    MMD^2 = mean_{i != j} k(x_i, x_j) + mean_{i != j} k(y_i, y_j) - 2 mean_{i, j} k(x_i, y_j)

    The unbiased estimate can be slightly negative when the samples coincide.
    Kernel sums are accumulated in row chunks: RAM is O(chunk_size * n).
    Returns (mmd2, sigma2); sigma2 == 0 means that every point is identical.
    """
    x, y = (torch.as_tensor(values).detach().to("cpu", torch.float64) for values in (first, second))
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or min(len(x), len(y)) < 2:
        raise ValueError("MMD needs two (n, d) samples with at least two rows each.")
    pooled = torch.cat((x, y))
    pooled = pooled[torch.from_numpy(evenly_spaced(len(pooled), bandwidth_points))]
    distances = torch.cdist(pooled, pooled).square()
    sigma2 = float(distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)].median())
    if not sigma2 > 0:
        return 0.0, 0.0

    def kernel_sum(a, b):
        total = 0.0
        for start in range(0, len(a), chunk_size):
            total += float(torch.exp(torch.cdist(a[start:start + chunk_size], b).square() / (-2 * sigma2)).sum())
        return total

    n, m = len(x), len(y)
    return ((kernel_sum(x, x) - n) / (n * (n - 1)) + (kernel_sum(y, y) - m) / (m * (m - 1))
            - 2 * kernel_sum(x, y) / (n * m)), sigma2


class ValidationMonitor:
    """Score stability, D feature discrepancy and MMD on the same validation cells at every epoch.

    The subset is `timestamps` validation targets evenly spaced over the whole
    validation period (first and last included) times every location: it does
    not depend on the seed or on the epoch. One deterministic forward (dropout
    off) gives, per cell, the two components of the anomaly score and D's
    penultimate activations for the observation and for its reconstruction.

    The monitoring score is the same sum of min-max components as the official
    score, but its four factors are fitted once, at the first monitored epoch
    of the run, and reused afterwards so that epochs are comparable. They are
    never used by the final evaluation.
    """

    def __init__(self, dataset, *, timestamps, batch_size, device, precision="fp32",
                 feature_mmd_every_n_epochs=1, feature_mmd_samples=1024, state=None):
        self.dataset, self.batch_size, self.device, self.precision = dataset, batch_size, torch.device(device), precision
        self.feature_mmd_every_n_epochs, self.feature_mmd_samples = feature_mmd_every_n_epochs, feature_mmd_samples
        self.positions = evenly_spaced(len(dataset.targets), timestamps)
        per_target = len(dataset) // len(dataset.targets)  # 1 sample per timestamp, or one per location.
        self.sample_indices = (self.positions[:, None] * per_target + np.arange(per_target)).ravel()
        self.n_cells = len(self.positions) * dataset.n_locations
        self.feature_mmd_rows = evenly_spaced(self.n_cells, feature_mmd_samples)
        self.normalization, self.previous_scores, self.previous_epoch = None, None, None
        if state is not None:
            if not np.array_equal(state["positions"], self.positions) or len(state["previous_scores"]) != self.n_cells:
                raise ValueError("Checkpoint monitoring subset differs from this run.")
            self.normalization = dict(state["normalization"])
            self.previous_scores = np.asarray(state["previous_scores"], dtype=np.float64)
            self.previous_epoch = int(state["previous_epoch"])

    def state(self):
        """What a later epoch needs to stay comparable; stored in epoch checkpoints."""
        if self.normalization is None:
            return None
        return {"positions": self.positions, "normalization": dict(self.normalization),
                "previous_scores": self.previous_scores.copy(), "previous_epoch": self.previous_epoch}

    def metadata(self):
        return {"subset": {"selection": "evenly_spaced_validation_targets_times_all_locations",
                           "timestamps": [str(t) for t in self.dataset.target_timestamps[self.positions]],
                           "n_timestamps": int(len(self.positions)), "n_locations": int(self.dataset.n_locations),
                           "cells": int(self.n_cells)},
                "forward": "single_deterministic_pass_dropout_off",
                "score_normalization": None if self.normalization is None else {
                    **self.normalization, "purpose": "monitoring_only_never_used_by_the_final_score"},
                "feature_space": FEATURE_SPACE,
                "feature_discrepancy": "mean_absolute_difference_of_paired_real_and_reconstructed_features",
                "discriminator_feature_mmd": {"estimator": "unbiased_squared_mmd", "kernel": "rbf", "bandwidth": "median_heuristic_pooled",
                        "every_n_epochs": int(self.feature_mmd_every_n_epochs), "max_samples_per_set": int(self.feature_mmd_samples),
                        "samples_per_set": int(len(self.feature_mmd_rows))},
                "labels_used": False}

    def _components(self, model):
        reconstruction, discriminator, real_rows, fake_rows = [], [], [], []
        absolute, elements, offset = 0.0, 0, 0
        for start in range(0, len(self.sample_indices), self.batch_size):
            batch = self.dataset.fetch_batch(self.sample_indices[start:start + self.batch_size])[:5]
            batch = [values.to(self.device) for values in batch]
            with autocast_context(self.precision, self.device):
                g, d, real, fake = model.monitoring_outputs(*batch)
            real, fake = real.float(), fake.float()
            absolute += float((real.double() - fake.double()).abs().sum())
            elements += real.numel()
            rows = self.feature_mmd_rows[(self.feature_mmd_rows >= offset) & (self.feature_mmd_rows < offset + len(g))] - offset
            real_rows.append(real[rows].cpu())
            fake_rows.append(fake[rows].cpu())
            reconstruction.append(g.double().cpu().numpy())
            discriminator.append(d.double().cpu().numpy())
            offset += len(g)
        reconstruction, discriminator = np.concatenate(reconstruction), np.concatenate(discriminator)
        if offset != self.n_cells or not (np.isfinite(reconstruction).all() and np.isfinite(discriminator).all()):
            raise FloatingPointError("Monitoring outputs are incomplete or non-finite.")
        return reconstruction, discriminator, absolute / elements, torch.cat(real_rows), torch.cat(fake_rows)

    def evaluate(self, model, epoch):
        """Metrics of one epoch, keyed as training-history columns; None where not available."""
        with undisturbed(model, self.device):
            reconstruction, discriminator, discrepancy, real, fake = self._components(model)
        if self.normalization is None:
            self.normalization = {"fitted_epoch": int(epoch)}
            for prefix, values in (("r", reconstruction), ("d", discriminator)):
                self.normalization[prefix + "_min"] = float(values.min())
                self.normalization[prefix + "_max"] = float(values.max())
        ranges = [self.normalization[p + "_max"] - self.normalization[p + "_min"] for p in ("r", "d")]
        scores = sum((values - self.normalization[prefix + "_min"]) / (scale if scale >= 1e-8 else 1.0)
                     for prefix, values, scale in zip(("r", "d"), (reconstruction, discriminator), ranges))
        stability = (score_stability(self.previous_scores, scores) if self.previous_scores is not None
                     else dict.fromkeys(("score_delta_mean", "score_delta_median", "score_spearman")))
        mmd = (rbf_mmd2(real, fake)[0]
               if self.feature_mmd_every_n_epochs and epoch % self.feature_mmd_every_n_epochs == 0 and len(real) >= 2 else None)
        self.previous_scores, self.previous_epoch = scores, int(epoch)
        return {"validation_discriminator_feature_discrepancy": discrepancy,
                **{"validation_" + name: value for name, value in stability.items()}, "validation_discriminator_feature_mmd": mmd}
