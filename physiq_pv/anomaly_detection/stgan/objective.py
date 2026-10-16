"""Validation statistics of the MC-mean reconstruction error, computed once after training.

Diagnostic output only: nothing here enters a loss, an optimizer step, the
official anomaly score or the final evaluation, and no anomaly label is used.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from .loading import close_loader, make_loader
from .monitoring import preserved_rng
from .precision import autocast_context, validate_precision
from .scoring import PROGRESS_SECONDS, component_statistics, scoring_mode

STATISTICS = ("mean", "std", "median", "p95", "median_plus_p95")
COUNTS = ("candidate_points", "valid_points", "excluded_points")


def reconstruction_statistics(values, valid, *, chunk_size=65536):
    """Mean, population std, median, P95 and median + P95 over the valid points only.

    A point is one (timestamp, cell). An invalid point is left out of the
    distribution altogether, whatever sits in its slot (zero, a huge number,
    NaN): nothing is imputed for it. Every included value must be finite.
    Returns the statistics with the number of candidate, valid and excluded points.
    """
    values, valid = np.asarray(values), np.asarray(valid)
    if valid.dtype != np.bool_ or valid.shape != values.shape:
        raise ValueError("valid must be a boolean array with the shape of values.")
    included = values[valid]
    counts = {"candidate_points": int(valid.size), "valid_points": int(included.size),
              "excluded_points": int(valid.size - included.size)}
    if not included.size:
        raise ValueError("No valid validation point: the reconstruction statistics are undefined.")
    if not np.isfinite(included).all():
        raise FloatingPointError("Non-finite reconstruction error at a valid validation point.")
    statistics = component_statistics(included, chunk_size=chunk_size)
    return {**counts, **{name: statistics[name] for name in STATISTICS[:4]},
            "median_plus_p95": statistics["median"] + statistics["p95"]}


def validation_objective(model, dataset, *, batch_size, device, mc_dropout_enabled, mc_samples,
                         loader_options=None, share_history=True, precision="fp32", chunk_size=65536,
                         excluded=None, error_path=None, fit_score_ranges=False):
    """Statistics of the reconstruction error of every validation point of `dataset`.

    The error is the reconstruction component of the anomaly score before any
    min-max: the squared error averaged over the cells of the mask that the
    score itself uses. With MC Dropout every draw uses that same mask, its
    error is computed on the valid cells only, and the draws are then averaged.
    A point whose mask has no cell is not a zero: it is excluded
    (`model.reconstruction_valid`), and counted. The random streams and the
    module modes are restored, so the test scoring that follows is unchanged.

    `excluded` [timestamp, location] marks points of unusual weather (normal_masks.py): they are
    left out of the statistics and counted apart. With `error_path` the MC-mean error of every
    point (NaN where not valid) is saved there, flagged points included.

    With `fit_score_ranges` the same draws also give the ranges of the two score components for
    the test (score_mode=validation): the min-max of the paper over the MC-mean reconstruction
    error and the MC-mean discriminator component D(real) - D(generated), on the points of the
    statistics (valid, not flagged). They are returned as `score_ranges`.
    """
    validate_precision(precision, device)
    samples = mc_samples if mc_dropout_enabled else 1
    shape = (len(dataset.targets), dataset.n_locations)
    if excluded is not None and (np.asarray(excluded).dtype != np.bool_ or np.shape(excluded) != shape):
        raise ValueError(f"excluded must be a boolean array {shape}")
    mean_error = np.full(shape, np.nan, dtype=np.float32)  # NaN marks a slot that is not in the distribution.
    mean_gap = np.full(shape, np.nan, dtype=np.float32) if fit_score_ranges else None
    valid = np.zeros(shape, dtype=np.bool_)
    started, loader, visited = perf_counter(), None, 0
    try:
        loader = make_loader(dataset, batch_size=batch_size, shuffle=False, device=device, **(loader_options or {}))
        total, logged = len(loader), None
        with preserved_rng(device), scoring_mode(model, mc_dropout_enabled), torch.no_grad():
            for index, batch in enumerate(loader, start=1):
                recent, trend, mask, calendar, observed = (x.to(device, non_blocking=True) for x in batch[:5])
                with autocast_context(precision, device):
                    draws = model.score_draws(recent, trend, mask, calendar, observed, samples,
                                              share_history=share_history)
                usable = model.reconstruction_valid(mask).cpu().numpy()
                errors = draws[:, :, 0].float().cpu().numpy().astype(np.float64)
                if not np.isfinite(errors[:, usable]).all():
                    raise FloatingPointError("Non-finite reconstruction error at a valid validation point.")
                time, location = batch[-2].numpy().reshape(-1), batch[-1].numpy().reshape(-1)
                mean_error[time, location] = np.where(usable, errors.mean(axis=0), np.nan)
                if mean_gap is not None:
                    gaps = draws[:, :, 1].float().cpu().numpy().astype(np.float64)
                    if not np.isfinite(gaps[:, usable]).all():
                        raise FloatingPointError("Non-finite discriminator component at a valid validation point.")
                    mean_gap[time, location] = np.where(usable, gaps.mean(axis=0), np.nan)
                valid[time, location] = usable
                visited += len(usable)
                now = perf_counter()
                if logged is None or now - logged >= PROGRESS_SECONDS or index == total:
                    logged, elapsed = now, now - started
                    print(f"[stgan] validation objective batch={index}/{total} mc_samples={samples} "
                          f"elapsed={elapsed / 3600:.2f}h "
                          f"eta={elapsed / index * (total - index) / 3600:.2f}h", flush=True)
    finally:
        close_loader(loader)
    if visited != valid.size:
        raise RuntimeError("Validation objective did not visit every validation point exactly once.")
    if error_path is not None:
        np.save(error_path, mean_error)
    flagged = {}
    if excluded is not None:
        flagged = {"flagged_points_left_out": int((valid & excluded).sum())}
        valid &= ~excluded
    result = {"quantity": "mc_mean_reconstruction_error_before_minmax", "split": "validation",
              "start": str(dataset.target_timestamps[0]), "end": str(dataset.target_timestamps[-1]),
              "timestamps": int(shape[0]), "locations": int(shape[1]), "effective_mc_samples": int(samples),
              "validity": "points_with_at_least_one_cell_in_the_reconstruction_mask",
              **reconstruction_statistics(mean_error, valid, chunk_size=chunk_size), **flagged,
              "seconds": perf_counter() - started}
    if mean_gap is not None:
        # Same points as the statistics above: valid, flagged ones left out.
        result["score_ranges"] = {
            "r_min": float(mean_error[valid].min()), "r_max": float(mean_error[valid].max()),
            "d_min": float(mean_gap[valid].min()), "d_max": float(mean_gap[valid].max()),
            "points": int(valid.sum())}
    print("[stgan] validation objective: " + " ".join(
        f"{name}={result[name]}" for name in COUNTS) + " " + " ".join(
        f"{name}={result[name]:.6g}" for name in STATISTICS), flush=True)
    return result
