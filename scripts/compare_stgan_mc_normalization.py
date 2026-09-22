"""Synthetic-only A/B audit of MC score normalization and ERA5 storage estimates.

No training, external datasets, or changes to production normalization.
B uses one pooled min/max per component over (M,T,N), held fixed across draws.
This historical same-population A/B experiment is not the production protocol:
production fits a separate calibration period and freezes its ranges for test.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from physiq_pv.anomaly_detection.stgan.scoring import fit_calibration_ranges, ScoreStore, normalize_mc_scores


def compare_components(generator, discriminator):
    """Historical same-population sensitivity comparison, not holdout evaluation."""
    if generator.shape != discriminator.shape or generator.ndim != 3:
        raise ValueError("Expected matching (M,T,N) components")
    if not np.isfinite(generator).all() or not np.isfinite(discriminator).all():
        raise ValueError("Comparison requires finite components")
    store = ScoreStore()
    try:
        # Experimental historical calculation, not an option in production.
        a_scores = []
        for g, d in zip(generator, discriminator):
            normalized = []
            for raw in (g, d):
                scale = float(raw.max()-raw.min())
                normalized.append((raw-float(raw.min()))/(scale if scale >= 1e-8 else 1.0))
            a_scores.append(normalized[0]+normalized[1])
        a_scores = np.asarray(a_scores, dtype=np.float64)
        a_mean, a_std = a_scores.mean(0), a_scores.std(0, ddof=0)
        b_mean, b_std, *_ = normalize_mc_scores(generator, discriminator, store, normalization=fit_calibration_ranges(generator, discriminator))
        ar, br = rankdata(-a_mean.ravel()), rankdata(-b_mean.ravel())
        count = a_mean.size
        k = max(1, int(np.ceil(count*.01)))
        # Stable flat-index tie breaking keeps the top 1% count exact.
        a_top = np.argsort(-a_mean.ravel(), kind="stable")[:k]
        b_top = np.argsort(-b_mean.ravel(), kind="stable")[:k]
        overlap = int(np.intersect1d(a_top, b_top).size)
        rho = float(spearmanr(ar, br).statistic) if np.ptp(ar) and np.ptp(br) else None
        mean_delta, std_delta = np.abs(a_mean-b_mean), np.abs(a_std-b_std)
        metrics = dict(samples=generator.shape[0], points=count,
            anomaly_mean_A=float(a_mean.mean(dtype=np.float64)), anomaly_mean_B=float(b_mean.mean()),
            anomaly_mean_mae=float(mean_delta.mean()), anomaly_mean_max_abs=float(mean_delta.max()),
            anomaly_std_A=float(a_std.mean(dtype=np.float64)), anomaly_std_B=float(b_std.mean()),
            anomaly_std_mae=float(std_delta.mean()), anomaly_std_max_abs=float(std_delta.max()),
            std_mean_ratio_A_over_B=(float(a_std.mean(dtype=np.float64)/b_std.mean())
                                     if b_std.mean() > 1e-12 else None),
            ranking_spearman=rho, ranking_mean_abs_shift=float(np.abs(ar-br).mean()),
            ranking_max_abs_shift=float(np.abs(ar-br).max()),
            top_1pct_count=k, top_1pct_overlap_count=overlap,
            top_1pct_overlap_fraction=overlap/k, top_1pct_jaccard=overlap/(2*k-overlap))
        return metrics, dict(A_mean=a_mean, A_std=a_std, B_mean=np.array(b_mean), B_std=np.array(b_std))
    finally:
        store.close()


def synthetic_cases(seed, samples=20, points=10000):
    rng = np.random.default_rng(seed)
    # Components retain STGAN signs: r>=0 and d in [-1,1].
    base_g = rng.uniform(.2, 1.2, size=(1, 100, points//100))
    base_d = rng.uniform(-.3, .3, size=base_g.shape)
    shape = (samples, *base_g.shape[1:])
    yield "identical_draws", np.broadcast_to(base_g, shape).astype(np.float32), np.broadcast_to(base_d, shape).astype(np.float32)
    yield "local_noise", (base_g+rng.normal(0, .015, shape)).astype(np.float32), (base_d+rng.normal(0, .008, shape)).astype(np.float32)
    # A cancels coherent offsets/scales even though every local raw score varies.
    scale_g = rng.uniform(.4, 2, (samples, 1, 1))
    scale_d = rng.uniform(.3, 1.5, (samples, 1, 1))
    shift_g = rng.uniform(0, .8, (samples, 1, 1))
    shift_d = rng.uniform(-.15, .15, (samples, 1, 1))
    yield "coherent_affine", (base_g*scale_g+shift_g).astype(np.float32), (base_d*scale_d+shift_d).astype(np.float32)
    # Only one remote cell changes; A injects variance into all fixed cells.
    remote_g = np.broadcast_to(base_g, shape).copy()
    remote_g[:, 0, 0] = np.linspace(2, 8, samples)
    yield "remote_extreme", remote_g.astype(np.float32), np.broadcast_to(base_d, shape).astype(np.float32)


def stgan_draws(seed, samples=20):
    """Small randomly initialized F=15 STGAN; strictly no optimizer or training."""
    import torch
    from physiq_pv.anomaly_detection.stgan import STGAN, STGANWindowDataset, build_spatial_grid
    from physiq_pv.anomaly_detection.stgan.scoring import fit_calibration_ranges, score_components
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    latitude, longitude = np.meshgrid([45., 44.5, 44.], [7., 7.5, 8.], indexing="ij")
    grid = build_spatial_grid(latitude.ravel(), longitude.ravel(), grid_crs="EPSG:4326",
                              angular_spacing=.5, audit_knn=False)
    values = np.random.default_rng(seed).normal(0, .5, (246, 9, 15)).astype(np.float32)
    dataset = STGANWindowDataset(values, pd.date_range("2005", periods=246, freq="3h"), grid,
        feature_minimum=np.full(15, -2, np.float32), feature_scale=np.full(15, 4, np.float32),
        recent_steps=1, trend_steps=6, stride=1)
    model = STGAN(n_features=15, hidden_size=8, n_layers=1, cnn_channels=4, cnn_layers=1)
    g, d, _, store = score_components(model, dataset, batch_size=128, device="cpu", n_features=15,
                                     mc_dropout_enabled=True, mc_samples=samples)
    try:
        return np.array(g), np.array(d)
    finally:
        store.close()


def storage_estimate(end_year=2025, *, n_locations=81*131, mc_samples=20, timestep_hours=3, features=15):
    """Payload bytes, no ERA5 allocations. Full lattice is a static-mask upper bound."""
    if end_year < 2005 or n_locations < 1 or mc_samples < 1 or timestep_hours not in (1, 3):
        raise ValueError("Invalid storage-estimate dimensions")
    timestamps = (date(end_year+1, 1, 1)-date(2005, 1, 1)).days*(24//timestep_hours)
    plane = timestamps*n_locations*4
    raw = 2*mc_samples*plane
    retained = (features+4)*plane
    def amount(value):
        return dict(bytes=value, GB=value/1e9, GiB=value/(1024**3))
    return dict(start_year=2005, end_year=end_year, timestamps=timestamps, locations=n_locations,
        grid_shape=[81, 131], mc_samples=mc_samples, timestep_hours=timestep_hours, features=features,
        dtype="float32", one_raw_component=amount(mc_samples*plane), both_raw_components=amount(raw),
        one_summary=amount(plane), feature_scores=amount(features*plane),
        retained_default=amount(retained), retained_debug=amount(retained+raw),
        peak_array_payload=amount(retained+raw), permanent_reduction_fraction=raw/(retained+raw),
        notes="Excludes model/input data, workers, filesystem headers, OS page cache, cubes and clustering. "
              "Peak includes temporary raw draws even when save_raw_mc=False. "
              "The memory backend allocates these bytes in RAM; memmap maps them on disk.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".analysis/stgan_mc_dropout"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 51, 79])
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--end-year", type=int, default=date.today().year-1)
    parser.add_argument("--n-locations", type=int, default=81*131)
    args = parser.parse_args(argv)
    if args.mc_samples < 1:
        parser.error("mc-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        cases = list(synthetic_cases(seed, samples=args.mc_samples))
        cases.append(("untrained_stgan_f15", *stgan_draws(seed, samples=args.mc_samples)))
        for name, g, d in cases:
            metrics, summaries = compare_components(g, d)
            if name == "remote_extreme":
                metrics["fixed_cells_std_A"] = float(summaries["A_std"].ravel()[1:].mean())
                metrics["fixed_cells_std_B"] = float(summaries["B_std"].ravel()[1:].mean())
            rows.append(dict(case=name, seed=seed, **metrics))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir/"normalization_comparison.csv", index=False)
    summary = frame.groupby("case", sort=False).mean(numeric_only=True).drop(columns=["seed"])
    summary.to_csv(args.output_dir/"normalization_summary.csv")
    report = dict(seeds=args.seeds, mc_samples=args.mc_samples,
        normalization_A="historical per-draw minmax over (T,N); experimental script only",
        normalization_B="common pooled minmax over (M,T,N), one range per component",
        std_ddof=0, ranking="descending anomaly_mean; Spearman uses average ranks for ties",
        top_1pct="ceil(0.01*T*N), stable flat-index tie breaking", training=False,
        results=rows)
    (args.output_dir/"normalization_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    estimates = [storage_estimate(args.end_year, n_locations=args.n_locations, mc_samples=args.mc_samples, timestep_hours=step)
                 for step in (3, 1)]
    (args.output_dir/"storage_estimate.json").write_text(json.dumps(estimates, indent=2), encoding="utf-8")
    print(summary[["anomaly_mean_mae", "anomaly_std_A", "anomaly_std_B", "ranking_spearman", "top_1pct_overlap_fraction"]].to_string())
    print(json.dumps(estimates[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
