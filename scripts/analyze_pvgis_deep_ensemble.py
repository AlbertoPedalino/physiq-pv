#!/usr/bin/env python
"""
Aggregate per-seed PVGIS-only ST-GNN predictions into a paper-style **Deep Ensemble**.

Seed-robustness != Deep Ensemble: that sweep reports metrics PER seed. A Deep
Ensemble combines, FOR EACH test sample, the mean prediction of every independently
trained seed model, then builds the predictive interval directly from those per-seed
predictions (empirical quantiles) and evaluates PICP / MPIW / NMPIL / CLC.

Inputs are the lightweight per-seed .npz files written by the runner under
`--save-ensemble-predictions`. NO post-hoc calibration. Anomaly labels are used
ONLY to stratify (normal vs rare_extreme), never as model input or target. Pipeline
stays PVGIS-only (no ENERGIA / Sentinel / kWp / compute_qs / real QS).

Usage:
  PYTHONPATH=$PWD python scripts/analyze_pvgis_deep_ensemble.py \
      --predictions-dir outputs/pvgis_deep_ensemble/predictions \
      --out-dir outputs/pvgis_deep_ensemble/analysis \
      --coverage-target 0.95 --clc-eta 10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the SAME pure interval metric used by the single-model paper-style pipeline.
from physiq_pv.experiments.pvgis_stgnn_runner import (
    ENSEMBLE_ANALYSIS_ROOT,
    compute_interval_metrics,
)

GROUP_RARE = "rare_or_extreme"
GROUP_NORMAL = "normal"
# Eval strata -> anomaly_group value (None = all rows). Anomaly labels are eval-only.
STRATA = {"global": None, "normal": GROUP_NORMAL, "rare_extreme": GROUP_RARE}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PVGIS-only ST-GNN Deep Ensemble aggregator.")
    p.add_argument("--predictions-dir", required=True,
                   help="Directory with per-seed .npz files from --save-ensemble-predictions.")
    p.add_argument("--out-dir", default=ENSEMBLE_ANALYSIS_ROOT)
    p.add_argument("--coverage-target", type=float, default=0.95,
                   help="gamma; PI = empirical quantiles at alpha/2, 1-alpha/2 (alpha=1-gamma).")
    p.add_argument("--clc-eta", type=float, default=10.0,
                   help="eta for CLC = NMPIL*(1+exp(-eta*(PICP-gamma))).")
    p.add_argument("--reference-predictions", default=None,
                   help="Optional predictions.csv of ONE ensemble member (same test "
                        "set). Joined on (timestamp, location) to recover "
                        "solar_irradiance_poa_target and anomaly_label, enabling "
                        "extended strata (daytime/normal_daytime/labels/production "
                        "bins) and an interval_miss-compatible "
                        "deep_ensemble_predictions_full.csv. Eval-only.")
    p.add_argument("--wandb", action="store_true", help="Log the ensemble metrics to a new W&B run.")
    p.add_argument("--wandb-entity", default="albertopedalino-politecnico-di-torino")
    p.add_argument("--wandb-project", default="PhysiQ-PV")
    p.add_argument("--wandb-run-name", default="pvgis_deep_ensemble")
    return p.parse_args()


def _load_members(predictions_dir: str) -> list[dict]:
    """Load every .npz, sorted by sample_id so seeds align regardless of row order."""
    files = sorted(glob.glob(os.path.join(predictions_dir, "*.npz")))
    if len(files) < 2:
        sys.exit(f"Deep Ensemble needs >= 2 seed files; found {len(files)} in {predictions_dir}.")
    members = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        sample_id = d["sample_id"].astype(str)
        order = np.argsort(sample_id, kind="stable")
        m = {
            "file": os.path.basename(f),
            "seed": int(d["seed"]) if "seed" in d.files else -1,
            "sample_id": sample_id[order],
            "y_true": d["y_true"].astype(float)[order],
            "y_pred_mean": d["y_pred_mean"].astype(float)[order],
            "anomaly_group": d["anomaly_group"].astype(str)[order],
        }
        if "y_pred_std_mc" in d.files:
            m["y_pred_std_mc"] = d["y_pred_std_mc"].astype(float)[order]
        members.append(m)
    return members


def _check_alignment(members: list[dict]) -> None:
    ref = members[0]
    ref_id = ref["sample_id"]
    for m in members[1:]:
        if m["sample_id"].shape != ref_id.shape or not np.array_equal(m["sample_id"], ref_id):
            sys.exit(
                f"sample_id mismatch between {ref['file']} and {m['file']}: the seeds "
                "do not cover the SAME test samples. Cannot form a Deep Ensemble."
            )
        if not np.allclose(m["y_true"], ref["y_true"], equal_nan=True):
            sys.exit(
                f"y_true differs between {ref['file']} and {m['file']} for aligned "
                "sample_ids — the test target is not identical across seeds."
            )


def _interval_metrics_by_group(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, group: np.ndarray,
    target_range: float, gamma: float, eta: float,
) -> dict:
    out = {}
    for name, gval in STRATA.items():
        mask = np.ones(len(y_true), bool) if gval is None else (group == gval)
        if mask.sum() == 0:
            continue
        out[name] = compute_interval_metrics(
            y_true[mask], lower[mask], upper[mask], target_range, gamma, eta
        )
    return out


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray, std: np.ndarray, group: np.ndarray) -> dict:
    out = {}
    for name, gval in STRATA.items():
        mask = np.ones(len(y_true), bool) if gval is None else (group == gval)
        if mask.sum() == 0:
            continue
        err = y_pred[mask] - y_true[mask]
        out[name] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean_std": float(np.mean(std[mask])),
            "count": int(mask.sum()),
        }
    return out


def _flatten_for_wandb(point: dict, iv_pi: dict, iv_mm: dict) -> dict:
    out: dict = {}
    for g in ("global", "normal", "rare_extreme"):
        if g in point:
            out[f"mae/{g}"] = point[g]["mae"]
            out[f"rmse/{g}"] = point[g]["rmse"]
            out[f"uncertainty/mean_std_{'rare_extreme' if g=='rare_extreme' else g}"] = point[g]["mean_std"]
        if g in iv_pi:
            for k in ("picp", "mpiw", "nmpil", "clc"):
                out[f"{k}_ensemble/{g}"] = iv_pi[g][k]
        if g in iv_mm:
            for k in ("picp", "mpiw", "nmpil", "clc"):
                out[f"{k}_ensemble_minmax/{g}"] = iv_mm[g][k]
    # ratios rare/normal
    if "normal" in point and "rare_extreme" in point:
        n, r = point["normal"], point["rare_extreme"]
        if n["mae"]:
            out["ratio/mae_rare_normal"] = r["mae"] / n["mae"]
        if n["rmse"]:
            out["ratio/rmse_rare_normal"] = r["rmse"] / n["rmse"]
        if n["mean_std"]:
            out["uncertainty/ratio_rare_normal"] = r["mean_std"] / n["mean_std"]
    return out


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if x is not None and np.isfinite(x) else "—"


def _render_report(meta: dict, point: dict, iv_pi: dict, iv_mm: dict, members: list[dict]) -> str:
    L = []
    A = L.append
    A("# PVGIS-only ST-GNN — Deep Ensemble report\n")
    A("**Deep Ensemble** = several ST-GNNs trained independently with different seeds, "
      "combined PER TEST SAMPLE. This differs from seed robustness: seed robustness "
      "aggregates metrics per seed; the Deep Ensemble combines the seeds' predictions "
      "for each sample. The predictive interval is built **directly from the seed "
      "predictions** (empirical quantiles) — **PICP is evaluated, not forced**, and "
      "**no post-hoc calibration** is used. With only "
      f"{meta['n_models']} models the empirical quantiles are coarse (close to "
      "min/max), so a min/max diagnostic interval is reported alongside.\n")

    A("## Setup\n")
    A(f"- Members (seeds): **{meta['n_models']}**  |  samples per member: **{meta['n_samples']}**")
    A(f"- coverage_target (gamma): **{meta['coverage_target']}**  |  clc_eta (eta): **{meta['clc_eta']}**")
    A(f"- PI quantiles: q{meta['q_lo']:.3f} / q{meta['q_hi']:.3f}  |  target_range (global): **{_fmt(meta['target_range'])}**")
    A("- Pipeline: PVGIS-only; anomaly labels eval-only (stratification), never input/target. No post-hoc calibration.\n")

    A("## Loaded seed files\n")
    A("| file | seed | n_samples |")
    A("|---|---|---|")
    for m in members:
        A(f"| {m['file']} | {m['seed']} | {len(m['sample_id'])} |")
    A("")

    A("## Point forecast & uncertainty\n")
    A("| stratum | count | MAE | RMSE | mean ensemble_std |")
    A("|---|---|---|---|---|")
    for g in ("global", "normal", "rare_extreme"):
        if g in point:
            p = point[g]
            A(f"| {g} | {p['count']} | {_fmt(p['mae'])} | {_fmt(p['rmse'])} | {_fmt(p['mean_std'])} |")
    if "normal" in point and "rare_extreme" in point:
        n, r = point["normal"], point["rare_extreme"]
        A("")
        A(f"- MAE rare/normal ratio: **{_fmt(r['mae']/n['mae'], 3) if n['mae'] else '—'}**  |  "
          f"RMSE rare/normal: **{_fmt(r['rmse']/n['rmse'], 3) if n['rmse'] else '—'}**  |  "
          f"ensemble_std rare/normal: **{_fmt(r['mean_std']/n['mean_std'], 3) if n['mean_std'] else '—'}**")
    A("")

    A("## Ensemble prediction interval (primary, empirical quantiles)\n")
    A("| stratum | PICP | MPIW | NMPIL | CLC |")
    A("|---|---|---|---|---|")
    for g in ("global", "normal", "rare_extreme"):
        if g in iv_pi:
            m = iv_pi[g]
            A(f"| {g} | {_fmt(m['picp'], 3)} | {_fmt(m['mpiw'])} | {_fmt(m['nmpil'])} | {_fmt(m['clc'])} |")
    A("")

    A("## Min/max diagnostic interval (secondary)\n")
    A("With few members the quantile band ~ min/max; this is a diagnostic, not the primary PI.\n")
    A("| stratum | PICP | MPIW | NMPIL | CLC |")
    A("|---|---|---|---|---|")
    for g in ("global", "normal", "rare_extreme"):
        if g in iv_mm:
            m = iv_mm[g]
            A(f"| {g} | {_fmt(m['picp'], 3)} | {_fmt(m['mpiw'])} | {_fmt(m['nmpil'])} | {_fmt(m['clc'])} |")
    A("")

    sm = meta.get("single_model")
    A("## Vs MC-Dropout single model (paper-style)\n")
    if sm:
        A(f"- Single-model MC-Dropout PICP (global): **{_fmt(sm.get('picp_pi_global'), 3)}**  |  "
          f"Deep Ensemble PICP (global): **{_fmt(iv_pi.get('global', {}).get('picp'), 3)}**")
        A("- A Deep Ensemble of independent seeds typically widens the interval vs a single "
          "MC-Dropout model, which was strongly under-dispersed (low PICP) in this PVGIS-only setting.\n")
    else:
        A("- No single-model paper-style metrics passed; compare against `picp_pi/global` from the "
          "seed-only paper-style sweep manually.\n")

    A("## Operational conclusion\n")
    pg = iv_pi.get("global", {}).get("picp")
    A(f"- Deep Ensemble of {meta['n_models']} independent ST-GNN seeds, combined per sample.")
    A(f"- Primary PI coverage (PICP global) = **{_fmt(pg, 3)}** — evaluated, not forced; no calibration.")
    A("- With 5 members the empirical quantile band is coarse; min/max is the diagnostic upper bound on width.")
    A("- If PICP stays far below gamma, the independent-seed disagreement alone does not yield calibrated "
      "intervals in the PVGIS-only setting (consistent with the under-dispersed MC-Dropout result).\n")
    return "\n".join(L) + "\n"


def _split_sample_ids(sample_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sample_id = '<int64 ns timestamp>_<location>' -> (timestamp ns, location).

    Split on the FIRST underscore: the timestamp part is all digits, so
    locations containing underscores survive intact.
    """
    ts = np.empty(len(sample_id), dtype=np.int64)
    loc = np.empty(len(sample_id), dtype=object)
    for i, sid in enumerate(sample_id):
        t, l = str(sid).split("_", 1)
        ts[i] = int(t)
        loc[i] = l
    return ts, loc.astype(str)


def build_full_predictions(
    summary: pd.DataFrame, reference_csv: str
) -> pd.DataFrame:
    """
    Join the ensemble summary with one member's predictions.csv on
    (timestamp, location) to recover solar_irradiance_poa_target and
    anomaly_label, and emit an interval_miss-compatible frame
    (y_pred_mean/y_pred_std/lower_pi/upper_pi naming). Eval-only metadata;
    the reference member's own predictions are NOT used.
    """
    ref = pd.read_csv(reference_csv, parse_dates=["timestamp"])
    needed = {"timestamp", "location", "y_true", "solar_irradiance_poa_target",
              "anomaly_group", "anomaly_label"}
    missing = needed - set(ref.columns)
    if missing:
        sys.exit(f"--reference-predictions missing columns: {sorted(missing)}")

    ts, loc = _split_sample_ids(summary["sample_id"].to_numpy())
    full = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts),
            "location": loc,
            "y_true": summary["y_true"].to_numpy(dtype=float),
            "y_pred": summary["ensemble_mean"].to_numpy(dtype=float),
            "y_pred_mean": summary["ensemble_mean"].to_numpy(dtype=float),
            # ensemble std plays the role of the predictive std in the
            # interval_miss diagnostics (required_multiplier, PICP-vs-k curve).
            "y_pred_std": summary["ensemble_std"].to_numpy(dtype=float),
            "y_pred_std_raw": summary["ensemble_std"].to_numpy(dtype=float),
            "lower_pi": summary["lower_ensemble_pi"].to_numpy(dtype=float),
            "upper_pi": summary["upper_ensemble_pi"].to_numpy(dtype=float),
            "anomaly_group": summary["anomaly_group"].to_numpy(),
        }
    )
    ref = ref.copy()
    ref["location"] = ref["location"].astype(str)
    meta_cols = ref[["timestamp", "location", "y_true",
                     "solar_irradiance_poa_target", "anomaly_label"]].rename(
        columns={"y_true": "y_true_ref"}
    )
    full = full.merge(meta_cols, on=["timestamp", "location"], how="left")
    if full["solar_irradiance_poa_target"].isna().any():
        n_bad = int(full["solar_irradiance_poa_target"].isna().sum())
        sys.exit(
            f"reference join failed for {n_bad}/{len(full)} rows: the reference "
            "predictions.csv does not cover the ensemble test samples."
        )
    if not np.allclose(full["y_true"], full["y_true_ref"], equal_nan=True):
        sys.exit(
            "y_true mismatch between ensemble .npz and --reference-predictions: "
            "not the same test set/protocol."
        )
    full = full.drop(columns=["y_true_ref"])
    full["anomaly_label"] = full["anomaly_label"].fillna("")
    return full


def _extended_strata_metrics(
    full: pd.DataFrame, target_range: float, gamma: float, eta: float
) -> list[dict]:
    """Per-stratum PICP/MPIW/NMPIL/CLC + MAE/RMSE/mean ensemble_std for the
    daytime/anomaly-label/production-bin strata (masks shared with the
    interval_miss diagnostics)."""
    from scripts.interval_miss_utils import build_strata_masks  # noqa: PLC0415

    y_true = full["y_true"].to_numpy(dtype=float)
    y_pred = full["y_pred_mean"].to_numpy(dtype=float)
    y_std = full["y_pred_std"].to_numpy(dtype=float)
    lower = full["lower_pi"].to_numpy(dtype=float)
    upper = full["upper_pi"].to_numpy(dtype=float)

    rows = []
    for stratum, mask in build_strata_masks(full).items():
        row = {"stratum": stratum, "count": int(mask.sum())}
        if mask.sum() == 0:
            rows.append(row)
            continue
        err = y_pred[mask] - y_true[mask]
        row["mae"] = float(np.mean(np.abs(err)))
        row["rmse"] = float(np.sqrt(np.mean(err ** 2)))
        row["mean_ensemble_std"] = float(np.mean(y_std[mask]))
        row.update(compute_interval_metrics(
            y_true[mask], lower[mask], upper[mask], target_range, gamma, eta
        ))
        rows.append(row)
    return rows


def main() -> None:
    args = _parse_args()
    gamma = float(args.coverage_target)
    eta = float(args.clc_eta)
    if not 0.0 < gamma < 1.0:
        sys.exit(f"--coverage-target must be in (0, 1), got {gamma}.")

    members = _load_members(args.predictions_dir)
    _check_alignment(members)

    ref = members[0]
    sample_id = ref["sample_id"]
    y_true = ref["y_true"]
    group = ref["anomaly_group"]
    n_samples = len(sample_id)

    # pred_matrix: [n_models, n_samples]
    pred_matrix = np.vstack([m["y_pred_mean"] for m in members])
    n_models = pred_matrix.shape[0]
    print(f"[ensemble] {n_models} members x {n_samples} samples; seeds={[m['seed'] for m in members]}")

    ensemble_mean = pred_matrix.mean(axis=0)
    ensemble_std = pred_matrix.std(axis=0)

    alpha = 1.0 - gamma
    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
    lower_pi = np.quantile(pred_matrix, q_lo, axis=0)
    upper_pi = np.quantile(pred_matrix, q_hi, axis=0)
    lower_mm = pred_matrix.min(axis=0)
    upper_mm = pred_matrix.max(axis=0)

    eps = 1e-6
    target_range = float(np.nanmax(y_true) - np.nanmin(y_true))
    if not np.isfinite(target_range) or target_range < eps:
        target_range = eps

    point = _point_metrics(y_true, ensemble_mean, ensemble_std, group)
    iv_pi = _interval_metrics_by_group(y_true, lower_pi, upper_pi, group, target_range, gamma, eta)
    iv_mm = _interval_metrics_by_group(y_true, lower_mm, upper_mm, group, target_range, gamma, eta)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "n_models": n_models, "n_samples": n_samples, "seeds": [m["seed"] for m in members],
        "coverage_target": gamma, "clc_eta": eta, "q_lo": q_lo, "q_hi": q_hi,
        "target_range": target_range, "posthoc_calibration": False,
        "files": [m["file"] for m in members],
    }
    payload = {"meta": meta, "point": point, "interval_pi": iv_pi, "interval_minmax": iv_mm}
    (out_dir / "deep_ensemble_metrics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    summary = pd.DataFrame({
        "sample_id": sample_id,
        "y_true": y_true,
        "ensemble_mean": ensemble_mean,
        "ensemble_std": ensemble_std,
        "lower_ensemble_pi": lower_pi,
        "upper_ensemble_pi": upper_pi,
        "anomaly_group": group,
    })
    summary.to_csv(out_dir / "deep_ensemble_predictions_summary.csv", index=False)

    # Optional extended strata: join one member's predictions.csv to recover
    # solar irradiance + anomaly labels (eval-only). Also writes an
    # interval_miss-compatible full CSV so the existing post-hoc script
    # (scripts/analyze_pvgis_interval_miss_distance.py) runs on the ensemble.
    if args.reference_predictions:
        full = build_full_predictions(summary, args.reference_predictions)
        full_path = out_dir / "deep_ensemble_predictions_full.csv"
        full.to_csv(full_path, index=False)
        extended = _extended_strata_metrics(full, target_range, gamma, eta)
        pd.DataFrame(extended).to_csv(
            out_dir / "deep_ensemble_extended_strata.csv", index=False
        )
        payload["extended_strata"] = extended
        (out_dir / "deep_ensemble_metrics.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        print(f"[ensemble] extended strata + full CSV written ({full_path})")
        print("[ensemble] run interval_miss on the ensemble with:")
        print(f"  python scripts/analyze_pvgis_interval_miss_distance.py "
              f"--predictions {full_path} --out-dir {out_dir / 'interval_miss'}")
        focus = ("global", "daytime", "normal_daytime", "rare_extreme_daytime",
                 "label:unusually_low_solar_potential",
                 "label:unusually_high_solar_potential", "daytime_gt_100")
        print("[ensemble] extended strata (PI = between-seed empirical quantiles):")
        for row in extended:
            if row["stratum"] in focus and row["count"]:
                print(
                    f"  {row['stratum']:42s} n={row['count']:8d}  "
                    f"picp={row['picp']:.3f}  mpiw={row['mpiw']:.2f}  "
                    f"clc={row['clc']:.3f}  mae={row['mae']:.3f}"
                )

    (out_dir / "deep_ensemble_report.md").write_text(
        _render_report(meta, point, iv_pi, iv_mm, members), encoding="utf-8"
    )

    flat = _flatten_for_wandb(point, iv_pi, iv_mm)
    print("[ensemble] key metrics:")
    for k in sorted(flat):
        print(f"  {k:34s} {flat[k]:.4f}")
    print(f"[ensemble] wrote: {out_dir}/deep_ensemble_metrics.json, deep_ensemble_report.md, "
          "deep_ensemble_predictions_summary.csv")

    if args.wandb:
        import wandb  # noqa: PLC0415
        run = wandb.init(
            entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_run_name,
            config={"mode": "pvgis_deep_ensemble", **meta},
        )
        run.log(flat)
        run.summary.update(flat)
        run.finish()
        print(f"[ensemble] logged to W&B run '{args.wandb_run_name}'")


if __name__ == "__main__":
    main()
