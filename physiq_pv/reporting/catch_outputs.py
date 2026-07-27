"""Stable tabular and report outputs for PVGIS CATCH experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def canonical_detector_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Return the detector-label contract consumed by the SDE pipelines."""

    score_column = (
        "anomaly_score" if "anomaly_score" in scores.columns else "global_score"
    )
    required = {"location", "timestamp", score_column, "threshold", "is_anomaly"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"CATCH scores missing canonical columns: {missing}")
    canonical = scores.loc[
        :, ["location", "timestamp", score_column, "threshold", "is_anomaly"]
    ].copy()
    canonical.insert(2, "method", "catch")
    if score_column != "anomaly_score":
        canonical = canonical.rename(columns={score_column: "anomaly_score"})
    return canonical


def merge_detection_intervals(detections: pd.DataFrame) -> pd.DataFrame:
    """Merge exactly consecutive hourly detections without label-based padding."""

    columns = ["location", "start", "end", "n_points", "max_global_score"]
    if detections.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict] = []
    for location, group in detections.sort_values("timestamp").groupby(
        "location", sort=False
    ):
        group = group.reset_index(drop=True)
        breaks = group["timestamp"].diff().ne(pd.Timedelta(hours=1)).cumsum()
        for _, interval in group.groupby(breaks):
            rows.append(
                {
                    "location": location,
                    "start": interval["timestamp"].iloc[0],
                    "end": interval["timestamp"].iloc[-1],
                    "n_points": int(len(interval)),
                    "max_global_score": float(interval["global_score"].max()),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def render_catch_report(meta: dict, summary: pd.DataFrame) -> str:
    sensors = ", ".join(meta["sensors"])
    train_years = ", ".join(str(year) for year in meta["train_years"])
    total_anomalies = int(summary["n_anomalies"].sum()) if not summary.empty else 0
    return "\n".join(
        [
            "# PVGIS CATCH report",
            "",
            "Channel-aware multivariate anomaly detection through frequency patching, "
            "time/frequency reconstruction, and learned channel masks.",
            "",
            "## Protocol",
            "",
            f"- Training years: **{train_years}**",
            f"- Held-out test year: **{meta['test_year']}**",
            f"- Sensors: {sensors}",
            f"- Locations fitted independently: **{meta['n_locations']}**",
            "- Labels used for training or threshold calibration: **none**",
            "- Windows cross year boundaries: **no**",
            f"- Window: **{meta['seq_len']} h**; frequency patch: "
            f"**{meta['patch_size']}**, stride **{meta['patch_stride']}**",
            f"- Backbone: cf_dim={meta['cf_dim']}, d_model={meta['d_model']}, "
            f"head_dim={meta['head_dim']}; mask source: **{meta['mask_source']}**",
            f"- Train-only expected contamination: **{meta['contamination']}**",
            f"- Total anomalous test points: **{total_anomalies}**",
            "",
            "## Implementation notes",
            "",
            "- The implementation is self-contained in physiq_pv and has no runtime "
            "dependency on the external research checkout.",
            "- The threshold is calibrated only from training scores, avoiding the "
            "train-plus-test percentile used by some benchmark scripts.",
            "- Every yearly segment is split chronologically; scaling is fitted only "
            "on the fitting portion, never on validation or test points.",
            "- Mask parameters are updated in the paper's outer loop before grouped "
            "inner updates of the remaining model parameters.",
            "- Frequency patch errors are averaged back onto every timestamp they cover, "
            "matching the point-granularity scoring described in the paper.",
            "",
        ]
    )


def write_catch_outputs(
    out_dir: str | Path,
    scores: pd.DataFrame,
    train_scores: pd.DataFrame,
    summary: pd.DataFrame,
    meta: dict,
) -> dict[str, Path]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    detections = scores[scores["is_anomaly"]].copy()
    intervals = merge_detection_intervals(detections)
    paths = {
        "scores": directory / "catch_scores.csv",
        "anomaly_scores": directory / "anomaly_scores.csv",
        "train_anomaly_scores": directory / "train_anomaly_scores.csv",
        "detections": directory / "catch_detections.csv",
        "intervals": directory / "catch_intervals.csv",
        "summary": directory / "catch_location_summary.csv",
        "meta": directory / "catch_meta.json",
        "report": directory / "report.md",
    }
    scores.to_csv(paths["scores"], index=False)
    canonical_detector_scores(scores).to_csv(paths["anomaly_scores"], index=False)
    canonical_detector_scores(train_scores).to_csv(
        paths["train_anomaly_scores"], index=False
    )
    detections.to_csv(paths["detections"], index=False)
    intervals.to_csv(paths["intervals"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["report"].write_text(render_catch_report(meta, summary), encoding="utf-8")
    return paths
