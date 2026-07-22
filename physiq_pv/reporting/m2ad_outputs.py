"""Pure tabular/report output helpers for PVGIS M2AD experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def merge_detection_intervals(detections: pd.DataFrame) -> pd.DataFrame:
    """Merge exactly consecutive hourly detections without point padding."""
    columns = [
        "location",
        "start",
        "end",
        "n_points",
        "max_global_score",
        "min_gamma_p_value",
    ]
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
                    "min_gamma_p_value": float(
                        interval["gamma_p_value"].min()
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def render_m2ad_report(meta: dict, summary: pd.DataFrame) -> str:
    """Render the human-readable experiment report without performing I/O."""
    sensors = ", ".join(meta["sensors"])
    total_anomalies = int(summary["n_anomalies"].sum()) if not summary.empty else 0
    train_years = ", ".join(str(year) for year in meta["train_years"])
    return "\n".join(
        [
            "# PVGIS M2AD report",
            "",
            "Paper-faithful label-unaware detector: stacked LSTM forecast, sensor-level GMM "
            "p-values, weighted Fisher global score, and moment-matched Gamma calibration.",
            "",
            "## Protocol",
            "",
            f"- Training years: **{train_years}** ({len(meta['train_years'])} separate yearly segments)",
            f"- Held-out test year: **{meta['test_year']}**",
            f"- Sensors: {sensors}",
            f"- Locations fitted independently: **{meta['n_locations']}**",
            "- Labels used for training/calibration/thresholding: **none**",
            "- Windows and EWMA state cross year boundaries: **no**",
            f"- Window: **{meta['window_size']} h**; error: **{meta['error']}**; "
            f"Gamma significance: **{meta['significance']}**",
            f"- Total anomalous test points: **{total_anomalies}**",
            "",
            "## Deliberate corrections to the public code",
            "",
            "- Missing-value statistics, scaling, and area-error scaling are fitted on training only.",
            "- A window ending at time t predicts t+1 for horizon=1 (no extra off-by-one gap).",
            "- Reported intervals merge only consecutive detections; benchmark-style ±50 point "
            "padding is not applied to operational PVGIS results.",
            "",
            "## Assumption and limitation",
            "",
            "M2AD assumes the training interval represents normal operation and that the "
            "test residual distribution does not undergo a wholesale regime shift. In this "
            "label-unaware protocol, training is not filtered using anomaly annotations; "
            "contamination can therefore make the detector conservative.",
            "",
        ]
    )


def write_m2ad_outputs(
    out_dir: str | Path,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    meta: dict,
) -> dict[str, Path]:
    """Write the complete stable output contract and return its paths."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    detections = scores[scores["is_anomaly"]].copy()
    intervals = merge_detection_intervals(detections)
    paths = {
        "scores": directory / "m2ad_scores.csv",
        "detections": directory / "m2ad_detections.csv",
        "intervals": directory / "m2ad_intervals.csv",
        "summary": directory / "m2ad_location_summary.csv",
        "meta": directory / "m2ad_meta.json",
        "report": directory / "report.md",
    }
    scores.to_csv(paths["scores"], index=False)
    detections.to_csv(paths["detections"], index=False)
    intervals.to_csv(paths["intervals"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["report"].write_text(
        render_m2ad_report(meta, summary), encoding="utf-8"
    )
    return paths
