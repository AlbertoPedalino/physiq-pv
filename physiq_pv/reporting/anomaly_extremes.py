"""Which days does MTGFlow consider extreme, and how far past its threshold?

The IQR threshold of Eq. 13 is binary: a window one unit past it and a window
ten times past it are both simply ``is_anomaly``. The forecasting pipeline only
ever sees that flag. This module keeps the continuous score and ranks days by
how far the detector's own tail is exceeded, so the analysis can start from the
windows the detector is most confident about rather than from a calendar guess.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("location", "timestamp", "anomaly_score")


def _validate(path: Path) -> set:
    if not path.exists():
        raise FileNotFoundError(f"{path} is required.")
    available = set(pd.read_csv(path, nrows=0).columns)
    missing = set(REQUIRED_COLUMNS) - available
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}.")
    return available


def score_reference(
    scores_path: str | Path,
    *,
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999, 0.9999),
) -> Dict[str, float]:
    """Return exact score quantiles and the detector threshold.

    Only the score column is read, so the full distribution fits in memory even
    when the file itself is large.
    """
    path = Path(scores_path)
    available = _validate(path)
    scores = pd.read_csv(path, usecols=["anomaly_score"])["anomaly_score"]
    values = pd.to_numeric(scores, errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        raise ValueError(f"{path} holds no finite score.")
    reference = {
        f"q{quantile:g}": float(np.quantile(values, quantile))
        for quantile in quantiles
    }
    reference["n_windows"] = float(values.size)
    reference["max"] = float(values.max())
    if "threshold" in available:
        thresholds = pd.to_numeric(
            pd.read_csv(path, usecols=["threshold"])["threshold"], errors="coerce"
        ).dropna()
        if not thresholds.empty:
            reference["threshold_min"] = float(thresholds.min())
            reference["threshold_median"] = float(thresholds.median())
            reference["threshold_max"] = float(thresholds.max())
    return reference


def rank_extreme_days(
    scores_path: str | Path,
    *,
    quantile: float = 0.999,
    chunksize: int = 2_000_000,
    n_locations: Optional[int] = None,
) -> pd.DataFrame:
    """Rank calendar days by how much of the score tail they contain.

    A day scores high when many of its windows sit in the extreme tail of the
    whole year, not merely past the binary threshold.
    """
    path = Path(scores_path)
    _validate(path)
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}.")
    cut = score_reference(path, quantiles=(quantile,))[f"q{quantile:g}"]

    parts: List[pd.DataFrame] = []
    # (day, location) pairs are kept separately: counting unique locations per
    # chunk and summing would count a location once per chunk it appears in.
    extreme_pairs: List[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        dtype={"location": "string"},
        chunksize=chunksize,
    )
    for chunk in reader:
        stamp = pd.to_datetime(chunk["timestamp"], errors="coerce")
        score = pd.to_numeric(chunk["anomaly_score"], errors="coerce")
        frame = pd.DataFrame({
            "day": stamp.dt.normalize(),
            "location": chunk["location"].astype(str),
            "score": score,
            "extreme": score >= cut,
        }).dropna(subset=["day", "score"])
        grouped = frame.groupby("day")
        parts.append(pd.DataFrame({
            "n_windows": grouped.size(),
            "n_extreme": grouped["extreme"].sum(),
            "score_sum": grouped["score"].sum(),
            "score_max": grouped["score"].max(),
        }))
        pairs = frame.loc[frame["extreme"], ["day", "location"]]
        if not pairs.empty:
            extreme_pairs.append(pairs.drop_duplicates())
    if not parts:
        raise ValueError(f"{path} produced no daily aggregate.")

    combined = pd.concat(parts)
    days = combined.groupby(level=0).agg(
        n_windows=("n_windows", "sum"),
        n_extreme=("n_extreme", "sum"),
        score_sum=("score_sum", "sum"),
        score_max=("score_max", "max"),
    )
    if extreme_pairs:
        unique_pairs = pd.concat(extreme_pairs, ignore_index=True).drop_duplicates()
        days["n_locations_extreme"] = (
            unique_pairs.groupby("day")["location"].nunique()
        )
    days["n_locations_extreme"] = days.get(
        "n_locations_extreme", pd.Series(dtype=float)
    ).reindex(days.index).fillna(0).astype(int)
    days["score_mean"] = days["score_sum"] / days["n_windows"]
    days["extreme_share"] = days["n_extreme"] / days["n_windows"]
    if n_locations:
        days["location_coverage"] = days["n_locations_extreme"] / float(n_locations)
    days = days.drop(columns="score_sum")
    days.index = days.index.rename("day")
    return days.sort_values(
        ["n_extreme", "score_max"], ascending=False
    ).reset_index()


def top_extreme_windows(
    scores_path: str | Path,
    *,
    k: int = 50,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Return the ``k`` single windows with the highest score."""
    path = Path(scores_path)
    available = _validate(path)
    if k < 1:
        raise ValueError("k must be positive.")
    usecols = [*REQUIRED_COLUMNS]
    if "threshold" in available:
        usecols.append("threshold")
    best: Optional[pd.DataFrame] = None
    reader = pd.read_csv(
        path, usecols=usecols, dtype={"location": "string"}, chunksize=chunksize
    )
    for chunk in reader:
        chunk = chunk.copy()
        chunk["anomaly_score"] = pd.to_numeric(
            chunk["anomaly_score"], errors="coerce"
        )
        candidate = chunk.nlargest(k, "anomaly_score")
        best = candidate if best is None else pd.concat([best, candidate])
        best = best.nlargest(k, "anomaly_score")
    if best is None or best.empty:
        raise ValueError(f"{path} produced no window.")
    best = best.copy()
    best["timestamp"] = pd.to_datetime(best["timestamp"])
    if "threshold" in best:
        best["excess_ratio"] = best["anomaly_score"] / best["threshold"]
    return best.reset_index(drop=True)


def summarise_extreme_days(
    days: pd.DataFrame,
    labels: Optional[pd.DataFrame] = None,
    *,
    top_n: int = 15,
) -> pd.DataFrame:
    """Join the ranked days with their driver mix, when labels are available."""
    selected = days.head(top_n).copy()
    if labels is None or labels.empty:
        return selected
    work = labels.copy()
    work["day"] = pd.to_datetime(work["timestamp"]).dt.normalize()
    mix = (
        work[work["day"].isin(selected["day"])]
        .groupby(["day", "driver"]).size().rename("n").reset_index()
    )
    if mix.empty:
        return selected
    shares = mix.pivot(index="day", columns="driver", values="n").fillna(0)
    shares = shares.div(shares.sum(axis=1), axis=0).add_prefix("share_")
    return selected.merge(shares.reset_index(), on="day", how="left")
