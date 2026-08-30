"""Does the forecaster react to an event, or trail it?

The model sees no future covariate: its input window ends before the hour it
predicts.  When a dust cloud arrives, the last hours of input still describe a
clear sky, so a purely reactive model would keep predicting clear-sky
production and only adapt once the drop has entered its input window.

That hypothesis is testable.  For every node the arrival hour is located, and
the error is then averaged by *hours since arrival*.  A reactive model shows a
large positive bias at lag 0 that decays towards zero within a few hours; a
model that genuinely anticipates the event shows no such profile.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

DAYTIME_IRRADIANCE_THRESHOLD_WM2 = 10.0


def _columns(available: Iterable[str]) -> Dict[str, Optional[str]]:
    columns = set(available)

    def choose(*candidates: str, required: bool = True) -> Optional[str]:
        found = next((name for name in candidates if name in columns), None)
        if found is None and required:
            raise ValueError(f"predictions.csv requires one of {list(candidates)}.")
        return found

    return {
        "timestamp": choose("timestamp", "target_timestamp", "time"),
        "location": choose("location", "location_id", "node_id"),
        "y_true": choose("y_true"),
        "y_pred": choose("y_pred_mean", "y_pred"),
        "lower_pi": choose("lower_pi", required=False),
        "upper_pi": choose("upper_pi", required=False),
        "solar": choose(
            "solar_irradiance_poa_target", "solar_irradiance_poa", "ghi_target",
            required=False,
        ),
        "horizon": choose("horizon_hours", "horizon", required=False),
    }


def load_event_window(
    out_dir: str | Path,
    *,
    event_days: Iterable[str],
    baseline_days: int = 10,
    exclude_days: Iterable[str] = (),
    chunksize: int = 500_000,
    horizon_hours: Optional[int] = None,
) -> pd.DataFrame:
    """Read the event days plus the quiet days that precede them.

    ``exclude_days`` drops days from the window entirely.  Events ramp up: the
    days just before the peak are already disturbed, and leaving them in the
    baseline would depress the reference and hide the very shortfall the
    analysis looks for.
    """
    out = Path(out_dir)
    predictions_path = out / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} is required.")
    days = sorted({pd.Timestamp(day).normalize() for day in event_days})
    if not days:
        raise ValueError("event_days must contain at least one date.")
    if baseline_days < 1:
        raise ValueError("baseline_days must be positive.")
    start = days[0] - pd.Timedelta(days=baseline_days)
    end = days[-1] + pd.Timedelta(hours=23, minutes=59)

    columns = _columns(pd.read_csv(predictions_path, nrows=0).columns)
    if horizon_hours is not None:
        horizon_hours = int(horizon_hours)
        if horizon_hours < 1:
            raise ValueError("horizon_hours must be a positive integer.")
        if columns["horizon"] is None:
            raise ValueError(
                "horizon_hours was requested but predictions.csv has no horizon column."
            )
    usecols = [name for name in columns.values() if name is not None]
    parts: List[pd.DataFrame] = []
    reader = pd.read_csv(
        predictions_path,
        usecols=list(dict.fromkeys(usecols)),
        dtype={columns["timestamp"]: "string", columns["location"]: "string"},
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        if horizon_hours is not None:
            selected_horizon = pd.to_numeric(
                chunk[columns["horizon"]], errors="coerce"
            ).eq(horizon_hours)
            chunk = chunk.loc[selected_horizon]
            if chunk.empty:
                continue
        stamp = pd.to_datetime(chunk[columns["timestamp"]], errors="coerce")
        if stamp.dt.tz is not None:
            stamp = stamp.dt.tz_convert("UTC").dt.tz_localize(None)
        keep = (stamp >= start) & (stamp <= end)
        if not keep.any():
            continue
        block = pd.DataFrame({
            "timestamp": stamp[keep],
            "location": chunk.loc[keep, columns["location"]].astype(str),
        })
        if horizon_hours is not None:
            block["horizon_hours"] = horizon_hours
        for role in ("y_true", "y_pred", "lower_pi", "upper_pi", "solar"):
            column = columns[role]
            if column is not None:
                block[role] = pd.to_numeric(
                    chunk.loc[keep, column], errors="coerce"
                ).to_numpy()
        parts.append(block)
    if not parts:
        raise ValueError("No prediction row falls in the requested window.")
    window = pd.concat(parts, ignore_index=True)
    window["day"] = window["timestamp"].dt.normalize()
    window["hour"] = window["timestamp"].dt.hour
    dropped = {pd.Timestamp(day).normalize() for day in exclude_days}
    if dropped:
        window = window.loc[~window["day"].isin(dropped)]
        if window.empty:
            raise ValueError("exclude_days removed every row from the window.")
    window["is_event"] = window["day"].isin(days)
    if not window["is_event"].any():
        raise ValueError("No event day survived the window and exclusions.")
    if not (~window["is_event"]).any():
        raise ValueError("No baseline day left; increase baseline_days.")
    return window.sort_values(["location", "timestamp"]).reset_index(drop=True)


def build_onset_response(
    window: pd.DataFrame,
    *,
    shortfall: float = 0.4,
    max_lag_hours: int = 12,
    min_daytime_rows: int = 3,
) -> Dict[str, object]:
    """Average the forecast error by hours since the event reached each node.

    The reference production is the same node at the same hour of day on the
    quiet days that precede the event, so the diurnal and seasonal cycles are
    removed without any climatology outside the window.  A node enters the
    event when its production first falls below ``1 - shortfall`` of that
    reference; nodes the event never reaches are reported separately instead of
    being mixed into lag 0.
    """
    if not 0.0 < shortfall < 1.0:
        raise ValueError(f"shortfall must be in (0, 1), got {shortfall}.")
    required = {"timestamp", "location", "y_true", "y_pred", "is_event", "hour"}
    missing = required - set(window.columns)
    if missing:
        raise ValueError(f"Window is missing columns {sorted(missing)}.")

    work = window.copy()
    if "solar" in work:
        daytime = work["solar"] > DAYTIME_IRRADIANCE_THRESHOLD_WM2
    else:
        daytime = work["y_true"] > 0.0
    work = work.loc[daytime & np.isfinite(work["y_true"]) & np.isfinite(work["y_pred"])]
    if work.empty:
        raise ValueError("No valid daytime row in the window.")

    baseline = (
        work.loc[~work["is_event"]]
        .groupby(["location", "hour"])["y_true"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "reference", "size": "n_baseline"})
    )
    baseline = baseline.loc[baseline["n_baseline"] >= min_daytime_rows]
    if baseline.empty:
        raise ValueError(
            "The quiet days provide no reference; increase baseline_days."
        )

    event = work.loc[work["is_event"]].merge(
        baseline, on=["location", "hour"], how="inner"
    )
    if event.empty:
        raise ValueError("No event row shares a node and hour with the baseline.")
    event["shortfall"] = 1.0 - event["y_true"] / event["reference"].replace(0.0, np.nan)
    event["error"] = event["y_pred"] - event["y_true"]
    event["abs_error"] = event["error"].abs()

    hit = event["shortfall"] >= shortfall
    onset = (
        event.loc[hit].groupby("location")["timestamp"].min().rename("onset")
    )
    event = event.merge(onset, on="location", how="left")
    reached = event["onset"].notna()
    event.loc[reached, "lag_hours"] = (
        (event.loc[reached, "timestamp"] - event.loc[reached, "onset"])
        / pd.Timedelta(hours=1)
    ).round().astype(int)

    covered = None
    if {"lower_pi", "upper_pi"} <= set(event.columns):
        covered = (event["y_true"] >= event["lower_pi"]) & (
            event["y_true"] <= event["upper_pi"]
        )
        event["covered"] = covered

    selected = event.loc[
        reached
        & event["lag_hours"].between(0, max_lag_hours)
    ]
    aggregations = {
        "n": ("error", "size"),
        "bias": ("error", "mean"),
        "mae": ("abs_error", "mean"),
        "over_share": ("error", lambda values: float(np.mean(values > 0))),
        "mean_shortfall": ("shortfall", "mean"),
        "mean_y_true": ("y_true", "mean"),
        "mean_y_pred": ("y_pred", "mean"),
        "mean_reference": ("reference", "mean"),
    }
    if covered is not None:
        aggregations["picp"] = ("covered", "mean")
    profile = (
        selected.groupby("lag_hours").agg(**aggregations).reset_index()
    )
    # How much of the drop the forecast reproduced, as a fraction of the drop
    # that actually happened. Scale-free, so it stays comparable across lags
    # even though the reference rises with the sun, and across models.
    actual_drop = 1.0 - profile["mean_y_true"] / profile["mean_reference"]
    predicted_drop = 1.0 - profile["mean_y_pred"] / profile["mean_reference"]
    profile["captured_share"] = np.where(
        actual_drop.abs() > 1e-6, predicted_drop / actual_drop, np.nan
    )

    never = event.loc[~reached]
    unaffected = {
        "n_nodes": int(never["location"].nunique()),
        "n_rows": int(len(never)),
        "bias": float(never["error"].mean()) if len(never) else float("nan"),
        "mae": float(never["abs_error"].mean()) if len(never) else float("nan"),
    }
    return {
        "profile": profile,
        "event_rows": event,
        "onset": onset,
        "n_nodes_reached": int(onset.size),
        "unaffected": unaffected,
        "shortfall": shortfall,
    }


def compare_onset_responses(responses: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """Put the onset profiles of several runs side by side.

    Absolute errors are not comparable between models trained on different
    feature sets, so the comparison leans on ``captured_share``: the fraction of
    the real drop each forecast reproduced at that lag.
    """
    if not responses:
        raise ValueError("At least one response is required.")
    frames = []
    for name, response in responses.items():
        profile = response["profile"].copy()
        if profile.empty:
            continue
        profile.insert(0, "run", name)
        frames.append(profile)
    if not frames:
        raise ValueError("Every profile is empty.")
    return pd.concat(frames, ignore_index=True)


def plot_onset_comparison(comparison: pd.DataFrame, *, title: str = ""):
    """Compare the reaction of several runs to the same event."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for name, block in comparison.groupby("run", sort=False):
        axes[0].plot(block["lag_hours"], block["captured_share"], lw=2, label=name)
        axes[1].plot(block["lag_hours"], block["bias"], lw=2, label=name)
        if "picp" in block:
            axes[2].plot(block["lag_hours"], block["picp"], lw=2, label=name)
    axes[0].axhline(1.0, color="black", ls="--", lw=1)
    axes[0].set(ylabel="quota di crollo catturata")
    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].set(ylabel="bias [W]")
    axes[2].axhline(0.95, color="black", ls="--", lw=1)
    axes[2].set(xlabel="ore dall'arrivo dell'evento sul nodo", ylabel="PICP")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(title or "Reazione all'evento: confronto fra run")
    fig.tight_layout()
    return fig


def plot_onset_response(response: Dict[str, object], *, title: str = ""):
    """Plot the error profile against hours since the event reached a node."""
    import matplotlib.pyplot as plt

    profile = response["profile"]
    if profile.empty:
        raise ValueError("The onset profile is empty; lower the shortfall.")
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    lag = profile["lag_hours"]

    # The gap between the two curves is the error; the grey line says what the
    # node would have produced without the event, so the distance between grey
    # and blue is the drop the model had to reproduce.
    axes[0].plot(lag, profile["mean_y_true"], label="produzione reale", lw=2)
    axes[0].plot(lag, profile["mean_y_pred"], label="previsione", lw=2)
    axes[0].plot(
        lag, profile["mean_reference"], label="riferimento pre-evento",
        ls="--", color="grey",
    )
    axes[0].fill_between(
        lag, profile["mean_y_true"], profile["mean_y_pred"],
        where=profile["mean_y_pred"] >= profile["mean_y_true"],
        color="tab:red", alpha=0.15, label="sovrastima",
    )
    axes[0].fill_between(
        lag, profile["mean_y_true"], profile["mean_y_pred"],
        where=profile["mean_y_pred"] < profile["mean_y_true"],
        color="tab:blue", alpha=0.15, label="sottostima",
    )
    axes[0].set(ylabel="W")
    axes[0].legend(fontsize=8)

    if "captured_share" in profile:
        axes[1].plot(lag, profile["captured_share"], color="tab:purple", lw=2)
        axes[1].axhline(1.0, color="black", ls="--", lw=1)
        axes[1].set(ylabel="crollo catturato")
        axes[1].annotate(
            "1.0 = il modello riproduce tutto il calo",
            xy=(lag.iloc[0], 1.0), xytext=(4, 4),
            textcoords="offset points", fontsize=8, color="grey",
        )

    axes[2].axhline(0.0, color="black", lw=1)
    axes[2].plot(lag, profile["bias"], color="tab:red", lw=2, label="bias")
    axes[2].fill_between(lag, 0.0, profile["bias"], color="tab:red", alpha=0.15)
    axes[2].set(ylabel="bias [W]")
    axes[2].legend(fontsize=8)

    if "picp" in profile:
        axes[3].plot(lag, profile["picp"], color="tab:green", lw=2, label="PICP")
        axes[3].axhline(0.95, color="black", ls="--", lw=1)
    axes[3].plot(
        lag, profile["mean_shortfall"], color="tab:orange", lw=2,
        label="calo di produzione",
    )
    axes[3].set(xlabel="ore dall'arrivo dell'evento sul nodo", ylabel="quota")
    axes[3].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(title or "Risposta del forecaster all'arrivo dell'evento")
    fig.tight_layout()
    return fig
