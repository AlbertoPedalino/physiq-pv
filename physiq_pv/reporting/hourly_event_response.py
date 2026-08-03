"""Hour-by-hour reaction to a single event, by expected production.

Binning on realised production conditions on the outcome: during a dust event a
node leaves the high bin *because* the event hit it, so a line for that bin
would describe the nodes the event spared rather than the model's reaction.
Bins here are therefore built on the production the node was expected to reach
at that hour, taken from the quiet days before the event, which makes bin
membership independent of what the event did.

Errors are reported relative to that same reference, so hours are comparable
despite the diurnal cycle, and signed, because the whole point is that the
forecast is too high while the event arrives and too low while it clears.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Expected-production bands, as a fraction of each node's own daily maximum
# reference.  Wider than the report bins: the split only has to separate hours
# when a node was meant to produce little from hours when it was meant to run
# near its peak.
EXPECTED_BANDS: Sequence[Tuple[str, float, Optional[float]]] = (
    ("atteso_0_25", 0.0, 0.25),
    ("atteso_25_50", 0.25, 0.50),
    ("atteso_50_75", 0.50, 0.75),
    ("atteso_75_100", 0.75, None),
)


def hourly_reference(
    window: pd.DataFrame,
    *,
    min_samples: int = 3,
) -> pd.DataFrame:
    """Expected production per node and hour of day, from the quiet days."""
    required = {"location", "hour", "y_true", "is_event"}
    missing = required - set(window.columns)
    if missing:
        raise ValueError(f"Window is missing columns {sorted(missing)}.")
    quiet = window.loc[~window["is_event"]]
    if quiet.empty:
        raise ValueError("The window holds no quiet day to build a reference.")
    reference = (
        quiet.groupby(["location", "hour"])["y_true"]
        .agg(reference="mean", n_reference="size")
        .reset_index()
    )
    reference = reference.loc[reference["n_reference"] >= min_samples]
    if reference.empty:
        raise ValueError("No node/hour has enough quiet samples.")
    peak = (
        reference.groupby("location")["reference"].max().rename("reference_peak")
    )
    reference = reference.merge(peak, on="location", how="left")
    return reference


def build_hourly_response(
    window: pd.DataFrame,
    *,
    bands: Sequence[Tuple[str, float, Optional[float]]] = EXPECTED_BANDS,
    min_reference_w: float = 5.0,
    min_samples: int = 3,
) -> Dict[str, object]:
    """Signed, reference-normalised error per hour and expected-production band."""
    reference = hourly_reference(window, min_samples=min_samples)
    work = window.merge(reference, on=["location", "hour"], how="inner")
    work = work.loc[
        np.isfinite(work["y_true"])
        & np.isfinite(work["y_pred"])
        & (work["reference"] > min_reference_w)
    ].copy()
    if work.empty:
        raise ValueError("No row survived the reference filter.")

    share = work["reference"] / work["reference_peak"].replace(0.0, np.nan)
    work["band"] = pd.NA
    for name, low, high in bands:
        mask = share >= low
        if high is not None:
            mask &= share < high
        work.loc[mask, "band"] = name
    work = work.dropna(subset=["band"])

    work["error"] = work["y_pred"] - work["y_true"]
    # Both normalised by the expected level, so an hour at dawn and an hour at
    # noon can sit on the same axis.
    work["rel_error"] = work["error"] / work["reference"]
    work["rel_true"] = work["y_true"] / work["reference"]
    work["rel_pred"] = work["y_pred"] / work["reference"]

    def quantile(name: str, q: float):
        return (name, lambda values, q=q: float(np.quantile(values, q)))

    grouped = work.groupby(["timestamp", "band"])
    profile = grouped.agg(
        n=("rel_error", "size"),
        rel_bias_mean=("rel_error", "mean"),
        rel_bias_median=("rel_error", "median"),
        rel_bias_q1=quantile("rel_error", 0.25),
        rel_bias_q3=quantile("rel_error", 0.75),
        rel_bias_p10=quantile("rel_error", 0.10),
        rel_bias_p90=quantile("rel_error", 0.90),
        mae=("error", lambda values: float(np.mean(np.abs(values)))),
        rel_mae=("rel_error", lambda values: float(np.mean(np.abs(values)))),
        rel_true=("rel_true", "mean"),
        rel_pred=("rel_pred", "mean"),
        mean_reference=("reference", "mean"),
    ).reset_index()
    if "lower_pi" in work and "upper_pi" in work:
        covered = (work["y_true"] >= work["lower_pi"]) & (
            work["y_true"] <= work["upper_pi"]
        )
        picp = (
            work.assign(covered=covered)
            .groupby(["timestamp", "band"])["covered"].mean()
            .rename("picp").reset_index()
        )
        profile = profile.merge(picp, on=["timestamp", "band"], how="left")
    profile["is_event"] = profile["timestamp"].isin(
        work.loc[work["is_event"], "timestamp"].unique()
    )
    return {
        "profile": profile.sort_values(["band", "timestamp"]).reset_index(drop=True),
        "rows": work,
        "bands": [name for name, _, _ in bands if (work["band"] == name).any()],
        "n_nodes": int(work["location"].nunique()),
    }


def persistence_check(
    response: Dict[str, object],
    *,
    event_only: bool = True,
    fast_change: float = 0.15,
) -> pd.DataFrame:
    """Is the forecast closer to the hour it predicts, or to the one before?

    Without a covariate for the target hour the best available predictor is the
    last observation, and a model can learn to reproduce it.  The comparison is
    made per node, not on hourly means, so it is not an artefact of averaging:
    for every node the forecast is scored against its own production at the
    target hour and at the hour before.

    Values are relative to the node's expected production, which keeps dawn and
    noon on the same scale.  The ``fast`` columns keep only the hours where
    production actually moved, since that is where persistence and prediction
    part company.
    """
    rows = response["rows"].copy()
    required = {"location", "timestamp", "rel_true", "rel_pred"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Rows are missing columns {sorted(missing)}.")
    rows = rows.sort_values(["location", "timestamp"])
    previous_time = rows.groupby("location")["timestamp"].shift(1)
    rows["rel_true_prev"] = rows.groupby("location")["rel_true"].shift(1)
    # Only consecutive hours qualify: a gap across the night is not persistence.
    consecutive = (rows["timestamp"] - previous_time) == pd.Timedelta(hours=1)
    rows = rows.loc[consecutive & rows["rel_true_prev"].notna()]
    if event_only:
        rows = rows.loc[rows["is_event"]]
    if rows.empty:
        raise ValueError("No consecutive-hour pair survived the filters.")

    rows["err_now"] = (rows["rel_pred"] - rows["rel_true"]).abs()
    rows["err_prev"] = (rows["rel_pred"] - rows["rel_true_prev"]).abs()
    rows["change"] = (rows["rel_true"] - rows["rel_true_prev"]).abs()

    def correlation(left: pd.Series, right: pd.Series) -> float:
        # A band whose values never move has no correlation to report; asking
        # for one only produces a divide-by-zero warning.
        if left.std(ddof=0) == 0 or right.std(ddof=0) == 0:
            return float("nan")
        return float(left.corr(right))

    def summarise(block: pd.DataFrame, band: str) -> dict:
        fast = block.loc[block["change"] > fast_change]
        entry = {
            "band": band,
            "n_ore_nodo": int(len(block)),
            "err_vs_ora_corrente": float(block["err_now"].mean()),
            "err_vs_ora_precedente": float(block["err_prev"].mean()),
            "corr_ora_corrente": correlation(block["rel_pred"], block["rel_true"]),
            "corr_ora_precedente": correlation(
                block["rel_pred"], block["rel_true_prev"]
            ),
            "n_transizioni": int(len(fast)),
        }
        entry["rapporto"] = (
            entry["err_vs_ora_corrente"] / entry["err_vs_ora_precedente"]
            if entry["err_vs_ora_precedente"] > 0
            else np.nan
        )
        if len(fast):
            entry["err_vs_corrente_transizioni"] = float(fast["err_now"].mean())
            entry["err_vs_precedente_transizioni"] = float(fast["err_prev"].mean())
            entry["rapporto_transizioni"] = (
                entry["err_vs_corrente_transizioni"]
                / entry["err_vs_precedente_transizioni"]
                if entry["err_vs_precedente_transizioni"] > 0
                else np.nan
            )
        return entry

    summary: List[dict] = [summarise(rows, "tutte")]
    for band, block in rows.groupby("band"):
        summary.append(summarise(block, str(band)))
    return pd.DataFrame(summary)


def plot_hourly_response(
    response: Dict[str, object],
    *,
    event_days: Iterable[str] = (),
    min_n: int = 20,
    title: str = "",
):
    """One panel per expected-production band, plus the sample size."""
    import matplotlib.pyplot as plt

    profile = response["profile"]
    bands = list(response["bands"])
    if profile.empty or not bands:
        raise ValueError("Nothing to plot.")
    days = [pd.Timestamp(day).normalize() for day in event_days]

    fig, axes = plt.subplots(
        len(bands) + 1, 1, figsize=(13, 2.4 * len(bands) + 2.4), sharex=True
    )
    colours = plt.get_cmap("tab10")
    for index, band in enumerate(bands):
        axis = axes[index]
        block = profile.loc[profile["band"] == band].sort_values("timestamp")
        thin = block["n"] < min_n
        axis.axhline(0.0, color="black", lw=1)
        axis.fill_between(
            block["timestamp"], block["rel_bias_p10"], block["rel_bias_p90"],
            color=colours(index), alpha=0.12,
        )
        axis.fill_between(
            block["timestamp"], block["rel_bias_q1"], block["rel_bias_q3"],
            color=colours(index), alpha=0.28,
        )
        axis.plot(
            block["timestamp"], block["rel_bias_median"], color=colours(index), lw=2,
        )
        axis.plot(
            block["timestamp"], block["rel_bias_mean"], color=colours(index),
            lw=1, ls=":",
        )
        # Hours computed on very few nodes are marked rather than hidden.
        if thin.any():
            axis.scatter(
                block.loc[thin, "timestamp"], block.loc[thin, "rel_bias_median"],
                s=12, facecolors="none", edgecolors="black", linewidths=0.6,
            )
        axis.set_ylabel(f"{band}\nbias relativo")
        axis.grid(alpha=0.25)

    for band, axis in zip(bands, axes):
        for day in days:
            axis.axvspan(day, day + pd.Timedelta(hours=23), color="tab:red", alpha=0.07)

    counts = axes[-1]
    for index, band in enumerate(bands):
        block = profile.loc[profile["band"] == band].sort_values("timestamp")
        counts.plot(block["timestamp"], block["n"], lw=1.5, label=band, color=colours(index))
    for day in days:
        counts.axvspan(day, day + pd.Timedelta(hours=23), color="tab:red", alpha=0.07)
    counts.set(ylabel="n nodi", xlabel="timestamp")
    counts.set_yscale("symlog")
    counts.grid(alpha=0.25)
    counts.legend(fontsize=8, ncol=len(bands))

    fig.suptitle(
        title
        or "Reazione oraria per fascia di produzione attesa "
           "(linea = mediana, punteggiata = media, bande = Q1-Q3 e P10-P90)"
    )
    fig.tight_layout()
    return fig


def plot_hourly_levels(
    response: Dict[str, object],
    *,
    band: str,
    event_days: Iterable[str] = (),
    title: str = "",
):
    """Expected, realised and predicted level for one band, as fractions."""
    import matplotlib.pyplot as plt

    profile = response["profile"]
    block = profile.loc[profile["band"] == band].sort_values("timestamp")
    if block.empty:
        raise ValueError(f"Band {band!r} holds no row.")
    fig, axis = plt.subplots(figsize=(13, 4.2))
    axis.axhline(1.0, color="grey", ls="--", lw=1, label="livello atteso")
    axis.plot(block["timestamp"], block["rel_true"], lw=2, label="reale / atteso")
    axis.plot(block["timestamp"], block["rel_pred"], lw=2, label="previsto / atteso")
    axis.fill_between(
        block["timestamp"], block["rel_true"], block["rel_pred"],
        where=block["rel_pred"] >= block["rel_true"],
        color="tab:red", alpha=0.15, label="sovrastima",
    )
    axis.fill_between(
        block["timestamp"], block["rel_true"], block["rel_pred"],
        where=block["rel_pred"] < block["rel_true"],
        color="tab:blue", alpha=0.15, label="sottostima",
    )
    for day in event_days:
        start = pd.Timestamp(day).normalize()
        axis.axvspan(start, start + pd.Timedelta(hours=23), color="tab:red", alpha=0.07)
    axis.set(ylabel="quota del livello atteso", xlabel="timestamp")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.suptitle(title or f"Livelli orari — {band}")
    fig.tight_layout()
    return fig
