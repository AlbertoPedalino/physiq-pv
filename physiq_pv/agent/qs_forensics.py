"""
QS drill-down: when aggregate QS drops, decompose into m1..m5 components
and produce a soft, adaptive suspicion score per component without hard
thresholds.

Two complementary signals fused per component:
  - fleet-relative: percentile of plant within fleet distribution NOW
  - self-baseline:  z-score of plant vs its own rolling history

Output is a continuous suspicion in [0, 1] per component. Downstream policy
consumes scores probabilistically; no if/elif on magic numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9
_M_KEYS = ("m1", "m2", "m3", "m4", "m5")

# Hypothesis mapping per component (lowest m_i = strongest evidence for cause)
_CAUSE_MAP = {
    "m1": "shape_distortion",       # corr collapsed -> shading / soiling / inverter
    "m2": "bias_drift",             # offset -> degradation / miscalibration
    "m3": "missing_data",           # NaN spike -> sensor offline / comms
    "m4": "noise_anomaly",          # variance spike -> sensor fault
    "m5": "physics_violation",      # eta(T) breach -> structural degradation
}

# Action family per cause (proposal, not enforced; downstream policy weighs them)
_ACTION_FAMILY = {
    "shape_distortion":   ("skip_update", "alert_cleaning"),
    "bias_drift":         ("trigger_update",),
    "missing_data":       ("fallback_pvgis", "alert_comms"),
    "noise_anomaly":      ("skip_update", "alert_sensor"),
    "physics_violation":  ("alert_maintenance", "preserve_replay"),
}


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def fleet_relative_score(m_now: np.ndarray) -> np.ndarray:
    """
    For one component at one time, returns per-plant suspicion in [0, 1].
    Low percentile -> high suspicion (1 = worst plant in fleet).

    Args:
        m_now: shape (N,), one m_i across fleet at time t.

    Returns:
        suspicion: shape (N,), in [0, 1]. NaN inputs propagate as NaN.
    """
    valid = ~np.isnan(m_now)
    susp = np.full_like(m_now, np.nan, dtype=float)
    if valid.sum() < 2:
        return susp
    ranks = np.empty_like(m_now)
    ranks[:] = np.nan
    sorted_vals = np.sort(m_now[valid])
    # left-side percentile: pct of plants with strictly larger m_i (= better)
    for i in np.where(valid)[0]:
        pct_worse_or_eq = np.searchsorted(sorted_vals, m_now[i], side="right") / valid.sum()
        ranks[i] = pct_worse_or_eq
    # low m_i -> low rank -> high suspicion
    susp = 1.0 - ranks
    return susp


def self_baseline_score(
    m_history: np.ndarray, m_now: np.ndarray
) -> np.ndarray:
    """
    Per-plant z-score vs own history, mapped through sigmoid to [0, 1].
    Negative z (current below own mean) -> high suspicion.

    Args:
        m_history: shape (N, H), rolling history per plant.
        m_now:     shape (N,), current value per plant.

    Returns:
        suspicion: shape (N,), in [0, 1]. Plants with <5 valid history points
                   get NaN.
    """
    N = m_now.shape[0]
    susp = np.full(N, np.nan, dtype=float)
    for p in range(N):
        hist = m_history[p][~np.isnan(m_history[p])]
        if len(hist) < 5 or np.isnan(m_now[p]):
            continue
        mu = float(hist.mean())
        sigma = float(hist.std())
        if sigma < _EPS:
            continue
        z = (m_now[p] - mu) / sigma
        susp[p] = float(_sigmoid(-z))  # negative z -> close to 1
    return susp


def combined_suspicion(
    m_now: np.ndarray,
    m_history: np.ndarray,
    w_fleet: float = 0.5,
) -> np.ndarray:
    """
    Convex combination of fleet-relative and self-baseline scores.

    Falls back gracefully:
      - cold-start plant (no history) -> fleet-only score.
      - homogeneous fleet (single valid plant) -> self-only score.
      - both unavailable -> NaN.
    """
    fleet = fleet_relative_score(m_now)
    self_ = self_baseline_score(m_history, m_now)

    both = ~np.isnan(fleet) & ~np.isnan(self_)
    only_f = ~np.isnan(fleet) & np.isnan(self_)
    only_s = np.isnan(fleet) & ~np.isnan(self_)

    out = np.full_like(m_now, np.nan, dtype=float)
    out[both] = w_fleet * fleet[both] + (1.0 - w_fleet) * self_[both]
    out[only_f] = fleet[only_f]
    out[only_s] = self_[only_s]
    return out


@dataclass(frozen=True)
class PlantForensics:
    """Per-plant drill-down report at a single time step."""
    scores: dict[str, float]        # suspicion per m_i in [0, 1]
    dominant: str                   # m_i with max suspicion
    confidence: float               # dominance of top vs rest, in [0, 1]
    cause: str                      # hypothesised cause label
    actions: tuple[str, ...]        # candidate action family
    mode: str                       # "auto" / "conservative" / "uncertain"


def diagnose_plant(scores: dict[str, float]) -> PlantForensics:
    """
    Reduce per-component suspicion dict into ranked diagnosis with
    confidence-driven mode selection.

    Mode bands derived from dominance, not raw suspicion:
      dominance = (top - mean_rest) / (top + eps)
      >0.5  -> auto
      >0.2  -> conservative
      else  -> uncertain
    """
    valid = {k: v for k, v in scores.items() if not np.isnan(v)}
    if not valid:
        return PlantForensics(
            scores=scores, dominant="none", confidence=0.0,
            cause="unknown", actions=(), mode="uncertain",
        )

    dominant = max(valid, key=valid.get)
    top = valid[dominant]
    rest = [v for k, v in valid.items() if k != dominant]
    mean_rest = float(np.mean(rest)) if rest else 0.0
    dominance = (top - mean_rest) / (top + _EPS) if top > _EPS else 0.0
    dominance = float(np.clip(dominance, 0.0, 1.0))

    if dominance > 0.5 and top > 0.5:
        mode = "auto"
    elif dominance > 0.2 and top > 0.3:
        mode = "conservative"
    else:
        mode = "uncertain"

    cause = _CAUSE_MAP.get(dominant, "unknown")
    actions = _ACTION_FAMILY.get(cause, ())

    return PlantForensics(
        scores=scores,
        dominant=dominant,
        confidence=dominance,
        cause=cause,
        actions=actions,
        mode=mode,
    )


def forensic_report(
    m_components: dict[str, np.ndarray],
    t_idx: int,
    history_window: int = 720,
    w_fleet: float = 0.5,
    recent_window: int = 24,
) -> dict[int, PlantForensics]:
    """
    Run drill-down for every plant around time t_idx.

    Robust to boundary NaNs: the "now" snapshot is the nanmean over the last
    `recent_window` timesteps (e.g. 24h). The self-baseline history is the
    block ending just before that window.

    Args:
        m_components: {"m1".."m5"} each ndarray (N, T).
        t_idx:        current time index (right edge, inclusive in slicing).
        history_window: rolling window for self-baseline (default 30 days).
        w_fleet:      weight of fleet-relative vs self-baseline score.
        recent_window: span used to summarise the "now" value of each m_i.

    Returns:
        dict plant_idx -> PlantForensics.
    """
    if not all(k in m_components for k in _M_KEYS):
        raise KeyError(f"m_components must contain {_M_KEYS}")

    N = m_components["m1"].shape[0]
    now_hi = t_idx + 1
    now_lo = max(0, now_hi - recent_window)
    hist_hi = now_lo
    hist_lo = max(0, hist_hi - history_window)

    per_component_susp: dict[str, np.ndarray] = {}
    for m in _M_KEYS:
        arr = m_components[m]
        if t_idx >= arr.shape[1]:
            raise IndexError(f"t_idx={t_idx} out of range for {m} with T={arr.shape[1]}")
        recent_block = arr[:, now_lo:now_hi]
        with np.errstate(invalid="ignore"):
            m_now = np.nanmean(recent_block, axis=1)
        m_hist = arr[:, hist_lo:hist_hi] if hist_hi > hist_lo else recent_block
        per_component_susp[m] = combined_suspicion(m_now, m_hist, w_fleet=w_fleet)

    out: dict[int, PlantForensics] = {}
    for p in range(N):
        scores_p = {m: float(per_component_susp[m][p]) for m in _M_KEYS}
        out[p] = diagnose_plant(scores_p)
    return out


def fleet_summary(report: dict[int, PlantForensics]) -> dict:
    """Aggregate plant-level forensics into fleet-level diagnostics."""
    causes: dict[str, int] = {}
    modes: dict[str, int] = {}
    susp_vals: list[float] = []
    for r in report.values():
        causes[r.cause] = causes.get(r.cause, 0) + 1
        modes[r.mode] = modes.get(r.mode, 0) + 1
        susp_vals.extend(v for v in r.scores.values() if not np.isnan(v))
    return {
        "n_plants": len(report),
        "cause_counts": causes,
        "mode_counts": modes,
        "mean_suspicion": float(np.mean(susp_vals)) if susp_vals else float("nan"),
        "p95_suspicion": float(np.percentile(susp_vals, 95)) if susp_vals else float("nan"),
    }
