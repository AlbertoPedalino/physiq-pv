"""
Parametric action policy for the ATSF Planning module.

Replaces hardcoded component-to-action mapping with a utility-based policy:

    state -> per-action utility -> softmax distribution -> sampled / argmax action

State is a structured dataclass aggregating the signals that the previous
diagnose step produced: per-component suspicion, drift flag, CI width,
forensic mode and a couple of fleet-level summaries.

Utilities are linear combinations of state features. Coefficients are
exposed in `default_action_weights` and meant to be tuned (or learned)
later. The MVP keeps them rule-derived but in a single dict, so swapping in
a learned policy later is a drop-in.

Reference: ATSF formalism (Cheng et al. 2026) Planning + Action separation;
A2ER (Frontiers AI 2026) output-gated continual learning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


ACTIONS: tuple[str, ...] = (
    "do_nothing",
    "trigger_update",
    "skip_update",
    "fallback_pvgis",
    "alert_sensor",
    "alert_cleaning",
    "alert_maintenance",
    "alert_comms",
    "preserve_replay",
)


@dataclass
class PolicyState:
    """
    Input to UtilityActionPolicy. All fields default-safe so callers can
    populate only what they have.

    suspicion: per-component suspicion scores in [0, 1]; NaN treated as 0.
    drift_flag: True if upstream drift detector fires.
    ci_width: width of the conformal interval (None if CP not wired yet).
    mode: forensic mode label ("auto", "conservative", "uncertain").
    qs_mean / fleet_qs: scalar summaries (any sample, any plant).
    """
    suspicion: dict[str, float] = field(default_factory=dict)
    drift_flag: bool = False
    ci_width: float | None = None
    mode: str = "uncertain"
    qs_mean: float = 1.0
    fleet_qs: float = 1.0

    def _susp(self, key: str) -> float:
        v = self.suspicion.get(key, 0.0)
        return 0.0 if v is None or (isinstance(v, float) and v != v) else float(v)


# ---------------------------------------------------------------------- #
# Default rule-derived utility weights (MVP).
# Each action maps to a list of (feature_extractor, coeff) pairs. Features
# read from PolicyState; coeff is the linear weight. A constant bias is set
# under key "_bias".
# ---------------------------------------------------------------------- #

def _bias(b: float):
    return ("_bias", b)


def _from(field_or_susp: str, c: float):
    return (field_or_susp, c)


def default_action_weights() -> dict[str, list[tuple[str, float]]]:
    return {
        "do_nothing":      [_bias(0.0), _from("drift_flag", -1.5), _from("max_susp", -1.5)],
        "trigger_update":  [_bias(-0.3), _from("drift_flag", 2.0),
                            _from("m2", 1.2), _from("max_susp", -2.0),
                            _from("qs_mean", 1.0)],
        "skip_update":     [_bias(-0.2), _from("max_susp", 1.8),
                            _from("m1", 0.8), _from("m4", 1.0)],
        "fallback_pvgis":  [_bias(-0.5), _from("m3", 2.5)],
        "alert_sensor":    [_bias(-0.7), _from("m4", 2.0), _from("max_susp", 0.5)],
        "alert_cleaning":  [_bias(-0.8), _from("m1", 1.5)],
        "alert_maintenance": [_bias(-0.8), _from("m5", 1.8), _from("m2", 0.6)],
        "alert_comms":     [_bias(-0.9), _from("m3", 1.8)],
        "preserve_replay": [_bias(-0.4), _from("m5", 0.6), _from("m1", 0.4)],
    }


# ---------------------------------------------------------------------- #
# Feature extraction
# ---------------------------------------------------------------------- #

_M_KEYS = ("m1", "m2", "m3", "m4", "m5")


def state_features(state: PolicyState) -> dict[str, float]:
    """Flatten PolicyState into a flat dict consumed by the linear utility."""
    susp_values = [state._susp(k) for k in _M_KEYS]
    max_susp = float(max(susp_values)) if susp_values else 0.0

    feat = {
        "_bias": 1.0,
        "drift_flag": 1.0 if state.drift_flag else 0.0,
        "ci_width": float(state.ci_width) if state.ci_width is not None else 0.0,
        "qs_mean": float(state.qs_mean),
        "fleet_qs": float(state.fleet_qs),
        "mode_auto": 1.0 if state.mode == "auto" else 0.0,
        "mode_conservative": 1.0 if state.mode == "conservative" else 0.0,
        "mode_uncertain": 1.0 if state.mode == "uncertain" else 0.0,
        "max_susp": max_susp,
    }
    for k in _M_KEYS:
        feat[k] = state._susp(k)
    return feat


# ---------------------------------------------------------------------- #
# Policy
# ---------------------------------------------------------------------- #

class UtilityActionPolicy:
    """
    Linear utility policy with softmax-over-utility action distribution.

    Designed so that the same interface (`distribution`, `sample`,
    `recommend`) survives a future swap to a learned policy (bandit, RL,
    neural net). Only `_utility` needs to change.
    """

    def __init__(
        self,
        weights: dict[str, list[tuple[str, float]]] | None = None,
        temperature: float = 1.0,
        uncertain_temperature: float = 2.0,
        actions: Sequence[str] = ACTIONS,
    ):
        self.weights = weights if weights is not None else default_action_weights()
        self.temperature = float(temperature)
        self.uncertain_temperature = float(uncertain_temperature)
        self.actions = tuple(actions)

        missing = [a for a in self.actions if a not in self.weights]
        if missing:
            raise ValueError(f"missing utility weights for actions: {missing}")

    def _utility(self, state: PolicyState, action: str) -> float:
        feats = state_features(state)
        u = 0.0
        for feat_name, coeff in self.weights[action]:
            u += float(coeff) * float(feats.get(feat_name, 0.0))
        return u

    def distribution(self, state: PolicyState) -> dict[str, float]:
        """Softmax over utilities. Temperature scales with mode uncertainty."""
        utilities = np.array([self._utility(state, a) for a in self.actions], dtype=np.float64)

        temp = self.uncertain_temperature if state.mode == "uncertain" else self.temperature
        temp = max(temp, 1e-3)

        z = utilities / temp
        z -= z.max()
        e = np.exp(z)
        p = e / e.sum()
        return dict(zip(self.actions, [float(x) for x in p]))

    def recommend(self, state: PolicyState) -> tuple[str, dict[str, float]]:
        """Argmax action + the full distribution."""
        dist = self.distribution(state)
        chosen = max(dist, key=dist.get)
        return chosen, dist

    def sample(
        self,
        state: PolicyState,
        rng: np.random.Generator | None = None,
    ) -> tuple[str, dict[str, float]]:
        """Stochastic sample from the distribution."""
        dist = self.distribution(state)
        rng = rng if rng is not None else np.random.default_rng()
        actions = list(dist.keys())
        probs = np.array(list(dist.values()), dtype=np.float64)
        probs /= probs.sum()
        choice = rng.choice(len(actions), p=probs)
        return actions[choice], dist
