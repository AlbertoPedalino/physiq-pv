"""Sensor GMMs and Gamma-calibrated Fisher aggregation for M2AD."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.special import ndtr
from scipy.stats import gamma
from sklearn.mixture import GaussianMixture


_P_VALUE_FLOOR = 1e-16
_EPS = 1e-12


def _gmm_cdf(model: GaussianMixture, values: np.ndarray) -> np.ndarray:
    means = model.means_.reshape(-1)
    if model.covariance_type == "spherical":
        variances = model.covariances_.reshape(-1)
    elif model.covariance_type == "diag":
        variances = model.covariances_.reshape(model.n_components, -1)[:, 0]
    elif model.covariance_type == "tied":
        variances = np.repeat(
            np.asarray(model.covariances_).reshape(-1)[0], model.n_components
        )
    else:
        variances = model.covariances_.reshape(model.n_components, -1)[:, 0]
    sigma = np.sqrt(np.maximum(variances, _EPS))
    z = (
        np.asarray(values, dtype=np.float64)[:, None] - means[None, :]
    ) / sigma[None, :]
    return np.sum(model.weights_[None, :] * ndtr(z), axis=1)


class M2ADCalibrator:
    """Fit GMM p-values and the moment-matched global Gamma distribution."""

    def __init__(
        self,
        sensor_names: Sequence[str],
        *,
        error: str,
        n_components: int | str | Mapping[str, int] = "bic",
        max_components: int = 3,
        covariance_type: str = "spherical",
        sensor_weights: Sequence[float] | None = None,
        significance: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.sensor_names = [str(name) for name in sensor_names]
        self.n_sensors = len(self.sensor_names)
        self.error = error
        self.n_components = n_components
        self.max_components = int(max_components)
        self.covariance_type = covariance_type
        self.significance = float(significance)
        self.seed = int(seed)
        if error not in {"point", "area"}:
            raise ValueError("error must be either 'point' or 'area'")
        if self.max_components < 1:
            raise ValueError("max_components must be >= 1")
        if not 0.0 < self.significance < 1.0:
            raise ValueError("significance must be in (0, 1)")
        if sensor_weights is None:
            self.sensor_weights = np.ones(self.n_sensors, dtype=np.float64)
        else:
            weights = np.asarray(sensor_weights, dtype=np.float64)
            if (
                weights.shape != (self.n_sensors,)
                or np.any(weights <= 0)
                or not np.isfinite(weights).all()
            ):
                raise ValueError(
                    "sensor_weights must contain one finite positive value per sensor"
                )
            self.sensor_weights = weights
        self.gmms: list[GaussianMixture] = []
        self.selected_components: dict[str, int] = {}
        self.is_fitted = False

    def _component_count(self, sensor: str, values: np.ndarray) -> int:
        if isinstance(self.n_components, Mapping):
            if sensor not in self.n_components:
                raise ValueError(f"n_components has no entry for sensor {sensor!r}")
            return int(self.n_components[sensor])
        if isinstance(self.n_components, (int, np.integer)):
            return int(self.n_components)
        if self.n_components != "bic":
            raise ValueError("n_components must be an int, a sensor mapping, or 'bic'")
        candidates: list[tuple[float, int]] = []
        for count in range(1, min(self.max_components, len(values)) + 1):
            candidate = self._new_gmm(count).fit(values[:, None])
            candidates.append((float(candidate.bic(values[:, None])), count))
        return min(candidates)[1]

    def _new_gmm(self, count: int) -> GaussianMixture:
        return GaussianMixture(
            n_components=count,
            covariance_type=self.covariance_type,
            reg_covar=1e-6,
            n_init=3,
            random_state=self.seed,
        )

    def fit(self, errors: np.ndarray) -> "M2ADCalibrator":
        values = np.asarray(errors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_sensors:
            raise ValueError(
                f"errors must have shape (time, {self.n_sensors}), got {values.shape}"
            )
        self.gmms = []
        self.selected_components = {}
        for sensor_index, sensor in enumerate(self.sensor_names):
            sensor_values = values[:, sensor_index]
            count = self._component_count(sensor, sensor_values)
            if count < 1:
                raise ValueError(f"GMM component count for {sensor!r} must be >= 1")
            self.gmms.append(self._new_gmm(count).fit(sensor_values[:, None]))
            self.selected_components[sensor] = count

        _, _, _, global_scores = self.score_errors(values)
        score_mean = float(np.mean(global_scores))
        score_variance = float(np.var(global_scores))
        if score_mean <= 0 or score_variance <= _EPS:
            raise ValueError(
                "training Fisher scores have insufficient variance for Gamma calibration"
            )
        self.gamma_shape = score_mean**2 / score_variance
        self.gamma_scale = score_variance / score_mean
        self.threshold = float(
            gamma.ppf(
                1.0 - self.significance,
                a=self.gamma_shape,
                scale=self.gamma_scale,
            )
        )
        self.is_fitted = True
        return self

    def score_errors(self, errors: np.ndarray) -> tuple[np.ndarray, ...]:
        if len(self.gmms) != self.n_sensors:
            raise RuntimeError("GMM calibration is not fitted")
        values = np.asarray(errors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_sensors:
            raise ValueError(
                f"errors must have shape (time, {self.n_sensors}), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("errors must be finite")
        p_values = np.empty_like(values)
        for sensor_index, model in enumerate(self.gmms):
            cdf = _gmm_cdf(model, values[:, sensor_index])
            p_value = 1.0 - cdf if self.error == "point" else 2.0 * np.minimum(cdf, 1.0 - cdf)
            p_values[:, sensor_index] = np.clip(
                p_value, _P_VALUE_FLOOR, 1.0
            )
        fisher = -2.0 * np.log(p_values)
        contributions = fisher * self.sensor_weights[None, :]
        global_scores = np.sum(contributions, axis=1)
        return p_values, fisher, contributions, global_scores

    def score(self, errors: np.ndarray) -> tuple[np.ndarray, ...]:
        if not self.is_fitted:
            raise RuntimeError("call fit before score")
        p_values, fisher, contributions, global_scores = self.score_errors(errors)
        gamma_p_values = gamma.sf(
            global_scores, a=self.gamma_shape, scale=self.gamma_scale
        )
        anomalies = global_scores > self.threshold
        return (
            p_values,
            fisher,
            contributions,
            global_scores,
            gamma_p_values,
            anomalies,
        )
