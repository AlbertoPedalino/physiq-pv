import numpy as np
import xarray as xr

from physiq_pv.data.quality_score import (
    compute_qs,
    temporal_qs_slope,
    spatial_qs,
    fleet_mean_qs,
)
from physiq_pv.agent.drift_monitor import ADWINDriftMonitor
from physiq_pv.agent.qs_clustering import QSClusterer
from physiq_pv.agent.causal_classifier import CausalClassifier


class PhysiQAgent:
    """
    Full agentic diagnosis cycle (ATSF paradigm - Cheng et al. 2026 [R32]).

    Perception  -> compute_qs(ds)
    Planning    -> drift detection + soft-DTW clustering + ML causal diagnosis
    Action      -> drift-triggered retraining decision
    Reflection  -> post-retraining loss comparison
    Memory      -> DER++ replay buffer (managed by QualityGatedUpdater)

    Two entry points:
      run(ds)             - batch diagnosis on full historical dataset
      step(ds_window)     - one online cycle step on a sliding window
    """

    def __init__(self, n_clusters: int = 4, drift_window: int = 720):
        self.n_clusters = n_clusters
        self.drift_window = drift_window
        self._monitors: dict[int, ADWINDriftMonitor] = {}
        self._clusterer = QSClusterer(n_clusters=n_clusters)
        self._classifier = CausalClassifier(seq_len=drift_window)

    # ---------------------------------------------------------------------- #
    # Classifier training
    # ---------------------------------------------------------------------- #

    def train_classifier(self, ds: xr.Dataset) -> dict:
        """
        Fit CausalClassifier on QS sequences extracted from ds.
        For synthetic datasets: uses known fault-injection labels.
        For real datasets: call _classifier.fit(X, y) directly after labeling.

        Returns a summary of training data statistics.
        """
        qs = compute_qs(ds)
        X, y = CausalClassifier.make_synthetic_training_data(
            qs, window=self.drift_window
        )
        if len(X) == 0:
            return {"trained": False, "reason": "no_valid_windows"}

        unique, counts = np.unique(y, return_counts=True)
        self._classifier.fit(X, y)
        return {
            "trained": True,
            "n_samples": len(X),
            "class_counts": dict(zip(unique.tolist(), counts.tolist())),
        }

    # ---------------------------------------------------------------------- #
    # Batch entry point
    # ---------------------------------------------------------------------- #

    def run(self, ds: xr.Dataset) -> dict:
        """Batch diagnosis on full dataset."""
        qs = compute_qs(ds)
        return self._diagnose(qs)

    # ---------------------------------------------------------------------- #
    # Online entry point (ATSF loop)
    # ---------------------------------------------------------------------- #

    def step(
        self,
        ds_window: xr.Dataset,
        updater=None,
        qs: xr.DataArray | None = None,
    ) -> dict:
        """
        One online agentic cycle step on a sliding window of data.

        If updater is provided, triggers retraining when model drift is detected.
        QS diagnoses data quality and weights the loss; it is not a hard gate.

        qs: precomputed QS DataArray (skips internal compute_qs call if provided).

        Returns report dict. After optional retraining, call reflect() to
        add reflection summary.
        """
        if qs is None:
            qs = compute_qs(ds_window)
        report = self._diagnose(qs)

        # Action decision:
        # QS diagnoses data quality; it does not gate training samples.
        if updater is not None:
            drift_plants = [
                p for p, d in report["plant_diagnoses"].items()
                if d["drift"]
                and d["cause"] not in ("sensor_failure", "regional_cloud_event")
            ]
            if drift_plants:
                report["action"] = "retrain_triggered"
                report["action_plants"] = drift_plants
            else:
                report["action"] = "no_action"
                report["action_plants"] = []
        else:
            report["action"] = "no_updater"

        return report

    # ---------------------------------------------------------------------- #
    # Reflection
    # ---------------------------------------------------------------------- #

    def reflect(
        self,
        loss_before: float,
        loss_after: float,
        report: dict,
        tolerance: float = 0.05,
    ) -> dict:
        """
        Post-retraining reflection: compare loss before and after update.

        Adds "reflection" key to report:
          improvement_pct > tolerance  -> "improved"
          improvement_pct < -tolerance -> "degraded" (retraining hurt)
          otherwise                    -> "stable"
        """
        improvement = loss_before - loss_after
        pct = improvement / (abs(loss_before) + 1e-9) * 100.0

        report["reflection"] = {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "improvement_pct": round(pct, 2),
            "verdict": (
                "improved"  if pct > tolerance * 100 else
                "degraded"  if pct < -tolerance * 100 else
                "stable"
            ),
        }
        return report

    # ---------------------------------------------------------------------- #
    # Core diagnosis (shared by run and step)
    # ---------------------------------------------------------------------- #

    def _diagnose(self, qs: xr.DataArray) -> dict:
        n_plants = int(qs.shape[0])

        # --- Per-plant drift detection (ADWIN/KS) ---
        drift_flags: dict[int, bool] = {}
        for p in range(n_plants):
            if p not in self._monitors:
                self._monitors[p] = ADWINDriftMonitor(window_size=self.drift_window)
            qs_p = qs.isel(plant=p).values
            valid = qs_p[~np.isnan(qs_p)]
            drift_flags[p] = bool(self._monitors[p].update_batch(valid).any())

        # --- Clustering on weekly-aggregated QS (soft-DTW) ---
        qs_weekly = qs.resample(time="1W").mean()
        qs_mat = qs_weekly.values.copy()
        col_mean = np.nanmean(qs_mat, axis=0, keepdims=True)
        row_mean = np.nanmean(qs_mat, axis=1, keepdims=True)
        qs_mat = np.where(np.isnan(qs_mat), col_mean, qs_mat)
        qs_mat = np.where(np.isnan(qs_mat), row_mean, qs_mat)
        qs_mat = np.nan_to_num(qs_mat, nan=0.5)

        cluster_summary: dict = {}
        if qs_mat.shape[1] >= self.n_clusters:
            self._clusterer.fit(qs_mat)
            cluster_summary = self._clusterer.cluster_summary()

        # --- Fleet-level anomaly scores ---
        slopes = temporal_qs_slope(qs)
        z = spatial_qs(qs)
        fleet = fleet_mean_qs(qs)
        fleet_mean = float(fleet.mean(skipna=True))

        # --- Per-plant ML causal diagnosis ---
        plant_diagnoses: dict[int, dict] = {}
        for p in range(n_plants):
            qs_seq = qs.isel(plant=p).values
            cause, confidence = self._classifier.diagnose(qs_seq)
            slope = float(slopes[p])
            z_mean = float(z.isel(plant=p).mean(skipna=True))

            plant_diagnoses[p] = {
                "drift": drift_flags.get(p, False),
                "slope": slope,
                "z_mean": z_mean,
                "cause": cause,
                "confidence": round(confidence, 3),
            }

        return {
            "plant_diagnoses": plant_diagnoses,
            "cluster_summary": cluster_summary,
            "fleet_mean_qs": fleet_mean,
            "n_drifting": sum(drift_flags.values()),
        }
