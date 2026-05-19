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
from physiq_pv.agent.qs_forensics import forensic_report, fleet_summary
from physiq_pv.agent.action_policy import UtilityActionPolicy, PolicyState


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

    def __init__(
        self,
        n_clusters: int = 4,
        drift_window: int = 720,
        action_policy: UtilityActionPolicy | None = None,
    ):
        self.n_clusters = n_clusters
        self.drift_window = drift_window
        self._monitors: dict[int, ADWINDriftMonitor] = {}
        self._clusterer = QSClusterer(n_clusters=n_clusters)
        self._classifier = CausalClassifier(seq_len=drift_window)
        self.action_policy = action_policy if action_policy is not None else UtilityActionPolicy()

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
        qs, m_components = compute_qs(ds, debug=True)
        return self._diagnose(qs, m_components=m_components)

    # ---------------------------------------------------------------------- #
    # Online entry point (ATSF loop)
    # ---------------------------------------------------------------------- #

    def step(
        self,
        ds_window: xr.Dataset,
        updater=None,
        qs: xr.DataArray | None = None,
        m_components: dict | None = None,
        ci_width: float | None = None,
    ) -> dict:
        """
        One online agentic cycle step on a sliding window of data.

        ci_width: optional conformal interval width feeding the action policy.

        Returns report dict. After optional retraining, call reflect() to
        add reflection summary.
        """
        if qs is None:
            qs, m_components = compute_qs(ds_window, debug=True)
        report = self._diagnose(qs, m_components=m_components)

        # Aggregate fleet-level signals for action policy.
        forensic_sum = report.get("forensic_summary", {}) or {}
        n_plants = len(report.get("plant_diagnoses", {}))
        n_drift = int(report.get("n_drifting", 0))
        drift_frac = n_drift / n_plants if n_plants else 0.0

        # Build fleet-level suspicion dict by averaging per-component scores
        # across plants. Missing forensics -> empty dict -> max_susp=0.
        component_scores: dict[str, list[float]] = {k: [] for k in ("m1", "m2", "m3", "m4", "m5")}
        modes: dict[str, int] = {}
        for diag in report.get("plant_diagnoses", {}).values():
            f = diag.get("forensics")
            if not f:
                continue
            for k, v in f.get("scores", {}).items():
                if v is not None and v == v:
                    component_scores[k].append(float(v))
            modes[f.get("mode", "uncertain")] = modes.get(f.get("mode", "uncertain"), 0) + 1
        susp_fleet = {
            k: (float(sum(vals) / len(vals)) if vals else 0.0)
            for k, vals in component_scores.items()
        }
        fleet_mode = max(modes, key=modes.get) if modes else "uncertain"

        if updater is not None:
            state = PolicyState(
                suspicion=susp_fleet,
                drift_flag=drift_frac > 0.1,
                ci_width=ci_width,
                mode=fleet_mode,
                qs_mean=float(forensic_sum.get("mean_suspicion", 0.0) and 1.0 - forensic_sum["mean_suspicion"])
                        if "mean_suspicion" in forensic_sum else float(report.get("fleet_mean_qs", 1.0)),
                fleet_qs=float(report.get("fleet_mean_qs", 1.0)),
            )
            chosen, dist = self.action_policy.recommend(state)
            report["policy_action"] = chosen
            report["policy_distribution"] = {k: round(v, 3) for k, v in dist.items()}

            if chosen == "trigger_update":
                report["action"] = "retrain_triggered"
            elif chosen == "do_nothing":
                report["action"] = "no_action"
            else:
                report["action"] = chosen
        else:
            report["action"] = "no_updater"
            report["policy_action"] = None

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

    def _diagnose(
        self,
        qs: xr.DataArray,
        m_components: dict | None = None,
    ) -> dict:
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

        # --- QS drill-down forensics (soft adaptive, no thresholds) ---
        forensics: dict[int, object] = {}
        forensic_fleet: dict = {}
        if m_components is not None:
            t_last = int(qs.shape[1]) - 1
            try:
                forensics = forensic_report(
                    m_components, t_idx=t_last, history_window=self.drift_window
                )
                forensic_fleet = fleet_summary(forensics)
            except (KeyError, IndexError):
                forensics = {}

        # --- Per-plant ML causal diagnosis ---
        plant_diagnoses: dict[int, dict] = {}
        for p in range(n_plants):
            qs_seq = qs.isel(plant=p).values
            cause, confidence = self._classifier.diagnose(qs_seq)
            slope = float(slopes[p])
            z_mean = float(z.isel(plant=p).mean(skipna=True))

            entry = {
                "drift": drift_flags.get(p, False),
                "slope": slope,
                "z_mean": z_mean,
                "cause": cause,
                "confidence": round(confidence, 3),
            }
            if p in forensics:
                f = forensics[p]
                entry["forensics"] = {
                    "scores": {k: round(v, 3) for k, v in f.scores.items()},
                    "dominant": f.dominant,
                    "dominance": round(f.confidence, 3),
                    "drill_cause": f.cause,
                    "actions": list(f.actions),
                    "mode": f.mode,
                }
            plant_diagnoses[p] = entry

        return {
            "plant_diagnoses": plant_diagnoses,
            "cluster_summary": cluster_summary,
            "fleet_mean_qs": fleet_mean,
            "n_drifting": sum(drift_flags.values()),
            "forensic_summary": forensic_fleet,
        }
