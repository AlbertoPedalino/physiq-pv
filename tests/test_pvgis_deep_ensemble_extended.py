"""Smoke test: Deep Ensemble aggregator + extended strata + interval_miss chain.

Builds 3 synthetic per-seed .npz members and a matching reference
predictions.csv, runs the aggregator with --reference-predictions, and then
runs the interval_miss CLI on the emitted deep_ensemble_predictions_full.csv.
All eval-only; no training involved.
"""

from pathlib import Path
import sys
import tempfile

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd


def _make_member_and_reference(tmp: Path) -> Path:
    """3 seeds x (4 timestamps x 2 locations) = 8 aligned samples per member."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2019-06-01 10:00", periods=4, freq="h")
    locs = ["loc_a", "loc_b"]
    rows = [(t, l) for t in times for l in locs]
    ts_ns = np.array([pd.Timestamp(t).value for t, _ in rows], dtype=np.int64)
    loc_arr = np.array([l for _, l in rows], dtype=str)
    sample_id = np.array([f"{t}_{l}" for t, l in zip(ts_ns, loc_arr)], dtype=str)
    y_true = rng.uniform(0.0, 150.0, size=len(rows))
    group = np.array(
        ["normal", "rare_or_extreme"] * (len(rows) // 2), dtype=str
    )

    pred_dir = tmp / "predictions"
    pred_dir.mkdir(parents=True)
    for seed in (1, 2, 3):
        member_rng = np.random.default_rng(seed)
        np.savez_compressed(
            pred_dir / f"run_seed{seed}.npz",
            sample_id=sample_id,
            y_true=y_true,
            y_pred_mean=y_true + member_rng.normal(0.0, 5.0, size=len(rows)),
            anomaly_group=group,
            seed=np.asarray(seed),
            timestamp=ts_ns,
            location_id=loc_arr,
            y_pred_std_mc=np.full(len(rows), 2.0),
        )

    reference = pd.DataFrame(
        {
            "timestamp": [t for t, _ in rows],
            "location": loc_arr,
            "y_true": y_true,
            "solar_irradiance_poa_target": rng.uniform(50.0, 800.0, size=len(rows)),
            "anomaly_group": group,
            "anomaly_label": np.where(
                group == "rare_or_extreme", "unusually_high_solar_potential", ""
            ),
        }
    )
    ref_path = tmp / "member_seed1_predictions.csv"
    reference.to_csv(ref_path, index=False)
    return ref_path


def test_aggregator_extended_and_interval_miss_chain() -> None:
    import analyze_pvgis_deep_ensemble as agg
    from analyze_pvgis_interval_miss_distance import main as interval_miss_main

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ref_path = _make_member_and_reference(tmp)
        out_dir = tmp / "analysis"

        argv_backup = sys.argv
        sys.argv = [
            "analyze_pvgis_deep_ensemble.py",
            "--predictions-dir", str(tmp / "predictions"),
            "--out-dir", str(out_dir),
            "--coverage-target", "0.95",
            "--clc-eta", "10",
            "--reference-predictions", str(ref_path),
        ]
        try:
            agg.main()
        finally:
            sys.argv = argv_backup

        for name in (
            "deep_ensemble_metrics.json",
            "deep_ensemble_report.md",
            "deep_ensemble_predictions_summary.csv",
            "deep_ensemble_predictions_full.csv",
            "deep_ensemble_extended_strata.csv",
        ):
            assert (out_dir / name).exists(), f"missing {name}"

        full = pd.read_csv(out_dir / "deep_ensemble_predictions_full.csv")
        for col in (
            "y_true", "y_pred_mean", "y_pred_std", "lower_pi", "upper_pi",
            "solar_irradiance_poa_target", "anomaly_group", "anomaly_label",
        ):
            assert col in full.columns, f"full csv missing {col}"
        assert len(full) == 8

        ext = pd.read_csv(out_dir / "deep_ensemble_extended_strata.csv")
        assert set(ext["stratum"]) >= {
            "global", "daytime", "normal_daytime", "rare_extreme_daytime",
            "label:unusually_high_solar_potential", "daytime_gt_100",
        }
        g = ext[ext["stratum"] == "global"].iloc[0]
        assert g["count"] == 8
        assert 0.0 <= g["picp"] <= 1.0
        assert g["mpiw"] > 0.0

        # Chain: interval_miss runs unchanged on the ensemble full CSV.
        miss_out = tmp / "interval_miss"
        paths = interval_miss_main([
            "--predictions", str(out_dir / "deep_ensemble_predictions_full.csv"),
            "--out-dir", str(miss_out),
        ])
        for path in paths.values():
            assert path.exists(), f"missing interval_miss output {path}"


def test_split_sample_ids_handles_underscored_locations() -> None:
    import analyze_pvgis_deep_ensemble as agg

    ts = pd.Timestamp("2019-06-01 10:00").value
    sample_id = np.array([f"{ts}_loc_with_underscores"], dtype=str)
    out_ts, out_loc = agg._split_sample_ids(sample_id)
    assert out_ts[0] == ts
    assert out_loc[0] == "loc_with_underscores"


if __name__ == "__main__":
    test_split_sample_ids_handles_underscored_locations()
    test_aggregator_extended_and_interval_miss_chain()
    print("PASS: PVGIS deep ensemble extended aggregation")
