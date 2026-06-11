"""Tests for the post-hoc interval-miss diagnostics (eval-only, watt space).

Exact hand-computed counts/distances/quantiles on synthetic predictions, PICP
curve monotonicity + exact values, empty-stratum behaviour, and a CLI smoke run
of scripts/analyze_pvgis_interval_miss_distance.py on a temp CSV.
"""

from pathlib import Path
import sys
import tempfile

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from physiq_pv.eval.interval_miss import (
    build_strata_masks,
    compute_interval_miss_table,
    compute_picp_curve_table,
    interval_miss_row,
    render_interval_miss_report,
)


def _toy_predictions() -> pd.DataFrame:
    """6 rows, watt space. Interval [10, 20] everywhere; std=2 everywhere.

    row  y_true  position        outside distance
    0    15      inside          0
    1    25      above by 5      5
    2    30      above by 10     10
    3    5       below by 5      5
    4    10      inside (bound)  0
    5    20      inside (bound)  0
    """
    n = 6
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2019-06-01 10:00", periods=n, freq="h"),
            "location": ["loc_a"] * n,
            "y_true": [15.0, 25.0, 30.0, 5.0, 10.0, 20.0],
            "y_pred_mean": [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            "y_pred_std_raw": [2.0] * n,
            "y_pred_std": [2.0] * n,
            "lower_pi": [10.0] * n,
            "upper_pi": [20.0] * n,
            "lower_gaussian": [10.0] * n,
            "upper_gaussian": [20.0] * n,
            "solar_irradiance_poa_target": [500.0, 500.0, 500.0, 500.0, 5.0, 500.0],
            "anomaly_group": [
                "normal", "rare_or_extreme", "rare_or_extreme",
                "normal", "normal", "normal",
            ],
            "anomaly_label": [
                "", "unusually_high_solar_potential",
                "unusually_high_solar_potential", "unusually_low_solar_potential",
                "", "",
            ],
        }
    )


def test_counts_and_percentages_exact() -> None:
    df = _toy_predictions()
    row = interval_miss_row(df, "global", interval="pi")
    assert row["n"] == 6
    assert row["inside_interval_count"] == 3
    assert row["outside_interval_count"] == 3
    assert abs(row["inside_interval_pct"] - 0.5) < 1e-12
    assert abs(row["outside_interval_pct"] - 0.5) < 1e-12
    # 2 above (rows 1,2), 1 below (row 3)
    assert abs(row["above_interval_pct"] - 2 / 6) < 1e-12
    assert abs(row["below_interval_pct"] - 1 / 6) < 1e-12


def test_distances_in_watt_conditional_on_misses() -> None:
    df = _toy_predictions()
    row = interval_miss_row(df, "global", interval="pi")
    # outside distances over outside rows only: {5, 10, 5}
    assert abs(row["mean_outside_distance"] - 20.0 / 3.0) < 1e-12
    assert abs(row["median_outside_distance"] - 5.0) < 1e-12
    # above distances over above rows only: {5, 10}
    assert abs(row["mean_above_distance"] - 7.5) < 1e-12
    # below distances over below rows only: {5}
    assert abs(row["mean_below_distance"] - 5.0) < 1e-12
    assert abs(row["median_below_distance"] - 5.0) < 1e-12


def test_required_multiplier_quantiles_exact() -> None:
    df = _toy_predictions()
    row = interval_miss_row(df, "global", interval="pi", eps=0.0)
    # |y_true - 15| / 2 = {0, 5, 7.5, 5, 2.5, 2.5}
    required = np.array([0.0, 5.0, 7.5, 5.0, 2.5, 2.5])
    for q, key in ((50, "p50_required_multiplier"),
                   (90, "p90_required_multiplier"),
                   (95, "p95_required_multiplier")):
        assert abs(row[key] - np.percentile(required, q)) < 1e-9


def test_strata_masks() -> None:
    df = _toy_predictions()
    masks = build_strata_masks(df)
    assert masks["global"].sum() == 6
    assert masks["daytime"].sum() == 5          # row 4 has solar=5 -> nighttime
    assert masks["nighttime"].sum() == 1
    assert masks["rare_extreme"].sum() == 2
    assert masks["normal_daytime"].sum() == 3   # rows 0, 3, 5
    assert masks["rare_extreme_daytime"].sum() == 2
    assert masks["label:unusually_high_solar_potential"].sum() == 2
    assert masks["label:unusually_low_solar_potential"].sum() == 1
    # daytime production bins use y_true: daytime rows y_true={15,25,30,5,20};
    # bins are [lower, upper) -> 20 falls in daytime_20_40, not daytime_0_20.
    assert masks["daytime_0_20"].sum() == 2     # 15, 5
    assert masks["daytime_20_40"].sum() == 3    # 25, 30, 20
    assert masks["daytime_gt_100"].sum() == 0


def test_empty_stratum_no_crash() -> None:
    df = _toy_predictions()
    table = compute_interval_miss_table(df)
    gt100 = table[table["group"] == "daytime_gt_100"].iloc[0]
    assert gt100["n"] == 0
    assert gt100["inside_interval_count"] == 0
    assert gt100["outside_interval_count"] == 0
    assert np.isnan(gt100["mean_outside_distance"])
    assert np.isnan(gt100["p95_required_multiplier"])


def test_picp_curve_exact_and_monotone() -> None:
    df = _toy_predictions()
    curve = compute_picp_curve_table(df, multipliers=(1.0, 1.96, 2.5, 5.0, 10.0))
    g = curve[curve["group"] == "global"].iloc[0]
    # band = 15 +/- 2k; |y-15| = {0, 10, 15, 10, 5, 5}
    # k=1.0  -> half-width 2   -> covered: {0}                 -> 1/6
    # k=1.96 -> 3.92           -> covered: {0}                 -> 1/6
    # k=2.5  -> 5.0            -> covered: {0, 5, 5}           -> 3/6
    # k=5.0  -> 10.0           -> covered: {0, 10, 10, 5, 5}   -> 5/6
    # k=10.0 -> 20.0           -> covered: all                 -> 6/6
    assert abs(g["picp_k_1"] - 1 / 6) < 1e-12
    assert abs(g["picp_k_1.96"] - 1 / 6) < 1e-12
    assert abs(g["picp_k_2.5"] - 3 / 6) < 1e-12
    assert abs(g["picp_k_5"] - 5 / 6) < 1e-12
    assert abs(g["picp_k_10"] - 1.0) < 1e-12
    # Monotone non-decreasing in k for every stratum with n > 0.
    k_cols = [c for c in curve.columns if c.startswith("picp_k_")]
    for _, r in curve.iterrows():
        vals = [r[c] for c in k_cols if np.isfinite(r[c])]
        assert vals == sorted(vals), f"PICP not monotone for {r['group']}"


def test_report_renders() -> None:
    df = _toy_predictions()
    miss = compute_interval_miss_table(df)
    curve = compute_picp_curve_table(df)
    report = render_interval_miss_report(miss, curve, meta={"predictions": "toy.csv"})
    assert "Interval-miss distances per stratum" in report
    assert "PICP curve" in report
    assert "Widen-vs-bias verdict" in report
    assert "daytime_gt_100" in report


def test_cli_smoke() -> None:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from analyze_pvgis_interval_miss_distance import main as cli_main

    df = _toy_predictions()
    with tempfile.TemporaryDirectory() as tmp:
        pred_path = Path(tmp) / "predictions.csv"
        out_dir = Path(tmp) / "out"
        df.to_csv(pred_path, index=False)
        paths = cli_main([
            "--predictions", str(pred_path),
            "--out-dir", str(out_dir),
        ])
        for path in paths.values():
            assert path.exists(), f"missing output {path}"
        written = pd.read_csv(paths["interval_miss_distance"])
        assert (written[written["group"] == "global"]["inside_interval_count"] == 3).all()


if __name__ == "__main__":
    test_counts_and_percentages_exact()
    test_distances_in_watt_conditional_on_misses()
    test_required_multiplier_quantiles_exact()
    test_strata_masks()
    test_empty_stratum_no_crash()
    test_picp_curve_exact_and_monotone()
    test_report_renders()
    test_cli_smoke()
    print("PASS: PVGIS interval-miss diagnostics")
