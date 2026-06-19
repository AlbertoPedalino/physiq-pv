#!/usr/bin/env python
"""Thin wrapper around physiq_pv.reporting.daytime_bin_anomaly_report.

Kept so the documented command
    python scripts/analyze_pvgis_daytime_report.py --predictions ... --out-dir ...
still works after the post-hoc report logic moved into the package
(physiq_pv/reporting/daytime_bin_anomaly_report.py). All CLI flags and outputs
(PICP, MAE, RMSE, mean_std, MPIW, NMPIL, sharpness_overview.csv) are unchanged.
"""
import sys
from pathlib import Path

# Allow `python scripts/analyze_pvgis_daytime_report.py` (sys.path[0] is the
# scripts/ dir) to import the physiq_pv package from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physiq_pv.reporting.daytime_bin_anomaly_report import main  # noqa: E402

if __name__ == "__main__":
    main()
