"""Paper-faithful import path for PVGIS climatology anomaly scoring.

The SDE paper-faithful scripts import this module name. The implementation in
this repo lives in pvgis_climatology_anomaly, so this file keeps the script API
aligned without duplicating the anomaly logic.
"""

from physiq_pv.data.pvgis_climatology_anomaly import *  # noqa: F401,F403
