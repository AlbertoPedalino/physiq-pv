"""PVGIS sensor and segment adapters for CATCH."""

from physiq_pv.data.pvgis_anomaly import (
    DEFAULT_ANOMALY_SENSORS,
    available_locations,
    extract_location_segments,
    prepare_pvgis_years,
)


DEFAULT_CATCH_SENSORS = list(DEFAULT_ANOMALY_SENSORS)


__all__ = [
    "DEFAULT_CATCH_SENSORS",
    "available_locations",
    "extract_location_segments",
    "prepare_pvgis_years",
]
