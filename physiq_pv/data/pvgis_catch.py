"""PVGIS sensor and segment adapters for CATCH."""

from physiq_pv.data.pvgis_anomaly import (
    available_locations,
    extract_location_segments,
    prepare_pvgis_years,
)


# PVGIS power is a theoretical target derived from the same irradiance and
# weather inputs. Keep detector labels independent from the SDE forecast target.
DEFAULT_CATCH_SENSORS = [
    "solar_irradiance_poa",
    "temperature_2m",
    "wind_speed_10m",
]


__all__ = [
    "DEFAULT_CATCH_SENSORS",
    "available_locations",
    "extract_location_segments",
    "prepare_pvgis_years",
]
