"""One shared feature contract for the raw downloader and STGAN ERA5 adapter."""

SINGLE = {
    "2m_temperature": ("t2m", "2t"),
    "10m_u_component_of_wind": ("u10", "10u"),
    "10m_v_component_of_wind": ("v10", "10v"),
    "mean_sea_level_pressure": ("msl",),
    "total_precipitation": ("tp",),
    "total_cloud_cover": ("tcc",),
    "surface_solar_radiation_downwards": ("ssrd",),
    "volumetric_soil_water_layer_1": ("swvl1",),
    "friction_velocity": ("zust",),
    "boundary_layer_height": ("blh",),
}
PRESSURE = {
    "u_component_of_wind": ("u",),
    "v_component_of_wind": ("v",),
    "vertical_velocity": ("w",),
    "temperature": ("t",),
    "specific_humidity": ("q",),
}
FEATURE_NAMES = tuple(SINGLE) + tuple(PRESSURE)
