# Input Features Reference

This document describes the 16 input features used by the STGNN model, how
each is computed from the raw dataset, and how they are assembled into the
tensor consumed by the model.

The features are defined in `physiq_pv/data/dataset.py` (`PVDataset` class) and
the QS components are computed in `physiq_pv/data/quality_score.py`
(`compute_qs`). `N_FEATURES = 16`, `SEQ_LEN = 24` (one calendar day of hourly
samples).

---

## Feature index

| # | Name | Group | Source | Normalisation | Range |
|---|------|-------|--------|---------------|-------|
| 0 | `temp` | Weather | `temperature_2m` (xarray) | z-score per dataset | ℝ |
| 1 | `solar_poa` | Weather | `solar_irradiance_poa` (W/m²) | z-score per dataset | ℝ |
| 2 | `wind` | Weather | `wind_speed_10m` (m/s) | z-score per dataset | ℝ |
| 3 | `sin_elev` | Geometry | `pvlib` solar elevation | none (already bounded) | [0, 1] |
| 4 | `cos_elev` | Geometry | `pvlib` solar elevation | none | [0, 1] |
| 5 | `m1` | QS | `compute_qs` Pearson rolling | NaN → 0.0 | {0} ∪ (0, 1] |
| 6 | `m2` | QS | `compute_qs` bias score | NaN → 0.0 | {0} ∪ (0, 1] |
| 7 | `m3` | QS | `compute_qs` completeness | NaN → 0.0 | {0} ∪ (0, 1] |
| 8 | `m4` | QS | `compute_qs` variance ratio | NaN → 0.0 | {0} ∪ (0, 1] |
| 9 | `m5` | QS | `compute_qs` η(T) consistency | NaN → 0.0 | {0} ∪ (0, 1] |
| 10 | `pv_lag` | Autoregressive | past `ENERGIA / p99_daytime` | per-plant scaling | [0, ~1.5] |
| 11 | `kt` | Cloud dynamics | `solar_poa / ghi_cs` | clamp | [0, 1.5] |
| 12 | `kt_std_3h` | Cloud dynamics | rolling 3h std of `kt` | none | ≥ 0 |
| 13 | `dghi_dt` | Cloud dynamics | first difference of `solar_poa` (kW/m²) | z-score per dataset | ℝ |
| 14 | `dni_norm` | Erbs split | `pvlib.irradiance.erbs` DNI | clamp, /1000 | [0, 1.5] kW/m² |
| 15 | `dhi_norm` | Erbs split | `pvlib.irradiance.erbs` DHI | clamp, /1000 | [0, 1.0] kW/m² |

---

## Per-feature definitions

### 0–2: Weather (NWP-style inputs)

| Feature | Variable in xarray | Units (raw) | Transform |
|---------|--------------------|-------------|-----------|
| `temp` | `temperature_2m` | °C | z-score: `(x - mean) / (std + 1e-6)` with NaN → mean |
| `solar_poa` | `solar_irradiance_poa` | W/m² | z-score (same as above) |
| `wind` | `wind_speed_10m` | m/s | z-score (same as above) |

z-score statistics are computed once across the **full** training period and
applied to every sample (no per-batch re-normalisation). Code:
`PVDataset._norm`.

### 3–4: Solar geometry

For each plant `p` with `(lat[p], lon[p])`, hourly times are fed to
`pvlib.location.Location(...).get_solarposition(times_utc)`, yielding apparent
elevation in degrees `elev_deg ∈ [0°, 90°]` (clamped). Then:

```
sin_elev = sin(radians(elev_deg))   # → [0, 1]
cos_elev = cos(radians(elev_deg))   # → [0, 1]
```

These are **deterministic** functions of time + plant coordinates; no
per-plant statistics are involved. Plants with NaN coordinates fall back to
the fleet-mean lat/lon. They are not z-scored because they are already
bounded.

### 5–9: Quality Score components m1–m5

QS is a per-plant, per-hour data-quality scalar in [0, 1] computed in
`compute_qs` over a rolling window of 720 hours (30 days). The five
components are emitted from `compute_qs(ds, debug=True)` and each
captures a different failure mode. Each is clipped to [0, 1] (1 = good).

| ID | Name | Formula (per window) | Penalises |
|----|------|----------------------|-----------|
| m1 | `corr_score` | Pearson(real, irradiance_ref) rolling | decoupled sensors |
| m2 | `bias_score` | `1 - |mean(real - ref)| / mean(ref)` | persistent offset |
| m3 | `nan_score` | `1 - nan_fraction(real)` | missingness |
| m4 | `var_score` | `min(1, std(real) / std(ref))` | stuck sensors |
| m5 | `eta_score` | `1 - mean( max(0, 1 - (real/ref) / η_T) )` | physical impossibility |

Where:
- `real = ENERGIA` (production in kWh/h)
- `irradiance_ref = solar_irradiance_poa / 1000` scaled per plant by
  `p99(real_day) / p99(ref_day)` so it matches the plant capacity
- `η_T = η_base * (1 - 0.004 * (T - 25))` is the IEC-61215 temperature-corrected
  performance ratio; `η_base` is the data-driven median ratio after capacity
  scaling, clamped to `[0.1, 2.0]`
- Window length = 720 hours, `min_periods = max(window/4, 10)`
- Nighttime samples (`irradiance_ref < 0.05 kW/m²`) are filtered to NaN inside
  the rolling computation. Special markers (0.0 or 1.0) are written for
  outside-daytime hours, depending on whether production is genuinely zero
  (correct sensor) or non-zero in darkness (faulty).

After computing `(m1·m2·m3·m4·m5)^(1/5)` per cell, the aggregate scalar `QS`
is also produced, but **it is not used as a model input**. Only the five
components `m1..m5` enter the feature tensor. The aggregate `QS` is used for
diagnostics and for the optional outlier-filter ablation in `main.py`. In the
default training path, neither `QS` nor any threshold is applied at training
time.

In `PVDataset.__init__` the raw `m_components` arrays come in as `(N, T)` and
are transposed to `(T, N)` with `np.nan_to_num(..., nan=0.0)` before stacking.
This means NaN values produced by `compute_qs` (nighttime samples and the
first ~`window/4` ≈ 180 hours of rolling cold-start) are **replaced by 0.0**
in the feature tensor. A literal `m_k = 0` in the input therefore encodes
either "this hour is night / outside daytime" or "we do not yet have a
720-hour history at this timestamp", not a degraded sensor. The model has to
learn to distinguish those two cases via the other features (geometry,
`ghi_cs`, `pv_lag`).

### 10: Autoregressive PV lag

`pv_lag` is the **same** normalised PV series that the model is trained to
predict, fed back as an input feature. It is therefore an **autoregressive**
input channel (in the AR/ARX sense), not an exogenous covariate.

```
pv_scale[p]      = percentile_99( ENERGIA[day_mask, p] > 0 )   # per plant
target_pv_norm   = clip( ENERGIA / pv_scale, 0.0, 1.5 )        # (T, N)
pv_lag           = target_pv_norm                              # alias of same array
```

The day mask used to estimate `pv_scale[p]` requires both `sin_elev > 0.05`
**and** `solar_irradiance_poa > 30 W/m²`, so that the percentile reflects
real production hours only.

**Anti-leakage by slicing.** At sample index `idx → t`, `__getitem__`
returns `feats[t - seq_len : t]`. The right-open slice excludes index `t`,
so the `pv_lag` channel only exposes `target_pv_norm[t - seq_len .. t - 1]`
— strictly past values. The target at time `t` itself is never visible to
the encoder. No masking is needed inside the model.

This is the only feature whose value at time `t'` equals (a transformation
of) the supervised target at time `t'`. All other features are exogenous to
the PV target.

### 11–13: Cloud dynamics

These three features were added to expose the model to cloud-induced
intermittency that cannot be inferred from POA alone.

| Feature | Computation | Rationale |
|---------|-------------|-----------|
| `kt`        | `kt = clip(solar_poa_kwm2 / (ghi_cs + 1e-6), 0, 1.5)` where `ghi_cs ≤ 0.1 kW/m²` is forced to 0 | clearness index in [0, 1.5]; values > 1 from cloud-edge enhancement |
| `kt_std_3h` | `pandas.Series(kt[:, p]).rolling(3, min_periods=1).std()` per plant, fill NaN with 0 | short-term variability proxy |
| `dghi_dt`   | first finite difference of `solar_irradiance_poa / 1000` over time (pad t=0 with 0), then z-score | ramp rate, transient regime signal |

`ghi_cs` is the Ineichen clear-sky GHI from `pvlib.location.Location.get_clearsky`
(falls back to `simplified_solis` if Ineichen fails). It is stored separately
on `dataset.ghi_cs` because it is also passed to the model as a multiplier
for the GHI head (see "Output decoding" below).

### 14–15: Erbs DNI/DHI split

Erbs (1982) decomposition splits GHI into the beam (DNI) and diffuse (DHI)
components using zenith angle and day-of-year. Per plant:

```python
erbs_out = pvlib.irradiance.erbs(
    ghi=solar_irradiance_poa_wm2[:, p],      # W/m²
    zenith=apparent_zenith_deg[:, p],
    datetime_or_doy=times.dayofyear,
)
dni_norm[:, p] = clip( erbs_out["dni"] / 1000, 0.0, 1.5 )   # kW/m²
dhi_norm[:, p] = clip( erbs_out["dhi"] / 1000, 0.0, 1.0 )   # kW/m²
```

The zenith comes from the same `pvlib.location.Location.get_solarposition`
call used for `sin_elev` / `cos_elev`. NaN values produced by the Erbs
`min_cos_zenith` / `max_zenith` clamps (which kick in at night and at the
horizon) are filled with 0. No further normalisation is applied because the
values are already in physical units bounded near unity.

These two features improve the cloud-state observability: `dni` collapses to
zero under overcast, `dhi` rises; in clear-sky `dni` dominates. The model
benefits in twilight and partly-cloudy regimes.

---

## How features are fed to the model

After per-feature transforms, the dataset constructs a single tensor:

```python
feature_arrays = [
    _norm(temp), _norm(solar), _norm(wind),       # weather, z-scored
    sin_elev, cos_elev,                           # geometry, bounded
    m1, m2, m3, m4, m5,                           # QS, bounded
    pv_lag,
    kt, kt_std_3h, _norm(dghi),                   # cloud dynamics
    dni_norm, dhi_norm,                           # Erbs split
]
self.feats = np.stack(feature_arrays, axis=-1).astype(np.float32)   # (T, N, 16)
```

A sample at index `idx → t` returns:

```
x       shape (N, seq_len=24, n_features=16)
y_pv    shape (N,)  — target PV at time t, normalised by p99 per plant
y_ghi   shape (N,)  — target GHI at time t, kW/m²
eta     shape (N,)  — per-plant efficiency proxy (weighted least squares fit)
ghi_cs  shape (N,)  — Ineichen clear-sky GHI at time t, kW/m²
```

with `x[:, t', :]` corresponding to `feats[t - seq_len + t', :, :]`
(transposed so the plant dimension comes first).

### Batching and graph context

The DataLoader stacks samples into:

```
x          (B, N, seq_len, n_features)        — input
edge_index (2, E)                              — geographic kNN, k = 10 km
edge_weight (E,)                               — 1 / dist_km
ghi_cs     (B, N)                              — clear-sky multiplier for head
```

`edge_index` and `edge_weight` are constructed once by
`physiq_pv.model.graph_builder.build_graph(lats, lons, max_dist_km=10.0)` and
do **not** depend on time — the geographic topology is shared across the
window.

### Inside the STGNN

```
x: (B, N, L, C)
  └── reshape → (B*N, L, C)
       └── PatchTSTEncoder (channel-independent)
             ├── unfold to patches of length patch_len=4, stride=2
             ├── linear patch_embed + sinusoidal positional encoding
             ├── 2-layer pre-norm Transformer (d_model=64, 4 heads)
             └── mean-pool over patches → (B*N, C*d_model)
       └── proj: Linear → GELU → LayerNorm → (B*N, gat_dim=96)
       └── reshape → (B, N, 96)
       └── GATLayer × 1   (8-head GAT over edge_index, attention scaled by log(1+edge_weight))
       └── head_ghi: 96 → 48 → 1  (sigmoid · KT_MAX = 1.2)
       └── head_pv : 96 → 48 → 1  (softplus)
```

### Output decoding

The GHI head outputs a clear-sky index `pred_kt = sigmoid(h) * 1.2`. The
final GHI prediction is:

```
pred_ghi = pred_kt * ghi_cs        # physical residual constraint
```

This forces zero GHI at night (`ghi_cs ≈ 0`) and caps any prediction at
`1.2 × clear-sky`, the snow-albedo physical bound. The PV head uses
`softplus` so production is non-negative but unbounded above.

---

## Provenance and notebook

For training-time noise, only weather channels (indices 0–2) are perturbed:

```python
noise = 1.0 + 0.05 * randn(B, N, L, 3)        # ±5% multiplicative
x = cat([x[..., :3] * noise, x[..., 3:]], dim=-1)
```

Geometry (3–4) and m-components (5–9) are deterministic by construction and
must not be perturbed; the cloud dynamics, Erbs split, and `pv_lag` are also
left untouched in the augmentation step (`train.py::_train_epoch`).

| Group | Origin | Stochastic? |
|-------|--------|-------------|
| Weather (0–2) | NWP or sentinel data | ±5% augmentation in train |
| Geometry (3–4) | `pvlib` solar position | deterministic |
| QS (5–9) | `compute_qs`, 720h rolling | deterministic |
| `pv_lag` (10) | past `ENERGIA` | deterministic |
| Cloud dynamics (11–13) | `solar_poa` + `ghi_cs` | deterministic |
| Erbs split (14–15) | `pvlib.irradiance.erbs` | deterministic |

---

## Quick map: dataset.py → feature index

```
feature_arrays = [
    _norm(temp),         # 0
    _norm(solar),        # 1
    _norm(wind),         # 2
    sin_elev,            # 3
    cos_elev,            # 4
    m1, m2, m3, m4, m5,  # 5..9
    pv_lag,              # 10
    kt,                  # 11
    kt_std_3h,           # 12
    _norm(dghi),         # 13
    dni_norm,            # 14
    dhi_norm,            # 15
]
```

To add a new feature: append the array in this list, update `N_FEATURES` in
`physiq_pv/data/dataset.py`, and update the `features` list logged to wandb
in `train.py`.
