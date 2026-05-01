# PhysiQ-PV - Training Architecture

## Entry Point

Training end-to-end:

```text
main.py -> train.py -> PVDataset -> STGNN
```

Il modello corrente e' una ST-GNN semplificata:
- encoder temporale PatchTST
- GAT geografico
- due teste: `pred_ghi` e `pred_pv`
- vincolo fisico sul rapporto PV/irradianza

## Input

Feature per impianto e timestep:

| Canale | Feature |
|---|---|
| 0 | `temperature_2m` normalizzata |
| 1 | `solar_irradiance_poa` normalizzata |
| 2 | `wind_speed_10m` normalizzata |
| 3 | `sin_solar_elev` |
| 4 | `cos_solar_elev` |
| 5 | `QS` |

Shape sample: `(N, 24, 6)`.

`pvgis_ref` non e' una feature.

## Target

```text
y_ghi = solar_irradiance_poa / 1000.0
y_pv  = clip(ENERGIA / pv_scale, 0.0, 1.5)
```

`pv_scale` e' calcolato come p99 della produzione osservata nelle ore diurne.

Le ore diurne sono definite con geometria solare e soglia di irradianza, non con PVGIS reference power.

## Eta Adjusted

`eta_adjusted` e' stimato da produzione normalizzata e irradianza normalizzata:

```text
solar_norm = (solar_irradiance_poa / 1000.0) / solar_p99
pv_norm    = ENERGIA / pv_scale
eta_adjusted = median(pv_norm / solar_norm)
```

Clip: `[0.1, 1.0]`.

Uso: target per `L_physics`.

## Quality Score

`compute_qs()` calcola un Quality Score per `(plant, time)` usando il riferimento irradiance-based.

```text
QS = (m1 * m2 * m3 * m4 * m5) ** 0.2
```

`QS` entra:
- come sesta feature
- come peso loss: `weight = QS^0.2`

## Modello

Configurazione corrente in `train.py`:

| Parametro | Valore |
|---|---|
| `seq_len` | 24 |
| `patch_len` | 4 |
| `stride` | 2 |
| `d_model` | 64 |
| `gat_dim` | 96 |
| `gat_heads` | 4 |
| `gat_layers` | 1 |
| `dropout` | 0.0 |

Forward:

```text
(B, N, 24, 6)
  -> PatchTSTEncoder
  -> projection
  -> GAT over geographic graph
  -> softplus heads
  -> pred_ghi, pred_pv
```

Le due teste usano `softplus`, quindi gli output raw del modello sono non negativi.

## Loss

`physics_loss_full()`:

```text
L_base = L_ghi + L_pv + lam * L_physics

L_ghi     = mean(weight * (pred_ghi - true_ghi)^2)
L_pv      = mean(weight * (pred_pv - true_pv)^2)
L_physics = mean(weight * (pred_pv / abs(pred_ghi) - eta_adjusted)^2)
weight    = QS^0.2
```

Training aggiunge una loss asimmetrica sui picchi PV:

```text
w_peak = 1 + peak_alpha * true_pv^peak_gamma
err = pred_pv - true_pv
asym = 2.0 * abs(err) se err < 0, altrimenti abs(err)
L_peak = mean(w_peak * asym)

L = L_base + peak_loss_weight * L_peak
```

Obiettivo: ridurre la sottostima dei picchi senza cambiare architettura.

## Split e Checkpoint

Split:
- stratificato per mese
- 80% train, 20% validation per ogni mese presente

Checkpoint:
- `best_state` viene aggiornato solo quando migliora la validation loss
- a fine training il modello ricarica `best_state`
- `main.py` salva quindi `checkpoints/model.pt` dal best validation epoch
- `loss_history.json` include `best_epoch`

## Calibrazione PV

Dopo il training viene stimata una calibrazione lineare su validation daytime:

```text
pred_cal = slope * pred_pv + intercept
pred_cal = clip(pred_cal, 0.0, None)
```

La calibrazione viene abilitata solo se migliora il KPI scelto:

| `calibration_kpi` | Criterio |
|---|---|
| `rmse` | RMSE dopo < RMSE prima |
| `mae` | MAE dopo < MAE prima |
| `both` | entrambi migliorano |
| `none` | disabilitata |

Il floor a zero e' parte del post-processing operativo, quindi anche i KPI `mae_after` e `rmse_after` sono calcolati dopo il floor.

## Output

```text
checkpoints/
  model.pt
  loss_history.json
  model_config.json
  pv_calibration.json
```

`pv_calibration.json` contiene anche `best_val_epoch`, KPI prima/dopo e conteggio delle predizioni che sarebbero negative prima del floor.
