# PhysiQ-PV

Forecasting fotovoltaico distribuito su flotta reale con BiLSTM, Graph
Attention Network, vincoli fisici e metriche di qualità.

La configurazione corrente usa dati orari Sentinel/SCADA e PVGIS. Nel file
PVGIS 2019 originale `solar_irradiance_poa` è nullo: durante il merge viene
quindi ricostruito come:

```text
POA = direct_irradiance_tilted + diffuse_irradiance_tilted
```

Non viene usato GHI come sostituto della radiazione sul piano inclinato.

## Varianti

- `fix/solar-poa-from-tilted`: usa i canali POA ricostruiti come input.
- `feat/solar-poa-ablation`: stessa architettura e stessi target, ma azzera
  esclusivamente i canali di input che dipendono da POA.

Questa separazione rende confrontabile una futura ablazione: il numero di
feature e i parametri del modello non cambiano tra i due branch.

## Modello

La finestra causale contiene le 24 ore precedenti. Per ogni impianto:

1. una BiLSTM bidirezionale a due layer codifica la finestra passata;
2. una GAT propaga informazione tra impianti vicini;
3. due head predicono POA e potenza PV normalizzata.

La bidirezionalità non legge il futuro: opera soltanto dentro la finestra
`[t-24, t)`. La head POA usa il clear-sky POA sullo stesso piano inclinato:

```text
pred_poa = sigmoid(head_poa) * 1.6 * poa_clear_sky
pred_pv  = softplus(head_pv)
```

Il vincolo fisico confronta grandezze con scala coerente:

```text
pred_pv ~= PR * (pred_poa / poa_scale)
```

Tutte le statistiche di preprocessing, le scale e il PR proxy sono stimati
solo sulla porzione di training. Lo split è cronologico 80/20 e il checkpoint
è selezionato tramite `rmse_pv_day`, confrontata anche con la persistence.

## Feature

L’input ha sempre 16 canali:

| # | Nome | Dipende da POA |
|---:|---|:---:|
| 0 | `temp_z` | no |
| 1 | `wind_z` | no |
| 2 | `sin_elev` | no |
| 3 | `cos_elev` | no |
| 4 | `pv_lag` | no |
| 5 | `poa_z` | sì |
| 6 | `diffuse_fraction` | sì |
| 7 | `kt_poa` | sì |
| 8 | `kt_poa_std_3h` | sì |
| 9 | `dpoa_z` | sì |
| 10 | `m1` | sì |
| 11 | `m2` | sì |
| 12 | `m3` | no |
| 13 | `m4` | sì |
| 14 | `m5` | sì |
| 15 | `quality_valid` | sì |

`m3` misura la completezza del segnale PV e resta disponibile anche senza
POA. Le altre metriche di qualità confrontano direttamente o indirettamente
PV e POA, quindi sono mascherate nell’ablazione.

## Avvio e test

```powershell
python main.py
python -m unittest discover -s tests -v
```

I checkpoint includono `model.pt`, configurazione, cronologia delle loss e
`preprocessing_state.json`, necessario per riprodurre lo stesso preprocessing
in inferenza.

La specifica canonica, incluse le istruzioni per portare le correzioni sugli
altri branch, è in
[`docs/BILSTM_GAT_STATE.md`](docs/BILSTM_GAT_STATE.md).
