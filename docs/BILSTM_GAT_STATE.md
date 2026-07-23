# Stato canonico BiLSTM + GAT — POA v2

Questo documento è la specifica da usare quando il BiLSTM+GAT viene corretto
o portato su altri branch. Descrive il contratto attuale, non i vecchi
checkpoint basati su GHI o sul campo PVGIS POA nullo.

## 1. Varianti mantenute

| Branch | `include_poa_inputs` | Significato |
|---|:---:|---|
| `fix/solar-poa-from-tilted` | `True` | input POA ricostruito |
| `feat/solar-poa-ablation` | `False` | stessi 16 canali, quelli POA-dipendenti a zero |

Entrambe le varianti mantengono POA come target ausiliario. La differenza deve
essere soltanto l’accesso del codificatore alle informazioni POA, non la forma
della rete, lo split o la loss.

## 2. Contratto dati

Variabili richieste dopo il merge:

```text
ENERGIA
temperature_2m
wind_speed_10m
direct_irradiance_tilted
diffuse_irradiance_tilted
solar_irradiance_poa
```

Il campo `solar_irradiance_poa` del NetCDF consegnato è interamente nullo e non
deve essere usato. Il loader richiede entrambe le componenti inclinate e
ricostruisce:

```text
solar_irradiance_poa =
    direct_irradiance_tilted + diffuse_irradiance_tilted
```

Unità PVGIS: W/m². Il dataset converte POA in kW/m².

La geometria PVGIS corrente è letta dagli attributi del file:

```text
surface_tilt = 30°
surface_azimuth = 180°
```

Il clear-sky reference viene calcolato con pvlib sullo stesso piano, usando
Ineichen e `get_total_irradiance`. Non usare GHI clear-sky per vincolare POA.

I timestamp SCADA senza timezone sono interpretati come `Europe/Rome`,
convertiti in UTC e memorizzati UTC-naive. I timestamp PVGIS sono trattati
come UTC. Il merge usa nearest-neighbour temporale con tolleranza 31 minuti e
fallisce esplicitamente se restano valori meteo non allineati.

## 3. Feature schema

L’ordine è un contratto persistente:

```text
 0 temp_z
 1 wind_z
 2 sin_elev
 3 cos_elev
 4 pv_lag
 5 poa_z
 6 diffuse_fraction
 7 kt_poa
 8 kt_poa_std_3h
 9 dpoa_z
10 m1
11 m2
12 m3
13 m4
14 m5
15 quality_valid
```

Formule principali:

```text
diffuse_fraction = diffuse_tilted / POA
kt_poa            = POA / POA_clear_sky
kt_poa_std_3h     = rolling_std(kt_poa, 3)
dpoa              = POA[t] - POA[t-1]
```

`kt_poa` di input è limitato a 2.5 per robustezza. La head usa invece un
limite distinto, 1.6, calibrato sul rapporto osservato POA/POA-clear-sky.

Gli indici mascherati quando `include_poa_inputs=False` sono:

```text
poa_z, diffuse_fraction, kt_poa, kt_poa_std_3h, dpoa_z,
m1, m2, m4, m5, quality_valid
```

`m3` resta attivo perché misura soltanto la completezza di `ENERGIA`.
`temp_z`, `wind_z`, geometria solare e `pv_lag` restano attivi.

La radiazione diretta non è duplicata come canale separato: POA e frazione
diffusa determinano implicitamente direct e diffuse, evitando forte
collinearità tra `POA`, `direct`, `diffuse` e `direct+diffuse`.

## 4. Split e preprocessing

Lo split è strettamente cronologico:

```text
train = primo 80% dei target validi
validation = ultimo 20% dei target validi
```

Non viene effettuato shuffle prima dello split. I DataLoader possono
mescolare soltanto le finestre già assegnate al training.

Usano esclusivamente `fit_time_mask=train`:

- media e deviazione standard di temperatura, vento, POA e delta POA;
- percentile 99 per `pv_scale` e `poa_scale`;
- capacity scale ed `eta_base` delle metriche di qualità;
- regressione del `pr_proxy`.

Le rolling metriche rimangono causali sulla timeline completa. Il loro fit
globale, quando necessario, non vede la validation.

Lo stato risultante viene salvato in `preprocessing_state.json`:

```text
feature_names
include_poa_inputs
surface_tilt / surface_azimuth
zscore
pv_scale / poa_scale
pr_proxy
fit_start / fit_end
```

Questo file deve accompagnare il checkpoint in inferenza.

## 5. Target e batch

Target:

```text
y_poa = POA in kW/m²
y_pv  = clip(ENERGIA / pv_scale, 0, 1.5)
```

Un elemento del dataset restituisce:

```text
x, y_poa, y_pv, pr_proxy, poa_cs, poa_scale
```

La forma di `x` è `[n_impianti, seq_len, 16]`; dopo batching è
`[batch, n_impianti, seq_len, 16]`.

## 6. Architettura operativa

Configurazione usata da `main.py`:

| Componente | Configurazione |
|---|---|
| contesto | 24 ore precedenti |
| BiLSTM | 2 layer, hidden 128, bidirezionale |
| pooling | attention di default, `last` supportato |
| input projection BiLSTM | nessuna |
| proiezione spaziale | 256 → 96, GELU, LayerNorm |
| GAT | 1 layer, 4 head, dimensione 96 |
| dropout | 0.2 |
| grafo | soglia 20 km, nearest-neighbour per isolati |
| prior distanza | gaussiano, scala predefinita 10 km |
| forza prior | 1.0, aggiunta ai logit in log-spazio |

La BiLSTM non viola la causalità: legge in entrambe le direzioni soltanto
all’interno della finestra storica che termina a `t-1`.

Head:

```text
pred_kt_poa = sigmoid(head_poa(h)) * 1.6
pred_poa    = pred_kt_poa * poa_clear_sky
pred_pv     = softplus(head_pv(h))
```

Moltiplicare per POA clear-sky forza la radiazione predetta a zero durante la
notte senza dipendere dall’input POA osservato.

## 7. Loss

Il termine fisico usa POA normalizzato con la stessa scala impiegata per
stimare il PR:

```text
poa_norm = pred_poa / poa_scale
L_poa    = weighted_MSE(pred_poa, y_poa)
L_pv     = weighted_MSE(pred_pv, y_pv)
L_phys   = weighted_MSE(pred_pv, pr_proxy * poa_norm)

L_base = L_poa + L_pv + lambda * L_phys
L      = L_base + peak_loss_weight * L_peak_asymmetric
```

Default:

```text
lambda = 0.1
night_loss_weight = 0.2
pr_max = 1.5
```

La configurazione di `main.py` usa:

```text
peak_alpha = 2.5
peak_gamma = 2.0
peak_loss_weight = 0.25
under_penalty = 3.0
```

Le ore notturne non sono eliminate, ma pesano 0.2 rispetto al giorno per non
dominare la loss.

## 8. Validazione e scelta checkpoint

Metriche obbligatorie:

- PV globale: MAE, RMSE, bias;
- PV giorno e notte separati;
- POA nelle ore diurne;
- persistence PV diurna, usando `pv_lag[t-1]`;
- metriche PV per bin di potenza.

Il checkpoint è scelto su `rmse_pv_day`, non sulla loss composita. La loss
resta riportata come diagnostica. W&B salva le metriche dell’epoca scelta con
prefisso `best_`.

## 9. Quality Score

Dipendenza da POA:

| Metrica | Contenuto | POA-dipendente |
|---|---|:---:|
| `m1` | correlazione rolling PV/POA | sì |
| `m2` | bias rolling PV/POA | sì |
| `m3` | completezza del PV | no |
| `m4` | rapporto di variabilità PV/POA | sì |
| `m5` | coerenza fisica PR/temperatura | sì |

Questo spiega perché il vecchio modello poteva funzionare con POA nullo:
maschera giorno/notte e geometria solare provenivano anche da pvlib, mentre
`pv_lag`, temperatura, vento e `m3` conservavano segnale. Tuttavia le metriche
POA-dipendenti erano degeneri o fuorvianti e non costituivano una correzione
valida del campo nullo.

## 10. Breaking changes

Quando si porta questa versione su un altro branch:

- rinominare semanticamente `GHI` in `POA`;
- `head_ghi` diventa `head_poa`;
- `ghi_cs` diventa `poa_cs`;
- il batch passa da 5 a 6 elementi e include `poa_scale`;
- la proxy `eta` viene sostituita da `pr_proxy` coerente con le scale;
- il grafo usa un prior gaussiano e non `1/distance`;
- lo split mensile o random deve essere sostituito da quello cronologico;
- ogni normalizzazione deve ricevere il train mask;
- l’ablazione non deve cambiare `N_FEATURES` né l’architettura.

I vecchi checkpoint non sono compatibili con `head_poa` e non devono essere
caricati silenziosamente.

## 11. Checklist di porting

1. Portare insieme `dataset.py`, `quality_score.py` e `physics_loss.py`.
2. Portare `graph_builder.py` e `st_gnn.py`.
3. Adeguare ogni training loop al batch a 6 elementi.
4. Creare lo split prima di `compute_qs` e `PVDataset`.
5. Passare lo stesso `fit_time_mask` a entrambi.
6. Salvare `preprocessing_state.json`.
7. Usare `rmse_pv_day` per early stopping/checkpoint.
8. Decidere soltanto `include_poa_inputs=True/False`.
9. Non cambiare forma o iperparametri tra i due lati dell’ablazione.
10. Eseguire i test e uno smoke training prima di pubblicare.

## 12. Verifica

Comandi minimi:

```powershell
python -m compileall -q physiq_pv train.py main.py scripts tests
python -m unittest discover -s tests -v
```

I test coprono:

- ricostruzione POA da direct+diffuse;
- errore se manca una componente;
- schema identico nell’ablazione e maschera selettiva;
- preprocessing invariato da modifiche nella validation;
- POA clear-sky nullo di notte;
- coerenza dimensionale della loss fisica;
- split cronologico;
- assenza di nodi isolati e prior geografico limitato.

## 13. Limiti noti

- Tilt e azimuth PVGIS sono assunti comuni a tutti gli impianti.
- Il fuso `Europe/Rome` è un’ipotesi esplicita sul formato SCADA.
- La validation finale non sostituisce un test set temporale indipendente.
- Il limite `kt_poa=1.6` è calibrato sul file 2019 e va rivalutato su altri
  anni o orientamenti.
- POA continua a essere necessario come target e per le metriche diurne anche
  nel branch senza input POA.
