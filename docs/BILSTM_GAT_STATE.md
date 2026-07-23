# Stato canonico BiLSTM + GAT — POA v2

Questo documento è la specifica da usare quando il BiLSTM+GAT viene corretto
o portato su altri branch. Descrive il contratto attuale, non i vecchi
checkpoint basati su GHI o sul campo PVGIS POA nullo.

## 1. Varianti mantenute

| Branch | `include_poa_inputs` | Significato |
|---|:---:|---|
| `fix/solar-poa-from-tilted` | `True` | input POA ricostruito |
| `feat/solar-poa-ablation` | `False` | stessi 17 canali, quelli POA-dipendenti a zero |

Entrambe le varianti mantengono POA come target ausiliario. La differenza deve
essere soltanto l’accesso del codificatore alle informazioni POA, non la forma
della rete, lo split o la loss.

`main.py` espone `INCLUDE_POA_INPUTS`, `FEATURE_SET`, `SEQ_LEN_MODEL` e
`CHECKPOINT_DIR_BASE`; il notebook di training importa queste costanti per non
creare una seconda configurazione divergente.

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

Il normalizzatore fallisce sui campi mancanti: non crea vento o irradianza
sintetici per un training reale.
Ogni impianto usato deve avere coordinate finite. Il loader conta le UPN
uniche valide nei metadati, interseca tale insieme con i CSV Sentinel
disponibili ed esclude esplicitamente le UPN prive di coordinate, riportandone
numero e un'anteprima dei codici. Il dataset conserva inoltre la coordinata
`upn` e i conteggi di copertura negli attributi. Il match al punto PVGIS più
vicino usa distanza great-circle (Haversine), non distanza euclidea in gradi.

Il campo `solar_irradiance_poa` del NetCDF consegnato è interamente nullo e non
deve essere usato. Il loader richiede entrambe le componenti inclinate e
ricostruisce:

```text
solar_irradiance_poa =
    direct_irradiance_tilted + diffuse_irradiance_tilted
```

Unità PVGIS: W/m². Il dataset converte POA in kW/m².
Il file non contiene una componente ground-reflected separata. Per confrontare
componenti omogenee, anche la clear-sky POA viene calcolata con `albedo=0`;
non si aggiunge artificialmente riflessione soltanto al denominatore di
`kt_poa`.

La geometria PVGIS corrente è letta dagli attributi del file:

```text
surface_tilt = 30°
surface_azimuth = 180°
```

L’azimuth usa la convenzione pvlib (`180° = sud`).
Il clear-sky reference viene calcolato con pvlib sullo stesso piano, usando
Ineichen e `get_total_irradiance`. Non usare GHI clear-sky per vincolare POA.

I timestamp SCADA senza timezone sono interpretati come `Europe/Rome`,
convertiti in UTC e memorizzati UTC-naive. I timestamp PVGIS sono trattati
come UTC. Il merge usa nearest-neighbour temporale con tolleranza 31 minuti e
fallisce esplicitamente se restano valori meteo non allineati.

Il loader materializza ogni ora tra il primo e l’ultimo timestamp. Le ore
SCADA assenti rimangono `NaN` in `ENERGIA`; non vengono eliminate e soprattutto
non vengono reinterpretate come produzione zero. `PVDataset` rifiuta timeline
duplicate, non ordinate o con intervalli diversi da un’ora.

## 3. Feature schema

L’ordine è un contratto persistente:

```text
 0 temp_z
 1 wind_z
 2 sin_elev
 3 cos_elev
 4 pv_lag
 5 pv_observed
 6 poa_z
 7 diffuse_fraction
 8 kt_poa
 9 kt_poa_std_3h
10 dpoa_z
11 m1
12 m2
13 m3
14 m4
15 m5
16 quality_valid
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

`pv_observed` distingue un lag nullo reale da un dato SCADA mancante. `m3`
resta attivo perché misura la completezza rolling di `ENERGIA`. `temp_z`,
`wind_z`, geometria solare, `pv_lag` e `pv_observed` restano attivi.

La radiazione diretta non è duplicata come canale separato: POA e frazione
diffusa determinano implicitamente direct e diffuse, evitando forte
collinearità tra `POA`, `direct`, `diffuse` e `direct+diffuse`.

## 4. Split e preprocessing

Lo split è strettamente cronologico:

```text
train = primo 80% dei timestamp target
validation = ultimo 20% dei timestamp target
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
pv_scale_fallback / poa_scale_fallback
pr_proxy
time_grid / missing_pv_fraction
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
x, y_poa, y_pv, pr_proxy, poa_cs, poa_scale,
pv_target_valid, pv_lag_valid
```

La forma di `x` è `[n_impianti, seq_len, 17]`; dopo batching è
`[batch, n_impianti, seq_len, 17]`.

`pv_target_valid` impedisce che un dato SCADA mancante, rappresentato
numericamente con zero nel tensore, venga supervisionato come produzione
nulla. `pv_lag_valid` serve a non valutare la persistence quando il valore
precedente manca.

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
| grafo | soglia 20 km, self-loop e nearest-neighbour per isolati |
| prior distanza | gaussiano, scala predefinita 10 km |
| forza prior | 1.0, aggiunta ai logit in log-spazio |

La BiLSTM non viola la causalità: legge in entrambe le direzioni soltanto
all’interno della finestra storica che termina a `t-1`.

Ogni nodo GAT ha un self-loop con prior 1. Un eventuale nearest-neighbour oltre
la soglia compete quindi con il messaggio del nodo stesso e viene attenuato
dal prior gaussiano, invece di ricevere attenzione 1 per assenza di alternative.

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
L_pv     = weighted_MSE(pred_pv, y_pv; mask=pv_target_valid)
L_phys   = weighted_MSE(
    pred_pv, pr_proxy * poa_norm; mask=pv_target_valid
)

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
dominare la loss. Giorno/notte è definito da `poa_clear_sky > 0.05 kW/m²`,
non dalla POA osservata: una giornata molto nuvolosa non diventa notte. La
maschera di validità PV si applica a `L_pv`, `L_phys` e alla peak loss;
`L_poa` resta attiva perché il meteo PVGIS è indipendente dalla disponibilità
SCADA.

## 8. Validazione e scelta checkpoint

Metriche obbligatorie:

- PV globale: MAE, RMSE, bias;
- PV giorno e notte separati;
- POA nelle ore diurne;
- persistence PV diurna, usando `pv_lag[t-1]`;
- metriche PV diurne per bin di potenza.

Il checkpoint è scelto su `rmse_pv_day`, non sulla loss composita. La loss
resta riportata come diagnostica. W&B salva le metriche dell’epoca scelta con
prefisso `best_`.

Tutte le metriche PV ignorano target SCADA mancanti. La persistence richiede
anche un `pv_lag[t-1]` osservato; le metriche POA restano disponibili.
Le metriche giorno/notte usano la stessa maschera clear-sky della loss.

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
- il batch passa da 5 a 8 elementi e include `poa_scale` e le due maschere PV;
- lo schema passa a 17 canali con `pv_observed`;
- la proxy `eta` viene sostituita da `pr_proxy` coerente con le scale;
- il grafo usa un prior gaussiano e non `1/distance`;
- lo split mensile o random deve essere sostituito da quello cronologico;
- la timeline deve essere materializzata e validata come griglia oraria;
- target e lag PV mancanti devono essere mascherati, non trasformati in zero;
- ogni normalizzazione deve ricevere il train mask;
- l’ablazione non deve cambiare `N_FEATURES` né l’architettura.

I vecchi checkpoint non sono compatibili con `head_poa` e non devono essere
caricati silenziosamente.

## 11. Checklist di porting

1. Portare insieme `dataset.py`, `quality_score.py` e `physics_loss.py`.
2. Portare `graph_builder.py` e `st_gnn.py`.
3. Adeguare ogni training loop al batch a 8 elementi.
4. Adeguare ogni loss e metrica a `pv_target_valid`.
5. Usare `pv_lag_valid` per la persistence.
6. Creare lo split prima di `compute_qs` e `PVDataset`.
7. Passare lo stesso `fit_time_mask` a entrambi.
8. Salvare `preprocessing_state.json`.
9. Usare `rmse_pv_day` per early stopping/checkpoint.
10. Decidere soltanto `include_poa_inputs=True/False`.
11. Non cambiare forma o iperparametri tra i due lati dell’ablazione.
12. Allineare anche `notebooks/run_training.ipynb` alle costanti del branch.
13. Eseguire i test e uno smoke training prima di pubblicare.

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
- griglia temporale strettamente oraria;
- target PV mancanti esclusi dalla supervisione;
- giorno/notte determinato dalla clear-sky POA;
- POA clear-sky nullo di notte;
- coerenza dimensionale della loss fisica;
- split cronologico;
- assenza di nodi isolati, self-loop GAT e prior geografico limitato;
- sintassi e API del notebook di training.

## 13. Limiti noti

- Tilt e azimuth PVGIS sono assunti comuni a tutti gli impianti.
- La componente PVGIS ground-reflected non è disponibile: POA osservata e
  clear-sky sono entrambe definite sulle sole componenti beam + diffuse.
- Il fuso `Europe/Rome` è un’ipotesi esplicita sul formato SCADA.
- La validation finale non sostituisce un test set temporale indipendente.
- Il limite `kt_poa=1.6` è calibrato sul file 2019 e va rivalutato su altri
  anni o orientamenti.
- POA continua a essere necessario come target e per le metriche diurne anche
  nel branch senza input POA.

### Sweep W&B sui seed

`notebooks/run_training.ipynb` registra nel progetto W&B `physiq_pv` uno
sweep `grid` il cui unico parametro variabile è:

```text
seed ∈ {42, 123, 2024}
```

L’agente esegue esattamente tre run (`count=3`). Tutti gli altri
iperparametri restano fissi: ogni run ha un massimo di 15 epoche e può
terminare prima tramite early stopping su `rmse_pv_day`. Checkpoint,
preprocessing e configurazioni sono salvati separatamente per seed.

I CSV di mapping/coordinate e il NetCDF non sono versionati. Il notebook
risolve i percorsi prima del caricamento e supporta:

```text
PHYSIQ_SENTINEL_DIR
PHYSIQ_PLANT_MAPPING_PATH
PHYSIQ_ENERGY_COORDS_PATH
PHYSIQ_PVGIS_PATH
```

`plant_mapping.csv` è opzionale quando è disponibile
`energy_with_coordinates.csv`; almeno uno dei due deve fornire coordinate
finite. Le UPN Sentinel non geolocalizzate vengono escluse prima del match
PVGIS e del grafo, senza imputare coordinate artificiali. Sul server il NetCDF
PVGIS viene cercato prima in
`/data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance/` e poi nella
directory `data/` del progetto.

## 14. Profilo di porting per branch PVGIS-only

Gli altri branch BiLSTM+GAT basati direttamente sul NetCDF PVGIS non devono
importare il percorso Sentinel/SCADA. In particolare, non servono:

- `load_sentinel_hourly`;
- conversione `Europe/Rome → UTC`;
- nearest-neighbour temporale Sentinel/PVGIS;
- logica specifica dei CSV UPN.

Il mapping dati PVGIS-only è:

```text
location                      -> plant
time                          -> time, già interpretato come UTC
pv_power_output               -> target PV
temperature_2m                -> temperatura
wind_speed_10m                -> vento
direct_irradiance_tilted      -> beam sul piano
diffuse_irradiance_tilted     -> diffuse sul piano
direct_tilted + diffuse_tilted -> target POA ricostruito
lat / lon                     -> coordinate del grafo
```

Il timestamp può essere, per esempio, `HH:10`: non deve essere arrotondato se
tutta la serie mantiene esattamente un passo di un’ora. La geometria pvlib
deve usare quegli stessi istanti come UTC.

Nel profilo PVGIS-only:

- `pv_target_valid = isfinite(pv_power_output)`;
- `pv_lag_valid` è la validità del timestamp precedente;
- `pv_observed` resta un canale causale, normalmente sempre uguale a 1;
- non si materializzano buchi Sentinel inesistenti, ma si valida comunque la
  cadenza oraria completa;
- split, normalizzazioni, scale e PR proxy restano train-only;
- giorno/notte resta basato su clear-sky POA;
- head POA, loss fisica e grafo GAT restano quelli descritti sopra.

### Quality Score nei branch PVGIS-only

`pv_power_output` è prodotto dal modello fisico PVGIS a partire dalla stessa
irradianza. Di conseguenza `m1`, `m2`, `m4` e `m5` possono risultare quasi
deterministici e non misurano la qualità di un sensore SCADA reale.

Regola di porting:

- se il branch non studia il Quality Score, non introdurre `m1..m5` soltanto
  per uniformarlo a questo branch;
- se il branch studia anomalie o degradazioni iniettate, mantenere metriche
  rolling causali e calibrazione train-only, dichiarando che misurano coerenza
  sintetica PVGIS e non qualità SCADA;
- per confronti POA-on/POA-off nello stesso branch, mantenere identici schema
  e architettura e mascherare soltanto i canali POA-dipendenti.

### Checklist PVGIS-only

1. Rinominare le dimensioni/variabili senza passare dal loader Sentinel.
2. Ricostruire POA da direct+diffuse e ignorare il campo POA nullo.
3. Conservare gli attributi `tilt_angle=30` e `azimuth_angle=180`.
4. Calcolare clear-sky POA sugli stessi timestamp UTC e con componenti
   coerenti (`albedo=0` quando `Gr(i)` è assente).
5. Validare cadenza oraria, coordinate finite e target PV finiti.
6. Creare split e fit mask prima di qualsiasi normalizzazione.
7. Applicare le maschere di validità a loss, fisica e metriche.
8. Selezionare il checkpoint su `rmse_pv_day`.
9. Portare il Quality Score soltanto se fa parte dell’esperimento.
10. Salvare configurazione e preprocessing insieme al checkpoint.
