# MTGFlow base su PVGIS

Il flusso neurale di anomaly detection descritto qui espone un solo detector
multivariato: MTGFlow base, implementato localmente dalle equazioni del paper
*Label-Free Multivariate Time Series Anomaly Detection*. Il checkout degli
autori in `.research` è soltanto un riferimento di audit e non è importato a
runtime.

## Gerarchia di fedeltà

La regola di implementazione è:

1. quando il paper v2 specifica chiaramente equazione, semantica o protocollo,
   prevale il paper;
2. quando il paper lascia un dettaglio ambiguo o non specificato, si segue il
   repository ufficiale;
3. gli adattamenti necessari a PVGIS e al downstream SDE sono dichiarati nei
   metadati e non vengono presentati come protocollo originale.

Di conseguenza, normalizzazione per riga dell'attenzione (Eq. 7),
condizionamento `A H W1 + H_previous W2` (Eq. 8), likelihood con aggregazione
temporale (Eq. 12) e soglie IQR (Eq. 13--15) seguono il paper anche quando
dettagli del codice ufficiale differiscono. La tensorizzazione pointwise
`input_size=1`, hidden size, numero di layer interni, weight decay, gradient
clipping e seed provengono invece dal repository ufficiale perché non sono
determinati completamente dal testo.

Nel caso scalare, i parametri affini del MAF dipendono dal contesto
grafo-LSTM e non dalla stessa variabile trasformata. Questo mantiene la
trasformazione bijettiva e il log-Jacobiano esatto richiesti dalle Eq. 2--4,
conservando al tempo stesso la tensorizzazione `input_size=1` del repository.

## Protocollo

```text
2005--2018  training, statistiche z-score e calibrazione delle soglie
2019        target/test
```

Configurazione predefinita:

```text
window_size=60, train_stride=10, score_stride=10, n_blocks=2, batch_size=256
Adam lr=2e-3, weight_decay=5e-4, gradient clipping=1
attention_dropout=0.2, epochs=40, seed=15,16,17,18,19
```

La soglia globale è `Q3 + 1.5*IQR`. Le soglie per entità applicano inoltre il
fattore `lambda=0.8` dell'Eq. 15. Tutte le soglie sono stimate esclusivamente
sugli score di training.

## Preparazione

Per il confronto controllato con CATCH e M2AD, ogni modello locale MTGFlow
riceve gli stessi tre canali fisici: `solar_irradiance_poa`,
`temperature_2m` e `wind_speed_10m`. Le feature climatiche derivate e le
codifiche temporali non entrano nel detector.

La modalità paper usa soltanto lo z-score interno training-only. Non passare
`--seasonal-normalization`:

```powershell
python scripts/prepare_pvgis_mtgflow.py `
  --pvgis-dir DATA_DIR `
  --out-dir outputs/pvgis_mtgflow/prepared `
  --train-start 2005 --train-end 2018 `
  --test-year 2019
```

Una validation separata non è necessaria perché il training usa un numero fisso
di epoche e le soglie sono calibrate sul training. `--validation-year` resta
disponibile come controllo opzionale di schema, senza influenzare scaler,
ottimizzazione o soglie.

## Esecuzione

```powershell
python scripts/run_pvgis_mtgflow.py `
  --manifest outputs/pvgis_mtgflow/prepared/manifest_shard_0000.csv `
  --out-dir outputs/pvgis_mtgflow/shard_0000 `
  --device cuda
```

Il runner esegue automaticamente i seed `15,16,17,18,19` e li conserva in
directory separate. `--seed 15` limita volontariamente l'esecuzione a un solo
seed. Le metriche possono essere aggregate; gli score non vengono mai mediati.
La directory passata a `--out-dir` deve essere nuova o vuota, così run diverse
non possono lasciare artifact mescolati.

`--score-stride 1` produce uno score ogni ora, ma è registrato nei metadati come
`dense_hourly_window_adaptation`: non è il protocollo letterale del paper. Anche
in questa modalità il flag riguarda l'intera finestra precedente di 60 ore, non
il singolo punto finale. Gli output espongono quindi sia `window_start` sia
`window_end`; `timestamp` resta un alias di `window_end` per il join post-hoc.

Per il downstream STGNN+SDE si usa un solo seed e lo score orario:

```powershell
python scripts/run_pvgis_mtgflow.py `
  --manifest outputs/pvgis_mtgflow/prepared/manifest_shard_0000.csv `
  --out-dir outputs/pvgis_mtgflow/downstream_dense `
  --device cuda --seed 15 --score-stride 1
```

La pipeline forecasting usa direttamente `is_anomaly`; non stima una seconda
soglia sugli score MTGFlow.

## Output

- `seed_<n>/anomaly_scores.csv`: score e flag globali test per finestra;
- `seed_<n>/train_anomaly_scores.csv`: score e flag globali aggregati del
  training, con la stessa soglia training-only;
- `entity_anomaly_scores.csv`: score, soglia e flag per entità;
- `train_scores.csv` e `train_entity_scores.csv` nelle directory delle singole
  località: calibrazione riproducibile;
- `metadata.json`: configurazione, feature e soglie impiegate;
- `checkpoint.pt`: pesi, scaler z-score, feature, configurazione e ambiente;
- `summary_by_seed.csv` e `summary_aggregate.csv`: statistiche operative
  separate e sintesi, non le metriche AUROC del paper;
- `run_metadata.json` e `environment.json`: protocollo e ambiente esatto della run.

L'ambiente risolto resta fissato da `uv.lock`. Ogni run registra inoltre versioni
Python/NumPy/Pandas/PyTorch, CUDA/cuDNN, GPU, commit/stato Git e stato
deterministico. `load_mtgflow_checkpoint(...)` ricostruisce direttamente il
modello dal bundle salvato. Il codice è organizzato nel package `mtgflow/`:
`config.py` definisce l'unica configurazione di riferimento, `model.py` contiene
esclusivamente l'architettura, `pipeline.py` gestisce training/checkpoint/scoring
e `result.py` definisce il risultato tipizzato.

Il MAF usa la tensorizzazione pointwise `input_size=1` del repository ufficiale,
un dettaglio non specificato completamente dal paper. I metadati registrano
`alignment_policy=paper_when_explicit_official_repo_when_underspecified` e
distinguono sempre l'allineamento del metodo dalla replica numerica degli
esperimenti: PVGIS sostituisce infatti i benchmark originali.

Il workflow interattivo è disponibile in
`notebooks/mtgflow_pvgis_workflow.ipynb`; orchestra gli stessi script senza
duplicare la logica di preparazione o training.
