# Run STGAN e collegamento agli sweep W&B

Progetto: `albertopedalino-politecnico-di-torino/physiq_pv`.
Il runner `scripts/run_stgan_wandb.py` riusa il training dei branch GAT e CNN
e registra configurazione effettiva, loss per epoca, tempi, memoria e ambiente.
Il default del runner W&B e' CUDA con precisione BF16.

## Registrare una nuova run

Sul server, dopo `wandb login`, avviare dal branch corrispondente:

```bash
# experiment/stgan-era5-gat-mc-dropout
python scripts/run_stgan_wandb.py --backend era5 \
  --prepared-dir /percorso/era5/prepared

# experiment/stgan-cnn
python scripts/run_stgan_wandb.py --backend pvgis \
  --manifest /percorso/manifest.csv
```

Si possono usare `STGAN_PREPARED_DIR` / `STGAN_MANIFEST` al posto degli
argomenti. `--dry-run` mostra la configurazione senza contattare W&B o
iniziare il training; `--wandb-mode offline` registra localmente.
`--model-config file.json` sovrascrive i valori predefiniti usando i nomi di
`STGANCNNConfig` (per esempio `learning_rate`, non `lr`). Non definisce uno
spazio di ricerca: contiene i valori di una singola configurazione.

Ogni run usa `outputs/stgan_wandb/<backend>/<run_id>/results` e conserva
`wandb_run.json` nella cartella superiore, con URL, eventuale sweep ID,
configurazione, seed e stato. `--output-root` cambia la radice. Una run
corrisponde a un seed; W&B puo' fornire il seed nella sua configurazione.
I file di dati, gli score e i checkpoint restano sul server; il runner
registra in W&B le metriche e i metadati, senza caricare i grandi artefatti.

Metriche gia' registrate:

- `train/generator_loss`, `train/discriminator_loss`, `train/epoch_seconds`,
  `train/samples`, con asse `epoch`;
- `performance/training_seconds`, `performance/scoring_seconds`,
  `performance/training_samples_per_second`,
  `performance/scoring_samples_per_second`, `performance/peak_cuda_memory_bytes`;
- gli altri scalari disponibili in `performance`, inclusa la calibrazione
  quando eseguita dal branch.

Queste sono metriche diagnostiche: nessuna e' stata scelta come obiettivo
dello sweep. Il codice non aggiunge split di validazione o metriche di
selezione: verranno definiti con la scelta del target.

## Bozza dello sweep bayesiano

`sweeps/stgan_bayes.draft.yaml` contiene progetto, `method: bayes` e comando
del runner del rispettivo branch. `metric.name`, `metric.goal` e
`parameters` restano da compilare dopo la ricerca. Non e' uno sweep remoto
gia' creato e non puo' essere avviato in questo stato.

Quando target e iperparametri saranno definiti, completare la configurazione
e verificare che il runner registri la metrica scelta. Quindi:

```bash
# Controllo locale; una bozza incompleta produce un errore esplicito.
python scripts/create_stgan_sweep.py sweeps/stgan_bayes.draft.yaml

# Registrazione in W&B, senza avviare training.
python scripts/create_stgan_sweep.py sweeps/stgan_bayes.draft.yaml --create

# Eseguire sul server il comando wandb agent stampato dal comando precedente.
```

Il comando dell'agente passa il backend; percorsi dei dati e radice output
arrivano dalle variabili d'ambiente. I parametri campionati vengono letti da
`wandb.config`, che prevale sui valori di base. I nomi sconosciuti vengono
rifiutati. `wandb.init` associa automaticamente la nuova run allo sweep
fornito dall'agente; un avvio manuale crea una run ordinaria nel progetto.

I processi di training gia' in esecuzione continuano con il codice con cui
sono stati avviati. Questa integrazione si applica ai nuovi avvii.

## Test

```bash
python tests/test_stgan_wandb.py
```

I test verificano applicazione dei parametri assegnati, metriche per epoca,
stato delle run fallite e protezione contro bozze incomplete. Eseguono anche
un piccolo training CPU FP32 con il vero SDK W&B in modalita' offline, senza
autenticazione o scritture remote. Nel branch GAT verificano il runner ERA5.
