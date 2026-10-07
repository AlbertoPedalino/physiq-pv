# Run STGAN CNN e collegamento agli sweep W&B

Progetto: `albertopedalino-politecnico-di-torino/physiq_pv`.
Il runner `scripts/run_stgan_wandb.py` riusa il training ConvGRU di questo
branch e registra configurazione effettiva, loss per epoca, tempi, memoria e
ambiente. Il default del runner W&B e' CUDA con precisione BF16, come nel
branch `experiment/stgan-era5-gat-mc-dropout`; il GAT resta in quel branch.
Per FP32 passare `precision: fp32` nella configurazione
([Precisione STGAN](STGAN_PRECISION.md)).

## Registrare una nuova run

Sul server, dopo `wandb login`:

```bash
python scripts/run_stgan_wandb.py --backend era5 \
  --prepared-dir /percorso/era5/prepared
```

Si puo' usare `STGAN_PREPARED_DIR` al posto dell'argomento. `--dry-run` mostra
la configurazione senza contattare W&B o iniziare il training;
`--wandb-mode offline` registra localmente. `--model-config file.json`
sovrascrive i valori predefiniti usando i nomi di `STGANCNNConfig` (per esempio
`learning_rate`, non `lr`). Non definisce uno spazio di ricerca: contiene i
valori di una singola configurazione. I valori di base del backend `era5` sono
quelli di `run_era5_stgan.py train` senza opzioni (batch 256, batch di scoring
1024, `trend_steps=56`, MC Dropout con 20 campioni, `time_encoding=onehot`,
`normalization=minmax`), con la sola differenza di `precision=bf16`. Il backend
`pvgis` resta disponibile con `--manifest` / `STGAN_MANIFEST`.

### Codifica temporale e normalizzazione degli input

`time_encoding=onehot` (default) usa giorno della settimana e ora come nel
paper; `cyclic` usa seno/coseno dell'ora solare locale di ogni cella e della
posizione nell'anno. Sostituisce il precedente `annual_cycle`, che non e' piu'
accettato. `normalization=minmax` (default) e' il min-max sui soli dati di
training; `seasonal` standardizza prima ogni cella per giorno dell'anno
(+-`seasonal_window_days`) e ora del giorno, con statistiche calcolate solo sul
training, poi applica lo stesso min-max. Entrambe le scelte richiedono un nuovo
training e sono registrate in metadati, checkpoint e configurazione W&B.

### Equilibrio generatore/discriminatore

`learning_rate` e' il learning rate Adam di G. `discriminator_lr_ratio`
imposta `lr_D / lr_G`: per esempio `learning_rate=0.0002` e
`discriminator_lr_ratio=2` danno `lr_D=0.0004`. Il rapporto predefinito e' 1,
quindi le configurazioni precedenti mantengono lo stesso LR per G e D.
In alternativa i due rate si impostano in modo indipendente con
`generator_learning_rate` e `discriminator_learning_rate`. Il default di
entrambi e' `null`: valgono allora `learning_rate` e
`learning_rate * discriminator_lr_ratio`, quindi le configurazioni precedenti
non cambiano. Ogni rate si indica in una sola forma: `generator_learning_rate`
insieme a un `learning_rate` diverso, o `discriminator_learning_rate` insieme a
un `discriminator_lr_ratio` diverso da 1, vengono rifiutati. Dove la scelta e'
esplicita il rifiuto non dipende dai valori: `--model-config` e lo spazio dello
sweep non possono contenere entrambi i nomi della stessa rete, e la CLI non
accetta insieme `--discriminator-lr-ratio` e `--discriminator-learning-rate`
(`--lr`, `--learning-rate` e `--generator-learning-rate` sono la stessa
opzione). Nella configurazione W&B di una run dello sweep `learning_rate` e
`discriminator_lr_ratio` restano ai default e non sono usati: i rate effettivi
sono `generator_learning_rate` e `discriminator_learning_rate`, riportati anche
nel summary.
`generator_reconstruction_weight` pesa solo la ricostruzione nella loss di G;
il termine avversario di G e la loss di D non ne dipendono. Il training
fa per default un aggiornamento di D e uno di G per batch.

### Numero di aggiornamenti di D e G

`discriminator_generator_update_ratio` e' il numero di `optimizer.step()` per
batch, scritto come `"D:G"`: `1:1` (default) e' un aggiornamento di D e uno di
G, `2:1` due di D e poi uno di G, `1:2` uno di D e poi due di G. Riguarda il
numero di aggiornamenti ed e' indipendente da `discriminator_lr_ratio`, che
scala il learning rate di D. Gli aggiornamenti ripetuti usano lo stesso batch:
ognuno rifa' per intero forward e backward (nuove maschere di dropout), prima
tutti quelli di D e poi tutti quelli di G, quindi ogni epoca resta un passaggio
completo sui dati per entrambe le reti. Con `1:1` il training e' identico al
precedente. Nel GAT i blocchi del discriminatore accumulano i gradienti in un
solo aggiornamento e non contano come aggiornamenti. Le loss registrate sono
la media sugli aggiornamenti di ciascuna rete. La CLI usa
`--discriminator-generator-update-ratio 2:1`; in YAML il valore va tra
virgolette (`"1:1"` senza virgolette viene letto come il numero 61 e
rifiutato). Il rapporto e' salvato in metadati e checkpoint, il resume lo
confronta (i checkpoint precedenti valgono `1:1`) e W&B registra il rapporto
nel summary e il numero effettivo di aggiornamenti per epoca in
`train/discriminator_updates` e `train/generator_updates`, letti dal contatore
di Adam.

Entrambi i runner CLI accettano `--lr` (alias `--learning-rate`),
`--discriminator-lr-ratio`, `--generator-learning-rate`,
`--discriminator-learning-rate` e `--generator-reconstruction-weight`.
Il runner W&B riceve gli stessi controlli attraverso i nomi della dataclass.
I learning rate effettivi sono stampati prima del training e registrati nei
metadati, nei checkpoint e nel summary W&B (`generator_learning_rate`,
`discriminator_learning_rate`).

### Ripresa da un checkpoint di epoca

`run_era5_stgan.py train --resume-from <run>/model_epoch_N.pt` continua una
run interrotta, anche nella stessa cartella. Il resume ripristina pesi e stati
Adam e rifiuta un checkpoint con learning rate di G o D, rapporto,
`generator_reconstruction_weight`, `time_encoding` o normalizzazione diversi
dalla configurazione richiesta, per evitare che la run ripresa cambi
silenziosamente l'equilibrio della GAN o gli input. Un checkpoint precedente
con `annual_cycle=False` equivale a `time_encoding=onehot`; con
`annual_cycle=True` non puo' essere ripreso. La precisione non viene
confrontata. I
checkpoint precedenti a questa modifica non contengono questi valori e vengono
accettati senza confronto (Adam riparte dai momenti iniziali). Con dropout
disattivato la ripresa e' identica bit a bit a una run non interrotta; con
dropout attivo le maschere successive alla ripresa seguono un flusso casuale
diverso, perche' lo stato del generatore casuale non e' nel checkpoint.

Ogni run usa `outputs/stgan_wandb/<backend>/<run_id>/results` e conserva
`wandb_run.json` nella cartella superiore, con URL, eventuale sweep ID,
configurazione, seed e stato. `--output-root` cambia la radice. Una run
corrisponde a un seed; W&B puo' fornire il seed nella sua configurazione.
I file di dati, gli score e i checkpoint restano sul server; il runner
registra in W&B le metriche e i metadati, senza caricare i grandi artefatti.

Metriche gia' registrate:

- `train/generator_loss`, `train/discriminator_loss`, `train/epoch_seconds`,
  `train/samples`, con asse `epoch`;
- `train/generator_reconstruction_loss`: errore di ricostruzione non pesato,
  cioe' la MSE mascherata prima della moltiplicazione per
  `generator_reconstruction_weight`; e' confrontabile tra run con pesi diversi;
- `train/generator_reconstruction_weighted_loss`: lo stesso valore moltiplicato
  per `generator_reconstruction_weight`;
- `train/generator_adversarial_loss`: termine avversario di G. Vale
  `generator_loss = generator_reconstruction_weighted_loss + generator_adversarial_loss`;
- `train/discriminator_real_loss`, `train/discriminator_fake_loss`: i due
  addendi di `train/discriminator_loss` (meta' della BCE sui campioni reali e
  meta' di quella sui generati);
- `train/discriminator_real_mean`, `train/discriminator_fake_mean`: media
  dell'uscita di D su reali e generati durante l'aggiornamento di D, con la
  convenzione del modello (0 = reale, 1 = generato);
- con la validation attiva: `validation/discriminator_feature_discrepancy`,
  `validation/score_delta_mean`, `validation/score_delta_median`,
  `validation/score_spearman`, `validation/discriminator_feature_mmd`, `validation/seconds`;
- `performance/training_seconds`, `performance/scoring_seconds`,
  `performance/training_samples_per_second`,
  `performance/scoring_samples_per_second`, `performance/peak_cuda_memory_bytes`;
- `score/reconstruction_min`, `score/reconstruction_max`, `score/discriminator_min`,
  `score/discriminator_max`: i quattro fattori min-max dello score, stimati sulle mappe
  MC-medie dell'intero test di quella run;
- `score/reconstruction_raw_*`, `score/discriminator_raw_*`, `score/anomaly_*` con
  `median`, `p95`, `mean`, `std`: componenti non normalizzate e score finale.
  Sono descrittive: non entrano nello score e non sono il target dello sweep.


Queste sono metriche diagnostiche: nessuna e' stata scelta come obiettivo
dello sweep.

### Validation e monitoraggio per epoca

Con `validation_holdout=true` (`--validation-holdout` in `run_era5_stgan.py`)
il training usa 1980-2003, il 2004 resta fuori dal training come validation e
il test 2005-2025 non cambia. Il default della CLI e della dataclass e' `false`
(training fino al 2004, come nelle run precedenti); il runner W&B ERA5 e la
bozza dello sweep usano `true`. Con la validation il min-max degli input e'
stimato su 1980-2003. La validation serve solo al monitoraggio per epoca e
all'objective dello sweep: non entra in nessuna loss, non calibra lo score, non
stima i min-max dello score finale, non definisce soglie o normalizzazioni del
test e non usa label. Lo score del test resta quello del paper, con i suoi
fattori stimati sul solo test. Come in ogni split cronologico, gli ultimi
istanti del 2004 sono la storia di input (`trend_steps`) dei primi istanti del
test. Il runner PVGIS non la supporta.

La cache preparata ha tre partizioni: `train` (1980-2003), `validation` (2004)
e `test` (dal 2005). Una cache preparata prima di questo split (seconda
partizione 2003-2004) viene letta senza riscriverla: il 2003 entra nel training
e il 2004 e' la validation.

Dopo ogni epoca il modello viene valutato su un sottoinsieme fisso della
validation: `monitoring_timestamps` istanti (default 32) equispaziati su tutto
il 2004, ciascuno con tutte le celle della griglia ERA5 81 x 131, cioe'
10.611 celle per istante (nessuna cella e' mascherata). In totale sono
32 x 10.611 = 339.552 coppie cella-istante, le stesse nei branch CNN e GAT:
cambia solo il batching (il ConvGRU legge una patch 3x3 centrata su ogni cella,
il GAT l'intera griglia di un istante). Il
sottoinsieme non dipende da seed o epoca; `0` disattiva il monitoraggio. La
valutazione usa un solo forward deterministico (dropout spento), senza
gradienti e senza passi degli ottimizzatori, e ripristina stato dei generatori
casuali e modalita' dei moduli: i pesi di una run con monitoraggio sono
identici bit a bit a quelli della stessa run senza.

- `validation/discriminator_feature_discrepancy`: media di `|f(x) - f(G(x))|`
  su celle e dimensioni, dove `f` sono le attivazioni del penultimo livello di
  D (quelle che entrano nell'ultimo livello lineare) per l'osservazione e per
  la sua ricostruzione. Non e' un termine di loss.
- `validation/score_delta_mean`, `validation/score_delta_median`: media e
  mediana di `|score_e - score_(e-1)|` sulle stesse celle nello stesso ordine.
  `validation/score_spearman`: correlazione di Spearman tra i due vettori.
  Mancano alla prima epoca monitorata. Lo score di monitoraggio e' la stessa
  somma di componenti min-max dello score ufficiale, ma i quattro fattori sono
  stimati una sola volta, alla prima epoca monitorata della run, e poi tenuti
  fissi; sono salvati sotto `monitoring` nei checkpoint e nei metadati e non
  sono mai usati dalla valutazione finale.
- `validation/discriminator_feature_mmd`: MMD^2 (stimatore non distorto) tra le attivazioni
  `f(x)` e `f(G(x))`, kernel RBF `exp(-||a-b||^2 / (2 sigma^2))` con `sigma^2`
  uguale alla mediana delle distanze al quadrato tra i punti dei due insiemi
  uniti. Usa al piu' `monitoring_feature_mmd_samples` celle equispaziate per insieme
  (default 1024) e viene calcolata ogni `monitoring_feature_mmd_every_n_epochs` epoche
  (default 1; `0` la disattiva). Puo' essere leggermente negativa quando i due
  insiemi coincidono.

Il resume ripristina lo stato del monitoraggio dal checkpoint e rifiuta un
checkpoint con split diversi (validation presente o assente, periodo di
training) o con un sottoinsieme di monitoraggio diverso. Un checkpoint senza
stato di monitoraggio viene accettato e il confronto riparte dall'epoca
successiva.

### Errore di ricostruzione sulla validation completa

Con `validation_holdout=true`, a training finito e prima dello scoring del test
il modello viene valutato una volta su tutta la validation 2004 (2.928
istanti x 10.611 celle = 31.069.008 coppie cella-istante), non sui 32 istanti
del monitoraggio per epoca. Per ogni coppia si calcola `reconstruction_raw`,
cioe' la componente di ricostruzione dello score prima del min-max (errore
quadratico medio sulle celle della patch e sulle feature), con MC Dropout: ogni
draw usa la stessa mask, l'errore di ogni draw e' calcolato solo sulle celle
valide, poi si fa la media sui `mc_samples` draw (20). Sulle coppie valide si
calcolano media, deviazione standard, mediana, P95 e mediana + P95.

La regola di validita' e' quella dell'errore di ricostruzione STGAN ed e' la
stessa per CNN e GAT: la mask e' spaziale (una cella della patch esiste o e'
assente, con tutte le sue feature) e gli input non finiti sono rifiutati prima
del training. Le celle assenti non entrano ne' nel numeratore ne' nel
denominatore dell'errore. Una coppia la cui mask non ha nessuna cella valida
viene esclusa dalla distribuzione e contata, non trasformata in zero; non c'e'
nessuna imputazione e i valori inclusi devono essere finiti. Su ERA5 la griglia
e' completa: candidate e valide coincidono e le escluse sono 0.

- `validation/reconstruction_raw_mean`, `_std`, `_median`, `_p95`,
  `validation/reconstruction_raw_median_plus_p95`;
- `validation/objective_candidate_points`, `validation/objective_valid_points`,
  `validation/objective_excluded_points`, `validation/objective_seconds`.

Gli stessi valori sono in `metadata.json` sotto `backend.validation_objective`.
Il calcolo legge soltanto il modello e ripristina lo stato dei generatori
casuali: pesi, score e incertezze del test sono identici bit a bit a quelli
della stessa run senza questo calcolo. E' una diagnostica: non e' l'obiettivo
dello sweep.

### Spazio PCA fisso (riferimento comune alle run)

Serve alla futura metrica MMD: uno spazio di feature fissato una sola volta,
uguale per ogni run, seed e architettura (CNN e GAT). Si costruisce prima
dello sweep:

```bash
python scripts/run_era5_stgan.py pca-reference \
  --prepared-dir /percorso/era5/prepared --output-dir /percorso/era5/pca_reference
```

Il comando legge solo la partizione di training (1980-2003): mai validation o
test, nessun modello, nessuna label. Un campione e' il campo completo di un
istante, 81 x 131 celle x 15 variabili, normalizzato per variabile con il
min-max del solo training che usa la pipeline e appiattito in 159.165 valori.
La PCA e' esatta (autovalori della matrice di Gram dei campioni) su
`--pca-samples` istanti di training (default 4096, circa il 5,8% dei 70.128
disponibili) equispaziati su tutto il periodo, primo e ultimo inclusi; gli
istanti usati sono salvati. Vengono conservate le prime `--pca-components`
componenti (default 100, meno se il rango e' inferiore): e' quanto si salva,
non una scelta di quante usarne.

Per vedere la curva oltre 100 componenti si costruisce un secondo riferimento
in una cartella nuova, con gli stessi dati, la stessa normalizzazione e gli
stessi istanti:

```bash
python scripts/run_era5_stgan.py pca-reference \
  --prepared-dir /percorso/era5/prepared --output-dir /percorso/era5/pca_reference_300 \
  --pca-components 300
```

Le prime 100 componenti sono quelle del riferimento a 100. La cartella
esistente non viene toccata: il comando rifiuta una cartella non vuota.

Nella cartella: `scaler_minimum.npy`, `scaler_scale.npy`, `pca_mean.npy`,
`pca_components.npy`, `pca_explained_variance.npy`,
`pca_explained_variance_ratio.npy`, `pca_fit_timestamps.npy`,
`metadata.json` (forma del campo, dimensione appiattita, numero e selezione
dei campioni, varianza spiegata per componente e cumulata, valori a 50, 75,
100, 150, 200, 250 e 300 componenti (`null` oltre quelle salvate), numero
minimo di componenti che raggiunge l'80%, 85%, 90% e 95% (`null` se le
componenti salvate non ci arrivano), SHA-256 di ogni file e impronta
complessiva) e il grafico
`pca_cumulative_explained_variance.png`. Il numero di componenti da usare non
e' ancora scelto (`chosen_components: null`).

Una run carica il riferimento con `--pca-reference-dir` (runner W&B:
`STGAN_PCA_REFERENCE_DIR`): ogni file e' verificato con il suo checksum e la
run viene rifiutata se variabili, celle o min-max non coincidono. Il
riferimento non viene mai ristimato da una run: se la cartella manca la run
fallisce con un messaggio esplicito, e una run di uno sweep ERA5 senza
riferimento viene rifiutata. L'impronta e' salvata nei checkpoint e nei
metadati (`backend.pca_reference`). Caricarlo non cambia training, score o
test. `PCAReference.transform` applica la stessa trasformazione a osservazioni
e ricostruzioni; `model_features` la calcola per un modello CNN o GAT.

### MMD nello spazio PCA (obiettivo dello sweep)

La MMD confronta, sulla validation, le osservazioni con le loro ricostruzioni
nello spazio PCA fisso. Usa un secondo riferimento, costruito una sola volta
sopra il riferimento PCA a 100 componenti (che viene solo caricato, mai
ristimato):

```bash
python scripts/run_era5_stgan.py mmd-reference \
  --prepared-dir /percorso/era5/prepared \
  --pca-reference-dir /percorso/era5/pca_reference \
  --output-dir /percorso/era5/mmd_reference
```

Il comando legge solo la validation (2004): mai training o test, nessun
modello, nessuna label. Un campione e' il campo completo di un istante
(`H x W x C`). I sottoinsiemi sono 5 da 512 istanti (`--subsets`,
`--subset-size`), estratti con una permutazione a seme fisso
(`--subset-seed`, default 0) degli istanti della validation e senza
sovrapposizioni finche' gli istanti bastano (5 x 512 = 2560 su 2928). Le
osservazioni dei sottoinsiemi passano per la normalizzazione e le prime
`--pca-components` componenti (default 100) del riferimento PCA; `sigma` e' la
mediana delle distanze euclidee fra quelle feature reali, tutti i
sottoinsiemi insieme. `sigma` e' calcolato qui e poi congelato: mai per run,
per epoca o sulle ricostruzioni.

Nella cartella: `subset_positions.npy`, `subset_timestamps.npy`,
`real_features.npy` e `metadata.json` (`sigma`, impronta del riferimento PCA
usato, numero di componenti, SHA-256 dei file e impronta complessiva).

Una run lo carica con `--mmd-reference-dir` insieme a `--pca-reference-dir`
(runner W&B: `STGAN_MMD_REFERENCE_DIR` e `STGAN_PCA_REFERENCE_DIR`). A ogni
epoca, per ciascun sottoinsieme, G ricostruisce gli stessi istanti (un
forward deterministico, dropout spento); osservazioni e ricostruzioni, nella
stessa normalizzazione, sono appiattite e proiettate sulle stesse componenti.
Fra le due nuvole si calcola la MMD^2 non distorta con kernel media di tre RBF
di banda `sigma/2`, `sigma` e `2 sigma`:

```text
k(a, b) = media_s exp(-||a - b||^2 / (2 (s sigma)^2)),   s in {1/2, 1, 2}
MMD^2   = media_{i != j} k(x_i, x_j) + media_{i != j} k(y_i, y_j) - 2 media_{i, j} k(x_i, y_j)
```

Metriche per epoca: `validation/pca_mmd_mean` e `validation/pca_mmd_variance`
(media e varianza campionaria delle 5 MMD^2) e
`validation/pca_mmd_rolling_mean`, la media di `pca_mmd_mean` sulle ultime
`mmd_objective_window` epoche (default 5; tutte quelle disponibili se sono
meno). Quest'ultima e' l'obiettivo dello sweep. `validation/pca_mmd_seconds`
e' il tempo del calcolo.

La run viene rifiutata se il riferimento MMD manca, se e' stato costruito su
un altro riferimento PCA, se i suoi istanti non sono nella validation o se le
osservazioni della run non riproducono le feature salvate; una run di uno
sweep ERA5 senza i due riferimenti viene rifiutata. L'impronta e' salvata nei
checkpoint (`mmd_reference`), nei metadati (`backend.pca_mmd`) e nel summary
W&B (`mmd_reference`); un resume su un altro riferimento viene rifiutato.
Training, loss, score e test non cambiano. La MMD sulle attivazioni di D resta
una diagnostica separata con il nome `validation/discriminator_feature_mmd`,
come `validation/reconstruction_raw_median_plus_p95`.

## Bozza dello sweep bayesiano

`sweeps/stgan_bayes.draft.yaml` contiene progetto, `method: bayes` e comando
del runner ERA5/CNN. Lo spazio di ricerca, uguale a quello del branch GAT,
contiene:

- `generator_learning_rate`: da `1e-5` a `1e-3`, distribuzione `log_uniform_values`;
- `discriminator_learning_rate`: da `1e-5` a `1e-3`, distribuzione
  `log_uniform_values`: lo stesso intervallo di G, perche' non c'e' una
  motivazione per esplorare D su un intervallo piu' ampio. Il rate di
  riferimento delle due reti, `1e-3`, e' l'estremo superiore;
- `generator_reconstruction_weight`: da `50` a `2000`, distribuzione logaritmica;
- `discriminator_generator_update_ratio`: `"1:1"`, `"2:1"`, `"1:2"`.

I due learning rate sono indipendenti: `learning_rate` e
`discriminator_lr_ratio` non fanno parte dello spazio, per non avere due modi
di fissare lo stesso valore. Le loss non hanno termini di regolarizzazione
espliciti (Adam senza weight decay), quindi lo spazio non ne contiene. Sono
fissi anche `validation_holdout=true`, `monitoring_timestamps=32`,
`monitoring_feature_mmd_every_n_epochs=1`, `monitoring_feature_mmd_samples=1024`
e `mmd_objective_window=5`.

Sono fissi: architettura ConvGRU (`hidden_size=64`, `n_layers=2`,
`cnn_channels=32`, `cnn_layers=2`, `patch_size=3`, `kernel_size=3`),
`recent_steps=1`, `trend_steps=56`, batch di training 256 e di scoring 1024,
`time_encoding=onehot`, `normalization=minmax`, dropout 0.2 con MC Dropout a
20 campioni, BF16, 6 epoche e seed 20.
L'obiettivo e' `metric.name: validation/pca_mmd_rolling_mean` con
`metric.goal: minimize` (vedi "MMD nello spazio PCA"). Non e' uno sweep remoto
gia' creato. Prima di registrarlo servono sul server i due riferimenti
(`STGAN_PCA_REFERENCE_DIR`, `STGAN_MMD_REFERENCE_DIR`). Quindi:

```bash
# Controllo locale; una bozza incompleta produce un errore esplicito.
python scripts/create_stgan_sweep.py sweeps/stgan_bayes.draft.yaml

# Registrazione in W&B, senza avviare training.
python scripts/create_stgan_sweep.py sweeps/stgan_bayes.draft.yaml --create

# Eseguire sul server il comando wandb agent stampato dal comando precedente.
```

Il comando dell'agente passa il backend; percorsi dei dati e radice output
arrivano dalle variabili d'ambiente. I parametri campionati vengono letti da
`wandb.config`, che prevale sui valori di base. I nomi sconosciuti (inclusi
quelli specifici del GAT) vengono rifiutati. `wandb.init` associa
automaticamente la nuova run allo sweep fornito dall'agente; un avvio manuale
crea una run ordinaria nel progetto.

## Test

```bash
python -m unittest discover -s tests -p "test_stgan_wandb.py"
python -m unittest discover -s tests -p "test_stgan_gan_balance.py"
```

I test verificano learning rate effettivi di G e D, peso della ricostruzione,
CLI, resume, applicazione dei parametri assegnati dall'agente, metriche per
epoca, stato delle run fallite e protezione contro bozze incomplete. Eseguono
anche un piccolo training CPU FP32 con il vero SDK W&B in modalita' offline, senza
autenticazione o scritture remote, e il runner ERA5 condiviso.

`tests/test_stgan_monitoring.py` verifica i termini separati delle loss, le
medie di D, le metriche di validation (stabilita' dello score, discrepanza
delle feature, MMD e sua periodicita'), l'assenza di effetti del monitoraggio
sul training, i due learning rate indipendenti, il resume e i nomi W&B.
