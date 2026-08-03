# Come funziona MTGFlow

Detector non supervisionato di anomalie su serie multivariate, da Zhou et al.,
*Label-Free Multivariate Time Series Anomaly Detection* (arXiv:2312.11549v2).
Qui è implementata la variante **base** (senza estensione `MTGFlow_cluster`).

Codice: `physiq_pv/anomaly_detection/mtgflow/` (`model.py` architettura,
`pipeline.py` finestre/training/scoring, `config.py` iperparametri) e
`physiq_pv/anomaly_detection/thresholds.py` (soglie).

## 1. Idea di fondo

Non si impara "cosa è normale" da un training set pulito (approccio one-class):
si **stima la densità di probabilità di tutti i dati di training**, contaminati
inclusi. L'ipotesi è che le anomalie stiano in regioni a bassa densità, e siano
comunque una minoranza, quindi la densità stimata resta dominata dal
comportamento normale.

Conseguenze pratiche:

- il training set può contenere anomalie: non serve etichettarle né rimuoverle;
- lo score non è un errore di ricostruzione ma una **log-verosimiglianza
  negativa**: più bassa la densità, più alto lo score;
- serve un modello di densità flessibile e con verosimiglianza *esatta*: da qui
  il normalizing flow.

Il problema è che la densità di una finestra multivariata dipende sia dal tempo
sia dalle relazioni tra entità. MTGFlow risolve stimando una densità
**condizionata**: il flow non modella `P(x)` in astratto, ma `P(x | C)` dove `C`
è una condizione spazio-temporale prodotta da un grafo dinamico + RNN.

## 2. Input: finestre

Serie multivariata con `K` entità (qui 3: `solar_irradiance_poa`,
`temperature_2m`, `wind_speed_10m`) campionate ogni ora.

Normalizzazione per entità (Eq. 5), statistiche calcolate **solo sul training**:

```python
x_k = (x_k - mean(x_k)) / std(x_k)
```

Finestra scorrevole di dimensione `M = 60` ore con stride `S`:

```text
train_stride = 10   finestre usate per ottimizzare
score_stride = 10   finestre usate per calibrare le soglie e per lo score
```

Una finestra è scartata se al suo interno c'è un salto temporale irregolare
(`_regular_window_starts`: cadenza dedotta come differenza modale, prefix-sum
sugli scarti). Il tensore di batch è:

```text
x : (B, K, M, 1)     batch, entità, tempo, canale
```

Il canale è 1 perché ogni entità è una serie scalare: tensorizzazione pointwise
`input_size=1` presa dal repository ufficiale (il paper non la specifica).

Il **livello di decisione è la finestra**, non il singolo punto: una finestra è
anomala se contiene almeno un punto anomalo. Lo score della finestra viene poi
registrato sul suo istante finale (`window_end`).

## 3. Grafo dinamico via self-attention (Eq. 6--7)

Ogni entità è un nodo. La finestra di ciascun nodo (`M` valori appiattiti) viene
proiettata in query e key e confrontata a coppie:

```python
e_ij = (x_i W_Q)(x_j W_K)^T / sqrt(M)
a_ij = exp(e_ij) / sum_j exp(e_ij)        # softmax per riga
A    = [a_ij]                              # (B, K, K)
```

`A` è la matrice di adiacenza del grafo, **ricalcolata per ogni finestra**:
quindi le dipendenze tra entità evolvono nel tempo, invece di essere un DAG
statico come in GANF. In `DynamicGraphAttention` seguono un dropout 0.2 e
proiezioni lineari senza bias; la scala è `sqrt(M * input_size)`.

Nota: non c'è softmax sulle colonne né simmetrizzazione — `a_ij` quantifica il
flusso da `j` verso `i` e la riga `i` somma a 1.

## 4. Condizione spazio-temporale (§4.4, Eq. 8)

**Parte temporale.** Un LSTM a 1 layer processa ogni entità indipendentemente:

```text
(B, K, M, 1) → reshape (B*K, M, 1) → LSTM → (B*K, M, 32) → (B, K, M, 32)
```

Gli stati nascosti `H^t_k` sono il "time encoding": riassumono il passato della
finestra fino a `t` per quell'entità. I pesi dell'LSTM sono condivisi tra entità.

**Parte spaziale.** Una convoluzione su grafo mescola gli stati dei nodi secondo
`A`, sommando anche lo stato precedente dello stesso nodo (termine di storia,
ereditato da GANF):

```python
neighbours = einsum("bij,bjth->bith", A, H)   # aggregazione dai vicini
C = ReLU(neighbours @ W1 + H_shift @ W2) @ W3
# H_shift[:, :, t] = H[:, :, t-1], azzerato a t = 0
```

`C` ha forma `(B, K, M, 32)`: per ogni entità e ogni istante, un vettore che
codifica insieme *dove* si trova l'entità nel grafo corrente e *cosa* è successo
prima. È l'unico ingresso informativo del flow.

## 5. Normalizing flow entity-aware (Eq. 2--4, 9, 11)

Un normalizing flow è una trasformazione invertibile `z = f(x)` che mappa il dato
su una distribuzione target nota. Il cambio di variabile dà la densità esatta:

```text
log P_X(x) = log P_Z(f(x)) + log |det df/dx|
```

Qui la trasformazione è **condizionata** da `C` e composta da `n_blocks = 2`
blocchi affini. Con variabile scalare, shift e scala non possono dipendere dalla
variabile stessa senza rompere la bigezione: sono prodotti da una MLP che riceve
**solo** la condizione.

```python
shift, log_scale = MLP(C)                 # Linear → Tanh → Linear → Tanh → Linear
z      = (x - shift) * exp(-log_scale)
logdet = -log_scale                        # jacobiano esatto, nessuna stima
```

**Entity-aware** significa che ogni entità ha una distribuzione target diversa:

```text
Z_k = N(mu_k, I),   mu_k ~ N(0, 1)
```

`mu_k` è estratto una volta sola ed è registrato come *buffer*, non come
parametro: non viene addestrato (coerente con Eq. 9). I pesi del flow sono invece
**condivisi tra tutte le entità** (Eq. 11) — è questo che evita la crescita di
memoria con `K` pur mantenendo densità distinte per entità.

Perché serve: entità con meccanismi fisici diversi (irraggiamento, temperatura,
vento) hanno anomalie con caratteristiche di sparsità diverse. Mapparle tutte su
`N(0, I)` come fa GANF comprime queste differenze.

Log-densità di un punto e aggregazione sulla finestra:

```python
log P(x_k,t) = -0.5*(z - mu_k)^2 - 0.5*log(2*pi) + logdet
entity_log_prob[b, k] = sum_t log P(x_k,t)      # (B, K)
```

## 6. Ottimizzazione congiunta (§4.6)

Attenzione, LSTM, conditioner e flow sono addestrati **insieme** con un solo
optimizer, massimizzando la verosimiglianza media:

```text
loss = - mean_{b,k} entity_log_prob[b, k]

Adam(lr=2e-3, weight_decay=5e-4)
clip_grad_value_(params, 1.0)
epochs=40, batch_size=256, shuffle=True, window=60, n_blocks=2, hidden=32
```

Nessuna etichetta, nessun early stopping, nessun model selection: si usano
epoche fisse (`checkpoint_selection = final_fixed_epoch`). Ottimizzare tutti i
moduli insieme evita che grafo e flow convergano a ottimi locali separati.

## 7. Score di anomalia (Eq. 12, 14)

Lo score di finestra è la NLL media sulle entità; la sua decomposizione dà il
contributo di ciascuna entità:

```python
entity_score = -entity_log_prob / K        # S_ck   (B, K)
global_score = entity_score.sum(dim=1)     # S_c = sum_k S_ck   (B,)
```

`sum_k (-llk_k / K)` è esattamente la media delle NLL dell'Eq. 12: `global_score`
è lo score di finestra e `entity_score` la sua attribuzione per entità
(anomaly interpretation, Eq. 14): dice *quale* variabile ha reso la finestra
improbabile.

Score alto = densità bassa = finestra più probabilmente anomala. Lo scoring gira
in `eval()` (dropout dell'attenzione spento) e senza gradienti.

## 8. Soglie IQR (Eq. 13, 15)

Non esiste un validation set pulito su cui prendere il massimo: anche training e
validation contengono anomalie. Si usa quindi una soglia robusta calcolata sugli
score di **training**:

```text
globale     T   = Q3 + 1.5 * (Q3 - Q1)
per entità  T_k = lambda * (Q3_k + 1.5 * (Q3_k - Q1_k)),  lambda = 0.8
flag        score >= soglia
```

La soglia per entità è separata perché ogni `S_ck` ha scala propria (target
gaussiani diversi): una soglia unica assegnerebbe implicitamente pesi diversi
alle entità. `lambda` corregge il fatto che anche le osservazioni normali
fluttuano con ampiezze diverse da entità a entità.

`thresholds.py` accetta **solo score**: le etichette non compaiono nell'API,
quindi non possono contaminare la calibrazione. Le soglie stimate sul training
vengono poi applicate invariate al test.

## 9. Schema riassuntivo

```text
serie grezze (K entità, oraria)
   │  z-score per entità, statistiche training-only            Eq. 5
   ▼
finestre M=60, stride S, senza salti temporali → (B, K, M, 1)
   │
   ├─ self-attention su finestra → A (B,K,K), softmax per riga  Eq. 6--7
   ├─ LSTM per entità            → H (B,K,M,32)                 §4.4
   ▼
C = ReLU(A H W1 + H^{t-1} W2) W3   → (B,K,M,32)                 Eq. 8
   │
   ▼
MAF condizionale (2 blocchi, pesi condivisi)                    Eq. 2--4, 11
   z = (x - shift(C)) * exp(-log_scale(C)),  logdet = -log_scale
   base: N(mu_k, I), mu_k ~ N(0,1) fisso per entità             Eq. 9
   │
   ▼
entity_log_prob (B,K) = somma su M delle log-densità
   │
   ├─ training: loss = -mean(entity_log_prob)                   §4.6
   └─ scoring : S_ck = -llk/K ; S_c = sum_k S_ck                Eq. 12, 14
                soglie IQR su score di training                 Eq. 13, 15
```

## 10. Fedeltà e scelte non specificate dal paper

Policy: `paper_when_explicit_official_repo_when_underspecified` (dettagli in
`docs/MTGFLOW_PAPER_ALIGNMENT.md`).

Dal paper: z-score per entità, attenzione normalizzata per riga, condizione
`ReLU(A H W1 + H^{t-1} W2) W3`, target gaussiani per entità con parametri di flow
condivisi, MLE congiunta, score come NLL media, decomposizione per entità, soglie
IQR globale e per entità con `lambda`.

Dal repository ufficiale (dettagli lasciati aperti dal testo): tensorizzazione
pointwise `input_size=1`, hidden size 32, 1 layer interno per blocco, 2 blocchi,
weight decay, gradient clipping, seed `15--19`.

Non implementato: `MTGFlow_cluster` (Eq. 10), cioè il clustering `KShape` delle
entità con target condiviso per cluster. Con `K = 3` entità la variante base
coincide di fatto con il caso in cui ogni cluster ha cardinalità 1.
