# Experiments

Documentazione completa di baseline, ablation e sanity check eseguiti su PhysiQ-PV. Tutti gli esperimenti loggano su W&B
(entity `albertopedalino-politecnico-di-torino`, project `PhysiQ-PV`).

## Indice

1. [Training principale ST-GNN](#1-training-principale-st-gnn)
2. [Persistence baseline (anti-naive)](#2-persistence-baseline-anti-naive)
3. [Ablation finestra temporale: L=1 vs L=24](#3-ablation-finestra-temporale-l1-vs-l24)
4. [Single-hour inference (case study)](#4-single-hour-inference-case-study)
5. [Sanity check anti-leakage (one-hour-training)](#5-sanity-check-anti-leakage-one-hour-training)
6. [Riepilogo risultati globali](#6-riepilogo-risultati-globali)

---

## 1. Training principale ST-GNN

**File**: `main.py` + `train.py`

**Scopo**: addestrare il modello ST-GNN dual-head (PatchTST + GAT) sull'intero dataset Piemonte 2019 (1116 plants × 5743 ore), con peak-aware asymmetric loss e cloud features (phaseA).

**Config corrente (`feat/peak-loss-ablation`, run 6 best)**:
- `seq_len=24, patch_len=4, stride=2, d_model=64, gat_dim=96`
- `peak_alpha=2.0, peak_gamma=2.0, peak_loss_weight=0.25, under_penalty=2.0`
- `N_FEATURES=14` (phaseA: kt thr=0.1, kt_std_3h, dghi_dt)
- 10 epoche, early stopping `patience=3`

**Comando**:
```bash
python main.py
```

**Output**:
- Checkpoint `checkpoints/model.pt` + JSON config
- W&B run `phaseA_cloud_kt01_f14_a2.0_g2.0_w0.25_u2.0`
- Metrics loggate per epoch: `train_loss`, `val_loss`, `mae_pv`, `rmse_pv`, `bias_pv` + per fascia campana (`mae_pv_60_80`, ecc.)

**Risultato atteso** (Run 6): MAE globale 2.94%, val_loss 0.0243, MAE picco (80-100%) 3.84%.

---

## 2. Persistence baseline (anti-naive)

**File**: `scripts/experiments/persistence_baseline.py`

**Scopo**: stabilire il pavimento assoluto. Nessun modello, nessun training. Predizione = copia dell'ora precedente:

```
y_pred(t) = y_true(t-1)
```

Risponde a: **"Il modello ML aggiunge valore rispetto a una regola naive?"** Se ST-GNN non batte persistence di almeno 2-3×, l'architettura non è giustificata.

**Comando**:
```bash
python scripts/experiments/persistence_baseline.py
python scripts/experiments/persistence_baseline.py --wandb
```

**CLI args utili**:
- `--year 2019`
- `--out-dir checkpoints/persistence_baseline`
- `--wandb` (log su dashboard)

**Output**:
- `checkpoints/persistence_baseline/metrics_global.json`
- `checkpoints/persistence_baseline/metrics_by_production_band.csv`
- W&B run `persistence-baseline-t-minus-1` (tag `persistence`, job_type `baseline`)

**Risultato osservato**: MAE globale **10.22%**, peggior fascia 40-60% MAE 17.25%. Il modello batte persistence di ~3.5× → giustificato.

---

## 3. Ablation finestra temporale: L=1 vs L=24

**File**: `main.py` configurato per L=1 sul branch `feat/persistence-baseline`. Training originale L=24 su `feat/peak-loss-ablation`.

**Scopo**: misurare l'impatto del contesto temporale. Si trainano due modelli identici con stessa pipeline ma `seq_len` differente.

- **L=24**: input window = 24 ore di storia (modello principale)
- **L=1**: input window = 1 ora (solo snapshot t-1)

Risponde a: **"Quanto valore aggiunge il lookback 24h vs solo 1 ora?"**

**Comando** (branch `feat/persistence-baseline`):
```bash
python main.py   # config attuale: SEQ_LEN_ABLATION = 1
```

**Output**:
- Checkpoint `checkpoints/seq_len_1/model.pt`
- W&B run `stgnn-seq-len-1-t-minus-1` (tag `ablation`, `seq_len_1`)
- L=24 baseline checkpoint resta in `checkpoints/` (non sovrascritto)

**Risultato osservato**:
- L=24 MAE globale: **2.94%**
- L=1 MAE globale: **3.17%** (+0.23pp)
- Differenza marginale → context 24h utile principalmente su regime transitorio (20-40, 40-60, 80-100), trascurabile su regime stabile

---

## 4. Single-hour inference (case study)

**File**: `scripts/experiments/single_hour_inference.py`

**Scopo**: caricare un checkpoint trainato e produrre la predizione su **un singolo timestamp** della val partition. Test puntuale per case study tesi (es. "ore 13:00 del 25 settembre 2019, cosa predice il modello su tutti 1116 plants?"). Non statisticamente significativo da solo — utile per visualizzazione.

**Comandi**:
```bash
# Default: midpoint val partition, checkpoint L=24
python scripts/experiments/single_hour_inference.py --wandb

# Timestamp specifico via indice (es. campione 500-esimo nella val)
python scripts/experiments/single_hour_inference.py --hour-index 500 --wandb

# Singolo impianto inspect dettagliato
python scripts/experiments/single_hour_inference.py --plant-index 42 --wandb

# Inferenza su checkpoint L=1 invece di L=24
python scripts/experiments/single_hour_inference.py --checkpoint-dir checkpoints/seq_len_1 --wandb
```

**CLI args**:
- `--checkpoint-dir` (default `checkpoints`, usa `checkpoints/seq_len_1` per L=1)
- `--hour-index N` (sample N nella val partition, default midpoint)
- `--plant-index P` (focus singolo impianto)
- `--out-dir` (default `checkpoints/single_hour_inference`)
- `--wandb`

**Output**:
- CSV `single_hour_<timestamp>.csv` per-plant con `actual_pv`, `pred_pv`, `err_pv`, `actual_ghi`, `pred_ghi`, `err_ghi`
- W&B summary: `pv/mae`, `pv/rmse`, `pv/bias`, `ghi/*` + Table per-plant
- Run name `single-hour-<safe_timestamp>` (tag `single-hour-inference`, `seq_len_N`)

**Caveat**: MAE su 1 ora è rumoroso (varia 5-10% con stagione/meteo). Non confondere con MAE aggregato. Utile solo per:
- Visualizzare scatter pred vs actual su 1116 plants in 1 istante
- Confrontare modelli sulla **stessa identica ora** (devi forzare `--hour-index` uguale)

---

## 5. Sanity check anti-leakage (one-hour-training)

**File**: `scripts/experiments/one_hour_training_sanity.py`

**Scopo**: validare l'integrità della pipeline. Addestra un modello fresh su **un solo sample** del training set per N step (overfit garantito), poi valuta sull'intera val partition.

Logica:
- Modello memorizza il sample (train_loss → 0)
- Su val partition (mai vista) il modello deve fallire (MAE alto)
- Se invece val MAE è simile al modello vero → **leakage sospetto**: target presente nelle feature, split sbagliato, o feature troppo informativa

Risponde a: **"La pipeline è pulita? Non c'è una scorciatoia che permette al modello di predire senza imparare?"**

**Comando base**:
```bash
python scripts/experiments/one_hour_training_sanity.py --seq-len 1 --train-steps 500 --wandb
```

**Comandi singoli** — 1 run per mese (maggio, luglio, settembre), sample daytime (h 10-15):
```bash
# Maggio
python scripts/experiments/one_hour_training_sanity.py --seq-len 1 --train-steps 500 --daytime-only --month 5 --wandb

# Luglio
python scripts/experiments/one_hour_training_sanity.py --seq-len 1 --train-steps 500 --daytime-only --month 7 --wandb

# Settembre
python scripts/experiments/one_hour_training_sanity.py --seq-len 1 --train-steps 500 --daytime-only --month 9 --wandb
```

**Comando stratificato unico** — esegue tutti e 3 in sequenza:
```bash
# Linux / bash
for m in 5 7 9; do python scripts/experiments/one_hour_training_sanity.py --seq-len 1 --train-steps 500 --daytime-only --month $m --wandb; done

# PowerShell
foreach ($m in 5,7,9) { python scripts/experiments/one_hour_training_sanity.py --seq-len 1 --train-steps 500 --daytime-only --month $m --wandb }
```

Verifica che il test fallisca in **tutti i regimi stagionali**, non solo su 1 caso isolato. Run name autogenerato: `one-hour-training-sanity-day-m05`, `-m07`, `-m09`.

**CLI args**:
- `--seq-len 1` (default; usa 24 per test L=24)
- `--train-steps 500` (overfit steps sul singolo sample)
- `--sample-index N` (opzionale, sample dal train pool; default midpoint dopo filtri)
- `--daytime-only` (filtra train pool a hours 10-15 → evita sample notturni con PV=0)
- `--month M` (filtra train pool al mese 1-12 → controllo stagionale)
- `--out-dir` (default `checkpoints/one_hour_training_sanity`)
- `--wandb`

**Perché stratificare**: la prima sanity (sample 2019-09-02 21:00, notte) ha dato MAE val 18.99% con bias -18.5% — modello ha imparato "predict 0 always" perché il sample era notturno. Test valido ma magnitude distorta da scelta sample. Forzare daytime + 3 mesi differenti rende il test robusto e replicabile.

**Output**:
- `metrics_global.json`, `metrics_by_production_band.csv`, `config.json` in `checkpoints/one_hour_training_sanity/`
- W&B run `one-hour-training-sanity` (tag `sanity`, `anti-leakage`, job_type `sanity-check`)
- Console: train loss finale + val MAE/RMSE/bias + tabella per fasce

**Interpretazione attesa**:
- Train loss singolo sample: **basso** (~0.001, overfit OK)
- Val MAE: **molto alto** (>10%, idealmente 15-20%)
- Se val MAE ≤ ~5% → **red flag**: rivedere `pv_lag`, split, target normalization

**Pipeline OK** ↔ test fallisce in validation. È disegnato per essere "rotto".

---

## 6. Riepilogo risultati globali

Run sul dataset Piemonte 2019 (1116 plants, daytime mask `solar_poa/1000 > 0.03`).

| Metodo | MAE globale | RMSE globale | bias | Note |
|---|---|---|---|---|
| **Persistence** (naive) | 10.22% | 15.02% | -2.34% | Pavimento ML-free |
| **ST-GNN L=1** | 3.17% | 6.90% | +1.33% | Solo snapshot t-1, contesto minimo |
| **ST-GNN L=24** (Run 6) | **2.94%** | 6.39% | +1.12% | Modello principale, phaseA features |

### Per fascia campana di produzione (best L=24 vs Persistence)

| Fascia | n samples | Persistence MAE | ST-GNN L=24 MAE | Δ |
|---|---|---|---|---|
| 0-20%   | 1.18M | 4.59%  | 1.45% | **-3.14** |
| 20-40%  | 450k  | 15.99% | 7.22% | **-8.77** |
| 40-60%  | 437k  | 17.25% | 8.03% | **-9.22** |
| 60-80%  | 426k  | 14.21% | 6.63% | **-7.58** |
| 80-100% | 415k  | 8.63%  | 3.84% | **-4.79** |
| >100%   | 28k   | 6.87%  | 4.19% | **-2.68** |

### Conclusioni difendibili in tesi

1. **Modello batte persistence di 3.5× globalmente** → giustifica l'architettura ML.
2. **Lookback 24h vs 1h dà solo -0.23pp** → contesto temporale lungo ha valore marginale; modello sfrutta principalmente feature istantanee (cloud + sensor + lag autoregressivo).
3. **Errore residuo concentrato in 40-60% (8.03%) e 20-40% (7.22%)** → noise transienti cloud / regime ambiguo. Ulteriori riduzioni richiedono DNI/DHI decomposition o feature meteo aggiuntive.

---

## W&B dashboard

Tutte le run sono organizzate per tag su https://wandb.ai/albertopedalino-politecnico-di-torino/PhysiQ-PV :

- `peak-tune` — training ablation peak loss
- `phaseA_cloud_kt01` — feature set cloud dynamics
- `persistence` — baseline naive
- `ablation`, `seq_len_1` — L=1 vs L=24
- `single-hour-inference` — case study puntuali
- `sanity`, `anti-leakage` — sanity check
- `month-05`, `month-07`, `month-09`, `daytime` — sanity stratificate per mese e regime

Filtra per tag per confronti diretti.
