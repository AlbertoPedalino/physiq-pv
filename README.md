# PhysiQ-PV

Sistema per forecasting di impianti fotovoltaici distribuiti basato su ST-GNN (PatchTST + GAT),
con Quality Score fisico-informato e loss pesata per qualità del sensore.

**Dataset**: 1,116 impianti PV Piemonte 2019, dati orari Sentinel/SCADA + PVGIS

---

## Struttura del repository

```
main.py          — entry point: pipeline completa end-to-end
train.py         — training ST-GNN con stratified monthly split

physiq_pv/
  data/
    sentinel_hourly_loader.py — carica CSV UPN orari, merge con PVGIS
    dataset.py                — PVDataset: normalizzazione, eta_adjusted, finestre
    load_kwp.py               — carica potenza di picco (kWp) dal registro GSE
    quality_score.py          — QS(plant, time) su 5 metriche fisiche
    synthetic_generator.py    — dataset sintetico (test/debug)

  model/
    patchtst_encoder.py  — encoder PatchTST (patch_len=4, stride=2, d_model=128)
    st_gnn.py            — ST-GNN: PatchTST + GAT (2 layer, 4 heads, gat_dim=256)
    graph_builder.py     — grafo geografico (archi <= 20 km, peso=1/dist)
    physics_loss.py      — L = L_ghi + L_pv + λ·L_physics, peso QS^0.2

  continual/
    replay_buffer.py          — ReplayBuffer (capacity=1000)
    quality_gated_update.py   — aggiornamento pesi con soglia QS

  agent/
    drift_monitor.py      — drift detection su QS(t) via KS test
    qs_clustering.py      — clustering soft-DTW su traiettorie QS
    causal_classifier.py  — MultiROCKET + Ridge: classifica causa drift
    cycle.py              — ciclo agentico (PhysiQAgent)

  uncertainty/
    mondrian_cp.py  — Mondrian CP stratificata per bande QS

  eval/
    benchmark.py  — metriche MAE/RMSE e baseline persistence

scripts/
  debug_loss.py           — debug isolato della loss function
  test_sentinel_loader.py — validazione pipeline di caricamento dati

docs/
  DATA_TYPES.md              — variabili, sorgenti, pipeline di caricamento
  TRAINING_ARCHITECTURE.md   — architettura modello, iperparametri, metriche

data/
  plant_mapping.csv            — UPN -> lat/lon, eta_base
  energy_with_coordinates.csv  — registro GSE Piemonte (kWp, coordinate)
  piedmont_pvgis_2019.nc       — riferimento PVGIS 2019 (1,149 locations)

checkpoints/
  model.pt           — state dict best val epoch
  loss_history.json  — {"train": [...], "val": [...]}
  model_config.json  — iperparametri architettura
```

---

## Pipeline (main.py)

1. **Caricamento dati reali** — 1,116 impianti Piemonte da CSV Sentinel orari + merge PVGIS
2. **Quality Score** — QS(plant, time) in [0,1], media geometrica di 5 metriche fisiche
3. **Training ST-GNN** — 20 epoche, split mensile stratificato (80/20), loss physics-informed
4. **Online loop** — disabilitato (sezione commentata in main.py, struttura pronta)

---

## Modello

```
Input (B=16, N=1116, L=24, C=5)
    ↓
PatchTST  (patch_len=4, stride=2 → 11 patch, d_model=128, channel-independent)
    ↓
Linear + GELU + LayerNorm  →  (B, N, 256)
    ↓
GAT × 2  (gat_dim=256, 4 heads, archi <= 20 km)
    ↓
Head GHI: Linear → softplus  →  pred_ghi (B, N)
Head PV:  Linear → softplus  →  pred_pv  (B, N)
```

~2.1M parametri totali.

---

## Loss

```
L = L_ghi + L_pv + 0.1 × L_physics

L_physics = mean(weight × (pred_pv / pred_ghi - eta_adjusted)²)
weight    = QS^0.2
```

`eta_adjusted[p]` = Performance Ratio stimato dai dati per ogni impianto (~0.757 media fleet).

---

## Quality Score — 5 metriche

| Metrica | Misura |
|---------|--------|
| Pearson(reale, pvgis_ref) rolling 720h | Forma profilo giornaliero |
| Bias relativo | Offset sistematico |
| Frazione NaN | Buchi temporali |
| Rapporto varianze | Sensore bloccato |
| PV/GHI vs eta(T) | Consistenza fisica termica |

QS basso → peso ridotto nella loss, retraining bloccato.

---

## Avvio

```bash
cd /home/apedalino/physiq_pv
source .venv/bin/activate
python main.py
```

Output: `checkpoints/model.pt`, `checkpoints/loss_history.json`

Documentazione dettagliata: `docs/DATA_TYPES.md`, `docs/TRAINING_ARCHITECTURE.md`
