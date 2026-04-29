# PhysiQ-PV

Sistema agentico per forecasting e diagnosi di impianti fotovoltaici distribuiti,
basato su un Quality Score fisico-informato derivato dal confronto tra produzione
reale (PV Sentinel) e riferimento teorico (PVGIS).

---

## Struttura del repository

```
main.py          — entry point: esegue la pipeline completa end-to-end
train.py         — training ST-GNN con DER++ quality-gated
online_loop.py   — ciclo agentico online (finestre scorrevoli)

physiq_pv/
  data/
    quality_score.py       — calcolo QS(t) su 5 metriche fisiche
    synthetic_generator.py — dataset sintetico con 4 fault iniettati
    pvgis_loader.py        — caricamento dati PVGIS (NetCDF)
    sentinel_loader.py     — caricamento dati PV Sentinel (CSV)
    upn_mapping.py         — mapping UPN -> coordinate geografiche

  model/
    patchtst_encoder.py    — encoder PatchTST (patch_len=16, d_model=128)
    st_gnn.py              — ST-GNN: PatchTST + GAT (2 layer, 4 heads)
    graph_builder.py       — grafo geografico (archi <= 50km, peso=1/dist)
    physics_loss.py        — L = L_ghi + L_pv + lambda*L_physics, peso QS^0.2

  continual/
    replay_buffer.py       — buffer DER++ per regressione (capacity=2000)
    quality_gated_update.py — aggiornamento pesi solo se QS > soglia

  agent/
    drift_monitor.py       — rilevamento drift su QS(t) via KS test
    qs_clustering.py       — clustering soft-DTW su traiettorie QS settimanali
    causal_classifier.py   — MultiROCKET + Ridge: classifica causa del drift
    cycle.py               — ciclo agentico completo (PhysiQAgent)

  uncertainty/
    mondrian_cp.py         — Mondrian CP stratificata per bande QS

  eval/
    benchmark.py           — metriche MAE/RMSE e baseline persistence
```

---

## Pipeline (main.py)

1. **Generazione dati sintetici** — 20 impianti, 8760 timestep, 4 fault iniettati
2. **Quality Score** — QS(plant, time) in [0,1], media geometrica di 5 metriche
3. **Training ST-GNN** — PatchTST per encoding temporale + GAT per propagazione spaziale, loss physics-informed e quality-weighted, DER++ per continual learning
4. **Loop agentico online** — finestre scorrevoli (720h), ad ogni step:
   - Perception: calcolo QS sulla finestra
   - Planning: drift detection + clustering + diagnosi causale (MultiROCKET)
   - Action: retraining quality-gated se drift rilevato e QS > 0.5
   - Reflection: confronto loss pre/post retraining

---

## Quality Score — 5 metriche

| Metrica | Cosa misura |
|---|---|
| Pearson(reale, PVGIS) | Forma del profilo giornaliero |
| Bias relativo | Offset sistematico |
| Frazione NaN | Buchi temporali |
| Rapporto varianze | Rumore anomalo |
| PV/GHI vs eta(T) | Consistenza fisica con efficienza termica |

QS basso -> intervalli CP piu ampi, peso ridotto nella loss, retraining bloccato.

---

## Swap dati sintetici -> reali

Modificare solo:
- `pvgis_loader.py` — puntare ai file NetCDF reali
- `sentinel_loader.py` — puntare ai CSV energy data
- `upn_mapping.py` — caricare il mapping UPN -> lat/lon

Tutte le interfacce del modello rimangono invariate.
