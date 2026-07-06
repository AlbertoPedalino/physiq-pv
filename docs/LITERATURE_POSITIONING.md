# Literature Positioning

Posizionamento di PhysiQ-PV rispetto alla letteratura su PV forecasting, data quality e physics-informed.

## Idea Centrale

Il contributo non è una architettura più complessa, ma combinare:

```text
data-centric (QS multi-componente)
+ physics-informed (clear-sky kt + L_physics multiplicativo + eta WLS)
```

su flotta **reale eterogenea** con dati meteo Open-Meteo, senza sky imagery.

## Run corrente vs SOTA

| Method | n plants | Region | Horizon | nMAE PV | Data | Note |
|---|---|---|---|---|---|---|
| Hasnat GAT 2025 | 1146 | USA | 30-min | 2.0% | synthetic NREL + lagged | best fleet-scale |
| Hasnat GAT 2025 | 1146 | USA | 3 h | 3.7% | synthetic + lagged | |
| Hasnat GAT 2025 | 1146 | USA | day | 4.4% | synthetic + lagged | |
| ST-GNN Fourier 2025 | multi | varied | 1-step | RMSE 0.027 (norm) | varied | |
| GCLSTM/GCTrafo 2021 | 304+1000 | CH | 6 h | ~6-8% est | real+sim, PV-only | |
| RTI-Net 2025 | 1 | varied | 5-15 min | 1-3% | sky images | not scalable |
| Holt-Winters | 1 | varied | short | 3.7% | persistence | classical |
| **PhysiQ-PV (run corrente)** | **1116** | **Piemonte** | **1 h** | **4.98%** | **real + lagged + meteo Open-Meteo** | **branch feat/openmeteo-training** |

Run corrente (outlier filter **disabilitato**, full fleet, n=3,061,186 daytime samples, 10 epoche):

- PV: MAE=0.0498, RMSE=0.0842, r=0.967, bias=+0.0048
- GHI: MAE=0.0564, RMSE=0.0830, r=0.952, bias=-0.0111
- Best val epoch 9 (val_loss=0.0253, train drop -68.2%)
- Per-plant time series r≈0.98 su plant 0/500/1115

## Riferimenti vicini

### Hasnat, Asadi, Alemazkoor (2025) — Tier A

*A graph attention network framework for generalized-horizon multi-plant solar power generation forecasting using heterogeneous data*  
Renewable Energy 243 (2025) 122520. DOI: 10.1016/j.renene.2025.122520.

GAT su 1146 plant USA, NREL synthetic 2024 (5-min downsampled), lagged power autoregressive primario, day-ahead MAE 4.4%.

**Differenza con PhysiQ-PV:**
- Hasnat: synthetic dataset (no degradation, no sensor noise) + lagged PV power primario
- PhysiQ-PV: real degraded fleet + lagged + meteo Open-Meteo + QS data-centric

Confronto **non apples-to-apples**: rimuovere lagged AR o passare a real noisy data degraderebbe i numeri di Hasnat.

### Sundararajan et al. (2022) — data quality-aware

*A Data Quality-Aware Framework to Reliably Forecast Photovoltaic Generation and Consumer Load for an Improved Resilience of Microgrids*  
IEEE PEDG 2022 / ORNL.

Framework data-quality-aware per attivare classi diverse di modelli. **Differenza:** ORNL usa qualità del dato per **selezionare strategia**. PhysiQ-PV la incorpora **direttamente** come 5 feature m1..m5 e come diagnostica post-hoc.

### Yu, Loskot, Gao (2026) — physics-guided

*PhysEmbedFormer: a physics-guided interpretable architecture for days-ahead forecasting of PV power*  
Scientific Reports 16, 4705.

Lavora su **architettura** scomposta physics-guided. **Differenza:** PhysiQ-PV usa QS fisicamente interpretabile come **segnale operativo** integrato nel training e diagnostica, non come decomposizione architetturale.

## Formulazione del contributo

```text
A differenza di approcci data-quality-aware che usano la qualità del dato
per selezionare strategie di forecasting differenti, e di approcci
physics-guided che lavorano sulla decomposizione architetturale, PhysiQ-PV
integra un Quality Score multi-componente (m1..m5) direttamente come
feature input, e usa QS aggregato come segnale diagnostico per analisi
post-hoc per fascia di qualità e per plant.

Il modello base è una ST-GNN con vincolo fisico moltiplicativo sul rapporto
PV/GHI e parametrizzazione clear-sky index sulla testa GHI.

La validazione è su flotta reale eterogenea (1116 impianti Piemonte 2019,
dati meteo Open-Meteo), non su dataset sintetici.
```

## Claim difendibile

```text
PhysiQ-PV si distingue per la combinazione operativa di:
  (a) Quality Score multi-componente fisicamente interpretabile,
      integrato come feature direttamente nell'input modello;
  (b) vincolo fisico moltiplicativo sul rapporto PV/GHI con
      parametrizzazione clear-sky sulla testa GHI;
validati su flotta reale eterogenea con dati meteo Open-Meteo.

Il contributo non è SOTA accuracy: il MAE PV 5.05% non batte Hasnat 4.4%
day-ahead, ma il confronto è apples-to-oranges (synthetic NREL vs real
degraded fleet). Il contributo è il framework data-centric +
physics-informed operativamente deployable su flotta reale.
```

## Claim da evitare

```text
"Nessuno ha mai usato la qualità del dato nel PV forecasting."
"PhysiQ-PV batte SOTA su nMAE."
"Il QS migliora l'accuratezza."
```

**Motivi:**
- Letteratura data-quality-aware esiste già (ORNL 2022).
- Confronto Hasnat è apples-to-oranges.
- Con `pv_lag` autoregressive on, la correlazione QS↔MAE è collassata a +0.056 (era −0.166): QS non predice più l'errore — segnale lagged satura.

## Posizionamento prudente

- NON "rivoluzionario": miglioramento solido ma non enorme su MAE.
- Confronto interno alla stessa architettura.
- No benchmark esteso vs SOTA su dataset pubblici equivalenti.
- Lagged power è il vero driver del MAE 5.05%.

## Reframe risultato sperimentale

**Vecchia ablation (senza lagged power):**
- QS baseline: MAE 0.0872
- No-QS base: MAE 0.0960
- Mid-low QS bin actual ≥0.8: MAE −64.5% (0.367 → 0.130)

**Run corrente (con lagged + clearsky kt + L_physics multiplicativo, outlier filter disabilitato):**
- MAE 0.0498, RMSE 0.0842, r 0.967, bias +0.0048
- Mid-low QS bin (quantile p5–p25, n=612,236): MAE 0.0564
- Correlazione QS_raw↔MAE bin (hardcoded thresholds): +0.181 (atteso negativo, contaminato da sparsity + soglie arbitrarie)
- Correlazione QS_shrunk↔MAE bin (quantile data-driven `[5, 25, 75, 95]`): **−0.956** (monotonia quasi perfetta)
- Combinazione vincente: shrinkage bayesiano data-driven (prior, n0, scale fittati dai dati) + bin quantile-based + asymmetric=False

**Run precedente (stesso setup, outlier filter ATTIVO):** MAE 0.0505, mid-low MAE 0.0805. Disabilitare filter migliora marginalmente il MAE e mantiene la validazione sulla full fleet.

**Lettura:** lagged power domina. m1..m5 contribuiscono marginalmente al MAE puro nel batch training. Il QS resta utile per diagnostica data-quality e analisi errori, non come claim di accuracy.

## Posizionamento finale tesi

> Il contributo principale di PhysiQ-PV non è proporre un'architettura più
> complessa, ma mostrare che, in scenari reali multi-impianto, la
> combinazione di vincoli fisici espliciti (clear-sky kt + L_physics
> multiplicativo + eta WLS), feature data-centric multi-componente
> (m1..m5 separati come canali input), e segnale autoregressive (pv_lag) è
> più efficace e interpretabile di un puro aumento della capacità del
> modello, ed è operativamente deployable su flotta reale eterogenea con
> dati meteo pubblici.

Vedi `SOTA_REFERENCES.md` per survey completa della letteratura.
