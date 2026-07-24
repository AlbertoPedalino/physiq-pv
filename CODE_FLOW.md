# PhysiQ-PV — Flusso di esecuzione (branch `feat/sde-net-true-minimal`)

Mappa **runtime** dell’adattamento della SDE U-Net di Monaco et al. (2025) alla
backbone BiLSTM+GAT. La U-Net non viene conservata; vengono conservati il doppio
encoder drift/diffusion, l’allineamento della diffusione a ogni livello, il
training ID/OOD e il campionamento Monte Carlo.

## Albero delle chiamate

```text
python -m physiq_pv.experiments.pvgis_stgnn_runner [argomenti CLI]
│
└─ main()                                                experiments/pvgis_stgnn_runner.py
   └─ run_from_args(args)
      │
      ├─ _validate(args)
      ├─ resolve_feature_set(feature_set)                data/pvgis_dataset.py
      │
      ├─ load_pvgis_years(train_years)                   data/pvgis_dataset.py
      │     └─ xr.open_dataset per ciascun anno di training
      ├─ load_pvgis_year(test_year)
      │     └─ anno held-out
      │
      ├─ build_datasets(train_map, test_ds, ...)          data/pvgis_dataset.py
      │     ├─ riallinea ID/coordinate dei nodi fra tutti gli anni
      │     ├─ separa train effettivo / validation / test
      │     ├─ build_year_raw() per anno
      │     │     ├─ with_effective_poa()                 data/pvgis_irradiance.py
      │     │     │     └─ POA = direct_tilted + diffuse_tilted (sempre)
      │     │     ├─ geometria solare + clear-sky POA inclinata (pvlib)
      │     │     ├─ verifica griglia oraria continua
      │     │     └─ kt_poa, kt_poa_std_3h, dpoa_dt, pv_lag_pvgis
      │     ├─ [se --train-normal-only]
      │     │     ├─ build_regional_event_protocol(scores)
      │     │     │     └─ q99 spaziale → soglie q0.975 fit su train
      │     │     ├─ rimuove finestre rare da train e validation
      │     │     └─ seleziona i timestamp normali usati dal training
      │     ├─ fit_normalization(train_raws, normal_time_masks)
      │     │     └─ fit soltanto sui timestamp normali del training effettivo
      │     └─ PVGISWindowDataset(train / validation / test)
      │           └─ x: (B,N,L,F), target PV normalizzato: (B,N)
      │
      ├─ build_graph(lat, lon, max_dist_km)               model/graph_builder.py
      │     └─ haversine → prior gaussiano, self-loop, fallback isolati
      │
      ├─ STGNN(...).to(device)                            model/st_gnn.py
      │     │
      │     ├─ PERCORSO DRIFT
      │     │     ├─ BiLSTM_f + proj_f
      │     │     ├─ temporal_refine + residual
      │     │     └─ GAT_f × 1                            (default)
      │     │
      │     ├─ PERCORSO DIFFUSION
      │     │     └─ MonacoDiffusionEncoder
      │     │           ├─ BiLSTM_g + proj_g → sigmoid g_0
      │     │           └─ GAT_g × 1 → sigmoid g_1
      │     │
      │     └─ head_pv puntuale + head_poa/kt_poa
      │
      ├─ train_model(train, validation, ...)              training/train_loop.py
      │     ├─ opt_f = Adam(drift BiLSTM/GAT + head)
      │     ├─ opt_g = Adam(diffusion BiLSTM/GAT)
      │     └─ per epoca, per batch:
      │          ├─ PREDICTION / DRIFT STEP
      │          │    ├─ model(..., stochastic=True)
      │          │    │    ├─ calcola g_0,g_1
      │          │    │    ├─ inserisce un Brownian kick a ogni livello
      │          │    │    └─ head_pv → previsione puntuale
      │          │    ├─ MSE(pred_pv, y)
      │          │    ├─ [+ irradiance_weight · MSE(kt_poa)]
      │          │    └─ opt_f.step()
      │          │
      │          └─ DIFFUSION STEP
      │               ├─ x_ood = inject_input_noise(x)
      │               │    └─ rumore sulle feature continue,
      │               │       esclusi sin_elev/cos_elev
      │               ├─ g_in_terms  = model.diffusion(x)
      │               ├─ g_ood_terms = model.diffusion(x_ood)
      │               ├─ loss_g = Σ_i BCE(g_i,in,0) + BCE(g_i,ood,1)
      │               └─ opt_g.step()
      │     └─ validation deterministica, early stopping e restore best epoch
      │
      ├─ [se --sde-uncertainty] predict_sde(...)          training/uncertainty.py
      │     └─ per batch:
      │          ├─ ripeti M volte model(..., stochastic=True)
      │          │     └─ ogni forward genera rumore nuovo a ogni stadio
      │          ├─ mean = media delle M previsioni
      │          ├─ std  = deviazione standard delle M previsioni
      │          └─ PI   = quantili empirici delle M previsioni
      │
      ├─ [altrimenti] predict(..., stochastic=False)
      │     └─ solo percorso drift, senza Brownian kick
      │
      ├─ attach_anomaly_labels(predictions, test_scores)  data/pvgis_labels.py
      ├─ build_interval/daytime/residual metrics          experiments/pvgis_stgnn_runner.py
      └─ write_outputs() + write_report()                 reporting/run_report.py
            └─ predictions.csv, metriche, best_model.pt e report.md
```

Il notebook e gli sweep costruiscono la stessa CLI tramite
`physiq_pv/experiments/sde_pipeline.py`; non contengono un modello alternativo.

## Catena critica per batch

```text
PVGISWindowDataset.__getitem__
  ├─ MonacoDiffusionEncoder
  │    → BiLSTM_g → g_0 → GAT_g1 → g_1
  └─ drift encoder
       → BiLSTM_f + Brownian(g_0)
       → GAT_f1 + Brownian(g_1)
       → head PV → MSE → opt_f

x e x_ood → diffusion encoder → Σ BCE per livello → opt_g
```

## SDE anche sulla BiLSTM

Sì. Il primo livello SDE è quello temporale:

```text
drift temporal:     BiLSTM_f + proj_f ───────────────┐
diffusion temporal: BiLSTM_g + proj_g → sigmoid g_0 ┤
                                                    └─ Brownian kick su h_0
```

I livelli successivi accoppiano un GAT drift e un GAT diffusion. Con i default:

```text
n_sde_steps = 2
gat_layers  = 1
stadi SDE   = 1 temporale + 1 spaziale
```

Il costruttore rifiuta configurazioni in cui
`n_sde_steps != 1 + numero_GAT_attivi`.

## Diffusione a livelli diversi

Non esiste un unico `SDEBlock` dopo la backbone. Ogni livello possiede il
proprio gate:

```text
g_i(x): (B,N,gat_dim), valori in (0,1)

h_i = drift_i(h_(i-1))
      + sigma_max · sqrt(dt) · g_i(x) · epsilon_i
epsilon_i ~ N(0,I)
```

- `g_0` viene dal ramo BiLSTM diffusion;
- `g_1` viene dall'unico GAT diffusion;
- ogni `g_i` ha la dimensione dello stato del livello associato;
- ogni livello estrae un rumore indipendente;
- lo stato diffusion non viene ricavato dal gate sigmoid precedente: prosegue
  nel suo encoder sequenziale parallelo, come nel codice Monaco.

In questa backbone `gat_dim` resta costante, quindi i gate hanno la stessa
forma. Il concetto “adattato alla dimensione del livello” rimane comunque
esplicito nell’accoppiamento uno-a-uno fra blocchi drift e diffusion.

## Dove si trova l’implementazione

Tutto il meccanismo Monaco adattato è in `physiq_pv/model/st_gnn.py`:

```text
GATLayer
MonacoDiffusionEncoder
STGNN
```

Su questo branch non serve `physiq_pv/model/sde_net.py`: non viene usato il
blocco vettoriale post-backbone di Kong. Il ramo diffusion è parte strutturale
della ST-GNN e accompagna la BiLSTM e ogni GAT.

## Training alternato Monaco

| Passo | Optimizer | Parametri | Loss |
|---|---|---|---|
| drift | Adam `opt_f` | BiLSTM/GAT drift + head | MSE PV + MSE kt opzionale |
| diffusion | Adam `opt_g` | BiLSTM/GAT diffusion | somma BCE per stadio |

Il pseudo-OOD è ottenuto perturbando le feature continue. Non vengono applicati
il clipping a `100` o il decadimento del solo `lr_f` del protocollo YearMSD di
Kong: non fanno parte del training pubblico Monaco seguito da questo branch.

## Inferenza e significato dell’intervallo

Il modello ha una sola head PV puntuale. Non produce una `sigma` aleatorica:

```text
mu_s = previsione della traiettoria browniana s
prediction = E_s[mu_s]
uncertainty = Std_s(mu_s)
PI = quantili empirici {mu_1, ..., mu_M}
```

L’intervallo descrive quindi lo spread delle traiettorie SDE. Non esiste qui la
decomposizione `epistemica + aleatorica` dei branch con head probabilistica. Il
target di copertura (default 95%) sceglie soltanto i quantili e le metriche di
valutazione; non entra nella loss.

## Feature PVGIS di default

```text
temperature_2m, solar_irradiance_poa, wind_speed_10m,
sin_elev, cos_elev,
kt_poa, kt_poa_std_3h, dpoa_dt,
direct_irradiance_tilted, diffuse_irradiance_tilted,
pv_lag_pvgis
```

Non vengono usati Sentinel o Quality Score `m1..m5`. Gli anni di train e
l’anno di test sono separati.

## Cosa resta letterale e cosa cambia rispetto a Monaco

| Parte | Stato |
|---|---|
| Encoder drift/diffusion separati e paralleli | Monaco |
| Gate sigmoid per feature a ogni livello | Monaco |
| Rumore indipendente per livello, `T=4`, `sigma=0.5` | Monaco |
| BCE ID/OOD sommata su tutti i livelli, due Adam, MSE | Monaco |
| BiLSTM+GAT al posto dei blocchi convoluzionali U-Net | adattamento PV |
| Nessun decoder e nessuna skip connection encoder-decoder | differenza di backbone |
| Risoluzione e `gat_dim` costanti fra i livelli | differenza di backbone |
| Softplus della previsione PV e head kt_poa supervisionata | adattamento PV |

Le label di anomalia sono usate per filtraggio/valutazione e non come feature.
