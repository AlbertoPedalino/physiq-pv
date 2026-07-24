# PhysiQ-PV — Flusso di esecuzione (branch `feat/sde-net-paper-faithful`)

Mappa **runtime** della pipeline SDE-Net adattata al fotovoltaico. La backbone
resta BiLSTM+GAT; dinamica SDE, diffusione, ottimizzazione alternata e head
gaussiana seguono il caso di regressione di Kong, Sun e Zhang (2020).

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
      │     └─ xr.open_dataset per ciascun anno di train
      ├─ load_pvgis_year(test_year)
      │     └─ anno held-out, mai usato per fit o normalizzazione
      │
      ├─ build_datasets(train_map, test_ds, ...)          data/pvgis_dataset.py
      │     ├─ build_year_raw() per anno
      │     │     ├─ with_effective_poa()                 data/pvgis_irradiance.py
      │     │     ├─ geometria solare e clear sky (pvlib)
      │     │     ├─ POA = direct_tilted + diffuse_tilted (sempre)
      │     │     └─ kt_poa, kt_poa_std_3h, dpoa_dt, pv_lag_pvgis
      │     ├─ fit_normalization(train_raws)
      │     │     └─ statistiche calcolate soltanto sul train
      │     └─ PVGISWindowDataset(train / validation / test)
      │           └─ x: (B,N,L,F), target PV normalizzato: (B,N)
      │
      ├─ build_graph(lat, lon, max_dist_km)               model/graph_builder.py
      │     └─ haversine → edge_index, edge_weight
      │
      ├─ [se --train-normal-only]
      │     ├─ train_dataset.attach_anomaly_mask(scores)
      │     └─ train_dataset.filter_normal_only_windows()
      │           └─ rimuove fisicamente finestre con target/storia anomali
      │
      ├─ STGNN(...).to(device)                            model/st_gnn.py
      │     ├─ BiLSTMEncoder                              model/bilstm_encoder.py
      │     ├─ proj: Linear + GELU + LayerNorm
      │     ├─ GATLayer × K
      │     ├─ PaperSDEBlock                              model/sde_net.py
      │     │     ├─ VectorDrift: Linear + ReLU
      │     │     └─ ScalarDiffusion: Linear-ReLU-Linear-Sigmoid
      │     ├─ head_poa/kt_poa inclinata (loss ausiliaria attiva)
      │     └─ head_pv → (media, deviazione standard)
      │
      ├─ train_model(...)                                 training/train_loop.py
      │     ├─ opt_f = SGD(backbone + drift + head,
      │     │                  momentum=0.9, weight_decay=5e-4)
      │     ├─ opt_g = SGD(diffusion_net,
      │     │                  momentum=0.9, weight_decay=5e-4)
      │     └─ per epoca, per batch:
      │          ├─ PREDICTION / DRIFT STEP
      │          │    ├─ model(..., stochastic=True)
      │          │    │    └─ BiLSTM → proj → GAT → SDE → head
      │          │    ├─ Gaussian NLL(y; mu, sigma)
      │          │    ├─ [+ irradiance_weight · MSE(kt), se abilitata]
      │          │    ├─ backward()
      │          │    ├─ clip_grad_norm_(..., 100)
      │          │    └─ opt_f.step()
      │          │
      │          └─ DIFFUSION STEP
      │               ├─ x_ood = x + 2 · N(0,I)          (default)
      │               ├─ x0_in  = model.encode(x)        [no_grad]
      │               ├─ x0_ood = model.encode(x_ood)    [no_grad]
      │               ├─ g_in  = diffusion(x0_in)
      │               ├─ g_ood = diffusion(x0_ood)
      │               ├─ BCE(g_in,0) + BCE(g_ood,1)
      │               └─ opt_g.step()
      │
      │     ├─ sigma SDE: 0.01 → 0.5 dopo 30 epoche (default)
      │     └─ dopo epoch index 20: lr_f × 0.1; lr_g invariato
      │
      ├─ [se --sde-uncertainty] predict_sde(...)          training/uncertainty.py
      │     └─ per batch:
      │          ├─ esegue M forward stocastici
      │          │     └─ una coppia (mu_s, sigma_s) per Brownian path
      │          ├─ media = E_s[mu_s]
      │          ├─ epistemic_var = Var_s(mu_s)
      │          ├─ aleatoric_var = E_s[sigma_s²]
      │          ├─ total_var = epistemic_var + aleatoric_var
      │          └─ PI: inversione numerica della CDF
      │                F(y) = (1/M) Σ_s Φ((y-mu_s)/sigma_s)
      │
      ├─ [altrimenti] predict(..., stochastic=False)
      │     └─ sola traiettoria drift
      │
      ├─ attach_anomaly_labels(predictions, test_scores)  data/pvgis_labels.py
      ├─ build_interval/daytime/residual metrics          experiments/pvgis_stgnn_runner.py
      └─ write_outputs() + write_report()                 reporting/run_report.py
            └─ predictions.csv, metriche, metadati e report.md
```

Il notebook `notebooks/pvgis_sde_pipeline.ipynb` e gli sweep non implementano
un secondo training loop: costruiscono la CLI tramite
`physiq_pv/experiments/sde_pipeline.py`.

## Catena critica per batch

```text
PVGISWindowDataset.__getitem__
  → STGNN.encode
      → BiLSTMEncoder → proiezione → GAT
  → PaperSDEBlock
      → Euler-Maruyama con diffusione scalare g(x0)
  → head_pv(mu, sigma)
  → Gaussian NLL → clipping → opt_f
  → pseudo-OOD → BCE(g_ID=0, g_OOD=1) → opt_g
```

## Ruolo dei due file SDE

```text
physiq_pv/model/sde_net.py
  ├─ VectorDrift
  ├─ ScalarDiffusion
  ├─ PaperSDEBlock
  ├─ YearMSDSDENet       riferimento eseguibile 90 → 50 → (mu,sigma)
  └─ diffusion_bce_loss / yearmsd_nll_loss

physiq_pv/model/st_gnn.py
  ├─ GATLayer
  └─ STGNN
       ├─ backbone BiLSTM+GAT
       ├─ self.sde = PaperSDEBlock(...)
       └─ head PV e head kt
```

`sde_net.py` contiene quindi il nucleo generico e testabile di Kong;
`st_gnn.py` lo innesta nella backbone fotovoltaica.

## Il blocco SDE di Kong

```text
x0 = encoder_BiLSTM_GAT(x)                       (B,N,D)
summary = mean_pool_nodes(x0)                    (B,D)
g = sigmoid(Linear(ReLU(Linear(summary))))       (B,1)
scale = sigma · g                                (B,1)

x = x0
per k = 0..n_steps-1:
    x = x + ReLU(Linear(x))·dt
    x = x + scale·sqrt(dt)·epsilon_k
```

- La diffusione è **un solo scalare per grafo**, trasmesso a tutti i nodi e
  canali.
- `g` dipende dallo stato iniziale `x0` e resta costante lungo la traiettoria.
- Ogni step usa un incremento browniano indipendente.
- Il default usa quattro step sull’orizzonte `[0,4]`.
- Il mean-pooling è il ponte necessario fra il latent grafico `(B,N,D)` e
  l’invariante di Kong “una diffusione per esempio”.

## Training alternato

| Passo | Parametri aggiornati | Obiettivo |
|---|---|---|
| prediction/drift | BiLSTM, proiezione, GAT, drift, head | Gaussian NLL PV + MSE kt opzionale |
| diffusion | solo `diffusion_net` | BCE ID→0 + pseudo-OOD→1 |

Nel passo diffusion l’encoder è eseguito senza gradiente. Questo impedisce alla
BCE di modificare la rappresentazione usata per la previsione. La traiettoria
predittiva usa comunque il rumore SDE durante il training, come nel protocollo
di Kong.

## Incertezza e intervallo

Ogni traiettoria browniana produce una Gaussiana condizionale:

```text
p(y|x) ≈ (1/M) Σ_s Normal(y; mu_s, sigma_s²)

epistemic_var = Var_s(mu_s)
aleatoric_var = E_s(sigma_s²)
total_var     = epistemic_var + aleatoric_var
```

La decomposizione è la legge della varianza totale. L’intervallo primario al
95% non sostituisce la miscela con una singola Gaussiana: calcola i quantili
invertendo la CDF della miscela. La banda
`E[mu] ± z·sqrt(total_var)` è conservata soltanto come diagnostica
moment-matched. Il 95% è una soglia di valutazione e non influenza il training.

## Feature PVGIS di default

```text
temperature_2m, solar_irradiance_poa, wind_speed_10m,
sin_elev, cos_elev,
kt_poa, kt_poa_std_3h, dpoa_dt,
direct_irradiance_tilted, diffuse_irradiance_tilted,
pv_lag_pvgis
```

Non vengono usati Sentinel né Quality Score `m1..m5`. Gli anni di train e
l’anno di test sono separati esplicitamente.

## Fedeltà e adattamenti dichiarati

| Parte | Stato |
|---|---|
| Diffusione scalare, drift ReLU, `[0,4]`, Euler–Maruyama | Kong |
| BCE ID/OOD, due SGD, sigma schedule, clip e LR decay | Kong/repository ufficiale |
| Head eteroscedastica gaussiana `(mu,sigma)` | Kong YearMSD |
| BiLSTM+GAT al posto di `Linear(90,50)` | adattamento PV |
| Mean-pooling dei nodi prima di `g` | adattamento dimensionale necessario |
| Softplus sulla media PV | vincolo di non negatività |
| Head/loss kt opzionale | task ausiliario PV |
| Report e label anomalie | valutazione post-hoc |

Le label di anomalia non sono input del modello.
