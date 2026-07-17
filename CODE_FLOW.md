# PhysiQ-PV — Flusso di esecuzione (branch `feat/student-t-beta-nll`)

Mappa **runtime** (chi chiama chi), non semplice elenco degli import. Il flusso
parte dalla CLI PVGIS; il notebook e gli sweep costruiscono la stessa chiamata
tramite `experiments/sde_pipeline.py`.

## Albero delle chiamate

```text
python -m physiq_pv.experiments.pvgis_stgnn_runner [argomenti CLI]
│
└─ main()                                                experiments/pvgis_stgnn_runner.py
   └─ run_from_args(args)
      │
      ├─ _validate(args)
      ├─ resolve_feature_set(feature_set)                data/pvgis_dataset.py
      │     └─ seleziona i canali PVGIS usati dal modello
      │
      ├─ load_pvgis_years(train_years)                   data/pvgis_dataset.py
      │     └─ xr.open_dataset per ciascun anno di training
      ├─ load_pvgis_year(test_year)
      │     └─ anno held-out usato soltanto per la valutazione
      │
      ├─ build_datasets(train_map, test_ds, ...)          data/pvgis_dataset.py
      │     ├─ build_year_raw() per ogni anno
      │     │     ├─ with_effective_poa()                 data/pvgis_irradiance.py
      │     │     │     └─ POA oppure direct_tilted + diffuse_tilted
      │     │     ├─ geometria solare + clear sky (pvlib)
      │     │     ├─ DNI/DHI tilted reali; Erbs solo come fallback
      │     │     └─ kt, kt_std_3h, dghi_dt, pv_lag_pvgis, maschera giorno
      │     ├─ fit_normalization(train_raws)
      │     │     └─ statistiche stimate esclusivamente sugli anni di train
      │     └─ PVGISWindowDataset(train) / PVGISWindowDataset(test)
      │           └─ x: (B,N,L,F), y PV normalizzato: (B,N)
      │
      ├─ build_graph(lat, lon, max_dist_km)               model/graph_builder.py
      │     └─ haversine → edge_index, edge_weight = 1 / distanza
      │
      ├─ [se --train-normal-only]
      │     ├─ load_anomaly_scores(train CSV)
      │     └─ train_dataset.attach_anomaly_mask(...)
      │           └─ esclude dalla task loss celle con target/storia anomali
      │
      ├─ STGNN(...).to(device)                            model/st_gnn.py
      │     ├─ BiLSTMEncoder                              model/bilstm_encoder.py
      │     ├─ proj: Linear + GELU + LayerNorm
      │     ├─ GATLayer × K
      │     ├─ PaperSDEBlock                              model/sde_net.py
      │     │     ├─ VectorDrift: Linear + ReLU
      │     │     └─ ScalarDiffusion: MLP + sigmoid
      │     ├─ head_ghi/kt opzionale
      │     └─ head_pv → (mu, sigma)
      │
      ├─ train_model(...)                                 training/train_loop.py
      │     ├─ opt_f = SGD(backbone + drift + head)
      │     ├─ opt_g = SGD(diffusion_net)
      │     └─ per epoca, per batch:
      │          ├─ PREDICTION / DRIFT STEP
      │          │    ├─ model(..., stochastic=True)
      │          │    │    └─ BiLSTM → proj → GAT → PaperSDEBlock → head
      │          │    ├─ Student-t beta-NLL(mu, sigma, nu, beta)
      │          │    │    └─ oppure Gaussian beta-NLL con --nll-dist gaussian
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
      │          ├─ ripeti M volte model(..., stochastic=True)
      │          │     └─ ottieni (mu_s, sigma_s) per traiettoria browniana
      │          ├─ media predittiva = E_s[mu_s]
      │          ├─ var epistemica = Var_s(mu_s)
      │          ├─ var aleatorica =
      │          │     E_s[sigma_s² · nu/(nu-2)]          (Student-t)
      │          ├─ var totale = epistemica + aleatorica
      │          └─ PI = quantili empirici della miscela completa:
      │                y_(s,r) ~ StudentT(nu, mu_s, sigma_s)
      │
      │     [con --nll-dist gaussian]
      │          └─ PI = inversione numerica della CDF della miscela gaussiana;
      │                la banda moment-matched resta solo diagnostica
      │
      ├─ [altrimenti] predict(..., stochastic=False)
      │     └─ sola traiettoria drift, senza campionamento browniano
      │
      ├─ attach_anomaly_labels(predictions, test_scores)  data/pvgis_labels.py
      ├─ build_interval/daytime/residual metrics          experiments/pvgis_stgnn_runner.py
      └─ write_outputs() + write_report()                 reporting/run_report.py
            └─ predictions.csv, metriche, metadati e report.md
```

## Catena critica di training

```text
PVGISWindowDataset.__getitem__
  → STGNN.encode
      → BiLSTMEncoder → proiezione → GAT
  → PaperSDEBlock
      → drift Euler-Maruyama + rumore scalato da g(x0)
  → head_pv(mu, sigma)
  → Student-t beta-NLL
  → clip gradienti → opt_f
  → pseudo-OOD → BCE diffusione → opt_g
```

## Il blocco SDE fedele a Kong

`PaperSDEBlock` è definito in `physiq_pv/model/sde_net.py` e viene istanziato
da `STGNN` in `physiq_pv/model/st_gnn.py`.

```text
x0 = BiLSTM + GAT latent                         (B,N,D)
g  = sigmoid(MLP(mean_pool_nodes(x0)))           (B,1)
scale = sigma_global · g                         (B,1)

x = x0
per k = 0..n_steps-1:
    x = x + f(x)·dt + scale·sqrt(dt)·epsilon_k
    epsilon_k ~ N(0,I)
```

Proprietà conservate dal codice YearMSD di Kong:

- `g(x0)` è **uno scalare per esempio/grafo**, non uno per nodo o feature;
- lo stesso valore di diffusione scala tutti gli incrementi della traiettoria;
- `g` è addestrata separatamente come discriminatore ID/pseudo-OOD;
- orizzonte temporale `[0,4]`, Euler–Maruyama e `sigma=0.5` finali;
- due SGD con momentum `0.9` e weight decay `5e-4`;
- clipping predittivo a `100` e decadimento del solo `lr_f`.

Il mean-pooling dei nodi è l’adattamento necessario per trasformare il latent
grafico `(B,N,D)` nell’unico vettore `(B,D)` richiesto dalla diffusione scalare
per esempio. Non modifica la backbone BiLSTM+GAT.

## Student-t e beta-NLL

Questa è l’estensione del branch rispetto alla versione Kong-gaussiana:

```text
head_pv(h) → mu = softplus(raw_mu)
             sigma = softplus(raw_sigma) + 1e-3

y | traiettoria s ~ StudentT(nu, mu_s, sigma_s)
loss = Student-t NLL pesata con beta
```

- `nu > 2` è fisso e configurabile (`5` di default);
- `beta=0.5` è il default; `beta=0` torna alla NLL ordinaria;
- il peso beta è trattato senza gradiente, come nella correzione beta-NLL;
- la modifica riguarda la likelihood aleatorica, non il blocco SDE di Kong.

## Decomposizione e intervallo predittivo

Con `M` traiettorie browniane:

```text
epistemic_var = Var_s(mu_s)
aleatoric_var = E_s[sigma_s² · nu/(nu-2)]
total_var     = epistemic_var + aleatoric_var
```

La somma segue la legge della varianza totale. L’intervallo primario non
approssima la miscela con una singola Student-t: campiona più valori
condizionali per ciascuna traiettoria e prende direttamente i quantili della
miscela. Il target di copertura (default `0.95`) è una soglia di valutazione e
non entra nella loss di training.

## Feature PVGIS di default

```text
temperature_2m, solar_irradiance_poa, wind_speed_10m,
sin_elev, cos_elev,
kt, kt_std_3h, dghi_dt,
dni_norm, dhi_norm,
pv_lag_pvgis
```

Non vengono usati Sentinel o i Quality Score `m1..m5`. Lo split è multi-anno:
gli anni indicati da `--train-years` addestrano il modello e `--test-year`
rimane held-out.

## Cosa è Kong e cosa è adattamento PV

| Parte | Provenienza |
|---|---|
| Diffusione scalare, Euler–Maruyama, training ID/OOD, due optimizer | Kong et al. |
| BiLSTM + GAT e mean-pooling dei nodi prima di `g` | adattamento PV |
| Softplus sulla media PV e head ausiliaria kt | vincoli/task PV |
| Head Student-t e beta-NLL | estensione di questo branch |
| Quantili della miscela Student-t | inferenza coerente con l’estensione |
| Label anomalie e report stratificato | valutazione post-hoc |

Le label di anomalia non sono feature del modello e il target di copertura non
influenza l’ottimizzazione.
