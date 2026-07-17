# PhysiQ-PV — Flusso SDE-Net (branch `feat/sde-net-true-minimal`)

Mappa **runtime** dei branch SDE. **Non** è una variante di `bilstm-gat`: è una
pipeline diversa per dati, split, modello, training e inferenza.
Riferimento: Monaco et al. (2025) "SDE U-Net" / Kong et al. (2020) "SDE-Net".

## Differenze chiave vs `feat/bilstm-gat`

| Aspetto | `bilstm-gat` | `sde-net-true-minimal` |
|---|---|---|
| Entry point | `main.py` → `train.py` | CLI `pvgis_stgnn_runner.py` (argparse) + notebook orchestratore |
| Dati | Sentinel **ENERGIA reale** + meteo PVGIS | **PVGIS-only** (nessun Sentinel) |
| Target | `ENERGIA / p99` (reale) | **`pv_power_output`** (PV simulato PVGIS) |
| Split | 80/20 **mensile** su 2019 | **multi-anno**: `--train-years` vs `--test-year` held-out |
| Feature | **16** (incl. QS m1..m5) | **11** (`PVGIS_STGNN_FEATURES`, **niente QS**) |
| Modello | BiLSTM → GAT → 2 head | BiLSTM → GAT → **SDEBlock** → 2 head |
| Optimizer | 1× AdamW | **2× alternati** (`opt_f` drift, `opt_g` diffusion) |
| Loss | MSE + physics + **peak asimmetrica** | **MSE puro** (+ kt aux opz.) / **BCE** su g |
| Incertezza | ❌ nessuna (stima puntuale) | ✅ **PI da spread dei campioni SDE** |
| Anomalie | — | label + `--train-normal-only` (protocollo Monaco) |
| Output | `model.pt` + json | CSV + report (`reporting/`) |

## Albero delle chiamate

```
python -m physiq_pv.experiments.pvgis_stgnn_runner  [CLI argparse]
   (oppure notebooks/pvgis_sde_pipeline.ipynb → costruisce il comando via
    experiments/sde_pipeline.py e lo lancia in subprocess)
│
└─ run_from_args(args)                          experiments/pvgis_stgnn_runner.py
   │
   ├─ resolve_feature_set(args.feature_set)     data/pvgis_dataset.py
   │     └─ subset di PVGIS_STGNN_FEATURES (11) → n_features
   │
   ├─ load_pvgis_years(pvgis_dir, train_years)  data/pvgis_dataset.py
   │     └─ xr.open_dataset per anno → dict {anno: ds}
   │   load_pvgis_year(test_year)               → ds di test
   │
   ├─ build_datasets(train_map, test_ds, ...)   data/pvgis_dataset.py
   │     └─ build_year_raw(ds) per anno:
   │          ├─ with_effective_poa(ds)          data/pvgis_irradiance.py
   │          │     └─ se POA vuota → direct_tilted + diffuse_tilted
   │          ├─ _solar_geometry(times,lat,lon)  data/pvgis_dataset.py
   │          │     └─ pvlib get_solarposition + get_clearsky (Ineichen)
   │          ├─ [tilted presenti?] DNI/DHI reali ELSE pvlib.irradiance.erbs()
   │          └─ kt, kt_std, dghi, pv_lag_pvgis, day mask
   │     └─ fit_normalization(train_raws) → normalizzazione fittata SOLO su train
   │     └─ PVGISWindowDataset(train) / PVGISWindowDataset(test)
   │
   ├─ build_graph(lats, lons, max_dist_km)      model/graph_builder.py
   │     └─ haversine → edge_index, edge_weight
   │
   ├─ [se --train-normal-only] filtro label     data/pvgis_labels.py
   │     └─ tiene solo finestre con target E storia "normali"
   │
   ├─ STGNN(n_features, n_sde_steps, sigma_max, ...)   model/st_gnn.py [__init__]
   │     ├─ BiLSTMEncoder(...)                   model/bilstm_encoder.py
   │     ├─ proj (Linear+GELU+LN)
   │     ├─ GATLayer(...) × gat_layers           model/st_gnn.py
   │     ├─ SDEBlock(dim, n_steps=4, sigma_max)  model/st_gnn.py  ← NOVITÀ
   │     │     ├─ drift        (Linear+Tanh)×2   f(x,t)
   │     │     └─ diffusion_net (Linear+Tanh)    g(x0) → sigmoid (0,1)
   │     └─ head_ghi (opz.), head_pv
   │
   ├─ train_model(model, dataset, ...)          training/train_loop.py
   │     ├─ make_loss_fn() → MSELoss             training/losses.py
   │     ├─ build_noise_feature_indices(...)     training/noise.py
   │     ├─ opt_f = Adam(drift + encoder + GAT + heads)
   │     ├─ opt_g = Adam(sde.diffusion_net)      ← solo diffusione
   │     └─ for epoch in 1..epochs:
   │          for batch (x, y, k) in loader:
   │            ┌─ DRIFT STEP (in-distribution) ────────────────────
   │            │  pred_ghi, pred_pv = model(x, ei, ew, None, stochastic=True)
   │            │        └─ STGNN.forward()      model/st_gnn.py
   │            │             ├─ encode(): BiLSTM → proj → GAT  ⇒ x0
   │            │             ├─ SDEBlock.forward(x0, stochastic=True)
   │            │             │     └─ Euler-Maruyama × n_steps:
   │            │             │          x ← x + f(x,t)·dt + sigma_max·g·√dt·Z
   │            │             └─ head_pv → softplus ; head_ghi → kt·ghi_cs
   │            │  loss_pv = MSE(pred_pv, y)        [+ irradiance_weight·MSE(kt)]
   │            │  opt_f.zero_grad(); loss.backward(); opt_f.step()
   │            └─ DIFFUSION STEP (pseudo-OOD) ─────────────────────
   │               x_ood = inject_input_noise(x, idx, ood_noise_std)   training/noise.py
   │               x0_in  = model.encode(x,     ei, ew)    [no_grad]
   │               x0_ood = model.encode(x_ood, ei, ew)    [no_grad]
   │               g_in  = model.sde.diffusion(x0_in)
   │               g_ood = model.sde.diffusion(x0_ood)
   │               loss_g = BCE(g_in → 0) + BCE(g_ood → 1)
   │               opt_g.zero_grad(); loss_g.backward(); opt_g.step()
   │
   ├─ predict_sde(model, test_ds, mc_samples, ...)   training/uncertainty.py
   │     └─ model.eval(); per batch:
   │          for s in 1..mc_samples:
   │              mu_s = model(x, ei, ew, None, stochastic=True)[1] · pv_scale
   │          mean  = E[mu]           ← predizione
   │          std   = Std(mu)         ← spread SDE
   │          lo_pi = quantile(mu, α/2)   hi_pi = quantile(mu, 1-α/2)  ← PI empirico
   │     (predict() = variante deterministica, uncertainty.py:19)
   │
   └─ reporting                                  reporting/
         ├─ build_interval_metrics / build_daytime_metrics / residual_bias
         ├─ write_outputs(...)                   reporting/run_report.py → CSV
         └─ write_report(...)                    → report finale
```

## Il cuore: `SDEBlock` (model/st_gnn.py:76)

Sostituisce il passaggio diretto GAT→head con l'integrazione di una SDE neurale.

```python
dt = 1.0 / n_steps
g  = sigmoid(diffusion_net(x0))          # (B,N,dim) in (0,1) — dipende solo da x0
x  = x0
for k in range(n_steps):                 # Euler-Maruyama
    t = k * dt
    x = x + drift(cat([x, t])) * dt                      # deriva deterministica
    if stochastic:
        x = x + sigma_max * g * sqrt(dt) * randn_like(x) # moto browniano
return x, g
```

- **drift `f(x,t)`** → la dinamica deterministica = **la predizione**.
- **diffusion `g(x0)`** → scala il rumore browniano = **incertezza epistemica**.
  Addestrata **bassa in-distribution**, **alta out-of-distribution**.
- `Tanh` mantiene f e g Lipschitz (esistenza/unicità, Teorema 1).
- `sigma_max` limita la diffusione efficace in `(0, sigma_max)` → non esplode.

## Ottimizzazione alternata (Algorithm 1, Kong et al.)

Due obiettivi, due optimizer, **nello stesso batch**:

| Step | Optimizer | Parametri | Loss | Scopo |
|---|---|---|---|---|
| **drift** | `opt_f` | encoder + GAT + drift + heads | **MSE**(pred_pv, y) | predire bene |
| **diffusion** | `opt_g` | `sde.diffusion_net` **only** | **BCE**(g_in→0) + **BCE**(g_ood→1) | g bassa se noto, alta se ignoto |

Il **pseudo-OOD** è sintetico: `inject_input_noise(x, ood_noise_std)` — rumore
gaussiano sui canali continui (`sin_elev`/`cos_elev` esclusi, sono deterministici).

## Incertezza: come si legge

Nessun MC-dropout, nessuna head di varianza. A inferenza:
1. Si campionano `mc_samples` **traiettorie browniane** (`stochastic=True`).
2. Ogni traiettoria → una predizione `mu`.
3. **mean** = E[mu], **std** = Std(mu), **PI** = quantili empirici di `mu`.

Monaco non separa aleatoria/epistemica: la banda **è** lo spread dei campioni SDE.

## Feature (11, `PVGIS_STGNN_FEATURES`)

```
temperature_2m, solar_irradiance_poa, wind_speed_10m,
sin_elev, cos_elev,
kt, kt_std_3h, dghi_dt,
dni_norm, dhi_norm,
pv_lag_pvgis
```
⚠️ **Nessun m1..m5 (QS)** — a differenza dei 16 canali di `bilstm-gat`.
Feature-set ablation via `resolve_feature_set()`: `full`, `no_pv_lag`,
`meteo_only`, `irradiance_only`, `no_derived_irradiance`.

## Parametri CLI specifici SDE

| Flag | Default | Cosa fa |
|---|---|---|
| `--n-sde-steps` | 4 | step di Euler-Maruyama |
| `--sigma-max` | 0.5 | tetto della diffusione |
| `--ood-noise-std` | 1.0 | std del rumore pseudo-OOD |
| `--lr-g` | — | learning rate del diffusion net |
| `--sde-uncertainty` | — | attiva inferenza stocastica |
| `--mc-samples` | 20 | traiettorie browniane a inferenza |
| `--train-normal-only` | — | protocollo Monaco: train solo su finestre normali |
| `--train-years` / `--test-year` | — | split multi-anno |

## Note

- **Le label anomalie NON sono input del modello**: filtrano solo quali finestre
  entrano in training (`--train-normal-only`); il test è sugli estremi held-out.
- **`with_effective_poa`** (`data/pvgis_irradiance.py`) risolve la POA vuota del
  `.nc` derivandola da `direct_tilted + diffuse_tilted`. Presente su
  `student-t-beta-nll`, `sde-net-true-minimal`, `sde-net-paper-faithful`;
  **assente** su `true-sde-uncertainty`, `beta-nll`, `clean/sde-proxy-minimal`.
- **DNI/DHI** ora letti reali dai tilted PVGIS (Erbs solo fallback).
- **Niente peak loss / physics loss** qui: solo MSE (scelta Monaco-literal).
- Il **notebook** non duplica logica: costruisce comandi CLI via
  `experiments/sde_pipeline.py` e legge i CSV prodotti.
