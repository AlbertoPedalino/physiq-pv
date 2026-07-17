# PhysiQ-PV — Flusso di esecuzione (branch `feat/bilstm-gat`)

Mappa **runtime** (chi chiama chi), non gli import. Ordine reale di esecuzione
partendo da `python main.py`. Riferimenti `file:funzione`.

## Albero delle chiamate

```
python main.py
│
└─ main()                                          main.py
   │
   ├─ load_sentinel_hourly()                       data/sentinel_hourly_loader.py
   │     └─ pd.read_csv (CSV UPN) → xr.Dataset ENERGIA (plant × time)
   │
   ├─ merge_with_weather(ds, pvgis_path)           data/sentinel_hourly_loader.py
   │     ├─ xr.open_dataset(pvgis .nc)
   │     ├─ scipy cdist() → nearest PVGIS location per impianto
   │     ├─ reindex(method='nearest') → temperature_2m, solar_irradiance_poa, wind_speed_10m
   │     ├─ [se presenti] estrae direct_irradiance_tilted + diffuse_irradiance_tilted
   │     └─ (fallback, solo se manca il .nc) pvlib.Location.get_clearsky
   │
   ├─ _normalize_dataset(ds)                        main.py
   │     └─ rename lat/lon, POA obbligatoria, wind fallback = 3.0
   │
   ├─ load_kwp()                                    data/load_kwp.py → kwp
   │
   └─ for SEED in [42, 123, 2024]:
        │
        └─ train(ds, kwp, seed, ...)                train.py
           │
           ├─ _set_global_seed(seed)                train.py
           │
           ├─ compute_qs(ds, debug=True)            data/quality_score.py
           │     └─ rolling corr / bias / nan / std / eta → m1..m5 (m_components)
           │
           ├─ build_graph(lats, lons, 20km)         model/graph_builder.py
           │     └─ haversine_km() per coppia → edge_index, edge_weight (1/dist)
           │
           ├─ PVDataset(ds, m_components, ...)       data/dataset.py  [__init__]
           │     ├─ _solar_geometry_and_clearsky()   data/dataset.py
           │     │     └─ pvlib.Location.get_solarposition + get_clearsky (Ineichen)
           │     │        → sin/cos elev, ghi_cs, zenith
           │     ├─ [direct/diffuse_tilted presenti?] lettura reale
           │     │        ELSE pvlib.irradiance.erbs()  (fallback stima)
           │     └─ costruisce feats (16 canali), target_pv, valid_starts
           │
           ├─ Subset(train) / Subset(val)  +  DataLoader × 2   (split mensile 80/20)
           │
           ├─ STGNN(...).to(device)                 model/st_gnn.py  [__init__]
           │     ├─ BiLSTMEncoder(...)               model/bilstm_encoder.py  (nn.LSTM + attn pooling)
           │     ├─ proj                             Linear + GELU + LayerNorm
           │     ├─ GATLayer(...) × gat_layers       model/st_gnn.py  (default 1; vuoto se use_gat=False)
           │     └─ head_ghi, head_pv                MLP dual-head
           │     (tutti i pesi inizializzati random)
           │
           ├─ AdamW(model.parameters())             lr=1e-3, wd=1e-4
           │
           └─ for epoca in 1..15 (early-stop patience=5):
                │
                ├─ _train_epoch(...)                train.py
                │    └─ per batch:
                │         ├─ DataLoader → PVDataset.__getitem__()   data/dataset.py
                │         │      └─ slice feats[t-24:t], target[t]
                │         ├─ noise ±5% su feature meteo (ch 0-2)
                │         ├─ model(x, ei, ew, ghi_cs)  STGNN.forward()   model/st_gnn.py
                │         │      ├─ BiLSTMEncoder.forward()          model/bilstm_encoder.py
                │         │      │     └─ nn.LSTM + attention pooling
                │         │      ├─ proj (Linear + GELU + LayerNorm)
                │         │      ├─ GATLayer.forward() × 1            model/st_gnn.py
                │         │      │     └─ edge-softmax sparso (scatter_reduce/scatter_add)
                │         │      └─ head_ghi (→ kt·ghi_cs), head_pv (softplus)
                │         ├─ physics_loss_full(...)   model/physics_loss.py
                │         │      └─ MSE(ghi) + MSE(pv) + lam·MSE(pv, eta·ghi)
                │         ├─ _asymmetric_peak_loss(...)  train.py  (under-pred × under_penalty)
                │         └─ loss.backward() + optimizer.step()
                │
                └─ _val_epoch(...)                  train.py
                     └─ model.forward() (no noise) + MAE/RMSE/bias + per-bin
        │
        ├─ model.load_state_dict(best_state)        (ripristina pesi miglior epoca val)
        └─ return model, loss_history, val_loss_history, edge_index, edge_weight, best_val_epoch
   │
   └─ main: torch.save(model.pt)
            + json (loss_history, model_config, training_config)
            + summary multi-seed
```

## Catena critica per-step (hot path, ripetuta ogni batch)

```
DataLoader → PVDataset.__getitem__ → STGNN.forward
   → BiLSTMEncoder.forward → GATLayer.forward → head_pv / head_ghi
   → physics_loss_full + _asymmetric_peak_loss → loss.backward()
```

## Foglie di calcolo (librerie esterne)

| Funzione | Libreria | Dove |
|---|---|---|
| `get_solarposition`, `get_clearsky` (Ineichen) | pvlib | `dataset.py` (+ fallback loader) |
| `irradiance.erbs` | pvlib | `dataset.py` (solo fallback DNI/DHI) |
| `cdist` (match PVGIS) | scipy | `sentinel_hourly_loader.py` |
| `nn.LSTM` | torch | `bilstm_encoder.py` |
| GAT (edge-softmax) scritto a mano (`scatter_add`/`scatter_reduce`) | torch | `st_gnn.py` |

## Note

- **DNI/DHI (ch 14-15)**: ora letti reali da `direct/diffuse_irradiance_tilted` del
  PVGIS; Erbs solo fallback quando i tilted mancano.
- **Calibrazione post-hoc**: rimossa (niente più `postprocessing.py` /
  `_fit_pv_linear_calibration`). Il modello restituito è già il best-val.
- **QS aggregato**: `compute_qs` calcola anche il QS composito (`_qs_da`), ma
  `train()` usa solo `m_components` (m1..m5); l'aggregato è scartato.
- **Loss deterministica**: MSE + peak asimmetrico. Nessun termine probabilistico
  su questo branch (Student-t / Beta-NLL sono sui branch fratelli).
