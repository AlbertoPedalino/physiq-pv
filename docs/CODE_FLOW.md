# Flusso BiLSTM + GAT

Questa è la catena runtime della configurazione POA v2.

```text
main.py
  │
  ├─ load_sentinel_hourly
  │    └─ timestamp SCADA Europe/Rome -> UTC
  │
  ├─ merge_with_weather
  │    ├─ match geografico impianto -> punto PVGIS
  │    ├─ allineamento orario con tolleranza 31 minuti
  │    └─ POA = direct_tilted + diffuse_tilted
  │
  └─ train
       ├─ split cronologico train/validation
       ├─ compute_qs(fit_time_mask=train)
       ├─ PVDataset(fit_time_mask=train)
       ├─ build_graph
       ├─ BiLSTMEncoder -> GATLayer -> head_poa/head_pv
       ├─ loss supervisionata + fisica + picchi
       └─ selezione checkpoint su rmse_pv_day
```

## Per-batch

`PVDataset.__getitem__` restituisce:

```text
x, y_poa, y_pv, pr_proxy, poa_cs, poa_scale
```

Il percorso caldo è:

```text
x [B,N,24,16]
  -> BiLSTM per nodo
  -> proiezione a 96 dimensioni
  -> GAT a 4 head
  -> pred_poa, pred_pv
  -> physics_loss_full + peak loss
```

La GAT usa archi geografici entro 20 km e collega al vicino più prossimo gli
eventuali nodi isolati. Il prior gaussiano della distanza viene sommato ai
logit di attenzione in log-spazio.

## Confine causale

- L’input di un target al tempo `t` è soltanto `[t-seq_len, t)`.
- `pv_lag` termina a `t-1`.
- Le rolling feature terminano nel timestamp dell’input e non usano il target.
- La BiLSTM è bidirezionale solo all’interno della finestra passata.
- Fit di z-score, scale, capacity score e PR usa soltanto il train.

## Branch di ablazione

Nel branch senza POA, il loader continua a costruire POA perché rimane target
ausiliario e serve a definire metriche giorno/notte. `PVDataset` azzera però
gli indici elencati in `POA_INPUT_INDICES`. Forma degli input, grafo, modello,
loss e split restano identici.

Per la specifica completa vedere
[`BILSTM_GAT_STATE.md`](BILSTM_GAT_STATE.md).
