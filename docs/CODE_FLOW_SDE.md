# PhysiQ-PV — flusso SDE Monaco su BiLSTM+GAT

Questo branch adatta la SDE U-Net di Monaco et al. (2025) alla backbone
spazio-temporale PV. La U-Net è sostituita da BiLSTM+GAT, mentre sono conservati
i due encoder paralleli, l'accoppiamento per stadio e il training ID/OOD della
diffusione.

Riferimenti:

- paper: <https://doi.org/10.1016/j.cageo.2025.105992>
- codice degli autori: <https://github.com/simone7monaco/probabilistic-rainprediction>

## Modello

Con `n_sde_steps=2`, gli stadi rispettano la backbone originale:
uno stadio temporale BiLSTM e un solo stadio GAT.

```text
input x: (B,N,L,F)
│
├─ drift temporal:     BiLSTM_f + proj_f ───────────────┐
└─ diffusion temporal: BiLSTM_g + proj_g → sigmoid g_0 ┤
                                                       └─ Brownian kick → h_0
│
├─ drift GAT_1(h_0) ───────────────────────────────────┐
└─ diffusion GAT_1(d_0) → sigmoid g_1 ────────────────┤
                                                      └─ Brownian kick → h_1
│
└─ head PV + head KT_POA supervisionata
```

Ogni gate ha forma `(B,N,gat_dim)` ed è limitata da `sigmoid`. A ogni stadio si
estrae un rumore indipendente:

```text
h_i = drift_i(h_{i-1}) + sigma_max * g_i * sqrt(dt) * epsilon_i
epsilon_i ~ N(0, I)
```

Come nel repository Monaco, `T=4`, `dt=4/(n_sde_stages+1)` e il default è
`sigma_max=0.5`. La diffusione è un encoder sequenziale separato: il suo stato
temporale alimenta il primo diffusion GAT, e così via. Non esiste più un singolo
`SDEBlock` dopo tutta la backbone.

## Training alternato

`training/train_loop.py` usa due Adam distinti:

```text
opt_f: BiLSTM_f + proj_f + drift GAT + head
       loss_f = MSE(PV) + peso * MSE(KT_POA)

opt_g: BiLSTM_g + proj_g + diffusion GAT
       loss_g = sum_i BCE(g_i(x_ID), 0)
              + sum_i BCE(g_i(x_ID + rumore), 1)
```

Il pseudo-OOD è costruito aggiungendo rumore gaussiano alle feature continue.
Con `--train-normal-only --train-anomaly-scores <csv>` le loss sono calcolate
solo sui nodi con target e propria storia normali. Un nodo raro non elimina
l'intera finestra regionale.

La validation è un anno distinto dal training effettivo; seleziona il best
checkpoint sul percorso deterministico. Il test viene eseguito solo dopo il
restore del best epoch.

## Inferenza

`training/uncertainty.py` esegue più forward con `stochastic=True`. Ogni forward
estrae un rumore nuovo a tutti gli stadi. La media delle traiettorie è la
predizione; deviazione standard e quantili empirici forniscono spread e
intervallo predittivo. `stochastic=False` esegue soltanto il percorso drift.

## Differenze inevitabili dalla SDE U-Net

- i blocchi convoluzionali sono sostituiti da un encoder temporale BiLSTM e da
  message passing GAT;
- non esistono decoder e skip connection U-Net; la head opera sullo stato GAT
  finale;
- la dimensione `(B,N,gat_dim)` resta costante tra gli stadi, invece di cambiare
  risoluzione e numero di canali tramite pooling.

Restano invece fedeli al codice Monaco: encoder diffusion parallelo e
sequenziale, gate per stadio, sigmoid per-feature, rumore indipendente per
stadio, `sigma=0.5`, BCE ID/OOD sommate, due optimizer, MSE e inferenza Monte
Carlo.
