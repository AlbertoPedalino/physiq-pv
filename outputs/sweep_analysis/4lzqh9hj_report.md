# Sweep `4lzqh9hj` - LSTM paper-style MC Dropout

Risultati eval-only dello sweep W&B. Nessun training aggiuntivo e nessuna calibrazione post-hoc.

## Setup sweep

| parametro | valore |
|---|---|
| sweep id | `4lzqh9hj` |
| model type | LSTM |
| feature set | `full` |
| target | `pv_power_output` |
| train years | 2016, 2017, 2018 |
| test year | 2019 |
| calibration | none |
| post-hoc calibration | disabled (`enable_posthoc_calibration=false`) |
| MC Dropout | enabled |
| mc_samples | 20 |
| dropout | 0.2 |
| epochs | 5 |
| batch size | 16 |
| lr | 0.001 |
| seeds | 1, 2, 3, 4, 5 |

W&B sweep container state: `RUNNING`. Run completate: **5/5**. Lo stato del container non altera il controllo sulle run, tutte `finished`.
Il valore configurativo predefinito `calibration_strategy=global` e' inattivo: non ci sono calibration years, il flag post-hoc e' falso e non sono state emesse metriche calibrate/fattori di calibrazione.

## Per-seed table

| seed | run_id | run_name | mae/global | mae/normal | mae/rare_extreme | ratio/mae_rare_normal | rmse/global | rmse/rare_extreme | picp_pi/global | picp_pi/rare_extreme | mpiw_pi/global | clc_pi/global | uncertainty/mean_std_global | uncertainty/ratio_rare_normal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4fb409bi | apricot-sweep-1 | 23.2150 | 21.9943 | 34.4550 | 1.5665 | 49.4920 | 71.8731 | 0.6669 | 0.5853 | 30.8285 | 0.6200 | 9.0131 | 1.0973 |
| 2 | b9lqndlt | volcanic-sweep-2 | 23.0750 | 21.8804 | 34.0743 | 1.5573 | 49.5247 | 72.9505 | 0.6715 | 0.5943 | 32.1095 | 0.6183 | 9.3881 | 1.1009 |
| 3 | c7v5cn6k | chocolate-sweep-3 | 21.9970 | 20.7278 | 33.6825 | 1.6250 | 49.4001 | 72.9756 | 0.6968 | 0.6196 | 32.3965 | 0.4926 | 9.4726 | 1.1008 |
| 4 | vl67moni | hearty-sweep-4 | 22.9006 | 21.4906 | 35.8832 | 1.6697 | 50.5780 | 76.1764 | 0.7009 | 0.6178 | 33.1199 | 0.4846 | 9.6831 | 1.0977 |
| 5 | 4ia0n619 | daily-sweep-5 | 22.2683 | 20.9814 | 34.1175 | 1.6261 | 50.0366 | 74.2361 | 0.6997 | 0.6223 | 32.1796 | 0.4764 | 9.4081 | 1.1037 |

## Aggregated mean/std/cv

### Point forecast metrics

| metric | mean | std | cv |
|---|---:|---:|---:|
| `mae/global` | 22.6912 | 0.5306 | 0.0234 |
| `mae/normal` | 21.4149 | 0.5518 | 0.0258 |
| `mae/rare_extreme` | 34.4425 | 0.8507 | 0.0247 |
| `ratio/mae_rare_normal` | 1.6089 | 0.0467 | 0.0290 |
| `rmse/global` | 49.8063 | 0.4979 | 0.0100 |
| `rmse/normal` | 46.4863 | 0.3333 | 0.0072 |
| `rmse/rare_extreme` | 73.6424 | 1.6453 | 0.0223 |
| `ratio/rmse_rare_normal` | 1.5841 | 0.0265 | 0.0167 |

### Uncertainty metrics

| metric | mean | std | cv |
|---|---:|---:|---:|
| `uncertainty/mean_std_global` | 9.3930 | 0.2424 | 0.0258 |
| `uncertainty/mean_std_normal` | 9.3018 | 0.2397 | 0.0258 |
| `uncertainty/mean_std_rare_extreme` | 10.2328 | 0.2684 | 0.0262 |
| `uncertainty/ratio_rare_normal` | 1.1001 | 0.0026 | 0.0024 |

### PI metrics

| metric | mean | std | cv |
|---|---:|---:|---:|
| `picp_pi/global` | 0.6872 | 0.0165 | 0.0241 |
| `picp_pi/normal` | 0.6958 | 0.0165 | 0.0238 |
| `picp_pi/rare_extreme` | 0.6078 | 0.0169 | 0.0278 |
| `mpiw_pi/global` | 32.1268 | 0.8288 | 0.0258 |
| `mpiw_pi/normal` | 31.8148 | 0.8194 | 0.0258 |
| `mpiw_pi/rare_extreme` | 34.9997 | 0.9183 | 0.0262 |
| `nmpil_pi/global` | 0.0360 | 0.0009 | 0.0258 |
| `nmpil_pi/normal` | 0.0356 | 0.0009 | 0.0258 |
| `nmpil_pi/rare_extreme` | 0.0392 | 0.0010 | 0.0262 |
| `clc_pi/global` | 0.5384 | 0.0740 | 0.1374 |
| `clc_pi/normal` | 0.4921 | 0.0671 | 0.1365 |
| `clc_pi/rare_extreme` | 1.2492 | 0.1821 | 0.1458 |

### Gaussian diagnostic metrics

| metric | mean | std | cv |
|---|---:|---:|---:|
| `picp_gaussian/global` | 0.7094 | 0.0134 | 0.0189 |
| `clc_gaussian/global` | 0.5008 | 0.0526 | 0.1051 |

## Interpretation

### Risposte alle domande

1. **Casi normal:** LSTM e' peggiore dello ST-GNN. `mae/normal` = 21.4149 contro circa 19.22 W, differenza +2.1949 W (11.4%).
2. **Casi rare/extreme:** LSTM e' peggiore. `mae/rare_extreme` = 34.4425 contro circa 30.74 W, differenza +3.7025 W (12.0%).
3. **Ratio rare/normal:** resta sostanzialmente invariato: 1.6089 contro circa 1.60 (delta +0.0089). La difficolta' relativa dei casi rari non dipende in modo evidente dal grafo.
4. **Dispersione MC Dropout:** LSTM produce molta piu' dispersione: mean std globale 9.3930 W contro circa 0.9 W (10.4x); MPIW = 32.1268 W contro circa 3.16 W (10.2x).
5. **PICP:** migliora nettamente rispetto allo ST-GNN: 0.6872 contro circa 0.028 (delta +0.6592). Resta sotto il target 0.95, quindi gli intervalli LSTM sono ancora under-covered.
6. **Origine del problema:** l'under-coverage resta anche senza grafo, quindi non e' esclusivamente un problema ST-GNN. Tuttavia il salto di dispersione e PICP mostra che l'architettura ST-GNN aggrava fortemente il collasso dell'incertezza; la LSTM lo attenua, pagando una point accuracy peggiore.

### Confronto sintetico

Le colonne PICP/MPIW confrontano intervalli empirici MC Dropout (`pi`) con intervalli ensemble; sono diagnostiche qualitative perche' il meccanismo di costruzione non e' identico.

| modello | mae/global | mae/normal | mae/rare_extreme | ratio mae r/n | PICP global | PICP rare | MPIW global | mean std global | std ratio r/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM MC Dropout `4lzqh9hj` | 22.6912 | 21.4149 | 34.4425 | 1.6089 | 0.6872 | 0.6078 | 32.1268 | 9.3930 | 1.1001 |
| ST-GNN MC Dropout full `7c8llckm` | 20.34 | 19.22 | 30.74 | 1.60 | 0.028 | n/a | 3.16 | 0.9 | n/a |
| Deep Ensemble full `vfry1hgx` | 19.1925 | 18.0754 | 29.4780 | 1.6308 | 0.1726 | 0.1917 | 15.1574 | 5.7790 | 1.3676 |
| Deep Ensemble no-pv-lag `02jeasjt` | 19.3293 | 18.2378 | 29.3792 | 1.6109 | 0.1971 | 0.2207 | 18.3331 | 7.0306 | 1.3302 |

La LSTM e' meno accurata di entrambi i Deep Ensemble e dello ST-GNN sul point forecast. In compenso genera intervalli molto piu' larghi e una PICP molto superiore a tutti e tre i riferimenti, ma ancora non affidabile rispetto al target 0.95. Il ratio MAE rare/normal resta nello stesso intervallo dei riferimenti, mentre il ratio della std LSTM e' piu' vicino a 1: la dispersione cresce poco sui casi rare/extreme rispetto ai Deep Ensemble.

## Controlli

- Run attese/completate: **5/5**, seed 1-5, tutte `finished`.
- Metriche richieste mancanti: **nessuna**.
- Metriche `*_calibrated`: **assenti**.
- `k_global`, `k_normal`, `k_rare_extreme` e `calibration/factor_*`: **assenti**.
- Protocollo paper-style: confermato da intervalli `picp_pi/mpiw_pi/nmpil_pi/clc_pi`, flag post-hoc falso, nessun calibration year e assenza del flag CLI `--enable-posthoc-calibration`.
- Nessun nuovo esperimento avviato.
