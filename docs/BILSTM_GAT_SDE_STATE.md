# Stato BiLSTM + GAT + SDE-Net (PVGIS-only)

Branch di riferimento: `feat/sde-net-paper-faithful`.

Questo documento descrive le invarianti da mantenere quando la stessa pipeline
viene riportata negli altri branch PVGIS-only.

## Obiettivo

Previsione causale della produzione PV sintetica a `t + 1 h` usando soltanto una
finestra passata PVGIS. La BiLSTM è bidirezionale esclusivamente all'interno
della finestra storica: non vede il target né osservazioni future. Il test resta
separato da preprocessing, scelta dell'epoca e tuning.

## Irradianza e feature fisiche

- `solar_irradiance_poa` presente nel NetCDF viene ignorata sempre.
- La POA effettiva è ricostruita come:
  `direct_irradiance_tilted + diffuse_irradiance_tilted`.
- Entrambe le componenti inclinate sono obbligatorie, finite e non negative.
- La clear-sky POA è calcolata per ogni nodo con pvlib: clear sky Ineichen
  (fallback Simplified Solis) e trasposizione sul piano definito dagli attributi
  `tilt_angle` e `azimuth_angle` (fallback 30°/180°).
- `kt_poa = POA / clear_sky_POA`, non POA/GHI orizzontale.
- La head ausiliaria `head_poa` è attiva e supervisionata per default con MSE su
  `kt_poa`; il peso è configurabile con `--irradiance-loss-weight`.
- Il bound coerente di dataset e head è `--kt-poa-max` (default 1.6).

Feature canoniche:

1. `temperature_2m`
2. `solar_irradiance_poa` ricostruita
3. `wind_speed_10m`
4. `sin_elev`
5. `cos_elev`
6. `kt_poa`
7. `kt_poa_std_3h`
8. `dpoa_dt`
9. `direct_irradiance_tilted`
10. `diffuse_irradiance_tilted`
11. `pv_lag_pvgis`

## Split e preprocessing

- Servono almeno due anni in `--train-years`.
- Per default l'ultimo anno di quel gruppo è validation; si può scegliere con
  `--validation-year`.
- p99 per impianto e statistiche z-score sono calcolati soltanto sugli anni
  rimasti nel training effettivo.
- Validation e test usano le statistiche congelate del training.
- Il test è valutato soltanto dopo aver ripristinato il best checkpoint scelto
  sulla validation.
- La metrica di selezione predefinita è `rmse_daytime`; sono disponibili anche
  `mae_daytime` e `nll`.
- I target PV sono limitati inferiormente a zero ma non hanno upper clipping per
  default. `--pv-target-clip-max` esiste soltanto per ablation.
- Il clipping a 1.5 non proviene da SDE-Net. Il paper normalizza feature e target
  a media zero/deviazione standard uno; il repository YearMSD usa
  `StandardScaler` e applica solo gradient clipping con norma 100.

## Integrità temporale e spaziale

- Ogni anno deve avere timestamp unici, crescenti e distanziati esattamente di
  un'ora. Un buco temporale interrompe la costruzione del dataset.
- Gli ID location devono essere unici e coincidere esattamente tra gli anni.
- Tutti gli anni e il test sono reindicizzati nell'ordine del primo anno di
  training.
- Latitudine e longitudine vengono verificate per ID dopo il riordino.
- In questo modo normalizzazioni, target e nodi del grafo restano associati allo
  stesso impianto.

## Grafo e GAT

- Gli archi entro `--max-dist-km` usano un prior gaussiano limitato in `(0, 1]`.
- Ogni nodo isolato riceve un collegamento al vicino più prossimo.
- Sono presenti self-loop per tutti i nodi.
- Il prior geografico è aggiunto ai logit di attenzione in log-space:
  `logit + strength * log(prior)`.
- Non si moltiplica più un logit con segno per un peso di distanza: tale
  operazione poteva favorire un arco lontano quando il logit era negativo.

## Training SDE-Net preservato

Restano invariati rispetto all'implementazione paper-faithful:

- Gaussian NLL eteroschedastica per la head PV;
- aggiornamento alternato drift/backbone/head e diffusion net;
- pseudo-OOD `x + 2 * N(0, I)` configurabile;
- schedule di `sigma`;
- gradient clipping a norma 100;
- decomposizione aleatorica/epistemica tramite head gaussiana e traiettorie
  Browniane.

La validation usa il percorso deterministico (`stochastic=False`) per una
selezione stabile. Dopo ogni epoca vengono loggate le metriche validation,
applicato early stopping e conservato lo stato migliore.

## Riproducibilità e W&B

- Progetto W&B predefinito: `physiq_pv`.
- Head e loss `kt_poa` sono abilitate per default e peso/metrica sono nella
  configurazione W&B.
- `best_model.pt` contiene:
  state dict, best epoch/metrica, anni dello split, ordine e coordinate dei
  nodi, feature, p99/z-score, configurazione grafo, configurazione modello/SDE
  e argomenti di training.
- Con upload artifact attivo vengono creati un artifact report e un artifact
  W&B di tipo `model` contenente il checkpoint.

## Checklist per altri branch

1. Portare `pvgis_irradiance.py` e la definizione POA/`kt_poa`.
2. Portare validazione temporale e reindicizzazione esatta delle location.
3. Verificare che fit p99/z-score usi soltanto il training effettivo.
4. Portare il grafo gaussiano e il prior GAT in log-space.
5. Collegare `head_poa` alla loss `kt_poa`; non lasciare una head casuale.
6. Aggiungere validation, best-state restore ed early stopping prima del test.
7. Salvare e caricare tutti i metadati del checkpoint, incluso l'ordine nodi.
8. Lasciare l'upper clipping PV disabilitato salvo ablation esplicita.
9. Eseguire `tests/test_sde_net.py`, `tests/test_sde_pipeline.py` e compileall.
