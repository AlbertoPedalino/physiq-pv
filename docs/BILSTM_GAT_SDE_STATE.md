# Stato BiLSTM + GAT + SDE-Net true-minimal (PVGIS-only)

Branch di riferimento: `feat/sde-net-true-minimal`.

## Protocollo

- Forecast causale della produzione PVGIS a `t + 1 h` da 24 ore passate.
- BiLSTM bidirezionale soltanto dentro la finestra storica.
- Training effettivo: 2016-2017; validation: 2018; test: 2019.
- p99 e z-score sono stimati soltanto sui timestamp usati dalle finestre
  normali del training effettivo.
- La validation usa `stochastic=False`, seleziona il best epoch ed effettua
  early stopping. Il test viene eseguito una volta dopo il restore.
- Il target PV è limitato inferiormente a zero e non ha upper clipping per
  default.

## Irradianza

- Il campo NetCDF `solar_irradiance_poa` viene ignorato sempre.
- `POA = direct_irradiance_tilted + diffuse_irradiance_tilted`.
- Piccoli residui negativi delle componenti, entro 20 W/m², vengono portati a
  zero; negatività più grandi o valori non finiti interrompono la pipeline.
- La clear-sky POA viene trasposta sul piano inclinato usando tilt e azimuth
  del dataset, con fallback 30°/180°.
- `kt_poa = POA / clear_sky_POA_inclinata`, limitato al bound condiviso
  `kt_poa_max=1.6`.
- `head_poa` e la MSE ausiliaria su `kt_poa` sono attive nel notebook con peso
  0.1.

Feature:

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

## Integrità spaziale e temporale

- I timestamp devono essere unici, crescenti e distanziati esattamente di
  un'ora.
- Gli ID location devono coincidere fra gli anni.
- Validation e test vengono reindicizzati nell'ordine del primo anno di
  training.
- Latitudine e longitudine vengono verificate dopo il riordino.
- Il grafo usa un prior gaussiano limitato in `(0, 1]`, self-loop e un
  nearest-neighbour per ogni nodo altrimenti isolato.
- Il GAT aggiunge `edge_prior_strength * log(prior)` ai logit.

## Normal-only

Le climatology labels non sono feature. Per ottenere l'analogo PV del
protocollo `non-intense`/`intense` di Monaco:

1. per timestamp e variabile, i punteggi locali sono aggregati sul grafo con
   il q99 spaziale di `|anomaly_score|`;
2. per ogni variabile, la soglia regionale q0.975 è stimata soltanto su
   2016-2017;
3. un timestamp è raro se supera almeno una soglia;
4. una finestra viene rimossa fisicamente se il target, una delle 24 ore di
   input o le due ore precedenti necessarie a `kt_poa_std_3h` è rara.

Train 2016-2017 e validation 2018 contengono quindi soltanto eventi regionali
normali e conservano sempre tutti i nodi. Non servono loss mask, maschere degli
archi o grafi dinamici. Il test 2019 resta completo e viene valutato sia per
`event_group` regionale sia tramite le etichette locali per cella.

Il file aggregato 2016-2018 viene usato per classificare entrambi gli split,
ma le soglie regionali e le normalizzazioni sono stimate soltanto sul training
effettivo 2016-2017.

## SDE true-minimal preservata

Questo branch non usa Gaussian NLL o Beta-NLL:

- head PV puntuale con `softplus`;
- MSE PV;
- encoder drift e diffusion BiLSTM+GAT paralleli;
- due stadi SDE: temporale BiLSTM e spaziale GAT;
- due optimizer Adam;
- BCE per stadio, `g(ID) -> 0` e `g(pseudo-OOD) -> 1`;
- pseudo-OOD ottenuto aggiungendo rumore alle feature continue;
- in inferenza, media, deviazione standard e intervalli derivano dalle
  traiettorie Browniane.

Queste scelte distinguono intenzionalmente il branch
`feat/sde-net-true-minimal` dal branch `feat/sde-net-paper-faithful`.

## Riproducibilità

`best_model.pt` salva:

- state dict del best epoch;
- metrica e score di validation;
- anni dello split;
- ordine, latitudine e longitudine dei nodi;
- feature e normalizzazioni;
- grafo e prior geografico;
- configurazione BiLSTM/GAT/SDE e argomenti del training.

Il progetto W&B predefinito è `physiq_pv`.
