# PVGIS continual-learning protocol

## Scope

Continual learning starts only after the frozen offline evaluation used in the
thesis. The 2019 test set must not be reused to tune anomaly thresholds,
trigger rules, replay composition, or update hyperparameters.

The current offline reference remains:

- MTGFlow and STGAN trained without anomaly labels;
- SDE-Net direct outputs t+1, ..., t+6;
- MTGFlow `k=1.5` and STGAN global top 1% as frozen reference decisions;
- 2019 used only for final offline evaluation.

If data after 2019 are available, the online stream begins in 2020. If only
historical data through 2019 are available, continual learning is a separate
simulation with a new chronological train/validation/stream split and must not
be compared directly with the frozen 2019 test numbers.

## Prequential cycle

For every incoming target timestamp:

1. predict all direct horizons before observing the target;
2. store issue timestamp, target timestamp, prediction and uncertainty;
3. when the target becomes available, compute residuals and detector scores;
4. report metrics before any update (test-then-train);
5. route the sample to the stable buffer or quarantine buffer;
6. update only at a scheduled boundary or after confirmed drift;
7. evaluate the candidate checkpoint on a frozen rolling validation window;
8. promote the candidate only when quality and forgetting guards both pass.

Predictions for a timestamp can never be recomputed after the model has seen
that timestamp.

## Detector routing

Detector output is not ground truth. It controls routing, not correctness:

- normal according to both detectors: eligible for the stable replay buffer;
- anomalous according to one detector: store in quarantine;
- anomalous according to both detectors: high-priority review/quarantine;
- disagreement: retain detector names and scores; do not collapse it silently.

Quarantined events are not automatically discarded. After a delay, they can be
admitted to a rare-event replay stratum when they are persistent, physically
plausible, and contain no data-quality failure. This prevents the model from
learning sensor corruption while also preventing permanent blindness to new
climate regimes.

## Replay composition

Use a bounded replay buffer stratified by:

- season/month;
- geographical cluster;
- production bin;
- normal versus reviewed rare event;
- forecast horizon.

Sampling must preserve a stable majority while reserving a small, explicit
share for reviewed rare regimes. Buffer policy and capacity are selected on
validation and then frozen for the stream evaluation.

## Drift confirmation

A single anomaly does not trigger training. Drift requires persistence across
multiple windows and evidence from at least two families:

- input drift: feature-distribution change;
- residual drift: sustained MAE/RMSE or bias change;
- spatial drift: change across several neighbouring locations;
- calibration drift: prediction-interval coverage degradation.

All drift statistics use only observations available at the current stream
time. Thresholds are calibrated on the offline validation period.

## Update and promotion

The default operational cadence is monthly, with an earlier update allowed
only after confirmed persistent drift. Each update starts from the currently
deployed checkpoint and trains on recent eligible data plus replay.

A candidate is promoted only if:

- recent-window MAE/RMSE improves or stays within the validation tolerance;
- historical replay performance does not exceed the forgetting tolerance;
- daytime and high-production performance pass their guards;
- uncertainty coverage does not materially degrade;
- normal and reviewed-rare strata both retain sufficient support.

Otherwise the deployed checkpoint is retained and the failed candidate is
archived for audit.

## Required artefacts per cycle

Every cycle writes:

- immutable pre-update predictions;
- detector scores and routing decisions;
- drift statistics and trigger decision;
- replay-buffer composition;
- candidate and deployed checkpoint identifiers;
- validation and forgetting metrics;
- promotion/rejection reason;
- code commit, configuration and random seed.

## Evaluation

Report at least:

- prequential MAE and RMSE at t+1 and t+6;
- normal/rare metrics under both detector references;
- performance by season, production bin and geographical cluster;
- backward transfer/forgetting on frozen replay slices;
- adaptation gain after confirmed drift;
- update count, rejected candidates and buffer composition.

The first implementation milestone should connect the existing replay and
quality-gated-update prototype to the direct multi-horizon SDE-Net checkpoint.
It must be developed on a dedicated branch because it changes model state,
whereas the spatial and threshold notebooks are evaluation-only.
