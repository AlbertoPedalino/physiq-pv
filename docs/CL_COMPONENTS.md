# CL_COMPONENTS.md

Architectural and technical reference for the continual / adaptive PV forecasting pipeline of PhysiQ-PV. This document is designed to be readable end-to-end without opening the source code or the dataset.

---

## 1. Overview

PhysiQ-PV is a continual and adaptive photovoltaic forecasting system built around a physics-informed spatio-temporal graph neural network (PatchTST/BiLSTM encoder + GAT spatial layer + dual GHI/PV heads). The forecasting model is trained once, offline, on a temporally bounded slice of the data, and is then deployed into a **streaming loop** that monitors data quality, detects drift, and decides — on a window-by-window basis — whether to update the model on new observations.

This streaming loop is organised as a **workflow-based ATSF implementation**. The Agentic Time Series Forecasting paradigm proposed by Cheng et al. (2026) is used here as an **architectural framework**: it provides a vocabulary and a five-block decomposition (perception, planning, action, reflection, memory) that maps cleanly to the components a continual forecaster needs in deployment. ATSF is **not** used as an LLM-driven autonomous agent and **not** as a reinforcement-learning policy. Both are perfectly compatible with the ATSF framework but neither is required by it. A deterministic orchestrator that composes explicit modules according to the same five-block decomposition is, in our reading of the paper, a legitimate "workflow-based" instantiation of ATSF.

Two concrete consequences follow from this interpretation.

- **The offline training phase is not part of the ATSF cycle.** It serves as initialisation: it produces the model state that the streaming loop will then maintain. The five ATSF blocks are exercised during the online phase, where new windows arrive sequentially and the orchestrator iterates perception → planning → action → reflection → memory.
- **A probabilistic action policy is not, by itself, reinforcement learning.** The current policy uses a softmax over hand-tuned linear utilities. The softmax is a deterministic transformation of state into a probability distribution; the weights are not updated from any reward signal. Calling the resulting choice "agentic" in the ATSF sense refers to the *structural role* of the planning block, not to autonomy or learning.

The rest of this document describes the pipeline at this level of granularity, lists the claims it can and cannot support, and records the known limitations and the work that is explicitly out of scope for this iteration.

---

## 2. Conceptual Background: ATSF

Cheng et al. (2026) argue that purely model-centric time-series forecasting is insufficient once the model is deployed in a non-stationary environment, and they propose a five-block decomposition shared with the broader agentic-AI literature.

- **Perception** is the block that turns raw data into structured signals the rest of the system can reason about: parsing, feature engineering, normalisation, completeness checks, quality scoring, and any other diagnostic that summarises the state of incoming data.
- **Planning** is the block that decides what the system should do given the perceived state: it consumes the perception signals, optional drift / uncertainty indicators, and any auxiliary diagnoses (such as fault attribution), and selects an action.
- **Action** is the block that actually changes the world or the model: it includes generating a forecast, updating model parameters, calibrating an auxiliary head, or invoking an external service.
- **Reflection** is the block that evaluates the consequences of an action: it inspects pre- and post-action metrics, compares against historical references, and can in principle veto an action that turns out to be harmful.
- **Memory** is the block that persists information across time: a replay buffer, an experience log, a calibration state, learned drift statistics, model checkpoints, structured reports.

In our pipeline these five blocks are implemented as **explicit software modules**, not as components of an autonomous agent. There is no LLM in the loop, no reinforcement-learning policy, and no meta-cognitive reasoning. The planning block is a parametric policy with hand-tuned weights; the reflection block is a metric-based check with a rollback gate; the memory block is a small set of files and in-memory data structures. None of this contradicts ATSF as an architectural framework, but it is important to keep three distinctions clean.

- **Probabilistic ≠ learned.** A softmax-over-utility policy yields a distribution over actions, but the weights of that utility are fixed by hand. The system is therefore probabilistic in its decision-making but not adaptive in its policy.
- **ATSF workflow ≠ RL.** Reinforcement learning is a specific way to make the policy adaptive (by feeding back a reward). It is not required for an ATSF instantiation. Our pipeline is rule-based and remains a valid workflow-based ATSF implementation.
- **ATSF workflow ≠ LLM agent.** The literature often instantiates ATSF with an LLM reasoning loop, but the paradigm itself is agnostic to the implementation substrate. A deterministic Python orchestrator that respects the five-block separation also qualifies.

---

## 3. Current Scope

This iteration of PhysiQ-PV concentrates on **synthetic / study mode** end-to-end. The objective is to deliver a complete, readable, piece-by-piece studiable architecture, not to publish empirical results on the real Piedmont 2019 fleet or to claim that any individual mechanism reduces forgetting in practice. Empirical validation is deferred to a later iteration.

The following elements are inside the current scope.

- The full orchestrator (`scripts/run_cl_experiment.py`) and the modules it composes.
- A synthetic dataset generator (`physiq_pv.data.synthetic_generator`) with four injected fault scenarios.
- The walk-forward split into train / calibration / holdout / stream subsets.
- The offline initial training routine (`train.train`).
- The QS-stratified Mondrian split conformal calibration (`MondriaNCP`).
- The five-component Quality Score and its soft forensics drill-down.
- The rule-based utility action policy (`UtilityActionPolicy`) and the agent that drives it (`PhysiQAgent`).
- The quality-gated online updater (`QualityGatedUpdater`) with a DER++ replay term.
- A snapshot / restore rollback when an update degrades the loss.
- Walk-forward continual-learning metrics (BWT, FWT, Average Forgetting) and a frozen-holdout retention curve.
- A reproducible per-task JSONL streaming log and a final JSON report.

The following elements are explicitly out of scope.

- Validation on the real Piedmont 2019 fleet. The loaders exist (`load_sentinel_hourly`, `merge_with_weather`, `load_kwp`) and the CLI flag `--data real` is plumbed, but real-data behaviour is not assessed here.
- Real-data causal labelling. The fault classifier is trained on labels hard-coded from the synthetic generator; on real fleets the rule-based fallback applies.
- Any form of policy learning: no reinforcement learning, no contextual bandit, no LLM-based reasoning.
- Adaptive or rolling online conformal recalibration; the conformal predictor is calibrated once at the start of the streaming loop.
- A controlled empirical comparison of DER++ against a no-replay baseline. DER++ is implemented as a mechanism; its effectiveness on this pipeline is not yet demonstrated.

---

## 4. ATSF Mapping to the Codebase

The table below pairs each ATSF block with the modules that implement it. The discursive subsections that follow describe each block in more depth.

| ATSF block | Role in the pipeline | Current implementation | Status | Notes |
|------------|----------------------|------------------------|--------|-------|
| **Perception** | Ingest PV + weather data, build features, score per-sample quality, partition time. | `physiq_pv/data/{synthetic_generator,sentinel_hourly_loader,load_kwp,dataset,quality_score}.py`, `physiq_pv/eval/streaming_protocol.py`, `physiq_pv/agent/qs_forensics.py`, `physiq_pv/model/graph_builder.py`. | Implemented. | Real loaders are present but unused in this iteration. QS forensics is soft and threshold-free. |
| **Planning** | Detect drift, diagnose faults, decide whether to update. | `physiq_pv/agent/{cycle,drift_monitor,qs_clustering,causal_classifier,qs_forensics,action_policy}.py`. | Implemented for the online loop. | Drift detection is KS-based (not true incremental ADWIN). Policy is rule-based and not learned. |
| **Action** | Train initially, forecast, perform online updates. | `train.py`, `online_loop.py`, `physiq_pv/continual/quality_gated_update.py`, `physiq_pv/model/{st_gnn,physics_loss}.py`. | Implemented. | Online updates fire only when the policy selects `trigger_update` or `--force_update` is set. |
| **Reflection** | Compare loss before / after each update, roll back if degraded, track CL metrics, monitor holdout retention. | `physiq_pv/agent/cycle.py::reflect`, `scripts/run_cl_experiment.py`, `physiq_pv/eval/cl_metrics.py`. | Implemented. | Reflection is metric-based, not meta-cognitive. Numerical significance of BWT / FWT / AF depends on updates actually firing. |
| **Memory** | Replay buffer, model checkpoints, conformal calibration state, per-plant drift state, persistent logs. | `physiq_pv/continual/replay_buffer.py`, `physiq_pv/uncertainty/{mondrian_cp,conformal_runtime}.py`, `outputs/cl_experiment/<run>/`. | Implemented. | DER++ replay distils the PV head only. CP is calibrated once offline. Per-plant drift state is not serialised across orchestrator restarts. |

### Perception

**Purpose.**
Transforms raw PV, weather and synthetic inputs into structured model features and quality signals that the rest of the pipeline consumes. Also produces the walk-forward temporal partition that underpins continual-learning evaluation.

**Main files.**
- `physiq_pv/data/dataset.py`
- `physiq_pv/data/quality_score.py`
- `physiq_pv/data/synthetic_generator.py`
- `physiq_pv/data/sentinel_hourly_loader.py`
- `physiq_pv/eval/streaming_protocol.py`
- `physiq_pv/agent/qs_forensics.py`

**Main outputs.**
- 16-channel feature tensor (`PVDataset`).
- Quality Score and m1..m5 components.
- Walk-forward train / calibration / holdout / stream split.
- Per-window perception signals fed to Planning.

The perception block is responsible for transforming raw multivariate inputs (energy readings, meteorological variables, plant metadata) into the structured signals that the rest of the pipeline consumes. The core feature builder is `PVDataset`, which assembles a per-(plant, time) tensor with 16 channels: three meteorological variables (`temperature_2m`, `solar_irradiance_poa`, `wind_speed_10m`), two solar-geometry channels (`sin_solar_elev`, `cos_solar_elev`), the five Quality Score components (`m1`, `m2`, `m3`, `m4`, `m5`), the lagged normalised PV target (`pv_lag`), and four cloud-dynamics channels (`kt`, `kt_std_3h`, `dghi_dt`) plus the Erbs decomposition (`dni_norm`, `dhi_norm`). The lagged target channel is sliced strictly up to `t-1` so that no target leakage is introduced into the encoder.

The most important diagnostic signal produced by perception is the **Quality Score**. Each component m1..m5 captures a distinct way in which a measurement can be untrustworthy: correlation with an irradiance reference (`m1`), bias against an expected baseline (`m2`), completeness (`m3`), variance ratio versus an expected envelope (`m4`), and coherence with the temperature-dependent efficiency profile (`m5`). The aggregate QS is the geometric mean of m1..m5, so a single low component already pulls the aggregate down. A subsequent `apply_qs_shrinkage` step smoothes the score with a data-driven Bayesian prior, reducing the impact of statistical noise on plants with sparse reliable samples.

A second perception output is the **walk-forward split** (`WalkForwardSplit`), which partitions the time axis into four temporally disjoint regions: an initial `train_ds` slice for offline supervised training, a small `calibration_ds` slice reserved for the conformal predictor, a `holdout_ds` slice that is never touched again and serves as a frozen retention reference, and a `stream_ds` region that the orchestrator scans as a sequence of sliding task windows.

### Planning

**Purpose.**
Reads the perception signals at every streaming window, runs drift and fault diagnostics, and decides whether (and how) the model should be updated. Active during the online loop; the offline training phase is not driven by Planning.

**Main files.**
- `physiq_pv/agent/cycle.py`
- `physiq_pv/agent/drift_monitor.py`
- `physiq_pv/agent/qs_forensics.py`
- `physiq_pv/agent/qs_clustering.py`
- `physiq_pv/agent/causal_classifier.py`
- `physiq_pv/agent/action_policy.py`

**Main outputs.**
- Per-plant drift flag and per-component suspicion scores.
- Soft mode label (`auto` / `conservative` / `uncertain`).
- Fault hypothesis from the causal classifier.
- `PolicyState` and the selected action (rule-based, hand-tuned utility).

The planning block lives inside `PhysiQAgent`. For every streaming window the agent runs three diagnostics in parallel and combines them into a `PolicyState`.

The first diagnostic is **drift detection** via `KSDriftMonitor` (historically aliased `ADWINDriftMonitor`). Each plant has its own monitor with a circular buffer of length `2 * window_size`; the buffer is split into an "older" and a "newer" half and a two-sample Kolmogorov-Smirnov test is applied. Drift is reported when the test rejects the null hypothesis at the `significance = 1e-3` level. This is an **approximation of ADWIN-style window comparison**, not the true incremental ADWIN of Bifet & Gavaldà (2007); the trade-off is documented in the module docstring.

The second diagnostic is the **QS forensics drill-down** (`forensic_report`). For each plant and each component m1..m5 it fuses two soft signals: a fleet-relative percentile and a self-baseline z-score versus the plant's own rolling history. The fusion yields a per-component "suspicion" in `[0, 1]`. Each plant is then assigned the component with the highest suspicion, a `dominance` score that measures how clearly one component stands out, and a mode label that summarises diagnostic confidence: `auto` when the dominant component is unambiguous, `conservative` when it is plausible but not strong, and `uncertain` when no component clearly dominates. The forensics block also maps each dominant component to a physical hypothesis (m1 → shape distortion, m2 → bias drift, m3 → missing data, m4 → noise anomaly, m5 → physics violation) and suggests an action family.

The third diagnostic is the **causal classifier** (`CausalClassifier`), a MultiROCKET + RidgeClassifierCV pipeline that classifies a QS time series into one of five fault categories (normal, gradual degradation, sensor failure, regional cloud event, soiling cycle). When the classifier has not been fitted, or when the input is too sparse, the system falls back to a rule-based heuristic (slope, recent-vs-early mean, 720-hour autocorrelation).

The `PolicyState` collected from these diagnostics is the input to the **utility action policy**. The policy uses a softmax over hand-tuned linear utilities (see Section 6 for details). The temperature of the softmax depends on the diagnostic mode: it is higher when the mode is `uncertain`, so the policy explores more in low-confidence regimes; it is lower when the mode is `auto`, so the policy mostly exploits.

### Action

**Purpose.**
Produces the model that the streaming loop maintains (offline phase) and applies online updates when Planning decides so (online phase). Includes the quality-gated DER++ updater and the forward pass used for forecasting.

**Main files.**
- `train.py`
- `online_loop.py`
- `physiq_pv/continual/quality_gated_update.py`
- `physiq_pv/model/st_gnn.py`
- `physiq_pv/model/physics_loss.py`
- `scripts/run_cl_experiment.py` (orchestrator entry)

**Main outputs.**
- Initial trained model weights (`checkpoints/initial.pt`).
- Per-batch online updates when the gate allows them.
- Forecasts on each streaming window (used by Reflection and CL metrics).

The action block has two distinct lives. The first is **offline initial training**, where the model is fit on `train_ds` with the standard supervised loss `physics_loss_full + peak_loss`. This phase is deterministic, classical, and does not involve any ATSF planning: it produces the initial weights that the streaming loop will then maintain.

The second life of the action block is **online updating** during the stream. When the policy selects `trigger_update` (or when `--force_update` is set as a debug aid), the orchestrator calls `_retrain_window`, which iterates up to `n_retrain_batches` batches over the current window and routes each batch through `QualityGatedUpdater.step`. The updater applies a DER++ step subject to the soft Bernoulli gate (Section 6). Inference itself (i.e. running the model forward to compute predictions for evaluation, conformal snapshots and R-matrix filling) is currently embedded inside the same helpers and does not have its own service module.

### Reflection

**Purpose.**
Evaluates the consequences of an action. At the single-task level it compares the loss before and after an update and rolls the model back on degradation. At the multi-task level it records the R-matrix and produces continual-learning metrics. Metric-based, not meta-cognitive.

**Main files.**
- `physiq_pv/agent/cycle.py` (`reflect`)
- `physiq_pv/eval/cl_metrics.py`
- `scripts/run_cl_experiment.py` (snapshot / restore on degraded; R-matrix filling; holdout retention)
- `online_loop.py` (`_eval_loss`)

**Main outputs.**
- `improved` / `stable` / `degraded` verdict per triggered update.
- Weight rollback when `degraded`.
- `R[i, j]` matrix, BWT, FWT, Average Forgetting, learning curve.
- Frozen-holdout retention curve.

The reflection block is metric-based and operates at two timescales. At the **single-task** scale, every time an update is applied the orchestrator evaluates the loss on the same window before and after the update; `PhysiQAgent.reflect` computes a relative improvement, classifies the outcome as `improved`, `stable` or `degraded`, and the orchestrator restores the previously snapshotted `state_dict` if the verdict is `degraded`. This is the rollback mechanism that protects the model from individual toxic updates.

At the **multi-task** scale, after each task `i` the orchestrator evaluates the current model on every task `j` and records the result in the `R[i, j]` matrix of `CLMetricsTracker`. The matrix yields the three GEM continual-learning metrics: Backward Transfer (mean of `R[T-1, i] - R[i, i]` for `i < T-1`), Forward Transfer (`R[i-1, i]` against a baseline), and Average Forgetting (`max_l R[l, i] - R[T-1, i]`). Because the metric is MAE, the tracker is constructed with `higher_is_better=False`, which flips the signs so that BWT > 0 still means "training on later tasks improved the earlier ones".

Reflection in this pipeline is **not meta-cognitive**: there is no language model reasoning over the outcome and no policy update from reflection. The verdict is consumed only by the rollback gate and the report.

### Memory

**Purpose.**
Persists information across time: past experience for replay, model snapshots for rollback and resume, conformal calibration state for uncertainty quantification, per-plant drift state, and structured logs for offline analysis.

**Main files.**
- `physiq_pv/continual/replay_buffer.py`
- `physiq_pv/uncertainty/mondrian_cp.py`
- `physiq_pv/uncertainty/conformal_runtime.py`
- `physiq_pv/agent/cycle.py` (`_monitors` for per-plant drift state)
- `scripts/run_cl_experiment.py` (writes `report.json`, `online_events.jsonl`, `checkpoints/*.pt`)

**Main outputs.**
- QS-weighted replay buffer with deterministic sampling.
- Initial and final model checkpoints.
- `MondriaNCP` per-bin residual quantiles.
- Per-plant `KSDriftMonitor` buffers (process-local, not serialised).
- `online_events.jsonl`, `report.json`, `cl_metrics.json`, `learning_curve.csv`, `holdout_retention.csv`.

The memory block is materialised by a small, explicit set of structures.

- The **replay buffer** (`ReplayBuffer`) stores `(x, y_pv, pred_pv_old, qs)` tuples and is sampled with probability proportional to `qs`, floored by `qs_floor` (default `_QS_FLOOR = 1e-3`, exposed as a constructor argument). The floor prevents zero-QS samples from being permanently silenced. Sampling is deterministic via an injected `numpy.random.Generator`.
- The **model checkpoints** under `outputs/cl_experiment/<run>/checkpoints/{initial,final}.pt` persist the model state at the boundary of the streaming loop.
- The **conformal calibration state** is held inside `MondriaNCP`: per-bin residual quantiles indexed by QS band. It is not updated online.
- The **per-plant drift state** lives inside `PhysiQAgent._monitors` and is reset whenever the orchestrator is restarted.
- The **persistent logs** are `online_events.jsonl` (one record per task, with all the per-step signals) and `report.json` (run-level summary), both produced by the orchestrator.

---

## 5. End-to-End Workflow

The complete workflow chains the modules described in Section 4 into a single sequence. The orchestrator (`scripts/run_cl_experiment.py`) is the only entry point and reuses existing helpers (`_eval_loss`, `_retrain_window`, `_qs_from_features`, `_jsonl_safe`) rather than re-implementing them. The high-level structure is shown below.

```
Offline initial training
       │
       ▼
Walk-forward split            (train / calibration / holdout / stream)
       │
       ▼
Conformal calibration         (MondriaNCP fit on calibration_ds)
       │
       ▼
Streaming tasks               (loop over WalkForwardSplit.stream_tasks())
       │
       ├── Perception signals          (QS, m1..m5, forensics, drift, ci_width)
       │
       ├── Planning / action selection (PhysiQAgent + UtilityActionPolicy)
       │
       ├── Optional quality-gated update
       │       └── DER++ replay step
       │
       ├── Reflection and rollback     (loss before / after, restore if degraded)
       │
       └── Memory and logging          (buffer growth, JSONL event, checkpoint)
       │
       ▼
CL metrics and final report
```

Two clarifications matter. First, **the offline training phase is not divided into the five ATSF blocks**. It is a classical supervised fit on the `train_ds` slice and produces the model that the streaming loop will then maintain. The ATSF cycle proper begins after the conformal calibration step, when the orchestrator starts iterating over `stream_tasks()`. Second, **not all paths inside the streaming loop are exercised on every task**. The update path (`_retrain_window` → `QualityGatedUpdater.step`), the replay sampling step, and the rollback path are all conditional on `report["action"] == "retrain_triggered"`. On synthetic high-QS data the rule-based policy often prefers `skip_update` or one of the logged-only actions; the CLI flag `--force_update` overrides this decision and is intended as a smoke-test for the DER++ / rollback path, not as a substantive experiment.

---

## 6. Continual Learning Components

### Quality Score

The Quality Score is a per-(plant, time) data-quality assessment built from five soft components m1..m5 (curve shape, bias, completeness, variance ratio, physics coherence). The aggregate QS is the geometric mean of the five, which means that a single weak component already lowers the score; this is the desired behaviour because a sensor that is correct in four ways but wrong in one is not trustworthy. Night-time samples are handled separately (no signal to measure), and the optional `apply_qs_shrinkage` step smoothes the score against statistical noise on sparsely sampled plants.

Quality Score has two roles in the pipeline. As a **feature**, the five components m1..m5 enter the model input as channels 5..9. As a **diagnostic signal**, the aggregate QS is consumed by perception, by replay weighting, by Mondrian conformal stratification, and by the forensics drill-down.

### Suspicion and Drift Signals

Two distinct quality-related signals coexist in this pipeline and they must be kept separate.

The **per-sample Quality Score** is what the replay buffer uses to bias its sampling. When the buffer samples a mini-batch for the DER++ distillation step, the probability that a stored item is selected is proportional to its QS, floored at `qs_floor`. The floor is small enough (default `1e-3`) that low-QS samples remain reachable but are picked rarely.

The **fleet-level suspicion** is what the updater uses as its **update gate**. It is derived from QS forensics and is the average of per-component suspicion values across the fleet. The updater interprets it as a "do not trust this update" signal: the probability that a gradient step is skipped equals `suspicion_mean`. A high suspicion therefore makes updates less likely.

These two signals act on different ATSF blocks. The per-sample QS belongs to memory (it controls how memory is sampled). The fleet-level suspicion belongs to action (it controls whether the update is applied). The orchestrator's JSONL log exposes both, including aliases `update_gate_signal` and `replay_weight_signal`, to make the distinction explicit.

### Utility-Based Action Policy

The action policy is `UtilityActionPolicy`. For every candidate action in the 9-way set (`do_nothing`, `trigger_update`, `skip_update`, `fallback_pvgis`, `alert_sensor`, `alert_cleaning`, `alert_maintenance`, `alert_comms`, `preserve_replay`) it computes a linear utility from the `PolicyState` features (per-component suspicion, drift flag, ci_width, mode flags, fleet QS). The utilities are then normalised through a softmax whose temperature depends on the diagnostic mode: higher temperature when the mode is `uncertain`, lower temperature when it is `auto`. The orchestrator either takes the argmax of the distribution (in `auto` mode) or samples from it (otherwise).

The policy is therefore **probabilistic**, but it is not learned. The coefficients of the linear utility live in `default_action_weights()` and are hand-tuned. No reward signal is fed back from reflection. This is consistent with a workflow-based ATSF implementation: the agentic role of the planning block is to *compose* perception signals into an action choice, not necessarily to *learn* the composition. A swap to a learned policy (LinUCB-style bandit, actor-critic over the action space) would only require overriding `_utility`; it is recorded as future work.

A note on action effects. The `ACTION_EFFECTS` dictionary in `action_policy.py` classifies each of the nine actions as either runtime-active or logged-only. Only `do_nothing`, `trigger_update` and `skip_update` actually drive code paths downstream of the policy. The remaining six actions are recorded in `online_events.jsonl` but have no consumer in this iteration; they describe an *intent* that a human operator or a future automation layer could act on.

### Quality-Gated Update

When the policy selects `trigger_update`, the orchestrator hands the current window over to `_retrain_window`, which in turn calls `QualityGatedUpdater.step` on each batch. The updater applies three filters in order before committing a gradient step.

The first filter is a **legacy hard floor**: if a scalar `qs_mean` is supplied and `qs_threshold` is configured, an update is unconditionally skipped when `qs_mean <= qs_threshold`. This filter is retained for backward compatibility and is disabled by default (`qs_threshold = None`).

The second filter is the **soft Bernoulli gate**. A Bernoulli draw with probability `suspicion_mean` decides whether to skip; the higher the suspicion, the more likely the update is dropped. This is the main online gate and corresponds to "do not trust this update".

The third filter is **implicit**: if the buffer is empty (`len(buffer) < replay_batch`), the DER++ distillation term is omitted and only the supervised loss is used. This is by construction a no-op for the first few updates after initialisation.

Whenever the gradient step is allowed, the buffer is updated **regardless of whether the step was actually taken**: the current sample is always added before the gate fires, so memory grows even when the update is skipped. This separation between "should we learn from this?" and "should we remember this?" is intentional.

### DER++ Replay

DER++ replay (Buzzega et al., NeurIPS 2020) is a memory-based mechanism that is *intended to mitigate forgetting* during online updates. Whenever a gradient step is allowed and the buffer has at least `replay_batch` items, the updater samples a mini-batch of past observations and adds two additional terms to the loss.

- The **α-distillation term** computes the mean-squared error between the current model output on the replayed inputs and the old model output that was stored together with each replayed sample. This stabilises the representation.
- The **β-retention term** computes the mean-squared error between the current model output on the replayed inputs and the stored ground-truth targets. This stabilises the task performance.

Default coefficients are `α = 0.2` and `β = 1.0`, following the original DER++ ablation. Gradients are clipped to `max_grad_norm = 1.0`. Sampling is QS-weighted via `ReplayBuffer.sample`, with the probability floor described above.

Two important caveats apply.

- **Replay covers the PV head only.** The buffer stores `pred_pv_old` and `y_pv`, not the corresponding GHI quantities. The DER++ terms therefore protect the PV head; the GHI head is left to drift under online updates. Dual-head replay is recorded as future work because it would touch every buffer-add call site and the loss assembly inside the updater.
- **Forgetting reduction is not yet validated.** DER++ is implemented as a mechanism, but this iteration does not include a controlled run with updates actually firing nor a no-replay baseline against which to measure forgetting reduction. Any claim to the contrary should be deferred to a later experimental iteration.

### Reflection and Rollback

After each triggered update the orchestrator evaluates the loss on the same window before and after the update. The relative improvement is computed as `(loss_before - loss_after) / |loss_before|`. `PhysiQAgent.reflect` classifies the outcome as `improved` when the improvement exceeds a positive tolerance (default 5%), `degraded` when the improvement is more negative than the symmetric tolerance, and `stable` otherwise. On `degraded` the orchestrator calls `model.load_state_dict(checkpoint_before)`, restoring the snapshot taken before the update; the rollback flag is recorded in the JSONL log.

At the run level, `CLMetricsTracker` records the per-task evaluation matrix and computes Backward Transfer, Forward Transfer and Average Forgetting on completion. The learning curve is the diagonal of the R matrix. Reflection here is metric-based: there is no language-model reasoning over the verdict and no policy update from it.

### Memory and Logging

Memory and logging together produce the artefacts that allow a run to be analysed offline. Five outputs are written to `outputs/cl_experiment/<run_name>/`.

- `report.json` summarises the run: random seed, configuration, split summary, offline training metrics, conformal coverage report, stream-level counters (triggers, rollbacks, final buffer size), CL metrics (BWT, FWT, AF, learning curve), holdout retention summary, and the list of known limitations.
- `online_events.jsonl` contains one record per stream task, with task id, timestamp range, action chosen, full policy distribution, ci_width, loss before / after, reflection verdict, rollback flag, batches updated, buffer size, task MAE and holdout MAE.
- `cl_metrics.json` contains the BWT, FWT and Average Forgetting values together with the full R-matrix.
- `learning_curve.csv` and `holdout_retention.csv` expose the diagonal of R and the per-task holdout evaluation in a flat, plot-ready format.
- `checkpoints/initial.pt` and `checkpoints/final.pt` store the model state at the boundaries of the streaming loop.

The replay buffer, the conformal predictor's per-bin quantiles and the per-plant drift buffers are all kept in process memory and are not serialised across orchestrator restarts.

---

## 7. Supported and Unsupported Claims

The following table records which claims the codebase supports today.

| Claim | Status | Explanation |
|-------|--------|-------------|
| ATSF-inspired pipeline | **Supported** | The five ATSF blocks are mapped to explicit software modules. |
| Workflow-based ATSF implementation | **Supported** | A deterministic orchestrator composes the modules according to the ATSF decomposition; the policy is rule-based. |
| ATSF-compatible architecture | **Supported** | Stable public interfaces allow drop-in replacement of policy, drift detector and conformal predictor without a structural refactor. |
| End-to-end ATSF-style adaptive forecasting pipeline | **Supported** | A single orchestrator covers perception → planning → action → reflection → memory on a walk-forward split and emits formal CL metrics. |
| Full autonomous ATSF agent | **Not supported** | No autonomy, no meta-cognitive reasoning, no hierarchical memory; the pipeline is a static workflow. |
| Reinforcement learning | **Not supported** | No reward signal is propagated back to the policy. |
| LLM-based agent | **Not supported** | No language model is involved anywhere in the pipeline. |
| Learned adaptive policy | **Not supported (future work)** | Policy weights are hand-tuned in `default_action_weights`. |
| Rule-based utility policy | **Supported** | Documented in `UtilityActionPolicy`; coefficients are static. |
| DER++ replay mechanism | **Supported (PV head only)** | Implemented in `QualityGatedUpdater.step` with `α = 0.2`, `β = 1.0`. |
| DER++ reduces forgetting | **Not yet supported** | No controlled run with updates firing and no no-replay baseline in this iteration. |
| Offline conformal calibration | **Supported** | `calibrate_conformal(split.calibration_ds, ...)` is called once before the stream. |
| Online adaptive conformal prediction | **Not supported (future work)** | `maybe_recalibrate_cp` is a documented no-op hook. |
| Synthetic / study mode | **Supported** | Canonical mode of this iteration; orchestrator exercised on synthetic data. |
| Real-data causal diagnosis fully supported | **Out of scope** | Real-data labelling is deferred; the classifier falls back to rule-based heuristics on real data. |

A short interpretation of the entries above is worth making explicit.

- "Workflow-based ATSF implementation" is supported because what ATSF requires of an instantiation is the **structural** five-block decomposition together with a runtime that connects them. Our orchestrator does both. The fact that the policy is rule-based and the runtime is deterministic does not invalidate the claim; it qualifies it as workflow-based as opposed to autonomous.
- "Full autonomous ATSF agent" is not supported because we lack the components that distinguish an autonomous agent from a static workflow: meta-cognitive reasoning over reflection outcomes, hierarchical memory (working memory vs. long-term knowledge), planning over multi-step horizons, and an adaptive policy that updates itself from experience. Each of these is recorded as future work.
- "Reinforcement learning" and "LLM-based agent" are unnecessary for an ATSF instantiation in our reading of the paper. They are common ways to implement the planning block but they are not requirements. We use neither and the pipeline still respects the ATSF decomposition.
- "DER++ replay mechanism" is supported because the mechanism is present and exercised whenever updates fire. "DER++ reduces forgetting" is a *different* claim that requires empirical evidence: it must be backed by a controlled run with multiple triggered updates and a comparison against a no-replay baseline. Both are out of scope for this iteration.

---

## 8. Known Limitations and Future Work

Limitations are grouped by nature rather than enumerated as a flat list.

### Current technical limitations

- **DER++ replay is PV-head only.** The buffer stores `pred_pv_old` and `y_pv`; the GHI head is not distilled and is allowed to drift under online updates. Extending the buffer schema to `(x, y_ghi, y_pv, pred_ghi_old, pred_pv_old)` would touch every buffer-add call site and the loss assembly inside the updater.
- **Conformal prediction is calibrated once on `calibration_ds`** and not updated during streaming. Under non-stationary drift the marginal coverage guarantee degrades. `maybe_recalibrate_cp` is a documented no-op hook that delimits the intended extension point.
- **Drift detection is KS-based, not full ADWIN.** `KSDriftMonitor` runs a two-sample Kolmogorov-Smirnov test on consecutive half-windows; latency is therefore `O(window_size)`. The historical alias `ADWINDriftMonitor` is retained for backward compatibility.
- **Six of nine policy actions are logged-only.** `fallback_pvgis`, all `alert_*` actions, and `preserve_replay` are recorded in `online_events.jsonl` but have no consumer downstream. They describe intent rather than producing side effects.
- **Policy weights are hand-tuned.** `default_action_weights` is a static dictionary; no learning loop updates it from reflection.
- **Per-plant drift state is process-local.** Restarting the orchestrator resets the per-plant `KSDriftMonitor._buf`.

### Experimental limitations

- **DER++ effectiveness against forgetting is not empirically demonstrated** in this iteration. Showing it requires runs where updates actually fire (i.e. with drift in the stream or with `--force_update`) and a no-replay baseline for comparison.
- **The update path must be exercised in controlled streaming scenarios** to produce meaningful BWT / FWT / AF values. On synthetic high-QS data the policy rarely chooses `trigger_update`, so the R matrix tends to have identical rows and the CL metrics collapse to zero (which is technically correct but uninformative).
- **Forgetting reduction must be quantified explicitly** through the holdout retention curve and a baseline that disables either the replay term or the entire update path. Such a baseline is not part of this iteration.

### Out of scope for now

- **Real-data validation.** The Piedmont 2019 loaders exist and the CLI flag `--data real` is plumbed, but real-data behaviour is not assessed here. This is a deliberate choice for the current iteration; it is not a permanent restriction.
- **Real-data causal labelling (G2).** The fault classifier's labels are hard-coded from the synthetic generator; real fleets have no oracle labels and the classifier falls back to its rule-based heuristic.
- **Policy learning.** No reinforcement learning, no contextual bandit. The interface is designed to make a future swap straightforward, but no learning loop is implemented today.
- **Full ATSF autonomous agent.** No meta-cognitive reasoning, no LLM in the loop, no hierarchical memory. This is consistent with the workflow-based positioning of the pipeline.

---

## 9. Suggested Wording

The following formulation can be used verbatim or adapted in the thesis. It is calibrated to match exactly what the code supports today and to avoid claims that would require validation we have not yet performed.

> *The proposed pipeline implements a workflow-based ATSF architecture for continual PV forecasting. The forecasting model is initialised through offline supervised training, while the subsequent online continual-learning loop operationalises the ATSF cycle through perception, planning, action, reflection and memory. The system does not rely on reinforcement learning or LLM-based agents; instead, it uses a deterministic orchestrator and a rule-based utility policy. DER++ replay, quality-gated updates and rollback are implemented as mechanisms for online adaptation, while their empirical contribution to forgetting reduction must be assessed through controlled experiments and suitable baselines.*

This wording is appropriate for three reasons. First, it states explicitly that the pipeline is *workflow-based*, which is the strongest claim that the code can support: there is no autonomous agent and no learned policy. Second, it separates the *initialisation* role of the offline training from the *adaptation* role of the online loop, mirroring the way ATSF is actually exercised in the codebase. Third, it positions DER++ as a *mechanism* whose empirical effect is yet to be measured, rather than as a proven cure for forgetting. Any stronger statement about forgetting reduction should be deferred to the experimental section of the thesis and backed by a controlled run with a no-replay baseline.

---

## 10. What This File Is For

This document is intentionally written to be readable end-to-end without opening the source code or the dataset. It serves several distinct purposes.

- As an **architectural reference**: it lists the modules involved in the continual-learning pipeline and explains how they fit together.
- As a **presentation document**: it can be shared with reviewers, supervisors and external readers who want to understand the system without running it.
- As a **study guide**: someone approaching the codebase for the first time can use the sections of this file as a roadmap to navigate the modules in the recommended order.
- As a **roadmap of limitations and future work**: it makes explicit which claims the codebase supports today and which require additional experiments or implementation work.

It is intentionally conservative on empirical claims. Any statement about the effectiveness of a specific mechanism — most notably DER++ against forgetting and conformal prediction under drift — belongs in the experimental sections of the thesis and must be backed by controlled runs.

---

## References

- Cheng et al., 2026 — *Position: Beyond Model-Centric Prediction — Agentic Time Series Forecasting*.
- Buzzega et al., NeurIPS 2020 — *Dark Experience for General Continual Learning (DER++)*.
- Lopez-Paz & Ranzato, NeurIPS 2017 — *Gradient Episodic Memory* (BWT, FWT, Average Forgetting definitions).
- Tan et al., 2022 — *MultiROCKET* (arXiv:2102.00457).
- Bifet & Gavaldà, 2007 — *Learning from Time-Changing Data with Adaptive Windowing* (true ADWIN; here approximated, not implemented in full).
- Cuturi & Blondel, 2017 — *Soft-DTW* (implementation in `tslearn`).
- Vovk et al., 2005; Boström et al., COPA 2021 — *Mondrian Conformal Prediction*.
- Gibbs & Candès, 2021 — *Adaptive Conformal Inference* (future work, not implemented).
