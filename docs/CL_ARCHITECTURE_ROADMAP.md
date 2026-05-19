# PhysiQ-PV — Roadmap Architetturale CL post-MVP

Componenti architetturali mancanti per allineare la pipeline Continual Learning allo stato dell'arte ATSF + CL-TS forecasting (2023–2026). Esclude validazione sperimentale, tuning, benchmark e documentazione: solo pezzi di codice nuovi o sostitutivi che chiudono gap rispetto alla letteratura.

---

## 1. Stato MVP (riferimento)

| Modulo ATSF | Implementazione attuale |
|---|---|
| Perception | `compute_qs(ds, debug=True)` → QS + m1..m5 |
| Planning | `PhysiQAgent._diagnose` (ADWIN + soft-DTW + causal classifier + QS forensics) |
| Action | `online_loop._retrain_window` + `QualityGatedUpdater` |
| Reflection | `agent.reflect` (loss before/after + rollback su `degraded`) |
| Memory | `ReplayBuffer` (DER++, QS-weighted sampling) |

End-to-end smoke test passa su dataset sintetico con scenari di guasto iniettati. QS controlla feature, diagnosi, gate, replay sampling e rollback. Coerenza data-centric verificata.

---

## 2. Gap architetturali prioritari

### A. Conformal Prediction stratificata QS-band

- **Stato:** `physiq_pv/uncertainty/mondrian_cp.py` scaffolded; non wireato in `run_online`.
- **Gap SOTA:** ATSF richiede modulo "Quantify" esplicito. Senza CP, le previsioni mancano di intervalli calibrati e PhysiQ-PV non risponde all'argomento "trustworthy forecasting" (Trustworthy PV Review MDPI 2026 [R12]).
- **Riferimenti:** Cordier et al. COPA 2023; Renkema et al. Solar Energy Advances 2024; CACP arXiv:2510.15780; Nguyen et al. Energy and AI 2025 ([R18]).
- **Interventi:**
  - Esporre `MapieRegressor` con `groups=qs_band(qs)` nel ciclo.
  - Calibration set fisso costruito durante la fase batch.
  - Output per step: `(pred_pv, pred_lo, pred_hi)` per ogni (batch, node).
  - Persistenza intervalli + width nella history del loop.
- **Interfaccia attesa:**
  ```python
  cp = StratifiedConformalRegressor(alpha=0.1)
  cp.fit(X_cal, y_cal, qs_bands_cal)
  pred, lo, hi = cp.predict(X_test, qs_band_test)
  ```

### B. CL Metrics module (BWT / FWT / Avg Forgetting)

- **Stato:** assente. `eval/benchmark.py` esiste solo come placeholder.
- **Gap SOTA:** standard di benchmark CL Buzzega NeurIPS 2020 (DER++); senza, nessun confronto credibile con Proceed KDD 2025, FSNet ICLR 2023, A²ER Frontiers 2026.
- **Interventi:**
  - Nuovo modulo `physiq_pv/eval/cl_metrics.py`.
  - Funzioni `backward_transfer`, `forward_transfer`, `average_forgetting`, `learning_curve`.
  - Holdout fisso "old task" + sliding "new task" derivati dal walk-forward split.
  - Aggregazione cross-seed.
- **Interfaccia attesa:**
  ```python
  tracker = CLMetricsTracker(tasks=task_splits)
  tracker.record(task_id, metric_name, value)
  report = tracker.compute()  # {bwt, fwt, af, learning_curve}
  ```

### C. Action policy parametrica

- **Stato:** mapping hardcoded in `qs_forensics._ACTION_FAMILY`.
- **Gap SOTA:** ATSF 2026 separa Planning come modulo dedicato con policy esplicita. A²ER Frontiers 2026 introduce gating delle azioni su segnali multipli. Hardcoded if/elif è anti-pattern dichiarato nella letteratura ATSF.
- **Interventi:**
  - Nuovo modulo `physiq_pv/agent/action_policy.py`.
  - Stato in input: `(suspicion_per_component, drift_flag, ci_width, mode, qs_mean)`.
  - Output: distribuzione su azioni `{skip_update, trigger_update, fallback_pvgis, alert_*, do_nothing}`.
  - MVP implementabile come utility-based (softmax su utility per azione); upgrade futuri verso bandit o RL.
- **Interfaccia attesa:**
  ```python
  policy = UtilityActionPolicy(weights=...)
  action_dist = policy(state)
  chosen = policy.sample(action_dist)
  ```

### D. Long-term memory + checkpoint store

- **Stato:** `ReplayBuffer` short-term DER++ + snapshot ad-hoc in `online_loop` per rollback.
- **Gap SOTA:** ATSF Memory module multi-tier (short/long/episodic). FSNet Pham ICLR 2023 distingue fast/slow memory. PhysiQ-PV ha solo fast. Manca timeline rollback e log eventi drift.
- **Interventi:**
  - Nuovo modulo `physiq_pv/continual/memory.py` con tre componenti:
    - `ShortTermBuffer` — wrapper sopra `ReplayBuffer` attuale.
    - `LongTermCheckpoints` — snapshot pesi periodici (es. ogni stride). Indice temporale + recovery point su `reflection=degraded` esteso a finestre passate.
    - `EpisodicLog` — registro drift events + diagnosi + esito update + segnali post-hoc (ground-truth quando disponibile).
- **Interfaccia attesa:**
  ```python
  mem = ATSFMemory(short=buf, long=ckpt_store, episodic=event_log)
  mem.long.checkpoint(model, step=t)
  mem.long.rollback_to(step=t-2)
  mem.episodic.log(event=drift_report)
  ```

### E. CP feedback loop nel ciclo ATSF

- **Stato:** assente. Anche dopo integrazione CP (A), gli intervalli sarebbero solo output finale.
- **Gap SOTA:** ATSF formalismo richiede reflection-on-action: incertezza alta = segnale di reflection che retroalimenta planning. Senza, il ciclo non è chiuso.
- **Interventi:**
  - `agent.step` riceve `ci_width` dal modulo CP.
  - Combinato con QS suspicion e drift flag, produce `uncertainty_trigger` (segnale aggiuntivo per planning).
  - Action policy (C) usa `ci_width` come feature di stato.
- **Interfaccia attesa:**
  ```python
  report = agent.step(ds_window, updater=...,
                     qs=qs, m_components=m, ci_width=cp_width)
  # report['uncertainty_flag'] = bool
  ```

### F. Hydra+MultiROCKET sostituzione causal_classifier

- **Stato:** `physiq_pv/agent/causal_classifier.py` allenato su synthetic labels.
- **Gap SOTA:** ROCKET superato 2023 (Dempster DMKD). Doc piano già indica swap. PhysiQ-PV cita Hydra+MultiROCKET come scelta ma codice non lo usa.
- **Interventi:**
  - Sostituzione interna mantenendo signature `diagnose(qs_seq) → (cause, confidence)`.
  - Wrapper via `sktime.classification.kernel_based.RocketClassifier` oppure direttamente `hydra` + `RidgeClassifierCV`.
  - Training su clustering soft-DTW reale post-labeling (richiede sessione con prof).
- **Interfaccia attesa:** invariata (drop-in).

### G. Cold-start strategy module

- **Stato:** gestione implicita NaN fallback in `qs_forensics.combined_suspicion` (cade su solo fleet-relative se history insufficiente).
- **Gap SOTA:** strategia esplicita per impianti nuovi raccomandata da Sarmas Sci. Reports 2022 (transfer learning under scarcity). Letteratura PV trasferimento cross-site richiede onboarding formalizzato.
- **Interventi:**
  - Nuovo modulo `physiq_pv/agent/cold_start.py`.
  - Parametri: `min_history_days`, `ramp_up_curve`.
  - Trust ramp-up: peso `w_self_baseline` cresce con storia accumulata, fino a saturazione.
  - Fallback fleet-only score sotto soglia minima storia.
- **Interfaccia attesa:**
  ```python
  cold = ColdStartManager(min_history_days=14)
  w_self = cold.self_weight(plant_history_days)
  # passata a forensic_report come override di w_fleet
  ```

### H. Planning module esplicito (decoupling)

- **Stato:** planning fuso in `_diagnose` (sceglie causa) + `step` (decide retraining).
- **Gap SOTA:** ATSF separa moduli Perception / Planning / Action / Reflection / Memory. PhysiQ-PV viola separazione: `_diagnose` mescola perception + planning.
- **Interventi:**
  - Nuovo modulo `physiq_pv/agent/planner.py`.
  - Riceve: `perception` (qs + m_components), `memory.episodic`, `reflection` history.
  - Produce: piano azione candidato (senza eseguirlo).
  - `_diagnose` ridotto a sola perception/diagnosi descrittiva.
  - `step` invoca `planner.plan(state)` → `action_policy.execute(plan)`.
- **Interfaccia attesa:**
  ```python
  plan = planner.plan(perception=p, memory=m, reflection=r)
  result = action_policy.execute(plan)
  ```

---

## 3. Tabella sintetica gap → reference

| ID | Componente | Stato attuale | SOTA reference | Modulo target |
|---|---|---|---|---|
| A | Conformal Prediction QS-band | scaffolded, non integrato | Cordier COPA 2023; Renkema SEJA 2024 | `uncertainty/mondrian_cp.py` (wire) |
| B | CL metrics BWT/FWT/AF | mancante | Buzzega NeurIPS 2020 | `eval/cl_metrics.py` (nuovo) |
| C | Action policy parametrica | hardcoded mapping | A²ER Frontiers 2026; ATSF 2026 | `agent/action_policy.py` (nuovo) |
| D | Long-term memory + checkpoint store | solo short-term | FSNet ICLR 2023; ATSF 2026 | `continual/memory.py` (nuovo) |
| E | CP feedback loop | assente | ATSF 2026 | `agent/cycle.py` (estensione) |
| F | Hydra+MultiROCKET causal classifier | placeholder synthetic | Dempster DMKD 2023 | `agent/causal_classifier.py` (rewrite) |
| G | Cold-start strategy | implicito | Sarmas Sci. Rep. 2022 | `agent/cold_start.py` (nuovo) |
| H | Planning module esplicito | fuso con diagnosi | ATSF 2026 | `agent/planner.py` (nuovo) |

---

## 4. Dipendenze fra interventi

```
A (Conformal) ──► E (CP feedback)
                    │
                    ▼
                  C (Action policy) ──► H (Planner)
                                          │
D (Memory tiers) ─────────────────────────┤
                                          │
G (Cold-start) ───► forensics override ──┘
                                          │
F (Hydra real) ────► causal classifier ──┘
                                          │
B (CL metrics) ──────► eval transversale  │
```

A è prerequisito di E. C abilita H. D è autonomo ma alimenta H tramite `memory.episodic`. F è drop-in indipendente. B è trasversale.

---

## 5. Priorità per coerenza ATSF

| Priorità | ID | Motivazione |
|---|---|---|
| Alta | A, B, C | Senza CP + metrics + policy parametrica, defense ATSF debole. Reviewer flag immediato. |
| Media | D, E, F | Espande ATSF Memory + chiude loop reflection. Hydra rimpiazza placeholder. |
| Bassa | G, H | Cold-start e planner separato sono raffinement; difendibili anche fusi nelle versioni attuali. |

---

## 6. Out-of-scope di questo documento

- Validazione su dataset reale Sentinel multi-anno → bloccato da mapping UPN.
- Benchmark vs Proceed, FSNet, A²ER, PTER, SolNet, GPVS-Faults.
- Tuning iperparametri (`w_fleet`, `recent_window`, dominance bands, DER++ α/β).
- Ablation studies (forensics on/off, gate on/off, sampling QS-weighted vs uniform).
- Notebook diagnostici e documentazione paper-style.
- Scelta scope tesi (full pipeline vs sotto-pipeline).

Questi appartengono al piano sperimentale, non al gap architetturale.

---

*Documento generato post-MVP CL (branch `feat/continual-learning`). Da rivedere quando ogni componente sopra è implementato o esplicitamente fuori scope tesi.*
