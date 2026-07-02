# Drift-robust smell classification: baseline-relative features + cross-session validation

## Context

The deployed smell classifier (`BalancedRFClassifier`) is trained on **absolute** resistance
features (`log1p(R)` → RobustScaler). MOS sensor baselines drift day-to-day (aging, temperature,
humidity), so the same smell produces different raw resistances on a different day and the model
fails unless it was trained the same day inference runs. The system currently only *detects* drift
(`/smell/drift`, `diagnose_sample`) — it never *corrects* it.

The fix, validated by both the operator's intuition and the machine-olfaction literature
(Marco & Gutiérrez-Gálvez; Fonollosa; Ziyatdinov; Vergara), is to stop classifying absolute values
and instead classify a **drift-invariant representation**: each sample referenced to a fresh
same-session clean-air baseline, plus per-sample pattern normalization. Because the target and the
baseline drift *together*, their relative response stays stable across days.

Decisions (confirmed with user):
- **Additive, not a rewrite (hard requirement).** The current absolute-feature pipeline works well for
  offline / same-day use and MUST stay fully runnable and unchanged by default. Drift-robustness is a new,
  **opt-in** mode selected at runtime; with the new flags off, every code path behaves exactly as today.
  See [[feedback_additive_not_rewrite]].
- A short clean-air baseline **is captured per session** → baseline-relative features are the primary path.
- A small **multi-day same-smell validation set will be collected** → we can honestly prove day-invariance.
- Scope this round = **Tier-1 core**: cross-session eval harness + baseline-relative + SNV features,
  wired for inference-time baseline capture. **Non-breaking** (config-gated, A/B against current).
- Keep the Random Forest (beats deep learning at this data size). **No deep model, no LLM/AI agent**
  this round — both are the wrong tool here (research-confirmed); CPCA/temporal-CNN are noted as future options.

### Backward-compatibility guarantee (how "additive" is enforced)
- New config flags default to the current behavior: `baseline_mode="none"`, `snv=False` → identical pipeline,
  identical features, identical results. Existing saved models keep loading and predicting unchanged.
- The drift-robust mode is chosen per run: via `SmellClassifierConfig` (or an `ENOSE_BASELINE_MODE` /
  `ENOSE_SNV` env override) and, for training, a flag on `/smell/learn_from_csv`. Both modes coexist;
  nothing is removed or hard-swapped.
- `POST /smell/baseline` and the client baseline capture are **optional**: if never called, classification
  works exactly as today. When a model was trained in absolute mode, the baseline endpoint is a no-op.
- The eval harness compares the modes side by side precisely so the current pipeline remains a first-class,
  measured baseline — not something we replace sight-unseen.

### What already exists to reuse (do NOT rebuild)
- Preprocessing is state-light and slot-friendly: `enose/classifier/preprocessing.py`
  (`clean_resistances`, `log1p_resistances`, `scale_resistances`, `compute_resistance_clip_bounds`).
- A **drift-correction mechanism already exists but is unwired** in `enose/classifier/xgb.py`:
  `update_drift_baseline` (EMA), `_apply_drift_correction`, `_init_drift_baseline_from_training`
  (inits baseline from the abundant `air` class). Reuse this pattern.
- The live client already computes a local clean-air baseline: `enose/client/live.py`
  `LiveSensorPlot._update_baseline_locked` (currently plot-only).
- Session/segment grouping for honest splits already flows through training:
  `enose/classifier/training.py` `_extract_groups` / `GroupShuffleSplit`; and
  `data/database_robodog_time_windows.csv` carries `session_id` / `segment_id` / `t_start`.
- The `air` class is abundant across datasets → the natural baseline reference.

---

## Phase A — Cross-session evaluation harness (build FIRST; proves the problem + measures each fix)

New `scripts/eval_drift.py` (read-only over data; no model/server writes):
- **Leave-one-session-out (LOSO)** / train-on-session(s)-A → test-on-held-out-session-B evaluation,
  grouping by `session_id`/`segment_id` from `data/database_robodog_time_windows.csv` (and any
  collected multi-day CSVs). This is the honest drift metric — unlike the current random/group split
  which can still leak same-session structure.
- Compare representations side by side: **(0)** current absolute · **(1)** baseline-relative ·
  **(2)** baseline-relative + SNV. Report balanced accuracy, macro-F1, and confusion per config,
  plus a "same-session (upper bound)" vs "cross-session (drift)" gap.
- Reuse `BalancedRFClassifier` + `enose/classifier/training.py` helpers; parameterize the feature
  config so the same code trains each variant.

Also add a short **data-collection protocol** doc (`docs/DRIFT_DATA_PROTOCOL.md`): record
air + ≥2 smells (e.g. rose, coffee) on 2–3 separate days, each with a clean-air baseline segment,
tagged with a `session_id`/day — the input the harness needs.

## Phase B — Drift-invariant feature representation (the core; non-breaking, config-gated)

1. **New transforms in `enose/classifier/preprocessing.py`:**
   - `baseline_relative_resistances(df, baseline, mode)` — `mode ∈ {delta: R−R₀, ratio: R/R₀, logratio: log1p(R)−log1p(R₀)}`.
     `logratio` composes naturally with the existing `log1p` step.
   - `snv_normalize(df)` — per-sample standardization across R1–R17 (subtract row mean, divide by row std);
     makes the odour *pattern* invariant to overall intensity/gain drift.
2. **Config flags in `enose/classifier/config.py` (`SmellClassifierConfig`):**
   `baseline_mode: str = "none"` (`none|delta|ratio|logratio`) and `snv: bool = False`.
   Defaults keep current behavior → nothing changes until enabled.
3. **Baseline source + storage in `enose/classifier/balanced_rf.py`:**
   - At `train()`: compute the reference air baseline (per-session mean of `air` rows, else first-N-frames
     per recording), store as `self.sensor_baseline_` / `self._original_baseline_`
     (mirror `xgb.py._init_drift_baseline_from_training`).
   - Insert baseline-relative → SNV into the transform chain in **both** `train()` and
     `process_sensor_data()` (they must stay identical), after `clean_resistances` and before scaling.
   - `_build_model`/augmentation unchanged; clip-bounds computed in the same (relative) space they're applied.
4. **Persistence** in `enose/classifier/persistence.py`: save/restore `baseline_mode`, `snv`,
   `sensor_baseline_`, `_original_baseline_` (extend `build_save_payload` + `load`, like the existing metric fields).

## Phase C — Inference-time baseline capture (operational plumbing)

- **Server:** add `POST /smell/baseline` (accepts a short batch of clean-air samples for the current
  `session_id`) that sets/EMA-updates the classifier's live baseline used in `process_sensor_data`.
  Reuse the `update_drift_baseline` EMA pattern from `xgb.py`. The live buffer already tracks `session_id`.
- **Client/robot:** at session start, capture ~30–60s clean air and POST it as the baseline.
  Reuse `live.py`'s existing baseline computation; the robot policy's SETTLE/PURGE phases already
  sit in clean air (`enose/robot/policy.py`).
- If no baseline has been set for a session, fall back to the persisted training baseline and surface
  a warning via the existing `/smell/drift` / OOD path.

## Critical files
- New: `scripts/eval_drift.py`, `docs/DRIFT_DATA_PROTOCOL.md`.
- Edit: `enose/classifier/preprocessing.py` (new transforms), `enose/classifier/config.py` (flags),
  `enose/classifier/balanced_rf.py` (`train` + `process_sensor_data` + baseline init/storage),
  `enose/classifier/persistence.py` (persist baseline + flags), `enose/server/routes/smell.py`
  (+ `schemas.py`) for `POST /smell/baseline`, and a client helper in `enose/client/api.py` / `live.py`.

## Verification (Docker-first; isolated data — never the live model, per [[project_server_writes_live_data]])
1. **Harness on existing sessions:** `python scripts/eval_drift.py` → show the cross-session accuracy
   gap for config (0) absolute, then the improvement for (1) baseline-relative and (2) +SNV.
   Success = cross-session balanced accuracy rises materially (target ≥85%, up from near-chance) while
   same-session accuracy is preserved.
2. **Round-trip:** train a model with `baseline_mode=logratio, snv=True` on session A, save/reload,
   confirm baseline + flags persist and `process_sensor_data` reproduces training-time features.
3. **Inference plumbing:** start a server (`ENOSE_NO_VLM=1`) against an **isolated copy** of the data/model
   dir; `POST /smell/baseline` with air samples, then classify a same-"day" and a different-"day" sample;
   confirm the different-day prediction is now correct.
4. **On collected multi-day data:** rerun the harness to confirm the win holds on true cross-day data.

## Explicitly out of scope this round (future options, noted for the record)
- Reference-free **CPCA / component correction** (for sessions with no baseline) — add later if needed.
- **1D-CNN / LSTM / DANN** — research says unlikely to beat RF+features at current data size; revisit only
  if data grows to 5k+ labeled non-air samples and drift proves nonlinear.
- **LLM / AI agent** — wrong tool for numeric sensor drift; not pursued.

_(On execution, copy this plan to `docs/` per the plan-location convention: [[feedback_plan_location]].)_
