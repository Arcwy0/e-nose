# Multi-day data-collection protocol (for drift validation)

To measure robustness to day-to-day sensor drift, collect recordings of the
**same smells on different days**, each with a clean-air baseline. This is the input the
cross-session eval harness (`scripts/eval_drift.py`) needs, and the data pattern the
drift-robust modes are designed for.

## Why

MOS resistances drift with sensor aging, temperature and humidity. A model trained and tested
on the *same* day looks perfect but tells you nothing about tomorrow. Only **cross-day** data
reveals the real failure — and lets us measure the fix (baseline-relative features).

## What to record

Per day (≥ 2 days, ideally 3–4; more days = stronger evidence):

1. **Clean-air baseline** — at least 60 s of stable clean air before each exposure;
   record longer if recovery/settling requires it. Label only confirmed clean air `air`.
   This is the R₀ reference the drift-robust mode subtracts.
2. **Each target smell** — e.g. `rose`, `coffee` (pick ≥ 2, keep the set identical across days).
   Record onset, full response, stable plateau, removal and recovery. A 30–60 s
   entire exposure is not sufficient for the current default plateau extractor:
   final 25% minus a 30 s end margin must contain a full 60 s window, so exposures
   shorter than roughly six minutes may yield none. Inspect preview; select
   scientifically justified settings rather than forcing every trace to pass.
3. Keep the sensor rig, gain and wiring **unchanged** across days — we want to capture *natural*
   drift, not a hardware change.

Spread the days out (different mornings, ideally different ambient conditions) so real drift
accumulates between sessions.

## File format

One CSV per recording is easiest to audit; combine only while retaining units
and identity. Use verified labels/annotations. Canonical columns:

| Column | Meaning |
|---|---|
| `R1`…`R17` | resistance sensors (as the client already logs them) |
| `T,H,CO2,H2S,CH2O` | Optional measured ambient/environmental channels; no heater temperature here |
| `resistance_unit` | Verified `ohm`, `kohm`, or `mohm`; never guessed from old RLOW |
| `smell_label` (or `Gas name`) | `air`, `rose`, `coffee`, … (lowercased) |
| **`session_id`** | **one distinct value per day/recording** — this is the grouping the harness splits on |
| `exposure_id` | One ID per complete air/analyte pair; avoid splitting one exposure across folds |
| `Timestamp` | Required for plateau extraction; retain real date and acquisition timing |

`session_id` is the critical field: it's how leave-one-session-out training simulates
"train on day 1–2, test on day 3". Time-only supplier exports need recording date
and session identity. Their `rs_ch*_ohm` channels are already resistances;
normalization maps them by index without proving cross-board sensor equivalence.

The client stamps a publisher session ID and records acquisition metadata in
NPZ. Keep physical board/configuration, channel mapping, calibration, heater
target/actual temperature/faults, flow/distance/concentration and event timing
in the experimental record. Supplier filenames do not establish chemical labels.
Retain raw sequences privately even when fitting only stable windows.

## How to validate once collected

```bash
# Absolute vs baseline-relative vs +SNV, leave-one-session-out:
docker exec -i enose-srv python scripts/eval_drift.py \
    --csv data/<your_multiday>.csv --session-col session_id
```

Compare absolute/logratio/SNV on the same independent held-out sessions; report
per-class errors and the protocol, including clean-air adaptation. Improvement
is an experimental result, not guaranteed. This harness uses held-out-session
clean air; registry paired evaluation instead uses persisted training-baseline
fallbacks. State which protocol produced a number. Never select a model from
different internal holdouts or random adjacent rows alone.

## Then deploy the drift-robust model

Use Train preview → Stable plateaus → Balanced RF + logratio, augmentation/SNV
off initially, verified units and candidate staging on. Historical merge stays
on for compatible Idea 1 data; choose replacement for a controlled dataset-only
experiment. Independently compare and explicitly promote in Model Info.

Capture fresh live clean air after board/configuration/session changes,
deployment reload or server restart. `--live-baseline` is only a plot option.
Environment overrides do not retroactively convert a saved model or replace
explicit CSV request settings. See [training details](MODEL_IMPROVEMENTS.md),
[client options](../enose/client/README.md), and [PCB safety](PCB_ACQUISITION.md).
