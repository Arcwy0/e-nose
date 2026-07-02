# Multi-day data-collection protocol (for drift validation)

To *prove* the classifier is robust to day-to-day sensor drift, we need recordings of the
**same smells on different days**, each with a clean-air baseline. This is the input the
cross-session eval harness (`scripts/eval_drift.py`) needs, and the data pattern the
drift-robust modes are designed for.

## Why

MOS resistances drift with sensor aging, temperature and humidity. A model trained and tested
on the *same* day looks perfect but tells you nothing about tomorrow. Only **cross-day** data
reveals the real failure — and lets us measure the fix (baseline-relative features).

## What to record

Per day (≥ 2 days, ideally 3–4; more days = stronger evidence):

1. **Clean-air baseline** — 30–60 s of clean air at the start, *before* any smell. Label `air`.
   This is the R₀ reference the drift-robust mode subtracts.
2. **Each target smell** — e.g. `rose`, `coffee` (pick ≥ 2, keep the set identical across days).
   ~30–60 s each. Purge with clean air between smells (the robot's PURGE phase, or just wait).
3. Keep the sensor rig, gain and wiring **unchanged** across days — we want to capture *natural*
   drift, not a hardware change.

Spread the days out (different mornings, ideally different ambient conditions) so real drift
accumulates between sessions.

## File format

One CSV (append days into it, or one CSV per day and concatenate). Required columns:

| Column | Meaning |
|---|---|
| `R1`…`R17` | resistance sensors (as the client already logs them) |
| `T,H,CO2,H2S,CH2O` | environmental channels |
| `smell_label` (or `Gas name`) | `air`, `rose`, `coffee`, … (lowercased) |
| **`session_id`** | **one distinct value per day/recording** — this is the grouping the harness splits on |

`session_id` is the critical field: it's how leave-one-session-out training simulates
"train on day 1–2, test on day 3". `timestamp` is nice to have but optional.

> The client already stamps a per-run `session_id` (`enose/client/live.py`), and
> `data/database_robodog_time_windows.csv` shows the exact schema (`session_id`, `segment_id`).

## How to validate once collected

```bash
# Absolute vs baseline-relative vs +SNV, leave-one-session-out:
docker exec -i enose-srv python scripts/eval_drift.py \
    --csv data/<your_multiday>.csv --session-col session_id
```

**Success:** the `logratio` / `logratio+snv` rows show materially higher cross-session balanced
accuracy than `none` (absolute). If `none` is already high, either drift is mild in your data or
sessions are too few/too similar — collect more spread-out days.

## Then deploy the drift-robust model

Train with the mode on (env override, no code change):

```bash
ENOSE_BASELINE_MODE=logratio ENOSE_SNV=1 python scripts/run_server.py
# …train via the UI / learn_from_csv on the multi-day CSV…
# at the start of each session, POST 30–60 s of clean air to /smell/baseline
```

The absolute pipeline stays the default (`ENOSE_BASELINE_MODE` unset) — nothing changes for
offline / same-day work.
