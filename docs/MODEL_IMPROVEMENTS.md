# Stable-plateau and drift-robust training

## Why raw-row training looked accurate but failed live

The recorder labels every row with the currently selected gas. That includes
the response transient just after air changes to an analyte and the recovery
transient after it changes back to air. Adjacent rows are also nearly copies of
one another. A random row split therefore leaks the same exposure trajectory
into training and testing and can report 100% while failing on a later session.

The stable-plateau profile fixes the data side of this problem. It:

- keeps clean-air windows immediately **before** each exposure;
- keeps only late, low-slope analyte windows;
- excludes the recovery portion labelled as air;
- summarizes each minute by its median;
- caps long exposures at 30 evenly spaced windows per phase;
- splits recorder gaps longer than five minutes;
- assigns `session_id` and `exposure_id` so adjacent samples stay together;
- merges textual aliases that share a numeric gas class, such as
  `amyl alcohol` and `amyl` in the supplied dataset.

The source CSV is never modified.

## Recommended UI settings

Open `/ui`, select **Train**, and use:

| Setting | Recommended value |
|---|---|
| Training data profile | Stable plateaus |
| Classifier | Balanced RF + baseline |
| Baseline representation | Log ratio to clean air |
| Data augmentation | Off |
| SNV | Off |
| Merge with historical data | On for normal Idea 1 learning; off for a controlled one-dataset experiment |

Stable-plateau training requires `Timestamp`, `Gas name`, and `R1`–`R17`.
Environmental columns are optional. A model trained without usable
environmental data automatically uses only the 17 resistance channels.

If the server's existing canonical history was created with the legacy
every-row workflow, make the first stable-plateau upload with **Merge history
off**. That one-time replacement prevents the old transient rows from being
mixed back into the cleaned dataset. Leave history merging on for subsequent
Idea 1 updates.

The two-stage response-shape classifier remains in the menu as an experimental
comparison. It separates air reliably in the available later recording, but it
confuses ethanol with acetone, so it is not the deployment default.

## Required live workflow

A baseline-relative model needs a baseline from the **same session**:

1. Put the array in clean air and allow it to recover and stabilize.
2. Stream readings from the client to the server.
3. In the UI's **Live** tab, click **Set clean-air baseline**. The server rejects
   a window that is too short or still changing.
4. Approach the sample, wait until the response stabilizes, then click
   **Classify stable window**. This classifies the median of the stable window,
   not a noisy instantaneous row.

For a quicker robot decision, choose a 5, 10, 15, or 30 second **Fast window**
and click **Fast prediction**. It averages probabilities from short median
bins, reports whether the signal is still changing, and returns `unknown` when
the top confidence or top-two margin is too small. This mode is deliberately
labelled provisional: the present recordings do not support a reliable
early-response identity model. The stable-window result remains the
measurement to use for quantitative evaluation.

Before starting another exposure, click **Check recovery**. It compares the
80th percentile of `abs(log(R/R0))` with the captured clean-air baseline and
also requires a low slope. This prevents a recovery transient from being
treated as clean air for the next sample.

Idea-1 online learning now applies the same discipline. A capture is reduced
to at most five ordered, zero-aware median windows; chunks with fewer than 12
active resistance channels or excessive relative MAD are rejected. For a
baseline-relative model, online learning is rejected until today's clean-air
baseline has been captured. That baseline is added as an `air` row with the
same `exposure_id` as the vision-confirmed odor, so retraining constructs the
correct session-local response. Raw frames remain in the client's local log,
while the persistent classifier history stores the compact stable windows.

The corresponding API endpoints are:

```text
POST /smell/baseline/live?window=60&max_relative_slope=0.002&min_samples=10
POST /smell/classify_window?window=15&bin_seconds=5&min_confidence=0.45&min_margin=0.10
POST /smell/classify_stable?window=60&max_relative_slope=0.002&min_samples=10
GET  /smell/recovery?window=15&response_threshold=0.12
```

`POST /smell/baseline` is also available when a client already has a batch of
clean-air readings. Sending only `R1`–`R17` is supported.

## Independent benchmark on the supplied recordings

Run the benchmark inside the server image:

```bash
docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app \
  enose-server:latest python scripts/eval_plateau.py \
  --train data/database_robodog_time.csv \
  --test data/dataset_robodog_online.csv
```

The earlier February recording (all four of its classes) is used only for
training and the September recording only for testing. On the 26 extracted
September windows:

| Model | Correct | Accuracy | Balanced accuracy | Main error |
|---|---:|---:|---:|---|
| Absolute-resistance RF | 8/26 | 30.8% | 33.3% | Predicts everything as air |
| Log-ratio RF | 18/26 | 69.2% | 54.2% | All five acetone windows become ethanol |
| Experimental two-stage | 13/26 | 50.0% | 66.7% | All 13 ethanol windows become acetone |

The log-ratio RF is materially more stable than the old absolute model, but the
remaining acetone/ethanol error is not solved by changing classifiers alone.
The September acetone response magnitude is much closer to February ethanol
than to the weak February acetone exposures. Likely causes include different
concentration/flow/distance, incomplete recovery, sensor drift, or inconsistent
label timing.

## Audit of all four historical exports

The two additional 2025 exports were audited together with the March 2026 and
online-test files. They contain 464,111 rows in total, but 116,400 rows are
duplicates because later recorder exports include much of an earlier export.
After deduplication there are 347,711 unique rows and only **34 independent
analyte exposure episodes across 15 days**:

| Label | Independent exposures |
|---|---:|
| ethanol | 10 |
| acetone | 8 |
| IPA | 6 |
| coffee_573 | 5 |
| petroleum | 3 |
| amyl | 2 |

Only 10 exposures have a preceding recorded clean-air baseline. The other 24
start with an analyte label and can only use their first seconds as an onset
proxy. This is why adding all rows does not produce the improvement suggested
by the file sizes.

`database_robodog_05_09_25.csv` also contains 6,180 variable-width rows where
UART environmental fragments appear before the gas name. A normal pandas load
silently interprets numeric fragments as smell labels. The server now detects
this recorder format, recovers the rightmost textual gas label, and marks the
environmental fields unreliable. Both 2025 files also use zero as a
"multiplexed channel not updated" placeholder; only 73.8% and 76.0% of their
rows respectively contain at least 12 positive R readings. Window medians now
ignore these zeros, and every upload returns an `input_quality` report.

## Early-response latency experiment

Run the reproducible grouped benchmark with all four exports:

```bash
python scripts/eval_latency.py \
  data/database_robodog_time.csv \
  data/dataset_robodog_online.csv \
  /path/to/database_robodog_05_09_25.csv \
  /path/to/database_robodog_07_10_25.csv
```

Leave-one-exposure-out results (no adjacent-row leakage) were:

| Response time | Extra Trees accuracy | RBF-SVM accuracy |
|---:|---:|---:|
| 20 s | 26.5% | 26.5% |
| 30 s | 23.5% | 23.5% |
| 60 s | 29.4% | **38.2%** |
| 120 s | **35.3%** | 32.4% |

For acetone versus ethanol alone, the best accuracy was 55.6% at 60 or 120
seconds. These results reject deployment of the experimental early-response
model for now. They also show that model architecture is not the only remaining
problem: the training episodes do not consistently connect clean air, exposure
onset, concentration, plateau, and recovery.

The server still supports low operational latency through probability
aggregation. In the Docker end-to-end smoke test, a batched five-second window
took a median **214 ms** of server compute after acquisition. Low-confidence
results abstain instead of forcing an odor.
This improves stability and decision safety, but it must not be reported as an
accuracy improvement until a held-out acquisition validates it.

## Next data-collection step

Collect at least 5–10 complete repetitions per analyte across several days,
using the same protocol and logging these fields: `session_id`, `exposure_id`,
concentration, source distance, flow, start/end event, and RLOW. Each repetition
should contain stable clean air, exposure onset, a sufficiently long saturated
plateau, removal, and full recovery. Keep the raw sequence for future temporal
models, but train the current point model on the extracted stable windows.

Evaluate by leaving out complete days or exposure repetitions. Do not use a
random row split as the scientific result. Once this dataset exists, compare
the log-ratio RF against a temporal model on complete sequences.

The Model Info page now states the validation strategy. If the available
exposures cannot produce a group-disjoint holdout containing every class, the
UI warns that its row-split metric is optimistic. The complete confusion matrix
is still shown, but it is not a cross-day performance claim.
