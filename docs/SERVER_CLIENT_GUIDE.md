# Server + client integration guide (3.3.0)

The server runs Florence-2 and the smell classifier; the acquisition client
owns USB, camera and heater monitoring. They communicate over HTTP. A server
restart cannot switch off a heater attached to another PC.

## Installation and entry points

Use [SETUP.md](SETUP.md) for the shared **server runtime lock**, pinned Florence
snapshot and CPU/CUDA setup. Use [CLIENT_QUICKSTART.md](CLIENT_QUICKSTART.md)
for a separate lightweight client environment. Do not mix native client OpenCV
with the server's headless OpenCV environment.

| Installation | Server | Client | NPZ replay |
|---|---|---|---|
| Installed wheel or editable package | `enose-server` | `enose-client` | `python -m enose.client.replay` |
| Installed repository checkout | `python scripts/run_server.py` | `python scripts/run_client.py` | `python scripts/replay.py` |

The wheel does not include repository `scripts/`. Docker server/client images
expect a source mount rather than an installed checkout baked into the image.

## Server launch options

Bash, from the installed server environment or server container:

```bash
# Smell only; vision endpoints return 503
ENOSE_NO_VLM=1 enose-server --host 127.0.0.1 --port 8016

# Full server; checkout verifier loads real weights and generates a caption
python scripts/verify_server_environment.py --vlm
ENOSE_NO_VLM=0 enose-server --host 0.0.0.0 --port 8016
```

Windows PowerShell, active server uv environment:

```powershell
$env:ENOSE_NO_VLM = "0"
uv run --active --no-project python scripts/verify_server_environment.py --vlm --device cpu
uv run --active --no-project enose-server --host 127.0.0.1 --port 8016
```

Set `$env:ENOSE_NO_VLM = "1"` for smell-only mode. `--device cpu` belongs to
the verifier, not the server launcher; startup auto-selects CPU when no CUDA
GPU is available. CPU inference uses host RAM and may need longer client
request deadlines. Full startup failure aborts the server rather than silently
pretending vision is available.

| Server flag | Default | Effect |
|---|---|---|
| `--host` | `SERVER_HOST`, otherwise `0.0.0.0` | Bind address; `127.0.0.1` is local-only |
| `--port` | `SERVER_PORT`, otherwise `8080` | Listen port; choose another for occupied-port errors |
| `--reload` | off | Development reload; resets live state and reloads models |
| `--workers` | `1` | Must remain 1: buffers, baselines and deployment locks are process-local |
| `-h`, `--help` | — | Print launcher help |

The browser is at `http://localhost:8016/ui` for these examples. Remote clients
use the server's actual IP, not `0.0.0.0`. The server API has no built-in
authentication: use a trusted private network/VPN/authenticated proxy. Do not
publish write/training/deployment routes on an unauthenticated public tunnel.

### Environment and paths

| Setting | Default | Scope |
|---|---|---|
| `ENOSE_NO_VLM` | `0` | `1` skips Florence |
| `ENOSE_CLASSIFIER` | `balanced_rf` | Startup loader backend; `xgb` selects XGBTabularClassifier |
| `ENOSE_BASELINE_MODE`, `ENOSE_SNV` | `none`, off | Defaults for new classifier config only; saved model/explicit CSV settings take precedence |
| `SERVER_HOST`, `SERVER_PORT` | `0.0.0.0`, `8080` | Launcher defaults overridden by flags |
| `ENOSE_RLOW` | `1.0` | Legacy **client**, not a server-side rescaling control |
| `ENOSE_VISION_TIMEOUT`, `ENOSE_SCENE_TIMEOUT` | `30`, `60` | Client HTTP deadlines, seconds |

`.env` is not auto-loaded. Export variables in the process or pass Docker
`--env-file`. Paths below come from `enose/config.py`, relative to the process
working directory; they are not currently server CLI flags:

| Path | Purpose |
|---|---|
| `model/Florence-2-Large/` | Separate Florence snapshot; one-time tokenizer recovery needs write access |
| `trained_models/smell_classifier_sgd_latest.joblib` | Active smell model |
| `database_robodog.csv` | Active canonical training history; preserve with the model |
| `trained_models/candidates/<id>/` | Candidate model/history/manifest |
| `trained_models/candidates/rollbacks/<id>/` | Recoverable deployment snapshots |

Rollback manifests bind absolute target paths. After migration to a different
working directory, old snapshots are archival; do not edit manifests to bypass
target checks. Make new snapshots in the new location.

### Docker server

Bash, repository root; source, weights and runtime state stay on the host:

```bash
docker build -t enose-server -f docker/Dockerfile-cu124 .
docker run --rm --name enose-server-run --gpus all -p 8016:8080 --mount type=bind,source="$PWD",target=/workspace/repo --workdir /workspace/repo -e PYTHONPATH=/workspace/repo -e ENOSE_NO_VLM=0 enose-server python scripts/run_server.py --workers 1
```

For CPU-only execution omit `--gpus all`, use `--runtime runc`, and add
`-e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=`. Set
`ENOSE_NO_VLM=1` for smell-only CPU mode; leave it `0` to test Florence on CPU.
Omitting `--gpus` alone may not isolate a host with an NVIDIA default runtime.
Do not grant the server USB/robot devices; acquisition belongs to the client.
Keep persistent files mounted before removing/replacing a container.

## Client launch and physical ownership

See the [complete flag/menu reference](../enose/client/README.md#complete-cli-reference).
New PCB is default, firmware-calibrated ohms at 250000 baud. Original hardware
requires `--sensor-backend legacy`, usually 9600 baud and experiment-specific
RLOW. Both can omit `--port_UART` for resistance-only operation.

```bash
# No serial/heater commands
enose-client --offline --server http://SERVER:8016

# New PCB with local recording and browser-only display
enose-client --port_enose /dev/ttyACM0 --live --headless --live-classify --record-session outputs/session-01.npz --server http://SERVER:8016

# Original board, no environmental UART
enose-client --sensor-backend legacy --port_enose /dev/ttyUSB0 --rlow 0.1 --server http://SERVER:8016
```

On Windows run through `uv run --active --no-project` and use COM port names.
The client connects the selected sensor before opening its menu; use
`--offline` for remote CSV/model/manual work without hardware.

`--heater-control external` is default: preserve supplier PI settings, no
automatic enable/setpoint write. Close the supplier tool before our client
opens the port. Explicit client heating uses `--heater-control client` with
configurable 300°C target and 350°C **host** trip threshold. This threshold is
not a certified safe temperature. Both modes request disable on exit and verify
fresh disabled/zero-duty state. Graceful signals are handled, but no host-loss
watchdog is documented; USB loss/forced kill/host failure cannot guarantee off.
See [PCB acquisition and safety](PCB_ACQUISITION.md). The server/browser cannot
enable heating. Legacy/replay never receive heater or binary status commands.

Preflight (`--preflight --diagnostic-report outputs/preflight-01.json`) checks
acquisition and model compatibility without camera/VLM inference. It does not
enable heating; PCB disconnect still requests off. JSON reports refuse overwrite.

## Live inference and recording

Acquisition callbacks record canonical local readings before HTTP. A separate
bounded latest-sample queue publishes to `/sensor/live/push`. Slow requests
may skip publication/predictions, not block local sampling. Polling at 5 Hz
does not mean every 10 Hz MCU frame is retained. `--prediction-confirmations 2`
adds repeated-label confirmation locally, not a new server accuracy estimate.

The transport uses R1–R17 plus optional environmental channels. Without an env
UART, default room values are marked unmeasured; 17-feature models ignore them.
PCB `temp_actual_c`, duty and fault flags are diagnostics, never ambient `T`.
Verified units must match across sample/model/history. Canonical R CSV values
are not rescaled just because an API unit was selected.

For baseline-relative models:

1. Stream stable clean air from the actual board/session.
2. Capture **Live → Set clean-air baseline**; local `--live-baseline` only
   changes the plot and is not this operation.
3. Approach the analyte and use fast provisional or strict stable-window inference.
4. Check recovery before the next exposure. Reconnects, changed acquisition
   identity/session and deployment/server reloads require a fresh baseline.

Local NPZ saves acquisition/units, network latency and confirmation metadata
without pickle. `--live-label` tags data but does not train. Reclassify via
menu 9 or `python -m enose.client.replay session.npz --server http://SERVER:8016`.
Supplier CSV sensor playback instead uses `--sensor-backend replay --sensor-csv`;
it never establishes smell labels or measured physical-board identity.

## Training and candidate deployment

### UI/menu defaults versus API defaults

| Setting | Browser Train / client menu 5 | Direct CSV API default |
|---|---|---|
| Data profile | Stable plateaus | Raw rows |
| Backend | Balanced RF | Balanced RF |
| Baseline | Log ratio | None |
| Augmentation | Off | On, 5 copies |
| SNV | Off | Off |
| Merge history | On | On |
| Stage candidate | On | Off: immediate update |

These API defaults preserve old callers; request recommended settings explicitly.
History policy and deployment policy are independent. `merge_history=false`
trains on this upload only; with staging enabled it does **not** replace the
active pair until promotion. Online Idea 1 updates retain historical learning.

Browser workflow:

1. Upload CSV, select units, recording date/session/recording identity and label column.
   R1–R17 suffice; supplier `rs_ch0_ohm`…`rs_ch16_ohm` are converted once.
2. Supply verified labels or source-row annotations. Rows are zero-based,
   header excluded, `[start_row,end_row)`. Unlabelled gaps are not classes.
3. Preview traces, retained/rejected windows and reasons before fitting. Unlabelled
   preview works; unlabelled training does not. Heater temperature is diagnostic only.
4. Train a candidate. The active model/history remains unchanged.
5. In Model Info, upload a separately collected labelled CSV and explicitly confirm
   independence. Evaluate both models on exactly the same rows.
6. Promote explicitly and retain the rollback ID. First deployment or unavailable
   evidence requires an explicit unevaluated override, not an accuracy claim.

Exact R+label overlap with either history is rejected. The operator must still
establish exposure/day independence; overlap screening cannot prove it.
Paired evaluation uses persisted training-baseline fallbacks, not test-label
baseline adaptation or the active live baseline. Evidence is bound to hashes of
both persisted model/history pairs; retraining invalidates stale comparisons.
Known regression needs a separate explicit override.

Within a fit, group-disjoint holdout and calibration keep exposures separate;
learned preprocessing is fit only on training folds. If no class-complete group
split exists, accuracy is null (`not_estimable`), not a row-wise fallback.
Final fitting happens after holdout diagnostics are frozen. Different fits'
training scores are not a paired comparison or proof of improvement.

Promotion snapshots the old pair; rollback restores it. Two-file replacement
is recoverable, **not crash-atomic**. Keep one worker and back up the model,
root history and candidate/snapshot tree together. Load only trusted joblib
files; they are executable Python serialization, not an untrusted upload format.

## API reference

Use the running server's `/docs` for full current request schemas. Selected routes:

| Method | Route | Purpose |
|---|---|---|
| GET | `/`, `/health`, `/smell/model_info` | Version, readiness and model/validation metadata |
| POST | `/smell/classify` | Named resistance fields, optional env, acquisition metadata and session ID |
| POST | `/smell/test_console` | 17 or 22 comma-separated values; declare `resistance_unit` for verified-unit models |
| POST | `/smell/debug_input` | Per-sample diagnostics |
| POST | `/sensor/live/push` | `sample`, optional `client_t`, `label`, `classify`, `session_id`, `acquisition` |
| GET | `/sensor/live/recent`, `/sensor/live/stats` | Incremental samples and acquisition/baseline diagnostics |
| DELETE | `/sensor/live/clear` | Clear live ring buffer |
| POST | `/smell/baseline`, `/smell/baseline/live` | Capture clean-air batch or live baseline with identity binding |
| POST | `/smell/classify_window`, `/smell/classify_stable` | Provisional fast bins or strict stable median |
| GET | `/smell/recovery`, `/smell/drift` | Recovery and live-vs-training drift diagnostics |
| POST | `/smell/preview_csv`, `/smell/learn_from_csv` | Shared CSV preparation; preview does not fit/change active state |
| POST | `/smell/online_learning` | Confirmed stable Idea 1 batch with optional acquisition/session/provenance |
| GET | `/smell/models/candidates` | List candidate manifests |
| POST | `/smell/models/evaluate` | `candidate_id`, `csv_data`, `target_column`, `resistance_unit`, `independent_confirmed` |
| POST | `/smell/models/promote` | `candidate_id`, optional `allow_unevaluated`, `allow_regression` |
| POST | `/smell/models/rollback` | `snapshot_id` |
| GET | `/smell/visualize_data`, `/smell/analyze_data`, `/smell/environmental_analysis` | Plots and analysis |

Example recommended CSV request from Bash (`jq` is an optional shell helper,
not a Python dependency):

```bash
jq -n --rawfile csv recording.csv '{csv_data:$csv,target_column:"smell_label",resistance_unit:"ohm",training_profile:"plateau",classifier_backend:"balanced_rf",baseline_mode:"logratio",use_augmentation:false,n_augmentations:0,merge_history:true,stage_candidate:true}' | curl -X POST http://SERVER:8016/smell/learn_from_csv -H 'Content-Type: application/json' --data-binary @-
```

For canonical R columns include their actual `resistance_unit` column; the
request unit controls supplier-ohm conversion, not an arbitrary unit rewrite.
For preview send the same body to `/smell/preview_csv`. Plateau options default
to 60 s window/stride, 600 s baseline tail, final 25% exposure, 30 s end margin,
relative slope limit 0.002. Short exposures can yield no accepted windows.
Set them from experimental evidence, not merely to force acceptance.

The Live tab polls the ring buffer, not a WebSocket. OOD/drift flags describe
training-distribution mismatch, not chemical proof. Current software checks do
not establish physical USB/thermal behavior or cross-day recognition accuracy;
see [TESTING_GUIDE.md](TESTING_GUIDE.md).
