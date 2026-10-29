# E-Nose — Client quickstart (3.3.0)

Use this guide to connect a sensor/camera to an existing lab server. Hosting
that server is a separate installation: see [SETUP.md](SETUP.md). Keep client
and server environments separate.

Replace `http://SERVER:8016` below with the base URL supplied by your admin,
without `/ui`. Check `http://SERVER:8016/health` before connecting hardware.
All [client options and menu items](../enose/client/README.md#complete-cli-reference)
are documented in the client README.

## Browser only — no installation

Open `http://SERVER:8016/ui`.

| Tab | Current workflow |
|---|---|
| Classify | Paste 17 or 22 values; declare units matching the model |
| Train | Upload/preview CSV, verify labels and retained windows, train a staged candidate |
| Live | View samples published by a client, acquisition/heater diagnostics, capture clean-air baseline, run fast/stable predictions |
| Model Info | Inspect classes and validation warnings; compare candidate and active model on independent data, promote or roll back |

The browser does not connect to USB or control heating. Closing it does not
stop the acquisition client. Training a candidate does not change predictions
until you explicitly promote it.

## Install the lightweight client

No Torch, CUDA, Florence weights or server packages are needed here.
Python 3.10–3.12, 64-bit, is the recommended shared setup range.

### Windows PowerShell

Use the [wheel installation guide](CLIENT_WINDOWS_INSTALL.md) for a no-clone
installation. For an editable checkout:

```powershell
git clone https://github.com/Arcwy0/e-nose.git
Set-Location e-nose
uv venv --python 3.11 .venv-client
.\.venv-client\Scripts\Activate.ps1
uv pip install --python .venv-client -e ".[client]"
uv pip check --python .venv-client
uv run --active --no-project enose-client --help
```

Run Windows examples below with `uv run --active --no-project` before the
`enose-client` command, and replace Linux device paths with `COM3`, for example:

```powershell
uv run --active --no-project enose-client --port_enose COM3 --server http://SERVER:8016
```

### Linux Bash — client machine

```bash
git clone https://github.com/Arcwy0/e-nose.git
cd e-nose
python3 -m venv .venv-client
source .venv-client/bin/activate
python -m pip install -e '.[client]'
python -m pip check
enose-client --help
```

On the remote Docker server use the container workflow rather than creating a
host Python environment. From an installed checkout,
`python scripts/run_client.py` is equivalent to `enose-client`. Wheel-only
users have no `scripts/` directory; use installed commands or modules.

## First software-only test

```bash
enose-client --offline --server http://SERVER:8016
```

Choose menu 4 for model information or 6 for manual 17/22-value classification.
Synthetic signals test plumbing, not real-smell accuracy. Vision options still
require a real camera and a server with Florence loaded.

## Select the correct board

```bash
# New PCB — default backend; calibrated ohms, 250000 baud
enose-client --port_enose /dev/ttyACM0 --server http://SERVER:8016

# Original board — ASCII ADC, 9600 baud, experiment-specific RLOW
enose-client --sensor-backend legacy --port_enose /dev/ttyUSB0 --rlow 0.1 --server http://SERVER:8016

# Legacy with an optional environmental board
enose-client --sensor-backend legacy --port_enose /dev/ttyUSB0 --port_UART /dev/ttyUSB1 --rlow 0.1 --server http://SERVER:8016
```

R1–R17 alone are supported. With no environmental UART the client sends
explicitly unmeasured defaults; a 17-feature model ignores them. The PCB's
~300°C heater temperature is **not ambient T**. Its firmware already converts
resistances; do not apply RLOW or the old ADC formula again.

The PCB default output is ohms. `--resistance-unit kohm` or `mohm` converts the
PCB/replay output, but never proves the units of historical data. Input/model
units must match. See [PCB acquisition](PCB_ACQUISITION.md).

### Preflight before measurements

```bash
enose-client --port_enose /dev/ttyACM0 --preflight --diagnostic-report outputs/preflight-01.json --server http://SERVER:8016
```

Use `--sensor-backend legacy` for the original board. No binary INFO/STATUS is
sent to legacy hardware. Reports include units, freshness, faults and model
compatibility; existing report files are not overwritten. PCB preflight does
not enable heat, but disconnect requests controlled heater-off.

### Heating — choose one owner

Default `--heater-control external` preserves supplier-prepared PI settings;
it does not start heat or write a setpoint. Close the supplier tool before our
client opens the same port. For explicit client-owned heating:

```bash
enose-client --port_enose /dev/ttyACM0 --heater-control client --heater-target-c 300 --heater-max-c 350 --server http://SERVER:8016
```

This enables heating after successful connection and safety checks. Menu 11
also enables after confirmation; menu 12 disables and verifies zero duty.
Both modes request disable on exit. Wait for firmware confirmation; disable
ramps down and zero duty is not a cool sensor. The 350°C host guard is not a
certified safe limit. USB/host loss and forced kills cannot guarantee shutdown:
this firmware has no documented host-loss watchdog. Read the
[full heater policy](../enose/client/README.md#pcb-heater-policy) before use.

## Live display and recording

```bash
enose-client --port_enose /dev/ttyACM0 --live --headless --live-classify --record-session outputs/session-01.npz --server http://SERVER:8016
```

Omit `--headless` for a local Matplotlib window. `--live-rate 5` requests a
5 Hz polling cadence; it is not the MCU's sampling rate. Local readings are
recorded before HTTP. Slow network inference may skip published frames without
blocking local capture. `--live-http-timeout 2` sets its separate deadline;
`--prediction-confirmations 2` requires repeated local labels.

For a baseline-relative model, first stream stable clean air, then click
**Live → Set clean-air baseline**. Recapture after reconnect, board/configuration
or session changes, model reload or server restart. `--live-baseline` only
changes the local plot and cannot substitute for this step. Fast predictions
are provisional; stable windows and independent sessions are the evaluation
path. Stop with Ctrl+C and allow cleanup/NPZ saving to finish.

## Camera workflows and CPU VLM deadlines

```bash
# Camera + simulated sensors, menu 2 for vision only
enose-client --offline --camera 0 --vision-timeout 180 --scene-timeout 300 --server http://SERVER:8016

# Camera + new PCB, menu 1 for the training cycle
enose-client --port_enose /dev/ttyACM0 --camera 0 --time 60 --samples 100 --vision-timeout 180 --scene-timeout 300 --server http://SERVER:8016
```

The defaults are 60 s capture and 100 target readings; short/slow recordings
need not reach that count. Timeout values above are initial CPU budgets, not
latency promises. A bottle/vial can be grounded visually; its chemical contents
cannot be established from appearance alone.

## Replay without hardware

Supplier **Rs-mode CSV**, streamed at recorded cadence:

```bash
enose-client --sensor-backend replay --sensor-csv /path/to/recording.csv --live --headless --server http://SERVER:8016
```

Saved **NPZ**, reclassified against the current model without unit conversion:

```bash
python -m enose.client.replay outputs/session-01.npz --server http://SERVER:8016 --rate 10 --limit 500
```

Menu 9 also replays NPZ. Neither replay mode actuates hardware. Unlabelled
supplier CSV supports viewing/replay, but training needs verified labels or
explicit source-row annotations.

## CSV training and deployment

1. In Train, upload CSV and declare actual units, date/session/recording identity.
2. Provide a label column or verified row annotations; Preview also works without labels.
3. For continuous recordings choose Stable plateaus, Balanced RF + baseline,
   log ratio, no augmentation/SNV. Inspect retained/rejected windows and reasons.
4. Leave historical merging on for normal Idea 1 updates. Turn it off for a
   one-dataset experiment; this affects candidate history, not current deployment.
5. Leave candidate staging on. Fit, then compare active/candidate on an independent
   labelled CSV in Model Info. Promote explicitly; keep the returned rollback ID.

A first deployment has no active comparison model; an unevaluated override is
explicit and is **not** evidence of accuracy. Impossible independent group
splits give null metrics, not optimistic row-split scores. Menu 5 also defaults
to staging; direct API defaults differ for backward compatibility. See
[model/training details](MODEL_IMPROVEMENTS.md).

## Headless client in Docker — Linux

```bash
docker build -t enose-client -f docker/Dockerfile-client .

# No hardware
docker run --rm -it --mount type=bind,source="$PWD",target=/workspace/repo --workdir /workspace/repo -e PYTHONPATH=/workspace/repo enose-client python scripts/run_client.py --offline --server http://SERVER:8016

# New PCB — only the USB device in use
docker run --rm -it --stop-timeout 300 --device=/dev/ttyACM0 --mount type=bind,source="$PWD",target=/workspace/repo --workdir /workspace/repo -e PYTHONPATH=/workspace/repo enose-client python scripts/run_client.py --port_enose /dev/ttyACM0 --live --headless --server http://SERVER:8016
```

The client image uses headless OpenCV/Matplotlib; it is not the server's version
lock. Native `[client]` is needed for interactive camera/plot windows. No
`--privileged` or GPU access is needed. A reachable remote server does not
require host networking; container `localhost` is not the host. Linux may need
`--add-host=host.docker.internal:host-gateway` for a server on the host. The
longer Docker stop grace permits controlled heater cooldown; it cannot guarantee
completion or replace a firmware interlock. Windows hardware should use the
native client, not assume COM ports appear inside Linux Docker.

## Troubleshooting

| Symptom | Check |
|---|---|
| Server unreachable | Base URL/port, VPN/firewall, `/health` from client PC |
| `classifier not fitted` after training | Candidate may only be staged; inspect Model Info and explicitly promote |
| Unit/baseline error | Verify actual units; capture a fresh same-board/session clean-air baseline |
| Serial busy | Close supplier tool/other clients; one process owns the port |
| Linux serial permission denied | Configure host serial-group/device access; do not broadly grant world-write access |
| VLM timeout/503 | Increase client deadlines for CPU; 503 may mean server started with `ENOSE_NO_VLM=1` |
| No local plot in Docker | Expected with Agg; use `--headless` and browser Live |
| `scripts/...` missing | Wheel-only install: use `enose-client` or packaged replay module |
| Heater `OFF UNVERIFIED` | Inspect hardware safely; never assume off from client exit alone |

Menu 0–13 and every startup flag are listed in the
[client reference](../enose/client/README.md). Setup/recovery details are in
[SETUP.md](SETUP.md), [PCB_ACQUISITION.md](PCB_ACQUISITION.md), and
[CHEMIST_GUIDE.md](CHEMIST_GUIDE.md).
