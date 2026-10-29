# enose/client — Client commands and options

Reference for **3.3.0**. The hardware/API client runs separately from the
server; it needs neither Torch, Florence weights nor a GPU. Install it using
[Client quickstart](../../docs/CLIENT_QUICKSTART.md) or the
[Windows wheel guide](../../docs/CLIENT_WINDOWS_INSTALL.md).

`enose-client`, `python -m enose.client.main`, and, from an installed checkout,
`python scripts/run_client.py` use the same parser. Wheel-only installations
have no repository `scripts/` directory. Run `enose-client --help` to check
which options your installed version supports.

## Choose an acquisition source

The default is the **new PCB**, not the original ASCII/ADC board.

```bash
# New PCB: firmware-calibrated resistance, default ohms, 250000 baud
enose-client --port_enose /dev/ttyACM0 --server http://SERVER:8016

# Legacy: host ADC conversion, 9600 baud; use the experiment's RLOW
enose-client --sensor-backend legacy --port_enose /dev/ttyUSB0 --rlow 0.1 --server http://SERVER:8016

# Simulation: no serial ports or heater commands
enose-client --offline --server http://SERVER:8016

# Supplier Rs CSV: recorded readings, no USB or heating
enose-client --sensor-backend replay --sensor-csv /path/to/recording.csv --live --headless --server http://SERVER:8016
```

On Windows use COM ports and the active uv environment, for example:

```powershell
uv run --active --no-project enose-client --port_enose COM3 --server http://SERVER:8016
```

Environmental UART is optional for both physical boards. Without it the
22-key transport uses `T=21`, `H=49`, `CO2=400`, `H2S=0`, `CH2O=5`: these are
**unmeasured defaults**. A 17-feature model ignores them. PCB heater temperature
is separate metadata, never ambient `T`.

## Complete CLI reference

Defaults below come from `main.build_parser()` and `enose/config.py`.
Paths are relative to the client process's working directory.

### Connection and acquisition

| Option | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Print help and exit without connecting |
| `--server URL` | `http://localhost:8080` | Server base URL, without `/ui` |
| `--sensor-backend {pcb,legacy,replay}` | `pcb` | New binary USB CDC, original ASCII/ADC, or supplier CSV playback |
| `--port_enose PORT` | interactive selection | Physical serial port; replay must not use serial ports |
| `--port_UART PORT` | omitted | Optional separate environmental UART |
| `--baud_enose N` | PCB `250000`; legacy `9600` | Hardware serial baud, not sampling rate |
| `--baud_UART N` | `9600` | Environmental UART baud |
| `--resistance-unit {ohm,kohm,mohm}` | `ohm` | PCB/replay output units; `mohm` means megaohms here; does not rescale legacy conversion |
| `--rlow VALUE` | `ENOSE_RLOW`, otherwise `1.0` | Positive finite load resistance for **legacy only**, in the experiment's units |
| `--offline` | off | Synthetic sensors; incompatible with `--sensor-csv` |
| `--sensor-csv PATH` | omitted | Supplier Rs-mode CSV, required with replay; no labels inferred |
| `--replay-speed N` | `1.0` | CSV time multiplier; `0` removes time gating, while polling still limits reads |

`--resistance-unit` converts supplier `rs_ch*_ohm` values once. It never
establishes the units of an old model or arbitrary canonical `R1`–`R17` CSV.
Verified model/input units must match; unverified legacy units block PCB
prediction/history merging. Do not change units merely to bypass that check.

### Capture and vision deadlines

| Option | Default | Meaning |
|---|---|---|
| `--camera N` | `0` | Camera index; opened only for vision workflows |
| `--time N` | `60` | Recording duration, seconds |
| `--samples N` | `100` | Target unique readings per recording, not a guarantee |
| `--vision-timeout SECONDS` | `ENOSE_VISION_TIMEOUT`, otherwise `30` | Object-detection HTTP deadline |
| `--scene-timeout SECONDS` | `ENOSE_SCENE_TIMEOUT`, otherwise `60` | Scene-description HTTP deadline |

For a slow CPU VLM, try `--vision-timeout 180 --scene-timeout 300` initially.
These are request budgets, not measured latency guarantees; increasing them
does not speed up inference. Florence must be loaded on the server. A bottle's
appearance cannot establish whether its contents are acetone.

### Preflight and diagnostics

| Option | Default | Meaning |
|---|---|---|
| `--preflight` | off | Connect, inspect acquisition/server compatibility, print diagnostics, then exit; no camera/VLM inference |
| `--preflight-seconds SECONDS` | `3.0` | Observation period; serial/HTTP waits can extend total runtime |
| `--diagnostic-report PATH` | omitted | Save JSON with preflight or menu 13; refuses to overwrite an existing file |

Preflight reports units, identity, freshness, read/fault counters, sample rate
and model compatibility. Only the new PCB receives INFO/STATUS requests;
legacy has neither. Preflight never enables heating, but PCB disconnect still
requests heater-off. It is not a passive observer of an externally heated board.

```bash
enose-client --port_enose /dev/ttyACM0 --preflight --diagnostic-report outputs/preflight-01.json --server http://SERVER:8016
```

### PCB heater policy

| Option | Default | Meaning |
|---|---|---|
| `--heater-control {external,client}` | `external` | Preserve supplier-prepared PI settings, or explicitly enable heating here |
| `--heater-target-c CELSIUS` | `300` | Integer PI setpoint, written only when enabling through this client |
| `--heater-max-c CELSIUS` | `350` | Host trip temperature, **not a certified hardware limit** |
| `--heater-max-seconds SECONDS` | `1800` | Maximum host-monitored heating session |
| `--heater-shutdown-timeout SECONDS` | `240` | Controlled cooldown deadline including lock waiting; in-flight serial I/O may finish later |

Limits must be finite, with `0 < target < maximum <= 32767` and positive
deadlines. Software acceptance does not establish thermal safety. Client
heating requires calibrated, fault-free feedback and normal PI mode; fixed-duty
operation is unsupported. No flash settings, calibration or gains are written.

`external` never enables heating or changes the setpoint. Close the supplier's
tool before opening the same port here. Our client monitors existing heating
and requests shutdown on exit in **both** modes. `client` starts PI heating
after connection; it is refused in offline/replay/preflight modes. Menu 11
asks for `yes`. The local plot's **Enable PI heater** button enables directly
with the configured policy; there is no additional confirmation dialog.

On normal exit, exceptions, Ctrl+C/SIGINT, SIGTERM and supported SIGBREAK,
cleanup requests disable and waits for fresh confirmation of disabled state
**and zero duty**. Disable ramps down; it is not an emergency cut-off.
`OFF UNVERIFIED` means off was not confirmed. Zero duty does not mean cool.
USB loss, forced kill, host/power failure cannot guarantee off: the supplied
firmware documents no host-loss watchdog. See
[PCB safety limits](../../docs/PCB_ACQUISITION.md#conversion-and-configuration).

### Live display, publication and recording

| Option | Default | Meaning |
|---|---|---|
| `--live` | off | Skip menu and start live acquisition/publication |
| `--live-rate HZ` | `5.0` | Requested client polling/publication cadence, not MCU stream/export rate |
| `--live-window SECONDS` | `60.0` | Local plot rolling window |
| `--live-baseline` | off | Subtract initial baseline **on local plot only**; does not set server clean-air baseline |
| `--live-classify` | off | Request predictions for published samples |
| `--live-label LABEL` | omitted | Operator session tag; does not automatically train the model |
| `--live-http-timeout SECONDS` | `2.0` | Live request deadline, separate from vision deadlines |
| `--headless` | off | No local plot; stream to browser. Agg also selects headless behavior |
| `--prediction-confirmations N` | `2` | Consecutive matching network predictions before local confirmation; adds delay |
| `--record-session PATH` | omitted | Save canonical readings to NPZ with live mode or menu 8 |

Local acquisition/recording runs before HTTP. A bounded latest-sample queue
prevents slow HTTP from blocking it; skipped network frames remain locally
without predictions. This does not guarantee recording every MCU frame.
NPZ includes units/acquisition metadata, network latency, candidate predictions
and confirmation flags. Graceful exit saves it; forced termination may not.

```bash
enose-client --port_enose /dev/ttyACM0 --live --headless --live-classify --record-session outputs/session-01.npz --server http://SERVER:8016
```

Baseline-relative models need **Live → Set clean-air baseline** before live
inference/online learning. The baseline binds to board/configuration,
connection segment and publisher session. Recapture after these change or the
server restarts. Local `--live-baseline` is unrelated.

## Menu and browser equivalents

Interactive workflows connect the selected sensor first; use `--offline`
for remote CSV/manual/model functions when no hardware is connected.

| Menu | Action | Browser equivalent / requirement |
|---|---|---|
| 1 | Full vision + smell training cycle | Camera + sensor or simulation; Florence required; not the same as browser CSV training |
| 2 | Vision only | Camera; Florence required |
| 3 | Current smell | Classify for manual values; Live for streamed samples |
| 4 | Model information | Model Info |
| 5 | CSV learning | Train; both default to staging, historical merge remains default |
| 6 | Manual input (historical “22-feature” menu name) | Classify accepts 17 or 22 values |
| 7 | Visualizations and analysis | Metrics/drift in UI; full plot generation via API |
| 8 | Live stream | Live displays it; browser cannot open client USB ports |
| 9 | Reclassify recorded NPZ | Packaged replay; no browser NPZ upload |
| 10 | Change legacy RLOW | No browser calibration control; refused for PCB/replay |
| 11 | Start monitored PCB PI heating | Explicit local confirmation; no HTTP/browser enable control |
| 12 | Stop heater and verify zero duty | Local CLI/plot only; browser heater diagnostics are read-only |
| 13 | Preflight / optional JSON report | Browser downloads server/live diagnostics, not local serial preflight |
| 0 | Exit with cleanup | Closing browser does not stop client/heater |

Browser Train also supports extraction preview, row annotations, recording
date/identity and plateau parameters. Model Info supports paired independent
comparison, promotion and rollback. Not all of these are CLI flags. Direct API
training defaults to immediate update for compatibility; staging is separate
from history merging. See the
[training workflow](../../docs/PCB_ACQUISITION.md#preview-annotation-evaluation-and-deployment).

## Two kinds of replay

`--sensor-backend replay --sensor-csv PATH` streams supplier Rs CSV as a sensor.
NPZ replay reclassifies canonical stored values without unit conversion:

```bash
python -m enose.client.replay outputs/session-01.npz --server http://SERVER:8016 --rate 10 --limit 500
```

NPZ options: positional `path`, `--server`, `--rate` (10 Hz; 0 = no throttling),
`--limit` (0 = all), `--timeout` (5 s), `--quiet`. Menu 9 and checkout wrapper
`scripts/replay.py` use this packaged implementation.

## Modules

`pcb.py` / `pcb_protocol.py` handle USB CDC; `pcb_replay.py` supplies the factory
and CSV replay. `sensors.py` retains legacy/synthetic acquisition. `heater.py`
/ `lifecycle.py` implement monitored PI and cleanup; `preflight.py` reports
diagnostics. `api.py`, `pipeline.py`, `webcam.py`, `live.py`, `session.py`,
`replay.py`, `main.py` handle HTTP, workflows, camera, streaming and entry points.

For serial permissions and headless Docker mapping see
[Client quickstart](../../docs/CLIENT_QUICKSTART.md). Physical USB timing,
thermal behavior and native Windows execution still need field validation.
