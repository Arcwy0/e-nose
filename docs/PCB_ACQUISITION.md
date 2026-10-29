# PCB acquisition, legacy compatibility, and recorded-data replay

The supported client entry points (`scripts/run_client.py` / `enose-client`)
now default to the new USB CDC PCB. The robot trainer's ROS2 launcher uses the
same factory and default. Simulation remains synthetic and does not open ports.
The old standalone `client_server.py` remains a legacy implementation; use the
package client for the new backend. This workflow is included in release 3.3.0;
previously published 3.2.2 wheels do not include it.

## Select acquisition

Commands below run in the installed client environment. The server can continue
running remotely in Docker; it does not need access to the USB port.

```bash
# New PCB, R1-R17 only; default backend and output units
python scripts/run_client.py --port_enose /dev/ttyACM0 --server http://SERVER:8016

# Old ASCII/ADC board: original RLOW transform and 9600 baud
python scripts/run_client.py --sensor-backend legacy --port_enose /dev/ttyUSB0 --rlow 0.1 --server http://SERVER:8016

# Replay the desktop tool's Rs-mode CSV at its recorded cadence; no USB access
python scripts/run_client.py --sensor-backend replay --sensor-csv /path/to/E-NOSE-recording.csv --live --server http://SERVER:8016
```

Windows PowerShell uses COM ports, for example:

```powershell
uv run --active --no-project python scripts/run_client.py --port_enose COM3 --server http://SERVER:8016
uv run --active --no-project python scripts/run_client.py --sensor-backend legacy --port_enose COM3 --rlow 0.1 --server http://SERVER:8016
```

The active environment must already contain the documented client dependencies.
No PyQt6, pyqtgraph, firmware-flashing tools, or new runtime dependencies are
required. Server dependency locks and CUDA/VLM libraries are unchanged.
Restart the server with the updated source for metadata/unit validation and UI
changes. The repository Dockerfile bind-mounts source at runtime, so no new
dependency layer is needed. Custom images that copy source must be rebuilt.

| Option | Meaning |
|---|---|
| `--sensor-backend pcb` | Default: USB CDC binary protocol, 250000 baud |
| `--sensor-backend legacy` | Original text UART/ADC reader, 9600 baud |
| `--sensor-backend replay --sensor-csv PATH` | Read resistance-mode desktop CSV without hardware |
| `--resistance-unit ohm/kohm/mohm` | PCB/replay output unit, default ohm; never changes legacy RLOW conversion |
| `--baud_enose N` | Override hardware backend's baud setting |
| `--port_UART PORT` | Optional separate environmental UART, unchanged 9600 default |
| `--replay-speed N` | Recorded time multiplier; 1 = original cadence, 0 = one row per read without time gating |
| `--offline` | Original synthetic smell profiles; not real-data replay |
| `--preflight --diagnostic-report PATH` | Backend-aware diagnostics then exit; refuses to overwrite report |
| `--heater-control external` | Default: preserve supplier-configured PI, never enable/write setpoint; monitor existing heat |
| `--heater-control client` | Explicitly enable normal PI heating; new PCB only |
| `--heater-target-c 300 --heater-max-c 350` | Configurable target and conservative host trip temperature, deg C |
| `--heater-max-seconds 1800` | Host-monitored maximum heating session duration, seconds |
| `--heater-shutdown-timeout 240` | Cooldown deadline including lock wait; serial I/O can finish later |
| `--vision-timeout 30 --scene-timeout 60` | Configurable HTTP deadlines in seconds; raise explicitly for CPU vision |
| `--headless` | Publish live without a local plot; also selected automatically on Agg |
| `--live-http-timeout 2 --prediction-confirmations 2` | Network deadline and local consecutive-prediction confirmation |

`--live-classify` requests predictions only when a fitted model with compatible
verified units is available. `--record-session PATH` saves canonical samples
and acquisition metadata. Source CSV playback and `scripts/replay.py` (NPZ
reclassification) are separate workflows. NPZ replay never converts units again.
Recorded acquisition ages remain provenance; NPZ replay is not a live freshness
measurement. Do not combine `--offline` and `--sensor-csv`.

## Conversion and configuration

The supported binary protocol uses frame header A5 5A, VERSION=0x01, little-endian
fields, and no application checksum. Current measurement payloads contain 17
integer resistance values in ohms at nominal 10 Hz. Documented older binary
payload variants are accepted when their value mode is established. The desktop
export interval can be much slower than the physical stream.

The host requests INFO/STATUS, validates 17 channels, requests runtime Rs mode,
and starts streaming. ACKs and data may arrive in either order. The reader
continuously drains fragmented/coalesced traffic independently of prediction
and network timing. Consumers receive the latest unique valid frame, not a
growing FIFO of old readings. Short recordings do not fabricate extra frames
to reach the requested sample count.

Firmware has already converted and calibrated Rs values. The new backend does
not apply the legacy RLOW formula or multiply by calibration coefficients again.
`--resistance-unit mohm` explicitly divides transmitted ohms by 1,000,000;
`kohm` divides by 1,000. Default ohms are justified by the protocol/CSV headers,
not an assumption about historical training data. RLOW's unit remains whatever
was used in the legacy experiment. Its startup/environment/menu controls remain
available on the legacy backend only. Direct `ENoseSensor()` construction keeps
legacy behavior for existing integrations; shared launchers use `create_sensor`.

The CH0 -> R1 ... CH16 -> R17 mapping follows logical index order. Equivalence
to the old physical sensor array and the MCU's complete conversion formula
remain unverified without further hardware/firmware information.

No calibration, gain, fixed-duty, factory-reset, or flash-save commands are sent.
The supplier's tool is the bundled **E-NOSE desktop tool**. Its Heater ON/OFF
and PI setpoint controls are independent from Start/Stop Stream. Close it before
our client opens the same port. Existing firmware/settings can continue heating
after that tool closes; automatic heating must not be assumed in either program.

Our default `external` mode preserves the supplier-prepared setpoint and never
enables heating. Explicit `client` mode, or client menu 11/local plot Enable,
uses normal PI control with a default 300 C target and 350 C host trip threshold.
These are configurable operating controls, not a certified hardware limit.
Do not assume temperatures above 350 C are safe. Temperature calibration,
fault-free feedback, normal PI mode, fresh measurements and acknowledged
setpoint/enable are required. Fixed-duty operation is not supported here.

Client menu 12/local plot Disable and graceful exit request heater-disable and
wait for matching-request STATUS plus ACK confirming both disabled and actual
zero duty. Faults, lost measurements, temperature limit and session timeout
also request disable and lock out automatic re-enable. Ctrl+C/SIGINT, SIGTERM,
normal return, exceptions and connect failures have cleanup paths. During
cleanup, repeated graceful signals do not skip verification. The robot runner
also wraps its existing mission in cleanup; it does not automatically enable
heating or add real-robot functionality.

**STREAM_STOP alone does not turn off the heater.** Firmware disable can ramp
down rather than instantly remove power. A timeout, failed ACK, missing matching
status, USB loss, SIGKILL, OS crash or host power loss is **OFF UNVERIFIED**,
not a successful shutdown. No host-loss watchdog/emergency-off command is
documented in the supplied firmware. A firmware watchdog or independent power
interlock is required for an unconditional fail-off guarantee. Residual heat can
remain even after duty reaches zero. Never leave this unvalidated setup unattended.

Legacy acquisition receives no INFO, STATUS or binary heater commands; its
existing board-controlled heating remains outside this integration. Sensor faults, calibration mode,
temperature open-circuit flags, RAW mode, non-positive values, and stale/duplicate
frames are rejected. No undocumented saturation threshold is guessed. Reconnect
explicitly after USB loss; there is no silent fallback to the old board or fake data.

## Labels, units, and history safety

Desktop `rs_ch0_ohm ... rs_ch16_ohm` exports are normalized on CSV upload. Choose
the matching output unit in the Train tab. Ordinary R1-R17 CSVs are not rescaled.
RAW_LPF exports are rejected as training input. Missing smell labels cause a
clear error: file names, temperatures, and signal transitions never become labels.
Add verified labels/exposure boundaries to a copy of the file before training.

`temp_actual_c` is the heated sensor's temperature, not ambient `T`. It remains
diagnostic metadata. Missing environmental inputs use compatibility defaults,
marked unmeasured; use a resistance-only model unless real environmental data
exist. Faulted/invalid CSV rows are dropped with a count; time gaps are retained.
Time-only files use a fixed date anchor for elapsed-time analysis. Replay can
recover the recording date from the standard filename; it never invents a date
from today's clock.

New models and canonical history persist explicit resistance units. Compatible
historical learning remains the default. Mismatched units or known PCB units
mixed with unverified historical units are refused before training/history writes.
For the first new-board model, back up existing artifacts and use replacement
training on labelled, correctly annotated data. Do not clear historical files
manually to bypass the checks. Correctly verified legacy CSVs can carry a
`resistance_unit` column on every row; this is an annotation, not a conversion.

Acquisition identity, units, board configuration, counters, host receive times,
temperatures, and faults remain separate from the 22 canonical numeric fields.
Live and manual acquired predictions enforce model-unit compatibility. Window
predictions reject stale or mixed-configuration input. Device changes require
a fresh compatible window and clean-air baseline before baseline-relative use.
Physical cross-board transfer accuracy is not established by unit agreement.

## Plots, sessions, and local logs

The Live tab and client plot show resistance units, unmeasured environmentals,
and freshness separately from browser polling. Read-only diagnostics describe
the last accepted frame; rejected fault frames are logged in the client and the
browser becomes stale if valid readings stop. These diagnostics are not heater
controls or safety interlocks. Frame gaps, resets, and configuration changes
break curves instead of implying uninterrupted acquisition.

Local recording runs on acquisition callbacks, before HTTP. A separate bounded
latest-sample network queue prevents slow requests from blocking local sampling.
Unsent frames remain in local recordings without predictions; this does not
promise capture of every MCU frame when client polling is slower. Idle polls,
read failures and network-skipped frames have separate counters. Local repeated
prediction confirmation adds delay; fast-window predictions remain provisional
when unstable. Agg/headless live mode keeps publishing until interruption or
the end of a replay instead of exiting at once.

## Preview, annotation, evaluation, and deployment

Train > Preview CSV does not fit a model or write active history. It plots the
source R channels and kept/rejected windows, including reasons and class counts.
Adjust window/stride, stable-tail fraction, end margin and slope explicitly.
Defaults remain 60 s windows, final 25% of an exposure, and 30 s end margin;
therefore exposures shorter than roughly 6 minutes may yield no analyte windows.
Do not loosen stability gates merely to make a file pass.

Supply verified `smell_label` rows, or explicit annotations with zero-based
source data rows `[start_row,end_row)` (header excluded). Source-row references
remain stable after fault filtering. Unlabelled gaps never become a chemical
class or connect an exposure to the wrong clean-air phase. Set recording date
and session/recording identity explicitly for time-only exports. File names and
waveform shapes never establish chemical ground truth.

Manual Classify requires a declared resistance unit for verified-unit models.
Live clean-air baselines are bound to board/configuration, connection segment,
and publisher session; changes require a fresh baseline. A recorded physical
identity is provenance, not proof of electrical cross-board equivalence.

The UI defaults to staging candidates without changing active model/history.
Historical merging is still the default data policy. API `stage_candidate=false`
retains the older immediate-update workflow. Model Info lists candidates,
supports explicitly independent CSV comparison on the SAME rows for both models,
explicit promotion, and recoverable rollback snapshots. Exact training-row
overlap is refused; near-duplicate/session independence still requires operator
confirmation. No automatic accuracy improvement is inferred from different
training holdouts. Paired comparison uses persisted training baseline fallbacks;
it does not adapt using the evaluation labels or modify the active live baseline.

Unestimable group holdout metrics are null with an explicit warning, not a
row-wise fallback accuracy. Learned preprocessing and probability calibration
are group-isolated. Final deployment is refitted after holdout diagnostics are
frozen. Promotion without independent paired evidence needs explicit override;
known regression is blocked unless separately overridden. Snapshots preserve
the old model/history pair, but two-file replacement is not crash-atomic. Keep
one server worker, protect the UI/API with private network access, and back up
artifacts before changing deployments. The browser never energizes heaters.

NPZ sessions keep their existing keys and add Unicode JSON acquisition metadata;
loading does not require pickle. Older recordings remain readable. Local PCB
training logs use a backend/unit-specific CSV filename and an explicit
`resistance_unit` column, plus a JSONL metadata sidecar. They never append ohms
to the old unannotated legacy log. Units and temperatures are never classifier
features.

## Validation without hardware

Focused critical tests use a simulated serial port and synthetic packets:

```bash
python -m unittest discover -s tests -p test_pcb_acquisition.py -v
```

Run them inside the documented Docker/client environment, not on real robot
hardware. They verify framing, ACK order, command allowlist, one-time scaling,
fault/RAW rejection, freshness/counters, CSV handling, units/history integrity,
and legacy preservation. No USB/GPU or robot-device access is needed.

The supplied private recording was checked through CSV replay, simulated binary
decoding, live API handling, and NPZ round-trip: 1,616 valid rows, zero dropped,
3,230 seconds, median export interval 2 seconds, sensor temperature 283-286 C.
The saved sample matrix is (1616, 22); its ambient values are defaults, not those
sensor temperatures. It contains no smell labels, so recognition accuracy cannot
be measured. This validates software/data plumbing, not physical USB timing,
electrical calibration, chemical identity, or the real channel correspondence.

Model weights, private CSVs, and the supplier's Windows distribution are not
included in the Python package or public source changes.
