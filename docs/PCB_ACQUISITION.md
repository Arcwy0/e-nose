# PCB acquisition, legacy compatibility, and recorded-data replay

The supported client entry points (`scripts/run_client.py` / `enose-client`)
now default to the new USB CDC PCB. The robot trainer's ROS2 launcher uses the
same factory and default. Simulation remains synthetic and does not open ports.
The old standalone `client_server.py` remains a legacy implementation; use the
package client for the new backend. This document describes unreleased source
changes, not functionality in the previously published v3.2.2 wheel.

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

No heater-enable, temperature-setpoint, calibration, gain, factory-reset, or
flash-save commands are sent. Use the supplier's tool to prepare hardware safely
before acquisition. Stop the supplier tool before opening the same port here.
**STREAM_STOP does not turn off the heater.** Sensor faults, calibration mode,
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
