# enose/client — Interactive Client

## Quickstart

```bash
# No hardware needed:
python scripts/run_client.py --offline --server http://localhost:8080

# Hardware, auto port-select:
python scripts/run_client.py --server http://SERVER_IP:8080

# Hardware, explicit ports (Linux):
python scripts/run_client.py --port_enose /dev/ttyUSB0 --port_UART /dev/ttyUSB1

# Hardware, explicit ports (Windows):
python scripts/run_client.py --port_enose COM3 --port_UART COM4
```

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--server` | `http://localhost:8080` | Server URL |
| `--port_enose` | (interactive) | E-nose serial port |
| `--port_UART` | (interactive) | UART environmental port |
| `--camera` | `0` | Camera index |
| `--time` | `60` | Recording duration in seconds |
| `--samples` | `100` | Target samples per recording |
| `--baud_enose` | `9600` | E-nose baud rate |
| `--baud_UART` | `9600` | UART baud rate |
| `--offline` | off | Simulate all sensors |
| `--live` | off | Skip the menu and open the live dashboard straight away |
| `--live-rate` | `5.0` | Live publisher rate in Hz |
| `--live-window` | `60.0` | Rolling window shown in the plot, in seconds |
| `--live-baseline` | off | Subtract the first seconds as an "air" baseline |
| `--live-classify` | off | Also classify every sample so the plot shows confidence |
| `--live-label` | — | Tag live samples with this label (recording a known odour) |
| `--record-session` | — | Also save a `.npz` session file (works with `--live` or menu 8) |

## Menu options

| # | Name | Hardware needed |
|---|---|---|
| 1 | Full training cycle (vision + smell) | Camera + both serial |
| 2 | Vision only | Camera only |
| 3 | Smell identification | E-nose + UART (or `--offline`) |
| 4 | Model info | None (server call) |
| 5 | Learn from CSV | None (server call) |
| 6 | Manual 22-feature input | None (type values) |
| 7 | Visualizations + analysis | None (server call) |
| 8 | Live sensor stream (plot + push to server) | E-nose + UART (or `--offline`) |
| 9 | Replay a recorded session | A `.npz` session file |

### Live streaming & session replay

```bash
# Stream to the server, classify each sample, tag as "coffee", record to disk
python scripts/run_client.py --offline \
    --live --live-classify --live-label coffee \
    --record-session runs/coffee_2026-04-23.npz \
    --server http://localhost:8080

# Replay a saved session later against a (possibly retrained) server
python scripts/replay.py runs/coffee_2026-04-23.npz \
    --server http://localhost:8080
```

The live dashboard opens a matplotlib window (R-sensor trace on top, env below, confidence at the bottom when `--live-classify` is set) and simultaneously pushes samples to `/sensor/live/push` so the UI's **Live** tab can plot them for anyone watching the server. The `--record-session` flag tees the same data stream into a `.npz` file that `scripts/replay.py` can replay offline.

## Module layout

| File | Class / Functions | Description |
|---|---|---|
| `sensors.py` | `ENoseSensor` | Dual UART reader, offline simulation, threaded recording |
| `webcam.py` | `WebcamHandler` | OpenCV preview + capture, detection overlay |
| `api.py` | `ServerAPI` | HTTP client wrapping every server endpoint |
| `pipeline.py` | `TrainingPipeline` | Orchestrates the full training and inference flows |
| `live.py` | `LivePublisher`, `LiveSensorPlot`, `run_live_dashboard` | Threaded sample publisher + matplotlib live plot |
| `session.py` | `SessionRecorder`, `tee_callbacks` | Records a live run to `.npz` for later replay |
| `main.py` | `main()` | argparse CLI + menu loop |

## Offline simulation

In offline mode, `ENoseSensor.generate_realistic_sensor_data(smell_name)` produces synthetic 22-feature samples using reference smell profiles (air, coffee, rose, lemon, vanilla, apple, mint, chocolate) with Gaussian noise. Add new profiles by editing `_SMELL_PROFILES` in `sensors.py`.

## Hardware wiring

```
E-nose board   →  USB/Serial  →  port_enose (sends 17 space-separated ADC counts)
UART env board →  USB/Serial  →  port_UART  (sends 5 tab-separated values: T H CO2 H2S CH2O)
```

ADC → resistance conversion happens client-side in `transform_sensor_values()` using constants from `enose/config.py` (RLOW, VCC, EG, VREF, COEF).

### Finding the serial device

| OS | How |
|---|---|
| Linux | `ls /dev/ttyUSB*` or `ls -l /dev/serial/by-id/` for stable names |
| macOS | `ls /dev/tty.usbserial-*` |
| Windows | Device Manager → Ports (COM & LPT) |

If opening the device fails with `permission denied` on Linux:

```bash
sudo usermod -aG dialout $USER   # once, then log out + back in
```

## Running the client in Docker with UART pass-through

The provided `docker/Dockerfile-client` gives colleagues a reproducible Python environment. USB-serial devices are shared with the container via `--device`.

```bash
# Build once
docker build -t enose-client -f docker/Dockerfile-client .

# Offline (verify server connection first, no hardware)
docker run --rm -it \
  --network host \
  -v "$(pwd)":/app -e PYTHONPATH=/app \
  enose-client \
  python scripts/run_client.py --offline --server http://<SERVER_HOST>:8080

# Hardware — pass both USB-serial devices through
docker run --rm -it \
  --device=/dev/ttyUSB0 \
  --device=/dev/ttyUSB1 \
  -v "$(pwd)":/app -e PYTHONPATH=/app \
  enose-client \
  python scripts/run_client.py \
      --port_enose /dev/ttyUSB0 \
      --port_UART  /dev/ttyUSB1 \
      --server     http://<SERVER_HOST>:8080
```

Notes:
- `--device` is enough; no `--privileged` required.
- `Dockerfile-client` is headless — webcam-dependent menu items (1, 2) will not work inside it. Colleagues should use menu **3** (smell identification) and **6** (manual input) when driving the remote server from Docker.
- On Windows/macOS hosts, USB-serial pass-through to a Linux container is not supported — run the client outside Docker instead.

See [`docs/SERVER_CLIENT_GUIDE.md`](../../docs/SERVER_CLIENT_GUIDE.md) for the full server / network setup.

## Programmatic use

```python
from enose.client.sensors import ENoseSensor
from enose.client.api import ServerAPI

sensor = ENoseSensor(offline_mode=True)
sensor.connect()
sensor.set_simulation_smell("coffee")
sample = sensor.read_single_measurement()   # dict with 22 keys

api = ServerAPI("http://localhost:8080")
result, err = api.classify_smell(sample)
print(result["predicted_smell"], result["confidence"])
```
