# Installing the E-Nose client on Windows

This guide covers the **3.3.0 wheel**, PowerShell, and uv. The client acquires
sensors/camera data and talks to a remote server; it does not need Torch, the
VLM, or classifier libraries.

For a server on this PC, follow [SETUP.md](SETUP.md#2-server-environment-the-same-runtime-lock-as-docker).
Keep client and server environments separate because their OpenCV distributions
differ. Upgrade both packages for matching acquisition/UI behaviour.

## 1. Download and verify

Install uv and use 64-bit Python 3.11 for this client setup. Download these two
assets from [release v3.3.0](https://github.com/Arcwy0/e-nose/releases/tag/v3.3.0):

- `enose-3.3.0-py3-none-any.whl`
- `SHA256SUMS`

In their download directory:

```powershell
$wheelName = "enose-3.3.0-py3-none-any.whl"
$checksumLine = Get-Content .\SHA256SUMS | Where-Object { $_ -match ([regex]::Escape($wheelName) + "$") }
if (@($checksumLine).Count -ne 1) { throw "Missing or ambiguous wheel checksum" }
$expectedHash = ($checksumLine -split "\s+")[0].ToLowerInvariant()
$actualHash = (Get-FileHash -Algorithm SHA256 ".\$wheelName").Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) { throw "Wheel checksum mismatch" }
```

The package wheel is pure Python. Dependencies such as NumPy and OpenCV still
need platform-specific native wheels and Windows runtime DLLs. This does not
guarantee installation on every Windows/Python combination; native Windows
hardware/GUI operation remains to be validated on the colleague's machine.

## 2. Create a client-only environment

```powershell
uv venv --python 3.11 .venv-client
.\.venv-client\Scripts\Activate.ps1
uv pip install ".\enose-3.3.0-py3-none-any.whl[client]"
uv pip check
uv run --active --no-project python -c "import enose.client.api, enose.client.live; print('e-nose client OK')"
uv run --active --no-project enose-client --help
```

Use `uv run --active --no-project` so a nearby project file cannot silently
select a different environment. No Git checkout is needed for `enose-client`
or packaged NPZ replay. Repository scripts require a checkout.

The lightweight client dependencies are NumPy, pandas, Pillow, requests,
pyserial, Matplotlib, and desktop OpenCV. FastAPI, Torch, scikit-learn and
Transformers are intentionally absent.

If using a trusted offline/mirror installation, obtain all native dependency
wheels for the target Windows/Python architecture first. With a matching
`requirements-client.txt` and configured local index:

```powershell
uv pip install -r requirements-client.txt
uv pip install --no-deps ".\enose-3.3.0-py3-none-any.whl"
uv pip check
```

The requirements file alone is not an air-gapped wheelhouse.

## 3. Test without hardware

```powershell
uv run --active --no-project enose-client --offline --server http://LAB_SERVER:8016 --preflight
uv run --active --no-project enose-client --offline --server http://LAB_SERVER:8016
```

Replace `LAB_SERVER` with the actual server address. The browser UI is
`http://LAB_SERVER:8016/ui`; test connectivity at `/health`.
No client installation is needed for browser-only classify/train/model-info.
Live sensor data still needs an acquisition client.

Replay a supplier resistance-mode CSV without USB hardware:

```powershell
uv run --active --no-project enose-client --sensor-backend replay --sensor-csv ".\E-NOSE-recording.csv" --server http://LAB_SERVER:8016 --live --headless
```

Replay an existing recorded NPZ session:

```powershell
uv run --active --no-project python -m enose.client.replay ".\session.npz" --server http://LAB_SERVER:8016 --limit 50
```

Replay is read-only acquisition: it sends no heater commands and infers no
chemical labels from filenames.

## 4. Hardware setup and options

Review [PCB_ACQUISITION.md](PCB_ACQUISITION.md) before connecting the board.
The default backend is the new binary USB CDC PCB, not the legacy board.
Preflight observes acquisition/status without enabling heating:

```powershell
uv run --active --no-project enose-client --sensor-backend pcb --port_enose COM3 --server http://LAB_SERVER:8016 --preflight --diagnostic-report ".\pcb-preflight.json"
```

Use a new report filename; diagnostic output refuses to overwrite a file.
For the old board select `--sensor-backend legacy`; INFO/STATUS are unavailable
by design. RLOW calibration applies to that legacy conversion, not PCB ohms.
Environmental UART is optional: omit `--port_UART` for R1–R17-only operation.

The default heater mode `external` preserves the supplier-set PI target and
monitors heating. Close the supplier GUI before our client opens the same COM
port. Explicit client heating requires `--heater-control client`; its defaults
are a 300°C target, 350°C host trip guard, and 1800-second limit. These are not
certified board safety ratings.

On graceful exit, the client requests heater disable and verifies fresh zero
duty; firmware cooldown is not an immediate cut-off. USB loss, process kill,
host crash and power failure cannot guarantee shutdown without a firmware
watchdog/interlock. `OFF UNVERIFIED` is not a safe-off confirmation.

For **all options and menu controls**, use the
[client README](../enose/client/README.md). Important groups include:

| Group | Flags |
|---|---|
| Acquisition | `--sensor-backend`, `--port_enose`, `--baud_enose`, `--resistance-unit`, `--rlow`, optional `--port_UART` |
| Replay / simulation | `--sensor-csv`, `--replay-speed`, `--offline` |
| Diagnostics | `--preflight`, `--preflight-seconds`, `--diagnostic-report` |
| Heating | `--heater-control`, `--heater-target-c`, `--heater-max-c`, `--heater-max-seconds`, `--heater-shutdown-timeout` |
| Vision deadlines | `--vision-timeout`, `--scene-timeout` |
| Live recording | `--live`, `--headless`, `--live-rate`, `--live-classify`, `--live-http-timeout`, `--prediction-confirmations`, `--record-session` |

Example offline CPU-server timing test:

```powershell
uv run --active --no-project enose-client --offline --server http://LAB_SERVER:8016 --vision-timeout 180 --scene-timeout 240
```

Those are request deadlines, not a guaranteed inference time. Before live
baseline-dependent predictions, capture current clean air after a new
board/session or server restart. PCB ohms and legacy scaling must not be mixed
without verified conversion and compatible training provenance.

## Troubleshooting

| Symptom | Check |
|---|---|
| Classifier/Torch dependencies install on the client | Use the current wheel's `[client]` extra in a clean client-only environment. |
| No native dependency wheel / compiler requested | Confirm 64-bit Python 3.11 and the target platform; the universal package wheel does not make its dependencies universal. |
| DLL load failed on Torch/OpenCV | Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe); do not mix headless and desktop OpenCV. |
| `No module named enose` | Activate `.venv-client` and use `uv run --active --no-project`. |
| COM port unavailable | Check Device Manager; close the supplier GUI or another process owning the port. |
| Detection timeout | Increase client deadlines and distinguish timeout from an empty detection response; CPU can be slow. |
| Baseline/unit mismatch | Declare actual units and reacquire clean air for the current session. |
| No predictions after training | Browser training stages a candidate by default; explicitly promote it after evaluation. |
