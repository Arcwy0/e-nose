# Installing the E-Nose client on Windows (prebuilt wheel)

This is the **no-compiler, no-`git clone`** way to install the e-nose client on
Windows. It's aimed at colleagues who just need to run the client (drive the
sensors/camera, stream live graphs, talk to the lab server) — not the server.

## Why the old way failed

Earlier attempts tried to build the whole package from source. That pulled in
the **server** dependencies — `scikit-learn<1.6` and `numpy<2` — which have **no
prebuilt Windows wheels** for newer Python versions, so `pip` tried to *compile*
them and failed (no Visual C++ / build toolchain).

The client doesn't need any of that. The package now ships as a **universal
prebuilt wheel** (`enose-3.1.0-py3-none-any.whl`, pure Python, works on any OS),
and a lightweight **`[client]`** dependency set with none of the heavy pins.

## What you need

- **Python 3.11 or 3.12** (64-bit). Download from <https://www.python.org/downloads/>.
  Tick *"Add python.exe to PATH"* during install.
  > Avoid 3.13 for now — a couple of the dependency wheels lag behind on the
  > newest Python. 3.11/3.12 have prebuilt wheels for everything.
- The file **`enose-3.1.0-py3-none-any.whl`** (ask the maintainer / grab it from
  the project's `dist/` folder).

## Install (PowerShell)

```powershell
# 1. Make a clean virtual environment next to the wheel
python -m venv enose-venv
.\enose-venv\Scripts\Activate.ps1

# 2. Upgrade pip (avoids old-resolver surprises)
python -m pip install --upgrade pip

# 3. Install the client — the [client] extra pulls only the light deps
pip install ".\enose-3.1.0-py3-none-any.whl[client]"
```

That's it. `pip` downloads prebuilt wheels for numpy, pandas, matplotlib,
opencv-python, pyserial, pillow and requests — no compiler involved.

### Verify

```powershell
python -c "import enose.client.api, enose.client.live; print('e-nose client OK')"
```

You should see `e-nose client OK`. After a `[client]` install, fastapi,
scikit-learn, joblib and uvicorn are intentionally **absent** — that's expected;
they live on the server.

## Running the client

The client is driven by `scripts/run_client.py` in the repo. If you installed
only the wheel (no repo checkout), download that one script, or clone the repo
for the scripts and run against the venv you just made. Typical commands:

```powershell
# Offline / simulated sensors, talking to the lab server's web API
python scripts\run_client.py --offline --server http://<LAB_GPU_IP>:18080

# Real hardware on the robot dog (serial ports differ on Windows, e.g. COM3/COM4)
python scripts\run_client.py --port_enose COM3 --port_UART COM4 `
    --server http://<LAB_GPU_IP>:18080 --live --live-classify
```

Or just use the **browser UI** at `http://<LAB_GPU_IP>:18080/ui` — no Python at
all for classify/train/live/model-info.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip` tries to build `scikit-learn`/`numpy` from source | You installed without the `[client]` extra (or an old wheel). Use exactly `pip install ".\enose-3.1.0-py3-none-any.whl[client]"`. |
| `ImportError: DLL load failed` on `import cv2` | Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), or use `pip install opencv-python-headless` if you don't need OpenCV windows. |
| `No module named enose` | The venv isn't activated, or you installed into a different Python. Re-activate `.\enose-venv\Scripts\Activate.ps1`. |
| Serial port not found | On Windows the e-nose shows up as `COMx` (check Device Manager), not `/dev/ttyUSB0`. |
| Can't reach the server | Confirm the lab server URL/port and that you're on the same network / VPN. Open `http://<LAB_GPU_IP>:18080/health` in a browser. |

## Air-gapped / mirror fallback

If installing the extra is inconvenient, install the deps explicitly then the
package with no deps:

```powershell
pip install -r requirements-client.txt
pip install --no-deps ".\enose-3.1.0-py3-none-any.whl"
```
