#!/usr/bin/env python3
"""Verify the Docker/pip/uv runtime lock and optionally run real VLM startup.

Run after installing requirements-server.txt and `pip install --no-deps -e .`:
    python scripts/verify_server_environment.py --vlm --device cpu
No datasets, model weights, or classifier artifacts are changed by the probe.
The loader may create missing tokenizer text files from the local tokenizer.json.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def check_lock(lock: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    errors: list[str] = []
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        name = canonicalize_name(requirement.name)
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f"{name}: missing")
            continue
        if not requirement.specifier.contains(installed, prereleases=True):
            errors.append(f"{name}: expected {requirement.specifier}, installed {installed}")
        versions[name] = installed
    if errors:
        raise RuntimeError("Runtime differs from the shared lock:\n" + "\n".join(errors))
    return versions


async def check_vlm(model_path: Path, device: str) -> dict:
    import torch
    from PIL import Image, ImageDraw

    from enose import config
    from enose.vision.florence import process_image_with_vlm

    selected = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    torch.set_num_threads(8)
    config.VLM_MODEL_PATH = str(model_path)
    # The app reads this flag at import time; the smoke check always enables VLM.
    os.environ["ENOSE_NO_VLM"] = "0"
    original_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="enose-vlm-smoke-") as work:
            os.chdir(work)
            server_module = importlib.import_module("enose.server.app")
            from enose.server import state
            from enose.server.routes.health import health_check

            # Exercise the real startup/lifespan, choosing the probe's device.
            with patch.object(server_module, "check_gpu_availability", return_value=selected):
                async with server_module.app.router.lifespan_context(server_module.app):
                    health = await health_check()
                    if not health["models"]["vlm"]:
                        raise RuntimeError("Server lifespan completed without a loaded VLM")
                    image_path = Path(work) / "probe.png"
                    image = Image.new("RGB", (128, 128), "white")
                    ImageDraw.Draw(image).rectangle((32, 32, 96, 96), fill="red")
                    image.save(image_path)
                    prediction = process_image_with_vlm(
                        state.vlm_model, state.vlm_processor, str(image_path), "<CAPTION>"
                    )
                    if not isinstance(prediction, dict) or "<CAPTION>" not in prediction:
                        raise RuntimeError(f"Unexpected VLM output: {prediction!r}")
                    return {
                        "device": selected,
                        "server_lifespan": "passed",
                        "model": type(state.vlm_model).__name__,
                        "processor": type(state.vlm_processor).__name__,
                        "tokenizer_fast": state.vlm_processor.tokenizer.is_fast,
                        "dtype": str(next(state.vlm_model.parameters()).dtype),
                        "prediction": prediction,
                    }
    finally:
        os.chdir(original_cwd)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=root / "requirements/server-lock.txt")
    parser.add_argument("--vlm", action="store_true", help="Load real weights and generate a caption")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model-path", type=Path, default=root / "model/Florence-2-Large")
    parser.add_argument("--output", type=Path, help="Optional JSON validation evidence")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        parser.error("the server lock supports CPython 3.10, 3.11, and 3.12")
    result = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "runtime_versions": check_lock(args.lock.resolve()),
    }
    try:
        result["flash_attn"] = metadata.version("flash-attn")
    except metadata.PackageNotFoundError:
        result["flash_attn"] = None
    if result["flash_attn"] not in (None, "2.7.3"):
        raise RuntimeError(f"Expected optional flash-attn 2.7.3, got {result['flash_attn']}")
    if args.vlm:
        result["vlm"] = asyncio.run(check_vlm(args.model_path.resolve(), args.device))
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
