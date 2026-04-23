"""GPU detection for Florence-2 loading.

Prefers CUDA when available, otherwise falls back to CPU. Prints diagnostics
so failures to get a GPU in Docker are obvious in logs.
"""

from __future__ import annotations

import os
import subprocess

import torch


def check_gpu_availability() -> str:
    """Return "cuda" if torch sees a GPU, else "cpu". Prints a verbose diagnostic block."""
    print("=== GPU Detection ===")
    print(f"PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in filter(None, (l.strip() for l in result.stdout.splitlines())):
                print(f"  nvidia-smi: {line}")
        else:
            print("  nvidia-smi failed or no GPUs")
    except Exception as e:
        print(f"  nvidia-smi unavailable: {e}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"  torch GPU {i}: {name} ({mem_gb:.1f} GB)")
        device = "cuda"
    else:
        print("  torch sees no CUDA — check drivers / `--gpus all` / torch build")
        device = "cpu"

    print(f"Selected device: {device}")
    print("=" * 30)
    return device
