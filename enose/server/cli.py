"""Installed server command; shared by the wheel and repository launcher."""
from __future__ import annotations

import argparse
import os

from enose.config import SERVER_HOST, SERVER_PORT


def main():
    parser = argparse.ArgumentParser(description="Run the E-Nose FastAPI server")
    parser.add_argument("--host", default=os.environ.get("SERVER_HOST", SERVER_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SERVER_PORT", SERVER_PORT)))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers != 1:
        parser.error("Use one worker: live baselines, buffers and deployment locks are process-local")
    import uvicorn
    uvicorn.run("enose.server.app:app", host=args.host, port=args.port, reload=args.reload, workers=1)


if __name__ == "__main__":
    main()
