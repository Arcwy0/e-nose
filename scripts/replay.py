#!/usr/bin/env python3
"""Repository wrapper for the packaged, wheel-installable replay implementation."""
from enose.client.replay import (
    load_session, replay_session, main, _row_to_payload, _summarize, _build_parser,
)


if __name__ == "__main__":
    raise SystemExit(main())
