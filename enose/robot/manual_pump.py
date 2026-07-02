"""Manual / human-in-the-loop air pump.

The Go-1's e-nose pump is mechanically switched only — no programmable
relay. The policy still issues ``pump.on()`` / ``pump.off()`` at SETTLE /
APPROACH / PURGE because those state transitions are the right behaviour;
this implementation just *logs* the requested state so the human watching
the mission knows when to flip the switch.

Two modes, picked at construction:

* ``mode="log"`` (default) — emit a single prompt line on each transition,
  return immediately. Fits a CLI run where a human is glancing at the
  terminal.
* ``mode="confirm"`` — also block until the user presses Enter. Useful for
  early bring-up when you don't trust yourself to flip the switch in time.

Either mode satisfies the :class:`enose.robot.interfaces.PumpController`
protocol so the policy doesn't care which is wired up.
"""

from __future__ import annotations

import sys
from typing import Callable, Literal, Optional


class ManualPump:
    def __init__(
        self,
        mode: Literal["log", "confirm"] = "log",
        prompt: Optional[Callable[[str], None]] = None,
    ) -> None:
        if mode not in ("log", "confirm"):
            raise ValueError(f"mode must be 'log' or 'confirm', got {mode!r}")
        self.mode = mode
        self._on: bool = False
        self._prompt = prompt or (lambda msg: print(msg, file=sys.stderr, flush=True))

    def _ask(self, want_on: bool) -> None:
        target = "ON" if want_on else "OFF"
        msg = f"[pump] please switch pump {target}"
        self._prompt(msg)
        if self.mode == "confirm":
            try:
                input(f"[pump] press Enter once pump is {target} ▸ ")
            except EOFError:  # non-interactive stdin
                pass

    def on(self) -> bool:
        if not self._on:
            self._ask(True)
            self._on = True
        return True

    def off(self) -> bool:
        if self._on:
            self._ask(False)
            self._on = False
        return True

    def is_on(self) -> bool:
        return self._on
