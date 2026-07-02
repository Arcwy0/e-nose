"""Sim pump — just a boolean flag."""

from __future__ import annotations


class SimPump:
    def __init__(self) -> None:
        self._on: bool = False

    def on(self) -> bool:
        self._on = True
        return True

    def off(self) -> bool:
        self._on = False
        return True

    def is_on(self) -> bool:
        return self._on
