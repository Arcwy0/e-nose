#!/usr/bin/env python3
"""Lightweight smoke test: AST-parse the package and verify public imports resolve.

Intentionally does NOT import heavy deps (sklearn, xgboost, torch, matplotlib, serial).
It parses every .py file in `enose/`, then checks that each package's `__init__.py`
exposes the names it advertises via `__all__`.

Run:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import ast
import pathlib
import sys
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent / "enose"


def parse_all() -> Tuple[int, List[str]]:
    failures: List[str] = []
    files = sorted(ROOT.rglob("*.py"))
    for p in files:
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            failures.append(f"{p}: {e}")
    return len(files), failures


def check_all_exports() -> List[str]:
    """For every enose sub-package, verify `__all__` names have a matching definition."""
    failures: List[str] = []
    for init in ROOT.rglob("__init__.py"):
        tree = ast.parse(init.read_text(encoding="utf-8"))

        exported: List[str] = []
        imported_names: set[str] = set()
        defined_names: set[str] = set()

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        if tgt.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    exported.append(elt.value)
                        else:
                            defined_names.add(tgt.id)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined_names.add(node.name)

        available = imported_names | defined_names
        for name in exported:
            if name not in available:
                failures.append(f"{init}: __all__ names '{name}' but it's not imported/defined")
    return failures


def main() -> int:
    n_files, parse_failures = parse_all()
    export_failures = check_all_exports()

    print(f"[smoke] AST-parsed {n_files} files — {len(parse_failures)} failures")
    for f in parse_failures:
        print(f"  parse: {f}")

    print(f"[smoke] __all__ export check — {len(export_failures)} failures")
    for f in export_failures:
        print(f"  export: {f}")

    total = len(parse_failures) + len(export_failures)
    print(f"[smoke] {'OK' if total == 0 else f'FAILED ({total})'}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
