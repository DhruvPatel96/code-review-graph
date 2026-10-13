"""Explicit boundary for the optional Rust core.

Importing code-review-graph never requires the extension. Callers must request
it explicitly; production parsing remains Python until parity gates pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class NativeCoreUnavailableError(RuntimeError):
    """Raised when an explicitly requested native operation cannot run."""


@dataclass(frozen=True)
class NativeDefinition:
    kind: str
    name: str
    parent_name: str | None
    line_start: int
    line_end: int


@dataclass(frozen=True)
class NativeParseResult:
    path: str
    sha256: str
    syntax_errors: int
    definitions: tuple[NativeDefinition, ...]


def _extension() -> Any:
    try:
        from . import _native
    except ImportError as exc:
        raise NativeCoreUnavailableError(
            "Rust core is not installed; run `maturin develop --release`"
        ) from exc
    return _native


def core_version() -> str:
    return str(_extension().core_version())


def parse_python_batch(paths: list[Path]) -> list[NativeParseResult]:
    """Read and parse Python files in one native batch call."""
    payload = [(path.as_posix(), path.read_bytes()) for path in paths]
    rows = json.loads(_extension().parse_python_batch(payload))
    return [
        NativeParseResult(
            path=row["path"],
            sha256=row["sha256"],
            syntax_errors=row["syntax_errors"],
            definitions=tuple(NativeDefinition(**definition) for definition in row["definitions"]),
        )
        for row in rows
    ]
