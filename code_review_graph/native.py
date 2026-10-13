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
    params: str | None
    return_type: str | None
    modifiers: str | None
    is_test: bool
    identity_name: str | None
    decorators: tuple[str, ...]
    line_start: int
    line_end: int


@dataclass(frozen=True)
class NativeEdge:
    kind: str
    source: str
    target: str
    line: int


@dataclass(frozen=True)
class NativeParseResult:
    path: str
    sha256: str
    syntax_errors: int
    definitions: tuple[NativeDefinition, ...]
    edges: tuple[NativeEdge, ...]


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


def parse_python_batch(
    paths: list[Path],
    repo_root: Path | None = None,
) -> list[NativeParseResult]:
    """Read and parse Python files in one native batch call."""
    payload = [(path.as_posix(), path.read_bytes()) for path in paths]
    rows = json.loads(_extension().parse_python_batch(payload))
    from .parser import _is_test_function, is_test_file

    results = []
    for row in rows:
        definitions = []
        file_is_test = is_test_file(row["path"], repo_root)
        for definition in row["definitions"]:
            decorators = tuple(definition.pop("decorators"))
            raw_kind = definition.pop("kind")
            function_is_test = raw_kind == "Function" and _is_test_function(
                definition["name"],
                row["path"],
                decorators,
                repo_root=repo_root,
            )
            definitions.append(
                NativeDefinition(
                    **definition,
                    kind="Test" if function_is_test else raw_kind,
                    modifiers=",".join(decorators) or None,
                    is_test=function_is_test or file_is_test,
                    identity_name=definition["name"] if raw_kind == "Function" else None,
                    decorators=decorators,
                )
            )
        results.append(
            NativeParseResult(
                path=row["path"],
                sha256=row["sha256"],
                syntax_errors=row["syntax_errors"],
                definitions=tuple(definitions),
                edges=tuple(NativeEdge(**edge) for edge in row["edges"]),
            )
        )
    return results
