"""Contract tests for the opt-in Rust parsing core."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from code_review_graph.native import (
    NativeCoreUnavailableError,
    core_version,
    parse_python_batch,
)
from code_review_graph.parser import CodeParser

FIXTURES = Path(__file__).parent / "fixtures"


def test_missing_extension_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(__import__("sys").modules, "code_review_graph._native", raising=False)
    try:
        version = core_version()
    except NativeCoreUnavailableError as exc:
        assert "maturin develop" in str(exc)
    else:
        assert version == "0.1.0"


def test_native_python_batch_is_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("code_review_graph._native")
    first = tmp_path / "first.py"
    first.write_text(
        "class Service:\n    def run(self):\n        return 1\n\ndef top():\n    return 2\n",
        encoding="utf-8",
    )
    second = tmp_path / "second.py"
    second.write_text("def other():\n    return 3\n", encoding="utf-8")

    result = parse_python_batch([first, second])
    assert [row.path for row in result] == [first.as_posix(), second.as_posix()]
    assert result[0].sha256 == hashlib.sha256(first.read_bytes()).hexdigest()
    assert result[0].syntax_errors == 0
    assert [(item.kind, item.name, item.parent_name) for item in result[0].definitions] == [
        ("Class", "Service", None),
        ("Function", "run", "Service"),
        ("Function", "top", None),
    ]


def test_native_python_batch_reports_syntax_errors(tmp_path: Path) -> None:
    pytest.importorskip("code_review_graph._native")
    source = tmp_path / "broken.py"
    source.write_text("def broken(:\n", encoding="utf-8")
    assert parse_python_batch([source])[0].syntax_errors > 0


def test_native_definition_projection_matches_python_fixtures() -> None:
    pytest.importorskip("code_review_graph._native")
    # Databricks exports embed R/SQL cells and intentionally use a separate
    # polyglot preprocessing path. This first native slice covers Python syntax.
    paths = sorted(path for path in FIXTURES.glob("*.py") if "databricks" not in path.name)
    native = parse_python_batch(paths)
    parser = CodeParser(FIXTURES)
    for path, actual in zip(paths, native, strict=True):
        nodes, _ = parser.parse_file(path)
        expected = sorted(
            (
                "Class" if node.kind == "Class" else "Function",
                node.name,
                node.parent_name,
                node.line_start,
                node.line_end,
            )
            for node in nodes
            if node.kind in {"Class", "Function", "Test"}
        )
        observed = sorted(
            (item.kind, item.name, item.parent_name, item.line_start, item.line_end)
            for item in actual.definitions
        )
        assert observed == expected, path.name
