"""Contract tests for the opt-in Rust parsing core."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from code_review_graph.native import (
    NativeCoreUnavailableError,
    core_version,
    parse_python_batch,
    parser_rows,
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

    result = parse_python_batch([first, second], tmp_path)
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
    native = parse_python_batch(paths, FIXTURES)
    parser = CodeParser(FIXTURES)
    for path, actual in zip(paths, native, strict=True):
        nodes, edges = parser.parse_file(path)
        expected = sorted(
            (
                (
                    node.kind,
                    node.name,
                    node.parent_name,
                    node.line_start,
                    node.line_end,
                    node.params,
                    node.return_type,
                    node.modifiers,
                    node.is_test,
                    node.identity_name,
                    node.extra,
                )
                for node in nodes
            ),
            key=repr,
        )
        observed = sorted(
            (
                (
                    item.kind,
                    item.name,
                    item.parent_name,
                    item.line_start,
                    item.line_end,
                    item.params,
                    item.return_type,
                    item.modifiers,
                    item.is_test,
                    item.identity_name,
                    item.extra,
                )
                for item in (actual.file_node, *actual.definitions)
            ),
            key=repr,
        )
        assert observed == expected, path.name
        expected_structural_edges = sorted(
            (
                (edge.kind, edge.source, edge.target, edge.line, edge.extra)
                for edge in edges
                if edge.kind
            ),
            key=repr,
        )
        observed_structural_edges = sorted(
            ((edge.kind, edge.source, edge.target, edge.line, edge.extra) for edge in actual.edges),
            key=repr,
        )
        assert observed_structural_edges == expected_structural_edges, path.name


def test_native_parser_rows_match_public_parser_contract() -> None:
    pytest.importorskip("code_review_graph._native")
    path = FIXTURES / "sample_python.py"
    native_nodes, native_edges = parser_rows(parse_python_batch([path], FIXTURES)[0])
    python_nodes, python_edges = CodeParser(FIXTURES).parse_file(path)

    assert native_nodes == python_nodes
    assert sorted(native_edges, key=repr) == sorted(python_edges, key=repr)


def test_native_feature_flag_selects_snapshot_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("code_review_graph._native")
    from code_review_graph import incremental

    path = FIXTURES / "sample_python.py"
    source = path.read_bytes()
    monkeypatch.setenv("CRG_NATIVE_CORE", "1")
    native_nodes, native_edges = incremental._parse_snapshot(
        path,
        source,
        CodeParser(FIXTURES),
        FIXTURES,
    )
    python_nodes, python_edges = CodeParser(FIXTURES).parse_bytes(path, source)

    assert native_nodes == python_nodes
    assert sorted(native_edges, key=repr) == sorted(python_edges, key=repr)
