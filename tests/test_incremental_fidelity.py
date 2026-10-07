"""Tests for the incremental-equals-rebuild fidelity benchmark.

The benchmark answers one question: after an incremental update, is the graph
byte-for-byte what a clean rebuild of the same tree would have produced? A
clean rebuild is a free and perfect oracle, so any divergence is drift the
persistent graph accumulated and never reported.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from code_review_graph.eval.benchmarks import incremental_fidelity as fidelity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HELPER_PY = '''"""Helper utilities."""


def greet(name):
    return f"Hello {name}"


def farewell(name):
    return f"Bye {name}"
'''

_SERVICE_PY = '''"""Service layer."""

from helper import farewell, greet


def welcome(name):
    return greet(name)


def dismiss(name):
    return farewell(name)
'''

_MAIN_PY = '''"""Entry point."""

from service import dismiss, welcome


def main():
    print(welcome("world"))
    print(dismiss("world"))


if __name__ == "__main__":
    main()
'''

_TEST_PY = '''"""Tests."""

from service import welcome


def test_welcome():
    assert welcome("a") == "Hello a"
'''


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=bench@example.com",
            "-c",
            "user.name=bench",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )


def _mock_repo(root: Path) -> Path:
    """A tiny but structurally real git repository."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "helper.py").write_text(_HELPER_PY, encoding="utf-8")
    (root / "service.py").write_text(_SERVICE_PY, encoding="utf-8")
    (root / "main.py").write_text(_MAIN_PY, encoding="utf-8")
    (root / "test_service.py").write_text(_TEST_PY, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--no-verify", "-m", "initial")
    return root


def _build_db(tree: Path, db_path: Path) -> None:
    fidelity._clean_build(tree, db_path)


# ---------------------------------------------------------------------------
# Registration and rendering
# ---------------------------------------------------------------------------


def test_incremental_fidelity_registered_in_the_runner():
    """A benchmark nothing dispatches to never runs."""
    yaml = pytest.importorskip("yaml")  # noqa: F841
    from code_review_graph.eval.runner import BENCHMARK_REGISTRY

    assert "incremental_fidelity" in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY["incremental_fidelity"] is fidelity.run


def test_incremental_fidelity_rendered_by_the_full_report(tmp_path):
    """The reporter must emit the benchmark's rows like the existing seven."""
    from code_review_graph.eval.reporter import generate_full_report

    csv_path = tmp_path / "mock_incremental_fidelity_2026-01-01.csv"
    csv_path.write_text(
        "repo,edit_kind,status,total_differing_rows\n"
        "mock,trailing_comment,ok,0\n",
        encoding="utf-8",
    )
    report = generate_full_report(tmp_path)
    assert "Incremental Fidelity" in report
    assert "trailing_comment" in report


def test_incremental_fidelity_rendered_by_the_readme_tables(tmp_path):
    from code_review_graph.eval.reporter import generate_readme_tables

    csv_path = tmp_path / "mock_incremental_fidelity_2026-01-01.csv"
    csv_path.write_text(
        "repo,edit_kind,status,total_differing_rows,seconds\n"
        "mock,delete_file,diverged,7,1.5\n",
        encoding="utf-8",
    )
    tables = generate_readme_tables(tmp_path)
    assert "delete_file" in tables
    assert "7" in tables


# ---------------------------------------------------------------------------
# The comparator itself
# ---------------------------------------------------------------------------


def test_two_clean_builds_of_the_same_tree_compare_equal(tmp_path):
    """The oracle must agree with itself, or every later number is noise."""
    tree = _mock_repo(tmp_path / "repo")
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    _build_db(tree, left)
    _build_db(tree, right)

    diff = fidelity.compare_graphs(left, right)
    assert diff["total_differing_rows"] == 0, diff["tables"]


def test_comparator_counts_differing_rows_per_table(tmp_path):
    """An injected divergence is attributed to the table that holds it."""
    tree = _mock_repo(tmp_path / "repo")
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    _build_db(tree, left)
    shutil.copy2(left, right)

    import sqlite3

    conn = sqlite3.connect(str(left))
    conn.execute(
        "DELETE FROM edges WHERE kind = 'CALLS' AND rowid IN "
        "(SELECT rowid FROM edges WHERE kind = 'CALLS' LIMIT 2)"
    )
    conn.commit()
    conn.close()

    diff = fidelity.compare_graphs(left, right)
    assert diff["tables"]["edges"]["differing_rows"] == 2
    assert diff["total_differing_rows"] >= 2


def test_comparator_names_a_concrete_example(tmp_path):
    """"Seven rows differ" is unactionable without one of the seven."""
    tree = _mock_repo(tmp_path / "repo")
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    _build_db(tree, left)
    shutil.copy2(left, right)

    import sqlite3

    conn = sqlite3.connect(str(left))
    conn.execute(
        "UPDATE nodes SET line_start = line_start + 99 WHERE name = 'greet'"
    )
    conn.commit()
    conn.close()

    diff = fidelity.compare_graphs(left, right)
    nodes = diff["tables"]["nodes"]
    assert nodes["differing_rows"] >= 1
    assert nodes["examples"], "a divergence with no example is not a report"
    assert any("greet" in ex for ex in nodes["examples"])


def test_comparator_covers_every_derived_table(tmp_path):
    """Nodes and edges alone leave the derived tables free to rot."""
    tree = _mock_repo(tmp_path / "repo")
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    _build_db(tree, left)
    _build_db(tree, right)

    diff = fidelity.compare_graphs(left, right)
    for table in (
        "nodes",
        "edges",
        "communities",
        "node_community",
        "flows",
        "flow_memberships",
        "nodes_fts",
        "community_summaries",
        "flow_snapshots",
        "risk_index",
        "metadata",
    ):
        assert table in diff["tables"], f"{table} is never compared"


def test_comparator_uses_the_unused_graph_diff_helpers(tmp_path):
    """graph_diff.take_snapshot/diff_snapshots exist and nothing called them."""
    tree = _mock_repo(tmp_path / "repo")
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    _build_db(tree, left)
    shutil.copy2(left, right)

    import sqlite3

    conn = sqlite3.connect(str(left))
    conn.execute("DELETE FROM nodes WHERE name = 'farewell'")
    conn.commit()
    conn.close()

    diff = fidelity.compare_graphs(left, right)
    # "added" is relative to the incremental side: the rebuild has a node the
    # incremental graph lost.
    assert diff["snapshot_summary"]["nodes_added"] == 1


# ---------------------------------------------------------------------------
# End-to-end benchmark
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _mock_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("fidelity")
    tree = _mock_repo(root / "repo")
    config = {"name": "mock"}
    from code_review_graph.graph import GraphStore

    store = GraphStore(root / "outer.db")
    try:
        return fidelity.run(tree, store, config)
    finally:
        store.close()


def test_every_edit_kind_gets_exactly_one_row(_mock_run):
    kinds = [r["edit_kind"] for r in _mock_run]
    assert kinds == list(fidelity.EDIT_KINDS)
    assert len(fidelity.EDIT_KINDS) == 7


def test_each_row_reports_wall_time_and_per_table_counts(_mock_run):
    for row in _mock_run:
        assert isinstance(row["seconds"], float)
        assert row["seconds"] >= 0.0
        assert isinstance(row["total_differing_rows"], int)
        for table in fidelity.COMPARED_TABLES:
            assert f"{table}_diff" in row


def test_known_failures_do_not_turn_the_run_red(_mock_run):
    """A currently-broken edit kind is recorded, not raised."""
    for row in _mock_run:
        assert row["status"] in {"ok", "known_failure", "diverged", "skipped", "error"}
        if row["edit_kind"] in fidelity.KNOWN_FAILURES:
            assert row["status"] != "diverged", (
                "a known failure must be reported as known_failure, "
                "not as an unexpected divergence"
            )


def test_edit_kinds_outside_the_known_failure_list_still_hold(_mock_run):
    """Regression guard: a passing edit kind must not start diverging."""
    unexpected = [
        (r["edit_kind"], r["total_differing_rows"], r["worst_table"])
        for r in _mock_run
        if r["status"] == "diverged"
    ]
    assert not unexpected, f"undeclared divergences: {unexpected}"


def test_run_reports_the_benchmark_wall_time(_mock_run):
    assert all("benchmark_seconds" in r for r in _mock_run)
    assert len({r["benchmark_seconds"] for r in _mock_run}) == 1


def test_known_failures_only_names_real_edit_kinds():
    """A stale entry would silence a regression guard for a kind that is gone."""
    assert fidelity.KNOWN_FAILURES <= set(fidelity.EDIT_KINDS)


# ---------------------------------------------------------------------------
# What the benchmark found. These document the defects at the level of the
# defect rather than at the level of the whole-graph diff. These must pass
# after full post-processing of an incremental update.
# ---------------------------------------------------------------------------


def test_incremental_update_leaves_no_dangling_flow_memberships(tmp_path):
    import sqlite3

    from code_review_graph.eval.benchmarks import incremental_fidelity as f

    tree = _mock_repo(tmp_path / "repo")
    db = tmp_path / "graph.db"
    f._clean_build(tree, db)
    base = f._head(tree)

    helper = tree / "helper.py"
    helper.write_text(helper.read_text(encoding="utf-8") + "# neutral\n",
                      encoding="utf-8")
    f._commit_all(tree, "neutral edit")
    f._incremental_build(tree, db, base=base)

    conn = sqlite3.connect(str(db))
    try:
        dangling = conn.execute(
            "SELECT count(*) FROM flow_memberships fm "
            "LEFT JOIN nodes n ON n.id = fm.node_id WHERE n.id IS NULL"
        ).fetchone()[0]
    finally:
        conn.close()
    assert dangling == 0


def test_reparsed_nodes_keep_a_community_assignment(tmp_path):
    import sqlite3

    from code_review_graph.eval.benchmarks import incremental_fidelity as f

    tree = _mock_repo(tmp_path / "repo")
    db = tmp_path / "graph.db"
    f._clean_build(tree, db)
    base = f._head(tree)

    def uncommunitied() -> int:
        conn = sqlite3.connect(str(db))
        try:
            return conn.execute(
                "SELECT count(*) FROM nodes WHERE community_id IS NULL"
            ).fetchone()[0]
        finally:
            conn.close()

    before = uncommunitied()
    helper = tree / "helper.py"
    helper.write_text(helper.read_text(encoding="utf-8") + "# neutral\n",
                      encoding="utf-8")
    f._commit_all(tree, "neutral edit")
    f._incremental_build(tree, db, base=base)
    assert uncommunitied() == before


@pytest.mark.parametrize("edit_kind", ["replace", "add", "delete"])
def test_repeated_mutation_and_revert_matches_fresh_derived_graph(tmp_path, edit_kind):
    from code_review_graph.eval.benchmarks.storage_growth import (
        _fingerprints,
        _health,
        _pipeline,
    )
    from code_review_graph.graph import GraphStore

    tree = _mock_repo(tmp_path / "repo")
    target = tree / {"replace": "helper.py", "add": "probe.py", "delete": "main.py"}[edit_kind]
    original = target.read_bytes() if target.exists() else None
    store = GraphStore(tmp_path / "incremental.db")
    try:
        _pipeline(store, tree, None)
        baseline = _fingerprints(store)
        for cycle in range(3):
            for phase in ("mutate", "revert"):
                content = original
                if phase == "mutate" and edit_kind == "delete":
                    content = None
                elif phase == "mutate" and edit_kind == "add":
                    content = (
                        b'from service import welcome\n\ndef main():\n'
                        b'    return welcome("x")\n'
                    )
                elif phase == "mutate":
                    assert original is not None
                    content = original.replace(
                        b'return f"Hello {name}"', b'return farewell(name)'
                    ) + b'\n\ndef storage_probe(value):\n    return greet(value)\n'
                if content is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_bytes(content)
                _git(tree, "add", "-A")
                _pipeline(store, tree, [target.name])
                assert _health(store)["ok"]
                fresh = GraphStore(tmp_path / f"fresh-{cycle}-{phase}.db")
                try:
                    _pipeline(fresh, tree, None)
                    assert _fingerprints(store) == _fingerprints(fresh)
                finally:
                    fresh.close()
                if phase == "revert":
                    assert _fingerprints(store) == baseline
    finally:
        store.close()


@pytest.mark.parametrize("discovery", ["git", "reconcile", "watch", "capped"])
def test_deleted_called_file_matches_rebuild(tmp_path, monkeypatch, discovery):
    from code_review_graph.eval.benchmarks.storage_growth import _fingerprints, _health
    from code_review_graph.graph import GraphStore
    from code_review_graph.incremental import incremental_update
    from code_review_graph.tools.build import _run_postprocess

    if discovery == "capped":
        monkeypatch.setattr("code_review_graph.incremental.find_dependents", lambda *args: [])
    tree = _mock_repo(tmp_path / "repo")
    db = tmp_path / "incremental.db"
    fidelity._clean_build(tree, db)
    base = fidelity._head(tree)
    (tree / "helper.py").unlink()
    _git(tree, "add", "-A")
    _git(tree, "commit", "-qm", "delete called helper")
    store = GraphStore(db)
    fresh = GraphStore(tmp_path / "fresh.db")
    try:
        result = incremental_update(
            tree, store, base=base,
            changed_files=None if discovery == "git" else (
                [] if discovery == "reconcile" else ["helper.py"]
            ),
            reconcile_stale=discovery != "watch",
        )
        assert not result["errors"]
        assert "service.py" in result["dependent_files"]
        assert not _run_postprocess(
            store, result, "full", changed_files=result["changed_files"],
            repo_root=str(tree),
        )
        from code_review_graph.eval.benchmarks.storage_growth import _pipeline
        _pipeline(fresh, tree, None)
        assert _health(store)["ok"]
        assert _fingerprints(store) == _fingerprints(fresh)
    finally:
        store.close()
        fresh.close()


@pytest.mark.parametrize("failure", ["parse", "write"])
def test_deleted_target_referrers_retry_after_failure(tmp_path, monkeypatch, failure):
    from code_review_graph.eval.benchmarks.storage_growth import _fingerprints, _pipeline
    from code_review_graph.graph import GraphStore
    from code_review_graph.incremental import incremental_update
    from code_review_graph.parser import CodeParser
    from code_review_graph.tools.build import _run_postprocess

    tree = _mock_repo(tmp_path / "repo")
    db = tmp_path / "incremental.db"
    fidelity._clean_build(tree, db)
    (tree / "helper.py").unlink()
    store = GraphStore(db)
    try:
        with monkeypatch.context() as patch:
            if failure == "parse":
                original = CodeParser.parse_bytes

                def fail_parse(self, path, source):
                    if path.name == "service.py":
                        raise OSError("temporary read failure")
                    return original(self, path, source)

                patch.setattr(CodeParser, "parse_bytes", fail_parse)
                result = incremental_update(tree, store, changed_files=["helper.py"])
                assert result["errors"]
            else:
                original_write = store.store_file_nodes_edges

                def fail_write(path, *args):
                    if Path(path).name == "service.py":
                        raise RuntimeError("temporary database failure")
                    return original_write(path, *args)

                patch.setattr(store, "store_file_nodes_edges", fail_write)
                with pytest.raises(RuntimeError, match="temporary database failure"):
                    incremental_update(tree, store, changed_files=["helper.py"])
        assert "service.py" in store.get_metadata("pending_dependent_files")
    finally:
        store.close()

    store = GraphStore(db)
    fresh = GraphStore(tmp_path / "fresh.db")
    try:
        # Retry without source events or a reconciliation scan. The removed
        # target and its incoming edges are already gone from the database.
        result = incremental_update(tree, store, changed_files=[], reconcile_stale=False)
        assert not result["errors"]
        assert "service.py" in result["changed_files"]
        assert store.get_metadata("pending_dependent_files") == "[]"
        assert not _run_postprocess(
            store, result, "full", changed_files=result["changed_files"], repo_root=str(tree),
        )
        _pipeline(fresh, tree, None)
        assert _fingerprints(store) == _fingerprints(fresh)
    finally:
        store.close()
        fresh.close()


def test_ambiguous_candidate_order_survives_file_replacement(tmp_path):
    from code_review_graph.graph import GraphStore
    from code_review_graph.parser import EdgeInfo, NodeInfo

    store = GraphStore(tmp_path / "graph.db")
    try:
        nodes = {
            file: NodeInfo(kind="Function", name="greet", file_path=file,
                           line_start=1, line_end=2, language="python")
            for file in ("a.py", "b.py")
        }
        for file, node in nodes.items():
            store.store_file_nodes_edges(file, [node], [], "hash")
        store.store_file_nodes_edges("caller.py", [], [
            EdgeInfo(kind="IMPORTS_FROM", source="caller.py", target="a.py",
                     file_path="caller.py", line=1),
            EdgeInfo(kind="IMPORTS_FROM", source="caller.py", target="b.py",
                     file_path="caller.py", line=2),
            EdgeInfo(kind="CALLS", source="caller.py::main", target="greet",
                     file_path="caller.py", line=4),
        ], "hash")
        store.resolve_bare_call_targets()
        before = store.get_edges_by_source("caller.py::main")[0].extra
        store.store_file_nodes_edges("a.py", [nodes["a.py"]], [], "new hash")
        store.resolve_bare_call_targets()
        assert store.get_edges_by_source("caller.py::main")[0].extra == before
    finally:
        store.close()
