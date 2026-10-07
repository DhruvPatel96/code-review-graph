"""Isolation, real churn, maintenance effects, and honest correctness reporting."""

from __future__ import annotations

import json
import sqlite3
import tarfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from code_review_graph.eval.benchmarks import storage_growth as growth
from code_review_graph.graph import GraphStore


def _repo(root: Path) -> Path:
    root.mkdir()
    for index in range(3):
        (root / f"service{index}.py").write_text(
            f"def work{index}(value):\n    return value + {index}\n", encoding="utf-8")
    (root / "main.py").write_text(
        "from service0 import work0\n\ndef main():\n    return work0(1)\n", encoding="utf-8")
    growth._git(root, "-c", "init.templateDir=", "init", "-q")
    growth._git(root, "add", "--all")
    growth._git(root, "commit", "-qm", "Fixture")
    return root


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return _repo(tmp_path / "source")


@pytest.fixture
def snapshot(source: Path, tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("CRG_SERIAL_PARSE", "1")
    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    destination = tmp_path / "snapshot"
    growth.snapshot_repository(source, growth._git(source, "rev-parse", "HEAD"), destination)
    return destination


def test_snapshot_uses_commit_and_skips_links_and_database(source: Path, tmp_path: Path) -> None:
    # A tracked graph must not seed or contaminate the fresh benchmark database.
    data_dir = source / ".code-review-graph"
    data_dir.mkdir()
    (data_dir / "graph.db").write_bytes(b"live database sentinel")
    outside = tmp_path / "outside.py"
    outside.write_text("external sentinel", encoding="utf-8")
    try:
        (source / "linked.py").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")
    growth._git(source, "add", "--all")
    growth._git(source, "commit", "-qm", "Tracked artifacts")
    revision = growth._git(source, "rev-parse", "HEAD")
    committed = (source / "main.py").read_bytes()
    (source / "main.py").write_bytes(b"uncommitted sentinel")
    (source / "untracked.py").write_bytes(b"untracked sentinel")
    before = growth._git(source, "status", "--porcelain")
    target = tmp_path / "snapshot"
    result = growth.snapshot_repository(source, revision, target)
    assert (target / "main.py").read_bytes() == committed
    assert not (target / "untracked.py").exists()
    assert not (target / "linked.py").exists()
    assert not (target / ".code-review-graph" / "graph.db").exists()
    assert set(result["skipped_archive_members"]) == {"linked.py", ".code-review-graph/graph.db"}
    assert (source / "main.py").read_bytes() == b"uncommitted sentinel"
    assert (data_dir / "graph.db").read_bytes() == b"live database sentinel"
    assert outside.read_text() == "external sentinel"
    assert growth._git(source, "status", "--porcelain") == before


@pytest.mark.parametrize("policy", growth.POLICIES)
def test_real_cycles_restore_graph_and_capture_maintenance(snapshot: Path, policy: str) -> None:
    config = growth.Config(cycles=2, files=2, functions_per_file=2, query_repeats=1,
                           maintenance_every=2, vector_dim=8)
    result = growth.run_case(snapshot, policy, config)
    assert result["auto_vacuum"] == (2 if policy == "incremental" else 0)
    assert result["maintenance_preserved_content"]
    assert len(result["cycles"]) == 2
    cold = result["cold_sample"]
    for cycle in result["cycles"]:
        added = cycle["mutate"]["storage"]["row_counts"]
        restored = cycle["revert"]["storage"]["row_counts"]
        assert added["nodes"] == cold["row_counts"]["nodes"] + 4
        assert added["embeddings"] == cold["row_counts"]["embeddings"] + 4
        assert restored["nodes"] == cold["row_counts"]["nodes"]
        assert restored["edges"] == cold["row_counts"]["edges"]
        assert restored["embeddings"] == cold["row_counts"]["embeddings"]
        assert cycle["revert"]["core_matches_expected"]
        for phase in ("mutate", "revert"):
            assert cycle[phase]["files_processed"] == config.files
            assert cycle[phase]["health"]["fts_integrity"] == "ok"
    assert "maintenance" not in result["cycles"][0]
    assert result["cycles"][1]["maintenance"]["policy"] == policy
    assert result["end_diagnostics"][-1]["storage"]["freelist_count"] == 0
    assert result["end_diagnostics"][-1]["storage"]["wal_bytes"] == 0
    assert "synthetic_vector_search" in result["final_queries"]["latencies"]
    assert result["final_sample"]["non_freelist_bytes"] == (
        result["final_sample"]["allocated_bytes"] - result["final_sample"]["free_reusable_bytes"]
    )
    assert not (snapshot / "graph.db").exists()
    assert growth._git(snapshot, "status", "--porcelain") == ""


def test_worker_ignores_live_data_directory_and_cloud_config(
    source: Path, tmp_path: Path, monkeypatch,
) -> None:
    live = tmp_path / "live"
    live.mkdir()
    sentinel = live / "graph.db"
    sentinel.write_bytes(b"do not open this database")
    monkeypatch.setenv("CRG_DATA_DIR", str(live))
    monkeypatch.setenv("CRG_OPENAI_API_KEY", "must-not-be-used")
    monkeypatch.setenv("CRG_OPENAI_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("CRG_OPENAI_MODEL", "must-not-be-used")
    result = growth.run_benchmark(
        source, growth.Config(cycles=1, files=1, query_repeats=1, vector_dim=4), ["none"],
    )
    case = result["cases"][0]
    assert "error" not in case
    assert len(case["cycles"]) == 1
    assert sentinel.read_bytes() == b"do not open this database"
    assert growth._git(source, "status", "--porcelain") == ""
    assert result["source_commit"] == growth._git(source, "rev-parse", "HEAD")


@pytest.mark.parametrize("policy", ["checkpoint", "vacuum", "incremental"])
def test_maintenance_reclaims_only_requested_space(tmp_path: Path, policy: str) -> None:
    path = tmp_path / "graph.db"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA auto_vacuum=" + ("INCREMENTAL" if policy == "incremental" else "NONE"))
    with GraphStore(path) as store:
        conn = store._conn
        conn.execute("CREATE TABLE payload (data BLOB)")
        conn.executemany("INSERT INTO payload VALUES (zeroblob(100000))", [()] * 20)
        conn.execute("DELETE FROM payload WHERE rowid > 1")
        before = growth.storage_sample(store)
        assert before["freelist_count"] > 10
        growth.apply_maintenance(store, policy, 3)
        after = growth.storage_sample(store)
        assert after["wal_bytes"] == 0
        assert conn.execute("SELECT length(data) FROM payload").fetchone()[0] == 100000
        if policy == "vacuum":
            assert after["freelist_count"] == 0
            assert after["allocated_bytes"] < before["allocated_bytes"]
        elif policy == "incremental":
            assert before["freelist_count"] - after["freelist_count"] == 3
        else:
            assert after["freelist_count"] == before["freelist_count"]


def test_diagnostics_detect_stale_references_and_broken_fts(snapshot: Path, tmp_path: Path) -> None:
    with GraphStore(tmp_path / "graph.db") as store:
        growth._pipeline(store, snapshot, None)
        assert growth._health(store)["ok"]
        store._conn.execute(
            "INSERT INTO flow_memberships (flow_id, node_id, position) VALUES (999999, 999999, 0)"
        )
        store._conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('delete-all')")
        health = growth._health(store)
        assert not health["ok"]
        assert health["dangling_flow_memberships"] >= 1
        assert health["fts_integrity"] != "ok"


def test_derived_drift_is_reported_not_repaired_or_ignored(snapshot: Path) -> None:
    original = growth._fingerprints
    calls = 0

    def drifting(store):
        nonlocal calls
        calls += 1
        result = original(store)
        if calls > 1:
            result["flows"] = "controlled drift"
        return result

    with patch.object(growth, "_fingerprints", side_effect=drifting):
        result = growth.run_case(
            snapshot, "none", growth.Config(cycles=1, files=1, query_repeats=1),
        )
    assert not result["correctness_ok"]
    assert "flows" in result["cycles"][0]["revert"]["mismatched_layers"]


@pytest.mark.parametrize("field", ["cycles", "files", "functions_per_file", "query_repeats",
                                    "maintenance_every", "incremental_pages"])
def test_config_rejects_zero_work(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        replace(growth.Config(), **{field: 0}).validate()


def test_cli_refuses_to_overwrite_and_reports_failed_correctness(tmp_path: Path) -> None:
    report = tmp_path / "result.json"
    with patch.object(growth, "run_benchmark", return_value={"correctness_ok": False}) as run:
        assert growth.main(["--output", str(report)]) == 2
        assert json.loads(report.read_text()) == {"correctness_ok": False}
        with pytest.raises(SystemExit) as exc:
            growth.main(["--output", str(report)])
        assert exc.value.code == 2
        assert run.call_count == 1


@pytest.mark.parametrize("name", ["../escaped.py", "/escaped.py", "C:/escaped.py",
                                  "subdir/C:/escaped.py"])
def test_snapshot_rejects_archive_paths_outside_destination(tmp_path: Path, name: str) -> None:
    def archive_command(*args, **kwargs):
        with tarfile.open(fileobj=kwargs["stdout"], mode="w") as archive:
            archive.addfile(tarfile.TarInfo(name))

    with patch.object(growth.subprocess, "run", side_effect=archive_command):
        with pytest.raises(ValueError, match="Unsafe archive path"):
            growth.snapshot_repository(tmp_path, "HEAD", tmp_path / "snapshot")
    assert not (tmp_path / "escaped.py").exists()


def test_worker_failure_is_not_reported_as_a_successful_case(source: Path) -> None:
    result = growth.run_benchmark(
        source, growth.Config(cycles=1, files=999, query_repeats=1), ["none"],
    )
    assert not result["correctness_ok"]
    assert result["cases"][0]["returncode"] == 1
    assert "indexed Python files" in result["cases"][0]["error"]
