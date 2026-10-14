"""SQLite maintenance preserves the forget workflow's data and transactions."""

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from code_review_graph.forget import _purge_orphan_embeddings, _referrer_files
from code_review_graph.graph import GraphStore
from code_review_graph.parser import EdgeInfo, NodeInfo
from code_review_graph.storage.base import FileMaintenanceStorage


def test_referrers_include_either_endpoint_and_preserve_stored_paths(tmp_path: Path) -> None:
    with GraphStore(tmp_path / "graph.db") as store:
        names = [f"module.py::symbol{i}" for i in range(1201)]
        for index in (0, 399, 400, 799, 800, 1200):
            store.upsert_edge(EdgeInfo("CALLS", "caller", names[index], "incoming.py", index))
        store.upsert_edge(EdgeInfo("TESTED_BY", names[-1], "test", "outgoing.py", 1))
        legacy_path = r"C:\repo\legacy.py"
        edge_id = store.upsert_edge(EdgeInfo("CALLS", "caller", names[0], legacy_path, 1))
        # Current parser records normalize paths; emulate a database written
        # before that normalization to exercise exact stored-path handling.
        store._conn.execute("UPDATE edges SET file_path = ? WHERE id = ?", (legacy_path, edge_id))
        store.upsert_edge(EdgeInfo("CALLS", "other", "unrelated", "unrelated.py", 1))
        # setlimit is available on Python 3.11+. Exercise the historical
        # parameter ceiling even on SQLite builds with a much higher default.
        if hasattr(store._conn, "setlimit"):
            store._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        assert store.find_referrer_files(iter(names + names)) == [
            legacy_path, "incoming.py", "outgoing.py",
        ]
        assert store.find_referrer_files([]) == []
        assert store.find_referrer_files(["absent"]) == []
        assert _referrer_files(store, set(names), {"incoming.py"}) == [
            legacy_path, "outgoing.py",
        ]


def test_purge_without_embeddings_does_not_create_storage(tmp_path: Path) -> None:
    with GraphStore(tmp_path / "graph.db") as store:
        assert store.purge_orphan_embeddings() == 0
        assert store._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'embeddings'"
        ).fetchone() is None


def test_purge_shares_transaction_and_retains_live_vectors(tmp_path: Path) -> None:
    with GraphStore(tmp_path / "graph.db") as store:
        store.upsert_node(NodeInfo(
            kind="Function", name="keep", file_path="module.py",
            line_start=1, line_end=2, language="python",
        ))
        store._conn.execute(
            "CREATE TABLE embeddings (qualified_name TEXT PRIMARY KEY, vector BLOB)"
        )
        store._conn.executemany(
            "INSERT INTO embeddings VALUES (?, ?)",
            [("module.py::keep", b"live-vector"), ("missing.py::gone", b"orphan-vector")],
        )
        store._conn.execute("BEGIN")
        store._conn.execute("INSERT INTO metadata VALUES ('uncommitted', 'value')")
        assert store.purge_orphan_embeddings() == 1
        assert store._conn.in_transaction
        assert [tuple(row) for row in store._conn.execute("SELECT * FROM embeddings")] == [
            ("module.py::keep", b"live-vector"),
        ]
        store.rollback()
        assert store.get_metadata("uncommitted") is None
        assert store._conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 2
        assert store.purge_orphan_embeddings() == 1
        assert store.purge_orphan_embeddings() == 0


def test_forget_helpers_accept_storage_without_a_sqlite_connection() -> None:
    class MemoryMaintenance:
        def find_referrer_files(self, qualified_names: Iterable[str]) -> list[str]:
            return ["forgotten.py", "survivor.py"] if "target" in qualified_names else []

        def purge_orphan_embeddings(self) -> int:
            return 2

    store: FileMaintenanceStorage = MemoryMaintenance()
    assert _referrer_files(store, {"target"}, {"forgotten.py"}) == ["survivor.py"]
    assert _purge_orphan_embeddings(store) == 2
