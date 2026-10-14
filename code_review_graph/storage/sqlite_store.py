"""SQLite operations extracted from application workflows.

``graph.GraphStore`` inherits these operations so its existing imports and
connection lifecycle remain unchanged. The rest of the SQLite implementation
still lives in ``graph.py``; this is not yet a standalone alternative backend.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

# Each name is bound twice, keeping both IN clauses below the historical
# SQLite limit of 999 parameters in a single statement.
_REFERRER_CHUNK = 400


class SqliteFileMaintenance:
    """File-maintenance operations sharing the owning graph's connection."""

    _conn: sqlite3.Connection

    def find_referrer_files(self, qualified_names: Iterable[str]) -> list[str]:
        """Return sorted, distinct edge-owning files for either endpoint.

        Query edges directly: endpoints need not resolve to indexed nodes.
        Return the stored path spelling so legacy paths remain addressable.
        Filtering out files being forgotten is the caller's responsibility.
        """
        names = list(dict.fromkeys(qualified_names))
        referrers: set[str] = set()
        for start in range(0, len(names), _REFERRER_CHUNK):
            window = names[start:start + _REFERRER_CHUNK]
            placeholders = ",".join("?" for _ in window)
            rows = self._conn.execute(
                f"SELECT DISTINCT file_path FROM edges "
                f"WHERE target_qualified IN ({placeholders}) "
                f"OR source_qualified IN ({placeholders})",
                window + window,
            ).fetchall()
            referrers.update(row["file_path"] for row in rows)
        return sorted(referrers)

    def purge_orphan_embeddings(self) -> int:
        """Remove vectors without graph nodes using the existing connection.

        A graph without an embeddings table is a no-op. Do not initialize an
        embedding provider, open another writer, or commit a caller-owned
        transaction. This preserves the forget workflow's transaction scope.
        """
        has_table = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embeddings'"
        ).fetchone()
        if has_table is None:
            return 0
        cursor = self._conn.execute(
            "DELETE FROM embeddings WHERE NOT EXISTS ("
            "SELECT 1 FROM nodes WHERE nodes.qualified_name = embeddings.qualified_name"
            ")"
        )
        return max(cursor.rowcount, 0)
