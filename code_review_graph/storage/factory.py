"""Store opening boundary, initially preserving the SQLite constructor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import GraphStore


def open_graph_store(db_path: str | Path) -> GraphStore:
    """Open the existing SQLite store without changing recovery or defaults.

    Resolve the legacy class at call time to preserve integrations that patch
    ``graph.GraphStore``. The concrete return type is intentional during the
    extraction: callers still use analysis and derived-data capabilities not
    yet covered by the core storage protocol. Backend selection is deferred
    until those contracts and the remaining direct SQL callers are migrated.
    """
    from ..graph import GraphStore

    return GraphStore(db_path)
