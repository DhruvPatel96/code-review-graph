"""Backend-neutral contracts for the first extracted storage capabilities.

These are structural protocols: the existing SQLite store implements them
without inheriting them. They intentionally expose no connection, cursor,
physical database path, SQL expression, or driver-specific row. They are not
yet the complete interface required by resolvers, search, and derived data.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Protocol

from .models import GraphEdge, GraphNode, GraphStats

if TYPE_CHECKING:
    from ..parser import EdgeInfo, NodeInfo


class FileMaintenanceStorage(Protocol):
    """Data access needed to repair a graph after forgetting files."""

    def find_referrer_files(self, qualified_names: Iterable[str]) -> list[str]:
        """Sorted unique stored file paths owning edges with either endpoint."""
        ...

    def purge_orphan_embeddings(self) -> int:
        """Remove orphan vectors, without provider calls or an implicit commit.

        Return the number removed; absent vector storage is a no-op.
        """
        ...


class GraphStorage(FileMaintenanceStorage, Protocol):
    """Core graph persistence contract; analysis capabilities remain separate.

    Qualified names identify nodes logically. Integer IDs remain available
    for existing consumers but need not survive a per-file replacement.
    Edges may have unresolved endpoints and retain distinct call sites.
    """

    def close(self) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def upsert_node(self, node: NodeInfo, file_hash: str = "") -> int: ...

    def upsert_edge(self, edge: EdgeInfo) -> int: ...

    def store_file_nodes_edges(
        self, file_path: str, nodes: list[NodeInfo], edges: list[EdgeInfo], fhash: str = ""
    ) -> None:
        """Atomically replace file-owned data, retaining other files' edges."""
        ...

    def store_file_batch(
        self, batch: list[tuple[str, list[NodeInfo], list[EdgeInfo], str]]
    ) -> None:
        """Apply all file replacements atomically, rolling back all on failure."""
        ...

    def remove_file_data(self, file_path: str) -> None:
        """Remove only file-owned nodes and edges; derived repair is separate."""
        ...

    def remove_file_permanently(self, file_path: str) -> int: ...

    def remove_files_permanently(
        self, file_paths: list[str], *, stored_paths: bool = False,
    ) -> int:
        """Atomically remove files, their endpoint references and embeddings.

        Return the number of distinct files removed. With ``stored_paths``,
        address exact inventory spellings rather than normalizing legacy paths.
        """
        ...

    def get_metadata(self, key: str) -> str | None: ...

    def set_metadata(self, key: str, value: str) -> None: ...

    def get_repo_root(self) -> str | None: ...

    def has_nodes(self) -> bool: ...

    def has_nodes_for_language(self, language: str) -> bool: ...

    def get_node(self, qualified_name: str) -> GraphNode | None: ...

    def get_node_by_id(self, node_id: int) -> GraphNode | None: ...

    def get_nodes_by_file(self, file_path: str) -> list[GraphNode]: ...

    def iter_nodes_by_file(self, file_path: str) -> Iterator[GraphNode]: ...

    def get_all_nodes(self, exclude_files: bool = True) -> list[GraphNode]: ...

    def get_edges_by_source(self, qualified_name: str) -> list[GraphEdge]: ...

    def get_edges_by_target(self, qualified_name: str) -> list[GraphEdge]: ...

    def iter_edges_by_source(self, qualified_name: str) -> Iterator[GraphEdge]: ...

    def iter_edges_by_target(self, qualified_name: str) -> Iterator[GraphEdge]: ...

    def get_all_edges(self) -> list[GraphEdge]: ...

    def get_all_files(self) -> list[str]: ...

    def get_file_hashes(self) -> dict[str, str]: ...

    def get_file_marker_paths(self) -> list[str]: ...

    def search_nodes(self, query: str, limit: int = 20) -> list[GraphNode]: ...

    def get_stats(self) -> GraphStats: ...
