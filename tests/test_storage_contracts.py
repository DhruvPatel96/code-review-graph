"""Core storage behavior through both the legacy constructor and the opener."""

from collections.abc import Callable
from pathlib import Path

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.parser import EdgeInfo, NodeInfo
from code_review_graph.storage.base import GraphStorage
from code_review_graph.storage.factory import open_graph_store

_OPENERS: list[Callable[[str | Path], GraphStorage]] = [GraphStore, open_graph_store]


def _function(path: str, name: str) -> NodeInfo:
    return NodeInfo(
        kind="Function", name=name, file_path=path,
        line_start=1, line_end=3, language="python",
    )


@pytest.mark.parametrize("opener", _OPENERS)
def test_replacement_and_deletion_preserve_edge_ownership(
    tmp_path: Path, opener: Callable[[str | Path], GraphStorage],
) -> None:
    """Replacement retains incoming edges; permanent removal deletes them."""
    path = tmp_path / "graph.db"
    first, second = str(tmp_path / "a.py"), str(tmp_path / "b.py")
    source, target = f"{first}::caller", f"{second}::callee"
    # This annotation checks structural compatibility when type-checking the
    # contract suite; assertions use only the backend-neutral interface.
    store: GraphStorage = opener(path)
    try:
        store.store_file_batch([
            (first, [_function(first, "caller")], [
                EdgeInfo("CALLS", source, target, first, 1),
                EdgeInfo("CALLS", source, target, first, 2),
                EdgeInfo("CALLS", source, "unresolved", first, 3),
            ], "first-hash"),
            (second, [_function(second, "callee")], [], "old-hash"),
        ])
        store.store_file_nodes_edges(second, [_function(second, "callee")], [], "new-hash")
        assert sorted(edge.line for edge in store.get_edges_by_target(target)) == [1, 2]
        assert len(store.get_edges_by_target("unresolved")) == 1
        assert store.find_referrer_files([target]) == [first]
        store.set_metadata("contract", "persisted")
    finally:
        store.close()

    store = opener(path)
    try:
        assert store.get_metadata("contract") == "persisted"
        node = store.get_node(target)
        assert node is not None and node.file_hash == "new-hash"
        assert store.remove_files_permanently([second, second]) == 1
        assert store.get_node(target) is None
        assert store.get_edges_by_target(target) == []
        assert [edge.target_qualified for edge in store.get_edges_by_source(source)] == [
            "unresolved",
        ]
    finally:
        store.close()


@pytest.mark.parametrize("opener", _OPENERS)
def test_failed_batch_preserves_all_committed_files(
    tmp_path: Path, opener: Callable[[str | Path], GraphStorage],
) -> None:
    """A serialization failure after replacement starts must roll it all back."""
    path = tmp_path / "graph.db"
    first, second = str(tmp_path / "a.py"), str(tmp_path / "b.py")
    store: GraphStorage = opener(path)
    try:
        store.store_file_batch([
            (first, [_function(first, "original")], [], "old-a"),
            (second, [_function(second, "original")], [], "old-b"),
        ])
        invalid = _function(second, "replacement")
        invalid.extra = {"not_serializable": object()}
        with pytest.raises(TypeError):
            store.store_file_batch([
                (first, [_function(first, "replacement")], [], "new-a"),
                (second, [invalid], [], "new-b"),
            ])
    finally:
        store.close()

    store = opener(path)
    try:
        assert {node.qualified_name for node in store.get_all_nodes()} == {
            f"{first}::original", f"{second}::original",
        }
        first_node = store.get_node(f"{first}::original")
        second_node = store.get_node(f"{second}::original")
        assert first_node is not None and first_node.file_hash == "old-a"
        assert second_node is not None and second_node.file_hash == "old-b"
    finally:
        store.close()
