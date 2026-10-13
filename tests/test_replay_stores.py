"""Core replay contract: foreign references, multiedges, rollback and persistence."""

import pytest

from code_review_graph.eval.benchmarks.replay_stores import (
    LadybugReplayStore,
    SqliteReplayStore,
    deleted_image,
    normalize,
    patch_for,
    read_sqlite,
    reference_traversal,
)
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build


@pytest.fixture(params=[SqliteReplayStore, LadybugReplayStore])
def replay(request, tmp_path):
    if request.param is LadybugReplayStore:
        pytest.importorskip("ladybug")
    store = request.param(tmp_path / "replay.db")
    yield store
    store.close()


@pytest.fixture
def image(tmp_path):
    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "a.py").write_text("def target():\n    return 1\n")
    (tree / "b.py").write_text(
        "from a import target\ndef caller():\n    target()\n    target()\n    missing()\n"
    )
    store = GraphStore(tmp_path / "oracle.db")
    full_build(tree, store)
    result = read_sqlite(store)
    store.close()
    return result


def test_replay_contract(replay, image):
    files = sorted({n["file_path"] for n in image["nodes"]})
    replay.replace_files(files, image)
    assert replay.snapshot() == image
    for n in image["nodes"]:
        qn = n["qualified_name"]
        for depth in (1, 2, 3):
            assert replay.traverse(qn, depth) == reference_traversal(image, qn, depth)
    target = [f for f in files if f.endswith("a.py")]
    replay.replace_files(target, {"nodes": [], "edges": []})
    expected = normalize(
        {k: [r for r in rows if r["file_path"] not in target] for k, rows in image.items()}
    )
    assert replay.snapshot() == expected  # Foreign references survive replacement.
    replay.replace_files(target, patch_for(image, target))
    assert replay.snapshot() == image
    replay.remove_files(target)
    assert replay.snapshot() == deleted_image(image, target)


def test_rollback_and_checkpoint(replay, image):
    files = sorted({n["file_path"] for n in image["nodes"]})
    replay.replace_files(files, image)
    with pytest.raises(RuntimeError, match="injected"):
        with replay.transaction():
            if isinstance(replay, SqliteReplayStore):
                replay.conn.execute("DELETE FROM edges")
            else:
                replay.query("MATCH ()-[e:CodeEdge]->() DELETE e")
            raise RuntimeError("injected")
    assert replay.snapshot() == image
    replay.checkpoint()


def test_reject_foreign_patch(replay, image):
    with pytest.raises(ValueError):
        replay.replace_files(["unrelated"], image)
    assert replay.snapshot() == {"nodes": [], "edges": []}


def test_duplicate_patch_is_atomic(replay, image):
    files = sorted({n["file_path"] for n in image["nodes"]})
    replay.replace_files(files, image)
    invalid = {"nodes": image["nodes"] * 2, "edges": image["edges"]}
    with pytest.raises(ValueError, match="Duplicate"):
        replay.replace_files(files, invalid)
    assert replay.snapshot() == image


def test_coordinator_runs_real_workers(tmp_path):
    pytest.importorskip("ladybug")
    from code_review_graph.eval.benchmarks.ladybug_replay import run
    from code_review_graph.eval.benchmarks.storage_growth import _git

    tree = tmp_path / "repo"
    tree.mkdir()
    (tree / "a.py").write_text("def a():\n    return 1\n")
    _git(tree, "init", "-q")
    _git(tree, "add", "a.py")
    _git(tree, "commit", "-qm", "fixture")
    report = run(tree, 1, 1, 128)
    assert len(report["results"]) == 2
    assert all(r["correctness_ok"] for r in report["results"]), report
    assert all(r["updates"] == 2 for r in report["results"])


def test_cycles_duplicates_and_unresolved_paths(replay, image):
    first, second = image["nodes"][:2]
    template = image["edges"][0]
    a, b = first["qualified_name"], second["qualified_name"]
    edges = []
    for source, target in ((a, b), (b, a), (a, b), ("missing", a)):
        edges.append(
            {**template, "kind": "CALLS", "source_qualified": source, "target_qualified": target}
        )
    image = normalize({"nodes": image["nodes"], "edges": edges})
    files = sorted({r["file_path"] for rows in image.values() for r in rows})
    replay.replace_files(files, image)
    assert replay.snapshot() == image
    assert len(replay.neighbors(b, True)) == 2
    assert replay.traverse(a, 3) == [b]
    assert replay.traverse("missing", 3) == []
