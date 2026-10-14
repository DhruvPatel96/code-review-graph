"""Isolated, offline SQLite churn experiment (not part of the default eval run).

Run with ``python -m code_review_graph.eval.benchmarks.storage_growth --help``.
Each policy runs in a fresh process over the same committed Git snapshot.
Only scratch databases receive maintenance; production defaults are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar
from unittest.mock import patch

from ... import __version__
from ...embeddings import EmbeddingProvider, EmbeddingStore, embed_all_nodes
from ...graph import GraphStore
from ...incremental import full_build, incremental_update
from ...search import _fts_search
from ...tools.build import _run_postprocess
from .incremental_fidelity import _PROJECTORS

_MODULE = "code_review_graph.eval.benchmarks.storage_growth"
POLICIES = ("none", "checkpoint", "vacuum", "incremental")
_TABLES = (
    "nodes", "edges", "metadata", "flows", "flow_memberships", "communities",
    "community_summaries", "flow_snapshots", "risk_index", "nodes_fts_state", "embeddings",
)
_T = TypeVar("_T")


@dataclass(frozen=True)
class Config:
    cycles: int = 100
    files: int = 20
    functions_per_file: int = 4
    query_repeats: int = 5
    maintenance_every: int = 10
    incremental_pages: int = 128
    vector_dim: int = 0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            minimum = 0 if name == "vector_dim" else 1
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


def _git_env() -> dict[str, str]:
    # Inherited worktree/index overrides must never redirect scratch Git writes.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    return env


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=" + os.devnull,
         "-c", "user.name=Storage benchmark", "-c", "user.email=benchmark@localhost",
         "-c", "commit.gpgsign=false", *args],
        cwd=root, env=_git_env(), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout.strip()


def snapshot_repository(source: Path, revision: str, destination: Path) -> dict[str, Any]:
    """Materialize a pinned commit, never the user's dirty working tree or DB.

    Extract regular archive members explicitly: no tar extraction of links,
    device files, or paths that could escape the newly created directory.
    """
    destination.mkdir()
    skipped: list[str] = []
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["git", "archive", "--format=tar", revision], cwd=source, env=_git_env(),
            stdin=subprocess.DEVNULL, stdout=archive, stderr=subprocess.PIPE,
            check=True, timeout=120,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:") as entries:
            for member in entries:
                parts = PurePosixPath(member.name).parts
                if (not parts or PurePosixPath(member.name).is_absolute()
                        or ".." in parts or "\\" in member.name or ":" in member.name):
                    raise ValueError(f"Unsafe archive path: {member.name!r}")
                if member.isdir():
                    continue
                if (not member.isfile() or parts[0] in {".git", ".beads"}
                        or (parts[0] == ".code-review-graph"
                            and member.name != ".code-review-graph/languages.toml")
                        or parts[0].startswith(".code-review-graph.db")):
                    skipped.append(member.name)
                    continue
                target = destination.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                content = entries.extractfile(member)
                if content is None:
                    raise ValueError(f"Cannot read archive member {member.name!r}")
                with content, target.open("xb") as output:
                    shutil.copyfileobj(content, output)
    _git(destination, "-c", "init.templateDir=", "init", "-q")
    _git(destination, "add", "--all", "--force")
    _git(destination, "commit", "-q", "--no-verify", "-m", "Benchmark snapshot")
    return {"skipped_archive_members": skipped, "snapshot_commit": _git(destination, "rev-parse",
                                                                                   "HEAD")}


def _timed(operation: Callable[[], _T]) -> tuple[_T, dict[str, float]]:
    before = os.times()
    start = time.perf_counter()
    value = operation()
    after = os.times()
    cpu = (after.user + after.system + after.children_user + after.children_system
           - before.user - before.system - before.children_user - before.children_system)
    return value, {"wall_s": time.perf_counter() - start, "cpu_s": max(0.0, cpu)}


def _sizes(path: Path) -> dict[str, int]:
    sizes = {}
    for name, suffix in (("db_bytes", ""), ("wal_bytes", "-wal"), ("shm_bytes", "-shm")):
        file = Path(str(path) + suffix)
        sizes[name] = file.stat().st_size if file.exists() else 0
    return {**sizes, "total_bytes": sum(sizes.values())}


def storage_sample(store: GraphStore) -> dict[str, Any]:
    """Observe without checkpointing: main-file size may lag committed WAL pages."""
    conn = store._conn
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    pages = conn.execute("PRAGMA page_count").fetchone()[0]
    free = conn.execute("PRAGMA freelist_count").fetchone()[0]
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    counts = {name: conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
              if name in tables else 0 for name in _TABLES}
    return {
        **_sizes(store.db_path), "page_size": page_size, "page_count": pages,
        "freelist_count": free, "allocated_bytes": pages * page_size,
        "free_reusable_bytes": free * page_size, "non_freelist_bytes": (pages - free) * page_size,
        "row_counts": counts,
    }


def apply_maintenance(store: GraphStore, policy: str, pages: int) -> dict[str, Any]:
    """Maintain only a scratch connection; drain incremental-vacuum results."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    if policy == "none":
        return {"policy": policy}
    if policy == "vacuum":
        store._conn.execute("VACUUM").fetchall()
    elif policy == "incremental":
        if pages < 1:
            raise ValueError("incremental vacuum page budget must be positive")
        # PRAGMA does not accept bound parameters. The interpolated value is
        # a validated integer, never SQL supplied by the user.
        store._conn.execute(f"PRAGMA incremental_vacuum({int(pages)})").fetchall()
    checkpoint = tuple(store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
    if checkpoint[0] != 0:
        raise RuntimeError(f"Checkpoint could not finish: {checkpoint}")
    return {"policy": policy, "checkpoint": checkpoint}


def _fingerprint(store: GraphStore) -> str:
    """Core content, including signatures and resolution, excluding generated IDs."""
    digest = hashlib.sha256()
    for table in ("nodes", "edges"):
        columns = [row[1] for row in store._conn.execute(f"PRAGMA table_info({table})")
                   if row[1] not in {"id", "updated_at", "community_id"}]
        rows = store._conn.execute(f"SELECT {','.join(columns)} FROM {table}").fetchall()
        for row in sorted(json.dumps(tuple(row), ensure_ascii=True) for row in rows):
            digest.update(row.encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _fingerprints(store: GraphStore) -> dict[str, str | None]:
    """Reuse row-ID-independent derived projections from the fidelity benchmark."""
    result: dict[str, str | None] = {"core": _fingerprint(store)}
    for name, project in _PROJECTORS.items():
        if name in {"nodes", "edges", "metadata"}:
            continue
        projection = project(store._conn)
        result[name] = None if projection is None else hashlib.sha256(
            json.dumps(projection, sort_keys=True).encode()).hexdigest()
    if store._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='embeddings'"
    ).fetchone():
        rows = [tuple(row) for row in store._conn.execute(
            "SELECT qualified_name, hex(vector), text_hash, provider FROM embeddings "
            "ORDER BY qualified_name"
        )]
        result["embeddings"] = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    return result


def _health(store: GraphStore) -> dict[str, Any]:
    """Check index consistency and dangling derived references without repairing them."""
    conn = store._conn
    result = {}
    for label, query in {
        "dangling_flow_memberships": "SELECT count(*) FROM flow_memberships m "
        "LEFT JOIN nodes n ON n.id=m.node_id LEFT JOIN flows f ON f.id=m.flow_id "
        "WHERE n.id IS NULL OR f.id IS NULL",
        "dangling_node_communities": "SELECT count(*) FROM nodes n LEFT JOIN communities c "
        "ON c.id=n.community_id WHERE n.community_id IS NOT NULL AND c.id IS NULL",
    }.items():
        result[label] = conn.execute(query).fetchone()[0]
    try:
        conn.execute("INSERT INTO nodes_fts(nodes_fts, rank) VALUES ('integrity-check', 1)")
        result["fts_integrity"] = "ok"
    except sqlite3.DatabaseError as exc:
        result["fts_integrity"] = str(exc)
    result["ok"] = all(result[key] == 0 for key in result if key.startswith("dangling_")) \
        and result["fts_integrity"] == "ok"
    return result


class _SyntheticProvider(EmbeddingProvider):
    """Deterministic vectors exercise storage/search, not model quality or inference."""

    def __init__(self, dimension: int):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"benchmark:synthetic:{self.dimension}"

    def embed_query(self, text: str) -> list[float]:
        raw = hashlib.shake_256(text.encode()).digest(self.dimension)
        return [(value - 127.5) / 127.5 for value in raw]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def _pipeline(store: GraphStore, tree: Path, changed: list[str] | None) -> dict[str, Any]:
    operation = (lambda: full_build(tree, store, recurse_submodules=False)) if changed is None \
        else (lambda: incremental_update(tree, store, changed_files=changed))
    result, build_timing = _timed(operation)
    warnings, post_timing = _timed(lambda: _run_postprocess(
        store, result, "full", full_rebuild=changed is None,
        changed_files=result.get("changed_files"), repo_root=str(tree),
    ))
    if result.get("errors") or warnings:
        raise RuntimeError(f"Incomplete build: errors={result.get('errors')}, warnings={warnings}")
    return {"build": build_timing, "postprocess": post_timing,
            "postprocess_stages": result.get("postprocess_timing", {}),
            "files_processed": result.get("files_parsed", result.get("files_updated", 0))}


def _latencies(operation: Callable[[], Any], repeats: int) -> dict[str, Any]:
    # Report the first call separately instead of silently folding warmup into p50.
    first_value, first = _timed(operation)
    samples = [_timed(operation)[1]["wall_s"] * 1000 for _ in range(repeats)]
    ordered = sorted(samples)
    result_count = len(first_value) if isinstance(first_value, list) else (
        first_value.get("total_impacted") if isinstance(first_value, dict) else None)
    return {"first_ms": first["wall_s"] * 1000, "first_result_count": result_count,
            "samples_ms": samples,
            "p50_ms": statistics.median(samples),
            "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1]}


def _queries(store: GraphStore, embeddings: EmbeddingStore | None, repeats: int) -> dict[str, Any]:
    row = store._conn.execute(
        "SELECT n.qualified_name, n.name, n.file_path FROM nodes n "
        "LEFT JOIN edges e ON e.target_qualified=n.qualified_name AND e.kind='CALLS' "
        "WHERE n.kind IN ('Function','Method') GROUP BY n.id "
        "ORDER BY count(e.id) DESC, n.qualified_name LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError("Snapshot has no function to query")
    qn, name, file_path = row
    operations: dict[str, Callable[[], Any]] = {
        "incoming_edges": lambda: store.get_edges_by_target(qn),
        "outgoing_edges": lambda: store.get_edges_by_source(qn),
        "impact_2hop": lambda: store.get_impact_radius([file_path], max_depth=2),
        "blast_radius_3hop": lambda: store.get_impact_radius([file_path], max_depth=3),
        "fts": lambda: _fts_search(store._conn, name, limit=20),
    }
    if embeddings is not None:
        operations["synthetic_vector_search"] = lambda: embeddings.search(name, limit=20)
    return {"target": qn, "query": name,
            "latencies": {key: _latencies(op, repeats) for key, op in operations.items()}}


def _allocation(store: GraphStore) -> dict[str, Any]:
    try:
        return {"available": True, "bytes_by_object": dict(store._conn.execute(
            "SELECT name, sum(pgsize) FROM dbstat GROUP BY name ORDER BY name"
        ))}
    except sqlite3.OperationalError as exc:
        return {"available": False, "reason": str(exc)}


def _peak_rss() -> int | None:
    try:
        import resource
    except ImportError:  # Windows: unavailable rather than a fabricated zero.
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    """Policy measurements only: exclude the end-of-experiment vacuum."""
    samples = [report["cold_sample"], report["final_sample"]]
    updates = []
    maintenance_s = 0.0
    for cycle in report["cycles"]:
        samples.append(cycle["after_maintenance"])
        maintenance_s += cycle.get("maintenance", {}).get("wall_s", 0.0)
        for phase in ("mutate", "revert"):
            step = cycle[phase]
            samples.append(step["storage"])
            updates.append(sum(step.get(stage, {}).get("wall_s", 0.0)
                               for stage in ("build", "postprocess", "embeddings")))
    ordered = sorted(updates)
    return {
        "update_count": len(updates),
        "update_p50_s": statistics.median(updates),
        "update_p95_s": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "maintenance_total_s": maintenance_s,
        "update_plus_maintenance_total_s": sum(updates) + maintenance_s,
        "observed_peak_database_bytes": max(sample["total_bytes"] for sample in samples),
        "growth_bytes_per_update": (report["final_sample"]["total_bytes"]
                                    - report["cold_sample"]["total_bytes"]) / len(updates),
        "cycles_with_drift": sum(bool(cycle["revert"]["mismatched_layers"])
                                 for cycle in report["cycles"]),
    }


def run_case(snapshot: Path, policy: str, config: Config) -> dict[str, Any]:
    """Run one policy in private directories. The coordinator uses a fresh process."""
    config.validate()
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    with tempfile.TemporaryDirectory(prefix="crg-storage-case-") as scratch:
        root = Path(scratch).resolve()
        tree = root / "repo"
        shutil.copytree(snapshot, tree, symlinks=True)
        db = root / "graph.db"
        # Must be set before the first schema object, not retrofitted to a live DB.
        with closing(sqlite3.connect(db)) as setup:
            setup.execute("PRAGMA auto_vacuum=" + ("INCREMENTAL" if policy == "incremental"
                                                  else "NONE"))
        store, open_timing = _timed(lambda: GraphStore(db))
        embeddings = None
        try:
            cold = _pipeline(store, tree, None)
            cold["open_store"] = open_timing
            candidates = sorted(Path(p).relative_to(tree).as_posix() for p in store.get_all_files()
                                if p.endswith(".py") and Path(p).is_file()
                                and not Path(p).is_symlink()
                                and Path(p).resolve().is_relative_to(tree))
            if len(candidates) < config.files:
                raise ValueError(
                    f"Need {config.files} indexed Python files; found {len(candidates)}")
            selected = candidates[:config.files]
            originals = {rel: (tree / rel).read_bytes() for rel in selected}
            if config.vector_dim:
                # Inject only the provider factory inside this isolated benchmark
                # process. The actual EmbeddingStore write/search paths are unchanged.
                with patch("code_review_graph.embeddings.get_provider",
                           return_value=_SyntheticProvider(config.vector_dim)):
                    embeddings = EmbeddingStore(db)
                _, cold["embeddings"] = _timed(lambda: embed_all_nodes(store, embeddings))
            baseline = _fingerprints(store)
            cold_health = _health(store)
            report: dict[str, Any] = {
                "policy": policy, "selected_files": selected, "cold_build": cold,
                "auto_vacuum": store._conn.execute("PRAGMA auto_vacuum").fetchone()[0],
                "wal_autocheckpoint": store._conn.execute(
                    "PRAGMA wal_autocheckpoint").fetchone()[0],
                "cold_sample": storage_sample(store), "cold_queries": _queries(
                    store, embeddings, config.query_repeats), "cycles": [],
                "cold_health": cold_health,
                "correctness_ok": cold_health["ok"],
                "unchecked_layers": [name for name, value in baseline.items() if value is None],
            }
            mutated_fingerprint = None
            for cycle in range(1, config.cycles + 1):
                record: dict[str, Any] = {"cycle": cycle}
                for phase in ("mutate", "revert"):
                    for index, (rel, original) in enumerate(originals.items()):
                        added = ""
                        if phase == "mutate":
                            for number in range(config.functions_per_file):
                                name = f"crg_storage_probe_{index}_{number}"
                                target = f"crg_storage_probe_{index}_{number - 1}(value)" \
                                    if number else "value"
                                added += f'\n\ndef {name}(value):\n    """Storage probe."""\n' \
                                    f"    return {target}\n"
                        (tree / rel).write_bytes(original + added.encode())
                    phase_result = _pipeline(store, tree, selected)
                    if embeddings is not None:
                        _, phase_result["embeddings"] = _timed(
                            lambda: embed_all_nodes(store, embeddings))
                    # Sample before diagnostics and before optional maintenance.
                    phase_result["storage"] = storage_sample(store)
                    actual, phase_result["verification"] = _timed(lambda: _fingerprints(store))
                    if phase == "mutate" and mutated_fingerprint is None:
                        mutated_fingerprint = actual
                    expected = baseline if phase == "revert" else mutated_fingerprint
                    phase_result["core_matches_expected"] = actual["core"] == expected["core"]
                    phase_result["mismatched_layers"] = [
                        name for name in set(actual) | set(expected)
                        if actual.get(name) != expected.get(name)
                    ]
                    phase_result["mismatched_layers"].sort()
                    phase_result["health"] = _health(store)
                    phase_result["processed_selected_files"] = phase_result["files_processed"] \
                        >= len(selected)
                    report["correctness_ok"] &= (actual == expected
                                                 and phase_result["health"]["ok"]
                                                 and phase_result["processed_selected_files"])
                    record[phase] = phase_result
                if cycle % config.maintenance_every == 0 or cycle == config.cycles:
                    details, timing = _timed(lambda: apply_maintenance(
                        store, policy, config.incremental_pages))
                    record["maintenance"] = {**details, **timing}
                record["after_maintenance"] = storage_sample(store)
                report["cycles"].append(record)
                if cycle % 10 == 0:
                    print(f"{policy}: {cycle}/{config.cycles} cycles", file=sys.stderr, flush=True)
            report["final_queries"] = _queries(store, embeddings, config.query_repeats)
            report["final_allocation"] = _allocation(store)
            report["final_sample"] = storage_sample(store)
            report["peak_policy_rss_bytes"] = _peak_rss()
            # End-of-experiment measurements are separate from the policy curve.
            report["end_diagnostics"] = []
            for action in ("checkpoint", "vacuum"):
                details, timing = _timed(lambda: apply_maintenance(
                    store, action, config.incremental_pages))
                report["end_diagnostics"].append({**details, **timing,
                                                   "storage": storage_sample(store)})
            report["maintenance_preserved_content"] = _fingerprints(store) == actual
            report["correctness_ok"] &= report["maintenance_preserved_content"]
        finally:
            if embeddings is not None:
                embeddings.close()
            store.close()
        report["after_close"] = _sizes(db)
        report["peak_worker_rss_bytes"] = _peak_rss()
        report["summary"] = _summary(report)
        return report


def run_benchmark(source: Path, config: Config, policies: list[str]) -> dict[str, Any]:
    config.validate()
    if not policies or any(policy not in POLICIES for policy in policies):
        raise ValueError(f"Choose policies from {POLICIES}")
    source = Path(_git(source.resolve(), "rev-parse", "--show-toplevel"))
    revision = _git(source, "rev-parse", "HEAD")
    versions = {}
    for package in ("tree-sitter", "tree-sitter-language-pack", "networkx", "igraph"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report: dict[str, Any] = {
        "format_version": 1, "source": str(source), "source_commit": revision,
        "working_tree_dirty": bool(_git(source, "status", "--porcelain")),
        "config": asdict(config), "python": platform.python_version(),
        "platform": platform.platform(), "sqlite": sqlite3.sqlite_version,
        "code_review_graph": __version__, "cases": [],
        "dependencies": versions,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "methodology": {
            "snapshot": "git archive HEAD; regular files only; dirty and untracked files excluded",
            "pipeline": "full_build / incremental_update + MCP full postprocessing",
            "updates": "explicit batches with stale reconciliation; no Git diff timing",
            "parser": "serial in fresh worker per policy; PYTHONHASHSEED=0",
            "cold": "fresh database, not a cold operating-system file cache",
            "queries": "first call plus warm samples at cold-build and final-revert boundaries",
            "memory": "worker process lifetime high-water RSS, including diagnostics",
            "disk": "file lengths at phase boundaries; excludes transient VACUUM temp files",
            "correctness": "core and derived revert parity against cold build; mutation "
                           "repeatability; FTS and reference checks. First mutated state is "
                           "not independently compared with a clean rebuild",
            "vectors": "synthetic deterministic provider; real BLOB writes and exact cosine scan; "
                       "no model inference or semantic quality measurement" if config.vector_dim
                       else "not measured; embeddings disabled",
            "maintenance": "after complete cycles at configured interval and last cycle; "
                           "end checkpoint/VACUUM diagnostics excluded from policy samples",
        },
    }
    with tempfile.TemporaryDirectory(prefix="crg-storage-benchmark-") as scratch:
        root = Path(scratch).resolve()
        snapshot = root / "snapshot"
        report["snapshot"] = snapshot_repository(source, revision, snapshot)
        for policy in policies:
            request = root / f"{policy}.json"
            request.write_text(json.dumps({"snapshot": str(snapshot), "policy": policy,
                                           "config": asdict(config)}), encoding="utf-8")
            env = _git_env()
            for key in list(env):
                if key.startswith("CRG_"):
                    del env[key]
            env.update(CRG_SERIAL_PARSE="1", CRG_HOME=str(root / "home"), PYTHONHASHSEED="0")
            print(f"Running {policy} on {revision[:12]}", file=sys.stderr, flush=True)
            child = subprocess.run(
                [sys.executable, "-m", _MODULE, "--worker", str(request)],
                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True,
                check=False,
            )
            if child.returncode:
                report["cases"].append({"policy": policy, "error": child.stdout[-4000:],
                                        "returncode": child.returncode, "correctness_ok": False})
            else:
                report["cases"].append(json.loads(child.stdout))
    report["correctness_ok"] = all(case["correctness_ok"] for case in report["cases"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help="New JSON report path (never overwritten)")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    for name, default in asdict(Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    if args.worker:
        request = json.loads(args.worker.read_text(encoding="utf-8"))
        try:
            result = run_case(
                Path(request["snapshot"]), request["policy"], Config(**request["config"]))
        except (OSError, RuntimeError, ValueError, sqlite3.Error,
                subprocess.SubprocessError) as exc:
            print(json.dumps({"error": str(exc), "type": type(exc).__name__}))
            return 1
        print(json.dumps(result))
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        parser.error("--output already exists; choose a new report path")
    if not args.output.parent.is_dir():
        parser.error("--output parent directory must already exist")
    config = Config(**{name: getattr(args, name) for name in asdict(Config())})
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    result = run_benchmark(args.repo, config, args.policies)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(f"Report: {args.output}")
    return 0 if result["correctness_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
