"""Isolated core-row replay; deliberately not a complete backend benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ...graph import GraphStore
from .replay_stores import (
    LadybugReplayStore,
    SqliteReplayStore,
    canonical,
    patch_for,
    read_sqlite,
    reference_traversal,
)
from .storage_growth import _git, _peak_rss, _pipeline, snapshot_repository


def worker(request: dict) -> dict:
    baseline, mutated = request["baseline"], request["mutated"]
    files = request["files"]
    cls = SqliteReplayStore if request["backend"] == "sqlite" else LadybugReplayStore
    with tempfile.TemporaryDirectory(prefix="crg-replay-db-") as scratch:
        root = Path(scratch)
        store = cls(root / "graph.db", buffer_mb=request["buffer_mb"])
        try:
            started = time.perf_counter()
            store.replace_files(
                sorted({r["file_path"] for rows in baseline.values() for r in rows}), baseline
            )
            load = time.perf_counter() - started
            if store.snapshot() != baseline:
                raise RuntimeError("Baseline mismatch")
            timings = []
            for _ in range(request["cycles"]):
                for expected in (mutated, baseline):
                    started = time.perf_counter()
                    store.replace_files(files, patch_for(expected, files))
                    timings.append(time.perf_counter() - started)
                    if store.snapshot() != expected:
                        raise RuntimeError("Replay mismatch")
            queries = []
            correctness_errors = []
            for seed in request["seeds"]:
                for depth in (2, 3):
                    expected = reference_traversal(baseline, seed, depth)
                    samples = []
                    for _ in range(3):
                        started = time.perf_counter()
                        actual = store.traverse(seed, depth)
                        samples.append(time.perf_counter() - started)
                        if actual != expected:
                            missing = sorted(set(expected) - set(actual))
                            extra = sorted(set(actual) - set(expected))
                            correctness_errors.append(
                                f"Traversal mismatch; missing={missing[:20]!r}, "
                                f"extra={extra[:20]!r}; seed={seed!r}, depth={depth}, "
                                f"expected_count={len(expected)}, actual_count={len(actual)}"
                            )
                            break
                    queries.append(
                        {
                            "seed": seed,
                            "depth": depth,
                            "results": len(expected),
                            "median_seconds": statistics.median(samples),
                        }
                    )
                for incoming in (True, False):
                    column = "target_qualified" if incoming else "source_qualified"
                    expected_edges = sorted(
                        [e for e in baseline["edges"] if e[column] == seed], key=canonical
                    )
                    if store.neighbors(seed, incoming) != expected_edges:
                        correctness_errors.append(
                            f"Neighbor mismatch for {seed!r}, incoming={incoming}"
                        )
                if correctness_errors:
                    break
            before = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            started = time.perf_counter()
            store.checkpoint()
            checkpoint = time.perf_counter() - started
            after = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            store.close()
            store = cls(root / "graph.db", buffer_mb=request["buffer_mb"])
            if store.snapshot() != baseline:
                raise RuntimeError("Reopen mismatch")
            return {
                "backend": request["backend"],
                "correctness_ok": not correctness_errors,
                "correctness_errors": correctness_errors,
                "load_seconds": load,
                "updates": len(timings),
                "update_median_seconds": statistics.median(timings),
                "update_max_seconds": max(timings),
                "update_seconds": timings,
                "worker_peak_rss_bytes": _peak_rss(),
                "query_samples": queries,
                "bytes_before_checkpoint": before,
                "bytes_after_checkpoint": after,
                "checkpoint_seconds": checkpoint,
            }
        finally:
            store.close()


def run(
    source: Path,
    cycles: int,
    count: int,
    buffer_mb: int = 128,
    backends: tuple[str, ...] = ("sqlite", "ladybug"),
) -> dict:
    if cycles < 1 or count < 1 or buffer_mb < 1:
        raise ValueError("cycles and files must be positive")
    revision = _git(source, "rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(prefix="crg-replay-source-") as scratch:
        root = Path(scratch)
        tree = root / "repo"
        snapshot_repository(source, revision, tree)
        oracle = GraphStore(root / "oracle.db")
        try:
            _pipeline(oracle, tree, None)
            baseline = read_sqlite(oracle)
            files = sorted(
                f for f in oracle.get_all_files() if f.endswith(".py") and Path(f).is_file()
            )[:count]
            if len(files) != count:
                raise ValueError("Not enough Python files")
            originals = {f: Path(f).read_bytes() for f in files}
            for i, (f, original) in enumerate(originals.items()):
                Path(f).write_bytes(
                    original
                    + (f"\n\ndef crg_ladybug_probe_{i}(value):\n    return value\n").encode()
                )
            _pipeline(oracle, tree, files)
            mutated = read_sqlite(oracle)
            for f, original in originals.items():
                Path(f).write_bytes(original)
            _pipeline(oracle, tree, files)
            if read_sqlite(oracle) != baseline:
                raise RuntimeError("SQLite source pipeline drift after revert")
        finally:
            oracle.close()
        owners = sorted(
            {
                r["file_path"]
                for image in (baseline, mutated)
                for rows in image.values()
                for r in rows
            }
        )
        changed = [f for f in owners if patch_for(baseline, [f]) != patch_for(mutated, [f])]
        names = {n["qualified_name"] for n in baseline["nodes"]}
        degrees: dict[str, int] = {}
        for e in baseline["edges"]:
            if e["target_qualified"] in names:
                qn = e["target_qualified"]
                degrees[qn] = degrees.get(qn, 0) + 1
        seeds = sorted(degrees, key=lambda q: (-degrees[q], q))[:5]
        report: dict[str, Any] = {
            "source_revision": revision,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ladybug": importlib.metadata.version("ladybug"),
            "scope": "core rows only; no derived data, FTS, vectors or MCP",
            "cycles": cycles,
            "input_files": count,
            "changed_owners": len(changed),
            "nodes": len(baseline["nodes"]),
            "edges": len(baseline["edges"]),
            "baseline_sha256": hashlib.sha256(canonical(baseline).encode()).hexdigest(),
            "buffer_mb": buffer_mb,
            "ladybug_threads": 1,
            "sqlite": sqlite3.sqlite_version,
            "harness_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(__file__).with_name("replay_stores.py"))
            },
            "results": [],
        }
        for backend in backends:
            request = root / f"{backend}.json"
            request.write_text(
                json.dumps(
                    {
                        "backend": backend,
                        "baseline": baseline,
                        "mutated": mutated,
                        "files": changed,
                        "seeds": seeds,
                        "cycles": cycles,
                        "buffer_mb": buffer_mb,
                    }
                )
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "code_review_graph.eval.benchmarks.ladybug_replay",
                    "--worker",
                    str(request),
                ],
                text=True,
                capture_output=True,
                timeout=3600,
            )
            if result.returncode:
                report["results"].append(
                    {
                        "backend": backend,
                        "correctness_ok": False,
                        "returncode": result.returncode,
                        "stderr": result.stderr[-4000:],
                    }
                )
            else:
                report["results"].append(json.loads(result.stdout))
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--files", type=int, default=20)
    parser.add_argument("--buffer-mb", type=int, default=128)
    parser.add_argument("--backend", choices=("both", "sqlite", "ladybug"), default="both")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.pop("CRG_DATA_DIR", None)
    os.environ["CRG_SERIAL_PARSE"] = "1"
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker.read_text()))))
        return 0
    if args.output is None or args.output.exists():
        parser.error("--output must name a new report file")
    backends = ("sqlite", "ladybug") if args.backend == "both" else (args.backend,)
    report = run(args.source.resolve(), args.cycles, args.files, args.buffer_mb, backends)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(r["correctness_ok"] for r in report["results"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
