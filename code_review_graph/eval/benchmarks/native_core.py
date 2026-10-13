"""Measure the opt-in Rust Python-definition/containment slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

from ...incremental import collect_all_files
from ...native import NativeParseResult, core_version, parse_python_batch
from ...parser import CodeParser
from .storage_growth import _git, snapshot_repository


def _projection(nodes: list, edges: list) -> tuple[list[tuple], list[tuple]]:
    definitions = sorted(
        (
            (
                node.kind,
                node.name,
                node.parent_name,
                node.line_start,
                node.line_end,
                node.params,
                node.return_type,
                node.modifiers,
                node.is_test,
                node.identity_name,
            )
            for node in nodes
            if node.kind in {"Class", "Function", "Test"}
        ),
        key=repr,
    )
    contains = sorted(
        (edge.kind, edge.source, edge.target, edge.line)
        for edge in edges
        if edge.kind == "CONTAINS"
    )
    return definitions, contains


def _native_projection(result: NativeParseResult) -> tuple[list[tuple], list[tuple]]:
    definitions = sorted(
        (
            (
                item.kind,
                item.name,
                item.parent_name,
                item.line_start,
                item.line_end,
                item.params,
                item.return_type,
                item.modifiers,
                item.is_test,
                item.identity_name,
            )
            for item in result.definitions
        ),
        key=repr,
    )
    contains = sorted((edge.kind, edge.source, edge.target, edge.line) for edge in result.edges)
    return definitions, contains


def _ordinary_python(path: Path) -> bool:
    try:
        return not path.read_bytes().startswith(b"# Databricks notebook source")
    except OSError:
        return False


def run(source: Path, repeats: int) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    revision = _git(source, "rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(prefix="crg-native-core-") as scratch:
        tree = Path(scratch) / "repo"
        snapshot_repository(source, revision, tree)
        paths = [
            tree / relative
            for relative in collect_all_files(tree, recurse_submodules=False)
            if relative.endswith(".py") and _ordinary_python(tree / relative)
        ]
        paths.sort()
        if not paths:
            raise ValueError("No ordinary Python files found")

        python_samples = []
        expected = None
        for _ in range(repeats):
            parser = CodeParser(tree)
            started = time.perf_counter()
            rows = []
            for path in paths:
                nodes, edges = parser.parse_file(path)
                rows.append(_projection(nodes, edges))
            python_samples.append(time.perf_counter() - started)
            expected = rows

        native_samples = []
        actual = None
        for _ in range(repeats):
            started = time.perf_counter()
            parsed = parse_python_batch(paths, tree)
            native_samples.append(time.perf_counter() - started)
            actual = [_native_projection(row) for row in parsed]

        mismatches = [
            paths[index].relative_to(tree).as_posix()
            for index, (left, right) in enumerate(zip(expected or [], actual or [], strict=True))
            if left != right
        ]
        python_median = statistics.median(python_samples)
        native_median = statistics.median(native_samples)
        return {
            "source_revision": revision,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "native_core": core_version(),
            "files": len(paths),
            "bytes": sum(path.stat().st_size for path in paths),
            "repeats": repeats,
            "scope": (
                "Rust hashes and extracts Python definitions and CONTAINS edges; "
                "Python timing is the complete current parser and is not equivalent work"
            ),
            "correctness_ok": not mismatches,
            "mismatched_files": mismatches,
            "python_full_parse_seconds": python_samples,
            "python_full_parse_median_seconds": python_median,
            "rust_definition_slice_seconds": native_samples,
            "rust_definition_slice_median_seconds": native_median,
            "observed_ratio": python_median / native_median,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must name a new file")
    report = run(args.source.resolve(), args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["correctness_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
