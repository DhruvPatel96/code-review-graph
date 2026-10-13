"""Explicit boundary for the optional Rust core.

Importing code-review-graph never requires the extension. Callers must request
it explicitly; production parsing remains Python until parity gates pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


class NativeCoreUnavailableError(RuntimeError):
    """Raised when an explicitly requested native operation cannot run."""


@dataclass(frozen=True)
class NativeDefinition:
    kind: str
    name: str
    parent_name: str | None
    params: str | None
    return_type: str | None
    modifiers: str | None
    is_test: bool
    identity_name: str | None
    decorators: tuple[str, ...]
    extra: dict[str, Any]
    line_start: int
    line_end: int


@dataclass(frozen=True)
class NativeEdge:
    kind: str
    source: str
    target: str
    line: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeParseResult:
    path: str
    sha256: str
    syntax_errors: int
    definitions: tuple[NativeDefinition, ...]
    edges: tuple[NativeEdge, ...]
    file_node: NativeDefinition


def _extension() -> Any:
    try:
        from . import _native
    except ImportError as exc:
        raise NativeCoreUnavailableError(
            "Rust core is not installed; run `maturin develop --release`"
        ) from exc
    return _native


def core_version() -> str:
    return str(_extension().core_version())


def _reference_edges(
    parser: Any,
    root: Any,
    source: bytes,
    path: str,
    import_map: dict[str, str],
    defined_names: set[str],
) -> list[Any]:
    """Run the compatibility reference rules over the native parse snapshot."""
    edges: list[Any] = []

    def walk(node: Any, enclosing_class: str | None, enclosing_function: str | None) -> None:
        next_class = enclosing_class
        next_function = enclosing_function
        if node.type == "class_definition":
            name = node.child_by_field_name("name")
            next_class = name.text.decode("utf-8", errors="replace") if name else enclosing_class
            next_function = None
        elif node.type == "function_definition":
            name = node.child_by_field_name("name")
            if name is not None:
                next_function = name.text.decode("utf-8", errors="replace")
        parser._extract_value_references(
            node,
            node.type,
            source,
            "python",
            path,
            edges,
            next_class,
            next_function,
            import_map,
            defined_names,
        )
        for child in node.children:
            walk(child, next_class, next_function)

    walk(root, None, None)
    return edges


def _docstrings_by_line(parser: Any, root: Any) -> dict[int, str]:
    summaries = {}

    def walk(node: Any) -> None:
        if node.type in {"class_definition", "function_definition"}:
            summary = parser._get_docstring_summary(node, "python")
            if summary:
                summaries[node.start_point[0] + 1] = summary
        for child in node.children:
            walk(child)

    walk(root)
    return summaries


def parse_python_batch(
    paths: list[Path],
    repo_root: Path | None = None,
) -> list[NativeParseResult]:
    """Read and parse Python files in one native batch call."""
    payload = [(path.as_posix(), path.read_bytes()) for path in paths]
    return parse_python_snapshots(payload, repo_root)


def parse_python_snapshots(
    payload: list[tuple[str, bytes]],
    repo_root: Path | None = None,
) -> list[NativeParseResult]:
    """Parse pre-read Python snapshots without reopening their files."""
    rows = json.loads(_extension().parse_python_batch(payload))
    from .parser import (
        _PY_MODULE_BINDING_KEY,
        CodeParser,
        EdgeInfo,
        NodeInfo,
        _is_test_function,
        _python_unreachable_call_positions,
        is_test_file,
    )

    results = []
    parser = CodeParser(repo_root)
    source_by_path = dict(payload)
    for row in rows:
        definitions = []
        defined_names = set()
        file_is_test = is_test_file(row["path"], repo_root)
        for definition in row["definitions"]:
            decorators = tuple(definition.pop("decorators"))
            definition.pop("bases")
            module_scope = definition.pop("module_scope")
            raw_kind = definition.pop("kind")
            if module_scope:
                defined_names.add(definition["name"])
            function_is_test = raw_kind == "Function" and _is_test_function(
                definition["name"],
                row["path"],
                decorators,
                repo_root=repo_root,
            )
            definitions.append(
                NativeDefinition(
                    **definition,
                    kind="Test" if function_is_test else raw_kind,
                    modifiers=",".join(decorators) or None,
                    is_test=function_is_test or file_is_test,
                    identity_name=definition["name"] if raw_kind == "Function" else None,
                    decorators=decorators,
                    extra={},
                )
            )
        edges = [NativeEdge(**edge) for edge in row["edges"]]
        import_map: dict[str, str] = {}
        for imported in row["imports"]:
            modules = [imported["module"]]
            if imported["is_from"]:
                package_dir = parser._python_package_dir(imported["module"], row["path"])
                if package_dir is not None:
                    namespace = not parser._file_exists_exact(package_dir / "__init__.py")
                    modules = [] if namespace else modules
                    prefix = (
                        imported["module"]
                        if imported["module"].endswith(".")
                        else f"{imported['module']}."
                    )
                    for name in imported["names"]:
                        if "." in name:
                            continue
                        if parser._python_submodule_file(
                            package_dir, name
                        ) is not None or parser._dir_exists_exact(package_dir / name):
                            modules.append(f"{prefix}{name}")
                    if not modules:
                        modules = [imported["module"]]
            for module in modules:
                target, _ = parser._resolve_import_target(module, row["path"], "python")
                edges.append(
                    NativeEdge(
                        kind="IMPORTS_FROM",
                        source=row["path"],
                        target=target or module,
                        line=imported["line"],
                    )
                )
            if imported["is_from"]:
                module = imported["module"]
                package_dir = parser._python_package_dir(module, row["path"])
                submodules = set()
                if package_dir is not None:
                    submodules = {
                        name
                        for name in imported["names"]
                        if parser._python_submodule_file(package_dir, name) is not None
                        or parser._dir_exists_exact(package_dir / name)
                    }
                if module.startswith("."):
                    resolved, _ = parser._resolve_import_target(module, row["path"], "python")
                    module = resolved or module
                for local_name, imported_name in imported["bindings"].items():
                    if imported_name in submodules and package_dir is not None:
                        target = parser._python_submodule_file(package_dir, imported_name)
                        if target is not None:
                            target_text = str(target).replace("\\", "/")
                            import_map[local_name] = target_text
                            import_map[f"{_PY_MODULE_BINDING_KEY}{local_name}"] = target_text
                            continue
                    import_map[local_name] = module

        python_parser = parser._get_parser("python")
        if python_parser is None:
            raise RuntimeError("Python grammar became unavailable during native parsing")
        tree = python_parser.parse(source_by_path[row["path"]])
        exact_import_map, exact_defined_names = parser._collect_file_scope(
            tree.root_node,
            "python",
            source_by_path[row["path"]],
            row["path"],
        )
        parser._expand_python_star_imports(
            tree.root_node,
            row["path"],
            exact_import_map,
        )
        typed_targets = parser._collect_typed_call_targets(
            tree.root_node,
            "python",
            row["path"],
            exact_import_map,
            exact_defined_names,
        )
        receiver_evidence = parser._collect_python_receiver_evidence(
            tree.root_node,
            row["path"],
        )
        docstrings = _docstrings_by_line(parser, tree.root_node)
        definitions = [
            replace(
                definition,
                extra={
                    **(
                        {"decorators": list(definition.decorators)} if definition.decorators else {}
                    ),
                    **(
                        {"docstring": docstrings[definition.line_start]}
                        if definition.line_start in docstrings
                        else {}
                    ),
                },
            )
            for definition in definitions
        ]
        call_start = len(edges)
        unreachable = _python_unreachable_call_positions(source_by_path[row["path"]])
        for call in row["calls"]:
            if (call["line"], call["column"]) in unreachable:
                continue
            receiver = call["receiver"]
            extra = {}
            if receiver:
                extra["receiver"] = receiver
            if call["receiver_form"]:
                extra["receiver_form"] = call["receiver_form"]
            enclosing_class = call["enclosing_class"]
            enclosing_function = call["enclosing_function"]
            if enclosing_function:
                source = (
                    f"{row['path']}::{enclosing_class}.{enclosing_function}"
                    if enclosing_class
                    else f"{row['path']}::{enclosing_function}"
                )
            else:
                source = row["path"]
            if receiver in {"self", "cls"} and enclosing_class:
                target = f"{row['path']}::{enclosing_class}.{call['name']}"
            elif receiver or call["receiver_form"] == "expression":
                target = call["name"]
            else:
                target = parser._resolve_call_target(
                    call["name"],
                    row["path"],
                    "python",
                    exact_import_map,
                    exact_defined_names,
                )
            edges.append(
                NativeEdge(
                    kind="CALLS",
                    source=source,
                    target=target,
                    line=call["line"],
                    extra=extra,
                )
            )
        call_edges = [
            EdgeInfo(
                kind=edge.kind,
                source=edge.source,
                target=edge.target,
                file_path=row["path"],
                line=edge.line,
                extra=edge.extra,
            )
            for edge in edges[call_start:]
        ]
        call_edges = parser._apply_typed_call_targets(
            call_edges,
            typed_targets,
            "python",
        )
        call_edges = parser._apply_python_receiver_evidence(
            call_edges,
            receiver_evidence,
        )
        parity_nodes = [
            NodeInfo(
                kind=definition.kind,
                name=definition.name,
                file_path=row["path"],
                line_start=definition.line_start,
                line_end=definition.line_end,
                language="python",
                parent_name=definition.parent_name,
                params=definition.params,
                return_type=definition.return_type,
                modifiers=definition.modifiers,
                is_test=definition.is_test,
                identity_name=definition.identity_name,
            )
            for definition in definitions
        ]
        call_edges = parser._resolve_call_targets(
            parity_nodes,
            call_edges,
            row["path"],
        )
        reference_edges = _reference_edges(
            parser,
            tree.root_node,
            source_by_path[row["path"]],
            row["path"],
            exact_import_map,
            exact_defined_names,
        )
        reference_edges = parser._resolve_call_targets(
            parity_nodes,
            reference_edges,
            row["path"],
        )
        test_qnames = set()
        for definition in definitions:
            if definition.kind != "Test":
                continue
            identity = definition.identity_name or definition.name
            qualified = (
                f"{row['path']}::{definition.parent_name}.{identity}"
                if definition.parent_name
                else f"{row['path']}::{identity}"
            )
            test_qnames.add(qualified)
        tested_by_edges = [
            EdgeInfo(
                kind="TESTED_BY",
                source=edge.target,
                target=edge.source,
                file_path=row["path"],
                line=edge.line,
                extra=dict(edge.extra),
            )
            for edge in call_edges
            if edge.source in test_qnames
        ]
        call_edges.extend(reference_edges)
        call_edges.extend(tested_by_edges)
        edges[call_start:] = [
            NativeEdge(
                kind=edge.kind,
                source=edge.source,
                target=edge.target,
                line=edge.line,
                extra=edge.extra,
            )
            for edge in call_edges
        ]
        results.append(
            NativeParseResult(
                path=row["path"],
                sha256=row["sha256"],
                syntax_errors=row["syntax_errors"],
                definitions=tuple(definitions),
                edges=tuple(edges),
                file_node=NativeDefinition(
                    kind="File",
                    name=row["path"],
                    parent_name=None,
                    params=None,
                    return_type=None,
                    modifiers=None,
                    is_test=file_is_test,
                    identity_name=None,
                    decorators=(),
                    extra={},
                    line_start=1,
                    line_end=source_by_path[row["path"]].count(b"\n") + 1,
                ),
            )
        )
    return results


def parser_rows(result: NativeParseResult) -> tuple[list[Any], list[Any]]:
    """Convert a parity-checked native result to the parser's public row types."""
    from .parser import EdgeInfo, NodeInfo

    nodes = [
        NodeInfo(
            kind=item.kind,
            name=item.name,
            file_path=result.path,
            line_start=item.line_start,
            line_end=item.line_end,
            language="python",
            parent_name=item.parent_name,
            params=item.params,
            return_type=item.return_type,
            modifiers=item.modifiers,
            is_test=item.is_test,
            extra=item.extra,
            identity_name=item.identity_name,
        )
        for item in (result.file_node, *result.definitions)
    ]
    edges = [
        EdgeInfo(
            kind=edge.kind,
            source=edge.source,
            target=edge.target,
            file_path=result.path,
            line=edge.line,
            extra=edge.extra,
        )
        for edge in result.edges
    ]
    return nodes, edges
