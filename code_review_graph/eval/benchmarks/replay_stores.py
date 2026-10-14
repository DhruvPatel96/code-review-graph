"""Experimental core-graph replay stores, NOT production GraphStorage backends.

SQLite uses the existing schema/indexes. Ladybug uses native relationships with
placeholder vertices for unresolved endpoints, and JSON payloads for lossless
core row round-tripping. Neither adapter serves MCP, resolvers or derived tables.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ...graph import GraphStore

NODE_COLUMNS = (
    "kind",
    "name",
    "qualified_name",
    "file_path",
    "line_start",
    "line_end",
    "language",
    "parent_name",
    "params",
    "return_type",
    "modifiers",
    "is_test",
    "file_hash",
    "extra",
    "symbol",
    "docstring",
    "name_tokens",
    "signature",
)
EDGE_COLUMNS = (
    "kind",
    "source_qualified",
    "target_qualified",
    "file_path",
    "line",
    "extra",
    "confidence",
    "confidence_tier",
    "target_resolution",
)
KINDS = ("CALLS", "IMPORTS_FROM", "INHERITS", "IMPLEMENTS", "DEPENDS_ON")
Image = dict[str, list[dict[str, Any]]]


def canonical(row: Any) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def normalize(image: Image) -> Image:
    return {table: sorted(image[table], key=canonical) for table in ("nodes", "edges")}


def read_sqlite(store: GraphStore) -> Image:
    return normalize(
        {
            table: [
                dict(row) for row in store._conn.execute(f"SELECT {','.join(columns)} FROM {table}")
            ]
            for table, columns in (("nodes", NODE_COLUMNS), ("edges", EDGE_COLUMNS))
        }
    )


def patch_for(image: Image, files: list[str]) -> Image:
    paths = set(files)
    return {
        table: [row for row in image[table] if row["file_path"] in paths]
        for table in ("nodes", "edges")
    }


def validate_patch(image: Image, files: list[str]) -> None:
    names = [n["qualified_name"] for n in image["nodes"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate node definitions")
    for table, columns in (("nodes", NODE_COLUMNS), ("edges", EDGE_COLUMNS)):
        for row in image[table]:
            if set(row) != set(columns) or row["file_path"] not in files:
                raise ValueError("Patch must contain complete rows owned by replaced files")


def deleted_image(image: Image, files: list[str]) -> Image:
    paths = set(files)
    names = {n["qualified_name"] for n in image["nodes"] if n["file_path"] in paths}
    return normalize(
        {
            "nodes": [n for n in image["nodes"] if n["file_path"] not in paths],
            "edges": [
                e
                for e in image["edges"]
                if e["file_path"] not in paths
                and e["source_qualified"] not in names
                and e["target_qualified"] not in names
            ],
        }
    )


def reference_traversal(image: Image, seed: str, depth: int) -> list[str]:
    names = {n["qualified_name"] for n in image["nodes"]}
    incoming: dict[str, set[str]] = {}
    for e in image["edges"]:
        if e["kind"] in KINDS and e["source_qualified"] in names and e["target_qualified"] in names:
            incoming.setdefault(e["target_qualified"], set()).add(e["source_qualified"])
    seen = {seed}
    frontier = {seed} if seed in names else set()
    for _ in range(depth):
        frontier = {p for node in frontier for p in incoming.get(node, set())} - seen
        seen.update(frontier)
    return sorted(seen - {seed})


class SqliteReplayStore:
    def __init__(self, path: Path, *, buffer_mb: int = 128):
        self.store = GraphStore(path)
        self.conn = self.store._conn
        self.conn.execute(f"PRAGMA cache_size=-{int(buffer_mb) * 1024}")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def _write(self, image: Image) -> None:
        for table, columns in (("nodes", NODE_COLUMNS), ("edges", EDGE_COLUMNS)):
            sql = (
                f"INSERT INTO {table} ({','.join(columns)},updated_at) VALUES "
                f"({','.join('?' for _ in columns)},0)"
            )
            self.conn.executemany(sql, [tuple(row[k] for k in columns) for row in image[table]])

    def replace_files(self, files: list[str], image: Image) -> None:
        validate_patch(image, files)
        with self.transaction():
            for file in files:
                self.conn.execute("DELETE FROM edges WHERE file_path=?", (file,))
                self.conn.execute("DELETE FROM nodes WHERE file_path=?", (file,))
            self._write(image)

    def remove_files(self, files: list[str]) -> None:
        self.store.remove_files_permanently(files, stored_paths=True)

    def snapshot(self) -> Image:
        return read_sqlite(self.store)

    def neighbors(self, qn: str, incoming: bool) -> list[dict[str, Any]]:
        column = "target_qualified" if incoming else "source_qualified"
        return sorted(
            [
                dict(row)
                for row in self.conn.execute(
                    f"SELECT {','.join(EDGE_COLUMNS)} FROM edges WHERE {column}=?", (qn,)
                )
            ],
            key=canonical,
        )

    def traverse(self, qn: str, depth: int) -> list[str]:
        if depth not in (1, 2, 3):
            raise ValueError("depth must be 1, 2 or 3")
        rows = self.conn.execute(
            "WITH RECURSIVE walk(q,depth) AS ("
            "SELECT qualified_name,0 FROM nodes WHERE qualified_name=? UNION "
            "SELECT e.source_qualified,w.depth+1 FROM walk w "
            "JOIN edges e ON e.target_qualified=w.q "
            "JOIN nodes n ON n.qualified_name=e.source_qualified "
            "WHERE w.depth < ? AND e.kind IN (?,?,?,?,?)) "
            "SELECT DISTINCT q FROM walk WHERE q<>? ORDER BY q",
            (qn, depth, *KINDS, qn),
        )
        return [r[0] for r in rows]

    def checkpoint(self) -> None:
        if self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
            raise RuntimeError("SQLite checkpoint busy")

    def close(self) -> None:
        self.store.close()


class LadybugReplayStore:
    """Native property graph with explicit unresolved endpoint vertices.

    File replacement demotes removed definitions to placeholders so foreign
    edges survive. Permanent deletion removes their incident edges. Unused
    placeholders are pruned, and placeholders never count as graph definitions.
    """

    def __init__(self, path: Path, *, buffer_mb: int = 128):
        import ladybug

        self.db = ladybug.Database(
            str(path), buffer_pool_size=buffer_mb * 1024 * 1024, max_num_threads=1
        )
        self.conn = ladybug.Connection(self.db)
        self.query(
            "CREATE NODE TABLE IF NOT EXISTS CodeNode("
            "qn STRING PRIMARY KEY, present BOOL, file_path STRING, payload STRING)"
        )
        self.query(
            "CREATE REL TABLE IF NOT EXISTS CodeEdge("
            "FROM CodeNode TO CodeNode, kind STRING, file_path STRING, payload STRING)"
        )

    def query(self, sql: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        result: Any = self.conn.execute(sql, params)
        try:
            rows = []
            while result.has_next():
                rows.append(result.get_next())
            return rows
        finally:
            result.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.query("BEGIN TRANSACTION")
        try:
            yield
            self.query("COMMIT")
        except BaseException:
            try:
                self.query("ROLLBACK")
            except RuntimeError as rollback_error:
                if "No active transaction" not in str(rollback_error):
                    raise
            raise

    def _batches(self, sql: str, rows: list[Any]) -> None:
        for start in range(0, len(rows), 1000):
            self.query(sql, {"rows": rows[start : start + 1000]})

    def _write(self, image: Image) -> None:
        names = {n["qualified_name"] for n in image["nodes"]}
        names.update(e[k] for e in image["edges"] for k in ("source_qualified", "target_qualified"))
        self._batches(
            "UNWIND $rows AS q MERGE (n:CodeNode {qn:q}) "
            "ON CREATE SET n.present=false,n.file_path='',n.payload=''",
            sorted(names),
        )
        self._batches(
            "UNWIND $rows AS r MATCH (n:CodeNode {qn:r.qn}) "
            "SET n.present=true,n.file_path=r.file_path,n.payload=r.payload",
            [
                {"qn": n["qualified_name"], "file_path": n["file_path"], "payload": canonical(n)}
                for n in image["nodes"]
            ],
        )
        self._batches(
            "UNWIND $rows AS r MATCH (a:CodeNode {qn:r.source}),"
            "(b:CodeNode {qn:r.target}) CREATE (a)-[:CodeEdge "
            "{kind:r.kind,file_path:r.file_path,payload:r.payload}]->(b)",
            [
                {
                    "source": e["source_qualified"],
                    "target": e["target_qualified"],
                    "kind": e["kind"],
                    "file_path": e["file_path"],
                    "payload": canonical(e),
                }
                for e in image["edges"]
            ],
        )

    def _prune(self) -> None:
        self.query(
            "MATCH (n:CodeNode) WHERE NOT n.present "
            "AND NOT EXISTS {MATCH (n)-[:CodeEdge]-()} DELETE n"
        )

    def replace_files(self, files: list[str], image: Image) -> None:
        validate_patch(image, files)
        with self.transaction():
            self.query(
                "MATCH ()-[e:CodeEdge]->() WHERE e.file_path IN $files DELETE e", {"files": files}
            )
            self.query(
                "MATCH (n:CodeNode) WHERE n.file_path IN $files "
                "SET n.present=false,n.file_path='',n.payload=''",
                {"files": files},
            )
            self._write(image)
            self._prune()

    def remove_files(self, files: list[str]) -> None:
        with self.transaction():
            self.query(
                "MATCH ()-[e:CodeEdge]->() WHERE e.file_path IN $files DELETE e", {"files": files}
            )
            self.query(
                "MATCH (n:CodeNode) WHERE n.present AND n.file_path IN $files DETACH DELETE n",
                {"files": files},
            )
            self._prune()

    def snapshot(self) -> Image:
        return normalize(
            {
                "nodes": [
                    json.loads(r[0])
                    for r in self.query("MATCH (n:CodeNode) WHERE n.present RETURN n.payload")
                ],
                "edges": [
                    json.loads(r[0])
                    for r in self.query("MATCH ()-[e:CodeEdge]->() RETURN e.payload")
                ],
            }
        )

    def neighbors(self, qn: str, incoming: bool) -> list[dict[str, Any]]:
        pattern = "()-[e:CodeEdge]->(n:CodeNode)" if incoming else "(n:CodeNode)-[e:CodeEdge]->()"
        return sorted(
            [
                json.loads(r[0])
                for r in self.query(f"MATCH {pattern} WHERE n.qn=$qn RETURN e.payload", {"qn": qn})
            ],
            key=canonical,
        )

    def traverse(self, qn: str, depth: int) -> list[str]:
        if depth not in (1, 2, 3):
            raise ValueError("depth must be 1, 2 or 3")
        # 0.21.2 asserts on a list parameter inside recursive ALL. These fixed
        # schema constants avoid that compiler path; the symbol stays bound.
        kinds = ",".join(f"'{kind}'" for kind in KINDS)
        rows = self.query(
            f"MATCH p=(s:CodeNode {{qn:$qn}})<-[:CodeEdge*1..{depth}]-(n:CodeNode) "
            f"WHERE ALL(v IN nodes(p) WHERE v.present) "
            f"AND ALL(r IN rels(p) WHERE r.kind IN [{kinds}]) "
            "AND n.qn<>$qn RETURN DISTINCT n.qn ORDER BY n.qn",
            {"qn": qn},
        )
        return [r[0] for r in rows]

    def checkpoint(self) -> None:
        self.query("CHECKPOINT")

    def close(self) -> None:
        self.conn.close()
        self.db.close()
