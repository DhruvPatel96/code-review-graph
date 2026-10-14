"""Run native extension mutation probes in disposable subprocesses."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def probe(feature: str) -> None:
    import ladybug

    with tempfile.TemporaryDirectory() as root:
        db = ladybug.Database(
            str(Path(root) / "graph"), buffer_pool_size=128 * 1024**2, max_num_threads=1
        )
        conn = ladybug.Connection(db)

        def query(sql: str, params: dict | None = None) -> list:
            print(json.dumps({"running": sql, "parameters": params}), flush=True)
            result: Any = conn.execute(sql, params)
            rows = []
            while result.has_next():
                rows.append(result.get_next())
            result.close()
            print(json.dumps({"result": rows}), flush=True)
            return rows

        try:
            query("CREATE NODE TABLE N(q STRING PRIMARY KEY, text STRING, v FLOAT[3])")
            query("CREATE (:N {q:'a',text:'alpha',v:[1,0,0]}),(:N {q:'b',text:'beta',v:[0,1,0]})")
            if feature in ("fts", "both"):
                query("INSTALL fts")
                query("LOAD fts")
                query("CALL CREATE_FTS_INDEX('N','text_idx',['text'],stemmer := 'none')")
            if feature in ("vector", "both"):
                query("INSTALL vector")
                query("LOAD vector")
                query("CALL CREATE_VECTOR_INDEX('N','vec_idx','v',metric := 'cosine')")
            query("CREATE (:N {q:'c',text:'gamma',v:[0,0,1]})")
            if feature in ("fts", "both"):
                assert query("CALL QUERY_FTS_INDEX('N','text_idx','gamma') RETURN node.q") == [
                    ["c"]
                ]
                assert (
                    query("CALL QUERY_FTS_INDEX('N','text_idx',$q) RETURN node.q", {"q": "delta"})
                    == []
                )
                query("MATCH (n:N {q:'c'}) SET n.text='delta'")
                assert query(
                    "CALL QUERY_FTS_INDEX('N','text_idx',$q) RETURN node.q", {"q": "delta"}
                ) == [["c"]]
            if feature in ("vector", "both"):
                assert query("CALL QUERY_VECTOR_INDEX('N','vec_idx',[0,0,1],1) RETURN node.q") == [
                    ["c"]
                ]
            query("MATCH (n:N {q:'c'}) DETACH DELETE n")
            if feature in ("fts", "both"):
                assert (
                    query("CALL QUERY_FTS_INDEX('N','text_idx',$q) RETURN node.q", {"q": "delta"})
                    == []
                )
            if feature in ("vector", "both"):
                assert all(
                    row[0] != "c"
                    for row in query(
                        "CALL QUERY_VECTOR_INDEX('N','vec_idx',[0,0,1],2) RETURN node.q"
                    )
                )
        finally:
            conn.close()
            db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=("fts", "vector", "both"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.probe:
        probe(args.probe)
        return 0
    if args.output is None or args.output.exists():
        parser.error("--output must name a new file")
    reports = []
    for feature in ("fts", "vector", "both"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "code_review_graph.eval.benchmarks.ladybug_capabilities",
                "--probe",
                feature,
            ],
            text=True,
            capture_output=True,
            timeout=120,
        )
        reports.append(
            {
                "feature": feature,
                "ladybug": importlib.metadata.version("ladybug"),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    return int(any(r["returncode"] for r in reports))


if __name__ == "__main__":
    raise SystemExit(main())
