# Rust core migration

The Rust migration is incremental. Python remains responsible for MCP, CLI,
configuration, external providers and compatibility orchestration. Rust takes
over bounded CPU-heavy operations only after exact parity and benchmark gates
pass. SQLite remains the storage format.

## Current milestone: Python definition and containment extraction

`native/` builds an ABI3 extension for CPython 3.10 and newer. Its first batch
API reads Python byte snapshots, computes SHA-256 hashes, parses them in
parallel with Tree-sitter, and returns deterministic class/function definitions
plus their `CONTAINS` edges. Definition rows include parameters, return types,
decorators, test classification and repository-aware identity fields.
The Python boundary is `code_review_graph.native`; importing the main package
does not require Rust. Production builds and updates do not call it yet.

```bash
. "$HOME/.cargo/env"
uv sync --extra dev
cargo fmt --manifest-path native/Cargo.toml -- --check
cargo clippy --manifest-path native/Cargo.toml --all-targets -- -D warnings
uv run maturin develop --release --manifest-path native/Cargo.toml
uv run pytest tests/test_native_core.py -q
uv run python -m code_review_graph.eval.benchmarks.native_core \
  --repeats 5 --output evaluate/results/native_core.json
```

The parity test covers every ordinary Python fixture and compares the expanded
definition rows and `CONTAINS` edges exactly. Databricks exports are
excluded because they use a separate polyglot Python/R/SQL preprocessing path.
The benchmark reports the native definition slice beside the complete Python
parser. Those timings represent different workloads and therefore measure an
optimization ceiling, not an end-to-end speedup.

On committed source revision `987f57a`, the expanded gate covered 277 Python
files (6.1 MB) with zero definition or `CONTAINS` differences. Across five
repetitions, the Rust slice had a 0.288-second median; the complete Python parser
had a 4.797-second median. The observed 16.7x ratio is an upper bound because
the Python side also extracts imports, calls and other graph metadata. Raw
evidence is tracked in `evaluate/results/native_core_987f57a_rows.json`.

## Activation gates

1. Extend native output to every Python node and edge field, including imports,
   calls, decorators, tests, dead-code guards and repository-aware resolution.
2. Pass fixture, real-repository and seven-edit incremental parity with zero
   differing rows.
3. Batch whole file sets across the PyO3 boundary; do not add per-node calls.
4. Demonstrate at least 2x faster full builds or 30% faster incremental updates
   without increasing peak memory.
5. Publish and test ABI3 wheels on Linux, macOS and Windows before enabling the
   native path by default.

Later milestones move resolver candidate generation, reachability, flows and
other profiled CPU hotspots. SQLite transaction orchestration moves only if
profiling shows Python row conversion or call frequency is material; SQL already
runs in native code. MCP schemas, CLI behavior and the database format remain
stable throughout the migration.
