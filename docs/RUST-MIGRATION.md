# Rust core migration

The Rust migration is incremental. Python remains responsible for MCP, CLI,
configuration, external providers and compatibility orchestration. Rust takes
over bounded CPU-heavy operations only after exact parity and benchmark gates
pass. SQLite remains the storage format.

## Current milestone: complete Python graph rows behind a feature flag

`native/` builds an ABI3 extension for CPython 3.10 and newer. Its first batch
API reads Python byte snapshots, computes SHA-256 hashes, parses them in
parallel with Tree-sitter, and returns deterministic nodes and edges. The native
path covers file/class/function/test nodes and `CONTAINS`, `IMPORTS_FROM`,
`INHERITS`, `CALLS`, `REFERENCES` and `TESTED_BY` edges, including resolver
metadata, decorators and docstrings.
The Python boundary is `code_review_graph.native`; importing the main package
does not require Rust. Set `CRG_NATIVE_CORE=1` to exercise it in full and
incremental builds. Python remains the default.

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

The parity test covers every ordinary Python fixture and compares all graph
rows exactly. Databricks exports are
excluded because they use a separate polyglot Python/R/SQL preprocessing path.
The benchmark reports equivalent complete graph-row workloads. Receiver and
resolver evidence still uses a Python compatibility scan, which preserves exact
behavior while that logic is migrated.

On committed source revision `cd1e1bb`, the complete-row gate covered 277 Python
files (6.1 MB) with zero differences. Across five repetitions, the opt-in path
had a 6.378-second median and the Python parser had a 5.603-second median, so the
native path was 13.8% slower. All seven incremental edit scenarios had zero
differing rows, but the measured benchmark work improved only 1.4%, below the
30% activation gate. Raw evidence is tracked in
`evaluate/results/native_core_cd1e1bb_full_rows.json` and the two
`incremental_fidelity_*_cd1e1bb.csv` files.

## Activation gates

1. Move the remaining compatibility evidence scan into Rust so each file is
   parsed once.
2. Batch whole file sets across the PyO3 boundary; do not add per-node calls.
3. Demonstrate at least 2x faster full builds or 30% faster incremental updates
   without increasing peak memory.
4. Publish and test ABI3 wheels on Linux, macOS and Windows before enabling the
   native path by default.

Later milestones move resolver candidate generation, reachability, flows and
other profiled CPU hotspots. SQLite transaction orchestration moves only if
profiling shows Python row conversion or call frequency is material; SQL already
runs in native code. MCP schemas, CLI behavior and the database format remain
stable throughout the migration.

## Graph-algorithm profile

A serial full build plus post-processing profile on revision `cd1e1bb` spent
11.62 seconds in parsing and 1.38 seconds in the complete post-processing
pipeline. Community detection used 0.54 seconds, bare-endpoint resolution used
0.50 seconds, and no flow or reachability routine appeared among the leading
CPU costs. The measured hotspots are Python syntax/evidence walks: extraction
used 6.43 seconds, dead-guard analysis 2.42 seconds, receiver evidence 2.14
seconds and typed-call evidence 1.40 seconds. Moving graph algorithms now would
optimize less than the compatibility scans that keep the native path slower, so
the next Rust work remains those evidence walks. The raw profile is reproducible
with `cProfile` around `full_build` followed by `run_post_processing`.

ABI3 wheel jobs build and import-test CPython 3.10 and 3.13 artifacts on Linux,
macOS and Windows for x86-64 and ARM64. Release publishing uploads those native
wheels alongside the existing pure-Python wheel and source distribution, so the
fallback remains installable on unsupported platforms.
