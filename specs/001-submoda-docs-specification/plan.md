# Implementation Plan: Submodular Optimization Platform (submoda)

**Branch**: `001-submoda-docs-specification` | **Date**: 2025-10-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-submoda-docs-specification/spec.md` and `/docs/specification.md`

## Summary

Build a production-grade submodular optimization platform in Rust with Python bindings that solves large-scale subset selection problems (select k items from n candidates to maximize submodular objective). System must handle millions of candidates × millions of demands × billions of non-zero interactions, processing in minutes on commodity hardware (32-core, 128GB RAM) with provable (1-1/e) ≈ 63% approximation guarantees, thread-safe parallel evaluation, deterministic execution across runs, and comprehensive observability via Prometheus metrics and JSON Lines audit logs.

## Technical Context

**Language/Version**: Rust 1.70+ (requires std::simd for SIMD vectorization)

**Primary Dependencies**:
- **rayon** (0.8+): Parallel iterators for data parallelism, thread pool management
- **arrow** (50.0+) / **parquet** (50.0+): Data ingestion, schema validation, predicate pushdown, row-group indexing
- **fixedbitset** (0.5+): Compact bitset for SelectionView with O(1) membership testing
- **pyo3** (0.20+) + **maturin** (1.4+): Python bindings with GIL release and zero-copy Arrow FFI
- **prometheus** (0.13+) / **hyper** (1.0+): Metrics exposure at /metrics endpoint
- **serde** (1.0+) / **serde_json** (1.0+): JSON Lines audit log serialization
- **rand** (0.8+) / **rand_chacha** (0.3+): Seeded RNG for deterministic sampling (StdRng for platform independence)
- **clap** (4.4+): CLI argument parsing
- **criterion** (0.5+): Benchmarking with regression detection (test/CI only)
- **proptest** (1.4+): Property-based testing for submodularity validation (test only)

**Storage**: Parquet files (read-only batch processing, CSR/CSC sparse matrices in memory, no persistent mutable state)

**Testing**:
- **Unit**: cargo test + proptest for submodularity properties
- **Integration**: end-to-end solver runs with determinism validation (10 identical runs)
- **Benchmarks**: criterion.rs with regression detection (alert on >1% degradation)
- **Python**: pytest for PyO3 bindings, NumPy/Pandas integration

**Target Platform**: Linux/macOS/Windows servers (32-core x86_64 with AVX2, 128GB RAM, SSD storage)

**Project Type**: Rust workspace with 6 crates (core, objectives, solver, io, bindings-py, service)

**Performance Goals**:
- Large-scale: <10 min for n=10^6 candidates, m=10^6 demands, nnz=10^8, k=250 selections
- Lazy Greedy efficiency: 5-10% of evaluations vs standard greedy
- Parallel speedup: ≥50x with 64 threads for embarrassingly parallel workloads
- Marginal gain latency: <200ms p95 per evaluation
- I/O throughput: 500MB/s sequential read from Parquet with async prefetch

**Constraints**:
- Memory: <150GB for 100 billion non-zero utilities (12 bytes per entry + overhead)
- No runtime type erasure overhead (use monomorphization for f32/f64)
- No GIL blocking in Python bindings (use Python::allow_threads() for long computations)
- Zero-copy I/O via Arrow FFI for NumPy/Pandas integration (no serialization)
- Deterministic execution: fixed-order parallel reduction adds ~10% overhead when enabled

**Scale/Scope**:
- Candidates: 4.3 billion (u32), configurable to u64 for extreme scale
- Demands: millions (tested up to 10^6)
- Non-zero utilities: billions (tested up to 10^8, target 10^11)
- Parquet row groups: 128-256MB (configurable via SUBMOD_ROW_GROUP_TARGET_MB)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Specification Adherence ✓

**Status**: PASS
**Evidence**: Plan references docs/specification.md and spec.md as primary sources. All design decisions trace to specification sections. No deviations planned.

### II. Theoretical Guarantees ✓

**Status**: PASS
**Evidence**:
- Lazy Greedy: (1-1/e) for monotone submodular + cardinality (specification.md §5.1)
- Stochastic Greedy: (1-1/e - ε) with explicit ε parameter (specification.md §5.2)
- Continuous Greedy: (1-1/e) for matroid constraints (specification.md §5.3)
- Knapsack: Explicit Practical (heuristic) vs Theoretical ((1-1/e)) modes (specification.md §6.3)

### III. Thread Safety ✓

**Status**: PASS
**Evidence**:
- `SubmodularOracle::gain(&self, view: &SelectionView, e: ItemId) -> f64` is immutable
- `SubmodularOracle::commit(&mut self, e: ItemId)` is the only mutating method
- All parallel evaluation uses immutable SelectionView snapshots
- Fork pattern: `Arc<immutable_data>` + thread-local caches (specification.md §3.3)

### IV. Deterministic Execution ✓

**Status**: PASS
**Evidence**:
- Hierarchical RNG seeding: master_seed → algo_seed → iteration_seed (specification.md §9.3)
- Tie-breaking: deterministic by ItemId (lowest wins) by default (specification.md §2.2)
- Fixed-order parallel reduction: sort by ItemId after parallel evaluation (specification.md §9.4)
- Heap operations: deterministic Ord with tie-breaking (specification.md §5.1)
- Audit log hashes: only integer decisions (ItemId sequence + seed), never floats (specification.md §8.3)

### V. Numerical Stability ✓

**Status**: PASS
**Evidence**:
- Facility Location, Saturating Coverage: f32 (2x memory, wider SIMD) (specification.md §8.1)
- Log-Determinant: f64 mandatory (Cholesky fails with f32) (specification.md §4.3)
- Continuous Greedy gradient: f64 for intermediate accumulation (specification.md §5.3)
- 5-layer defense for Log-Determinant: regularization, safe Schur, log1p, refactorization, degradation (specification.md §8.2)
- Fixed-order reduction: f64 aggregation when enabled (specification.md §8.3)
- Knapsack: relative epsilon tolerance for budget comparison (specification.md §8.4)

### VI. Observability ✓

**Status**: PASS
**Evidence**:
- Prometheus metrics: 15+ metrics (counters, gauges, histograms) (specification.md §10.1)
- Audit logs: JSON Lines with 9 mandatory fields per iteration (specification.md §10.2)
- Termination reasons: semantic labels (cardinality_reached, stagnation, timeout, etc.) (specification.md §5.6)
- Threshold logging: all numerical thresholds with semantic names (specification.md §10.2)
- Fallback events: log-det degradation with action taken (specification.md §4.3)
- HTTP endpoint: /metrics on configurable port (default 9090) (specification.md §10.1)

**GATE RESULT**: ✅ ALL CHECKS PASSED - Proceed to Phase 0 research

## Project Structure

### Documentation (this feature)

```
specs/001-submoda-docs-specification/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output: Technology decisions and rationale
├── data-model.md        # Phase 1 output: Core entities, traits, relationships
├── quickstart.md        # Phase 1 output: Minimal working example
└── contracts/           # Phase 1 output: Rust API contracts (JSON schema format)
    ├── core-api.json            # SubmodularOracle, Constraint, Matroid traits
    ├── solver-api.json          # solve() function, SolverConfig
    └── python-api.json          # PyO3 bindings, NumPy integration
```

### Source Code (repository root)

```
submoda/ (Cargo workspace root)
├── Cargo.toml                   # Workspace manifest with 6 member crates
├── Cargo.lock                   # Locked dependency versions
├── .github/
│   └── workflows/
│       ├── ci.yml               # Test matrix: {Linux, macOS, Windows} × {stable, nightly}
│       └── benchmarks.yml       # Criterion benchmarks with regression detection
├── examples/
│   ├── minimal.rs               # Quick start: 50 candidates, 100 demands, k=10
│   ├── demand_mini.parquet      # 100 rows: [i:u32, w:f32]
│   └── utility_mini.parquet     # 500 rows: [i:u32, s:u32, u:f32]
├── submod-core/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs               # Re-exports
│   │   ├── traits.rs            # SubmodularOracle, Constraint, Matroid
│   │   ├── types.rs             # ItemId, Weight, SelectionView, Selection
│   │   ├── config.rs            # Strategy, Determinism, TieBreak enums
│   │   └── metrics.rs           # Prometheus metric definitions
│   └── tests/
│       ├── submodularity.rs     # Proptest: diminishing returns property
│       └── selection_view.rs    # FixedBitSet membership tests
├── submod-objectives/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs               # Re-exports
│   │   ├── facility_location.rs # FacilityLocation oracle with CSR
│   │   ├── saturating.rs        # Saturating Coverage with concave functions
│   │   ├── logdet.rs            # Log-Determinant with Cholesky + 5-layer defense
│   │   └── diversity.rs         # Diversity-Aware Facility Location
│   └── tests/
│       ├── facility_location_correctness.rs
│       └── logdet_numerical_stability.rs
├── submod-solver/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs               # Public solve() entry point
│   │   ├── lazy_greedy.rs       # Epoch-based heap, stale bound elimination
│   │   ├── stochastic.rs        # Reservoir sampling, O(n log(1/ε)) complexity
│   │   ├── continuous.rs        # View-based gradient estimation, rounding
│   │   └── sieve_streaming.rs   # Single-pass streaming (Phase 4)
│   ├── benches/
│   │   └── solvers.criterion.rs # Criterion benchmarks: 1M×1M problem
│   └── tests/
│       ├── determinism.rs       # 10 identical runs validation
│       └── approximation.rs     # Compare to brute-force optimal on small datasets
├── submod-io/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs               # Re-exports
│   │   ├── parquet_loader.rs    # Predicate pushdown, row-group pruning
│   │   ├── csr.rs               # CSRMatrix construction with zero filtering
│   │   ├── validation.rs        # Fail-fast input validation
│   │   └── sharding.rs          # Hash-based sharding (128 shards), async prefetch
│   └── tests/
│       ├── parquet_schema.rs    # Schema conformance tests
│       └── validation.rs        # NaN/Inf/bounds rejection tests
├── submod-bindings-py/
│   ├── Cargo.toml               # [lib] crate-type = ["cdylib"]
│   ├── pyproject.toml           # Maturin build config
│   ├── src/
│   │   ├── lib.rs               # #[pymodule] submoda
│   │   ├── oracle.rs            # #[pyclass] FacilityLocation, LogDeterminant
│   │   ├── constraint.rs        # #[pyclass] CardinalityConstraint, Knapsack
│   │   ├── solver.rs            # #[pyfunction] solve() with GIL release
│   │   └── arrow_ffi.rs         # Zero-copy NumPy/Pandas via Arrow C Data Interface
│   ├── python/
│   │   └── submoda/
│   │       ├── __init__.py      # Type stubs for IDE
│   │       └── py.typed         # PEP 561 marker
│   └── tests/
│       ├── test_facility_location.py
│       └── test_determinism.py
├── submod-service/              # Phase 5: gRPC/REST endpoints
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs              # Axum HTTP server + Prometheus /metrics
│   │   ├── grpc.rs              # Tonic gRPC service (future)
│   │   └── rest.rs              # REST API (future)
│   └── tests/
│       └── metrics.rs           # HTTP /metrics endpoint tests
└── docs/
    ├── specification.md         # Complete technical specification (source of truth)
    └── README.md                # Project overview, quick start
```

**Structure Decision**: Rust workspace with 6 crates provides clear separation of concerns (specification.md §2.1), enables parallel development, supports selective dependencies (Python bindings don't need service layer), and facilitates incremental testing. Each crate has focused responsibilities: core (traits/types), objectives (oracles), solver (algorithms), io (Parquet), bindings-py (PyO3), service (HTTP/gRPC).

## Complexity Tracking

*No constitution violations detected. Standard project structure (6 crates) is within complexity bounds.*

**Justification**: Workspace organization follows specification.md §2.1 exactly. Clear separation enables:
- Independent versioning (core traits stable, solvers iterate)
- Parallel team development (oracles, algorithms, I/O, bindings)
- Selective compilation (Python-only users don't build service layer)
- Incremental testing (test core traits before algorithm implementations)

No simpler alternative exists that satisfies the scale, performance, and modularity requirements.
