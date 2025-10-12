# Technology Stack: Submoda

## Architecture Overview

**Type**: Multi-crate Rust workspace for production-grade submodular optimization

**Design Pattern**: Layered architecture with clear separation of concerns
- Core traits and types (submod-core)
- Objective function implementations (submod-objectives)
- Algorithm implementations (submod-solver)
- Data I/O layer (submod-io)
- Language bindings (submod-bindings-py)
- Service layer (submod-service)

## Core Technology

### Primary Language: Rust
- **Version**: Stable channel (latest)
- **Edition**: 2021
- **Rationale**:
  - Memory safety without garbage collection
  - Zero-cost abstractions for performance
  - Fearless concurrency via ownership system
  - Strong type system prevents numerical errors
  - Excellent SIMD support via std::simd

### Workspace Structure

**6 Crates:**

1. **submod-core**: Foundation
   - Core trait definitions (SubmodularOracle, Constraint, Matroid)
   - Type system (ItemId as u32, Weight as f32/f64, SelectionView)
   - Determinism infrastructure (RNG hierarchy, fixed-order reduction)
   - Metrics and observability framework

2. **submod-objectives**: Oracle Implementations
   - Facility Location with CSR sparse matrix
   - Saturating Coverage with concave functions
   - Log-Determinant with 5-layer numerical stability
   - Extensibility framework for custom objectives

3. **submod-solver**: Algorithms
   - Lazy Greedy with epoch-based heap
   - Stochastic Greedy with reservoir sampling
   - Continuous Greedy with view-based gradient estimation
   - Sieve-Streaming for online scenarios

4. **submod-io**: Data Layer
   - Parquet/Arrow reader with predicate pushdown
   - CSR/CSC sparse matrix construction
   - Row-group optimization and sharding
   - Validation and zero filtering

5. **submod-bindings-py**: Python Integration
   - PyO3 bindings with GIL release
   - Zero-copy NumPy/Pandas integration via Arrow
   - Error propagation to Python exceptions

6. **submod-service**: Service Layer
   - gRPC endpoints (tonic)
   - REST API (axum)
   - Job queue management
   - Authentication and authorization

## Key Dependencies

### Core Rust Libraries

**Data Structures:**
- `fixedbitset = "0.5"` - O(1) membership testing for SelectionView
- `rayon = "1.10"` - Data parallelism and parallel iterators

**Numerical Computing:**
- `nalgebra = "0.32"` - Linear algebra for Log-Determinant Cholesky
- `ndarray = "0.15"` - N-dimensional arrays (optional, for kernel matrices)

**Data I/O:**
- `arrow = "52"` - Apache Arrow for columnar data
- `parquet = "52"` - Parquet file format with predicate pushdown

**Randomness:**
- `rand = "0.8"` - RNG framework
- `rand_chacha = "0.3"` - ChaCha RNG for determinism (StdRng)

**Observability:**
- `prometheus = "0.13"` - Metrics collection and export
- `tracing = "0.1"` - Structured logging
- `serde_json = "1.0"` - JSON Lines audit logs

**Service Layer:**
- `tonic = "0.12"` - gRPC framework
- `axum = "0.7"` - REST API framework
- `tokio = "1.0"` - Async runtime

**Python Bindings:**
- `pyo3 = "0.21"` - Rust-Python FFI
- `numpy = "0.21"` - NumPy integration

### Development Tools

**Testing:**
- `cargo-nextest` - Fast parallel test runner
- `proptest = "1.0"` - Property-based testing
- `criterion = "0.5"` - Benchmarking framework

**Code Quality:**
- `clippy` - Rust linter (via rustup component)
- `rustfmt` - Code formatter (via rustup component)

**Environment Management:**
- `mise` - Development environment and task runner
  - Manages Rust toolchain
  - Defines development tasks (test, lint, format, etc.)
  - Python dependency compilation via pip-tools

## Data Formats

### Input Formats
- **Parquet**: Primary data format
  - Demand data: columns `[i: u32, w: f32]`
  - Utility data: columns `[i: u32, s: u32, u: f32]`
  - Compression: Snappy (default) or Zstd level 3
  - Row-group size: 128-256MB (configurable via SUBMOD_ROW_GROUP_TARGET_MB)

- **Arrow**: In-memory columnar format
  - Zero-copy interchange with Python
  - Efficient memory layout for SIMD

### Output Formats
- **JSON**: Selection results and configuration
- **JSON Lines**: Structured audit logs (one JSON object per line)
- **Prometheus**: Metrics export at `/metrics` endpoint

## Type System Decisions

### ItemId
- **Type**: `u32` (default) or `u64` (configurable)
- **Capacity**: 4.3 billion elements (u32)
- **Rationale**: Balance between range and memory efficiency

### Weight
- **Type**: `f32` (default) or `f64` (objective-specific)
- **Policy**:
  - `f32` for Facility Location, Saturating Coverage (2x memory reduction)
  - `f64` mandatory for Log-Determinant (Cholesky stability)
  - `f64` for Continuous Greedy gradient accumulation
- **Validation**: No NaN, no infinity, non-negative where required

### SelectionView
- **Implementation**: `fixedbitset::FixedBitSet` + `usize`
- **Memory**: |V|/8 bytes for bitset + 8 bytes for size
- **Rationale**: O(1) membership testing vs O(n) for Vec::contains

## Numerical Computing Strategy

### SIMD Vectorization
- Use `std::simd` for portable SIMD (Rust 1.75+)
- Vectorize inner loops: 4-8 elements per instruction
- Target: AVX2 (x86_64) and NEON (aarch64)
- Typical speedup: 4-8x on modern CPUs

### Parallel Evaluation
- Thread pool via Rayon's parallel iterators
- Immutable oracle pattern (`&self`) enables safe parallelism
- Fork pattern for thread-local caches (Arc for shared data)

### Floating-Point Robustness
- **Kahan summation**: Compensated summation for accuracy
- **Pairwise summation**: Recursive pair-wise for O(log n) error
- **Mixed-precision aggregation**: f64 for reductions in deterministic mode
- **Epsilon clipping**: Safe Schur complement computation

## Determinism Framework

### RNG Hierarchy
- **Master seed**: User-provided (e.g., 42)
- **Algorithm seed**: `master ^ ALGO_TAG` (unique per algorithm)
- **Iteration seed**: `algo_seed + iteration_num`
- **Implementation**: `rand::rngs::StdRng` (platform-independent)

### Fixed-Order Reduction
- Sort parallel results by ItemId when `determinism.fixed_order = true`
- Aggregate in f64 precision for reproducibility
- Trade-off: ~10% slowdown for determinism

### Tie-Breaking
- **Default**: `TieBreak::ById` (prefer lower ItemId)
- **Alternatives**: `ByUpperBound`, `Random` (seeded)

## Development Environment

### Setup Command
```bash
mise run setup
```

This single command:
- Installs Rust toolchain via mise
- Installs Python dependencies via pip-tools
- Configures development tools (rustfmt, clippy)

### Common Commands

**Testing:**
```bash
mise run test         # Run tests with nextest
mise run watch        # Watch and auto-test
```

**Code Quality:**
```bash
mise run check        # Quick compilation check
mise run lint         # Run clippy linter
mise run fmt          # Format code
```

**Benchmarking:**
```bash
mise run bench        # Run criterion benchmarks
```

**Python Dependencies:**
```bash
mise run pip-compile  # Compile requirements-dev.txt from requirements-dev.in
```

### Python Dependency Management

Uses **pip-tools** for full reproducibility:
1. Edit `requirements-dev.in` to add/update dependencies
2. Run `mise run pip-compile` to generate `requirements-dev.txt`
3. Run `mise run setup` to install updated dependencies
4. Commit both `.in` and `.txt` files

## Monitoring and Observability

### Prometheus Metrics
- **Naming**: snake_case with `submod_` prefix
- **Counters**: `_total` suffix (e.g., `submod_gain_eval_total`)
- **Gauges**: Current values (e.g., `submod_objective_value`)
- **Histograms**: Latency distributions (e.g., `submod_gain_compute_seconds`)
- **Labels**: `{algorithm="LazyGreedy"}` for dimensions
- **Endpoint**: HTTP `/metrics` on port 9090 (configurable)

### Structured Logging
- **Framework**: `tracing` with structured fields
- **Format**: JSON Lines for audit logs
- **Required fields**: iteration, element, gain, size, objective, algorithm, seed, timestamp

### Audit Log Schema
```json
{
  "iteration": 42,
  "selected_element": 1337,
  "marginal_gain": 123.456,
  "selection_size": 42,
  "objective_value": 5432.1,
  "algorithm": "LazyGreedy",
  "rng_seed": 42,
  "timestamp_ms": 15234,
  "git_hash": "a3f21c9"
}
```

## Environment Variables

### Configuration
- `SUBMOD_ROW_GROUP_TARGET_MB`: Parquet row-group size (default: 192)
- `RUST_LOG`: Logging level (e.g., `debug`, `info`, `warn`)
- `RUST_BACKTRACE`: Enable backtraces (set to `1` or `full`)

### Service Configuration
- `SUBMOD_METRICS_PORT`: Prometheus metrics port (default: 9090)
- `SUBMOD_GRPC_PORT`: gRPC service port (default: 50051)
- `SUBMOD_REST_PORT`: REST API port (default: 8080)

## Port Configuration

| Service | Port | Purpose |
|---------|------|---------|
| Prometheus Metrics | 9090 | `/metrics` endpoint |
| gRPC Service | 50051 | gRPC API |
| REST API | 8080 | HTTP REST endpoints |

All ports are configurable via environment variables.

## Build and Deployment

### Build Profiles
```toml
[profile.dev]
opt-level = 0  # Fast compilation

[profile.release]
opt-level = 3  # Maximum optimization
lto = "thin"   # Link-time optimization
codegen-units = 1  # Better optimization (slower build)
```

### CI/CD Matrix
- **Platforms**: Ubuntu, macOS, Windows
- **Rust Versions**: stable, nightly
- **Test Strategy**: nextest with 10-iteration determinism tests
- **Benchmarks**: Regression detection with criterion

## Security Considerations

### Input Validation
- Reject NaN, infinity, negative values at data loading
- Bounds checking: ItemId < universe_size
- Schema validation for Parquet files

### Service Layer
- Authentication: API key, JWT, or OAuth (configurable)
- Authorization: Role-based access control
- Rate limiting: Protect against abuse

### Dependencies
- Regular `cargo audit` for vulnerability scanning
- Minimal dependency tree to reduce attack surface
- Pin major versions for stability

## Performance Targets

### Baseline Problem
- **Scale**: n=10⁶, m=10⁶, nnz=10⁸, k=250
- **Hardware**: 32-core, 128GB RAM
- **Target**: < 10 minutes runtime
- **Lazy Efficiency**: < 10% evaluations vs. standard greedy

### Memory Budget
- **Sparse Matrix**: ~120-150 GB for 100 billion non-zero entries
- **Per non-zero**: 12 bytes (4B col_index + 4B value + 4B overhead)
- **Sharding**: Partition into ~128 shards for parallel loading

## License

**Apache-2.0** - Production-friendly open source license with patent protection
