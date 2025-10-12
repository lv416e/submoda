# Project Structure: Submoda

## Repository Organization

```
submoda/
├── .kiro/                    # Kiro spec-driven development
│   ├── specs/               # Feature specifications
│   │   └── submod-platform/ # Main platform spec
│   │       ├── spec.json    # Workflow state tracking
│   │       ├── requirements.md  # 25 requirements, 183 criteria
│   │       ├── design.md    # Complete technical design
│   │       └── tasks.md     # 74 implementation tasks
│   └── steering/            # Project steering documents
│       ├── product.md       # Product overview (this was just created)
│       ├── tech.md          # Technology stack (this was just created)
│       └── structure.md     # This file
├── .claude/                 # Claude Code configuration
│   └── commands/            # Custom slash commands
│       ├── kiro:*.md        # Kiro workflow commands
│       └── speckit.*.md     # Speckit workflow commands
├── docs/                    # Project documentation
│   └── specification.md     # Complete technical specification (2,469 lines)
├── submod-core/             # Core traits and types (to be created)
├── submod-objectives/       # Objective function implementations (to be created)
├── submod-solver/           # Algorithm implementations (to be created)
├── submod-io/               # Data I/O layer (to be created)
├── submod-bindings-py/      # Python bindings (to be created)
├── submod-service/          # Service layer (to be created)
├── Cargo.toml               # Workspace configuration (to be created)
├── README.md                # Project overview
├── CLAUDE.md                # Development guidelines for Claude Code
└── LICENSE                  # Apache-2.0 license (to be created)
```

## Development Phase

**Current Status**: Ready for Implementation (Phase 0)

The project is in specification phase with all design documents completed. Implementation will begin with Phase 1 (Core Framework + Facility Location).

## Planned Crate Structure

### submod-core/ (Foundation Crate)

**Purpose**: Core traits, types, and infrastructure

```
submod-core/
├── src/
│   ├── lib.rs              # Crate root and module exports
│   ├── oracle.rs           # SubmodularOracle trait
│   ├── constraint.rs       # Constraint and Matroid traits
│   ├── types.rs            # ItemId, Weight, SelectionView
│   ├── selection.rs        # Selection result structure
│   ├── config.rs           # Strategy, Determinism, Termination configs
│   ├── determinism.rs      # RNG hierarchy and fixed-order reduction
│   ├── metrics.rs          # Prometheus metrics registry
│   └── validation.rs       # Input validation functions
├── tests/
│   ├── types_test.rs       # Type system validation tests
│   └── determinism_test.rs # Determinism framework tests
└── Cargo.toml
```

**Key Exports**:
- `SubmodularOracle` trait with `gain()`, `commit()`, `upper_bound()`
- `Constraint` trait for feasibility checking
- `SelectionView` with `FixedBitSet` for O(1) membership
- Configuration types: `Strategy`, `Constraint`, `Determinism`

### submod-objectives/ (Oracle Implementations)

**Purpose**: Concrete objective function implementations

```
submod-objectives/
├── src/
│   ├── lib.rs              # Crate root
│   ├── facility.rs         # FacilityLocation oracle
│   ├── saturating.rs       # SaturatingCoverage oracle
│   ├── logdet.rs           # LogDeterminant oracle
│   │   ├── mod.rs          # Main implementation
│   │   ├── cholesky.rs     # Incremental Cholesky updates
│   │   ├── safety.rs       # 5-layer numerical stability
│   │   └── lowrank.rs      # Low-rank approximation
│   └── diversity.rs        # Diversity-aware Facility Location
├── tests/
│   ├── facility_test.rs    # Facility Location tests
│   ├── saturating_test.rs  # Saturating Coverage tests
│   └── logdet_test.rs      # Log-Determinant numerical safety tests
└── Cargo.toml
```

**Key Types**:
- `FacilityLocation` with CSR sparse matrix
- `SaturatingCoverage` with concave function library
- `LogDeterminant` with 5-layer numerical safety

### submod-solver/ (Algorithm Implementations)

**Purpose**: Optimization algorithms

```
submod-solver/
├── src/
│   ├── lib.rs              # Crate root
│   ├── lazy_greedy.rs      # Lazy Greedy with epoch-based heap
│   │   ├── mod.rs          # Main algorithm
│   │   ├── heap.rs         # HeapEntry with epoch tracking
│   │   └── parallel.rs     # Parallel evaluation
│   ├── stochastic.rs       # Stochastic Greedy
│   │   ├── mod.rs          # Main algorithm
│   │   └── sampling.rs     # Reservoir sampling
│   ├── continuous.rs       # Continuous Greedy
│   │   ├── mod.rs          # Main algorithm
│   │   ├── gradient.rs     # View-based gradient estimation
│   │   └── rounding.rs     # Pipage and Swap rounding
│   ├── sieve.rs            # Sieve-Streaming
│   ├── termination.rs      # Termination condition checking
│   └── utils.rs            # Shared utilities
├── tests/
│   ├── lazy_test.rs        # Lazy Greedy correctness tests
│   ├── stochastic_test.rs  # Stochastic Greedy approximation tests
│   └── continuous_test.rs  # Continuous Greedy with matroids
└── Cargo.toml
```

**Key Algorithms**:
- `LazyGreedy::solve()` with epoch-based optimization
- `StochasticGreedy::solve()` with O(n log 1/ε) complexity
- `ContinuousGreedy::solve()` for matroid constraints

### submod-io/ (Data Layer)

**Purpose**: Parquet/Arrow I/O and sparse matrix construction

```
submod-io/
├── src/
│   ├── lib.rs              # Crate root
│   ├── parquet.rs          # Parquet reader with predicate pushdown
│   │   ├── mod.rs          # Main reader
│   │   ├── validation.rs   # Schema and data validation
│   │   └── rowgroup.rs     # Row-group optimization
│   ├── sparse.rs           # CSR/CSC sparse matrix
│   │   ├── csr.rs          # Compressed Sparse Row
│   │   ├── csc.rs          # Compressed Sparse Column
│   │   └── builder.rs      # Construction from triplets
│   ├── sharding.rs         # Hash-based sharding
│   └── prefetch.rs         # Async prefetching
├── tests/
│   ├── parquet_test.rs     # Parquet loading tests
│   └── sparse_test.rs      # Sparse matrix construction tests
└── Cargo.toml
```

**Key Functionality**:
- Parquet reading with row-group clustering
- CSR matrix construction with zero filtering
- Parallel shard loading with producer-consumer pattern

### submod-bindings-py/ (Python Integration)

**Purpose**: PyO3 bindings for Python integration

```
submod-bindings-py/
├── src/
│   ├── lib.rs              # PyO3 module definition
│   ├── solver.rs           # PySubmodSolver class
│   ├── io.rs               # Zero-copy data loading
│   ├── errors.rs           # Rust-to-Python error conversion
│   └── types.rs            # Python type wrappers
├── tests/
│   └── integration_test.py # Python integration tests
├── pyproject.toml          # Python package metadata
└── Cargo.toml
```

**Key Exports**:
- `PySubmodSolver` class with GIL release
- `load_from_parquet()` and `load_from_numpy()` functions
- Exception classes for error propagation

### submod-service/ (Service Layer)

**Purpose**: gRPC and REST service endpoints

```
submod-service/
├── src/
│   ├── main.rs             # Service entry point
│   ├── grpc.rs             # gRPC endpoints (tonic)
│   │   ├── mod.rs          # Service implementation
│   │   └── proto.rs        # Generated protobuf code
│   ├── rest.rs             # REST API (axum)
│   │   ├── mod.rs          # Route handlers
│   │   └── handlers.rs     # Request/response handling
│   ├── queue.rs            # Job queue management
│   ├── auth.rs             # Authentication and authorization
│   └── metrics.rs          # Prometheus metrics endpoint
├── proto/
│   └── submod.proto        # Protobuf schema
├── tests/
│   ├── grpc_test.rs        # gRPC endpoint tests
│   └── rest_test.rs        # REST API tests
└── Cargo.toml
```

**Key Endpoints**:
- gRPC: `submit_job`, `get_job_status`, `get_job_result`
- REST: `POST /api/v1/jobs`, `GET /api/v1/jobs/:id`, `GET /metrics`

## Code Organization Patterns

### Trait-Based Abstraction

**Pattern**: Abstract interfaces with concrete implementations

```rust
// Core trait (submod-core)
pub trait SubmodularOracle: Send + Sync {
    fn gain(&self, view: &SelectionView, e: ItemId) -> f64;
    fn commit(&mut self, e: ItemId);
    fn universe_size(&self) -> usize;
}

// Concrete implementation (submod-objectives)
impl SubmodularOracle for FacilityLocation {
    fn gain(&self, view: &SelectionView, e: ItemId) -> f64 {
        // Implementation
    }
    // ...
}
```

**Benefits**:
- Algorithm code is oracle-agnostic
- Easy to add new objective functions
- Testable with mock oracles

### Builder Pattern for Configuration

**Pattern**: Fluent API for complex configuration

```rust
let config = SolverConfig::builder()
    .strategy(Strategy::LazyGreedy { epsilon: 0.0 })
    .constraint(Constraint::Cardinality { k: 250 })
    .determinism(Determinism::default())
    .build()?;
```

**Benefits**:
- Type-safe configuration construction
- Required vs optional parameters enforced
- Clear API for users

### Arc Pattern for Shared Data

**Pattern**: Immutable shared data with Arc

```rust
pub struct FacilityLocation {
    demand_weights: Arc<Vec<f32>>,      // Shared, immutable
    utility_matrix: Arc<CsrMatrix>,     // Shared, immutable
    best_u: Vec<f32>,                   // Mutable, per-instance
}

impl SubmodularOracle for FacilityLocation {
    fn fork(&self) -> Box<dyn SubmodularOracle> {
        Box::new(Self {
            demand_weights: self.demand_weights.clone(),  // Arc clone: O(1)
            utility_matrix: self.utility_matrix.clone(),  // Arc clone: O(1)
            best_u: vec![0.0; self.n_demands],           // New mutable state
        })
    }
}
```

**Benefits**:
- Cheap cloning for parallel workers
- Clear separation of shared vs per-thread state
- Thread-safe via Send + Sync

### View-Based Evaluation Pattern

**Critical Pattern**: Temporary views for gradient estimation

```rust
// ✓ CORRECT: View-based evaluation (no oracle mutation)
fn estimate_gradient(oracle: &impl SubmodularOracle, x: &[f64]) -> Vec<f64> {
    let mut grad = vec![0.0; x.len()];

    for _ in 0..num_samples {
        let mut view = SelectionView::new(x.len());

        // Build temporary view from fractional solution
        for i in 0..x.len() {
            if random() < x[i] {
                view.in_set.insert(i);
                view.size += 1;
            }
        }

        // Evaluate with immutable oracle
        for i in 0..x.len() {
            grad[i] += oracle.gain(&view, i);  // ✓ &self, no mutation
        }
    }

    grad
}
```

**Why Critical**:
- Oracle state remains valid during sampling
- Multiple samples can be evaluated in parallel
- Prevents state corruption in Continuous Greedy

## File Naming Conventions

### Rust Files
- **Modules**: `snake_case.rs` (e.g., `lazy_greedy.rs`, `facility_location.rs`)
- **Tests**: `*_test.rs` suffix (e.g., `determinism_test.rs`)
- **Benchmarks**: `*_bench.rs` suffix (e.g., `lazy_bench.rs`)

### Documentation Files
- **Markdown**: `PascalCase.md` or `lowercase.md` (e.g., `README.md`, `specification.md`)
- **Specification**: `spec.md` in feature directories
- **Design**: `design.md` for technical design
- **Tasks**: `tasks.md` for implementation tasks

### Configuration Files
- **Cargo**: `Cargo.toml` (standard Rust)
- **Mise**: `.mise.toml` or `mise.toml`
- **Python**: `pyproject.toml`, `requirements-dev.in`, `requirements-dev.txt`

## Import Organization

### Standard Order
```rust
// 1. Standard library
use std::collections::HashMap;
use std::sync::Arc;

// 2. External crates (alphabetically)
use fixedbitset::FixedBitSet;
use rayon::prelude::*;

// 3. Internal crates (alphabetically)
use submod_core::{ItemId, SelectionView, SubmodularOracle};
use submod_io::CsrMatrix;

// 4. Internal modules (alphabetically)
use crate::config::Strategy;
use crate::heap::HeapEntry;
```

### Re-exports in lib.rs
```rust
// Public API exports
pub use crate::oracle::SubmodularOracle;
pub use crate::types::{ItemId, Weight, SelectionView};
pub use crate::config::{Strategy, Constraint, Determinism};
```

**Benefits**:
- Users import from crate root: `use submod_core::{ItemId, SelectionView}`
- Internal organization hidden from public API
- Easy refactoring without breaking API

## Key Architectural Principles

### 1. Immutable Read, Mutable Write Separation

**Principle**: Marginal gain evaluation (`gain()`) is a read operation using `&self`. Only `commit()` mutates state with `&mut self`.

**Impact**:
- Enables safe parallel evaluation
- Clear concurrency boundaries
- Prevents subtle race conditions

### 2. SelectionView as Canonical Representation

**Principle**: During optimization, `SelectionView` with `FixedBitSet` is the single source of truth for current selection.

**Impact**:
- O(1) membership testing vs O(n) for `Vec::contains`
- Memory-efficient: |V|/8 bytes vs 4|S| bytes for Vec<ItemId>
- Thread-safe cloning for parallel evaluations

### 3. Epoch-Based Staleness Tracking

**Principle**: Each heap entry carries an epoch timestamp. On commit, increment global epoch to invalidate all cached bounds.

**Impact**:
- 90-95% reduction in evaluations vs standard greedy
- Prevents accumulation of obsolete upper bounds
- Simple invariant: only current-epoch entries are valid

### 4. View-Based Gradient Estimation (Continuous Greedy)

**Principle**: Never mutate oracle during gradient estimation. Use temporary `SelectionView` instances for sampling.

**Impact**:
- Preserves oracle correctness during multi-sample gradient estimation
- Enables parallel gradient computation
- Critical for mathematical correctness

### 5. Defense-in-Depth for Numerical Stability

**Principle**: Multiple layers of safety for numerically sensitive operations (Log-Determinant).

**5 Layers**:
1. Epsilon regularization (K + εI)
2. Safe Schur computation (clipping)
3. log1p for small values
4. Immediate refactorization on NaN
5. Graceful degradation after failures

**Impact**:
- Robustness on ill-conditioned matrices
- Clear failure modes and recovery paths
- Production-grade reliability

### 6. Determinism by Design

**Principle**: Reproducibility is a first-class requirement, not an afterthought.

**Mechanisms**:
- Hierarchical RNG seeding (master → algo → iteration)
- Fixed-order parallel reduction
- Deterministic tie-breaking by ItemId
- Platform-independent StdRng

**Impact**:
- Debugging reproducible bugs
- CI test stability
- Provable approximation quality

### 7. Zero-Copy Data Interchange

**Principle**: Avoid memory copies between Rust and Python via Arrow/FFI.

**Impact**:
- NumPy arrays accessed directly from Rust
- Minimal serialization overhead
- Efficient Python integration

### 8. Fail-Fast Validation

**Principle**: Validate all inputs at data loading time. Reject NaN, infinity, negative values, out-of-bounds IDs.

**Impact**:
- Errors caught early with clear messages
- No silent corruption or late failures
- Production-grade reliability

## Testing Strategy

### Unit Tests
- **Location**: `tests/` directory in each crate
- **Naming**: `*_test.rs` files
- **Scope**: Single function or module
- **Runner**: `cargo nextest run`

### Integration Tests
- **Location**: Top-level `tests/` directory (future)
- **Scope**: Cross-crate workflows
- **Examples**: End-to-end solver runs, data I/O pipeline

### Property-Based Tests
- **Framework**: `proptest`
- **Use Cases**: Submodularity property, matroid axioms, determinism
- **Example**: Verify `f(A∪B) + f(A∩B) ≤ f(A) + f(B)` for random sets

### Benchmarks
- **Framework**: `criterion`
- **Location**: `benches/` directory in each crate
- **Purpose**: Regression detection, performance profiling

## Documentation Standards

### Code Documentation
- **Public items**: Rustdoc comments with examples
- **Format**: `///` for functions, `//!` for modules
- **Examples**: `#[doc = include_str!("../README.md")]` for crate docs

### Specification Documents
- **Location**: `.kiro/specs/` for feature specifications
- **Format**: Markdown with EARS acceptance criteria
- **Sections**: Requirements, Design, Tasks

### User Guides
- **Location**: `docs/` directory
- **Examples**: Quick start, tutorials, API reference
- **Generation**: `cargo doc` for API reference

## Current Implementation Status

**Phase**: Specification Complete, Ready for Implementation

**Next Steps**:
1. Initialize Cargo workspace with 6 crates
2. Begin Phase 1: Core Framework + Facility Location
3. Follow tasks.md implementation sequence

**Estimated Timeline**: 22-30 weeks across 6 phases
