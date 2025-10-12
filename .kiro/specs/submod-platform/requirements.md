# Requirements Document: Submodular Optimization Platform

## Introduction

The Submodular Optimization Platform is a production-grade Rust implementation designed to solve large-scale subset selection problems with provable approximation guarantees. The platform addresses problems with millions of candidates, millions of demand points, and billions of non-zero interactions, processing them in minutes on commodity hardware.

**Business Value:**
- Enable researchers and engineers to solve complex subset selection problems with theoretical guarantees
- Provide a thread-safe, deterministic, and numerically stable optimization framework
- Scale from toy problems (50 candidates) to massive datasets (millions of candidates, billions of interactions)
- Support multiple objective functions, algorithms, and constraint types through extensible architecture
- Integrate seamlessly with Python data science workflows and production service environments

**Core Design Pillars:**
1. **Theoretical Rigor:** Maintain provable (1-1/e) ≈ 0.632 approximation guarantees
2. **Thread Safety:** Enable safe parallel evaluation through immutable oracle interfaces
3. **Deterministic Execution:** Ensure reproducible results across runs, platforms, and parallelism levels
4. **Numerical Stability:** Handle floating-point arithmetic robustly with graceful degradation
5. **Production Readiness:** Provide comprehensive monitoring, auditing, and failure recovery

---

## Requirements

### Requirement 1: Core Framework and Type System
**Objective:** As a platform developer, I want a foundational type system and trait architecture, so that I can build thread-safe, efficient, and extensible optimization components.

#### Acceptance Criteria

1. WHEN the platform initializes THEN the Core Framework SHALL provide `ItemId` type as `u32` supporting up to 4.3 billion candidates
2. WHEN numerical precision is required THEN the Core Framework SHALL provide configurable `Weight` type defaulting to `f32` with optional `f64` for numerically sensitive operations
3. WHEN representing current selection state THEN the Core Framework SHALL provide `SelectionView` structure with `FixedBitSet` for O(1) membership testing
4. WHEN multiple elements have equal marginal gains THEN the Core Framework SHALL break ties deterministically by ItemId (prefer lower ID)
5. WHEN validating numerical inputs THEN the Core Framework SHALL reject NaN, infinity, and negative values where non-negativity is required
6. WHERE configuration is needed THEN the Core Framework SHALL provide `Strategy` enum supporting LazyGreedy, StochasticGreedy, SieveStreaming, and ContinuousGreedy algorithms
7. WHERE constraints are defined THEN the Core Framework SHALL provide `Constraint` enum supporting Cardinality, Knapsack, Partition, and Matroid constraints
8. WHEN deterministic execution is required THEN the Core Framework SHALL provide `Determinism` configuration with seed, fixed_order flag, and tie-breaking strategy

### Requirement 2: Thread-Safe Oracle Interface
**Objective:** As a solver developer, I want a thread-safe oracle interface with immutable queries, so that I can parallelize marginal gain evaluations without data races.

#### Acceptance Criteria

1. WHEN evaluating marginal gain THEN the Oracle Interface SHALL expose `gain(&self, view: &SelectionView, e: ItemId) -> f64` as immutable read operation
2. WHEN committing selected element THEN the Oracle Interface SHALL expose `commit(&mut self, e: ItemId)` as exclusive write operation
3. WHEN querying universe size THEN the Oracle Interface SHALL expose `universe_size(&self) -> usize` as immutable operation
4. WHEN lazy greedy optimization is used THEN the Oracle Interface SHALL expose `upper_bound(&self, e: ItemId) -> f64` returning most recently evaluated marginal gain
5. WHEN batch evaluation is beneficial THEN the Oracle Interface SHALL provide optional `gain_batch(&self, view: &SelectionView, candidates: &[ItemId]) -> Vec<f64>` for SIMD optimization
6. WHEN advanced parallelism is needed THEN the Oracle Interface SHALL provide optional `fork(&self) -> Box<dyn SubmodularOracle>` for thread-local caches with shared immutable data
7. WHERE forking is implemented THEN the Oracle Interface SHALL guarantee that forked instances only observe parent commit() operations
8. WHEN multiple threads evaluate candidates THEN the Oracle Interface SHALL ensure `gain()` calls are safe to execute concurrently via `&self` reference

### Requirement 3: Facility Location Objective Function
**Objective:** As a coverage optimization user, I want a Facility Location oracle implementation, so that I can maximize weighted coverage of demand points with selected candidates.

#### Acceptance Criteria

1. WHEN computing objective value THEN Facility Location SHALL calculate f(S) = Σᵢ wᵢ · max_{s∈S} u_{i,s} for all demands
2. WHEN evaluating marginal gain for candidate e THEN Facility Location SHALL compute Δ(e|S) = Σᵢ wᵢ · max(0, u_{i,e} - best_u[i]) in O(nnz(e)) time
3. WHEN committing selected candidate e THEN Facility Location SHALL update best_u[i] ← max(best_u[i], u[i,e]) for all demands with non-zero utility
4. WHEN loading data THEN Facility Location SHALL accept demand weights as `[i: u32, w: f32]` schema
5. WHEN loading utility matrix THEN Facility Location SHALL accept sparse utilities as `[i: u32, s: u32, u: f32]` schema with only non-zero entries
6. WHEN representing utility matrix internally THEN Facility Location SHALL use Compressed Sparse Row (CSR) format for O(nnz(e)) access
7. WHERE SIMD optimization is available THEN Facility Location SHALL vectorize inner loops processing 4-8 elements per instruction
8. WHEN batch evaluation is requested THEN Facility Location SHALL evaluate multiple candidates with shared demand access for cache efficiency

### Requirement 4: Saturating Coverage Objective Function
**Objective:** As a diminishing returns optimization user, I want a Saturating Coverage oracle with concave saturation functions, so that I can model objectives with sublinear benefits.

#### Acceptance Criteria

1. WHEN computing objective value THEN Saturating Coverage SHALL calculate f(S) = Σᵢ φᵢ(Σ_{s∈S} a_{i,s}) with concave functions φᵢ
2. WHEN evaluating marginal gain THEN Saturating Coverage SHALL compute Δ(e|S) = Σᵢ [φᵢ(cumulative[i] + a[i,e]) - φᵢ(cumulative[i])]
3. WHEN committing element THEN Saturating Coverage SHALL update cumulative[i] ← cumulative[i] + a[i,e] for all affected demands
4. WHERE concave functions are logarithmic THEN Saturating Coverage SHALL support φ(x) = log(1 + x)
5. WHERE concave functions are square root THEN Saturating Coverage SHALL support φ(x) = √x
6. WHERE concave functions are saturating THEN Saturating Coverage SHALL support φ(x) = min(x, τ) with threshold τ
7. WHEN evaluating concave functions THEN Saturating Coverage SHALL use precomputed lookup tables (LUT) for efficiency
8. WHERE LUT is used THEN Saturating Coverage SHALL ensure strict monotonicity to prevent numerical violations

### Requirement 5: Log-Determinant Objective Function with Numerical Safety
**Objective:** As a diversity optimization user, I want a Log-Determinant oracle with defense-in-depth numerical stability, so that I can select diverse subsets with graceful degradation on numerical failures.

#### Acceptance Criteria

1. WHEN computing objective value THEN Log-Determinant SHALL calculate f(S) = log det(K_{S,S} + εI) where ε ∈ [10⁻⁶, 10⁻⁴]
2. WHEN evaluating marginal gain THEN Log-Determinant SHALL compute Δ(e|S) = log(d) where d is Schur complement diagonal entry
3. WHEN maintaining state THEN Log-Determinant SHALL use Cholesky decomposition L where K_{S,S} + εI = LL^T in f64 precision (never f32)
4. WHERE numerical stability is critical THEN Log-Determinant SHALL apply epsilon regularization K ← K + εI as Layer 1 defense
5. WHEN computing Schur diagonal THEN Log-Determinant SHALL clip result to d_safe = max(d, ε · 0.1) as Layer 2 defense
6. IF Schur diagonal is small relative to mean THEN Log-Determinant SHALL use log1p for numerical accuracy as Layer 3 defense
7. WHEN NaN is detected in Cholesky update THEN Log-Determinant SHALL trigger full Cholesky recomputation as Layer 4 defense
8. IF consecutive failures exceed 5 THEN Log-Determinant SHALL degrade to Facility Location only mode and log warning as Layer 5 defense
9. WHERE kernel matrix is massive THEN Log-Determinant SHALL support low-rank approximation K ≈ ZZ^T with rank r ≪ n

### Requirement 6: Lazy Greedy Algorithm with Epoch-Based Optimization
**Objective:** As a performance-sensitive user, I want Lazy Greedy algorithm with epoch-based stale bound elimination, so that I can achieve 5-10% evaluation counts of standard greedy with provable guarantees.

#### Acceptance Criteria

1. WHEN selecting k elements THEN Lazy Greedy SHALL achieve (1-1/e) approximation for cardinality constraints
2. WHEN initializing heap THEN Lazy Greedy SHALL insert all candidates with upper bounds from `oracle.upper_bound(e)`
3. WHEN committing element THEN Lazy Greedy SHALL increment current_epoch by 1 to invalidate all existing heap entries
4. WHEN popping heap entry THEN Lazy Greedy SHALL discard entry if entry.epoch < current_epoch (stale bound)
5. WHEN evaluating candidate THEN Lazy Greedy SHALL compute actual marginal gain delta ← oracle.gain(&view, e)
6. IF delta + epsilon ≥ next_ub THEN Lazy Greedy SHALL commit element as definitively best
7. IF delta + epsilon < next_ub THEN Lazy Greedy SHALL re-insert with updated bound and current_epoch
8. WHERE parallel evaluation is enabled THEN Lazy Greedy SHALL evaluate top-M candidates concurrently using oracle.gain()
9. WHEN determinism is required THEN Lazy Greedy SHALL sort parallel results by ItemId before aggregation
10. WHEN comparing marginal gains THEN Lazy Greedy SHALL break ties by ItemId (prefer lower ID) for deterministic selection

### Requirement 7: Stochastic Greedy Algorithm
**Objective:** As a user optimizing massive datasets, I want Stochastic Greedy algorithm with O(n log(1/ε)) complexity, so that I can achieve (1-1/e-ε) approximation with dramatically reduced evaluations.

#### Acceptance Criteria

1. WHEN selecting k elements THEN Stochastic Greedy SHALL achieve E[f(S_k)] ≥ (1-1/e-ε)·f(S_opt) approximation
2. WHEN sampling at iteration t THEN Stochastic Greedy SHALL draw random subset R_t of size ceil((n/k)·ln(1/ε))
3. WHEN finding best in sample THEN Stochastic Greedy SHALL evaluate e_t = argmax_{e∈R_t} Δ(e|S_{t-1})
4. WHEN deterministic execution is required THEN Stochastic Greedy SHALL use seeded RNG StdRng::seed_from_u64(seed)
5. WHERE sampling is performed THEN Stochastic Greedy SHALL use reservoir sampling for uniform random selection
6. WHEN parallel evaluation is available THEN Stochastic Greedy SHALL evaluate sampled candidates concurrently
7. WHERE lazy optimization applies THEN Stochastic Greedy SHALL maintain epoch-based heap only for sampled candidates as hybrid approach

### Requirement 8: Continuous Greedy Algorithm for Matroid Constraints
**Objective:** As a user with complex constraints, I want Continuous Greedy algorithm optimizing multilinear extension, so that I can achieve (1-1/e) approximation for matroid-constrained problems.

#### Acceptance Criteria

1. WHEN optimizing over matroid THEN Continuous Greedy SHALL achieve (1-1/e) approximation guarantee
2. WHEN discretizing time THEN Continuous Greedy SHALL divide [0,1] into T steps with Δt = 1/T
3. WHEN estimating gradient THEN Continuous Greedy SHALL compute ĝᵢ = (1/G)·Σⱼ[f(Rⱼ∪{i}) - f(Rⱼ)] where Rⱼ ~ x^(t)
4. WHEN sampling for gradient THEN Continuous Greedy SHALL create temporary SelectionView without mutating oracle (view-based evaluation)
5. WHEN selecting direction THEN Continuous Greedy SHALL find maximum weight BASE (maximal independent set of size = rank) not just any independent set
6. WHEN updating fractional solution THEN Continuous Greedy SHALL compute x^(t+1)_i = min(x^(t)_i + 1_{i∈B^(t)}/T, 1.0)
7. WHERE partition matroid is used THEN Continuous Greedy SHALL apply Pipage rounding for deterministic integral solution
8. WHERE general matroid is used THEN Continuous Greedy SHALL apply Swap rounding with seeded RNG
9. WHEN materializing final solution THEN Continuous Greedy SHALL commit all selected items sequentially to oracle
10. IF determinism is required THEN Continuous Greedy SHALL derive iteration seeds as algo_seed + iteration_num

### Requirement 9: Sieve-Streaming Algorithm
**Objective:** As a streaming data user, I want Sieve-Streaming algorithm with single-pass processing, so that I can optimize subsets with O(k·log(1/ε)) memory in streaming scenarios.

#### Acceptance Criteria

1. WHEN processing stream THEN Sieve-Streaming SHALL achieve (1/2-ε) approximation for monotone submodular functions
2. WHEN initializing THEN Sieve-Streaming SHALL maintain threshold levels τ₁ > τ₂ > ... > τ_L in geometric sequence
3. WHEN element e arrives THEN Sieve-Streaming SHALL compute Δ(e|S_τ) for each threshold τ
4. IF Δ(e|S_τ) ≥ τ AND |S_τ| < k THEN Sieve-Streaming SHALL add e to solution S_τ
5. WHEN algorithm completes THEN Sieve-Streaming SHALL use O(k·L) memory where L = O(log(1/ε))

### Requirement 10: Standard Termination Conditions
**Objective:** As a user optimizing diverse problems, I want standard termination conditions across all algorithms, so that I can stop optimization early when appropriate without compromising solution quality.

#### Acceptance Criteria

1. WHEN cardinality constraint is reached THEN Algorithm SHALL terminate after exactly k iterations
2. IF upper_bound_threshold τ is configured AND next_ub < τ THEN Algorithm SHALL terminate with reason "upper_bound_threshold"
3. WHEN marginal gain Δ_t < ε·f(S_t) for m consecutive iterations THEN Algorithm SHALL terminate with reason "stagnation" where default m=3, ε=10⁻⁶
4. IF wall-clock time exceeds configured timeout THEN Algorithm SHALL terminate with reason "timeout" and return best solution found
5. WHEN termination occurs THEN Algorithm SHALL log specific termination_reason in audit log
6. WHERE timeout is configured THEN Algorithm SHALL check elapsed time at iteration boundaries for responsiveness

### Requirement 11: Cardinality Constraint
**Objective:** As a user selecting fixed-size subsets, I want cardinality constraint |S| ≤ k, so that I can limit selection to exactly k elements.

#### Acceptance Criteria

1. WHEN checking feasibility THEN Cardinality Constraint SHALL return true if view.size < k
2. WHEN element is committed THEN Cardinality Constraint SHALL require no internal state update beyond view.size
3. WHEN constraint is reset THEN Cardinality Constraint SHALL reset to zero state

### Requirement 12: Knapsack Constraint with Dual Modes
**Objective:** As a budget-constrained user, I want knapsack constraint with practical and theoretical modes, so that I can choose between speed (heuristic) and provable guarantees.

#### Acceptance Criteria

1. WHEN checking feasibility THEN Knapsack Constraint SHALL return true if used + cost_fn(e) ≤ budget
2. WHEN element is committed THEN Knapsack Constraint SHALL update used ← used + cost_fn(e)
3. WHEN comparing budget with floating-point THEN Knapsack Constraint SHALL use relative epsilon tolerance (not exact equality)
4. WHERE Practical mode is selected THEN Knapsack Constraint SHALL enumerate top-1 to top-3 items plus cost-benefit ratio greedy (no (1-1/e) guarantee)
5. WHERE Theoretical mode is selected THEN Knapsack Constraint SHALL apply Continuous Greedy achieving (1-1/e) guarantee (Sviridenko 2004)
6. WHEN mode is selected THEN Knapsack Constraint SHALL log mode choice and approximation guarantee in metadata
7. WHERE cost function is defined THEN Knapsack Constraint SHALL accept Arc<dyn Fn(ItemId) -> f64 + Send + Sync> for dynamic lookup

### Requirement 13: Partition Matroid Constraint
**Objective:** As a diversity-aware user, I want partition matroid constraint with per-category capacities, so that I can enforce balanced selection across predefined categories.

#### Acceptance Criteria

1. WHEN ground set is partitioned THEN Partition Matroid SHALL partition V = C₁ ⊔ C₂ ⊔ ... ⊔ C_m into disjoint categories
2. WHEN checking independence THEN Partition Matroid SHALL return true if |S ∩ C_j| ≤ cap_j for all j
3. WHEN computing rank THEN Partition Matroid SHALL return Σⱼ cap_j as total capacity
4. WHEN finding max-weight base THEN Partition Matroid SHALL greedily select top-cap_j elements by weight from each partition
5. WHERE partition function is defined THEN Partition Matroid SHALL accept Arc<dyn Fn(ItemId) -> usize + Send + Sync> for category assignment
6. WHEN ties occur in weight sorting THEN Partition Matroid SHALL break ties by ItemId for determinism

### Requirement 14: Graphic Matroid Constraint
**Objective:** As a graph optimization user, I want graphic matroid constraint detecting cycles, so that I can select acyclic edge subsets (forests) with maximum submodular value.

#### Acceptance Criteria

1. WHEN checking feasibility THEN Graphic Matroid SHALL return true if adding edge e does not create cycle
2. WHEN committing edge e THEN Graphic Matroid SHALL union endpoints (u,v) in Union-Find structure
3. WHEN finding max-weight base THEN Graphic Matroid SHALL apply Kruskal's algorithm: sort edges by weight descending, add greedily if no cycle
4. WHEN resetting THEN Graphic Matroid SHALL reinitialize Union-Find with n_vertices singleton components

### Requirement 15: Parquet Data I/O with Row-Group Optimization
**Objective:** As a data engineer, I want efficient Parquet/Arrow data loading with predicate pushdown, so that I can load sparse utility matrices with billions of non-zero entries efficiently.

#### Acceptance Criteria

1. WHEN loading demand data THEN Data I/O SHALL accept schema `[i: u32, w: f32]` from demand.parquet
2. WHEN loading utility data THEN Data I/O SHALL accept schema `[i: u32, s: u32, u: f32]` from utility.parquet with only non-zero entries
3. WHEN organizing row groups THEN Data I/O SHALL cluster by demand ID `i` for sequential access during marginal gain computation
4. WHERE row group size is configured THEN Data I/O SHALL use SUBMOD_ROW_GROUP_TARGET_MB environment variable (default 192MB)
5. WHEN applying compression THEN Data I/O SHALL use Snappy by default with configurable Zstd level 3
6. WHERE predicate pushdown is applicable THEN Data I/O SHALL push filters like `i IN (...)` to Parquet reader for row-group skipping
7. WHEN validating data THEN Data I/O SHALL reject NaN, infinity, negative values where non-negativity required, and out-of-bound IDs (fail-fast)
8. IF duplicate (i,s) pairs exist THEN Data I/O SHALL apply configurable policy: max utility (default), sum, average, or error
9. WHEN constructing sparse matrix THEN Data I/O SHALL explicitly drop entries with u = 0 during CSR construction

### Requirement 16: Sparse Matrix Construction and Sharding
**Objective:** As a performance engineer, I want CSR sparse matrix construction with parallel sharding, so that I can load and process massive datasets with bounded memory.

#### Acceptance Criteria

1. WHEN representing utility matrix THEN Sparse Matrix SHALL use Compressed Sparse Row (CSR) format with row_ptr, col_indices, values vectors
2. WHEN accessing row i THEN Sparse Matrix SHALL retrieve indices as col_indices[row_ptr[i]..row_ptr[i+1]] in O(nnz(i)) time
3. WHERE candidate-centric access dominates THEN Sparse Matrix SHALL transpose to Compressed Sparse Column (CSC) format
4. WHEN sharding data THEN Sparse Matrix SHALL partition into P shards (P ≈ 128) by hash(i) mod P
5. WHEN loading shards THEN Sparse Matrix SHALL use parallel loading with producer-consumer pattern via crossbeam channel
6. WHERE prefetching is beneficial THEN Sparse Matrix SHALL async load shards i+1, i+2 while processing shard i
7. WHEN bounding memory THEN Sparse Matrix SHALL limit prefetch channel capacity to prevent memory explosion

### Requirement 17: Numerical Stability and Floating-Point Type Selection
**Objective:** As a numerical computing user, I want robust floating-point handling with dtype selection per objective, so that I can avoid numerical failures while maintaining efficiency.

#### Acceptance Criteria

1. WHEN using Facility Location or Saturating Coverage THEN Platform SHALL default to f32 for 2x memory reduction and wider SIMD (8 lanes vs 4)
2. WHEN using Log-Determinant THEN Platform SHALL mandate f64 as f32 fails rapidly in Cholesky accumulation
3. WHEN using Continuous Greedy THEN Platform SHALL use f64 for gradient accumulation even if oracle uses f32
4. WHERE type is configurable THEN Platform SHALL enable compile-time monomorphization avoiding runtime branching overhead
5. WHEN summing in parallel THEN Platform SHALL apply Kahan summation or pairwise summation for improved accuracy
6. WHERE determinism requires fixed order THEN Platform SHALL aggregate in f64 for reductions when determinism.fixed_order = true
7. WHEN hashing for audit THEN Platform SHALL hash only integer decisions (ItemId sequence, seed, algorithm) never float values

### Requirement 18: Deterministic Execution Framework
**Objective:** As a reproducible research user, I want deterministic execution guarantees, so that identical inputs produce identical selection sequences, objective values, and audit log hashes across runs and parallelism levels.

#### Acceptance Criteria

1. WHEN identical inputs are provided THEN Platform SHALL produce identical selection sequence S₁, S₂, ..., S_k
2. WHEN identical inputs are provided THEN Platform SHALL produce identical objective value within floating-point epsilon
3. WHEN identical inputs are provided THEN Platform SHALL produce identical audit log hash across runs
4. WHERE determinism is configured THEN Platform SHALL use hierarchical RNG seeding: master_seed → algo_seed (master ^ ALGO_TAG) → iter_seed (algo_seed + iter)
5. WHEN parallel reduction is performed THEN Platform SHALL optionally fix reduction order by sorting results by ItemId before aggregation (determinism.fixed_order = true)
6. WHERE tie-breaking occurs THEN Platform SHALL use TieBreak::ById (prefer lower ItemId) as default for deterministic selection
7. WHEN using StdRng THEN Platform SHALL ensure platform-independent random number generation
8. WHERE FP differences across platforms exist THEN Platform SHALL accept objective value differences < 10⁻⁶ as tolerance

### Requirement 19: Prometheus Metrics for Monitoring
**Objective:** As an operations engineer, I want comprehensive Prometheus metrics, so that I can monitor solver performance, detect bottlenecks, and alert on anomalies.

#### Acceptance Criteria

1. WHEN exposing metrics THEN Platform SHALL follow naming convention: snake_case with submod_ prefix and unit suffixes (_seconds, _bytes, _total)
2. WHERE counters are needed THEN Platform SHALL expose submod_gain_eval_total, submod_commit_total, submod_heap_pop_total, submod_heap_reinsert_total, submod_fallback_total
3. WHERE gauges are needed THEN Platform SHALL expose submod_objective_value, submod_selection_size, submod_upper_bound_max, submod_gap_estimate
4. WHERE histograms are needed THEN Platform SHALL expose submod_gain_compute_seconds, submod_iteration_seconds, submod_io_wait_seconds
5. WHEN using Continuous Greedy THEN Platform SHALL expose submod_grad_variance, submod_base_weight_sum, submod_rounding_loss
6. WHERE metrics endpoint is exposed THEN Platform SHALL serve HTTP /metrics endpoint on configurable port (default 9090)
7. WHEN labeling metrics THEN Platform SHALL include algorithm dimension as label (e.g., {algorithm="LazyGreedy"})

### Requirement 20: Structured Audit Logs (JSON Lines)
**Objective:** As a debugging user, I want structured audit logs with mandatory fields, so that I can reconstruct optimization history, perform post-mortem analysis, and verify correctness.

#### Acceptance Criteria

1. WHEN logging iteration THEN Platform SHALL include mandatory fields: iteration, selected_element, marginal_gain, selection_size, objective_value, algorithm, rng_seed, timestamp_ms
2. WHERE git tracking is available THEN Platform SHALL include optional git_hash field
3. WHEN logging thresholds THEN Platform SHALL use semantic names: lazy_epsilon, stagnation_threshold, ub_threshold with values
4. IF termination occurs THEN Platform SHALL log termination_reason as one of: "cardinality_reached", "upper_bound_threshold", "stagnation", "timeout"
5. WHERE counterfactuals are useful THEN Platform SHALL log top-K runner-up candidates with element, marginal_gain, deficit, reason fields
6. WHEN using Facility Location THEN Platform SHALL optionally log top_demand_contributions with demand_id, contribution, utility
7. IF numerical fallback occurs THEN Platform SHALL log event: "fallback" with component, reason, element, action fields
8. WHERE audit log is written THEN Platform SHALL use JSON Lines format (one JSON object per line) for streaming parsability

### Requirement 21: Python Bindings with GIL Release
**Objective:** As a Python data scientist, I want PyO3 bindings with GIL release and zero-copy NumPy/Pandas integration, so that I can use the platform from Python without blocking or expensive data copies.

#### Acceptance Criteria

1. WHEN calling solver from Python THEN Python Bindings SHALL release Global Interpreter Lock during computation using Python::allow_threads()
2. WHERE NumPy arrays are passed THEN Python Bindings SHALL use zero-copy Arrow memory layout for interchange
3. WHERE Pandas DataFrames are passed THEN Python Bindings SHALL avoid unnecessary copies between Rust and Python heaps
4. WHEN Rust error occurs THEN Python Bindings SHALL convert Result<T,E> to Python exception with detailed context (file, line, error chain)
5. WHERE long-running computation occurs THEN Python Bindings SHALL enable true parallelism by releasing GIL preventing blocking of other Python threads
6. WHEN returning results THEN Python Bindings SHALL convert Selection to Python dict with items, objective, used_budget, counts_by_part, trace fields

### Requirement 22: Service Layer with gRPC and REST
**Objective:** As a service integration user, I want gRPC and REST endpoints with job queue management, so that I can integrate optimization into microservice architectures.

#### Acceptance Criteria

1. WHEN exposing RPC interface THEN Service Layer SHALL provide gRPC endpoints using tonic framework
2. WHEN exposing HTTP interface THEN Service Layer SHALL provide REST endpoints using axum framework
3. WHERE async job processing is needed THEN Service Layer SHALL implement job queue with status tracking (queued, running, completed, failed)
4. WHEN authentication is required THEN Service Layer SHALL support configurable auth mechanisms (API key, JWT, OAuth)
5. WHERE Prometheus integration exists THEN Service Layer SHALL expose /metrics endpoint for scraping
6. WHEN job completes THEN Service Layer SHALL store results and audit logs for retrieval
7. IF job fails THEN Service Layer SHALL return detailed error response with failure reason and partial results if available

### Requirement 23: Testing and Quality Assurance
**Objective:** As a platform maintainer, I want comprehensive testing coverage, so that I can verify correctness, approximation guarantees, determinism, and performance.

#### Acceptance Criteria

1. WHEN testing properties THEN Platform SHALL use proptest for property-based tests verifying submodularity and monotonicity
2. WHEN testing approximation quality THEN Platform SHALL verify f(S_greedy) ≥ (1-1/e)·f(S_opt) on small problems with brute-force optimal
3. WHEN testing determinism THEN Platform SHALL run 10 identical iterations and assert identical outputs (selection sequence, objective, audit hash)
4. WHERE cross-platform testing is performed THEN Platform SHALL test on {Ubuntu, macOS, Windows} × {stable, nightly} matrix
5. WHEN benchmarking THEN Platform SHALL use criterion for performance regression detection
6. WHERE baseline is defined THEN Platform SHALL target <10 min runtime for n=10⁶, m=10⁶, nnz=10⁸, k=250 on 32-core 128GB RAM
7. IF objective value regresses >1% THEN Platform SHALL alert on standard datasets
8. WHEN testing numerical edge cases THEN Platform SHALL verify NaN rejection, infinity rejection, negative value rejection, and graceful degradation on Log-Determinant failures

### Requirement 24: Quick Start Example and Documentation
**Objective:** As a new user, I want a runnable minimal example and comprehensive documentation, so that I can quickly validate the platform and learn how to use it effectively.

#### Acceptance Criteria

1. WHEN providing minimal example THEN Platform SHALL include 10KB dataset: 100 demands, 50 candidates, 10% density (~500 utilities)
2. WHERE quick start is documented THEN Platform SHALL provide generation script (Python) creating demand_mini.parquet and utility_mini.parquet
3. WHEN running minimal example THEN Platform SHALL complete in <1 second demonstrating end-to-end pipeline
4. IF user runs quick start THEN Platform SHALL output deterministic selection sequence with seed 42 achieving ~45.2 objective
5. WHERE verification is needed THEN Platform SHALL provide checklist: coverage curve shows diminishing returns, lazy efficiency <10%, determinism verified, audit log complete
6. WHEN documenting API THEN Platform SHALL provide user guide, API reference, and tutorials
7. WHERE examples are provided THEN Platform SHALL include examples for each objective function, algorithm, and constraint type

### Requirement 25: Continuous Integration and Release
**Objective:** As a project maintainer, I want automated CI/CD with cross-platform testing and benchmarks, so that I can ensure quality and catch regressions before release.

#### Acceptance Criteria

1. WHEN committing code THEN CI SHALL run test matrix on {Ubuntu, macOS, Windows} × {stable, nightly}
2. WHERE determinism is critical THEN CI SHALL run determinism test with 10 identical iterations asserting identical outputs
3. WHEN benchmarks are run THEN CI SHALL use criterion for performance measurement and regression detection
4. IF cross-platform differences exist THEN CI SHALL accept objective value differences <10⁻⁶ as tolerance
5. WHERE release is created THEN CI SHALL build binaries for all supported platforms
6. WHEN publishing crate THEN CI SHALL verify version bumps, changelog updates, and documentation completeness

---

## Phased Implementation Alignment

The requirements above align with the 6-phase implementation roadmap (22-30 weeks total):

**Phase 1 (4-6 weeks):** Requirements 1-3, 6, 11 - Core + Facility Location
**Phase 2 (3-4 weeks):** Requirements 15-16 - Data I/O + Parallel
**Phase 3 (4-5 weeks):** Requirements 4-5 - Additional Objectives
**Phase 4 (6-8 weeks):** Requirements 8, 12-14 - Matroid + Continuous Greedy
**Phase 5 (3-4 weeks):** Requirements 21-22 - Python + Service
**Phase 6 (2-3 weeks):** Requirements 23-25 - Hardening + Testing

Requirements 7, 9-10, 17-20 are cross-cutting and apply throughout all phases.

---

## Success Criteria

The platform successfully meets requirements when:

1. **Correctness:** All algorithms achieve stated approximation guarantees verified by tests
2. **Scale:** Processes 10⁶ candidates, 10⁶ demands, 10⁸ non-zero entries in <10 min
3. **Determinism:** 10 identical runs produce identical selection sequences and audit hashes
4. **Stability:** Log-Determinant gracefully degrades after 5 consecutive numerical failures
5. **Usability:** Quick start example runs in <1 second with <10 lines of code
6. **Production:** Prometheus metrics expose all core counters, gauges, and histograms
7. **Integration:** Python bindings release GIL and support zero-copy NumPy/Pandas
8. **Quality:** CI passes on all platforms with <1% performance regression tolerance
