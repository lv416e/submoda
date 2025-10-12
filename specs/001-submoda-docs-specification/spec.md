# Feature Specification: Submodular Optimization Platform (submoda)

**Feature Branch**: `001-submoda-docs-specification`
**Created**: 2025-10-12
**Status**: Draft
**Input**: User description: "submoda を実現するために、 @docs/specification.md のみを解析。 WHAT と WHY に焦点を当てて抽出すること。要件/非機能/KPI/スコープ/除外/主要ユーザフロー/失敗時挙動/計測指標を抽出。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Large-Scale Subset Selection (Priority: P1)

A data scientist needs to select an optimal subset of 250 candidate items from 1 million candidates to maximize coverage of 1 million demand points, processing billions of non-zero utility relationships in minutes on commodity hardware.

**Why this priority**: This is the core use case - without it, the platform has no value. This represents the primary problem domain (facility location, content recommendation, sensor placement) that justifies building the entire system.

**Independent Test**: Can be fully tested by loading a dataset with 1M candidates, 1M demands, 100M non-zero utilities, running greedy algorithm with cardinality constraint k=250, and verifying completion within 10 minutes on a 32-core machine.

**Acceptance Scenarios**:

1. **Given** a Parquet dataset with demand weights and sparse utility matrix, **When** user runs lazy greedy with k=250, **Then** system returns 250 selected items with objective value within 63.2% of optimal (1-1/e approximation guarantee)
2. **Given** the same dataset and seed=42, **When** user runs the algorithm twice, **Then** both runs produce identical selection sequences and objective values (deterministic execution)
3. **Given** a dataset with 100 billion non-zero utilities, **When** system loads data, **Then** memory usage stays under 150GB and processing completes within target time

---

### User Story 2 - Production-Grade Reliability (Priority: P1)

A production system requires numerical stability, graceful degradation, and comprehensive audit logs to maintain service reliability when processing diverse datasets with varying numerical properties.

**Why this priority**: Without reliability guarantees, the platform cannot be deployed in production environments. Numerical instability or silent failures would make results untrustworthy.

**Independent Test**: Can be tested by running log-determinant objective on ill-conditioned kernel matrices, verifying fallback to facility-location-only mode after 5 consecutive Cholesky failures, and confirming all decisions are logged to JSON audit trail.

**Acceptance Scenarios**:

1. **Given** a log-determinant objective encounters NaN during Cholesky update, **When** incremental update fails, **Then** system immediately refactorizes using full Cholesky and continues without user intervention
2. **Given** 5 consecutive numerical failures in log-determinant computation, **When** system detects repeated failures, **Then** system disables diversity term, switches to facility-location-only mode, logs degradation warning, and continues optimization
3. **Given** any optimization run, **When** run completes, **Then** system generates JSON Lines audit log with all mandatory fields (iteration, selected_element, marginal_gain, objective_value, algorithm, rng_seed, timestamp_ms)

---

### User Story 3 - Reproducible Results Across Parallelism (Priority: P2)

A researcher needs to reproduce exact results from a published experiment, regardless of whether they run with 1 thread or 64 threads, to validate research findings.

**Why this priority**: Reproducibility is critical for scientific validation and debugging. Without it, users cannot trust results or compare experiments across different hardware configurations.

**Independent Test**: Can be tested by running same dataset with seed=42 on 1 thread, 8 threads, and 64 threads, computing deterministic hash of selection sequence, and verifying all three hashes match exactly.

**Acceptance Scenarios**:

1. **Given** determinism config with seed=42 and fixed_order=true, **When** user runs lazy greedy 10 times in parallel, **Then** all 10 runs produce identical selection sequences with matching audit log hashes
2. **Given** same config run on 1 thread vs 64 threads, **When** both complete, **Then** objective values differ by less than 10^-6 (floating-point epsilon)
3. **Given** parallel marginal gain evaluation, **When** multiple candidates have equal gains, **Then** system breaks ties deterministically by ItemId (lowest ID wins)

---

### User Story 4 - Advanced Constraint Handling (Priority: P2)

An operations researcher needs to solve knapsack-constrained problems where each candidate has a cost and total budget is limited, choosing between fast heuristic mode and provably optimal mode based on requirements.

**Why this priority**: Knapsack constraints are common in real-world applications (budget allocation, resource-constrained selection). The dual-mode design allows users to trade speed for quality guarantees.

**Independent Test**: Can be tested by running knapsack problem in both Practical mode (fast, heuristic) and Theoretical mode (slow, (1-1/e) guarantee), comparing solution quality and runtime, and verifying Theoretical mode meets approximation bound.

**Acceptance Scenarios**:

1. **Given** knapsack constraint with budget B and cost function, **When** user selects Practical mode, **Then** system enumerates top-3 items, runs cost-benefit greedy, completes in O(nk) time
2. **Given** same problem with Theoretical mode, **When** optimization runs, **Then** system uses Continuous Greedy multilinear relaxation, achieves (1-1/e) approximation, completes with higher runtime
3. **Given** partition matroid constraint with per-category capacities, **When** selection violates any capacity, **Then** system rejects candidate before committing to oracle state

---

### User Story 5 - Real-Time Monitoring and Observability (Priority: P3)

A production operator needs to monitor optimization progress via Prometheus metrics and diagnose issues via structured audit logs when troubleshooting performance degradation or unexpected results.

**Why this priority**: Production observability is essential for operations but not blocking for initial research/development use. Can be added after core algorithms prove correct.

**Independent Test**: Can be tested by running optimization with Prometheus exporter enabled, querying metrics endpoint at /metrics, verifying presence of all required metrics (submod_gain_eval_total, submod_objective_value, submod_iteration_seconds), and parsing audit.jsonl for termination_reason field.

**Acceptance Scenarios**:

1. **Given** optimization in progress, **When** operator queries /metrics endpoint, **Then** system exposes all core counters (gain_eval_total, commit_total), gauges (objective_value, selection_size), and histograms (gain_compute_seconds, iteration_seconds)
2. **Given** optimization stops early, **When** operator checks audit log, **Then** log contains termination_reason field with semantic value (cardinality_reached, upper_bound_threshold, stagnation, timeout)
3. **Given** lazy greedy iteration, **When** multiple candidates are evaluated but not selected, **Then** audit log includes counterfactuals array with runner-up candidates, their marginal gains, deficits, and rejection reasons

---

### Edge Cases

- What happens when utility matrix has duplicate (i,s) pairs? System applies configurable duplicate handling: default is max utility, alternatives are sum, average, or error
- How does system handle NaN or Infinity values in input data? System rejects during fail-fast validation phase before optimization begins
- What if candidate IDs are not dense or zero-indexed? System requires dense zero-indexed IDs for bitset efficiency; user must preprocess data
- How does epoch-based lazy greedy handle heap exhaustion? When heap is empty mid-iteration, system terminates gracefully, logs stagnation reason, returns best solution found so far
- What if log-determinant encounters negative Schur complement? System applies epsilon clipping (d_safe = d.max(epsilon * 0.1)), if still negative triggers full Cholesky refactorization
- How does continuous greedy handle fractional solutions near 0 or 1? System applies explicit clipping: x[i] = min(x[i] + dt, 1.0) to prevent floating-point drift
- What if Parquet row groups are clustered incorrectly (not by demand ID)? Performance degrades due to random access, but correctness preserved; system logs warning if statistics indicate poor clustering

## Requirements *(mandatory)*

### Functional Requirements

**Core Optimization**

- **FR-001**: System MUST solve subset selection problems by selecting k items from n candidates to maximize a submodular objective function
- **FR-002**: System MUST support Lazy Greedy algorithm with epoch-based stale bound elimination achieving 5-10% evaluation rate vs naive greedy
- **FR-003**: System MUST support Stochastic Greedy algorithm with O(n log(1/epsilon)) complexity via random sampling
- **FR-004**: System MUST support Continuous Greedy for matroid constraints via multilinear extension relaxation in [0,1]^n domain
- **FR-005**: System MUST guarantee (1-1/e) approximation for monotone submodular functions under cardinality constraints

**Objective Functions**

- **FR-006**: System MUST implement Facility Location objective: sum over demands of weighted max utility
- **FR-007**: System MUST implement Saturating Coverage objective with concave saturation functions (log, sqrt, min)
- **FR-008**: System MUST implement Log-Determinant objective for diversity via Schur complement incremental update
- **FR-009**: System MUST support Diversity-Aware Facility Location combining coverage and pairwise similarity penalty

**Constraints**

- **FR-010**: System MUST support Cardinality constraint (|S| ≤ k) with O(1) feasibility check
- **FR-011**: System MUST support Knapsack constraint in two modes: Practical (fast heuristic) and Theoretical ((1-1/e) guarantee)
- **FR-012**: System MUST support Partition Matroid constraint with per-category capacity limits
- **FR-013**: System MUST support Graphic Matroid constraint for acyclic edge selection using Union-Find

**Data I/O**

- **FR-014**: System MUST load sparse utility matrices from Parquet with predicate pushdown and row-group pruning
- **FR-015**: System MUST construct CSR (Compressed Sparse Row) format for demand-centric access with O(nnz(e)) marginal gain complexity
- **FR-016**: System MUST validate input data: non-negativity, no NaN/Inf, ID bounds, schema conformance (fail-fast)
- **FR-017**: System MUST support hash-based sharding (128 shards) with parallel loading and prefetch

**Numerical Stability**

- **FR-018**: System MUST use f32 for Facility Location and Saturating Coverage objectives (memory efficiency, SIMD)
- **FR-019**: System MUST use f64 for Log-Determinant objective and Continuous Greedy gradient estimation (error accumulation)
- **FR-020**: System MUST apply 5-layer defense for log-determinant: regularization, safe Schur, log1p, refactorization, graceful degradation
- **FR-021**: System MUST gracefully degrade from log-determinant to facility-location-only after 5 consecutive numerical failures

**Deterministic Execution**

- **FR-022**: System MUST produce identical selection sequences for identical inputs when determinism config is enabled (seed, fixed_order, tie_breaking)
- **FR-023**: System MUST use hierarchical RNG seeding: master_seed ^ ALGO_TAG + iteration_num for platform-independent reproducibility
- **FR-024**: System MUST implement fixed-order parallel reduction: sort by ItemId after parallel evaluation when fixed_order=true
- **FR-025**: System MUST break marginal gain ties deterministically by ItemId (lowest ID wins) by default

**Monitoring**

- **FR-026**: System MUST expose Prometheus metrics at /metrics endpoint: counters, gauges, histograms with snake_case naming and unit suffixes
- **FR-027**: System MUST emit structured audit logs in JSON Lines format with mandatory fields (iteration, selected_element, marginal_gain, objective_value, algorithm, rng_seed, timestamp_ms)
- **FR-028**: System MUST log termination reason semantically (cardinality_reached, upper_bound_threshold, stagnation, timeout)
- **FR-029**: System MUST include counterfactuals in audit logs: runner-up candidates with marginal gains, deficits, rejection reasons

**Thread Safety**

- **FR-030**: System MUST implement immutable oracle interface: gain() takes &self and SelectionView, never &mut self
- **FR-031**: System MUST separate read operations (gain, upper_bound) from write operations (commit)
- **FR-032**: System MUST support fork() pattern: shared Arc<immutable_data> with thread-local mutable caches
- **FR-033**: System MUST enable parallel marginal gain evaluation via Rayon with par_iter() over candidates

**Termination Conditions**

- **FR-034**: System MUST support 4 termination conditions: cardinality reached, upper bound below threshold, consecutive stagnation, timeout
- **FR-035**: System MUST allow configurable stagnation detection: window size (default 3), relative epsilon (default 10^-6)
- **FR-036**: System MUST return best solution found when timeout expires (no failure on timeout)

### Non-Functional Requirements

**Performance**

- **NFR-001**: System MUST process 1M candidates, 1M demands, 100M non-zero utilities, k=250 in under 10 minutes on 32-core 128GB machine
- **NFR-002**: Lazy Greedy MUST achieve 5-10% evaluation rate compared to naive greedy (empirical target)
- **NFR-003**: Parallel evaluation MUST scale near-linearly up to 64 threads for embarassingly parallel workloads
- **NFR-004**: Memory usage MUST stay under 150GB for 100 billion non-zero utilities at 12 bytes per entry + overhead

**Scalability**

- **NFR-005**: System MUST support up to 4.3 billion candidates using u32 ItemId (configurable to u64 for larger scale)
- **NFR-006**: System MUST handle sparse matrices with arbitrary sparsity (1% to 50% non-zero density)
- **NFR-007**: Parquet row groups MUST be 128-256MB for parallelism/metadata balance (configurable via SUBMOD_ROW_GROUP_TARGET_MB)

**Reliability**

- **NFR-008**: Log-determinant MUST refactorize immediately on NaN detection (zero-tolerance for silent corruption)
- **NFR-009**: System MUST degrade gracefully from log-determinant to facility-location after 5 failures (service continuity)
- **NFR-010**: All constraint checks MUST use epsilon-tolerant comparison for floating-point budget/cost arithmetic

**Maintainability**

- **NFR-011**: Platform MUST be organized into specialized Rust crates with clear separation: core, objectives, solver, io, bindings-py, service
- **NFR-012**: All numerical thresholds MUST be logged with semantic names (lazy_epsilon, stagnation_threshold, ub_threshold)
- **NFR-013**: Configuration types MUST be strongly typed enums (Strategy, Constraint, KnapsackMode, TieBreak) not strings

**Testability**

- **NFR-014**: System MUST compute deterministic hash of selection sequence for regression testing (hash ItemId sequence + seed, not floats)
- **NFR-015**: All approximation guarantees MUST be empirically validated on small problems vs brute-force optimal
- **NFR-016**: Quick start example MUST run in under 1 second with 10KB dataset (50 candidates, 100 demands, 10% density)

### Key Entities

- **ItemId**: Element identifier in ground set V, represented as u32 (up to ~4.3B candidates) or u64 (extreme scale), must be dense and zero-indexed for bitset efficiency
- **SelectionView**: Immutable snapshot of current selection with FixedBitSet for O(1) membership testing and size counter, canonical representation during optimization
- **Selection**: Final output structure with Vec<ItemId> items, objective value, budget usage, partition counts, iteration trace
- **Demand**: Point to be covered/satisfied, has weight w_i and utility values u_{i,s} to each candidate
- **Utility Matrix**: Sparse (i,s,u) triplets stored in CSR format, row-major for demand-centric access, Parquet schema [i:u32, s:u32, u:f32]
- **Oracle State**: Mutable state maintained by objective function (e.g., best_u for facility location, cumulative for saturating coverage, Cholesky factor L for log-determinant)
- **HeapEntry**: Lazy greedy data structure with upper_bound, epoch timestamp, ItemId for stale bound elimination
- **Constraint State**: Mutable state per constraint type (used budget for knapsack, per-partition counts for partition matroid, union-find for graphic matroid)
- **Audit Log Entry**: JSON object per iteration with mandatory fields (iteration, selected_element, marginal_gain, objective_value, algorithm, rng_seed, timestamp_ms)
- **Metric**: Prometheus counter/gauge/histogram with snake_case name, unit suffix, algorithm label for observability

## Success Criteria *(mandatory)*

### Measurable Outcomes

**Correctness**

- **SC-001**: Lazy Greedy on monotone submodular functions MUST achieve objective value ≥ 63% of optimal (1-1/e lower bound) on small datasets validated against brute-force
- **SC-002**: Same input with same seed MUST produce bit-identical selection sequences across 10 runs (deterministic hash collision rate = 0%)
- **SC-003**: Log-determinant MUST maintain numerical stability: NaN rate < 0.1% on diverse kernel matrices, graceful degradation to facility-location after 5 failures

**Performance**

- **SC-004**: Large-scale problem (1M candidates, 1M demands, 100M non-zero utilities, k=250) MUST complete in under 10 minutes on 32-core 128GB machine
- **SC-005**: Lazy Greedy MUST evaluate only 5-10% of (n × k) marginal gains compared to naive greedy on typical datasets
- **SC-006**: Parallel evaluation with 64 threads MUST achieve ≥50x speedup compared to single-threaded for embarrassingly parallel workloads (fork() pattern)

**Reliability**

- **SC-007**: System MUST handle 100 billion non-zero utilities without out-of-memory errors (memory < 150GB)
- **SC-008**: Data validation MUST reject 100% of malformed inputs (NaN, Inf, out-of-bounds IDs) before optimization begins (fail-fast)
- **SC-009**: Numerical failures (negative Schur complement, NaN) MUST trigger refactorization or degradation in 100% of cases (zero silent failures)

**Reproducibility**

- **SC-010**: Same algorithm/seed/dataset run on 1 thread vs 64 threads MUST produce objective values differing by < 10^-6 (floating-point epsilon)
- **SC-011**: Selection sequence hash MUST be identical across platforms (Linux, macOS, Windows) for same seed and integer-based tie-breaking (aspirational cross-platform goal)

**Usability**

- **SC-012**: Quick start example (50 candidates, 100 demands, k=10) MUST complete in under 1 second and validate core pipeline end-to-end
- **SC-013**: Audit logs MUST include all mandatory fields (9 fields minimum) in 100% of iterations for production debugging

**Observability**

- **SC-014**: Prometheus /metrics endpoint MUST expose minimum 15 metrics (5 counters, 5 gauges, 5 histograms) covering evaluation count, objective value, iteration latency, I/O wait time
- **SC-015**: Termination reason MUST be logged semantically (not error codes) in 100% of runs for post-mortem analysis

## Scope

### In Scope

**Core Algorithms**
- Lazy Greedy with epoch-based heap and stale bound elimination
- Stochastic Greedy with reservoir sampling and configurable epsilon
- Continuous Greedy with view-based gradient estimation and maximum weight base selection
- Sieve-Streaming for single-pass streaming scenarios (future phase)

**Objective Functions**
- Facility Location (weighted max coverage)
- Saturating Coverage with concave saturation functions
- Log-Determinant with incremental Cholesky and 5-layer numerical safety
- Diversity-Aware Facility Location (coverage minus similarity penalty)

**Constraints**
- Cardinality (|S| ≤ k)
- Knapsack with Practical (fast heuristic) and Theoretical ((1-1/e)) modes
- Partition Matroid with per-category capacities
- Graphic Matroid with Union-Find cycle detection

**Data I/O**
- Parquet ingestion with predicate pushdown, row-group pruning, schema validation
- CSR/CSC sparse matrix construction with zero filtering
- Hash-based sharding (128 shards) with parallel loading
- Minimal example dataset (10KB) for quick start

**Determinism & Reliability**
- Hierarchical RNG seeding (master → algo → iteration)
- Fixed-order parallel reduction for reproducibility
- Deterministic tie-breaking by ItemId
- Graceful degradation (log-determinant → facility-location)
- Fail-fast input validation (non-negativity, no NaN/Inf, ID bounds)

**Monitoring & Observability**
- Prometheus metrics (counters, gauges, histograms) at /metrics
- JSON Lines audit logs with mandatory fields + counterfactuals
- Structured termination reasons (not error codes)
- Explainability: coverage report, residual demand, objective curve

**Platform Support**
- Rust implementation with crate organization (core, objectives, solver, io)
- Python bindings via PyO3 with GIL release and zero-copy Arrow/FFI
- CLI interface (clap)
- 32-core commodity hardware target (128GB RAM)

### Out of Scope (Explicitly Excluded)

**Algorithms Not Included**
- Double Greedy for non-monotone submodular functions (monotone-only in initial release)
- Non-streaming variants of Sieve-Streaming (deferred to future phase)
- Hybrid Lazy-Stochastic Greedy (mentioned as future optimization)

**Constraint Types Not Included**
- Matroid intersection constraints (only single matroid)
- General polymatroid constraints (only specific matroid types)
- Multi-objective optimization (single objective only)

**Objective Functions Not Included**
- Custom user-defined objective functions via plugins (framework only supports built-in objectives)
- Non-submodular objectives (no approximation guarantees)
- Time-varying or dynamic objectives (static optimization only)

**Data Formats Not Included**
- Real-time streaming ingestion (batch processing only via Parquet)
- SQL database direct integration (Parquet export required)
- Arrow IPC streaming format (Parquet files only)

**Service Features Not Included**
- gRPC and REST endpoints (Phase 5 only)
- Job queue management and scheduling (Phase 5 only)
- Authentication and authorization (Phase 5 only)
- Multi-tenancy support

**Platform Limitations**
- Cross-platform bitwise floating-point reproducibility (aspirational, not guaranteed due to FMA/architecture differences)
- Distributed computing across multiple machines (single-machine parallelism only)
- GPU acceleration (CPU-only via Rayon)
- Mobile/embedded platforms (commodity server hardware only)

**Explainability Beyond Scope**
- Interactive visualization dashboards (audit logs only)
- Counterfactual "what-if" simulation UI (data in logs but no UI)
- Automated hyperparameter tuning (manual config only)

**Scale Limitations**
- ItemId u32 limits to 4.3 billion candidates (u64 requires manual configuration)
- Memory-bound by sparse matrix size (no disk-based out-of-core computation)
- Parquet row group size fixed at 128-256MB (no adaptive sizing)

## Assumptions

1. **Hardware Assumptions**: Target deployment is 32-core commodity server with 128GB RAM and SSD storage; performance targets assume this baseline
2. **Data Preprocessing**: Input data is preprocessed to have dense zero-indexed ItemIds; system does not handle arbitrary string IDs or sparse ID spaces
3. **Network Reliability**: For Parquet sharding, assume reliable local filesystem or network storage with low latency (<10ms); high-latency remote storage degrades performance
4. **Numerical Stability**: Log-determinant assumes kernel matrices are positive semidefinite; indefinite matrices may fail gracefully but results are undefined
5. **Monotonicity Assumption**: Approximation guarantees assume objective functions are monotone submodular; non-monotone functions accepted but no quality bounds
6. **Single-Objective**: Only one objective function optimized per run; multi-objective requires external Pareto frontier search
7. **Batch Processing**: All data loaded upfront from Parquet; no real-time ingestion or dynamic dataset updates during optimization
8. **Rust Ecosystem**: Relies on stable Rust toolchain and crates (Rayon, Arrow, Parquet, fixedbitset); breaking changes in dependencies may require updates
9. **Determinism Trade-off**: Fixed-order parallel reduction adds ~10% overhead; users requiring maximum performance can disable via fixed_order=false
10. **Cross-Platform FP**: Bitwise floating-point reproducibility across platforms is aspirational; accept differences <10^-6 due to CPU architecture (FMA, rounding modes)
11. **Parquet Schema**: Utility matrix Parquet must have columns [i:u32, s:u32, u:f32]; other schemas require user conversion
12. **Concave Functions**: Saturating coverage assumes user-provided concave functions are correctly implemented; no runtime convexity validation
13. **Matroid Independence**: User-provided matroid implementations must correctly satisfy independence axioms (I1, I2, I3); incorrect implementations produce undefined behavior
14. **Cost Functions**: Knapsack cost functions must be non-negative and finite; negative or infinite costs cause constraint violations
15. **Licensing**: Apache-2.0 license assumed for open-source release; patent protection and commercial-friendly terms

## Dependencies

**External Crates (Rust)**
- **rayon**: Parallel iterators for data parallelism (thread pool management)
- **arrow / parquet**: Data ingestion, schema validation, predicate pushdown, row-group indexing
- **fixedbitset**: Compact bitset for SelectionView with O(1) membership testing
- **rand / rand_chacha**: Seeded RNG for deterministic sampling (StdRng for platform independence)
- **prometheus / hyper**: Metrics exposure at /metrics endpoint
- **serde / serde_json**: JSON Lines audit log serialization
- **pyo3**: Python bindings with GIL release and Arrow FFI
- **clap**: CLI argument parsing
- **tonic / axum**: gRPC and REST endpoints (Phase 5 only)
- **criterion**: Benchmarking with regression detection (test/CI only)
- **proptest**: Property-based testing for submodularity validation (test only)

**System Requirements**
- **Rust Toolchain**: Stable Rust 1.70+ with std::simd for SIMD vectorization
- **LLVM/Clang**: For compilation with SIMD intrinsics and optimization
- **Python 3.8+**: For PyO3 bindings and NumPy/Pandas integration
- **Parquet Tools**: For dataset generation (pyarrow, pandas in Python)

**Hardware Requirements**
- **CPU**: 32-core x86_64 with AVX2 (AVX-512 optional for 8-wide SIMD)
- **Memory**: 128GB RAM for large-scale problems (100B non-zero utilities)
- **Storage**: SSD with 500MB/s sequential read for Parquet loading
- **Network**: Optional, for distributed Parquet storage (not required for single-machine)

**Testing Infrastructure**
- **GitHub Actions**: CI/CD for test matrix (Ubuntu, macOS, Windows) × (stable, nightly Rust)
- **Criterion Benchmarks**: Baseline datasets for performance regression detection
- **Small Ground Truth Dataset**: Brute-force optimal solutions for approximation validation

**Documentation Tools**
- **mdBook**: User guide, API reference, tutorials (assumed for documentation phase)
- **rustdoc**: API documentation generation from inline comments

## Risks & Mitigations

### Technical Risks

**Risk: Numerical Instability in Log-Determinant**
- **Impact**: High - Silent corruption or optimization failure
- **Probability**: Medium - Ill-conditioned kernels common in practice
- **Mitigation**: 5-layer defense (regularization, safe Schur, log1p, refactorization, degradation to facility-location); f64 mandatory; graceful degradation after 5 failures; extensive testing on diverse kernel matrices

**Risk: Floating-Point Non-Determinism Across Platforms**
- **Impact**: Medium - Breaks reproducibility for cross-platform validation
- **Probability**: High - FMA, rounding modes differ across CPU architectures
- **Mitigation**: Hash only integer decisions (ItemId sequence), not floats; accept <10^-6 objective differences; document as aspirational goal; StdRng for platform-independent RNG

**Risk: Memory Exhaustion on Large Datasets**
- **Impact**: High - Out-of-memory crash, data loss
- **Probability**: Low - With proper sharding and memory estimation
- **Mitigation**: Memory estimate formula (12 bytes per non-zero + overhead); fail-fast validation on dataset load; sharding (128 shards) with bounded prefetch queue; optional TopK compression (retain top-50 utilities per candidate)

**Risk: Performance Degradation from Poor Parquet Clustering**
- **Impact**: Medium - 10-100x slowdown from random access
- **Probability**: Medium - Users may not understand row-group clustering requirements
- **Mitigation**: Document clustering requirements (by demand ID); validate statistics on load; log warning if clustering poor; provide example generation scripts

### Operational Risks

**Risk: Silent Correctness Violations from Malformed Input**
- **Impact**: Critical - Wrong optimization results, no detection
- **Probability**: Low - With fail-fast validation
- **Mitigation**: Mandatory validation (non-negativity, no NaN/Inf, ID bounds, schema conformance); reject before optimization begins; comprehensive error messages with file/line context

**Risk: Insufficient Observability for Production Debugging**
- **Impact**: Medium - Cannot diagnose performance issues or unexpected results
- **Probability**: Low - With comprehensive metrics and audit logs
- **Mitigation**: Prometheus metrics (15+ metrics covering all stages); JSON Lines audit logs with mandatory fields + counterfactuals; structured termination reasons; example Grafana dashboards (Phase 5)

**Risk: Lack of User Understanding of Approximation Guarantees**
- **Impact**: Medium - Users expect optimal but get 63% approximation
- **Probability**: High - Non-expert users unfamiliar with submodularity theory
- **Mitigation**: Explicit KnapsackMode enum (Practical vs Theoretical); documentation of approximation factors; validation scripts comparing to brute-force on small problems; theoretical gap bound in audit logs

### Schedule Risks

**Risk: Continuous Greedy Complexity Underestimated**
- **Impact**: Medium - Phase 4 delay (6-8 weeks → 10-12 weeks)
- **Probability**: Medium - Gradient estimation, rounding, matroid integration complex
- **Mitigation**: Phased delivery (Lazy Greedy first in Phase 1, Continuous Greedy deferred to Phase 4); extensive testing with partition matroid before general matroid; budget buffer in Phase 4 estimate

**Risk: Python Bindings GIL/FFI Issues**
- **Impact**: Medium - Phase 5 delay or reduced parallelism
- **Probability**: Low - PyO3 mature, GIL release well-documented
- **Mitigation**: Early prototype in Phase 2 (parallel evaluation test); Arrow FFI for zero-copy; consult PyO3 examples; allocate contingency time in Phase 5

### Dependency Risks

**Risk: Breaking Changes in Arrow/Parquet Crates**
- **Impact**: Low - Compilation failure, require code updates
- **Probability**: Low - Arrow crates stabilizing, major versions rare
- **Mitigation**: Pin dependency versions in Cargo.toml; monitor crate changelogs; CI runs on stable + nightly to catch breakage early; allocate maintenance buffer

**Risk: Rayon Performance Regressions**
- **Impact**: Low - Parallel speedup degrades
- **Probability**: Very Low - Rayon mature and stable
- **Mitigation**: Criterion benchmarks with regression detection; lock Rayon version after validation; fallback to manual thread pool if needed

## Success Metrics (KPIs)

### Correctness Metrics

- **Approximation Ratio**: ≥63% of optimal on small datasets (validated against brute-force), 100% of test cases pass
- **Determinism Hash Collision Rate**: 0% across 10 runs with same seed (bit-identical selections)
- **Numerical Stability**: NaN rate <0.1% on diverse kernel matrices, 100% graceful degradation after 5 failures
- **Data Validation Coverage**: 100% rejection of malformed inputs (NaN, Inf, out-of-bounds IDs) before optimization

### Performance Metrics

- **Large-Scale Completion Time**: <10 minutes for 1M×1M problem with 100M non-zero utilities on 32-core machine
- **Lazy Greedy Efficiency**: 5-10% evaluation rate vs naive greedy (empirical)
- **Parallel Speedup**: ≥50x with 64 threads for embarrassingly parallel workloads (fork pattern)
- **Memory Footprint**: <150GB for 100 billion non-zero utilities

### Reliability Metrics

- **Silent Failure Rate**: 0% (all numerical errors trigger refactorization or degradation)
- **OOM Failure Rate**: 0% for datasets within memory estimate (12 bytes per non-zero + overhead)
- **Termination Success Rate**: 100% (all runs complete or timeout gracefully, no crashes)

### Usability Metrics

- **Quick Start Time**: <1 second for 10KB example (50 candidates, 100 demands, k=10)
- **Documentation Completeness**: 100% of public APIs documented with rustdoc, user guide with 5+ examples
- **Error Message Clarity**: 90% of validation errors include file/line context and suggested fixes

### Observability Metrics

- **Prometheus Metric Coverage**: 15+ metrics (5 counters, 5 gauges, 5 histograms) covering all pipeline stages
- **Audit Log Completeness**: 100% of iterations include 9 mandatory fields
- **Termination Reason Coverage**: 100% of runs log semantic termination reason (not error codes)

### Adoption Metrics (Post-Release)

- **GitHub Stars**: Target 100 stars within 6 months of open-source release (indicator of community interest)
- **PyPI Downloads**: Target 500 downloads/month for Python bindings within 3 months of release
- **Issue Response Time**: 90% of GitHub issues triaged within 7 days (maintainer responsiveness)
- **Test Coverage**: ≥80% line coverage for core crates (submod-core, submod-solver, submod-objectives)

### Research Impact Metrics

- **Reproducibility**: 100% of published experiments reproducible with provided seed and dataset (audit hash matches)
- **Citation Count**: Target 5 citations in academic papers within 1 year of release (research adoption)
- **Benchmark Dataset Coverage**: 10+ public datasets with documented baseline results (community benchmarks)

## Open Questions

None - specification is complete with reasonable defaults and documented assumptions. Any remaining clarifications can be addressed during planning phase via `/speckit.clarify` if needed.
