# Tasks: Submoda Submodular Optimization Platform

**Feature**: Complete implementation of submoda platform based on `docs/specification.md`
**Total Tasks**: 265 tasks (updated: +30 total: 14 performance/explainability + 16 production-readiness)
**MVP Scope**: 50 tasks (Phase 1-3) → 5-6 weeks (includes SIMD, I/O, quickstart, CI)
**Input**: spec.md, plan.md, docs/specification.md, contracts/*.json

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in task descriptions
- Each task has: Files, Dependencies (Deps), Verification (Verify)

---

## Phase 1: Project Setup (4 tasks)

**Purpose**: Initialize Rust workspace and CI infrastructure
**Completion**: All 6 crates compile with `cargo build`

- [ ] **T001**: Create Cargo workspace with 6 crates (submod-core, submod-objectives, submod-solver, submod-io, submod-bindings-py, submod-service) | Files: Cargo.toml, */Cargo.toml | Deps: None | Verify: `cargo build --workspace` succeeds

- [ ] **T002**: Configure dependencies in each Cargo.toml (rayon, arrow, parquet, fixedbitset, pyo3, prometheus, anyhow, thiserror per specification.md §2.1) | Files: All Cargo.toml | Deps: T001 | Verify: `cargo check --workspace` succeeds

- [ ] **T003**: Setup GitHub Actions CI (Linux/macOS/Windows × stable/nightly, tests + clippy) | Files: .github/workflows/ci.yml | Deps: T001 | Verify: Push triggers successful CI run

- [ ] **T004**: Add project configuration files (.gitignore, LICENSE MIT/Apache-2.0, README.md, .rustfmt.toml, clippy.toml) | Files: Root config files | Deps: T001 | Verify: Files exist, follow Rust conventions

---

## Phase 2: Foundational Types and Traits (8 tasks) [BLOCKING]

**Purpose**: Core abstractions that ALL user stories depend on
**Completion**: All traits compile with rustdoc
**⚠️ CRITICAL**: No user story work can begin until this phase completes

- [ ] **T005** [P]: Define SubmodularOracle trait (universe_size, gain(&self), commit(&mut self), upper_bound, optional fork/prefetch/gain_batch) with Send+Sync bounds | Files: submod-core/src/oracle.rs | Deps: T001-T002 | Verify: Trait compiles, rustdoc generated

- [ ] **T006** [P]: Define Constraint trait (can_add, commit, reset) | Files: submod-core/src/constraint.rs | Deps: T001-T002 | Verify: Trait compiles, rustdoc with examples

- [ ] **T007** [P]: Implement SelectionView struct (in_set: FixedBitSet, size: usize, methods: new/contains/clone_with_element) | Files: submod-core/src/selection.rs | Deps: T001-T002 | Verify: Unit tests for O(1) membership, clone cost |V|/8 bytes

- [ ] **T008** [P]: Implement Selection struct (items: Vec<ItemId>, objective/used_budget: f64, counts_by_part, trace, termination_reason) | Files: submod-core/src/selection.rs | Deps: T001-T002 | Verify: All fields public, invariants documented

- [ ] **T009** [P]: Define ItemId type alias (pub type ItemId = u32) with dense zero-indexing documentation | Files: submod-core/src/types.rs | Deps: T001-T002 | Verify: Type exists, rustdoc explains 0 to 2^32-1 range

- [ ] **T010** [P]: Define Strategy enum (LazyGreedy{epsilon}, StochasticGreedy{epsilon, sample_factor}, ContinuousGreedy{steps, grad_samples, rounding}) | Files: submod-core/src/config.rs | Deps: T001-T002 | Verify: Enum compiles, rustdoc describes approximation guarantees

- [ ] **T011** [P]: Define Determinism config struct (seed: u64, fixed_order: bool, tie_breaking: TieBreak) and TieBreak enum (ById/ByUpperBound/Random) | Files: submod-core/src/config.rs | Deps: T001-T002 | Verify: Default impl (seed=42, fixed_order=true, tie_breaking=ById)

- [ ] **T012** [P]: Define TerminationConfig struct (max_iterations, ub_threshold: Option<f64>, stagnation_window/epsilon, timeout: Option<Duration>) | Files: submod-core/src/config.rs | Deps: T001-T002 | Verify: Struct compiles, rustdoc lists 4 termination reasons

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Large-Scale Subset Selection (40 tasks) [P1 - MVP] 🎯

**Goal**: Select 250 facilities from 1M candidates in <10 minutes (with SIMD, I/O, quickstart, CI, docs)
**Independent Test**: Load 1M×1M dataset, run lazy greedy, verify <10 min + ≥63% approximation + quickstart reproducible

### US1.1: FacilityLocation Oracle (submod-objectives)

- [ ] **T013** [P]: Implement FacilityLocation struct (demand_weights: Vec<f32>, utility_matrix: CsrMatrix<f32>, best_u: Vec<f32>) | Files: submod-objectives/src/facility_location.rs | Deps: T005, T007 | Verify: CSR layout enables O(nnz(e)) iteration

- [ ] **T014** [P]: Implement gain() with demand-centric loop (READ-ONLY &self, iterate nnz utilities, compute weighted improvements) | Files: submod-objectives/src/facility_location.rs | Deps: T013 | Verify: Unit test gain([]) = sum max utilities, diminishing returns

- [ ] **T015** [P]: Implement commit() updating best_u vector (WRITE &mut self, max(best_u[d], utility[e,d]) for all demands) | Files: submod-objectives/src/facility_location.rs | Deps: T013 | Verify: After commit(e), gain(e) = 0

- [ ] **T016** [P]: Implement upper_bound() returning cached gains (f64::INFINITY initially, lazy init for Lazy Greedy) | Files: submod-objectives/src/facility_location.rs | Deps: T013 | Verify: Initial bounds INFINITY, decrease after evaluation

- [ ] **T017** [P]: Add from_parquet() constructor (load demand_weights, utility_matrix, validate schema/non-negative/no NaN) | Files: submod-objectives/src/facility_location.rs | Deps: T013 | Verify: Load quickstart.parquet, verify universe_size

- [ ] **T017a** [P]: Implement gain_batch() for FacilityLocation (evaluate multiple candidates in one call, SIMD-friendly batch processing) | Files: submod-objectives/src/facility_location.rs | Deps: T014 | Verify: Batch evaluation returns same results as sequential gain() calls

- [ ] **T017b** [P]: Add SIMD optimization to marginal gain inner loop (std::simd with 4-8 lane processing for utility calculations) | Files: submod-objectives/src/facility_location.rs | Deps: T014, add std::simd | Verify: Flamegraph shows vectorized operations, 3-4x speedup on AVX2 CPUs

- [ ] **T017c**: Benchmark gain() vs gain_batch() performance (compare throughput on 1K-10K candidates with varying sparsity) | Files: benches/oracle_batch.rs | Deps: T017a-T017b | Verify: gain_batch() achieves 4-8x speedup over sequential gain()

- [ ] **T018** [P]: Add unit tests for monotonicity and submodularity (property tests: f(S) ≤ f(T) for S⊆T, diminishing returns) | Files: submod-objectives/src/facility_location.rs | Deps: T013-T015 | Verify: 100 random test cases pass

### US1.2: Lazy Greedy Algorithm (submod-solver)

- [ ] **T019**: Define HeapEntry struct (bound, epoch: u64, item_id) with Ord by (bound desc, item_id asc) | Files: submod-solver/src/lazy_greedy.rs | Deps: T009 | Verify: Unit test heap orders correctly, ties by ItemId

- [ ] **T020**: Implement heap initialization (BinaryHeap with oracle.upper_bound(e) for all e, epoch=0) | Files: submod-solver/src/lazy_greedy.rs | Deps: T019, T005 | Verify: Heap size = universe_size after init

- [ ] **T021**: Implement main loop (pop, check epoch, evaluate gain, commit or re-insert with current_epoch) | Files: submod-solver/src/lazy_greedy.rs | Deps: T019-T020, T005-T007 | Verify: Stale entries discarded, candidates re-inserted

- [ ] **T022**: Add termination logic (stop at |S|=max_iterations, set termination_reason="cardinality_reached") | Files: submod-solver/src/lazy_greedy.rs | Deps: T021, T012 | Verify: Stops exactly at k selections

- [ ] **T023**: Add iteration trace logging (record IterationTrace: iteration, selected_item, marginal_gain, objective_value, upper_bound_max, evaluations, heap_pops) | Files: submod-solver/src/lazy_greedy.rs | Deps: T021, T008 | Verify: trace.len() = k after k selections

- [ ] **T024**: Unit tests for epoch-based stale elimination (verify current_epoch increments after commit, old entries discarded) | Files: submod-solver/src/lazy_greedy.rs | Deps: T019-T023 | Verify: Insert entry, commit different item, verify old entry discarded

### US1.3: Parquet I/O (submod-io)

- [ ] **T025** [P]: Implement Parquet schema validation (check candidate_id: UInt32, demand_id: UInt32, utility: Float32, demand_weight: Float32) | Files: submod-io/src/parquet.rs | Deps: T001-T002 (arrow/parquet deps) | Verify: Valid schema passes, invalid returns errors

- [ ] **T026** [P]: Implement row-group clustering validation (verify demand_id monotonic within row groups) | Files: submod-io/src/parquet.rs | Deps: T025 | Verify: Clustered file passes, unclustered logs warning

- [ ] **T027** [P]: Implement load_facility_location() (read Parquet, validate schema, construct CSR matrix from tuples, load demand_weights) | Files: submod-io/src/parquet.rs | Deps: T013, T025-T026 | Verify: Load 10K×50K file, verify CSR dimensions

- [ ] **T027a** [P]: Implement hash-based sharding (partition utility matrix into ~128 shards by demand_id hash, distribute uniformly) | Files: submod-io/src/sharding.rs | Deps: T027 | Verify: 1M demands distributed uniformly across 128 shards (χ² test p>0.05)

- [ ] **T027b** [P]: Implement async prefetch strategy (bounded channel with parallelism×2 capacity, load shards i+1, i+2 while processing shard i) | Files: submod-io/src/prefetch.rs | Deps: T027a, add tokio | Verify: I/O wait time <5% of total runtime on 10+ shard dataset

- [ ] **T027c** [P]: Add row-group size configuration via SUBMOD_ROW_GROUP_TARGET_MB env var (default 192MB, validate range 64-512MB) | Files: submod-io/src/parquet.rs | Deps: T027 | Verify: Env var changes row group size, value logged at startup

- [ ] **T027d**: Integration test with sharded 1B+ non-zeros (generate synthetic 1M×1M×0.001 density, verify prefetch works, no memory explosion) | Files: submod-io/tests/large_scale_sharded.rs | Deps: T027a-T027c | Verify: Loads 1B non-zeros in <5 minutes, peak memory <16GB

- [ ] **T028** [P]: Add error handling for schema mismatches (anyhow::Context for missing columns, type mismatches, NaN/Inf) | Files: submod-io/src/parquet.rs | Deps: T027 | Verify: Malformed files produce clear error messages

- [ ] **T029**: Integration test with synthetic 10K×50K dataset (generate, save to Parquet, load, run lazy greedy) | Files: submod-io/tests/integration.rs | Deps: T027, T013-T024 | Verify: Test passes in <1 second

### US1.4: CardinalityConstraint (submod-core)

- [ ] **T030** [P]: Implement CardinalityConstraint struct (max_size, current_size: usize) | Files: submod-core/src/constraints/cardinality.rs | Deps: T006-T007 | Verify: Struct compiles with Constraint trait

- [ ] **T031** [P]: Implement can_add() checking size < max_size (check against view, not internal state) | Files: submod-core/src/constraints/cardinality.rs | Deps: T030 | Verify: Returns true when size<max, false when size=max

- [ ] **T032** [P]: Implement commit() incrementing counter | Files: submod-core/src/constraints/cardinality.rs | Deps: T030 | Verify: current_size increases after each commit

- [ ] **T033**: Unit tests for boundary conditions (k=0, k=1, k=universe_size) | Files: submod-core/src/constraints/cardinality.rs | Deps: T030-T032 | Verify: All edge cases handled correctly

### US1.5: Integration and Performance Validation

- [ ] **T034**: End-to-end test with 50K×100K dataset (run lazy greedy k=250, verify <1 second) | Files: tests/integration/us1_mvp.rs | Deps: T013-T033 | Verify: Runtime <1s, selection size=250

- [ ] **T035**: Benchmark suite with Criterion (1M×1M, 100M non-zeros, k=250, target <10 min) | Files: benches/large_scale.rs | Deps: T013-T033 | Verify: Benchmark completes, report runtime

- [ ] **T036**: Profile and optimize hot paths if needed (flamegraph/perf, optimize SIMD for gain(), cache prefetching, reduce allocations) | Files: submod-objectives/src/facility_location.rs, submod-solver/src/lazy_greedy.rs | Deps: T035 | Verify: Runtime <10 minutes for 1M×1M

- [ ] **T037**: Verify ≥63% approximation ratio (small problem 50×100, k=5, brute force optimal comparison) | Files: tests/integration/approximation_ratio.rs | Deps: T013-T024 | Verify: Ratio ≥0.63 (actual ≥0.66 typically)

### US1.6: Quickstart & CI Enhancements [CRITICAL - MVP]

- [ ] **T037d** [P]: Create quickstart dataset (10KB, 100 demands, 50 candidates, 10% density per spec §11.1) | Files: data/quickstart_demand.parquet, data/quickstart_utility.parquet, scripts/generate_quickstart.py | Deps: T027 | Verify: Dataset is 10KB total, loads successfully, 500 non-zeros

- [ ] **T037e**: Write quickstart.sh with exact commands and expected output (cargo run with seed=42, objective ~45.2±0.1, selection deterministic) | Files: quickstart.sh, docs/quickstart.md | Deps: T037d, T013-T037 | Verify: Script runs in <1 second, outputs hash matching spec §11.2

- [ ] **T037f**: Verify quickstart deterministic hash (run twice with seed=42, verify identical audit log hash) | Files: tests/integration/quickstart_deterministic.rs | Deps: T037e, T058 | Verify: Two runs produce identical hash, documented in quickstart.md

- [ ] **T037g**: Add determinism CI job (GitHub Actions: run on 1/8/64 threads, verify all audit log hashes match) | Files: .github/workflows/determinism.yml | Deps: T037f, T003 | Verify: CI passes with 3 identical hashes across thread counts

- [ ] **T037h**: Add performance regression CI (compare with baseline from T035, alert if >5% degradation) | Files: .github/workflows/perf-regression.yml | Deps: T035, T003 | Verify: CI tracks runtime, fails if regression >5%

- [ ] **T037i**: Add cross-platform hash comparison (log hashes from Linux/macOS/Windows, manual review for now) | Files: .github/workflows/ci.yml (extend) | Deps: T037g, T003 | Verify: Hashes logged per platform, aspirational equality per spec §9.6

### US1.7: MVP Documentation [CRITICAL]

- [ ] **T037j**: Write MVP quickstart guide Rust-only (FacilityLocation + LazyGreedy, uses quickstart dataset from T037d) | Files: docs/quickstart_mvp.md | Deps: T037e, T013-T037 | Verify: Guide follows spec §11.1-11.3 format, runnable example

- [ ] **T037k**: Generate rustdoc for Phase 1-3 crates (core/objectives/solver/io, verify all public items documented) | Files: Run `cargo doc`, fix warnings | Deps: T005-T037 | Verify: `cargo doc --no-deps` produces no warnings, all traits/structs have docs

**Checkpoint**: MVP complete - User Story 1 fully functional and testable independently

---

## Phase 4: User Story 2 - Production-Grade Reliability (32 tasks) [P1]

**Goal**: Numerical stability, error handling, audit trails, epsilon arithmetic, resource limits, DoS protection
**Independent Test**: Log-determinant on ill-conditioned matrices completes without panic, DoS scenarios rejected, audit logs validate

### US2.1: Log-Determinant Oracle (submod-objectives) - 5-Layer Defense

- [ ] **T038** [P]: Implement LogDeterminant struct (vectors: DMatrix<f64>, lambda: f64, inverse: DMatrix<f64>, failure_count, refactorization_needed) | Files: submod-objectives/src/log_determinant.rs | Deps: T005, add nalgebra | Verify: Uses f64 (mandatory)

- [ ] **T039** [P]: Layer 1 - λI regularization in constructor (validate λ>0, default 1e-6, add to initial covariance) | Files: submod-objectives/src/log_determinant.rs | Deps: T038 | Verify: Matrix with λ=1e-6 is positive definite

- [ ] **T040** [P]: Layer 2 - Safe Schur complement (check c=v^T A^{-1}v ≤ 0, return 0.0 if non-positive) | Files: submod-objectives/src/log_determinant.rs | Deps: T038 | Verify: Linearly dependent vectors return 0.0, no panic

- [ ] **T041** [P]: Layer 3 - log1p transformation (use log1p(c) instead of log(1+c) for numerical accuracy) | Files: submod-objectives/src/log_determinant.rs | Deps: T040 | Verify: Compare log1p vs log(1+x) for x=1e-10, verify precision

- [ ] **T042** [P]: Layer 4 - Refactorization trigger (track failure_count, after 5 consecutive failures recompute full inverse) | Files: submod-objectives/src/log_determinant.rs | Deps: T040 | Verify: After 5 failures, inverse refactored

- [ ] **T043** [P]: Layer 5 - Graceful degradation (if refactorization fails, return 0.0 for all remaining, terminate with stagnation) | Files: submod-objectives/src/log_determinant.rs | Deps: T042 | Verify: Rank-deficient matrix returns 0.0 without panic

- [ ] **T044** [P]: Integrate all 5 layers in gain() | Files: submod-objectives/src/log_determinant.rs | Deps: T039-T043 | Verify: Ill-conditioned matrix (cond >10^12) completes without panic

- [ ] **T045** [P]: Implement commit() with Woodbury identity (A^{-1} ← A^{-1} - (A^{-1}vv^T A^{-1})/(1 + v^T A^{-1}v)) | Files: submod-objectives/src/log_determinant.rs | Deps: T044 | Verify: After k commits, A * A^{-1} ≈ I

- [ ] **T046**: Unit tests with ill-conditioned matrices (condition numbers 10^6, 10^9, 10^12, 10^15) | Files: submod-objectives/src/log_determinant.rs | Deps: T038-T045 | Verify: All tests pass, layers trigger appropriately

- [ ] **T047**: Property test for refactorization (proptest, verify refactorization after exactly 5 consecutive zero gains) | Files: submod-objectives/src/log_determinant.rs | Deps: T042-T045 | Verify: 100 random cases pass

### US2.2: Error Handling (submod-solver)

- [ ] **T048** [P]: Define SolverError enum (OracleError, ConstraintError, NumericalError, ConfigurationError, TimeoutError with context/source fields) | Files: submod-solver/src/error.rs | Deps: T001, add thiserror | Verify: Enum compiles, impl std::error::Error

- [ ] **T049** [P]: Implement context chain with anyhow (From<anyhow::Error>, enable .context() chaining) | Files: submod-solver/src/error.rs | Deps: T048, add anyhow | Verify: Error chain displays full context stack

- [ ] **T050** [P]: Add NaN/Inf detection in marginal gains (check gain.is_nan() || gain.is_infinite(), return NumericalError) | Files: submod-solver/src/lazy_greedy.rs | Deps: T021, T048 | Verify: Oracle returning NaN triggers NumericalError

- [ ] **T051** [P]: Add validation checks in solve() (validate epsilon>0, max_iterations>0, universe_size>0, initial feasibility) | Files: submod-solver/src/lib.rs | Deps: T048 | Verify: Each invalid config returns ConfigurationError

- [ ] **T051a** [P]: Implement epsilon-aware constraint arithmetic for Knapsack (use epsilon=1e-9, check used + epsilon*budget.abs() >= budget per spec §8.4) | Files: submod-core/src/constraints/knapsack.rs | Deps: T051 | Verify: Boundary comparison uses epsilon, no exact float equality

- [ ] **T051b**: Test boundary conditions for constraint arithmetic (budget ± 1e-9, exactly at limit, verify epsilon prevents false positives/negatives) | Files: submod-core/tests/constraint_arithmetic.rs | Deps: T051a | Verify: Tests pass for edge cases within epsilon of boundary

- [ ] **T051c**: Document floating-point comparison policy (docs/numerics.md section on constraint arithmetic, epsilon values for different precision) | Files: docs/numerics.md | Deps: T051a-T051b | Verify: Policy documented with examples from spec §8.4

- [ ] **T052**: Unit tests for each error variant (test all 5 error types with realistic scenarios) | Files: submod-solver/src/error.rs | Deps: T048-T051 | Verify: All error variants tested, messages include file:line

### US2.3: Audit Logging (submod-io)

- [ ] **T053** [P]: Define IterationTrace struct (9 mandatory fields: iteration, selected_item, marginal_gain, objective_value, upper_bound_max, evaluations, heap_pops, timestamp_ns, termination_reason: Option<String>) | Files: submod-io/src/audit.rs | Deps: T009 | Verify: All fields public

- [ ] **T054** [P]: Implement JSON Lines writer (BufWriter, newline-delimited JSON, flush after each iteration) | Files: submod-io/src/audit.rs | Deps: T053, add serde_json | Verify: Write 10 traces, verify 10 lines of valid JSON

- [ ] **T055** [P]: Add termination_reason to Selection (String field: cardinality_reached/upper_bound_threshold/stagnation/timeout) | Files: submod-core/src/selection.rs | Deps: T008 | Verify: Field documented with 4 values

- [ ] **T056** [P]: Add counterfactuals logging (top-3 runner-ups per iteration with gains) | Files: submod-io/src/audit.rs | Deps: T053 | Verify: Audit log includes counterfactuals array with 3 entries

- [ ] **T057**: Integration test for audit log schema (run optimization with logging, parse output, validate fields) | Files: submod-io/tests/audit_schema.rs | Deps: T053-T056, T019-T024 | Verify: Parse 100 iterations, all fields valid

- [ ] **T058**: Add audit log hash computation (deterministic hash from (iteration, selected_item) pairs only, exclude floats) | Files: submod-io/src/audit.rs | Deps: T053 | Verify: Same integer selections produce identical hash

### US2.4: Integration and Reliability Validation

- [ ] **T059**: End-to-end test with log-determinant on ill-conditioned data (condition number 10^15, verify no panic) | Files: tests/integration/us2_reliability.rs | Deps: T038-T047, T019-T024 | Verify: Selection size >0, no errors

- [ ] **T060**: Verify graceful degradation (rank-deficient matrix rank<k, verify stagnation after exhausting row space) | Files: tests/integration/us2_reliability.rs | Deps: T043, T059 | Verify: Terminates gracefully, termination_reason="stagnation"

- [ ] **T061**: Verify audit log completeness for failed runs (inject oracle error mid-optimization, verify log has all iterations up to failure) | Files: tests/integration/us2_reliability.rs | Deps: T053-T058 | Verify: Audit log has N entries where N=iterations before error

- [ ] **T062**: Load test with 100 consecutive runs (monitor memory usage, verify no leaks) | Files: tests/integration/us2_reliability.rs | Deps: T013-T047 | Verify: Memory stable across 100 runs

### US2.5: Resource Limits & DoS Protection [IMPORTANT]

- [ ] **T062a** [P]: Add memory limit validation (estimate memory from n×m×density, reject if estimated_size > max_memory config) | Files: submod-io/src/parquet.rs | Deps: T027 | Verify: Loading 10GB dataset with 1GB limit fails with clear error

- [ ] **T062b** [P]: Add resource limits to SolverConfig (max_memory: Option<usize>, max_walltime: Option<Duration>, default None=unlimited) | Files: submod-solver/src/config.rs | Deps: T066, T012 | Verify: Config compiles, limits documented as optional

- [ ] **T062c**: Test DoS scenarios (k=10^9 fails validation, 10GB Parquet exceeds memory limit, timeout after 1s works) | Files: tests/integration/dos_protection.rs | Deps: T062a-T062b | Verify: All 3 attack vectors rejected with appropriate errors

- [ ] **T062d**: Update MVP quickstart with LogDeterminant example (add ill-conditioned matrix test, show 5-layer defense) | Files: docs/quickstart_mvp.md | Deps: T037j, T038-T047 | Verify: Example demonstrates graceful degradation

**Checkpoint**: Production-ready reliability - US1 and US2 both work independently

---

## Phase 5: User Story 3 - Reproducible Results (21 tasks) [P2]

**Goal**: Identical results across 1-64 threads with same seed, comprehensive determinism guide
**Independent Test**: Audit log hashes match for runs with seed=42 on 1/8/64 threads, guide examples reproducible

### US3.1-3.5: Determinism Infrastructure

- [ ] **T063** [P]: Implement hierarchical RNG seeding function (derive_iteration_seed(master_seed, algo_tag, iteration) = master_seed ^ (algo_tag + iteration)) | Files: submod-core/src/determinism.rs | Deps: T011 | Verify: Same inputs produce same seed

- [ ] **T064** [P]: Define algorithm seed constants (LAZY_GREEDY_TAG=0x01, STOCHASTIC_GREEDY_TAG=0x02, CONTINUOUS_GREEDY_TAG=0x03) | Files: submod-core/src/determinism.rs | Deps: T010 | Verify: Constants defined, documented

- [ ] **T065** [P]: Implement seed derivation in Lazy Greedy (derive iteration_seed from master_seed + LAZY_GREEDY_TAG + iteration_num) | Files: submod-solver/src/lazy_greedy.rs | Deps: T063-T064, T021 | Verify: Two runs with same seed produce identical selections

- [ ] **T066** [P]: Add Determinism to SolverConfig (determinism: Determinism field with Default trait) | Files: submod-solver/src/config.rs | Deps: T011, T010 | Verify: Config compiles, default uses seed=42, fixed_order=true, tie_breaking=ById

- [ ] **T067**: Unit tests for seed derivation consistency (100 random test cases verify consistency) | Files: submod-core/src/determinism.rs | Deps: T063-T065 | Verify: Tests pass

- [ ] **T068** [P]: Implement ById tie-breaking in HeapEntry Ord (primary: bound desc, secondary: item_id asc) | Files: submod-solver/src/lazy_greedy.rs | Deps: T019 | Verify: Equal bounds break tie by lower ItemId

- [ ] **T069** [P]: Implement ByUpperBound tie-breaking (if gains equal, prefer higher cached upper_bound) | Files: submod-solver/src/lazy_greedy.rs | Deps: T019, T011 | Verify: Equal gains prefer higher upper bound

- [ ] **T070** [P]: Implement Random tie-breaking (seeded RNG for reproducible tie-breaks) | Files: submod-solver/src/lazy_greedy.rs | Deps: T063-T065, T011 | Verify: Same seed produces same tie-break decisions

- [ ] **T071**: Unit tests for tie-breaking (synthetic equal-gain candidates, verify all 3 modes) | Files: submod-solver/src/lazy_greedy.rs | Deps: T068-T070 | Verify: All modes tested

- [ ] **T072** [P]: Implement parallel evaluation with Rayon (candidates.par_iter().map(|&e| oracle.gain(&view, e)).collect()) | Files: submod-solver/src/lazy_greedy.rs | Deps: T014, add rayon | Verify: 6-8x speedup on 8 cores

- [ ] **T073** [P]: Add conditional sort-by-ItemId before reduction (if fixed_order==true, sort by ItemId then find max) | Files: submod-solver/src/lazy_greedy.rs | Deps: T072, T066 | Verify: fixed_order=true produces identical max across runs

- [ ] **T074**: Determinism validation test (1-thread vs 8-thread with fixed_order=true, verify selections identical) | Files: tests/integration/us3_determinism.rs | Deps: T072-T073, T013-T024 | Verify: Selections match, audit log hashes match

- [ ] **T075**: Measure ~10% overhead for fixed_order=true (benchmark comparison) | Files: benches/determinism_overhead.rs | Deps: T073 | Verify: Overhead documented, typically 8-12%

- [ ] **T076** [P]: Implement reservoir sampling with iteration-seeded RNG for Stochastic Greedy (sample R_t with |R_t|=ceil((n/k)log(1/ε)) using seeded RNG) | Files: submod-solver/src/stochastic_greedy.rs | Deps: T063-T064, T005-T006 | Verify: Same seed produces same samples

- [ ] **T077**: Determinism validation for stochastic samples (run twice with same seed, verify sampled sets R_t identical) | Files: tests/integration/us3_determinism.rs | Deps: T076 | Verify: Samples match exactly

- [ ] **T078**: Unit tests for sample reproducibility (edge cases: small samples, boundary iterations) | Files: submod-solver/src/stochastic_greedy.rs | Deps: T076-T077 | Verify: All edge cases produce reproducible samples

- [ ] **T079**: End-to-end test with seed=42 on 1/8/64 threads (facility location k=100) | Files: tests/integration/us3_determinism.rs | Deps: T065, T073, T013-T024 | Verify: All 3 runs produce identical selections

- [ ] **T080**: Compute audit log hash for each run (hash from audit log, integer decisions only) | Files: tests/integration/us3_determinism.rs | Deps: T058, T079 | Verify: Hash excludes float fields

- [ ] **T081**: Verify all hashes identical (assert 1/8/64 thread hashes are bitwise identical) | Files: tests/integration/us3_determinism.rs | Deps: T079-T080 | Verify: Test passes, all 3 hashes match

- [ ] **T082**: Document determinism guarantees (docs/determinism.md: hierarchical seeding, tie-breaking modes, fixed-order overhead, limitations) | Files: docs/determinism.md | Deps: T063-T081 | Verify: Documentation complete with examples

- [ ] **T082a**: Write determinism guide with concrete examples (seed=42 walkthrough, comparison table of tie-breaking modes, troubleshooting section) | Files: docs/determinism_guide.md | Deps: T082 | Verify: Guide includes 3+ runnable examples, cross-referenced from quickstart

**Checkpoint**: Determinism verified - US1, US2, US3 all work independently

---

## Phase 6: User Story 4 - Advanced Constraints (36 tasks) [P2]

**Goal**: Knapsack, partition matroid, graphic matroid + Continuous Greedy with view-based safety
**Independent Test**: Continuous Greedy on matroid achieves ≥60% of optimal

### US4.1-4.6: Advanced Constraints (Condensed)

- [ ] **T083-T088**: KnapsackConstraint (struct with mode enum Practical/Theoretical, can_add() implementations, commit(), unit tests comparing modes, integration test measuring quality/runtime tradeoff) | Files: submod-core/src/constraints/knapsack.rs, tests/integration/knapsack_modes.rs | Deps: T006, T009 | Verify: Both modes work, Practical ≥ Theoretical in objective

- [ ] **T089-T093**: Matroid trait (trait Matroid: Constraint, rank() method, max_weight_base() method, document greedy algorithm, specify deterministic tie-breaking by ItemId) | Files: submod-core/src/matroid.rs | Deps: T006 | Verify: Trait compiles, rustdoc explains matroid axioms

- [ ] **T094-T099**: PartitionMatroid (struct with part_assignment/limits/counts, can_add(), rank(), max_weight_base() with per-partition greedy, unit tests for independence axioms, integration test with imbalanced partitions) | Files: submod-core/src/constraints/partition_matroid.rs, tests/integration/partition_matroid.rs | Deps: T089, T009 | Verify: Returns BASE with rank elements, respects partition limits

- [ ] **T100-T105**: GraphicMatroid (struct with union-find, can_add() checking acyclicity, rank() = |V|-1, max_weight_base() with Kruskal's algorithm, unit tests for forest maintenance, integration test with dense graph K_100) | Files: submod-core/src/constraints/graphic_matroid.rs, tests/integration/graphic_matroid.rs | Deps: T089, add union-find | Verify: Returns maximum weight spanning tree

- [ ] **T106**: Define ContinuousGreedyConfig struct (steps: usize, grad_samples: usize, rounding: RoundingMode enum, use_common_random_numbers: bool) | Files: submod-solver/src/continuous_greedy.rs | Deps: T089 | Verify: Config compiles with sensible defaults (steps=100, grad_samples=500)

- [ ] **T107**: Implement estimate_gradient() with VIEW-BASED evaluation (create temporary SelectionView per sample, iterate without mutating oracle) | Files: submod-solver/src/continuous_greedy.rs | Deps: T106, T005, T007 | Verify: Unit test confirms oracle.best_u unchanged after gradient estimation

- [ ] **T107a**: Add anti-pattern test for oracle mutation (demonstrate why mutating oracle during gradient estimation breaks correctness) | Files: submod-solver/tests/continuous_greedy_safety.rs | Deps: T107 | Verify: Test shows incorrect results when oracle is mutated

- [ ] **T108**: Implement direction selection via matroid.max_weight_base() (select BASE not just independent set, using greedy algorithm) | Files: submod-solver/src/continuous_greedy.rs | Deps: T107, T089 | Verify: Selected set has size = matroid.rank(), satisfies independence

- [ ] **T109**: Implement fractional solution update (x[e] ← min(x[e] + 1/T, 1.0) for e in base, clip to [0,1]) | Files: submod-solver/src/continuous_greedy.rs | Deps: T108 | Verify: After T steps, sum(x) ≈ matroid.rank() ± 1

- [ ] **T110**: Implement Pipage rounding for partition matroids (deterministic pairwise adjustments until integral solution) | Files: submod-solver/src/rounding/pipage.rs | Deps: T109, T094 | Verify: Produces integral solution, maintains partition constraints, deterministic

- [ ] **T111**: Implement Swap rounding for general matroids (probabilistic base decomposition + sampling with seeded RNG) | Files: submod-solver/src/rounding/swap.rs | Deps: T109, T063 | Verify: Same seed produces same rounding, respects matroid constraints

- [ ] **T111a**: Add common random numbers for variance reduction (optional, reuse sample sets R_1..R_G across gradient elements) | Files: submod-solver/src/continuous_greedy.rs | Deps: T107 | Verify: Gradient variance 2-3x lower with use_common_random_numbers=true

- [ ] **T112**: Integration test verifying (1-1/e) approximation (small problem with known optimum, compare fractional vs rounded solution) | Files: submod-solver/tests/continuous_greedy_approximation.rs | Deps: T106-T111 | Verify: Achieves ≥63% of fractional optimum on 10 test instances

- [ ] **T113-T116**: Integration tests for all constraint types (knapsack both modes, partition matroid with 5 partitions, graphic matroid spanning tree, continuous greedy on matroid ≥60% of optimal) | Files: tests/integration/us4_constraints.rs | Deps: T083-T112 | Verify: All constraint types work correctly

**Checkpoint**: Advanced constraints complete - US1-US4 all work independently

---

## Phase 7: User Story 5 - Monitoring (39 tasks) [P3]

**Goal**: Prometheus metrics, HTTP /metrics endpoint, explainability, baselines, alerting, <0.1% overhead
**Independent Test**: Query /metrics endpoint, verify all 15+ metrics present, alerts fire on violations, coverage reports generated

### US5.1-5.5: Monitoring Infrastructure (Condensed)

- [ ] **T117-T121**: Metrics Infrastructure (MetricsConfig struct, prometheus crate dependency, 15+ metric descriptors, lazy_static! registry, unit tests) | Files: submod-core/src/metrics.rs | Deps: T001 | Verify: Metrics registered

- [ ] **T122-T131**: Solver Instrumentation (8 metrics: gain_eval_total, commit_total, heap_pop_total, objective_value, selection_size, upper_bound_max, gain_compute_seconds, iteration_seconds, conditional updates, verify <0.1% overhead) | Files: submod-solver/src/lazy_greedy.rs | Deps: T117-T121 | Verify: Metrics increment correctly, overhead <0.1%

- [ ] **T132-T137**: HTTP Metrics Endpoint (axum/tokio dependencies, /metrics endpoint with Prometheus text format, graceful shutdown on SIGTERM, unit and integration tests) | Files: submod-service/src/metrics.rs, tests/ | Deps: T117-T131 | Verify: Endpoint returns metrics

- [ ] **T138-T147**: Audit Log Enhancements + Validation (add termination_reason/counterfactuals to audit logs, end-to-end validation tests: metrics enabled, query endpoint, verify all metrics present, compare runtime with/without metrics <0.1% difference) | Files: submod-io/src/audit.rs, tests/integration/us5_monitoring.rs | Deps: T053-T058, T117-T137 | Verify: All enhancements work, overhead acceptable

- [ ] **T147a** [P]: Implement coverage report generator (for each selected element, list demand IDs where it provided best utility) | Files: submod-io/src/explainability.rs | Deps: T053, T013 | Verify: Report shows demands covered by each selection with contribution scores

- [ ] **T147b** [P]: Implement residual demand analysis (identify demands with max_utility < threshold, list as uncovered) | Files: submod-io/src/explainability.rs | Deps: T147a | Verify: Lists demands with current best utility below 0.5×max, sorted by deficit

- [ ] **T147c** [P]: Implement theoretical gap bound reporting (compute max Δ(e|S) over remaining candidates per iteration) | Files: submod-solver/src/lazy_greedy.rs | Deps: T021 | Verify: Gap bound logged per iteration, decreases monotonically, justifies early stopping

- [ ] **T147d** [P]: Implement curvature κ computation (measure diminishing returns rate: κ = 1 - min_{e,S} Δ(e|S)/Δ(e|∅)) | Files: submod-objectives/src/utils.rs | Deps: T005 | Verify: κ ∈ [0,1], κ→0 for modular (additive) functions, κ→1 for strongly submodular

- [ ] **T147e**: Add summation accuracy helpers (Kahan summation for f32, pairwise summation for deterministic mode) | Files: submod-core/src/numerics.rs | Deps: T011 | Verify: Kahan reduces summation error by 100x on 10^6 random floats, pairwise matches sequential sum

### US5.6: Monitoring Baselines & Alerting [IMPORTANT]

- [ ] **T147f** [P]: Define healthy metric baselines (heap_reinsert_total/heap_pop_total < 0.1, io_wait_seconds/iteration_seconds < 0.05, gap_estimate < 0.01 for early stop) | Files: docs/monitoring_baselines.md | Deps: T117-T147 | Verify: Baselines documented with rationale from empirical data

- [ ] **T147g** [P]: Create Prometheus alerting rules (YAML rules for: gap_estimate > 0.05 warning, io_wait > 0.5 critical, reinsert_ratio > 0.15 warning) | Files: deploy/prometheus_rules.yml | Deps: T147f, T132-T137 | Verify: Rules load in Prometheus, alerts fire on synthetic violations

- [ ] **T147h**: Document metric interpretation guide (what each metric means, healthy ranges, troubleshooting steps for anomalies) | Files: docs/metrics_guide.md | Deps: T147f-T147g | Verify: Guide covers all 15+ metrics with interpretation

**Checkpoint**: Monitoring complete - All P1/P2/P3 user stories independently functional

---

## Phase 8: Python Bindings (21 tasks) [Cross-Cutting]

**Goal**: pyo3 bindings with zero-copy numpy, GIL release
**Completion**: Python can call Rust solver with <1% overhead

- [ ] **T148-T157**: Core Bindings (setup pyo3 with maturin, trait wrappers PySubmodularOracle/PyConstraint, export FacilityLocation.from_numpy(), CardinalityConstraint, KnapsackConstraint, SolverConfig, Selection, GIL release in solve(), zero-copy numpy tests, round-trip integration test) | Files: submod-bindings-py/* | Deps: T005-T008, T013-T037 | Verify: Python imports work, zero-copy confirmed

- [ ] **T158-T163**: Custom Oracle Support (Python SubmodularOracle base class, callbacks for gain/commit/upper_bound, error handling for Python exceptions, custom oracle example) | Files: submod-bindings-py/src/custom_oracle.rs, examples/ | Deps: T148-T157 | Verify: Custom Python oracle works

- [ ] **T164-T168**: Documentation + Examples (generate Python stub files with type annotations, docstrings, quickstart.py example, advanced example with custom oracle, performance benchmark comparing Python vs Rust overhead) | Files: submod-bindings-py/*.pyi, examples/, benches/ | Deps: T148-T163 | Verify: Examples run, docs complete

---

## Phase 9: CLI and Service (20 tasks) [Cross-Cutting]

**Goal**: CLI tool for generation/solving, HTTP service
**Deps**: Requires Phase 3 (solver), Phase 7 (metrics)

- [ ] **T169-T178**: Command-Line Interface (clap arg parsing, `generate` subcommand, `solve` and `verify` subcommands with flags: --input, --output, --algorithm, --cardinality, --seed, --trace, --metrics-port, integration test, --help documentation) | Files: submod-service/src/cli.rs, tests/ | Deps: T013-T037, T117-T147 | Verify: All commands work

- [ ] **T179-T183**: Data Generation Tools (facility location/coverage generators, Parquet output with proper schema, schema validation) | Files: submod-service/src/generators.rs | Deps: T169-T178 | Verify: Generated data loads correctly

- [ ] **T184-T188**: HTTP Service (Axum service, /metrics and /health endpoints, graceful shutdown, structured logging with tracing crate, integration tests) | Files: submod-service/src/service.rs, tests/ | Deps: T132-T137 | Verify: Service starts, endpoints respond

---

## Phase 10: Additional Oracle Implementations (18 tasks) [Cross-Cutting]

**Goal**: SaturatingCoverage, DiversityAwareFacilityLocation oracles
**Deps**: Requires T005 (SubmodularOracle trait)

- [ ] **T189-T195**: SaturatingCoverage Oracle (struct with saturation thresholds, gain/commit/upper_bound/from_parquet implementations, unit tests for saturation behavior, property test for submodularity) | Files: submod-objectives/src/saturating_coverage.rs | Deps: T005 | Verify: Saturation works correctly

- [ ] **T196-T202**: DiversityAwareFacilityLocation Oracle (struct with diversity penalties, gain combining coverage+diversity terms, commit updating both states, from_parquet with diversity matrix loading, unit tests, integration test vs vanilla FacilityLocation) | Files: submod-objectives/src/diversity_aware_facility_location.rs | Deps: T005, T013-T018 | Verify: Diversity penalties applied correctly

- [ ] **T203-T206**: Oracle Utilities (validation helpers for universe_size consistency, submodularity/monotonicity verification tools with random sampling, benchmark suite for oracle evaluations gain() throughput) | Files: submod-objectives/src/utils.rs, benches/ | Deps: T005 | Verify: Utilities work on all oracles

---

## Phase 11: Documentation and Finalization (30 tasks)

**Goal**: Complete documentation, testing, release preparation, Appendix B validation
**Deps**: All phases

### T207-T235: Documentation, Testing, Release (Condensed)

- [ ] **T207-T213**: API Documentation (write rustdoc for all public APIs in core/objectives/solver/io/bindings, add code examples in rustdoc comments, generate docs with `cargo doc --no-deps --open`, verify all public items documented) | Files: */src/*.rs | Deps: All phases | Verify: Docs complete, no warnings

- [ ] **T214-T219**: User Guides (quickstart.md Rust/Python from spec, algorithms.md explaining Lazy Greedy/Stochastic/Continuous, benchmarks.md with performance table, troubleshooting.md with common errors, update README.md with project overview) | Files: docs/*.md, README.md | Deps: All phases | Verify: Guides follow spec quickstart.md format

- [ ] **T220-T223**: Architecture Documentation (architecture.md with trait design patterns/thread safety, data-model.md with SelectionView/CSR matrix/Parquet schema, determinism.md from T082, numerical-stability.md explaining f32 vs f64/log-determinant defenses) | Files: docs/*.md | Deps: All phases | Verify: Architecture documented

- [ ] **T224-T229**: Comprehensive Testing (achieve >80% code coverage across all crates, add property tests for submodularity preservation, add property tests for constraint feasibility, add integration tests for all user story acceptance criteria, add performance regression tests with Criterion, add determinism regression tests with hash verification) | Files: tests/*, benches/* | Deps: All phases | Verify: Coverage >80%, all tests pass

- [ ] **T230-T235**: Release Preparation (set up semantic versioning starting at 0.1.0, write CHANGELOG.md with initial release notes, add LICENSE file MIT or Apache-2.0, add CONTRIBUTING.md with development guidelines, set up cargo publish workflow, create GitHub release with binaries) | Files: Cargo.toml, CHANGELOG.md, LICENSE, CONTRIBUTING.md, .github/workflows/release.yml | Deps: All phases | Verify: Release artifacts ready

- [ ] **T236**: Validate implementation against Appendix B Critical Implementation Checklist (verify all checkboxes from spec §B: oracle interface, selection representation, constraint callbacks, lazy greedy, continuous greedy, knapsack, log-determinant, determinism, I/O, monitoring) | Files: docs/appendix_b_validation.md | Deps: All phases | Verify: All 10 sections validated, documented deviations if any

---

## Phase 12: Advanced Features (3 tasks) [Nice-to-Have]

**Goal**: Specialized algorithms for niche use cases (streaming, large-scale kernel approximation, sparse compression)
**Deps**: Requires Phase 2 (traits), Phase 3 (solver), Phase 4 (oracles)
**Priority**: Optional for v0.1.0, defer to v0.2.0 if time constrained

- [ ] **T240**: Implement Sieve-Streaming algorithm (single-pass streaming, memory O(k log(1/ε)), geometric threshold levels per spec §5.5) | Files: submod-solver/src/sieve_streaming.rs | Deps: T005-T006, T010 | Verify: 1M stream processed in single pass, memory <10MB for k=250, achieves ≥50% approximation

- [ ] **T241**: Implement Nyström low-rank approximation for LogDeterminant (K ≈ ZZ^T with rank r << n, compute log det(I + Z_S^T Z_S) in O(r^2 |S| + r^3) per spec §4.3 lines 681-691) | Files: submod-objectives/src/log_determinant_nystrom.rs | Deps: T038-T047 | Verify: 10K×10K kernel with r=100 completes in <1s, quality within 5% of full-rank

- [ ] **T242**: Implement TopK compression for sparse matrices (retain top-L utilities per candidate, L~50, bounded degree per spec §7.5) | Files: submod-io/src/topk_compression.rs | Deps: T027 | Verify: 1M×1M matrix with 1000 avg degree compressed to L=50, marginal gains within 95% of full

**Completion**: Advanced features enable specialized deployments (streaming pipelines, massive kernel DPPs, extreme-scale sparsity)

---

## Dependencies & Execution Order

### Critical Path
```
Phase 1 (Setup) → Phase 2 (Foundational) [BLOCKS ALL] → Phase 3 (MVP - US1) →
{Phases 4-7 parallel} → Phases 8-11
```

### Phase Dependencies
- **Phase 1** → **Phase 2** [BLOCKING]
- **Phase 2** → ALL subsequent phases [BLOCKING]
- **Phase 3 (US1)** → Phases 4-7 can start
- **Phases 4-7** are independent (parallel)
- **Phase 8 (Python)** can start after Phase 3
- **Phase 9 (CLI)** needs Phase 3 + Phase 7 (metrics)
- **Phase 10 (Oracles)** needs Phase 2 only
- **Phase 11 (Docs)** needs all phases

### MVP vs Full Feature Set
- **MVP Critical Path** (50 tasks): T001-T004 → T005-T012 → T013-T037k (includes SIMD, I/O, quickstart, CI, docs)
- **Production Critical Path** (79 tasks): MVP + Phase 4 (T038-T062d, includes reliability + resource limits)
- **Full Feature Set** (265 tasks): All phases including Phase 12 (advanced features)

### Parallel Opportunities
- **Phase 2**: All 8 tasks [P] (independent type definitions)
- **Phase 3**: T013-T018 [P], T025-T029 [P], T030-T033 [P] (3 parallel tracks)
- **Phase 4**: T038-T047 [P], T048-T052 [P], T053-T058 [P] (3 parallel tracks)
- **Phases 4-7**: Fully independent after Phase 3
- **Phases 8-10**: Can all proceed in parallel

---

## Summary

**Total Tasks**: 265 (baseline 235 + 14 perf/explain + 16 production-readiness)
**MVP Scope** (Phase 1-3): **50 tasks → 5-6 weeks** (SIMD, I/O, quickstart, CI, MVP docs)
**Extended Scope** (+ Phase 4): **79 tasks → 8-10 weeks** (+ reliability, security)
**Full Feature Set**: **265 tasks → 20-26 weeks** (1 developer, production-grade)

**Task Breakdown by Phase**:
- Phase 1 (Setup): 4 tasks
- Phase 2 (Foundational): 8 tasks ← **BLOCKS ALL**
- Phase 3 (US1 - Large-Scale P1): **40 tasks** (+8 from v1: quickstart, CI, MVP docs; includes SIMD/I/O from prior update) ← **MVP ENDS**
- Phase 4 (US2 - Reliability P1): **32 tasks** (+7: epsilon arithmetic, resource limits, security)
- Phase 5 (US3 - Reproducibility P2): **21 tasks** (+1: determinism guide)
- Phase 6 (US4 - Constraints P2): 36 tasks
- Phase 7 (US5 - Monitoring P3): **39 tasks** (+3: baselines, alerting, metric guide)
- Phase 8 (Python Bindings): 21 tasks
- Phase 9 (CLI/Service): 20 tasks
- Phase 10 (Additional Oracles): 18 tasks
- Phase 11 (Documentation/Finalization): **30 tasks** (+1: Appendix B validation)
- Phase 12 (Advanced Features): **3 tasks** [Nice-to-Have] (Sieve-Streaming, Nyström, TopK)

**User Story Verification** (Independent Tests):
- **US1** (T034-T037): Load 1M×1M, run lazy greedy, <10 min, ≥63% approximation
- **US2** (T059-T062): Log-determinant on ill-conditioned matrices, graceful degradation
- **US3** (T079-T082): Run with seed=42 on 1/8/64 threads, verify identical hashes
- **US4** (T113-T116): Continuous greedy on matroid, ≥60% of optimal
- **US5** (T143-T147): Query /metrics endpoint, all metrics present, <0.1% overhead

**Recommended Approach**: Start with **MVP (T001-T037k, 50 tasks)**, validate with stakeholders, then add **Phase 4 (reliability + security)** for production deployment.

**Developer Allocation**:
- **1 developer**: Sequential execution, **20-26 weeks** (realistic for production-grade with all Critical/Important tasks)
- **2 developers**: Parallel tracks in Phase 3-7, **13-18 weeks**
- **3 developers**: MVP + 2 parallel P1 tracks, **10-14 weeks**

**Key Additions** (30 new tasks total):

**Critical (MVP-blocking, +8 tasks):**
- T037d-f: Quickstart dataset + verification script (spec §11 compliance)
- T037g-i: CI enhancements (determinism, performance regression, cross-platform)
- T037j-k: MVP documentation (guide + rustdoc)

**Critical (Production, +3 tasks):**
- T051a-c: Epsilon-aware constraint arithmetic (spec §8.4)

**Important (+5 tasks):**
- T062a-d: Resource limits & DoS protection + quickstart update
- T082a: Determinism guide with examples

**Important (Monitoring, +3 tasks):**
- T147f-h: Metric baselines, alerting rules, interpretation guide

**Nice-to-Have (Phase 12, +3 tasks):**
- T240-242: Sieve-Streaming, Nyström, TopK compression (defer to v0.2.0)

**Finalization (+1 task):**
- T236: Appendix B checklist validation

**Original Additions (+14 from previous update):**
- **Performance**: SIMD optimization (T017a-c), I/O sharding & prefetch (T027a-d)
- **Correctness**: Continuous Greedy view-based safety (T107a), common RNs (T111a)
- **Explainability**: Coverage reports, gap bounds, curvature, summation accuracy (T147a-e)
