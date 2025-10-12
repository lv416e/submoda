<!--
Sync Impact Report
==================

Version Change: 1.0.0 → 1.0.0 (Initial ratification)

Principles Defined:
- I. Specification Adherence (new)
- II. Theoretical Guarantees (new)
- III. Thread Safety (new)
- IV. Deterministic Execution (new)
- V. Numerical Stability (new)
- VI. Observability (new)

Added Sections:
- Technical Context (production-scale requirements)
- Quality Gates (development workflow)
- Governance (amendment procedures)

Templates Status:
✅ plan-template.md - Constitution Check section aligns with principles
✅ spec-template.md - Requirements format supports principle validation
✅ tasks-template.md - Task organization supports incremental testing
✅ No command files present in .specify/templates/commands/
✅ README.md - Project description aligns with constitution scope

Follow-up TODOs:
- RATIFICATION_DATE needs confirmation (using 2025-10-12 as initial)
-->

# Submoda Constitution

## Core Principles

### I. Specification Adherence

**PRIMARY SOURCE**: `docs/specification.md` is the single source of truth for all
implementation decisions, algorithms, data structures, and numerical methods.

**NON-NEGOTIABLE RULES**:
- All design decisions MUST reference specific sections of specification.md
- Deviation from specification.md requires explicit amendment of specification.md first
- Implementation ambiguities MUST be resolved by clarifying specification.md
- No undocumented algorithms, optimizations, or data structure changes

**RATIONALE**: Large-scale optimization systems require rigorous specification to ensure
correctness, reproducibility, and maintainability. Specification drift leads to undefined
behavior in production.

### II. Theoretical Guarantees

**APPROXIMATION QUALITY**: All algorithms MUST maintain provable approximation guarantees
as specified in specification.md Section 1 (Mathematical Foundations).

**NON-NEGOTIABLE RULES**:
- Lazy Greedy: (1-1/e) ≈ 0.632 for monotone submodular + cardinality constraint
- Stochastic Greedy: (1-1/e - ε) with explicit ε parameter
- Continuous Greedy: (1-1/e) for matroid constraints
- Knapsack constraint: Explicit Practical vs Theoretical modes with documented trade-offs
- Algorithm modifications that weaken guarantees MUST be rejected unless specification.md
  explicitly documents the degradation

**RATIONALE**: Users depend on approximation guarantees for production decision-making.
Silent degradation violates contract with users.

### III. Thread Safety

**ORACLE INTERFACE**: All submodular function evaluations MUST be thread-safe and
immutable as per specification.md Section 3 (Thread-Safe Oracle Interface).

**NON-NEGOTIABLE RULES**:
- `gain(&self, view: &SelectionView, e: ItemId) -> f64` MUST be immutable (&self)
- `commit(&mut self, e: ItemId)` is the ONLY mutating method, called sequentially
- All evaluation uses immutable `SelectionView` snapshots, never direct selection
- No shared mutable state between threads (use Arc + thread-local caches via fork())
- Parallel evaluation via Rayon MUST use immutable references only

**RATIONALE**: Rust's borrow checker enforces safety, but design must support parallelism.
Mutable gain() prevents parallel evaluation, violating performance requirements.

### IV. Deterministic Execution

**REPRODUCIBILITY**: Identical inputs MUST produce identical selection sequences, objective
values, and audit logs across runs, parallelism levels, and platforms (best-effort for
cross-platform) as per specification.md Section 9 (Deterministic Execution Framework).

**NON-NEGOTIABLE RULES**:
- Hierarchical RNG seeding: master_seed → algorithm_seed → iteration_seed
- Tie-breaking MUST be deterministic (default: by ItemId, configurable)
- Fixed-order parallel reduction when `determinism.fixed_order = true`
- Heap operations use deterministic Ord implementation with tie-breaking
- Audit log hashes computed only from integer decisions (ItemId sequence, seed, config)
- Never hash float values (non-deterministic across platforms)

**RATIONALE**: Production systems require reproducible debugging, compliance auditing,
and regression testing. Non-deterministic behavior prevents root-cause analysis.

### V. Numerical Stability

**TYPE SELECTION**: Floating-point precision MUST match algorithmic requirements as per
specification.md Section 8 (Numerical Stability and Robustness).

**NON-NEGOTIABLE RULES**:
- Facility Location, Saturating Coverage: f32 (2x memory reduction, wider SIMD)
- Log-Determinant: f64 mandatory (Cholesky fails with f32)
- Continuous Greedy gradient: f64 for intermediate accumulation
- Defense-in-depth for Log-Determinant:
  * Layer 1: Epsilon regularization (K ← K + εI)
  * Layer 2: Safe Schur complement with epsilon clipping
  * Layer 3: log1p for small diagonal entries
  * Layer 4: Immediate refactorization on NaN
  * Layer 5: Graceful degradation (disable log-det after 5 failures)
- Fixed-order reduction uses f64 aggregation for determinism when enabled
- Knapsack budget comparison with relative epsilon tolerance (not exact equality)

**RATIONALE**: Numerical instability causes silent correctness failures. Multi-layer defense
prevents catastrophic cancellation, overflow, and NaN propagation.

### VI. Observability

**MANDATORY INSTRUMENTATION**: All production code MUST expose Prometheus metrics and
structured audit logs as per specification.md Section 10 (Monitoring and Observability).

**NON-NEGOTIABLE RULES**:
- Prometheus metrics: counters (gain_eval_total, commit_total), gauges (objective_value,
  selection_size), histograms (gain_compute_seconds, iteration_seconds)
- Audit logs: JSON Lines format with mandatory fields (iteration, selected_element,
  marginal_gain, objective_value, algorithm, rng_seed, timestamp_ms)
- All numerical thresholds logged with semantic names (lazy_epsilon, ub_threshold, etc.)
- Termination reason explicitly logged (cardinality_reached, upper_bound_threshold,
  stagnation, timeout)
- Fallback events (log-det degradation, numerical errors) logged with action taken
- HTTP /metrics endpoint exposed on configurable port (default 9090)

**RATIONALE**: Production systems require real-time monitoring for performance debugging,
compliance auditing, and incident response. Missing metrics prevent root-cause analysis.

## Technical Context

**SCALE REQUIREMENTS**: System MUST handle millions of candidates × millions of demands ×
billions of non-zero interactions, processing in minutes on commodity hardware (32-core,
128GB RAM).

**IMPLEMENTATION LANGUAGE**: Rust (leveraging type safety, zero-cost abstractions, Rayon
parallelism, and memory safety without garbage collection).

**PERFORMANCE TARGETS** (specification.md Section 12.2):
- Baseline: n=10^6 candidates, m=10^6 demands, nnz=10^8, k=250 selections
- Target: < 10 minutes on 32-core, 128GB RAM
- Lazy Greedy efficiency: 5-10% of evaluations vs standard greedy
- I/O: Parquet with predicate pushdown, row-group pruning, async prefetch

**CONSTRAINTS**:
- No runtime type erasure overhead (use monomorphization for f32/f64)
- No GIL blocking in Python bindings (use Python::allow_threads())
- Zero-copy I/O via Arrow FFI for NumPy/Pandas integration
- Memory: O(m + nnz) for CSR, bounded prefetch queue to prevent OOM

## Quality Gates

**DEVELOPMENT WORKFLOW**: All features MUST follow the specify→plan→tasks→implement
pipeline with mandatory reviews at each stage.

**REVIEW GATES**:
1. **Specification Phase**: User scenarios, acceptance criteria, functional requirements
   reviewed and approved
2. **Planning Phase**: Constitution Check passed, technical design reviewed, structure
   decided
3. **Tasks Phase**: Task dependencies validated, parallelism opportunities identified,
   test strategy approved
4. **Implementation Phase**: Code review, test coverage, benchmark validation, metrics
   instrumentation verified

**DEFINITION OF DONE** (per task):
- [ ] Code implements specification.md requirements exactly
- [ ] Unit tests validate submodularity properties (if oracle change)
- [ ] Integration tests validate end-to-end correctness (if solver change)
- [ ] Determinism test: 10 identical runs produce identical selection sequences
- [ ] Benchmark: no >1% regression on standard datasets
- [ ] Prometheus metrics exposed (if new component)
- [ ] Audit log fields documented (if new algorithm/constraint)
- [ ] Approximation guarantee validated (if algorithm change)

## Governance

**AMENDMENT PROCEDURE**:
1. Propose amendment with rationale (GitHub issue or spec discussion)
2. Update specification.md if principle change affects algorithm/design
3. Update constitution.md with version bump (MAJOR: breaking, MINOR: new principle,
   PATCH: clarification)
4. Propagate changes to dependent templates (plan, spec, tasks)
5. Update runtime guidance (README.md, CLAUDE.md, quickstart.md if added)
6. Commit with message: `docs: amend constitution to vX.Y.Z (<summary>)`

**COMPLIANCE VERIFICATION**:
- All PRs MUST reference constitution principles in description
- Code reviews MUST verify specification.md adherence
- CI MUST run determinism tests (10 runs, assert identical selection sequences)
- Benchmark regressions >1% require justification or rejection
- Complexity violations (e.g., new approximation mode) require documented rationale

**PRECEDENCE**: This constitution supersedes all other development practices. In case of
conflict between constitution and other guidance, constitution wins. Ambiguities MUST be
resolved by amending constitution.

**Version**: 1.0.0 | **Ratified**: 2025-10-12 | **Last Amended**: 2025-10-12
