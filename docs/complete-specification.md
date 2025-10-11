# Complete Technical Specification: Submodular Optimization Platform
## Large-Scale Subset Selection System with Rust Implementation

**Status:** Production-Ready Specification
**Target Audience:** Implementation teams, system architects, researchers

---

## Executive Summary

This document provides a complete technical specification for a production-grade submodular optimization platform implemented in Rust. The platform addresses large-scale subset selection problems with the following design pillars:

1. **Theoretical Rigor:** Algorithms maintain provable approximation guarantees (e.g., $(1-1/e) \approx 0.632$ for monotone submodular functions under cardinality constraints)
2. **Thread Safety:** Immutable oracle interfaces enable safe parallel evaluation without data races
3. **Deterministic Execution:** Reproducible results across runs, platforms, and parallelism levels
4. **Numerical Stability:** Robust handling of floating-point arithmetic with graceful degradation
5. **Production Readiness:** Comprehensive monitoring, auditing, and failure recovery mechanisms

The platform is designed to handle problems with millions of candidates, millions of demand points, and billions of non-zero interactions, processing them in minutes on commodity hardware.

---

## Table of Contents

1. [Mathematical Foundations](#1-mathematical-foundations)
2. [Core Architecture and Critical Design Decisions](#2-core-architecture-and-critical-design-decisions)
3. [Thread-Safe Oracle Interface](#3-thread-safe-oracle-interface)
4. [Objective Functions](#4-objective-functions)
5. [Optimization Algorithms](#5-optimization-algorithms)
6. [Constraint Systems](#6-constraint-systems)
7. [Data Management and I/O](#7-data-management-and-io)
8. [Numerical Stability and Robustness](#8-numerical-stability-and-robustness)
9. [Deterministic Execution Framework](#9-deterministic-execution-framework)
10. [Monitoring and Observability](#10-monitoring-and-observability)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Theoretical Foundations and References](#12-theoretical-foundations-and-references)

---

## 1. Mathematical Foundations

### 1.1 Submodular Functions: Rigorous Definition

Let $V$ be a finite ground set of candidate elements with $|V| = n$. A set function $f: 2^V \to \mathbb{R}$ is **submodular** if for all subsets $A \subseteq B \subseteq V$ and all elements $e \in V \setminus B$:

$$
f(A \cup \{e\}) - f(A) \geq f(B \cup \{e\}) - f(B)
$$

This inequality expresses the **diminishing returns property**: the marginal gain from adding element $e$ decreases (or remains constant) as the context set grows from $A$ to $B$.

**Essential Properties:**

- **Monotonicity:** $f$ is monotone if $A \subseteq B$ implies $f(A) \leq f(B)$
- **Normalization:** $f(\emptyset) = 0$ (zero baseline)
- **Non-negativity:** $f(S) \geq 0$ for all $S \subseteq V$

This specification primarily addresses **monotone, normalized, non-negative submodular functions**, with architectural support for future extensions to non-monotone cases.

### 1.2 The Subset Selection Problem

**Cardinality-Constrained Maximization:**

$$
\max_{S \subseteq V} f(S) \quad \text{subject to} \quad |S| \leq k
$$

**Fundamental Theorem (Nemhauser, Wolsey, Fisher 1978):**

For monotone submodular $f$ and cardinality constraint $k$, the greedy algorithm achieves:

$$
f(S_{\text{greedy}}) \geq \left(1 - \frac{1}{e}\right) \cdot f(S_{\text{opt}}) \approx 0.632 \cdot f(S_{\text{opt}})
$$

This is the best polynomial-time approximation achievable unless P=NP.

### 1.3 Marginal Gain

For a set $S \subseteq V$ and element $e \in V \setminus S$, the **marginal gain** (or **marginal contribution**) is:

$$
\Delta(e \mid S) := f(S \cup \{e\}) - f(S)
$$

By submodularity, if $S \subseteq T$, then $\Delta(e \mid S) \geq \Delta(e \mid T)$.

### 1.4 Extended Constraint Formulations

**Knapsack Constraint:**

$$
\sum_{s \in S} \text{cost}(s) \leq B
$$

where $\text{cost}: V \to \mathbb{R}_{\geq 0}$ assigns a cost to each element and $B > 0$ is the budget.

**Partition Matroid Constraint:**

The ground set is partitioned into disjoint categories $V = C_1 \sqcup C_2 \sqcup \cdots \sqcup C_m$, with capacity limits:

$$
|S \cap C_j| \leq \text{cap}_j \quad \forall j \in [m]
$$

**General Matroid Constraint:**

A matroid $\mathcal{M} = (V, \mathcal{I})$ consists of a finite set $V$ and a family of independent sets $\mathcal{I} \subseteq 2^V$ satisfying:

- **(I1)** Empty set is independent: $\emptyset \in \mathcal{I}$
- **(I2)** Downward closure: $A \in \mathcal{I}$ and $B \subseteq A$ implies $B \in \mathcal{I}$
- **(I3)** Augmentation property: If $A, B \in \mathcal{I}$ and $|A| < |B|$, then there exists $b \in B \setminus A$ such that $A \cup \{b\} \in \mathcal{I}$

The optimization problem becomes:

$$
\max_{S \in \mathcal{I}} f(S)
$$

Examples include partition matroids and graphic matroids (forests in graphs).

### 1.5 Multilinear Extension

For Continuous Greedy algorithm, we use the **multilinear extension** $F: [0,1]^n \to \mathbb{R}$:

$$
F(x) := \mathbb{E}_{R \sim x}[f(R)]
$$

where $R \sim x$ means each element $i$ is included in $R$ independently with probability $x_i$.

The gradient component is:

$$
\frac{\partial F}{\partial x_i}(x) = \mathbb{E}_{R \sim x}[f(R \cup \{i\}) - f(R)]
$$

This represents the expected marginal gain of element $i$ for fractional solution $x$.

---

## 2. Core Architecture and Critical Design Decisions

### 2.1 Crate Organization

The platform is organized into specialized Rust crates with clear separation of concerns:

**submod-core**
- Core trait definitions (`SubmodularOracle`, `Constraint`, `Matroid`)
- Type system primitives (`ItemId`, `Weight`, `SelectionView`)
- Strategy configuration
- Determinism and metrics infrastructure

**submod-objectives**
- Concrete oracle implementations (Facility Location, Saturating Coverage, Log-Determinant)
- Domain-specific objectives
- Extensibility framework

**submod-solver**
- Algorithm implementations (Lazy Greedy, Stochastic Greedy, Continuous Greedy, Sieve-Streaming)
- Shared optimization utilities

**submod-io**
- Parquet/Arrow data loading with predicate pushdown
- Schema validation
- Sparse matrix construction (CSR/CSC)
- Row-group indexing for efficient access

**submod-bindings-py**
- PyO3-based Python bindings
- NumPy/Pandas integration

**submod-service**
- gRPC and REST endpoints
- Job queue management
- Prometheus metrics exposure

### 2.2 Fundamental Type System

**ItemId**: Element identifier
- Base type: `u32` (up to ~4.3 billion candidates)
- Configurable to `u64` for extreme-scale problems
- Must be dense and zero-indexed for efficient bitset operations

**Weight**: Numerical values
- Default: `f32` (memory-efficient, fast SIMD)
- Configurable to `f64` for numerically sensitive operations (log-determinant)
- Strict validation: no NaN, no infinity, non-negative where required

**SelectionView**: Immutable snapshot of current selection

```rust
pub struct SelectionView {
    pub in_set: fixedbitset::FixedBitSet,  // O(1) membership testing
    pub size: usize,                        // Cardinality |S|
}
```

**Why FixedBitSet is Critical:**
- **O(1) membership testing** vs O(n) for `Vec::contains`
- **Compact memory**: 1 bit per element vs 4 bytes for `Vec<ItemId>`
- **Fast set operations**: union, intersection for constraint checking
- **Thread-safe cloning**: enables parallel evaluation

**Selection**: Final result structure

```rust
pub struct Selection {
    pub items: Vec<ItemId>,              // Derived from SelectionView at output
    pub objective: f64,                   // Final objective value
    pub used_budget: f64,                 // For knapsack constraints
    pub counts_by_part: Vec<usize>,      // For partition constraints
    pub trace: Vec<IterationTrace>,      // Per-iteration audit trail
}
```

**Design Principle:** `SelectionView` with its `FixedBitSet` is the **canonical representation** during optimization. The `Vec<ItemId>` in `Selection` is generated only for final output, preventing duplicate entries and maintaining O(1) membership tests throughout execution.

**Tie-Breaking:** When multiple elements have equal marginal gains, deterministic tie-breaking is essential for reproducibility. **Default policy: break ties by ItemId** (prefer lower ID). This ensures consistent selection order across runs. Alternative tie-breaking strategies (by upper bound, random with seeded RNG) are configurable via `TieBreak` enum.

### 2.3 Configuration Types

**Strategy**: Algorithm selection

```rust
pub enum Strategy {
    LazyGreedy {
        epsilon: f64,  // ε-approximation tolerance (0.0 = exact)
    },
    StochasticGreedy {
        epsilon: f64,       // Approximation parameter
        sample_factor: f64, // Sample size multiplier
    },
    SieveStreaming {
        epsilon: f64,
        threshold_levels: usize,
    },
    ContinuousGreedy {
        steps: usize,           // Time discretization T
        grad_samples: usize,    // Gradient estimation samples G
        rounding: RoundingMode, // Pipage or Swap
    },
}
```

**Constraint**: Problem constraints with flexible callbacks

```rust
pub enum Constraint {
    Cardinality {
        k: usize,
    },
    Knapsack {
        budget: f64,
        cost_fn: Arc<dyn Fn(ItemId) -> f64 + Send + Sync>,  // Dynamic cost lookup
        mode: KnapsackMode,
    },
    Partition {
        part_fn: Arc<dyn Fn(ItemId) -> usize + Send + Sync>,  // Category assignment
        capacities: Vec<usize>,
    },
    Matroid {
        matroid: Arc<dyn Matroid + Send + Sync>,
    },
}
```

**Why `Arc<dyn Fn>` instead of `fn` pointers:**
- Supports closures that capture environment
- Enables dynamic computation (e.g., database lookups, complex rules)
- Much more flexible than raw function pointers
- Thread-safe with `Send + Sync` bounds

**KnapsackMode**: Explicit approximation strategy

```rust
pub enum KnapsackMode {
    /// Fast practical mode: enumerate top-1 to top-3 items + cost-benefit ratio greedy
    /// Approximation: No (1-1/e) guarantee, but fast and empirically good
    Practical {
        enumerate_top_k: usize,  // Typically 1-3
    },
    /// Theoretical mode: Continuous Greedy via multilinear extension
    /// Approximation: (1-1/e) guarantee (Sviridenko 2004)
    /// Much slower but provable quality
    Theoretical {
        cg_steps: usize,
        grad_samples: usize,
    },
}
```

This explicit separation ensures users understand the approximation quality vs speed tradeoff.

**Determinism**: Reproducibility configuration

```rust
pub struct Determinism {
    pub seed: u64,              // RNG seed
    pub fixed_order: bool,      // Fix parallel reduction order
    pub tie_breaking: TieBreak, // Equal marginal gain resolution
}

pub enum TieBreak {
    ById,            // Prefer lower ItemId (default, deterministic)
    ByUpperBound,    // Prefer higher cached upper bound
    Random,          // Use seeded RNG (less deterministic)
}
```

---

## 3. Thread-Safe Oracle Interface

### 3.1 The Critical Design Pattern

**The Problem with Mutable gain():**

```rust
// ❌ ANTI-PATTERN: Cannot parallelize
fn gain(&mut self, sol: &Selection, e: ItemId) -> f64
```

This design forces sequential evaluation because Rust's borrow checker correctly prevents multiple `&mut self` references. Parallel evaluation becomes impossible.

**The Solution: Immutable Queries with SelectionView:**

```rust
pub trait SubmodularOracle: Send + Sync {
    /// Number of elements in ground set
    fn universe_size(&self) -> usize;

    /// Compute marginal gain Δ(e|S) - READ-ONLY, THREAD-SAFE
    /// view: immutable snapshot of current selection
    /// Returns: f(S ∪ {e}) - f(S)
    fn gain(&self, view: &SelectionView, e: ItemId) -> f64;

    /// Commit element to internal state - WRITE, NOT THREAD-SAFE
    /// Must be called sequentially after selecting element
    fn commit(&mut self, e: ItemId);

    /// Upper bound for Lazy Greedy - READ-ONLY, THREAD-SAFE
    ///
    /// **Semantics:** Returns the most recently evaluated marginal gain Δ(e|S')
    /// for this element (tight upper bound). Never use theoretical upper bounds.
    ///
    /// **Initialization Policy (FIXED):**
    /// - Default: f64::INFINITY (lazy initialization on first pop)
    /// - Optional: Pre-evaluate with oracle.gain(&empty_view, e) if config.eager_init = true
    ///   (costs O(n) evaluations upfront but may reduce later re-evaluations)
    ///
    /// Update rule: After evaluation, upper_bound(e) ← most recent Δ(e|S)
    fn upper_bound(&self, e: ItemId) -> f64;

    /// Create lightweight clone for parallel workers - OPTIONAL
    /// Shares immutable data, clones only mutable caches
    fn fork(&self) -> Box<dyn SubmodularOracle> {
        unimplemented!("fork() not supported by this oracle")
    }

    /// Prefetch hint for element e - OPTIONAL
    fn prefetch(&self, _e: ItemId) {
        // Default: no-op
    }

    /// Batch evaluation for SIMD/cache efficiency - OPTIONAL
    /// Evaluate marginal gains for multiple candidates in one call
    /// Default: fallback to sequential gain() calls
    fn gain_batch(&self, view: &SelectionView, candidates: &[ItemId]) -> Vec<f64> {
        candidates.iter().map(|&e| self.gain(view, e)).collect()
    }
}
```

**Why Batch Evaluation:**

For objectives like Facility Location or Saturating Coverage, evaluating multiple candidates simultaneously enables:
- **SIMD vectorization:** Process 4-8 candidates per instruction
- **Cache efficiency:** Amortize demand/row loads across candidates
- **Reduced dispatch overhead:** One virtual call instead of N

Solvers should prefer `gain_batch()` when available and fall back to sequential `gain()` otherwise.

### 3.2 Separation of Read and Write Operations

**Key Insight:** Marginal gain evaluation is a **read operation** that should not mutate oracle state. Only the final commitment after selection should mutate.

**Parallel Evaluation Pattern:**

```rust
// Safe parallel evaluation
let deltas: Vec<(ItemId, f64)> = candidates
    .par_iter()
    .map(|&e| {
        let delta = oracle.gain(&view, e);  // Multiple &self OK!
        (e, delta)
    })
    .collect();

// Sequential commit after selection
let best = deltas.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
oracle.commit(best.0);
view.in_set.insert(best.0 as usize);
view.size += 1;
```

### 3.3 The fork() Pattern for Advanced Parallelism

For oracles where cloning entire state is wasteful, `fork()` enables shared immutable data with thread-local caches:

```rust
impl SubmodularOracle for FacilityLocation {
    fn fork(&self) -> Box<dyn SubmodularOracle> {
        Box::new(Self {
            demand_weights: self.demand_weights.clone(),  // Arc, cheap
            utility_matrix: self.utility_matrix.clone(),  // Arc, cheap
            best_u: vec![0.0; self.n_demands],            // New per-thread cache
            last_epoch: 0,                                 // Fresh epoch tracker
        })
    }
}
```

**Fork Contract (Observability Rules):**

1. **Shared Immutable Data:** Use `Arc<T>` for read-only structures (utility matrices, weights, kernel matrices). Cloning `Arc` is O(1).

2. **Thread-Local Mutable Caches:** Each forked oracle maintains its own mutable state (`best_u`, `cumulative`, etc.). These caches are **independent** across forks.

3. **Visibility Guarantee:** Forked oracles only observe state updates from `commit()` calls on the **parent** oracle. Evaluations on forked instances (`gain()` calls) do **not** affect each other or the parent until results are aggregated and committed sequentially.

4. **No Cross-Fork Communication:** Forked oracles must not share mutable state via `RefCell`, `Mutex`, or similar mechanisms. All parallelism is embarrassingly parallel.

**Usage:** Thread pool can spawn worker threads, each with forked oracle, for embarrassingly parallel evaluation of large candidate sets.

---

## 4. Objective Functions

### 4.1 Facility Location

**Mathematical Definition:**

Given:
- Demand set $\mathcal{D} = \{1, 2, \ldots, m\}$
- Utility matrix $u_{i,s} \geq 0$ for demand $i \in \mathcal{D}$, candidate $s \in V$
- Demand weights $w_i \geq 0$

Objective:
$$
f_{\text{fac}}(S) = \sum_{i \in \mathcal{D}} w_i \cdot \max_{s \in S} u_{i,s}
$$

**Submodularity:** Each term $\max_{s \in S} u_{i,s}$ is submodular; non-negative weighted sums preserve submodularity.

**Data Schema:**
- `demand.parquet`: columns `[i: u32, w: f32]`
- `u.parquet`: columns `[i: u32, s: u32, u: f32]` (sparse, only non-zero entries)
- Row groups clustered by demand ID for sequential access

**Internal Representation:**

Compressed Sparse Row (CSR) format for demand-centric access:
- `row_ptr: Vec<usize>`: CSR row pointers
- `col_indices: Vec<ItemId>`: Candidate IDs
- `values: Vec<f32>`: Utility values
- `demand_weights: Vec<f32>`: $w_i$

**State Management:**
- `best_u: Vec<f32>`: Current best utility per demand, $b_i = \max_{s \in S} u_{i,s}$
- Initialized to zero
- Updated only during `commit()`

**Marginal Gain Computation (gain() method):**

```
Algorithm: gain(view, e)
Input: view (immutable SelectionView), e (candidate ItemId)
Output: Marginal gain Δ(e | S)

delta ← 0.0
for each demand i where u[i,e] > 0:
    improvement ← max(0, u[i,e] - best_u[i])
    delta ← delta + demand_weights[i] × improvement
return delta
```

**Commit Operation (commit() method):**

```
Algorithm: commit(e)
Input: e (selected ItemId)

for each demand i where u[i,e] > 0:
    best_u[i] ← max(best_u[i], u[i,e])
```

**Complexity:**
- `gain()`: $O(\text{nnz}(e))$ where $\text{nnz}(e)$ is number of non-zero utilities for candidate $e$
- `commit()`: $O(\text{nnz}(e))$
- Space: $O(m + \text{nnz})$ where $\text{nnz}$ is total non-zero entries

**SIMD Optimization:**

The inner loop is vectorizable using `std::simd`:
- Process utilities in chunks of 4-8 elements
- Vectorized max and weighted sum operations
- Typical speedup: 4-8x on modern CPUs with AVX2/AVX-512

### 4.2 Saturating Coverage

**Mathematical Definition:**

$$
f_{\text{sat}}(S) = \sum_{i \in \mathcal{D}} \phi_i\left(\sum_{s \in S} a_{i,s}\right)
$$

where:
- $a_{i,s} \geq 0$: contribution of candidate $s$ to demand $i$
- $\phi_i: \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$: increasing, concave function

**Common Concave Functions:**
- Logarithmic: $\phi(x) = \log(1 + x)$
- Square root: $\phi(x) = \sqrt{x}$
- Saturating: $\phi(x) = \min(x, \tau)$ for threshold $\tau$

**Submodularity:** Concave-over-modular composition is submodular.

**State:**
- `cumulative: Vec<f32>`: $x_i = \sum_{s \in S} a_{i,s}$ for each demand

**Marginal Gain:**

```
gain(view, e):
    delta ← 0.0
    for each demand i where a[i,e] > 0:
        delta ← delta + phi[i](cumulative[i] + a[i,e]) - phi[i](cumulative[i])
    return delta
```

**Commit:**

```
commit(e):
    for each demand i where a[i,e] > 0:
        cumulative[i] ← cumulative[i] + a[i,e]
```

**Concave Function Evaluation:**

For efficiency:
- Precompute lookup tables (LUT) with fine-grained bins
- Use piecewise polynomial approximations for smooth functions
- Minimize branching for SIMD-friendly execution
- Ensure strict monotonicity in LUT to prevent numerical violations

### 4.3 Log-Determinant (Determinantal Point Process)

**Mathematical Definition:**

Given symmetric positive semidefinite kernel matrix $K \in \mathbb{R}^{n \times n}$:

$$
f_{\log\det}(S) = \log \det(K_{S,S} + \epsilon I)
$$

where:
- $K_{S,S}$: principal submatrix indexed by $S$
- $\epsilon > 0$: regularization parameter (e.g., $10^{-6}$)

**Interpretation:** Logarithm of volume of parallelepiped; measures diversity in feature space.

**Submodularity:** Log-determinant of positive semidefinite matrix is submodular.

**Marginal Gain via Schur Complement:**

For $S \subseteq V$ and $e \notin S$:

$$
\Delta(e \mid S) = \log\left(K_{e,e} + \epsilon - K_{e,S}(K_{S,S} + \epsilon I)^{-1}K_{S,e}\right)
$$

Let $d = K_{e,e} + \epsilon - K_{e,S}(K_{S,S} + \epsilon I)^{-1}K_{S,e}$ (Schur complement's diagonal entry).

Then $\Delta(e \mid S) = \log(d)$.

**State Maintenance:**

Maintain Cholesky decomposition $L$ such that $K_{S,S} + \epsilon I = LL^T$:
- `L: Matrix<f64>`: Lower triangular Cholesky factor (**always f64**, f32 fails rapidly)
- `S_current: Vec<ItemId>`: Current selection for indexing
- `mean_diagonal: f64`: Mean of diagonal entries for numerical conditioning

**Incremental Update (during commit):**

```
Algorithm: commit(e)
Input: e (selected element)

1. Solve L x = K[S, e] via forward substitution
2. Compute d = K[e,e] + epsilon - dot(x, x)
3. If d ≤ 0 or isnan(d):
       Fallback to full Cholesky recomputation
       If still fails: disable log-det term, switch to Facility Location only
       Log degradation warning
4. Compute l_new = sqrt(d)
5. Extend L with new row/column: append x^T and l_new
```

**Numerical Safety Layers (Defense in Depth):**

**Layer 1: Regularization**
- Always add $\epsilon I$ with $\epsilon \in [10^{-6}, 10^{-4}]$
- Ensures positive definiteness

**Layer 2: Safe Schur Computation**

```rust
fn safe_schur_diagonal(
    K_ee: f64,
    K_eS: &[f64],
    L_inv_K_eS: &[f64],
    epsilon: f64,
    mean_diag: f64
) -> f64 {
    let x_norm_sq = L_inv_K_eS.iter().map(|&x| x * x).sum::<f64>();
    let d = K_ee + epsilon - x_norm_sq;

    // Epsilon clipping
    let d_safe = d.max(epsilon * 0.1);
    d_safe
}
```

**Layer 3: log1p for Small d**

When $d$ is small relative to mean diagonal:

```rust
let log_d = if d < 0.1 * mean_diagonal {
    // Use log1p for better numerical accuracy
    let adjusted = (d - mean_diagonal) / mean_diagonal;
    adjusted.ln_1p() + mean_diagonal.ln()
} else {
    d.ln()
};
```

**Layer 4: Immediate Refactorization on NaN**

```rust
fn commit_with_fallback(&mut self, e: ItemId) {
    match self.try_incremental_update(e) {
        Ok(()) => {},
        Err(NumericalError::NaN) => {
            warn!("Incremental Cholesky NaN; full recomputation");
            self.recompute_full_cholesky();
        },
        Err(NumericalError::Negative) => {
            warn!("Negative Schur complement; full recomputation");
            self.recompute_full_cholesky();
        }
    }
}
```

**Layer 5: Graceful Degradation**

After $N_{\text{fail}} = 5$ consecutive failures:

```rust
if self.consecutive_failures > 5 {
    error!("Log-determinant numerically unstable; disabling diversity term");
    self.mode = FallbackMode::FacilityLocationOnly;
    // Continue optimization with coverage only
}
```

**Low-Rank Approximation for Scale:**

For massive kernel matrices, use Nyström or random features:

$$K \approx ZZ^T$$

where $Z \in \mathbb{R}^{n \times r}$ with rank $r \ll n$.

Then compute:
$$\log \det(I + Z_S^T Z_S)$$

in $O(r^2 |S| + r^3)$ instead of $O(|S|^3)$.

### 4.4 Diversity-Aware Facility Location

**Combined Objective:**

$$
f_{\text{div}}(S) = f_{\text{fac}}(S) - \lambda \sum_{\{s,t\} \subseteq S} \text{sim}(s,t)
$$

where:
- $f_{\text{fac}}$: Facility Location (coverage)
- $\lambda > 0$: diversity penalty weight
- $\text{sim}(s,t) \geq 0$: similarity between candidates

**Note:** Penalty term may break strict submodularity depending on $\lambda$ and similarity structure. However, with careful tuning, the function remains "nearly submodular" and greedy performs empirically well.

**Implementation Strategy:**

1. Sparsify similarity: keep only top-$L$ neighbors per candidate ($L \approx 50$)
2. Maintain: $\sigma(S) = \sum_{\{s,t\} \subseteq S} \text{sim}(s,t)$
3. Incremental update: $\Delta\sigma = \sum_{s \in S} \text{sim}(e, s)$
4. Combined gain: $\Delta(e \mid S) = \Delta_{\text{fac}}(e \mid S) - \lambda \cdot \Delta\sigma(e \mid S)$

---

## 5. Optimization Algorithms

### 5.1 Lazy Greedy with Epoch-Based Stale Bound Elimination

**The Stale Bound Problem:**

In naive Lazy Greedy, outdated upper bounds persist in the heap indefinitely, causing unnecessary re-evaluations even when bounds are obsolete.

**Solution: Epoch Tracking**

Assign each cached bound a timestamp (epoch). When selection grows, increment global epoch. On heap pop, discard entries with outdated epochs.

**Data Structures:**

```rust
struct HeapEntry {
    upper_bound: f64,
    epoch: u64,      // When was this bound evaluated?
    element: ItemId,
}

// Implement Ord with deterministic tie-breaking
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.upper_bound.partial_cmp(&other.upper_bound)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.element.cmp(&other.element))  // Deterministic
    }
}

struct LazyGreedyState {
    heap: BinaryHeap<HeapEntry>,
    current_epoch: u64,
    view: SelectionView,
}
```

**Algorithm:**

```
Algorithm: LazyGreedy
Input: oracle, constraint, k
Output: Selection S

Initialize:
    heap ← empty max-heap
    current_epoch ← 0
    view ← empty SelectionView

    for each e in V:
        ub ← oracle.upper_bound(e)
        heap.push(HeapEntry { upper_bound: ub, epoch: 0, element: e })

Main Loop:
    for iteration in 1..=k:
        loop:
            if heap.is_empty():
                break  // No more candidates

            entry ← heap.pop()

            // Discard stale entries from previous epochs
            if entry.epoch < current_epoch:
                continue  // Skip, this bound is obsolete

            // Evaluate actual marginal gain
            delta ← oracle.gain(&view, entry.element)

            // Check constraint feasibility
            if not constraint.can_add(&view, entry.element):
                continue  // Not feasible

            // Check if this is definitively the maximum
            next_ub ← heap.peek().map(|e| e.upper_bound).unwrap_or(0.0)
            if delta + epsilon ≥ next_ub:
                // Confirmed best; commit
                oracle.commit(entry.element)
                view.in_set.insert(entry.element as usize)
                view.size += 1
                current_epoch += 1
                constraint.commit(entry.element)
                break  // Next iteration
            else:
                // Re-insert with updated bound and CURRENT epoch
                heap.push(HeapEntry {
                    upper_bound: delta,
                    epoch: current_epoch,  // Fresh timestamp
                    element: entry.element,
                })

Return: materialize_selection(oracle, &view)
```

**Epoch-Based Invariants:**

> **Invariant 1 (Validity):** Only heap entries with `epoch(entry) == current_epoch` are valid.
> Entries with `epoch(entry) < current_epoch` are stale and must be discarded on pop.
>
> **Invariant 2 (Increment):** `commit` increments `current_epoch` by 1, invalidating all existing heap entries.
>
> **Invariant 3 (TOCTOU Prevention):** When re-evaluating an element popped from the heap, the evaluation must use the **current** `SelectionView` (reflecting `current_epoch`). The upper bound stored with an earlier epoch is invalid for the current selection state.
>
> These invariants prevent accumulation of obsolete upper bounds, ensure correctness of lazy evaluation, and eliminate time-of-check-to-time-of-use races.

**Parallel Evaluation:**

During evaluation phase, compute marginal gains for top-M candidates in parallel:

```rust
let top_candidates: Vec<ItemId> = heap.iter()
    .take(parallelism * 2)
    .map(|entry| entry.element)
    .collect();

let mut gains: Vec<(ItemId, f64)> = top_candidates
    .par_iter()
    .map(|&e| (e, oracle.gain(&view, e)))
    .collect();

if config.determinism.fixed_order {
    gains.sort_by_key(|(e, _)| *e);  // Deterministic order
}
```

**Approximation Guarantee:** $(1-1/e)$ for cardinality, $1/2$ for general matroid.

**Empirical Performance:** Typically 5-10% of evaluations compared to standard greedy.

### 5.2 Stochastic Greedy

**Algorithm:**

At each iteration $t = 1, \ldots, k$:

1. Sample random subset $R_t \subseteq V \setminus S_{t-1}$ with size:
   $$|R_t| = \left\lceil \frac{n}{k} \log\frac{1}{\epsilon} \right\rceil$$

   (Mirzasoleiman et al. 2015 - see Section 12.2)

2. Find best element in sample:
   $$e_t = \arg\max_{e \in R_t} \Delta(e \mid S_{t-1})$$

3. Add to solution: $S_t = S_{t-1} \cup \{e_t\}$

**Approximation Guarantee:**

$$\mathbb{E}[f(S_k)] \geq \left(1 - \frac{1}{e} - \epsilon\right) f(S_{\text{opt}})$$

**Total Complexity:** $O(n \log(1/\epsilon))$ evaluations (vs $O(nk)$ for standard greedy).

**Deterministic Sampling:**

```
Algorithm: StochasticGreedy
Input: oracle, k, epsilon, seed

rng ← StdRng::seed_from_u64(seed)
view ← empty SelectionView

for iteration in 1..=k:
    sample_size ← ceil((n / k) * ln(1 / epsilon))
    available ← V \ {elements in view}
    sample ← reservoir_sample(&mut rng, available, sample_size)

    // Evaluate sample (parallelizable with fork())
    best ← sample.par_iter()
        .map(|&e| (e, oracle.gain(&view, e)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()

    oracle.commit(best.0)
    view.in_set.insert(best.0 as usize)
    view.size += 1

Return: materialize_selection(oracle, &view)
```

**Hybrid: Lazy Stochastic Greedy**

Apply Lazy strategy within each sample:
- Maintain epoch-based heap only for sampled candidates
- Reduces evaluations further when sample size is large

### 5.3 Continuous Greedy for Matroid Constraints (Maximum Weight **BASE** Selection)

**Core Idea:**

1. Relax discrete problem to continuous domain $[0,1]^n$
2. Optimize multilinear extension $F(x)$ via gradient ascent on matroid **BASE** polytope
3. Round fractional solution to integer solution

**Critical:** At each iteration, select a **BASE** (maximal independent set of size = rank), not just any independent set.

**Time Discretization:**

Divide interval $[0,1]$ into $T$ steps with $\Delta t = 1/T$.

**Gradient Estimation with View-Based Evaluation**

At fractional solution $x^{(t)}$, estimate gradient:

$$
\widehat{g}_i = \frac{1}{G} \sum_{j=1}^G \left[ f(R_j \cup \{i\}) - f(R_j) \right]
$$

where $R_j \sim x^{(t)}$.

**Variance Reduction via Common Random Numbers (Optional):**

For lower variance, use the **same** random sample set $R_1, \ldots, R_G$ when comparing different elements. This eliminates sample-to-sample noise and improves gradient estimation quality. Enable via `use_common_random_numbers: true` in configuration.

**CRITICAL: View-Based Evaluation (No Oracle Mutation)**

```
Algorithm: estimate_gradient
Input: oracle, x (fractional solution), G (samples), seed
Output: Gradient estimate g

g ← vector of zeros (length n)
rng ← StdRng::seed_from_u64(seed)

for sample_id in 1..=G:
    // Build TEMPORARY view without mutating oracle
    view ← SelectionView::new(n)
    for i in 0..n:
        if rng.gen::<f64>() < x[i]:
            view.in_set.insert(i)
            view.size += 1

    // Evaluate all elements with this TEMPORARY view
    for i in 0..n:
        if not view.in_set[i]:  // Element not in sample
            delta ← oracle.gain(&view, i)  // ✓ Immutable query
            g[i] ← g[i] + delta

// Average over samples
for i in 0..n:
    g[i] ← g[i] / G

return g
```

**Why This is Critical:**

**PRINCIPLE: The oracle is never mutated during gradient estimation.** All sampling and evaluation must use temporary views to preserve oracle correctness.

❌ **ANTI-PATTERN (mutates oracle):**
```rust
for sample in 1..=G {
    let mut R = Selection::new();
    for i in V {
        if random() < x[i] {
            oracle.commit(i);  // ❌ Mutates oracle state!
            R.add(i);
        }
    }
    // Oracle state is now corrupted
}
```

✓ **CORRECT PATTERN (view-based):**
```rust
for sample in 1..=G {
    let mut view = SelectionView::new(n);
    for i in V {
        if random() < x[i] {
            view.in_set.insert(i);  // ✓ Local view only
            view.size += 1;
        }
    }
    // Oracle remains untouched
    for i in V {
        if !view.in_set[i] {
            gain[i] += oracle.gain(&view, i);  // ✓ Immutable
        }
    }
}
```

**Direction Selection: Maximum Weight BASE (not just independent set)**

Given gradient $\widehat{g}$, find **base** $B$ of matroid $\mathcal{M}$ maximizing:

$$
B^{(t)} = \arg\max_{B \text{ is a BASE}} \sum_{i \in B} \widehat{g}_i
$$

**Critical Distinction:** We select a **BASE** (maximal independent set of size = rank), not just any independent set. Bases are vertices of the matroid base polytope.

**Greedy Algorithm for Max-Weight Base:**

```
Algorithm: max_weight_base
Input: matroid M, weights w
Output: Base B

order ← sort elements by weights descending
    (tie-break by ID for determinism)

state ← matroid.new_state()  // e.g., partition counters, union-find
base ← empty list

for e in order:
    if state.can_add(e) and |base| < matroid.rank():
        state.add(e)
        base.append(e)
    if |base| == matroid.rank():
        break

return base
```

**Fractional Solution Update:**

$$
x^{(t+1)}_i = \min\left(x^{(t)}_i + \frac{\mathbf{1}_{i \in B^{(t)}}}{T}, 1\right)
$$

Each component is clipped to $[0,1]$.

**Complete Continuous Greedy:**

```
Algorithm: ContinuousGreedy
Input: oracle, matroid, T (steps), G (grad samples), rounding_mode
Output: Selection S

n ← oracle.universe_size()
x ← vector of zeros (length n)
dt ← 1.0 / T

for t in 0..T:
    // Estimate gradient (VIEW-BASED, no oracle mutation)
    grad ← estimate_gradient(oracle, x, G, seed + t)

    // Find maximum weight BASE
    base ← matroid.max_weight_base(grad)

    // Update fractional solution
    for e in base:
        x[e] ← min(x[e] + dt, 1.0)

// Round to integer solution
items ← match rounding_mode:
    Pipage => pipage_rounding(x, matroid)  // Deterministic, partition matroid
    Swap   => swap_rounding(x, matroid)     // Probabilistic, general matroid

// Materialize: commit selected items to oracle
view ← SelectionView::from_items(items, n)
for e in items:
    oracle.commit(e)

return Selection { items, objective: oracle.evaluate(&view), ... }
```

**Approximation Guarantee:** $(1-1/e)$ for monotone submodular + matroid constraint.

### 5.4 Rounding: Pipage vs Swap

**Pipage Rounding (Partition Matroid Only)**

Deterministic rounding for partition matroids. Within each partition, pairs of fractional variables are adjusted until all are integral.

**Algorithm:**

```
Algorithm: pipage_rounding
Input: x (fractional solution), partition_matroid
Output: Integral solution S

while exists fractional variables:
    Pick partition j with two fractional vars p, q (0 < x[p], x[q] < 1)

    // Adjust along direction preserving expectation
    alpha_max ← min(1 - x[p], x[q])
    alpha_min ← max(-x[p], -(1 - x[q]))

    // Submodularity ⇒ F(x) concave in alpha ⇒ extremum optimal
    if F(x + alpha_max * (e_p - e_q)) ≥ F(x + alpha_min * (e_p - e_q)):
        x[p] += alpha_max
        x[q] -= alpha_max
    else:
        x[p] += alpha_min
        x[q] -= alpha_min

return {i : x[i] == 1}
```

**Property:** Maintains $\mathbb{E}[f(x)]$ non-decreasing.

**Swap Rounding (General Matroid)**

Probabilistic rounding via base decomposition.

1. Decompose $x$ into convex combination of bases:
   $$x = \sum_{\ell=1}^L \lambda_\ell \mathbf{1}_{B_\ell}, \quad \sum_\ell \lambda_\ell = 1$$

2. Greedily construct bases: given residual $x'$, find max-weight base $B$ with weights $x'$, set $\lambda$ maximal, subtract, repeat.

3. Sample base $B_\ell$ with probability $\lambda_\ell$

4. Apply swap operations to merge bases stochastically

**Approximation:** Expected objective within $(1-1/e)$ factor.

**Determinism:** Use seeded RNG derived from global seed: `seed_swap = seed ^ 0x5377617052746E64` (hex for "SwapRtnd").

### 5.5 Sieve-Streaming

**Use Case:** Single-pass streaming, memory $O(k \log(1/\epsilon))$.

**Algorithm:**

Maintain thresholds $\tau_1 > \tau_2 > \cdots > \tau_L$ (geometric sequence).

For each element $e$ in stream:
- Compute $\Delta(e \mid S_\tau)$ for current solution at threshold $\tau$
- If $\Delta(e \mid S_\tau) \geq \tau$ and $|S_\tau| < k$: add $e$ to $S_\tau$

**Approximation:** $(1/2 - \epsilon)$ for monotone submodular + cardinality.

**Memory:** $O(kL)$ where $L = O(\log(1/\epsilon))$.

### 5.6 Standard Termination Conditions

All algorithms support the following standard termination conditions. Optimization terminates when **any** of these conditions is satisfied:

**1. Cardinality Reached:** $|S| = k$
- Primary stopping condition for cardinality constraints
- Guaranteed to terminate in exactly $k$ iterations

**2. Upper Bound Below Threshold:** $\text{next\_ub} < \tau$
- For Lazy Greedy: `heap.peek().unwrap_or(0.0) < tau`
- Where $\tau$ is a user-defined threshold (e.g., $10^{-3}$)
- Stops when remaining candidates contribute negligibly

**3. Consecutive Stagnation:** Marginal gain $\Delta \approx 0$ for $m$ consecutive iterations
- Default: $m = 3$, threshold $= 10^{-6}$
- Condition: $|\Delta_t| < \epsilon \cdot f(S_t)$ for $t-m+1, \ldots, t$
- Indicates practical saturation even if $k$ not reached

**4. Timeout:** Wall-clock time exceeds budget
- Configurable: seconds, minutes, or hours
- Ensures responsiveness in production systems
- Returns best solution found so far

**Configuration:**

```rust
pub struct TerminationConfig {
    pub max_iterations: usize,        // k for cardinality (required)
    pub ub_threshold: Option<f64>,    // Optional: stop if next_ub < tau
    pub stagnation_window: usize,     // m consecutive iters (default: 3)
    pub stagnation_epsilon: f64,      // Relative tolerance (default: 1e-6)
    pub timeout: Option<Duration>,    // Optional: wall-clock limit
}
```

**Example:**

```rust
let config = TerminationConfig {
    max_iterations: 250,
    ub_threshold: Some(1e-3),
    stagnation_window: 3,
    stagnation_epsilon: 1e-6,
    timeout: Some(Duration::from_secs(600)),  // 10 minutes
};
```

**Audit Logging:**

When termination occurs, log the specific reason:
- `"termination_reason": "cardinality_reached"`
- `"termination_reason": "upper_bound_threshold"`
- `"termination_reason": "stagnation"`
- `"termination_reason": "timeout"`

---

## 6. Constraint Systems

### 6.1 Constraint Trait

```rust
pub trait Constraint {
    /// Can element e be added to current selection?
    fn can_add(&self, view: &SelectionView, e: ItemId) -> bool;

    /// Notify constraint of committed element
    fn commit(&mut self, e: ItemId);

    /// Reset constraint state
    fn reset(&mut self);
}
```

### 6.2 Cardinality Constraint

Simplest: $|S| \leq k$.

```rust
impl Constraint for CardinalityConstraint {
    fn can_add(&self, view: &SelectionView, _e: ItemId) -> bool {
        view.size < self.k
    }

    fn commit(&mut self, _e: ItemId) {
        // No internal state beyond view.size
    }

    fn reset(&mut self) {}
}
```

### 6.3 Knapsack Constraint with Explicit Approximation Modes

**Mode Comparison:**

| Aspect | Practical | Theoretical |
|--------|-----------|-------------|
| **Approximation** | Heuristic (no guarantee) | $(1-1/e) \approx 0.632$ guaranteed |
| **Time Complexity** | $O(nk)$ + small enumeration | $O(Tn \cdot G)$ where $T \approx 100$, $G \approx 500$ |
| **Reproducibility** | Deterministic (fixed-order) | Deterministic with seeded RNG |
| **Use Case** | Speed-critical production | Provable quality, benchmarking |

**Two Distinct Modes:**

**Mode 1: Practical (Fast, Weaker Guarantee)**

- Enumerate best 1-3 individual items (highest single-item value)
- Run cost-benefit ratio greedy: $\arg\max_e \frac{\Delta(e \mid S)}{\text{cost}(e)}$
- Return better of enumerated singletons vs greedy solution
- **Approximation:** No $(1-1/e)$ guarantee, but fast and often excellent in practice
- **Use when:** Speed critical, heuristic quality acceptable

**Mode 2: Theoretical (Slow, Provable (1-1/e))**

- Treat as weighted matroid with cost-scaled rank function
- Apply Continuous Greedy with multilinear relaxation
- **Approximation:** $(1-1/e)$ (Sviridenko 2004)
- **Use when:** Provable quality required, can afford extra computation

**Implementation Roadmap for Theoretical Mode:**

*Phase 1 (Initial Release):* Partition Matroid + budget slicing approximation. Split budget into slices, treat each as a partition constraint. Weaker guarantee but tractable.

*Phase 2 (Full Implementation):* True Continuous Greedy over knapsack polytope with cost-scaled relaxation. Requires matroid rank function that incorporates costs. Enables full $(1-1/e)$ Sviridenko guarantee.

Users should expect Phase 1 initially; Phase 2 delivers the provable guarantee.

**Implementation:**

```rust
struct KnapsackConstraint {
    budget: f64,
    cost_fn: Arc<dyn Fn(ItemId) -> f64 + Send + Sync>,
    mode: KnapsackMode,
    used: f64,
}

impl Constraint for KnapsackConstraint {
    fn can_add(&self, _view: &SelectionView, e: ItemId) -> bool {
        self.used + (self.cost_fn)(e) <= self.budget
    }

    fn commit(&mut self, e: ItemId) {
        self.used += (self.cost_fn)(e);
    }

    fn reset(&mut self) {
        self.used = 0.0;
    }
}
```

**Why `Arc<dyn Fn>` is Essential:**

- Enables dynamic cost computation (database lookups, runtime rules)
- Supports closures capturing environment
- Much more flexible than `fn` pointers
- Thread-safe with `Send + Sync`

### 6.4 Partition Matroid

**Definition:**

Ground set partitioned: $V = C_1 \sqcup C_2 \sqcup \cdots \sqcup C_m$.

Independence: $S \in \mathcal{I}$ iff $|S \cap C_j| \leq \text{cap}_j$ for all $j$.

**Implementation:**

```rust
struct PartitionMatroid {
    part_fn: Arc<dyn Fn(ItemId) -> usize + Send + Sync>,
    capacities: Vec<usize>,
    counts: Vec<usize>,  // Current per-partition counts
}

impl Constraint for PartitionMatroid {
    fn can_add(&self, _view: &SelectionView, e: ItemId) -> bool {
        let part = (self.part_fn)(e);
        self.counts[part] < self.capacities[part]
    }

    fn commit(&mut self, e: ItemId) {
        let part = (self.part_fn)(e);
        self.counts[part] += 1;
    }

    fn reset(&mut self) {
        self.counts.fill(0);
    }
}

impl Matroid for PartitionMatroid {
    fn rank(&self) -> usize {
        self.capacities.iter().sum()
    }

    fn max_weight_base(&self, weights: &[f64]) -> Vec<ItemId> {
        // For each partition, greedily select top-cap[j] by weight
        let mut base = Vec::new();

        for j in 0..self.capacities.len() {
            let mut part_items: Vec<ItemId> = (0..weights.len())
                .filter(|&i| (self.part_fn)(i as ItemId) == j)
                .map(|i| i as ItemId)
                .collect();

            part_items.sort_by(|&a, &b| {
                weights[b as usize].partial_cmp(&weights[a as usize])
                    .unwrap_or(Ordering::Equal)
                    .then(a.cmp(&b))  // Deterministic tie-break
            });

            base.extend(part_items.iter()
                .take(self.capacities[j]));
        }

        base
    }
}
```

### 6.5 Graphic Matroid

**Definition:**

For undirected graph $G = (V, E)$, independent sets are forests (acyclic edge subsets).

**Application:** Select edges with maximum submodular value while maintaining acyclicity.

**Implementation:**

Use Union-Find for cycle detection:

```rust
struct GraphicMatroid {
    n_vertices: usize,
    edge_endpoints: Vec<(usize, usize)>,  // e -> (u, v)
    uf: UnionFind,
}

impl Constraint for GraphicMatroid {
    fn can_add(&self, _view: &SelectionView, e: ItemId) -> bool {
        let (u, v) = self.edge_endpoints[e as usize];
        !self.uf.connected(u, v)  // Would create cycle?
    }

    fn commit(&mut self, e: ItemId) {
        let (u, v) = self.edge_endpoints[e as usize];
        self.uf.union(u, v);
    }

    fn reset(&mut self) {
        self.uf = UnionFind::new(self.n_vertices);
    }
}
```

**Max-Weight Base:** Kruskal's algorithm: sort edges by weight descending, add greedily if no cycle.

---

## 7. Data Management and I/O

### 7.1 Parquet Schema Design with Row-Group Optimization

**Demand Data (`demand.parquet`):**

| Column | Type | Description | Encoding | Notes |
|--------|------|-------------|----------|-------|
| `i` | `u32` | Demand ID | Dictionary | Enable predicate pushdown |
| `w` | `f32` | Weight | Plain | Statistics for filtering |

**Utility Data (`u.parquet`):**

| Column | Type | Description | Encoding | Notes |
|--------|------|-------------|----------|-------|
| `i` | `u32` | Demand ID | Dictionary | Cluster row groups by this |
| `s` | `u32` | Candidate ID | Dictionary | Secondary sort key |
| `u` | `f32` | Utility | Plain | Store only non-zero |

**Row Group Organization:**

- **Cluster by demand ID `i`:** Sequential access during marginal gain computation
- **Row group size:** 128MB-256MB (balance parallelism and memory)
  - Configurable via environment variable: `SUBMOD_ROW_GROUP_TARGET_MB` (default: 192)
- **Statistics:** Min/max for each column enables Parquet row-group skipping
- **Predicate pushdown:** Filter `i IN (...)` pushed to Parquet reader, skips irrelevant row groups

**Compression:**

- Snappy: Fast decompression, moderate ratio (default)
- Zstd level 3: Better compression, slightly slower (archival)

**Operational Best Practices:**

1. **Schema Minimalism:** Store only essential columns (`i`, `s`, `u`). Drop auxiliary columns not used in computation to enable Parquet column pruning.

2. **Row Group Sizing:** Target 128-256MB per row group. Configurable via `SUBMOD_ROW_GROUP_TARGET_MB` environment variable (default: 192). Smaller groups increase parallelism but add overhead; larger groups reduce metadata but limit pruning granularity.

3. **Prefetch Depth Auto-Tuning:** Monitor `submod_io_wait_seconds` metric. If I/O stalls dominate, increase async prefetch queue depth (typical range: 2-8 shards ahead). Link prefetch depth to available memory budget dynamically.

### 7.2 Data Validation (Fail-Fast)

**Mandatory Checks:**

1. **Non-negativity:** All $w_i, u_{i,s}, \text{cost}(s) \geq 0$
2. **No NaN/Inf:** Reject IEEE-754 special values
3. **ID bounds:** $i < n_{\text{demands}}$, $s < n_{\text{candidates}}$
4. **Schema conformance:** Column types and names match spec

**Duplicate Handling:**

If multiple $(i, s)$ pairs:
- Default: max utility
- Configurable: sum, average, or error

**Zero Filtering:**

Explicitly drop entries with $u = 0$ during CSR construction.

### 7.3 Sparse Matrix Construction

**CSR Format (Row-Major):**

```rust
struct CSRMatrix {
    row_ptr: Vec<usize>,        // Length: n_demands + 1
    col_indices: Vec<ItemId>,   // Length: nnz
    values: Vec<f32>,           // Length: nnz
}
```

Access row $i$: indices `col_indices[row_ptr[i]..row_ptr[i+1]]`.

**CSC Format (Column-Major):**

Transpose when candidate-centric access dominates (rare for Facility Location).

**Memory Estimate:**

Per non-zero: 12 bytes + index overhead.
100 billion non-zero: ~120-150 GB RAM.

### 7.4 Sharding and Prefetch

**Hash-Based Sharding:**

Partition into $P$ shards ($P \approx 128$) by:
$$\text{shard}(i) = \text{hash}(i) \mod P$$

Files: `u_shard_000.parquet`, ..., `u_shard_127.parquet`

**Parallel Loading:**

```rust
use rayon::prelude::*;
use crossbeam::channel::bounded;

let (sender, receiver) = bounded(parallelism * 2);

// Producer thread
thread::spawn(move || {
    (0..n_shards).into_par_iter().for_each(|shard| {
        let matrix = load_shard(shard);  // Parquet read with row-group pruning
        sender.send(matrix).unwrap();
    });
});

// Consumer (solver)
for matrix in receiver {
    oracle.ingest(matrix);
}
```

**Prefetch Strategy:**

While processing shard $i$, async load shards $i+1, i+2$.
Bound channel capacity to prevent memory explosion.

### 7.5 Approximate TopK Compression (Optional)

**Motivation:** For candidates with thousands of non-zero utilities, full evaluation is expensive.

**Strategy:** Retain only top-$L$ utilities per candidate ($L \approx 50$).

**Construction:**

For each candidate $s$: sort demands by $u_{i,s}$ descending, keep top-$L$.

**Impact:**

- Memory: Bounded degree, drastically reduced
- Accuracy: Lower bound on marginal gain (missing small contributions)
- Theory: Approximation degrades, but empirically robust with $L \geq 50$

**When to Use:**

- Stochastic Greedy with massive candidate sets
- Preliminary screening
- Real-time constraints

---

## 8. Numerical Stability and Robustness

### 8.1 Floating-Point Type Selection

**Dtype Assignment Table:**

| Objective Function | Weight Type | Rationale |
|-------------------|-------------|-----------|
| Facility Location | `f32` | Sums of products; sufficient precision |
| Saturating Coverage | `f32` | Concave transforms; well-conditioned |
| Log-Determinant | `f64` | Cholesky accumulates errors; f32 fails |
| Continuous Greedy (gradient estimation) | `f64` | Accumulated sampling noise requires higher precision |

**Policy:**

- **Facility Location, Saturating Coverage:** `f32`
  - Reason: Sums of products; f32 precision sufficient
  - Benefit: 2x memory reduction, wider SIMD (8 lanes vs 4)

- **Log-Determinant:** `f64` (mandatory)
  - Reason: Cholesky accumulates errors; f32 fails rapidly
  - Even with f64, epsilon regularization essential

- **Continuous Greedy:** `f64` for gradient computation
  - Reason: Summing many small gradient samples amplifies rounding errors
  - Use f64 for intermediate gradient accumulation, even if oracle uses f32

**Type Monomorphization:**

Avoid runtime branching on dtype:

```rust
match config.dtype {
    DType::F32 => solve_mono::<f32>(oracle),
    DType::F64 => solve_mono::<f64>(oracle),
}

fn solve_mono<F: Float>(oracle: impl SubmodularOracle<Weight=F>) {
    // Hot loop uses concrete type F; full SIMD optimization
}
```

This enables aggressive compiler optimization without type erasure overhead.

### 8.2 Log-Determinant Robustness (Defense in Depth)

**Failure Modes:**

1. Negative Schur complement: $d < 0$ (not positive definite)
2. NaN propagation: Catastrophic cancellation
3. Overflow: Large kernel values

**Layer 1: Regularization**

$$K \leftarrow K + \epsilon I, \quad \epsilon \in [10^{-6}, 10^{-4}]$$

**Layer 2: Safe Schur Computation**

```rust
fn safe_schur_diagonal(
    K_ee: f64,
    K_eS: &[f64],
    L_inv_K_eS: &[f64],
    epsilon: f64
) -> f64 {
    let x_norm_sq: f64 = L_inv_K_eS.iter().map(|&x| x * x).sum();
    let d = K_ee + epsilon - x_norm_sq;

    // Clip to epsilon floor
    d.max(epsilon * 0.1)
}
```

**Layer 3: log1p for Small d**

When $d$ is small relative to mean diagonal $\mu$:

```rust
let log_d = if d < 0.1 * mean_diagonal {
    let adjusted = (d - mean_diagonal) / mean_diagonal;
    adjusted.ln_1p() + mean_diagonal.ln()
} else {
    d.ln()
};
```

**Layer 4: Immediate Refactorization on NaN**

```rust
match self.try_incremental_cholesky(e) {
    Ok(delta) => delta,
    Err(NumericalError::NaN) => {
        warn!("NaN detected; full Cholesky refactorization");
        self.recompute_full_cholesky();
        self.compute_marginal(e)  // Retry
    }
}
```

**Layer 5: Graceful Degradation**

After 5 consecutive failures:

```rust
if self.failures > 5 {
    error!("Log-det unstable; disabling diversity term");
    self.mode = FallbackMode::FacilityOnly;
}
```

### 8.3 Summation Accuracy

**Problem:** Parallel reduction with different thread orders yields different sums (non-associative).

**Solution 1: Kahan Summation**

```rust
fn kahan_sum(values: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut c = 0.0;  // Compensation

    for &v in values {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}
```

**Solution 2: Pairwise Summation**

Recursively sum pairs: error bound $O(\log n \cdot \epsilon)$ vs $O(n \cdot \epsilon)$.

**Deterministic Parallel Sum:**

```rust
fn deterministic_parallel_sum(values: &[f32], fixed_order: bool) -> f32 {
    if !fixed_order {
        return values.par_iter().sum();  // Fast, non-deterministic
    }

    let chunk_size = 1024;
    let mut partials: Vec<f32> = values
        .par_chunks(chunk_size)
        .map(|chunk| chunk.iter().sum())
        .collect();

    // Sequential sum of partials (deterministic)
    partials.iter().sum()
}
```

**Mixed-Precision Aggregation Policy:**

For maximum reproducibility in deterministic mode:
- **Storage:** Use native oracle precision (f32 for Facility Location, f64 for Log-Determinant)
- **Aggregation:** Always use **f64 for reductions** when `determinism.fixed_order = true`
- **Final Output:** Cast back to oracle precision after fixed-order aggregation

Example:
```rust
if config.determinism.fixed_order {
    // Aggregate in f64 for reproducibility
    let sum_f64: f64 = partials.iter().map(|&v| v as f64).sum();
    sum_f64 as f32  // Cast back to storage precision
} else {
    // Fast path: native precision
    partials.iter().sum()
}
```

This eliminates floating-point non-associativity across parallelism levels while maintaining storage efficiency.

**Deterministic Hash for Audit Integrity:**

For audit log verification, compute deterministic hash of selection sequence:
```rust
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn compute_selection_hash(items: &[ItemId], seed: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);  // Include seed for reproducibility
    for &item in items {
        item.hash(&mut hasher);
    }
    hasher.finish()
}
```

**Fixed Hash Range:** Output is `u64` (0 to 2^64-1). Never hash float values (non-deterministic). Hash only:
- ItemId sequence (deterministic)
- RNG seed
- Algorithm name
- Configuration parameters (as integers or strings)

This enables CI snapshot comparison: "Run with seed 42 → hash must be 0x1a2b3c4d5e6f7890"

### 8.4 Constraint Arithmetic

**Knapsack Budget Comparison:**

```rust
// ❌ BAD
if used == budget { ... }

// ✓ GOOD
const EPSILON: f64 = 1e-9;
if used + EPSILON * budget.abs() >= budget { ... }
```

**Partition Counts:** Integer counters; no FP issues.

---

## 9. Deterministic Execution Framework

### 9.1 Reproducibility Requirements

**Definition:** Identical inputs → identical outputs:

1. Identical selection sequence $S_1, S_2, \ldots, S_k$
2. Identical objective value (within FP epsilon)
3. Identical audit log hashes

**Across:**
- Multiple runs, same machine
- Different parallelism (1 vs 64 threads)
- Different platforms (aspirational; FP differences may preclude exact match)

### 9.2 Configuration

```rust
pub struct Determinism {
    pub seed: u64,
    pub fixed_order: bool,      // Fix parallel reduction order
    pub tie_breaking: TieBreak,
}

impl Default for Determinism {
    fn default() -> Self {
        Self {
            seed: 42,
            fixed_order: true,
            tie_breaking: TieBreak::ById,
        }
    }
}
```

### 9.3 RNG Hierarchy

**Seeding Strategy:**

1. **Master seed:** User-provided (e.g., 42)
2. **Algorithm seed:** `master ^ ALGO_TAG` where `ALGO_TAG` is unique per algorithm
3. **Iteration seed:** `algo_seed + iteration_num`

Example (Stochastic Greedy):

```rust
const STOCHASTIC_TAG: u64 = 0x5374476565647947; // "StGreedyG" in hex

let algo_seed = config.seed ^ STOCHASTIC_TAG;

for iter in 0..k {
    let iter_seed = algo_seed.wrapping_add(iter as u64);
    let mut rng = StdRng::seed_from_u64(iter_seed);

    let sample = sample_uniform(&mut rng, &candidates, sample_size);
    // ...
}
```

**Why Hierarchical:**

- Different algorithms don't correlate
- Iterations get unique, reproducible seeds
- Platform-independent (StdRng is portable)

### 9.4 Parallel Reduction Ordering

**Problem:**

```rust
// Non-deterministic
let sum: f32 = values.par_iter().map(compute).sum();
```

**Solution:**

```rust
if config.determinism.fixed_order {
    let mut results: Vec<(ItemId, f32)> = values
        .par_iter()
        .map(|&e| (e, compute(e)))
        .collect();

    results.sort_by_key(|(e, _)| *e);  // Deterministic order

    let sum = results.iter().map(|(_, v)| v).sum();
} else {
    // Fast path
    let sum = values.par_iter().map(compute).sum();
}
```

**Trade-off:** ~10% slowdown for determinism.

### 9.5 Heap Tie-Breaking

```rust
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.config.tie_breaking {
            TieBreak::ById => {
                self.ub.partial_cmp(&other.ub)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| self.element.cmp(&other.element))
            },
            // ... other modes
        }
    }
}
```

**Recommendation:** Default `TieBreak::ById` for simplicity and performance.

### 9.6 Platform Independence (Aspirational)

**Challenges:** FP operations differ subtly across CPU architectures, compilers, FMA instructions.

**Mitigation:**

1. Use `StdRng` (platform-independent)
2. Avoid platform intrinsics; use portable `std::simd`
3. Accept objective value differences $< 10^{-6}$ across platforms
4. Hash only integer decisions (ItemId sequence), not float values

**Practical Stance:**

- **Intra-platform reproducibility:** Guaranteed
- **Cross-platform:** Best-effort; bitwise match not guaranteed

---

## 10. Monitoring and Observability

### 10.1 Prometheus Metrics

**Naming Convention:**

All metrics follow these rules:
1. **snake_case** with `submod_` prefix
2. **Unit suffix** for measurements: `_seconds`, `_bytes`, `_total` (for counters)
3. **Labels** in curly braces for dimensions (e.g., `{algorithm="LazyGreedy"}`)

**Counters:**

| Metric | Description |
|--------|-------------|
| `submod_gain_eval_total{algorithm}` | Total marginal gain evaluations |
| `submod_commit_total{algorithm}` | Elements committed |
| `submod_heap_pop_total` | Lazy Greedy heap pops |
| `submod_heap_reinsert_total` | Re-insertions |
| `submod_fallback_total{reason}` | Log-det fallbacks |

**Gauges:**

| Metric | Description |
|--------|-------------|
| `submod_objective_value` | Current objective |
| `submod_selection_size` | Current $|S|$ |
| `submod_upper_bound_max` | Max upper bound in heap |
| `submod_gap_estimate` | Estimated optimality gap |

**Histograms:**

| Metric | Description |
|--------|-------------|
| `submod_gain_compute_seconds` | Marginal gain latency |
| `submod_iteration_seconds` | Per-iteration wall time |
| `submod_io_wait_seconds` | I/O stall time |

**Additional Metrics for Continuous Greedy:**

| Metric | Description |
|--------|-------------|
| `submod_grad_variance` | Gradient estimate variance |
| `submod_base_weight_sum` | $\sum_{i \in B_t} \widehat{g}_i$ |
| `submod_rounding_loss` | $F(x^{(T)}) - f(S)$ |

**Exporter:** HTTP endpoint at `/metrics` on configurable port (default 9090).

### 10.2 Structured Audit Logs (JSON Lines)

**Standard Schema (Mandatory Fields):**

Every iteration log entry must include:
- `iteration` (integer): Iteration number (1-indexed)
- `selected_element` (integer): Chosen ItemId
- `marginal_gain` (float): $\Delta(e \mid S)$ for selected element
- `selection_size` (integer): Current $|S|$ after selection
- `objective_value` (float): Current $f(S)$
- `algorithm` (string): Algorithm name (e.g., "LazyGreedy", "StochasticGreedy")
- `rng_seed` (integer): RNG seed for this iteration (for reproducibility)
- `timestamp_ms` (integer): Wall-clock milliseconds since start
- `git_hash` (string, optional): Git commit hash of running code

**Named Thresholds (for debugging and auditing):**

All numerical thresholds must be logged with semantic names:
- `lazy_epsilon` (float): ε-approximation tolerance for Lazy Greedy
- `stagnation_threshold` (float): Relative tolerance for consecutive stagnation detection
- `ub_threshold` (float): Upper bound stopping threshold τ
- `early_stop_reason` (string): Human-readable reason ("next_ub_below_tau", "stagnation", "cardinality", "timeout")

Example log entry with thresholds:
```json
{
  "iteration": 250,
  "termination_reason": "upper_bound_threshold",
  "next_ub": 0.00085,
  "ub_threshold": 0.001,
  "threshold_name": "tau_early_stop"
}
```

This enables post-mortem analysis: "Why did it stop at iteration 250?" → "Because next_ub (0.00085) fell below tau (0.001)"

**Iteration Event:**

```json
{
  "iteration": 42,
  "selected_element": 1337,
  "marginal_gain": 123.456,
  "upper_bound": 125.0,
  "num_evaluations": 3,
  "selection_size": 42,
  "objective_value": 5432.1,
  "constraint_status": {
    "budget_used": 0.85,
    "partition_counts": [10, 15, 17]
  },
  "algorithm": "LazyGreedy",
  "rng_seed": 42,
  "timestamp_ms": 15234,
  "git_hash": "a3f21c9"
}
```

**Contribution Breakdown (Facility Location):**

```json
{
  "iteration": 42,
  "selected_element": 1337,
  "top_demand_contributions": [
    {"demand_id": 5, "contribution": 23.4, "utility": 0.95},
    {"demand_id": 102, "contribution": 18.7, "utility": 0.88}
  ],
  "num_demands_covered": 47
}
```

**Runner-Up Candidates (Sensitivity Analysis & Counterfactuals):**

For Lazy Greedy, record top-K evaluated candidates that were **not** selected. This enables counterfactual analysis ("what if we had chosen element 999 instead?").

```json
{
  "iteration": 42,
  "selected_element": 1337,
  "counterfactuals": [
    {"element": 999, "marginal_gain": 122.1, "deficit": 1.356, "reason": "lower_than_selected"},
    {"element": 1500, "marginal_gain": 120.8, "deficit": 2.656, "reason": "lower_than_selected"},
    {"element": 2048, "marginal_gain": 0.0, "deficit": 123.456, "reason": "infeasible"}
  ]
}
```

**Fields:**
- `element`: Candidate ItemId
- `marginal_gain`: Evaluated $\Delta(e \mid S)$
- `deficit`: Difference from selected element's gain
- `reason`: Why not selected ("lower_than_selected", "infeasible", "stale_bound")

This is especially valuable for Lazy Greedy where many candidates are evaluated but only one is chosen per iteration.

**Fallback Event:**

```json
{
  "event": "fallback",
  "component": "log_determinant",
  "reason": "incremental_cholesky_nan",
  "element": 2048,
  "action": "full_recomputation"
}
```

### 10.3 Explainability

**Coverage Report:** For each selected $s$, list demands where $u_{i,s}$ was best.

**Residual Demand:** Demands with $\max_{s \in S} u_{i,s} < \theta$ (uncovered).

**Diversity Metrics (Log-Determinant):**
- Determinant value
- Pairwise similarities heatmap
- Eigenvalue spectrum of $K_{S,S}$

**Objective Curve:** Plot $f(S_t)$ vs $t$ (diminishing returns visualization).

**Theoretical Gap Bound:**

$$\text{gap} \leq \max_{e \notin S} \Delta(e \mid S)$$

Report at each iteration; justifies early stopping.

**Curvature κ (Early Stopping Justification):**

For modular functions (additive), curvature $\kappa = 0$; for strongly submodular, $\kappa$ is large. Small $\kappa$ indicates rapid saturation with few elements, providing theoretical justification for early stopping.

---

## 11. Quick Start with Minimal Example

**Why This Matters:** OSS adoption hinges on a runnable minimal example. "動く最小例" proves the platform works and provides a template for users.

### 11.1 Minimal Dataset (10KB scale)

**Problem:** Select 10 candidates from 50 to maximize coverage of 100 demand points.

**Files:**
- `demand_mini.parquet`: 100 rows, columns `[i: u32, w: f32]`
- `utility_mini.parquet`: 500 rows (10% density), columns `[i: u32, s: u32, u: f32]`

**Generation Script (Python):**
```python
import numpy as np
import pandas as pd

n_demands, n_candidates = 100, 50
demands = pd.DataFrame({
    'i': np.arange(n_demands, dtype=np.uint32),
    'w': np.random.rand(n_demands).astype(np.float32)
})

# Sparse utility: ~10% density
rows, cols = np.where(np.random.rand(n_demands, n_candidates) < 0.1)
utilities = pd.DataFrame({
    'i': rows.astype(np.uint32),
    's': cols.astype(np.uint32),
    'u': np.random.rand(len(rows)).astype(np.float32)
})

demands.to_parquet('demand_mini.parquet', index=False)
utilities.to_parquet('utility_mini.parquet', index=False)
```

### 11.2 Quick Start Command

```bash
# Build and run
cargo run --release -- \
  --algorithm lazy_greedy \
  --oracle facility_location \
  --demand demand_mini.parquet \
  --utility utility_mini.parquet \
  --k 10 \
  --seed 42 \
  --output selection.json \
  --audit audit.jsonl

# Expected output:
# Objective: ~45.2 (±0.1 due to data randomness)
# Selection: [3, 7, 12, 18, 23, 29, 34, 38, 41, 47]
# Hash: 0x... (deterministic with seed 42)
```

### 11.3 Verification Checklist

After running, verify:
1. **Coverage Curve:** Plot `f(S_t)` vs `t` → should show diminishing returns
2. **Lazy Efficiency:** Check `num_evaluations / (n * k)` → should be < 10% for this dataset
3. **Determinism:** Re-run with same seed → identical selection sequence and hash
4. **Audit Log:** Parse `audit.jsonl` → all mandatory fields present

This 10KB example runs in < 1 second and validates the core pipeline end-to-end.

---

## 12. Implementation Roadmap

### 12.1 Phased Development

**Phase 1: Core + Facility Location (4-6 weeks)**

Deliverables:
- `submod-core` crate: traits, types, determinism
- `FacilityLocation` oracle with CSR
- Lazy Greedy with epoch-based heap
- Stochastic Greedy
- Cardinality constraint
- Unit tests: submodularity, correctness, determinism

**Phase 2: Data I/O + Parallel (3-4 weeks)**

Deliverables:
- `submod-io`: Parquet with predicate pushdown, row-group indexing
- Parallel evaluation (Rayon)
- Fixed-order reduction
- Benchmarks on 1M+ problems

**Phase 3: Additional Objectives (4-5 weeks)**

Deliverables:
- Saturating Coverage
- Log-Determinant with full safety stack
- Diversity-aware Facility Location
- Approximation guarantee validation

**Phase 4: Matroid + Continuous Greedy (6-8 weeks)**

Deliverables:
- Matroid trait: Partition, Graphic
- Continuous Greedy with **view-based gradient estimation**
- Pipage rounding (partition)
- Swap rounding (general)
- Knapsack: practical + theoretical modes
- Empirical $(1-1/e)$ validation

**Phase 5: Python + Service (3-4 weeks)**

Deliverables:
- PyO3 bindings with **GIL release** for long computations
- NumPy/Pandas integration via **zero-copy Arrow/FFI**
- CLI (clap)
- gRPC (tonic) + REST (axum)
- Auth, job queue, metrics

**Python Binding Requirements:**
1. **GIL Release:** All solver entry points must use `Python::allow_threads()` to release the Global Interpreter Lock during computation. This enables true parallelism and prevents blocking other Python threads.
2. **Zero-Copy I/O:** Prefer Arrow memory layout for NumPy/Pandas interchange. Avoid unnecessary copies between Rust and Python heaps.
3. **Error Propagation:** Convert Rust `Result<T, E>` to Python exceptions with detailed context (file, line, error chain).

**Phase 6: Hardening (2-3 weeks)**

Deliverables:
- Error handling, fallbacks
- Performance profiling
- Documentation: user guide, API reference, tutorials
- Real-world dataset integration tests

**Total: 22-30 weeks (5.5-7.5 months)**

### 12.2 Testing Strategy

**Unit Tests:**

- Property-based (proptest): submodularity, monotonicity
- Constraint invariants
- Numerical edge cases

**Integration Tests:**

- End-to-end solver runs
- Determinism: 10 runs, assert identical outputs
- Cross-platform (CI: Linux, macOS, Windows)

**Approximation Quality:**

- Small problems: compare vs optimal (brute-force)
- Verify $\geq (1-1/e)$
- Stochastic: sweep $\epsilon$, validate quality/speed

**Regression:**

- Lock objective values for standard datasets
- Alert on $> 1\%$ degradation

**Performance:**

- Baseline: $n=10^6$, $m=10^6$, $\text{nnz}=10^8$, $k=250$
- Target: $< 10$ min on 32-core, 128GB RAM
- Measure: evals/sec, peak memory, I/O fraction

### 12.3 Continuous Integration

**GitHub Actions:**

- Test matrix: {Ubuntu, macOS, Windows} × {stable, nightly}
- Determinism test: 10 identical runs
- Benchmarks with `criterion`, regression detection
- Cross-platform reproducibility (accept $< 10^{-6}$ difference)

---

## 12. Theoretical Foundations and References

### 12.1 Foundational Results

**Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978)**
"An analysis of approximations for maximizing submodular set functions—I"
*Mathematical Programming*, 14(1), 265-294.

**Result:** Greedy achieves $(1-1/e)$ for monotone submodular + cardinality.

### 12.2 Acceleration Techniques

**Minoux, M. (1978)**
"Accelerated greedy algorithms for maximizing submodular set functions"
*Optimization Techniques*, Springer, 234-243.

**Contribution:** Lazy evaluation; dramatic reduction in evaluations.

**Mirzasoleiman, B., Badanidiyuru, A., Karbasi, A., Vondrák, J., & Krause, A. (2015)**
"Lazier than lazy greedy"
*AAAI Conference on Artificial Intelligence*

**Contribution:** Stochastic Greedy with $O(n \log(1/\epsilon))$ complexity.

### 12.3 Streaming

**Badanidiyuru, A., Mirzasoleiman, B., Karbasi, A., & Krause, A. (2014)**
"Streaming submodular maximization: Massive data summarization on the fly"
*KDD*

**Contribution:** Single-pass, limited memory, $1/2-\epsilon$ approximation.

### 12.4 Matroid Constraints

**Călinescu, G., Chekuri, C., Pál, M., & Vondrák, J. (2011)**
"Maximizing a monotone submodular function subject to a matroid constraint"
*SIAM Journal on Computing*, 40(6), 1740-1766.

**Contribution:** Continuous Greedy achieves $(1-1/e)$ for matroid constraints.

**Chekuri, C., Vondrák, J., & Zenklusen, R. (2014)**
"Submodular function maximization via the multilinear relaxation and contention resolution schemes"
*SIAM Journal on Computing*, 43(6), 1831-1879.

**Contribution:** Framework unifying continuous relaxation and rounding.

### 12.5 Non-Monotone

**Buchbinder, N., Feldman, M., Naor, J., & Schwartz, R. (2015)**
"A tight linear time $(1/2)$-approximation for unconstrained submodular maximization"
*SIAM Journal on Computing*, 44(5), 1384-1402.

**Contribution:** Double Greedy for non-monotone.

### 12.6 Knapsack

**Sviridenko, M. (2004)**
"A note on maximizing a submodular set function subject to a knapsack constraint"
*Operations Research Letters*, 32(1), 41-43.

**Contribution:** $(1-1/e)$ for submodular + knapsack via partial enumeration and continuous relaxation.

### 12.7 Log-Determinant and DPP

**Kulesza, A., & Taskar, B. (2012)**
"Determinantal point processes for machine learning"
*Foundations and Trends in Machine Learning*, 5(2-3), 123-286.

**Contribution:** Comprehensive DPP treatment; log-determinant submodularity.

### 12.8 Surveys

**Bilmes, J. A. (2022)**
"Submodularity in Machine Learning and Artificial Intelligence"
*arXiv preprint arXiv:2202.00132*

**Contribution:** Encyclopedic survey of theory, algorithms, applications.

**Krause, A., & Golovin, D. (2014)**
"Submodular function maximization"
*Tractability: Practical Approaches to Hard Problems*, Cambridge University Press.

**Contribution:** Accessible introduction with practical algorithms.

---

## Appendix A: Glossary

**Cardinality Constraint:** $|S| \leq k$

**Continuous Greedy:** Algorithm optimizing multilinear extension in $[0,1]^n$

**CSR (Compressed Sparse Row):** Sparse matrix format, row-major

**Determinism:** Reproducible outputs for identical inputs

**Diminishing Returns:** Core property: $\Delta(e \mid A) \geq \Delta(e \mid B)$ for $A \subseteq B$

**Epoch:** Timestamp for upper bounds in Lazy Greedy

**Facility Location:** Coverage objective: $\sum_i w_i \max_{s \in S} u_{i,s}$

**FixedBitSet:** Compact bitset with O(1) membership

**Greedy Algorithm:** Iteratively select max marginal gain

**Knapsack Constraint:** $\sum_{s \in S} \text{cost}(s) \leq B$

**Lazy Greedy:** Accelerated greedy using upper bounds

**Log-Determinant:** Diversity objective: $\log \det(K_{S,S} + \epsilon I)$

**Marginal Gain:** $\Delta(e \mid S) = f(S \cup \{e\}) - f(S)$

**Matroid:** Combinatorial structure generalizing independence

**Multilinear Extension:** Continuous relaxation $F: [0,1]^n \to \mathbb{R}$

**Oracle:** Abstract interface for submodular function

**Partition Matroid:** Matroid with per-category capacity constraints

**Pipage Rounding:** Deterministic rounding for partition matroids

**Saturating Coverage:** Objective with concave saturation: $\sum_i \phi_i(\sum_{s \in S} a_{i,s})$

**Schur Complement:** Matrix operation for incremental determinant

**SelectionView:** Immutable selection snapshot (FixedBitSet + size). Memory: $|V|/8$ bytes for bitset + 8 bytes for size.

**Sieve-Streaming:** Single-pass streaming algorithm

**Stochastic Greedy:** Randomized greedy sampling candidates

**Submodular Function:** Set function with diminishing returns

**Swap Rounding:** Probabilistic rounding for general matroids

---

## Appendix B: Critical Implementation Checklist

Before production deployment, verify:

**☐ Oracle Interface**
- [ ] `gain()` is `&self`, never `&mut self`
- [ ] All evaluation uses `SelectionView`, not direct selection
- [ ] `commit()` is only called after final selection decision

**☐ Selection Representation**
- [ ] `SelectionView` with `FixedBitSet` is canonical during optimization
- [ ] `Vec<ItemId>` generated only for final output
- [ ] No `Vec::contains` in hot paths

**☐ Constraint Callbacks**
- [ ] Use `Arc<dyn Fn + Send + Sync>`, not `fn` pointers
- [ ] Knapsack cost function supports dynamic lookup
- [ ] Partition function supports closures

**☐ Lazy Greedy**
- [ ] Epoch tracking implemented
- [ ] Stale bounds discarded on pop
- [ ] Tie-breaking deterministic (by ItemId)

**☐ Continuous Greedy**
- [ ] Gradient estimation uses **temporary views**
- [ ] Oracle never mutated during gradient estimation
- [ ] Direction selection finds **BASE**, not just independent set
- [ ] Rounding uses seeded RNG

**☐ Knapsack**
- [ ] Practical vs Theoretical modes explicit
- [ ] Approximation guarantees documented
- [ ] Mode choice logged

**☐ Log-Determinant**
- [ ] Always f64
- [ ] Epsilon regularization applied
- [ ] log1p for small Schur diagonal
- [ ] NaN detection → immediate refactorization
- [ ] 5-failure degradation to Facility Location

**☐ Determinism**
- [ ] Hierarchical RNG seeding (master → algo → iteration)
- [ ] Fixed-order reduction implemented
- [ ] Tie-breaking consistent

**☐ I/O**
- [ ] Parquet predicate pushdown enabled
- [ ] Row-group indexing used
- [ ] Zero filtering during CSR construction

**☐ Monitoring**
- [ ] All core metrics instrumented
- [ ] CG-specific metrics (grad variance, base weight, rounding loss)
- [ ] Audit logs in JSON Lines format

---

## Appendix C: License and Governance

**Recommended License:** Apache-2.0

The Apache License 2.0 is well-suited for this project because:
- **Patent Protection:** Explicit patent grant protects contributors and users
- **Commercial-Friendly:** Permissive terms enable commercial deployment without restrictions
- **Ecosystem Compatibility:** Compatible with most OSS and proprietary projects
- **Clear Contribution Terms:** Well-understood contribution model

**Governance Structure:**

For open-source release:
1. **SECURITY.md:** Vulnerability disclosure policy with responsible disclosure timeline (90 days)
2. **CODE_OF_CONDUCT.md:** Community standards (Contributor Covenant recommended)
3. **CONTRIBUTING.md:** Contribution guidelines, code review process, testing requirements
4. **MAINTAINERS.md:** List of active maintainers and decision-making process

This structure lowers barriers for external contributions while maintaining code quality and security standards.

---

**End of Complete Technical Specification**
