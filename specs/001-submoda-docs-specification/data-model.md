# Data Model: Submodular Optimization Platform

**Branch**: `001-submoda-docs-specification` | **Date**: 2025-10-12

This document defines the core entities, traits, and data structures for the submoda platform, extracted from the feature specification and technical specification.

---

## Core Type Definitions

### ItemId
**Purpose**: Unique identifier for elements in the ground set V

```rust
pub type ItemId = u32;  // Default: up to 4.3B candidates
// Alternative for extreme scale:
// pub type ItemId = u64;  // Up to 18 quintillion candidates
```

**Properties:**
- Must be dense and zero-indexed (0, 1, 2, ..., n-1)
- Used as array indices for O(1) bitset operations
- Configurable at compile-time via type alias

**Validation Rules:**
- No gaps in ID sequence
- Maximum value < universe_size
- Non-negative by type system

---

### Weight
**Purpose**: Numerical values for objectives, utilities, costs

```rust
pub type Weight = f32;  // Default: memory-efficient, SIMD-friendly
// Objective-specific overrides:
// LogDeterminant: always f64 (Cholesky numerical stability)
// ContinuousGreedy gradient: f64 (error accumulation)
```

**Properties:**
- Monomorphic types (f32 or f64), not trait objects
- Strict validation: no NaN, no Infinity, non-negative where required
- SIMD-aligned for vectorization

**Type Selection Table:**

| Objective Function | Type | Rationale |
|--------------------|------|-----------|
| Facility Location | `f32` | Sums of products, sufficient precision, 2x memory savings |
| Saturating Coverage | `f32` | Concave transforms, well-conditioned |
| Log-Determinant | `f64` | Cholesky accumulates errors, f32 fails |
| Continuous Greedy | `f64` | Gradient estimation, sampling noise |

---

## 1. SelectionView (Immutable Snapshot)

**Purpose**: Canonical representation of current selection during optimization, enabling thread-safe parallel evaluation.

```rust
pub struct SelectionView {
    pub in_set: fixedbitset::FixedBitSet,  // Bitset for O(1) membership testing
    pub size: usize,                         // Cardinality |S|
}
```

**Properties:**
- **Immutable**: Never modified in place; cloned for oracle queries
- **Compact**: 1 bit per element + 8 bytes size = |V|/8 bytes total
- **Fast Membership**: O(1) `in_set.contains(e)` vs O(n) for `Vec::contains`
- **Thread-Safe**: Clone is cheap (bitset copy), enables parallel evaluation

**Operations:**
```rust
impl SelectionView {
    pub fn new(universe_size: usize) -> Self {
        Self {
            in_set: FixedBitSet::with_capacity(universe_size),
            size: 0,
        }
    }

    pub fn contains(&self, e: ItemId) -> bool {
        self.in_set.contains(e as usize)
    }

    pub fn clone_with_element(&self, e: ItemId) -> Self {
        let mut new_view = self.clone();
        new_view.in_set.insert(e as usize);
        new_view.size += 1;
        new_view
    }
}
```

**Relationships:**
- Used by `SubmodularOracle::gain(&self, view: &SelectionView, e: ItemId)`
- Used by `Constraint::can_add(&self, view: &SelectionView, e: ItemId)`
- Converted to `Selection` only at final output

**Invariants:**
- `size == in_set.count_ones()`
- All set bits correspond to ItemId values < universe_size
- Never mutated during parallel evaluation

---

## 2. Selection (Final Output)

**Purpose**: Materializes final solution with full audit trail and metadata.

```rust
pub struct Selection {
    pub items: Vec<ItemId>,              // Selected elements (derived from SelectionView)
    pub objective: f64,                   // Final objective value f(S)
    pub used_budget: f64,                 // For knapsack constraints
    pub counts_by_part: Vec<usize>,      // For partition matroid constraints
    pub trace: Vec<IterationTrace>,      // Full audit log
}
```

**Properties:**
- Generated once at algorithm completion
- `items` derived from `SelectionView::in_set` by collecting set bits
- Includes full iteration history for reproducibility
- Immutable after creation

**Validation Rules:**
- `items.len() == SelectionView.size`
- No duplicate ItemIds (enforced by bitset)
- All ItemIds < universe_size
- `objective` finite and non-NaN

**Relationships:**
- Output of `solve()` function
- Contains `Vec<IterationTrace>` for full audit trail
- Serializable to JSON for persistence

---

## 3. IterationTrace (Audit Log Entry)

**Purpose**: Captures state of each algorithm iteration for reproducibility and debugging.

```rust
pub struct IterationTrace {
    pub iteration: usize,                 // Iteration number (1-indexed)
    pub selected_element: ItemId,         // Chosen element
    pub marginal_gain: f64,              // Δ(e | S) for selected element
    pub num_evaluations: usize,          // Evaluations performed this iteration
    pub selection_size: usize,            // Current |S| after selection
    pub objective_value: f64,            // Current f(S)
    pub algorithm: String,                // Algorithm name (e.g., "LazyGreedy")
    pub rng_seed: u64,                   // RNG seed for this iteration
    pub timestamp_ms: u64,               // Wall-clock milliseconds since start
    pub constraint_status: ConstraintStatus,  // Constraint-specific state
}
```

**Optional Fields (Lazy Greedy):**
```rust
pub struct LazyGreedyTrace {
    pub upper_bound: f64,                // Cached upper bound for selected element
    pub heap_size: usize,                 // Heap size before selection
    pub counterfactuals: Vec<Counterfactual>,  // Runner-up candidates
}

pub struct Counterfactual {
    pub element: ItemId,
    pub marginal_gain: f64,
    pub deficit: f64,                     // Difference from selected gain
    pub reason: String,                   // "lower_than_selected", "infeasible"
}
```

**Serialization:** JSON Lines format (one entry per line)

**Relationships:**
- Multiple `IterationTrace` collected in `Selection::trace`
- Used for post-mortem analysis and reproducibility validation
- Enables deterministic hash computation (hash ItemId sequence + seed)

**Invariants:**
- `iteration` strictly increasing
- `marginal_gain >= 0` (non-negative submodular functions)
- `selection_size == iteration` for cardinality constraints
- `rng_seed` changes deterministically per iteration

---

## 4. SubmodularOracle (Core Trait)

**Purpose**: Abstract interface for submodular function evaluation with thread-safe parallel queries.

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
    /// Returns: Most recently evaluated marginal gain Δ(e|S')
    /// Initialization: f64::INFINITY (lazy init on first pop)
    fn upper_bound(&self, e: ItemId) -> f64;

    /// Create lightweight clone for parallel workers - OPTIONAL
    /// Shares immutable data (Arc), clones mutable caches
    fn fork(&self) -> Box<dyn SubmodularOracle> {
        unimplemented!("fork() not supported by this oracle")
    }

    /// Prefetch hint for element e - OPTIONAL
    fn prefetch(&self, _e: ItemId) {
        // Default: no-op
    }

    /// Batch evaluation for SIMD/cache efficiency - OPTIONAL
    /// Default: fallback to sequential gain() calls
    fn gain_batch(&self, view: &SelectionView, candidates: &[ItemId]) -> Vec<f64> {
        candidates.iter().map(|&e| self.gain(view, e)).collect()
    }
}
```

**Critical Design Principles:**
1. **gain() is &self**: Enables parallel evaluation via Rayon
2. **commit() is &mut self**: Sequential state updates after selection
3. **View-based queries**: All evaluation uses immutable SelectionView snapshots
4. **Fork pattern**: `Arc<immutable_data>` + thread-local caches

**Concrete Implementations:**

### FacilityLocation
```rust
pub struct FacilityLocation {
    demand_weights: Arc<Vec<f32>>,          // Shared immutable
    utility_matrix: Arc<sprs::CsMat<f32>>,  // Shared immutable (CSR format)
    best_u: Vec<f32>,                        // Mutable: current best utility per demand
    upper_bounds: Vec<f64>,                  // Mutable: cached marginal gains
    last_epoch: u64,                         // Mutable: epoch tracking
}
```

**State Management:**
- `best_u[i] = max_{s ∈ S} u_{i,s}` (updated only in commit())
- `upper_bounds[e]` cached from last `gain()` evaluation
- `last_epoch` tracks selection iteration for stale bound detection

### LogDeterminant
```rust
pub struct LogDeterminant {
    kernel_matrix: Arc<nalgebra::DMatrix<f64>>,  // Shared immutable
    cholesky_L: nalgebra::DMatrix<f64>,          // Mutable: Cholesky factor
    selected: Vec<ItemId>,                        // Mutable: current selection
    epsilon: f64,                                 // Regularization parameter
    consecutive_failures: usize,                  // Mutable: degradation counter
    mode: ComputationMode,                        // Mutable: Normal or FallbackOnly
}
```

**State Management:**
- `cholesky_L` incrementally updated (Schur complement)
- 5-layer numerical defense (regularization, safe Schur, log1p, refactorization, degradation)
- Graceful degradation to Facility Location after 5 failures

---

## 5. Constraint (Trait)

**Purpose**: Feasibility checking and state management for constraint types.

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

**Concrete Implementations:**

### CardinalityConstraint
```rust
pub struct CardinalityConstraint {
    pub k: usize,  // Maximum cardinality
}

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

### KnapsackConstraint
```rust
pub struct KnapsackConstraint {
    pub budget: f64,
    pub cost_fn: Arc<dyn Fn(ItemId) -> f64 + Send + Sync>,
    pub mode: KnapsackMode,
    used: f64,  // Mutable: current budget usage
}

pub enum KnapsackMode {
    Practical { enumerate_top_k: usize },     // Fast heuristic
    Theoretical { cg_steps: usize, grad_samples: usize },  // (1-1/e) guarantee
}

impl Constraint for KnapsackConstraint {
    fn can_add(&self, _view: &SelectionView, e: ItemId) -> bool {
        self.used + (self.cost_fn)(e) <= self.budget + EPSILON
    }

    fn commit(&mut self, e: ItemId) {
        self.used += (self.cost_fn)(e);
    }

    fn reset(&mut self) {
        self.used = 0.0;
    }
}
```

**Why Arc<dyn Fn>:**
- Enables closures capturing environment (e.g., database lookups)
- Supports dynamic cost computation at runtime
- Thread-safe with `Send + Sync` bounds

### PartitionMatroid
```rust
pub struct PartitionMatroid {
    pub part_fn: Arc<dyn Fn(ItemId) -> usize + Send + Sync>,
    pub capacities: Vec<usize>,
    counts: Vec<usize>,  // Mutable: current per-partition counts
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
```

---

## 6. Matroid (Trait)

**Purpose**: Abstract interface for matroid independence testing and max-weight base computation.

```rust
pub trait Matroid: Constraint {
    /// Rank of the matroid (size of maximum independent set)
    fn rank(&self) -> usize;

    /// Find maximum weight base
    /// Returns: Base B (maximal independent set) maximizing Σ_{i ∈ B} weights[i]
    fn max_weight_base(&self, weights: &[f64]) -> Vec<ItemId>;

    /// Check if adding element maintains independence (redundant with Constraint::can_add)
    fn is_independent(&self, view: &SelectionView, e: ItemId) -> bool {
        self.can_add(view, e)
    }
}
```

**PartitionMatroid Implementation:**
```rust
impl Matroid for PartitionMatroid {
    fn rank(&self) -> usize {
        self.capacities.iter().sum()
    }

    fn max_weight_base(&self, weights: &[f64]) -> Vec<ItemId> {
        let mut base = Vec::new();

        for part_id in 0..self.capacities.len() {
            // Collect elements in this partition
            let mut part_items: Vec<(ItemId, f64)> = (0..weights.len())
                .filter(|&i| (self.part_fn)(i as ItemId) == part_id)
                .map(|i| (i as ItemId, weights[i]))
                .collect();

            // Sort by weight descending, tie-break by ItemId (deterministic)
            part_items.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });

            // Take top capacity[part_id] elements
            base.extend(part_items.iter().take(self.capacities[part_id]).map(|(id, _)| *id));
        }

        base
    }
}
```

**GraphicMatroid (Forest/Acyclic Edge Sets):**
```rust
pub struct GraphicMatroid {
    n_vertices: usize,
    edge_endpoints: Vec<(usize, usize)>,  // e -> (u, v)
    uf: UnionFind,                         // Mutable: union-find for cycle detection
}

impl Constraint for GraphicMatroid {
    fn can_add(&self, _view: &SelectionView, e: ItemId) -> bool {
        let (u, v) = self.edge_endpoints[e as usize];
        !self.uf.connected(u, v)  // Would adding e create a cycle?
    }

    fn commit(&mut self, e: ItemId) {
        let (u, v) = self.edge_endpoints[e as usize];
        self.uf.union(u, v);
    }

    fn reset(&mut self) {
        self.uf = UnionFind::new(self.n_vertices);
    }
}

impl Matroid for GraphicMatroid {
    fn rank(&self) -> usize {
        self.n_vertices - 1  // Maximum spanning tree size
    }

    fn max_weight_base(&self, weights: &[f64]) -> Vec<ItemId> {
        // Kruskal's algorithm: sort edges by weight, add greedily if no cycle
        let mut edges: Vec<(ItemId, f64)> = (0..weights.len())
            .map(|i| (i as ItemId, weights[i]))
            .collect();

        edges.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0))
        });

        let mut forest = Vec::new();
        let mut uf = UnionFind::new(self.n_vertices);

        for (e, _) in edges {
            let (u, v) = self.edge_endpoints[e as usize];
            if !uf.connected(u, v) {
                uf.union(u, v);
                forest.push(e);
                if forest.len() == self.rank() {
                    break;
                }
            }
        }

        forest
    }
}
```

---

## 7. Strategy (Algorithm Selection)

**Purpose**: Configuration for algorithm variant and parameters.

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

pub enum RoundingMode {
    Pipage,  // Deterministic, partition matroid only
    Swap,    // Probabilistic, general matroid
}
```

**Validation Rules:**
- `epsilon > 0` for all algorithms
- `sample_factor >= 1.0` for Stochastic Greedy
- `steps >= 10` for Continuous Greedy (discretization granularity)
- `grad_samples >= 100` for Continuous Greedy (gradient estimation)

**Relationships:**
- Passed to `solve()` function
- Determines algorithm variant instantiation
- Logged in `IterationTrace::algorithm` field

---

## 8. Determinism (Reproducibility Configuration)

**Purpose**: Control reproducibility behavior across runs and parallelism levels.

```rust
pub struct Determinism {
    pub seed: u64,              // Master RNG seed
    pub fixed_order: bool,      // Fix parallel reduction order
    pub tie_breaking: TieBreak, // Equal marginal gain resolution
}

pub enum TieBreak {
    ById,            // Prefer lower ItemId (default, deterministic)
    ByUpperBound,    // Prefer higher cached upper bound
    Random,          // Use seeded RNG (less deterministic)
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

**Hierarchical RNG Seeding:**
```rust
const STOCHASTIC_TAG: u64 = 0x5374476565647947;  // "StGreedyG" in hex

pub fn algorithm_seed(master_seed: u64, algorithm: &str) -> u64 {
    master_seed ^ algorithm_tag(algorithm)
}

pub fn iteration_seed(algo_seed: u64, iteration: usize) -> u64 {
    algo_seed.wrapping_add(iteration as u64)
}
```

**Guarantees:**
- Same seed + same algorithm → identical selection sequence
- Different parallelism (1 vs 64 threads) → identical results (if `fixed_order = true`)
- Cross-platform: <10^-6 objective differences acceptable (FMA variations)

---

## 9. CSRMatrix (Sparse Storage)

**Purpose**: Compressed Sparse Row format for demand-centric utility matrix access.

```rust
pub struct CSRMatrix<T> {
    pub row_ptr: Vec<usize>,        // Length: n_demands + 1
    pub col_indices: Vec<ItemId>,   // Length: nnz
    pub values: Vec<T>,             // Length: nnz
}
```

**Properties:**
- Generic over `T: Float` (f32 or f64)
- Row-major layout for sequential demand access
- O(1) row start/end lookup via `row_ptr`
- O(nnz(row)) access time for row data

**Construction from Triplets:**
```rust
impl<T: Float> CSRMatrix<T> {
    pub fn from_triplets(
        shape: (usize, usize),
        mut triplets: Vec<(usize, usize, T)>,
    ) -> Self {
        // Sort by (row, col) for CSR construction
        triplets.sort_by_key(|(r, c, _)| (*r, *c));

        let (n_rows, _) = shape;
        let mut row_ptr = vec![0; n_rows + 1];
        let mut col_indices = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());

        let mut current_row = 0;
        for (row, col, val) in triplets {
            // Skip zero entries
            if val.is_zero() {
                continue;
            }

            // Update row_ptr for row transitions
            while current_row < row {
                current_row += 1;
                row_ptr[current_row] = col_indices.len();
            }

            col_indices.push(col as ItemId);
            values.push(val);
        }

        // Fill remaining row_ptr entries
        while current_row < n_rows {
            current_row += 1;
            row_ptr[current_row] = col_indices.len();
        }

        Self { row_ptr, col_indices, values }
    }

    pub fn row(&self, i: usize) -> RowView<'_, T> {
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        RowView {
            indices: &self.col_indices[start..end],
            values: &self.values[start..end],
        }
    }
}

pub struct RowView<'a, T> {
    pub indices: &'a [ItemId],
    pub values: &'a [T],
}
```

**Memory Layout:**
```
Example: 3 demands × 5 candidates, 7 non-zeros

row_ptr:      [0, 2, 4, 7]
col_indices:  [1, 3, 0, 4, 2, 3, 4]
values:       [0.5, 0.8, 0.3, 0.9, 0.6, 0.4, 0.7]

Demand 0: candidates [1, 3] with utilities [0.5, 0.8]
Demand 1: candidates [0, 4] with utilities [0.3, 0.9]
Demand 2: candidates [2, 3, 4] with utilities [0.6, 0.4, 0.7]
```

**Relationships:**
- Used by `FacilityLocation::utility_matrix`
- Constructed from Parquet triplets (i, s, u) by `submod-io`
- Enables efficient SIMD-optimized marginal gain computation

---

## 10. ConstraintStatus (State Snapshot)

**Purpose**: Capture constraint-specific state for audit logs.

```rust
pub enum ConstraintStatus {
    Cardinality {
        current: usize,
        limit: usize,
    },
    Knapsack {
        used_budget: f64,
        total_budget: f64,
        remaining: f64,
    },
    Partition {
        counts: Vec<usize>,
        capacities: Vec<usize>,
    },
    Matroid {
        rank: usize,
        current_size: usize,
    },
}
```

**Serialization Example:**
```json
{
  "constraint_status": {
    "Knapsack": {
      "used_budget": 0.85,
      "total_budget": 1.0,
      "remaining": 0.15
    }
  }
}
```

---

## Entity Relationship Diagram

```
┌─────────────────┐
│ SelectionView   │ ───────┐
│ - in_set        │        │
│ - size          │        │
└─────────────────┘        │
        │                  │
        │ used by          │ used by
        ↓                  ↓
┌─────────────────┐  ┌─────────────────┐
│SubmodularOracle │  │   Constraint    │
│ - gain()        │  │ - can_add()     │
│ - commit()      │  │ - commit()      │
│ - upper_bound() │  └─────────────────┘
└─────────────────┘            │
        │                      │
        │ implements           │ implements
        ↓                      ↓
┌─────────────────┐  ┌─────────────────┐
│FacilityLocation │  │  Cardinality    │
│  LogDeterminant │  │   Knapsack      │
│  Saturating     │  │  PartitionMat.  │
└─────────────────┘  └─────────────────┘
                              │
                              │ extends
                              ↓
                     ┌─────────────────┐
                     │    Matroid      │
                     │ - rank()        │
                     │ - max_weight_   │
                     │   base()        │
                     └─────────────────┘

┌─────────────────┐
│   Selection     │ ◄──┐
│ - items         │    │ contains
│ - objective     │    │
│ - trace         │ ───┘
└─────────────────┘
        │
        │ contains Vec<>
        ↓
┌─────────────────┐
│ IterationTrace  │
│ - iteration     │
│ - selected_elem │
│ - marginal_gain │
└─────────────────┘
```

---

## State Transitions

### Optimization Loop State Machine

```
[Initialize SelectionView: empty bitset]
           ↓
    ┌──────────────┐
    │ Evaluate     │ ← Parallel: oracle.gain(view, candidates)
    └──────────────┘
           ↓
    ┌──────────────┐
    │ Select Best  │ ← Deterministic tie-breaking
    └──────────────┘
           ↓
    ┌──────────────┐
    │ Check        │ ← constraint.can_add(view, element)
    │ Constraint   │
    └──────────────┘
           ↓
    ┌──────────────┐
    │ Commit       │ ← Sequential: oracle.commit(e)
    │ Element      │    constraint.commit(e)
    │              │    view.in_set.insert(e)
    └──────────────┘
           ↓
    [Check termination: cardinality / stagnation / timeout]
           ↓
    ┌──────────────┐
    │ Materialize  │ ← Convert SelectionView → Selection
    │ Selection    │    Collect trace, compute final objective
    └──────────────┘
```

---

## Validation Rules Summary

### Data Integrity
- All `ItemId` values < `universe_size`
- All `Weight` values finite, non-NaN
- Non-negative weights/utilities/costs (objective-dependent)
- No duplicate ItemIds in `Selection::items` (enforced by bitset)

### Algorithm Invariants
- `SelectionView.size == in_set.count_ones()`
- `oracle.gain(view, e) >= 0` for monotone submodular functions
- `constraint.can_add(view, e)` checked before every `commit()`
- Determinism: same seed → same selection sequence (audit hash matches)

### Performance Constraints
- Memory: <150GB for 100B non-zeros
- Marginal gain latency: <200ms p95
- Lazy Greedy efficiency: 5-10% of naive evaluations
- Parallel speedup: ≥50x with 64 threads (embarrassingly parallel workloads)

---

**Status**: ✅ Data model complete, ready for API contracts generation
