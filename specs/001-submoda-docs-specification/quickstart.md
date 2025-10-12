# Quick Start Guide

## Overview

This guide demonstrates the minimal working example for the submoda submodular optimization platform. The example uses a small dataset (50 candidates, 100 demands, k=10) and validates the end-to-end pipeline in <1 second.

## Problem Setup

**Objective**: Facility Location (maximize weighted coverage)
- **Candidates**: 50 facility locations
- **Demands**: 100 demand points with weights
- **Constraint**: Cardinality k=10 (select 10 facilities)
- **Algorithm**: Lazy Greedy with ε=0.0 (exact)
- **Seed**: 42 (deterministic)

**Dataset Size**: ~10KB (50 candidates × 100 demands × 4 bytes)
**Expected Runtime**: <1 second
**Expected Approximation**: (1-1/e) ≈ 0.632 guarantee

---

## Rust CLI Example

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/submoda.git
cd submoda

# Build release binary
cargo build --release

# Verify installation
./target/release/submoda --version
```

### Generate Sample Data

```bash
# Create synthetic Facility Location dataset
./target/release/submoda generate \
  --objective facility-location \
  --candidates 50 \
  --demands 100 \
  --output data/quickstart.parquet \
  --seed 42
```

**Output**: `data/quickstart.parquet` (Parquet file with Arrow schema)

**Schema**:
```
candidate_id: UInt32
demand_id: UInt32
utility: Float32
demand_weight: Float32
```

### Run Optimization

```bash
# Lazy Greedy optimization
./target/release/submoda solve \
  --input data/quickstart.parquet \
  --algorithm lazy-greedy \
  --cardinality 10 \
  --seed 42 \
  --output results/quickstart_solution.json \
  --trace results/quickstart_trace.jsonl
```

**Expected Output**:

```json
{
  "selection": {
    "items": [3, 7, 12, 18, 23, 29, 34, 38, 42, 47],
    "objective": 8234.67,
    "cardinality": 10,
    "used_budget": null,
    "counts_by_part": [],
    "termination_reason": "cardinality_reached"
  },
  "config": {
    "algorithm": "lazy_greedy",
    "epsilon": 0.0,
    "seed": 42,
    "max_iterations": 10
  },
  "performance": {
    "total_seconds": 0.023,
    "evaluations": 487,
    "heap_pops": 63
  }
}
```

**Trace File** (`results/quickstart_trace.jsonl`):
```jsonl
{"iteration":1,"selected":3,"gain":1234.5,"objective":1234.5,"upper_bound":1234.5,"evaluations":50,"heap_pops":1}
{"iteration":2,"selected":7,"gain":987.3,"objective":2221.8,"upper_bound":1102.1,"evaluations":49,"heap_pops":6}
...
```

### Verify Approximation Guarantee

```bash
# Compare against brute-force optimal (for small k)
./target/release/submoda verify \
  --input data/quickstart.parquet \
  --solution results/quickstart_solution.json \
  --brute-force
```

**Output**:
```
Lazy Greedy Objective:  8234.67
Brute-Force Optimal:    12456.23
Approximation Ratio:    0.661 (≥ 0.632 guaranteed)
```

---

## Python Example

### Installation

```bash
# Install from PyPI (when published)
pip install submoda

# Or install from source
pip install maturin
cd submoda/submod-bindings-py
maturin develop --release
```

### Generate Sample Data

```python
import numpy as np
import pandas as pd
import submoda

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic Facility Location dataset
n_candidates = 50
n_demands = 100

# Random utilities: U[0, 100]
utilities = np.random.uniform(0, 100, size=(n_demands, n_candidates)).astype(np.float32)

# Random demand weights: U[1, 10]
demand_weights = np.random.uniform(1, 10, size=n_demands).astype(np.float32)

print(f"Dataset shape: {utilities.shape}")
print(f"Memory: {utilities.nbytes + demand_weights.nbytes} bytes")
```

### Run Optimization

```python
# Create Facility Location oracle (zero-copy from NumPy)
oracle = submoda.FacilityLocation.from_numpy(
    demand_weights=demand_weights,
    utility_matrix=utilities
)

# Create cardinality constraint
constraint = submoda.CardinalityConstraint(max_size=10)

# Configure Lazy Greedy solver
config = submoda.SolverConfig.lazy_greedy() \
    .with_seed(42) \
    .with_epsilon(0.0)

# Solve (GIL released during optimization)
selection = submoda.solve(oracle, constraint, config)

# Display results
print(f"Selected items: {selection.items}")
print(f"Objective value: {selection.objective:.2f}")
print(f"Cardinality: {len(selection.items)}")
print(f"Termination: {selection.termination_reason}")
```

**Expected Output**:
```
Selected items: [3, 7, 12, 18, 23, 29, 34, 38, 42, 47]
Objective value: 8234.67
Cardinality: 10
Termination: cardinality_reached
```

### Access Iteration Trace

```python
# Get detailed trace
selection_with_trace = submoda.solve_with_trace(oracle, constraint, config)

# Convert trace to DataFrame
trace_df = pd.DataFrame([
    {
        'iteration': t.iteration,
        'selected': t.selected_item,
        'gain': t.marginal_gain,
        'objective': t.objective_value,
        'upper_bound': t.upper_bound_max,
        'evaluations': t.evaluations,
        'heap_pops': t.heap_pops
    }
    for t in selection_with_trace.trace
])

print(trace_df)
```

**Output**:
```
   iteration  selected      gain  objective  upper_bound  evaluations  heap_pops
0          1         3  1234.50    1234.50      1234.50           50          1
1          2         7   987.30    2221.80      1102.10           49          6
2          3        12   856.12    3077.92       989.45           48          8
3          4        18   743.89    3821.81       891.23           47          7
...
```

### Save to Parquet

```python
# Save solution
selection_df = pd.DataFrame({
    'candidate_id': selection.items,
    'marginal_gain': [t.marginal_gain for t in selection_with_trace.trace]
})

selection_df.to_parquet('results/python_solution.parquet', compression='snappy')

# Save trace
trace_df.to_parquet('results/python_trace.parquet', compression='snappy')
```

### Parallel Execution Benchmark

```python
import time
import multiprocessing

# Larger problem for parallelism demonstration
n_candidates_large = 1000
n_demands_large = 5000

utilities_large = np.random.uniform(0, 100, size=(n_demands_large, n_candidates_large)).astype(np.float32)
demand_weights_large = np.random.uniform(1, 10, size=n_demands_large).astype(np.float32)

oracle_large = submoda.FacilityLocation.from_numpy(demand_weights_large, utilities_large)
constraint_large = submoda.CardinalityConstraint(max_size=50)

# Benchmark
start = time.time()
selection_large = submoda.solve(oracle_large, constraint_large, config)
elapsed = time.time() - start

print(f"Problem size: {n_candidates_large}×{n_demands_large}")
print(f"Selected: {len(selection_large.items)} items")
print(f"Objective: {selection_large.objective:.2f}")
print(f"Runtime: {elapsed:.2f}s")
print(f"CPUs used: {multiprocessing.cpu_count()} (GIL released)")
```

**Expected Output** (8-core machine):
```
Problem size: 1000×5000
Selected: 50 items
Objective: 234567.89
Runtime: 2.34s
CPUs used: 8 (GIL released)
```

---

## Advanced: Custom Oracle

### Rust Implementation

```rust
use submod_core::{SubmodularOracle, SelectionView, ItemId};

pub struct CustomOracle {
    universe_size: usize,
    data: Vec<f64>,
    upper_bounds: Vec<f64>,
}

impl SubmodularOracle for CustomOracle {
    fn universe_size(&self) -> usize {
        self.universe_size
    }

    fn gain(&self, view: &SelectionView, e: ItemId) -> f64 {
        // Thread-safe: no mutation, only reads
        let e_idx = e as usize;

        // Example: diminishing returns based on set size
        let base_value = self.data[e_idx];
        let discount = 1.0 / (1.0 + view.size() as f64);

        base_value * discount
    }

    fn commit(&mut self, e: ItemId) {
        // Update internal state after selection
        let e_idx = e as usize;
        self.upper_bounds[e_idx] = 0.0; // Invalidate upper bound
    }

    fn upper_bound(&self, e: ItemId) -> f64 {
        self.upper_bounds[e as usize]
    }
}
```

### Python Implementation

```python
import submoda
import numpy as np

class CustomOracle(submoda.SubmodularOracle):
    def __init__(self, data: np.ndarray):
        self.data = data
        self.universe_size = len(data)
        self.upper_bounds = np.full(self.universe_size, np.inf)

    def gain(self, view: submoda.SelectionView, e: int) -> float:
        """Thread-safe: no state mutation"""
        base_value = self.data[e]
        discount = 1.0 / (1.0 + view.size())
        return base_value * discount

    def commit(self, e: int):
        """Update state after selection"""
        self.upper_bounds[e] = 0.0

    def upper_bound(self, e: int) -> float:
        return self.upper_bounds[e]

# Usage
data = np.random.uniform(0, 100, size=100).astype(np.float64)
oracle = CustomOracle(data)
constraint = submoda.CardinalityConstraint(max_size=10)
config = submoda.SolverConfig.lazy_greedy().with_seed(42)

selection = submoda.solve(oracle, constraint, config)
print(f"Custom oracle result: {selection.items}")
```

---

## Performance Validation

### Expected Characteristics

| Metric | Small (50×100) | Medium (1K×5K) | Large (100K×100K) |
|--------|----------------|----------------|-------------------|
| **Dataset Size** | 10 KB | 10 MB | 20 GB |
| **Runtime (Lazy Greedy)** | <1 second | 2-5 seconds | 5-10 minutes |
| **Evaluations** | ~500 | ~50,000 | ~5,000,000 |
| **Memory** | <1 MB | ~50 MB | ~10 GB |
| **Parallel Speedup (8 cores)** | ~1.5x | ~6x | ~7.5x |

### Verification Steps

1. **Correctness**: Verify submodularity property
```bash
./target/release/submoda verify --check-submodularity \
  --input data/quickstart.parquet \
  --samples 1000 \
  --seed 42
```

2. **Approximation Ratio**: Compare against brute-force (small k only)
```bash
./target/release/submoda verify --brute-force \
  --input data/quickstart.parquet \
  --solution results/quickstart_solution.json
```

3. **Determinism**: Verify identical results across runs
```bash
for i in {1..5}; do
  ./target/release/submoda solve \
    --input data/quickstart.parquet \
    --algorithm lazy-greedy \
    --cardinality 10 \
    --seed 42 \
    --output results/run_${i}.json
done

# Check all solutions are identical
diff results/run_1.json results/run_2.json
# (should output nothing if identical)
```

---

## Troubleshooting

### Issue: "Universe size exceeds u32::MAX"

**Solution**: Use `--item-id-type u64` flag (slower, 8-byte indices)

### Issue: Python GIL contention

**Symptoms**: Parallel speedup <2x on 8 cores

**Solution**: Ensure `solve()` not called in Python loop; batch operations

```python
# ❌ Bad: GIL reacquired between calls
for config in configs:
    selection = submoda.solve(oracle, constraint, config)

# ✅ Good: Single long-running call
config = submoda.SolverConfig.lazy_greedy()
selection = submoda.solve(oracle, constraint, config)  # GIL released for entire call
```

### Issue: Numerical instability (NaN in trace)

**Symptoms**: `SolverError::NumericalError` with NaN values

**Solution**: Check for:
- Negative utilities (invalid for Facility Location)
- Infinity in input data
- Log-Determinant near-singularity (λ too small)

```python
# Validate input
assert not np.isnan(utilities).any(), "NaN in utilities"
assert not np.isinf(utilities).any(), "Infinity in utilities"
assert (utilities >= 0).all(), "Negative utilities"
```

---

## Next Steps

1. **Explore algorithms**: Try `StochasticGreedy` (100x faster) or `ContinuousGreedy` (matroids)
2. **Larger datasets**: Load real Parquet datasets with `--input your_data.parquet`
3. **Custom oracles**: Implement `SubmodularOracle` trait for domain-specific objectives
4. **Constraints**: Use `KnapsackConstraint`, `PartitionMatroid`, or custom constraints
5. **Monitoring**: Enable Prometheus metrics with `--metrics-port 9090`

**Documentation**:
- Full API reference: `docs/api/`
- Algorithm details: `docs/algorithms.md`
- Benchmarks: `docs/benchmarks.md`

**Support**:
- GitHub Issues: https://github.com/your-org/submoda/issues
- Discussions: https://github.com/your-org/submoda/discussions
