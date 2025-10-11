# Submoda

**Production-grade submodular optimization platform in Rust**

Large-scale subset selection with provable approximation guarantees.

## Status

🚧 **Under Active Development** - Specification complete, implementation in progress.

## Features

- **Provable Quality**: (1-1/e) ≈ 0.632 approximation guarantee for monotone submodular functions
- **Thread-Safe**: Immutable oracle interface enables safe parallel evaluation
- **Deterministic**: Reproducible results across runs and platforms
- **Scalable**: Handles millions of candidates with lazy evaluation and streaming algorithms

## Algorithms

- **Lazy Greedy** - Epoch-based stale bound elimination (5-10% evaluations vs standard greedy)
- **Stochastic Greedy** - O(n log(1/ε)) complexity with random sampling
- **Continuous Greedy** - For general matroid constraints
- **Sieve-Streaming** - Single-pass streaming with limited memory

## Objectives

- Facility Location (coverage maximization)
- Saturating Coverage (concave-over-modular)
- Log-Determinant (diversity via DPP)
- Custom objectives via trait interface

## Quick Start

```bash
# Generate minimal test dataset
python scripts/generate_mini_data.py

# Run optimization (coming soon)
cargo run --release -- \
  --algorithm lazy_greedy \
  --oracle facility_location \
  --demand demand_mini.parquet \
  --utility utility_mini.parquet \
  --k 10 \
  --seed 42
```

## Documentation

- **[Complete Specification](docs/complete-specification.md)** - 2,400+ lines of technical specification
- Implementation roadmap: 5.5-7.5 months, 6 phases

## Phase 1 (Current)

- Core traits: `SubmodularOracle`, `Constraint`, `SelectionView`
- Facility Location oracle with CSR sparse matrix
- Lazy Greedy with epoch-based heap
- Stochastic Greedy
- Cardinality constraints

**ETA: 4-6 weeks**

## License

Apache-2.0

## Reference

Based on foundational work:
- Nemhauser et al. (1978) - (1-1/e) approximation for greedy
- Minoux (1978) - Lazy evaluation
- Mirzasoleiman et al. (2015) - Stochastic greedy
