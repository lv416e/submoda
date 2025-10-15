# Implementation Tasks: Submodular Optimization Platform

## Overview

This document outlines the implementation tasks for building a production-grade submodular optimization platform in Rust. Tasks are organized into 6 phases following the approved design, with estimated timeline of 22-30 weeks.

**Implementation Principles:**
- Build incrementally with working, tested code at each step
- Integrate early and often to avoid orphaned components
- Validate core functionality before adding complexity
- Maintain determinism and numerical stability throughout

---

## Phase 1: Core Framework and Facility Location (4-6 weeks)

### 1. Initialize Rust workspace and foundational infrastructure

- [ ] 1.1 Set up Rust workspace with 6 crates
  - Create workspace root with proper Cargo.toml configuration
  - Initialize submod-core as the foundation crate
  - Initialize submod-objectives for oracle implementations
  - Initialize submod-solver for optimization algorithms
  - Initialize submod-io for data handling
  - Initialize submod-bindings-py for Python integration
  - Initialize submod-service for service layer
  - Configure workspace dependencies and feature flags
  - Set up development environment configuration (mise, rustfmt, clippy)
  - _Requirements: 1.1, 1.2_

- [ ] 1.2 Implement core type system and validation
  - Define ItemId as u32 type with up to 4.3 billion capacity
  - Define Weight type with f32 default and f64 configuration
  - Implement validation functions rejecting NaN, infinity, negative values
  - Create error hierarchy with ValidationError types
  - Add bounds checking for ItemId against universe size
  - Implement deterministic tie-breaking by ItemId
  - _Requirements: 1.1, 1.2, 1.4, 1.5, 17.1_

- [ ] 1.3 Build SelectionView with FixedBitSet representation
  - Implement SelectionView structure with FixedBitSet for membership testing
  - Provide O(1) membership testing functionality
  - Enable efficient cloning for immutable snapshots
  - Implement deterministic iteration by ItemId order
  - Add size tracking and capacity management
  - Support thread-safe sharing across parallel evaluations
  - _Requirements: 1.3, 18.1_

- [ ] 1.4 Create configuration types for strategies and constraints
  - Implement Strategy enum with LazyGreedy, StochasticGreedy, ContinuousGreedy, SieveStreaming variants
  - Implement Constraint enum with Cardinality, Knapsack, PartitionMatroid, GraphicMatroid variants
  - Create DeterminismConfig with seed, fixed_order, and tie-breaking options
  - Define TerminationConfig with threshold, stagnation, and timeout settings
  - Add ObservabilityConfig for metrics and audit logging
  - _Requirements: 1.6, 1.7, 1.8, 10.1, 10.2, 10.3_

### 2. Implement thread-safe oracle interface and constraint system

- [ ] 2.1 Define SubmodularOracle trait with thread-safe methods
  - Define immutable gain() method with &self for concurrent evaluation
  - Define mutable commit() method with &mut self for exclusive state updates
  - Add universe_size() query method
  - Provide upper_bound() method for lazy evaluation optimization
  - Add optional gain_batch() for SIMD-optimized batch evaluation
  - Include optional fork() for thread-local cache creation
  - Enforce Send + Sync bounds for thread safety
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

- [ ] 2.2 Define Constraint trait for feasibility checking
  - Create is_feasible() method checking if element can be added
  - Add commit() method for constraint state updates
  - Provide reset() method for constraint reinitialization
  - Include rank() method returning maximum selection size
  - Add max_weight_base() method for matroid base computation
  - _Requirements: 11.1, 11.2, 13.2, 13.3, 13.4_

- [ ] 2.3 Implement cardinality constraint
  - Check feasibility based on current selection size vs k
  - Track selection size through SelectionView
  - Require no internal state beyond view size
  - Support reset to zero state
  - _Requirements: 11.1, 11.2, 11.3_

### 3. Build Facility Location objective function with CSR sparse matrix

- [ ] 3.1 Implement CSR sparse matrix data structure
  - Create CsrMatrix with row_ptr, col_indices, values vectors
  - Provide O(nnz(row)) access time for row retrieval
  - Support construction from triplet format with sorting
  - Enable transpose to CSC format when needed
  - Implement memory-efficient storage for billions of entries
  - _Requirements: 3.6, 16.1, 16.2, 16.3_

- [ ] 3.2 Build Facility Location oracle implementation
  - Implement gain() computing marginal contribution in O(nnz(e)) time
  - Calculate improvement as max(0, u[i,e] - best_u[i]) for each demand
  - Implement commit() updating best_u[i] for affected demands
  - Track objective value incrementally during commits
  - Use Arc for shared immutable data (utility matrix, weights)
  - Maintain mutable state for best utilities per demand
  - Support fork() creating thread-local cache with shared data
  - _Requirements: 3.1, 3.2, 3.3, 3.6_

- [ ] 3.3 Add SIMD optimization and batch evaluation
  - Vectorize inner loops processing 4-8 elements per instruction
  - Implement gain_batch() for multiple candidate evaluation
  - Share demand access across batch for cache efficiency
  - Use f32 precision for 2x memory reduction and wider SIMD
  - _Requirements: 3.7, 3.8, 17.1_

### 4. Implement Lazy Greedy algorithm with epoch-based optimization

- [ ] 4.1 Build epoch-based heap with stale bound elimination
  - Create HeapEntry structure with item_id, upper_bound, epoch fields
  - Implement max-heap ordering with deterministic tie-breaking by ItemId
  - Track current_epoch counter incremented on each commit
  - Discard heap entries where entry.epoch < current_epoch
  - Achieve 5-10% evaluation counts compared to standard greedy
  - _Requirements: 6.2, 6.3, 6.4, 6.10_

- [ ] 4.2 Implement Lazy Greedy selection logic
  - Initialize heap with upper bounds from oracle.upper_bound()
  - Pop heap entries and check for staleness
  - Evaluate actual marginal gain for fresh entries
  - Commit element if gain + epsilon ≥ next upper bound
  - Re-insert with updated bound if not definitively best
  - Increment epoch on commit to invalidate all cached bounds
  - Maintain (1-1/e) approximation guarantee for cardinality constraints
  - _Requirements: 6.1, 6.2, 6.5, 6.6, 6.7_

- [ ] 4.3 Add parallel evaluation with deterministic aggregation
  - Evaluate top-M candidates concurrently using oracle.gain()
  - Collect results in parallel using rayon
  - Sort parallel results by ItemId when fixed_order is enabled
  - Aggregate in f64 precision for deterministic reductions
  - Release GIL equivalent for long-running computations
  - _Requirements: 6.8, 6.9, 17.6, 18.5_

- [ ] 4.4 Implement termination conditions
  - Terminate when cardinality constraint k is reached
  - Check upper bound threshold and terminate if next_ub < tau
  - Detect stagnation with consecutive low marginal gains
  - Support timeout with wall-clock time checking at iteration boundaries
  - Log specific termination reason to audit log
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

### 5. Add deterministic execution and metrics infrastructure

- [ ] 5.1 Implement hierarchical RNG seeding
  - Create master seed from user configuration
  - Derive algorithm-specific seeds using master ^ ALGO_TAG
  - Compute iteration seeds as algo_seed + iteration_num
  - Use StdRng for platform-independent random generation
  - Ensure reproducible selection sequences across runs
  - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.7_

- [ ] 5.2 Build Prometheus metrics collection
  - Expose counters for gain evaluations, commits, heap operations
  - Track gauges for objective value, selection size, gap estimate
  - Record histograms for gain computation and iteration times
  - Follow naming convention with snake_case and submod_ prefix
  - Label metrics with algorithm dimension
  - Serve metrics on configurable HTTP endpoint (default 9090)
  - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.6, 19.7_

- [ ] 5.3 Implement JSON Lines audit logging
  - Log iteration with mandatory fields (iteration, element, gain, size, objective, algorithm, seed, timestamp)
  - Include optional git_hash when available
  - Record named thresholds with semantic labels
  - Log termination reason on algorithm completion
  - Write in JSON Lines format for streaming parsability
  - Generate deterministic hash from integer decisions only
  - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.8, 17.7_

### 6. Create minimal working example and foundational tests

- [ ] 6.1 Build quick start example with toy dataset
  - Generate 10KB dataset: 100 demands, 50 candidates, 10% density
  - Create demand_mini.parquet and utility_mini.parquet files
  - Implement end-to-end pipeline completing in <1 second
  - Produce deterministic selection with seed 42
  - Validate coverage curve showing diminishing returns
  - Verify lazy efficiency <10% of standard greedy
  - _Requirements: 24.1, 24.2, 24.3, 24.4, 24.5_

- [ ] 6.2 Write unit tests for core components
  - Test SelectionView operations (insert, contains, size, iteration)
  - Validate type system and validation functions
  - Verify deterministic tie-breaking by ItemId
  - Test constraint feasibility checking
  - Confirm CSR matrix construction and access
  - _Requirements: 23.1, 23.8_

- [ ] 6.3 Add integration tests for Facility Location + Lazy Greedy
  - Test end-to-end optimization pipeline
  - Verify (1-1/e) approximation on small brute-force solvable problems
  - Run 10 identical iterations asserting identical outputs
  - Check selection sequence, objective value, audit log hash consistency
  - Validate metrics collection and audit log format
  - _Requirements: 23.2, 23.3_

---

## Phase 2: Data I/O and Parallel Processing (3-4 weeks)

### 7. Build Parquet data loading with validation

- [ ] 7.1 Implement Parquet reader for demand and utility data
  - Load demand data with schema [i: u32, w: f32]
  - Load utility data with schema [i: u32, s: u32, u: f32]
  - Process row groups with configurable target size (default 192MB)
  - Apply Snappy compression by default with Zstd level 3 option
  - Validate data rejecting NaN, infinity, negative values, out-of-bound IDs
  - Fail-fast on validation errors with detailed error messages
  - _Requirements: 15.1, 15.2, 15.4, 15.5, 15.7_

- [ ] 7.2 Add row-group optimization and predicate pushdown
  - Cluster utility data by demand ID for sequential access
  - Push filters like i IN (...) to Parquet reader for row-group skipping
  - Use row-group statistics (min/max) to skip irrelevant data
  - Read SUBMOD_ROW_GROUP_TARGET_MB environment variable
  - _Requirements: 15.3, 15.6_

- [ ] 7.3 Handle duplicate entries and zero filtering
  - Apply configurable duplicate policy: max (default), sum, average, error
  - Explicitly drop entries with u = 0 during CSR construction
  - Report duplicate handling in logs
  - _Requirements: 15.8, 15.9_

### 8. Implement parallel sharding and prefetching

- [ ] 8.1 Build hash-based sharding system
  - Partition data into P shards (P ≈ 128) by hash(i) mod P
  - Support loading individual shards independently
  - Enable parallel loading across multiple threads
  - Validate shard integrity and completeness
  - _Requirements: 16.4_

- [ ] 8.2 Add producer-consumer pattern for parallel loading
  - Implement crossbeam channel for shard communication
  - Create producer thread pool for parallel shard loading
  - Set up consumer processing loaded shards
  - Bound channel capacity to prevent memory explosion
  - _Requirements: 16.5, 16.7_

- [ ] 8.3 Enable async prefetching for streaming
  - Async load shards i+1, i+2 while processing shard i
  - Monitor SUBMOD_IO_WAIT_SECONDS metric for tuning
  - Auto-tune prefetch depth based on memory budget
  - Balance prefetch queue depth vs memory pressure
  - _Requirements: 16.6, 16.7_

### 9. Integrate data I/O with Facility Location oracle

- [ ] 9.1 Connect Parquet loader to Facility Location
  - Load demand weights and utility matrix from Parquet files
  - Construct CSR sparse matrix from loaded triplets
  - Initialize Facility Location oracle with loaded data
  - Validate matrix dimensions and non-zero count
  - _Requirements: 3.4, 3.5, 15.1, 15.2_

- [ ] 9.2 Test with large-scale synthetic datasets
  - Generate datasets with 10⁴-10⁵ candidates and demands
  - Verify correct loading and processing
  - Measure I/O wait time and prefetch efficiency
  - Profile memory usage during loading
  - _Requirements: 16.5, 16.6_

### 10. Add Stochastic Greedy algorithm

- [ ] 10.1 Implement Stochastic Greedy sampling
  - Sample random subset R_t of size ceil((n/k)·ln(1/ε))
  - Use reservoir sampling for uniform random selection
  - Derive iteration seeds from master seed hierarchically
  - Evaluate best element in sample using argmax
  - _Requirements: 7.2, 7.3, 7.4, 7.5_

- [ ] 10.2 Add parallel evaluation for sampled candidates
  - Evaluate all candidates in sample concurrently
  - Sort results by ItemId for deterministic selection
  - Break ties by lower ItemId when gains are equal
  - Maintain E[f(S_k)] ≥ (1-1/e-ε) approximation guarantee
  - _Requirements: 7.1, 7.6_

- [ ] 10.3 Implement hybrid Lazy Stochastic Greedy
  - Maintain epoch-based heap for sampled candidates only
  - Apply lazy optimization within each sample
  - Reduce evaluations further when sample size is large
  - _Requirements: 7.7_

---

## Phase 3: Additional Objective Functions (4-5 weeks)

### 11. Implement Saturating Coverage objective function

- [ ] 11.1 Build Saturating Coverage oracle with concave functions
  - Implement gain() computing Σᵢ [φᵢ(cumulative[i] + a[i,e]) - φᵢ(cumulative[i])]
  - Implement commit() updating cumulative[i] for affected demands
  - Track cumulative contribution per demand across selections
  - Maintain objective value f(S) = Σᵢ φᵢ(cumulative[i])
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 11.2 Add concave saturation function library
  - Support logarithmic φ(x) = log(1 + x)
  - Support square root φ(x) = √x
  - Support threshold φ(x) = min(x, τ)
  - Enable user-defined concave functions
  - _Requirements: 4.4, 4.5, 4.6_

- [ ] 11.3 Optimize with precomputed lookup tables
  - Build lookup tables for common concave functions
  - Ensure strict monotonicity to prevent numerical violations
  - Use LUT for efficient function evaluation
  - Minimize branching for SIMD-friendly execution
  - _Requirements: 4.7, 4.8_

### 12. Implement Log-Determinant objective with numerical safety

- [ ] 12.1 Build Log-Determinant oracle with f64 precision
  - Mandate f64 precision for all Cholesky computations
  - Compute f(S) = log det(K_{S,S} + εI) with ε ∈ [10⁻⁶, 10⁻⁴]
  - Evaluate marginal gain Δ(e|S) = log(d) from Schur complement
  - Maintain Cholesky decomposition L where K_{S,S} + εI = LL^T
  - Track selected_indices for matrix indexing
  - _Requirements: 5.1, 5.2, 5.3, 17.2_

- [ ] 12.2 Apply epsilon regularization as Layer 1 defense
  - Add εI to kernel matrix at initialization
  - Use epsilon in range [10⁻⁶, 10⁻⁴]
  - Ensure positive definiteness through regularization
  - _Requirements: 5.4_

- [ ] 12.3 Implement safe Schur complement computation as Layer 2
  - Compute Schur diagonal d from Cholesky decomposition
  - Clip result to d_safe = max(d, ε · 0.1)
  - Prevent negative or zero values in logarithm
  - _Requirements: 5.5_

- [ ] 12.4 Add log1p optimization as Layer 3
  - Use log1p when d is small relative to mean diagonal
  - Apply condition: if d < 0.01 · mean(diag) use log1p(d-1)
  - Improve numerical accuracy for small Schur values
  - _Requirements: 5.6_

- [ ] 12.5 Detect NaN and trigger recomputation as Layer 4
  - Monitor for NaN in Cholesky update operations
  - Trigger full Cholesky recomputation on NaN detection
  - Log recomputation events to audit log
  - Track consecutive failure count
  - _Requirements: 5.7_

- [ ] 12.6 Enable graceful degradation as Layer 5
  - Count consecutive numerical failures
  - Degrade to Facility Location mode after 5 failures
  - Log warning about degraded mode activation
  - Continue optimization without diversity term
  - _Requirements: 5.8_

- [ ] 12.7 Add low-rank approximation support
  - Support kernel approximation K ≈ ZZ^T with rank r ≪ n
  - Enable Nyström or random features for massive matrices
  - Compute log det(I + Z_S^T Z_S) in O(r² |S| + r³) time
  - Scale to n > 10,000 with controlled accuracy loss
  - _Requirements: 5.9_

### 13. Test numerical stability and multi-objective integration

- [ ] 13.1 Test Log-Determinant numerical edge cases
  - Verify all 5 layers of numerical defense
  - Test graceful degradation after consecutive failures
  - Validate f64 precision requirement
  - Check NaN rejection and Cholesky recomputation
  - Verify epsilon regularization effectiveness
  - _Requirements: 23.8, 5.4, 5.5, 5.6, 5.7, 5.8_

- [ ] 13.2 Integrate multiple objectives with Lazy Greedy
  - Test Facility Location + Lazy Greedy
  - Test Saturating Coverage + Lazy Greedy
  - Test Log-Determinant + Lazy Greedy
  - Verify deterministic execution across objectives
  - Validate approximation guarantees for each objective
  - _Requirements: 23.2, 18.1, 18.2_

- [ ] 13.3 Add dtype selection and monomorphization
  - Default f32 for Facility Location and Saturating Coverage
  - Enforce f64 for Log-Determinant
  - Enable compile-time monomorphization avoiding runtime branching
  - Apply Kahan summation for improved accuracy in reductions
  - _Requirements: 17.1, 17.2, 17.4, 17.5_

---

## Phase 4: Matroid Constraints and Continuous Greedy (6-8 weeks)

### 14. Implement Continuous Greedy algorithm foundation

- [ ] 14.1 Build multilinear extension optimization
  - Initialize fractional solution x as zero vector
  - Divide time interval [0,1] into T steps with Δt = 1/T
  - Iterate T steps updating fractional solution
  - Use f64 precision for gradient accumulation
  - _Requirements: 8.1, 8.2, 17.3_

- [ ] 14.2 Implement view-based gradient estimation
  - Estimate gradient ĝᵢ = (1/G)·Σⱼ[f(Rⱼ∪{i}) - f(Rⱼ)] where Rⱼ ~ x^(t)
  - Create temporary SelectionView for each sample
  - Never mutate oracle during gradient estimation
  - Evaluate gains using immutable oracle.gain() calls
  - Derive sample seeds hierarchically from master seed
  - _Requirements: 8.3, 8.4, 8.10_

- [ ] 14.3 Add parallel gradient computation
  - Evaluate gradient contributions across elements in parallel
  - Use thread-safe oracle.gain() with temporary views
  - Aggregate gradient estimates in f64 precision
  - Sort contributions by ItemId for deterministic aggregation
  - _Requirements: 8.4, 17.3, 18.5_

- [ ] 14.4 Implement maximum weight base selection
  - Find BASE (maximal independent set of size = rank) not just independent set
  - Compute base using max_weight_base() on constraint
  - Use greedy algorithm for matroid base computation
  - Break ties by ItemId for determinism
  - _Requirements: 8.5, 13.6_

- [ ] 14.5 Update fractional solution
  - Compute x^(t+1)_i = min(x^(t)_i + 1_{i∈B^(t)}/T, 1.0)
  - Clip each component to [0,1] interval
  - Track convergence and objective value progress
  - _Requirements: 8.6_

### 15. Implement matroid constraint library

- [ ] 15.1 Build Partition Matroid constraint
  - Partition ground set V = C₁ ⊔ C₂ ⊔ ... ⊔ C_m into categories
  - Check independence with |S ∩ C_j| ≤ cap_j for all j
  - Compute rank as Σⱼ cap_j
  - Accept Arc<dyn Fn(ItemId) -> usize + Send + Sync> for category assignment
  - Break ties by ItemId when sorting by weight
  - _Requirements: 13.1, 13.2, 13.3, 13.5, 13.6_

- [ ] 15.2 Implement max-weight base for Partition Matroid
  - Greedily select top-cap_j elements by weight from each partition
  - Sort elements within partition by weight descending
  - Apply deterministic tie-breaking by ItemId
  - Return base of size equal to rank
  - _Requirements: 13.4, 13.6_

- [ ] 15.3 Build Graphic Matroid constraint
  - Represent edges with endpoint pairs (u, v)
  - Use Union-Find for cycle detection
  - Check feasibility by testing if edge creates cycle
  - Union endpoints on commit
  - Reset Union-Find to singleton components
  - _Requirements: 14.1, 14.2, 14.4_

- [ ] 15.4 Implement Kruskal's algorithm for Graphic Matroid base
  - Sort edges by weight descending
  - Greedily add edges that don't create cycles
  - Use Union-Find for efficient cycle detection
  - Break ties by ItemId for determinism
  - _Requirements: 14.3_

### 16. Implement Knapsack constraint with dual modes

- [ ] 16.1 Build Knapsack constraint checking
  - Check feasibility with used + cost_fn(e) ≤ budget
  - Accept Arc<dyn Fn(ItemId) -> f64 + Send + Sync> for cost function
  - Update used budget on commit
  - Use relative epsilon tolerance for floating-point comparison
  - _Requirements: 12.1, 12.2, 12.3, 12.7_

- [ ] 16.2 Implement Practical mode (fast heuristic)
  - Enumerate best 1-3 individual items by value
  - Run cost-benefit ratio greedy: argmax Δ(e|S)/cost(e)
  - Return better of enumerated singletons vs greedy solution
  - No (1-1/e) guarantee but fast and empirically good
  - _Requirements: 12.4_

- [ ] 16.3 Implement Theoretical mode (provable guarantee)
  - Apply Continuous Greedy with multilinear relaxation
  - Treat as weighted matroid with cost-scaled rank function
  - Achieve (1-1/e) guarantee via Sviridenko 2004
  - Log mode choice and approximation guarantee
  - _Requirements: 12.5, 12.6_

### 17. Implement rounding algorithms for Continuous Greedy

- [ ] 17.1 Build Pipage rounding for Partition Matroid
  - Identify fractional variable pairs within partitions
  - Iteratively adjust pairs preserving matroid constraints
  - Move mass maintaining expectation non-decreasing
  - Produce deterministic integral solution
  - Threshold at 0.5 to determine final selection
  - _Requirements: 8.7_

- [ ] 17.2 Implement Swap rounding for general Matroid
  - Decompose fractional solution into convex combination of bases
  - Sample base with probability proportional to weight
  - Apply swap operations to merge bases stochastically
  - Use seeded RNG derived from master seed
  - Maintain (1-1/e) expected approximation
  - _Requirements: 8.8_

- [ ] 17.3 Materialize and commit rounded solution
  - Convert integral solution to SelectionView
  - Commit all selected items sequentially to oracle
  - Update constraint state for final selection
  - Record rounding loss in metrics
  - _Requirements: 8.9_

### 18. Add Sieve-Streaming algorithm

- [ ] 18.1 Implement Sieve-Streaming with threshold levels
  - Maintain threshold levels τ₁ > τ₂ > ... > τ_L in geometric sequence
  - Initialize solution sets S_τ for each threshold
  - Process elements in stream order
  - Use O(k·L) memory where L = O(log(1/ε))
  - _Requirements: 9.2, 9.5_

- [ ] 18.2 Add element processing for streaming
  - Compute Δ(e|S_τ) for each threshold when element arrives
  - Add e to S_τ if Δ(e|S_τ) ≥ τ AND |S_τ| < k
  - Maintain multiple solution candidates simultaneously
  - Return best solution at stream end
  - _Requirements: 9.3, 9.4_

- [ ] 18.3 Test Sieve-Streaming on streaming scenarios
  - Verify (1/2-ε) approximation guarantee
  - Validate memory usage O(k·log(1/ε))
  - Test with single-pass data streams
  - Compare against Lazy Greedy baseline
  - _Requirements: 9.1_

### 19. Test matroid constraints and Continuous Greedy

- [ ] 19.1 Write property-based tests for matroids
  - Verify empty set independence property
  - Test downward closure property
  - Validate augmentation property
  - Use proptest for randomized matroid testing
  - _Requirements: 23.1_

- [ ] 19.2 Test Continuous Greedy with matroid constraints
  - Validate (1-1/e) approximation for matroid-constrained problems
  - Test view-based gradient estimation preserves oracle immutability
  - Verify base selection returns maximum weight BASE
  - Test rounding produces valid matroid members
  - _Requirements: 8.1, 8.4, 8.5_

- [ ] 19.3 Add Continuous Greedy specific metrics
  - Expose submod_grad_variance metric
  - Track submod_base_weight_sum metric
  - Record submod_rounding_loss metric
  - Validate metrics collection during optimization
  - _Requirements: 19.5_

---

## Phase 5: Python Bindings and Service Layer (3-4 weeks)

### 20. Build Python bindings with PyO3

- [ ] 20.1 Create PySubmodSolver class with GIL release
  - Implement __init__ accepting objective, strategy, constraint, k, seed
  - Release GIL during solve() using Python::allow_threads()
  - Enable true parallelism preventing blocking of other threads
  - Return solution as Python dict with items, objective, iterations, wall_time
  - _Requirements: 21.1, 21.5, 21.6_

- [ ] 20.2 Add zero-copy data loading from Python
  - Support load_from_parquet() for Parquet file loading
  - Implement load_from_numpy() with zero-copy NumPy access
  - Use Arrow memory layout for efficient interchange
  - Avoid unnecessary copies between Rust and Python heaps
  - _Requirements: 21.2, 21.3_

- [ ] 20.3 Implement error propagation to Python
  - Convert Rust Result<T,E> to Python exceptions
  - Include detailed context (file, line, error chain)
  - Provide meaningful error messages for debugging
  - Handle validation, I/O, algorithm, numerical errors
  - _Requirements: 21.4_

- [ ] 20.4 Write Python integration tests
  - Test solver initialization and configuration
  - Validate solve() with GIL release
  - Test Parquet and NumPy data loading
  - Verify zero-copy performance
  - Check error handling and exception messages
  - _Requirements: 23.3_

### 21. Implement gRPC service layer

- [ ] 21.1 Build gRPC service with tonic
  - Define protobuf schema for JobRequest, JobResponse, JobStatus
  - Implement submit_job endpoint with validation
  - Add get_job_status endpoint for status queries
  - Create get_job_result endpoint returning solution and audit log
  - _Requirements: 22.1, 22.6, 22.7_

- [ ] 21.2 Implement job queue with status tracking
  - Create job queue with queued, running, completed, failed states
  - Assign unique job IDs to submitted jobs
  - Track job progress and completion
  - Store results and audit logs for retrieval
  - _Requirements: 22.3, 22.6_

- [ ] 21.3 Add authentication and authorization
  - Support configurable auth mechanisms (API key, JWT, OAuth)
  - Validate authentication tokens on requests
  - Implement role-based access control for job submission
  - Secure sensitive endpoints
  - _Requirements: 22.4_

### 22. Implement REST service layer

- [ ] 22.1 Build REST API with axum
  - Create POST /api/v1/jobs endpoint for job submission
  - Add GET /api/v1/jobs/:id for status queries
  - Implement GET /api/v1/jobs/:id/result for result retrieval
  - Include GET /metrics for Prometheus metrics export
  - _Requirements: 22.2, 22.5_

- [ ] 22.2 Integrate with job queue
  - Share job queue state between gRPC and REST
  - Enqueue jobs from REST requests
  - Query job status and results
  - Return detailed error responses on failures
  - _Requirements: 22.3, 22.7_

- [ ] 22.3 Test service endpoints
  - Test job submission via gRPC and REST
  - Verify status tracking and result retrieval
  - Validate authentication and authorization
  - Test metrics endpoint for Prometheus scraping
  - Check concurrent job processing
  - _Requirements: 22.5_

---

## Phase 6: Testing, Documentation, and CI/CD (2-3 weeks)

### 23. Comprehensive testing and quality assurance

- [ ] 23.1 Write property-based tests with proptest
  - Test submodularity property: f(S∪T) + f(S∩T) ≤ f(S) + f(T)
  - Verify monotonicity: f(S) ≤ f(S∪{e}) for all e
  - Validate constraint invariants (independence, augmentation)
  - Generate random inputs for stress testing
  - _Requirements: 23.1_

- [ ] 23.2 Test approximation quality guarantees
  - Compare greedy solutions to brute-force optimal on small problems
  - Verify f(S_greedy) ≥ (1-1/e)·f(S_opt) for cardinality constraints
  - Test Stochastic Greedy expected approximation E[f(S)] ≥ (1-1/e-ε)
  - Validate Continuous Greedy approximation for matroids
  - _Requirements: 23.2_

- [ ] 23.3 Implement determinism validation tests
  - Run 10 identical iterations with same seed and configuration
  - Assert identical selection sequence across runs
  - Verify identical objective value within floating-point epsilon
  - Check identical audit log hash across runs
  - Test with different parallelism levels (1, 4, 16, 64 threads)
  - _Requirements: 23.3, 18.1, 18.2, 18.3_

- [ ] 23.4 Set up cross-platform CI matrix
  - Configure GitHub Actions for {Ubuntu, macOS, Windows} × {stable, nightly}
  - Run full test suite on all platforms
  - Execute determinism tests across platforms
  - Accept objective value differences <10⁻⁶ across platforms
  - _Requirements: 23.4, 18.8, 25.1, 25.4_

- [ ] 23.5 Add performance benchmarks with criterion
  - Benchmark Lazy Greedy on standard datasets
  - Measure gain evaluations per second
  - Track heap operations per iteration
  - Monitor wall time for baseline problem (n=10⁶, m=10⁶, nnz=10⁸, k=250)
  - Target <10 min runtime on 32-core 128GB RAM
  - Detect >1% performance regression on standard datasets
  - _Requirements: 23.5, 23.6, 23.7, 25.3_

- [ ] 23.6 Test numerical edge cases
  - Verify NaN rejection in validation
  - Test infinity rejection
  - Validate negative value rejection
  - Confirm Log-Determinant graceful degradation after 5 failures
  - Test epsilon clipping and log1p accuracy
  - _Requirements: 23.8_

### 24. Create documentation and examples

- [ ] 24.1 Write user guide and tutorials
  - Provide getting started guide
  - Document common use cases for each objective function
  - Include examples for each algorithm
  - Show constraint usage patterns
  - Explain determinism configuration
  - _Requirements: 24.6_

- [ ] 24.2 Generate API reference documentation
  - Document all public traits, types, and functions
  - Include code examples in doc comments
  - Generate rustdoc with cargo doc
  - Publish documentation to docs.rs
  - _Requirements: 24.6_

- [ ] 24.3 Create examples for all features
  - Facility Location example with Lazy Greedy
  - Saturating Coverage example with Stochastic Greedy
  - Log-Determinant example with deterministic execution
  - Continuous Greedy example with Partition Matroid
  - Knapsack constraint example with practical mode
  - Python binding usage example
  - Service layer example with gRPC and REST
  - _Requirements: 24.7_

- [ ] 24.4 Enhance quick start example
  - Include verification checklist in documentation
  - Provide Python generation script for mini dataset
  - Add expected output for validation
  - Document performance expectations (<1 second)
  - Show coverage curve visualization
  - _Requirements: 24.2, 24.5_

### 25. Set up CI/CD and release automation

- [ ] 25.1 Configure continuous integration
  - Run test matrix on all commits
  - Execute determinism tests with 10 iterations
  - Run criterion benchmarks for regression detection
  - Check code formatting with rustfmt
  - Lint with clippy
  - _Requirements: 25.1, 25.2, 25.3_

- [ ] 25.2 Add release automation
  - Build binaries for all platforms on release
  - Create release artifacts with checksums
  - Publish crates to crates.io
  - Verify version bumps and changelog updates
  - Check documentation completeness
  - _Requirements: 25.5, 25.6_

- [ ] 25.3 Create integration test suite
  - Test complete workflow from data loading to result
  - Validate Python bindings integration
  - Test service layer with real requests
  - Run large-scale tests on synthetic datasets
  - Verify monitoring and observability features
  - _Requirements: 23.4_

---

## Requirements Coverage Summary

All 25 requirements are covered by the tasks above:

- **Req 1-2:** Tasks 1.2, 1.3, 1.4, 2.1, 2.2 (Core Framework and Oracle Interface)
- **Req 3:** Tasks 3.1, 3.2, 3.3, 9.1 (Facility Location)
- **Req 4:** Tasks 11.1, 11.2, 11.3 (Saturating Coverage)
- **Req 5:** Tasks 12.1-12.7, 13.1 (Log-Determinant)
- **Req 6:** Tasks 4.1-4.4 (Lazy Greedy)
- **Req 7:** Tasks 10.1-10.3 (Stochastic Greedy)
- **Req 8:** Tasks 14.1-14.5, 17.1-17.3, 19.2 (Continuous Greedy)
- **Req 9:** Tasks 18.1-18.3 (Sieve-Streaming)
- **Req 10:** Task 4.4 (Termination Conditions)
- **Req 11:** Task 2.3 (Cardinality Constraint)
- **Req 12:** Tasks 16.1-16.3 (Knapsack Constraint)
- **Req 13:** Tasks 15.1-15.2 (Partition Matroid)
- **Req 14:** Tasks 15.3-15.4 (Graphic Matroid)
- **Req 15:** Tasks 7.1-7.3, 9.1 (Parquet I/O)
- **Req 16:** Tasks 8.1-8.3, 9.2 (Sparse Matrix and Sharding)
- **Req 17:** Tasks 1.2, 3.3, 13.3 (Numerical Stability)
- **Req 18:** Tasks 5.1, 23.3, 23.4 (Deterministic Execution)
- **Req 19:** Tasks 5.2, 19.3 (Prometheus Metrics)
- **Req 20:** Task 5.3 (Audit Logs)
- **Req 21:** Tasks 20.1-20.4 (Python Bindings)
- **Req 22:** Tasks 21.1-21.3, 22.1-22.3 (Service Layer)
- **Req 23:** Tasks 23.1-23.6 (Testing)
- **Req 24:** Tasks 6.1, 24.1-24.4 (Documentation)
- **Req 25:** Tasks 23.4, 25.1-25.3 (CI/CD)

---

## Implementation Notes

**Integration Strategy:**
- Each phase builds on previous phases
- Test integration continuously, not just at the end
- Maintain working end-to-end pipeline throughout
- Commit working code frequently

**Testing Approach:**
- Write tests alongside implementation
- Use TDD for complex algorithms
- Validate correctness before optimizing performance
- Run determinism tests regularly

**Quality Gates:**
- All tests must pass before proceeding to next phase
- Code coverage >80% for core components
- Performance benchmarks within 10% of targets
- Documentation complete for public APIs

**Timeline Flexibility:**
- Phase estimates are ranges (e.g., 4-6 weeks)
- Adjust based on actual progress and complexity
- Prioritize correctness over speed
- Budget extra time for numerical stability debugging
