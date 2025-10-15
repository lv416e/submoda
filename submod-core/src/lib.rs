//! # submod-core
//!
//! Core types, traits, and infrastructure for the submodular optimization platform.
//!
//! ## Currently Implemented
//!
//! - Type system primitives (`ItemId`, `Weight`)
//!
//! ## Planned (TODO)
//!
//! The following features will be implemented in future tasks:
//! - `SubmodularOracle` trait for thread-safe marginal gain evaluation (Task 1.2)
//! - `SelectionView` with `FixedBitSet` for O(1) membership testing (Task 1.2)
//! - Configuration types for strategies, constraints, and determinism (Task 1.2)
//! - Validation functions for numerical inputs (Task 1.2)

#![warn(missing_docs)]
#![warn(clippy::all)]

/// Core type for element identifiers.
///
/// Uses `u32` by default, supporting up to 4.3 billion candidates.
/// Can be configured to `u64` for extreme-scale problems.
pub type ItemId = u32;

/// Core type for numerical weights and objective values.
///
/// Currently: `f32` for memory efficiency and wider SIMD.
/// Future: May support `f64` via feature flag for numerically sensitive operations (e.g., Log-Determinant).
pub type Weight = f32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_id_type() {
        let _id: ItemId = 0;
        assert_eq!(std::mem::size_of::<ItemId>(), 4);
    }

    #[test]
    fn test_weight_type() {
        let _weight: Weight = 1.0;
        assert_eq!(std::mem::size_of::<Weight>(), 4);
    }
}
