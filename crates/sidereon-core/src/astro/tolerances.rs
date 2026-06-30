//! Shared named numerical tolerances for deterministic scalar kernels.
//!
//! These are algorithmic thresholds, not physical-truth validation bounds.

/// Singular-pivot threshold for deterministic Gaussian-elimination kernels.
pub const PIVOT_EPSILON: f64 = 1.0e-12;
