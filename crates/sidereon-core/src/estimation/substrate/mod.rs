//! Shared estimation substrate kernels (Phase-2).
//!
//! The substrate owns the parity-sensitive primitives the SPP / RTK / PPP
//! strategies share, exposing each as a recipe-keyed entry so a strategy selects
//! the floating-point operation order it needs by enum value (see
//! [`crate::estimation::recipe`]) rather than owning a private copy of the
//! helper. P1 lands the frame and range kernels; P2 adds the parameter layout,
//! the weighted measurement row, the measurement covariance block, and the
//! normal-equation assembler; P3 adds the shared integer-ambiguity resolver and
//! the residual-screening / QC layer; later phases add the owned solver.

pub(crate) mod ambiguity;
pub(crate) mod frames;
pub(crate) mod normal;
pub(crate) mod parameters;
pub(crate) mod qc;
pub(crate) mod range;
pub(crate) mod rows;
