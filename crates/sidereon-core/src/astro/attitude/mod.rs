//! Spacecraft attitude laws.
//!
//! [`gnss`] holds the yaw-steering laws of the GNSS satellite blocks: nominal
//! yaw steering, the block-specific noon and midnight turns and shadow
//! crossings, orbit-normal mode, and the Sun-fixed DYB frame the empirical
//! solar radiation pressure models are expressed in.

pub mod gnss;
