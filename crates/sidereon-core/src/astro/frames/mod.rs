//! High-accuracy frame transforms (Skyfield-compatible, 0-ULP).
//!
//! The precise frame-transform substrate that used to be `pub(crate)` inside
//! `orbis_nif`, now a public part of the core crate. It exposes:
//!
//! - [`nutation`] - IAU 2000A nutation in longitude/obliquity, mean obliquity,
//!   the nutation rotation matrix, and the equation-of-equinoxes complementary
//!   terms. Depends on [`crate::astro::data::iau2000a`] + [`crate::astro::math::mat3`].
//! - [`precession`] - IAU 2006 precession matrix and the ICRS->J2000 frame bias.
//!   Depends on [`crate::astro::math::mat3`].
//! - [`transforms`] - the transform engine: TEME->GCRS, GCRS->ITRS,
//!   ITRS->geodetic (WGS84), geodetic->ITRS, and topocentric az/el/range.
//!   Depends on [`nutation`], [`precession`], [`crate::astro::math::mat3`],
//!   and [`crate::astro::time::scales`].
//!
//! The numerics are byte-for-byte identical to the `orbis_nif` originals so the
//! existing Skyfield 0-ULP parity (`test/skyfield_parity_test.exs`) holds. The
//! only changes on relocation are visibility (`pub(crate)` -> `pub`) and import
//! paths; the operation order, summation order, transcendental sequence, and the
//! single sanctioned `mul_add` site are preserved exactly.
//!
//! Per the crate-boundary invariant, the Rustler decode/encode shims
//! (`*_impl`, `parse_datetime_tuple`) stay in `orbis_nif`; only the pure
//! float-producing compute functions live here.

pub mod nutation;
pub mod precession;
pub mod transforms;
