//! High-accuracy frame transforms (Skyfield-compatible, 0-ULP).
//!
//! The precise frame-transform substrate, a public part of the core crate. It
//! exposes:
//!
//! - [`nutation`] - IAU 2000A nutation in longitude/obliquity, mean obliquity,
//!   the nutation rotation matrix, and the equation-of-equinoxes complementary
//!   terms. Depends on [`crate::astro::data::iau2000a`] + [`crate::astro::math::mat3`].
//! - [`precession`] - IAU 2006 precession matrix and the ICRS->J2000 frame bias.
//!   Depends on [`crate::astro::math::mat3`].
//! - [`orientation`] - a cacheable full GCRF<->ITRF Earth-orientation
//!   evaluation built from the existing transform substrate.
//! - [`transforms`] - the transform engine: TEME->GCRS, GCRS->ITRS,
//!   ITRS->geodetic (WGS84), geodetic->ITRS, and topocentric az/el/range.
//!   Depends on [`nutation`], [`precession`], [`crate::astro::math::mat3`],
//!   and [`crate::astro::time::scales`].
//!
//! The operation order, summation order, transcendental sequence, and the
//! single sanctioned `mul_add` site follow Skyfield exactly, so the transforms
//! agree with Skyfield to 0 ULP; `crates/sidereon/tests/skyfield_parity.rs`
//! checks that bit for bit against captured Skyfield 1.49 vectors.
//!
//! Only the pure float-producing compute functions live here; the language
//! bindings add nothing but decode/encode shims around them.

pub mod nutation;
pub mod orientation;
pub mod precession;
pub mod transforms;

pub use orientation::{
    EarthOrientation, EarthOrientationProvider, PolarMotionSample,
    PolarMotionSeriesEarthOrientationProvider, TdbEarthOrientationProvider,
};
