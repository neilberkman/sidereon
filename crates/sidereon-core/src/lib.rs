//! # sidereon-core
//!
//! The complete Sidereon engine in one crate. It folds the numerical
//! astrodynamics core (orbit propagation, force models, frames, time, SGP4)
//! together with the GNSS domain layer (SP3, broadcast ephemeris, multi-GNSS
//! positioning, RTK/PPP, ionosphere/troposphere, DOP).
//!
//! - The propagation/astro layer is always present under the [`astro`] module.
//! - The GNSS layer lives behind the default-on `gnss` cargo feature, so a
//!   propagation-only consumer can build with `--no-default-features` (plus
//!   any astro features it wants) and never compile the IONEX/SP3 parsers.
//!
//! The GNSS façade is organized by user-facing tasks:
//!
//! - [`ephemeris`] - precise SP3 and broadcast ephemeris products,
//! - [`rinex`] - RINEX navigation/observation parsing and CRINEX decoding,
//! - [`antex`] - ANTEX receiver and satellite antenna calibration parsing,
//! - [`combinations`] - observable linear combinations such as ionosphere-free,
//! - [`observables`] - forward range, Doppler, and azimuth/elevation prediction,
//! - [`velocity`] - receiver velocity and clock-drift solve from range-rate data,
//! - [`positioning`] - single-point positioning and DOP diagnostics,
//! - [`dgnss`] - code-differential pseudorange correction and rover pairing,
//! - [`quality`] - pseudorange weighting, RAIM, and FDE integrity checks,
//! - [`signal`] - GPS C/A code generation, correlation, and acquisition,
//! - [`ppp_corrections`] - static-arc PPP correction precomputation,
//! - [`atmosphere`] - ionosphere and troposphere corrections,
//! - [`orbit`] - compact reduced-orbit fitting/evaluation.
//!
//! Implementation modules (`sp3`, `rinex_nav`, `spp`, etc.) are crate-private.
//! This is a clean public surface rather than a compatibility shim around the
//! original implementation-shaped module layout.
//!
//! ## Units policy (internal representation)
//!
//! All quantities are stored and computed in **SI base units**, with the frame
//! and datum encoded in the type name (per the spec's frames-in-the-type-system
//! rule), never hidden behind a bare `position_m`:
//!
//! - **Length / position:** meters (`_m`). SP3 positions are ITRF/IGS-frame
//!   ECEF meters; SPP receiver positions are WGS84/ITRF-compatible ECEF meters.
//!   (The [`astro`] state layer works in kilometers; conversions happen
//!   explicitly at the boundary, never implicitly.)
//! - **Time / clock:** seconds (`_s`). Epochs are represented by the [`astro`]
//!   time family (`Instant`/`TimeScale`), always scale-tagged; there is no bare
//!   ambiguous epoch.
//! - **Velocity:** meters per second (`_m_s`).
//! - **Angles:** radians (`_rad`) internally. Degrees appear only at I/O edges
//!   and are named `_deg`.
//! - **Frequency:** hertz (`_hz`).
//!
//! Field and parameter names carry the unit suffix so the unit is visible at
//! every call site. Matrix/vector linear algebra uses `nalgebra`
//! (`DMatrix`/`DVector`) per the spec.

// ---------------------------------------------------------------------------
// Astro / propagation layer. Always present. The GNSS layer below depends on
// it via `crate::astro::*`.
// ---------------------------------------------------------------------------

mod validate;

#[cfg(all(test, sidereon_repo_tests))]
mod test_parity;

pub mod astro;
pub(crate) mod format;

// ---------------------------------------------------------------------------
// GNSS domain layer. Behind the default-on `gnss` feature so a propagation-only
// consumer can opt out. Additional product modules are added as each lands.
// ---------------------------------------------------------------------------

mod ambiguity; // shared RTK/PPP cycle-slip policy + wide-lane/narrow-lane prep
mod antenna; // shared ANTEX PCV/PCO zenith/azimuth interpolation kernels
pub mod antex; // ANTEX receiver/satellite antenna parser + PCO/PCV lookup
mod broadcast; // broadcast-ephemeris (GPS LNAV / Galileo I/NAV) orbit + clock
pub mod broadcast_comparison; // broadcast-vs-precise (SISRE orbit/clock) accuracy
pub mod carrier_phase; // carrier-phase combinations, cycle-slip detection, Hatch smoothing
pub mod constants; // shared physical/time constants (used by astro + gnss)
pub mod constellation; // GNSS constellation identity catalog (CelesTrak/NAVCEN)
mod crinex; // Hatanaka (CRINEX) observation-file decoder
mod dop; // dilution-of-precision geometry (GDOP/PDOP/HDOP/VDOP/TDOP)
pub mod frequencies; // canonical GNSS carrier-frequency table
mod glonass; // GLONASS PZ-90.11 state-vector RK4 propagation
mod ionex; // Klobuchar broadcast model + IONEX ionospheric maps
pub mod navigation; // navigation-message bit-level codecs (GPS LNAV)
pub mod observables; // forward GNSS observable prediction
pub mod ppp_corrections; // static-arc PPP correction tables
pub mod precise_positioning; // static multi-epoch PPP float solve
mod reduced_orbit; // compact mean-element orbit approximation (fitted)
mod rinex_clock; // RINEX clock satellite-bias parsing and interpolation
mod rinex_common; // shared RINEX header concepts (time-system label mapping)
mod rinex_nav; // RINEX 3 navigation-message parsing (GPS/Galileo broadcast)
mod rinex_obs; // RINEX 3 observation parsing + single-frequency pseudoranges
pub mod rtcm; // RTCM 3 differential-GNSS stream decode/encode (MSM, station, ephemeris)
pub mod rtk; // RTK double-difference construction
pub mod signal; // GPS C/A code, coherent correlation, and acquisition
mod sp3; // SP3-c / SP3-d parser + arbitrary-epoch interpolation
mod spp; // single-point positioning (least-squares PVT)
pub mod staleness; // product-staleness graceful degradation for time-varying products
mod tropo; // Saastamoinen zenith + Niell (NMF) mapping troposphere
pub mod velocity; // receiver velocity / clock-drift least-squares solve

mod error;
pub mod frame;
mod id;

pub mod atmosphere;
pub mod combinations;
pub mod dgnss;
pub mod ephemeris;
pub mod estimation; // Phase-2 estimation substrate: named operation-order recipes
pub mod geoid; // geoid undulation grid + bilinear interpolation (orthometric heights)
pub mod geometry;
pub mod ils; // integer least squares ambiguity-resolution kernels
pub mod orbit;
pub mod positioning;
pub mod prelude;
pub mod quality; // measurement weighting, RAIM, and FDE integrity checks
pub mod rinex;
pub mod rtk_filter; // sequential RTK baseline filter - serializable state ABI (kernel migration)
pub mod terrain;
pub mod tides;
pub mod tolerances;

pub use error::{Error, Result};
pub use frame::{
    geodetic_to_itrf, itrf_to_geodetic, FrameValueError, ItrfPositionM, ItrfVelocityMS,
    Wgs84Geodetic,
};
pub use geoid::{
    ellipsoidal_height_m, geoid_undulation, orthometric_height_m, GeoidError, GeoidGrid,
};
pub use id::{GnssSatelliteId, GnssSystem, SatelliteIdError};
