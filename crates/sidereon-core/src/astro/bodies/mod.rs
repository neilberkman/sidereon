//! Low-precision analytic Sun and Moon ephemerides.
//!
//! [`sun_moon`] ports the analytic low-precision solar and lunar position
//! formulae used for site-displacement and antenna-geometry corrections in
//! precise GNSS processing (solid-earth tide, carrier-phase wind-up, and
//! satellite antenna phase-center offset). The analytic ECI positions follow
//! the standard low-precision series (Montenbruck & Gill, "Satellite Orbits",
//! sections 3.3.2 / 3.3.3), expressed here in the same form used by the
//! widely-available open GNSS implementations. The series are referred to the
//! mean equator and equinox of date, so they are rotated to the Earth-fixed
//! frame with [`crate::astro::frames::transforms::mean_of_date_to_itrs_matrix`]
//! (nutation + GAST, using the crate's IAU 2000A nutation). Precession is already
//! implicit in the of-date series, so the full GCRS->ITRS transform is NOT used
//! here, which would otherwise double-count precession; this mirrors the
//! GMST/GAST + nutation rotation those references consume the series with.
//!
//! Precision is at the few-centimetre / sub-degree level, sufficient for the
//! tidal and antenna-geometry corrections that consume these positions.

pub mod rise_set;
pub mod sun_moon;

pub use rise_set::{
    find_sun_elevation_crossings, sun_elevation_deg, SunElevationCrossing,
    SunElevationCrossingKind, SunElevationOptions,
};
pub use sun_moon::{sun_moon_ecef, sun_moon_eci, sun_moon_eci_at, SunMoon, SunMoonError};

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
