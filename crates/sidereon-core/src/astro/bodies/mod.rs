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

pub mod observe;
pub mod rise_set;
pub mod sun_moon;

pub use observe::{
    moon_az_el, moon_az_el_with_validity, moon_illumination, moon_illumination_with_validity,
    observe, observe_spk_body, observe_spk_body_with_validity, observe_with_time_scales,
    observe_with_validity, sun_az_el, sun_az_el_with_validity, BodyAzEl, BodyObservationError,
    Ecliptic, Equatorial, Horizontal, MoonIllumination, Observation, ObserveError, ObserveOptions,
    Refraction, Target,
};
pub use rise_set::{
    find_moon_elevation_crossings, find_moon_elevation_crossings_with_validity, find_moon_transits,
    find_moon_transits_with_validity, find_sun_elevation_crossings,
    find_sun_elevation_crossings_with_validity, moon_elevation_deg,
    moon_elevation_deg_with_validity, sun_elevation_deg, sun_elevation_deg_with_validity,
    MoonElevationCrossing, MoonElevationCrossingKind, MoonElevationOptions, MoonTransit,
    MoonTransitKind, SunElevationCrossing, SunElevationCrossingKind, SunElevationOptions,
};
pub use sun_moon::{
    sun_moon_ecef, sun_moon_ecef_with_polar_motion, sun_moon_eci, sun_moon_eci_at, SunMoon,
    SunMoonError,
};

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
