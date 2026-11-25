pub mod albedo;
/// Combines the accelerations returned by multiple [`ForceModel`]
/// components in insertion order and returns the first component error.
pub mod composite;
pub mod drag;
pub mod ecom;
pub mod empirical;
pub mod geopotential;
pub mod icgem;
/// Provides [`J2Gravity`], a degree-2 zonal-harmonic perturbation model built
/// after [`TwoBodyGravity`] by [`crate::astro::propagator::ForceModelKind::TwoBodyJ2`].
pub mod j2;
pub mod relativity;
mod scaled;
pub mod srp;
pub mod third_body;
pub mod tides;
/// Defines the thread-safe [`ForceModel`] acceleration interface used by
/// numerical propagation. Its [`ForceModel::acceleration`] method returns an
/// acceleration in km/s² or [`crate::astro::error::PropagationError`]; the
/// partial-derivative and estimable-parameter methods serve the variational
/// equations and orbit determination.
pub mod r#trait;
/// Provides [`TwoBodyGravity`], which returns central point-mass acceleration
/// and reports an exact zero position norm as
/// [`crate::astro::error::PropagationError::NumericalFailure`].
pub mod two_body;
pub mod zonal;

pub use crate::astro::events::eclipse::ShadowFunction;
pub use albedo::EarthRadiationPressure;
pub use composite::CompositeForceModel;
pub use drag::{DragForce, DragParameters, SourcedDragForce, SpaceWeather, SpaceWeatherSource};
pub use ecom::{
    ecom_geometry, ecom_geometry_with_shadow, Ecom1Coefficients, Ecom1SolarRadiationPressure,
    Ecom2D2b1Coefficients, Ecom2D4b1Coefficients, Ecom2SolarRadiationPressure, EcomGeometry,
    ECOM1_COEFFICIENT_COUNT, ECOM2_D2B1_COEFFICIENT_COUNT, ECOM2_D4B1_COEFFICIENT_COUNT,
};
pub use empirical::{EmpiricalHarmonic, EmpiricalRtnAcceleration, RtnAxis};
pub use geopotential::{
    SphericalHarmonicCoefficient, SphericalHarmonicGravity, SphericalHarmonicGravityConfig,
    TideSystem, EGM96_DEGREE_ORDER_36, EGM96_EMBEDDED_MAX_DEGREE, EGM96_EMBEDDED_MAX_ORDER,
    EGM96_MU_KM3_S2, EGM96_REFERENCE_RADIUS_KM,
};
pub use icgem::{
    IcgemError, IcgemErrors, IcgemFormat, IcgemGravityField, IcgemHeader, IcgemNormalization,
    IcgemSigmas, IcgemTermKind, IcgemTimeVariableTerm,
};
pub use j2::J2Gravity;
pub use r#trait::{
    difference_quotient_partials, expect_parameter_count, AccelerationPartials, ForceModel,
    ForceParameter,
};
pub use relativity::SchwarzschildRelativity;
pub use srp::SolarRadiationPressure;
pub use third_body::{ThirdBodyBodies, ThirdBodyGravity};
pub use tides::{
    SolidEarthPoleTideGravity, SolidEarthTideGravity, PERMANENT_TIDE_H0_M,
    SOLID_EARTH_POLE_TIDE_IMAG_COUPLING, SOLID_EARTH_POLE_TIDE_SCALE, SOLID_EARTH_TIDE_A0_PER_M,
    SOLID_EARTH_TIDE_K20_IMAG, SOLID_EARTH_TIDE_K20_PLUS, SOLID_EARTH_TIDE_K20_REAL,
    SOLID_EARTH_TIDE_K21_IMAG, SOLID_EARTH_TIDE_K21_PLUS, SOLID_EARTH_TIDE_K21_REAL,
    SOLID_EARTH_TIDE_K22_IMAG, SOLID_EARTH_TIDE_K22_PLUS, SOLID_EARTH_TIDE_K22_REAL,
    SOLID_EARTH_TIDE_K30_REAL, SOLID_EARTH_TIDE_K31_REAL, SOLID_EARTH_TIDE_K32_REAL,
    SOLID_EARTH_TIDE_K33_REAL,
};
pub use two_body::TwoBodyGravity;
pub use zonal::{ZonalCoefficients, ZonalDegrees, ZonalGravity};

#[cfg(test)]
mod finite_tests;
#[cfg(test)]
mod partials_tests;

#[cfg(test)]
pub(crate) mod test_helpers {
    use crate::astro::bodies::sun_moon::sun_moon_eci;
    use crate::astro::constants::time::{DAYS_PER_JULIAN_CENTURY, SECONDS_PER_DAY};
    use crate::astro::constants::units::M_PER_KM;
    use crate::astro::forces::third_body::sun_moon_positions_km;
    use crate::astro::time::civil::j2000_seconds;
    use crate::astro::time::scales::tt_from_tdb_j2000_seconds;
    use nalgebra::Vector3;

    /// TDB epochs from 1990 to 2050, one at the 2024-04-08 solar eclipse.
    pub(crate) fn epochs_1990_to_2050() -> [f64; 4] {
        [
            j2000_seconds(1990, 2, 1, 0, 0, 0.0),
            j2000_seconds(2024, 4, 8, 18, 17, 0.0),
            j2000_seconds(2026, 11, 23, 0, 0, 0.0),
            j2000_seconds(2050, 6, 30, 12, 0, 0.0),
        ]
    }

    /// Geocentric GCRS Sun and Moon positions (km) via the force-model
    /// production path.
    pub(crate) fn gcrs_sun_moon_km(epoch_tdb_seconds: f64) -> (Vector3<f64>, Vector3<f64>) {
        let bodies = sun_moon_positions_km(
            epoch_tdb_seconds,
            &crate::astro::propagator::PropagationContext::default(),
        )
        .expect("GCRS Sun/Moon");
        (bodies.sun_km, bodies.moon_km)
    }

    /// Geocentric mean-of-date Sun and Moon positions (km) at the TT of
    /// `epoch_tdb_seconds`.
    pub(crate) fn mean_of_date_sun_moon_km(epoch_tdb_seconds: f64) -> (Vector3<f64>, Vector3<f64>) {
        let t = tt_from_tdb_j2000_seconds(epoch_tdb_seconds)
            / (DAYS_PER_JULIAN_CENTURY * SECONDS_PER_DAY);
        let bodies = sun_moon_eci(t).expect("mean of date");
        let km = |v: [f64; 3]| Vector3::new(v[0] / M_PER_KM, v[1] / M_PER_KM, v[2] / M_PER_KM);
        (km(bodies.sun), km(bodies.moon))
    }
}
