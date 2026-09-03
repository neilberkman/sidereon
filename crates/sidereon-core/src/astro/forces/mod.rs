pub mod albedo;
/// Combines the accelerations returned by multiple [`ForceModel`]
/// components in insertion order and returns the first component error.
pub mod composite;
pub mod drag;
pub mod geopotential;
/// Provides [`J2Gravity`], a degree-2 zonal-harmonic perturbation model built
/// after [`TwoBodyGravity`] by [`crate::astro::propagator::ForceModelKind::TwoBodyJ2`].
pub mod j2;
pub mod relativity;
pub mod srp;
pub mod third_body;
pub mod tides;
/// Defines the thread-safe [`ForceModel`] acceleration interface used by
/// numerical propagation. Its [`ForceModel::acceleration`] method returns an
/// acceleration in km/s² or [`crate::astro::error::PropagationError`].
pub mod r#trait;
/// Provides [`TwoBodyGravity`], which returns central point-mass acceleration
/// and reports an exact zero position norm as
/// [`crate::astro::error::PropagationError::NumericalFailure`].
pub mod two_body;
pub mod zonal;

pub use albedo::EarthRadiationPressure;
pub use composite::CompositeForceModel;
pub use drag::{DragForce, DragParameters, SourcedDragForce, SpaceWeather, SpaceWeatherSource};
pub use geopotential::{
    SphericalHarmonicCoefficient, SphericalHarmonicGravity, SphericalHarmonicGravityConfig,
    EGM96_DEGREE_ORDER_36, EGM96_EMBEDDED_MAX_DEGREE, EGM96_EMBEDDED_MAX_ORDER, EGM96_MU_KM3_S2,
    EGM96_REFERENCE_RADIUS_KM,
};
pub use j2::J2Gravity;
pub use r#trait::ForceModel;
pub use relativity::SchwarzschildRelativity;
pub use srp::SolarRadiationPressure;
pub use third_body::{ThirdBodyBodies, ThirdBodyGravity};
pub use tides::{
    SolidEarthPoleTideGravity, SolidEarthTideGravity, SOLID_EARTH_POLE_TIDE_IMAG_COUPLING,
    SOLID_EARTH_POLE_TIDE_SCALE, SOLID_EARTH_TIDE_K20_IMAG, SOLID_EARTH_TIDE_K20_PLUS,
    SOLID_EARTH_TIDE_K20_REAL, SOLID_EARTH_TIDE_K21_IMAG, SOLID_EARTH_TIDE_K21_PLUS,
    SOLID_EARTH_TIDE_K21_REAL, SOLID_EARTH_TIDE_K22_IMAG, SOLID_EARTH_TIDE_K22_PLUS,
    SOLID_EARTH_TIDE_K22_REAL, SOLID_EARTH_TIDE_K30_REAL, SOLID_EARTH_TIDE_K31_REAL,
    SOLID_EARTH_TIDE_K32_REAL, SOLID_EARTH_TIDE_K33_REAL,
};
pub use two_body::TwoBodyGravity;
pub use zonal::{ZonalCoefficients, ZonalDegrees, ZonalGravity};
