//! Satellite angular geometry against celestial bodies.
//!
//! Computes nadir/Sun, nadir/Moon, Sun-elevation, phase, and Earth angular
//! radius angles from GCRS position vectors (km), returning degrees. This is
//! the authoritative implementation; the Elixir binding is a thin marshaling
//! layer over it, and the high-level `compute` orchestration (TLE propagation
//! plus ephemeris lookup) stays caller-side over the already-core kernels.

use crate::astro::constants::earth::WGS84_A_KM;
use crate::astro::constants::units::DEGREES_PER_SEMICIRCLE;
use crate::astro::math::vec3;

/// A right angle in degrees: elevation is the complement of the zenith angle.
const RIGHT_ANGLE_DEG: f64 = 90.0;

/// Error while computing satellite angular geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AngleError {
    #[error("invalid angle input {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

/// Radians to degrees in the reference operation order (`rad * 180 / pi`,
/// multiply before divide), required for bit-exact parity with the prior
/// Elixir reference rather than a single rounded `RAD_TO_DEG` constant.
#[inline]
pub fn rad_to_deg_ref(rad: f64) -> f64 {
    rad * DEGREES_PER_SEMICIRCLE / std::f64::consts::PI
}

/// Snap a geodetic longitude (radians) off the `-pi` branch cut onto `+pi`.
///
/// WGS84 geodetic longitude is reported on the half-open interval `(-pi, pi]`;
/// a value at exactly `-pi` denotes the same meridian as `+pi`, so it is folded
/// up. Every other value passes through unchanged. Shared by the precise
/// positioning, RTK, and geometry receiver-geodetic paths so the branch-cut
/// convention is defined once.
#[inline]
pub fn normalize_geodetic_lon_rad(lon_rad: f64) -> f64 {
    if lon_rad <= -std::f64::consts::PI {
        std::f64::consts::PI
    } else {
        lon_rad
    }
}

/// Angle (degrees) between two vectors via the clamped cosine, in the
/// reference operation order.
#[inline]
fn angle_between(
    a: [f64; 3],
    a_field: &'static str,
    b: [f64; 3],
    b_field: &'static str,
) -> Result<f64, AngleError> {
    validate_nonzero_vec3(a, a_field)?;
    validate_nonzero_vec3(b, b_field)?;
    let cos_theta = vec3::dot3(a, b) / (vec3::norm3(a) * vec3::norm3(b));
    // Clamp into the valid cosine domain for numerical safety.
    let cos_theta = cos_theta.clamp(-1.0, 1.0);
    Ok(rad_to_deg_ref(cos_theta.acos()))
}

/// Angle (degrees) between the satellite nadir (toward Earth center) and the
/// direction from the satellite to `body`.
#[inline]
fn nadir_body_angle(
    sat_pos: [f64; 3],
    body_pos: [f64; 3],
    body_field: &'static str,
) -> Result<f64, AngleError> {
    validate_nonzero_vec3(sat_pos, "sat_pos")?;
    validate_nonzero_vec3(body_pos, body_field)?;
    let nadir = vec3::neg3(sat_pos);
    let body_from_sat = vec3::sub3(body_pos, sat_pos);
    angle_between(nadir, "sat_pos", body_from_sat, body_field)
}

/// Angle (degrees) between satellite nadir and the Sun direction.
///
/// `sat_pos` is the satellite GCRS position (km); `sun_pos` is the Sun position
/// relative to Earth center (km).
pub fn sun_angle(sat_pos: [f64; 3], sun_pos: [f64; 3]) -> Result<f64, AngleError> {
    nadir_body_angle(sat_pos, sun_pos, "sun_pos")
}

/// Angle (degrees) between satellite nadir and the Moon direction.
pub fn moon_angle(sat_pos: [f64; 3], moon_pos: [f64; 3]) -> Result<f64, AngleError> {
    nadir_body_angle(sat_pos, moon_pos, "moon_pos")
}

/// Sun elevation (degrees) above the satellite's local horizontal plane.
///
/// Positive means the Sun is on the sunlit (zenith) side. The zenith direction
/// is the satellite position itself, since Earth is at the GCRS origin.
pub fn sun_elevation(sat_pos: [f64; 3], sun_pos: [f64; 3]) -> Result<f64, AngleError> {
    validate_nonzero_vec3(sat_pos, "sat_pos")?;
    validate_nonzero_vec3(sun_pos, "sun_pos")?;
    let sun_from_sat = vec3::sub3(sun_pos, sat_pos);
    let zenith_angle = angle_between(sat_pos, "sat_pos", sun_from_sat, "sun_pos")?;
    Ok(RIGHT_ANGLE_DEG - zenith_angle)
}

/// Sun-satellite-observer phase angle (degrees): the angle at the satellite
/// between the Sun and the observer.
pub fn phase_angle(
    sat_pos: [f64; 3],
    sun_pos: [f64; 3],
    observer_pos: [f64; 3],
) -> Result<f64, AngleError> {
    validate_nonzero_vec3(sat_pos, "sat_pos")?;
    validate_nonzero_vec3(sun_pos, "sun_pos")?;
    validate_nonzero_vec3(observer_pos, "observer_pos")?;
    let sun_from_sat = vec3::sub3(sun_pos, sat_pos);
    let observer_from_sat = vec3::sub3(observer_pos, sat_pos);
    angle_between(sun_from_sat, "sun_pos", observer_from_sat, "observer_pos")
}

/// Angular radius (degrees) of the Earth as seen from the satellite:
/// `asin(R_earth / |sat_pos|)`, clamped to the `asin` domain.
pub fn earth_angular_radius(sat_pos: [f64; 3]) -> Result<f64, AngleError> {
    validate_nonzero_vec3(sat_pos, "sat_pos")?;
    let distance = vec3::norm3(sat_pos);
    let ratio = (WGS84_A_KM / distance).min(1.0);
    Ok(rad_to_deg_ref(ratio.asin()))
}

fn validate_nonzero_vec3(v: [f64; 3], field: &'static str) -> Result<(), AngleError> {
    if !v.iter().all(|value| value.is_finite()) {
        return Err(invalid_angle_input(field, "not finite"));
    }
    let norm = vec3::norm3(v);
    if norm == 0.0 {
        return Err(invalid_angle_input(field, "zero vector"));
    }
    if !norm.is_finite() {
        return Err(invalid_angle_input(field, "out of range"));
    }
    Ok(())
}

fn invalid_angle_input(field: &'static str, reason: &'static str) -> AngleError {
    AngleError::InvalidInput { field, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Frozen bits captured from the reference (Elixir) `Sidereon.Angles`
    // implementation. Cross-language 0-ULP equality.

    #[test]
    fn sun_angle_matches_reference_bits() {
        let sat = [6778.0, 0.0, 0.0];
        let skew_sat = [6778.0, 123.0, -456.0];
        assert_eq!(
            sun_angle(sat, [149_597_870.0, 0.0, 0.0])
                .expect("valid angle geometry")
                .to_bits(),
            0x4066_8000_0000_0000
        );
        assert_eq!(
            sun_angle(sat, [-149_597_870.0, 0.0, 0.0])
                .expect("valid angle geometry")
                .to_bits(),
            0x0000_0000_0000_0000
        );
        assert_eq!(
            sun_angle(skew_sat, [149_597_870.0, 1_000_000.0, -500_000.0])
                .expect("valid angle geometry")
                .to_bits(),
            0x4066_091c_484a_7158
        );
    }

    #[test]
    fn moon_angle_matches_reference_bits() {
        let sat = [6778.0, 0.0, 0.0];
        let skew_sat = [6778.0, 123.0, -456.0];
        assert_eq!(
            moon_angle(sat, [200_000.0, 300_000.0, 50_000.0])
                .expect("valid angle geometry")
                .to_bits(),
            0x405e_9b67_b2be_cf9b
        );
        assert_eq!(
            moon_angle(skew_sat, [-384_400.0, 12_345.0, 6_789.0])
                .expect("valid angle geometry")
                .to_bits(),
            0x400f_c228_50bd_874f
        );
    }

    #[test]
    fn sun_elevation_matches_reference_bits() {
        let sat = [6778.0, 0.0, 0.0];
        let skew_sat = [6778.0, 123.0, -456.0];
        assert_eq!(
            sun_elevation(sat, [149_597_870.0, 0.0, 0.0])
                .expect("valid angle geometry")
                .to_bits(),
            0x4056_8000_0000_0000
        );
        assert_eq!(
            sun_elevation(sat, [0.0, 149_597_870.0, 0.0])
                .expect("valid angle geometry")
                .to_bits(),
            0xbf65_4421_f2e3_8000
        );
        assert_eq!(
            sun_elevation(skew_sat, [149_597_870.0, 1_000_000.0, -500_000.0])
                .expect("valid angle geometry")
                .to_bits(),
            0x4055_9238_9094_e2b1
        );
    }

    #[test]
    fn phase_angle_matches_reference_bits() {
        let sat = [6778.0, 0.0, 0.0];
        let skew_sat = [6778.0, 123.0, -456.0];
        assert_eq!(
            phase_angle(sat, [149_597_870.0, 1_000_000.0, 0.0], [0.0, 6378.0, 0.0],)
                .expect("valid angle geometry")
                .to_bits(),
            0x4061_0b78_cc20_1866
        );
        assert_eq!(
            phase_angle(
                skew_sat,
                [149_597_870.0, 1_000_000.0, -500_000.0],
                [-6378.0, 100.0, 50.0]
            )
            .expect("valid angle geometry")
            .to_bits(),
            0x4066_3f01_b89b_b002
        );
    }

    #[test]
    fn earth_angular_radius_matches_reference_bits() {
        assert_eq!(
            earth_angular_radius([6778.0, 0.0, 0.0])
                .expect("valid angle geometry")
                .to_bits(),
            0x4051_8e27_583c_2f41
        );
        assert_eq!(
            earth_angular_radius([42_164.0, 0.0, 0.0])
                .expect("valid angle geometry")
                .to_bits(),
            0x4021_66aa_1bd9_bda5
        );
        assert_eq!(
            earth_angular_radius([7000.0, 1234.0, -567.0])
                .expect("valid angle geometry")
                .to_bits(),
            0x404f_b89e_165a_1133
        );
    }

    #[test]
    fn angle_helpers_reject_invalid_vectors() {
        assert_invalid_angle_field(
            sun_angle([0.0, 0.0, 0.0], [149_597_870.0, 0.0, 0.0]).unwrap_err(),
            "sat_pos",
            "zero vector",
        );
        assert_invalid_angle_field(
            moon_angle([6778.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]).unwrap_err(),
            "moon_pos",
            "not finite",
        );
        assert_invalid_angle_field(
            sun_elevation([6778.0, 0.0, 0.0], [6778.0, 0.0, 0.0]).unwrap_err(),
            "sun_pos",
            "zero vector",
        );
        assert_invalid_angle_field(
            phase_angle(
                [6778.0, 0.0, 0.0],
                [149_597_870.0, 0.0, 0.0],
                [6778.0, 0.0, 0.0],
            )
            .unwrap_err(),
            "observer_pos",
            "zero vector",
        );
        assert_invalid_angle_field(
            earth_angular_radius([f64::INFINITY, 0.0, 0.0]).unwrap_err(),
            "sat_pos",
            "not finite",
        );
    }

    fn assert_invalid_angle_field(
        error: AngleError,
        expected: &'static str,
        expected_reason: &'static str,
    ) {
        let AngleError::InvalidInput { field, reason } = error;
        assert_eq!(field, expected);
        assert_eq!(reason, expected_reason);
    }
}
