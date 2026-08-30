//! Angular geometry helpers for sky directions and satellite body geometry.
//!
//! Computes general angular separation between arbitrary directions or
//! `(lon, lat)` / `(RA, Dec)` pairs, position angle measured North through East,
//! and satellite nadir/Sun, nadir/Moon, Sun-elevation, phase, solar beta, and
//! Earth angular radius angles. Public angle-producing helpers in this module
//! return degrees unless their names state otherwise.

use crate::astro::constants::earth::WGS84_A_KM;
use crate::astro::constants::units::{DEGREES_PER_CIRCLE, DEGREES_PER_SEMICIRCLE};
use crate::astro::elements::clamp_acos;
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

#[inline]
pub(crate) fn beta_angle_from_cos_rad(cos: f64) -> f64 {
    std::f64::consts::FRAC_PI_2 - libm::acos(cos.clamp(-1.0, 1.0))
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
    Ok(rad_to_deg_ref(clamp_acos(cos_theta)))
}

/// On-sky angle (degrees) between two direction vectors, via the stable
/// `atan2(|a x b|, a . b)` form.
///
/// Vectors need not be normalized, but each must be finite, non-zero, and have
/// a finite positive squared norm under `dot3`. Extreme magnitudes whose squared
/// norm overflows to infinity or underflows to zero are rejected with
/// `AngleError::InvalidInput`. Fields `"a"` / `"b"` identify the offending
/// argument.
pub fn angular_separation(a: [f64; 3], b: [f64; 3]) -> Result<f64, AngleError> {
    validate_nonzero_vec3(a, "a")?;
    validate_nonzero_vec3(b, "b")?;
    let u = vec3::unit3(a).ok_or_else(|| invalid_angle_input("a", "zero vector"))?;
    let v = vec3::unit3(b).ok_or_else(|| invalid_angle_input("b", "zero vector"))?;
    let sin_theta = vec3::norm3(vec3::cross3(u, v));
    let cos_theta = vec3::dot3(u, v);
    Ok(rad_to_deg_ref(libm::atan2(sin_theta, cos_theta)))
}

/// On-sky angle (degrees) between two `(lon, lat)` / `(RA, Dec)` pairs in
/// degrees. The second tuple component is the latitude or declination in
/// `[-90, 90]`.
pub fn angular_separation_coords(
    a_lon_lat_deg: (f64, f64),
    b_lon_lat_deg: (f64, f64),
) -> Result<f64, AngleError> {
    validate_lon_lat_deg(a_lon_lat_deg, "a")?;
    validate_lon_lat_deg(b_lon_lat_deg, "b")?;
    angular_separation(
        unit_from_lon_lat_deg(a_lon_lat_deg),
        unit_from_lon_lat_deg(b_lon_lat_deg),
    )
}

/// Position angle (degrees, `[0, 360)`) of `to` as seen from `from`, measured
/// from North (`+lat`) through East (`+lon`). Inputs are `(lon, lat)` in degrees.
pub fn position_angle(
    from_lon_lat_deg: (f64, f64),
    to_lon_lat_deg: (f64, f64),
) -> Result<f64, AngleError> {
    validate_lon_lat_deg(from_lon_lat_deg, "from")?;
    validate_lon_lat_deg(to_lon_lat_deg, "to")?;

    let lon1 = reduce_lon_deg(from_lon_lat_deg.0);
    let lat1 = from_lon_lat_deg.1.to_radians();
    let lon2 = reduce_lon_deg(to_lon_lat_deg.0);
    let lat2 = to_lon_lat_deg.1.to_radians();
    let dlon = (lon2 - lon1).to_radians();

    let numerator = libm::cos(lat2) * libm::sin(dlon);
    let denominator =
        libm::cos(lat1) * libm::sin(lat2) - libm::sin(lat1) * libm::cos(lat2) * libm::cos(dlon);
    let pa = rad_to_deg_ref(libm::atan2(numerator, denominator)).rem_euclid(DEGREES_PER_CIRCLE);
    Ok(if pa == DEGREES_PER_CIRCLE || pa == 0.0 {
        0.0
    } else {
        pa
    })
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

/// Solar beta angle (degrees, signed, in [-90, 90]): the elevation of the Sun
/// above the orbit plane, `beta = 90 - angle(orbit_normal, sun)`, computed in
/// the normative `pi/2 - acos(clamp(cos))` operation order.
///
/// `orbit_normal` is the orbit-plane normal, for example `r x v`; `sun` is the
/// Earth-to-Sun vector. Both must be in the same inertial frame. Only their
/// directions matter. Sign is positive when the Sun is on the `+orbit_normal`
/// side of the plane.
pub fn beta_angle(orbit_normal: [f64; 3], sun: [f64; 3]) -> Result<f64, AngleError> {
    validate_nonzero_vec3(orbit_normal, "orbit_normal")?;
    validate_nonzero_vec3(sun, "sun")?;
    let cos_theta = vec3::dot3(orbit_normal, sun) / (vec3::norm3(orbit_normal) * vec3::norm3(sun));
    Ok(rad_to_deg_ref(beta_angle_from_cos_rad(cos_theta)))
}

/// Solar beta angle (degrees) from an inertial Cartesian state.
///
/// Computes the orbit normal as `r x v` and delegates to [`beta_angle`]. `r`,
/// `v`, and `sun` must all be in the same inertial frame. Returns
/// `AngleError::InvalidInput { field: "orbit_normal", reason: "zero vector" }`
/// when `r` and `v` are parallel, since no orbit plane is defined.
pub fn beta_angle_from_state(r: [f64; 3], v: [f64; 3], sun: [f64; 3]) -> Result<f64, AngleError> {
    validate_finite_vec3(r, "r")?;
    validate_finite_vec3(v, "v")?;
    beta_angle(vec3::cross3(r, v), sun)
}

/// Angular radius (degrees) of the Earth as seen from the satellite:
/// `asin(R_earth / |sat_pos|)`, clamped to the `asin` domain.
pub fn earth_angular_radius(sat_pos: [f64; 3]) -> Result<f64, AngleError> {
    validate_nonzero_vec3(sat_pos, "sat_pos")?;
    let distance = vec3::norm3(sat_pos);
    let ratio = (WGS84_A_KM / distance).min(1.0);
    Ok(rad_to_deg_ref(libm::asin(ratio)))
}

fn unit_from_lon_lat_deg(lon_lat_deg: (f64, f64)) -> [f64; 3] {
    let lon = reduce_lon_deg(lon_lat_deg.0).to_radians();
    let lat = lon_lat_deg.1.to_radians();
    let cos_lat = libm::cos(lat);
    [
        cos_lat * libm::cos(lon),
        cos_lat * libm::sin(lon),
        libm::sin(lat),
    ]
}

fn validate_lon_lat_deg(lon_lat_deg: (f64, f64), field: &'static str) -> Result<(), AngleError> {
    if !lon_lat_deg.0.is_finite() || !lon_lat_deg.1.is_finite() {
        return Err(invalid_angle_input(field, "not finite"));
    }
    if !(-90.0..=90.0).contains(&lon_lat_deg.1) {
        return Err(invalid_angle_input(field, "latitude out of range"));
    }
    Ok(())
}

fn reduce_lon_deg(lon: f64) -> f64 {
    let reduced = lon.rem_euclid(DEGREES_PER_CIRCLE);
    if reduced == DEGREES_PER_CIRCLE || reduced == 0.0 {
        0.0
    } else {
        reduced
    }
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

fn validate_finite_vec3(v: [f64; 3], field: &'static str) -> Result<(), AngleError> {
    if !v.iter().all(|value| value.is_finite()) {
        return Err(invalid_angle_input(field, "not finite"));
    }
    Ok(())
}

fn invalid_angle_input(field: &'static str, reason: &'static str) -> AngleError {
    AngleError::InvalidInput { field, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ReferenceCase {
        name: &'static str,
        a_lon_lat_deg: (f64, f64),
        b_lon_lat_deg: (f64, f64),
        expected_sep_deg: f64,
        expected_pa_deg: f64,
    }

    #[test]
    fn angular_separation_and_position_angle_match_reference_cases() {
        // Frozen canonical table generated with astropy.coordinates.
        // angular_separation and position_angle from astropy 7.2.0 on
        // Python 3.14.6. Inputs are ICRS/J2000 degrees
        // (lon1, lat1, lon2, lat2); outputs are degrees. Skyfield 1.49
        // separation_from cross-checked the star separations within 1e-9 deg.
        let cases = [
            ReferenceCase {
                name: "sirius->procyon",
                a_lon_lat_deg: (101.287155333, -16.716115861),
                b_lon_lat_deg: (114.825493028, 5.224993306),
                expected_sep_deg: 25.7013646403623,
                expected_pa_deg: 32.51673660099302,
            },
            ReferenceCase {
                name: "sirius->canopus",
                a_lon_lat_deg: (101.287155333, -16.716115861),
                b_lon_lat_deg: (95.987958333, -52.695661111),
                expected_sep_deg: 36.22078689550111,
                expected_pa_deg: 185.43546880831877,
            },
            ReferenceCase {
                name: "polaris->vega",
                a_lon_lat_deg: (37.954561667, 89.264108889),
                b_lon_lat_deg: (279.234734792, 38.783688958),
                expected_sep_deg: 51.57282525600137,
                expected_pa_deg: 299.23384966888466,
            },
            ReferenceCase {
                name: "near-coincident",
                a_lon_lat_deg: (10.0, 20.0),
                b_lon_lat_deg: (10.0000001, 20.0000001),
                expected_sep_deg: 1.372232571517946e-07,
                expected_pa_deg: 43.219178406844925,
            },
            ReferenceCase {
                name: "wrap-crossing",
                a_lon_lat_deg: (359.9, -5.0),
                b_lon_lat_deg: (0.1, 5.0),
                expected_sep_deg: 10.001994721516857,
                expected_pa_deg: 1.147219154245781,
            },
        ];

        for case in cases {
            let coord_sep =
                angular_separation_coords(case.a_lon_lat_deg, case.b_lon_lat_deg).expect(case.name);
            assert_close_deg(coord_sep, case.expected_sep_deg, 1.0e-9);

            let vector_sep = angular_separation(
                unit_from_lon_lat_deg(case.a_lon_lat_deg),
                unit_from_lon_lat_deg(case.b_lon_lat_deg),
            )
            .expect(case.name);
            assert_close_deg(vector_sep, coord_sep, 1.0e-9);

            let pa = position_angle(case.a_lon_lat_deg, case.b_lon_lat_deg).expect(case.name);
            assert_pa_close(pa, case.expected_pa_deg, 1.0e-9);
        }
    }

    #[test]
    fn position_angle_uses_north_through_east_convention() {
        assert_pa_close(
            position_angle((0.0, 0.0), (0.0, 10.0)).unwrap(),
            0.0,
            1.0e-12,
        );
        assert_pa_close(
            position_angle((0.0, 0.0), (10.0, 0.0)).unwrap(),
            90.0,
            1.0e-12,
        );
        assert_pa_close(
            position_angle((0.0, 0.0), (0.0, -10.0)).unwrap(),
            180.0,
            1.0e-12,
        );
        assert_pa_close(
            position_angle((0.0, 0.0), (-10.0, 0.0)).unwrap(),
            270.0,
            1.0e-12,
        );

        let off_axis = position_angle((15.0, 20.0), (75.0, -10.0)).unwrap();
        assert_pa_close(off_axis, 111.24565371752205, 1.0e-9);
    }

    #[test]
    fn angular_separation_keeps_small_vector_angles_stable() {
        let theta = 1.0e-8_f64;
        let expected_deg = rad_to_deg_ref(theta);
        let axis_a = [1.0, 0.0, 0.0];
        let axis_b = [libm::cos(theta), libm::sin(theta), 0.0];

        let axis_atan2 = angular_separation(axis_a, axis_b).unwrap();
        let axis_acos = acos_separation_deg(axis_a, axis_b);
        assert!(
            relative_error(axis_atan2, expected_deg) <= 1.0e-9,
            "axis atan2 actual={axis_atan2}, expected={expected_deg}"
        );
        assert!(
            relative_error(axis_acos, expected_deg) >= 1.0e-3,
            "axis acos actual={axis_acos}, expected={expected_deg}"
        );

        let oblique_u = vec3::unit3([1.0, 2.0, 3.0]).expect("nonzero vector");
        let oblique_k =
            vec3::unit3(vec3::cross3(oblique_u, [0.0, 0.0, 1.0])).expect("nonzero axis");
        let oblique_v = rotate_about_axis(oblique_u, oblique_k, theta);

        let oblique_atan2 = angular_separation(oblique_u, oblique_v).unwrap();
        let oblique_acos = acos_separation_deg(oblique_u, oblique_v);
        assert!(
            relative_error(oblique_atan2, expected_deg) <= 1.0e-6,
            "oblique atan2 actual={oblique_atan2}, expected={expected_deg}"
        );
        assert!(
            relative_error(oblique_acos, expected_deg) >= 1.0e-3,
            "oblique acos actual={oblique_acos}, expected={expected_deg}"
        );
    }

    #[test]
    fn angular_separation_coords_has_absolute_small_angle_accuracy() {
        let sep = angular_separation_coords((0.0, 0.0), (1.0e-6, 0.0)).unwrap();
        assert_close_deg(sep, 1.0e-6, 1.0e-9);
    }

    #[test]
    fn angular_separation_handles_antipodal_and_near_antipodal_cases() {
        let exact = angular_separation([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]).unwrap();
        assert_close_deg(exact, 180.0, 1.0e-12);

        let eps_deg = 1.0e-7_f64;
        let eps = eps_deg.to_radians();
        let a = [1.0, 0.0, 0.0];
        let b = [-libm::cos(eps), libm::sin(eps), 0.0];
        let sep = angular_separation(a, b).unwrap();
        let complement = 180.0 - sep;
        assert!(
            relative_error(complement, eps_deg) <= 1.0e-6,
            "complement={complement}, expected={eps_deg}, sep={sep}"
        );

        let acos_sep = acos_separation_deg(a, b);
        let acos_complement = 180.0 - acos_sep;
        assert!(
            relative_error(acos_complement, eps_deg) >= 1.0e-3,
            "acos complement={acos_complement}, expected={eps_deg}"
        );

        assert_finite_pa(position_angle((0.0, 0.0), (180.0, 0.0)).unwrap());
        assert_finite_pa(position_angle((0.0, 90.0), (0.0, -90.0)).unwrap());
    }

    #[test]
    fn coincident_inputs_return_zero_separation_and_zero_position_angle() {
        assert_eq!(
            angular_separation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
                .unwrap()
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            angular_separation_coords((123.0, 45.0), (123.0, 45.0))
                .unwrap()
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            position_angle((123.0, 45.0), (123.0, 45.0))
                .unwrap()
                .to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn pole_edge_cases_are_finite_and_documented() {
        assert_close_deg(
            angular_separation_coords((0.0, 90.0), (0.0, -90.0)).unwrap(),
            180.0,
            1.0e-12,
        );
        let same_north_pole = angular_separation_coords((0.0, 90.0), (123.0, 90.0)).unwrap();
        assert!(
            same_north_pole <= 1.0e-9,
            "same-pole separation={same_north_pole}"
        );

        assert_pa_close(
            position_angle((0.0, 0.0), (123.0, 90.0)).unwrap(),
            0.0,
            1.0e-6,
        );
        assert_pa_close(
            position_angle((0.0, 0.0), (123.0, -90.0)).unwrap(),
            180.0,
            1.0e-6,
        );

        assert_finite_pa(position_angle((0.0, 89.999999999999), (90.0, 90.0)).unwrap());
        assert_finite_pa(position_angle((0.0, 90.0), (45.0, 10.0)).unwrap());
        assert_finite_pa(position_angle((0.0, 90.0), (90.0, 90.0)).unwrap());
    }

    #[test]
    fn general_angle_helpers_reject_invalid_inputs() {
        assert_invalid_angle_field(
            angular_separation([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap_err(),
            "a",
            "zero vector",
        );
        assert_invalid_angle_field(
            angular_separation([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]).unwrap_err(),
            "b",
            "zero vector",
        );
        assert_invalid_angle_field(
            angular_separation([f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap_err(),
            "a",
            "not finite",
        );
        assert_invalid_angle_field(
            angular_separation([1.0, 0.0, 0.0], [f64::INFINITY, 0.0, 0.0]).unwrap_err(),
            "b",
            "not finite",
        );
        assert_invalid_angle_field(
            angular_separation([1.0e300, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap_err(),
            "a",
            "out of range",
        );
        assert_invalid_angle_field(
            angular_separation([1.0e-300, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap_err(),
            "a",
            "zero vector",
        );

        assert_invalid_angle_field(
            angular_separation_coords((0.0, 91.0), (0.0, 0.0)).unwrap_err(),
            "a",
            "latitude out of range",
        );
        assert_invalid_angle_field(
            angular_separation_coords((0.0, 0.0), (0.0, -91.0)).unwrap_err(),
            "b",
            "latitude out of range",
        );
        assert_invalid_angle_field(
            angular_separation_coords((f64::NAN, 0.0), (0.0, 0.0)).unwrap_err(),
            "a",
            "not finite",
        );
        assert_invalid_angle_field(
            angular_separation_coords((0.0, 0.0), (0.0, f64::INFINITY)).unwrap_err(),
            "b",
            "not finite",
        );

        assert_invalid_angle_field(
            position_angle((0.0, 91.0), (0.0, 0.0)).unwrap_err(),
            "from",
            "latitude out of range",
        );
        assert_invalid_angle_field(
            position_angle((0.0, 0.0), (0.0, -91.0)).unwrap_err(),
            "to",
            "latitude out of range",
        );
        assert_invalid_angle_field(
            position_angle((f64::NAN, 0.0), (0.0, 0.0)).unwrap_err(),
            "from",
            "not finite",
        );
        assert_invalid_angle_field(
            position_angle((0.0, 0.0), (0.0, f64::INFINITY)).unwrap_err(),
            "to",
            "not finite",
        );
    }

    #[test]
    fn longitude_reduction_wraps_and_canonicalizes_zero() {
        assert_eq!(reduce_lon_deg(-0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            reduce_lon_deg(DEGREES_PER_CIRCLE).to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(reduce_lon_deg(720.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(reduce_lon_deg(-10.0).to_bits(), 350.0_f64.to_bits());
        assert_eq!(reduce_lon_deg(123.456).to_bits(), 123.456_f64.to_bits());

        let sep = angular_separation_coords((359.999999, 0.0), (0.000001, 0.0)).unwrap();
        assert_close_deg(sep, 2.0e-6, 1.0e-9);
    }

    // Frozen bits captured from the reference Elixir angles implementation.
    // Cross-language 0-ULP equality.

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
    fn beta_angle_reference_r1() {
        let normal = normal_from_elements(51.6_f64.to_radians(), 90.0_f64.to_radians());
        let sun = [1.0, 0.0, 0.0];
        let beta = beta_angle(normal, sun).expect("valid beta geometry");
        assert_close_deg(beta, 51.6, 1.0e-9);
    }

    #[test]
    fn beta_angle_reference_r2_r3() {
        let alpha = 90.0_f64.to_radians();
        let delta = 23.4392911_f64.to_radians();
        let sun = sun_from_ra_dec(alpha, delta);

        for (inclination_deg, raan_deg) in [(51.64_f64, 0.0_f64), (98.0_f64, 30.0_f64)] {
            let inclination = inclination_deg.to_radians();
            let raan = raan_deg.to_radians();
            let normal = normal_from_elements(inclination, raan);
            let expected = closed_form_beta_deg(inclination, raan, alpha, delta);
            let actual = beta_angle(normal, sun).expect("valid beta geometry");
            assert_close_deg(actual, expected, 1.0e-9);
        }
    }

    #[test]
    fn beta_angle_sun_in_plane_is_zero() {
        let normal = [0.25, -0.5, 0.75];
        let sun = perpendicular_via_least_parallel_axis(normal);
        let beta = beta_angle(normal, sun).expect("valid beta geometry");
        assert_close_deg(beta, 0.0, 1.0e-12);
    }

    #[test]
    fn beta_angle_sun_along_normal_is_ninety() {
        let orbit_normal = [0.0, 0.0, 1.0];
        let beta_positive = beta_angle(orbit_normal, [0.0, 0.0, 5.0]).expect("valid beta geometry");
        let beta_negative =
            beta_angle(orbit_normal, [0.0, 0.0, -5.0]).expect("valid beta geometry");
        assert_close_deg(beta_positive, 90.0, 1.0e-9);
        assert_close_deg(beta_negative, -90.0, 1.0e-9);
    }

    #[test]
    fn beta_angle_from_state_matches_beta_angle() {
        let r = [7000.0, -1200.0, 350.0];
        let v = [1.25, 7.35, -0.42];
        let sun = [149_597_870.0, 3_000_000.0, 1_000_000.0];
        let from_state = beta_angle_from_state(r, v, sun).expect("valid beta geometry");
        let from_normal =
            beta_angle(vec3::cross3(r, v), sun).expect("valid beta geometry from normal");
        assert_eq!(from_state.to_bits(), from_normal.to_bits());
    }

    #[test]
    fn beta_angle_from_state_rejects_parallel_rv() {
        assert_invalid_angle_field(
            beta_angle_from_state([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap_err(),
            "orbit_normal",
            "zero vector",
        );
    }

    #[test]
    fn beta_angle_rejects_invalid_vectors() {
        assert_invalid_angle_field(
            beta_angle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap_err(),
            "orbit_normal",
            "zero vector",
        );
        assert_invalid_angle_field(
            beta_angle([f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap_err(),
            "orbit_normal",
            "not finite",
        );
        assert_invalid_angle_field(
            beta_angle([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]).unwrap_err(),
            "sun",
            "zero vector",
        );
        assert_invalid_angle_field(
            beta_angle([0.0, 0.0, 1.0], [f64::NAN, 0.0, 0.0]).unwrap_err(),
            "sun",
            "not finite",
        );
    }

    #[test]
    fn sat_yaw_beta_parity() {
        let eps = f64::EPSILON;
        for cos in [-1.0 - eps, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0 + eps] {
            let old = std::f64::consts::PI / 2.0 - libm::acos(cos.clamp(-1.0, 1.0));
            assert_eq!(beta_angle_from_cos_rad(cos).to_bits(), old.to_bits());
        }
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

    fn assert_close_deg(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    fn assert_pa_close(actual: f64, expected: f64, tolerance: f64) {
        assert_finite_pa(actual);
        let diff = circular_diff_deg(actual, expected);
        assert!(
            diff <= tolerance,
            "actual={actual}, expected={expected}, diff={diff}, tolerance={tolerance}"
        );
    }

    fn assert_finite_pa(pa: f64) {
        assert!(pa.is_finite(), "pa={pa}");
        assert!((0.0..DEGREES_PER_CIRCLE).contains(&pa), "pa={pa}");
    }

    fn circular_diff_deg(actual: f64, expected: f64) -> f64 {
        let diff = (actual - expected).abs();
        diff.min(DEGREES_PER_CIRCLE - diff)
    }

    fn relative_error(actual: f64, expected: f64) -> f64 {
        ((actual - expected) / expected).abs()
    }

    fn acos_separation_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
        let u = vec3::unit3(a).expect("nonzero vector");
        let v = vec3::unit3(b).expect("nonzero vector");
        rad_to_deg_ref(libm::acos(vec3::dot3(u, v).clamp(-1.0, 1.0)))
    }

    fn rotate_about_axis(v: [f64; 3], axis: [f64; 3], theta: f64) -> [f64; 3] {
        let cos_theta = libm::cos(theta);
        let sin_theta = libm::sin(theta);
        let cross = vec3::cross3(axis, v);
        let dot = vec3::dot3(axis, v);
        [
            v[0] * cos_theta + cross[0] * sin_theta + axis[0] * dot * (1.0 - cos_theta),
            v[1] * cos_theta + cross[1] * sin_theta + axis[1] * dot * (1.0 - cos_theta),
            v[2] * cos_theta + cross[2] * sin_theta + axis[2] * dot * (1.0 - cos_theta),
        ]
    }

    fn normal_from_elements(inclination: f64, raan: f64) -> [f64; 3] {
        [
            libm::sin(inclination) * libm::sin(raan),
            -libm::sin(inclination) * libm::cos(raan),
            libm::cos(inclination),
        ]
    }

    fn sun_from_ra_dec(alpha: f64, delta: f64) -> [f64; 3] {
        [
            libm::cos(delta) * libm::cos(alpha),
            libm::cos(delta) * libm::sin(alpha),
            libm::sin(delta),
        ]
    }

    fn closed_form_beta_deg(inclination: f64, raan: f64, alpha: f64, delta: f64) -> f64 {
        let sin_beta = libm::cos(delta) * libm::sin(inclination) * libm::sin(raan - alpha)
            + libm::sin(delta) * libm::cos(inclination);
        rad_to_deg_ref(libm::asin(sin_beta))
    }

    fn perpendicular_via_least_parallel_axis(v: [f64; 3]) -> [f64; 3] {
        let axis = if v[0].abs() <= v[1].abs() && v[0].abs() <= v[2].abs() {
            [1.0, 0.0, 0.0]
        } else if v[1].abs() <= v[2].abs() {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        vec3::cross3(v, axis)
    }
}
