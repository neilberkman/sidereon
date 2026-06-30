//! Observational-astronomy geometry primitives.
//!
//! Small, I/O-free geometry helpers used when reducing or planning ground- and
//! space-based observations. Each function takes already-resolved geometry
//! (vectors, angles, ephemeris-derived positions) so it stays a pure numerical
//! kernel; callers supply the ephemeris and frame transforms from the rest of
//! the engine (for example [`crate::astro::bodies::sun_moon::sun_moon_ecef`] for
//! the Earth-fixed Sun vector that feeds the sub-solar point and terminator).
//!
//! Conventions used throughout this module:
//!
//! - Latitudes are geocentric (the direction of the position vector), returned
//!   in degrees on `[-90, 90]`.
//! - Longitudes are returned in degrees on `(-180, 180]` from `atan2(y, x)`,
//!   i.e. positive toward the body-fixed `+y` axis (east of the prime meridian
//!   for a standard right-handed body-fixed frame).
//! - Angles passed in (hour angle, declination, phase angle, pole orientation)
//!   are in degrees unless the name says otherwise.

use crate::astro::constants::units::{DEG_TO_RAD, RAD_TO_DEG};
use crate::astro::math::vec3;
use crate::validate;

/// Error returned by observation-geometry helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ObservationError {
    /// A public input was non-finite or outside its valid domain.
    #[error("invalid observation input {field}: {reason}")]
    InvalidInput {
        /// Offending field name.
        field: &'static str,
        /// Human-readable reason.
        reason: &'static str,
    },
}

fn invalid(field: &'static str, reason: &'static str) -> ObservationError {
    ObservationError::InvalidInput { field, reason }
}

fn map_field(error: validate::FieldError) -> ObservationError {
    invalid(error.field(), error.reason())
}

/// A point on a body surface expressed as geocentric latitude and longitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfacePoint {
    /// Geocentric latitude, degrees on `[-90, 90]`.
    pub latitude_deg: f64,
    /// Longitude, degrees on `(-180, 180]`.
    pub longitude_deg: f64,
}

/// Geocentric latitude/longitude of the direction of an Earth-fixed vector.
///
/// `vec_ecef` is any non-zero vector in an Earth-fixed (ITRS/ECEF) frame; the
/// returned point is the sub-point where that direction pierces the sphere.
fn sub_point(vec_ecef: [f64; 3], field: &'static str) -> Result<SurfacePoint, ObservationError> {
    validate::finite_vec3(vec_ecef, field).map_err(map_field)?;
    let [x, y, z] = vec_ecef;
    let horizontal = (x * x + y * y).sqrt();
    if horizontal == 0.0 && z == 0.0 {
        return Err(invalid(field, "zero vector"));
    }
    let latitude_deg = z.atan2(horizontal) * RAD_TO_DEG;
    let longitude_deg = y.atan2(x) * RAD_TO_DEG;
    Ok(SurfacePoint {
        latitude_deg,
        longitude_deg,
    })
}

/// Sub-solar point: the geographic point where the Sun is at the zenith.
///
/// `sun_ecef` is the geocentric Sun position in an Earth-fixed (ITRS/ECEF)
/// frame, in any length unit (only its direction matters). The returned
/// latitude is the solar declination of date and the longitude is the meridian
/// where it is local solar noon. Pair with
/// [`crate::astro::bodies::sun_moon::sun_moon_ecef`] to obtain `sun_ecef` from a
/// time.
pub fn sub_solar_point(sun_ecef: [f64; 3]) -> Result<SurfacePoint, ObservationError> {
    sub_point(sun_ecef, "sun_ecef")
}

/// Latitude (degrees) of the day-night terminator at a given longitude.
///
/// The terminator is the great circle of points whose solar zenith angle is
/// exactly 90 degrees, i.e. the locus 90 degrees of angular distance from the
/// sub-solar point. With sub-solar latitude `delta` (solar declination) and
/// sub-solar longitude `lambda_s`, the great-circle condition
/// `sin(delta) sin(lat) + cos(delta) cos(lat) cos(lon - lambda_s) = 0` gives
///
/// ```text
/// lat(lon) = atan( -cos(lon - lambda_s) / tan(delta) )
/// ```
///
/// `sub_solar` is the sub-solar point (see [`sub_solar_point`]); `longitude_deg`
/// is the query longitude. The result is the terminator latitude on `[-90, 90]`.
///
/// At an equinox the sub-solar declination is ~0 and the terminator degenerates
/// to the meridian circle through the poles; this is reported as `+90` or `-90`
/// (the limit of the formula) for all longitudes except the two where the
/// terminator crosses the equator.
pub fn terminator_latitude_deg(
    sub_solar: SurfacePoint,
    longitude_deg: f64,
) -> Result<f64, ObservationError> {
    let delta = validate::finite(sub_solar.latitude_deg, "sub_solar.latitude_deg")
        .map_err(map_field)?
        .to_radians();
    let lambda_s = validate::finite(sub_solar.longitude_deg, "sub_solar.longitude_deg")
        .map_err(map_field)?
        .to_radians();
    let lon = validate::finite(longitude_deg, "longitude_deg")
        .map_err(map_field)?
        .to_radians();

    let tan_delta = delta.tan();
    let cos_dlon = (lon - lambda_s).cos();
    // Quadrature meridians (lon = lambda_s +/- 90) are where the terminator
    // crosses the equator: the numerator cos(lon - lambda_s) is zero there, so
    // the latitude is 0 regardless of declination. `cos(pi/2)` is not exactly
    // zero in floating point (~6e-17), so at an equinox (tan(delta) -> 0) the raw
    // ratio is tiny/zero, which atan would collapse to a spurious +/-90 instead
    // of the equator crossing. Handle the near-zero numerator explicitly.
    const QUADRATURE_EPS: f64 = 1e-9;
    if cos_dlon.abs() < QUADRATURE_EPS {
        return Ok(0.0);
    }
    // Away from quadrature, atan handles the `tan(delta) -> 0` (equinox) limit
    // cleanly: the ratio diverges and atan saturates to the +/-90 degree poleward
    // meridian, with the sign following that of -cos(lon - lambda_s), without a
    // divide-by-zero blowing up to NaN.
    let lat = (-cos_dlon / tan_delta).atan();
    Ok(lat * RAD_TO_DEG)
}

/// Parallactic angle (degrees) of a target at a station.
///
/// The parallactic angle is the angle at the target between the directions to
/// the celestial pole and to the observer's zenith, i.e. the position angle by
/// which an alt-azimuth field is rotated relative to the equatorial frame. The
/// standard formula is
///
/// ```text
/// q = atan2( sin H, tan(phi) cos(dec) - sin(dec) cos H )
/// ```
///
/// (Meeus, *Astronomical Algorithms*, 2nd ed., ch. 14), with `phi` the observer
/// geodetic latitude, `H` the local hour angle (positive west of the meridian),
/// and `dec` the target declination. The result is on `(-180, 180]`: it is `0`
/// on the meridian, positive west of it, negative east of it.
pub fn parallactic_angle_deg(
    observer_latitude_deg: f64,
    hour_angle_deg: f64,
    declination_deg: f64,
) -> Result<f64, ObservationError> {
    let phi = validate::finite(observer_latitude_deg, "observer_latitude_deg")
        .map_err(map_field)?
        .to_radians();
    let h = validate::finite(hour_angle_deg, "hour_angle_deg")
        .map_err(map_field)?
        .to_radians();
    let dec = validate::finite(declination_deg, "declination_deg")
        .map_err(map_field)?
        .to_radians();

    let numerator = h.sin();
    let denominator = phi.tan() * dec.cos() - dec.sin() * h.cos();
    Ok(numerator.atan2(denominator) * RAD_TO_DEG)
}

/// Apparent visual magnitude of a sunlit body from a diffuse-sphere phase law.
///
/// Model: the body is treated as a Lambertian (diffuse) sphere of fixed size
/// and albedo. Its apparent magnitude scales with the inverse square of range
/// (the `5 log10` distance term) and with the normalized diffuse-sphere phase
/// function
///
/// ```text
/// p(phi) = [ (pi - phi) cos(phi) + sin(phi) ] / pi ,   p(0) = 1
/// ```
///
/// where `phi` is the solar phase angle (Sun-body-observer), `phi = 0` being
/// fully illuminated (Karttunen et al., *Fundamental Astronomy*; this is the
/// classic Lambert sphere phase integral normalized to unity at opposition).
/// The apparent magnitude is then
///
/// ```text
/// m = M_std + 5 log10(range / range_ref) - 2.5 log10( p(phi) )
/// ```
///
/// where `standard_magnitude` (`M_std`) is the body's brightness at the
/// reference range `reference_range_km` and zero phase. With the common
/// satellite convention `reference_range_km = 1000.0`, `M_std` is the familiar
/// "standard (intrinsic) magnitude" tabulated for many satellites.
///
/// `range_km` and `reference_range_km` must be positive; `phase_angle_deg` is
/// clamped to `[0, 180]` (the physical range of a phase angle).
pub fn satellite_visual_magnitude(
    range_km: f64,
    phase_angle_deg: f64,
    standard_magnitude: f64,
    reference_range_km: f64,
) -> Result<f64, ObservationError> {
    let range_km = validate::finite_positive(range_km, "range_km").map_err(map_field)?;
    let reference_range_km =
        validate::finite_positive(reference_range_km, "reference_range_km").map_err(map_field)?;
    let standard_magnitude =
        validate::finite(standard_magnitude, "standard_magnitude").map_err(map_field)?;
    let phase_angle_deg =
        validate::finite(phase_angle_deg, "phase_angle_deg").map_err(map_field)?;

    let phi = phase_angle_deg.clamp(0.0, 180.0) * DEG_TO_RAD;
    let pi = std::f64::consts::PI;
    let phase = ((pi - phi) * phi.cos() + phi.sin()) / pi;

    let distance_term = 5.0 * (range_km / reference_range_km).log10();
    let phase_term = -2.5 * phase.log10();
    Ok(standard_magnitude + distance_term + phase_term)
}

/// Sub-observer point (planetary central meridian) on a rotating body.
///
/// Given the observer's position relative to the body center in an inertial
/// frame (ICRF/J2000 equatorial) and the body's IAU orientation, returns the
/// body-fixed latitude/longitude beneath the observer: the central meridian of
/// the visible disk.
///
/// The body-fixed frame follows the IAU WGCCRE convention: the north pole of
/// rotation points to right ascension `pole_ra_deg` (alpha0), declination
/// `pole_dec_deg` (delta0), and the prime meridian is at angle
/// `prime_meridian_deg` (W) measured along the body equator from its ascending
/// node on the ICRF equator. The rotation from inertial to body-fixed
/// coordinates is
///
/// ```text
/// M = Rz(W) . Rx(90 deg - delta0) . Rz(90 deg + alpha0)
/// ```
///
/// (Archinal et al., "Report of the IAU Working Group on Cartographic
/// Coordinates and Rotational Elements"). The returned longitude is the
/// planetocentric longitude `atan2(y, x)` in the body-fixed frame, on
/// `(-180, 180]`; latitude is planetocentric on `[-90, 90]`. Bodies whose
/// official maps use west or `[0, 360)` longitude need the caller to remap.
pub fn sub_observer_point(
    observer_from_body: [f64; 3],
    pole_ra_deg: f64,
    pole_dec_deg: f64,
    prime_meridian_deg: f64,
) -> Result<SurfacePoint, ObservationError> {
    validate::finite_vec3(observer_from_body, "observer_from_body").map_err(map_field)?;
    if vec3::norm3(observer_from_body) == 0.0 {
        return Err(invalid("observer_from_body", "zero vector"));
    }
    let alpha0 = validate::finite(pole_ra_deg, "pole_ra_deg")
        .map_err(map_field)?
        .to_radians();
    let delta0 = validate::finite(pole_dec_deg, "pole_dec_deg")
        .map_err(map_field)?
        .to_radians();
    let w = validate::finite(prime_meridian_deg, "prime_meridian_deg")
        .map_err(map_field)?
        .to_radians();

    let half_pi = std::f64::consts::FRAC_PI_2;
    let body_fixed = rot_z(
        rot_x(
            rot_z(observer_from_body, half_pi + alpha0),
            half_pi - delta0,
        ),
        w,
    );
    sub_point(body_fixed, "observer_from_body")
}

/// Passive rotation of a vector about the z-axis by `theta` (radians).
#[inline]
fn rot_z(v: [f64; 3], theta: f64) -> [f64; 3] {
    let (s, c) = theta.sin_cos();
    [c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]]
}

/// Passive rotation of a vector about the x-axis by `theta` (radians).
#[inline]
fn rot_x(v: [f64; 3], theta: f64) -> [f64; 3] {
    let (s, c) = theta.sin_cos();
    [v[0], c * v[1] + s * v[2], -s * v[1] + c * v[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    const AU_KM: f64 = 149_597_870.7;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn sub_solar_point_reads_declination_and_meridian() {
        // Sun in the Earth-fixed frame, 23.44 deg above the equatorial plane at
        // longitude 0: sub-solar point is at that declination and longitude.
        let delta = 23.44_f64.to_radians();
        let sun = [AU_KM * delta.cos(), 0.0, AU_KM * delta.sin()];
        let point = sub_solar_point(sun).expect("valid sun vector");
        assert!(close(point.latitude_deg, 23.44, 1e-9));
        assert!(close(point.longitude_deg, 0.0, 1e-9));

        // A longitude offset shows up directly.
        let sun_east = [0.0, AU_KM * delta.cos(), AU_KM * delta.sin()];
        let east = sub_solar_point(sun_east).expect("valid sun vector");
        assert!(close(east.longitude_deg, 90.0, 1e-9));
    }

    #[test]
    fn sub_solar_point_rejects_zero_vector() {
        let err = sub_solar_point([0.0, 0.0, 0.0]).unwrap_err();
        assert_eq!(
            err,
            ObservationError::InvalidInput {
                field: "sun_ecef",
                reason: "zero vector"
            }
        );
    }

    #[test]
    fn terminator_matches_polar_circles_at_solstice() {
        // Solstice sub-solar declination 23.44 deg at longitude 0. The
        // terminator on the noon meridian sits at the Antarctic Circle and on
        // the midnight meridian at the Arctic Circle (= 90 - 23.44).
        let sub_solar = SurfacePoint {
            latitude_deg: 23.44,
            longitude_deg: 0.0,
        };
        let noon = terminator_latitude_deg(sub_solar, 0.0).expect("valid");
        let midnight = terminator_latitude_deg(sub_solar, 180.0).expect("valid");
        assert!(close(noon, -66.56, 1e-9), "noon terminator {noon}");
        assert!(
            close(midnight, 66.56, 1e-9),
            "midnight terminator {midnight}"
        );

        // 90 deg from the sub-solar meridian the terminator crosses the equator.
        let quad = terminator_latitude_deg(sub_solar, 90.0).expect("valid");
        assert!(close(quad, 0.0, 1e-9), "quadrature terminator {quad}");
    }

    #[test]
    fn terminator_equinox_quadrature_crosses_equator() {
        // Equinox: sub-solar declination 0. The terminator is the meridian
        // circle through the poles, crossing the equator exactly at the two
        // quadrature meridians lambda_s +/- 90. There the latitude must be 0,
        // not the spurious -90 that cos(pi/2) being ~6e-17 (not exactly 0) over
        // tan(0) would otherwise produce.
        let sub_solar = SurfacePoint {
            latitude_deg: 0.0,
            longitude_deg: 0.0,
        };
        let east_quad = terminator_latitude_deg(sub_solar, 90.0).expect("valid");
        let west_quad = terminator_latitude_deg(sub_solar, -90.0).expect("valid");
        assert!(close(east_quad, 0.0, 1e-9), "east quadrature {east_quad}");
        assert!(close(west_quad, 0.0, 1e-9), "west quadrature {west_quad}");

        // Off the quadrature meridians the degenerate equinox terminator is the
        // poleward limit: the noon meridian heads to one pole, the midnight
        // meridian to the other.
        let noon = terminator_latitude_deg(sub_solar, 0.0).expect("valid");
        let midnight = terminator_latitude_deg(sub_solar, 180.0).expect("valid");
        assert!(close(noon.abs(), 90.0, 1e-9), "noon poleward limit {noon}");
        assert!(
            close(midnight.abs(), 90.0, 1e-9),
            "midnight poleward limit {midnight}"
        );
    }

    #[test]
    fn terminator_near_equinox_quadrature_crosses_equator() {
        // A small but non-zero declination (near, not exactly, an equinox) at a
        // shifted sub-solar meridian: the quadrature longitude still crosses the
        // equator at ~0, and the result stays continuous through it.
        let sub_solar = SurfacePoint {
            latitude_deg: 1.0e-4,
            longitude_deg: 30.0,
        };
        let quad = terminator_latitude_deg(sub_solar, 120.0).expect("valid");
        assert!(close(quad, 0.0, 1e-6), "near-equinox quadrature {quad}");
    }

    #[test]
    fn terminator_offset_subsolar_longitude() {
        let sub_solar = SurfacePoint {
            latitude_deg: 10.0,
            longitude_deg: 30.0,
        };
        let lat = terminator_latitude_deg(sub_solar, 100.0).expect("valid");
        assert!(close(lat, -62.726_830_443_196_36, 1e-9), "terminator {lat}");
    }

    #[test]
    fn parallactic_angle_zero_on_meridian() {
        let q = parallactic_angle_deg(45.0, 0.0, 10.0).expect("valid");
        assert!(close(q, 0.0, 1e-12), "meridian parallactic {q}");
    }

    #[test]
    fn parallactic_angle_known_values() {
        // Equatorial target (dec 0) from latitude 45 at hour angle 45:
        // q = atan2(sin45, tan45) = atan(sin45) = 35.2644 deg.
        let q = parallactic_angle_deg(45.0, 45.0, 0.0).expect("valid");
        assert!(close(q, 35.264_389_682_754_654, 1e-9), "{q}");

        let q2 = parallactic_angle_deg(40.0, 30.0, 20.0).expect("valid");
        assert!(close(q2, 45.444_731_720_286_68, 1e-9), "{q2}");

        // West of the meridian (positive H) gives positive q; east gives the
        // mirror-image negative angle.
        let west = parallactic_angle_deg(40.0, 30.0, 20.0).expect("valid");
        let east = parallactic_angle_deg(40.0, -30.0, 20.0).expect("valid");
        assert!(close(west, -east, 1e-9), "antisymmetry {west} {east}");
    }

    #[test]
    fn visual_magnitude_distance_and_phase_terms() {
        // At the reference range and full phase, m = standard magnitude.
        let base = satellite_visual_magnitude(1000.0, 0.0, 0.0, 1000.0).expect("valid");
        assert!(close(base, 0.0, 1e-12), "base {base}");

        // Doubling the range adds 5 log10(2) ~= 1.505 mag.
        let farther = satellite_visual_magnitude(2000.0, 0.0, 0.0, 1000.0).expect("valid");
        assert!(
            close(farther, 1.505_149_978_319_906, 1e-9),
            "farther {farther}"
        );

        // 90 deg phase dims a diffuse sphere by 2.5 log10(pi) ~= 1.2429 mag.
        let quarter = satellite_visual_magnitude(1000.0, 90.0, 0.0, 1000.0).expect("valid");
        assert!(
            close(quarter, 1.242_874_681_735_334_7, 1e-9),
            "quarter {quarter}"
        );
    }

    #[test]
    fn visual_magnitude_phase_is_monotonic_and_clamped() {
        // Fainter as the phase angle opens up.
        let full = satellite_visual_magnitude(1000.0, 0.0, -1.3, 1000.0).expect("valid");
        let half = satellite_visual_magnitude(1000.0, 90.0, -1.3, 1000.0).expect("valid");
        let thin = satellite_visual_magnitude(1000.0, 150.0, -1.3, 1000.0).expect("valid");
        assert!(full < half && half < thin, "{full} {half} {thin}");

        // Phase angle is clamped to the physical [0, 180] range.
        let clamped = satellite_visual_magnitude(1000.0, 200.0, -1.3, 1000.0).expect("valid");
        let at_180 = satellite_visual_magnitude(1000.0, 180.0, -1.3, 1000.0).expect("valid");
        assert!(close(clamped, at_180, 1e-12), "clamp {clamped} {at_180}");
    }

    #[test]
    fn visual_magnitude_rejects_nonpositive_range() {
        let err = satellite_visual_magnitude(0.0, 0.0, 0.0, 1000.0).unwrap_err();
        assert_eq!(
            err,
            ObservationError::InvalidInput {
                field: "range_km",
                reason: "not positive"
            }
        );
    }

    #[test]
    fn sub_observer_point_pole_along_z() {
        // Body north pole along ICRF +z (alpha0 = 0, delta0 = 90), prime
        // meridian W = 0. Then body-fixed +x points to ICRF +y.
        let on_pm = sub_observer_point([0.0, 1.0, 0.0], 0.0, 90.0, 0.0).expect("valid");
        assert!(
            close(on_pm.latitude_deg, 0.0, 1e-9),
            "lat {}",
            on_pm.latitude_deg
        );
        assert!(
            close(on_pm.longitude_deg, 0.0, 1e-9),
            "lon {}",
            on_pm.longitude_deg
        );

        // An observer over the rotation pole sees the pole as central.
        let polar = sub_observer_point([0.0, 0.0, 1.0], 0.0, 90.0, 0.0).expect("valid");
        assert!(
            close(polar.latitude_deg, 90.0, 1e-9),
            "polar lat {}",
            polar.latitude_deg
        );

        // ICRF +x is 90 deg of longitude before the prime meridian.
        let before = sub_observer_point([1.0, 0.0, 0.0], 0.0, 90.0, 0.0).expect("valid");
        assert!(
            close(before.longitude_deg, -90.0, 1e-9),
            "lon {}",
            before.longitude_deg
        );
    }

    #[test]
    fn sub_observer_point_prime_meridian_rotation() {
        // Rotating the prime meridian by +90 deg shifts the sub-observer
        // longitude by -90 deg.
        let point = sub_observer_point([0.0, 1.0, 0.0], 0.0, 90.0, 90.0).expect("valid");
        assert!(
            close(point.longitude_deg, -90.0, 1e-9),
            "lon {}",
            point.longitude_deg
        );
    }

    #[test]
    fn sub_observer_point_iau_mars_orientation() {
        // Mars IAU pole (alpha0 = 317.68, delta0 = 52.89) with W = 176 and an
        // observer toward ICRF +x. Reference value from the IAU rotation matrix.
        let point = sub_observer_point([1.0, 0.0, 0.0], 317.68, 52.89, 176.0).expect("valid");
        assert!(
            close(point.latitude_deg, 26.494_542_592_970_532, 1e-9),
            "lat {}",
            point.latitude_deg
        );
        assert!(
            close(point.longitude_deg, 142.788_019_764_132_46, 1e-9),
            "lon {}",
            point.longitude_deg
        );
    }

    #[test]
    fn sub_observer_point_rejects_zero_vector() {
        let err = sub_observer_point([0.0, 0.0, 0.0], 0.0, 90.0, 0.0).unwrap_err();
        assert_eq!(
            err,
            ObservationError::InvalidInput {
                field: "observer_from_body",
                reason: "zero vector"
            }
        );
    }
}
