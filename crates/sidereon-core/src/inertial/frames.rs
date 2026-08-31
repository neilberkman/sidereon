//! WGS84 normal gravity and ECEF plumb-bob gravity vectors.

use crate::astro::constants::earth::{
    GM_EARTH_M3_S2, OMEGA_E_DOT_RAD_S, WGS84_A_M, WGS84_E2, WGS84_F,
};
use crate::astro::math::vec3::{norm3, scale3};
use crate::frame::{itrf_to_geodetic, ItrfPositionM};

use super::{invalid_input, validate_finite, validate_vec3, InertialError};

/// WGS84 normal gravity at the equator in m/s^2.
pub const WGS84_NORMAL_GRAVITY_EQUATOR_MPS2: f64 = 9.780_325_335_9;

/// WGS84 normal gravity at the pole in m/s^2.
pub const WGS84_NORMAL_GRAVITY_POLE_MPS2: f64 = 9.832_184_937_8;

/// Somigliana normal-gravity formula constant derived from WGS84 table values.
pub const WGS84_SOMIGLIANA_K: f64 = ((WGS84_A_M * (1.0 - WGS84_F))
    * WGS84_NORMAL_GRAVITY_POLE_MPS2)
    / (WGS84_A_M * WGS84_NORMAL_GRAVITY_EQUATOR_MPS2)
    - 1.0;

/// WGS84 Somigliana normal gravity magnitude at geodetic latitude and height.
///
/// The returned magnitude is positive downward along the geodetic normal. The
/// height continuation is the WGS84 Taylor expression for normal gravity above
/// the ellipsoid, using the same WGS84 rotation and `GM` constants as the rest
/// of the core.
pub fn normal_gravity_mps2(lat_rad: f64, height_m: f64) -> Result<f64, InertialError> {
    validate_finite(lat_rad, "lat_rad")?;
    validate_finite(height_m, "height_m")?;
    if !(-core::f64::consts::FRAC_PI_2..=core::f64::consts::FRAC_PI_2).contains(&lat_rad) {
        return Err(invalid_input("lat_rad", "must be in [-pi/2, pi/2]"));
    }

    let sin_lat = libm::sin(lat_rad);
    let sin2 = sin_lat * sin_lat;
    let surface = WGS84_NORMAL_GRAVITY_EQUATOR_MPS2 * (1.0 + WGS84_SOMIGLIANA_K * sin2)
        / (1.0 - WGS84_E2 * sin2).sqrt();
    let b_m = WGS84_A_M * (1.0 - WGS84_F);
    let m = OMEGA_E_DOT_RAD_S * OMEGA_E_DOT_RAD_S * WGS84_A_M * WGS84_A_M * b_m / GM_EARTH_M3_S2;
    let height_scale = 1.0
        - (2.0 / WGS84_A_M) * (1.0 + WGS84_F + m - 2.0 * WGS84_F * sin2) * height_m
        + 3.0 * height_m * height_m / (WGS84_A_M * WGS84_A_M);
    let gravity = surface * height_scale;
    if gravity.is_finite() && gravity > 0.0 {
        Ok(gravity)
    } else {
        Err(invalid_input(
            "height_m",
            "outside normal-gravity Taylor domain",
        ))
    }
}

/// WGS84 normal gravity vector in ECEF coordinates, in m/s^2.
///
/// The vector includes the centrifugal contribution through Somigliana normal
/// gravity and points downward along the geodetic normal at the supplied ECEF
/// position.
pub fn gravity_ecef_mps2(position_ecef_m: [f64; 3]) -> Result<[f64; 3], InertialError> {
    validate_vec3(position_ecef_m, "position_ecef_m")?;
    if norm3(position_ecef_m) <= 0.0 {
        return Err(invalid_input("position_ecef_m", "must be nonzero"));
    }
    let position = ItrfPositionM::new(position_ecef_m[0], position_ecef_m[1], position_ecef_m[2])
        .map_err(|_| invalid_input("position_ecef_m", "must be finite"))?;
    let geodetic = itrf_to_geodetic(position)
        .map_err(|_| invalid_input("position_ecef_m", "must convert to WGS84 geodetic"))?;
    let gravity = normal_gravity_mps2(geodetic.lat_rad, geodetic.height_m)?;
    let normal = geodetic_surface_normal_ecef(geodetic.lat_rad, geodetic.lon_rad);
    Ok(scale3(normal, -gravity))
}

fn geodetic_surface_normal_ecef(lat_rad: f64, lon_rad: f64) -> [f64; 3] {
    let (sin_lat, cos_lat) = libm::sincos(lat_rad);
    let (sin_lon, cos_lon) = libm::sincos(lon_rad);
    [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
}

#[cfg(test)]
mod tests {
    //! Provenance: WGS84 normal-gravity validation uses WGS84 Technical Report
    //! TR8350.2, Chapter 4.2, Somigliana surface gravity, and WGS84 table
    //! values gamma_e = 9.7803253359 m/s^2 and gamma_p = 9.8321849378 m/s^2.

    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.17e}"
        );
    }

    #[test]
    fn somigliana_hits_wgs84_equator_and_pole_constants() {
        let equator = normal_gravity_mps2(0.0, 0.0).expect("equator");
        assert_eq!(
            equator.to_bits(),
            WGS84_NORMAL_GRAVITY_EQUATOR_MPS2.to_bits()
        );

        let pole = normal_gravity_mps2(core::f64::consts::FRAC_PI_2, 0.0).expect("pole gravity");
        assert_close(pole, WGS84_NORMAL_GRAVITY_POLE_MPS2, 2.0e-15);
    }

    #[test]
    fn gravity_ecef_points_down_at_equator_and_pole() {
        let equator = gravity_ecef_mps2([WGS84_A_M, 0.0, 0.0]).expect("equator vector");
        assert_eq!(
            equator[0].to_bits(),
            (-WGS84_NORMAL_GRAVITY_EQUATOR_MPS2).to_bits()
        );
        assert_eq!(equator[1].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(equator[2].to_bits(), (-0.0_f64).to_bits());

        let b_m = WGS84_A_M * (1.0 - WGS84_F);
        let pole = gravity_ecef_mps2([0.0, 0.0, b_m]).expect("pole vector");
        assert_close(pole[0], 0.0, 1.0e-15);
        assert_close(pole[1], 0.0, 1.0e-15);
        assert_close(pole[2], -WGS84_NORMAL_GRAVITY_POLE_MPS2, 2.0e-15);
    }
}
