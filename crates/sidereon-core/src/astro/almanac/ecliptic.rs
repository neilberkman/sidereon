use crate::astro::almanac::{norm_checked, wrap360, AlmanacError};
use crate::astro::frames::nutation::{skyfield_iau2000a_radians, skyfield_mean_obliquity_radians};
use crate::astro::time::scales::TimeScales;

/// Geocentric apparent ecliptic longitude and latitude of a body, degrees, of date.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EclipticLonLat {
    pub longitude_deg: f64,
    pub latitude_deg: f64,
}

/// Convert a geocentric true-equator-and-equinox-of-date apparent position,
/// metres, to true ecliptic longitude and latitude of date.
pub fn geocentric_ecliptic(
    pos_true_of_date_m: [f64; 3],
    ts: &TimeScales,
) -> Result<EclipticLonLat, AlmanacError> {
    let distance_m = norm_checked(pos_true_of_date_m, "pos_true_of_date_m")?;
    let mean =
        skyfield_mean_obliquity_radians(ts.jd_tdb).map_err(|_| AlmanacError::Frame("obliquity"))?;
    let (_dpsi, deps) =
        skyfield_iau2000a_radians(ts.jd_tt).map_err(|_| AlmanacError::Frame("nutation"))?;
    let eps = mean + deps;
    if !eps.is_finite() {
        return Err(AlmanacError::Frame("obliquity"));
    }

    let cos_eps = libm::cos(eps);
    let sin_eps = libm::sin(eps);
    let x = pos_true_of_date_m[0] / distance_m;
    let y_eq = pos_true_of_date_m[1] / distance_m;
    let z_eq = pos_true_of_date_m[2] / distance_m;
    let y = cos_eps * y_eq + sin_eps * z_eq;
    let z = -sin_eps * y_eq + cos_eps * z_eq;
    Ok(EclipticLonLat {
        longitude_deg: wrap360(libm::atan2(y, x).to_degrees()),
        latitude_deg: libm::asin(z.clamp(-1.0, 1.0)).to_degrees(),
    })
}
