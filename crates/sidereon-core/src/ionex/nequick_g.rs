//! Galileo NeQuick-G three-dimensional ionospheric correction.
//!
//! This is the full NeQuick-G slant total-electron-content model: the
//! three-dimensional NeQuick 2 electron-density profiler driven by the Galileo
//! broadcast effective-ionisation coefficients (`ai0`, `ai1`, `ai2`), integrated
//! along the receiver-to-satellite ray with the adaptive Gauss G7-Kronrod K15
//! quadrature in ray-perigee coordinates.
//!
//! Reference: *European GNSS (Galileo) Open Service - Ionospheric Correction
//! Algorithm for Galileo Single Frequency Users*, issue 1.2, September 2016
//! (the algorithm annexed to the Galileo OS SIS ICD). The MODIP grid and the
//! ITU-R/CCIR foF2 and M(3000)F2 coefficient maps are reference data tables (see
//! [`super::nequick_g_data`]).
//!
//! The model returns the slant TEC in TECU and, via [`nequick_g_delay_m`], the
//! single-frequency group delay in metres with the dispersive `40.3e16 / f^2`
//! relation. It is the reference-grade companion to the compact broadcast-driven
//! [`super::galileo_nequick_g_native`] helper.

use crate::error::{Error, Result};

use super::nequick_g_data::{
    CCIR_F2, CCIR_FM3, F2_COEFF_MAX_DEGREE, FM3_COEFF_MAX_DEGREE, MODIP_GRID, MODIP_LAT_POINTS,
    MODIP_LONG_POINTS,
};
use super::GalileoNequickCoeffs;

const PI: f64 = core::f64::consts::PI;
const DEG_TO_RAD: f64 = PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / PI;

/// Mean Earth radius used by the NeQuick-G geometry, km (reference 2.5.2).
const EARTH_RADIUS_KM: f64 = 6371.2;

/// Receiver/satellite ray geometry and epoch for a NeQuick-G evaluation.
///
/// Geodetic longitudes and latitudes are in degrees; heights are in metres above
/// the reference sphere. `month` is `1..=12` and `utc_hours` is the UTC time of
/// day in `[0, 24]`. These are exactly the inputs the NeQuick-G reference
/// algorithm consumes for a single ray.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NequickGRayEval {
    /// Month of the year, `1..=12`.
    pub month: u8,
    /// UTC time of day in hours, `[0, 24]`.
    pub utc_hours: f64,
    /// Receiver geodetic longitude, degrees.
    pub station_lon_deg: f64,
    /// Receiver geodetic latitude, degrees.
    pub station_lat_deg: f64,
    /// Receiver height above the reference sphere, metres.
    pub station_height_m: f64,
    /// Satellite geodetic longitude, degrees.
    pub satellite_lon_deg: f64,
    /// Satellite geodetic latitude, degrees.
    pub satellite_lat_deg: f64,
    /// Satellite height above the reference sphere, metres.
    pub satellite_height_m: f64,
}

/// Slant total electron content along the ray, in TECU (`1e16` electrons/m^2).
///
/// This evaluates the full NeQuick-G model: it forms the effective ionisation
/// level `Az` from the broadcast coefficients at the receiver MODIP, the
/// effective ionisation sunspot driver, the F2 Fourier coefficients for the
/// epoch, and integrates the NeQuick 2 electron density along the ray.
///
/// Returns [`Error::InvalidInput`] for an out-of-range month, UTC, or latitude,
/// or for a geometrically invalid ray (a sub-surface line of sight).
pub fn nequick_g_stec_tecu(coeffs: &GalileoNequickCoeffs, ray: &NequickGRayEval) -> Result<f64> {
    validate(coeffs, ray)?;

    let station = Position::new(
        ray.station_lon_deg,
        ray.station_lat_deg,
        ray.station_height_m,
    );
    let satellite = Position::new(
        ray.satellite_lon_deg,
        ray.satellite_lat_deg,
        ray.satellite_height_m,
    );

    let station_modip = modip_degree(station.lon.deg, station.lat.deg);
    let az_sfu = effective_ionisation_level_sfu(coeffs, station_modip);
    let azr = effective_sunspot_number(az_sfu);

    let model = NequickModel::new(ray.month, ray.utc_hours, az_sfu, azr);
    let geometry = RayGeometry::new(&station, &satellite)?;

    let tec = integrate_tec(&model, &geometry, &station, &satellite);
    let stec_tecu = tec / 1.0e13;
    if !stec_tecu.is_finite() {
        return Err(invalid("nequick_g_stec_tecu", "not finite"));
    }
    Ok(stec_tecu)
}

/// NeQuick-G slant ionospheric group delay (positive metres) on `frequency_hz`.
///
/// The slant TEC from [`nequick_g_stec_tecu`] is mapped to a group delay with the
/// dispersive `40.3e16 / f^2` relation, so the result is the positive metres that
/// increase the measured pseudorange on the given carrier.
pub fn nequick_g_delay_m(
    coeffs: &GalileoNequickCoeffs,
    ray: &NequickGRayEval,
    frequency_hz: f64,
) -> Result<f64> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(invalid("frequency_hz", "not positive"));
    }
    let stec_tecu = nequick_g_stec_tecu(coeffs, ray)?;
    let delay = stec_tecu * (40.3e16 / (frequency_hz * frequency_hz));
    if !delay.is_finite() {
        return Err(invalid("nequick_g_delay_m", "not finite"));
    }
    Ok(delay)
}

fn validate(coeffs: &GalileoNequickCoeffs, ray: &NequickGRayEval) -> Result<()> {
    for (value, field) in [
        (coeffs.ai0, "ai0"),
        (coeffs.ai1, "ai1"),
        (coeffs.ai2, "ai2"),
        (ray.utc_hours, "utc_hours"),
        (ray.station_lon_deg, "station_lon_deg"),
        (ray.station_lat_deg, "station_lat_deg"),
        (ray.station_height_m, "station_height_m"),
        (ray.satellite_lon_deg, "satellite_lon_deg"),
        (ray.satellite_lat_deg, "satellite_lat_deg"),
        (ray.satellite_height_m, "satellite_height_m"),
    ] {
        if !value.is_finite() {
            return Err(invalid(field, "not finite"));
        }
    }
    if !(1..=12).contains(&ray.month) {
        return Err(invalid("month", "out of range"));
    }
    if !(0.0..=24.0).contains(&ray.utc_hours) {
        return Err(invalid("utc_hours", "out of range"));
    }
    if !(-90.0..=90.0).contains(&ray.station_lat_deg) {
        return Err(invalid("station_lat_deg", "out of range"));
    }
    if !(-90.0..=90.0).contains(&ray.satellite_lat_deg) {
        return Err(invalid("satellite_lat_deg", "out of range"));
    }
    Ok(())
}

fn invalid(field: &'static str, reason: &'static str) -> Error {
    Error::InvalidInput(format!("{field} {reason}"))
}

// ---------------------------------------------------------------------------
// math helpers (NeQuick-G clamped exponential and join)
// ---------------------------------------------------------------------------

fn nq_exp(power: f64) -> f64 {
    if power > 80.0 {
        5.5406e34
    } else if power < -80.0 {
        1.8049e-35
    } else {
        power.exp()
    }
}

/// Smooth transition between two functions (reference "NeqJoin").
fn nq_join(func1: f64, func2: f64, alpha: f64, x: f64) -> f64 {
    let temp = nq_exp(alpha * x);
    ((func1 * temp) + func2) / (temp + 1.0)
}

fn square(v: f64) -> f64 {
    v * v
}

fn cos_from_sin(s: f64) -> f64 {
    (1.0 - s * s).sqrt()
}

fn sin_from_cos(c: f64) -> f64 {
    (1.0 - c * c).sqrt()
}

// ---------------------------------------------------------------------------
// angles and positions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Angle {
    deg: f64,
    rad: f64,
    sin: f64,
    cos: f64,
}

impl Angle {
    fn from_deg(deg: f64) -> Self {
        let rad = deg * DEG_TO_RAD;
        Self {
            deg,
            rad,
            sin: rad.sin(),
            cos: rad.cos(),
        }
    }

    fn from_rad(rad: f64) -> Self {
        Self {
            deg: rad * RAD_TO_DEG,
            rad,
            sin: rad.sin(),
            cos: rad.cos(),
        }
    }

    fn from_sin(sin: f64) -> Self {
        let cos = cos_from_sin(sin);
        let rad = sin.atan2(cos);
        Self {
            deg: rad * RAD_TO_DEG,
            rad,
            sin,
            cos,
        }
    }

    fn from_cos(cos: f64) -> Self {
        let sin = sin_from_cos(cos);
        let rad = sin.atan2(cos);
        Self {
            deg: rad * RAD_TO_DEG,
            rad,
            sin,
            cos,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Position {
    lon: Angle,
    lat: Angle,
    height_km: f64,
    radius_km: f64,
}

impl Position {
    fn new(lon_deg: f64, lat_deg: f64, height_m: f64) -> Self {
        let lon_norm = (lon_deg + 360.0).rem_euclid(360.0);
        let height_km = height_m * 1.0e-3;
        Self {
            lon: Angle::from_deg(lon_norm),
            lat: Angle::from_deg(lat_deg),
            height_km,
            radius_km: EARTH_RADIUS_KM + height_km,
        }
    }
}

// ---------------------------------------------------------------------------
// MODIP grid interpolation (reference F.2.4)
// ---------------------------------------------------------------------------

/// Third-order (4-point) interpolation, reference 2.5.7.1.
fn interpolate_third_order(z: &[f64; 4], offset: f64) -> f64 {
    if offset.abs() < 5.0e-11 {
        return z[1];
    }
    let delta = (2.0 * offset) - 1.0;
    let g1 = z[2] + z[1];
    let g2 = z[2] - z[1];
    let g3 = z[3] + z[0];
    let g4 = (z[3] - z[0]) / 3.0;
    let a0 = (9.0 * g1) - g3;
    let a1 = (9.0 * g2) - g4;
    let a2 = g3 - g1;
    let a3 = g4 - g2;
    (((a3 * delta + a2) * delta + a1) * delta + a0) / 16.0
}

fn modip_degree(lon_deg: f64, lat_deg: f64) -> f64 {
    if lat_deg <= -90.0 {
        return -90.0;
    }
    if lat_deg >= 90.0 {
        return 90.0;
    }

    // longitude grid <index, offset>, 10 deg step, 36 unique columns
    let lon_index_with_offset = (lon_deg + 180.0) / 10.0;
    let lon_floor = lon_index_with_offset.floor();
    let lon_offset = lon_index_with_offset - lon_floor;
    let mut lon_index = lon_floor as i32;
    if lon_index < 0 {
        lon_index += 36;
    } else if lon_index >= 36 {
        lon_index -= 36;
    }
    let lon_index = lon_index as usize;

    // latitude grid <index, offset>, 5 deg step, with the 1e-6 offset correction
    let lat_index_with_offset = (lat_deg + 90.0) / 5.0;
    let lat_index = (lat_index_with_offset - 1.0e-6).floor() as usize;
    let lat_offset = lat_index_with_offset - (lat_index as f64);

    let mut lon_points = [0.0_f64; 4];
    for (i, point) in lon_points.iter_mut().enumerate() {
        let col = lon_index + i;
        debug_assert!(col < MODIP_LONG_POINTS);
        let mut lat_points = [0.0_f64; 4];
        for (k, lp) in lat_points.iter_mut().enumerate() {
            let row = lat_index + k;
            debug_assert!(row < MODIP_LAT_POINTS);
            *lp = MODIP_GRID[row][col];
        }
        *point = interpolate_third_order(&lat_points, lat_offset);
    }
    interpolate_third_order(&lon_points, lon_offset)
}

// ---------------------------------------------------------------------------
// solar activity (reference 2.5.4, 3.x)
// ---------------------------------------------------------------------------

fn effective_ionisation_level_sfu(coeffs: &GalileoNequickCoeffs, modip: f64) -> f64 {
    let zero = |v: f64| v.abs() < 1.0e-7;
    if zero(coeffs.ai0) && zero(coeffs.ai1) && zero(coeffs.ai2) {
        return 63.7;
    }
    let az = coeffs.ai0 + (coeffs.ai1 * modip) + (coeffs.ai2 * modip * modip);
    az.clamp(0.0, 400.0)
}

/// ITU-R P.371-8 effective sunspot number from solar flux.
fn effective_sunspot_number(az_sfu: f64) -> f64 {
    (167273.0 + (az_sfu - 63.7) * 1123.6).sqrt() - 408.99
}

// ---------------------------------------------------------------------------
// epoch-fixed model state (solar declination + F2 Fourier coefficients)
// ---------------------------------------------------------------------------

struct NequickModel {
    month: u8,
    az_sfu: f64,
    azr: f64,
    solar_decl_sin: f64,
    solar_decl_cos: f64,
    utc_hours: f64,
    cf2: [f64; F2_COEFF_MAX_DEGREE],
    cm3: [f64; FM3_COEFF_MAX_DEGREE],
}

impl NequickModel {
    fn new(month: u8, utc_hours: f64, az_sfu: f64, azr: f64) -> Self {
        let (solar_decl_sin, solar_decl_cos) = solar_declination(month, utc_hours);
        let (cf2, cm3) = f2_fourier_coefficients(month, utc_hours, azr);
        Self {
            month,
            az_sfu,
            azr,
            solar_decl_sin,
            solar_decl_cos,
            utc_hours,
            cf2,
            cm3,
        }
    }
}

fn day_of_year(month: u8, utc_hours: f64) -> f64 {
    let mid = 30.5 * (month as f64) - 15.0;
    mid + (18.0 - utc_hours) / 24.0
}

fn solar_declination(month: u8, utc_hours: f64) -> (f64, f64) {
    let doy = day_of_year(month, utc_hours);
    let mean_anomaly = (0.9856 * doy - 3.289) * DEG_TO_RAD;
    let ecliptic_longitude = mean_anomaly
        + (282.634 * DEG_TO_RAD)
        + (1.916 * DEG_TO_RAD) * mean_anomaly.sin()
        + (0.02 * DEG_TO_RAD) * (2.0 * mean_anomaly).sin();
    let sin = 0.39782 * ecliptic_longitude.sin();
    (sin, cos_from_sin(sin))
}

fn solar_longitude_rad(utc_hours: f64) -> f64 {
    (15.0 * utc_hours - 180.0) * DEG_TO_RAD
}

/// Interpolate the CCIR maps by `Azr/100` and form the time-Fourier series.
fn f2_fourier_coefficients(
    month: u8,
    utc_hours: f64,
    azr: f64,
) -> ([f64; F2_COEFF_MAX_DEGREE], [f64; FM3_COEFF_MAX_DEGREE]) {
    let month0 = (month - 1) as usize;
    let f2 = &CCIR_F2[month0];
    let fm3 = &CCIR_FM3[month0];
    let ssn = azr / 100.0;

    // sin/cos harmonics of the solar longitude (max 6 needed for foF2)
    const MAX_HARMONICS: usize = 6;
    let mut sin_h = [0.0_f64; MAX_HARMONICS];
    let mut cos_h = [0.0_f64; MAX_HARMONICS];
    let solar_long = solar_longitude_rad(utc_hours);
    sin_h[0] = solar_long.sin();
    cos_h[0] = solar_long.cos();
    for i in 1..MAX_HARMONICS {
        sin_h[i] = sin_h[i - 1] * cos_h[0] + cos_h[i - 1] * sin_h[0];
        cos_h[i] = cos_h[i - 1] * cos_h[0] - sin_h[i - 1] * sin_h[0];
    }

    let mut cf2 = [0.0_f64; F2_COEFF_MAX_DEGREE];
    // foF2: 6 harmonics over 13 orders (1 + 2*6)
    for (i, c) in cf2.iter_mut().enumerate() {
        let af2_0 = f2[0][i][0] * (1.0 - ssn) + f2[1][i][0] * ssn;
        let mut value = af2_0;
        for (j, (&s, &cc)) in sin_h.iter().zip(cos_h.iter()).enumerate() {
            let order = 2 * (j + 1);
            let af2_s = f2[0][i][order - 1] * (1.0 - ssn) + f2[1][i][order - 1] * ssn;
            let af2_c = f2[0][i][order] * (1.0 - ssn) + f2[1][i][order] * ssn;
            value += (s * af2_s) + (cc * af2_c);
        }
        *c = value;
    }

    let mut cm3 = [0.0_f64; FM3_COEFF_MAX_DEGREE];
    // M(3000)F2: 4 harmonics over 9 orders (1 + 2*4)
    for (i, c) in cm3.iter_mut().enumerate() {
        let am3_0 = fm3[0][i][0] * (1.0 - ssn) + fm3[1][i][0] * ssn;
        let mut value = am3_0;
        for j in 0..4 {
            let order = 2 * (j + 1);
            let am3_s = fm3[0][i][order - 1] * (1.0 - ssn) + fm3[1][i][order - 1] * ssn;
            let am3_c = fm3[0][i][order] * (1.0 - ssn) + fm3[1][i][order] * ssn;
            value += (sin_h[j] * am3_s) + (cos_h[j] * am3_c);
        }
        *c = value;
    }

    (cf2, cm3)
}

// Legendre degrees per order for the foF2 and M(3000)F2 expansions.
const F2_CRIT_GRADES: [usize; 9] = [12, 12, 9, 5, 2, 1, 1, 1, 1];
const FM3_GRADES: [usize; 7] = [7, 8, 6, 3, 2, 1, 1];

fn freq_to_density(crit_freq_mhz: f64) -> f64 {
    0.124 * crit_freq_mhz * crit_freq_mhz
}

fn legendre_expansion(
    coeff: &[f64],
    grades: &[usize],
    modip_coeff: &[f64; 12],
    long_sin: &[f64; 8],
    long_cos: &[f64; 8],
    cos_lat: f64,
) -> f64 {
    let mut parameter = 0.0;
    let mut degree_index = 0usize;

    for &mc in modip_coeff.iter().take(grades[0]) {
        parameter += coeff[degree_index] * mc;
        degree_index += 1;
    }

    let mut lat_coeff = cos_lat;
    for (i, &grade) in grades.iter().enumerate().skip(1) {
        for &mc in modip_coeff.iter().take(grade) {
            parameter += mc
                * lat_coeff
                * ((coeff[degree_index] * long_cos[i - 1])
                    + (coeff[degree_index + 1] * long_sin[i - 1]));
            degree_index += 2;
        }
        lat_coeff *= cos_lat;
    }
    parameter
}

fn modip_legendre_coeff(modip_deg: f64) -> [f64; 12] {
    let mut c = [0.0_f64; 12];
    c[0] = 1.0;
    let s = (modip_deg * DEG_TO_RAD).sin();
    c[1] = s;
    for i in 2..12 {
        let mut term = c[i - 1] * c[1];
        if term.abs() <= 1.0e-30 {
            term = 0.0;
        }
        c[i] = term;
    }
    c
}

fn longitude_legendre_coeff(lon: &Angle) -> ([f64; 8], [f64; 8]) {
    let mut sin = [0.0_f64; 8];
    let mut cos = [0.0_f64; 8];
    sin[0] = lon.sin;
    cos[0] = lon.cos;
    let mut n_long = 2.0 * lon.rad;
    for i in 1..8 {
        sin[i] = n_long.sin();
        cos[i] = n_long.cos();
        n_long += lon.rad;
    }
    (sin, cos)
}

// ---------------------------------------------------------------------------
// ionospheric profile at a point (reference 2.5.5, 2.5.6)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct Peak {
    height_km: f64,
    thickness_top_km: f64,
    thickness_bottom_km: f64,
    electron_density: f64,
    amplitude: f64,
}

struct Profile {
    e: Peak,
    f1: Peak,
    f2: Peak,
}

impl NequickModel {
    fn profile_at(&self, position: &Position) -> Profile {
        let modip = modip_degree(position.lon.deg, position.lat.deg);

        // critical frequencies
        let e_crit_freq = self.e_layer_critical_freq(position);
        let modip_coeff = modip_legendre_coeff(modip);
        let (long_sin, long_cos) = longitude_legendre_coeff(&position.lon);
        let cos_lat = position.lat.cos;

        let f2_crit_freq = legendre_expansion(
            &self.cf2,
            &F2_CRIT_GRADES,
            &modip_coeff,
            &long_sin,
            &long_cos,
            cos_lat,
        );
        let trans_factor = legendre_expansion(
            &self.cm3,
            &FM3_GRADES,
            &modip_coeff,
            &long_sin,
            &long_cos,
            cos_lat,
        )
        .max(1.0);

        let f1_crit_freq = f1_critical_freq(e_crit_freq, f2_crit_freq);

        let mut e = Peak {
            electron_density: freq_to_density(e_crit_freq),
            ..Peak::default()
        };
        let mut f1 = Peak {
            electron_density: freq_to_density(f1_crit_freq),
            ..Peak::default()
        };
        let mut f2 = Peak {
            electron_density: freq_to_density(f2_crit_freq),
            ..Peak::default()
        };

        // peak heights
        e.height_km = 120.0;
        f2.height_km = f2_peak_height(trans_factor, f2_crit_freq, e_crit_freq);
        f1.height_km = (e.height_km + f2.height_km) / 2.0;

        // peak thicknesses
        f2.thickness_top_km = f64::INFINITY;
        f2.thickness_bottom_km = {
            let grad = 0.01
                * (-3.467 + 0.857 * (f2_crit_freq * f2_crit_freq).ln() + 2.02 * trans_factor.ln())
                    .exp();
            (0.385 * f2.electron_density) / grad
        };
        f1.thickness_top_km = 0.3 * (f2.height_km - f1.height_km);
        f1.thickness_bottom_km = 0.5 * (f1.height_km - e.height_km);
        e.thickness_top_km = f1.thickness_bottom_km.max(7.0);
        e.thickness_bottom_km = 5.0;

        // peak amplitudes
        compute_peak_amplitudes(&mut e, &mut f1, &mut f2, f1_crit_freq);

        // exosphere (topside) adjustment
        self.exosphere_adjust(&mut f2);

        Profile { e, f1, f2 }
    }

    fn e_layer_critical_freq(&self, position: &Position) -> f64 {
        let effective_zenith = self.solar_effective_zenith_angle(position);
        let season = match self.month {
            1 | 2 | 11 | 12 => -1.0,
            3 | 4 | 9 | 10 => 0.0,
            _ => 1.0,
        };
        let ee = nq_exp(0.3 * position.lat.deg);
        let lat_param = (season * (ee - 1.0)) / (ee + 1.0);

        let mut crit = (1.112 - 0.019 * lat_param) * self.az_sfu.sqrt().sqrt();
        crit *= nq_exp((effective_zenith * DEG_TO_RAD).cos().ln() * 0.3);
        (crit * crit + 0.49).sqrt()
    }

    fn solar_effective_zenith_angle(&self, position: &Position) -> f64 {
        let mut local_time = self.utc_hours + (position.lon.deg / 15.0);
        if local_time < 0.0 {
            local_time += 24.0;
        } else if local_time >= 24.0 {
            local_time -= 24.0;
        }
        let cos_hour = (PI * (12.0 - local_time) / 12.0).cos();
        let zenith_cos = position.lat.sin * self.solar_decl_sin
            + position.lat.cos * self.solar_decl_cos * cos_hour;
        let zenith = Angle::from_cos(zenith_cos);

        let func1 = 90.0 - 0.24 * nq_exp(20.0 - 0.2 * zenith.deg);
        nq_join(func1, zenith.deg, 12.0, zenith.deg - 86.232_927_962_116_15)
    }

    fn exosphere_adjust(&self, f2: &mut Peak) {
        let shape_factor = {
            let mut sf = if self.month > 3 && self.month < 10 {
                6.705 - 0.014 * self.azr - 0.008 * f2.height_km
            } else {
                -7.77
                    + 0.097 * square(f2.height_km / f2.thickness_bottom_km)
                    + 0.153 * f2.electron_density
            };
            sf = nq_join(sf, 2.0, 1.0, sf - 2.0);
            nq_join(8.0, sf, 1.0, sf - 8.0)
        };

        let mut top = shape_factor * f2.thickness_bottom_km;
        let x = (top - 150.0) / 100.0;
        let v = ((0.041163 * x - 0.183981) * x) + 1.424472;
        top /= v;
        f2.thickness_top_km = top;
        f2.electron_density = f64::NAN;
    }
}

fn f1_critical_freq(e_crit: f64, f2_crit: f64) -> f64 {
    let mut f1 = nq_join(1.4 * e_crit, 0.0, 1000.0, e_crit - 2.0);
    f1 = nq_join(0.0, f1, 1000.0, e_crit - f1);
    f1 = nq_join(f1, 0.85 * f1, 60.0, (0.85 * f2_crit) - f1);
    if f1 < 1.0e-6 {
        0.0
    } else {
        f1
    }
}

fn f2_peak_height(trans_factor: f64, f2_crit: f64, e_crit: f64) -> f64 {
    let m2 = trans_factor * trans_factor;
    let numerator = 1490.0 * trans_factor * ((0.0196 * m2 + 1.0) / (1.2967 * m2 - 1.0)).sqrt();
    let mut denominator = trans_factor + (-0.012);

    if e_crit >= 1.0e-30 {
        let mut r = f2_crit / e_crit;
        r = nq_join(r, 1.75, 20.0, r - 1.75);
        denominator += 0.253 / (r - 1.215);
    }
    (numerator / denominator) - 176.0
}

fn peak_amplitude_contribution(peak: &Peak, height_km: f64) -> f64 {
    let thickness = if peak.height_km > height_km {
        peak.thickness_bottom_km
    } else {
        peak.thickness_top_km
    };
    let mut ed = nq_exp((height_km - peak.height_km) / thickness);
    ed = peak.amplitude * ed / square(ed + 1.0);
    4.0 * ed
}

fn compute_peak_amplitudes(e: &mut Peak, f1: &mut Peak, f2: &mut Peak, f1_crit_freq: f64) {
    f2.amplitude = 4.0 * f2.electron_density;

    let e_sub_f2 = (4.0 * e.electron_density) - peak_amplitude_contribution(f2, e.height_km);

    if f1_crit_freq >= 0.5 {
        let f1_sub_f2 = (4.0 * f1.electron_density) - peak_amplitude_contribution(f2, f1.height_km);
        e.amplitude = 4.0 * e.electron_density;
        for _ in 0..5 {
            let amp_e_at_f1 = peak_amplitude_contribution(e, f1.height_km);
            f1.amplitude = f1_sub_f2 - amp_e_at_f1;
            f1.amplitude = nq_join(
                f1.amplitude,
                0.8 * f1.electron_density,
                1.0,
                f1.amplitude - (0.8 * f1.electron_density),
            );
            e.amplitude = e_sub_f2 - peak_amplitude_contribution(f1, e.height_km);
        }
    } else {
        f1.amplitude = 0.0;
        e.amplitude = e_sub_f2;
    }

    e.amplitude = nq_join(e.amplitude, 0.05, 60.0, e.amplitude - 0.005);
}

// ---------------------------------------------------------------------------
// electron density at a height given a profile (reference 2.5.6)
// ---------------------------------------------------------------------------

fn electron_density(profile: &mut Profile, height_km: f64) -> f64 {
    if height_km > profile.f2.height_km {
        top_side(profile, height_km)
    } else {
        bottom_side(profile, height_km)
    }
}

fn top_side(profile: &mut Profile, height_km: f64) -> f64 {
    let h_above = height_km - profile.f2.height_km;
    let delta_height = 0.125 * h_above;
    let denom = profile.f2.thickness_top_km
        * (1.0 + ((100.0 * delta_height) / ((100.0 * profile.f2.thickness_top_km) + delta_height)));
    let mut temp = nq_exp(h_above / denom);

    if temp > 1.0e11 {
        temp = 1.0 / temp;
    } else {
        temp /= square(1.0 + temp);
    }

    if profile.f2.electron_density.is_nan() {
        profile.f2.electron_density = bottom_side(profile, profile.f2.height_km);
    }
    (4.0 * temp) * profile.f2.electron_density
}

struct BottomLayer {
    b_param: f64,
    exp_value: f64,
    s: f64,
    above_threshold: bool,
}

fn bottom_side(profile: &Profile, height_km: f64) -> f64 {
    let f2_b = profile.f2.thickness_bottom_km;
    let f1_b = if height_km > profile.f1.height_km {
        profile.f1.thickness_top_km
    } else {
        profile.f1.thickness_bottom_km
    };
    let e_b = if height_km > profile.e.height_km {
        profile.e.thickness_top_km
    } else {
        profile.e.thickness_bottom_km
    };

    let h = height_km.max(100.0);
    let h_above_f2 = h - profile.f2.height_km;

    let mut f2_arg = h_above_f2 / f2_b;
    let mut f1_arg = (h - profile.f1.height_km) / f1_b;
    let mut e_arg = (h - profile.e.height_km) / e_b;

    let temp = (10.0 / (h_above_f2.abs() + 1.0)).exp();
    f1_arg *= temp;
    e_arg *= temp;

    let f2_above = f2_arg.abs() > 25.0;
    let f1_above = f1_arg.abs() > 25.0;
    let e_above = e_arg.abs() > 25.0;

    f2_arg = f2_arg.exp();
    f1_arg = f1_arg.exp();
    e_arg = e_arg.exp();

    let s = |amp: f64, exp_value: f64, above: bool| {
        if above {
            0.0
        } else {
            amp * exp_value / square(exp_value + 1.0)
        }
    };

    let f2_layer = BottomLayer {
        b_param: f2_b,
        exp_value: f2_arg,
        s: s(profile.f2.amplitude, f2_arg, f2_above),
        above_threshold: f2_above,
    };
    let f1_layer = BottomLayer {
        b_param: f1_b,
        exp_value: f1_arg,
        s: s(profile.f1.amplitude, f1_arg, f1_above),
        above_threshold: f1_above,
    };
    let e_layer = BottomLayer {
        b_param: e_b,
        exp_value: e_arg,
        s: s(profile.e.amplitude, e_arg, e_above),
        above_threshold: e_above,
    };

    let density = if height_km < 100.0 {
        bottom_side_low(&f2_layer, &f1_layer, &e_layer, height_km)
    } else {
        f2_layer.s + f1_layer.s + e_layer.s
    };
    1.0e11 * density
}

fn bottom_side_low(f2: &BottomLayer, f1: &BottomLayer, e: &BottomLayer, height_km: f64) -> f64 {
    let ds = |layer: &BottomLayer| {
        if layer.above_threshold {
            0.0
        } else {
            ((1.0 - layer.exp_value) / (1.0 + layer.exp_value)) / layer.b_param
        }
    };
    let s_sum = f2.s + f1.s + e.s;
    let s_ds_sum = (f2.s * ds(f2)) + (f1.s * ds(f1)) + (e.s * ds(e));
    let bc = 1.0 - ((s_ds_sum / s_sum) * 10.0);
    let z = (height_km - 100.0) / 10.0;
    s_sum * nq_exp(1.0 - ((bc * z) + nq_exp(-z)))
}

// ---------------------------------------------------------------------------
// ray geometry (reference 2.5.8) and slant sampling
// ---------------------------------------------------------------------------

struct RayGeometry {
    is_vertical: bool,
    perigee_radius_km: f64,
    azimuth_sin: f64,
    azimuth_cos: f64,
    perigee_lat: Angle,
    perigee_lon: Angle,
    // receiver latitude is replaced by perigee latitude for non-vertical rays
    receiver_lat_sin: f64,
    receiver_lat_cos: f64,
    receiver_height_km: f64,
    receiver_distance_km: f64,
    satellite_height_km: f64,
    satellite_distance_km: f64,
}

fn slant_distance(perigee_radius_km: f64, radius_km: f64) -> f64 {
    (square(radius_km) - square(perigee_radius_km)).abs().sqrt()
}

impl RayGeometry {
    fn new(station: &Position, satellite: &Position) -> Result<Self> {
        let is_vertical_above = (satellite.lat.deg - station.lat.deg).abs() < 1.0e-5
            && (satellite.lon.deg - station.lon.deg).abs() < 1.0e-5;

        if is_vertical_above {
            return Ok(Self {
                is_vertical: true,
                perigee_radius_km: 0.0,
                azimuth_sin: 0.0,
                azimuth_cos: 0.0,
                perigee_lat: station.lat,
                perigee_lon: station.lon,
                receiver_lat_sin: station.lat.sin,
                receiver_lat_cos: station.lat.cos,
                receiver_height_km: station.height_km,
                receiver_distance_km: 0.0,
                satellite_height_km: satellite.height_km,
                satellite_distance_km: 0.0,
            });
        }

        let lon_delta = satellite.lon.rad - station.lon.rad;
        let lon_delta_sin = lon_delta.sin();
        let lon_delta_cos = lon_delta.cos();

        let delta_cos = (station.lat.sin * satellite.lat.sin)
            + (station.lat.cos * satellite.lat.cos * lon_delta_cos);
        let delta_sin = sin_from_cos(delta_cos);

        // zenith angle of the satellite seen from the receiver
        let zenith_rad = delta_sin.atan2(delta_cos - (station.radius_km / satellite.radius_km));
        let zenith_sin = zenith_rad.sin();

        let perigee_radius_km = station.radius_km * zenith_sin;
        if zenith_rad.abs() > (90.0 * DEG_TO_RAD) && perigee_radius_km < EARTH_RADIUS_KM {
            return Err(invalid("nequick_g_ray", "line of sight below the surface"));
        }
        let is_vertical = perigee_radius_km < 0.1;

        // ray-perigee sigma (azimuth at the receiver of the great circle)
        let sigma_sin = (lon_delta_sin * satellite.lat.cos) / delta_sin;
        let sigma_cos =
            ((satellite.lat.sin - (delta_cos * station.lat.sin)) / delta_sin) / station.lat.cos;

        let delta_p_rad = (90.0 * DEG_TO_RAD) - zenith_rad;
        let delta_p_sin = delta_p_rad.sin();
        let delta_p_cos = delta_p_rad.cos();

        let perigee_lat_sin =
            (station.lat.sin * delta_p_cos) - (station.lat.cos * delta_p_sin * sigma_cos);
        let perigee_lat = Angle::from_sin(perigee_lat_sin);

        let sin_lamp = (-sigma_sin * delta_p_sin) / perigee_lat.cos;
        let cos_lamp = ((delta_p_cos - (station.lat.sin * perigee_lat.sin)) / station.lat.cos)
            / perigee_lat.cos;
        let perigee_lon = Angle::from_rad(sin_lamp.atan2(cos_lamp) + station.lon.rad);

        let mut geometry = Self {
            is_vertical,
            perigee_radius_km,
            azimuth_sin: 0.0,
            azimuth_cos: 0.0,
            perigee_lat,
            perigee_lon,
            receiver_lat_sin: station.lat.sin,
            receiver_lat_cos: station.lat.cos,
            receiver_height_km: station.height_km,
            receiver_distance_km: 0.0,
            satellite_height_km: satellite.height_km,
            satellite_distance_km: 0.0,
        };

        if !is_vertical {
            // receiver latitude becomes the ray-perigee latitude
            geometry.receiver_lat_sin = perigee_lat.sin;
            geometry.receiver_lat_cos = perigee_lat.cos;
            geometry.compute_azimuth(satellite);
            geometry.receiver_distance_km = slant_distance(perigee_radius_km, station.radius_km);
            geometry.satellite_distance_km = slant_distance(perigee_radius_km, satellite.radius_km);
        }

        Ok(geometry)
    }

    fn compute_azimuth(&mut self, satellite: &Position) {
        if (self.perigee_lat.deg.abs() - 90.0).abs() < 1.0e-10 {
            self.azimuth_sin = 0.0;
            self.azimuth_cos = if self.perigee_lat.deg > 0.0 {
                -1.0
            } else {
                1.0
            };
            return;
        }
        let delta_rad = satellite.lon.rad - self.perigee_lon.rad;
        let psi_cos = (self.perigee_lat.sin * satellite.lat.sin)
            + self.perigee_lat.cos * satellite.lat.cos * delta_rad.cos();
        let psi_sin = sin_from_cos(psi_cos);

        self.azimuth_sin = (satellite.lat.cos * delta_rad.sin()) / psi_sin;
        self.azimuth_cos = (satellite.lat.sin - (self.perigee_lat.sin * psi_cos))
            / (psi_sin * self.perigee_lat.cos);
    }

    /// Geodetic position of the ray point at slant distance `s_km` from perigee.
    fn point_at_slant(&self, s_km: f64) -> Position {
        let radius_km = (square(s_km) + square(self.perigee_radius_km)).sqrt();
        let height_km = if radius_km < EARTH_RADIUS_KM {
            0.0
        } else {
            radius_km - EARTH_RADIUS_KM
        };

        let tan_delta = s_km / self.perigee_radius_km;
        let delta_cos = 1.0 / (1.0 + square(tan_delta)).sqrt();
        let delta_sin = tan_delta * delta_cos;

        let lat_sin = (self.receiver_lat_sin * delta_cos)
            + (self.receiver_lat_cos * delta_sin * self.azimuth_cos);
        let lat = Angle::from_sin(lat_sin);

        let dlam_sin = delta_sin * self.azimuth_sin * self.receiver_lat_cos;
        let dlam_cos = delta_cos - (self.receiver_lat_sin * lat.sin);
        let dlam = dlam_sin.atan2(dlam_cos);
        let lon = Angle::from_rad(dlam + self.perigee_lon.rad);

        Position {
            lon,
            lat,
            height_km,
            radius_km,
        }
    }
}

// ---------------------------------------------------------------------------
// TEC integration (reference F.2.3 - adaptive Gauss-Kronrod over breakpoints)
// ---------------------------------------------------------------------------

const RECURSION_MAX: u32 = 50;

const KRONROD_NODES: [f64; 15] = [
    -0.9914553711208126,
    -0.9491079123427585,
    -0.8648644233597691,
    -0.7415311855993945,
    -0.5860872354676911,
    -0.4058451513773972,
    -0.20778495500789848,
    0.0,
    0.20778495500789848,
    0.4058451513773972,
    0.5860872354676911,
    0.7415311855993945,
    0.8648644233597691,
    0.9491079123427585,
    0.9914553711208126,
];

const KRONROD_WEIGHTS: [f64; 15] = [
    0.022935322010529224,
    0.06309209262997856,
    0.10479001032225019,
    0.14065325971552592,
    0.1690047266392679,
    0.19035057806478542,
    0.20443294007529889,
    0.20948214108472782,
    0.20443294007529889,
    0.19035057806478542,
    0.1690047266392679,
    0.14065325971552592,
    0.10479001032225019,
    0.06309209262997856,
    0.022935322010529224,
];

const GAUSS_WEIGHTS: [f64; 7] = [
    0.1294849661688697,
    0.27970539148927664,
    0.3818300505051189,
    0.4179591836734694,
    0.3818300505051189,
    0.27970539148927664,
    0.1294849661688697,
];

struct Integrator<'a> {
    model: &'a NequickModel,
    geometry: &'a RayGeometry,
}

impl Integrator<'_> {
    fn density_at(&self, s_km: f64) -> f64 {
        if self.geometry.is_vertical {
            let lat_deg = self.geometry.perigee_lat.deg;
            let lon_deg = self.geometry.perigee_lon.deg;
            let position = Position {
                lon: Angle::from_deg(lon_deg),
                lat: Angle::from_deg(lat_deg),
                height_km: s_km,
                radius_km: EARTH_RADIUS_KM + s_km,
            };
            let mut profile = self.model.profile_at(&position);
            electron_density(&mut profile, s_km)
        } else {
            let position = self.geometry.point_at_slant(s_km);
            let mut profile = self.model.profile_at(&position);
            electron_density(&mut profile, position.height_km)
        }
    }

    fn integrate(&self, a: f64, b: f64, tolerance: f64, level: u32) -> f64 {
        let mid = (a + b) / 2.0;
        let half = (b - a) / 2.0;

        let mut k15 = 0.0;
        let mut g7 = 0.0;
        let mut g7_index = 0usize;
        for i in 0..15 {
            let height = mid + half * KRONROD_NODES[i];
            let value = self.density_at(height);
            k15 += value * KRONROD_WEIGHTS[i];
            if i % 2 == 1 {
                g7 += value * GAUSS_WEIGHTS[g7_index];
                g7_index += 1;
            }
        }
        k15 *= half;
        g7 *= half;

        let within = (((k15 - g7) / k15).abs() <= tolerance) || ((k15 - g7).abs() <= tolerance);
        if within || level == RECURSION_MAX {
            return k15;
        }

        let left = self.integrate(a, a + half, tolerance, level + 1);
        let right = self.integrate(a + half, b, tolerance, level + 1);
        left + right
    }
}

const TOLERANCE_LOW: f64 = 0.001;
const TOLERANCE_HIGH: f64 = 0.01;
const FIRST_POINT_KM: f64 = 1000.0;
const SECOND_POINT_KM: f64 = 2000.0;

fn integrate_tec(
    model: &NequickModel,
    geometry: &RayGeometry,
    station: &Position,
    satellite: &Position,
) -> f64 {
    let integrator = Integrator { model, geometry };

    // map a geodetic height (km) to the integration variable
    let point_height = |height_km: f64| -> f64 {
        if geometry.is_vertical {
            height_km
        } else {
            slant_distance(geometry.perigee_radius_km, EARTH_RADIUS_KM + height_km)
        }
    };
    let point_zero = || -> f64 { point_height(station.height_km.max(0.0)) };

    let receiver_var = if geometry.is_vertical {
        geometry.receiver_height_km
    } else {
        geometry.receiver_distance_km
    };
    let satellite_var = if geometry.is_vertical {
        geometry.satellite_height_km
    } else {
        geometry.satellite_distance_km
    };

    let sat_height = satellite.height_km;
    let rx_height = station.height_km;

    let run = |a: f64, b: f64, tol: f64| integrator.integrate(a, b, tol, 0);

    if sat_height <= FIRST_POINT_KM {
        return run(point_zero(), satellite_var, TOLERANCE_LOW);
    }

    if sat_height <= SECOND_POINT_KM {
        if rx_height >= FIRST_POINT_KM {
            return run(receiver_var, satellite_var, TOLERANCE_LOW);
        }
        let first = point_height(FIRST_POINT_KM);
        let a = run(point_zero(), first, TOLERANCE_LOW);
        let b = run(first, satellite_var, TOLERANCE_HIGH);
        return a + b;
    }

    if rx_height >= SECOND_POINT_KM {
        return run(receiver_var, satellite_var, TOLERANCE_HIGH);
    }

    if rx_height >= FIRST_POINT_KM {
        let second = point_height(SECOND_POINT_KM);
        let a = run(receiver_var, second, TOLERANCE_HIGH);
        let b = run(second, satellite_var, TOLERANCE_HIGH);
        return a + b;
    }

    let first = point_height(FIRST_POINT_KM);
    let second = point_height(SECOND_POINT_KM);
    let a = run(point_zero(), first, TOLERANCE_LOW);
    let b = run(first, second, TOLERANCE_HIGH);
    let c = run(second, satellite_var, TOLERANCE_HIGH);
    a + b + c
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
