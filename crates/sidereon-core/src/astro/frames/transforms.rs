//! Coordinate transformation pipeline.
//!
//! TEME -> GCRS replicates Skyfield's exact computation path including AU/day
//! unit scaling for bit-exact (0 ULP) parity.
//!
//! Also provides GCRS -> ITRS, ITRS -> geodetic (WGS84), and topocentric
//! (az/el/range) transformations.
//!
//! The pure compute functions live here in the core crate; the Rustler
//! decode/encode shims that used to wrap them stay in `orbis_nif` as glue, so
//! no domain formula lives in the NIF layer. The numerics, summation order,
//! transcendental sequence, and the single sanctioned `mul_add` site
//! (`mat3_vec3_mul_fma`) are preserved exactly so the existing Skyfield 0-ULP
//! parity holds.

use crate::astro::frames::nutation::{
    build_skyfield_nutation_matrix_unchecked,
    skyfield_equation_of_the_equinoxes_complimentary_terms_unchecked,
    skyfield_iau2000a_radians_unchecked, skyfield_mean_obliquity_radians_unchecked,
};
use crate::astro::frames::precession::{
    build_icrs_to_j2000, compute_skyfield_precession_matrix_unchecked,
};
use crate::astro::math::mat3::{inline_mxmxm, inline_rxr, inline_tr, Mat3};
use crate::astro::time::scales::TimeScales;
use crate::astro::{
    constants::astro::AU_KM,
    constants::earth::{WGS84_A_KM, WGS84_E2, WGS84_F},
    constants::models::proj::{
        HALF_PI as PROJ_HALF_PI, RAD_TO_DEG as PROJ_RAD_TO_DEG, WGS84_A_M as PROJ_WGS84_A_M,
        WGS84_B_M as PROJ_WGS84_B_M, WGS84_E2S as PROJ_WGS84_E2S, WGS84_ES as PROJ_WGS84_ES,
    },
    constants::time::{DAYS_PER_JULIAN_CENTURY, J2000_JD, SECONDS_PER_DAY},
};

const TAU: f64 = std::f64::consts::TAU;
const ARCSECONDS_TO_RADIANS: f64 = 4.848_136_811_095_36e-6;

/// Error returned when public frame-transform inputs are outside the valid domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FrameTransformError {
    /// A transform input was non-finite or otherwise invalid.
    #[error("invalid frame transform {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

fn invalid_input(field: &'static str, reason: &'static str) -> FrameTransformError {
    FrameTransformError::InvalidInput { field, reason }
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), FrameTransformError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "must be finite"))
    }
}

fn validate_vec3(field: &'static str, values: &[f64; 3]) -> Result<(), FrameTransformError> {
    for value in values {
        if !value.is_finite() {
            return Err(invalid_input(field, "components must be finite"));
        }
    }
    Ok(())
}

fn validate_tuple3(field: &'static str, values: Vec3) -> Result<Vec3, FrameTransformError> {
    if values.0.is_finite() && values.1.is_finite() && values.2.is_finite() {
        Ok(values)
    } else {
        Err(invalid_input(field, "components must be finite"))
    }
}

fn validate_array3(field: &'static str, values: [f64; 3]) -> Result<[f64; 3], FrameTransformError> {
    validate_vec3(field, &values)?;
    Ok(values)
}

fn validate_mat3(field: &'static str, values: Mat3) -> Result<Mat3, FrameTransformError> {
    for row in &values {
        validate_vec3(field, row)?;
    }
    Ok(values)
}

fn validate_time_scales(ts: &TimeScales) -> Result<(), FrameTransformError> {
    validate_finite("jd_whole", ts.jd_whole)?;
    validate_finite("ut1_fraction", ts.ut1_fraction)?;
    validate_finite("tt_fraction", ts.tt_fraction)?;
    validate_finite("tdb_fraction", ts.tdb_fraction)?;
    validate_finite("jd_ut1", ts.jd_ut1)?;
    validate_finite("jd_tt", ts.jd_tt)?;
    validate_finite("jd_tdb", ts.jd_tdb)
}

fn validate_polar_motion(pole: PolarMotion) -> Result<(), FrameTransformError> {
    validate_finite("xp_rad", pole.xp_rad)?;
    validate_finite("yp_rad", pole.yp_rad)
}

fn validate_geodetic_degrees_km(
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_km: f64,
) -> Result<(), FrameTransformError> {
    validate_finite("latitude_deg", latitude_deg)?;
    if !(-90.0..=90.0).contains(&latitude_deg) {
        return Err(invalid_input("latitude_deg", "must be in [-90, 90]"));
    }
    validate_finite("longitude_deg", longitude_deg)?;
    if !(-180.0..=180.0).contains(&longitude_deg) {
        return Err(invalid_input("longitude_deg", "must be in [-180, 180]"));
    }
    validate_finite("altitude_km", altitude_km)
}

/// A bare Cartesian triple (km or km/s depending on context).
///
/// This is the internal compute-layer return shape. Typed input structs
/// ([`TemeStateKm`], [`GeodeticStationKm`]) bundle the public entry points'
/// arguments, but the numerics below operate on raw triples to preserve the
/// original operation order exactly.
pub type Vec3 = (f64, f64, f64);

/// TEME-frame position and velocity (km, km/s): the input to
/// [`teme_to_gcrs_compute`].
pub struct TemeStateKm {
    pub position_km: [f64; 3],
    pub velocity_km_s: [f64; 3],
}

/// Geodetic ground-station position (WGS84) for topocentric look angles.
pub struct GeodeticStationKm {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_km: f64,
}

/// Polar-motion coordinates of the Celestial Intermediate Pole.
///
/// `xp_rad` and `yp_rad` are radians. The embedded EOP table currently carries
/// UT1-UTC only, so the historical transforms use [`PolarMotion::ZERO`] by
/// default. Precision callers with pole coordinates should use the explicit
/// `*_with_polar_motion` entry points below.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolarMotion {
    pub xp_rad: f64,
    pub yp_rad: f64,
}

impl PolarMotion {
    /// No polar-motion rotation; preserves the historical transform exactly.
    pub const ZERO: Self = Self {
        xp_rad: 0.0,
        yp_rad: 0.0,
    };

    /// Construct polar-motion coordinates from radians.
    pub fn from_radians(xp_rad: f64, yp_rad: f64) -> Result<Self, FrameTransformError> {
        validate_finite("xp_rad", xp_rad)?;
        validate_finite("yp_rad", yp_rad)?;
        Ok(Self { xp_rad, yp_rad })
    }

    /// Construct polar-motion coordinates from arcseconds.
    pub fn from_arcseconds(xp_arcsec: f64, yp_arcsec: f64) -> Result<Self, FrameTransformError> {
        validate_finite("xp_arcsec", xp_arcsec)?;
        validate_finite("yp_arcsec", yp_arcsec)?;
        Self::from_radians(
            xp_arcsec * ARCSECONDS_TO_RADIANS,
            yp_arcsec * ARCSECONDS_TO_RADIANS,
        )
    }

    fn is_zero(self) -> bool {
        self.xp_rad == 0.0 && self.yp_rad == 0.0
    }
}

impl Default for PolarMotion {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Final matrix-vector multiply using explicit FMA.
/// This matches numpy's vectorized behavior and is the ONLY place
/// where f64::mul_add() should be used.
fn mat3_vec3_mul_fma(r: &Mat3, p: &[f64; 3]) -> [f64; 3] {
    let mut result = [0.0_f64; 3];
    for i in 0..3 {
        let sum = r[i][0] * p[0];
        let sum = f64::mul_add(r[i][1], p[1], sum);
        let sum = f64::mul_add(r[i][2], p[2], sum);
        result[i] = sum;
    }
    result
}

fn build_rot_z(angle: f64) -> Mat3 {
    let c = angle.cos();
    let s = angle.sin();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// IERS polar-motion matrix, omitting the tiny TIO locator term `s'`.
///
/// The matrix maps TIRS pseudo-Earth-fixed coordinates to ITRS:
/// `W = R_y(xp) * R_x(yp)`, whose small-angle form is
/// `[[1, 0, xp], [0, 1, -yp], [-xp, yp, 1]]`.
pub fn polar_motion_matrix(pole: PolarMotion) -> Result<Mat3, FrameTransformError> {
    validate_polar_motion(pole)?;
    validate_mat3("polar_motion_matrix", polar_motion_matrix_unchecked(pole))
}

fn polar_motion_matrix_unchecked(pole: PolarMotion) -> Mat3 {
    if pole.is_zero() {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let cx = pole.xp_rad.cos();
    let sx = pole.xp_rad.sin();
    let cy = pole.yp_rad.cos();
    let sy = pole.yp_rad.sin();

    [
        [cx, sx * sy, sx * cy],
        [0.0, cy, -sy],
        [-sx, cx * sy, cx * cy],
    ]
}

fn apply_polar_motion_to_itrs_matrix(mat: Mat3, pole: PolarMotion) -> Mat3 {
    if pole.is_zero() {
        mat
    } else {
        inline_rxr(&polar_motion_matrix_unchecked(pole), &mat)
    }
}

fn earth_rotation_angle(jd_whole: f64, ut1_fraction: f64) -> f64 {
    let days_since_j2000 = jd_whole - J2000_JD + ut1_fraction;
    // Force separate rounded operations to match Skyfield/Python's path.
    let spins_since_j2000: f64 = {
        let v = 0.00273781191135448 * days_since_j2000;
        // Use black_box-like pattern to prevent optimization
        let v_stored: f64 = v;
        v_stored
    };
    let th = 0.7790572732640 + spins_since_j2000;
    let mut result = (th % 1.0 + jd_whole % 1.0 + ut1_fraction) % 1.0;
    if result < 0.0 {
        result += 1.0;
    }
    result
}

fn compute_theta_gmst1982(jd_whole: f64, ut1_fraction: f64) -> f64 {
    let t = (jd_whole - J2000_JD + ut1_fraction) / DAYS_PER_JULIAN_CENTURY;
    let g = 67310.54841 + (8640184.812866 + (0.093104 + (-6.2e-6) * t) * t) * t;
    let mut theta = ((jd_whole % 1.0) + ut1_fraction + (g / SECONDS_PER_DAY) % 1.0) % 1.0 * TAU;
    if theta < 0.0 {
        theta += TAU;
    }
    theta
}

fn sidereal_time_hours(jd_whole: f64, ut1_fraction: f64, tdb_fraction: f64) -> f64 {
    let theta = earth_rotation_angle(jd_whole, ut1_fraction);
    let t = (jd_whole - J2000_JD + tdb_fraction) / DAYS_PER_JULIAN_CENTURY;
    let st = 0.014506
        + ((((-0.0000000368 * t - 0.000029956) * t - 0.00000044) * t + 1.3915817) * t
            + 4612.156534)
            * t;
    let mut result = (st / 54000.0 + theta * 24.0) % 24.0;
    if result < 0.0 {
        result += 24.0;
    }
    result
}

fn gast_radians(ts: &TimeScales, dpsi: f64) -> f64 {
    let gmst_hours = sidereal_time_hours(ts.jd_whole, ts.ut1_fraction, ts.tdb_fraction);
    let mean_ob = skyfield_mean_obliquity_radians_unchecked(ts.jd_tdb);
    let c_terms = skyfield_equation_of_the_equinoxes_complimentary_terms_unchecked(ts.jd_tt);
    let eq_eq = dpsi * mean_ob.cos() + c_terms;
    let mut gast_hours = (gmst_hours + eq_eq / TAU * 24.0) % 24.0;
    if gast_hours < 0.0 {
        gast_hours += 24.0;
    }
    gast_hours / 24.0 * TAU
}

/// Greenwich Mean Sidereal Time for an instant, radians in `[0, 2pi)`.
///
/// The IAU-1982 GMST used internally by the frame pipeline, surfaced as a public
/// entry point. This is a thin wrapper over the existing private sidereal-time
/// computation: it adds no new numerics, so the value is bit-identical to the
/// quantity the transforms consume.
pub fn greenwich_mean_sidereal_time_radians(ts: &TimeScales) -> Result<f64, FrameTransformError> {
    validate_time_scales(ts)?;
    let radians = greenwich_mean_sidereal_time_radians_unchecked(ts);
    validate_finite("gmst_radians", radians)?;
    Ok(radians)
}

fn greenwich_mean_sidereal_time_radians_unchecked(ts: &TimeScales) -> f64 {
    let hours = sidereal_time_hours(ts.jd_whole, ts.ut1_fraction, ts.tdb_fraction);
    hours / 24.0 * TAU
}

/// Greenwich Apparent Sidereal Time for an instant, radians in `[0, 2pi)`.
///
/// GMST plus the equation of the equinoxes (nutation in longitude projected on
/// the true equator, with the IAU 2000 complementary terms). A thin wrapper over
/// the existing private GAST computation; bit-identical to the value the
/// GCRS<->ITRS transforms apply.
pub fn greenwich_apparent_sidereal_time_radians(
    ts: &TimeScales,
) -> Result<f64, FrameTransformError> {
    validate_time_scales(ts)?;
    let radians = greenwich_apparent_sidereal_time_radians_unchecked(ts);
    validate_finite("gast_radians", radians)?;
    Ok(radians)
}

fn greenwich_apparent_sidereal_time_radians_unchecked(ts: &TimeScales) -> f64 {
    let (dpsi, _deps) = skyfield_iau2000a_radians_unchecked(ts.jd_tt);
    gast_radians(ts, dpsi)
}

/// Build the TEME->GCRS rotation matrix T from time scales.
fn build_teme_to_gcrs_matrix(ts: &TimeScales, skyfield_compat: bool) -> Mat3 {
    let (dpsi, deps) = skyfield_iau2000a_radians_unchecked(ts.jd_tt);
    let mean_ob = skyfield_mean_obliquity_radians_unchecked(ts.jd_tdb);
    let true_ob = mean_ob + deps;

    let n = build_skyfield_nutation_matrix_unchecked(mean_ob, true_ob, dpsi);
    let p = compute_skyfield_precession_matrix_unchecked(ts.jd_tdb);
    let b = build_icrs_to_j2000();

    // Skyfield uses Kahan-compensated triple product (matching numpy einsum).
    // Direct mode uses standard sequential multiply (more precise).
    let m = if skyfield_compat {
        inline_mxmxm(&n, &p, &b)
    } else {
        let np = inline_rxr(&n, &p);
        inline_rxr(&np, &b)
    };

    let gast = gast_radians(ts, dpsi);
    let theta = compute_theta_gmst1982(ts.jd_whole, ts.ut1_fraction);
    let angle = theta - gast;

    let r = build_rot_z(angle);
    let g = inline_rxr(&r, &m);
    inline_tr(&g)
}

/// Build the TEME->GCRS rotation matrix T from time scales.
pub(crate) fn teme_to_gcrs_matrix(ts: &TimeScales, skyfield_compat: bool) -> Mat3 {
    build_teme_to_gcrs_matrix(ts, skyfield_compat)
}

/// Standard (non-FMA) matrix-vector multiply.
pub fn mat3_vec3_mul(r: &Mat3, p: &[f64; 3]) -> Result<[f64; 3], FrameTransformError> {
    validate_mat3("matrix", *r)?;
    validate_vec3("vector", p)?;
    validate_array3("matrix_vector_product", mat3_vec3_mul_unchecked(r, p))
}

pub(crate) fn mat3_vec3_mul_unchecked(r: &Mat3, p: &[f64; 3]) -> [f64; 3] {
    let mut result = [0.0_f64; 3];
    for i in 0..3 {
        let mut sum = 0.0;
        for j in 0..3 {
            sum += r[i][j] * p[j];
        }
        result[i] = sum;
    }
    result
}

/// Core TEME->GCRS transform. Returns ((px,py,pz), (vx,vy,vz)).
pub fn teme_to_gcrs_compute(
    state: &TemeStateKm,
    ts: &TimeScales,
    skyfield_compat: bool,
) -> Result<(Vec3, Vec3), FrameTransformError> {
    validate_time_scales(ts)?;
    validate_vec3("position_km", &state.position_km)?;
    validate_vec3("velocity_km_s", &state.velocity_km_s)?;
    let (position, velocity) = teme_to_gcrs_compute_unchecked(state, ts, skyfield_compat);
    Ok((
        validate_tuple3("gcrs_position_km", position)?,
        validate_tuple3("gcrs_velocity_km_s", velocity)?,
    ))
}

fn teme_to_gcrs_compute_unchecked(
    state: &TemeStateKm,
    ts: &TimeScales,
    skyfield_compat: bool,
) -> (Vec3, Vec3) {
    let [x, y, z] = state.position_km;
    let [vx, vy, vz] = state.velocity_km_s;
    let t = build_teme_to_gcrs_matrix(ts, skyfield_compat);

    if skyfield_compat {
        // AU/day scaling + FMA multiply matching Skyfield's _at() path.
        let r_au = [x / AU_KM, y / AU_KM, z / AU_KM];
        let r_gcrs_au = mat3_vec3_mul_fma(&t, &r_au);
        let r_gcrs = (
            r_gcrs_au[0] * AU_KM,
            r_gcrs_au[1] * AU_KM,
            r_gcrs_au[2] * AU_KM,
        );

        let v_au_d = [
            vx / AU_KM * SECONDS_PER_DAY,
            vy / AU_KM * SECONDS_PER_DAY,
            vz / AU_KM * SECONDS_PER_DAY,
        ];
        let v_gcrs_au_d = mat3_vec3_mul_fma(&t, &v_au_d);
        let v_gcrs = (
            v_gcrs_au_d[0] * AU_KM / SECONDS_PER_DAY,
            v_gcrs_au_d[1] * AU_KM / SECONDS_PER_DAY,
            v_gcrs_au_d[2] * AU_KM / SECONDS_PER_DAY,
        );
        (r_gcrs, v_gcrs)
    } else {
        // Direct km/s multiply -- no AU round-trip, no FMA.
        let r_teme = [x, y, z];
        let r_g = mat3_vec3_mul_unchecked(&t, &r_teme);
        let v_teme = [vx, vy, vz];
        let v_g = mat3_vec3_mul_unchecked(&t, &v_teme);
        ((r_g[0], r_g[1], r_g[2]), (v_g[0], v_g[1], v_g[2]))
    }
}

// ---------------------------------------------------------------------------
// GCRS -> ITRS (Earth-fixed / ECEF)
// ---------------------------------------------------------------------------

/// Build the historical GCRS->ITRS rotation matrix for a given time.
///
/// This combines precession, nutation, and Earth rotation with zero polar
/// motion, preserving the original bit-exact path. Use
/// [`gcrs_to_itrs_matrix_with_polar_motion`] when `xp`/`yp` pole coordinates are
/// available.
pub fn gcrs_to_itrs_matrix(ts: &TimeScales) -> Result<Mat3, FrameTransformError> {
    validate_time_scales(ts)?;
    validate_mat3("gcrs_to_itrs_matrix", gcrs_to_itrs_matrix_unchecked(ts))
}

fn gcrs_to_itrs_matrix_unchecked(ts: &TimeScales) -> Mat3 {
    let (dpsi, deps) = skyfield_iau2000a_radians_unchecked(ts.jd_tt);
    let mean_ob = skyfield_mean_obliquity_radians_unchecked(ts.jd_tdb);
    let true_ob = mean_ob + deps;

    let n = build_skyfield_nutation_matrix_unchecked(mean_ob, true_ob, dpsi);
    let p = compute_skyfield_precession_matrix_unchecked(ts.jd_tdb);
    let b = build_icrs_to_j2000();

    // Celestial-to-terrestrial: combine precession, nutation, frame bias
    let m = inline_mxmxm(&n, &p, &b);

    let gast = gast_radians(ts, dpsi);

    // GAST rotation takes us from true-equator-equinox to ITRS
    let r_gast = build_rot_z(-gast);

    // GCRS->ITRS = R_z(-GAST) * (N * P * B)
    inline_rxr(&r_gast, &m)
}

/// Build the GCRS->ITRS rotation matrix with explicit polar motion.
///
/// The embedded Earth-orientation table supplies UT1-UTC but not `xp`/`yp`, so
/// callers that do not have pole coordinates should pass [`PolarMotion::ZERO`]
/// or use [`gcrs_to_itrs_matrix`].
pub fn gcrs_to_itrs_matrix_with_polar_motion(
    ts: &TimeScales,
    pole: PolarMotion,
) -> Result<Mat3, FrameTransformError> {
    validate_time_scales(ts)?;
    validate_polar_motion(pole)?;
    validate_mat3(
        "gcrs_to_itrs_matrix",
        gcrs_to_itrs_matrix_with_polar_motion_unchecked(ts, pole),
    )
}

fn gcrs_to_itrs_matrix_with_polar_motion_unchecked(ts: &TimeScales, pole: PolarMotion) -> Mat3 {
    apply_polar_motion_to_itrs_matrix(gcrs_to_itrs_matrix_unchecked(ts), pole)
}

/// Rotation from the **mean equator and equinox of date** to ITRS, i.e.
/// `R_z(-GAST) * N` (nutation + Earth rotation, *without* precession or frame
/// bias).
///
/// This is [`gcrs_to_itrs_matrix`] with the precession (`P`) and frame-bias
/// (`B`) factors removed. Use it for vectors that are already referred to the
/// mean equator/equinox of date (for example the low-precision analytic Sun/Moon
/// series in [`crate::astro::bodies::sun_moon`], whose mean longitude and obliquity are
/// of-date), so precession is not applied a second time. It mirrors the
/// `eci2ecef` (GMST/GAST + nutation) rotation those series are designed to be
/// consumed with, but uses the crate's IAU 2000A nutation and GAST.
pub fn mean_of_date_to_itrs_matrix(ts: &TimeScales) -> Result<Mat3, FrameTransformError> {
    validate_time_scales(ts)?;
    validate_mat3(
        "mean_of_date_to_itrs_matrix",
        mean_of_date_to_itrs_matrix_unchecked(ts),
    )
}

fn mean_of_date_to_itrs_matrix_unchecked(ts: &TimeScales) -> Mat3 {
    let (dpsi, deps) = skyfield_iau2000a_radians_unchecked(ts.jd_tt);
    let mean_ob = skyfield_mean_obliquity_radians_unchecked(ts.jd_tdb);
    let true_ob = mean_ob + deps;

    let n = build_skyfield_nutation_matrix_unchecked(mean_ob, true_ob, dpsi);
    let gast = gast_radians(ts, dpsi);
    let r_gast = build_rot_z(-gast);

    // mean-of-date -> ITRS = R_z(-GAST) * N
    inline_rxr(&r_gast, &n)
}

/// Mean-of-date to ITRS rotation with explicit polar motion.
pub fn mean_of_date_to_itrs_matrix_with_polar_motion(
    ts: &TimeScales,
    pole: PolarMotion,
) -> Result<Mat3, FrameTransformError> {
    validate_time_scales(ts)?;
    validate_polar_motion(pole)?;
    validate_mat3(
        "mean_of_date_to_itrs_matrix",
        mean_of_date_to_itrs_matrix_with_polar_motion_unchecked(ts, pole),
    )
}

fn mean_of_date_to_itrs_matrix_with_polar_motion_unchecked(
    ts: &TimeScales,
    pole: PolarMotion,
) -> Mat3 {
    apply_polar_motion_to_itrs_matrix(mean_of_date_to_itrs_matrix_unchecked(ts), pole)
}

/// Core GCRS->ITRS transform. Returns (x, y, z) in km.
pub fn gcrs_to_itrs_compute(
    x: f64,
    y: f64,
    z: f64,
    ts: &TimeScales,
    skyfield_compat: bool,
) -> Result<(f64, f64, f64), FrameTransformError> {
    validate_vec3("gcrs_position_km", &[x, y, z])?;
    validate_time_scales(ts)?;
    validate_tuple3(
        "itrs_position_km",
        gcrs_to_itrs_compute_unchecked(x, y, z, ts, skyfield_compat),
    )
}

fn gcrs_to_itrs_compute_unchecked(
    x: f64,
    y: f64,
    z: f64,
    ts: &TimeScales,
    skyfield_compat: bool,
) -> (f64, f64, f64) {
    let mat = gcrs_to_itrs_matrix_unchecked(ts);

    if skyfield_compat {
        // Skyfield: mxv(R, pos_au) in AU, then convert to km.
        // For ITRS, scalar (non-FMA) multiply matches einsum's rounding.
        // (Unlike TEME->GCRS where FMA is needed -- the difference is due to
        // the specific matrix/vector values and how rounding interacts.)
        let pos_au = [x / AU_KM, y / AU_KM, z / AU_KM];
        let r = mat3_vec3_mul_unchecked(&mat, &pos_au);
        (r[0] * AU_KM, r[1] * AU_KM, r[2] * AU_KM)
    } else {
        let pos = [x, y, z];
        let r = mat3_vec3_mul_unchecked(&mat, &pos);
        (r[0], r[1], r[2])
    }
}

/// Core GCRS->ITRS transform with explicit polar motion.
pub fn gcrs_to_itrs_compute_with_polar_motion(
    x: f64,
    y: f64,
    z: f64,
    ts: &TimeScales,
    skyfield_compat: bool,
    pole: PolarMotion,
) -> Result<(f64, f64, f64), FrameTransformError> {
    validate_vec3("gcrs_position_km", &[x, y, z])?;
    validate_time_scales(ts)?;
    validate_polar_motion(pole)?;
    validate_tuple3(
        "itrs_position_km",
        gcrs_to_itrs_compute_with_polar_motion_unchecked(x, y, z, ts, skyfield_compat, pole),
    )
}

fn gcrs_to_itrs_compute_with_polar_motion_unchecked(
    x: f64,
    y: f64,
    z: f64,
    ts: &TimeScales,
    skyfield_compat: bool,
    pole: PolarMotion,
) -> (f64, f64, f64) {
    let mat = gcrs_to_itrs_matrix_with_polar_motion_unchecked(ts, pole);

    if skyfield_compat {
        let pos_au = [x / AU_KM, y / AU_KM, z / AU_KM];
        let r = mat3_vec3_mul_unchecked(&mat, &pos_au);
        (r[0] * AU_KM, r[1] * AU_KM, r[2] * AU_KM)
    } else {
        let pos = [x, y, z];
        let r = mat3_vec3_mul_unchecked(&mat, &pos);
        (r[0], r[1], r[2])
    }
}

// ---------------------------------------------------------------------------
// ITRS -> GCRS (Earth-fixed / ECEF back to inertial)
// ---------------------------------------------------------------------------

/// Build the ITRS->GCRS rotation matrix for a given time.
///
/// This is the transpose of [`gcrs_to_itrs_matrix`]: the same precession,
/// nutation, frame-bias, and Earth-rotation pipeline, taken the other way.
pub fn itrs_to_gcrs_matrix(ts: &TimeScales) -> Result<Mat3, FrameTransformError> {
    validate_time_scales(ts)?;
    validate_mat3("itrs_to_gcrs_matrix", itrs_to_gcrs_matrix_unchecked(ts))
}

fn itrs_to_gcrs_matrix_unchecked(ts: &TimeScales) -> Mat3 {
    inline_tr(&gcrs_to_itrs_matrix_unchecked(ts))
}

/// Build the ITRS->GCRS rotation matrix with explicit polar motion.
pub fn itrs_to_gcrs_matrix_with_polar_motion(
    ts: &TimeScales,
    pole: PolarMotion,
) -> Result<Mat3, FrameTransformError> {
    validate_time_scales(ts)?;
    validate_polar_motion(pole)?;
    validate_mat3(
        "itrs_to_gcrs_matrix",
        itrs_to_gcrs_matrix_with_polar_motion_unchecked(ts, pole),
    )
}

fn itrs_to_gcrs_matrix_with_polar_motion_unchecked(ts: &TimeScales, pole: PolarMotion) -> Mat3 {
    inline_tr(&gcrs_to_itrs_matrix_with_polar_motion_unchecked(ts, pole))
}

/// Core ITRS->GCRS transform. Returns (x, y, z) in km.
///
/// Uses the plain (non-FMA, no AU round-trip) km path. The Skyfield AU-scaled
/// `mul_add` path is reserved for the GCRS->ITRS / TEME->GCRS directions that
/// carry the 0-ULP parity contract; this reverse direction is an ordinary
/// matrix-vector product.
pub fn itrs_to_gcrs_compute(
    x: f64,
    y: f64,
    z: f64,
    ts: &TimeScales,
) -> Result<(f64, f64, f64), FrameTransformError> {
    validate_vec3("itrs_position_km", &[x, y, z])?;
    validate_time_scales(ts)?;
    validate_tuple3(
        "gcrs_position_km",
        itrs_to_gcrs_compute_unchecked(x, y, z, ts),
    )
}

fn itrs_to_gcrs_compute_unchecked(x: f64, y: f64, z: f64, ts: &TimeScales) -> (f64, f64, f64) {
    let mat = itrs_to_gcrs_matrix_unchecked(ts);
    let r = mat3_vec3_mul_unchecked(&mat, &[x, y, z]);
    (r[0], r[1], r[2])
}

/// Core ITRS->GCRS transform with explicit polar motion.
pub fn itrs_to_gcrs_compute_with_polar_motion(
    x: f64,
    y: f64,
    z: f64,
    ts: &TimeScales,
    pole: PolarMotion,
) -> Result<(f64, f64, f64), FrameTransformError> {
    validate_vec3("itrs_position_km", &[x, y, z])?;
    validate_time_scales(ts)?;
    validate_polar_motion(pole)?;
    validate_tuple3(
        "gcrs_position_km",
        itrs_to_gcrs_compute_with_polar_motion_unchecked(x, y, z, ts, pole),
    )
}

fn itrs_to_gcrs_compute_with_polar_motion_unchecked(
    x: f64,
    y: f64,
    z: f64,
    ts: &TimeScales,
    pole: PolarMotion,
) -> (f64, f64, f64) {
    let mat = itrs_to_gcrs_matrix_with_polar_motion_unchecked(ts, pole);
    let r = mat3_vec3_mul_unchecked(&mat, &[x, y, z]);
    (r[0], r[1], r[2])
}

// ---------------------------------------------------------------------------
// ITRS -> Geodetic (WGS84 lat/lon/alt)
// ---------------------------------------------------------------------------

/// Convert ECEF/ITRS (km) to geodetic coordinates.
/// Returns (latitude_deg, longitude_deg, altitude_km).
///
/// Replicates Skyfield's exact algorithm (wgs84.subpoint / _compute_latitude)
/// which works in AU with exactly 3 iterations.
pub fn itrs_to_geodetic_compute(
    x: f64,
    y: f64,
    z: f64,
) -> Result<(f64, f64, f64), FrameTransformError> {
    validate_vec3("itrs_position_km", &[x, y, z])?;
    validate_tuple3("geodetic", itrs_to_geodetic_compute_unchecked(x, y, z))
}

fn itrs_to_geodetic_compute_unchecked(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // Convert to AU to match Skyfield's computation path.
    let x_au = x / AU_KM;
    let y_au = y / AU_KM;
    let z_au = z / AU_KM;

    let a_au = WGS84_A_KM / AU_KM; // Earth equatorial radius in AU
    let r_xy = (x_au * x_au + y_au * y_au).sqrt();

    // Longitude: match Skyfield's exact normalization:
    // (arctan2(y, x) - pi) % tau - pi
    // Python's % always returns positive; Rust's can be negative.
    let lon_raw = y_au.atan2(x_au);
    let pi = std::f64::consts::PI;
    let mut lon_shifted = (lon_raw - pi) % TAU;
    if lon_shifted < 0.0 {
        lon_shifted += TAU;
    }
    let lon = lon_shifted - pi;

    // Latitude: 3 iterations matching Skyfield exactly
    let mut lat = z_au.atan2(r_xy);
    let mut a_c = 0.0_f64;
    let mut hyp = 0.0_f64;

    for _ in 0..3 {
        let sin_lat = lat.sin();
        let e2_sin_lat = WGS84_E2 * sin_lat;
        a_c = a_au / (1.0 - e2_sin_lat * sin_lat).sqrt();
        hyp = z_au + a_c * e2_sin_lat;
        lat = hyp.atan2(r_xy);
    }

    // Elevation in AU, then convert to km
    let height_au = (hyp * hyp + r_xy * r_xy).sqrt() - a_c;
    let alt = height_au * AU_KM;

    // Skyfield's Angle.degrees uses: radians * 360.0 / tau
    // This gives different rounding than radians * (180.0 / PI).
    (lat * 360.0 / TAU, lon * 360.0 / TAU, alt)
}

fn proj_normal_radius_of_curvature(sinphi: f64) -> f64 {
    if PROJ_WGS84_ES == 0.0 {
        return PROJ_WGS84_A_M;
    }
    PROJ_WGS84_A_M / (1.0 - (PROJ_WGS84_ES * sinphi) * sinphi).sqrt()
}

fn proj_geocentric_radius(cosphi: f64, sinphi: f64) -> f64 {
    ((PROJ_WGS84_A_M * PROJ_WGS84_A_M) * cosphi).hypot((PROJ_WGS84_B_M * PROJ_WGS84_B_M) * sinphi)
        / (PROJ_WGS84_A_M * cosphi).hypot(PROJ_WGS84_B_M * sinphi)
}

/// Convert ECEF meters to `(longitude_degrees, latitude_degrees, altitude_m)`.
///
/// This is an additive PROJ parity variant and does not replace
/// [`itrs_to_geodetic_compute`]. It matches pyproj 3.6.1 / PROJ 9.3.0 for
/// `EPSG:4978 -> EPSG:4979` with `always_xy=True`; its Tier 1 bit fixture is
/// `crates/sidereon-core/tests/fixtures/geodetic/geodetic_proj.json`, generated
/// by `crates/sidereon-core/fixtures-generators/generate_geodetic_proj.py`.
pub fn geodetic_from_ecef_proj(x: f64, y: f64, z: f64) -> Result<[f64; 3], FrameTransformError> {
    validate_vec3("ecef_m", &[x, y, z])?;
    validate_array3("geodetic_proj", geodetic_from_ecef_proj_unchecked(x, y, z))
}

fn geodetic_from_ecef_proj_unchecked(x: f64, y: f64, z: f64) -> [f64; 3] {
    let p = x.hypot(y);

    let y_theta = z * PROJ_WGS84_A_M;
    let x_theta = p * PROJ_WGS84_B_M;
    let norm = y_theta.hypot(x_theta);
    let c = if norm == 0.0 { 1.0 } else { x_theta / norm };
    let s = if norm == 0.0 { 0.0 } else { y_theta / norm };

    let y_phi = z + ((((PROJ_WGS84_E2S * PROJ_WGS84_B_M) * s) * s) * s);
    let x_phi = p - ((((PROJ_WGS84_ES * PROJ_WGS84_A_M) * c) * c) * c);
    let norm_phi = y_phi.hypot(x_phi);
    let mut cosphi = if norm_phi == 0.0 {
        1.0
    } else {
        x_phi / norm_phi
    };
    let mut sinphi = if norm_phi == 0.0 {
        0.0
    } else {
        y_phi / norm_phi
    };

    let phi = if x_phi <= 0.0 {
        cosphi = 0.0;
        if z >= 0.0 {
            sinphi = 1.0;
            PROJ_HALF_PI
        } else {
            sinphi = -1.0;
            -PROJ_HALF_PI
        }
    } else {
        (y_phi / x_phi).atan()
    };

    let lam = y.atan2(x);
    let alt = if cosphi < 1e-6 {
        z.abs() - proj_geocentric_radius(cosphi, sinphi)
    } else {
        p / cosphi - proj_normal_radius_of_curvature(sinphi)
    };

    [lam * PROJ_RAD_TO_DEG, phi * PROJ_RAD_TO_DEG, alt]
}

// ---------------------------------------------------------------------------
// Topocentric (az/el/range) from ground station to satellite
// ---------------------------------------------------------------------------

/// Convert geodetic (lat_deg, lon_deg, alt_km) to ECEF/ITRS (km).
pub fn geodetic_to_itrs(
    lat_deg: f64,
    lon_deg: f64,
    alt_km: f64,
) -> Result<(f64, f64, f64), FrameTransformError> {
    validate_geodetic_degrees_km(lat_deg, lon_deg, alt_km)?;
    validate_tuple3(
        "itrs_position_km",
        geodetic_to_itrs_unchecked(lat_deg, lon_deg, alt_km),
    )
}

fn geodetic_to_itrs_unchecked(lat_deg: f64, lon_deg: f64, alt_km: f64) -> (f64, f64, f64) {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();

    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lon = lon.sin();
    let cos_lon = lon.cos();

    let n = WGS84_A_KM / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();

    let x = (n + alt_km) * cos_lat * cos_lon;
    let y = (n + alt_km) * cos_lat * sin_lon;
    let z = (n * (1.0 - WGS84_E2) + alt_km) * sin_lat;

    (x, y, z)
}

/// Compute station ECEF/ITRS position directly in AU.
/// Matches Skyfield's Geoid.latlon which works in AU from the start,
/// avoiding the km->AU_KM division that introduces 1 ULP rounding.
fn geodetic_to_itrs_au(lat_deg: f64, lon_deg: f64, alt_km: f64) -> [f64; 3] {
    let lat = lat_deg * TAU / 360.0;
    let lon = lon_deg * TAU / 360.0;

    let sinphi = lat.sin();
    let cosphi = lat.cos();

    let radius_au = WGS84_A_KM / AU_KM;
    let elevation_au = alt_km / AU_KM;

    let omf2 = (1.0 - WGS84_F) * (1.0 - WGS84_F);
    let c = 1.0 / (cosphi * cosphi + sinphi * sinphi * omf2).sqrt();
    let s = omf2 * c;

    let radius_xy = radius_au * c;
    let xy = (radius_xy + elevation_au) * cosphi;
    let x = xy * lon.cos();
    let y = xy * lon.sin();

    let radius_z = radius_au * s;
    let z = (radius_z + elevation_au) * sinphi;

    [x, y, z]
}

/// Build the ECEF->ENU rotation matrix for a given geodetic position.
fn ecef_to_enu_matrix(lat_deg: f64, lon_deg: f64) -> Mat3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();

    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lon = lon.sin();
    let cos_lon = lon.cos();

    // ENU rotation matrix:
    // E = [-sin(lon),           cos(lon),          0       ]
    // N = [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)]
    // U = [ cos(lat)*cos(lon),  cos(lat)*sin(lon), sin(lat)]
    [
        [-sin_lon, cos_lon, 0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ]
}

/// Compute topocentric az/el/range from a ground station to a satellite.
///
/// Returns (azimuth_deg, elevation_deg, range_km).
pub fn gcrs_to_topocentric_compute(
    sat_gcrs_km: [f64; 3],
    station: &GeodeticStationKm,
    ts: &TimeScales,
    skyfield_compat: bool,
) -> Result<(f64, f64, f64), FrameTransformError> {
    validate_vec3("sat_gcrs_km", &sat_gcrs_km)?;
    validate_geodetic_degrees_km(
        station.latitude_deg,
        station.longitude_deg,
        station.altitude_km,
    )?;
    validate_time_scales(ts)?;
    validate_tuple3(
        "topocentric",
        gcrs_to_topocentric_compute_unchecked(sat_gcrs_km, station, ts, skyfield_compat),
    )
}

fn gcrs_to_topocentric_compute_unchecked(
    sat_gcrs_km: [f64; 3],
    station: &GeodeticStationKm,
    ts: &TimeScales,
    skyfield_compat: bool,
) -> (f64, f64, f64) {
    let [sat_x, sat_y, sat_z] = sat_gcrs_km;
    let station_lat_deg = station.latitude_deg;
    let station_lon_deg = station.longitude_deg;
    let station_alt_km = station.altitude_km;
    if skyfield_compat {
        return gcrs_to_topocentric_skyfield(
            sat_x,
            sat_y,
            sat_z,
            station_lat_deg,
            station_lon_deg,
            station_alt_km,
            ts,
        );
    }

    // Standard path: GCRS->ITRS->subtract->ENU
    let (sat_itrs_x, sat_itrs_y, sat_itrs_z) =
        gcrs_to_itrs_compute_unchecked(sat_x, sat_y, sat_z, ts, false);

    let (stn_x, stn_y, stn_z) =
        geodetic_to_itrs_unchecked(station_lat_deg, station_lon_deg, station_alt_km);

    let dx = sat_itrs_x - stn_x;
    let dy = sat_itrs_y - stn_y;
    let dz = sat_itrs_z - stn_z;

    let enu_mat = ecef_to_enu_matrix(station_lat_deg, station_lon_deg);
    let enu = mat3_vec3_mul_unchecked(&enu_mat, &[dx, dy, dz]);
    let east = enu[0];
    let north = enu[1];
    let up = enu[2];

    // Range
    let range = (east * east + north * north + up * up).sqrt();

    // Elevation
    let elevation = (up / range).asin().to_degrees();

    // Azimuth (measured clockwise from north)
    let mut azimuth = east.atan2(north).to_degrees();
    if azimuth < 0.0 {
        azimuth += 360.0;
    }

    (azimuth, elevation, range)
}

/// Skyfield-compatible topocentric: stays in GCRS AU the entire time.
///
/// Replicates Skyfield's altaz computation:
/// 1. R_lat = rot_y(lat)[::-1]  (row-reversed Y rotation)
/// 2. R_latlon = mxm(R_lat, rot_z(-lon))
/// 3. R_full = mxm(R_latlon, itrs_rotation)
/// 4. station_gcrs_au = transpose(itrs_rotation) * station_itrs_au
/// 5. diff_au = sat_gcrs_au - station_gcrs_au
/// 6. enu_au = mxv(R_full, diff_au)
/// 7. to_spherical(enu_au) -> (range_au, elevation_rad, azimuth_rad)
fn gcrs_to_topocentric_skyfield(
    sat_x: f64,
    sat_y: f64,
    sat_z: f64,
    station_lat_deg: f64,
    station_lon_deg: f64,
    station_alt_km: f64,
    ts: &TimeScales,
) -> (f64, f64, f64) {
    let lat_rad = station_lat_deg * TAU / 360.0;
    let lon_rad = station_lon_deg * TAU / 360.0;

    // Build R_lat = rot_y(lat)[::-1]  (rows reversed)
    let cy = lat_rad.cos();
    let sy = lat_rad.sin();
    // rot_y(lat) = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    // [::-1] reverses rows: [[-sy, 0, cy], [0, 1, 0], [cy, 0, sy]]
    let r_lat: Mat3 = [[-sy, 0.0, cy], [0.0, 1.0, 0.0], [cy, 0.0, sy]];

    // R_latlon = mxm(R_lat, rot_z(-lon))
    let rz_neg_lon = build_rot_z(-lon_rad);
    let r_latlon = inline_rxr(&r_lat, &rz_neg_lon);

    // R_full = mxm(R_latlon, itrs_rotation)
    let r_itrs = gcrs_to_itrs_matrix_unchecked(ts);
    let r_full = inline_rxr(&r_latlon, &r_itrs);

    // Station ITRS position directly in AU, matching Skyfield's Geoid.latlon
    // which computes in AU from the start (not km then / AU_KM).
    let stn_itrs_au = geodetic_to_itrs_au(station_lat_deg, station_lon_deg, station_alt_km);

    // Station GCRS AU = transpose(R_itrs) * station_itrs_au
    let r_itrs_t = inline_tr(&r_itrs);
    let stn_gcrs_au = mat3_vec3_mul_unchecked(&r_itrs_t, &stn_itrs_au);

    // Satellite GCRS in AU
    let sat_au = [sat_x / AU_KM, sat_y / AU_KM, sat_z / AU_KM];

    // Difference vector in GCRS AU
    let diff_au = [
        sat_au[0] - stn_gcrs_au[0],
        sat_au[1] - stn_gcrs_au[1],
        sat_au[2] - stn_gcrs_au[2],
    ];

    // Rotate to ENU-ish frame: mxv(R_full, diff_au)
    let enu_au = mat3_vec3_mul_unchecked(&r_full, &diff_au);

    // to_spherical: r, theta (elevation), phi (azimuth)
    let ex = enu_au[0];
    let ey = enu_au[1];
    let ez = enu_au[2];

    let r_au = (ex * ex + ey * ey + ez * ez).sqrt();
    let elevation_rad = ez.atan2((ex * ex + ey * ey).sqrt());
    let mut azimuth_rad = ey.atan2(ex) % TAU;
    if azimuth_rad < 0.0 {
        azimuth_rad += TAU;
    }

    let range_km = r_au * AU_KM;
    let elevation_deg = elevation_rad * 360.0 / TAU;
    let azimuth_deg = azimuth_rad * 360.0 / TAU;

    (azimuth_deg, elevation_deg, range_km)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::time::scales::TimeScales;

    fn assert_mat3_bits_eq(actual: &Mat3, expected: &Mat3) {
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    actual[i][j].to_bits(),
                    expected[i][j].to_bits(),
                    "matrix[{i}][{j}]: {} vs {}",
                    actual[i][j],
                    expected[i][j]
                );
            }
        }
    }

    fn assert_vec3_bits_eq(actual: [f64; 3], expected: [f64; 3]) {
        for i in 0..3 {
            assert_eq!(
                actual[i].to_bits(),
                expected[i].to_bits(),
                "vector[{i}]: {} vs {}",
                actual[i],
                expected[i]
            );
        }
    }

    #[test]
    fn itrs_to_gcrs_inverts_gcrs_to_itrs() {
        // On a real epoch and a real-magnitude ECI vector, ITRS->GCRS recovers
        // the GCRS->ITRS input bit-for-bit on the plain (non-Skyfield) km path:
        // the two directions must be exact transposes, not just approximately so.
        let ts = TimeScales::from_utc(2020, 6, 24, 12, 34, 56.0).expect("valid UTC instant");
        let (x, y, z) = (4321.0_f64, -5678.0, 3210.0);

        let (ix, iy, iz) =
            gcrs_to_itrs_compute(x, y, z, &ts, false).expect("valid frame transform");
        // The rotation actually moved the vector (it is not a no-op).
        assert!(((ix - x).abs() + (iy - y).abs() + (iz - z).abs()) > 100.0);

        let (bx, by, bz) = itrs_to_gcrs_compute(ix, iy, iz, &ts).expect("valid frame transform");
        assert!((bx - x).abs() < 1e-9, "x {bx} vs {x}");
        assert!((by - y).abs() < 1e-9, "y {by} vs {y}");
        assert!((bz - z).abs() < 1e-9, "z {bz} vs {z}");

        // Magnitude is preserved by the rotation.
        let n0 = (x * x + y * y + z * z).sqrt();
        let n1 = (ix * ix + iy * iy + iz * iz).sqrt();
        assert!((n0 - n1).abs() < 1e-9);
    }

    #[test]
    fn polar_motion_matrix_matches_documented_convention() {
        let pole = PolarMotion::from_arcseconds(0.25, -0.35).expect("valid polar motion");
        let cx = pole.xp_rad.cos();
        let sx = pole.xp_rad.sin();
        let cy = pole.yp_rad.cos();
        let sy = pole.yp_rad.sin();

        let expected = [
            [cx, sx * sy, sx * cy],
            [0.0, cy, -sy],
            [-sx, cx * sy, cx * cy],
        ];
        let got = polar_motion_matrix(pole).expect("valid polar motion matrix");
        assert_mat3_bits_eq(&got, &expected);

        let small_angle = [
            [1.0, 0.0, pole.xp_rad],
            [0.0, 1.0, -pole.yp_rad],
            [-pole.xp_rad, pole.yp_rad, 1.0],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (got[i][j] - small_angle[i][j]).abs() < 1.0e-11,
                    "matrix[{i}][{j}] {} vs small-angle {}",
                    got[i][j],
                    small_angle[i][j]
                );
            }
        }
    }

    #[test]
    fn gcrs_to_itrs_with_polar_motion_premultiplies_legacy_rotation() {
        let ts = TimeScales::from_utc(2020, 6, 24, 12, 34, 56.0).expect("valid UTC instant");
        let pole = PolarMotion::from_arcseconds(0.18, -0.24).expect("valid polar motion");
        let legacy = gcrs_to_itrs_matrix(&ts).expect("valid frame transform");
        let expected = inline_rxr(
            &polar_motion_matrix(pole).expect("valid polar motion matrix"),
            &legacy,
        );
        let got = gcrs_to_itrs_matrix_with_polar_motion(&ts, pole).expect("valid frame transform");

        assert_mat3_bits_eq(&got, &expected);

        let pos = [4321.0_f64, -5678.0, 3210.0];
        let actual_vec =
            gcrs_to_itrs_compute_with_polar_motion(pos[0], pos[1], pos[2], &ts, false, pole)
                .expect("valid frame transform");
        let expected_vec = mat3_vec3_mul(&expected, &pos).expect("finite matrix-vector product");
        assert_vec3_bits_eq([actual_vec.0, actual_vec.1, actual_vec.2], expected_vec);

        let legacy_vec =
            gcrs_to_itrs_compute(pos[0], pos[1], pos[2], &ts, false).expect("valid transform");
        let delta = (actual_vec.0 - legacy_vec.0).abs()
            + (actual_vec.1 - legacy_vec.1).abs()
            + (actual_vec.2 - legacy_vec.2).abs();
        assert!(
            delta > 1.0e-4,
            "nonzero polar motion should move the vector"
        );

        let inverse =
            itrs_to_gcrs_matrix_with_polar_motion(&ts, pole).expect("valid frame transform");
        assert_mat3_bits_eq(&inverse, &inline_tr(&got));
    }

    #[test]
    fn zero_polar_motion_matches_legacy_transform_bits() {
        let ts = TimeScales::from_utc(2020, 6, 24, 12, 34, 56.0).expect("valid UTC instant");
        let legacy = gcrs_to_itrs_matrix(&ts).expect("valid frame transform");
        let zero = gcrs_to_itrs_matrix_with_polar_motion(&ts, PolarMotion::ZERO)
            .expect("valid frame transform");
        assert_mat3_bits_eq(&zero, &legacy);

        let mean_legacy = mean_of_date_to_itrs_matrix(&ts).expect("valid frame transform");
        let mean_zero = mean_of_date_to_itrs_matrix_with_polar_motion(&ts, PolarMotion::ZERO)
            .expect("valid frame transform");
        assert_mat3_bits_eq(&mean_zero, &mean_legacy);

        let pos = [4321.0_f64, -5678.0, 3210.0];
        for skyfield_compat in [false, true] {
            let legacy_vec = gcrs_to_itrs_compute(pos[0], pos[1], pos[2], &ts, skyfield_compat)
                .expect("valid frame transform");
            let zero_vec = gcrs_to_itrs_compute_with_polar_motion(
                pos[0],
                pos[1],
                pos[2],
                &ts,
                skyfield_compat,
                PolarMotion::ZERO,
            )
            .expect("valid frame transform");
            assert_vec3_bits_eq(
                [zero_vec.0, zero_vec.1, zero_vec.2],
                [legacy_vec.0, legacy_vec.1, legacy_vec.2],
            );
        }

        let legacy_back =
            itrs_to_gcrs_compute(pos[0], pos[1], pos[2], &ts).expect("valid frame transform");
        let zero_back =
            itrs_to_gcrs_compute_with_polar_motion(pos[0], pos[1], pos[2], &ts, PolarMotion::ZERO)
                .expect("valid frame transform");
        assert_vec3_bits_eq(
            [zero_back.0, zero_back.1, zero_back.2],
            [legacy_back.0, legacy_back.1, legacy_back.2],
        );
    }

    #[test]
    fn frame_transforms_reject_nonfinite_time() {
        let mut ts = TimeScales::from_utc(2020, 6, 24, 12, 34, 56.0).expect("valid UTC instant");
        ts.jd_tt = f64::NAN;

        assert!(greenwich_mean_sidereal_time_radians(&ts).is_err());
        assert!(gcrs_to_itrs_matrix(&ts).is_err());
        assert!(itrs_to_gcrs_compute(1.0, 2.0, 3.0, &ts).is_err());
    }

    #[test]
    fn frame_transforms_reject_nonfinite_pole_coordinates() {
        let ts = TimeScales::from_utc(2020, 6, 24, 12, 34, 56.0).expect("valid UTC instant");
        assert!(PolarMotion::from_radians(f64::NAN, 0.0).is_err());
        assert!(PolarMotion::from_arcseconds(0.0, f64::INFINITY).is_err());

        let pole = PolarMotion {
            xp_rad: f64::NAN,
            yp_rad: 0.0,
        };
        assert!(polar_motion_matrix(pole).is_err());
        assert!(gcrs_to_itrs_matrix_with_polar_motion(&ts, pole).is_err());
    }

    #[test]
    fn frame_transforms_reject_nonfinite_vectors() {
        let ts = TimeScales::from_utc(2020, 6, 24, 12, 34, 56.0).expect("valid UTC instant");
        let bad_state = TemeStateKm {
            position_km: [1.0, f64::NAN, 3.0],
            velocity_km_s: [0.1, 0.2, 0.3],
        };
        assert!(teme_to_gcrs_compute(&bad_state, &ts, false).is_err());
        assert!(gcrs_to_itrs_compute(1.0, f64::INFINITY, 3.0, &ts, false).is_err());
        assert!(itrs_to_gcrs_compute(1.0, 2.0, f64::NEG_INFINITY, &ts).is_err());
    }

    #[test]
    fn validated_frame_transform_preserves_valid_bits() {
        let ts = TimeScales::from_utc(2020, 6, 24, 12, 34, 56.0).expect("valid UTC instant");
        let pos = [4321.0_f64, -5678.0, 3210.0];
        let expected = gcrs_to_itrs_compute_unchecked(pos[0], pos[1], pos[2], &ts, true);
        let got =
            gcrs_to_itrs_compute(pos[0], pos[1], pos[2], &ts, true).expect("valid frame transform");
        assert_vec3_bits_eq([got.0, got.1, got.2], [expected.0, expected.1, expected.2]);
    }

    #[test]
    fn geodetic_transforms_reject_invalid_coordinates() {
        assert!(itrs_to_geodetic_compute(f64::NAN, 0.0, 0.0).is_err());
        assert!(geodetic_from_ecef_proj(0.0, f64::INFINITY, 0.0).is_err());
        assert!(geodetic_to_itrs(90.000_001, 0.0, 0.0).is_err());
        assert!(geodetic_to_itrs(0.0, -180.000_001, 0.0).is_err());
        assert!(geodetic_to_itrs(0.0, 0.0, f64::NAN).is_err());
    }

    #[test]
    fn topocentric_transform_rejects_invalid_coordinates() {
        let ts = TimeScales::from_utc(2020, 6, 24, 12, 34, 56.0).expect("valid UTC instant");
        let station = GeodeticStationKm {
            latitude_deg: f64::NAN,
            longitude_deg: 0.0,
            altitude_km: 0.0,
        };
        assert!(gcrs_to_topocentric_compute([7000.0, 0.0, 0.0], &station, &ts, false).is_err());

        let station = GeodeticStationKm {
            latitude_deg: 0.0,
            longitude_deg: 181.0,
            altitude_km: 0.0,
        };
        assert!(gcrs_to_topocentric_compute([7000.0, 0.0, 0.0], &station, &ts, false).is_err());

        let station = GeodeticStationKm {
            latitude_deg: 0.0,
            longitude_deg: 0.0,
            altitude_km: 0.0,
        };
        assert!(
            gcrs_to_topocentric_compute([7000.0, f64::NAN, 0.0], &station, &ts, false).is_err()
        );
    }

    #[test]
    fn validated_geodetic_transform_preserves_valid_bits() {
        let (lat, lon, alt) = (51.4779, -0.0015, 0.046);
        let expected = geodetic_to_itrs_unchecked(lat, lon, alt);
        let got = geodetic_to_itrs(lat, lon, alt).expect("valid geodetic coordinates");
        assert_eq!(got.0.to_bits(), expected.0.to_bits());
        assert_eq!(got.1.to_bits(), expected.1.to_bits());
        assert_eq!(got.2.to_bits(), expected.2.to_bits());

        let expected = itrs_to_geodetic_compute_unchecked(got.0, got.1, got.2);
        let roundtrip =
            itrs_to_geodetic_compute(got.0, got.1, got.2).expect("valid ITRS coordinates");
        assert_eq!(roundtrip.0.to_bits(), expected.0.to_bits());
        assert_eq!(roundtrip.1.to_bits(), expected.1.to_bits());
        assert_eq!(roundtrip.2.to_bits(), expected.2.to_bits());
    }

    #[test]
    fn sidereal_time_wrappers_are_in_range_and_consistent() {
        let ts = TimeScales::from_utc(2020, 6, 24, 12, 34, 56.0).expect("valid UTC instant");
        let gmst = greenwich_mean_sidereal_time_radians(&ts).expect("valid sidereal time");
        let gast = greenwich_apparent_sidereal_time_radians(&ts).expect("valid sidereal time");

        // Both land in [0, 2pi).
        assert!((0.0..TAU).contains(&gmst), "gmst {gmst}");
        assert!((0.0..TAU).contains(&gast), "gast {gast}");

        // The equation of the equinoxes is a small (sub-arcminute) offset, so the
        // apparent and mean sidereal times stay close (handle the seam at 2pi).
        let diff = (gast - gmst).rem_euclid(TAU);
        let eq_eq = diff.min(TAU - diff);
        assert!(eq_eq < 1.0e-3, "equation of equinoxes too large: {eq_eq}");

        // The mean wrapper equals the underlying private hours computation exactly.
        let gmst_hours = sidereal_time_hours(ts.jd_whole, ts.ut1_fraction, ts.tdb_fraction);
        assert_eq!(gmst, gmst_hours / 24.0 * TAU);
    }
}
