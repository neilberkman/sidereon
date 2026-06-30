//! Broadcast-ephemeris orbit and clock evaluation (GPS LNAV, Galileo I/NAV,
//! BeiDou D1/D2).
//!
//! Evaluates a broadcast navigation message into an ECEF satellite position and
//! a satellite clock offset, by the standard Keplerian construction of
//! IS-GPS-200 (Section 20.3.3.4.3.1, Table 20-IV) and the equivalent Galileo OS
//! and BeiDou SIS ICD sections. The constellations share the algorithm and
//! differ in the gravitational constant, the Earth-rotation rate, and the
//! relativistic clock constant ([`ConstellationConstants`]). BeiDou
//! geostationary satellites additionally take a custom-frame-to-ECEF rotation
//! (the `is_geo` path of [`satellite_position_ecef`]); GPS, Galileo, and BeiDou
//! MEO/IGSO satellites use the direct rotation.
//!
//! This is a 0-ULP parity target: the operation order reproduces the canonical
//! reference recipe (`parity/generator/broadcast_eval.py`) bit-for-bit. The
//! transcendentals are the libm scalar `sin`/`cos`/`sqrt`/`atan2` (matching the
//! recipe's CPython `math` calls on the pinned Apple-libm target); separate
//! `.sin()` and `.cos()` calls are used deliberately (never `.sin_cos()`, whose
//! fused evaluation can differ in the last bit). Integer powers are explicit
//! repeated multiplies and there is no fused multiply-add (Rust does not
//! auto-contract `a * b + c`).

use crate::astro::constants::models::broadcast::{
    BEIDOU_OMEGA_E_RAD_S, GALILEO_BEIDOU_DTR_F, GALILEO_GM_M3_S2, GPS_DTR_F,
    GPS_GALILEO_OMEGA_E_RAD_S, GPS_GM_M3_S2,
};
use crate::error::{Error, Result};
use crate::frame::{FrameValueError, ItrfPositionM};

/// Half a week, the fold threshold for a time difference against `toe`/`toc`.
pub use crate::constants::HALF_WEEK_S;
/// Seconds in one GPS/Galileo week.
pub use crate::constants::SECONDS_PER_WEEK;

/// Eccentric-anomaly fixed-point convergence threshold (radians).
pub const KEPLER_TOL: f64 = 1.0e-12;
/// Maximum eccentric-anomaly fixed-point iterations.
pub const KEPLER_MAX_ITER: usize = 30;
/// Satellite-clock time-argument refinement count (RTKLIB `eph2clk` convention).
pub const CLOCK_MAX_ITER: usize = 2;

/// Per-constellation physical constants used by the broadcast evaluation.
///
/// The literals match the values the broadcast reference recipe and the `rinex`
/// crate use, so the Python and Rust sides share identical `f64` bit patterns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstellationConstants {
    /// Gravitational constant GM (m^3 / s^2).
    pub gm_m3_s2: f64,
    /// Earth rotation rate used by the longitude-of-node (Sagnac) term (rad/s).
    pub omega_e_rad_s: f64,
    /// Relativistic clock constant `F = -2 * sqrt(GM) / c^2` (s / sqrt(m)).
    pub dtr_f: f64,
}

impl ConstellationConstants {
    /// GPS constants (IS-GPS-200).
    pub const GPS: Self = Self {
        gm_m3_s2: GPS_GM_M3_S2,
        omega_e_rad_s: GPS_GALILEO_OMEGA_E_RAD_S,
        dtr_f: GPS_DTR_F,
    };
    /// Galileo constants (OS SIS ICD); shares the GPS rotation rate.
    pub const GALILEO: Self = Self {
        gm_m3_s2: GALILEO_GM_M3_S2,
        omega_e_rad_s: GPS_GALILEO_OMEGA_E_RAD_S,
        dtr_f: GALILEO_BEIDOU_DTR_F,
    };
    /// BeiDou constants (BDS-SIS-ICD); its own Earth-rotation rate.
    pub const BEIDOU: Self = Self {
        gm_m3_s2: GALILEO_GM_M3_S2,
        omega_e_rad_s: BEIDOU_OMEGA_E_RAD_S,
        dtr_f: GALILEO_BEIDOU_DTR_F,
    };
}

/// Broadcast Keplerian orbital elements (SI units; angles in radians; `toe_sow`
/// in seconds of the constellation's week).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeplerianElements {
    /// Square root of the semi-major axis (sqrt(m)).
    pub sqrt_a: f64,
    /// Eccentricity (dimensionless).
    pub e: f64,
    /// Mean anomaly at reference time (rad).
    pub m0: f64,
    /// Mean motion difference from computed value (rad/s).
    pub delta_n: f64,
    /// Longitude of ascending node at weekly epoch (rad).
    pub omega0: f64,
    /// Inclination at reference time (rad).
    pub i0: f64,
    /// Argument of perigee (rad).
    pub omega: f64,
    /// Rate of right ascension (rad/s).
    pub omega_dot: f64,
    /// Rate of inclination (rad/s).
    pub idot: f64,
    /// Latitude argument cosine correction (rad).
    pub cuc: f64,
    /// Latitude argument sine correction (rad).
    pub cus: f64,
    /// Orbit radius cosine correction (m).
    pub crc: f64,
    /// Orbit radius sine correction (m).
    pub crs: f64,
    /// Inclination cosine correction (rad).
    pub cic: f64,
    /// Inclination sine correction (rad).
    pub cis: f64,
    /// Ephemeris reference time, seconds of week.
    pub toe_sow: f64,
}

/// Broadcast satellite-clock polynomial about `toc_sow`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockPolynomial {
    /// Clock bias (s).
    pub af0: f64,
    /// Clock drift (s/s).
    pub af1: f64,
    /// Clock drift rate (s/s^2).
    pub af2: f64,
    /// Clock reference time, seconds of week.
    pub toc_sow: f64,
}

/// A solved eccentric anomaly and the iteration count that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EccentricAnomaly {
    /// Eccentric anomaly (rad).
    pub value: f64,
    /// Fixed-point iterations performed.
    pub iterations: usize,
}

/// The full intermediate substrate of a broadcast orbit evaluation.
///
/// Every field is exposed so a 0-ULP parity test can localize a mismatch to a
/// single operation. [`OrbitState::position`] returns the ECEF position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitState {
    /// Semi-major axis (m).
    pub a: f64,
    /// Computed mean motion (rad/s).
    pub n0: f64,
    /// Corrected mean motion (rad/s).
    pub n: f64,
    /// Time from ephemeris reference epoch, half-week folded (s).
    pub tk: f64,
    /// Mean anomaly (rad).
    pub mk: f64,
    /// Eccentric anomaly (rad).
    pub eccentric_anomaly: f64,
    /// Number of Kepler iterations.
    pub kepler_iterations: usize,
    /// sin(E).
    pub sin_e: f64,
    /// cos(E).
    pub cos_e: f64,
    /// True anomaly (rad).
    pub nu: f64,
    /// Argument of latitude before correction (rad).
    pub phi: f64,
    /// sin(2*phi).
    pub s2: f64,
    /// cos(2*phi).
    pub c2: f64,
    /// Argument-of-latitude correction (rad).
    pub du: f64,
    /// Radius correction (m).
    pub dr: f64,
    /// Inclination correction (rad).
    pub di: f64,
    /// Corrected argument of latitude (rad).
    pub u: f64,
    /// Corrected radius (m).
    pub r: f64,
    /// Corrected inclination (rad).
    pub i: f64,
    /// Orbital-plane x (m).
    pub xp: f64,
    /// Orbital-plane y (m).
    pub yp: f64,
    /// Corrected longitude of ascending node (rad).
    pub omega_k: f64,
    /// ECEF x (m).
    pub x_m: f64,
    /// ECEF y (m).
    pub y_m: f64,
    /// ECEF z (m).
    pub z_m: f64,
}

impl OrbitState {
    /// The Earth-fixed (ITRF/ECEF) satellite position in meters.
    pub const fn position(&self) -> core::result::Result<ItrfPositionM, FrameValueError> {
        ItrfPositionM::new(self.x_m, self.y_m, self.z_m)
    }
}

/// The satellite clock offset, split into its components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockOffset {
    /// Polynomial term (s).
    pub dt_clock_poly_s: f64,
    /// Relativistic eccentricity term (s).
    pub dt_rel_s: f64,
    /// Group delay subtracted for the single-frequency user (s).
    pub tgd_s: f64,
    /// Total satellite clock offset (s).
    pub dt_clock_total_s: f64,
}

/// Broadcast relativistic satellite-clock correction, seconds.
///
/// Evaluates the periodic eccentric-orbit term
/// `F * e * sqrt(A) * sin(E)` from IS-GPS-200 and the equivalent Galileo/BeiDou
/// broadcast-clock models. Use [`ConstellationConstants::dtr_f`] for `F` when
/// evaluating a full broadcast record.
pub fn relativistic_clock_correction_s(
    dtr_f_s_sqrt_m: f64,
    eccentricity: f64,
    sqrt_a_m_sqrt: f64,
    eccentric_anomaly_sin: f64,
) -> Result<f64> {
    validate_finite(dtr_f_s_sqrt_m, "dtr_f_s_sqrt_m")?;
    validate_eccentricity(eccentricity)?;
    validate_positive(sqrt_a_m_sqrt, "sqrt_a_m_sqrt")?;
    validate_finite(eccentric_anomaly_sin, "eccentric_anomaly_sin")?;
    let correction = relativistic_clock_correction_s_unchecked(
        dtr_f_s_sqrt_m,
        eccentricity,
        sqrt_a_m_sqrt,
        eccentric_anomaly_sin,
    );
    validate_finite(correction, "relativistic_clock_correction_s")?;
    Ok(correction)
}

#[inline]
pub(crate) fn relativistic_clock_correction_s_unchecked(
    dtr_f_s_sqrt_m: f64,
    eccentricity: f64,
    sqrt_a_m_sqrt: f64,
    eccentric_anomaly_sin: f64,
) -> f64 {
    dtr_f_s_sqrt_m * eccentricity * sqrt_a_m_sqrt * eccentric_anomaly_sin
}

/// Time difference `t - t_ref` (seconds), folded into the +/- half-week range.
///
/// Used for both `tk = t - toe` (orbit) and `t - toc` (clock); the fold handles
/// a query that straddles the week rollover relative to the reference.
pub fn time_from_reference_s(t_sow_s: f64, t_ref_sow_s: f64) -> f64 {
    let mut dt = t_sow_s - t_ref_sow_s;
    if dt > HALF_WEEK_S {
        dt -= SECONDS_PER_WEEK;
    }
    if dt < -HALF_WEEK_S {
        dt += SECONDS_PER_WEEK;
    }
    dt
}

/// Solve Kepler's equation by fixed-point iteration `E = M + e*sin(E)`, seeded
/// at `E = M`; stops on `|dE| <= KEPLER_TOL` or after `KEPLER_MAX_ITER` steps.
pub fn eccentric_anomaly(mean_anomaly_rad: f64, eccentricity: f64) -> Result<EccentricAnomaly> {
    validate_finite(mean_anomaly_rad, "mean_anomaly_rad")?;
    validate_eccentricity(eccentricity)?;

    Ok(eccentric_anomaly_unchecked(mean_anomaly_rad, eccentricity))
}

pub(crate) fn eccentric_anomaly_unchecked(
    mean_anomaly_rad: f64,
    eccentricity: f64,
) -> EccentricAnomaly {
    let mut e_k = mean_anomaly_rad;
    let mut iterations = 0usize;
    while iterations < KEPLER_MAX_ITER {
        let e_prev = e_k;
        e_k = mean_anomaly_rad + eccentricity * e_prev.sin();
        iterations += 1;
        let delta = (e_k - e_prev).abs();
        if delta <= KEPLER_TOL {
            break;
        }
    }
    EccentricAnomaly {
        value: e_k,
        iterations,
    }
}

/// Evaluate the broadcast Keplerian orbit at `t_sow_s` (seconds of week).
///
/// `is_geo` selects the BeiDou geostationary path (the node omits the
/// Earth-rotation-during-`tk` term and the position is rotated to ECEF by
/// `Rz(omega_e*tk) . Rx(-5deg)`); GPS, Galileo, and BeiDou MEO/IGSO use `false`.
/// The statement order reproduces `broadcast_eval.satellite_position_ecef`.
pub fn satellite_position_ecef(
    elements: &KeplerianElements,
    consts: &ConstellationConstants,
    t_sow_s: f64,
    is_geo: bool,
) -> Result<OrbitState> {
    validate_elements(elements)?;
    validate_constants(consts)?;
    validate_finite(t_sow_s, "t_sow_s")?;

    let state = satellite_position_ecef_unchecked(elements, consts, t_sow_s, is_geo);
    validate_orbit_state(&state)?;
    Ok(state)
}

pub(crate) fn satellite_position_ecef_unchecked(
    elements: &KeplerianElements,
    consts: &ConstellationConstants,
    t_sow_s: f64,
    is_geo: bool,
) -> OrbitState {
    let sqrt_a = elements.sqrt_a;
    let e = elements.e;
    let gm = consts.gm_m3_s2;
    let omega_e = consts.omega_e_rad_s;

    // 1. Semi-major axis and mean motion. a^3 as an explicit multiply chain.
    let a = sqrt_a * sqrt_a;
    let n0 = (gm / (a * a * a)).sqrt();
    let n = n0 + elements.delta_n;

    // 2. Time from ephemeris reference epoch (half-week folded).
    let tk = time_from_reference_s(t_sow_s, elements.toe_sow);

    // 3. Mean anomaly and eccentric anomaly.
    let mk = elements.m0 + n * tk;
    let kepler = eccentric_anomaly_unchecked(mk, e);
    let ecc_anom = kepler.value;
    let sin_e = ecc_anom.sin();
    let cos_e = ecc_anom.cos();

    // 4. True anomaly (atan2 form) and argument of latitude.
    let e2 = e * e;
    let nu = ((1.0 - e2).sqrt() * sin_e).atan2(cos_e - e);
    let phi = nu + elements.omega;

    // 5. Second-harmonic corrections (sine term first).
    let two_phi = 2.0 * phi;
    let s2 = two_phi.sin();
    let c2 = two_phi.cos();
    let du = elements.cus * s2 + elements.cuc * c2;
    let dr = elements.crs * s2 + elements.crc * c2;
    let di = elements.cis * s2 + elements.cic * c2;

    // 6. Corrected argument of latitude, radius, inclination.
    let u = phi + du;
    let r = a * (1.0 - e * cos_e) + dr;
    let i = elements.i0 + di + elements.idot * tk;

    // 7. Position in the orbital plane.
    let xp = r * u.cos();
    let yp = r * u.sin();

    // 8. Corrected longitude of ascending node. The BeiDou GEO node omits the
    // Earth-rotation-during-tk term (applied by the final rotation instead).
    let omega_k = if is_geo {
        elements.omega0 + elements.omega_dot * tk - omega_e * elements.toe_sow
    } else {
        elements.omega0 + (elements.omega_dot - omega_e) * tk - omega_e * elements.toe_sow
    };

    // 9. Coordinates in the (custom) frame from the node rotation.
    let sin_o = omega_k.sin();
    let cos_o = omega_k.cos();
    let sin_i = i.sin();
    let cos_i = i.cos();
    let xg = xp * cos_o - yp * cos_i * sin_o;
    let yg = xp * sin_o + yp * cos_i * cos_o;
    let zg = yp * sin_i;

    // 10. Earth-fixed coordinates. The standard path is the identity; the BeiDou
    // GEO path applies Rz(omega_e*tk) . Rx(-5deg) (BDS-SIS-ICD).
    let (x, y, z) = if is_geo {
        let deg5 = 5.0_f64.to_radians();
        let cos_phi = deg5.cos();
        let sin_phi = -deg5.sin();
        let z_ang = omega_e * tk;
        let cos_z = z_ang.cos();
        let sin_z = z_ang.sin();
        let yr = yg * cos_phi + zg * sin_phi;
        let zr = -yg * sin_phi + zg * cos_phi;
        (xg * cos_z + yr * sin_z, -xg * sin_z + yr * cos_z, zr)
    } else {
        (xg, yg, zg)
    };

    OrbitState {
        a,
        n0,
        n,
        tk,
        mk,
        eccentric_anomaly: ecc_anom,
        kepler_iterations: kepler.iterations,
        sin_e,
        cos_e,
        nu,
        phi,
        s2,
        c2,
        du,
        dr,
        di,
        u,
        r,
        i,
        xp,
        yp,
        omega_k,
        x_m: x,
        y_m: y,
        z_m: z,
    }
}

/// Evaluate the broadcast satellite clock offset (seconds).
///
/// `sin_e` is the eccentric-anomaly sine from the position evaluation at the
/// same instant; `tgd_s` is the single-frequency group delay. The statement
/// order reproduces `broadcast_eval.satellite_clock_offset_s`.
pub fn satellite_clock_offset_s(
    clock: &ClockPolynomial,
    consts: &ConstellationConstants,
    elements: &KeplerianElements,
    sin_e: f64,
    t_sow_s: f64,
    tgd_s: f64,
) -> Result<ClockOffset> {
    validate_clock(clock)?;
    validate_constants(consts)?;
    validate_elements(elements)?;
    validate_finite(sin_e, "sin_e")?;
    validate_finite(t_sow_s, "t_sow_s")?;
    validate_finite(tgd_s, "tgd_s")?;

    let offset = satellite_clock_offset_s_unchecked(clock, consts, elements, sin_e, t_sow_s, tgd_s);
    validate_clock_offset(&offset)?;
    Ok(offset)
}

pub(crate) fn satellite_clock_offset_s_unchecked(
    clock: &ClockPolynomial,
    consts: &ConstellationConstants,
    elements: &KeplerianElements,
    sin_e: f64,
    t_sow_s: f64,
    tgd_s: f64,
) -> ClockOffset {
    let af0 = clock.af0;
    let af1 = clock.af1;
    let af2 = clock.af2;

    // Time from clock reference, folded; then refine out the SV clock itself.
    let dt0 = time_from_reference_s(t_sow_s, clock.toc_sow);
    let mut dt = dt0;
    let mut refine = 0usize;
    while refine < CLOCK_MAX_ITER {
        dt = dt0 - (af0 + af1 * dt + af2 * dt * dt);
        refine += 1;
    }
    let dt_poly = af0 + af1 * dt + af2 * dt * dt;

    // Relativistic eccentricity term (sqrt_a is the broadcast sqrt(A)).
    let dt_rel =
        relativistic_clock_correction_s_unchecked(consts.dtr_f, elements.e, elements.sqrt_a, sin_e);

    let dt_total = dt_poly + dt_rel - tgd_s;

    ClockOffset {
        dt_clock_poly_s: dt_poly,
        dt_rel_s: dt_rel,
        tgd_s,
        dt_clock_total_s: dt_total,
    }
}

/// A satellite's broadcast orbit and clock evaluated together at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SatelliteState {
    /// The orbit evaluation (ECEF position and all intermediates).
    pub orbit: OrbitState,
    /// The clock offset evaluation.
    pub clock: ClockOffset,
}

/// Evaluate the broadcast orbit and clock at the same instant.
///
/// This is the intended public entry point: it solves the orbit and feeds that
/// solution's eccentric-anomaly sine into the clock's relativistic term, so a
/// caller cannot accidentally pair a clock evaluation with `sin(E)` from a
/// different epoch. [`satellite_position_ecef`] and [`satellite_clock_offset_s`]
/// remain available for component-level parity testing.
pub fn satellite_state(
    elements: &KeplerianElements,
    clock: &ClockPolynomial,
    consts: &ConstellationConstants,
    t_sow_s: f64,
    tgd_s: f64,
    is_geo: bool,
) -> Result<SatelliteState> {
    validate_elements(elements)?;
    validate_clock(clock)?;
    validate_constants(consts)?;
    validate_finite(t_sow_s, "t_sow_s")?;
    validate_finite(tgd_s, "tgd_s")?;

    let state = satellite_state_unchecked(elements, clock, consts, t_sow_s, tgd_s, is_geo);
    validate_orbit_state(&state.orbit)?;
    validate_clock_offset(&state.clock)?;
    Ok(state)
}

pub(crate) fn satellite_state_unchecked(
    elements: &KeplerianElements,
    clock: &ClockPolynomial,
    consts: &ConstellationConstants,
    t_sow_s: f64,
    tgd_s: f64,
    is_geo: bool,
) -> SatelliteState {
    let orbit = satellite_position_ecef_unchecked(elements, consts, t_sow_s, is_geo);
    let clock =
        satellite_clock_offset_s_unchecked(clock, consts, elements, orbit.sin_e, t_sow_s, tgd_s);
    SatelliteState { orbit, clock }
}

fn validate_elements(elements: &KeplerianElements) -> Result<()> {
    validate_positive(elements.sqrt_a, "elements.sqrt_a")?;
    validate_eccentricity(elements.e)?;
    validate_finite(elements.m0, "elements.m0")?;
    validate_finite(elements.delta_n, "elements.delta_n")?;
    validate_finite(elements.omega0, "elements.omega0")?;
    validate_finite(elements.i0, "elements.i0")?;
    validate_finite(elements.omega, "elements.omega")?;
    validate_finite(elements.omega_dot, "elements.omega_dot")?;
    validate_finite(elements.idot, "elements.idot")?;
    validate_finite(elements.cuc, "elements.cuc")?;
    validate_finite(elements.cus, "elements.cus")?;
    validate_finite(elements.crc, "elements.crc")?;
    validate_finite(elements.crs, "elements.crs")?;
    validate_finite(elements.cic, "elements.cic")?;
    validate_finite(elements.cis, "elements.cis")?;
    validate_sow(elements.toe_sow, "elements.toe_sow")
}

fn validate_clock(clock: &ClockPolynomial) -> Result<()> {
    validate_finite(clock.af0, "clock.af0")?;
    validate_finite(clock.af1, "clock.af1")?;
    validate_finite(clock.af2, "clock.af2")?;
    validate_sow(clock.toc_sow, "clock.toc_sow")
}

fn validate_constants(consts: &ConstellationConstants) -> Result<()> {
    validate_positive(consts.gm_m3_s2, "consts.gm_m3_s2")?;
    validate_finite(consts.omega_e_rad_s, "consts.omega_e_rad_s")?;
    validate_finite(consts.dtr_f, "consts.dtr_f")
}

fn validate_orbit_state(state: &OrbitState) -> Result<()> {
    validate_finite(state.a, "orbit.a")?;
    validate_finite(state.n0, "orbit.n0")?;
    validate_finite(state.n, "orbit.n")?;
    validate_finite(state.tk, "orbit.tk")?;
    validate_finite(state.mk, "orbit.mk")?;
    validate_finite(state.eccentric_anomaly, "orbit.eccentric_anomaly")?;
    validate_finite(state.sin_e, "orbit.sin_e")?;
    validate_finite(state.cos_e, "orbit.cos_e")?;
    validate_finite(state.nu, "orbit.nu")?;
    validate_finite(state.phi, "orbit.phi")?;
    validate_finite(state.s2, "orbit.s2")?;
    validate_finite(state.c2, "orbit.c2")?;
    validate_finite(state.du, "orbit.du")?;
    validate_finite(state.dr, "orbit.dr")?;
    validate_finite(state.di, "orbit.di")?;
    validate_finite(state.u, "orbit.u")?;
    validate_finite(state.r, "orbit.r")?;
    validate_finite(state.i, "orbit.i")?;
    validate_finite(state.xp, "orbit.xp")?;
    validate_finite(state.yp, "orbit.yp")?;
    validate_finite(state.omega_k, "orbit.omega_k")?;
    validate_finite(state.x_m, "orbit.x_m")?;
    validate_finite(state.y_m, "orbit.y_m")?;
    validate_finite(state.z_m, "orbit.z_m")
}

fn validate_clock_offset(clock: &ClockOffset) -> Result<()> {
    validate_finite(clock.dt_clock_poly_s, "clock.dt_clock_poly_s")?;
    validate_finite(clock.dt_rel_s, "clock.dt_rel_s")?;
    validate_finite(clock.tgd_s, "clock.tgd_s")?;
    validate_finite(clock.dt_clock_total_s, "clock.dt_clock_total_s")
}

fn validate_eccentricity(eccentricity: f64) -> Result<()> {
    validate_finite(eccentricity, "eccentricity")?;
    if (0.0..1.0).contains(&eccentricity) {
        Ok(())
    } else {
        Err(invalid_input("eccentricity", "out of range"))
    }
}

fn validate_sow(value: f64, field: &'static str) -> Result<()> {
    validate_finite(value, field)?;
    if (0.0..SECONDS_PER_WEEK).contains(&value) {
        Ok(())
    } else {
        Err(invalid_input(field, "out of range"))
    }
}

fn validate_positive(value: f64, field: &'static str) -> Result<()> {
    validate_finite(value, field)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "not positive"))
    }
}

fn validate_finite(value: f64, field: &'static str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "not finite"))
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> Error {
    Error::InvalidInput(format!("{field} {reason}"))
}

#[cfg(test)]
mod public_api_tests {
    use super::*;

    #[test]
    fn relativistic_clock_correction_exposes_broadcast_formula() {
        let dtr_f = ConstellationConstants::GPS.dtr_f;
        let eccentricity = 0.013_456_789;
        let sqrt_a = 5_153.795_477_5;
        let sin_e = -0.625;
        let got = relativistic_clock_correction_s(dtr_f, eccentricity, sqrt_a, sin_e)
            .expect("valid relativistic correction");
        let want = dtr_f * eccentricity * sqrt_a * sin_e;
        assert_eq!(got.to_bits(), want.to_bits());
    }

    #[test]
    fn relativistic_clock_correction_rejects_invalid_inputs() {
        assert!(relativistic_clock_correction_s(f64::NAN, 0.01, 5_153.0, 0.5).is_err());
        assert!(relativistic_clock_correction_s(
            ConstellationConstants::GPS.dtr_f,
            1.0,
            5_153.0,
            0.5
        )
        .is_err());
        assert!(
            relativistic_clock_correction_s(ConstellationConstants::GPS.dtr_f, 0.01, 0.0, 0.5)
                .is_err()
        );
    }
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
