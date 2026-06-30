//! Conjunction geometry and collision-probability (`Pc`) computation.
//!
//! Builds the orthonormal encounter frame from two objects' relative state,
//! projects the combined position covariance into the 2D encounter plane, and
//! integrates the collision probability over the hard-body disk with several
//! independent methods (Foster 2D equal-area, Foster 2D numerical, and Alfano
//! 2005). This is the authoritative implementation; the Elixir binding is a
//! thin marshaling and input-validation layer over it.
//!
//! All positions are in km, velocities in km/s, and covariances in km^2.

use crate::astro::covariance::{positive_semidefinite, symmetric};
use crate::astro::math::special::erf;
use crate::astro::math::vec3;
use crate::validate;
use std::f64::consts::PI;

/// Relative velocities below this (km/s) leave the encounter frame undefined.
const ZERO_REL_SPEED_EPS_KM_S: f64 = 1.0e-12;
/// Orthogonal miss distances below this (km) are treated as a collision course,
/// where the cross-track axis is degenerate and chosen arbitrarily.
const COLLISION_COURSE_EPS_KM: f64 = 1.0e-12;
/// On a collision course, pick the world axis least aligned with `y_hat` to
/// seed an orthogonal vector; switch axes once a component dominates.
const COLLISION_TRIAL_AXIS_THRESHOLD: f64 = 0.9;
/// Standard deviations at or below this (km) collapse the Gaussian to a point,
/// for which the disk-integral collision probability is zero.
const SIGMA_FLOOR_KM: f64 = 1.0e-12;
/// Magnitudes below this are treated as zero when choosing/normalising the
/// encounter-plane eigenvector.
const EIGENVECTOR_EPS: f64 = 1.0e-30;

/// Number of Simpson intervals for the Alfano (2005) 1D quadrature. 200 gives
/// sub-nanometer precision on the validated CARA test cases.
const ALFANO_SIMPSON_INTERVALS: usize = 200;
/// Radial samples for the Foster 2D polar-grid numerical integration.
const FOSTER_NUMERICAL_RADIAL_STEPS: usize = 20;
/// Angular samples for the Foster 2D polar-grid numerical integration.
const FOSTER_NUMERICAL_ANGULAR_STEPS: usize = 40;

/// Orthonormal encounter frame and the relative state it was built from.
///
/// Axes: `x_hat` is the in-plane cross-track axis, `y_hat` is along the
/// relative velocity, and `z_hat` is the encounter-plane normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncounterFrame {
    pub x_hat: [f64; 3],
    pub y_hat: [f64; 3],
    pub z_hat: [f64; 3],
    pub relative_position_km: [f64; 3],
    pub relative_velocity_km_s: [f64; 3],
    pub miss_km: f64,
    pub relative_speed_km_s: f64,
}

/// Collision-probability integration method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcMethod {
    /// Foster 2D with the equal-area square approximation.
    FosterEqualArea,
    /// Foster 2D with polar-grid numerical integration.
    FosterNumerical,
    /// Alfano (2005) 1D Simpson's rule with analytical cross-axis integration.
    Alfano2005,
}

/// One object's state for a conjunction: ECI position (km), velocity (km/s),
/// and 3x3 position covariance (km^2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConjunctionState {
    pub position_km: [f64; 3],
    pub velocity_km_s: [f64; 3],
    pub covariance_km2: [[f64; 3]; 3],
}

/// Collision-probability result and the encounter-plane summary it came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionPc {
    pub pc: f64,
    pub miss_km: f64,
    pub relative_speed_km_s: f64,
    pub sigma_x_km: f64,
    pub sigma_z_km: f64,
}

/// Why a conjunction geometry or collision-probability solve could not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConjunctionError {
    /// A numeric input or derived probability was non-finite.
    NonFinite { field: &'static str },
    /// A strictly positive input was zero or negative.
    NotPositive { field: &'static str },
    /// The encounter frame is undefined because relative velocity is zero.
    UndefinedFrame,
}

impl core::fmt::Display for ConjunctionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite { field } => write!(f, "{field} must be finite"),
            Self::NotPositive { field } => write!(f, "{field} must be positive"),
            Self::UndefinedFrame => write!(f, "encounter frame is undefined"),
        }
    }
}

impl std::error::Error for ConjunctionError {}

/// Build the orthonormal encounter frame from two objects' states.
///
/// Returns an error when any state component is non-finite or the relative
/// velocity is below
/// [`ZERO_REL_SPEED_EPS_KM_S`], which leaves the frame undefined.
pub fn encounter_frame(
    r1: [f64; 3],
    v1: [f64; 3],
    r2: [f64; 3],
    v2: [f64; 3],
) -> Result<EncounterFrame, ConjunctionError> {
    validate::finite_vec3(r1, "object1.position_km").map_err(map_conjunction_field_error)?;
    validate::finite_vec3(v1, "object1.velocity_km_s").map_err(map_conjunction_field_error)?;
    validate::finite_vec3(r2, "object2.position_km").map_err(map_conjunction_field_error)?;
    validate::finite_vec3(v2, "object2.velocity_km_s").map_err(map_conjunction_field_error)?;
    let frame =
        encounter_frame_unchecked(r1, v1, r2, v2).ok_or(ConjunctionError::UndefinedFrame)?;
    validate_encounter_frame(&frame)?;
    Ok(frame)
}

fn encounter_frame_unchecked(
    r1: [f64; 3],
    v1: [f64; 3],
    r2: [f64; 3],
    v2: [f64; 3],
) -> Option<EncounterFrame> {
    let dr = vec3::sub3(r2, r1);
    let dv = vec3::sub3(v2, v1);

    let rel_speed = vec3::norm3(dv);
    if rel_speed < ZERO_REL_SPEED_EPS_KM_S {
        return None;
    }

    // Unit vector along the relative velocity. The division form (not the
    // reciprocal-multiply of `vec3::unit3`) is required to stay bit-exact with
    // the reference. `rel_speed` is exactly `norm3(dv)`.
    let y_hat = [dv[0] / rel_speed, dv[1] / rel_speed, dv[2] / rel_speed];

    // Component of the relative position orthogonal to the relative velocity.
    let dr_dot_y = vec3::dot3(dr, y_hat);
    let dr_ortho = vec3::sub3(dr, vec3::scale3(y_hat, dr_dot_y));
    let miss_ortho = vec3::norm3(dr_ortho);

    let x_hat = if miss_ortho < COLLISION_COURSE_EPS_KM {
        // Collision course: the cross-track direction is degenerate, so build
        // any unit vector orthogonal to `y_hat`.
        let trial = if y_hat[0].abs() < COLLISION_TRIAL_AXIS_THRESHOLD {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let cr = vec3::cross3(y_hat, trial);
        let n = vec3::norm3(cr);
        [cr[0] / n, cr[1] / n, cr[2] / n]
    } else {
        // `miss_ortho` is exactly `norm3(dr_ortho)`.
        [
            dr_ortho[0] / miss_ortho,
            dr_ortho[1] / miss_ortho,
            dr_ortho[2] / miss_ortho,
        ]
    };

    let z_hat = vec3::cross3(y_hat, x_hat);

    Some(EncounterFrame {
        x_hat,
        y_hat,
        z_hat,
        relative_position_km: dr,
        relative_velocity_km_s: dv,
        miss_km: miss_ortho,
        relative_speed_km_s: rel_speed,
    })
}

/// Project a 3x3 ECI position covariance into the 2D encounter plane `(x, z)`.
///
/// Computes `R * C * R^T` where the rows of `R` are `x_hat` and `z_hat`. The
/// accumulation order matches the reference so the result is bit-exact.
pub fn encounter_plane_covariance(
    frame: &EncounterFrame,
    cov: &[[f64; 3]; 3],
) -> Result<[[f64; 2]; 2], ConjunctionError> {
    validate_encounter_frame(frame)?;
    validate_covariance(cov, "covariance_km2")?;
    let projected = encounter_plane_covariance_unchecked(frame, cov);
    validate_plane_covariance(&projected, "encounter_plane_covariance")?;
    Ok(projected)
}

fn encounter_plane_covariance_unchecked(
    frame: &EncounterFrame,
    cov: &[[f64; 3]; 3],
) -> [[f64; 2]; 2] {
    let r = [frame.x_hat, frame.z_hat];

    // M = R * C  (2x3).
    let mut m = [[0.0_f64; 3]; 2];
    for (i, m_row) in m.iter_mut().enumerate() {
        for (j, m_ij) in m_row.iter_mut().enumerate() {
            let mut s = 0.0_f64;
            s += r[i][0] * cov[0][j];
            s += r[i][1] * cov[1][j];
            s += r[i][2] * cov[2][j];
            *m_ij = s;
        }
    }

    // C_enc = M * R^T  (2x2); column `jj` of R^T is row `jj` of R.
    let mut out = [[0.0_f64; 2]; 2];
    for (i, out_row) in out.iter_mut().enumerate() {
        for (jj, out_ij) in out_row.iter_mut().enumerate() {
            let mut s = 0.0_f64;
            s += m[i][0] * r[jj][0];
            s += m[i][1] * r[jj][1];
            s += m[i][2] * r[jj][2];
            *out_ij = s;
        }
    }

    out
}

/// Compute the collision probability for two objects' states and covariances.
///
/// Returns `None` when [`encounter_frame`] is undefined (zero relative
/// velocity). The covariances are combined by element-wise addition.
pub fn collision_probability(
    object1: &ConjunctionState,
    object2: &ConjunctionState,
    hard_body_radius_km: f64,
    method: PcMethod,
) -> Result<CollisionPc, ConjunctionError> {
    validate_conjunction_state(
        object1,
        "object1.position_km",
        "object1.velocity_km_s",
        "object1.covariance_km2",
    )?;
    validate_conjunction_state(
        object2,
        "object2.position_km",
        "object2.velocity_km_s",
        "object2.covariance_km2",
    )?;
    validate::finite_positive(hard_body_radius_km, "hard_body_radius_km")
        .map_err(map_conjunction_field_error)?;

    let frame = encounter_frame_unchecked(
        object1.position_km,
        object1.velocity_km_s,
        object2.position_km,
        object2.velocity_km_s,
    )
    .ok_or(ConjunctionError::UndefinedFrame)?;
    validate_encounter_frame(&frame)?;
    let combined = add_cov(&object1.covariance_km2, &object2.covariance_km2);
    validate_covariance(&combined, "combined_covariance_km2")?;
    let c_enc = encounter_plane_covariance_unchecked(&frame, &combined);
    validate_plane_covariance(&c_enc, "encounter_plane_covariance")?;
    let (sigma_x, sigma_z, xm, zm) = principal_components(&frame, &c_enc);
    validate_principal_components(sigma_x, sigma_z, xm, zm)?;

    let pc = match method {
        PcMethod::FosterEqualArea => {
            foster_equal_area(sigma_x, sigma_z, xm, zm, hard_body_radius_km)
        }
        PcMethod::FosterNumerical => {
            foster_numerical(sigma_x, sigma_z, xm, zm, hard_body_radius_km)
        }
        PcMethod::Alfano2005 => alfano_2005(sigma_x, sigma_z, xm, zm, hard_body_radius_km),
    };
    let pc = validate::finite(pc, "collision probability").map_err(map_conjunction_field_error)?;

    Ok(CollisionPc {
        pc: if pc < 0.0 { 0.0 } else { pc },
        miss_km: frame.miss_km,
        relative_speed_km_s: frame.relative_speed_km_s,
        sigma_x_km: sigma_x,
        sigma_z_km: sigma_z,
    })
}

fn validate_conjunction_state(
    state: &ConjunctionState,
    position_field: &'static str,
    velocity_field: &'static str,
    covariance_field: &'static str,
) -> Result<(), ConjunctionError> {
    validate::finite_vec3(state.position_km, position_field)
        .map_err(map_conjunction_field_error)?;
    validate::finite_vec3(state.velocity_km_s, velocity_field)
        .map_err(map_conjunction_field_error)?;
    validate_covariance(&state.covariance_km2, covariance_field)?;
    Ok(())
}

fn validate_encounter_frame(frame: &EncounterFrame) -> Result<(), ConjunctionError> {
    validate::finite_vec3(frame.x_hat, "frame.x_hat").map_err(map_conjunction_field_error)?;
    validate::finite_vec3(frame.y_hat, "frame.y_hat").map_err(map_conjunction_field_error)?;
    validate::finite_vec3(frame.z_hat, "frame.z_hat").map_err(map_conjunction_field_error)?;
    validate::finite_vec3(frame.relative_position_km, "frame.relative_position_km")
        .map_err(map_conjunction_field_error)?;
    validate::finite_vec3(frame.relative_velocity_km_s, "frame.relative_velocity_km_s")
        .map_err(map_conjunction_field_error)?;
    validate::finite(frame.miss_km, "frame.miss_km").map_err(map_conjunction_field_error)?;
    validate::finite(frame.relative_speed_km_s, "frame.relative_speed_km_s")
        .map_err(map_conjunction_field_error)?;
    Ok(())
}

fn validate_plane_covariance(
    cov: &[[f64; 2]; 2],
    field: &'static str,
) -> Result<(), ConjunctionError> {
    for row in cov {
        validate::finite_slice(row, field).map_err(map_conjunction_field_error)?;
    }
    Ok(())
}

fn validate_principal_components(
    sigma_x: f64,
    sigma_z: f64,
    xm: f64,
    zm: f64,
) -> Result<(), ConjunctionError> {
    validate::finite(sigma_x, "sigma_x_km").map_err(map_conjunction_field_error)?;
    validate::finite(sigma_z, "sigma_z_km").map_err(map_conjunction_field_error)?;
    validate::finite(xm, "miss_x_km").map_err(map_conjunction_field_error)?;
    validate::finite(zm, "miss_z_km").map_err(map_conjunction_field_error)?;
    Ok(())
}

fn validate_covariance(cov: &[[f64; 3]; 3], field: &'static str) -> Result<(), ConjunctionError> {
    for row in cov {
        validate::finite_slice(row, field).map_err(map_conjunction_field_error)?;
    }
    if !symmetric(cov) || !positive_semidefinite(cov) {
        return Err(ConjunctionError::NotPositive { field });
    }
    Ok(())
}

fn map_conjunction_field_error(error: validate::FieldError) -> ConjunctionError {
    match error {
        validate::FieldError::NonFinite { field } => ConjunctionError::NonFinite { field },
        validate::FieldError::NotPositive { field }
        | validate::FieldError::Negative { field }
        | validate::FieldError::OutOfRange { field, .. } => ConjunctionError::NotPositive { field },
        validate::FieldError::Missing { field }
        | validate::FieldError::FloatParse { field, .. }
        | validate::FieldError::IntParse { field, .. }
        | validate::FieldError::InvalidCivilDate { field, .. }
        | validate::FieldError::InvalidCivilTime { field, .. } => {
            ConjunctionError::NonFinite { field }
        }
    }
}

fn add_cov(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

/// Standard deviations, transformed miss components, of the principal-axes
/// encounter-plane Gaussian. Returns `(sigma_x, sigma_z, xm, zm)`.
fn principal_components(frame: &EncounterFrame, c_enc: &[[f64; 2]; 2]) -> (f64, f64, f64, f64) {
    let a = c_enc[0][0];
    let b = c_enc[0][1];
    let c = c_enc[1][0];
    let d = c_enc[1][1];

    let trace = a + d;
    let det = a * d - b * c;
    let disc = (trace * trace / 4.0 - det).max(0.0).sqrt();

    let l1 = trace / 2.0 + disc;
    let l2 = trace / 2.0 - disc;

    let sigma_x = l1.max(0.0).sqrt();
    let sigma_z = l2.max(0.0).sqrt();

    let (vx, vy) = if b.abs() > EIGENVECTOR_EPS {
        normalize_2d(l1 - d, c)
    } else if c.abs() > EIGENVECTOR_EPS {
        normalize_2d(c, l1 - a)
    } else {
        (1.0, 0.0)
    };

    let theta = vy.atan2(vx);
    let xm = frame.miss_km * theta.cos();
    let zm = -frame.miss_km * theta.sin();

    (sigma_x, sigma_z, xm, zm)
}

fn normalize_2d(x: f64, y: f64) -> (f64, f64) {
    let m = (x * x + y * y).sqrt();
    if m > EIGENVECTOR_EPS {
        (x / m, y / m)
    } else {
        (1.0, 0.0)
    }
}

/// Foster 2D collision probability via the equal-area square approximation.
fn foster_equal_area(sigma_x: f64, sigma_z: f64, xm: f64, zm: f64, hbr: f64) -> f64 {
    if sigma_x > SIGMA_FLOOR_KM && sigma_z > SIGMA_FLOOR_KM {
        let hsq = (PI / 4.0).sqrt() * hbr;
        let sqrt2 = 2.0_f64.sqrt();
        0.25 * (erf((xm + hsq) / (sqrt2 * sigma_x)) - erf((xm - hsq) / (sqrt2 * sigma_x)))
            * (erf((zm + hsq) / (sqrt2 * sigma_z)) - erf((zm - hsq) / (sqrt2 * sigma_z)))
    } else {
        0.0
    }
}

/// Alfano (2005) collision probability: 1D Simpson's composite rule along the
/// wide axis with the cross-axis integrated analytically as a difference of
/// error functions.
fn alfano_2005(sigma_x: f64, sigma_z: f64, xm: f64, zm: f64, hbr: f64) -> f64 {
    if sigma_x > SIGMA_FLOOR_KM && sigma_z > SIGMA_FLOOR_KM {
        let n = ALFANO_SIMPSON_INTERVALS;
        let h = 2.0 * hbr / n as f64;
        let sqrt2 = 2.0_f64.sqrt();
        let sqrt_2pi = (2.0 * PI).sqrt();

        let integrand = |i: usize| -> f64 {
            let x = -hbr + i as f64 * h;
            let y_top_sq = hbr * hbr - x * x;
            if y_top_sq <= 0.0 {
                0.0
            } else {
                let y_top = y_top_sq.sqrt();
                let arg = (x - xm) * (x - xm) / (2.0 * sigma_x * sigma_x);
                let exp_term = (-arg).exp();
                let erf_top = erf((y_top - zm) / (sigma_z * sqrt2));
                let erf_bot = erf((-y_top - zm) / (sigma_z * sqrt2));
                exp_term * (erf_top - erf_bot)
            }
        };

        let mut simpson_sum = 0.0_f64;
        for i in 0..=n {
            let weight = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            simpson_sum += weight * integrand(i);
        }

        let prefactor = 1.0 / (2.0 * sigma_x * sqrt_2pi);
        prefactor * (h / 3.0) * simpson_sum
    } else {
        0.0
    }
}

/// Foster 2D collision probability via polar-grid numerical integration.
fn foster_numerical(sigma_x: f64, sigma_z: f64, xm: f64, zm: f64, hbr: f64) -> f64 {
    if sigma_x > SIGMA_FLOOR_KM && sigma_z > SIGMA_FLOOR_KM {
        let steps_r = FOSTER_NUMERICAL_RADIAL_STEPS;
        let steps_theta = FOSTER_NUMERICAL_ANGULAR_STEPS;
        let dr = hbr / steps_r as f64;
        let dtheta = 2.0 * PI / steps_theta as f64;

        let mut acc = 0.0_f64;
        for i in 0..steps_r {
            for j in 0..steps_theta {
                let r = (i as f64 + 0.5) * dr;
                let theta = (j as f64 + 0.5) * dtheta;
                let x = r * theta.cos();
                let z = r * theta.sin();

                // `powf(2.0)` (not `x * x`) matches the reference `:math.pow(_, 2)`
                // for bit-exact parity.
                let arg = (x - xm).powf(2.0) / (2.0 * sigma_x * sigma_x)
                    + (z - zm).powf(2.0) / (2.0 * sigma_z * sigma_z);
                let f = (-arg).exp() / (2.0 * PI * sigma_x * sigma_z);
                acc += f * r * dr * dtheta;
            }
        }
        acc
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NASA CARA Omitron test case: states in ECI km / km/s, covariances in km^2.
    const OMITRON_OBJ1: ConjunctionState = ConjunctionState {
        position_km: [378.39559, 4305.721887, 5752.767554],
        velocity_km_s: [2.360800244, 5.580331936, -4.322349039],
        covariance_km2: [
            [44.5757544811362, 81.6751751052616, -67.8687662707124],
            [81.6751751052616, 158.453402956163, -128.616921644857],
            [-67.8687662707124, -128.616921644858, 105.490542562701],
        ],
    };
    const OMITRON_OBJ2: ConjunctionState = ConjunctionState {
        position_km: [374.5180598, 4307.560983, 5751.130418],
        velocity_km_s: [-5.388125081, -3.946827739, 3.322820358],
        covariance_km2: [
            [2.31067077720423, 1.69905293875632, -1.4170164577661],
            [1.69905293875632, 1.24957388457206, -1.04174164279599],
            [-1.4170164577661, -1.04174164279599, 0.869260558223714],
        ],
    };
    const OMITRON_HBR_KM: f64 = 0.020;

    fn omitron_pc(method: PcMethod, hbr: f64) -> CollisionPc {
        collision_probability(&OMITRON_OBJ1, &OMITRON_OBJ2, hbr, method)
            .expect("omitron frame is well defined")
    }

    #[test]
    fn collision_probability_matches_cara_published_value() {
        // External-product published value (NASA CARA Analysis Tools, Omitron
        // case). A tolerance is legitimate here: this is a computed quantity
        // compared against an external product's published number, not an
        // implementation-vs-implementation comparison.
        let result = omitron_pc(PcMethod::FosterEqualArea, OMITRON_HBR_KM);
        assert!((result.pc - 2.706_015_734_901_11e-5).abs() < 1.0e-9);
    }

    #[test]
    fn collision_probability_has_frozen_bits() {
        // Frozen-bits regression golden of the deterministic implementation's
        // own output. The Pc values come from the pure-Rust `libm::erf`, which
        // is identical on every platform; pinning them here guards against
        // accidental op-order or implementation drift. The geometry fields
        // (miss, relative speed, sigmas) do not depend on erf.
        let ea = omitron_pc(PcMethod::FosterEqualArea, OMITRON_HBR_KM);
        assert_eq!(ea.pc.to_bits(), 0x3efc_5fe7_e374_e761);
        assert_eq!(ea.miss_km.to_bits(), 0x4012_5f76_bb63_5a77);
        assert_eq!(ea.relative_speed_km_s.to_bits(), 0x402c_ee85_c45b_2fca);
        assert_eq!(ea.sigma_x_km.to_bits(), 0x400b_2f8b_338c_261f);
        assert_eq!(ea.sigma_z_km.to_bits(), 0x3fe9_dc26_3a96_e94c);

        let num = omitron_pc(PcMethod::FosterNumerical, OMITRON_HBR_KM);
        assert_eq!(num.pc.to_bits(), 0x3efc_5fed_5966_2e48);

        let alf = omitron_pc(PcMethod::Alfano2005, OMITRON_HBR_KM);
        assert_eq!(alf.pc.to_bits(), 0x3efc_5edd_3b73_ec59);

        assert_eq!(
            omitron_pc(PcMethod::Alfano2005, 0.010).pc.to_bits(),
            0x3edc_5f31_c28e_87bb
        );
        assert_eq!(
            omitron_pc(PcMethod::Alfano2005, 0.040).pc.to_bits(),
            0x3f1c_5d8b_338e_291a
        );
    }

    #[test]
    fn methods_agree_for_well_conditioned_geometry() {
        let ea = omitron_pc(PcMethod::FosterEqualArea, OMITRON_HBR_KM).pc;
        let num = omitron_pc(PcMethod::FosterNumerical, OMITRON_HBR_KM).pc;
        let alf = omitron_pc(PcMethod::Alfano2005, OMITRON_HBR_KM).pc;

        assert!((num - ea).abs() < ea * 0.01);
        assert!((alf - ea).abs() < ea * 0.01);
        assert!((alf - num).abs() < num * 0.01);
        assert!(alf > 0.0);
    }

    #[test]
    fn larger_hard_body_radius_increases_pc() {
        for method in [
            PcMethod::FosterEqualArea,
            PcMethod::FosterNumerical,
            PcMethod::Alfano2005,
        ] {
            let small = omitron_pc(method, 0.010).pc;
            let large = omitron_pc(method, 0.040).pc;
            assert!(large > small);
        }
    }

    #[test]
    fn swapping_objects_preserves_pc() {
        let forward = omitron_pc(PcMethod::FosterEqualArea, OMITRON_HBR_KM).pc;
        let swapped = collision_probability(
            &OMITRON_OBJ2,
            &OMITRON_OBJ1,
            OMITRON_HBR_KM,
            PcMethod::FosterEqualArea,
        )
        .unwrap()
        .pc;
        assert!((forward - swapped).abs() < 1.0e-15);
    }

    #[test]
    fn zero_relative_velocity_has_no_frame() {
        let cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(encounter_frame(
            [7000.0, 0.0, 0.0],
            [0.0, 7.5, 0.0],
            [7000.01, 0.0, 0.0],
            [0.0, 7.5, 0.0],
        )
        .is_err_and(|err| err == ConjunctionError::UndefinedFrame));
        let obj1 = ConjunctionState {
            position_km: [7000.0, 0.0, 0.0],
            velocity_km_s: [0.0, 7.5, 0.0],
            covariance_km2: cov,
        };
        let obj2 = ConjunctionState {
            position_km: [7000.01, 0.0, 0.0],
            velocity_km_s: [0.0, 7.5, 0.0],
            covariance_km2: cov,
        };
        assert_eq!(
            collision_probability(&obj1, &obj2, 0.015, PcMethod::FosterEqualArea),
            Err(ConjunctionError::UndefinedFrame)
        );
    }

    #[test]
    fn non_finite_state_inputs_are_rejected() {
        let mut object = OMITRON_OBJ1;
        object.position_km[0] = f64::NAN;
        assert_eq!(
            collision_probability(
                &object,
                &OMITRON_OBJ2,
                OMITRON_HBR_KM,
                PcMethod::FosterEqualArea,
            ),
            Err(ConjunctionError::NonFinite {
                field: "object1.position_km"
            })
        );

        assert_eq!(
            encounter_frame(
                [f64::INFINITY, 0.0, 0.0],
                [0.0, 7.5, 0.0],
                [7000.01, 0.0, 0.0],
                [0.0, -7.5, 0.0],
            ),
            Err(ConjunctionError::NonFinite {
                field: "object1.position_km"
            })
        );
    }

    #[test]
    fn non_finite_encounter_frame_outputs_are_rejected() {
        assert!(matches!(
            encounter_frame(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [f64::MAX, 0.0, 0.0],
                [0.0, f64::MAX, 0.0],
            ),
            Err(ConjunctionError::NonFinite {
                field: "frame.x_hat"
                    | "frame.y_hat"
                    | "frame.z_hat"
                    | "frame.relative_position_km"
                    | "frame.relative_velocity_km_s"
                    | "frame.miss_km"
                    | "frame.relative_speed_km_s"
            })
        ));
    }

    #[test]
    fn invalid_covariance_shape_is_rejected_before_pc() {
        for covariance_km2 in [
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.25, 0.0], [0.20, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ] {
            let object = ConjunctionState {
                covariance_km2,
                ..OMITRON_OBJ1
            };

            assert_eq!(
                collision_probability(
                    &object,
                    &OMITRON_OBJ2,
                    OMITRON_HBR_KM,
                    PcMethod::FosterEqualArea,
                ),
                Err(ConjunctionError::NotPositive {
                    field: "object1.covariance_km2"
                })
            );
        }
    }

    #[test]
    fn covariance_and_hbr_inputs_are_rejected() {
        let mut object = OMITRON_OBJ2;
        object.covariance_km2[1][2] = f64::INFINITY;
        assert_eq!(
            collision_probability(
                &OMITRON_OBJ1,
                &object,
                OMITRON_HBR_KM,
                PcMethod::FosterEqualArea,
            ),
            Err(ConjunctionError::NonFinite {
                field: "object2.covariance_km2"
            })
        );

        assert_eq!(
            collision_probability(&OMITRON_OBJ1, &OMITRON_OBJ2, 0.0, PcMethod::FosterEqualArea),
            Err(ConjunctionError::NotPositive {
                field: "hard_body_radius_km"
            })
        );

        let frame = encounter_frame(
            [7000.0, 0.0, 0.0],
            [0.0, 7.5, 0.0],
            [7000.1, 0.0, 0.0],
            [0.0, -7.5, 0.0],
        )
        .unwrap();
        let cov = [[1.0, 0.0, 0.0], [0.0, f64::NAN, 0.0], [0.0, 0.0, 3.0]];
        assert_eq!(
            encounter_plane_covariance(&frame, &cov),
            Err(ConjunctionError::NonFinite {
                field: "covariance_km2"
            })
        );
    }

    #[test]
    fn head_on_frame_matches_reference_bits() {
        let frame = encounter_frame(
            [7000.0, 0.0, 0.0],
            [0.0, 7.5, 0.0],
            [7000.1, 0.0, 0.0],
            [0.0, -7.5, 0.0],
        )
        .unwrap();

        assert_eq!(frame.x_hat[0].to_bits(), 1.0_f64.to_bits());
        assert_eq!(frame.x_hat[1].to_bits(), 0.0_f64.to_bits());
        assert_eq!(frame.x_hat[2].to_bits(), 0.0_f64.to_bits());
        assert_eq!(frame.y_hat[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(frame.y_hat[1].to_bits(), (-1.0_f64).to_bits());
        assert_eq!(frame.y_hat[2].to_bits(), 0.0_f64.to_bits());
        assert_eq!(frame.z_hat[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(frame.z_hat[1].to_bits(), 0.0_f64.to_bits());
        assert_eq!(frame.z_hat[2].to_bits(), 1.0_f64.to_bits());
        assert_eq!(frame.miss_km.to_bits(), 0x3fb9_9999_999a_0000);
        assert_eq!(frame.relative_speed_km_s.to_bits(), 0x402e_0000_0000_0000);
    }

    #[test]
    fn encounter_plane_covariance_projects_onto_xz() {
        let frame = encounter_frame(
            [7000.0, 0.0, 0.0],
            [0.0, 7.5, 0.0],
            [7000.1, 0.0, 0.0],
            [0.0, -7.5, 0.0],
        )
        .unwrap();
        let cov = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let c_enc = encounter_plane_covariance(&frame, &cov).unwrap();
        assert_eq!(c_enc, [[1.0, 0.0], [0.0, 3.0]]);
    }
}
