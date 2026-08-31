//! Navigation state and attitude conversion helpers.

use crate::astro::math::mat3::{inline_rxr, mul_vec3, Mat3};
use crate::astro::math::vec3::{cross3, dot3, norm3, scale3, sub3};

use super::{invalid_input, validate_finite, validate_mat3, validate_vec3, InertialError};

/// Navigation state used by the ECEF strapdown mechanizer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavState {
    /// State time in seconds since J2000 on the caller's GNSS time scale.
    pub t_j2000_s: f64,
    /// IMU position in ECEF meters.
    pub position_ecef_m: [f64; 3],
    /// IMU velocity in ECEF meters per second.
    pub velocity_ecef_mps: [f64; 3],
    /// Direction cosine matrix from body axes to ECEF axes.
    pub attitude_body_to_ecef: Mat3,
    /// Closed-loop accelerometer bias estimate in m/s^2.
    pub accel_bias_mps2: [f64; 3],
    /// Closed-loop gyroscope bias estimate in rad/s.
    pub gyro_bias_rps: [f64; 3],
}

impl NavState {
    /// Build a navigation state with zero IMU biases.
    pub fn new(
        t_j2000_s: f64,
        position_ecef_m: [f64; 3],
        velocity_ecef_mps: [f64; 3],
        attitude_body_to_ecef: Mat3,
    ) -> Result<Self, InertialError> {
        let state = Self {
            t_j2000_s,
            position_ecef_m,
            velocity_ecef_mps,
            attitude_body_to_ecef,
            accel_bias_mps2: [0.0; 3],
            gyro_bias_rps: [0.0; 3],
        };
        state.validate()?;
        Ok(state)
    }

    /// Return this state with closed-loop IMU bias estimates.
    pub fn with_biases(
        mut self,
        accel_bias_mps2: [f64; 3],
        gyro_bias_rps: [f64; 3],
    ) -> Result<Self, InertialError> {
        self.accel_bias_mps2 = accel_bias_mps2;
        self.gyro_bias_rps = gyro_bias_rps;
        self.validate()?;
        Ok(self)
    }

    /// Validate finite state fields and a nondegenerate attitude matrix.
    pub fn validate(&self) -> Result<(), InertialError> {
        validate_finite(self.t_j2000_s, "t_j2000_s")?;
        validate_vec3(self.position_ecef_m, "position_ecef_m")?;
        validate_vec3(self.velocity_ecef_mps, "velocity_ecef_mps")?;
        validate_mat3(&self.attitude_body_to_ecef, "attitude_body_to_ecef")?;
        validate_vec3(self.accel_bias_mps2, "accel_bias_mps2")?;
        validate_vec3(self.gyro_bias_rps, "gyro_bias_rps")?;
        validate_dcm_orthonormal(&self.attitude_body_to_ecef, "attitude_body_to_ecef")?;
        Ok(())
    }

    /// Body-to-ECEF attitude as a unit quaternion.
    pub fn attitude_quaternion_body_to_ecef(&self) -> Result<AttitudeQuaternion, InertialError> {
        dcm_to_quaternion(&self.attitude_body_to_ecef)
    }

    /// Body-to-ECEF attitude as yaw, pitch, and roll angles in radians.
    pub fn attitude_yaw_pitch_roll_rad(&self) -> [f64; 3] {
        attitude_yaw_pitch_roll_rad(&self.attitude_body_to_ecef)
    }
}

/// Unit quaternion using scalar-first `(w, x, y, z)` storage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttitudeQuaternion {
    /// Scalar component.
    pub w: f64,
    /// X vector component.
    pub x: f64,
    /// Y vector component.
    pub y: f64,
    /// Z vector component.
    pub z: f64,
}

impl AttitudeQuaternion {
    /// Build and normalize a quaternion.
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Result<Self, InertialError> {
        validate_finite(w, "quaternion.w")?;
        validate_finite(x, "quaternion.x")?;
        validate_finite(y, "quaternion.y")?;
        validate_finite(z, "quaternion.z")?;
        let norm = (w * w + x * x + y * y + z * z).sqrt();
        if norm <= 0.0 {
            return Err(InertialError::DegenerateAttitude);
        }
        Ok(Self {
            w: w / norm,
            x: x / norm,
            y: y / norm,
            z: z / norm,
        })
    }
}

/// Convert a body-to-ECEF DCM to a unit quaternion.
pub fn dcm_to_quaternion(dcm: &Mat3) -> Result<AttitudeQuaternion, InertialError> {
    validate_mat3(dcm, "dcm")?;
    let dcm = reorthonormalize_dcm(dcm)?;
    let trace = dcm[0][0] + dcm[1][1] + dcm[2][2];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        AttitudeQuaternion::new(
            0.25 * s,
            (dcm[2][1] - dcm[1][2]) / s,
            (dcm[0][2] - dcm[2][0]) / s,
            (dcm[1][0] - dcm[0][1]) / s,
        )
    } else if dcm[0][0] > dcm[1][1] && dcm[0][0] > dcm[2][2] {
        let s = (1.0 + dcm[0][0] - dcm[1][1] - dcm[2][2]).sqrt() * 2.0;
        AttitudeQuaternion::new(
            (dcm[2][1] - dcm[1][2]) / s,
            0.25 * s,
            (dcm[0][1] + dcm[1][0]) / s,
            (dcm[0][2] + dcm[2][0]) / s,
        )
    } else if dcm[1][1] > dcm[2][2] {
        let s = (1.0 + dcm[1][1] - dcm[0][0] - dcm[2][2]).sqrt() * 2.0;
        AttitudeQuaternion::new(
            (dcm[0][2] - dcm[2][0]) / s,
            (dcm[0][1] + dcm[1][0]) / s,
            0.25 * s,
            (dcm[1][2] + dcm[2][1]) / s,
        )
    } else {
        let s = (1.0 + dcm[2][2] - dcm[0][0] - dcm[1][1]).sqrt() * 2.0;
        AttitudeQuaternion::new(
            (dcm[1][0] - dcm[0][1]) / s,
            (dcm[0][2] + dcm[2][0]) / s,
            (dcm[1][2] + dcm[2][1]) / s,
            0.25 * s,
        )
    }
}

/// Convert a unit quaternion to a body-to-ECEF DCM.
pub fn quaternion_to_dcm(quaternion: AttitudeQuaternion) -> Mat3 {
    let w = quaternion.w;
    let x = quaternion.x;
    let y = quaternion.y;
    let z = quaternion.z;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Convert a body-to-ECEF DCM to yaw, pitch, roll radians using a ZYX sequence.
pub fn attitude_yaw_pitch_roll_rad(dcm: &Mat3) -> [f64; 3] {
    let pitch = libm::asin(-dcm[2][0]);
    let roll = libm::atan2(dcm[2][1], dcm[2][2]);
    let yaw = libm::atan2(dcm[1][0], dcm[0][0]);
    [yaw, pitch, roll]
}

/// Maximum elementwise departure of `C * C^T` from identity accepted by
/// [`validate_dcm_orthonormal`]. Numerical attitude drift under per-step
/// renormalization sits many orders of magnitude below this; scaled, sheared,
/// or otherwise non-orthonormal frames are rejected rather than repaired.
pub(crate) const DCM_ORTHONORMALITY_TOL: f64 = 1.0e-6;

/// Validate that `dcm` is an orthonormal, right-handed rotation within
/// [`DCM_ORTHONORMALITY_TOL`]. Rejects inputs that only become rotations
/// after re-orthonormalization, so callers cannot feed a degenerate attitude
/// and receive a precise result.
pub(crate) fn validate_dcm_orthonormal(
    dcm: &Mat3,
    field: &'static str,
) -> Result<(), InertialError> {
    validate_mat3(dcm, field)?;
    for i in 0..3 {
        for j in 0..3 {
            let mut g = 0.0;
            for row in dcm {
                g += row[i] * row[j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            if (g - expected).abs() > DCM_ORTHONORMALITY_TOL {
                return Err(invalid_input(field, "matrix is not orthonormal"));
            }
        }
    }
    let det = dcm[0][0] * (dcm[1][1] * dcm[2][2] - dcm[1][2] * dcm[2][1])
        - dcm[0][1] * (dcm[1][0] * dcm[2][2] - dcm[1][2] * dcm[2][0])
        + dcm[0][2] * (dcm[1][0] * dcm[2][1] - dcm[1][1] * dcm[2][0]);
    if det <= 0.5 {
        return Err(invalid_input(
            field,
            "matrix is not a right-handed rotation",
        ));
    }
    Ok(())
}

/// Re-orthonormalize a direction-cosine matrix into a right-handed frame.
pub fn reorthonormalize_dcm(dcm: &Mat3) -> Result<Mat3, InertialError> {
    validate_mat3(dcm, "dcm")?;
    let c0 = [dcm[0][0], dcm[1][0], dcm[2][0]];
    let c1_raw = [dcm[0][1], dcm[1][1], dcm[2][1]];
    let e0 = unit(c0)?;
    let c1_orth = sub3(c1_raw, scale3(e0, dot3(e0, c1_raw)));
    let e1 = unit(c1_orth)?;
    let e2 = unit(cross3(e0, e1))?;
    let e1 = unit(cross3(e2, e0))?;
    Ok([
        [e0[0], e1[0], e2[0]],
        [e0[1], e1[1], e2[1]],
        [e0[2], e1[2], e2[2]],
    ])
}

pub(crate) fn mat3_identity() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

pub(crate) fn mat3_add(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

pub(crate) fn mat3_scale(a: &Mat3, scale: f64) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] * scale;
        }
    }
    out
}

pub(crate) fn mat3_mul_vec(a: &Mat3, v: [f64; 3]) -> [f64; 3] {
    mul_vec3(a, v)
}

pub(crate) fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    inline_rxr(a, b)
}

pub(crate) fn skew(v: [f64; 3]) -> Mat3 {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn unit(v: [f64; 3]) -> Result<[f64; 3], InertialError> {
    let norm = norm3(v);
    if norm > 0.0 {
        Ok(scale3(v, 1.0 / norm))
    } else {
        Err(InertialError::DegenerateAttitude)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quaternion_roundtrip_preserves_right_angle_yaw() {
        let half = core::f64::consts::FRAC_1_SQRT_2;
        let q = AttitudeQuaternion::new(half, 0.0, 0.0, half).expect("quaternion");
        let dcm = quaternion_to_dcm(q);
        let ypr = attitude_yaw_pitch_roll_rad(&dcm);
        assert_eq!(ypr[0].to_bits(), 0x3ff9_21fb_5444_2d1a);
        assert!(ypr[1].abs() <= 1.0e-16);
        assert!(ypr[2].abs() <= 1.0e-16);

        let back = dcm_to_quaternion(&dcm).expect("back");
        assert!((back.w - half).abs() <= 2.0e-16);
        assert!((back.z - half).abs() <= 2.0e-16);
    }
}

#[cfg(test)]
mod orthonormality_tests {
    //! Regression for the review finding that a scaled non-rotation attitude
    //! was accepted and silently repaired instead of rejected.
    use super::*;

    fn state_with_attitude(attitude: Mat3) -> NavState {
        NavState {
            t_j2000_s: 0.0,
            position_ecef_m: [6_378_137.0, 0.0, 0.0],
            velocity_ecef_mps: [0.0, 0.0, 0.0],
            attitude_body_to_ecef: attitude,
            accel_bias_mps2: [0.0, 0.0, 0.0],
            gyro_bias_rps: [0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn scaled_non_rotation_attitude_is_rejected() {
        let scaled = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        assert!(validate_dcm_orthonormal(&scaled, "attitude").is_err());
        assert!(state_with_attitude(scaled).validate().is_err());
    }

    #[test]
    fn left_handed_frame_is_rejected() {
        let reflected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
        assert!(validate_dcm_orthonormal(&reflected, "attitude").is_err());
    }

    #[test]
    fn identity_and_small_drift_are_accepted() {
        assert!(validate_dcm_orthonormal(&mat3_identity(), "attitude").is_ok());
        let mut drifted = mat3_identity();
        drifted[0][0] += 1.0e-9;
        assert!(validate_dcm_orthonormal(&drifted, "attitude").is_ok());
    }
}
