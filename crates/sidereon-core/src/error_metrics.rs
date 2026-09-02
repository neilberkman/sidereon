//! Position-error metrics from supplied covariance matrices.
//!
//! The functions in this module consume a covariance supplied by another solver
//! and report standard radial accuracy scalars. No covariance estimation is
//! performed here.

use crate::astro::frames::transforms::geodetic_from_ecef_proj;
use crate::astro::math::special::erf;
use crate::dop::{self, DopError};
use crate::frame::Wgs84Geodetic;
use crate::integrity::{self, erfc_inv};
use crate::precise_positioning::KinematicEpochSolution;

const PSD_DIAGONAL_EPS: f64 = 1.0e-15;
const PSD_MINOR_EPS: f64 = 1.0e-12;
const SYMMETRY_EPS: f64 = 1.0e-12;
const DEGENERATE_REL_EPS: f64 = 1.0e-15;
const CEP_RGCSP_MIN_RATIO: f64 = 0.25;
const SEP_APPROX_MIN_RATIO: f64 = 0.35;
const NORMAL_MEDIAN_ABS: f64 = 0.674_490;
const SEP_ISOTROPIC_MEDIAN: f64 = 1.537_580;
const GL_INTERVALS: usize = 64;

#[allow(clippy::excessive_precision)]
const GL64_POSITIVE_NODES: [f64; 32] = [
    2.43502926634244325e-02,
    7.29931217877990424e-02,
    1.21462819296120544e-01,
    1.69644420423992831e-01,
    2.17423643740007083e-01,
    2.64687162208767424e-01,
    3.11322871990210970e-01,
    3.57220158337668126e-01,
    4.02270157963991570e-01,
    4.46366017253464087e-01,
    4.89403145707052956e-01,
    5.31279464019894565e-01,
    5.71895646202634000e-01,
    6.11155355172393278e-01,
    6.48965471254657311e-01,
    6.85236313054233270e-01,
    7.19881850171610771e-01,
    7.52819907260531940e-01,
    7.83972358943341385e-01,
    8.13265315122797539e-01,
    8.40629296252580316e-01,
    8.65999398154092770e-01,
    8.89315445995114140e-01,
    9.10522137078502825e-01,
    9.29569172131939570e-01,
    9.46411374858402765e-01,
    9.61008799652053658e-01,
    9.73326827789910975e-01,
    9.83336253884625977e-01,
    9.91013371476744287e-01,
    9.96340116771955220e-01,
    9.99305041735772170e-01,
];

#[allow(clippy::excessive_precision)]
const GL64_POSITIVE_WEIGHTS: [f64; 32] = [
    4.86909570091397237e-02,
    4.85754674415033935e-02,
    4.83447622348029404e-02,
    4.79993885964583034e-02,
    4.75401657148303222e-02,
    4.69681828162099788e-02,
    4.62847965813143539e-02,
    4.54916279274180727e-02,
    4.45905581637566079e-02,
    4.35837245293234157e-02,
    4.24735151236535352e-02,
    4.12625632426234581e-02,
    3.99537411327204536e-02,
    3.85501531786155635e-02,
    3.70551285402399844e-02,
    3.54722132568822401e-02,
    3.38051618371418630e-02,
    3.20579283548514254e-02,
    3.02346570724024884e-02,
    2.83396726142594486e-02,
    2.63774697150548978e-02,
    2.43527025687111306e-02,
    2.22701738083829967e-02,
    2.01348231535300216e-02,
    1.79517157756973571e-02,
    1.57260304760250269e-02,
    1.34630478967191179e-02,
    1.11681394601309738e-02,
    8.84675982636339009e-03,
    6.50445796897848854e-03,
    4.14703326056443250e-03,
    1.78328072169469699e-03,
];

/// A horizontal one-sigma error ellipse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorEllipse {
    /// Semi-major axis length, metres.
    pub semi_major_m: f64,
    /// Semi-minor axis length, metres.
    pub semi_minor_m: f64,
    /// Semi-major-axis orientation in radians, from east toward north.
    pub orientation_rad: f64,
}

/// A percentile circle or sphere radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PercentileRadius {
    /// Probability mass inside the reported radius.
    pub probability: f64,
    /// Exact circle or sphere radius, metres.
    pub radius_m: f64,
    /// Approximate radius when a named approximation is applicable.
    pub approx_m: f64,
    /// True when `approx_m` is inside the approximation's stated ratio range.
    pub approx_valid: bool,
}

/// Standard position-error metrics from one ENU covariance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionErrorMetrics {
    /// Horizontal one-sigma covariance ellipse.
    pub ellipse: ErrorEllipse,
    /// East standard deviation, metres.
    pub sigma_e_m: f64,
    /// North standard deviation, metres.
    pub sigma_n_m: f64,
    /// Up standard deviation, metres.
    pub sigma_u_m: f64,
    /// Horizontal 50 percent circle radius.
    pub cep_m: PercentileRadius,
    /// Horizontal 95 percent circle radius.
    pub r95_m: PercentileRadius,
    /// Horizontal 99 percent circle radius.
    pub r99_m: PercentileRadius,
    /// Distance root mean square, metres.
    pub drms_m: f64,
    /// Two times distance root mean square, metres.
    pub two_drms_m: f64,
    /// Vertical 50 percent one-dimensional radius, metres.
    pub vep_m: f64,
    /// Three-dimensional 50 percent sphere radius, metres.
    pub sep_m: PercentileRadius,
    /// Mean radial spherical error, metres.
    pub mrse_m: f64,
}

/// Error returned by position-error metric functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorMetricsError {
    /// At least one numeric input was NaN or infinite.
    NonFinite,
    /// The covariance was not positive semidefinite within tolerance.
    NotPositiveSemidefinite,
    /// A probability argument was outside `(0, 1)`.
    InvalidProbability,
    /// ECEF-to-ENU rotation failed.
    Rotation(DopError),
}

/// Compute all standard metrics from an ENU covariance in square metres.
pub fn metrics_from_enu_covariance_m2(
    covariance_enu_m2: [[f64; 3]; 3],
) -> Result<PositionErrorMetrics, ErrorMetricsError> {
    let covariance = validate_enu_covariance(covariance_enu_m2)?;
    let ellipse = error_ellipse_from_enu_m2(covariance)?;
    let lambda_h = horizontal_eigenvalues(covariance)?;
    let lambda_3d = eigenvalues_symmetric_3x3(covariance)?;
    let sigma_e_m = covariance[0][0].max(0.0).sqrt();
    let sigma_n_m = covariance[1][1].max(0.0).sqrt();
    let sigma_u_m = covariance[2][2].max(0.0).sqrt();
    let drms_m = (lambda_h[0].max(0.0) + lambda_h[1].max(0.0)).sqrt();
    let mrse_m = (lambda_3d[0].max(0.0) + lambda_3d[1].max(0.0) + lambda_3d[2].max(0.0)).sqrt();

    Ok(PositionErrorMetrics {
        ellipse,
        sigma_e_m,
        sigma_n_m,
        sigma_u_m,
        cep_m: horizontal_radius_from_eigenvalues(lambda_h, 0.5)?,
        r95_m: horizontal_radius_from_eigenvalues(lambda_h, 0.95)?,
        r99_m: horizontal_radius_from_eigenvalues(lambda_h, 0.99)?,
        drms_m,
        two_drms_m: 2.0 * drms_m,
        vep_m: NORMAL_MEDIAN_ABS * sigma_u_m,
        sep_m: spherical_radius_from_eigenvalues(lambda_3d, 0.5)?,
        mrse_m,
    })
}

/// Rotate an ECEF covariance to ENU and compute all standard metrics.
pub fn metrics_from_ecef_covariance_m2(
    covariance_ecef_m2: [[f64; 3]; 3],
    receiver: Wgs84Geodetic,
) -> Result<PositionErrorMetrics, ErrorMetricsError> {
    validate_finite_matrix(covariance_ecef_m2)?;
    let enu = dop::rotate_covariance_ecef_to_enu_m2(covariance_ecef_m2, receiver)
        .map_err(ErrorMetricsError::Rotation)?;
    metrics_from_enu_covariance_m2(enu)
}

/// Compute all metrics from a DOP position covariance without re-rotating it.
pub fn metrics_from_position_covariance(
    covariance: &dop::PositionCovariance,
) -> Result<PositionErrorMetrics, ErrorMetricsError> {
    metrics_from_enu_covariance_m2(covariance.enu_m2)
}

/// Compute all metrics from a kinematic PPP epoch solution.
///
/// The solution covariance is stored in ECEF square metres. The receiver
/// position is converted to WGS84 geodetic coordinates for the ENU rotation.
pub fn metrics_from_kinematic_solution(
    solution: &KinematicEpochSolution,
) -> Result<PositionErrorMetrics, ErrorMetricsError> {
    let [lon_deg, lat_deg, height_m] = geodetic_from_ecef_proj(
        solution.position_m[0],
        solution.position_m[1],
        solution.position_m[2],
    )
    .map_err(|_| {
        ErrorMetricsError::Rotation(DopError::InvalidInput {
            field: "position_m",
            reason: "geodetic conversion failed",
        })
    })?;
    let receiver = Wgs84Geodetic::new(lat_deg.to_radians(), lon_deg.to_radians(), height_m)
        .map_err(|_| {
            ErrorMetricsError::Rotation(DopError::InvalidInput {
                field: "position_m",
                reason: "geodetic conversion failed",
            })
        })?;
    metrics_from_ecef_covariance_m2(solution.position_covariance_m2, receiver)
}

/// Horizontal one-sigma ellipse from an ENU covariance in square metres.
pub fn error_ellipse_from_enu_m2(
    covariance_enu_m2: [[f64; 3]; 3],
) -> Result<ErrorEllipse, ErrorMetricsError> {
    let covariance = validate_enu_covariance(covariance_enu_m2)?;
    let block = [
        [covariance[0][0], covariance[0][1]],
        [covariance[1][0], covariance[1][1]],
    ];
    let ellipse = dop::error_ellipse_2x2_unit(block).map_err(map_dop_ellipse_error)?;
    Ok(ErrorEllipse {
        semi_major_m: ellipse.semi_major,
        semi_minor_m: ellipse.semi_minor,
        orientation_rad: ellipse.orientation_rad,
    })
}

/// Exact horizontal percentile circle radius from an ENU covariance.
///
/// For eigenvalues `a^2` and `b^2`, the circle probability is
/// `1 - (1 / (2 pi a b)) int exp(-R^2 g(phi) / 2) / g(phi) dphi`, with
/// `g(phi) = cos(phi)^2 / a^2 + sin(phi)^2 / b^2`. The integral uses the
/// fixed 64-node Gauss-Legendre rule over one quadrant and four-fold symmetry.
pub fn horizontal_radius_at(
    covariance_enu_m2: [[f64; 3]; 3],
    probability: f64,
) -> Result<PercentileRadius, ErrorMetricsError> {
    let covariance = validate_enu_covariance(covariance_enu_m2)?;
    let eigenvalues = horizontal_eigenvalues(covariance)?;
    horizontal_radius_from_eigenvalues(eigenvalues, probability)
}

/// Exact three-dimensional percentile sphere radius from an ENU covariance.
///
/// The full-rank case integrates the Gaussian density over a spherical shell
/// with fixed Gauss-Legendre angular nodes and analytic radial integration.
/// Rank-one and rank-two covariances reduce to the corresponding 1D or 2D
/// radial formulas.
pub fn spherical_radius_at(
    covariance_enu_m2: [[f64; 3]; 3],
    probability: f64,
) -> Result<PercentileRadius, ErrorMetricsError> {
    let covariance = validate_enu_covariance(covariance_enu_m2)?;
    let eigenvalues = eigenvalues_symmetric_3x3(covariance)?;
    spherical_radius_from_eigenvalues(eigenvalues, probability)
}

/// Vertical one-dimensional percentile radius from an up variance.
pub fn vertical_radius_at(sigma_u_m2: f64, probability: f64) -> Result<f64, ErrorMetricsError> {
    validate_probability(probability)?;
    if !sigma_u_m2.is_finite() {
        return Err(ErrorMetricsError::NonFinite);
    }
    if sigma_u_m2 < -PSD_DIAGONAL_EPS {
        return Err(ErrorMetricsError::NotPositiveSemidefinite);
    }
    let sigma = sigma_u_m2.max(0.0).sqrt();
    one_dimensional_radius(sigma, probability)
}

fn horizontal_radius_from_eigenvalues(
    eigenvalues: [f64; 2],
    probability: f64,
) -> Result<PercentileRadius, ErrorMetricsError> {
    validate_probability(probability)?;
    let lambda_major = eigenvalues[0].max(0.0);
    let lambda_minor = eigenvalues[1].max(0.0);
    if lambda_major == 0.0 {
        return Ok(percentile(probability, 0.0, 0.0, true));
    }

    let sigma_major = lambda_major.sqrt();
    let sigma_minor = lambda_minor.sqrt();
    let radius_m = if is_degenerate(lambda_major, lambda_minor) {
        one_dimensional_radius(sigma_major, probability)?
    } else {
        bisect_radius(
            probability,
            sigma_major * (-2.0 * libm::log(1.0 - probability)).sqrt() + 6.0 * sigma_major,
            |radius| horizontal_probability_radius(lambda_major, lambda_minor, radius),
        )
    };

    let (approx_m, approx_valid) = if probability == 0.5 {
        let ratio = if sigma_major == 0.0 {
            1.0
        } else {
            sigma_minor / sigma_major
        };
        (
            0.6152 * sigma_major + 0.5620 * sigma_minor,
            (CEP_RGCSP_MIN_RATIO..=1.0).contains(&ratio),
        )
    } else {
        (radius_m, false)
    };
    Ok(percentile(probability, radius_m, approx_m, approx_valid))
}

fn spherical_radius_from_eigenvalues(
    eigenvalues: [f64; 3],
    probability: f64,
) -> Result<PercentileRadius, ErrorMetricsError> {
    validate_probability(probability)?;
    let values = [
        eigenvalues[0].max(0.0),
        eigenvalues[1].max(0.0),
        eigenvalues[2].max(0.0),
    ];
    let positive = values.iter().filter(|&&value| value > 0.0).count();
    if positive == 0 {
        return Ok(percentile(probability, 0.0, 0.0, true));
    }

    let radius_m = match positive {
        1 => one_dimensional_radius(values[0].sqrt(), probability)?,
        2 => horizontal_radius_from_eigenvalues([values[0], values[1]], probability)?.radius_m,
        _ if nearly_equal(values[0], values[2]) && probability == 0.5 => {
            SEP_ISOTROPIC_MEDIAN * values[0].sqrt()
        }
        _ if nearly_equal(values[0], values[2]) => maxwell_radius(values[0].sqrt(), probability),
        _ => {
            let sigma_major = values[0].sqrt();
            bisect_radius(
                probability,
                sigma_major * (-2.0 * libm::log(1.0 - probability)).sqrt() + 6.0 * sigma_major,
                |radius| spherical_probability_radius(values, radius),
            )
        }
    };

    let sigmas = [values[0].sqrt(), values[1].sqrt(), values[2].sqrt()];
    let ratio = if sigmas[0] == 0.0 {
        1.0
    } else {
        sigmas[2] / sigmas[0]
    };
    let approx_m = if probability == 0.5 {
        0.51 * (sigmas[0] + sigmas[1] + sigmas[2])
    } else {
        radius_m
    };
    let approx_valid = probability == 0.5 && ratio >= SEP_APPROX_MIN_RATIO;
    Ok(percentile(probability, radius_m, approx_m, approx_valid))
}

fn horizontal_probability_radius(lambda_major: f64, lambda_minor: f64, radius: f64) -> f64 {
    if radius < 0.0 {
        return 0.0;
    }
    if lambda_major == 0.0 {
        return 1.0;
    }
    if is_degenerate(lambda_major, lambda_minor) {
        let sigma = lambda_major.sqrt();
        return erf(radius / (core::f64::consts::SQRT_2 * sigma));
    }

    let sigma_major = lambda_major.sqrt();
    let sigma_minor = lambda_minor.sqrt();
    if sigma_minor / sigma_major < 0.1 {
        return erf(radius / (core::f64::consts::SQRT_2 * sigma_major));
    }
    let quarter = integrate_gl64(0.0, core::f64::consts::FRAC_PI_2, |phi| {
        let cos_phi = libm::cos(phi);
        let sin_phi = libm::sin(phi);
        let guarded_minor = lambda_minor.max(lambda_major * DEGENERATE_REL_EPS);
        let g = cos_phi * cos_phi / lambda_major + sin_phi * sin_phi / guarded_minor;
        libm::exp(-0.5 * radius * radius * g) / g
    });
    let integral = 4.0 * quarter;
    let outside = integral / (2.0 * core::f64::consts::PI * sigma_major * sigma_minor);
    (1.0 - outside).clamp(0.0, 1.0)
}

fn spherical_probability_radius(eigenvalues: [f64; 3], radius: f64) -> f64 {
    if radius < 0.0 {
        return 0.0;
    }
    let sigmas = [
        eigenvalues[0].sqrt(),
        eigenvalues[1].sqrt(),
        eigenvalues[2].sqrt(),
    ];
    let theta_integral = integrate_gl64(0.0, core::f64::consts::FRAC_PI_2, |theta| {
        let sin_theta = libm::sin(theta);
        let cos_theta = libm::cos(theta);
        integrate_gl64(0.0, core::f64::consts::FRAC_PI_2, |phi| {
            let cos_phi = libm::cos(phi);
            let sin_phi = libm::sin(phi);
            let u0 = sin_theta * cos_phi;
            let u1 = sin_theta * sin_phi;
            let u2 = cos_theta;
            let h = u0 * u0 / eigenvalues[0] + u1 * u1 / eigenvalues[1] + u2 * u2 / eigenvalues[2];
            radial_integral(radius, h) * sin_theta
        })
    });
    let shell_integral = 8.0 * theta_integral;
    let norm = libm::pow(2.0 * core::f64::consts::PI, -1.5) / (sigmas[0] * sigmas[1] * sigmas[2]);
    (norm * shell_integral).clamp(0.0, 1.0)
}

fn radial_integral(radius: f64, h: f64) -> f64 {
    let a = 0.5 * h;
    let sqrt_a = a.sqrt();
    let ar2 = a * radius * radius;
    core::f64::consts::PI.sqrt() * erf(radius * sqrt_a) / (4.0 * a * sqrt_a)
        - radius * libm::exp(-ar2) / (2.0 * a)
}

fn one_dimensional_radius(sigma: f64, probability: f64) -> Result<f64, ErrorMetricsError> {
    if sigma == 0.0 {
        return Ok(0.0);
    }
    let inv = erfc_inv(1.0 - probability).ok_or(ErrorMetricsError::InvalidProbability)?;
    Ok(sigma * core::f64::consts::SQRT_2 * inv)
}

fn maxwell_radius(sigma: f64, probability: f64) -> f64 {
    if sigma == 0.0 {
        return 0.0;
    }
    bisect_radius(probability, 8.0 * sigma, |radius| {
        let x = radius / sigma;
        erf(x * core::f64::consts::FRAC_1_SQRT_2)
            - (2.0 / core::f64::consts::PI).sqrt() * x * libm::exp(-0.5 * x * x)
    })
}

fn bisect_radius<F>(probability: f64, hi_initial: f64, mut cdf: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    let mut lo = 0.0;
    let mut hi = hi_initial;
    for _ in 0..8 {
        if cdf(hi) >= probability || !hi.is_finite() {
            break;
        }
        hi *= 2.0;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < probability {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn integrate_gl64<F>(a: f64, b: f64, mut f: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    debug_assert_eq!(GL_INTERVALS, GL64_POSITIVE_NODES.len() * 2);
    let center = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut sum = 0.0;
    for idx in 0..GL64_POSITIVE_NODES.len() {
        let node = GL64_POSITIVE_NODES[idx];
        let weight = GL64_POSITIVE_WEIGHTS[idx];
        sum += weight * (f(center - half * node) + f(center + half * node));
    }
    half * sum
}

fn horizontal_eigenvalues(covariance: [[f64; 3]; 3]) -> Result<[f64; 2], ErrorMetricsError> {
    let block = [
        [covariance[0][0], covariance[0][1]],
        [covariance[1][0], covariance[1][1]],
    ];
    let ellipse = integrity::error_ellipse_2x2_unit(block).map_err(map_integrity_error)?;
    Ok([ellipse.semi_major.powi(2), ellipse.semi_minor.powi(2)])
}

fn eigenvalues_symmetric_3x3(covariance: [[f64; 3]; 3]) -> Result<[f64; 3], ErrorMetricsError> {
    let a00 = covariance[0][0];
    let a11 = covariance[1][1];
    let a22 = covariance[2][2];
    let a01 = 0.5 * (covariance[0][1] + covariance[1][0]);
    let a02 = 0.5 * (covariance[0][2] + covariance[2][0]);
    let a12 = 0.5 * (covariance[1][2] + covariance[2][1]);
    let p1 = a01 * a01 + a02 * a02 + a12 * a12;
    let mut values = if p1 == 0.0 {
        [a00, a11, a22]
    } else {
        let q = (a00 + a11 + a22) / 3.0;
        let b00 = a00 - q;
        let b11 = a11 - q;
        let b22 = a22 - q;
        let p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();
        let inv_p = 1.0 / p;
        let b = [
            [b00 * inv_p, a01 * inv_p, a02 * inv_p],
            [a01 * inv_p, b11 * inv_p, a12 * inv_p],
            [a02 * inv_p, a12 * inv_p, b22 * inv_p],
        ];
        let r = (det3(b) / 2.0).clamp(-1.0, 1.0);
        let phi = libm::acos(r) / 3.0;
        let lambda1 = q + 2.0 * p * libm::cos(phi);
        let lambda3 = q + 2.0 * p * libm::cos(phi + 2.0 * core::f64::consts::PI / 3.0);
        let lambda2 = a00 + a11 + a22 - lambda1 - lambda3;
        [lambda1, lambda2, lambda3]
    };
    values.sort_by(|a, b| b.total_cmp(a));
    if values[2] < -PSD_MINOR_EPS {
        return Err(ErrorMetricsError::NotPositiveSemidefinite);
    }
    Ok(values)
}

fn validate_enu_covariance(covariance: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], ErrorMetricsError> {
    validate_finite_matrix(covariance)?;
    let scale = covariance_scale(&covariance);
    let symmetry_tol = SYMMETRY_EPS * scale.max(1.0);
    for i in 0..3 {
        if covariance[i][i] < -PSD_DIAGONAL_EPS {
            return Err(ErrorMetricsError::NotPositiveSemidefinite);
        }
        for j in (i + 1)..3 {
            if (covariance[i][j] - covariance[j][i]).abs() > symmetry_tol {
                return Err(ErrorMetricsError::NotPositiveSemidefinite);
            }
        }
    }
    eigenvalues_symmetric_3x3(covariance)?;
    Ok(covariance)
}

fn validate_finite_matrix(covariance: [[f64; 3]; 3]) -> Result<(), ErrorMetricsError> {
    for row in covariance {
        if row.iter().any(|value| !value.is_finite()) {
            return Err(ErrorMetricsError::NonFinite);
        }
    }
    Ok(())
}

fn validate_probability(probability: f64) -> Result<(), ErrorMetricsError> {
    if probability.is_finite() && (0.0..1.0).contains(&probability) {
        Ok(())
    } else {
        Err(ErrorMetricsError::InvalidProbability)
    }
}

fn covariance_scale(covariance: &[[f64; 3]; 3]) -> f64 {
    covariance
        .iter()
        .flatten()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()))
}

fn det3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn is_degenerate(lambda_major: f64, lambda_minor: f64) -> bool {
    lambda_minor <= lambda_major * DEGENERATE_REL_EPS
}

fn nearly_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= (a.abs().max(b.abs()).max(1.0) * 1.0e-14)
}

fn percentile(
    probability: f64,
    radius_m: f64,
    approx_m: f64,
    approx_valid: bool,
) -> PercentileRadius {
    PercentileRadius {
        probability,
        radius_m,
        approx_m,
        approx_valid,
    }
}

fn map_integrity_error(error: integrity::IntegrityError) -> ErrorMetricsError {
    match error {
        integrity::IntegrityError::NonFinite => ErrorMetricsError::NonFinite,
        integrity::IntegrityError::NotPositiveSemidefinite => {
            ErrorMetricsError::NotPositiveSemidefinite
        }
        integrity::IntegrityError::InvalidProbability { .. } => {
            ErrorMetricsError::InvalidProbability
        }
        integrity::IntegrityError::InvalidInput { .. } | integrity::IntegrityError::Singular => {
            ErrorMetricsError::NotPositiveSemidefinite
        }
    }
}

fn map_dop_ellipse_error(error: DopError) -> ErrorMetricsError {
    match error {
        DopError::InvalidInput {
            reason: "not finite",
            ..
        } => ErrorMetricsError::NonFinite,
        DopError::InvalidInput { .. } => ErrorMetricsError::NotPositiveSemidefinite,
        DopError::TooFewSatellites | DopError::Singular => {
            ErrorMetricsError::NotPositiveSemidefinite
        }
    }
}

#[cfg(test)]
mod tests {
    //! Validation references: Rayleigh horizontal radius, Maxwell 3D radius,
    //! normal one-dimensional radius, and published CEP linear approximations.

    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "actual={actual} expected={expected} diff={}",
            (actual - expected).abs()
        );
    }

    fn assert_rel(actual: f64, expected: f64, rel: f64) {
        let tol = expected.abs() * rel;
        assert_close(actual, expected, tol);
    }

    #[test]
    fn isotropic_rayleigh_constants_match_closed_form() {
        let sigma = 2.5;
        let cov = [
            [sigma * sigma, 0.0, 0.0],
            [0.0, sigma * sigma, 0.0],
            [0.0, 0.0, sigma * sigma],
        ];
        let metrics = metrics_from_enu_covariance_m2(cov).expect("metrics");
        assert_rel(metrics.cep_m.radius_m, 1.177_410 * sigma, 1.0e-6);
        assert_rel(metrics.r95_m.radius_m, 2.447_747 * sigma, 1.0e-6);
        assert_rel(metrics.r99_m.radius_m, 3.034_854 * sigma, 1.0e-6);
        assert_rel(metrics.drms_m, core::f64::consts::SQRT_2 * sigma, 1.0e-6);
        assert_rel(
            metrics.two_drms_m,
            2.0 * core::f64::consts::SQRT_2 * sigma,
            1.0e-6,
        );
        assert_eq!(metrics.ellipse.orientation_rad.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn hand_eigen_horizontal_ellipse_matches_closed_form() {
        let cov = [[4.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let ellipse = error_ellipse_from_enu_m2(cov).expect("ellipse");
        assert_close(
            ellipse.semi_major_m,
            (3.0_f64 + 2.0_f64.sqrt()).sqrt(),
            1.0e-12,
        );
        assert_close(
            ellipse.semi_minor_m,
            (3.0_f64 - 2.0_f64.sqrt()).sqrt(),
            1.0e-12,
        );
        assert_close(
            ellipse.orientation_rad,
            core::f64::consts::PI / 8.0,
            1.0e-12,
        );
    }

    #[test]
    fn two_drms_coverage_spans_isotropic_to_elongated_cases() {
        let isotropic_r = 2.0 * (2.0_f64).sqrt();
        let isotropic = horizontal_probability_radius(1.0, 1.0, isotropic_r);
        assert_close(isotropic, 1.0 - libm::exp(-4.0_f64), 1.0e-12);

        let elongated_r = 2.0 * (100.0_f64 + 1.0e-4).sqrt();
        let elongated = horizontal_probability_radius(100.0, 1.0e-4, elongated_r);
        assert!((0.954..0.955).contains(&elongated), "elongated={elongated}");
    }

    #[test]
    fn isotropic_three_dimensional_constants_match_spec_values() {
        let sigma = 3.0;
        let cov = [
            [sigma * sigma, 0.0, 0.0],
            [0.0, sigma * sigma, 0.0],
            [0.0, 0.0, sigma * sigma],
        ];
        let metrics = metrics_from_enu_covariance_m2(cov).expect("metrics");
        assert_close(metrics.sep_m.radius_m, 1.537_580 * sigma, 1.0e-6);
        assert_close(metrics.mrse_m, 1.732_051 * sigma, 1.0e-6);
        assert_close(metrics.vep_m, 0.674_490 * sigma, 1.0e-6);
    }

    #[test]
    fn anisotropic_cep_reports_exact_and_validity_flag() {
        let cov = [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let cep = horizontal_radius_at(cov, 0.5).expect("cep");
        assert_close(cep.radius_m, 1.740_834_856_488_325, 1.0e-12);
        assert!(cep.approx_valid);

        let thin = [[4.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]];
        let thin_cep = horizontal_radius_at(thin, 0.5).expect("thin cep");
        assert!(!thin_cep.approx_valid);
        assert_close(thin_cep.radius_m, 1.348_980, 1.0e-3);
    }

    #[test]
    fn quadrature_recovers_isotropic_rayleigh_closed_form() {
        let sigma = 4.0;
        let probability = 0.95_f64;
        let radius = sigma * (-2.0_f64 * libm::log(1.0 - probability)).sqrt();
        let got = horizontal_probability_radius(sigma * sigma, sigma * sigma, radius);
        assert_close(got, probability, 1.0e-12);
    }

    #[test]
    fn degenerate_and_bad_covariances_do_not_produce_tight_numbers() {
        let rank1 = [[9.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let cep = horizontal_radius_at(rank1, 0.5).expect("rank one cep");
        let ellipse = error_ellipse_from_enu_m2(rank1).expect("rank one ellipse");
        assert_close(cep.radius_m, 2.023_469, 1.0e-6);
        assert_eq!(ellipse.semi_minor_m.to_bits(), 0.0_f64.to_bits());

        let non_psd = [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(
            horizontal_radius_at(non_psd, 0.5),
            Err(ErrorMetricsError::NotPositiveSemidefinite)
        );

        let huge = [[1.0e12, 0.0, 0.0], [0.0, 1.0e12, 0.0], [0.0, 0.0, 1.0e12]];
        let huge_metrics = metrics_from_enu_covariance_m2(huge).expect("huge finite");
        assert!(huge_metrics.cep_m.radius_m.is_finite());
        assert!(huge_metrics.cep_m.radius_m > 1.0e6);

        let non_finite = [[f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(
            metrics_from_enu_covariance_m2(non_finite),
            Err(ErrorMetricsError::NonFinite)
        );
    }

    #[test]
    fn horizontal_ellipse_uses_same_unit_ellipse_source_as_dop() {
        let cov = [[16.0, 3.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 9.0]];
        let metrics_ellipse = error_ellipse_from_enu_m2(cov).expect("metrics ellipse");
        let dop_ellipse =
            dop::error_ellipse_2x2_unit([[16.0, 3.0], [3.0, 4.0]]).expect("dop ellipse");
        assert_eq!(
            metrics_ellipse.semi_major_m.to_bits(),
            dop_ellipse.semi_major.to_bits()
        );
    }
}
