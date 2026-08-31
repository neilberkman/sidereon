//! WGS84 geodesic direct and inverse helpers.
//!
//! The computations implement the order-6 series from Karney (2013),
//! "Algorithms for geodesics", on the WGS84 ellipsoid. Public angles are
//! degrees and distances are meters.

use crate::constants::{DEG_TO_RAD, RAD_TO_DEG, WGS84_A_M, WGS84_F};

const SERIES_ORDER: usize = 6;
const HALF_TURN_RAD: f64 = std::f64::consts::PI;
const HALF_TURN_DEG: f64 = 180.0;
const TURN_DEG: f64 = 360.0;
const TINY: f64 = 1.491_668_146_240_041_3e-154;
const NEWTON_MAX: usize = 100;
const NEWTON_FAST_MAX: usize = 20;
const WGS84_B_M: f64 = WGS84_A_M * (1.0 - WGS84_F);
const WGS84_ONE_MINUS_F: f64 = 1.0 - WGS84_F;
const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);
const WGS84_N: f64 = WGS84_F / (2.0 - WGS84_F);
const WGS84_EP2: f64 = WGS84_F * (2.0 - WGS84_F) / ((1.0 - WGS84_F) * (1.0 - WGS84_F));

/// Error returned when geodesic inputs are outside the accepted domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GeodesicError {
    /// A geodesic input was non-finite or outside its documented range.
    #[error("invalid geodesic input {field}: {reason}")]
    InvalidInput {
        /// Name of the invalid field.
        field: &'static str,
        /// Reason the field was rejected.
        reason: &'static str,
    },
}

#[derive(Debug, Clone, Copy)]
struct Series {
    a1: f64,
    c1: [f64; SERIES_ORDER],
    c1p: [f64; SERIES_ORDER],
    a3: f64,
    c3: [f64; SERIES_ORDER - 1],
}

#[derive(Debug, Clone, Copy)]
struct HybridSolution {
    residual_rad: f64,
    distance_m: f64,
    salp1: f64,
    calp1: f64,
    salp2: f64,
    calp2: f64,
    derivative_rad: f64,
}

#[derive(Debug, Clone, Copy)]
struct Lengths {
    distance_over_b: f64,
    reduced_length_over_b: f64,
}

#[derive(Debug, Clone, Copy)]
struct ReducedPoint {
    sbet: f64,
    cbet: f64,
    dn: f64,
}

#[derive(Debug, Clone, Copy)]
struct SigmaPoint {
    ssig: f64,
    csig: f64,
    dn: f64,
}

#[derive(Debug, Clone, Copy)]
struct LongitudeTarget {
    slam: f64,
    clam: f64,
}

fn invalid_input(field: &'static str, reason: &'static str) -> GeodesicError {
    GeodesicError::InvalidInput { field, reason }
}

fn validate_latitude(field: &'static str, value: f64) -> Result<(), GeodesicError> {
    if !value.is_finite() {
        return Err(invalid_input(field, "must be finite"));
    }
    if !(-90.0..=90.0).contains(&value) {
        return Err(invalid_input(field, "must be in [-90, 90] degrees"));
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), GeodesicError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "must be finite"))
    }
}

fn sq(value: f64) -> f64 {
    value * value
}

fn hypot(value1: f64, value2: f64) -> f64 {
    libm::hypot(value1, value2)
}

fn sin_cos_deg(value: f64) -> (f64, f64) {
    let value = angle_round(value);
    let mut reduced_deg = value % TURN_DEG;
    let quadrant = (reduced_deg / 90.0).round();
    reduced_deg -= 90.0 * quadrant;
    let reduced = reduced_deg * DEG_TO_RAD;
    let (sin_reduced, cos_reduced) = libm::sincos(reduced);
    let (sin_reduced, cos_reduced) = if reduced_deg.abs() == 45.0 {
        let value = 0.5_f64.sqrt();
        (value.copysign(reduced), value)
    } else if reduced_deg.abs() == 30.0 {
        (0.5_f64.copysign(reduced), 0.75_f64.sqrt())
    } else {
        (sin_reduced, cos_reduced)
    };
    match (quadrant as i64).rem_euclid(4) {
        0 => (sin_reduced, cos_reduced),
        1 => (cos_reduced, -sin_reduced),
        2 => (-sin_reduced, -cos_reduced),
        _ => (-cos_reduced, sin_reduced),
    }
}

fn angle_round(value: f64) -> f64 {
    let z = 1.0 / 16.0;
    let y = value.abs();
    let rounded = if y < z { z - (z - y) } else { y };
    rounded.copysign(value)
}

fn normalize_pair(y: f64, x: f64) -> (f64, f64) {
    let r = hypot(y, x);
    if r == 0.0 {
        (0.0, 1.0)
    } else {
        (y / r, x / r)
    }
}

fn reduced_lat_sin_cos_deg(phi_deg: f64) -> (f64, f64) {
    let (sin_phi, cos_phi) = sin_cos_deg(phi_deg);
    let (sin_beta, cos_beta) = normalize_pair((1.0 - WGS84_F) * sin_phi, cos_phi);
    (sin_beta, cos_beta.abs().max(TINY))
}

fn atan2_deg(y: f64, x: f64) -> f64 {
    let mut y = y;
    let mut x = x;
    let mut quadrant = 0;
    if y.abs() > x.abs() {
        std::mem::swap(&mut y, &mut x);
        quadrant = 2;
    }
    if x.is_sign_negative() {
        x = -x;
        quadrant += 1;
    }
    let mut angle = libm::atan2(y, x) * RAD_TO_DEG;
    match quadrant {
        1 => angle = HALF_TURN_DEG.copysign(y) - angle,
        2 => angle = 90.0 - angle,
        3 => angle -= 90.0,
        _ => {}
    }
    normalize_angle_deg(angle)
}

fn normalize_angle_deg(value: f64) -> f64 {
    let normalized = (value + HALF_TURN_DEG).rem_euclid(TURN_DEG) - HALF_TURN_DEG;
    if normalized == -HALF_TURN_DEG {
        HALF_TURN_DEG
    } else {
        normalized
    }
}

fn normalize_lon_deg(value: f64) -> f64 {
    if value > -HALF_TURN_DEG && value <= HALF_TURN_DEG {
        value
    } else {
        normalize_angle_deg(value)
    }
}

fn longitude_delta_rad(lon1_deg: f64, lon2_deg: f64) -> f64 {
    normalize_angle_deg(lon2_deg - lon1_deg) * DEG_TO_RAD
}

fn reduced_point_deg(phi_deg: f64) -> ReducedPoint {
    let (sbet, cbet) = reduced_lat_sin_cos_deg(phi_deg);
    ReducedPoint {
        sbet,
        cbet,
        dn: (1.0 + WGS84_EP2 * sq(sbet)).sqrt(),
    }
}

fn series_from_salp0(salp0: f64) -> Series {
    let calp0 = (1.0 - sq(salp0)).max(0.0).sqrt();
    let k2 = WGS84_EP2 * sq(calp0);
    let root = (1.0 + k2).sqrt();
    let eps = k2 / (2.0 * (1.0 + root) + k2);
    let a1 = 1.0 + a1_minus1(eps);
    let c1 = c1_coefficients(eps);
    let c1p = c1p_coefficients(eps);
    let a3x = a3_polynomial_coefficients();
    let c3x = c3_polynomial_coefficients();
    let a3 = polyval(a3x.len() - 1, &a3x, 0, eps);
    let c3 = c3_coefficients(eps, &c3x);

    Series {
        a1,
        c1,
        c1p,
        a3,
        c3,
    }
}

fn polyval(order: usize, coeffs: &[f64], offset: usize, x: f64) -> f64 {
    let mut value = coeffs[offset];
    for idx in 1..=order {
        value = value * x + coeffs[offset + idx];
    }
    value
}

fn a1_minus1(eps: f64) -> f64 {
    let eps2 = sq(eps);
    let coeffs = [1.0, 4.0, 64.0, 0.0, 256.0];
    let value = polyval(3, &coeffs, 0, eps2) / coeffs[4];
    (value + eps) / (1.0 - eps)
}

fn c1_coefficients(eps: f64) -> [f64; SERIES_ORDER] {
    let coeffs = [
        -1.0, 6.0, -16.0, 32.0, -9.0, 64.0, -128.0, 2048.0, 9.0, -16.0, 768.0, 3.0, -5.0, 512.0,
        -7.0, 1280.0, -7.0, 2048.0,
    ];
    coefficient_series(eps, &coeffs)
}

fn c1p_coefficients(eps: f64) -> [f64; SERIES_ORDER] {
    let coeffs = [
        205.0, -432.0, 768.0, 1536.0, 4005.0, -4736.0, 3840.0, 12288.0, -225.0, 116.0, 384.0,
        -7173.0, 2695.0, 7680.0, 3467.0, 7680.0, 38081.0, 61440.0,
    ];
    coefficient_series(eps, &coeffs)
}

fn a2_minus1(eps: f64) -> f64 {
    let eps2 = sq(eps);
    let coeffs = [-11.0, -28.0, -192.0, 0.0, 256.0];
    let value = polyval(3, &coeffs, 0, eps2) / coeffs[4];
    (value - eps) / (1.0 + eps)
}

fn c2_coefficients(eps: f64) -> [f64; SERIES_ORDER] {
    let coeffs = [
        1.0, 2.0, 16.0, 32.0, 35.0, 64.0, 384.0, 2048.0, 15.0, 80.0, 768.0, 7.0, 35.0, 512.0, 63.0,
        1280.0, 77.0, 2048.0,
    ];
    coefficient_series(eps, &coeffs)
}

fn coefficient_series(eps: f64, coeffs: &[f64]) -> [f64; SERIES_ORDER] {
    let eps2 = sq(eps);
    let mut result = [0.0; SERIES_ORDER];
    let mut scale = eps;
    let mut offset = 0;
    for l in 1..=SERIES_ORDER {
        let order = (SERIES_ORDER - l) / 2;
        result[l - 1] = scale * polyval(order, coeffs, offset, eps2) / coeffs[offset + order + 1];
        offset += order + 2;
        scale *= eps;
    }
    result
}

fn a3_polynomial_coefficients() -> [f64; SERIES_ORDER] {
    let coeffs = [
        -3.0, 128.0, -2.0, -3.0, 64.0, -1.0, -3.0, -1.0, 16.0, 3.0, -1.0, -2.0, 8.0, 1.0, -1.0,
        2.0, 1.0, 1.0,
    ];
    let mut result = [0.0; SERIES_ORDER];
    let mut offset = 0;
    for (out, j) in (0..SERIES_ORDER).rev().enumerate() {
        let order = (SERIES_ORDER - j - 1).min(j);
        result[out] = polyval(order, &coeffs, offset, WGS84_N) / coeffs[offset + order + 1];
        offset += order + 2;
    }
    result
}

fn c3_polynomial_coefficients() -> [f64; SERIES_ORDER * (SERIES_ORDER - 1) / 2] {
    let coeffs = [
        3.0, 128.0, 2.0, 5.0, 128.0, -1.0, 3.0, 3.0, 64.0, -1.0, 0.0, 1.0, 8.0, -1.0, 1.0, 4.0,
        5.0, 256.0, 1.0, 3.0, 128.0, -3.0, -2.0, 3.0, 64.0, 1.0, -3.0, 2.0, 32.0, 7.0, 512.0,
        -10.0, 9.0, 384.0, 5.0, -9.0, 5.0, 192.0, 7.0, 512.0, -14.0, 7.0, 512.0, 21.0, 2560.0,
    ];
    let mut result = [0.0; SERIES_ORDER * (SERIES_ORDER - 1) / 2];
    let mut offset = 0;
    let mut out = 0;
    for l in 1..SERIES_ORDER {
        for j in (l..SERIES_ORDER).rev() {
            let order = (SERIES_ORDER - j - 1).min(j);
            result[out] = polyval(order, &coeffs, offset, WGS84_N) / coeffs[offset + order + 1];
            out += 1;
            offset += order + 2;
        }
    }
    result
}

fn c3_coefficients(
    eps: f64,
    coeffs: &[f64; SERIES_ORDER * (SERIES_ORDER - 1) / 2],
) -> [f64; SERIES_ORDER - 1] {
    let mut result = [0.0; SERIES_ORDER - 1];
    let mut scale = 1.0;
    let mut offset = 0;
    for l in 1..SERIES_ORDER {
        let order = SERIES_ORDER - l - 1;
        scale *= eps;
        result[l - 1] = scale * polyval(order, coeffs, offset, eps);
        offset += order + 1;
    }
    result
}

fn sine_series(coeffs: &[f64], sigma: f64) -> f64 {
    let (sin_sigma, cos_sigma) = libm::sincos(sigma);
    sine_series_sin_cos(coeffs, sin_sigma, cos_sigma)
}

fn sine_series_sin_cos(coeffs: &[f64], sin_sigma: f64, cos_sigma: f64) -> f64 {
    let two_cos = 2.0 * (cos_sigma - sin_sigma) * (cos_sigma + sin_sigma);
    let mut k = coeffs.len() + 1;
    let mut n = coeffs.len();
    let mut y1 = 0.0;
    let mut y0;
    if n & 1 == 1 {
        k -= 1;
        y0 = coeffs[k - 1];
    } else {
        y0 = 0.0;
    }
    n /= 2;
    while n > 0 {
        n -= 1;
        k -= 1;
        y1 = two_cos * y0 - y1 + coeffs[k - 1];
        k -= 1;
        y0 = if k == 0 {
            two_cos * y1 - y0
        } else {
            two_cos * y1 - y0 + coeffs[k - 1]
        };
    }
    2.0 * sin_sigma * cos_sigma * y0
}

fn i1(series: Series, sigma: f64) -> f64 {
    series.a1 * (sigma + sine_series(&series.c1, sigma))
}

fn sigma1_sin_cos(sbet1: f64, cbet1: f64, salp1: f64, calp1: f64) -> (f64, f64, f64) {
    let (salp0, calp0) = normalize_pair(salp1 * cbet1, hypot(calp1, salp1 * sbet1));
    let sigma1 = libm::atan2(sbet1, calp1 * cbet1);
    (sigma1, salp0, calp0)
}

fn direct_raw(sbet1: f64, cbet1: f64, salp1: f64, calp1: f64, s12_m: f64) -> (f64, f64, f64) {
    let (_sigma1, salp0, calp0) = sigma1_sin_cos(sbet1, cbet1, salp1, calp1);
    let series = series_from_salp0(salp0);
    let (ssig1, csig1) = normalize_pair(
        sbet1,
        if sbet1 != 0.0 || calp1 != 0.0 {
            cbet1 * calp1
        } else {
            1.0
        },
    );
    let b11 = sine_series_sin_cos(&series.c1, ssig1, csig1);
    let (sin_b11, cos_b11) = libm::sincos(b11);
    let stau1 = ssig1 * cos_b11 + csig1 * sin_b11;
    let ctau1 = csig1 * cos_b11 - ssig1 * sin_b11;
    let tau12 = s12_m / (WGS84_B_M * series.a1);
    let (stau12, ctau12) = libm::sincos(tau12);
    let stau2 = stau1 * ctau12 + ctau1 * stau12;
    let ctau2 = ctau1 * ctau12 - stau1 * stau12;
    let b12 = -sine_series_sin_cos(&series.c1p, stau2, ctau2);
    let sig12 = tau12 - (b12 - b11);
    let (ssig12, csig12) = libm::sincos(sig12);
    let ssig2 = ssig1 * csig12 + csig1 * ssig12;
    let csig2 = csig1 * csig12 - ssig1 * ssig12;

    let sbet2 = calp0 * ssig2;
    let cbet2 = hypot(salp0, calp0 * csig2);
    let lat2_rad = libm::atan2(sbet2, (1.0 - WGS84_F) * cbet2);
    let azi2_rad = libm::atan2(salp0, calp0 * csig2);
    let somg1 = salp0 * ssig1;
    let comg1 = csig1;
    let somg2 = salp0 * ssig2;
    let comg2 = csig2;
    let omg12 = libm::atan2(somg2 * comg1 - comg2 * somg1, comg2 * comg1 + somg2 * somg1);
    let b31 = sine_series_sin_cos(&series.c3, ssig1, csig1);
    let b32 = sine_series_sin_cos(&series.c3, ssig2, csig2);
    let lambda12 = omg12 - WGS84_F * salp0 * series.a3 * (sig12 + b32 - b31);

    (lat2_rad, lambda12, azi2_rad)
}

fn inverse_lengths(eps: f64, sig12: f64, point1: SigmaPoint, point2: SigmaPoint) -> Lengths {
    let a1_minus1 = a1_minus1(eps);
    let a1 = 1.0 + a1_minus1;
    let c1 = c1_coefficients(eps);
    let b1 = sine_series_sin_cos(&c1, point2.ssig, point2.csig)
        - sine_series_sin_cos(&c1, point1.ssig, point1.csig);

    let a2_minus1 = a2_minus1(eps);
    let a2 = 1.0 + a2_minus1;
    let c2 = c2_coefficients(eps);
    let b2 = sine_series_sin_cos(&c2, point2.ssig, point2.csig)
        - sine_series_sin_cos(&c2, point1.ssig, point1.csig);
    let j12 = (a1_minus1 - a2_minus1) * sig12 + (a1 * b1 - a2 * b2);

    Lengths {
        distance_over_b: a1 * (sig12 + b1),
        reduced_length_over_b: point2.dn * (point1.csig * point2.ssig)
            - point1.dn * (point1.ssig * point2.csig)
            - point1.csig * point2.csig * j12,
    }
}

fn hybrid_solution_from_pair(
    point1: ReducedPoint,
    point2: ReducedPoint,
    mut salp1: f64,
    mut calp1: f64,
    target: LongitudeTarget,
) -> HybridSolution {
    if point1.sbet == 0.0 && calp1 == 0.0 {
        calp1 = -TINY;
    }
    (salp1, calp1) = normalize_pair(salp1, calp1);
    let salp0 = salp1 * point1.cbet;
    let calp0 = hypot(calp1, salp1 * point1.sbet);
    let mut ssig1 = point1.sbet;
    let mut csig1 = calp1 * point1.cbet;
    let somg1 = salp0 * point1.sbet;
    let comg1 = csig1;
    (ssig1, csig1) = normalize_pair(ssig1, csig1);

    let salp2 = if point2.cbet != point1.cbet {
        salp0 / point2.cbet
    } else {
        salp1
    };
    let calp2 = if point2.cbet != point1.cbet || point2.sbet.abs() != -point1.sbet {
        let sensitive = if point1.cbet < -point1.sbet {
            (point2.cbet - point1.cbet) * (point1.cbet + point2.cbet)
        } else {
            (point1.sbet - point2.sbet) * (point1.sbet + point2.sbet)
        };
        (sq(calp1 * point1.cbet) + sensitive).max(0.0).sqrt() / point2.cbet
    } else {
        calp1.abs()
    };
    let mut ssig2 = point2.sbet;
    let mut csig2 = calp2 * point2.cbet;
    let somg2 = salp0 * point2.sbet;
    let comg2 = csig2;
    (ssig2, csig2) = normalize_pair(ssig2, csig2);

    let sig12 = libm::atan2(
        (csig1 * ssig2 - ssig1 * csig2).max(0.0),
        csig1 * csig2 + ssig1 * ssig2,
    );
    let somg12 = (comg1 * somg2 - somg1 * comg2).max(0.0);
    let comg12 = comg1 * comg2 + somg1 * somg2;
    let k2 = WGS84_EP2 * sq(calp0);
    let root = (1.0 + k2).sqrt();
    let eps = k2 / (2.0 * (1.0 + root) + k2);
    let series = series_from_salp0(salp0);
    let b3 = sine_series_sin_cos(&series.c3, ssig2, csig2)
        - sine_series_sin_cos(&series.c3, ssig1, csig1);
    let eta = libm::atan2(
        somg12 * target.clam - comg12 * target.slam,
        comg12 * target.clam + somg12 * target.slam,
    );
    let residual_rad = eta - WGS84_F * salp0 * series.a3 * (sig12 + b3);
    let sigma1 = SigmaPoint {
        ssig: ssig1,
        csig: csig1,
        dn: point1.dn,
    };
    let sigma2 = SigmaPoint {
        ssig: ssig2,
        csig: csig2,
        dn: point2.dn,
    };
    let lengths = inverse_lengths(eps, sig12, sigma1, sigma2);
    let derivative_rad = if calp2 == 0.0 {
        -2.0 * WGS84_ONE_MINUS_F * point1.dn / point1.sbet
    } else {
        lengths.reduced_length_over_b * WGS84_ONE_MINUS_F / (calp2 * point2.cbet)
    };
    let distance_m = WGS84_B_M * lengths.distance_over_b;
    HybridSolution {
        residual_rad,
        distance_m,
        salp1,
        calp1,
        salp2,
        calp2,
        derivative_rad,
    }
}

fn hybrid_solution(
    point1: ReducedPoint,
    point2: ReducedPoint,
    alpha1_rad: f64,
    lambda12_rad: f64,
) -> HybridSolution {
    let (salp1, calp1) = libm::sincos(alpha1_rad);
    let (slam12, clam12) = libm::sincos(lambda12_rad);
    hybrid_solution_from_pair(
        point1,
        point2,
        salp1,
        calp1,
        LongitudeTarget {
            slam: slam12,
            clam: clam12,
        },
    )
}

fn equatorial_inverse(lambda12_rad: f64) -> Option<HybridSolution> {
    if lambda12_rad <= (1.0 - WGS84_F) * HALF_TURN_RAD {
        Some(HybridSolution {
            residual_rad: 0.0,
            distance_m: WGS84_A_M * lambda12_rad,
            salp1: 1.0,
            calp1: 0.0,
            salp2: 1.0,
            calp2: 0.0,
            derivative_rad: f64::NAN,
        })
    } else if (lambda12_rad - HALF_TURN_RAD).abs() <= f64::EPSILON {
        let series = series_from_salp0(0.0);
        Some(HybridSolution {
            residual_rad: 0.0,
            distance_m: WGS84_B_M * (i1(series, HALF_TURN_RAD) - i1(series, 0.0)),
            salp1: 0.0,
            calp1: 1.0,
            salp2: 0.0,
            calp2: -1.0,
            derivative_rad: f64::NAN,
        })
    } else {
        None
    }
}

fn spherical_starting_azimuth(
    point1: ReducedPoint,
    point2: ReducedPoint,
    lambda12_rad: f64,
) -> (f64, f64) {
    let cbetm = 0.5 * (point1.cbet + point2.cbet);
    let wbar = (1.0 - WGS84_E2 * sq(cbetm)).sqrt();
    let omega12 = lambda12_rad / wbar;
    let (somg12, comg12) = libm::sincos(omega12);
    let salp1 = point2.cbet * somg12;
    let calp1 = point1.cbet * point2.sbet - point1.sbet * point2.cbet * comg12;
    let (salp1, calp1) = normalize_pair(salp1, calp1);
    if salp1 > 0.0 {
        (salp1, calp1)
    } else {
        (TINY, calp1.signum())
    }
}

fn astroid_mu(x: f64, y: f64) -> f64 {
    let y2 = sq(y);
    let mut low = 0.0;
    let mut high = 1.0;
    let eval = |mu: f64| sq(x / (1.0 + mu)) + y2 / sq(mu) - 1.0;

    while eval(high) > 0.0 {
        high *= 2.0;
    }

    for _ in 0..80 {
        let mid = 0.5 * (low + high);
        if eval(mid) > 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

fn astroid_starting_azimuth(
    point1: ReducedPoint,
    point2: ReducedPoint,
    lambda12_rad: f64,
) -> Option<(f64, f64)> {
    if lambda12_rad < HALF_TURN_RAD - 0.05 {
        return None;
    }

    let scale_series = series_from_salp0(point1.cbet);
    let lam_scale = WGS84_F * HALF_TURN_RAD * point1.cbet * scale_series.a3;
    let bet_scale = lam_scale * point1.cbet;
    if lam_scale <= 0.0 || bet_scale <= 0.0 {
        return None;
    }

    let x = (lambda12_rad - HALF_TURN_RAD) / lam_scale;
    let sbet_sum = point2.sbet * point1.cbet + point2.cbet * point1.sbet;
    let cbet_sum = point1.cbet * point2.cbet - point1.sbet * point2.sbet;
    let y = libm::atan2(sbet_sum, cbet_sum) / bet_scale;

    if !(x <= 0.0 && y <= 0.0 && x >= -5.0 && y >= -5.0) {
        return None;
    }

    let (salp1, calp1) = if y == 0.0 {
        (-x, -(1.0 - sq(x)).max(0.0).sqrt())
    } else {
        let mu = astroid_mu(x, y);
        (-x / (1.0 + mu), y / mu)
    };
    Some(normalize_pair(salp1, calp1))
}

fn starting_azimuth(
    point1: ReducedPoint,
    point2: ReducedPoint,
    lambda12_rad: f64,
) -> (f64, f64, bool) {
    if let Some((salp1, calp1)) = astroid_starting_azimuth(point1, point2, lambda12_rad) {
        (salp1, calp1, true)
    } else {
        let (salp1, calp1) = spherical_starting_azimuth(point1, point2, lambda12_rad);
        (salp1, calp1, false)
    }
}

fn canonical_inverse(phi1_deg: f64, phi2_deg: f64, lambda12_rad: f64) -> HybridSolution {
    let point1 = reduced_point_deg(phi1_deg);
    let point2 = reduced_point_deg(phi2_deg);

    if lambda12_rad == 0.0 {
        return hybrid_solution(point1, point2, 0.0, lambda12_rad);
    }
    if lambda12_rad == HALF_TURN_RAD {
        return hybrid_solution(point1, point2, HALF_TURN_RAD, lambda12_rad);
    }
    if point1.sbet == 0.0 && point2.sbet == 0.0 {
        if let Some(solution) = equatorial_inverse(lambda12_rad) {
            return solution;
        }
    }
    if point2.sbet == -point1.sbet && lambda12_rad > HALF_TURN_RAD - 0.05 {
        let east_solution = hybrid_solution(point1, point2, 0.5 * HALF_TURN_RAD, lambda12_rad);
        if east_solution.residual_rad.abs() <= 1.0e-12 {
            return east_solution;
        }
    }
    if let Some(solution) = short_inverse(point1, point2, lambda12_rad) {
        return solution;
    }

    let (slam12, clam12) = libm::sincos(lambda12_rad);
    let target = LongitudeTarget {
        slam: slam12,
        clam: clam12,
    };
    let (mut salp1, mut calp1, astroid_start) = starting_azimuth(point1, point2, lambda12_rad);
    let mut salp1a = TINY;
    let mut calp1a = 1.0;
    let mut salp1b = TINY;
    let mut calp1b = -1.0;
    let mut candidate = hybrid_solution_from_pair(point1, point2, salp1, calp1, target);
    let mut tripn = false;

    for numit in 0..NEWTON_MAX {
        let solution = hybrid_solution_from_pair(point1, point2, salp1, calp1, target);
        candidate = solution;
        let residual = solution.residual_rad;
        let tolerance = if tripn {
            8.0 * f64::EPSILON
        } else {
            f64::EPSILON
        };
        let alpha_correction = if astroid_start && solution.derivative_rad > 0.0 {
            (-residual / solution.derivative_rad).abs()
        } else {
            0.0
        };
        if residual.abs() < tolerance && alpha_correction <= 8.0 * f64::EPSILON {
            break;
        }

        if residual > 0.0 {
            salp1b = salp1;
            calp1b = calp1;
        } else if residual < 0.0 {
            salp1a = salp1;
            calp1a = calp1;
        }

        if numit < NEWTON_FAST_MAX && solution.derivative_rad > 0.0 {
            let dalp1 = -residual / solution.derivative_rad;
            if dalp1.abs() < HALF_TURN_RAD {
                let (sdalp1, cdalp1) = libm::sincos(dalp1);
                let next_salp1 = salp1 * cdalp1 + calp1 * sdalp1;
                if next_salp1 > 0.0 {
                    calp1 = calp1 * cdalp1 - salp1 * sdalp1;
                    salp1 = next_salp1;
                    (salp1, calp1) = normalize_pair(salp1, calp1);
                    tripn = residual.abs() <= 16.0 * f64::EPSILON;
                    continue;
                }
            }
        }

        let next_salp1 = 0.5 * (salp1a + salp1b);
        let next_calp1 = 0.5 * (calp1a + calp1b);
        if next_salp1 == salp1 && next_calp1 == calp1 {
            break;
        }
        salp1 = next_salp1;
        calp1 = next_calp1;
        (salp1, calp1) = normalize_pair(salp1, calp1);
        tripn = false;
    }
    candidate
}

fn short_line_threshold_rad() -> f64 {
    0.1 * f64::EPSILON.sqrt()
        / (0.5 * WGS84_F.abs().max(0.001) * (1.0 - WGS84_F / 2.0).min(1.0)).sqrt()
}

fn short_inverse(
    point1: ReducedPoint,
    point2: ReducedPoint,
    lambda12_rad: f64,
) -> Option<HybridSolution> {
    let sbet12 = point2.sbet * point1.cbet - point2.cbet * point1.sbet;
    let cbet12 = point2.cbet * point1.cbet + point2.sbet * point1.sbet;
    if !(cbet12 >= 0.0 && sbet12 < 0.5 && point2.cbet * lambda12_rad < 0.5) {
        return None;
    }
    let sbetm_sum = point1.sbet + point2.sbet;
    let cbetm_sum = point1.cbet + point2.cbet;
    let sbetm2 = sq(sbetm_sum) / (sq(sbetm_sum) + sq(cbetm_sum));
    let dnm = (1.0 + WGS84_EP2 * sbetm2).sqrt();
    let omega12 = lambda12_rad / (WGS84_ONE_MINUS_F * dnm);
    let (somg12, comg12) = libm::sincos(omega12);
    let sbet12a = point2.sbet * point1.cbet + point2.cbet * point1.sbet;
    let mut salp1 = point2.cbet * somg12;
    let mut calp1 = if comg12 >= 0.0 {
        sbet12 + point2.cbet * point1.sbet * sq(somg12) / (1.0 + comg12)
    } else {
        sbet12a - point2.cbet * point1.sbet * sq(somg12) / (1.0 - comg12)
    };
    let ssig12 = hypot(salp1, calp1);
    let csig12 = point1.sbet * point2.sbet + point1.cbet * point2.cbet * comg12;
    if ssig12 >= short_line_threshold_rad() {
        return None;
    }
    (salp1, calp1) = normalize_pair(salp1, calp1);
    let mut salp2 = point1.cbet * somg12;
    let mut calp2 = sbet12
        - point1.cbet
            * point2.sbet
            * if comg12 >= 0.0 {
                sq(somg12) / (1.0 + comg12)
            } else {
                1.0 - comg12
            };
    (salp2, calp2) = normalize_pair(salp2, calp2);
    let sig12 = libm::atan2(ssig12, csig12);
    Some(HybridSolution {
        residual_rad: 0.0,
        distance_m: sig12 * WGS84_B_M * dnm,
        salp1,
        calp1,
        salp2,
        calp2,
        derivative_rad: f64::NAN,
    })
}

fn transform_azimuths(
    mut solution: HybridSolution,
    swapped: bool,
    lat_sign: f64,
    lon_sign: f64,
) -> (f64, f64) {
    if swapped {
        (solution.salp1, solution.salp2) = (-solution.salp2, -solution.salp1);
        (solution.calp1, solution.calp2) = (-solution.calp2, -solution.calp1);
    }
    solution.calp1 *= lat_sign;
    solution.calp2 *= lat_sign;
    solution.salp1 *= lon_sign;
    solution.salp2 *= lon_sign;
    (
        atan2_deg(solution.salp1, solution.calp1),
        atan2_deg(solution.salp2, solution.calp2),
    )
}

/// Solve the WGS84 inverse geodesic problem.
///
/// Inputs are point 1 latitude and longitude followed by point 2 latitude and
/// longitude, all in degrees. Longitudes may be outside `[-180, 180]`; they are
/// reduced by the geodesic solver. The returned tuple is `(s12_m, azi1_deg,
/// azi2_deg)`, where `s12_m` is the geodesic distance in meters, `azi1_deg` is
/// the forward azimuth at point 1, and `azi2_deg` is the forward azimuth at
/// point 2.
pub fn geodesic_inverse(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
) -> Result<(f64, f64, f64), GeodesicError> {
    validate_latitude("lat1_deg", lat1_deg)?;
    validate_finite("lon1_deg", lon1_deg)?;
    validate_latitude("lat2_deg", lat2_deg)?;
    validate_finite("lon2_deg", lon2_deg)?;

    let mut phi1 = lat1_deg;
    let mut phi2 = lat2_deg;
    let lambda12 = longitude_delta_rad(lon1_deg, lon2_deg);
    let mut lon_sign = if lambda12 < 0.0 { -1.0 } else { 1.0 };
    let canonical_lambda12 = lambda12.abs();

    let swapped = phi1.abs() < phi2.abs();
    if swapped {
        std::mem::swap(&mut phi1, &mut phi2);
        lon_sign = -lon_sign;
    }
    let lat_sign = if phi1 > 0.0 { -1.0 } else { 1.0 };
    phi1 *= lat_sign;
    phi2 *= lat_sign;

    let solution = canonical_inverse(phi1, phi2, canonical_lambda12);
    let (azi1_deg, azi2_deg) = transform_azimuths(solution, swapped, lat_sign, lon_sign);
    let distance_m = solution.distance_m.abs();
    Ok((distance_m, azi1_deg, azi2_deg))
}

/// Solve the WGS84 direct geodesic problem.
///
/// Inputs are point 1 latitude, longitude, forward azimuth, and geodesic
/// distance. Angles are degrees and `s12_m` is meters. Longitudes and azimuths
/// may be outside one turn; they are reduced by the geodesic solver. The
/// returned tuple is `(lat2_deg, lon2_deg, azi2_deg)`.
pub fn geodesic_direct(
    lat1_deg: f64,
    lon1_deg: f64,
    azi1_deg: f64,
    s12_m: f64,
) -> Result<(f64, f64, f64), GeodesicError> {
    validate_latitude("lat1_deg", lat1_deg)?;
    validate_finite("lon1_deg", lon1_deg)?;
    validate_finite("azi1_deg", azi1_deg)?;
    validate_finite("s12_m", s12_m)?;

    let (sbet1, cbet1) = reduced_lat_sin_cos_deg(lat1_deg);
    let (salp1, calp1) = sin_cos_deg(azi1_deg);
    let (lat2_rad, lambda12_rad, azi2_rad) = direct_raw(sbet1, cbet1, salp1, calp1, s12_m);
    let lat2_deg = lat2_rad * RAD_TO_DEG;
    let lon2_deg = normalize_lon_deg(lambda12_rad / DEG_TO_RAD + normalize_lon_deg(lon1_deg));
    let azi2_deg = normalize_angle_deg(azi2_rad * RAD_TO_DEG);
    Ok((lat2_deg, lon2_deg, azi2_deg))
}
