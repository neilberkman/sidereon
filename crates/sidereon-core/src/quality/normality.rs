//! Residual-distribution diagnostics: sample moments and normality tests.
//!
//! Post-fit residuals from a converged least-squares solve should look like
//! zero-mean Gaussian noise. These primitives quantify departures from that
//! ideal on an arbitrary residual set: sample skewness and kurtosis, the
//! Jarque-Bera moment test, and the Shapiro-Wilk W test.
//!
//! # Conventions and references
//!
//! The moment definitions match `scipy.stats` so a caller can cross-check
//! against the reference implementation.
//!
//! - [`skewness`] is the Fisher-Pearson coefficient `g1 = m3 / m2^(3/2)`
//!   (`scipy.stats.skew`, `bias=true`). The bias-corrected sample skewness
//!   `G1 = sqrt(n(n-1)) / (n-2) * g1` is selected with `bias = false`
//!   (`scipy.stats.skew`, `bias=false`).
//! - [`kurtosis`] is the Fisher (excess) kurtosis `g2 = m4 / m2^2 - 3` by
//!   default (`fisher = true`, matching `scipy.stats.kurtosis`); pass
//!   `fisher = false` for the Pearson definition `m4 / m2^2` (no `-3`). The
//!   bias-corrected estimator (`scipy.stats.kurtosis`, `bias=false`) is
//!   selected with `bias = false`.
//! - The central moments are the population (biased) moments
//!   `m_k = (1/n) sum_i (x_i - xbar)^k`, exactly as `scipy.stats._moment`
//!   forms them.
//! - [`jarque_bera`] is `JB = n/6 * (S^2 + K^2/4)` with `S` the biased
//!   skewness and `K` the biased excess kurtosis, and a chi-square(2) survival
//!   `p = exp(-JB/2)` (the closed form of `scipy.stats.distributions.chi2.sf`
//!   at two degrees of freedom), matching `scipy.stats.jarque_bera`.
//! - [`shapiro_wilk`] is a direct double-precision port of Royston's Remark
//!   AS R94 (1995), the same algorithm `scipy.stats.shapiro` uses.
//!
//! # Reproducibility
//!
//! The reductions here are plain left-to-right `f64` folds. `scipy`/`numpy`
//! reduce with pairwise summation, so agreement is to a tight tolerance rather
//! than bit-for-bit (observed against `scipy` 1.18.0): `< 1e-12` relative on the
//! moment statistics, and `~1e-10` on the Shapiro-Wilk `W` with `~1e-9` on its
//! p-value. `scipy` 1.18.0 adjusted its Shapiro-Wilk path, widening the gap from
//! this AS R94 port by `~5e-11` (`W`) / `~5e-10` (p) relative to earlier
//! releases; both stay well inside the test tolerances. The polynomial and
//! rational approximations inside the Shapiro-Wilk path use the same constants as
//! the `scipy` translation.

use crate::astro::math::special::erf;

/// Why a residual-distribution diagnostic could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NormalityError {
    /// A residual was non-finite.
    #[error("residual set has a non-finite value")]
    NonFinite,
    /// Fewer residuals than the statistic needs (the required minimum is
    /// reported).
    #[error("residual set too small: need at least {need} values, got {got}")]
    InsufficientData {
        /// Minimum number of residuals the statistic requires.
        need: usize,
        /// Number of residuals supplied.
        got: usize,
    },
    /// The residual set has zero (or numerically zero) variance, so a moment
    /// ratio is undefined.
    #[error("residual set has zero variance")]
    ZeroVariance,
    /// The residual set has zero range (all values equal), so the Shapiro-Wilk
    /// statistic is undefined.
    #[error("residual set has zero range")]
    ZeroRange,
}

/// Sample mean, variance, skewness, and kurtosis of a residual set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentStats {
    /// Arithmetic mean `(1/n) sum_i x_i`.
    pub mean: f64,
    /// Population (biased) variance `m2 = (1/n) sum_i (x_i - mean)^2`, the same
    /// second central moment `scipy.stats` divides by.
    pub variance: f64,
    /// Sample skewness, biased or bias-corrected per the `bias` flag passed to
    /// [`moments`] (see [`skewness`]).
    pub skewness: f64,
    /// Sample kurtosis. With `fisher = true` this is the excess kurtosis
    /// (Gaussian -> 0); with `fisher = false` it is the Pearson kurtosis
    /// (Gaussian -> 3). Biased or bias-corrected per the `bias` flag (see
    /// [`kurtosis`]).
    pub kurtosis_excess: f64,
}

/// Population central moments `m2, m3, m4` and the mean, formed exactly as
/// `scipy.stats._moment`: `m_k = (1/n) sum_i (x_i - mean)^k`.
fn central_moments(x: &[f64]) -> Result<(usize, f64, f64, f64, f64), NormalityError> {
    let n = x.len();
    if n == 0 {
        return Err(NormalityError::InsufficientData { need: 1, got: 0 });
    }
    for &v in x {
        if !v.is_finite() {
            return Err(NormalityError::NonFinite);
        }
    }
    let mut sum = 0.0;
    for &v in x {
        sum += v;
    }
    let mean = sum / n as f64;
    let (mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0);
    for &v in x {
        let d = v - mean;
        let d2 = d * d;
        s2 += d2;
        s3 += d2 * d;
        s4 += d2 * d2;
    }
    let inv_n = 1.0 / n as f64;
    Ok((n, mean, s2 * inv_n, s3 * inv_n, s4 * inv_n))
}

/// Sample skewness of a residual set.
///
/// `bias = true` returns the Fisher-Pearson coefficient `g1 = m3 / m2^(3/2)`
/// (`scipy.stats.skew`, default). `bias = false` applies the sample correction
/// `G1 = sqrt(n(n-1)) / (n-2) * g1` (`scipy.stats.skew(bias=False)`), which
/// needs at least three residuals.
pub fn skewness(x: &[f64], bias: bool) -> Result<f64, NormalityError> {
    let (n, _mean, m2, m3, _m4) = central_moments(x)?;
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(m2 > 0.0) {
        return Err(NormalityError::ZeroVariance);
    }
    let g1 = m3 / m2.powf(1.5);
    if bias {
        return Ok(g1);
    }
    if n < 3 {
        return Err(NormalityError::InsufficientData { need: 3, got: n });
    }
    let nf = n as f64;
    Ok(((nf - 1.0) * nf).sqrt() / (nf - 2.0) * g1)
}

/// Sample kurtosis of a residual set.
///
/// `fisher = true` returns the excess kurtosis `m4 / m2^2 - 3` (Gaussian -> 0,
/// `scipy.stats.kurtosis` default); `fisher = false` returns the Pearson
/// kurtosis `m4 / m2^2` (Gaussian -> 3). `bias = false` applies the sample
/// correction (`scipy.stats.kurtosis(bias=False)`), which needs at least four
/// residuals.
pub fn kurtosis(x: &[f64], fisher: bool, bias: bool) -> Result<f64, NormalityError> {
    let (n, _mean, m2, _m3, m4) = central_moments(x)?;
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(m2 > 0.0) {
        return Err(NormalityError::ZeroVariance);
    }
    let mut vals = m4 / (m2 * m2);
    if !bias {
        if n < 4 {
            return Err(NormalityError::InsufficientData { need: 4, got: n });
        }
        let nf = n as f64;
        vals = 1.0 / (nf - 2.0) / (nf - 3.0) * ((nf * nf - 1.0) * vals - 3.0 * (nf - 1.0).powi(2))
            + 3.0;
    }
    Ok(if fisher { vals - 3.0 } else { vals })
}

/// Mean, variance, skewness, and kurtosis in one pass.
///
/// `fisher` and `bias` select the kurtosis convention and the skewness/kurtosis
/// bias correction, exactly as in [`skewness`] and [`kurtosis`]. The reported
/// `variance` is always the biased second central moment.
pub fn moments(x: &[f64], fisher: bool, bias: bool) -> Result<MomentStats, NormalityError> {
    let (_n, mean, m2, _m3, _m4) = central_moments(x)?;
    Ok(MomentStats {
        mean,
        variance: m2,
        skewness: skewness(x, bias)?,
        kurtosis_excess: kurtosis(x, fisher, bias)?,
    })
}

/// Jarque-Bera goodness-of-fit test against normality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JarqueBera {
    /// Test statistic `JB = n/6 * (S^2 + K^2/4)`.
    pub statistic: f64,
    /// Upper-tail p-value under the chi-square(2) null, `exp(-JB/2)`.
    pub p_value: f64,
}

/// Jarque-Bera normality test on a residual set.
///
/// Uses the biased skewness and biased excess kurtosis, matching
/// `scipy.stats.jarque_bera`. The p-value is the closed-form chi-square(2)
/// survival function `exp(-statistic/2)`. Needs at least two residuals.
pub fn jarque_bera(x: &[f64]) -> Result<JarqueBera, NormalityError> {
    let (n, _mean, m2, m3, m4) = central_moments(x)?;
    if n < 2 {
        return Err(NormalityError::InsufficientData { need: 2, got: n });
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(m2 > 0.0) {
        return Err(NormalityError::ZeroVariance);
    }
    let s = m3 / m2.powf(1.5);
    let k = m4 / (m2 * m2) - 3.0;
    let statistic = n as f64 / 6.0 * (s * s + k * k / 4.0);
    let p_value = (-statistic / 2.0).exp();
    Ok(JarqueBera { statistic, p_value })
}

/// Shapiro-Wilk normality test on a residual set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapiroWilk {
    /// The Shapiro-Wilk `W` statistic in `(0, 1]`; closer to one is more
    /// Gaussian.
    pub w: f64,
    /// Upper-tail p-value for the null hypothesis of normality.
    pub p_value: f64,
}

// --- Royston AS R94 (1995) Shapiro-Wilk, double-precision port -------------

const SW_C1: [f64; 6] = [
    0.,
    0.221157,
    -0.147981,
    -0.207119e1,
    0.4434685e1,
    -0.2706056e1,
];
const SW_C2: [f64; 6] = [
    0.,
    0.42981e-1,
    -0.293762,
    -0.1752461e1,
    0.5682633e1,
    -0.3582633e1,
];
const SW_C3: [f64; 4] = [0.5440, -0.39978, 0.25054e-1, -0.6714e-3];
const SW_C4: [f64; 4] = [0.13822e1, -0.77857, 0.62767e-1, -0.20322e-2];
const SW_C5: [f64; 4] = [-0.15861e1, -0.31082, -0.83751e-1, 0.38915e-2];
const SW_C6: [f64; 3] = [-0.4803, -0.82676e-1, 0.30302e-2];
const SW_G: [f64; 2] = [-0.2273e1, 0.459];
const SW_SMALL: f64 = 1e-19;

/// Horner polynomial evaluation matching AS R94's `poly`:
/// `c[0] + x*(c[1] + x*(c[2] + ...))`.
fn sw_poly(c: &[f64], x: f64) -> f64 {
    let nord = c.len();
    let mut res = c[0];
    if nord == 1 {
        return res;
    }
    let mut p = x * c[nord - 1];
    for j in (1..nord - 1).rev() {
        p = (p + c[j]) * x;
    }
    res += p;
    res
}

/// Standard-normal tail area (AS 66 `alnorm`). With `upper = true` returns
/// `P(Z > x)`, else `P(Z <= x)`.
fn sw_alnorm(x: f64, upper: bool) -> f64 {
    // Reuse the crate's deterministic erf rather than the AS 66 rational core:
    // P(Z > x) = 0.5 * erfc(x / sqrt(2)). erfc(z) = 1 - erf(z).
    let phi_upper = 0.5 * (1.0 - erf(x / std::f64::consts::SQRT_2));
    if upper {
        phi_upper
    } else {
        1.0 - phi_upper
    }
}

/// Inverse standard-normal CDF (AS 111 rational approximation), the `ppnd`
/// helper from the `scipy` Shapiro-Wilk translation.
fn sw_ppnd(p: f64) -> f64 {
    const A0: f64 = 2.50662823884;
    const A1: f64 = -18.61500062529;
    const A2: f64 = 41.39119773534;
    const A3: f64 = -25.44106049637;
    const B1: f64 = -8.47351093090;
    const B2: f64 = 23.08336743743;
    const B3: f64 = -21.06224101826;
    const B4: f64 = 3.13082909833;
    const C0: f64 = -2.78718931138;
    const C1: f64 = -2.29796479134;
    const C2: f64 = 4.85014127135;
    const C3: f64 = 2.32121276858;
    const D1: f64 = 3.54388924762;
    const D2: f64 = 1.63706781897;
    const SPLIT: f64 = 0.42;

    let q = p - 0.5;
    if q.abs() <= SPLIT {
        let r = q * q;
        let temp = q * (((A3 * r + A2) * r + A1) * r + A0);
        return temp / ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.0);
    }
    let mut r = if q > 0.0 { 1.0 - p } else { p };
    if r > 0.0 {
        r = (-r.ln()).sqrt();
    } else {
        return 0.0;
    }
    let temp = (((C3 * r + C2) * r + C1) * r + C0) / ((D2 * r + D1) * r + 1.0);
    if q < 0.0 {
        -temp
    } else {
        temp
    }
}

/// Shapiro-Wilk W test for normality, a port of Royston's Remark AS R94 (1995),
/// the algorithm `scipy.stats.shapiro` uses.
///
/// Needs at least three residuals. Returns [`NormalityError::ZeroRange`] when
/// every residual is equal (the statistic is undefined). The `W` statistic
/// matches `scipy.stats.shapiro` to a tight tolerance; for `n > 5000` the
/// statistic is reliable but the p-value approximation degrades (as documented
/// for the reference implementation).
#[allow(clippy::needless_range_loop)]
pub fn shapiro_wilk(x: &[f64]) -> Result<ShapiroWilk, NormalityError> {
    let n = x.len();
    if n < 3 {
        return Err(NormalityError::InsufficientData { need: 3, got: n });
    }
    for &v in x {
        if !v.is_finite() {
            return Err(NormalityError::NonFinite);
        }
    }

    // Match scipy: sort ascending, then subtract the value at original index
    // n/2 (a centering for numerical conditioning that leaves W unchanged).
    let mut y = vec![0.0_f64; n + 1]; // 1-based y[1..=n]
    {
        let mut sorted = x.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let shift = x[n / 2];
        for (i, v) in sorted.into_iter().enumerate() {
            y[i + 1] = v - shift;
        }
    }

    let n2 = n / 2;
    let mut a = vec![0.0_f64; n2 + 1]; // 1-based a[1..=n2]

    if n == 3 {
        a[1] = std::f64::consts::FRAC_1_SQRT_2;
    } else {
        let an = n as f64;
        let an25 = an + 0.25;
        let mut summ2 = 0.0;
        for i in 1..=n2 {
            a[i] = sw_ppnd((i as f64 - 0.375) / an25);
            summ2 += a[i] * a[i];
        }
        summ2 *= 2.0;
        let ssumm2 = summ2.sqrt();
        let rsn = 1.0 / an.sqrt();
        let a1 = sw_poly(&SW_C1, rsn) - a[1] / ssumm2;

        let (i1, fac);
        if n > 5 {
            i1 = 3;
            let a2 = -a[2] / ssumm2 + sw_poly(&SW_C2, rsn);
            fac = ((summ2 - 2.0 * a[1] * a[1] - 2.0 * a[2] * a[2])
                / (1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2))
                .sqrt();
            a[1] = a1;
            a[2] = a2;
        } else {
            i1 = 2;
            fac = ((summ2 - 2.0 * a[1] * a[1]) / (1.0 - 2.0 * a1 * a1)).sqrt();
            a[1] = a1;
        }
        for i in i1..=n2 {
            a[i] = -a[i] / fac;
        }
    }

    // Antisymmetric coefficient for the i-th order statistic (1-based), built
    // from the half-length `a` exactly as the AS R94 W loops do.
    let coeff = |i: usize, j: usize| -> f64 {
        let sign = if i >= j { 1.0 } else { -1.0 };
        sign * a[i.min(j)]
    };

    let rng = y[n] - y[1];
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let nonpositive_or_nan = !(rng > 0.0);
    if nonpositive_or_nan {
        return Err(NormalityError::ZeroRange);
    }

    let mut sa = 0.0;
    let mut sx = 0.0;
    let mut j = n;
    for i in 1..=n {
        let asa = if i != j { coeff(i, j) } else { 0.0 };
        sa += asa;
        sx += y[i] / rng;
        j -= 1;
    }
    sa /= n as f64;
    sx /= n as f64;

    let mut ssa = 0.0;
    let mut ssx = 0.0;
    let mut sax = 0.0;
    let mut j = n;
    for i in 1..=n {
        let asa = if i != j { coeff(i, j) - sa } else { -sa };
        let xsx = y[i] / rng - sx;
        ssa += asa * asa;
        ssx += xsx * xsx;
        sax += asa * xsx;
        j -= 1;
    }
    let ssassx = (ssa * ssx).sqrt();
    let w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx);
    let w = 1.0 - w1;

    let p_value = if n == 3 {
        let pi6 = 6.0 / std::f64::consts::PI;
        let stqr = (0.75_f64).sqrt().asin();
        (pi6 * (w.sqrt().asin() - stqr)).clamp(0.0, 1.0)
    } else if w1 <= 0.0 {
        // Degenerate: W >= 1 (the residuals are essentially perfectly
        // Gaussian-ordered), so the null is not rejected.
        1.0
    } else {
        let an = n as f64;
        let mut y_t = w1.ln();
        let xx = an.ln();
        let (m, s);
        if n <= 11 {
            let gamma = sw_poly(&SW_G, an);
            if y_t >= gamma {
                return Ok(ShapiroWilk {
                    w,
                    p_value: SW_SMALL,
                });
            }
            y_t = -(gamma - y_t).ln();
            m = sw_poly(&SW_C3, an);
            s = sw_poly(&SW_C4, an).exp();
        } else {
            m = sw_poly(&SW_C5, xx);
            s = sw_poly(&SW_C6, xx).exp();
        }
        sw_alnorm((y_t - m) / s, true)
    };

    Ok(ShapiroWilk { w, p_value })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fixed residual vectors with golden values from scipy 1.18.0
    // (scipy.stats.skew / kurtosis / jarque_bera / shapiro), regenerated by
    // `fixtures-generators/generate_normality.py`. Agreement is to a tight
    // tolerance, not bit-for-bit, because numpy reduces with pairwise summation
    // while these folds are left-to-right.
    const V1: [f64; 15] = [
        0.12, -0.34, 0.05, 0.88, -1.21, 0.42, -0.07, 0.63, -0.55, 0.19, 0.27, -0.91, 1.04, -0.16,
        0.33,
    ];
    const V2: [f64; 12] = [
        1.0, -2.0, 0.5, 3.2, -1.1, 0.0, 2.3, -0.7, 4.5, -3.1, 0.9, -1.8,
    ];

    const TOL: f64 = 1e-11;

    fn close(got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol + tol * want.abs(),
            "got {got}, want {want}, diff {}",
            (got - want).abs()
        );
    }

    #[test]
    fn skew_matches_scipy() {
        // scipy.stats.skew(V1) and skew(V1, bias=False)
        close(skewness(&V1, true).unwrap(), -3.990_837_649_877_545e-1, TOL);
        close(skewness(&V1, false).unwrap(), -4.448_671_685_942_52e-1, TOL);
        close(skewness(&V2, true).unwrap(), 3.471_961_494_435_007e-1, TOL);
        close(
            skewness(&V2, false).unwrap(),
            3.988_980_062_229_937_6e-1,
            TOL,
        );
    }

    #[test]
    fn kurtosis_matches_scipy() {
        // scipy.stats.kurtosis(V1, fisher=True/False, bias=True/False)
        close(
            kurtosis(&V1, true, true).unwrap(),
            -3.608_466_739_341_209_5e-1,
            TOL,
        );
        close(
            kurtosis(&V1, false, true).unwrap(),
            2.639_153_326_065_879,
            TOL,
        );
        close(
            kurtosis(&V1, true, false).unwrap(),
            2.032_272_460_741_557_7e-2,
            TOL,
        );
        close(
            kurtosis(&V2, true, true).unwrap(),
            -7.089_134_727_921_165e-1,
            TOL,
        );
        close(
            kurtosis(&V2, false, true).unwrap(),
            2.291_086_527_207_883_5,
            TOL,
        );
        close(
            kurtosis(&V2, true, false).unwrap(),
            -3.930_514_067_696_959_7e-1,
            TOL,
        );
    }

    #[test]
    fn moments_bundle_matches_components() {
        let m = moments(&V1, true, true).unwrap();
        close(m.mean, 4.6e-2, TOL);
        close(m.variance, 3.582_106_666_666_667_3e-1, TOL);
        close(m.skewness, skewness(&V1, true).unwrap(), 0.0);
        close(m.kurtosis_excess, kurtosis(&V1, true, true).unwrap(), 0.0);
    }

    #[test]
    fn jarque_bera_matches_scipy() {
        let jb1 = jarque_bera(&V1).unwrap();
        close(jb1.statistic, 4.795_510_799_978_267_6e-1, TOL);
        close(jb1.p_value, 7.868_044_473_746_433e-1, TOL);
        let jb2 = jarque_bera(&V2).unwrap();
        close(jb2.statistic, 4.923_694_883_298_767_6e-1, TOL);
        close(jb2.p_value, 7.817_777_826_998_267e-1, TOL);
    }

    #[test]
    fn shapiro_wilk_matches_scipy() {
        let sw1 = shapiro_wilk(&V1).unwrap();
        close(sw1.w, 9.760_100_117_114_072e-1, 1e-10);
        close(sw1.p_value, 9.349_583_655_477_645e-1, 1e-9);
        let sw2 = shapiro_wilk(&V2).unwrap();
        close(sw2.w, 9.773_113_095_849_641e-1, 1e-10);
        close(sw2.p_value, 9.706_201_224_239_078e-1, 1e-9);
    }

    #[test]
    fn shapiro_wilk_n3_path() {
        // n == 3 exercises the arcsin p-value branch.
        let x = [0.1, -0.4, 0.9];
        let sw = shapiro_wilk(&x).unwrap();
        assert!(sw.w > 0.0 && sw.w <= 1.0 + 1e-12);
        assert!((0.0..=1.0).contains(&sw.p_value));
    }

    #[test]
    fn rejects_degenerate_inputs() {
        assert_eq!(
            skewness(&[1.0, 1.0, 1.0], true),
            Err(NormalityError::ZeroVariance)
        );
        assert_eq!(
            shapiro_wilk(&[2.0, 2.0, 2.0]),
            Err(NormalityError::ZeroRange)
        );
        assert!(matches!(
            skewness(&[1.0, 2.0], false),
            Err(NormalityError::InsufficientData { need: 3, got: 2 })
        ));
        assert!(matches!(
            kurtosis(&[1.0, 2.0, 3.0], true, false),
            Err(NormalityError::InsufficientData { need: 4, got: 3 })
        ));
        assert_eq!(
            skewness(&[1.0, f64::NAN, 2.0], true),
            Err(NormalityError::NonFinite)
        );
        assert!(matches!(
            shapiro_wilk(&[1.0, 2.0]),
            Err(NormalityError::InsufficientData { need: 3, got: 2 })
        ));
    }
}
