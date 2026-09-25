//! Independent interval evaluation of the ideal GPS RTKLIB center model.
//!
//! The certificate treats all supplied binary64 measurements and state values
//! as exact inputs. It computes geometry and model terms independently with
//! `Interval`; it does not call the SPP production model. Inputs whose model
//! branches cannot be certified are refused.

use super::interval_certificate::Interval;

const PI: f64 = core::f64::consts::PI;
const C_M_S: f64 = 299_792_458.0;
const OMEGA_E_DOT_RAD_S: f64 = 7.292_115_146_7e-5;
const WGS84_A_M: f64 = 6_378_137.0;
const WGS84_E2: f64 = 6.694_379_990_141_316_5e-3;
const BROADCAST_IONOSPHERE_ERROR_FACTOR: f64 = 0.5;
const CODE_BIAS_ERROR_M: f64 = 0.3;
const UNCORRECTED_IONOSPHERE_ERROR_M: f64 = 5.0;
const UNCORRECTED_TROPOSPHERE_ERROR_M: f64 = 3.0;
const TROPOSPHERE_MODEL_ERROR_M: f64 = 0.3;
const PHASE_ERROR_BASE_M: f64 = 0.003;
const PHASE_ERROR_ELEVATION_M: f64 = 0.003;
const CODE_PHASE_ERROR_RATIO: f64 = 300.0;

/// Binary64 inputs for one GPS L1 RTKLIB-placement measurement at a fixed state.
pub(super) struct CenterInputs {
    pub receiver_ecef_m: [f64; 3],
    /// Candidate order is the three-iteration Skyfield result, then RTKLIB ecef2pos.
    pub finite_inverse_candidates: [[f64; 3]; 2],
    pub receiver_clock_m: f64,
    pub satellite_ecef_m: [f64; 3],
    pub satellite_clock_s: f64,
    pub group_delay_s: f64,
    pub pseudorange_m: f64,
    pub ephemeris_variance_m2: f64,
    pub second_of_day_s: f64,
    pub alpha: [f64; 4],
    pub beta: [f64; 4],
    pub apply_ionosphere: bool,
    pub apply_troposphere: bool,
}

/// Independently enclosed model rows and center geodetic/look-angle inputs.
#[derive(Debug)]
pub(super) struct CenterIntervals {
    pub ideal_latitude_rad: Interval,
    pub ideal_longitude_rad: Interval,
    pub ideal_height_m: Interval,
    pub ideal_azimuth_rad: Interval,
    pub ideal_elevation_rad: Interval,
    pub ideal_klobuchar: Option<KlobucharIntervals>,
    pub latitude_rad: Interval,
    pub longitude_rad: Interval,
    pub height_m: Interval,
    pub azimuth_rad: Interval,
    pub elevation_rad: Interval,
    pub design_row: [Interval; 4],
    pub weight: Interval,
    pub residual_m: Interval,
    pub finite_inverse_forward_residuals_m: [[Interval; 3]; 2],
    pub klobuchar: Option<KlobucharIntervals>,
}

#[derive(Debug)]
pub(super) struct CenterGeometry {
    pub ideal_latitude_rad: Interval,
    pub ideal_longitude_rad: Interval,
    pub ideal_height_m: Interval,
    pub ideal_azimuth_rad: Interval,
    pub ideal_elevation_rad: Interval,
    pub latitude_rad: Interval,
    pub longitude_rad: Interval,
    pub height_m: Interval,
    pub azimuth_rad: Interval,
    pub elevation_rad: Interval,
    pub design_row: [Interval; 4],
    pub finite_inverse_forward_residuals_m: [[Interval; 3]; 2],
}

#[derive(Debug)]
pub(super) struct KlobucharIntervals {
    pub raw_phi_i_semicircles: Interval,
    pub phi_m_semicircles: Interval,
    pub local_time_seconds: Interval,
    pub phase_radians: Interval,
    pub delay_m: Interval,
}

/// Why an input could not be certified within the supported GPS model domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CenterCertificateError {
    InvalidInput,
    GeodeticDomain,
    FiniteInverseMismatch,
    GeometryDomain,
    CorrectionBranch,
    TroposphereDomain,
}

type GeodeticResult = Result<[Interval; 3], CenterCertificateError>;
type GeodeticEvaluator = fn([Interval; 3]) -> GeodeticResult;
type GeodeticCacheEntry = Option<([[u64; 2]; 3], GeodeticResult)>;

std::thread_local! {
    static IDEAL_GEODETIC_CACHE: std::cell::RefCell<GeodeticCacheEntry> = const { std::cell::RefCell::new(None) };
    static SKYFIELD_GEODETIC_CACHE: std::cell::RefCell<GeodeticCacheEntry> = const { std::cell::RefCell::new(None) };
    static RTKLIB_GEODETIC_CACHE: std::cell::RefCell<GeodeticCacheEntry> = const { std::cell::RefCell::new(None) };
}

/// Reuse a pure receiver enclosure only when every input bound has identical bits.
/// Each model retains one entry per thread; no interval is widened or reused for
/// a different receiver, and the uncached arithmetic is unchanged.
fn cached_geodetic(
    cache: &'static std::thread::LocalKey<std::cell::RefCell<GeodeticCacheEntry>>,
    receiver: [Interval; 3],
    evaluate: GeodeticEvaluator,
) -> GeodeticResult {
    let key = receiver.map(|interval| [interval.lower().to_bits(), interval.upper().to_bits()]);
    if let Some(result) = cache.with(|entry| {
        entry
            .borrow()
            .as_ref()
            .filter(|(cached_key, _)| *cached_key == key)
            .map(|(_, result)| *result)
    }) {
        return result;
    }
    let result = evaluate(receiver);
    cache.with(|entry| *entry.borrow_mut() = Some((key, result)));
    result
}

pub(super) fn evaluate_geometry(
    inputs: &CenterInputs,
) -> Result<CenterGeometry, CenterCertificateError> {
    if !finite_inputs(inputs) {
        return Err(CenterCertificateError::InvalidInput);
    }
    let receiver = inputs.receiver_ecef_m.map(Interval::point);
    let satellite = inputs.satellite_ecef_m.map(Interval::point);
    let [ideal_latitude, ideal_longitude, ideal_height] = geodetic(receiver)?;
    let delta = [
        satellite[0].sub(receiver[0]),
        satellite[1].sub(receiver[1]),
        satellite[2].sub(receiver[2]),
    ];
    let (ideal_azimuth_rad, ideal_elevation_rad) =
        look_angles(receiver, delta, ideal_latitude, ideal_longitude)?;
    let reproduced = [skyfield_geodetic(receiver)?, rtklib_geodetic(receiver)?];
    let mut latitude_rad = ideal_latitude;
    let mut longitude_rad = ideal_longitude;
    let mut height_m = ideal_height;
    let mut finite_inverse_forward_residuals_m = [[Interval::point(0.0); 3]; 2];
    for candidate_index in 0..2 {
        let candidate = inputs.finite_inverse_candidates[candidate_index];
        let candidate_interval = candidate.map(Interval::point);
        if !(0..3)
            .all(|component| reproduced[candidate_index][component].contains(candidate[component]))
        {
            return Err(CenterCertificateError::FiniteInverseMismatch);
        }
        latitude_rad = latitude_rad.hull(candidate_interval[0]);
        longitude_rad = longitude_rad.hull(candidate_interval[1]);
        height_m = height_m.hull(candidate_interval[2]);
        finite_inverse_forward_residuals_m[candidate_index] =
            forward_residual(receiver, candidate_interval)?;
    }
    if longitude_rad.width() >= PI {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    let range = sum_squares(&delta).sqrt();
    if range.lower() <= 0.0 {
        return Err(CenterCertificateError::GeometryDomain);
    }
    let los = [
        delta[0].div(range),
        delta[1].div(range),
        delta[2].div(range),
    ];
    let (azimuth_rad, elevation_rad) = look_angles(receiver, delta, latitude_rad, longitude_rad)?;
    let design_row = [
        Interval::point(-1.0).mul(los[0]),
        Interval::point(-1.0).mul(los[1]),
        Interval::point(-1.0).mul(los[2]),
        Interval::point(1.0),
    ];
    Ok(CenterGeometry {
        ideal_latitude_rad: ideal_latitude,
        ideal_longitude_rad: ideal_longitude,
        ideal_height_m: ideal_height,
        ideal_azimuth_rad,
        ideal_elevation_rad,
        latitude_rad,
        longitude_rad,
        height_m,
        azimuth_rad,
        elevation_rad,
        design_row,
        finite_inverse_forward_residuals_m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(receiver_ecef_m: [f64; 3]) -> CenterInputs {
        let receiver = receiver_ecef_m.map(Interval::point);
        let candidate_values = [
            skyfield_geodetic(receiver).unwrap(),
            rtklib_geodetic(receiver).unwrap(),
        ];
        let candidates = candidate_values.map(|candidate| {
            candidate.map(|value| value.lower() + (value.upper() - value.lower()) * 0.5)
        });
        CenterInputs {
            receiver_ecef_m,
            finite_inverse_candidates: candidates,
            receiver_clock_m: 0.0,
            satellite_ecef_m: [20_200_000.0, -13_000_000.0, 9_000_000.0],
            satellite_clock_s: 0.0,
            group_delay_s: 0.0,
            pseudorange_m: 13_800_000.0,
            ephemeris_variance_m2: 1.0,
            second_of_day_s: 40_000.0,
            alpha: [0.0; 4],
            beta: [0.0; 4],
            apply_ionosphere: true,
            apply_troposphere: false,
        }
    }

    #[test]
    fn known_ecef_returns_distinct_ideal_and_finite_hull_geometry() {
        let mut inputs = inputs([WGS84_A_M + 100.0, 0.0, 0.0]);
        inputs.finite_inverse_candidates[1] = [0.0, 0.0, 100.0];
        let result = evaluate_geometry(&inputs).unwrap();
        assert!(result.ideal_latitude_rad.contains(0.0));
        assert!(result.ideal_longitude_rad.contains(0.0));
        assert!(result.ideal_height_m.contains(100.0));
        assert!(result
            .latitude_rad
            .contains(inputs.finite_inverse_candidates[0][0]));
        assert!(result
            .longitude_rad
            .contains(inputs.finite_inverse_candidates[1][1]));
        assert!(result.azimuth_rad.upper() < 0.0);
    }

    #[test]
    fn evaluation_exposes_ideal_and_finite_hull_klobuchar_enclosures() {
        let mut inputs = inputs([WGS84_A_M + 100.0, 0.0, 0.0]);
        inputs.alpha[0] = 1.0e-8;
        inputs.beta[0] = 100_000.0;
        let result = evaluate(&inputs).unwrap();
        assert!(result.ideal_klobuchar.is_some());
        assert!(result.klobuchar.is_some());
    }

    #[test]
    fn equatorial_geometry_encloses_an_exact_integer_direction() {
        let mut inputs = inputs([WGS84_A_M + 100.0, 0.0, 0.0]);
        inputs.finite_inverse_candidates[1] = [0.0, 0.0, 100.0];
        inputs.satellite_ecef_m = [WGS84_A_M + 103.0, 4.0, 12.0];
        let result = evaluate_geometry(&inputs).unwrap();
        assert!(result.height_m.contains(100.0));
        for (interval, expected) in
            result
                .design_row
                .into_iter()
                .zip([-3.0 / 13.0, -4.0 / 13.0, -12.0 / 13.0, 1.0])
        {
            assert!(interval.contains(expected));
        }
        for residual in result.finite_inverse_forward_residuals_m[1] {
            assert!(residual.contains(0.0));
        }
    }

    #[test]
    fn invalid_ecef_domain_is_refused() {
        let mut inputs = inputs([WGS84_A_M + 100.0, 0.0, 0.0]);
        inputs.receiver_ecef_m = [0.0; 3];
        let result = evaluate_geometry(&inputs);
        assert_eq!(result.unwrap_err(), CenterCertificateError::GeodeticDomain);
    }

    #[test]
    fn cached_geodetic_enclosures_retain_exact_inputs_and_results() {
        let evaluators: [(GeodeticEvaluator, GeodeticEvaluator); 3] = [
            (geodetic, geodetic_uncached),
            (skyfield_geodetic, skyfield_geodetic_uncached),
            (rtklib_geodetic, rtklib_geodetic_uncached),
        ];
        let receiver = [WGS84_A_M + 100.0, 0.0, 0.0].map(Interval::point);
        let mut adjacent = receiver;
        adjacent[0] = Interval::point(f64::from_bits(receiver[0].lower().to_bits() + 1));
        let mut widened = receiver;
        widened[0] = receiver[0].hull(adjacent[0]);
        let bits = |result: GeodeticResult| {
            result.map(|coordinates| {
                coordinates.map(|interval| [interval.lower().to_bits(), interval.upper().to_bits()])
            })
        };
        for (cached, uncached) in evaluators {
            for input in [receiver, receiver, adjacent, widened, receiver] {
                assert_eq!(bits(cached(input)), bits(uncached(input)));
            }
        }
    }

    #[test]
    fn incorrect_finite_inverse_candidate_is_refused() {
        let mut inputs = inputs([WGS84_A_M + 100.0, 0.0, 0.0]);
        inputs.finite_inverse_candidates[0][0] += 1.0e-3;
        assert_eq!(
            evaluate_geometry(&inputs).unwrap_err(),
            CenterCertificateError::FiniteInverseMismatch
        );
    }

    #[test]
    fn ambiguous_klobuchar_branch_is_refused() {
        let second_of_day_s = 50_400.0 + 1.57 * 72_000.0 / (2.0 * PI);
        let result = klobuchar(
            Interval::point(0.0),
            Interval::point(0.0),
            Interval::point(0.0),
            Interval::point(0.5),
            second_of_day_s,
            [1.0e-8, 0.0, 0.0, 0.0],
            [71_000.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(
            result.unwrap_err(),
            CenterCertificateError::CorrectionBranch
        );
    }

    #[test]
    fn asin_zero_has_a_certified_singleton_enclosure() {
        assert_eq!(
            asin_interval(Interval::point(0.0)),
            Ok(Interval::point(0.0))
        );
    }

    #[test]
    fn asin_unit_endpoints_enclose_the_exact_quadrant_limits() {
        let pi_upper = f64::from_bits(PI.to_bits() + 1);
        let positive = asin_interval(Interval::point(1.0)).unwrap();
        let negative = asin_interval(Interval::point(-1.0)).unwrap();
        assert!(positive.lower() <= PI / 2.0);
        assert!(positive.upper() >= pi_upper / 2.0);
        assert!(negative.lower() <= -pi_upper / 2.0);
        assert!(negative.upper() >= -PI / 2.0);
    }

    #[test]
    fn western_finite_inverse_longitudes_share_the_principal_branch() {
        let receiver = [-3_194_469.0, -3_194_469.0, 4_487_419.0].map(Interval::point);
        let skyfield = skyfield_geodetic(receiver).unwrap();
        let rtklib = rtklib_geodetic(receiver).unwrap();
        assert!(skyfield[1].upper() < 0.0);
        assert!(rtklib[1].upper() < 0.0);

        let mut inputs = inputs([-3_194_469.0, -3_194_469.0, 4_487_419.0]);
        inputs.finite_inverse_candidates = [skyfield, rtklib].map(|candidate| {
            candidate.map(|value| value.lower() + (value.upper() - value.lower()) * 0.5)
        });
        evaluate_geometry(&inputs).expect("western longitude conventions align");
    }
}

/// Enclose ideal WGS-84 GPS L1 design, inverse-variance, and residual inputs.
pub(super) fn evaluate(inputs: &CenterInputs) -> Result<CenterIntervals, CenterCertificateError> {
    if !finite_inputs(inputs) {
        return Err(CenterCertificateError::InvalidInput);
    }
    let receiver = inputs.receiver_ecef_m.map(Interval::point);
    let satellite = inputs.satellite_ecef_m.map(Interval::point);
    let [ideal_latitude, ideal_longitude, ideal_height] = geodetic(receiver)?;
    let dx = satellite[0].sub(receiver[0]);
    let dy = satellite[1].sub(receiver[1]);
    let dz = satellite[2].sub(receiver[2]);
    let ideal_angles = look_angles(receiver, [dx, dy, dz], ideal_latitude, ideal_longitude)?;
    let reproduced_candidates = [skyfield_geodetic(receiver)?, rtklib_geodetic(receiver)?];
    let mut latitude_rad = ideal_latitude;
    let mut longitude_rad = ideal_longitude;
    let mut height_m = ideal_height;
    let mut finite_inverse_forward_residuals_m = [[Interval::point(0.0); 3]; 2];
    for candidate_index in 0..2 {
        let candidate = inputs.finite_inverse_candidates[candidate_index];
        let candidate_intervals = candidate.map(Interval::point);
        if !reproduced_candidates[candidate_index][0].contains(candidate[0])
            || !reproduced_candidates[candidate_index][1].contains(candidate[1])
            || !reproduced_candidates[candidate_index][2].contains(candidate[2])
        {
            return Err(CenterCertificateError::FiniteInverseMismatch);
        }
        latitude_rad = latitude_rad.hull(candidate_intervals[0]);
        longitude_rad = longitude_rad.hull(candidate_intervals[1]);
        height_m = height_m.hull(candidate_intervals[2]);
        finite_inverse_forward_residuals_m[candidate_index] =
            forward_residual(receiver, candidate_intervals)?;
    }
    if longitude_rad.width() >= PI {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    let dx = satellite[0].sub(receiver[0]);
    let dy = satellite[1].sub(receiver[1]);
    let dz = satellite[2].sub(receiver[2]);
    let range = sum_squares(&[dx, dy, dz]).sqrt();
    if range.lower() <= 0.0 {
        return Err(CenterCertificateError::GeometryDomain);
    }
    let los = [dx.div(range), dy.div(range), dz.div(range)];
    let (azimuth_rad, elevation_rad) =
        look_angles(receiver, [dx, dy, dz], latitude_rad, longitude_rad)?;
    if elevation_rad.lower() <= super::ELEVATION_MASK_RAD {
        return Err(CenterCertificateError::GeometryDomain);
    }
    let design_row = [
        Interval::point(-1.0).mul(los[0]),
        Interval::point(-1.0).mul(los[1]),
        Interval::point(-1.0).mul(los[2]),
        Interval::point(1.0),
    ];
    let cross = satellite[0]
        .mul(receiver[1])
        .sub(satellite[1].mul(receiver[0]));
    let geometric_range = range.add(
        Interval::point(OMEGA_E_DOT_RAD_S)
            .div(Interval::point(C_M_S))
            .mul(cross),
    );
    let corrected_clock =
        Interval::point(inputs.satellite_clock_s).sub(Interval::point(inputs.group_delay_s));
    let ideal_klobuchar = if inputs.apply_ionosphere {
        Some(klobuchar(
            ideal_latitude,
            ideal_longitude,
            ideal_angles.0,
            ideal_angles.1,
            inputs.second_of_day_s,
            inputs.alpha,
            inputs.beta,
        )?)
    } else {
        None
    };
    let klobuchar = if inputs.apply_ionosphere {
        Some(klobuchar(
            latitude_rad,
            longitude_rad,
            azimuth_rad,
            elevation_rad,
            inputs.second_of_day_s,
            inputs.alpha,
            inputs.beta,
        )?)
    } else {
        None
    };
    let ionosphere_m = klobuchar
        .as_ref()
        .map(|components| components.delay_m)
        .unwrap_or_else(|| Interval::point(0.0));
    let mut troposphere_m = Interval::point(0.0);
    if inputs.apply_troposphere {
        troposphere_m = rtklib_troposphere(latitude_rad, height_m, elevation_rad)?;
    }
    let predicted = geometric_range
        .add(Interval::point(inputs.receiver_clock_m))
        .sub(Interval::point(C_M_S).mul(corrected_clock))
        .add(ionosphere_m)
        .add(troposphere_m);
    let residual_m = Interval::point(inputs.pseudorange_m).sub(predicted);
    let ionosphere_variance = if inputs.apply_ionosphere {
        ionosphere_m
            .mul(Interval::point(BROADCAST_IONOSPHERE_ERROR_FACTOR))
            .square()
    } else {
        Interval::point(UNCORRECTED_IONOSPHERE_ERROR_M).square()
    };
    let sine_elevation = elevation_rad.sin();
    if sine_elevation.lower() <= 0.0 {
        return Err(CenterCertificateError::GeometryDomain);
    }
    let troposphere_variance = if inputs.apply_troposphere {
        Interval::point(TROPOSPHERE_MODEL_ERROR_M)
            .div(sine_elevation.add(Interval::point(0.1)))
            .square()
    } else {
        Interval::point(UNCORRECTED_TROPOSPHERE_ERROR_M).square()
    };
    let code_variance = Interval::point(CODE_PHASE_ERROR_RATIO).square().mul(
        Interval::point(PHASE_ERROR_BASE_M).square().add(
            Interval::point(PHASE_ERROR_ELEVATION_M)
                .square()
                .div(sine_elevation),
        ),
    );
    let variance = Interval::point(inputs.ephemeris_variance_m2)
        .add(Interval::point(CODE_BIAS_ERROR_M).square())
        .add(ionosphere_variance)
        .add(troposphere_variance)
        .add(code_variance);
    if variance.lower() <= 0.0 {
        return Err(CenterCertificateError::InvalidInput);
    }
    let weight = Interval::point(1.0).div(variance);
    Ok(CenterIntervals {
        ideal_latitude_rad: ideal_latitude,
        ideal_longitude_rad: ideal_longitude,
        ideal_height_m: ideal_height,
        ideal_azimuth_rad: ideal_angles.0,
        ideal_elevation_rad: ideal_angles.1,
        ideal_klobuchar,
        latitude_rad,
        longitude_rad,
        height_m,
        azimuth_rad,
        elevation_rad,
        design_row,
        weight,
        residual_m,
        finite_inverse_forward_residuals_m,
        klobuchar,
    })
}

fn finite_inputs(inputs: &CenterInputs) -> bool {
    inputs
        .receiver_ecef_m
        .iter()
        .chain(inputs.satellite_ecef_m.iter())
        .chain(inputs.finite_inverse_candidates.iter().flatten())
        .chain([
            &inputs.receiver_clock_m,
            &inputs.satellite_clock_s,
            &inputs.group_delay_s,
            &inputs.pseudorange_m,
            &inputs.ephemeris_variance_m2,
            &inputs.second_of_day_s,
        ])
        .chain(inputs.alpha.iter())
        .chain(inputs.beta.iter())
        .all(|value| value.is_finite())
        && inputs
            .receiver_ecef_m
            .iter()
            .all(|value| value.abs() <= 10_000_000.0)
        && inputs
            .satellite_ecef_m
            .iter()
            .all(|value| value.abs() <= 30_000_000.0)
        && inputs.finite_inverse_candidates.iter().all(|candidate| {
            candidate[0].abs() <= core::f64::consts::FRAC_PI_2
                && candidate[1].abs() <= 2.0 * PI
                && (0.0..10_000.0).contains(&candidate[2])
        })
        && inputs.receiver_clock_m.abs() <= 1.0e6
        && inputs.satellite_clock_s.abs() <= 1.0
        && inputs.group_delay_s.abs() <= 1.0
        && inputs.pseudorange_m.abs() <= 1.0e8
        && inputs.ephemeris_variance_m2 >= 0.0
        && inputs.ephemeris_variance_m2 <= 1.0e6
        && inputs
            .alpha
            .iter()
            .chain(inputs.beta.iter())
            .all(|value| value.abs() <= 1.0e6)
        && (0.0..86_400.0).contains(&inputs.second_of_day_s)
}

fn forward_residual(
    receiver: [Interval; 3],
    geodetic: [Interval; 3],
) -> Result<[Interval; 3], CenterCertificateError> {
    let (sine_latitude, cosine_latitude, prime_vertical) = ellipsoid_terms(geodetic[0]);
    let radial = prime_vertical.add(geodetic[2]).mul(cosine_latitude);
    let x = radial.mul(interval_cos_periodic(geodetic[1])?);
    let y = radial.mul(interval_sin_periodic(geodetic[1])?);
    let z = prime_vertical
        .mul(Interval::point(1.0).sub(Interval::point(WGS84_E2)))
        .add(geodetic[2])
        .mul(sine_latitude);
    Ok([x.sub(receiver[0]), y.sub(receiver[1]), z.sub(receiver[2])])
}

fn geodetic(ecef: [Interval; 3]) -> Result<[Interval; 3], CenterCertificateError> {
    cached_geodetic(&IDEAL_GEODETIC_CACHE, ecef, geodetic_uncached)
}

fn geodetic_uncached(ecef: [Interval; 3]) -> GeodeticResult {
    let horizontal = sum_squares(&[ecef[0], ecef[1]]).sqrt();
    if horizontal.lower() <= 0.0 {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    let one_minus_eccentricity = Interval::point(1.0).sub(Interval::point(WGS84_E2));
    let derivative_lower_bound = Interval::point(WGS84_E2)
        .mul(Interval::point(WGS84_A_M))
        .div(one_minus_eccentricity.mul(one_minus_eccentricity.sqrt()));
    if horizontal.lower() <= derivative_lower_bound.upper() {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    let mut lower = -core::f64::consts::FRAC_PI_2;
    let mut upper = core::f64::consts::FRAC_PI_2;
    if latitude_equation(horizontal, ecef[2], Interval::point(lower)).upper() >= 0.0
        || latitude_equation(horizontal, ecef[2], Interval::point(upper)).lower() <= 0.0
    {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    for _ in 0..64 {
        let midpoint = lower + (upper - lower) * 0.5;
        let equation = latitude_equation(horizontal, ecef[2], Interval::point(midpoint));
        if equation.upper() < 0.0 {
            lower = midpoint;
        } else if equation.lower() > 0.0 {
            upper = midpoint;
        } else {
            let lower_quartile = lower + (midpoint - lower) * 0.5;
            let upper_quartile = midpoint + (upper - midpoint) * 0.5;
            let lower_equation =
                latitude_equation(horizontal, ecef[2], Interval::point(lower_quartile));
            let upper_equation =
                latitude_equation(horizontal, ecef[2], Interval::point(upper_quartile));
            if lower_equation.upper() < 0.0 && upper_equation.lower() > 0.0 {
                lower = lower_quartile;
                upper = upper_quartile;
            }
        }
    }
    let latitude = Interval::new(lower, upper);
    let (_, cosine, prime_vertical) = ellipsoid_terms(latitude);
    if cosine.lower() <= 0.0 {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    let height = horizontal.div(cosine).sub(prime_vertical);
    if height.lower() <= 0.0 || height.upper() >= 10_000.0 {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    let longitude = asin_interval(ecef[1].div(horizontal))?;
    let longitude = if ecef[0].upper() < 0.0 {
        if ecef[1].lower() >= 0.0 {
            Interval::point(PI).sub(longitude)
        } else if ecef[1].upper() < 0.0 {
            Interval::point(-PI).sub(longitude)
        } else {
            return Err(CenterCertificateError::GeodeticDomain);
        }
    } else {
        longitude
    };
    Ok([latitude, longitude, height])
}

fn latitude_equation(horizontal: Interval, z: Interval, latitude: Interval) -> Interval {
    let (sine, cosine, prime_vertical) = ellipsoid_terms(latitude);
    horizontal
        .sub(Interval::point(WGS84_E2).mul(prime_vertical).mul(cosine))
        .mul(sine)
        .sub(z.mul(cosine))
}

fn ellipsoid_terms(latitude: Interval) -> (Interval, Interval, Interval) {
    let sine = latitude.sin();
    let cosine = latitude.cos();
    let denominator = Interval::point(1.0).sub(Interval::point(WGS84_E2).mul(sine.square()));
    let prime_vertical = Interval::point(WGS84_A_M).div(denominator.sqrt());
    (sine, cosine, prime_vertical)
}

fn skyfield_geodetic(receiver: [Interval; 3]) -> Result<[Interval; 3], CenterCertificateError> {
    cached_geodetic(
        &SKYFIELD_GEODETIC_CACHE,
        receiver,
        skyfield_geodetic_uncached,
    )
}

fn skyfield_geodetic_uncached(receiver: [Interval; 3]) -> GeodeticResult {
    let kilometre_scale = Interval::point(crate::constants::KM_TO_M);
    let astronomical_unit_km = Interval::point(crate::constants::AU_KM);
    let x_km = receiver[0].div(kilometre_scale);
    let y_km = receiver[1].div(kilometre_scale);
    let z_km = receiver[2].div(kilometre_scale);
    let x_au = x_km.div(astronomical_unit_km);
    let y_au = y_km.div(astronomical_unit_km);
    let z_au = z_km.div(astronomical_unit_km);
    let semimajor_axis_au = Interval::point(crate::constants::WGS84_A_KM).div(astronomical_unit_km);
    let horizontal = sum_squares(&[x_au, y_au]).sqrt();
    if horizontal.lower() <= 0.0 {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    let longitude = atan2_from_horizontal(y_au, x_au)?;
    let mut latitude = atan2_from_horizontal(z_au, horizontal)?;
    let mut coefficient = Interval::point(0.0);
    let mut hypotenuse = Interval::point(0.0);
    for _ in 0..3 {
        let sin_latitude = latitude.sin();
        let eccentricity_term = Interval::point(WGS84_E2).mul(sin_latitude);
        let denominator = Interval::point(1.0).sub(eccentricity_term.mul(sin_latitude));
        if denominator.lower() <= 0.0 {
            return Err(CenterCertificateError::GeodeticDomain);
        }
        coefficient = semimajor_axis_au.div(denominator.sqrt());
        hypotenuse = z_au.add(coefficient.mul(eccentricity_term));
        latitude = atan2_from_horizontal(hypotenuse, horizontal)?;
    }
    let height_au = sum_squares(&[hypotenuse, horizontal])
        .sqrt()
        .sub(coefficient);
    let height_m = height_au.mul(astronomical_unit_km).mul(kilometre_scale);
    Ok([latitude, longitude, height_m])
}

fn rtklib_geodetic(receiver: [Interval; 3]) -> Result<[Interval; 3], CenterCertificateError> {
    cached_geodetic(&RTKLIB_GEODETIC_CACHE, receiver, rtklib_geodetic_uncached)
}

fn rtklib_geodetic_uncached(receiver: [Interval; 3]) -> GeodeticResult {
    let horizontal_squared = sum_squares(&[receiver[0], receiver[1]]);
    let horizontal = horizontal_squared.sqrt();
    if horizontal.lower() <= 0.0 {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    let flattening = Interval::point(crate::constants::WGS84_F);
    let eccentricity_squared = flattening.mul(Interval::point(2.0).sub(flattening));
    let mut z = receiver[2];
    let mut previous_z = Interval::point(0.0);
    let mut prime_vertical = Interval::point(crate::constants::WGS84_A_M);
    let mut converged = false;
    for _ in 0..32 {
        let delta = z.sub(previous_z);
        if delta.lower() > -1.0e-4 && delta.upper() < 1.0e-4 {
            converged = true;
            break;
        }
        previous_z = z;
        let sine_latitude = z.div(sum_squares(&[horizontal, z]).sqrt());
        let denominator =
            Interval::point(1.0).sub(eccentricity_squared.mul(sine_latitude.square()));
        if denominator.lower() <= 0.0 {
            return Err(CenterCertificateError::GeodeticDomain);
        }
        prime_vertical = Interval::point(crate::constants::WGS84_A_M).div(denominator.sqrt());
        z = receiver[2].add(prime_vertical.mul(eccentricity_squared).mul(sine_latitude));
    }
    if !converged {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    let latitude = atan2_from_horizontal(z, horizontal)?;
    let longitude_principal = atan2_from_horizontal(receiver[1], receiver[0])?;
    let height = sum_squares(&[horizontal, z]).sqrt().sub(prime_vertical);
    Ok([latitude, longitude_principal, height])
}

fn sum_squares(values: &[Interval]) -> Interval {
    let sum = values
        .iter()
        .copied()
        .fold(Interval::point(0.0), |sum, value| sum.add(value.square()));
    Interval::new(sum.lower().max(0.0), sum.upper())
}

fn atan2_from_horizontal(
    vertical: Interval,
    horizontal: Interval,
) -> Result<Interval, CenterCertificateError> {
    if horizontal.lower() <= 0.0 && horizontal.upper() >= 0.0 {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    let magnitude = sum_squares(&[vertical, horizontal]).sqrt();
    let acute = asin_interval(vertical.div(magnitude))?;
    if horizontal.lower() > 0.0 {
        Ok(acute)
    } else if horizontal.upper() < 0.0 && vertical.lower() >= 0.0 {
        Ok(Interval::point(PI).sub(acute))
    } else if horizontal.upper() < 0.0 && vertical.upper() < 0.0 {
        Ok(Interval::point(-PI).sub(acute))
    } else {
        Err(CenterCertificateError::GeodeticDomain)
    }
}

fn asin_interval(value: Interval) -> Result<Interval, CenterCertificateError> {
    if value.lower() < -1.0 || value.upper() > 1.0 {
        return Err(CenterCertificateError::GeodeticDomain);
    }
    let lower = asin_endpoint(value.lower());
    let upper = asin_endpoint(value.upper());
    Ok(Interval::new(lower.0, upper.1))
}

fn asin_endpoint(value: f64) -> (f64, f64) {
    if value == 0.0 {
        return (0.0, 0.0);
    }
    let pi_upper = f64::from_bits(PI.to_bits() + 1);
    if value == 1.0 {
        return (PI / 2.0, pi_upper / 2.0);
    }
    if value == -1.0 {
        return (-pi_upper / 2.0, -PI / 2.0);
    }
    let mut lower = -core::f64::consts::FRAC_PI_2;
    let mut upper = core::f64::consts::FRAC_PI_2;
    for _ in 0..64 {
        let midpoint = lower + (upper - lower) * 0.5;
        let sine = Interval::point(midpoint).sin();
        if sine.upper() < value {
            lower = midpoint;
        } else if sine.lower() > value {
            upper = midpoint;
        } else {
            let lower_quartile = lower + (midpoint - lower) * 0.5;
            let upper_quartile = midpoint + (upper - midpoint) * 0.5;
            let lower_sine = Interval::point(lower_quartile).sin();
            let upper_sine = Interval::point(upper_quartile).sin();
            if lower_sine.upper() < value {
                lower = lower_quartile;
            } else if lower_sine.lower() > value {
                upper = lower_quartile;
            }
            if upper_sine.upper() < value {
                lower = upper_quartile;
            } else if upper_sine.lower() > value {
                upper = upper_quartile;
            }
        }
    }
    (lower, upper)
}

fn look_angles(
    receiver: [Interval; 3],
    delta: [Interval; 3],
    latitude: Interval,
    longitude: Interval,
) -> Result<(Interval, Interval), CenterCertificateError> {
    let horizontal = sum_squares(&[receiver[0], receiver[1]]).sqrt();
    if horizontal.lower() <= 0.0 {
        return Err(CenterCertificateError::GeometryDomain);
    }
    let sin_lon = interval_sin_periodic(longitude)?;
    let cos_lon = interval_cos_periodic(longitude)?;
    let (sin_lat, cos_lat, _) = ellipsoid_terms(latitude);
    let east = [
        Interval::point(-1.0).mul(sin_lon),
        cos_lon,
        Interval::point(0.0),
    ];
    let north = [
        Interval::point(-1.0).mul(sin_lat).mul(cos_lon),
        Interval::point(-1.0).mul(sin_lat).mul(sin_lon),
        cos_lat,
    ];
    let up = [cos_lat.mul(cos_lon), cos_lat.mul(sin_lon), sin_lat];
    let east_projection = dot(delta, east);
    let north_projection = dot(delta, north);
    let up_projection = dot(delta, up);
    let direction_range = sum_squares(&[east_projection, north_projection, up_projection]).sqrt();
    let horizontal_projection = sum_squares(&[east_projection, north_projection]).sqrt();
    if horizontal_projection.lower() <= 0.0 {
        return Err(CenterCertificateError::GeometryDomain);
    }
    let elevation = asin_interval(up_projection.div(direction_range))?;
    let azimuth = asin_interval(east_projection.div(horizontal_projection))?;
    let azimuth = if north_projection.lower() >= 0.0 {
        azimuth
    } else if north_projection.upper() < 0.0 {
        if east_projection.lower() >= 0.0 {
            Interval::point(PI).sub(azimuth)
        } else if east_projection.upper() < 0.0 {
            Interval::point(-PI).sub(azimuth)
        } else {
            return Err(CenterCertificateError::GeometryDomain);
        }
    } else {
        return Err(CenterCertificateError::GeometryDomain);
    };
    Ok((azimuth, elevation))
}

fn dot(left: [Interval; 3], right: [Interval; 3]) -> Interval {
    left[0]
        .mul(right[0])
        .add(left[1].mul(right[1]))
        .add(left[2].mul(right[2]))
}

fn klobuchar(
    latitude: Interval,
    longitude: Interval,
    azimuth: Interval,
    elevation: Interval,
    second_of_day_s: f64,
    alpha: [f64; 4],
    beta: [f64; 4],
) -> Result<KlobucharIntervals, CenterCertificateError> {
    let pi = Interval::point(PI);
    let phi_u = latitude.div(pi);
    let lambda_u = longitude.div(pi);
    let elevation_sc = elevation.div(pi);
    let psi = Interval::point(0.0137)
        .div(elevation_sc.add(Interval::point(0.11)))
        .sub(Interval::point(0.022));
    let phi_i_raw = phi_u.add(psi.mul(azimuth.cos()));
    let phi_i = if phi_i_raw.lower() > 0.416 {
        Interval::point(0.416)
    } else if phi_i_raw.upper() < -0.416 {
        Interval::point(-0.416)
    } else if phi_i_raw.lower() >= -0.416 && phi_i_raw.upper() <= 0.416 {
        phi_i_raw
    } else {
        return Err(CenterCertificateError::CorrectionBranch);
    };
    let lambda_i = lambda_u.add(psi.mul(azimuth.sin()).div(phi_i.mul(pi).cos()));
    let phi_m = phi_i.add(Interval::point(0.064).mul(interval_cos_periodic(
        lambda_i.sub(Interval::point(1.617)).mul(pi),
    )?));
    let local_unwrapped = Interval::point(43_200.0)
        .mul(lambda_i)
        .add(Interval::point(second_of_day_s));
    let local_time = if local_unwrapped.lower() >= 86_400.0 {
        local_unwrapped.sub(Interval::point(86_400.0))
    } else if local_unwrapped.upper() < 0.0 {
        local_unwrapped.add(Interval::point(86_400.0))
    } else if local_unwrapped.lower() >= 0.0 && local_unwrapped.upper() < 86_400.0 {
        local_unwrapped
    } else {
        return Err(CenterCertificateError::CorrectionBranch);
    };
    let d = Interval::point(0.53).sub(elevation_sc);
    let factor = Interval::point(1.0).add(Interval::point(16.0).mul(d.square()).mul(d));
    let amplitude_raw = polynomial(phi_m, alpha);
    let amplitude = if amplitude_raw.upper() < 0.0 {
        Interval::point(0.0)
    } else if amplitude_raw.lower() > 0.0 {
        amplitude_raw
    } else {
        return Err(CenterCertificateError::CorrectionBranch);
    };
    let period_raw = polynomial(phi_m, beta);
    let period = if period_raw.upper() < 72_000.0 {
        Interval::point(72_000.0)
    } else if period_raw.lower() > 72_000.0 {
        period_raw
    } else {
        return Err(CenterCertificateError::CorrectionBranch);
    };
    let phase = Interval::point(2.0)
        .mul(Interval::point(PI))
        .mul(local_time.sub(Interval::point(50_400.0)))
        .div(period);
    let abs_phase = phase.lower().abs().max(phase.upper().abs());
    let seconds = if abs_phase < 1.57 {
        let phase_squared = phase.square();
        let phase_fourth = phase_squared.square();
        factor.mul(
            Interval::point(5.0e-9).add(
                amplitude.mul(
                    Interval::point(1.0)
                        .sub(phase_squared.div(Interval::point(2.0)))
                        .add(phase_fourth.div(Interval::point(24.0))),
                ),
            ),
        )
    } else if phase.lower() > 1.57 || phase.upper() < -1.57 {
        factor.mul(Interval::point(5.0e-9))
    } else {
        return Err(CenterCertificateError::CorrectionBranch);
    };
    Ok(KlobucharIntervals {
        raw_phi_i_semicircles: phi_i_raw,
        phi_m_semicircles: phi_m,
        local_time_seconds: local_time,
        phase_radians: phase,
        delay_m: Interval::point(C_M_S).mul(seconds),
    })
}

fn polynomial(value: Interval, coefficients: [f64; 4]) -> Interval {
    Interval::point(coefficients[0]).add(value.mul(
        Interval::point(coefficients[1]).add(value.mul(
            Interval::point(coefficients[2]).add(value.mul(Interval::point(coefficients[3]))),
        )),
    ))
}

fn interval_sin_periodic(angle: Interval) -> Result<Interval, CenterCertificateError> {
    Ok(reduce_periodic(angle)
        .ok_or(CenterCertificateError::GeometryDomain)?
        .sin())
}

fn interval_cos_periodic(angle: Interval) -> Result<Interval, CenterCertificateError> {
    Ok(reduce_periodic(angle)
        .ok_or(CenterCertificateError::CorrectionBranch)?
        .cos())
}

fn reduce_periodic(mut angle: Interval) -> Option<Interval> {
    let true_pi = Interval::new(PI, f64::from_bits(PI.to_bits() + 1));
    let true_tau = Interval::point(2.0).mul(true_pi);
    while angle.lower() > 4.0 {
        angle = angle.sub(true_tau);
    }
    while angle.upper() < -4.0 {
        angle = angle.add(true_tau);
    }
    (angle.lower() >= -4.0 && angle.upper() <= 4.0).then_some(angle)
}

fn rtklib_troposphere(
    latitude: Interval,
    height: Interval,
    elevation: Interval,
) -> Result<Interval, CenterCertificateError> {
    if height.lower() <= 0.0 || height.upper() >= 10_000.0 || elevation.lower() <= 0.0 {
        return Err(CenterCertificateError::TroposphereDomain);
    }
    let pressure_factor = Interval::point(1.0).sub(Interval::point(2.2557e-5).mul(height));
    if pressure_factor.lower() <= 0.0 {
        return Err(CenterCertificateError::TroposphereDomain);
    }
    let pressure =
        Interval::point(1013.25).mul(pressure_factor.ln().mul(Interval::point(5.2568)).exp());
    let temperature = Interval::point(288.16).sub(Interval::point(0.0065).mul(height));
    if temperature.lower() <= 38.45 {
        return Err(CenterCertificateError::TroposphereDomain);
    }
    let vapor_exponent = Interval::point(17.15)
        .mul(temperature)
        .sub(Interval::point(4684.0))
        .div(temperature.sub(Interval::point(38.45)));
    let vapor = Interval::point(6.108)
        .mul(Interval::point(0.7))
        .mul(vapor_exponent.exp());
    let denominator = Interval::point(1.0)
        .sub(Interval::point(0.00266).mul(Interval::point(2.0).mul(latitude).cos()))
        .sub(Interval::point(2.8e-7).mul(height));
    if denominator.lower() <= 0.0 {
        return Err(CenterCertificateError::TroposphereDomain);
    }
    let sine_elevation = elevation.sin();
    if sine_elevation.lower() <= 0.0 {
        return Err(CenterCertificateError::TroposphereDomain);
    }
    let dry = Interval::point(0.0022768)
        .mul(pressure)
        .div(denominator)
        .div(sine_elevation);
    let wet = Interval::point(0.002277)
        .mul(
            Interval::point(1255.0)
                .div(temperature)
                .add(Interval::point(0.05)),
        )
        .mul(vapor)
        .div(sine_elevation);
    Ok(dry.add(wet))
}
