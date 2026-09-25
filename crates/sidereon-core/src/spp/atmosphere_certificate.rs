//! Conditional position-region bounds for GPS RTKLIB SPP atmosphere and weights.
//!
//! This certificate is deliberately limited to the reference RTKLIB
//! pseudorange-placement recipe, GPS single-frequency codes, Klobuchar, and
//! RTKLIB's Saastamoinen troposphere model. It refuses unsupported models and
//! any region whose satellite selection or Klobuchar branches cannot be shown
//! fixed. Regional derivatives are for the ideal WGS-84 map. Callers provide
//! certified enclosures for center geodetic and satellite-angle values; these
//! bounds do not certify the finite-inverse or endpoint arithmetic errors.
//! The troposphere envelope follows RTKLIB's signed-height branches: zero delay
//! below -100 m, and a sea-level-clamped standard atmosphere from -100 m to
//! 10000 m with humidity 0.7. Regions crossing either discontinuous cutoff are
//! refused. The clamp at zero is continuous and 1-Lipschitz.
//! The certified receiver-height domain is `[-1000, 10000)` m; a region that
//! reaches below the ionosphere cutoff is rejected.
//! Since sine is globally 1-Lipschitz, subtracting the regional elevation
//! variation from a certified center sine lower bound bounds the whole ball.

use std::collections::BTreeSet;

use super::interval_certificate::Interval;
use crate::ionex::klobuchar_l1_components;
use crate::spp::{
    GnssSatelliteId, PseudorangeCode, RejectionReason, Selection, SolveInputs, TroposphereModel,
    BROADCAST_IONOSPHERE_ERROR_FACTOR, CODE_PHASE_ERROR_RATIO, C_M_S, ELEVATION_MASK_RAD, F_L1_HZ,
    OMEGA_E_DOT_RAD_S, PHASE_ERROR_ELEVATION_M, TROPOSPHERE_MODEL_ERROR_M,
};

const PI: f64 = core::f64::consts::PI;
const WGS84_M_MIN_M: f64 = 6_335_439.32;
pub(super) const IONOSPHERE_HEIGHT_CUTOFF_M: f64 = -1_000.0;
pub(super) const TROPOSPHERE_HEIGHT_CUTOFF_M: f64 = -100.0;
pub(super) const HEIGHT_MAX_M: f64 = 10_000.0;
const TROP_HUMIDITY: f64 = 0.7;
const TROP_PRESSURE_SEA_LEVEL_HPA: f64 = 1013.25;
const TROP_PRESSURE_SCALE: f64 = 2.2557e-5;
const TROP_PRESSURE_EXPONENT: f64 = 5.2568;
const TROP_TEMP_SEA_LEVEL_K: f64 = 288.16;
const TROP_TEMP_LAPSE_K_M: f64 = 0.0065;
const TROP_DENOMINATOR_LAT: f64 = 0.00266;
const TROP_DENOMINATOR_HEIGHT: f64 = 2.8e-7;

/// Position-region terms to combine with the independently derived center errors.
pub(super) struct SatelliteAtmosphereBounds {
    /// Bound for the directional derivative norm of the geometric design row.
    pub design_derivative: f64,
    /// Bound for the omitted Sagnac and atmospheric range-gradient terms.
    pub correction_gradient: f64,
    /// Upper bound for inverse pseudorange variance throughout the region.
    pub weight_max: f64,
    /// Bound for the inverse-variance gradient norm throughout the region.
    pub weight_gradient: f64,
}

/// Ideal WGS-84 center geodetic values with caller-certified absolute errors.
pub(super) struct CenterGeodeticEnclosure {
    /// Center geodetic latitude in radians.
    pub latitude_rad: f64,
    /// Absolute latitude error bound in radians.
    pub latitude_error_rad: f64,
    /// Center geodetic longitude in radians.
    pub longitude_rad: f64,
    /// Absolute longitude error bound in radians.
    pub longitude_error_rad: f64,
    /// Center signed ellipsoidal height in metres.
    pub height_m: f64,
    /// Absolute height error bound in metres.
    pub height_error_m: f64,
}

/// Ideal WGS-84 center look angles with caller-certified absolute errors.
pub(super) struct CenterSatelliteAngles {
    /// Satellite identifier matching the native state and selection.
    pub satellite_id: GnssSatelliteId,
    /// Center azimuth in radians.
    pub azimuth_rad: f64,
    /// Absolute azimuth error bound in radians.
    pub azimuth_error_rad: f64,
    /// Center elevation in radians.
    pub elevation_rad: f64,
    /// Absolute elevation error bound in radians.
    pub elevation_error_rad: f64,
    /// Certified lower bound for sine of the ideal center elevation.
    pub sin_elevation_lower: f64,
    /// Absolute center inverse-weight error from the caller's endpoint enclosure.
    pub weight_error: f64,
    /// Certified discrepancies of source-kernel center values from ideal center
    /// values, including degree conversion and libm evaluation errors.
    pub klobuchar_error: KlobucharCenterError,
}

/// Center evaluation error enclosures for discontinuous Klobuchar decisions.
pub(super) struct KlobucharCenterError {
    /// Unclamped pierce-point latitude discrepancy, semicircles.
    pub raw_phi_i_semicircles: f64,
    /// Geomagnetic latitude discrepancy, semicircles.
    pub phi_m_semicircles: f64,
    /// Wrapped local-time discrepancy, seconds.
    pub local_time_seconds: f64,
    /// Diurnal phase discrepancy, radians.
    pub phase_radians: f64,
}

/// Why a receiver ball could not be bounded under the documented preconditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AtmosphereBoundError {
    /// A non-finite value, invalid radius, or inconsistent satellite list was supplied.
    InvalidInput,
    /// The requested solve is outside the documented GPS/RTKLIB/single-code domain.
    UnsupportedModel,
    /// A selected or rejected satellite can cross the elevation mask in the ball.
    SelectionMayChange,
    /// A geodetic/troposphere validity boundary intersects the ball.
    HeightBranchMayChange,
    /// The local-time wrap or Klobuchar night/day branch is not fixed in the ball.
    IonosphereBranchMayChange,
    /// Total variance could approach or cross zero under the derivative bound.
    VarianceNotPositive,
}

#[derive(Clone, Copy)]
struct GeometryBounds {
    latitude_per_m: f64,
    longitude_per_m: f64,
    elevation_per_m: f64,
    azimuth_per_m: f64,
}

fn distance_lower(left: [f64; 3], right: [f64; 3]) -> f64 {
    let sum = (0..3).fold(Interval::point(0.0), |sum, axis| {
        sum.add(
            Interval::point(left[axis])
                .sub(Interval::point(right[axis]))
                .square(),
        )
    });
    sum.sqrt().lower()
}

fn xy_radius_lower(position: [f64; 3]) -> f64 {
    Interval::point(position[0])
        .square()
        .add(Interval::point(position[1]).square())
        .sqrt()
        .lower()
}

fn upper_product(left: f64, right: f64) -> f64 {
    Interval::point(left).mul(Interval::point(right)).upper()
}

fn upper_quotient(numerator: f64, denominator: f64) -> f64 {
    Interval::point(numerator)
        .div(Interval::point(denominator))
        .upper()
}

fn upper_sum(left: f64, right: f64) -> f64 {
    Interval::point(left).add(Interval::point(right)).upper()
}

fn lower_difference(left: f64, right: f64) -> f64 {
    Interval::point(left).sub(Interval::point(right)).lower()
}

fn lower_product(left: f64, right: f64) -> f64 {
    Interval::point(left).mul(Interval::point(right)).lower()
}

fn geometry_bounds(
    receiver_ecef_m: [f64; 3],
    satellite_ecef_m: [f64; 3],
    radius_m: f64,
    center_geodetic: &CenterGeodeticEnclosure,
) -> Result<GeometryBounds, AtmosphereBoundError> {
    let height = Interval::point(center_geodetic.height_m);
    let height_error = Interval::point(center_geodetic.height_error_m);
    let radius = Interval::point(radius_m);
    let height_min = height.sub(height_error).sub(radius).lower();
    let height_max = height.add(height_error).add(radius).upper();
    if height_min < IONOSPHERE_HEIGHT_CUTOFF_M {
        return Err(AtmosphereBoundError::HeightBranchMayChange);
    }
    if height_max >= HEIGHT_MAX_M {
        return Err(AtmosphereBoundError::HeightBranchMayChange);
    }
    let rho_min = Interval::point(distance_lower(satellite_ecef_m, receiver_ecef_m))
        .sub(radius)
        .lower();
    let xy_min = Interval::point(xy_radius_lower(receiver_ecef_m))
        .sub(radius)
        .lower();
    if !(rho_min > 0.0 && xy_min > 0.0) {
        return Err(AtmosphereBoundError::InvalidInput);
    }

    let meridional_radius_min = Interval::point(WGS84_M_MIN_M)
        .add(Interval::point(height_min))
        .lower();
    if meridional_radius_min <= 0.0 {
        return Err(AtmosphereBoundError::InvalidInput);
    }
    let latitude_per_m = upper_quotient(1.0, meridional_radius_min);
    let longitude_per_m = upper_quotient(1.0, xy_min);
    let range_direction_per_m = upper_quotient(1.0, rho_min);
    const COS_ELEVATION_FLOOR: f64 = 0.09;
    let lat_lon_rate = upper_sum(latitude_per_m, longitude_per_m);
    let elevation_per_m = upper_quotient(
        upper_sum(range_direction_per_m, lat_lon_rate),
        COS_ELEVATION_FLOOR,
    );
    let azimuth_per_m = upper_quotient(
        upper_sum(range_direction_per_m, upper_product(2.0, lat_lon_rate)),
        COS_ELEVATION_FLOOR,
    );
    if elevation_per_m >= 5.5e-6 || azimuth_per_m >= 1.04e-5 {
        return Err(AtmosphereBoundError::InvalidInput);
    }
    Ok(GeometryBounds {
        latitude_per_m,
        longitude_per_m,
        elevation_per_m,
        azimuth_per_m,
    })
}

fn tropo_gradient_bound(
    sin_min: f64,
    geometry: GeometryBounds,
    height_min: f64,
    height_max: f64,
) -> Result<f64, AtmosphereBoundError> {
    if !(sin_min > 0.0 && sin_min <= 1.0) {
        return Err(AtmosphereBoundError::SelectionMayChange);
    }
    if height_max < TROPOSPHERE_HEIGHT_CUTOFF_M {
        return Ok(0.0);
    }
    if height_min < TROPOSPHERE_HEIGHT_CUTOFF_M || height_max >= HEIGHT_MAX_M {
        return Err(AtmosphereBoundError::HeightBranchMayChange);
    }

    let pressure_min_factor =
        lower_difference(1.0, upper_product(TROP_PRESSURE_SCALE, HEIGHT_MAX_M));
    let pressure_max = TROP_PRESSURE_SEA_LEVEL_HPA;
    let temperature_min = lower_difference(
        TROP_TEMP_SEA_LEVEL_K,
        upper_product(TROP_TEMP_LAPSE_K_M, HEIGHT_MAX_M),
    );
    let denominator_min = lower_difference(
        lower_difference(1.0, TROP_DENOMINATOR_LAT),
        upper_product(0.00028, 10.0),
    );
    let temperature_max = TROP_TEMP_SEA_LEVEL_K;
    let exponent_max = Interval::point(17.15)
        .mul(Interval::point(temperature_max))
        .sub(Interval::point(4684.0))
        .div(Interval::point(temperature_max).sub(Interval::point(38.45)));
    if exponent_max.upper() > 8.0 {
        return Err(AtmosphereBoundError::InvalidInput);
    }
    let vapor_max = Interval::point(6.108)
        .mul(Interval::point(TROP_HUMIDITY))
        .mul(exponent_max.exp())
        .upper();
    let temperature_offset = lower_difference(temperature_min, 38.45);
    let vapor_log_derivative_max = upper_quotient(
        Interval::point(4684.0)
            .sub(Interval::point(17.15).mul(Interval::point(38.45)))
            .upper(),
        lower_product(temperature_offset, temperature_offset),
    );
    let pressure_height_max = upper_quotient(
        upper_product(
            upper_product(TROP_PRESSURE_SEA_LEVEL_HPA, TROP_PRESSURE_EXPONENT),
            TROP_PRESSURE_SCALE,
        ),
        pressure_min_factor,
    );
    let vapor_height_max = upper_product(
        upper_product(vapor_max, vapor_log_derivative_max),
        TROP_TEMP_LAPSE_K_M,
    );
    let dry_zenith_max = upper_quotient(upper_product(0.0022768, pressure_max), denominator_min);
    let wet_temperature_term = upper_sum(upper_quotient(1255.0, temperature_min), 0.05);
    let wet_zenith_max = upper_product(upper_product(0.002277, wet_temperature_term), vapor_max);
    let zenith_max = upper_sum(dry_zenith_max, wet_zenith_max);
    let denominator_squared = lower_product(denominator_min, denominator_min);
    let pressure_height_term = upper_quotient(pressure_height_max, denominator_min);
    let denominator_height_term = upper_quotient(
        upper_product(pressure_max, TROP_DENOMINATOR_HEIGHT),
        denominator_squared,
    );
    let dry_height_max = upper_product(
        0.0022768,
        upper_sum(pressure_height_term, denominator_height_term),
    );
    let temperature_squared = lower_product(temperature_min, temperature_min);
    let wet_temperature_derivative = upper_quotient(
        upper_product(1255.0, TROP_TEMP_LAPSE_K_M),
        temperature_squared,
    );
    let wet_height_max = upper_product(
        0.002277,
        upper_sum(
            upper_product(upper_product(wet_temperature_derivative, vapor_max), 1.0),
            upper_product(wet_temperature_term, vapor_height_max),
        ),
    );
    let zenith_height_max = upper_sum(dry_height_max, wet_height_max);
    let zenith_latitude_max = upper_quotient(
        upper_product(
            upper_product(upper_product(0.0022768, pressure_max), TROP_DENOMINATOR_LAT),
            2.0,
        ),
        denominator_squared,
    );
    if !(sin_min > 0.0 && pressure_min_factor > 0.0 && temperature_min > 38.45) {
        return Err(AtmosphereBoundError::InvalidInput);
    }

    let sine_squared_lower = Interval::point(sin_min).square().lower();
    if sine_squared_lower <= 0.0 {
        return Err(AtmosphereBoundError::SelectionMayChange);
    }
    // On the active branch RTKLIB evaluates the atmosphere at max(h, 0).
    // Its clamp is continuous and 1-Lipschitz; the interval envelope above
    // covers [0, 10000], so integrating the derivative bound on either side of
    // the zero-height kink bounds the whole segment without assuming a
    // derivative exists at the kink itself.
    let height_term = upper_quotient(zenith_height_max, sin_min);
    let latitude_term = upper_product(
        upper_quotient(zenith_latitude_max, sin_min),
        geometry.latitude_per_m,
    );
    let elevation_term = upper_product(
        upper_quotient(zenith_max, sine_squared_lower),
        geometry.elevation_per_m,
    );
    Ok(upper_sum(
        upper_sum(height_term, latitude_term),
        elevation_term,
    ))
}

fn klobuchar_night_bounds(
    inputs: &SolveInputs,
    geodetic: &CenterGeodeticEnclosure,
    angles: &CenterSatelliteAngles,
    geometry: GeometryBounds,
    radius_m: f64,
) -> Result<(f64, f64), AtmosphereBoundError> {
    let height = Interval::point(geodetic.height_m);
    let height_error = Interval::point(geodetic.height_error_m);
    let radius = Interval::point(radius_m);
    let height_min = height.sub(height_error).sub(radius).lower();
    let height_max = height.add(height_error).add(radius).upper();
    if height_max < IONOSPHERE_HEIGHT_CUTOFF_M {
        return Ok((0.0, 0.0));
    }
    if height_min < IONOSPHERE_HEIGHT_CUTOFF_M {
        return Err(AtmosphereBoundError::HeightBranchMayChange);
    }
    let components = klobuchar_l1_components(
        geodetic.latitude_rad.to_degrees(),
        geodetic.longitude_rad.to_degrees(),
        angles.azimuth_rad.to_degrees(),
        angles.elevation_rad.to_degrees(),
        inputs.t_rx_second_of_day_s,
        inputs.klobuchar.alpha,
        inputs.klobuchar.beta,
    );
    if ![
        components.phi_i,
        components.lambda_i,
        components.phi_m,
        components.t,
        components.f,
        components.per,
        components.x,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        return Err(AtmosphereBoundError::IonosphereBranchMayChange);
    }
    let total_elevation_error = angles.elevation_error_rad.max(0.0);
    let total_elevation_error = Interval::point(total_elevation_error)
        .add(Interval::point(radius_m).mul(Interval::point(geometry.elevation_per_m)))
        .upper();
    let elevation_interval = Interval::point(angles.elevation_rad)
        .add(Interval::new(-total_elevation_error, total_elevation_error))
        .div(Interval::point(PI));
    let e_min = elevation_interval.lower();
    let e_max = elevation_interval.upper();
    if !(e_min > 0.0 && e_max < 0.53) {
        return Err(AtmosphereBoundError::IonosphereBranchMayChange);
    }

    let le = upper_quotient(geometry.elevation_per_m, PI);
    let l_phi_u = upper_quotient(geometry.latitude_per_m, PI);
    let psi_denominator = Interval::point(e_min).add(Interval::point(0.11));
    let psi_abs = Interval::point(0.0137)
        .div(psi_denominator)
        .add(Interval::point(0.022))
        .upper();
    let psi_derivative = Interval::point(0.0137)
        .div(psi_denominator.square())
        .upper();
    let psi_error = Interval::point(psi_derivative)
        .mul(Interval::point(total_elevation_error))
        .div(Interval::point(PI))
        .upper();
    let azimuth_error = Interval::point(angles.azimuth_error_rad)
        .add(Interval::point(radius_m).mul(Interval::point(geometry.azimuth_per_m)))
        .upper();
    let longitude_error = Interval::point(geodetic.longitude_error_rad)
        .add(Interval::point(radius_m).mul(Interval::point(geometry.longitude_per_m)))
        .upper();
    let raw_phi_i = Interval::point(components.phi_i);
    let raw_phi_i_error = Interval::point(geodetic.latitude_error_rad)
        .div(Interval::point(PI))
        .add(Interval::point(radius_m).mul(Interval::point(l_phi_u)))
        .add(Interval::point(psi_error))
        .add(Interval::point(psi_abs).mul(Interval::point(azimuth_error)))
        .add(Interval::point(
            angles.klobuchar_error.raw_phi_i_semicircles,
        ));
    if raw_phi_i.sub(raw_phi_i_error).lower() <= -0.416
        || raw_phi_i.add(raw_phi_i_error).upper() >= 0.416
    {
        return Err(AtmosphereBoundError::IonosphereBranchMayChange);
    }

    let cosine_min = 0.258;
    let cosine_min_squared = lower_product(cosine_min, cosine_min);
    let lambda_i_error = Interval::point(longitude_error)
        .div(Interval::point(PI))
        .add(Interval::point(psi_error).div(Interval::point(cosine_min)))
        .add(
            Interval::point(psi_abs)
                .mul(Interval::point(azimuth_error))
                .div(Interval::point(cosine_min)),
        )
        .add(
            Interval::point(psi_abs)
                .mul(Interval::point(PI))
                .mul(raw_phi_i_error)
                .div(Interval::point(cosine_min_squared)),
        )
        .upper();
    let phi_m_error = raw_phi_i_error
        .add(
            Interval::point(0.064)
                .mul(Interval::point(PI))
                .mul(Interval::point(lambda_i_error)),
        )
        .add(Interval::point(angles.klobuchar_error.phi_m_semicircles));
    let phi_m_error = phi_m_error.upper();
    let phi_m = components.phi_m;
    let phi_m_max = upper_sum(phi_m.abs(), phi_m_error);
    let beta = inputs.klobuchar.beta;
    let phi_m_squared = upper_product(phi_m_max, phi_m_max);
    let period_derivative = upper_sum(
        beta[1].abs(),
        upper_sum(
            upper_product(upper_product(2.0, beta[2].abs()), phi_m_max),
            upper_product(
                upper_product(upper_product(3.0, beta[3].abs()), phi_m_squared),
                1.0,
            ),
        ),
    );
    let local_time_error = Interval::point(43_200.0)
        .mul(Interval::point(lambda_i_error))
        .add(Interval::point(angles.klobuchar_error.local_time_seconds))
        .upper();
    if components.t <= local_time_error
        || Interval::point(86_400.0)
            .sub(Interval::point(components.t))
            .lower()
            <= local_time_error
    {
        return Err(AtmosphereBoundError::IonosphereBranchMayChange);
    }
    let phase_period = Interval::point(72_000.0).square();
    let phase_fraction_error = Interval::point(local_time_error)
        .div(Interval::point(72_000.0))
        .add(
            Interval::point(50_400.0)
                .mul(Interval::point(period_derivative))
                .mul(Interval::point(phi_m_error))
                .div(phase_period),
        );
    let phase_error = Interval::point(2.0)
        .mul(Interval::point(PI))
        .mul(phase_fraction_error)
        .add(Interval::point(angles.klobuchar_error.phase_radians))
        .upper();
    if lower_difference(components.x.abs(), phase_error) <= 1.57 {
        return Err(AtmosphereBoundError::IonosphereBranchMayChange);
    }

    let mapping_excess = Interval::point(0.53).sub(Interval::point(e_min));
    let f_max = Interval::point(1.0)
        .add(
            Interval::point(16.0)
                .mul(mapping_excess.square())
                .mul(mapping_excess),
        )
        .upper();
    let f_gradient = Interval::point(48.0)
        .mul(mapping_excess.square())
        .mul(Interval::point(le))
        .upper();
    let carrier_ratio = Interval::point(F_L1_HZ).div(Interval::point(F_L1_HZ));
    let carrier_scale = carrier_ratio.square();
    let delay_scale = Interval::point(C_M_S)
        .mul(carrier_scale)
        .mul(Interval::point(5.0e-9));
    let delay_max = delay_scale.mul(Interval::point(f_max)).upper();
    let delay_gradient = delay_scale.mul(Interval::point(f_gradient)).upper();
    Ok((delay_max, delay_gradient))
}

/// Derive conditional per-satellite design, correction, and inverse-variance bounds.
///
/// The geodetic and angle inputs must enclose the ideal WGS-84 center geometry;
/// the per-satellite Klobuchar errors must enclose the source-kernel center
/// intermediates, and `weight_error` must enclose the center inverse-weight
/// difference. ECEF distance lower bounds use the interval arithmetic module.
/// Candidate positions include every observation with a usable native
/// transmit-epoch state. Missing-ephemeris
/// rejects are fixed only for RTKLIB's receiver-independent transmit placement.
/// There must be exactly one angle enclosure per candidate position, and the
/// candidate list must include every observation with a usable native state.
///
/// This helper does not establish its input enclosures, certify floating-point
/// errors in the caller's endpoint map, or support non-GPS, ionosphere-free,
/// robust, or non-RTKLIB troposphere solves.
pub(super) fn satellite_regions(
    inputs: &SolveInputs,
    center_ecef_m: [f64; 3],
    radius_m: f64,
    selection: &Selection,
    candidate_positions_ecef_m: &[(GnssSatelliteId, [f64; 3])],
    center_geodetic: &CenterGeodeticEnclosure,
    center_angles: &[CenterSatelliteAngles],
) -> Result<Vec<SatelliteAtmosphereBounds>, AtmosphereBoundError> {
    if !radius_m.is_finite()
        || radius_m <= 0.0
        || center_ecef_m.iter().any(|value| !value.is_finite())
        || !center_geodetic.latitude_rad.is_finite()
        || !center_geodetic.longitude_rad.is_finite()
        || !center_geodetic.height_m.is_finite()
        || ![
            center_geodetic.latitude_error_rad,
            center_geodetic.longitude_error_rad,
            center_geodetic.height_error_m,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
        || center_geodetic.latitude_error_rad > 2.0 * PI
        || center_geodetic.longitude_error_rad > 2.0 * PI
        || center_geodetic.height_error_m > 1.0e6
        || center_geodetic.latitude_rad.abs() >= PI / 2.0
        || Interval::point(center_geodetic.latitude_rad.abs())
            .add(Interval::point(center_geodetic.latitude_error_rad))
            .upper()
            >= PI / 2.0
        || center_geodetic.longitude_rad.abs() > PI
        || center_ecef_m.iter().any(|value| value.abs() > 1.0e7)
        || radius_m > 1.0e6
        || inputs.pseudorange_code != PseudorangeCode::SingleFrequency
        || inputs.troposphere_model != TroposphereModel::Rtklib
        || inputs.robust.is_some()
        || !inputs.t_rx_second_of_day_s.is_finite()
        || !(0.0..86_400.0).contains(&inputs.t_rx_second_of_day_s)
        || inputs
            .klobuchar
            .alpha
            .iter()
            .chain(inputs.klobuchar.beta.iter())
            .any(|value| !value.is_finite() || value.abs() > 1.0e6)
        || inputs
            .observations
            .iter()
            .any(|observation| observation.satellite_id.system != crate::id::GnssSystem::Gps)
    {
        return Err(AtmosphereBoundError::UnsupportedModel);
    }
    if selection.used.len() < 4
        || selection.used.len() > 64
        || selection.used.len() != selection.weights.len()
        || selection.used.len() != selection.lines_of_sight.len()
        || selection.used.len() != selection.residuals_m.len()
        || selection
            .used
            .iter()
            .any(|sat| sat.system != crate::id::GnssSystem::Gps)
    {
        return Err(AtmosphereBoundError::InvalidInput);
    }

    let height = Interval::point(center_geodetic.height_m);
    let height_error = Interval::point(center_geodetic.height_error_m);
    let radius = Interval::point(radius_m);
    let height_min = height.sub(height_error).sub(radius).lower();
    let height_max = height.add(height_error).add(radius).upper();
    if height_min < IONOSPHERE_HEIGHT_CUTOFF_M || height_max >= HEIGHT_MAX_M {
        return Err(AtmosphereBoundError::HeightBranchMayChange);
    }
    if height_min < TROPOSPHERE_HEIGHT_CUTOFF_M && height_max >= TROPOSPHERE_HEIGHT_CUTOFF_M {
        return Err(AtmosphereBoundError::HeightBranchMayChange);
    }
    let observation_ids: BTreeSet<_> = inputs
        .observations
        .iter()
        .map(|observation| observation.satellite_id)
        .collect();
    if observation_ids.len() != inputs.observations.len() {
        return Err(AtmosphereBoundError::InvalidInput);
    }
    let mut candidate_ids = BTreeSet::new();
    for (satellite, position) in candidate_positions_ecef_m {
        if !observation_ids.contains(satellite)
            || !candidate_ids.insert(*satellite)
            || position
                .iter()
                .any(|value| !value.is_finite() || value.abs() > 1.0e9)
        {
            return Err(AtmosphereBoundError::InvalidInput);
        }
    }
    let mut angle_ids = BTreeSet::new();
    for angles in center_angles {
        if !candidate_ids.contains(&angles.satellite_id)
            || !angle_ids.insert(angles.satellite_id)
            || !angles.azimuth_rad.is_finite()
            || !angles.elevation_rad.is_finite()
            || angles.azimuth_rad.abs() > 2.0 * PI
            || angles.elevation_rad.abs() > PI / 2.0
            || ![
                angles.azimuth_error_rad,
                angles.elevation_error_rad,
                angles.weight_error,
                angles.klobuchar_error.raw_phi_i_semicircles,
                angles.klobuchar_error.phi_m_semicircles,
                angles.klobuchar_error.local_time_seconds,
                angles.klobuchar_error.phase_radians,
            ]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            || angles.azimuth_error_rad > 2.0 * PI
            || angles.elevation_error_rad > PI
            || angles.weight_error > 1.0e12
            || angles.klobuchar_error.raw_phi_i_semicircles > 2.0
            || angles.klobuchar_error.phi_m_semicircles > 2.0
            || angles.klobuchar_error.local_time_seconds > 86_400.0
            || angles.klobuchar_error.phase_radians > 1.0e6
            || !angles.sin_elevation_lower.is_finite()
            || !(-1.0..=1.0).contains(&angles.sin_elevation_lower)
        {
            return Err(AtmosphereBoundError::InvalidInput);
        }
    }
    if angle_ids != candidate_ids {
        return Err(AtmosphereBoundError::InvalidInput);
    }
    for satellite in &selection.used {
        if !candidate_ids.contains(satellite) {
            return Err(AtmosphereBoundError::InvalidInput);
        }
    }

    let mut states = Vec::with_capacity(selection.used.len());
    for (satellite, position) in candidate_positions_ecef_m {
        let angles = center_angles
            .iter()
            .find(|entry| entry.satellite_id == *satellite)
            .ok_or(AtmosphereBoundError::InvalidInput)?;
        let geometry = geometry_bounds(center_ecef_m, *position, radius_m, center_geodetic)?;
        let elevation_error = Interval::point(angles.elevation_error_rad)
            .add(Interval::point(radius_m).mul(Interval::point(geometry.elevation_per_m)))
            .upper();
        if upper_sum(angles.elevation_rad.abs(), elevation_error) >= 1.47 {
            return Err(AtmosphereBoundError::SelectionMayChange);
        }
        let selected = selection.used.contains(satellite);
        if selected {
            if Interval::point(angles.elevation_rad)
                .sub(Interval::point(elevation_error))
                .lower()
                <= ELEVATION_MASK_RAD
            {
                return Err(AtmosphereBoundError::SelectionMayChange);
            }
            states.push((*satellite, *position, angles, geometry));
        } else {
            let rejected_low = selection.rejected.iter().any(|rejected| {
                rejected.satellite_id == *satellite
                    && rejected.reason == RejectionReason::LowElevation
            });
            if rejected_low
                && Interval::point(angles.elevation_rad)
                    .add(Interval::point(elevation_error))
                    .upper()
                    >= ELEVATION_MASK_RAD
            {
                return Err(AtmosphereBoundError::SelectionMayChange);
            }
            if !rejected_low
                && !selection.rejected.iter().any(|rejected| {
                    rejected.satellite_id == *satellite
                        && rejected.reason == RejectionReason::NoEphemeris
                })
            {
                return Err(AtmosphereBoundError::UnsupportedModel);
            }
        }
    }
    for rejected in &selection.rejected {
        if rejected.reason == RejectionReason::LowElevation
            && !candidate_ids.contains(&rejected.satellite_id)
        {
            return Err(AtmosphereBoundError::InvalidInput);
        }
        if rejected.reason == RejectionReason::NoEphemeris
            && candidate_ids.contains(&rejected.satellite_id)
        {
            return Err(AtmosphereBoundError::InvalidInput);
        }
        if !matches!(
            rejected.reason,
            RejectionReason::NoEphemeris | RejectionReason::LowElevation
        ) {
            return Err(AtmosphereBoundError::UnsupportedModel);
        }
    }
    let mut classified_ids = selection.used.iter().copied().collect::<BTreeSet<_>>();
    if selection
        .rejected
        .iter()
        .any(|rejected| !classified_ids.insert(rejected.satellite_id))
        || classified_ids != observation_ids
    {
        return Err(AtmosphereBoundError::InvalidInput);
    }

    let mut bounds = Vec::with_capacity(selection.used.len());
    for (index, satellite) in selection.used.iter().enumerate() {
        let (_, sat_position, angles, geometry) = states
            .iter()
            .find(|(candidate, _, _, _)| candidate == satellite)
            .ok_or(AtmosphereBoundError::InvalidInput)?;
        let rho_min = Interval::point(distance_lower(*sat_position, center_ecef_m))
            .sub(radius)
            .lower();
        if rho_min <= 0.0 {
            return Err(AtmosphereBoundError::InvalidInput);
        }
        let design_derivative = upper_quotient(1.0, rho_min);
        let sagnac_gradient = Interval::point(OMEGA_E_DOT_RAD_S)
            .div(Interval::point(C_M_S))
            .mul(
                Interval::point(sat_position[0])
                    .square()
                    .add(Interval::point(sat_position[1]).square())
                    .sqrt(),
            )
            .upper();

        let mut delay_gradient = 0.0;
        let mut ionosphere_delay_max = 0.0;
        if inputs.corrections.ionosphere
            && height_max >= IONOSPHERE_HEIGHT_CUTOFF_M
            && height_min < IONOSPHERE_HEIGHT_CUTOFF_M
        {
            return Err(AtmosphereBoundError::HeightBranchMayChange);
        }
        if inputs.corrections.ionosphere && height_max >= IONOSPHERE_HEIGHT_CUTOFF_M {
            (ionosphere_delay_max, delay_gradient) =
                klobuchar_night_bounds(inputs, center_geodetic, angles, *geometry, radius_m)?;
        }
        let sin_min = regional_sine_lower(
            angles.sin_elevation_lower,
            geometry.elevation_per_m,
            radius_m,
        );
        if sin_min <= 0.0 {
            return Err(AtmosphereBoundError::SelectionMayChange);
        }
        let sine_squared_lower = Interval::point(sin_min).square().lower();
        if sine_squared_lower <= 0.0 {
            return Err(AtmosphereBoundError::SelectionMayChange);
        }
        let troposphere_gradient = if inputs.corrections.troposphere {
            tropo_gradient_bound(sin_min, *geometry, height_min, height_max)?
        } else {
            0.0
        };
        let correction_gradient = upper_sum(
            sagnac_gradient,
            upper_sum(delay_gradient, troposphere_gradient),
        );

        let ionosphere_variance_gradient = if inputs.corrections.ionosphere {
            Interval::point(BROADCAST_IONOSPHERE_ERROR_FACTOR)
                .square()
                .mul(Interval::point(2.0))
                .mul(Interval::point(ionosphere_delay_max))
                .mul(Interval::point(delay_gradient))
                .upper()
        } else {
            0.0
        };
        let troposphere_variance_gradient = if inputs.corrections.troposphere {
            let sine_plus_floor = Interval::point(sin_min).add(Interval::point(0.1));
            Interval::point(2.0)
                .mul(Interval::point(TROPOSPHERE_MODEL_ERROR_M).square())
                .div(sine_plus_floor.square().mul(sine_plus_floor))
                .mul(Interval::point(geometry.elevation_per_m))
                .upper()
        } else {
            0.0
        };
        let code_variance_scale = Interval::point(CODE_PHASE_ERROR_RATIO)
            .square()
            .mul(Interval::point(PHASE_ERROR_ELEVATION_M).square());
        let code_variance_gradient = code_variance_scale
            .div(Interval::point(sine_squared_lower))
            .mul(Interval::point(geometry.elevation_per_m))
            .upper();
        let variance_gradient = upper_sum(
            ionosphere_variance_gradient,
            upper_sum(troposphere_variance_gradient, code_variance_gradient),
        );
        let center_weight = selection.weights[index];
        if !center_weight.is_finite() || center_weight <= 0.0 || center_weight > 1.0e12 {
            return Err(AtmosphereBoundError::InvalidInput);
        }
        let center_weight_upper = Interval::point(center_weight)
            .add(Interval::point(angles.weight_error))
            .upper();
        let variance_min = Interval::point(1.0)
            .div(Interval::point(center_weight_upper))
            .sub(Interval::point(radius_m).mul(Interval::point(variance_gradient)))
            .lower();
        if !variance_min.is_finite() || variance_min <= 0.0 {
            return Err(AtmosphereBoundError::VarianceNotPositive);
        }
        let variance_min_squared = Interval::point(variance_min).square().lower();
        if variance_min_squared <= 0.0 {
            return Err(AtmosphereBoundError::VarianceNotPositive);
        }
        let weight_gradient = upper_quotient(variance_gradient, variance_min_squared);
        let weight_max = upper_sum(
            center_weight_upper,
            upper_product(radius_m, weight_gradient),
        );
        if ![
            design_derivative,
            correction_gradient,
            weight_max,
            weight_gradient,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            return Err(AtmosphereBoundError::InvalidInput);
        }
        bounds.push(SatelliteAtmosphereBounds {
            design_derivative,
            correction_gradient,
            weight_max,
            weight_gradient,
        });
    }
    Ok(bounds)
}

fn regional_sine_lower(center_lower: f64, elevation_per_m: f64, radius_m: f64) -> f64 {
    Interval::point(center_lower)
        .sub(Interval::point(radius_m).mul(Interval::point(elevation_per_m)))
        .lower()
}

#[cfg(test)]
mod tests {
    use super::{
        geometry_bounds, regional_sine_lower, tropo_gradient_bound, AtmosphereBoundError,
        CenterGeodeticEnclosure, GeometryBounds, WGS84_M_MIN_M,
    };

    #[test]
    fn regional_sine_bound_includes_the_entire_elevation_variation() {
        let center_lower = 0.5;
        let elevation_per_m = 0.125;
        let radius_m = 2.0;
        let lower = regional_sine_lower(center_lower, elevation_per_m, radius_m);
        assert!(lower <= 0.25);
        assert!(lower > 0.0);
        assert!(regional_sine_lower(center_lower, elevation_per_m, 4.0) <= 0.0);
        assert!(regional_sine_lower(center_lower, elevation_per_m, 0.0) <= center_lower);
    }

    #[test]
    fn signed_height_uses_the_reduced_meridional_radius() {
        let geodetic = CenterGeodeticEnclosure {
            latitude_rad: 0.0,
            latitude_error_rad: 0.0,
            longitude_rad: 0.0,
            longitude_error_rad: 0.0,
            height_m: -500.0,
            height_error_m: 0.0,
        };
        let bounds = geometry_bounds(
            [6_377_637.0, 0.0, 0.0],
            [20_200_000.0, 0.0, 0.0],
            1.0,
            &geodetic,
        )
        .unwrap();
        assert!(bounds.latitude_per_m > 1.0 / WGS84_M_MIN_M);

        let below_ionosphere_cutoff = CenterGeodeticEnclosure {
            height_m: -1_001.0,
            ..geodetic
        };
        assert!(matches!(
            geometry_bounds(
                [6_377_136.0, 0.0, 0.0],
                [20_200_000.0, 0.0, 0.0],
                1.0,
                &below_ionosphere_cutoff,
            ),
            Err(AtmosphereBoundError::HeightBranchMayChange)
        ));
        let above_troposphere_domain = CenterGeodeticEnclosure {
            height_m: 10_000.0,
            ..geodetic
        };
        assert!(matches!(
            geometry_bounds(
                [6_388_137.0, 0.0, 0.0],
                [20_200_000.0, 0.0, 0.0],
                1.0,
                &above_troposphere_domain,
            ),
            Err(AtmosphereBoundError::HeightBranchMayChange)
        ));
    }

    #[test]
    fn troposphere_gradient_preserves_zero_delay_branch_and_sea_level_clamp() {
        let geometry = GeometryBounds {
            latitude_per_m: 1.0 / WGS84_M_MIN_M,
            longitude_per_m: 1.0e-7,
            elevation_per_m: 1.0e-6,
            azimuth_per_m: 1.0e-6,
        };
        assert_eq!(
            tropo_gradient_bound(0.5, geometry, -150.0, -101.0).unwrap(),
            0.0
        );
        assert!(tropo_gradient_bound(0.5, geometry, -100.0, 0.0).unwrap() > 0.0);
        assert_eq!(
            tropo_gradient_bound(0.5, geometry, -101.0, -99.0).unwrap_err(),
            AtmosphereBoundError::HeightBranchMayChange
        );
    }
}
