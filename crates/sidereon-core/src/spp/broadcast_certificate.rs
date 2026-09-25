//! Independent interval enclosure for GPS LNAV position and broadcast clock.
//!
//! This test-only model uses the standard non-geostationary GPS equations. It
//! accepts GPS-week records with `0 <= e < 1`, finite LNAV parameters, and
//! transmit epochs within half a GPS week of toe/toc. Reduced trigonometric
//! arguments must fit the interval primitive's `[-8, 8]` domain. Inputs outside
//! those conditions are refused. The eccentric-anomaly root is enclosed by
//! monotonic interval-sign bisection; the iteration cap follows from the
//! 16-radian initial bracket and the binary64 subnormal spacing, not an
//! observed discrepancy.

use crate::astro::time::{ExactEpoch, ExactEpochQuery};
use crate::broadcast::{ClockPolynomial, KeplerianElements};

use super::interval_certificate::Interval;

const GPS_GM_M3_S2: f64 = 3.986_005_0e14;
const GPS_OMEGA_E_RAD_S: f64 = 7.292_115_146_7e-5;
const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;
const SECONDS_PER_GPS_WEEK: i64 = 604_800;
const GPS_EPOCH_J2000_S: i64 = -630_763_200;
const MAX_ROOT_BISECTIONS: usize = 1_080;

#[derive(Clone, Copy, Debug)]
pub(super) struct GpsStateEnclosure {
    pub(super) position_m: [Interval; 3],
    pub(super) clock_s: Interval,
}

pub(super) fn query_from_rtklib_time(
    j2000_whole_s: i64,
    fraction_bits: &str,
) -> Option<ExactEpochQuery> {
    if j2000_whole_s.unsigned_abs() >= (1_u64 << 53) || fraction_bits.len() != 16 {
        return None;
    }
    let fraction = binary_fraction(fraction_bits)?;
    ExactEpoch::J2000
        .query()
        .checked_add_binary_seconds(j2000_whole_s as f64)?
        .checked_add_binary_seconds(fraction)
}

pub(super) fn gps_time_deltas(
    week: u32,
    toe_sow: f64,
    toc_sow: f64,
    j2000_whole_s: i64,
    fraction_bits: &str,
) -> Option<(Interval, Interval)> {
    if !toe_sow.is_finite()
        || !toc_sow.is_finite()
        || !(0.0..SECONDS_PER_GPS_WEEK as f64).contains(&toe_sow)
        || !(0.0..SECONDS_PER_GPS_WEEK as f64).contains(&toc_sow)
        || j2000_whole_s.unsigned_abs() >= (1_u64 << 53)
    {
        return None;
    }
    let fraction = binary_fraction(fraction_bits)?;
    let week_start =
        GPS_EPOCH_J2000_S.checked_add(i64::from(week).checked_mul(SECONDS_PER_GPS_WEEK)?)?;
    let whole_delta = j2000_whole_s.checked_sub(week_start)?;
    if whole_delta.unsigned_abs() >= (1_u64 << 53) {
        return None;
    }
    let query_seconds = Interval::point(whole_delta as f64).add(Interval::point(fraction));
    let toe_delta = query_seconds.sub(Interval::point(toe_sow));
    let toc_delta = query_seconds.sub(Interval::point(toc_sow));
    let half_week = SECONDS_PER_GPS_WEEK as f64 / 2.0;
    if toe_delta.lower() <= -half_week
        || toe_delta.upper() >= half_week
        || toc_delta.lower() <= -half_week
        || toc_delta.upper() >= half_week
    {
        return None;
    }
    Some((toe_delta, toc_delta))
}

pub(super) fn gps_time_deltas_at_query(
    toe_week: u32,
    toe_sow: f64,
    toc_week: u32,
    toc_sow: f64,
    query: &ExactEpochQuery,
) -> Option<(Interval, Interval)> {
    let toe_epoch = gps_record_epoch_query(toe_week, toe_sow)?;
    let toc_epoch = gps_record_epoch_query(toc_week, toc_sow)?;
    let toe_delta_s = query.seconds_since_query(&toe_epoch);
    let toc_delta_s = query.seconds_since_query(&toc_epoch);
    let half_week = SECONDS_PER_GPS_WEEK as f64 / 2.0;
    if !toe_delta_s.is_finite()
        || !toc_delta_s.is_finite()
        || toe_delta_s.abs() >= half_week
        || toc_delta_s.abs() >= half_week
    {
        return None;
    }
    let toe_delta = Interval::point(toe_delta_s).add(Interval::point(0.0));
    let toc_delta = Interval::point(toc_delta_s).add(Interval::point(0.0));
    if toe_delta.lower() <= -half_week
        || toe_delta.upper() >= half_week
        || toc_delta.lower() <= -half_week
        || toc_delta.upper() >= half_week
    {
        return None;
    }
    Some((toe_delta, toc_delta))
}

fn gps_record_epoch_query(week: u32, seconds_of_week: f64) -> Option<ExactEpochQuery> {
    if !seconds_of_week.is_finite()
        || !(0.0..SECONDS_PER_GPS_WEEK as f64).contains(&seconds_of_week)
    {
        return None;
    }
    let week_start_s =
        GPS_EPOCH_J2000_S.checked_add(i64::from(week).checked_mul(SECONDS_PER_GPS_WEEK)?)?;
    if week_start_s.unsigned_abs() >= (1_u64 << 53) {
        return None;
    }
    ExactEpoch::from_j2000_seconds(week_start_s as f64)?
        .checked_add_seconds(seconds_of_week)
        .map(ExactEpoch::query)
}

pub(super) fn enclose_gps_lnav_state(
    elements: &KeplerianElements,
    clock: &ClockPolynomial,
    tk: Interval,
    toc_delta: Interval,
) -> Option<GpsStateEnclosure> {
    if !valid_elements(elements)
        || !valid_clock(clock)
        || tk.lower() <= -SECONDS_PER_GPS_WEEK as f64 / 2.0
        || tk.upper() >= SECONDS_PER_GPS_WEEK as f64 / 2.0
        || toc_delta.lower() <= -SECONDS_PER_GPS_WEEK as f64 / 2.0
        || toc_delta.upper() >= SECONDS_PER_GPS_WEEK as f64 / 2.0
    {
        return None;
    }

    let sqrt_a = Interval::point(elements.sqrt_a);
    let eccentricity = Interval::point(elements.e);
    let semi_major_axis = sqrt_a.mul(sqrt_a);
    let mean_motion = Interval::point(GPS_GM_M3_S2)
        .div(semi_major_axis.mul(semi_major_axis).mul(semi_major_axis))
        .sqrt()
        .add(Interval::point(elements.delta_n));
    let mean_anomaly = Interval::point(elements.m0).add(mean_motion.mul(tk));
    let eccentric_anomaly = kepler_root_enclosure(mean_anomaly, eccentricity)?;
    let (sin_eccentric_anomaly, cos_eccentric_anomaly) = interval_sin_cos(eccentric_anomaly)?;

    let one = Interval::point(1.0);
    let eccentricity_squared = eccentricity.square();
    let eccentric_denominator = one.sub(eccentricity.mul(cos_eccentric_anomaly));
    if eccentric_denominator.lower() <= 0.0 {
        return None;
    }
    let sin_true_anomaly = Interval::new(
        one.sub(eccentricity_squared).lower().max(0.0),
        one.sub(eccentricity_squared).upper(),
    )
    .sqrt()
    .mul(sin_eccentric_anomaly)
    .div(eccentric_denominator);
    let cos_true_anomaly = cos_eccentric_anomaly
        .sub(eccentricity)
        .div(eccentric_denominator);
    let (sin_omega, cos_omega) = interval_sin_cos(Interval::point(elements.omega))?;
    let sin_phi = sin_true_anomaly
        .mul(cos_omega)
        .add(cos_true_anomaly.mul(sin_omega));
    let cos_phi = cos_true_anomaly
        .mul(cos_omega)
        .sub(sin_true_anomaly.mul(sin_omega));
    let sin_two_phi = Interval::point(2.0).mul(sin_phi).mul(cos_phi);
    let cos_two_phi = cos_phi.square().sub(sin_phi.square());

    let delta_u = Interval::point(elements.cus)
        .mul(sin_two_phi)
        .add(Interval::point(elements.cuc).mul(cos_two_phi));
    let delta_r = Interval::point(elements.crs)
        .mul(sin_two_phi)
        .add(Interval::point(elements.crc).mul(cos_two_phi));
    let delta_i = Interval::point(elements.cis)
        .mul(sin_two_phi)
        .add(Interval::point(elements.cic).mul(cos_two_phi));
    let (sin_delta_u, cos_delta_u) = interval_sin_cos(delta_u)?;
    let sin_u = sin_phi.mul(cos_delta_u).add(cos_phi.mul(sin_delta_u));
    let cos_u = cos_phi.mul(cos_delta_u).sub(sin_phi.mul(sin_delta_u));

    let radius = semi_major_axis
        .mul(one.sub(eccentricity.mul(cos_eccentric_anomaly)))
        .add(delta_r);
    let inclination = Interval::point(elements.i0)
        .add(Interval::point(elements.idot).mul(tk))
        .add(delta_i);
    let (sin_inclination, cos_inclination) = interval_sin_cos(inclination)?;
    let (sin_node, cos_node) = interval_sin_cos(
        Interval::point(elements.omega0)
            .add(
                Interval::point(elements.omega_dot)
                    .sub(Interval::point(GPS_OMEGA_E_RAD_S))
                    .mul(tk),
            )
            .sub(Interval::point(GPS_OMEGA_E_RAD_S).mul(Interval::point(elements.toe_sow))),
    )?;

    let orbital_x = radius.mul(cos_u);
    let orbital_y = radius.mul(sin_u);
    let position_x = orbital_x
        .mul(cos_node)
        .sub(orbital_y.mul(cos_inclination).mul(sin_node));
    let position_y = orbital_x
        .mul(sin_node)
        .add(orbital_y.mul(cos_inclination).mul(cos_node));
    let position_z = orbital_y.mul(sin_inclination);

    let clock_polynomial = Interval::point(clock.af0)
        .add(Interval::point(clock.af1).mul(toc_delta))
        .add(Interval::point(clock.af2).mul(toc_delta).mul(toc_delta));
    let relativistic_clock = Interval::point(-2.0)
        .mul(Interval::point(GPS_GM_M3_S2).mul(semi_major_axis).sqrt())
        .mul(eccentricity)
        .mul(sin_eccentric_anomaly)
        .div(Interval::point(SPEED_OF_LIGHT_M_S * SPEED_OF_LIGHT_M_S));

    Some(GpsStateEnclosure {
        position_m: [position_x, position_y, position_z],
        clock_s: clock_polynomial.add(relativistic_clock),
    })
}

fn valid_elements(elements: &KeplerianElements) -> bool {
    let values = [
        elements.sqrt_a,
        elements.e,
        elements.m0,
        elements.delta_n,
        elements.omega0,
        elements.i0,
        elements.omega,
        elements.omega_dot,
        elements.idot,
        elements.cuc,
        elements.cus,
        elements.crc,
        elements.crs,
        elements.cic,
        elements.cis,
        elements.toe_sow,
    ];
    values.iter().all(|value| value.is_finite())
        && elements.sqrt_a > 0.0
        && (0.0..1.0).contains(&elements.e)
        && (0.0..SECONDS_PER_GPS_WEEK as f64).contains(&elements.toe_sow)
}

fn valid_clock(clock: &ClockPolynomial) -> bool {
    [clock.af0, clock.af1, clock.af2, clock.toc_sow]
        .iter()
        .all(|value| value.is_finite())
        && (0.0..SECONDS_PER_GPS_WEEK as f64).contains(&clock.toc_sow)
}

fn binary_fraction(fraction_bits: &str) -> Option<f64> {
    if fraction_bits.len() != 16 {
        return None;
    }
    let fraction = f64::from_bits(u64::from_str_radix(fraction_bits, 16).ok()?);
    (fraction.is_finite() && (0.0..1.0).contains(&fraction)).then_some(fraction)
}

fn kepler_root_enclosure(mean_anomaly: Interval, eccentricity: Interval) -> Option<Interval> {
    if eccentricity.lower() < 0.0 || eccentricity.upper() >= 1.0 {
        return None;
    }
    let mut lower = Interval::point(mean_anomaly.lower())
        .sub(Interval::point(eccentricity.upper()))
        .lower();
    let mut upper = Interval::point(mean_anomaly.upper())
        .add(Interval::point(eccentricity.upper()))
        .upper();
    if lower < -8.0 || upper > 8.0 || !lower.is_finite() || !upper.is_finite() {
        return None;
    }
    for _ in 0..MAX_ROOT_BISECTIONS {
        let midpoint = lower * 0.5 + upper * 0.5;
        if midpoint <= lower || midpoint >= upper {
            return Some(Interval::new(lower, upper));
        }
        let residual = Interval::point(midpoint)
            .sub(eccentricity.mul(Interval::point(midpoint).sin()))
            .sub(mean_anomaly);
        if residual.upper() < 0.0 {
            lower = midpoint;
        } else if residual.lower() > 0.0 {
            upper = midpoint;
        } else {
            let residual_magnitude = residual.lower().abs().max(residual.upper().abs());
            let derivative_lower = Interval::point(1.0)
                .sub(Interval::point(eccentricity.upper()))
                .lower();
            let root_radius = Interval::point(residual_magnitude)
                .div(Interval::point(derivative_lower))
                .upper();
            let contracted =
                Interval::point(midpoint).sub(Interval::new(-root_radius, root_radius));
            let contracted_lower = lower.max(contracted.lower());
            let contracted_upper = upper.min(contracted.upper());
            return Some(Interval::new(contracted_lower, contracted_upper));
        }
    }
    None
}

fn interval_sin_cos(angle: Interval) -> Option<(Interval, Interval)> {
    let pi = pi_enclosure();
    let tau = pi.mul(Interval::point(2.0));
    let midpoint = angle.lower() * 0.5 + angle.upper() * 0.5;
    let turns = (midpoint / core::f64::consts::TAU).round();
    if !turns.is_finite() || turns.abs() >= (1_u64 << 52) as f64 {
        return None;
    }
    let reduced = angle.sub(tau.mul(Interval::point(turns)));
    if reduced.lower() < -8.0 || reduced.upper() > 8.0 {
        return None;
    }
    Some((reduced.sin(), reduced.cos()))
}

fn pi_enclosure() -> Interval {
    Interval::point(16.0)
        .mul(atan_small(Interval::point(1.0).div(Interval::point(5.0))))
        .sub(Interval::point(4.0).mul(atan_small(Interval::point(1.0).div(Interval::point(239.0)))))
}

fn atan_small(argument: Interval) -> Interval {
    let square = argument.square();
    let mut power = argument;
    let mut sum = Interval::point(0.0);
    for term_index in 0..32 {
        let denominator = (2 * term_index + 1) as f64;
        let term = power.div(Interval::point(denominator));
        sum = if term_index % 2 == 0 {
            sum.add(term)
        } else {
            sum.sub(term)
        };
        power = power.mul(square);
    }
    let remainder = power.div(Interval::point(65.0));
    sum.add(Interval::new(-remainder.upper(), remainder.upper()))
}

#[cfg(test)]
mod tests {
    use super::{
        enclose_gps_lnav_state, gps_time_deltas, gps_time_deltas_at_query, kepler_root_enclosure,
        query_from_rtklib_time, GpsStateEnclosure,
    };
    use crate::astro::time::ExactEpoch;
    use crate::broadcast::{ClockPolynomial, KeplerianElements};

    use super::super::interval_certificate::Interval;

    fn circular_elements() -> KeplerianElements {
        KeplerianElements {
            sqrt_a: 2.0,
            e: 0.0,
            m0: 0.0,
            delta_n: 0.0,
            omega0: 0.0,
            i0: 0.0,
            omega: 0.0,
            omega_dot: 0.0,
            idot: 0.0,
            cuc: 0.0,
            cus: 0.0,
            crc: 0.0,
            crs: 0.0,
            cic: 0.0,
            cis: 0.0,
            toe_sow: 0.0,
        }
    }

    fn zero_clock() -> ClockPolynomial {
        ClockPolynomial {
            af0: 0.0,
            af1: 0.0,
            af2: 0.0,
            toc_sow: 0.0,
        }
    }

    fn contains_analytic_position(enclosure: &GpsStateEnclosure, expected: [f64; 3]) {
        for axis in 0..3 {
            assert!(enclosure.position_m[axis].contains(expected[axis]));
        }
    }

    #[test]
    fn exact_oracle_time_pair_preserves_binary_fraction() {
        let epoch_query = query_from_rtklib_time(123_456, "3fc8000000000000")
            .expect("finite exact RTKLIB time pair");
        assert_eq!(epoch_query.seconds_since(ExactEpoch::J2000), 123_456.1875);
        assert!(query_from_rtklib_time(0, "3ff0000000000000").is_none());
        assert!(query_from_rtklib_time(0, "not-a-float-bits").is_none());
    }

    #[test]
    fn kepler_interval_contains_a_nonzero_analytic_root() {
        let eccentricity = 0.125;
        let expected_root = 0.7;
        let expected_sine = Interval::point(expected_root).sin();
        let mean_anomaly =
            Interval::point(expected_root).sub(Interval::point(eccentricity).mul(expected_sine));
        let root = kepler_root_enclosure(mean_anomaly, Interval::point(eccentricity))
            .expect("bounded unique root");

        assert!(root.contains(expected_root));
        assert!(kepler_root_enclosure(mean_anomaly, Interval::point(1.0)).is_none());
    }

    #[test]
    fn circular_orbit_and_polynomial_clock_match_closed_form() {
        let elements = circular_elements();
        let clock = ClockPolynomial {
            af0: 0.125,
            af1: 0.0625,
            af2: 0.03125,
            toc_sow: 0.0,
        };
        let enclosure = enclose_gps_lnav_state(
            &elements,
            &clock,
            Interval::point(0.0),
            Interval::point(2.0),
        )
        .expect("finite GPS LNAV state");

        contains_analytic_position(&enclosure, [4.0, 0.0, 0.0]);
        assert!(enclosure.clock_s.contains(0.375));
    }

    #[test]
    fn eccentric_orbit_at_perigee_has_analytic_radius_and_zero_relativity() {
        let mut elements = circular_elements();
        elements.e = 0.25;
        let enclosure = enclose_gps_lnav_state(
            &elements,
            &zero_clock(),
            Interval::point(0.0),
            Interval::point(0.0),
        )
        .expect("finite eccentric GPS LNAV state");

        contains_analytic_position(&enclosure, [3.0, 0.0, 0.0]);
        assert!(enclosure.clock_s.contains(0.0));
    }

    #[test]
    fn clock_delta_at_either_half_week_boundary_is_refused() {
        let elements = circular_elements();
        let clock = zero_clock();
        let half_week = super::SECONDS_PER_GPS_WEEK as f64 / 2.0;

        assert!(enclose_gps_lnav_state(
            &elements,
            &clock,
            Interval::point(0.0),
            Interval::point(-half_week),
        )
        .is_none());
        assert!(enclose_gps_lnav_state(
            &elements,
            &clock,
            Interval::point(0.0),
            Interval::point(half_week),
        )
        .is_none());
    }

    #[test]
    fn nonzero_eccentric_anomaly_has_positive_cross_track_and_negative_relativity() {
        let mut elements = circular_elements();
        elements.e = 0.125;
        elements.m0 = 0.5;
        let enclosure = enclose_gps_lnav_state(
            &elements,
            &zero_clock(),
            Interval::point(0.0),
            Interval::point(0.0),
        )
        .expect("finite eccentric GPS LNAV state");

        assert!(enclosure.position_m[1].lower() > 0.0);
        assert!(enclosure.position_m[2].contains(0.0));
        assert!(enclosure.clock_s.upper() < 0.0);
    }

    #[test]
    fn gps_time_delta_matches_week_and_tow_at_a_fractional_instant() {
        let week = 2_111;
        let toe_sow = 120_000.0;
        let toc_sow = 120_010.0;
        let j2000_whole_s = -630_763_200 + i64::from(week) * 604_800 + 120_100;
        let (toe_delta, toc_delta) =
            gps_time_deltas(week, toe_sow, toc_sow, j2000_whole_s, "3fc0000000000000")
                .expect("within-week GPS time pair");

        assert!(toe_delta.contains(100.125));
        assert!(toc_delta.contains(90.125));
        assert!(
            gps_time_deltas(week, toe_sow, toc_sow, j2000_whole_s, "3ff0000000000000").is_none()
        );
    }

    #[test]
    fn exact_gps_time_deltas_preserve_sub_f64_epoch_offsets_and_adjacent_weeks() {
        let toe_week = 2_111;
        let toe_sow = 604_799.1;
        let toc_week = 2_112;
        let toc_sow = 2.1;
        let toe_week_start_s = -630_763_200 + i64::from(toe_week) * 604_800;
        let toe_epoch = ExactEpoch::from_j2000_seconds(toe_week_start_s as f64)
            .expect("representable GPS week start")
            .checked_add_seconds(toe_sow)
            .expect("valid decimal GPS toe")
            .query();
        let query = toe_epoch
            .clone()
            .checked_add_binary_seconds(1.0e-10)
            .expect("finite sub-f64 offset");
        let toc_week_start_s = -630_763_200 + i64::from(toc_week) * 604_800;
        let toc_epoch = ExactEpoch::from_j2000_seconds(toc_week_start_s as f64)
            .expect("representable adjacent GPS week start")
            .checked_add_seconds(toc_sow)
            .expect("valid decimal GPS toc")
            .query();
        let (toe_delta, toc_delta) =
            gps_time_deltas_at_query(toe_week, toe_sow, toc_week, toc_sow, &query)
                .expect("both reference deltas are within half a week");

        assert_eq!(
            toe_epoch.j2000_seconds().to_bits(),
            query.j2000_seconds().to_bits()
        );
        assert!(toe_delta.contains(1.0e-10));
        assert!(toc_delta.contains(query.seconds_since_query(&toc_epoch)));
        assert!(toe_delta.lower() > 0.0);
        assert!(toc_delta.upper() < 0.0);
    }
}
