//! Unit tests for the analytic propagators that need no external oracle:
//! input validation, the domain limits, the element conversions, the
//! critical-inclination factor against its closed form, and the
//! mean/osculating round trips. The Orekit comparison is in
//! `tests/analytic_propagators_orekit.rs`.

use super::brouwer_lyddane::t2_series_for_tests;
use super::elements::{
    circular_mean_to_eccentric, elliptic_mean_to_eccentric, CircularElements, KeplerianElements,
};
use super::*;
use crate::astro::state::CartesianState;

fn sso_state() -> CartesianState {
    let mean = KeplerianElements {
        a_km: 7078.137,
        e: 0.0012,
        i_rad: 98.19f64.to_radians(),
        arg_perigee_rad: 90f64.to_radians(),
        raan_rad: 30f64.to_radians(),
        mean_anomaly_rad: 10f64.to_radians(),
    };
    let (p, v) = mean.to_cartesian(MU_EARTH).unwrap();
    CartesianState::new(100.0, p, v)
}

fn gps_state() -> CartesianState {
    let mean = KeplerianElements {
        a_km: 26_559.7,
        e: 0.01,
        i_rad: 55f64.to_radians(),
        arg_perigee_rad: 30f64.to_radians(),
        raan_rad: 250f64.to_radians(),
        mean_anomaly_rad: 100f64.to_radians(),
    };
    let (p, v) = mean.to_cartesian(MU_EARTH).unwrap();
    CartesianState::new(0.0, p, v)
}

fn angle_diff(a: f64, b: f64) -> f64 {
    normalize_angle(a - b, 0.0).abs()
}

#[test]
fn rejects_non_finite_and_invalid_inputs() {
    let g = ZonalHarmonics::earth();
    let opts = MeanConversionOptions::ECKSTEIN_HECHLER_DEFAULT;
    let mut bad = sso_state();
    bad.position_km.x = f64::NAN;
    assert!(matches!(
        EcksteinHechlerPropagator::new(&bad, &g, opts),
        Err(AnalyticPropagationError::NonFinite {
            field: "position_km"
        })
    ));
    let mut bad = sso_state();
    bad.epoch_tdb_seconds = f64::INFINITY;
    assert!(matches!(
        BrouwerLyddanePropagator::new(&bad, &g, 0.0, opts),
        Err(AnalyticPropagationError::NonFinite {
            field: "epoch_tdb_seconds"
        })
    ));
    assert!(matches!(
        BrouwerLyddanePropagator::new(&gps_state(), &g, f64::NAN, opts),
        Err(AnalyticPropagationError::NonFinite { field: "m2_rad_s2" })
    ));
    let zero_mu = ZonalHarmonics {
        mu_km3_s2: 0.0,
        ..g
    };
    assert!(matches!(
        EcksteinHechlerPropagator::new(&sso_state(), &zero_mu, opts),
        Err(AnalyticPropagationError::InvalidParameter {
            field: "mu_km3_s2",
            ..
        })
    ));
    let no_j2 = ZonalHarmonics { c20: 0.0, ..g };
    assert!(matches!(
        BrouwerLyddanePropagator::new(&gps_state(), &no_j2, 0.0, opts),
        Err(AnalyticPropagationError::InvalidParameter { field: "c20", .. })
    ));
    let nan_c40 = ZonalHarmonics { c40: f64::NAN, ..g };
    assert!(matches!(
        brouwer_lyddane_mean_elements(&gps_state(), &nan_c40, opts),
        Err(AnalyticPropagationError::NonFinite { field: "c40" })
    ));
    for (tolerance, max_iterations, field) in [
        (0.0, 10, "tolerance"),
        (-1.0e-12, 10, "tolerance"),
        (1.0e-12, 0, "max_iterations"),
    ] {
        let result = eckstein_hechler_mean_elements(
            &sso_state(),
            &g,
            MeanConversionOptions::new(tolerance, max_iterations),
        );
        assert!(
            matches!(result, Err(AnalyticPropagationError::InvalidParameter { field: f, .. }) if f == field),
            "{result:?}"
        );
    }
    assert!(matches!(
        eckstein_hechler_mean_elements(&sso_state(), &g, MeanConversionOptions::new(f64::NAN, 10)),
        Err(AnalyticPropagationError::NonFinite { field: "tolerance" })
    ));
    let propagator = EcksteinHechlerPropagator::new(&sso_state(), &g, opts).unwrap();
    assert!(matches!(
        propagator.propagate(f64::NAN),
        Err(AnalyticPropagationError::NonFinite {
            field: "epoch_tdb_seconds"
        })
    ));
}

#[test]
fn rejects_open_and_degenerate_states() {
    let g = ZonalHarmonics::earth();
    let escape = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [0.0, 11.0, 1.0]);
    assert!(matches!(
        eckstein_hechler_mean_elements(
            &escape,
            &g,
            MeanConversionOptions::ECKSTEIN_HECHLER_DEFAULT
        ),
        Err(AnalyticPropagationError::NotElliptic { .. })
    ));
    let radial = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    assert!(matches!(
        brouwer_lyddane_mean_elements(&radial, &g, MeanConversionOptions::BROUWER_LYDDANE_DEFAULT),
        Err(AnalyticPropagationError::NotElliptic { .. })
    ));
    // Angular momentum of 1e-212 km^2/s: the eccentricity rounds to 1.
    let nearly_rectilinear = CartesianState::new(
        0.0,
        [
            -3.007469634836511e101,
            5.392008948635528e67,
            1.72723357266931e-77,
        ],
        [2.1216488546e-314, 0.0, 0.0],
    );
    assert!(matches!(
        CircularElements::from_cartesian(
            nearly_rectilinear.position_array(),
            nearly_rectilinear.velocity_array(),
            g.mu_km3_s2
        ),
        Err(AnalyticPropagationError::NotElliptic { field: "state" })
    ));
    // At apogee the radius is 1.5 a = 2.55e308 km, beyond f64::MAX.
    let huge = KeplerianElements {
        a_km: 1.7e308,
        e: 0.5,
        i_rad: 0.5,
        arg_perigee_rad: 0.0,
        raan_rad: 0.0,
        mean_anomaly_rad: core::f64::consts::PI,
    };
    assert!(matches!(
        huge.to_cartesian(g.mu_km3_s2),
        Err(AnalyticPropagationError::InvalidParameter {
            field: "elements",
            ..
        })
    ));
    let origin = CartesianState::new(0.0, [0.0; 3], [0.0, 7.5, 0.0]);
    assert!(matches!(
        brouwer_lyddane_mean_elements(&origin, &g, MeanConversionOptions::BROUWER_LYDDANE_DEFAULT),
        Err(AnalyticPropagationError::InvalidParameter {
            field: "position_km",
            ..
        })
    ));
    // Equatorial and retrograde: the equinoctial elements are undefined.
    let retro = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [0.0, -7.546, 0.0]);
    assert!(matches!(
        brouwer_lyddane_mean_elements(&retro, &g, MeanConversionOptions::BROUWER_LYDDANE_DEFAULT),
        Err(AnalyticPropagationError::RetrogradeEquatorial { .. })
    ));
    let bad_mean = KeplerianElements {
        a_km: 7000.0,
        e: 1.0,
        i_rad: 0.5,
        arg_perigee_rad: 0.0,
        raan_rad: 0.0,
        mean_anomaly_rad: 0.0,
    };
    assert!(matches!(
        BrouwerLyddanePropagator::from_mean_elements(0.0, bad_mean, &g, 0.0),
        Err(AnalyticPropagationError::NotElliptic { .. })
    ));
    let negative_a = KeplerianElements {
        a_km: -7000.0,
        e: 0.1,
        ..bad_mean
    };
    assert!(matches!(
        brouwer_lyddane_osculating_elements(&negative_a, &g),
        Err(AnalyticPropagationError::NotElliptic { .. })
    ));
}

#[test]
fn eckstein_hechler_domain_limits() {
    let g = ZonalHarmonics::earth();
    let base = CircularElements {
        a_km: 7200.0,
        ex: 0.0,
        ey: 0.001,
        i_rad: 50f64.to_radians(),
        raan_rad: 0.3,
        mean_latitude_argument_rad: 1.0,
    };
    let ok = |m: CircularElements| EcksteinHechlerPropagator::from_mean_elements(0.0, m, &g);
    assert!(ok(base).is_ok());
    // e = 0.1 is accepted, anything above it is not.
    assert!(ok(CircularElements {
        ex: 0.1,
        ey: 0.0,
        ..base
    })
    .is_ok());
    assert!(matches!(
        ok(CircularElements { ex: 0.0, ey: 0.1000001, ..base }),
        Err(AnalyticPropagationError::EccentricityTooLarge { limit, .. }) if limit == 0.1
    ));
    for i_deg in [0.0, 180.0, 1.0e-4] {
        assert!(
            matches!(
                ok(CircularElements {
                    i_rad: f64::to_radians(i_deg),
                    ..base
                }),
                Err(AnalyticPropagationError::NearlyEquatorial { .. })
            ),
            "i = {i_deg}"
        );
    }
    // sin^2 i = 4/5 at 63.4349 and 116.5651 degrees; the band is |sin^2 i - 0.8| < 1e-3.
    let critical = libm::asin(libm::sqrt(0.8));
    for i in [
        critical,
        core::f64::consts::PI - critical,
        critical + 1.0e-3,
    ] {
        assert!(
            matches!(
                ok(CircularElements { i_rad: i, ..base }),
                Err(AnalyticPropagationError::NearCriticalInclination { .. })
            ),
            "i = {i}"
        );
    }
    // d(sin^2 i)/di = sin(2i) = 0.8 there, so 2e-3 rad moves sin^2 i by 1.6e-3.
    assert!(ok(CircularElements {
        i_rad: critical + 2.0e-3,
        ..base
    })
    .is_ok());
    assert!(ok(CircularElements {
        i_rad: critical - 2.0e-3,
        ..base
    })
    .is_ok());
}

#[test]
fn inside_reference_sphere_is_refused() {
    let g = ZonalHarmonics::earth();
    let low = KeplerianElements {
        a_km: 6000.0,
        e: 0.001,
        i_rad: 0.8,
        arg_perigee_rad: 0.0,
        raan_rad: 0.0,
        mean_anomaly_rad: 0.0,
    };
    let (p, v) = low.to_cartesian(g.mu_km3_s2).unwrap();
    let state = CartesianState::new(0.0, p, v);
    for result in [
        eckstein_hechler_mean_elements(&state, &g, MeanConversionOptions::ECKSTEIN_HECHLER_DEFAULT)
            .map(|_| ()),
        brouwer_lyddane_mean_elements(&state, &g, MeanConversionOptions::BROUWER_LYDDANE_DEFAULT)
            .map(|_| ()),
    ] {
        assert!(matches!(
            result,
            Err(AnalyticPropagationError::InsideReferenceSphere { reference_radius_km, .. })
                if reference_radius_km == g.re_km
        ));
    }
}

#[test]
fn non_convergence_is_a_typed_error() {
    let g = ZonalHarmonics::earth();
    let result =
        eckstein_hechler_mean_elements(&gps_state(), &g, MeanConversionOptions::new(1.0e-11, 1));
    assert_eq!(
        result,
        Err(AnalyticPropagationError::MeanElementsNotConverged {
            theory: AnalyticTheory::EcksteinHechler,
            iterations: 1,
        })
    );
    let result =
        brouwer_lyddane_mean_elements(&gps_state(), &g, MeanConversionOptions::new(1.0e-13, 2));
    assert_eq!(
        result,
        Err(AnalyticPropagationError::MeanElementsNotConverged {
            theory: AnalyticTheory::BrouwerLyddane,
            iterations: 2,
        })
    );
    // A tolerance below the rounding of the residuals cannot be met.
    let result =
        brouwer_lyddane_mean_elements(&gps_state(), &g, MeanConversionOptions::new(1.0e-20, 50));
    assert!(matches!(
        result,
        Err(AnalyticPropagationError::MeanElementsNotConverged { iterations: 50, .. })
    ));
    let error: PropagationError = result.unwrap_err().into();
    assert!(matches!(error, PropagationError::NumericalFailure(_)));
}

#[test]
fn critical_inclination_factor_matches_its_closed_form() {
    // Phipps (1992) Eq. 2.47: T2 = (1 - exp(-100 x^2)) / x with
    // x = 1 - 5 cos^2 i, evaluated here with expm1, which keeps full precision
    // as x goes to zero. The series-and-product form agrees to a few units in
    // the 13th digit, the truncation of its 13-term series at small |x|.
    for i_deg in [
        0.0,
        10.0,
        30.0,
        45.0,
        55.0,
        60.0,
        62.0,
        63.0,
        63.4349,
        63.43494882,
        64.0,
        65.0,
        70.0,
        90.0,
        98.0,
        116.5651,
        120.0,
        180.0,
    ] {
        let c = libm::cos(f64::to_radians(i_deg));
        let x = 1.0 - 5.0 * c * c;
        let closed = if x == 0.0 {
            0.0
        } else {
            -libm::expm1(-100.0 * x * x) / x
        };
        let t2 = t2_series_for_tests(c);
        assert!(
            (t2 - closed).abs() <= 1.0e-12 * closed.abs().max(1.0e-3),
            "i = {i_deg}: {t2} vs {closed}"
        );
        // Away from the critical band it is 1 / x to double precision.
        if x.abs() > 0.6 {
            assert!((t2 * x - 1.0).abs() < 1.0e-12, "i = {i_deg}");
        }
    }
}

#[test]
fn element_conversions_round_trip_and_agree() {
    let mu = MU_EARTH;
    for (a, e, i, w, raan, m) in [
        (7078.137, 0.0012, 98.19, 90.0, 30.0, 10.0),
        (26_600.0, 0.72, 63.4, 270.0, 40.0, 10.0),
        (42_164.0, 0.0, 0.0, 0.0, 0.0, 77.0),
        (7000.0, 0.05, 120.0, 300.0, 200.0, 359.0),
    ] {
        let kep = KeplerianElements {
            a_km: a,
            e,
            i_rad: f64::to_radians(i),
            arg_perigee_rad: f64::to_radians(w),
            raan_rad: f64::to_radians(raan),
            mean_anomaly_rad: f64::to_radians(m),
        };
        let (p, v) = kep.to_cartesian(mu).unwrap();
        // Independent two-body check: vis-viva and angular momentum.
        let r = libm::sqrt(p.iter().map(|x| x * x).sum());
        let v2: f64 = v.iter().map(|x| x * x).sum();
        assert!((v2 - mu * (2.0 / r - 1.0 / a)).abs() < 1.0e-12 * v2);
        let h = [
            p[1] * v[2] - p[2] * v[1],
            p[2] * v[0] - p[0] * v[2],
            p[0] * v[1] - p[1] * v[0],
        ];
        let h_norm = libm::sqrt(h.iter().map(|x| x * x).sum());
        assert!((h_norm - libm::sqrt(mu * a * (1.0 - e * e))).abs() < 1.0e-12 * h_norm);

        let circ = CircularElements::from_cartesian(p, v, mu).unwrap();
        let (p2, v2v) = circ.to_cartesian(mu).unwrap();
        let back = KeplerianElements::from_cartesian(p, v, mu).unwrap();
        let (p3, v3) = back.to_cartesian(mu).unwrap();
        for k in 0..3 {
            // Round trips through the angles lose a few units in the last
            // place of the radius (1e-12 km is 1 nm).
            assert!((p[k] - p2[k]).abs() < 1.0e-8, "{p:?} {p2:?}");
            assert!((p[k] - p3[k]).abs() < 1.0e-8, "{p:?} {p3:?}");
            assert!((v[k] - v2v[k]).abs() < 1.0e-11, "{v:?} {v2v:?}");
            assert!((v[k] - v3[k]).abs() < 1.0e-11, "{v:?} {v3:?}");
        }
        assert!((back.a_km - a).abs() < 1.0e-8 * a);
        assert!((back.e - e).abs() < 1.0e-12);
    }
}

#[test]
fn kepler_solvers_satisfy_their_equations() {
    for e in [0.0, 1.0e-9, 0.01, 0.3, 0.7, 0.95, 0.999_999] {
        for k in -40..=40 {
            let m = 0.1 * k as f64 + 0.013;
            let big_e = elliptic_mean_to_eccentric(m, e).unwrap();
            assert!((big_e - e * libm::sin(big_e) - m).abs() < 4.0e-15 * (1.0 + m.abs()));
        }
    }
    for (ex, ey) in [(0.0, 0.0), (0.01, -0.02), (-0.07, 0.07), (0.0, 0.0999)] {
        for k in -40..=40 {
            let alpha_m = 0.1 * k as f64 + 0.007;
            let alpha_e = circular_mean_to_eccentric(ex, ey, alpha_m).unwrap();
            let back = alpha_e - ex * libm::sin(alpha_e) + ey * libm::cos(alpha_e);
            assert!((back - alpha_m).abs() < 4.0e-15 * (1.0 + alpha_m.abs()));
        }
    }
}

#[test]
fn eckstein_hechler_velocity_is_the_position_derivative() {
    let g = ZonalHarmonics::earth();
    let p = EcksteinHechlerPropagator::new(
        &gps_state(),
        &g,
        MeanConversionOptions::ECKSTEIN_HECHLER_DEFAULT,
    )
    .unwrap();
    for t in [0.0, 5000.0, 86_400.0] {
        let h = 0.5;
        let a = p.propagate(t - h).unwrap();
        let b = p.propagate(t + h).unwrap();
        let s = p.propagate(t).unwrap();
        for k in 0..3 {
            let fd = (b.position_km[k] - a.position_km[k]) / (2.0 * h);
            // Central difference over 1 s of a 12-hour orbit: truncation
            // error (h^2 / 6) |d3x/dt3| ~ 1e-9 km/s, rounding 1e-14 km / 1 s.
            assert!((fd - s.velocity_km_s[k]).abs() < 1.0e-8, "t = {t}");
        }
    }
}

#[test]
fn mean_osculating_round_trips() {
    let g = ZonalHarmonics::earth();
    // Eckstein-Hechler: mean -> osculating state -> mean.
    let mean = CircularElements {
        a_km: 7078.137,
        ex: 1.0e-4,
        ey: 1.2e-3,
        i_rad: 98.19f64.to_radians(),
        raan_rad: 0.5,
        mean_latitude_argument_rad: 2.0,
    };
    let osc = eckstein_hechler_osculating_elements(&mean, &g).unwrap();
    let (p, v) = osc.to_cartesian(g.mu_km3_s2).unwrap();
    let back = eckstein_hechler_mean_elements(
        &CartesianState::new(0.0, p, v),
        &g,
        MeanConversionOptions::new(1.0e-13, 100),
    )
    .unwrap();
    let m = back.elements;
    // The fixed point stops at residuals of 1e-13 relative; the inverse map is
    // well conditioned, so the mean elements come back to about that.
    assert!((m.a_km - mean.a_km).abs() < 1.0e-9);
    assert!((m.ex - mean.ex).abs() < 1.0e-12);
    assert!((m.ey - mean.ey).abs() < 1.0e-12);
    assert!((m.i_rad - mean.i_rad).abs() < 1.0e-12);
    assert!(angle_diff(m.raan_rad, mean.raan_rad) < 1.0e-12);
    assert!(
        angle_diff(
            m.mean_latitude_argument_rad,
            mean.mean_latitude_argument_rad
        ) < 1.0e-12
    );

    // Brouwer-Lyddane, eccentric and critically inclined.
    let mean = KeplerianElements {
        a_km: 26_600.0,
        e: 0.72,
        i_rad: 63.43f64.to_radians(),
        arg_perigee_rad: 270f64.to_radians(),
        raan_rad: 40f64.to_radians(),
        mean_anomaly_rad: 10f64.to_radians(),
    };
    let osc = brouwer_lyddane_osculating_elements(&mean, &g).unwrap();
    let (p, v) = osc.to_cartesian(g.mu_km3_s2).unwrap();
    let back = brouwer_lyddane_mean_elements(
        &CartesianState::new(0.0, p, v),
        &g,
        MeanConversionOptions::BROUWER_LYDDANE_DEFAULT,
    )
    .unwrap();
    let m = back.elements;
    assert!((m.a_km - mean.a_km).abs() < 1.0e-8);
    assert!((m.e - mean.e).abs() < 1.0e-12);
    assert!((m.i_rad - mean.i_rad).abs() < 1.0e-12);
    assert!(angle_diff(m.raan_rad, mean.raan_rad) < 1.0e-12);
    assert!(angle_diff(m.arg_perigee_rad, mean.arg_perigee_rad) < 1.0e-11);
    assert!(angle_diff(m.mean_anomaly_rad, mean.mean_anomaly_rad) < 1.0e-11);

    // The propagator started from the osculating state reproduces it at its
    // own epoch to the conversion tolerance.
    let start = CartesianState::new(42.0, p, v);
    let prop = BrouwerLyddanePropagator::new(
        &start,
        &g,
        0.0,
        MeanConversionOptions::BROUWER_LYDDANE_DEFAULT,
    )
    .unwrap();
    let again = prop.propagate(42.0).unwrap();
    assert!((again.position_km - start.position_km).norm() < 1.0e-7);
    assert!((again.velocity_km_s - start.velocity_km_s).norm() < 1.0e-10);
}

#[test]
fn drag_term_adds_m2_dt_squared_to_the_mean_anomaly() {
    // With M2 the mean anomaly gains M2 dt^2 and the mean semi-major axis
    // drops by (4/3) a M2 dt / (n0 (1 + dsl)) (Phipps 1992, Eq. 2.38 and
    // 2.41): its mean motion rises by 2 M2 dt, the derivative of M2 dt^2.
    let g = ZonalHarmonics::earth();
    let mean = KeplerianElements {
        a_km: 6778.137,
        e: 0.01,
        i_rad: 51.64f64.to_radians(),
        arg_perigee_rad: 1.0,
        raan_rad: 2.0,
        mean_anomaly_rad: 3.0,
    };
    let m2 = 1.0e-14;
    let with = BrouwerLyddanePropagator::from_mean_elements(0.0, mean, &g, m2).unwrap();
    let without = BrouwerLyddanePropagator::from_mean_elements(0.0, mean, &g, 0.0).unwrap();
    assert_eq!(
        with.propagate(0.0).unwrap(),
        without.propagate(0.0).unwrap()
    );
    let dt = 5.0 * 86_400.0;
    let a = with.osculating_elements(dt).unwrap();
    let b = without.osculating_elements(dt).unwrap();
    let lon = |k: &KeplerianElements| k.arg_perigee_rad + k.raan_rad + k.mean_anomaly_rad;
    let gain = normalize_angle(lon(&a) - lon(&b), 0.0);
    // M2 dt^2 = 1.866e-3 rad; the periodic terms move by the change of a and
    // e, a part in 1e3 of the gain here.
    assert!(
        (gain - m2 * dt * dt).abs() < 2.0e-3 * m2 * dt * dt,
        "{gain}"
    );
    assert!(a.a_km < b.a_km);
}

#[test]
fn eckstein_hechler_refuses_mean_elements_inside_the_reference_sphere() {
    let g = ZonalHarmonics::earth();
    let mean = CircularElements {
        a_km: 10.0,
        ex: 0.0,
        ey: 0.0,
        i_rad: 0.5,
        raan_rad: 0.0,
        mean_latitude_argument_rad: 0.0,
    };
    let inside = |r: Result<(), AnalyticPropagationError>| {
        matches!(
            r,
            Err(AnalyticPropagationError::InsideReferenceSphere { semi_major_axis_km, reference_radius_km })
                if semi_major_axis_km == 10.0 && reference_radius_km == g.re_km
        )
    };
    assert!(inside(
        eckstein_hechler_osculating_elements(&mean, &g).map(|_| ())
    ));
    assert!(inside(
        EcksteinHechlerPropagator::from_mean_elements(0.0, mean, &g).map(|_| ())
    ));
    let kep = KeplerianElements {
        a_km: 10.0,
        e: 0.0,
        i_rad: 0.5,
        arg_perigee_rad: 0.0,
        raan_rad: 0.0,
        mean_anomaly_rad: 0.0,
    };
    assert!(inside(
        brouwer_lyddane_osculating_elements(&kep, &g).map(|_| ())
    ));
    assert!(inside(
        BrouwerLyddanePropagator::from_mean_elements(0.0, kep, &g, 0.0).map(|_| ())
    ));
}

#[test]
fn eckstein_hechler_osculating_elements_are_elliptic_or_refused() {
    // A field with J2 = 5 at a = re: the periodic terms of the eccentricity
    // reach several times their mean value, so the osculating orbit is not
    // elliptic for some mean arguments of latitude.
    let g = ZonalHarmonics {
        c20: -5.0,
        c30: 0.0,
        c40: 0.0,
        c50: 0.0,
        c60: 0.0,
        ..ZonalHarmonics::earth()
    };
    let mut refused = 0;
    for k in 0..36 {
        let mean = CircularElements {
            a_km: g.re_km,
            ex: 0.001,
            ey: 0.001,
            i_rad: 60f64.to_radians(),
            raan_rad: 0.2,
            mean_latitude_argument_rad: 0.1 + 0.17 * k as f64,
        };
        let osc = eckstein_hechler_osculating_elements(&mean, &g);
        let state = EcksteinHechlerPropagator::from_mean_elements(0.0, mean, &g)
            .unwrap()
            .propagate(0.0);
        match osc {
            Ok(o) => {
                assert!(
                    o.a_km > 0.0 && o.ex * o.ex + o.ey * o.ey < 1.0,
                    "{k}: {o:?}"
                );
                assert!(state.is_ok(), "{k}: {state:?}");
            }
            Err(AnalyticPropagationError::NotElliptic {
                field: "osculating elements",
            }) => {
                refused += 1;
                assert!(
                    matches!(
                        state,
                        Err(AnalyticPropagationError::NotElliptic {
                            field: "osculating elements"
                        })
                    ),
                    "{k}: {state:?}"
                );
            }
            Err(other) => panic!("{k}: {other:?}"),
        }
    }
    assert!(refused > 0);
}

#[test]
fn keplerian_eccentricity_is_below_one_whenever_the_vector_is() {
    // CircularElements accepts ex^2 + ey^2 < 1 and KeplerianElements takes
    // e = sqrt(ex^2 + ey^2). Every double s < 1 is at most 1 - 2^-53, and
    // sqrt(1 - 2^-53) = 1 - 2^-54 - 2^-109 - ... lies below the midpoint
    // 1 - 2^-54 of 1 - 2^-53 and 1, so the correctly rounded square root
    // of any s < 1 is below 1.
    let mut s = 1.0f64;
    for _ in 0..100_000 {
        s = f64::from_bits(s.to_bits() - 1);
        assert!(libm::sqrt(s) < 1.0, "{s:e}");
    }
    let c = CircularElements {
        a_km: 7000.0,
        ex: 1.0536712127723509e-8,
        ey: 0.9999999999999999,
        i_rad: 0.5,
        raan_rad: 0.0,
        mean_latitude_argument_rad: 0.0,
    };
    assert!(c.ex * c.ex + c.ey * c.ey < 1.0);
    assert!(c.eccentricity() < 1.0);
    for alpha in [0.0, 1.0e-9, 0.5, 3.0] {
        let c = CircularElements {
            mean_latitude_argument_rad: alpha,
            ..c
        };
        let Ok((p, v)) = c.to_cartesian(MU_EARTH) else {
            continue;
        };
        match KeplerianElements::from_cartesian(p, v, MU_EARTH) {
            Ok(k) => {
                assert!(k.e < 1.0, "{k:?}");
                assert!(k.to_cartesian(MU_EARTH).is_ok(), "{k:?}");
            }
            Err(error) => assert!(
                matches!(error, AnalyticPropagationError::NotElliptic { .. }),
                "{error:?}"
            ),
        }
    }
}

/// The reduction of `x` to `(-pi, pi]` by libm's sine and cosine, whose
/// argument reduction is exact.
fn reduced(x: f64) -> f64 {
    libm::atan2(libm::sin(x), libm::cos(x))
}

#[test]
fn angles_far_from_zero_are_reduced_exactly() {
    let g = ZonalHarmonics::earth();
    for big in [1.0e16, -3.3e15, 7.0e12, 123_456.789, 1.0e300] {
        // Circular elements: the position is that of the reduced angle.
        let c = CircularElements {
            a_km: 7000.0,
            ex: 0.1,
            ey: 0.0,
            i_rad: 0.5,
            raan_rad: 0.0,
            mean_latitude_argument_rad: big,
        };
        let c_red = CircularElements {
            mean_latitude_argument_rad: reduced(big),
            ..c
        };
        let (p, v) = c.to_cartesian(MU_EARTH).unwrap();
        let (p_red, v_red) = c_red.to_cartesian(MU_EARTH).unwrap();
        for k in 0..3 {
            assert!(
                (p[k] - p_red[k]).abs() < 1.0e-8,
                "{big}: {p:?} vs {p_red:?}"
            );
            assert!((v[k] - v_red[k]).abs() < 1.0e-11, "{big}");
        }
        // And it satisfies Kepler's equation for the given angle.
        let back = CircularElements::from_cartesian(p, v, MU_EARTH).unwrap();
        assert!(
            angle_diff(back.mean_latitude_argument_rad, reduced(big)) < 1.0e-12,
            "{big}: {}",
            back.mean_latitude_argument_rad
        );

        let kep = KeplerianElements {
            a_km: 7000.0,
            e: 0.1,
            i_rad: 0.5,
            arg_perigee_rad: big,
            raan_rad: big,
            mean_anomaly_rad: big,
        };
        let kep_red = KeplerianElements {
            arg_perigee_rad: reduced(big),
            raan_rad: reduced(big),
            mean_anomaly_rad: reduced(big),
            ..kep
        };
        let (p, _) = kep.to_cartesian(MU_EARTH).unwrap();
        let (p_red, _) = kep_red.to_cartesian(MU_EARTH).unwrap();
        for k in 0..3 {
            assert!(
                (p[k] - p_red[k]).abs() < 1.0e-8,
                "{big}: {p:?} vs {p_red:?}"
            );
        }

        // Both theories, from mean elements with the angle far from zero.
        let eh = |m: CircularElements| {
            EcksteinHechlerPropagator::from_mean_elements(0.0, m, &g)
                .unwrap()
                .propagate(600.0)
                .unwrap()
                .position_km
        };
        let mean_c = CircularElements {
            ex: 0.001,
            raan_rad: big,
            ..c
        };
        let mean_c_red = CircularElements {
            ex: 0.001,
            raan_rad: reduced(big),
            ..c_red
        };
        assert!((eh(mean_c) - eh(mean_c_red)).norm() < 1.0e-8, "{big}");
        let bl = |m: KeplerianElements| {
            BrouwerLyddanePropagator::from_mean_elements(0.0, m, &g, 0.0)
                .unwrap()
                .propagate(600.0)
                .unwrap()
                .position_km
        };
        assert!((bl(kep) - bl(kep_red)).norm() < 1.0e-8, "{big}");
    }
}

#[test]
fn cartesian_conversion_handles_extreme_scales() {
    // r v^2 / mu = 1 with half the speed radial: a = r and e = sqrt(1/2).
    // mu * a = 2e320 overflows; sqrt(mu) * sqrt(a) does not.
    let mu = 2.0e300;
    let (p, v) = ([1.0e20, 0.0, 0.0], [1.0e140, 1.0e140, 0.0]);
    let c = CircularElements::from_cartesian(p, v, mu).unwrap();
    assert!((c.a_km / 1.0e20 - 1.0).abs() < 1.0e-14, "{c:?}");
    assert!(
        (c.eccentricity() - libm::sqrt(0.5)).abs() < 1.0e-14,
        "{c:?}"
    );
    let k = KeplerianElements::from_cartesian(p, v, mu).unwrap();
    assert!((k.e - libm::sqrt(0.5)).abs() < 1.0e-14, "{k:?}");

    // Circular orbits whose r^2 underflows, or whose v^2 overflows.
    for (r, mu) in [
        (1.0e-200, MU_EARTH),
        (1.0e-200, 1.0e120),
        (1.0e200, 1.0e300),
    ] {
        let speed = libm::sqrt(mu) / libm::sqrt(r);
        let (p, v) = ([r, 0.0, 0.0], [0.0, speed * 0.6, speed * 0.8]);
        let c = CircularElements::from_cartesian(p, v, mu).unwrap();
        assert!((c.a_km / r - 1.0).abs() < 1.0e-14, "{r} {mu}: {c:?}");
        assert!(c.eccentricity() < 1.0e-14, "{r} {mu}: {c:?}");
        // h = r x v = r speed (0, -0.8, 0.6).
        assert!(
            (c.i_rad - libm::acos(0.6)).abs() < 1.0e-14,
            "{r} {mu}: {c:?}"
        );
        let (p2, v2) = c.to_cartesian(mu).unwrap();
        for k in 0..3 {
            assert!((p2[k] - p[k]).abs() <= 1.0e-14 * r, "{r} {mu}: {p2:?}");
            assert!((v2[k] - v[k]).abs() <= 1.0e-14 * speed, "{r} {mu}: {v2:?}");
        }
        let k = KeplerianElements::from_cartesian(p, v, mu).unwrap();
        let (p3, v3) = k.to_cartesian(mu).unwrap();
        for j in 0..3 {
            assert!((p3[j] - p[j]).abs() <= 1.0e-14 * r, "{r} {mu}: {p3:?}");
            assert!((v3[j] - v[j]).abs() <= 1.0e-14 * speed, "{r} {mu}: {v3:?}");
        }
    }
}

/// Brouwer-Lyddane position at `dt` from mean elements `(a, e, i)` with the
/// other angles fixed.
fn bl_position(a: f64, e: f64, i: f64, dt: f64) -> [f64; 3] {
    let mean = KeplerianElements {
        a_km: a,
        e,
        i_rad: i,
        arg_perigee_rad: 0.4,
        raan_rad: 1.1,
        mean_anomaly_rad: 0.3,
    };
    let state =
        BrouwerLyddanePropagator::from_mean_elements(0.0, mean, &ZonalHarmonics::earth(), 0.0)
            .unwrap_or_else(|error| panic!("i = {i}: {error:?}"))
            .propagate(dt)
            .unwrap_or_else(|error| panic!("i = {i}: {error:?}"));
    state.position_array()
}

fn distance(p: [f64; 3], q: [f64; 3]) -> f64 {
    libm::sqrt((0..3).map(|k| (p[k] - q[k]) * (p[k] - q[k])).sum())
}

#[test]
fn brouwer_lyddane_is_continuous_at_small_inclinations() {
    // Tilting the mean orbit plane by d rad moves a point at radius r by at
    // most r d; anything more is the theory, not the geometry.
    let (a, e) = (7000.0, 0.5);
    let r_max = a * (1.0 + e);
    for dt in [0.0, 3_600.0, 86_400.0] {
        let equatorial = bl_position(a, e, 0.0, dt);
        for d in [1.0e-17, 1.0e-12, 1.0e-8, 1.0e-4] {
            let near = bl_position(a, e, d, dt);
            assert!(
                distance(near, equatorial) <= 2.0 * r_max * d + 1.0e-9,
                "i = {d}, dt {dt}: {} km",
                distance(near, equatorial)
            );
        }
    }
}

#[test]
fn brouwer_lyddane_is_regular_at_retrograde_inclinations() {
    use core::f64::consts::{FRAC_PI_2, PI};
    let (a, e) = (7000.0, 0.5);
    let r_max = a * (1.0 + e);
    for dt in [0.0, 3_600.0, 86_400.0] {
        let equatorial = bl_position(a, e, PI, dt);
        for d in [1.0e-17, 1.0e-12, 1.0e-8, 1.0e-4] {
            let near = bl_position(a, e, PI - d, dt);
            assert!(
                distance(near, equatorial) <= 2.0 * r_max * d + 1.0e-9,
                "i = pi - {d}, dt {dt}: {} km",
                distance(near, equatorial)
            );
        }
        // Well inside the band where the prograde form of Lyddane's
        // variables leaves the unit disc.
        bl_position(a, e, PI - 0.01, dt);
        // Across the switch to the time-reversed evaluation at i = pi / 2.
        let below = bl_position(a, e, FRAC_PI_2 - 1.0e-12, dt);
        let above = bl_position(a, e, FRAC_PI_2 + 1.0e-12, dt);
        assert!(
            distance(below, above) <= 4.0 * r_max * 1.0e-12 + 1.0e-9,
            "dt {dt}: {} km",
            distance(below, above)
        );
    }
}

/// Mean Keplerian elements whose inclination is outside `[0, pi]`, each with
/// the same orbit written out by hand with the inclination in `[0, pi]`: the
/// inclination reduced modulo `2 pi`, and a negative one replaced by
/// `(i, w, raan) -> (-i, w + pi, raan + pi)`.
fn keplerian_inclination_cases() -> Vec<(KeplerianElements, KeplerianElements)> {
    use core::f64::consts::PI;
    let base = KeplerianElements {
        a_km: 7000.0,
        e: 0.01,
        i_rad: 0.0,
        arg_perigee_rad: 0.5,
        raan_rad: 1.0,
        mean_anomaly_rad: 0.3,
    };
    let flipped = |i: f64| KeplerianElements {
        i_rad: i,
        arg_perigee_rad: base.arg_perigee_rad + PI,
        raan_rad: base.raan_rad + PI,
        ..base
    };
    let eccentric = KeplerianElements {
        a_km: 7000.0,
        e: 0.5,
        i_rad: -PI,
        arg_perigee_rad: 0.4,
        raan_rad: 1.1,
        mean_anomaly_rad: 0.3,
    };
    vec![
        // i in (-pi, -pi/2).
        (
            KeplerianElements {
                i_rad: -2.0,
                ..base
            },
            flipped(2.0),
        ),
        // i in (3 pi/2, 2 pi].
        (
            KeplerianElements { i_rad: 5.0, ..base },
            flipped(2.0 * PI - 5.0),
        ),
        // i in (2 pi, 5 pi/2).
        (
            KeplerianElements {
                i_rad: 2.0 * PI + 0.7,
                ..base
            },
            KeplerianElements {
                i_rad: reduced(2.0 * PI + 0.7),
                ..base
            },
        ),
        // i = -pi, eccentric.
        (
            eccentric,
            KeplerianElements {
                i_rad: PI,
                arg_perigee_rad: 0.4 + PI,
                raan_rad: 1.1 + PI,
                ..eccentric
            },
        ),
        // Far from zero.
        (
            KeplerianElements {
                i_rad: 1.0e16,
                ..base
            },
            KeplerianElements {
                i_rad: reduced(1.0e16),
                ..base
            },
        ),
        (
            KeplerianElements {
                i_rad: -1.0e16,
                ..base
            },
            flipped(-reduced(-1.0e16)),
        ),
        // i in (-pi/2, 0).
        (
            KeplerianElements {
                i_rad: -0.3,
                ..base
            },
            flipped(0.3),
        ),
    ]
}

#[test]
fn brouwer_lyddane_accepts_inclinations_outside_zero_to_pi() {
    use core::f64::consts::PI;
    let g = ZonalHarmonics::earth();
    for (given, same) in keplerian_inclination_cases() {
        assert!((0.0..=PI).contains(&same.i_rad), "{same:?}");
        // The hand-written elements are the same orbit.
        let (p_given, _) = given.to_cartesian(g.mu_km3_s2).unwrap();
        let (p_same, _) = same.to_cartesian(g.mu_km3_s2).unwrap();
        assert!(distance(p_given, p_same) < 1.0e-8, "{given:?}");

        // Mean to osculating at the epoch.
        let osc = brouwer_lyddane_osculating_elements(&given, &g).unwrap();
        let osc_same = brouwer_lyddane_osculating_elements(&same, &g).unwrap();
        assert!((0.0..=PI).contains(&osc.i_rad), "{given:?}: {osc:?}");
        let (p, v) = osc.to_cartesian(g.mu_km3_s2).unwrap();
        let (p_ref, v_ref) = osc_same.to_cartesian(g.mu_km3_s2).unwrap();
        assert!(
            distance(p, p_ref) < 1.0e-9,
            "{given:?}: {p:?} vs {p_ref:?}, {} km",
            distance(p, p_ref)
        );
        assert!(distance(v, v_ref) < 1.0e-12, "{given:?}");
        // The periodic terms move a 7000 km orbit by kilometres, not across
        // the equator.
        if given.e < 0.1 {
            assert!(
                distance(p, p_given) < 50.0,
                "{given:?}: {} km from the mean orbit",
                distance(p, p_given)
            );
        }

        // The propagator, with the drag term, on both sides of the epoch.
        let propagator =
            BrouwerLyddanePropagator::from_mean_elements(0.0, given, &g, 1.0e-14).unwrap();
        let reference =
            BrouwerLyddanePropagator::from_mean_elements(0.0, same, &g, 1.0e-14).unwrap();
        let mean = propagator.mean_elements();
        assert!((0.0..=PI).contains(&mean.i_rad), "{given:?}: {mean:?}");
        for dt in [0.0, 3_600.0, 86_400.0, -86_400.0] {
            let x = propagator.propagate(dt).unwrap().position_array();
            let y = reference.propagate(dt).unwrap().position_array();
            assert!(
                distance(x, y) < 1.0e-8,
                "{given:?} dt {dt}: {} km",
                distance(x, y)
            );
            let osc = propagator.osculating_elements(dt).unwrap();
            assert!(
                (0.0..=PI).contains(&osc.i_rad),
                "{given:?} dt {dt}: {osc:?}"
            );
        }
    }
}

#[test]
fn eckstein_hechler_accepts_inclinations_outside_zero_to_pi() {
    use core::f64::consts::PI;
    let g = ZonalHarmonics::earth();
    let base = CircularElements {
        a_km: 7000.0,
        ex: 0.001,
        ey: 0.0005,
        i_rad: 0.0,
        raan_rad: 1.0,
        mean_latitude_argument_rad: 0.8,
    };
    // w -> w + pi negates (ex, ey) and moves the argument of latitude by pi.
    let flipped = |i: f64| CircularElements {
        ex: -base.ex,
        ey: -base.ey,
        i_rad: i,
        raan_rad: base.raan_rad + PI,
        mean_latitude_argument_rad: base.mean_latitude_argument_rad + PI,
        ..base
    };
    let unflipped = |i: f64| CircularElements { i_rad: i, ..base };
    for (i, same) in [
        (-2.0, flipped(2.0)),
        (5.0, flipped(2.0 * PI - 5.0)),
        (2.0 * PI + 0.7, unflipped(reduced(2.0 * PI + 0.7))),
        (1.0e16, unflipped(reduced(1.0e16))),
        (-1.0e16, flipped(-reduced(-1.0e16))),
        (-0.3, flipped(0.3)),
    ] {
        let given = CircularElements { i_rad: i, ..base };
        assert!((0.0..=PI).contains(&same.i_rad), "{same:?}");
        let (p_given, _) = given.to_cartesian(g.mu_km3_s2).unwrap();
        let (p_same, _) = same.to_cartesian(g.mu_km3_s2).unwrap();
        assert!(distance(p_given, p_same) < 1.0e-8, "i = {i}");

        let osc = eckstein_hechler_osculating_elements(&given, &g).unwrap();
        let osc_same = eckstein_hechler_osculating_elements(&same, &g).unwrap();
        assert!((0.0..=PI).contains(&osc.i_rad), "i = {i}: {osc:?}");
        let (p, _) = osc.to_cartesian(g.mu_km3_s2).unwrap();
        let (p_ref, _) = osc_same.to_cartesian(g.mu_km3_s2).unwrap();
        assert!(
            distance(p, p_ref) < 1.0e-9,
            "i = {i}: {} km",
            distance(p, p_ref)
        );
        assert!(distance(p, p_given) < 50.0, "i = {i}");

        let propagator = EcksteinHechlerPropagator::from_mean_elements(0.0, given, &g).unwrap();
        let reference = EcksteinHechlerPropagator::from_mean_elements(0.0, same, &g).unwrap();
        let mean = propagator.mean_elements();
        assert!((0.0..=PI).contains(&mean.i_rad), "i = {i}: {mean:?}");
        for dt in [0.0, 3_600.0, 86_400.0, -86_400.0] {
            let x = propagator.propagate(dt).unwrap();
            let y = reference.propagate(dt).unwrap();
            assert!(
                distance(x.position_array(), y.position_array()) < 1.0e-8,
                "i = {i} dt {dt}: {} km",
                distance(x.position_array(), y.position_array())
            );
            assert!(
                distance(x.velocity_array(), y.velocity_array()) < 1.0e-11,
                "i = {i} dt {dt}"
            );
        }
    }
}

#[test]
fn eckstein_hechler_osculating_inclination_stays_in_zero_to_pi() {
    // Near the equator the J3 inclination term, proportional to e J3 / J2 and
    // not to sin i, is larger than a small mean inclination, so the mean
    // inclination plus its periodic terms leaves [0, pi]. The returned
    // elements are the same orbit with the inclination folded back, the
    // node and the argument of latitude moved by pi.
    use core::f64::consts::PI;
    let g = ZonalHarmonics::earth();
    for i in [2.0e-5, PI - 2.0e-5] {
        let mean = CircularElements {
            a_km: 7000.0,
            ex: 0.0,
            ey: 0.05,
            i_rad: i,
            raan_rad: 1.0,
            mean_latitude_argument_rad: 0.8,
        };
        let propagator = EcksteinHechlerPropagator::from_mean_elements(0.0, mean, &g).unwrap();
        for dt in [0.0, 600.0] {
            let osc = propagator.osculating_elements(dt).unwrap();
            assert!((0.0..=PI).contains(&osc.i_rad), "i = {i} dt {dt}: {osc:?}");
            // The node moved by pi: the case is not vacuous.
            assert!(
                angle_diff(osc.raan_rad, mean.raan_rad + PI) < 0.1,
                "i = {i} dt {dt}: {osc:?}"
            );
            let (p, _) = osc.to_cartesian(g.mu_km3_s2).unwrap();
            let state = propagator.propagate(dt).unwrap();
            assert!(
                distance(p, state.position_array()) < 1.0e-9,
                "i = {i} dt {dt}: {} km",
                distance(p, state.position_array())
            );
        }
        let osc = eckstein_hechler_osculating_elements(&mean, &g).unwrap();
        assert!((0.0..=PI).contains(&osc.i_rad), "i = {i}: {osc:?}");
    }
}

#[test]
fn eckstein_hechler_position_reduces_a_large_argument_of_latitude() {
    // A contrived field with J3 and J5 of order 1e16, balanced so that the
    // frozen eccentricity eps2 cancels to rounding: at alpha = 0 the
    // eccentricity and semi-major axis terms stay small while the J3/J5
    // argument-of-latitude term puts the osculating argument of latitude
    // far beyond 1e16, where adding the eccentric correction to it rounds
    // the correction away.
    let (a, i) = (7000.0, 1.0);
    let s2 = libm::sin(i) * libm::sin(i);
    let q = RE_EARTH / a;
    let c30 = 1.0e16;
    let c50 = c30 * (4.0 - 5.0 * s2) / ((10.0 - 35.0 * s2 + 26.25 * s2 * s2) * q * q);
    let g = ZonalHarmonics {
        c30,
        c50,
        ..ZonalHarmonics::earth()
    };
    let mean = CircularElements {
        a_km: a,
        ex: 0.05,
        ey: 0.0,
        i_rad: i,
        raan_rad: 0.0,
        mean_latitude_argument_rad: 0.0,
    };
    let propagator = EcksteinHechlerPropagator::from_mean_elements(0.0, mean, &g).unwrap();
    let osc = propagator.osculating_elements(0.0).unwrap();
    assert!(osc.mean_latitude_argument_rad.abs() > 1.0e16, "{osc:?}");
    let (expected, _) = osc.to_cartesian(g.mu_km3_s2).unwrap();
    let position = propagator.propagate(0.0).unwrap().position_array();
    assert!(
        distance(position, expected) < 1.0e-6,
        "{position:?} vs {expected:?}: {} km",
        distance(position, expected)
    );
}

#[test]
fn inclination_canonicalization_keeps_the_cartesian_state() {
    // The angles move by multiples of pi, which no double represents, so the
    // state agrees to the rounding of a few units in the last place of the
    // angles, not bit for bit. Elements already in [0, pi] are returned
    // unchanged, bit for bit.
    use core::f64::consts::{FRAC_PI_2, PI};
    let norm = |x: [f64; 3]| libm::sqrt(x.iter().map(|c| c * c).sum());
    let outside = [
        -2.0,
        5.0,
        -PI,
        f64::from_bits(PI.to_bits() + 1),
        2.0 * PI,
        2.0 * PI + 0.7,
        1.5 * PI + 0.2,
        -FRAC_PI_2 - 0.3,
        -FRAC_PI_2,
        -0.3,
        -1.0e-300,
        7.5,
        -7.5,
        4.0 * PI + 3.0,
        1.0e16,
        -1.0e16,
        1.0e300,
        -1.0e300,
    ];
    let inside = [0.0, 1.0e-300, 0.3, FRAC_PI_2, 2.0, PI];
    let mut flipped = 0;
    for (w, raan, anomaly) in [(0.5, 1.0, 0.3), (-2.9, 3.1, -1.0), (1.0e16, -7.0, 40.0)] {
        for e in [0.0, 0.01, 0.5] {
            for i in outside.into_iter().chain(inside) {
                let k = KeplerianElements {
                    a_km: 7000.0,
                    e,
                    i_rad: i,
                    arg_perigee_rad: w,
                    raan_rad: raan,
                    mean_anomaly_rad: anomaly,
                };
                let c = CircularElements {
                    a_km: 7000.0,
                    ex: e * libm::cos(w),
                    ey: e * libm::sin(w),
                    i_rad: i,
                    raan_rad: raan,
                    mean_latitude_argument_rad: anomaly,
                };
                let (kc, cc) = (k.canonical(), c.canonical());
                assert!((0.0..=PI).contains(&kc.i_rad), "{k:?} -> {kc:?}");
                assert!((0.0..=PI).contains(&cc.i_rad), "{c:?} -> {cc:?}");
                assert_eq!(kc.canonical(), kc);
                assert_eq!(cc.canonical(), cc);
                if inside.contains(&i) {
                    assert_eq!(kc, k);
                    assert_eq!(cc, c);
                }
                if cc.ex != c.ex || cc.raan_rad != c.raan_rad {
                    flipped += 1;
                }
                for ((p, v), (pc, vc), label) in [
                    (
                        k.to_cartesian(MU_EARTH).unwrap(),
                        kc.to_cartesian(MU_EARTH).unwrap(),
                        "keplerian",
                    ),
                    (
                        c.to_cartesian(MU_EARTH).unwrap(),
                        cc.to_cartesian(MU_EARTH).unwrap(),
                        "circular",
                    ),
                ] {
                    assert!(
                        distance(p, pc) <= 1.0e-14 * norm(p),
                        "{label} i = {i:e} e {e} w {w}: {p:?} vs {pc:?}"
                    );
                    assert!(
                        distance(v, vc) <= 1.0e-14 * norm(v),
                        "{label} i = {i:e} e {e} w {w}: {v:?} vs {vc:?}"
                    );
                }
            }
        }
    }
    // Both branches are exercised.
    assert!(flipped > 0);
}
