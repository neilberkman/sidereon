#![cfg(sidereon_repo_tests)]
//! The Eckstein-Hechler and Brouwer-Lyddane propagators and their
//! mean/osculating conversions against Orekit 13.1.10.
//!
//! Oracle: Orekit `EcksteinHechlerPropagator`, `BrouwerLyddanePropagator` and
//! `FixedPointConverter` with `EcksteinHechlerTheory` and
//! `BrouwerLyddaneTheory`, run by
//! `fixtures-generators/orekit_analytic_oracle/generate.sh` into
//! `tests/fixtures/analytic/orekit_analytic_propagators.json`, which records
//! the Orekit, Hipparchus and orekit-data versions. Both sides use the same
//! gravity (mu, reference radius, unnormalized C20..C60) and start from the
//! same osculating Cartesian doubles. Theories: Eckstein and Hechler, ESRO
//! SR-13 (1970); Brouwer, AJ 64 (1959); Lyddane, AJ 68 (1963).
//!
//! Eckstein-Hechler evaluates the same closed-form expressions as Orekit, so
//! the residual disagreement is floating-point rounding: libm against
//! Hipparchus FastMath for the trigonometric functions, km against m for the
//! units, and element conversions taking different but equivalent paths
//! (Orekit converts equinoctial elements to Keplerian through a Cartesian
//! state). A tolerance is given per quantity with its reason where it is
//! defined.
//!
//! Brouwer-Lyddane follows Brouwer (1959) in three terms where Orekit 13.1
//! does not (see `astro::propagator::analytic`). The comparison with Orekit
//! runs the test-only replay of Orekit's arrangement of those terms, which
//! agrees to rounding and so shows that the public propagator differs from
//! Orekit by those terms alone; the published terms themselves are tested in
//! the crate's unit tests and against numerical integration below.
//!
//! The independent check of the formulas themselves, against numerical
//! integration of the same zonal field, is at the end of this file.

use std::path::PathBuf;

use serde_json::Value;
use sidereon_core::astro::forces::{TideSystem, ZonalCoefficients, ZonalDegrees, ZonalGravity};
use sidereon_core::astro::propagator::analytic::brouwer_lyddane_orekit_replay as orekit_bl;
use sidereon_core::astro::propagator::analytic::{
    brouwer_lyddane_mean_elements, eckstein_hechler_mean_elements,
    eckstein_hechler_osculating_elements, AnalyticPropagationError, AnalyticTheory,
    BrouwerLyddanePropagator, CircularElements, EcksteinHechlerPropagator, KeplerianElements,
    MeanConversionOptions, ZonalHarmonics,
};
use sidereon_core::astro::propagator::api::IntegratorOptions;
use sidereon_core::astro::propagator::{
    ForceModelComponents, ForceModelKind, IntegratorKind, StatePropagator,
};
use sidereon_core::astro::state::CartesianState;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/analytic/orekit_analytic_propagators.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn gravity(fx: &Value) -> ZonalHarmonics {
    let g = &fx["gravity"];
    ZonalHarmonics {
        mu_km3_s2: f(&g["mu_m3_s2"]) * 1.0e-9,
        re_km: f(&g["re_m"]) * 1.0e-3,
        c20: f(&g["c20"]),
        c30: f(&g["c30"]),
        c40: f(&g["c40"]),
        c50: f(&g["c50"]),
        c60: f(&g["c60"]),
    }
}

fn state(v: &Value, epoch: f64) -> CartesianState {
    let p: Vec<f64> = v["position_m"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| f(x) * 1.0e-3)
        .collect();
    let w: Vec<f64> = v["velocity_m_s"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| f(x) * 1.0e-3)
        .collect();
    CartesianState::new(epoch, [p[0], p[1], p[2]], [w[0], w[1], w[2]])
}

fn circular(v: &Value) -> CircularElements {
    CircularElements {
        a_km: f(&v["a_m"]) * 1.0e-3,
        ex: f(&v["ex"]),
        ey: f(&v["ey"]),
        i_rad: f(&v["i"]),
        raan_rad: f(&v["raan"]),
        mean_latitude_argument_rad: f(&v["alpha_m"]),
    }
}

fn keplerian(v: &Value) -> KeplerianElements {
    KeplerianElements {
        a_km: f(&v["a_m"]) * 1.0e-3,
        e: f(&v["e"]),
        i_rad: f(&v["i"]),
        arg_perigee_rad: f(&v["pa"]),
        raan_rad: f(&v["raan"]),
        mean_anomaly_rad: f(&v["m"]),
    }
}

fn wrap(a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    a - two_pi * ((a + std::f64::consts::PI) / two_pi).floor()
}

/// Position (m) and velocity (m/s) differences.
fn pv_diff(ours: &CartesianState, orekit: &CartesianState) -> (f64, f64) {
    (
        (ours.position_km - orekit.position_km).norm() * 1.0e3,
        (ours.velocity_km_s - orekit.velocity_km_s).norm() * 1.0e3,
    )
}

/// Largest difference of circular elements: semi-major axis in m, the rest
/// as angles or eccentricity components.
fn circ_diff(a: &CircularElements, b: &CircularElements) -> (f64, f64) {
    let da = (a.a_km - b.a_km).abs() * 1.0e3;
    let rest = [
        (a.ex - b.ex).abs(),
        (a.ey - b.ey).abs(),
        (a.i_rad - b.i_rad).abs(),
        wrap(a.raan_rad - b.raan_rad).abs(),
        wrap(a.mean_latitude_argument_rad - b.mean_latitude_argument_rad).abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max);
    (da, rest)
}

/// Keplerian elements compared in a form defined for circular and equatorial
/// orbits: a (m), and e cos(w + raan), e sin(w + raan), tan(i/2) cos(raan),
/// tan(i/2) sin(raan), w + raan + M.
fn kep_diff(a: &KeplerianElements, b: &KeplerianElements) -> (f64, f64) {
    let nonsingular = |k: &KeplerianElements| {
        let lp = k.arg_perigee_rad + k.raan_rad;
        let t = libm::tan(0.5 * k.i_rad);
        [
            k.e * libm::cos(lp),
            k.e * libm::sin(lp),
            t * libm::cos(k.raan_rad),
            t * libm::sin(k.raan_rad),
            lp + k.mean_anomaly_rad,
        ]
    };
    let (x, y) = (nonsingular(a), nonsingular(b));
    let mut rest = 0.0f64;
    for k in 0..4 {
        rest = rest.max((x[k] - y[k]).abs());
    }
    rest = rest.max(wrap(x[4] - y[4]).abs());
    ((a.a_km - b.a_km).abs() * 1.0e3, rest)
}

#[derive(Default, Debug)]
struct Worst {
    mean_a_m: f64,
    mean_angle: f64,
    pos_m: f64,
    vel_m_s: f64,
    osc_a_m: f64,
    osc_angle: f64,
}

impl Worst {
    fn pv(&mut self, d: (f64, f64)) {
        self.pos_m = self.pos_m.max(d.0);
        self.vel_m_s = self.vel_m_s.max(d.1);
    }
    fn osc(&mut self, d: (f64, f64)) {
        self.osc_a_m = self.osc_a_m.max(d.0);
        self.osc_angle = self.osc_angle.max(d.1);
    }
}

fn options(case: &Value) -> MeanConversionOptions {
    MeanConversionOptions::new(
        f(&case["tolerance"]),
        case["max_iterations"].as_u64().unwrap() as u32,
    )
}

#[test]
fn eckstein_hechler_matches_orekit() {
    let fx = fixture();
    let g = gravity(&fx);
    let mut report = Vec::new();
    for case in fx["eckstein_hechler"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let initial = state(&case["initial"], 0.0);
        let mut worst = Worst::default();

        // Osculating to mean.
        let opts = options(case);
        let mean = eckstein_hechler_mean_elements(&initial, &g, opts).unwrap();
        let d = circ_diff(&mean.elements, &circular(&case["mean"]));
        worst.mean_a_m = d.0;
        worst.mean_angle = d.1;
        assert_eq!(
            mean.iterations,
            case["mean_iterations"].as_u64().unwrap() as u32,
            "{name}: iteration count"
        );

        // Propagation from the osculating state.
        let prop = EcksteinHechlerPropagator::new(&initial, &g, opts).unwrap();
        for s in case["samples"].as_array().unwrap() {
            let dt = f(&s["dt_s"]);
            worst.pv(pv_diff(
                &prop.propagate(dt).unwrap(),
                &state(&s["state"], dt),
            ));
            worst.osc(circ_diff(
                &prop.osculating_elements(dt).unwrap(),
                &circular(&s["osculating"]),
            ));
        }

        // Mean to osculating, and propagation from mean elements.
        let from_mean = &case["from_mean"];
        let mean_in = circular(&from_mean["mean"]);
        let prop = EcksteinHechlerPropagator::from_mean_elements(0.0, mean_in, &g).unwrap();
        for s in from_mean["samples"].as_array().unwrap() {
            let dt = f(&s["dt_s"]);
            worst.pv(pv_diff(
                &prop.propagate(dt).unwrap(),
                &state(&s["state"], dt),
            ));
            let osc = circular(&s["osculating"]);
            worst.osc(circ_diff(&prop.osculating_elements(dt).unwrap(), &osc));
            if dt == 0.0 {
                worst.osc(circ_diff(
                    &eckstein_hechler_osculating_elements(&mean_in, &g).unwrap(),
                    &osc,
                ));
            }
        }
        report.push((name.to_string(), worst));
    }
    for (name, w) in &report {
        println!("eckstein_hechler {name}: {w:?}");
    }
    for (name, w) in &report {
        check(name, w, &ECKSTEIN_HECHLER);
    }
}

#[test]
fn brouwer_lyddane_matches_orekit() {
    let fx = fixture();
    let g = gravity(&fx);
    let mut report = Vec::new();
    for case in fx["brouwer_lyddane"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let m2 = f(&case["m2"]);
        let initial = state(&case["initial"], 0.0);
        let mut worst = Worst::default();

        if let Some(error) = case.get("orekit_conversion_error") {
            // Orekit's fixed point found no mean elements at any tolerance
            // from 1e-13 to 1e-9 for these near-circular LEO orbits, and
            // neither does the replay of its terms, whose eccentricity terms
            // are off by their own size: the iteration does not settle. With
            // Brouwer's terms the fixed point converges in a few
            // iterations, and the propagator started from it reproduces the
            // osculating state.
            assert_eq!(error.as_str().unwrap(), "UNABLE_TO_COMPUTE_MEAN_PARAMETERS");
            for tolerance in [1.0e-13, 1.0e-12, 1.0e-11, 1.0e-10, 1.0e-9] {
                let result = orekit_bl::mean_elements(
                    &initial,
                    &g,
                    MeanConversionOptions::new(tolerance, 200),
                );
                assert!(
                    matches!(
                        result,
                        Err(AnalyticPropagationError::MeanElementsNotConverged {
                            theory: AnalyticTheory::BrouwerLyddane,
                            iterations: 200
                        })
                    ),
                    "{name} at {tolerance}: {result:?}"
                );
            }
            let mean = brouwer_lyddane_mean_elements(
                &initial,
                &g,
                MeanConversionOptions::BROUWER_LYDDANE_DEFAULT,
            )
            .unwrap();
            assert!(
                mean.iterations <= 10,
                "{name}: {} iterations",
                mean.iterations
            );
            let prop = BrouwerLyddanePropagator::new(
                &initial,
                &g,
                m2,
                MeanConversionOptions::BROUWER_LYDDANE_DEFAULT,
            )
            .unwrap();
            let again = prop.propagate(0.0).unwrap();
            // Converged to 1e-13 relative in the elements: about 1e-6 m in
            // position and 1e-9 m/s in velocity at 7000 km.
            assert!(
                (again.position_km - initial.position_km).norm() < 1.0e-8,
                "{name}"
            );
            assert!(
                (again.velocity_km_s - initial.velocity_km_s).norm() < 1.0e-11,
                "{name}"
            );
        } else {
            let opts = options(case);
            let mean = orekit_bl::mean_elements(&initial, &g, opts).unwrap();
            let d = kep_diff(&mean.elements, &keplerian(&case["mean"]));
            worst.mean_a_m = d.0;
            worst.mean_angle = d.1;
            assert_eq!(
                mean.iterations,
                case["mean_iterations"].as_u64().unwrap() as u32,
                "{name}: iteration count"
            );
            let prop = orekit_bl::new(&initial, &g, m2, opts).unwrap();
            for s in case["samples"].as_array().unwrap() {
                let dt = f(&s["dt_s"]);
                worst.pv(pv_diff(
                    &prop.propagate(dt).unwrap(),
                    &state(&s["state"], dt),
                ));
                worst.osc(kep_diff(
                    &prop.osculating_elements(dt).unwrap(),
                    &keplerian(&s["osculating"]),
                ));
            }
        }

        let from_mean = &case["from_mean"];
        let mean_in = keplerian(&from_mean["mean"]);
        let prop = orekit_bl::from_mean_elements(0.0, mean_in, &g, m2).unwrap();
        for s in from_mean["samples"].as_array().unwrap() {
            let dt = f(&s["dt_s"]);
            worst.pv(pv_diff(
                &prop.propagate(dt).unwrap(),
                &state(&s["state"], dt),
            ));
            worst.osc(kep_diff(
                &prop.osculating_elements(dt).unwrap(),
                &keplerian(&s["osculating"]),
            ));
        }
        report.push((name.to_string(), worst));
    }
    for (name, w) in &report {
        println!("brouwer_lyddane {name}: {w:?}");
    }
    for (name, w) in &report {
        check(name, w, &BROUWER_LYDDANE);
    }
}

/// Largest disagreement allowed with Orekit for one theory.
struct Tolerance {
    mean_a_m: f64,
    mean_angle: f64,
    pos_m: f64,
    vel_m_s: f64,
    osc_a_m: f64,
    osc_angle: f64,
}

/// Eckstein-Hechler. Both sides evaluate the same expressions in the same
/// order; what differs is the last bit of the trigonometric functions (libm
/// against FastMath) and of the km/m scaling. Those rounding differences in
/// the mean elements and rates are carried by the secular phase, up to about
/// 6e2 rad after seven days in LEO, so a 1e-16 relative difference reaches a
/// few micrometres. Largest seen: mean a 1.5e-8 m, mean angle 1.3e-15,
/// position 2.3e-6 m (LEO, seven days), velocity 2.6e-9 m/s, osculating a
/// 2.2e-8 m, osculating angle 3.4e-13. The tolerances are about five times
/// those.
const ECKSTEIN_HECHLER: Tolerance = Tolerance {
    mean_a_m: 1.0e-7,
    mean_angle: 1.0e-14,
    pos_m: 1.0e-5,
    vel_m_s: 1.0e-8,
    osc_a_m: 1.0e-7,
    osc_angle: 2.0e-12,
};

/// Brouwer-Lyddane against Orekit's arrangement of the terms (the replay).
/// The same rounding as for Eckstein-Hechler, plus two amplifiers: the GTO
/// and Molniya cases (e above 0.7) solve Kepler's equation where
/// dE/dM = 1 / (1 - e cos E) reaches 3.7, and Orekit converts every
/// fixed-point iterate to Keplerian elements through a Cartesian state where
/// this code converts directly. Largest seen (GTO, seven days): mean a 3.8e-7 m,
/// mean angle 3.9e-15, position 7.6e-5 m, velocity 3.3e-8 m/s, osculating a
/// 5.0e-7 m, osculating angle 1.7e-12. The tolerances are about four times
/// those.
const BROUWER_LYDDANE: Tolerance = Tolerance {
    mean_a_m: 2.0e-6,
    mean_angle: 2.0e-14,
    pos_m: 3.0e-4,
    vel_m_s: 1.5e-7,
    osc_a_m: 2.0e-6,
    osc_angle: 1.0e-11,
};

fn check(name: &str, w: &Worst, t: &Tolerance) {
    assert!(w.mean_a_m <= t.mean_a_m, "{name}: mean a {} m", w.mean_a_m);
    assert!(
        w.mean_angle <= t.mean_angle,
        "{name}: mean elements {}",
        w.mean_angle
    );
    assert!(w.pos_m <= t.pos_m, "{name}: position {} m", w.pos_m);
    assert!(w.vel_m_s <= t.vel_m_s, "{name}: velocity {} m/s", w.vel_m_s);
    assert!(
        w.osc_a_m <= t.osc_a_m,
        "{name}: osculating a {} m",
        w.osc_a_m
    );
    assert!(
        w.osc_angle <= t.osc_angle,
        "{name}: osculating elements {}",
        w.osc_angle
    );
}

#[test]
fn refusals_match_orekit() {
    let fx = fixture();
    let g = gravity(&fx);
    for case in fx["errors"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let initial = state(&case["initial"], 0.0);
        let opts = MeanConversionOptions::new(
            f(&case["tolerance"]),
            case["max_iterations"].as_u64().unwrap() as u32,
        );
        let result = match case["theory"].as_str().unwrap() {
            "eckstein_hechler" => EcksteinHechlerPropagator::new(&initial, &g, opts).map(|_| ()),
            _ => BrouwerLyddanePropagator::new(&initial, &g, 0.0, opts).map(|_| ()),
        };
        let error = result.expect_err(name);
        let expected = case["orekit_error"].as_str().unwrap();
        let matches = match expected {
            "TOO_LARGE_ECCENTRICITY_FOR_PROPAGATION_MODEL" => {
                matches!(error, AnalyticPropagationError::EccentricityTooLarge { .. })
            }
            "ALMOST_CRITICALLY_INCLINED_ORBIT" => {
                matches!(
                    error,
                    AnalyticPropagationError::NearCriticalInclination { .. }
                )
            }
            "ALMOST_EQUATORIAL_ORBIT" => {
                matches!(error, AnalyticPropagationError::NearlyEquatorial { .. })
            }
            "TRAJECTORY_INSIDE_BRILLOUIN_SPHERE" => {
                matches!(
                    error,
                    AnalyticPropagationError::InsideReferenceSphere { .. }
                )
            }
            "UNABLE_TO_COMPUTE_MEAN_PARAMETERS" => {
                matches!(
                    error,
                    AnalyticPropagationError::MeanElementsNotConverged { .. }
                )
            }
            other => panic!("{name}: unmapped Orekit error {other}"),
        };
        assert!(matches, "{name}: Orekit {expected}, ours {error:?}");
    }
}

/// Radial, along-track and cross-track components (km) of `ours - truth`.
fn rtn(ours: &CartesianState, truth: &CartesianState) -> [f64; 3] {
    let r = truth.position_km.normalize();
    let n = truth.position_km.cross(&truth.velocity_km_s).normalize();
    let t = n.cross(&r);
    let d = ours.position_km - truth.position_km;
    [d.dot(&r), d.dot(&t), d.dot(&n)]
}

/// Independent check of the formulas: both theories against numerical
/// integration (Dormand-Prince 5(4), tolerances 1e-12 km and 1e-13, steps of
/// at most 60 s) of the same zonal field, J2 through J6 for Eckstein-Hechler
/// and J2 through J5 for Brouwer-Lyddane, over one day from the same
/// osculating state. The analytic theories truncate the perturbation series,
/// so they agree with the integration only to their accuracy; a wrong sign or
/// factor in a periodic term shows as a kilometre-level radial error within
/// one revolution, as the Orekit arrangement of the Brouwer-Lyddane terms
/// does here.
#[test]
fn theories_follow_numerical_integration_of_the_zonal_field() {
    let g = ZonalHarmonics::earth();
    let integrate =
        |initial: &CartesianState, max_degree: u8, t: f64| integrate(&g, initial, max_degree, t);
    // (name, a km, e, i deg, EH bound km, BL bound km)
    //
    // Eckstein-Hechler carries the J2..J6 short-period terms and the J2^2
    // secular terms: tens of metres per day in LEO for e near 1e-3, a few
    // hundred metres at e = 0.02, where its accuracy is documented to degrade
    // (e above about 0.005). Brouwer-Lyddane carries only the J2 short-period
    // terms, so the mean semi-major axis it recovers is off by the omitted
    // J2^2 and J4 short-period amplitudes, about 1e-6 relative in LEO; the
    // mean motion error 1.5e-6 n then drifts 1 to 3 km along-track per day.
    // Its radial and cross-track errors stay below 200 m and 100 m.
    for (name, a, e, i_deg, eh_bound, bl_bound) in [
        ("leo_sso", 7078.137, 0.0012, 98.19, 0.1, 3.0),
        ("leo_iss", 6778.137, 0.0005, 51.64, 0.1, 3.0),
        ("leo_e0p02", 7500.0, 0.02, 30.0, 0.5, 3.0),
        ("gps", 26_559.7, 0.01, 55.0, 0.05, 0.05),
    ] {
        let kep = KeplerianElements {
            a_km: a,
            e,
            i_rad: f64::to_radians(i_deg),
            arg_perigee_rad: 0.7,
            raan_rad: 1.3,
            mean_anomaly_rad: 2.1,
        };
        let (p, v) = kep.to_cartesian(g.mu_km3_s2).unwrap();
        let initial = CartesianState::new(0.0, p, v);
        let num6 = integrate(&initial, 6, 86_400.0);
        let num5 = integrate(&initial, 5, 86_400.0);
        let mut bl_radial_cross = 0.0f64;
        let mut bl_radial = 0.0f64;
        let mut bl_cross = 0.0f64;
        let mut replay_radial = 0.0f64;
        let eh_prop = EcksteinHechlerPropagator::new(
            &initial,
            &g,
            MeanConversionOptions::ECKSTEIN_HECHLER_DEFAULT,
        )
        .unwrap();
        let bl_prop = BrouwerLyddanePropagator::new(
            &initial,
            &g,
            0.0,
            MeanConversionOptions::BROUWER_LYDDANE_DEFAULT,
        )
        .unwrap();
        let eh = (eh_prop.propagate(86_400.0).unwrap().position_km - num6.position_km).norm();
        let bl = (bl_prop.propagate(86_400.0).unwrap().position_km - num5.position_km).norm();
        for t in [1200.0, 2400.0, 3600.0, 21_600.0, 86_400.0] {
            let truth = integrate(&initial, 5, t);
            let d = rtn(&bl_prop.propagate(t).unwrap(), &truth);
            bl_radial_cross = bl_radial_cross.max(d[0].abs() / 0.2).max(d[2].abs() / 0.1);
            bl_radial = bl_radial.max(d[0].abs());
            bl_cross = bl_cross.max(d[2].abs());
            if let Ok(replay) = orekit_bl::new(
                &initial,
                &g,
                0.0,
                MeanConversionOptions::BROUWER_LYDDANE_DEFAULT,
            ) {
                replay_radial =
                    replay_radial.max(rtn(&replay.propagate(t).unwrap(), &truth)[0].abs());
            }
        }
        println!(
            "{name}: EH {eh:.4} km, BL {bl:.4} km (radial {bl_radial:.4}, cross {bl_cross:.4}), \
             Orekit-arrangement radial {replay_radial:.3} km"
        );
        assert!(eh < eh_bound, "{name}: Eckstein-Hechler {eh} km");
        assert!(bl < bl_bound, "{name}: Brouwer-Lyddane {bl} km");
        assert!(
            bl_radial_cross < 1.0,
            "{name}: Brouwer-Lyddane radial or cross-track"
        );
        if a < 10_000.0 && replay_radial > 0.0 {
            // The Orekit arrangement misses the J2 short-period eccentricity
            // terms by their own size, J2 (Re/a)^2 a, several kilometres in LEO.
            assert!(
                replay_radial > 2.0,
                "{name}: Orekit arrangement radial {replay_radial} km"
            );
        }
    }
}

/// The state of a numerical integration of the zonal field through degree
/// `max_degree` from `initial` to `t`.
fn integrate(
    g: &ZonalHarmonics,
    initial: &CartesianState,
    max_degree: u8,
    t: f64,
) -> CartesianState {
    let zonal = ZonalGravity::new(
        g.mu_km3_s2,
        g.re_km,
        ZonalDegrees::through(max_degree).unwrap(),
        ZonalCoefficients {
            j2: -g.c20,
            j3: -g.c30,
            j4: -g.c40,
            j5: -g.c50,
            j6: -g.c60,
            tide_system: TideSystem::TideFree,
        },
    );
    let components = ForceModelComponents {
        two_body_mu_km3_s2: Some(g.mu_km3_s2),
        zonal: Some(zonal),
        ..ForceModelComponents::EMPTY
    };
    let mut options = IntegratorOptions::default();
    options.abs_tol = 1.0e-12;
    options.rel_tol = 1.0e-13;
    options.max_step = 60.0;
    StatePropagator::new(
        initial.epoch_tdb_seconds,
        initial.position_array(),
        initial.velocity_array(),
        ForceModelKind::composite(components),
        IntegratorKind::Dp54,
    )
    .with_options(options)
    .propagate_to(t)
    .unwrap()
    .final_state
}

/// Brouwer-Lyddane at the equatorial inclinations, prograde and retrograde,
/// against the same integration. Lyddane's variables are regular at i = 0;
/// the retrograde orbits go through the time-reversed evaluation, so they
/// must follow the integration as closely as their prograde mirrors.
#[test]
fn brouwer_lyddane_follows_numerical_integration_near_equatorial() {
    let g = ZonalHarmonics::earth();
    // (a km, e, i deg, bound km)
    for (a, e, i_deg, bound) in [
        (7078.137, 0.0012, 0.1, 3.0),
        (7078.137, 0.0012, 179.9, 3.0),
        (7078.137, 0.0012, 180.0, 3.0),
        (7078.137, 0.0012, 120.0, 3.0),
        (12_000.0, 0.3, 1.0, 3.0),
        (12_000.0, 0.3, 179.0, 3.0),
        (12_000.0, 0.3, 179.5, 3.0),
    ] {
        let mean = KeplerianElements {
            a_km: a,
            e,
            i_rad: f64::to_radians(i_deg),
            arg_perigee_rad: 0.7,
            raan_rad: 1.3,
            mean_anomaly_rad: 2.1,
        };
        let prop = BrouwerLyddanePropagator::from_mean_elements(0.0, mean, &g, 0.0).unwrap();
        let initial = prop.propagate(0.0).unwrap();
        let mut worst = 0.0f64;
        let mut worst_radial_cross = 0.0f64;
        for t in [1200.0, 2400.0, 3600.0, 21_600.0, 86_400.0] {
            let truth = integrate(&g, &initial, 5, t);
            let ours = prop.propagate(t).unwrap();
            worst = worst.max((ours.position_km - truth.position_km).norm());
            let d = rtn(&ours, &truth);
            worst_radial_cross = worst_radial_cross.max(d[0].abs()).max(d[2].abs());
        }
        println!("a {a} e {e} i {i_deg}: {worst:.4} km, radial/cross {worst_radial_cross:.4} km");
        assert!(worst < bound, "a {a} e {e} i {i_deg}: {worst} km");
        assert!(
            worst_radial_cross < 0.2,
            "a {a} e {e} i {i_deg}: radial or cross-track {worst_radial_cross} km"
        );
    }
}
