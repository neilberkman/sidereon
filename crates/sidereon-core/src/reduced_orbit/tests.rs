//! Deterministic correctness tests for the compact mean-element model: a
//! synthetic orbit generated from known elements is recovered by the fitter,
//! evaluation invariants hold, drift against the generating samples is ~0, and
//! every degenerate input maps to its typed error.

use super::*;
use crate::astro::frames::transforms::{
    gcrs_to_itrs_compute, mat3_vec3_mul, teme_to_gcrs_compute, TemeStateKm,
};
use crate::astro::sgp4::{JulianDate, Satellite};
use crate::astro::time::civil::split_julian_date;
use crate::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use crate::sp3::Sp3;
use crate::{GnssSatelliteId, GnssSystem};

/// Base epoch for the synthetic orbits (UTC). Windows stay within the day so the
/// helper below only has to roll seconds into minutes/hours.
fn base_epoch() -> CalendarEpoch {
    CalendarEpoch::new(2020, 6, 24, 0, 0, 0.0)
}

/// `base + dt_s`, valid for `dt_s` up to a day (hour stays < 24 for the windows
/// used here).
fn epoch_at(base: CalendarEpoch, dt_s: f64) -> CalendarEpoch {
    let whole = dt_s.floor() as i64;
    let frac = dt_s - whole as f64;
    let h = base.hour + (whole / 3600) as i32;
    let m = (whole % 3600) / 60;
    let s = (whole % 60) as f64 + frac;
    CalendarEpoch::new(base.year, base.month, base.day, h, m as i32, s)
}

/// Compare two angles modulo 2π.
fn angle_close(a: f64, b: f64, tol: f64) -> bool {
    let two_pi = 2.0 * std::f64::consts::PI;
    let d = ((a - b + std::f64::consts::PI).rem_euclid(two_pi)) - std::f64::consts::PI;
    d.abs() < tol
}

#[test]
fn gpst_epoch_after_positive_leap_uses_true_utc_leap_offset() {
    let expected = crate::astro::time::scales::TimeScales::from_utc(2016, 12, 31, 23, 59, 48.0)
        .expect("valid UTC instant");

    let got = CalendarEpoch::new(2017, 1, 1, 0, 0, 5.0).time_scales(TimeScale::Gpst);

    assert_eq!(
        got, expected,
        "2017-01-01 00:00:05 GPST is 2016-12-31 23:59:48 UTC, not one second early"
    );
}

#[test]
fn gpst_epoch_inside_positive_leap_second_maps_to_utc_leap_label() {
    let expected = crate::astro::time::scales::TimeScales::from_utc(2016, 12, 31, 23, 59, 60.5)
        .expect("valid UTC leap-second instant");

    let got = CalendarEpoch::new(2017, 1, 1, 0, 0, 17.5).time_scales(TimeScale::Gpst);

    assert_eq!(got, expected);
}

#[test]
fn gpst_epoch_away_from_leap_second_keeps_existing_offset() {
    let expected = crate::astro::time::scales::TimeScales::from_utc(2020, 6, 23, 23, 59, 42.0)
        .expect("valid UTC instant");

    let got = CalendarEpoch::new(2020, 6, 24, 0, 0, 0.0).time_scales(TimeScale::Gpst);

    assert_eq!(got, expected);
}

/// Generate ECEF samples from a known `circular_secular` orbit. The generator
/// uses the SAME GCRS->ECEF rotation (`gcrs_to_itrs_matrix`) and the SAME
/// `dt_seconds` the fitter inverts, so a correct fit recovers the parameters to
/// LM tolerance. `raan_rate` is passed in so a test can choose a value far from
/// the J2 seed to prove the rate is genuinely fitted.
fn synth_samples(
    a_km: f64,
    i: f64,
    raan0: f64,
    raan_rate: f64,
    arg_lat0: f64,
    n_samples: usize,
    cadence_s: f64,
) -> ([f64; N_PARAMS], Vec<EcefSample>) {
    let base = base_epoch();
    let t0 = base.time_scales(TimeScale::Utc);
    let n = (MU_EARTH / (a_km * a_km * a_km)).sqrt();
    let params = [a_km, i, raan0, raan_rate, arg_lat0, n];

    let mut samples = Vec::with_capacity(n_samples);
    for k in 0..n_samples {
        let ep = epoch_at(base, k as f64 * cadence_s);
        let ts = ep.time_scales(TimeScale::Utc);
        let dt = dt_seconds(&t0, &ts);
        let r_gcrs = eval_gcrs_km(&params, dt);
        let mat = gcrs_to_itrs_matrix(&ts).expect("valid frame transform");
        let r_itrs = mat3_vec3_mul(&mat, &r_gcrs).expect("finite matrix-vector product");
        samples.push(EcefSample::new(
            ep,
            r_itrs[0] * M_PER_KM,
            r_itrs[1] * M_PER_KM,
            r_itrs[2] * M_PER_KM,
        ));
    }
    (params, samples)
}

#[test]
fn recovers_synthetic_orbit_and_raan_rate_is_fitted() {
    // GPS-like MEO. Choose a nodal rate far larger than the J2 seed so the fit
    // must move off the seed: an exaggerated 1e-5 rad/s (0.36 rad over the 10h
    // window), versus a J2 seed of order 1e-8.
    let a_km = 26_560.0;
    let i = 0.9599; // ~55 deg
    let raan0 = 1.0;
    let arg_lat0 = 0.5;
    let raan_rate = 1.0e-5;
    let (params, samples) = synth_samples(a_km, i, raan0, raan_rate, arg_lat0, 41, 900.0);
    let n_true = params[5];

    let fitted = fit(&samples, TimeScale::Utc).expect("fit should succeed");
    let e = fitted.elements;

    assert!((e.a_m - a_km * M_PER_KM).abs() < 1.0, "a_m = {}", e.a_m);
    assert!((e.i_rad - i).abs() < 1.0e-7, "i = {}", e.i_rad);
    assert!(
        angle_close(e.raan_rad, raan0, 1.0e-7),
        "raan = {}",
        e.raan_rad
    );
    assert!(
        angle_close(e.arg_lat_rad, arg_lat0, 1.0e-7),
        "u = {}",
        e.arg_lat_rad
    );
    assert!(
        (e.mean_motion_rad_s - n_true).abs() < 1.0e-11,
        "n = {}",
        e.mean_motion_rad_s
    );

    // The nodal rate is genuinely fitted to the exaggerated value, NOT pinned to
    // the J2 seed (which is ~1e-8, three orders smaller).
    assert!(
        (e.raan_rate_rad_s - raan_rate).abs() < 1.0e-10,
        "fitted raan_rate = {}",
        e.raan_rate_rad_s
    );
    let j2_seed = raan_rate_j2(n_true, i, a_km);
    assert!((e.raan_rate_j2_rad_s - j2_seed).abs() < 1.0e-18);
    assert!(
        (e.raan_rate_rad_s - e.raan_rate_j2_rad_s).abs() > 1.0e-6,
        "fitted rate must differ from the J2 seed"
    );

    // The fit reproduces the samples to well under a millimetre.
    assert!(
        fitted.stats.rms_m < 1.0e-3,
        "rms_m = {}",
        fitted.stats.rms_m
    );
    assert!(
        fitted.stats.max_m < 1.0e-2,
        "max_m = {}",
        fitted.stats.max_m
    );
    assert_eq!(fitted.stats.n_samples, 41);
}

#[test]
fn fit_is_independent_of_sample_order() {
    // A shuffled (non-monotonic) sample list fits the identical model: the fitter
    // orders by time, so the earliest epoch is always t0 and the plane seed is
    // unaffected by the caller's ordering.
    let (_p, samples) = synth_samples(26_560.0, 0.9599, 1.0, 1.0e-5, 0.5, 21, 900.0);
    let mut shuffled = samples.clone();
    shuffled.reverse();
    shuffled.rotate_left(7);

    let a = fit(&samples, TimeScale::Utc).expect("fit ordered").elements;
    let b = fit(&shuffled, TimeScale::Utc)
        .expect("fit shuffled")
        .elements;

    assert_eq!(a.epoch, b.epoch);
    assert!((a.a_m - b.a_m).abs() < 1.0e-6, "a {} vs {}", a.a_m, b.a_m);
    assert!((a.i_rad - b.i_rad).abs() < 1.0e-12);
    assert!((a.raan_rate_rad_s - b.raan_rate_rad_s).abs() < 1.0e-15);
    assert!((a.arg_lat_rad - b.arg_lat_rad).abs() < 1.0e-12);
}

#[test]
fn gcrs_position_has_constant_radius() {
    // A circular orbit keeps |r| == a in the inertial frame at every epoch.
    let a_km = 26_560.0;
    let i = 0.9599;
    let (_p, samples) = synth_samples(a_km, i, 1.0, 0.0, 0.5, 8, 1800.0);
    let fitted = fit(&samples, TimeScale::Utc).expect("fit");
    for k in 0..12 {
        let ep = epoch_at(base_epoch(), k as f64 * 1200.0);
        let r = position(&fitted.elements, ep, TimeScale::Utc, Frame::Gcrs)
            .expect("valid reduced-orbit position");
        let radius = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        assert!(
            (radius - fitted.elements.a_m).abs() < 1.0e-3,
            "radius {} vs a {}",
            radius,
            fitted.elements.a_m
        );
    }
}

#[test]
fn ecef_velocity_matches_finite_difference() {
    // The analytic ECEF velocity (with the Earth-rotation transport term) agrees
    // with a central finite difference of the ECEF position.
    let (_p, samples) = synth_samples(26_560.0, 0.9599, 1.0, 0.0, 0.5, 8, 1800.0);
    let e = fit(&samples, TimeScale::Utc).expect("fit").elements;

    let t = 3000.0;
    let h = 0.5;
    let mid = epoch_at(base_epoch(), t);
    let (_r, v) = position_velocity(&e, mid, TimeScale::Utc, Frame::Ecef)
        .expect("valid reduced-orbit position/velocity");
    let rp = position(
        &e,
        epoch_at(base_epoch(), t + h),
        TimeScale::Utc,
        Frame::Ecef,
    )
    .expect("valid reduced-orbit position");
    let rm = position(
        &e,
        epoch_at(base_epoch(), t - h),
        TimeScale::Utc,
        Frame::Ecef,
    )
    .expect("valid reduced-orbit position");
    for c in 0..3 {
        let fd = (rp[c] - rm[c]) / (2.0 * h);
        assert!(
            (v[c] - fd).abs() < 1.0e-3,
            "axis {}: v {} vs fd {}",
            c,
            v[c],
            fd
        );
    }
}

#[test]
fn drift_against_generating_samples_is_near_zero() {
    let (_p, samples) = synth_samples(26_560.0, 0.9599, 1.0, 1.0e-6, 0.5, 41, 900.0);
    let fitted = fit(&samples, TimeScale::Utc).expect("fit");

    let report =
        drift(&fitted.elements, &samples, TimeScale::Utc, 100.0).expect("valid drift report");
    assert_eq!(report.per_epoch.len(), samples.len());
    assert!(report.max_m < 1.0e-2, "max_m = {}", report.max_m);
    assert!(report.rms_m < 1.0e-2, "rms_m = {}", report.rms_m);
    // The error never approaches 100 m, so there is no threshold crossing.
    assert!(report.threshold_horizon.is_none());
}

#[test]
fn drift_reports_threshold_crossing() {
    let (_p, samples) = synth_samples(26_560.0, 0.9599, 1.0, 0.0, 0.5, 41, 900.0);
    let fitted = fit(&samples, TimeScale::Utc).expect("fit");
    // A sub-nanometre threshold is crossed at the very first sample.
    let report =
        drift(&fitted.elements, &samples, TimeScale::Utc, 1.0e-12).expect("valid drift report");
    assert_eq!(report.threshold_horizon, Some(samples[0].epoch));
}

#[test]
fn piecewise_segment_selection_rejects_unsorted_segments() {
    let (_p, samples) = synth_samples(26_560.0, 0.9599, 1.0, 0.0, 0.5, 8, 900.0);
    let orbit = fit(&samples, TimeScale::Utc).expect("fit");
    let base = base_epoch();
    let first = PiecewiseSegment {
        t0: base,
        t1: epoch_at(base, 1800.0),
        orbit,
    };
    let second = PiecewiseSegment {
        t0: epoch_at(base, 1800.0),
        t1: epoch_at(base, 3600.0),
        orbit,
    };
    let piecewise = PiecewiseOrbit {
        model: Model::CircularSecular,
        t0: base,
        t1: second.t1,
        segment_s: 1800,
        segments: vec![second, first],
    };

    assert!(matches!(
        select_piecewise_segment(&piecewise, epoch_at(base, 900.0)),
        Err(PiecewiseOrbitError::Reduced(
            ReducedOrbitError::InvalidInput {
                field: "piecewise.segments.t0",
                reason: "must be strictly increasing",
            }
        ))
    ));
}

#[test]
fn public_evaluation_rejects_invalid_elements_and_epochs() {
    let (_p, samples) = synth_samples(26_560.0, 0.9599, 1.0, 0.0, 0.5, 8, 1800.0);
    let mut elements = fit(&samples, TimeScale::Utc).expect("fit").elements;
    elements.a_m = f64::NAN;
    assert!(matches!(
        position(&elements, base_epoch(), TimeScale::Utc, Frame::Gcrs),
        Err(ReducedOrbitError::InvalidInput {
            field: "elements.a_m",
            ..
        })
    ));

    elements.a_m = 26_560_000.0;
    elements.e = 1.0;
    assert!(matches!(
        position_velocity(&elements, base_epoch(), TimeScale::Utc, Frame::Ecef),
        Err(ReducedOrbitError::InvalidInput {
            field: "elements.e",
            ..
        })
    ));

    elements.e = 0.0;
    let bad_epoch = CalendarEpoch::new(2020, 6, 24, 0, 0, f64::NAN);
    assert!(matches!(
        position(&elements, bad_epoch, TimeScale::Utc, Frame::Gcrs),
        Err(ReducedOrbitError::InvalidInput { field: "epoch", .. })
    ));
}

#[test]
fn drift_rejects_invalid_threshold_and_truth_samples() {
    let (_p, mut samples) = synth_samples(26_560.0, 0.9599, 1.0, 0.0, 0.5, 8, 1800.0);
    let elements = fit(&samples, TimeScale::Utc).expect("fit").elements;
    assert!(matches!(
        drift(&elements, &samples, TimeScale::Utc, f64::INFINITY),
        Err(ReducedOrbitError::InvalidInput {
            field: "threshold_m",
            ..
        })
    ));

    samples[0].x_m = f64::NAN;
    assert!(matches!(
        drift(&elements, &samples, TimeScale::Utc, 100.0),
        Err(ReducedOrbitError::InvalidInput {
            field: "truth.x_m",
            ..
        })
    ));

    let (_p, mut samples) = synth_samples(26_560.0, 0.9599, 1.0, 0.0, 0.5, 8, 1800.0);
    samples[0].x_m = f64::MAX;
    assert!(matches!(
        drift(&elements, &samples, TimeScale::Utc, 100.0),
        Err(ReducedOrbitError::InvalidInput {
            field: "drift.error_m",
            ..
        })
    ));
}

#[test]
fn too_few_samples_is_typed() {
    let (_p, samples) = synth_samples(26_560.0, 0.9599, 1.0, 0.0, 0.5, 3, 900.0);
    assert!(matches!(
        fit(&samples, TimeScale::Utc),
        Err(ReducedOrbitError::TooFewSamples {
            got: 3,
            required: 4
        })
    ));
}

#[test]
fn fit_rejects_invalid_sample_epoch_without_panic() {
    let (_p, mut samples) = synth_samples(26_560.0, 0.9599, 1.0, 0.0, 0.5, 4, 900.0);
    samples[1].epoch = CalendarEpoch::new(2020, 2, 30, 0, 0, 0.0);

    assert!(matches!(
        fit(&samples, TimeScale::Utc),
        Err(ReducedOrbitError::InvalidInput { field: "epoch", .. })
    ));
}

#[test]
fn fit_rejects_invalid_sample_coordinates_without_panic() {
    let (_p, mut samples) = synth_samples(26_560.0, 0.9599, 1.0, 0.0, 0.5, 4, 900.0);
    samples[1].x_m = f64::NAN;

    assert!(matches!(
        fit(&samples, TimeScale::Utc),
        Err(ReducedOrbitError::InvalidInput {
            field: "sample.x_m",
            ..
        })
    ));
    assert!(matches!(
        fit_with_model(&samples, TimeScale::Utc, Model::EccentricSecular),
        Err(ReducedOrbitError::InvalidInput {
            field: "sample.x_m",
            ..
        })
    ));
}

#[test]
fn single_epoch_window_is_invalid() {
    // Four samples that all share one epoch span no time.
    let ep = base_epoch();
    let (_p, good) = synth_samples(26_560.0, 0.9599, 1.0, 0.0, 0.5, 4, 900.0);
    let stacked: Vec<EcefSample> = good
        .iter()
        .map(|s| EcefSample::new(ep, s.x_m, s.y_m, s.z_m))
        .collect();
    assert!(matches!(
        fit(&stacked, TimeScale::Utc),
        Err(ReducedOrbitError::InvalidWindow)
    ));
}

#[test]
fn collinear_gcrs_samples_are_singular_plane() {
    // Radially aligned positions have vanishing r_k x r_{k+1}: no orbital plane.
    let s = vec![
        GcrsSample {
            dt: 0.0,
            r_km: [7000.0, 0.0, 0.0],
        },
        GcrsSample {
            dt: 1.0,
            r_km: [7100.0, 0.0, 0.0],
        },
        GcrsSample {
            dt: 2.0,
            r_km: [7200.0, 0.0, 0.0],
        },
        GcrsSample {
            dt: 3.0,
            r_km: [7300.0, 0.0, 0.0],
        },
    ];
    assert!(matches!(
        seed_params(&s),
        Err(ReducedOrbitError::SingularPlaneFit)
    ));
}

#[test]
fn near_equatorial_samples_are_raan_ambiguous() {
    // A near-equatorial circle: the normal is ~+z, inclination ~7e-5 rad, so the
    // ascending node (and RAAN) is undefined.
    let s = vec![
        GcrsSample {
            dt: 0.0,
            r_km: [7000.0, 0.0, 0.5],
        },
        GcrsSample {
            dt: 1.0,
            r_km: [0.0, 7000.0, 0.5],
        },
        GcrsSample {
            dt: 2.0,
            r_km: [-7000.0, 0.0, 0.5],
        },
        GcrsSample {
            dt: 3.0,
            r_km: [0.0, -7000.0, 0.5],
        },
    ];
    assert!(matches!(
        seed_params(&s),
        Err(ReducedOrbitError::RaanAmbiguous)
    ));
}

// -------------------------------------------------------------------------
// Eccentric model.
// -------------------------------------------------------------------------

/// Generate ECEF samples for a known `eccentric_secular` orbit given `e` and
/// `omega`. Uses the SAME eccentric GCRS eval and GCRS->ITRS rotation the fitter
/// inverts, so a correct fit recovers `(a, i, raan, raan_rate, h, k, L0, n)` to
/// LM tolerance.
/// Keplerian elements for a synthetic `eccentric_secular` arc: semi-major axis,
/// eccentricity, inclination, ascending-node longitude and its rate, argument of
/// perigee, and mean anomaly at epoch.
struct EccOrbit {
    a_km: f64,
    e: f64,
    i: f64,
    raan0: f64,
    raan_rate: f64,
    omega: f64,
    big_m0: f64,
}

/// The GPS-like MEO eccentric orbit (e ~ 0.02, no nodal drift) several tests
/// reuse to generate an arc and then vary only the sample count and cadence.
fn gps_meo_ecc_orbit() -> EccOrbit {
    EccOrbit {
        a_km: 26_560.0,
        e: 0.02,
        i: 0.9599,
        raan0: 1.0,
        raan_rate: 0.0,
        omega: 0.7,
        big_m0: 0.3,
    }
}

fn synth_ecc(
    orbit: EccOrbit,
    n_samples: usize,
    cadence_s: f64,
) -> ([f64; N_PARAMS_ECC], Vec<EcefSample>) {
    let EccOrbit {
        a_km,
        e,
        i,
        raan0,
        raan_rate,
        omega,
        big_m0,
    } = orbit;
    let base = base_epoch();
    let t0 = base.time_scales(TimeScale::Utc);
    let n = (MU_EARTH / (a_km * a_km * a_km)).sqrt();
    let h = e * omega.sin();
    let k = e * omega.cos();
    let l0 = omega + big_m0;
    let params = [a_km, i, raan0, raan_rate, h, k, l0, n];

    let mut samples = Vec::with_capacity(n_samples);
    for s in 0..n_samples {
        let ep = epoch_at(base, s as f64 * cadence_s);
        let ts = ep.time_scales(TimeScale::Utc);
        let dt = dt_seconds(&t0, &ts);
        let r_gcrs = eval_gcrs_km_ecc(&params, dt);
        let mat = gcrs_to_itrs_matrix(&ts).expect("valid frame transform");
        let r_itrs = mat3_vec3_mul(&mat, &r_gcrs).expect("finite matrix-vector product");
        samples.push(EcefSample::new(
            ep,
            r_itrs[0] * M_PER_KM,
            r_itrs[1] * M_PER_KM,
            r_itrs[2] * M_PER_KM,
        ));
    }
    (params, samples)
}

#[test]
fn recovers_synthetic_eccentric_orbit() {
    // A BeiDou-like e ~ 0.01 and a GPS-like e ~ 0.02, both recovered tightly.
    for (e_true, omega_true) in [(0.01_f64, 0.7_f64), (0.02_f64, -1.3_f64)] {
        let a_km = 26_560.0;
        let i = 0.9599;
        let raan0 = 1.0;
        let raan_rate = 1.0e-6;
        let big_m0 = 0.3;
        let (params, samples) = synth_ecc(
            EccOrbit {
                a_km,
                e: e_true,
                i,
                raan0,
                raan_rate,
                omega: omega_true,
                big_m0,
            },
            41,
            900.0,
        );

        let fitted = fit_with_model(&samples, TimeScale::Utc, Model::EccentricSecular)
            .expect("eccentric fit should succeed");
        let el = fitted.elements;

        assert_eq!(el.model, Model::EccentricSecular);
        assert!((el.a_m - a_km * M_PER_KM).abs() < 5.0, "a_m = {}", el.a_m);
        assert!((el.i_rad - i).abs() < 1.0e-7, "i = {}", el.i_rad);
        assert!(
            angle_close(el.raan_rad, raan0, 1.0e-7),
            "raan = {}",
            el.raan_rad
        );
        assert!(
            (el.e - e_true).abs() < 1.0e-6,
            "e = {} (want {})",
            el.e,
            e_true
        );
        assert!(
            angle_close(el.arg_perigee_rad, omega_true, 1.0e-5),
            "omega = {} (want {})",
            el.arg_perigee_rad,
            omega_true
        );
        assert!(
            (el.mean_motion_rad_s - params[7]).abs() < 1.0e-11,
            "n = {}",
            el.mean_motion_rad_s
        );
        assert!(
            fitted.stats.rms_m < 1.0e-2,
            "rms_m = {}",
            fitted.stats.rms_m
        );
    }
}

#[test]
fn eccentric_fit_reduces_to_circular_at_zero_e() {
    // Fed perfectly circular samples, the eccentric model recovers e ~ 0 and its
    // positions match the circular fit to well under a millimetre.
    let a_km = 29_600.0;
    let i = 0.9774;
    let (_p, samples) = synth_samples(a_km, i, 1.0, 1.0e-6, 0.5, 25, 900.0);

    let circ = fit(&samples, TimeScale::Utc).expect("circular fit");
    let ecc =
        fit_with_model(&samples, TimeScale::Utc, Model::EccentricSecular).expect("eccentric fit");

    assert!(ecc.elements.e < 1.0e-6, "recovered e = {}", ecc.elements.e);
    assert!(ecc.elements.h.abs() < 1.0e-6 && ecc.elements.k.abs() < 1.0e-6);

    for s in 0..30 {
        let ep = epoch_at(base_epoch(), s as f64 * 800.0);
        let rc = position(&circ.elements, ep, TimeScale::Utc, Frame::Gcrs)
            .expect("valid reduced-orbit position");
        let re = position(&ecc.elements, ep, TimeScale::Utc, Frame::Gcrs)
            .expect("valid reduced-orbit position");
        let d =
            ((rc[0] - re[0]).powi(2) + (rc[1] - re[1]).powi(2) + (rc[2] - re[2]).powi(2)).sqrt();
        assert!(
            d < 1.0e-3,
            "circular-vs-eccentric position differs by {} m",
            d
        );
    }
}

#[test]
fn eccentric_velocity_matches_finite_difference() {
    let (_p, samples) = synth_ecc(gps_meo_ecc_orbit(), 25, 900.0);
    let e = fit_with_model(&samples, TimeScale::Utc, Model::EccentricSecular)
        .expect("fit")
        .elements;

    let t = 3000.0;
    let h = 0.5;
    let mid = epoch_at(base_epoch(), t);
    let (_r, v) = position_velocity(&e, mid, TimeScale::Utc, Frame::Ecef)
        .expect("valid reduced-orbit position/velocity");
    let rp = position(
        &e,
        epoch_at(base_epoch(), t + h),
        TimeScale::Utc,
        Frame::Ecef,
    )
    .expect("valid reduced-orbit position");
    let rm = position(
        &e,
        epoch_at(base_epoch(), t - h),
        TimeScale::Utc,
        Frame::Ecef,
    )
    .expect("valid reduced-orbit position");
    for c in 0..3 {
        let fd = (rp[c] - rm[c]) / (2.0 * h);
        assert!(
            (v[c] - fd).abs() < 1.0e-3,
            "axis {}: v {} vs fd {}",
            c,
            v[c],
            fd
        );
    }
}

#[test]
fn eccentric_gcrs_radius_varies_with_anomaly() {
    // Unlike the circular model, |r| sweeps between a(1-e) and a(1+e).
    let (_p, samples) = synth_ecc(gps_meo_ecc_orbit(), 25, 900.0);
    let el = fit_with_model(&samples, TimeScale::Utc, Model::EccentricSecular)
        .expect("fit")
        .elements;
    let mut rmin = f64::INFINITY;
    let mut rmax = 0.0_f64;
    // Sample across roughly a full orbital period.
    let period = 2.0 * std::f64::consts::PI / el.mean_motion_rad_s;
    for s in 0..60 {
        let ep = epoch_at(base_epoch(), s as f64 * period / 60.0);
        let r =
            position(&el, ep, TimeScale::Utc, Frame::Gcrs).expect("valid reduced-orbit position");
        let rad = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        rmin = rmin.min(rad);
        rmax = rmax.max(rad);
    }
    let a = el.a_m;
    assert!(
        (rmax - rmin) > 0.5 * a * el.e,
        "radius did not vary: {} to {}",
        rmin,
        rmax
    );
    assert!(rmin > a * (1.0 - el.e) - 1.0e3 && rmax < a * (1.0 + el.e) + 1.0e3);
}

#[test]
fn eccentric_too_few_samples_is_typed() {
    let (_p, samples) = synth_ecc(gps_meo_ecc_orbit(), 3, 900.0);
    assert!(matches!(
        fit_with_model(&samples, TimeScale::Utc, Model::EccentricSecular),
        Err(ReducedOrbitError::TooFewSamples {
            got: 3,
            required: 4
        })
    ));
}

#[test]
fn eccentric_fit_rejects_invalid_sample_epoch_without_panic() {
    let (_p, mut samples) = synth_ecc(gps_meo_ecc_orbit(), 4, 900.0);
    samples[2].epoch = CalendarEpoch::new(2020, 6, 24, 0, 0, 61.0);

    assert!(matches!(
        fit_with_model(&samples, TimeScale::Utc, Model::EccentricSecular),
        Err(ReducedOrbitError::InvalidInput { field: "epoch", .. })
    ));
}

#[test]
fn eccentric_single_epoch_window_is_invalid() {
    let ep = base_epoch();
    let (_p, good) = synth_ecc(gps_meo_ecc_orbit(), 4, 900.0);
    let stacked: Vec<EcefSample> = good
        .iter()
        .map(|s| EcefSample::new(ep, s.x_m, s.y_m, s.z_m))
        .collect();
    assert!(matches!(
        fit_with_model(&stacked, TimeScale::Utc, Model::EccentricSecular),
        Err(ReducedOrbitError::InvalidWindow)
    ));
}

#[test]
fn eccentric_dramatically_beats_circular_on_gps_sp3() {
    // The headline source-backed gate: GPS G21 (e ~ 0.024, a*e ~ 634 km radial
    // signal) is the hardest case for the circular model. Fit both models over
    // six hours and drift over the full day; the eccentric model must recover the
    // radial signal the circular model cannot.
    let (sp3, node) = gps_g21_nodes();

    let fit_samples: Vec<EcefSample> = (0..25).filter_map(&node).collect();
    assert!(fit_samples.len() >= 20, "got {} nodes", fit_samples.len());

    let circ = fit(&fit_samples, TimeScale::Gpst).expect("circular fit");
    let ecc = fit_with_model(&fit_samples, TimeScale::Gpst, Model::EccentricSecular)
        .expect("eccentric fit");

    // The eccentric fit recovers a real GPS eccentricity.
    assert!(
        ecc.elements.e > 0.015 && ecc.elements.e < 0.035,
        "recovered e = {}",
        ecc.elements.e
    );

    let truth: Vec<EcefSample> = (0..96).step_by(2).filter_map(&node).collect();
    let dc = drift(&circ.elements, &truth, TimeScale::Gpst, 1.0e9).expect("valid drift report");
    let de = drift(&ecc.elements, &truth, TimeScale::Gpst, 1.0e9).expect("valid drift report");

    // Real bounds: the circular model leaves tens of km of unmodelled a*e signal;
    // the eccentric model brings it down to a few km, dramatically better.
    assert!(dc.max_m > 50_000.0, "circular drift max {} m", dc.max_m);
    assert!(de.max_m < 10_000.0, "eccentric drift max {} m", de.max_m);
    assert!(
        de.max_m < dc.max_m / 5.0,
        "eccentric {} not <<  circular {}",
        de.max_m,
        dc.max_m
    );
    assert!(sp3.header.time_scale == TimeScale::Gpst);
}

#[test]
fn eccentric_is_comparable_to_circular_on_galileo_sp3() {
    // The nonsingular guarantee end-to-end: on near-circular Galileo E01 the
    // eccentric model must not regress relative to the circular model.
    let (_sp3, node) = galileo_e01_nodes();

    let fit_samples: Vec<EcefSample> = (0..25).filter_map(&node).collect();
    let circ = fit(&fit_samples, TimeScale::Gpst).expect("circular fit");
    let ecc = fit_with_model(&fit_samples, TimeScale::Gpst, Model::EccentricSecular)
        .expect("eccentric fit");

    // Galileo is genuinely near-circular: the recovered e is tiny.
    assert!(ecc.elements.e < 1.0e-3, "Galileo e = {}", ecc.elements.e);

    let truth: Vec<EcefSample> = (0..96).step_by(2).filter_map(&node).collect();
    let dc = drift(&circ.elements, &truth, TimeScale::Gpst, 1.0e9).expect("valid drift report");
    let de = drift(&ecc.elements, &truth, TimeScale::Gpst, 1.0e9).expect("valid drift report");

    assert!(de.max_m < 20_000.0, "eccentric drift max {} m", de.max_m);
    // Comparable: a near-circular orbit must not regress under the eccentric
    // model. A tight relative bound (not a fixed multi-km slack) so a genuine
    // regression on this small-error case would be caught.
    assert!(
        de.max_m < dc.max_m * 1.5,
        "eccentric {} regressed vs circular {}",
        de.max_m,
        dc.max_m
    );
}

/// Load the SP3 fixture and return a node accessor for GPS G21.
fn gps_g21_nodes() -> (Sp3, impl Fn(usize) -> Option<EcefSample>) {
    sp3_nodes_for(GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id"))
}

/// Load the SP3 fixture and return a node accessor for Galileo E01.
fn galileo_e01_nodes() -> (Sp3, impl Fn(usize) -> Option<EcefSample>) {
    sp3_nodes_for(GnssSatelliteId::new(GnssSystem::Galileo, 1).expect("valid satellite id"))
}

fn sp3_nodes_for(sat: GnssSatelliteId) -> (Sp3, impl Fn(usize) -> Option<EcefSample>) {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
    ))
    .expect("read SP3 fixture");
    let sp3 = Sp3::parse(&bytes).expect("parse SP3");
    let interval = sp3.header.epoch_interval_s;
    let start = CalendarEpoch::new(2020, 6, 24, 0, 0, 0.0);
    // Clone the small handle into the closure so the returned closure is 'static
    // with respect to the borrow; Sp3 is returned separately for header checks.
    let sp3_for_closure = sp3.clone();
    let node = move |idx: usize| -> Option<EcefSample> {
        let st = sp3_for_closure.state(sat, idx).ok()?;
        Some(EcefSample::new(
            epoch_at(start, idx as f64 * interval),
            st.position.x_m,
            st.position.y_m,
            st.position.z_m,
        ))
    };
    (sp3, node)
}

fn instant_for_epoch(epoch: CalendarEpoch, scale: TimeScale) -> Instant {
    let (jd_whole, fraction) = split_julian_date(
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second,
    );
    let split = JulianDateSplit::new(jd_whole, fraction).expect("valid split JD");
    Instant::from_julian_date(scale, split)
}

fn manual_sp3_samples(
    sp3: &Sp3,
    sat: GnssSatelliteId,
    start: CalendarEpoch,
    count: usize,
    cadence_s: f64,
) -> Vec<EcefSample> {
    (0..=count)
        .filter_map(|idx| {
            let epoch = epoch_at(start, idx as f64 * cadence_s);
            let instant = instant_for_epoch(epoch, sp3.header.time_scale);
            let state = sp3.position(sat, instant).ok()?;
            Some(EcefSample::new(
                epoch,
                state.position.x_m,
                state.position.y_m,
                state.position.z_m,
            ))
        })
        .collect()
}

fn manual_sgp4_samples(
    satellite: &Satellite,
    start: CalendarEpoch,
    count: usize,
    cadence_s: f64,
) -> Vec<EcefSample> {
    (0..=count)
        .filter_map(|idx| {
            let epoch = epoch_at(start, idx as f64 * cadence_s);
            let (jd_whole, fraction) = split_julian_date(
                epoch.year,
                epoch.month,
                epoch.day,
                epoch.hour,
                epoch.minute,
                epoch.second,
            );
            let prediction = satellite
                .propagate_jd(JulianDate(jd_whole, fraction))
                .ok()?;
            let ts = epoch.time_scales(TimeScale::Utc);
            let (gcrs, _) = teme_to_gcrs_compute(
                &TemeStateKm {
                    position_km: prediction.position,
                    velocity_km_s: prediction.velocity,
                },
                &ts,
                false,
            )
            .ok()?;
            let (x_km, y_km, z_km) =
                gcrs_to_itrs_compute(gcrs.0, gcrs.1, gcrs.2, &ts, false).ok()?;
            Some(EcefSample::new(
                epoch,
                x_km * 1000.0,
                y_km * 1000.0,
                z_km * 1000.0,
            ))
        })
        .collect()
}

#[test]
fn source_backed_sp3_fit_drift_and_piecewise_match_manual_composition() {
    let (sp3, _node) = galileo_e01_nodes();
    let sat = GnssSatelliteId::new(GnssSystem::Galileo, 1).expect("valid satellite id");
    let start = CalendarEpoch::new(2020, 6, 24, 0, 0, 0.0);
    let fit_end = CalendarEpoch::new(2020, 6, 24, 6, 0, 0.0);
    let drift_end = CalendarEpoch::new(2020, 6, 24, 12, 0, 0.0);
    let fit_sampling = ReducedOrbitSourceSampling::new(start, fit_end, 1800.0);
    let drift_sampling = ReducedOrbitSourceSampling::new(start, drift_end, 1800.0);
    let source = ReducedOrbitSource::Sp3 {
        product: &sp3,
        satellite: sat,
    };

    let fit_driver = fit_reduced_orbit_source(
        source,
        ReducedOrbitSourceFitOptions {
            sampling: fit_sampling,
            model: Model::EccentricSecular,
        },
    )
    .expect("source fit");
    let fit_samples = manual_sp3_samples(&sp3, sat, start, 12, 1800.0);
    let fit_manual =
        fit_with_model(&fit_samples, TimeScale::Gpst, Model::EccentricSecular).expect("manual fit");
    assert_eq!(fit_driver.requested_samples, 13);
    assert_eq!(fit_driver.orbit, fit_manual);

    let drift_driver = drift_reduced_orbit_source(
        &fit_driver.orbit.elements,
        source,
        ReducedOrbitSourceDriftOptions {
            sampling: drift_sampling,
            threshold_m: 1.0e9,
        },
    )
    .expect("source drift");
    let drift_samples = manual_sp3_samples(&sp3, sat, start, 24, 1800.0);
    let drift_manual = drift(
        &fit_driver.orbit.elements,
        &drift_samples,
        TimeScale::Gpst,
        1.0e9,
    )
    .expect("manual drift");
    assert_eq!(drift_driver.requested_samples, 25);
    assert_eq!(drift_driver.report, drift_manual);

    let piecewise_driver = fit_piecewise_reduced_orbit_source(
        source,
        PiecewiseOrbitSourceFitOptions {
            sampling: drift_sampling,
            model: Model::EccentricSecular,
            segment_s: 7200.0,
        },
    )
    .expect("source piecewise fit");
    let piecewise_manual = fit_piecewise(
        &drift_samples,
        TimeScale::Gpst,
        Model::EccentricSecular,
        start,
        drift_end,
        7200,
    )
    .expect("manual piecewise fit");
    assert_eq!(piecewise_driver.requested_samples, 25);
    assert_eq!(piecewise_driver.orbit, piecewise_manual);

    let piecewise_drift_driver = drift_piecewise_reduced_orbit_source(
        &piecewise_driver.orbit,
        source,
        ReducedOrbitSourceDriftOptions {
            sampling: drift_sampling,
            threshold_m: 1.0e9,
        },
    )
    .expect("source piecewise drift");
    let piecewise_drift_manual = piecewise_drift(
        &piecewise_driver.orbit,
        &drift_samples,
        TimeScale::Gpst,
        1.0e9,
    )
    .expect("manual piecewise drift");
    assert_eq!(piecewise_drift_driver.requested_samples, 25);
    assert_eq!(piecewise_drift_driver.report, piecewise_drift_manual);
}

#[test]
fn source_backed_sgp4_fit_matches_manual_sampling_and_fit() {
    const ISS_L1: &str = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993";
    const ISS_L2: &str = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106";
    let satellite = Satellite::from_tle(ISS_L1, ISS_L2).expect("ISS TLE parses");
    let start = CalendarEpoch::new(2018, 7, 4, 0, 0, 0.0);
    let end = CalendarEpoch::new(2018, 7, 4, 4, 0, 0.0);
    let sampling = ReducedOrbitSourceSampling::new(start, end, 600.0);

    let driver = fit_reduced_orbit_source(
        ReducedOrbitSource::Sgp4 {
            satellite: &satellite,
        },
        ReducedOrbitSourceFitOptions {
            sampling,
            model: Model::EccentricSecular,
        },
    )
    .expect("SGP4 source fit");
    let samples = manual_sgp4_samples(&satellite, start, 24, 600.0);
    let manual =
        fit_with_model(&samples, TimeScale::Utc, Model::EccentricSecular).expect("manual fit");

    assert_eq!(driver.requested_samples, 25);
    assert_eq!(driver.orbit, manual);
}

#[test]
fn fits_a_real_galileo_sp3_track_within_a_few_km() {
    // A genuine source-backed gate (not self-consistency): fit and drift against
    // real precise-ephemeris nodes for a near-circular Galileo satellite, with
    // the product's own GPST scale.
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
    ))
    .expect("read SP3 fixture");
    let sp3 = Sp3::parse(&bytes).expect("parse SP3");
    assert_eq!(sp3.header.time_scale, TimeScale::Gpst);

    // 15-minute nodes from 2020-06-24 00:00:00 GPST (filename DOY 176).
    let interval = sp3.header.epoch_interval_s;
    let start = CalendarEpoch::new(2020, 6, 24, 0, 0, 0.0);
    let sat = GnssSatelliteId::new(GnssSystem::Galileo, 1).expect("valid satellite id"); // E01, near-circular

    let node = |idx: usize| -> Option<EcefSample> {
        let st = sp3.state(sat, idx).ok()?;
        Some(EcefSample::new(
            epoch_at(start, idx as f64 * interval),
            st.position.x_m,
            st.position.y_m,
            st.position.z_m,
        ))
    };

    // Fit the first six hours (25 nodes).
    let fit_samples: Vec<EcefSample> = (0..25).filter_map(node).collect();
    assert!(fit_samples.len() >= 20, "got {} nodes", fit_samples.len());
    let fitted = fit(&fit_samples, TimeScale::Gpst).expect("fit");

    // Near-circular Galileo: a few-km in-window residual and the ~29 600 km axis.
    assert!(fitted.stats.rms_m < 5_000.0, "rms {}", fitted.stats.rms_m);
    assert!(
        (fitted.elements.a_m - 29_600_000.0).abs() < 200_000.0,
        "a {}",
        fitted.elements.a_m
    );

    // Extrapolated drift against the rest of the day's nodes stays bounded.
    let truth: Vec<EcefSample> = (0..96).step_by(2).filter_map(node).collect();
    let report =
        drift(&fitted.elements, &truth, TimeScale::Gpst, 100_000.0).expect("valid drift report");
    assert!(report.max_m < 20_000.0, "drift max {}", report.max_m);
    assert!(report.threshold_horizon.is_none());
}

#[test]
fn eccentric_beats_circular_on_beidou_meo_and_igso_sp3() {
    // BeiDou source-backed gate. The GRG product carries no BeiDou, so a trimmed
    // GBM MGEX product is used: C21 is a MEO (e ~ 9e-4), C08 an IGSO (e ~ 5e-3).
    // Even at these small eccentricities the unmodelled a*e radial signal makes
    // the circular model drift hundreds-to-thousands of km over a day, which the
    // eccentric model recovers to a few km.
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sp3/GBM_BDS_C21_C08_trim.sp3"
    ))
    .expect("read BeiDou SP3 fixture");
    let sp3 = Sp3::parse(&bytes).expect("parse BeiDou SP3");
    assert_eq!(sp3.header.time_scale, TimeScale::Gpst);

    let interval = sp3.header.epoch_interval_s;
    let start = CalendarEpoch::new(2020, 6, 25, 0, 0, 0.0);

    for prn in [21u8, 8u8] {
        let sat = GnssSatelliteId::new(GnssSystem::BeiDou, prn).expect("valid satellite id");
        let node = |idx: usize| -> Option<EcefSample> {
            let st = sp3.state(sat, idx).ok()?;
            Some(EcefSample::new(
                epoch_at(start, idx as f64 * interval),
                st.position.x_m,
                st.position.y_m,
                st.position.z_m,
            ))
        };

        // Fit the first six hours (5-minute nodes), drift over the full day.
        let fit_samples: Vec<EcefSample> = (0..72).filter_map(&node).collect();
        assert!(
            fit_samples.len() >= 60,
            "C{prn}: only {} nodes",
            fit_samples.len()
        );
        let circ = fit(&fit_samples, TimeScale::Gpst).expect("circular fit");
        let ecc = fit_with_model(&fit_samples, TimeScale::Gpst, Model::EccentricSecular)
            .expect("eccentric fit");

        let truth: Vec<EcefSample> = (0..288).step_by(4).filter_map(&node).collect();
        let dc = drift(&circ.elements, &truth, TimeScale::Gpst, 1.0e9).expect("valid drift report");
        let de = drift(&ecc.elements, &truth, TimeScale::Gpst, 1.0e9).expect("valid drift report");

        assert!(
            dc.max_m > 50_000.0,
            "C{prn} circular drift max {} m",
            dc.max_m
        );
        assert!(
            de.max_m < 20_000.0,
            "C{prn} eccentric drift max {} m",
            de.max_m
        );
        assert!(
            de.max_m < dc.max_m / 5.0,
            "C{prn} eccentric {} not << circular {}",
            de.max_m,
            dc.max_m
        );
    }
}
