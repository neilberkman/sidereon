//! Unit tests for ECOM2 (Arnold et al. 2015, Sect. 4.2 eq. 5) and ECOM1
//! (Beutler et al. 1994; Springer, Beutler and Rothacher 1999; Arnold et al.
//! 2015 eq. 3): state and parameter partial derivatives against central
//! differences, undefined-frame handling (beta = 0 noon, Sun along orbit
//! normal, equatorial orbits, rectilinear states), umbra zeroing, shadow
//! configuration options, GCRS Sun consistency through `ForceModel`, and typed
//! errors.

use super::*;
use crate::astro::constants::astro::AU_KM;
use crate::astro::forces::test_helpers::{
    epochs_1990_to_2050, gcrs_sun_moon_km, mean_of_date_sun_moon_km,
};
use std::f64::consts::{FRAC_PI_2, PI};

const A_KM: f64 = 26_560.0;

/// A Moon position far off the Sun line so lunar occultation does not fire.
const CLEAR_MOON_KM: Vector3<f64> = Vector3::new(0.0, 384_400.0, 0.0);

fn all_ecom2_shadow_configs(
    base: &Ecom2SolarRadiationPressure,
) -> [Ecom2SolarRadiationPressure; 6] {
    [
        base.clone(),
        base.clone().with_moon_eclipse(false),
        base.clone()
            .with_earth_shadow_model(EarthShadowModel::Spherical),
        base.clone()
            .with_earth_shadow_model(EarthShadowModel::Spherical)
            .with_moon_eclipse(false),
        base.clone().with_shadow_function(ShadowFunction::Conical),
        base.clone()
            .with_shadow_function(ShadowFunction::Conical)
            .with_earth_shadow_model(EarthShadowModel::Spherical),
    ]
}

fn all_ecom1_shadow_configs(base: Ecom1SolarRadiationPressure) -> [Ecom1SolarRadiationPressure; 6] {
    [
        base,
        base.with_moon_eclipse(false),
        base.with_earth_shadow_model(EarthShadowModel::Spherical),
        base.with_earth_shadow_model(EarthShadowModel::Spherical)
            .with_moon_eclipse(false),
        base.with_shadow_function(ShadowFunction::Conical),
        base.with_shadow_function(ShadowFunction::Conical)
            .with_earth_shadow_model(EarthShadowModel::Spherical),
    ]
}

/// Circular orbit of inclination `inc` and node `raan` at argument of
/// latitude `u`, with the speed of a 26 560 km GPS orbit.
fn state(inc: f64, raan: f64, u: f64) -> (Vector3<f64>, Vector3<f64>) {
    let p = Vector3::new(libm::cos(raan), libm::sin(raan), 0.0);
    let q = Vector3::new(
        -libm::cos(inc) * libm::sin(raan),
        libm::cos(inc) * libm::cos(raan),
        libm::sin(inc),
    );
    let speed = (398_600.441_8_f64 / A_KM).sqrt();
    (
        A_KM * (libm::cos(u) * p + libm::sin(u) * q),
        speed * (-libm::sin(u) * p + libm::cos(u) * q),
    )
}

/// Sun at `distance` with elevation `beta` above the orbit plane and in-plane
/// argument of latitude `u_sun`.
fn sun(inc: f64, raan: f64, u_sun: f64, beta: f64, distance: f64) -> Vector3<f64> {
    let p = Vector3::new(libm::cos(raan), libm::sin(raan), 0.0);
    let q = Vector3::new(
        -libm::cos(inc) * libm::sin(raan),
        libm::cos(inc) * libm::cos(raan),
        libm::sin(inc),
    );
    let n = p.cross(&q);
    distance
        * (libm::cos(beta) * (libm::cos(u_sun) * p + libm::sin(u_sun) * q) + libm::sin(beta) * n)
}

fn mixed_d4b1() -> Ecom2D4b1Coefficients {
    Ecom2D4b1Coefficients {
        b0: -1.0e-9,
        b1c: 3.0e-10,
        b1s: -4.0e-10,
        d0: -1.0e-7,
        d2c: 2.0e-9,
        d4c: 1.0e-10,
        d2s: -5.0e-10,
        d4s: 2.0e-10,
        y0: 5.0e-10,
    }
}

fn mixed_d2b1() -> Ecom2D2b1Coefficients {
    Ecom2D2b1Coefficients {
        b0: -1.0e-9,
        b1c: 3.0e-10,
        b1s: -4.0e-10,
        d0: -1.0e-7,
        d2c: 2.0e-9,
        d2s: -5.0e-10,
        y0: 5.0e-10,
    }
}

fn mixed1() -> Ecom1Coefficients {
    Ecom1Coefficients {
        d0: -1.0e-7,
        dc: 1.5e-9,
        ds: -2.0e-9,
        y0: 5.0e-10,
        yc: 3.0e-10,
        ys: -6.0e-10,
        b0: -1.0e-9,
        bc: 4.0e-10,
        bs: 2.0e-10,
    }
}

/// Fourth-order central difference quotient of `f(r, v)` wrt position and velocity.
fn state_central_diff_4(
    f: impl Fn(Vector3<f64>, Vector3<f64>) -> Vector3<f64>,
    r: Vector3<f64>,
    v: Vector3<f64>,
    h_r: f64,
    h_v: f64,
) -> AccelerationPartials {
    let mut partials = AccelerationPartials::zeros();
    for axis in 0..3 {
        let at_r = |step: f64| {
            let mut moved = r;
            moved[axis] += step * h_r;
            f(moved, v)
        };
        let col_r = (at_r(-2.0) - 8.0 * at_r(-1.0) + 8.0 * at_r(1.0) - at_r(2.0)) / (12.0 * h_r);
        partials.wrt_position.set_column(axis, &col_r);

        let at_v = |step: f64| {
            let mut moved = v;
            moved[axis] += step * h_v;
            f(r, moved)
        };
        let col_v = (at_v(-2.0) - 8.0 * at_v(-1.0) + 8.0 * at_v(1.0) - at_v(2.0)) / (12.0 * h_v);
        partials.wrt_velocity.set_column(axis, &col_v);
    }
    partials
}

#[test]
fn delta_u_is_u_minus_u_sun_in_the_direction_of_motion() {
    let (inc, raan, u_sun, beta) = (0.96, 0.4, 1.1, 0.35);
    let s = sun(inc, raan, u_sun, beta, AU_KM);
    for du in [-3.0, -FRAC_PI_2, -0.2, 0.0, 0.7, FRAC_PI_2, 2.5] {
        let (r, v) = state(inc, raan, u_sun + du);
        let g = ecom_geometry(r, v, s).expect("geometry");
        let got_du = g.delta_u_rad.expect("defined delta_u");
        assert!((got_du - du).abs() < 1e-14, "{du}: {got_du}");
        let u = g.argument_of_latitude_rad.expect("inclined orbit");
        let expected = (u_sun + du + PI).rem_euclid(2.0 * PI) - PI;
        assert!((u - expected).abs() < 1e-14, "{u} vs {expected}");
    }
}

#[test]
fn delta_u_is_zero_at_orbit_noon_and_pi_at_orbit_midnight() {
    let (inc, raan, u_sun) = (0.96, 0.4, 1.1);
    let s = sun(inc, raan, u_sun, 0.5, AU_KM);
    let (r, v) = state(inc, raan, u_sun);
    let noon = ecom_geometry(r, v, s).expect("noon");
    assert!(noon.delta_u_rad.expect("du").abs() < 1e-15);
    let (r, v) = state(inc, raan, u_sun + PI);
    let midnight = ecom_geometry(r, v, s).expect("midnight");
    assert!((midnight.delta_u_rad.expect("du").abs() - PI).abs() < 1e-15);
}

#[test]
fn ecom2_parameter_names_and_order_follow_orekit_convention() {
    let d0b0 = Ecom2SolarRadiationPressure::new(0, 0, vec![-1.0e-9, -1.0e-7, 5.0e-10]).unwrap();
    let names: Vec<String> = d0b0.parameters().into_iter().map(|p| p.name).collect();
    assert_eq!(names, ["ecom2.b0", "ecom2.d0", "ecom2.y0"]);
    assert_eq!(d0b0.parameter_count(), 3);

    let d2b1 = Ecom2SolarRadiationPressure::d2b1(mixed_d2b1()).unwrap();
    let names: Vec<String> = d2b1.parameters().into_iter().map(|p| p.name).collect();
    assert_eq!(names, Ecom2D2b1Coefficients::PARAMETER_NAMES);
    assert_eq!(d2b1.parameter_count(), 7);

    let d4b1 = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let names: Vec<String> = d4b1.parameters().into_iter().map(|p| p.name).collect();
    assert_eq!(names, Ecom2D4b1Coefficients::PARAMETER_NAMES);
    assert_eq!(d4b1.parameter_count(), 9);

    let d6b3 = Ecom2SolarRadiationPressure::parameter_names(3, 2).unwrap();
    assert_eq!(
        d6b3,
        [
            "ecom2.b0",
            "ecom2.b1c",
            "ecom2.b3c",
            "ecom2.b1s",
            "ecom2.b3s",
            "ecom2.d0",
            "ecom2.d2c",
            "ecom2.d4c",
            "ecom2.d6c",
            "ecom2.d2s",
            "ecom2.d4s",
            "ecom2.d6s",
            "ecom2.y0",
        ]
    );

    let ecom1 = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    let names1: Vec<String> = ecom1.parameters().into_iter().map(|p| p.name).collect();
    assert_eq!(names1, Ecom1Coefficients::PARAMETER_NAMES);
    assert_eq!(ecom1.parameter_count(), 9);
}

#[test]
fn default_shadow_is_wgs84_apparent_disc_with_moon_and_all_options_work() {
    let e2 = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    assert_eq!(e2.shadow_function(), ShadowFunction::ApparentDisc);
    assert_eq!(e2.earth_shadow_model(), EarthShadowModel::Wgs84Oblate);
    assert!(e2.moon_eclipse_enabled());

    let e2_earth_disc = e2.clone().with_moon_eclipse(false);
    assert_eq!(
        e2_earth_disc.shadow_function(),
        ShadowFunction::ApparentDisc
    );
    assert_eq!(
        e2_earth_disc.earth_shadow_model(),
        EarthShadowModel::Wgs84Oblate
    );
    assert!(!e2_earth_disc.moon_eclipse_enabled());

    let e2_conical = e2.clone().with_shadow_function(ShadowFunction::Conical);
    assert_eq!(e2_conical.shadow_function(), ShadowFunction::Conical);
    assert_eq!(
        e2_conical.earth_shadow_model(),
        EarthShadowModel::Wgs84Oblate
    );
    assert!(!e2_conical.moon_eclipse_enabled());
    assert_eq!(
        e2_conical,
        e2_earth_disc
            .clone()
            .with_shadow_function(ShadowFunction::Conical)
    );
    assert_eq!(e2_conical.clone().with_moon_eclipse(true), e2);
    assert_eq!(
        e2_conical
            .clone()
            .with_shadow_function(ShadowFunction::ApparentDisc),
        e2_earth_disc
    );

    let e1 = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    assert_eq!(e1.shadow_function(), ShadowFunction::ApparentDisc);
    assert_eq!(e1.earth_shadow_model(), EarthShadowModel::Wgs84Oblate);
    assert!(e1.moon_eclipse_enabled());

    let e1_earth_disc = e1.with_moon_eclipse(false);
    assert_eq!(
        e1_earth_disc.shadow_function(),
        ShadowFunction::ApparentDisc
    );
    assert!(!e1_earth_disc.moon_eclipse_enabled());

    let e1_conical = e1.with_shadow_function(ShadowFunction::Conical);
    assert_eq!(e1_conical.shadow_function(), ShadowFunction::Conical);
    assert!(!e1_conical.moon_eclipse_enabled());
    assert_eq!(
        e1_conical,
        e1_earth_disc.with_shadow_function(ShadowFunction::Conical)
    );
    assert_eq!(e1_conical.with_moon_eclipse(true), e1);
    assert_eq!(
        e1_conical.with_shadow_function(ShadowFunction::ApparentDisc),
        e1_earth_disc
    );

    // Conical disables Moon so *_with_sun succeeds and differs from apparent-disc in penumbra.
    let s = Vector3::new(AU_KM, 0.0, 0.0);
    let r = Vector3::new(-26_000.0, 6_000.0, 2_100.0);
    let v = Vector3::new(0.8, 2.5, 2.9);
    let a2_disc = e2_earth_disc.acceleration_with_sun(r, v, s).unwrap();
    let a2_con = e2_conical.acceleration_with_sun(r, v, s).unwrap();
    assert!((a2_disc - a2_con).norm() > 1.0e-12);
    let a1_disc = e1_earth_disc.acceleration_with_sun(r, v, s).unwrap();
    let a1_con = e1_conical.acceleration_with_sun(r, v, s).unwrap();
    assert!((a1_disc - a1_con).norm() > 1.0e-12);

    // Lunar umbra zeroes default ECOM2 and ECOM1 while Earth-only options ignore the Moon.
    let r_sun = Vector3::new(7_000.0, 1_000.0, 0.0);
    let v_sun = Vector3::new(-1.0, 7.4, 0.5);
    let moon_total = r_sun + (s - r_sun).normalize() * 356_000.0;
    assert_eq!(
        e2.acceleration_with_sun_moon(r_sun, v_sun, s, moon_total)
            .unwrap(),
        Vector3::zeros()
    );
    assert_eq!(
        e1.acceleration_with_sun_moon(r_sun, v_sun, s, moon_total)
            .unwrap(),
        Vector3::zeros()
    );
    assert!(
        e2_earth_disc
            .acceleration_with_sun_moon(r_sun, v_sun, s, moon_total)
            .unwrap()
            .norm()
            > 0.0
    );
    assert!(
        e2_conical
            .acceleration_with_sun_moon(r_sun, v_sun, s, moon_total)
            .unwrap()
            .norm()
            > 0.0
    );
    assert!(
        e1_earth_disc
            .acceleration_with_sun_moon(r_sun, v_sun, s, moon_total)
            .unwrap()
            .norm()
            > 0.0
    );
    assert!(
        e1_conical
            .acceleration_with_sun_moon(r_sun, v_sun, s, moon_total)
            .unwrap()
            .norm()
            > 0.0
    );
}

#[test]
fn state_and_parameter_partials_match_central_differences_in_sunlight_and_penumbra() {
    // 1. Sunlit GPS orbit, testing D2B1, D4B1, D6B3, and ECOM1 under the default shadow.
    let (inc, raan, u_sun) = (0.96, 0.4, 1.1);
    let s = sun(inc, raan, u_sun, 0.4, 0.985 * AU_KM);
    let (r, v) = state(inc, raan, u_sun + 2.2);
    let m = CLEAR_MOON_KM;

    let d2b1 = Ecom2SolarRadiationPressure::d2b1(mixed_d2b1()).unwrap();
    let d4b1 = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let d6b3 = Ecom2SolarRadiationPressure::new(
        3,
        2,
        vec![
            -1.0e-9, 3.0e-10, -1.5e-10, -4.0e-10, 2.5e-10, -1.0e-7, 2.0e-9, 1.0e-10, -8.0e-11,
            -5.0e-10, 2.0e-10, 6.0e-11, 5.0e-10,
        ],
    )
    .unwrap();

    for (label, model) in [("D2B1", &d2b1), ("D4B1", &d4b1), ("D6B3", &d6b3)] {
        let ad = model
            .acceleration_partials_with_sun_moon(r, v, s, m)
            .unwrap();
        let num = state_central_diff_4(
            |rr, vv| model.acceleration_with_sun_moon(rr, vv, s, m).unwrap(),
            r,
            v,
            1.0e-1,
            1.0e-5,
        );
        let pos_scale = ad.wrt_position.norm();
        let vel_scale = ad.wrt_velocity.norm();
        assert!(pos_scale > 0.0 && vel_scale > 0.0);
        assert!(
            (ad.wrt_position - num.wrt_position).norm() <= 1.0e-8 * pos_scale,
            "{label} da/dr: {:?} vs {:?}",
            ad.wrt_position,
            num.wrt_position
        );
        assert!(
            (ad.wrt_velocity - num.wrt_velocity).norm() <= 1.0e-8 * vel_scale,
            "{label} da/dv: {:?} vs {:?}",
            ad.wrt_velocity,
            num.wrt_velocity
        );

        let p_ad = model.parameter_partials_with_sun_moon(r, v, s, m).unwrap();
        let base = model.coefficients().to_vec();
        let h_c = 1.0e-8;
        for (k, partial) in p_ad.iter().enumerate() {
            let shifted = |delta: f64| {
                let mut c = base.clone();
                c[k] += delta;
                Ecom2SolarRadiationPressure::new(model.n_d(), model.n_b(), c)
                    .unwrap()
                    .acceleration_with_sun_moon(r, v, s, m)
                    .unwrap()
            };
            let fd = (shifted(h_c) - shifted(-h_c)) / (2.0 * h_c);
            assert!(
                (fd - partial).norm() <= 1.0e-10 * partial.norm().max(1.0e-3),
                "{label} param[{k}]: {fd:?} vs {partial:?}"
            );
        }
    }

    // ECOM1 in sunlight
    let ecom1 = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    let ad1 = ecom1
        .acceleration_partials_with_sun_moon(r, v, s, m)
        .unwrap();
    let num1 = state_central_diff_4(
        |rr, vv| ecom1.acceleration_with_sun_moon(rr, vv, s, m).unwrap(),
        r,
        v,
        1.0e-1,
        1.0e-3,
    );
    assert!(
        (ad1.wrt_position - num1.wrt_position).norm() <= 1.0e-8 * ad1.wrt_position.norm(),
        "ECOM1 da/dr: {:?} vs {:?}",
        ad1.wrt_position,
        num1.wrt_position
    );
    assert!(
        (ad1.wrt_velocity - num1.wrt_velocity).norm() <= 1.0e-8 * ad1.wrt_velocity.norm(),
        "ECOM1 da/dv: {:?} vs {:?}",
        ad1.wrt_velocity,
        num1.wrt_velocity
    );
    let p1 = ecom1.parameter_partials_with_sun_moon(r, v, s, m).unwrap();
    let base1 = mixed1().to_array();
    for (k, partial) in p1.iter().enumerate() {
        let h_c = 1.0e-8;
        let shifted = |delta: f64| {
            let mut c = base1;
            c[k] += delta;
            Ecom1SolarRadiationPressure::new(Ecom1Coefficients::from_array(c))
                .unwrap()
                .acceleration_with_sun_moon(r, v, s, m)
                .unwrap()
        };
        let fd = (shifted(h_c) - shifted(-h_c)) / (2.0 * h_c);
        assert!(
            (fd - partial).norm() <= 1.0e-10 * partial.norm().max(1.0e-3),
            "ECOM1 param[{k}]: {fd:?} vs {partial:?}"
        );
    }

    // 2. Penumbra across all six shadow configurations for ECOM2 and ECOM1.
    let s_pen = Vector3::new(AU_KM, 0.0, 0.0);
    let r_pen = Vector3::new(-26_000.0, 6_000.0, 2_100.0);
    let v_pen = Vector3::new(0.8, 2.5, 2.9);
    for (m2, m1) in all_ecom2_shadow_configs(&d4b1)
        .into_iter()
        .zip(all_ecom1_shadow_configs(ecom1))
    {
        let ad2 = m2
            .acceleration_partials_with_sun_moon(r_pen, v_pen, s_pen, m)
            .unwrap();
        let num2 = state_central_diff_4(
            |rr, vv| m2.acceleration_with_sun_moon(rr, vv, s_pen, m).unwrap(),
            r_pen,
            v_pen,
            1.0e-2,
            1.0e-5,
        );
        assert!(
            (ad2.wrt_position - num2.wrt_position).norm() <= 1.0e-7 * ad2.wrt_position.norm(),
            "penumbra ECOM2 {:?}/{:?}/moon={} da/dr",
            m2.shadow_function(),
            m2.earth_shadow_model(),
            m2.moon_eclipse_enabled()
        );
        assert!(
            (ad2.wrt_velocity - num2.wrt_velocity).norm() <= 1.0e-7 * ad2.wrt_velocity.norm(),
            "penumbra ECOM2 {:?}/{:?}/moon={} da/dv",
            m2.shadow_function(),
            m2.earth_shadow_model(),
            m2.moon_eclipse_enabled()
        );

        let ad1p = m1
            .acceleration_partials_with_sun_moon(r_pen, v_pen, s_pen, m)
            .unwrap();
        let num1p = state_central_diff_4(
            |rr, vv| m1.acceleration_with_sun_moon(rr, vv, s_pen, m).unwrap(),
            r_pen,
            v_pen,
            1.0e-2,
            1.0e-3,
        );
        assert!(
            (ad1p.wrt_position - num1p.wrt_position).norm() <= 1.0e-7 * ad1p.wrt_position.norm(),
            "penumbra ECOM1 {:?}/{:?}/moon={} da/dr",
            m1.shadow_function(),
            m1.earth_shadow_model(),
            m1.moon_eclipse_enabled()
        );
        assert!(
            (ad1p.wrt_velocity - num1p.wrt_velocity).norm() <= 1.0e-7 * ad1p.wrt_velocity.norm(),
            "penumbra ECOM1 {:?}/{:?}/moon={} da/dv",
            m1.shadow_function(),
            m1.earth_shadow_model(),
            m1.moon_eclipse_enabled()
        );

        // Parameter partials match finite differences in penumbra under every option.
        let p2 = m2
            .parameter_partials_with_sun_moon(r_pen, v_pen, s_pen, m)
            .unwrap();
        let base2 = m2.coefficients().to_vec();
        let h_c = 1.0e-8;
        for (k, partial) in p2.iter().enumerate() {
            let shifted = |delta: f64| {
                let mut c = base2.clone();
                c[k] += delta;
                let mut copy = Ecom2SolarRadiationPressure::new(m2.n_d(), m2.n_b(), c)
                    .unwrap()
                    .with_earth_shadow_model(m2.earth_shadow_model());
                copy = if m2.shadow_function() == ShadowFunction::Conical {
                    copy.with_shadow_function(ShadowFunction::Conical)
                } else {
                    copy.with_moon_eclipse(m2.moon_eclipse_enabled())
                };
                copy.acceleration_with_sun_moon(r_pen, v_pen, s_pen, m)
                    .unwrap()
            };
            let fd = (shifted(h_c) - shifted(-h_c)) / (2.0 * h_c);
            assert!(
                (fd - partial).norm() <= 1.0e-10 * partial.norm().max(1.0e-3),
                "penumbra ECOM2 param[{k}]: {fd:?} vs {partial:?}"
            );
        }
    }
}

#[test]
fn no_sun_distance_scaling_is_applied() {
    let (inc, raan) = (0.96, 0.4);
    let (r, v) = state(inc, raan, 0.5);
    let model = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
        d0: -1.0e-7,
        ..Default::default()
    })
    .unwrap();
    let near = model
        .acceleration_with_sun_moon(r, v, sun(inc, raan, 0.0, 0.3, AU_KM), CLEAR_MOON_KM)
        .unwrap();
    let far = model
        .acceleration_with_sun_moon(r, v, sun(inc, raan, 0.0, 0.3, 2.0 * AU_KM), CLEAR_MOON_KM)
        .unwrap();
    // Both have |a| = 1e-7 m/s^2 = 1e-10 km/s^2 (no (AU/|r_sun|)^2 factor).
    assert!((near.norm() - 1.0e-10).abs() <= 1.0e-24);
    assert!((far.norm() - 1.0e-10).abs() <= 1.0e-24);
}

#[test]
fn umbra_gives_exact_zero_including_degenerate_frame() {
    // Beta = 0 midnight: satellite directly behind Earth; eD || satPos so eY is undefined.
    let r = Vector3::new(-A_KM, 0.0, 0.0);
    let v = Vector3::new(0.0, 3.87, 0.0);
    let s = Vector3::new(AU_KM, 0.0, 0.0);
    let m = CLEAR_MOON_KM;

    let ecom2_base = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    for ecom2 in all_ecom2_shadow_configs(&ecom2_base) {
        assert_eq!(
            ecom2.acceleration_with_sun_moon(r, v, s, m).unwrap(),
            Vector3::zeros()
        );
        assert_eq!(
            ecom2
                .acceleration_partials_with_sun_moon(r, v, s, m)
                .unwrap(),
            AccelerationPartials::zeros()
        );
        assert!(ecom2
            .parameter_partials_with_sun_moon(r, v, s, m)
            .unwrap()
            .iter()
            .all(|p| *p == Vector3::zeros()));
    }

    let ecom1_base = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    for ecom1 in all_ecom1_shadow_configs(ecom1_base) {
        assert_eq!(
            ecom1.acceleration_with_sun_moon(r, v, s, m).unwrap(),
            Vector3::zeros()
        );
        assert_eq!(
            ecom1
                .acceleration_partials_with_sun_moon(r, v, s, m)
                .unwrap(),
            AccelerationPartials::zeros()
        );
        assert!(ecom1
            .parameter_partials_with_sun_moon(r, v, s, m)
            .unwrap()
            .iter()
            .all(|p| *p == Vector3::zeros()));
    }

    assert!(matches!(
        ecom_geometry(r, v, s),
        Err(PropagationError::NumericalFailure(_))
    ));
}

#[test]
fn beta_zero_noon_handles_zero_and_nonzero_y_b_terms() {
    // Beta = 0 lit noon: satellite directly between Earth and Sun -> eD || satPos -> eY undefined.
    let r = Vector3::new(A_KM, 0.0, 0.0);
    let v = Vector3::new(0.0, 3.87, 0.0);
    let s = Vector3::new(AU_KM, 0.0, 0.0);
    let m = CLEAR_MOON_KM;

    // Nonzero Y0 or B terms must return NumericalFailure.
    let full = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    assert!(matches!(
        full.acceleration_with_sun_moon(r, v, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        full.acceleration_partials_with_sun_moon(r, v, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        full.parameter_partials_with_sun_moon(r, v, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));

    // Only D terms nonzero (Y0 = B0 = B1c = B1s = 0): succeeds and matches Orekit's
    // three-term linear combination with zero Y and B terms to the bit.
    let d_only = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
        d0: -1.0e-7,
        d2c: 2.0e-9,
        d4c: 1.0e-10,
        d2s: -5.0e-10,
        d4s: 2.0e-10,
        ..Default::default()
    })
    .unwrap();
    let accel = d_only.acceleration_with_sun_moon(r, v, s, m).unwrap();
    let e_d = Vec3::new(1.0, 0.0, 0.0);
    let d_u = -1.0e-7 + 2.0e-9 + 1.0e-10;
    let orekit_lincomb = Vec3::lincomb3(
        d_u,
        e_d,
        0.0,
        Vec3::new(0.0, 1.0, 0.0),
        0.0,
        Vec3::new(0.0, 0.0, 1.0),
    );
    assert_eq!(accel.x.to_bits(), (orekit_lincomb.x / M_PER_KM).to_bits());
    assert_eq!(accel.y.to_bits(), (orekit_lincomb.y / M_PER_KM).to_bits());
    assert_eq!(accel.z.to_bits(), (orekit_lincomb.z / M_PER_KM).to_bits());

    let ad_noon = d_only
        .acceleration_partials_with_sun_moon(r, v, s, m)
        .unwrap();
    assert_eq!(ad_noon.wrt_velocity, Matrix3::zeros());
    let d_prime = 2.0 * (-5.0e-10) + 4.0 * 2.0e-10;
    let dist_m = (AU_KM - A_KM) * M_PER_KM;
    let a_m = A_KM * M_PER_KM;
    let expected_pos = Matrix3::new(
        0.0,
        d_prime / a_m,
        0.0,
        0.0,
        -d_u / dist_m,
        0.0,
        0.0,
        0.0,
        -d_u / dist_m,
    );
    assert!(
        (ad_noon.wrt_position - expected_pos).norm() <= 1.0e-14 * expected_pos.norm(),
        "beta=0 noon ECOM2 da/dr: {:?} vs {:?}",
        ad_noon.wrt_position,
        expected_pos
    );
    let num_noon = state_central_diff_4(
        |rr, vv| d_only.acceleration_with_sun_moon(rr, vv, s, m).unwrap(),
        r,
        v,
        10.0,
        1.0e-5,
    );
    assert!(
        (ad_noon.wrt_position - num_noon.wrt_position).norm()
            <= 1.0e-9 * ad_noon.wrt_position.norm(),
        "beta=0 noon ECOM2 da/dr vs FD: {:?} vs {:?}",
        ad_noon.wrt_position,
        num_noon.wrt_position
    );

    // Parameter partials still error because Y0 and B0 partials require eY and eB.
    assert!(matches!(
        d_only.parameter_partials_with_sun_moon(r, v, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));

    // ECOM1 at beta = 0 noon: same rule.
    let ecom1_full = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    assert!(matches!(
        ecom1_full.acceleration_with_sun_moon(r, v, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    let ecom1_d0 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        d0: -1.0e-7,
        ..Default::default()
    })
    .unwrap();
    let a1 = ecom1_d0.acceleration_with_sun_moon(r, v, s, m).unwrap();
    assert_eq!(a1.x.to_bits(), (-1.0e-7 / M_PER_KM).to_bits());
    assert_eq!(a1.y, 0.0);
    assert_eq!(a1.z, 0.0);
    let ad1_noon = ecom1_d0
        .acceleration_partials_with_sun_moon(r, v, s, m)
        .unwrap();
    assert_eq!(ad1_noon.wrt_velocity, Matrix3::zeros());
    let expected_pos1 = Matrix3::new(
        0.0,
        0.0,
        0.0,
        0.0,
        1.0e-7 / dist_m,
        0.0,
        0.0,
        0.0,
        1.0e-7 / dist_m,
    );
    assert!(
        (ad1_noon.wrt_position - expected_pos1).norm() <= 1.0e-14 * expected_pos1.norm(),
        "beta=0 noon ECOM1 da/dr: {:?} vs {:?}",
        ad1_noon.wrt_position,
        expected_pos1
    );
}

#[test]
fn sun_along_orbit_normal_and_equatorial_orbits_only_fail_when_needed() {
    // 1. Sun along orbit normal on an inclined orbit (i = atan2(3, 4) = 36.87 deg,
    //    u = atan2(5, 4) = 0.896 rad): exact Pythagorean vectors give
    //    Z = r x v = (0, -72_000, 96_000) strictly parallel to s = (0, -9e7, 1.2e8),
    //    so delta_u is undefined while ECOM1's u and its velocity partial are nonzero.
    let r = Vector3::new(16_000.0, 16_000.0, 12_000.0);
    let v = Vector3::new(-4.0, 2.0, 1.5);
    let s = Vector3::new(0.0, -90_000_000.0, 120_000_000.0);
    let m = Vector3::new(384_400.0, 0.0, 0.0);
    let u0 = libm::atan2(5.0, 4.0);

    let g = ecom_geometry(r, v, s).expect("geometry");
    assert_eq!(g.delta_u_rad, None);
    let u = g.argument_of_latitude_rad.expect("defined u");
    assert!((u - u0).abs() <= 1.0e-15, "u={u} vs {u0}");
    let (su, cu) = (libm::sin(u), libm::cos(u));
    let e_d = Vec3::new(g.dyb.d[0], g.dyb.d[1], g.dyb.d[2]);
    let e_y = Vec3::new(g.dyb.y[0], g.dyb.y[1], g.dyb.y[2]);
    let e_b = Vec3::new(g.dyb.b[0], g.dyb.b[1], g.dyb.b[2]);
    let e_d_km = Vector3::new(e_d.x / M_PER_KM, e_d.y / M_PER_KM, e_d.z / M_PER_KM);
    let e_y_km = Vector3::new(e_y.x / M_PER_KM, e_y.y / M_PER_KM, e_y.z / M_PER_KM);
    let e_b_km = Vector3::new(e_b.x / M_PER_KM, e_b.y / M_PER_KM, e_b.z / M_PER_KM);

    // ECOM1 succeeds with all 9 coefficients nonzero!
    let m1 = mixed1();
    let ecom1 = Ecom1SolarRadiationPressure::new(m1).unwrap();
    let a1 = ecom1.acceleration_with_sun_moon(r, v, s, m).unwrap();
    let exp1 = Vec3::lincomb3(
        m1.d0 + (m1.dc * cu + m1.ds * su),
        e_d,
        m1.y0 + (m1.yc * cu + m1.ys * su),
        e_y,
        m1.b0 + (m1.bc * cu + m1.bs * su),
        e_b,
    );
    // ECOM1 is evaluated in exponent-separated arithmetic directly from km and rounded once
    // at output, so it agrees with the unscaled Orekit-order reference within rounding
    // (measured max 1 ULP, 1.5e-16 relative).
    let exp1_km = Vector3::new(exp1.x / M_PER_KM, exp1.y / M_PER_KM, exp1.z / M_PER_KM);
    assert!(
        (a1 - exp1_km).norm() <= 4.0 * f64::EPSILON * exp1_km.norm(),
        "polar ECOM1 accel: {a1:?} vs {exp1_km:?} (rel={:e})",
        (a1 - exp1_km).norm() / exp1_km.norm()
    );

    let ad1 = ecom1
        .acceleration_partials_with_sun_moon(r, v, s, m)
        .unwrap();
    let num1 = state_central_diff_4(
        |rr, vv| ecom1.acceleration_with_sun_moon(rr, vv, s, m).unwrap(),
        r,
        v,
        1.0e-1,
        1.0e-3,
    );
    assert!(ad1.wrt_position.norm() > 0.0 && ad1.wrt_velocity.norm() > 0.0);
    assert!(
        (ad1.wrt_position - num1.wrt_position).norm() <= 1.0e-8 * ad1.wrt_position.norm(),
        "polar ECOM1 da/dr: {:?} vs {:?}",
        ad1.wrt_position,
        num1.wrt_position
    );
    assert!(
        (ad1.wrt_velocity - num1.wrt_velocity).norm() <= 1.0e-8 * ad1.wrt_velocity.norm(),
        "polar ECOM1 da/dv: {:?} vs {:?}",
        ad1.wrt_velocity,
        num1.wrt_velocity
    );

    let p1 = ecom1.parameter_partials_with_sun_moon(r, v, s, m).unwrap();
    // [D0, Dc, Ds, Y0, Yc, Ys, B0, Bc, Bs].
    let expected_p1 = [
        e_d_km,
        e_d_km * cu,
        e_d_km * su,
        e_y_km,
        e_y_km * cu,
        e_y_km * su,
        e_b_km,
        e_b_km * cu,
        e_b_km * su,
    ];
    for (k, (got, exp)) in p1.iter().zip(&expected_p1).enumerate() {
        assert!(
            (got - exp).norm() <= 1.0e-18,
            "polar ECOM1 param[{k}]: {got:?} vs {exp:?}"
        );
    }

    // ECOM2 with nonzero periodic coefficients errors.
    let ecom2_full = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    assert!(matches!(
        ecom2_full.acceleration_with_sun_moon(r, v, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        ecom2_full.parameter_partials_with_sun_moon(r, v, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));

    // ECOM2 with all periodic coefficients zero succeeds and equals Orekit's linear combination.
    let ecom2_const = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
        b0: -1.0e-9,
        d0: -1.0e-7,
        y0: 5.0e-10,
        ..Default::default()
    })
    .unwrap();
    let a2 = ecom2_const.acceleration_with_sun_moon(r, v, s, m).unwrap();
    let expected = Vec3::lincomb3(-1.0e-7, e_d, 5.0e-10, e_y, -1.0e-9, e_b);
    assert_eq!(a2.x.to_bits(), (expected.x / M_PER_KM).to_bits());
    assert_eq!(a2.y.to_bits(), (expected.y / M_PER_KM).to_bits());
    assert_eq!(a2.z.to_bits(), (expected.z / M_PER_KM).to_bits());

    let ad2 = ecom2_const
        .acceleration_partials_with_sun_moon(r, v, s, m)
        .unwrap();
    assert_eq!(ad2.wrt_velocity, Matrix3::zeros());
    let num2 = state_central_diff_4(
        |rr, vv| {
            ecom2_const
                .acceleration_with_sun_moon(rr, vv, s, m)
                .unwrap()
        },
        r,
        v,
        1.0e-1,
        1.0e-3,
    );
    assert!(ad2.wrt_position.norm() > 0.0);
    assert!(
        (ad2.wrt_position - num2.wrt_position).norm() <= 1.0e-8 * ad2.wrt_position.norm(),
        "sun-normal ECOM2 const da/dr: {:?} vs {:?}",
        ad2.wrt_position,
        num2.wrt_position
    );

    // D0B0 (no periodic parameters at all) also succeeds on parameter_partials.
    let d0b0 = Ecom2SolarRadiationPressure::new(0, 0, vec![-1.0e-9, -1.0e-7, 5.0e-10]).unwrap();
    assert_eq!(
        d0b0.acceleration_partials_with_sun_moon(r, v, s, m)
            .unwrap(),
        ad2
    );
    let p_d0b0 = d0b0.parameter_partials_with_sun_moon(r, v, s, m).unwrap();
    assert_eq!(p_d0b0, vec![e_b_km, e_d_km, e_y_km]);

    // 2. Equatorial GEO orbit: ECOM1 succeeds when all 1pr terms are zero, fails otherwise.
    let r_eq = Vector3::new(42_164.0, 0.0, 0.0);
    let v_eq = Vector3::new(0.0, 3.0747, 0.0);
    let s_eq = sun(0.0, 0.0, 1.0, 0.3, AU_KM);
    let g_eq = ecom_geometry(r_eq, v_eq, s_eq).expect("equatorial geometry");
    assert_eq!(g_eq.argument_of_latitude_rad, None);
    assert!(g_eq.delta_u_rad.is_some());

    let ecom1_const = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        d0: -1.0e-7,
        y0: 5.0e-10,
        b0: -1.0e-9,
        ..Default::default()
    })
    .unwrap();
    let a1_eq = ecom1_const
        .acceleration_with_sun_moon(r_eq, v_eq, s_eq, m)
        .unwrap();
    let expected_eq = Vec3::lincomb3(
        -1.0e-7,
        Vec3::new(g_eq.dyb.d[0], g_eq.dyb.d[1], g_eq.dyb.d[2]),
        5.0e-10,
        Vec3::new(g_eq.dyb.y[0], g_eq.dyb.y[1], g_eq.dyb.y[2]),
        -1.0e-9,
        Vec3::new(g_eq.dyb.b[0], g_eq.dyb.b[1], g_eq.dyb.b[2]),
    );
    let expected_eq_km = Vector3::new(
        expected_eq.x / M_PER_KM,
        expected_eq.y / M_PER_KM,
        expected_eq.z / M_PER_KM,
    );
    assert!(
        (a1_eq - expected_eq_km).norm() <= 4.0 * f64::EPSILON * expected_eq_km.norm(),
        "equatorial ECOM1 accel: {a1_eq:?} vs {expected_eq_km:?}"
    );

    let ad1_eq = ecom1_const
        .acceleration_partials_with_sun_moon(r_eq, v_eq, s_eq, m)
        .unwrap();
    assert_eq!(ad1_eq.wrt_velocity, Matrix3::zeros());
    let num1_eq = state_central_diff_4(
        |rr, vv| {
            ecom1_const
                .acceleration_with_sun_moon(rr, vv, s_eq, m)
                .unwrap()
        },
        r_eq,
        v_eq,
        1.0e-1,
        1.0e-3,
    );
    assert!(ad1_eq.wrt_position.norm() > 0.0);
    assert!(
        (ad1_eq.wrt_position - num1_eq.wrt_position).norm() <= 1.0e-8 * ad1_eq.wrt_position.norm(),
        "equatorial ECOM1 const da/dr: {:?} vs {:?}",
        ad1_eq.wrt_position,
        num1_eq.wrt_position
    );
    assert!(matches!(
        ecom1_const.parameter_partials_with_sun_moon(r_eq, v_eq, s_eq, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        ecom1.acceleration_with_sun_moon(r_eq, v_eq, s_eq, m),
        Err(PropagationError::NumericalFailure(_))
    ));
}

#[test]
fn extreme_velocity_preserves_defined_angles_and_zero_velocity_only_fails_when_needed() {
    let r = Vector3::new(26_560.0, 0.0, 0.0);
    let v_huge_eq = Vector3::new(0.0, 1.0e306, 0.0);
    let v_nom_eq = Vector3::new(0.0, 3.874, 0.0);
    let v_huge_inc = Vector3::new(0.0, 1.0e306, 1.0e306);
    let v_nom_inc = Vector3::new(0.0, 3.874, 3.874);
    let v_zero = Vector3::zeros();
    let s = Vector3::new(0.0, 149_597_870.7, 0.0);
    let m = Vector3::new(384_400.0, 0.0, 0.0);

    // 1. Huge velocity (> f64::MAX / 1000 km/s, overflowing m/s conversion) still has
    //    well-defined delta_u (and well-defined u when inclined), matching nominal velocity.
    let g_nom_eq = ecom_geometry(r, v_nom_eq, s).expect("nominal equatorial geometry");
    let g_huge_eq = ecom_geometry(r, v_huge_eq, s).expect("huge-velocity equatorial geometry");
    assert_eq!(g_huge_eq.dyb, g_nom_eq.dyb);
    assert_eq!(g_huge_eq.lighting_ratio.to_bits(), 1.0_f64.to_bits());
    assert!(
        (g_huge_eq.delta_u_rad.expect("huge delta_u") - g_nom_eq.delta_u_rad.unwrap()).abs()
            <= 4.0 * f64::EPSILON
    );
    // Equatorial orbit: ECOM1 argument of latitude is genuinely undefined.
    assert_eq!(g_huge_eq.argument_of_latitude_rad, None);

    let g_nom_inc = ecom_geometry(r, v_nom_inc, s).expect("nominal inclined geometry");
    let g_huge_inc = ecom_geometry(r, v_huge_inc, s).expect("huge-velocity inclined geometry");
    assert!(
        (g_huge_inc.delta_u_rad.expect("inclined delta_u") - g_nom_inc.delta_u_rad.unwrap()).abs()
            <= 4.0 * f64::EPSILON
    );
    assert!(
        (g_huge_inc.argument_of_latitude_rad.expect("inclined u")
            - g_nom_inc.argument_of_latitude_rad.unwrap())
        .abs()
            <= 4.0 * f64::EPSILON
    );

    let d4b1_full = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let a2_nom = d4b1_full
        .acceleration_with_sun_moon(r, v_nom_inc, s, m)
        .unwrap();
    let a2_huge = d4b1_full
        .acceleration_with_sun_moon(r, v_huge_inc, s, m)
        .unwrap();
    assert!((a2_huge - a2_nom).norm() <= 4.0 * f64::EPSILON * a2_nom.norm());
    let pp2_nom = d4b1_full
        .parameter_partials_with_sun_moon(r, v_nom_inc, s, m)
        .unwrap();
    let pp2_huge = d4b1_full
        .parameter_partials_with_sun_moon(r, v_huge_inc, s, m)
        .unwrap();
    for (p, pr) in pp2_huge.iter().zip(pp2_nom.iter()) {
        assert!((p - pr).norm() <= 4.0 * f64::EPSILON * pr.norm().max(1.0 / M_PER_KM));
    }
    let ad2_nom = d4b1_full
        .acceleration_partials_with_sun_moon(r, v_nom_inc, s, m)
        .unwrap();
    let ad2_huge = d4b1_full
        .acceleration_partials_with_sun_moon(r, v_huge_inc, s, m)
        .unwrap();
    assert!(
        (ad2_huge.wrt_position - ad2_nom.wrt_position).norm()
            <= 1.0e-12 * ad2_nom.wrt_position.norm()
    );
    assert!(ad2_huge.wrt_velocity.iter().all(|x| x.is_finite()));

    let ecom1_full = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    let a1_nom = ecom1_full
        .acceleration_with_sun_moon(r, v_nom_inc, s, m)
        .unwrap();
    let a1_huge = ecom1_full
        .acceleration_with_sun_moon(r, v_huge_inc, s, m)
        .unwrap();
    assert!((a1_huge - a1_nom).norm() <= 4.0 * f64::EPSILON * a1_nom.norm());
    let ad1_huge = ecom1_full
        .acceleration_partials_with_sun_moon(r, v_huge_inc, s, m)
        .unwrap();
    assert!(ad1_huge.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad1_huge.wrt_velocity.iter().all(|x| x.is_finite()));
    let pp1_huge = ecom1_full
        .parameter_partials_with_sun_moon(r, v_huge_inc, s, m)
        .unwrap();
    assert!(pp1_huge.iter().all(|v| v.iter().all(|x| x.is_finite())));

    // 2. Zero velocity (genuinely undefined r x v == 0): constant-only models succeed
    //    with da/dv == 0, while periodic terms and parameter partials error.
    let g_zero = ecom_geometry(r, v_zero, s).expect("zero-velocity geometry");
    assert_eq!(g_zero.dyb, g_nom_eq.dyb);
    assert_eq!(g_zero.delta_u_rad, None);
    assert_eq!(g_zero.argument_of_latitude_rad, None);

    let e_d = Vec3::new(g_zero.dyb.d[0], g_zero.dyb.d[1], g_zero.dyb.d[2]);
    let e_y = Vec3::new(g_zero.dyb.y[0], g_zero.dyb.y[1], g_zero.dyb.y[2]);
    let e_b = Vec3::new(g_zero.dyb.b[0], g_zero.dyb.b[1], g_zero.dyb.b[2]);
    let e_d_km = Vector3::new(e_d.x / M_PER_KM, e_d.y / M_PER_KM, e_d.z / M_PER_KM);
    let e_y_km = Vector3::new(e_y.x / M_PER_KM, e_y.y / M_PER_KM, e_y.z / M_PER_KM);
    let e_b_km = Vector3::new(e_b.x / M_PER_KM, e_b.y / M_PER_KM, e_b.z / M_PER_KM);
    let expected_raw = Vec3::lincomb3(-1.0e-7, e_d, 5.0e-10, e_y, -1.0e-9, e_b);
    let expected_accel = Vector3::new(
        expected_raw.x / M_PER_KM,
        expected_raw.y / M_PER_KM,
        expected_raw.z / M_PER_KM,
    );

    let d0b0 = Ecom2SolarRadiationPressure::new(0, 0, vec![-1.0e-9, -1.0e-7, 5.0e-10]).unwrap();
    let ad_nom = d0b0
        .acceleration_partials_with_sun_moon(r, v_nom_eq, s, m)
        .unwrap();
    let a_d0b0 = d0b0.acceleration_with_sun_moon(r, v_zero, s, m).unwrap();
    for axis in 0..3 {
        assert_eq!(a_d0b0[axis].to_bits(), expected_accel[axis].to_bits());
    }
    let ad_d0b0 = d0b0
        .acceleration_partials_with_sun_moon(r, v_zero, s, m)
        .unwrap();
    assert_eq!(ad_d0b0.wrt_velocity, Matrix3::zeros());
    assert_eq!(ad_d0b0.wrt_position, ad_nom.wrt_position);
    assert_eq!(
        d0b0.parameter_partials_with_sun_moon(r, v_zero, s, m)
            .unwrap(),
        vec![e_b_km, e_d_km, e_y_km]
    );

    let d4b1_const = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
        b0: -1.0e-9,
        d0: -1.0e-7,
        y0: 5.0e-10,
        ..Default::default()
    })
    .unwrap();
    let a_d4b1_c = d4b1_const
        .acceleration_with_sun_moon(r, v_zero, s, m)
        .unwrap();
    for axis in 0..3 {
        assert_eq!(a_d4b1_c[axis].to_bits(), expected_accel[axis].to_bits());
    }
    let ad_d4b1_c = d4b1_const
        .acceleration_partials_with_sun_moon(r, v_zero, s, m)
        .unwrap();
    assert_eq!(ad_d4b1_c.wrt_velocity, Matrix3::zeros());
    assert_eq!(ad_d4b1_c.wrt_position, ad_nom.wrt_position);
    assert!(matches!(
        d4b1_const.parameter_partials_with_sun_moon(r, v_zero, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        d4b1_full.acceleration_with_sun_moon(r, v_zero, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        d4b1_full.acceleration_partials_with_sun_moon(r, v_zero, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        d4b1_full.parameter_partials_with_sun_moon(r, v_zero, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));

    let ecom1_const = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        d0: -1.0e-7,
        y0: 5.0e-10,
        b0: -1.0e-9,
        ..Default::default()
    })
    .unwrap();
    let a_e1_c = ecom1_const
        .acceleration_with_sun_moon(r, v_zero, s, m)
        .unwrap();
    // ECOM1 is evaluated in exponent-separated arithmetic directly from km and rounded once
    // at output, so it agrees with the unscaled ECOM2 reference within rounding (measured 1 ULP).
    assert!(
        (a_e1_c - expected_accel).norm() <= 4.0 * f64::EPSILON * expected_accel.norm(),
        "zero-vel ECOM1 accel: {a_e1_c:?} vs {expected_accel:?}"
    );
    let ad_e1_nom = ecom1_const
        .acceleration_partials_with_sun_moon(r, v_nom_eq, s, m)
        .unwrap();
    let ad_e1_c = ecom1_const
        .acceleration_partials_with_sun_moon(r, v_zero, s, m)
        .unwrap();
    assert_eq!(ad_e1_c.wrt_velocity, Matrix3::zeros());
    assert_eq!(ad_e1_c.wrt_position, ad_e1_nom.wrt_position);
    assert!(
        mat3_rel_err(ad_e1_c.wrt_position, ad_nom.wrt_position).unwrap() <= 8.0 * f64::EPSILON,
        "zero-vel ECOM1 wrt_position vs ECOM2 ref"
    );
    assert!(matches!(
        ecom1_const.parameter_partials_with_sun_moon(r, v_zero, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        ecom1_full.acceleration_with_sun_moon(r, v_zero, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        ecom1_full.acceleration_partials_with_sun_moon(r, v_zero, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        ecom1_full.parameter_partials_with_sun_moon(r, v_zero, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));

    // 3. ForceModel trait methods agree on the huge-velocity inclined state and zero-velocity state.
    let ctx = PropagationContext::default();
    let st_huge = CartesianState::new(
        0.0,
        [r.x, r.y, r.z],
        [v_huge_inc.x, v_huge_inc.y, v_huge_inc.z],
    );
    let st_nom = CartesianState::new(
        0.0,
        [r.x, r.y, r.z],
        [v_nom_inc.x, v_nom_inc.y, v_nom_inc.z],
    );
    assert!(
        (d4b1_full.acceleration(&st_huge, &ctx).unwrap()
            - d4b1_full.acceleration(&st_nom, &ctx).unwrap())
        .norm()
            <= 4.0 * f64::EPSILON * d4b1_full.acceleration(&st_nom, &ctx).unwrap().norm()
    );
    assert!(d4b1_full
        .acceleration_partials(d4b1_full.coefficients(), &st_huge, &ctx)
        .is_ok());
    assert!(d4b1_full
        .parameter_partials(d4b1_full.coefficients(), &st_huge, &ctx)
        .is_ok());

    let st_zero = CartesianState::new(0.0, [r.x, r.y, r.z], [0.0; 3]);
    let (sun_gcrs, _) = gcrs_sun_moon_km(0.0);
    let g_gcrs = ecom_geometry(r, v_zero, sun_gcrs).expect("GCRS geometry");
    assert_eq!(g_gcrs.delta_u_rad, None);
    assert_eq!(g_gcrs.argument_of_latitude_rad, None);
    let eb_g = Vector3::from(g_gcrs.dyb.b) / M_PER_KM;
    let ed_g = Vector3::from(g_gcrs.dyb.d) / M_PER_KM;
    let ey_g = Vector3::from(g_gcrs.dyb.y) / M_PER_KM;

    let d0b0_p = d0b0.coefficients();
    let d0b0_fm_a = d0b0.acceleration(&st_zero, &ctx).unwrap();
    let d0b0_fm_ad = d0b0.acceleration_partials(d0b0_p, &st_zero, &ctx).unwrap();
    assert_eq!(d0b0_fm_ad.wrt_velocity, Matrix3::zeros());
    let d0b0_fm_pp = d0b0.parameter_partials(d0b0_p, &st_zero, &ctx).unwrap();
    assert_eq!(d0b0_fm_pp, vec![eb_g, ed_g, ey_g]);

    let d4b1_c_p = d4b1_const.coefficients();
    assert_eq!(d4b1_const.acceleration(&st_zero, &ctx).unwrap(), d0b0_fm_a);
    assert_eq!(
        d4b1_const
            .acceleration_partials(d4b1_c_p, &st_zero, &ctx)
            .unwrap(),
        d0b0_fm_ad
    );
    assert!(matches!(
        d4b1_const.parameter_partials(d4b1_c_p, &st_zero, &ctx),
        Err(PropagationError::NumericalFailure(_))
    ));

    let e1_c_p = ecom1_const.coefficients().to_array();
    let e1_fm_a = ecom1_const.acceleration(&st_zero, &ctx).unwrap();
    // ECOM1 is evaluated in exponent-separated arithmetic directly from km and rounded once
    // at output, so it agrees with the unscaled ECOM2 reference within rounding (measured <= 2 ULP).
    assert!(
        (e1_fm_a - d0b0_fm_a).norm() <= 4.0 * f64::EPSILON * d0b0_fm_a.norm(),
        "zero-vel ForceModel ECOM1 accel: {e1_fm_a:?} vs {d0b0_fm_a:?}"
    );
    let e1_fm_ad = ecom1_const
        .acceleration_partials(&e1_c_p, &st_zero, &ctx)
        .unwrap();
    assert_eq!(e1_fm_ad.wrt_velocity, Matrix3::zeros());
    assert!(
        mat3_rel_err(e1_fm_ad.wrt_position, d0b0_fm_ad.wrt_position).unwrap() <= 8.0 * f64::EPSILON,
        "zero-vel ForceModel ECOM1 wrt_position vs ECOM2 ref"
    );
    assert!(matches!(
        ecom1_const.parameter_partials(&e1_c_p, &st_zero, &ctx),
        Err(PropagationError::NumericalFailure(_))
    ));

    let d4b1_f_p = d4b1_full.coefficients();
    assert!(matches!(
        d4b1_full.acceleration(&st_zero, &ctx),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        d4b1_full.acceleration_partials(d4b1_f_p, &st_zero, &ctx),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        d4b1_full.parameter_partials(d4b1_f_p, &st_zero, &ctx),
        Err(PropagationError::NumericalFailure(_))
    ));

    let e1_f_p = ecom1_full.coefficients().to_array();
    assert!(matches!(
        ecom1_full.acceleration(&st_zero, &ctx),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        ecom1_full.acceleration_partials(&e1_f_p, &st_zero, &ctx),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        ecom1_full.parameter_partials(&e1_f_p, &st_zero, &ctx),
        Err(PropagationError::NumericalFailure(_))
    ));
}

#[test]
fn force_model_uses_gcrs_sun_and_moon_matching_explicit_bodies_bit_for_bit() {
    let ctx = PropagationContext::default();
    let ecom2_base = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let ecom1_base = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();

    let (r, v) = state(0.96, 0.4, 1.2);
    for epoch in epochs_1990_to_2050() {
        let st = CartesianState::new(epoch, [r.x, r.y, r.z], [v.x, v.y, v.z]);
        let (sun_gcrs, moon_gcrs) = gcrs_sun_moon_km(epoch);

        for (ecom2, ecom1) in all_ecom2_shadow_configs(&ecom2_base)
            .into_iter()
            .zip(all_ecom1_shadow_configs(ecom1_base))
        {
            let a2_fm = ecom2.acceleration(&st, &ctx).unwrap();
            let a2_exp = ecom2
                .acceleration_with_sun_moon(st.position_km, st.velocity_km_s, sun_gcrs, moon_gcrs)
                .unwrap();
            for axis in 0..3 {
                assert_eq!(a2_fm[axis].to_bits(), a2_exp[axis].to_bits());
            }
            if !ecom2.moon_eclipse_enabled() {
                let a2_sun = ecom2
                    .acceleration_with_sun(st.position_km, st.velocity_km_s, sun_gcrs)
                    .unwrap();
                for axis in 0..3 {
                    assert_eq!(a2_fm[axis].to_bits(), a2_sun[axis].to_bits());
                }
            }
            let p2_fm = ecom2
                .acceleration_partials(ecom2.coefficients(), &st, &ctx)
                .unwrap();
            let p2_exp = ecom2
                .acceleration_partials_with_sun_moon(
                    st.position_km,
                    st.velocity_km_s,
                    sun_gcrs,
                    moon_gcrs,
                )
                .unwrap();
            assert_eq!(p2_fm, p2_exp);
            let pp2_fm = ecom2
                .parameter_partials(ecom2.coefficients(), &st, &ctx)
                .unwrap();
            let pp2_exp = ecom2
                .parameter_partials_with_sun_moon(
                    st.position_km,
                    st.velocity_km_s,
                    sun_gcrs,
                    moon_gcrs,
                )
                .unwrap();
            assert_eq!(pp2_fm, pp2_exp);

            let a1_fm = ecom1.acceleration(&st, &ctx).unwrap();
            let a1_exp = ecom1
                .acceleration_with_sun_moon(st.position_km, st.velocity_km_s, sun_gcrs, moon_gcrs)
                .unwrap();
            for axis in 0..3 {
                assert_eq!(a1_fm[axis].to_bits(), a1_exp[axis].to_bits());
            }
            if !ecom1.moon_eclipse_enabled() {
                let a1_sun = ecom1
                    .acceleration_with_sun(st.position_km, st.velocity_km_s, sun_gcrs)
                    .unwrap();
                for axis in 0..3 {
                    assert_eq!(a1_fm[axis].to_bits(), a1_sun[axis].to_bits());
                }
            }
            let c1 = ecom1.coefficients().to_array();
            let p1_fm = ecom1.acceleration_partials(&c1, &st, &ctx).unwrap();
            let p1_exp = ecom1
                .acceleration_partials_with_sun_moon(
                    st.position_km,
                    st.velocity_km_s,
                    sun_gcrs,
                    moon_gcrs,
                )
                .unwrap();
            assert_eq!(p1_fm, p1_exp);
            let pp1_fm = ecom1.parameter_partials(&c1, &st, &ctx).unwrap();
            let pp1_exp = ecom1
                .parameter_partials_with_sun_moon(
                    st.position_km,
                    st.velocity_km_s,
                    sun_gcrs,
                    moon_gcrs,
                )
                .unwrap();
            assert_eq!(pp1_fm, pp1_exp.to_vec());
        }
    }

    // Confirm GCRS differs from mean-of-date at 2026 (precession ~0.37 deg).
    let epoch_2026 = epochs_1990_to_2050()[2];
    let st = CartesianState::new(epoch_2026, [r.x, r.y, r.z], [v.x, v.y, v.z]);
    let (sun_mod, moon_mod) = mean_of_date_sun_moon_km(epoch_2026);
    let a_gcrs = ecom2_base.acceleration(&st, &ctx).unwrap();
    let a_mod = ecom2_base
        .acceleration_with_sun_moon(st.position_km, st.velocity_km_s, sun_mod, moon_mod)
        .unwrap();
    assert!((a_gcrs - a_mod).norm() > 5.0e-13);
}

#[test]
fn invalid_inputs_and_moon_eclipse_guard_are_typed_errors() {
    assert!(matches!(
        Ecom2SolarRadiationPressure::new(1, 1, vec![0.0; 6]),
        Err(PropagationError::InvalidInput(_))
    ));
    let mut bad = mixed_d4b1();
    bad.d4s = f64::NAN;
    assert!(matches!(
        Ecom2SolarRadiationPressure::d4b1(bad),
        Err(PropagationError::InvalidInput(_))
    ));
    let mut bad1 = mixed1();
    bad1.yc = f64::INFINITY;
    assert!(matches!(
        Ecom1SolarRadiationPressure::new(bad1),
        Err(PropagationError::InvalidInput(_))
    ));

    let default_model = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let earth_only_model = default_model.clone().with_moon_eclipse(false);
    let (r, v) = state(0.96, 0.4, 0.5);
    let s = sun(0.96, 0.4, 0.0, 0.3, AU_KM);
    let nan = Vector3::new(f64::NAN, 0.0, 0.0);
    let huge = Vector3::new(f64::MAX, f64::MAX, 0.0);
    for (rr, vv, ss) in [
        (nan, v, s),
        (r, nan, s),
        (r, v, nan),
        (Vector3::zeros(), v, s),
        (r, v, Vector3::zeros()),
        (huge, v, s),
        (r, v, huge),
        // Satellite inside Earth or coincident with Sun
        (Vector3::new(6_000.0, 0.0, 0.0), v, s),
        (s, v, s),
    ] {
        assert!(
            matches!(
                earth_only_model.acceleration_with_sun(rr, vv, ss),
                Err(PropagationError::InvalidInput(_))
            ),
            "{rr:?} {vv:?} {ss:?}"
        );
    }

    // Parallel r and v with nonzero periodic terms -> NumericalFailure.
    assert!(matches!(
        earth_only_model.acceleration_with_sun(r, 2.0 * r, s),
        Err(PropagationError::NumericalFailure(_))
    ));

    // Calling _with_sun when moon eclipse is enabled (including the default) is InvalidInput.
    assert!(matches!(
        default_model.acceleration_with_sun(r, v, s),
        Err(PropagationError::InvalidInput(_))
    ));
    assert!(matches!(
        default_model.acceleration_partials_with_sun(r, v, s),
        Err(PropagationError::InvalidInput(_))
    ));
    assert!(matches!(
        default_model.parameter_partials_with_sun(r, v, s),
        Err(PropagationError::InvalidInput(_))
    ));

    let default_ecom1 = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    assert!(matches!(
        default_ecom1.acceleration_with_sun(r, v, s),
        Err(PropagationError::InvalidInput(_))
    ));
    assert!(matches!(
        default_ecom1.acceleration_partials_with_sun(r, v, s),
        Err(PropagationError::InvalidInput(_))
    ));
    assert!(matches!(
        default_ecom1.parameter_partials_with_sun(r, v, s),
        Err(PropagationError::InvalidInput(_))
    ));
}

#[test]
fn overflowing_norm_vectors_normalize_accurately_across_ecom_paths() {
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};

    // Direct Vec3 / FieldVec3 normalize check on finite vectors whose norm_sq overflows.
    let big_z = Vec3::new(0.0, 0.0, -1.0e155)
        .normalize()
        .expect("big_z unit");
    assert_eq!(big_z.x, 0.0);
    assert_eq!(big_z.y, 0.0);
    assert_eq!(big_z.z, -1.0);

    let big_diag = Vec3::new(-1.0e155, 1.0e155, 0.0)
        .normalize()
        .expect("big_diag unit");
    assert!((big_diag.x + FRAC_1_SQRT_2).abs() <= 2.0 * f64::EPSILON);
    assert!((big_diag.y - FRAC_1_SQRT_2).abs() <= 2.0 * f64::EPSILON);
    assert_eq!(big_diag.z, 0.0);

    let big_field = FieldVec3::<3>::new(
        Grad::variable(0, -1.0e155),
        Grad::variable(1, 1.0e155),
        Grad::variable(2, 0.0),
    )
    .normalize()
    .expect("big_field unit");
    assert!((big_field.x.value + FRAC_1_SQRT_2).abs() <= 2.0 * f64::EPSILON);
    assert!((big_field.y.value - FRAC_1_SQRT_2).abs() <= 2.0 * f64::EPSILON);
    assert_eq!(big_field.z.value, 0.0);
    assert!(
        big_field.x.is_all_finite() && big_field.y.is_all_finite() && big_field.z.is_all_finite()
    );
    let expected_inv_norm = 1.0 / (2.0_f64.sqrt() * 1.0e155);
    assert!(
        (big_field.z.grad[2] - expected_inv_norm).abs() <= 4.0 * f64::EPSILON * expected_inv_norm
    );

    // 1. Conical case where e_Y = e_D x sat_pos overflows norm_sq (|sat_pos| = 1e155 m).
    let r_conical = Vector3::new(1.0e152, 0.0, 0.0);
    let v_conical = Vector3::new(0.0, 0.0, 1.0e-160);
    let s_conical = Vector3::new(1.0e152, 1.0e100, 0.0);

    let ecom2_conical = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1())
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
    let a2_c = ecom2_conical
        .acceleration_with_sun(r_conical, v_conical, s_conical)
        .expect("conical ECOM2 accel");
    let p2_c = ecom2_conical
        .parameter_partials_with_sun(r_conical, v_conical, s_conical)
        .expect("conical ECOM2 param partials");
    let ad2_c = ecom2_conical
        .acceleration_partials_with_sun(r_conical, v_conical, s_conical)
        .expect("conical ECOM2 state partials");

    // True frame: e_D = (0, 1, 0), e_Y = (0, 0, -1), e_B = (-1, 0, 0), delta_u = 0.
    let inv_km = 1.0 / M_PER_KM;
    assert_eq!(p2_c[0], Vector3::new(-inv_km, 0.0, 0.0)); // B0 -> e_B
    assert_eq!(p2_c[1], Vector3::new(-inv_km, 0.0, 0.0)); // B1c * cos(0)
    assert_eq!(p2_c[2], Vector3::zeros()); // B1s * sin(0)
    assert_eq!(p2_c[3], Vector3::new(0.0, inv_km, 0.0)); // D0 -> e_D
    assert_eq!(p2_c[8], Vector3::new(0.0, 0.0, -inv_km)); // Y0 -> e_Y
    let recon2: Vector3<f64> = ecom2_conical
        .coefficients()
        .iter()
        .zip(p2_c.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a2_c - recon2).norm() <= 4.0 * f64::EPSILON * a2_c.norm());
    // Y and B components are preserved (non-zero).
    assert!(a2_c.x.abs() > 1.0e-13 && a2_c.z.abs() > 1.0e-13);
    assert!(ad2_c.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad2_c.wrt_velocity.iter().all(|x| x.is_finite()));
    // Exact d(delta_u)/d(vel_y) = -1e105 rad/(m/s) -> da/dv_y = (b1s * e_B + (2*d2s + 4*d4s) * e_D) * (-1e105).
    let d_du_dv_y = -1.0e105;
    let c_d4b1 = mixed_d4b1();
    let expected_da_dvy_x = -c_d4b1.b1s * d_du_dv_y;
    let expected_da_dvy_y = (2.0 * c_d4b1.d2s + 4.0 * c_d4b1.d4s) * d_du_dv_y;
    assert!(
        (ad2_c.wrt_velocity[(0, 1)] - expected_da_dvy_x).abs() <= 1.0e-12 * expected_da_dvy_x.abs()
    );
    assert!(
        (ad2_c.wrt_velocity[(1, 1)] - expected_da_dvy_y).abs() <= 1.0e-12 * expected_da_dvy_y.abs()
    );

    let ecom1_conical = Ecom1SolarRadiationPressure::new(mixed1())
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
    let a1_c = ecom1_conical
        .acceleration_with_sun(r_conical, v_conical, s_conical)
        .expect("conical ECOM1 accel");
    let p1_c = ecom1_conical
        .parameter_partials_with_sun(r_conical, v_conical, s_conical)
        .expect("conical ECOM1 param partials");
    let ad1_c = ecom1_conical
        .acceleration_partials_with_sun(r_conical, v_conical, s_conical)
        .expect("conical ECOM1 state partials");
    assert_eq!(p1_c[0], Vector3::new(0.0, inv_km, 0.0)); // D0 -> e_D
    assert_eq!(p1_c[3], Vector3::new(0.0, 0.0, -inv_km)); // Y0 -> e_Y
    assert_eq!(p1_c[6], Vector3::new(-inv_km, 0.0, 0.0)); // B0 -> e_B
    let recon1: Vector3<f64> = ecom1_conical
        .coefficients()
        .to_array()
        .iter()
        .zip(p1_c.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a1_c - recon1).norm() <= 4.0 * f64::EPSILON * a1_c.norm());
    assert!(ad1_c.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad1_c.wrt_velocity.iter().all(|x| x.is_finite()));

    // 2. Conical case where e_D = normalize(sun - sat) itself overflows norm_sq.
    //    With v_conical along +z, Z = r x v is along -y which is parallel to s_diag,
    //    so delta_u is genuinely undefined while e_D, e_Y, e_B are well-defined:
    //    D0-only d4b1 acceleration & state partials succeed (and D0B0 parameter
    //    partials succeed), while d4b1 parameter partials refuse undefined delta_u.
    let s_diag = Vector3::new(0.0, 1.0e152, 0.0);
    let d0_only = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
        d0: -1.0e-7,
        ..Default::default()
    })
    .unwrap()
    .with_shadow_function(ShadowFunction::Conical);
    let a_d0 = d0_only
        .acceleration_with_sun(r_conical, v_conical, s_diag)
        .expect("D0-only conical accel");
    let ad_d0 = d0_only
        .acceleration_partials_with_sun(r_conical, v_conical, s_diag)
        .expect("D0-only conical state partials");
    assert!(matches!(
        d0_only.parameter_partials_with_sun(r_conical, v_conical, s_diag),
        Err(PropagationError::NumericalFailure(_))
    ));
    let d0b0_conical = Ecom2SolarRadiationPressure::new(0, 0, vec![0.0, -1.0e-7, 0.0])
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
    let pp_d0b0 = d0b0_conical
        .parameter_partials_with_sun(r_conical, v_conical, s_diag)
        .expect("D0B0 conical param partials");
    let expected_d0 = Vector3::new(
        -1.0e-7 * (-FRAC_1_SQRT_2) * inv_km,
        -1.0e-7 * FRAC_1_SQRT_2 * inv_km,
        0.0,
    );
    assert!((a_d0 - expected_d0).norm() <= 4.0 * f64::EPSILON * expected_d0.norm());
    assert!(
        (pp_d0b0[1] - Vector3::new(-FRAC_1_SQRT_2 * inv_km, FRAC_1_SQRT_2 * inv_km, 0.0)).norm()
            <= 4.0 * f64::EPSILON * inv_km
    );
    assert!((pp_d0b0[2] - Vector3::new(0.0, 0.0, -inv_km)).norm() <= 4.0 * f64::EPSILON * inv_km);
    assert!(ad_d0.wrt_position.iter().all(|x| x.is_finite()));
    assert_eq!(ad_d0.wrt_velocity, Matrix3::zeros());
    // Analytical d(D0 * e_D)/d(r_z) = -D0 / |sun - sat| * e_z.
    let expected_daz_drz = -(-1.0e-7) / (2.0_f64.sqrt() * 1.0e155);
    assert!(
        (ad_d0.wrt_position[(2, 2)] - expected_daz_drz).abs()
            <= 4.0 * f64::EPSILON * expected_daz_drz.abs()
    );

    // With velocity along +y (orbit normal along +z, perpendicular to s_diag),
    // delta_u is also well-defined and all three d4b1 methods agree.
    let v_diag = Vector3::new(0.0, 1.0e-160, 0.0);
    let pp_d0_diag = d0_only
        .parameter_partials_with_sun(r_conical, v_diag, s_diag)
        .expect("D0-only d4b1 param partials with defined delta_u");
    assert!(
        (pp_d0_diag[3] - Vector3::new(-FRAC_1_SQRT_2 * inv_km, FRAC_1_SQRT_2 * inv_km, 0.0)).norm()
            <= 4.0 * f64::EPSILON * inv_km
    );
    assert!(
        (pp_d0_diag[8] - Vector3::new(0.0, 0.0, -inv_km)).norm() <= 4.0 * f64::EPSILON * inv_km
    );

    // 3. Default ApparentDisc case where X = normalize(Y x Z) overflows norm_sq
    //    and Y = normalize(Z x sun) overflows norm_sq.grad.
    let r_disc = Vector3::new(26_560.0, 0.0, 26_560.0);
    let v_disc = Vector3::new(-1.0e148, 0.0, 1.0e148);
    let s_disc = Vector3::new(1.0e-9, AU_KM, 0.0);
    let m_disc = Vector3::new(0.0, 0.0, 384_400.0);

    let geom = ecom_geometry(r_disc, v_disc, s_disc).expect("ApparentDisc geometry");
    assert!(
        (geom.delta_u_rad.expect("delta_u") - FRAC_PI_4).abs() <= 1.0e-14,
        "expected delta_u = pi/4, got {:?}",
        geom.delta_u_rad
    );
    assert!(
        (geom.argument_of_latitude_rad.expect("u") - FRAC_PI_4).abs() <= 1.0e-14,
        "expected u = pi/4, got {:?}",
        geom.argument_of_latitude_rad
    );

    let ecom2_def = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let a2_d = ecom2_def
        .acceleration_with_sun_moon(r_disc, v_disc, s_disc, m_disc)
        .expect("default ECOM2 accel");
    let ad2_d = ecom2_def
        .acceleration_partials_with_sun_moon(r_disc, v_disc, s_disc, m_disc)
        .expect("default ECOM2 state partials");
    let pp2_d = ecom2_def
        .parameter_partials_with_sun_moon(r_disc, v_disc, s_disc, m_disc)
        .expect("default ECOM2 param partials");

    // Reference evaluated with nominal velocity magnitude along the exact same direction
    // (delta_u depends only on the direction of v).
    let v_disc_unit = Vector3::new(-3.0, 0.0, 3.0);
    let a2_ref = ecom2_def
        .acceleration_with_sun_moon(r_disc, v_disc_unit, s_disc, m_disc)
        .expect("reference ECOM2 accel");
    let pp2_ref = ecom2_def
        .parameter_partials_with_sun_moon(r_disc, v_disc_unit, s_disc, m_disc)
        .expect("reference ECOM2 param partials");
    assert!((a2_d - a2_ref).norm() <= 4.0 * f64::EPSILON * a2_ref.norm());
    for (p, p_ref) in pp2_d.iter().zip(pp2_ref.iter()) {
        assert!((p - p_ref).norm() <= 4.0 * f64::EPSILON * p_ref.norm().max(inv_km));
    }
    assert!(ad2_d.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad2_d.wrt_velocity.iter().all(|x| x.is_finite()));
    // Scaling velocity by alpha scales da/dv by 1/alpha and leaves da/dr unchanged.
    let ad2_ref = ecom2_def
        .acceleration_partials_with_sun_moon(r_disc, v_disc_unit, s_disc, m_disc)
        .expect("reference ECOM2 state partials");
    assert!(
        (ad2_d.wrt_position - ad2_ref.wrt_position).norm() <= 1.0e-12 * ad2_ref.wrt_position.norm()
    );
    let v_scale = 1.0e148 / 3.0;
    assert!(
        (ad2_d.wrt_velocity * v_scale - ad2_ref.wrt_velocity).norm()
            <= 1.0e-12 * ad2_ref.wrt_velocity.norm()
    );
}

#[test]
fn parameter_partials_succeed_on_exact_annular_lunar_eclipse() {
    let r = Vector3::new(7_000.0, 0.0, 0.0);
    let v = Vector3::new(0.0, 5.0, 5.0);
    let s = Vector3::new(7_000.0, AU_KM, 0.0);
    let m = Vector3::new(7_000.0, 405_000.0, 0.0);

    // Confirm field lighting fails on exact annular collinearity while double lighting succeeds.
    assert!(matches!(
        lighting_ratio_field(
            r,
            s,
            Some(m),
            ShadowFunction::ApparentDisc,
            EarthShadowModel::Wgs84Oblate,
            true
        ),
        Err(PropagationError::NumericalFailure(_))
    ));
    let nu = lighting_ratio_double(
        r,
        s,
        Some(m),
        ShadowFunction::ApparentDisc,
        EarthShadowModel::Wgs84Oblate,
        true,
    )
    .expect("double annular ratio");
    assert!(
        nu > 0.14 && nu < 0.16,
        "expected annular nu ~ 0.149, got {nu}"
    );

    let inv_km = 1.0 / M_PER_KM;
    let e_d = Vector3::new(0.0, 1.0, 0.0);
    let e_y = Vector3::new(0.0, 0.0, -1.0);
    let e_b = Vector3::new(-1.0, 0.0, 0.0);

    // ECOM2 (default ApparentDisc + Wgs84Oblate + Moon)
    let ecom2 = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let a2 = ecom2
        .acceleration_with_sun_moon(r, v, s, m)
        .expect("annular ECOM2 accel");
    let pp2 = ecom2
        .parameter_partials_with_sun_moon(r, v, s, m)
        .expect("annular ECOM2 parameter partials");
    assert!(matches!(
        ecom2.acceleration_partials_with_sun_moon(r, v, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!((pp2[0] - e_b * (nu * inv_km)).norm() <= 4.0 * f64::EPSILON * inv_km);
    assert!((pp2[3] - e_d * (nu * inv_km)).norm() <= 4.0 * f64::EPSILON * inv_km);
    assert!((pp2[8] - e_y * (nu * inv_km)).norm() <= 4.0 * f64::EPSILON * inv_km);
    let recon2: Vector3<f64> = ecom2
        .coefficients()
        .iter()
        .zip(pp2.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a2 - recon2).norm() <= 4.0 * f64::EPSILON * a2.norm());

    // ECOM1 (default ApparentDisc + Wgs84Oblate + Moon)
    let ecom1 = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    let a1 = ecom1
        .acceleration_with_sun_moon(r, v, s, m)
        .expect("annular ECOM1 accel");
    let pp1 = ecom1
        .parameter_partials_with_sun_moon(r, v, s, m)
        .expect("annular ECOM1 parameter partials");
    assert!(matches!(
        ecom1.acceleration_partials_with_sun_moon(r, v, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!((pp1[0] - e_d * (nu * inv_km)).norm() <= 4.0 * f64::EPSILON * inv_km);
    assert!((pp1[3] - e_y * (nu * inv_km)).norm() <= 4.0 * f64::EPSILON * inv_km);
    assert!((pp1[6] - e_b * (nu * inv_km)).norm() <= 4.0 * f64::EPSILON * inv_km);
    let recon1: Vector3<f64> = ecom1
        .coefficients()
        .to_array()
        .iter()
        .zip(pp1.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a1 - recon1).norm() <= 4.0 * f64::EPSILON * a1.norm());
}

#[test]
fn ecom_geometry_defaults_to_wgs84_oblate_in_polar_penumbra() {
    let sat = Vector3::new(-7_000.0, 0.0, 6_350.0);
    let vel = Vector3::new(0.0, 7.5, 0.0);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);

    let g_default = ecom_geometry(sat, vel, sun).expect("default ecom_geometry");
    let g_wgs84 = ecom_geometry_with_shadow(sat, vel, sun, None, EarthShadowModel::Wgs84Oblate)
        .expect("WGS84 ecom_geometry");
    let g_spherical = ecom_geometry_with_shadow(sat, vel, sun, None, EarthShadowModel::Spherical)
        .expect("spherical ecom_geometry");

    assert_eq!(g_default, g_wgs84);
    assert_eq!(
        g_default.lighting_ratio.to_bits(),
        g_wgs84.lighting_ratio.to_bits()
    );
    assert!(
        g_default.lighting_ratio > 0.2 && g_default.lighting_ratio < 0.8,
        "expected polar penumbra under WGS84, got {}",
        g_default.lighting_ratio
    );
    assert!(
        g_spherical.lighting_ratio < 0.15,
        "expected deep penumbra/umbra under spherical Earth, got {}",
        g_spherical.lighting_ratio
    );
    assert!(
        (g_default.lighting_ratio - g_spherical.lighting_ratio).abs() > 0.3,
        "WGS84 ({}) and spherical ({}) must differ sharply in polar penumbra",
        g_default.lighting_ratio,
        g_spherical.lighting_ratio
    );
}

#[test]
fn near_axis_finite_norm_preserves_lit_umbra_and_annular_fallback() {
    use crate::astro::events::eclipse::{
        shadow_fraction_with_model_moon_and_gradient, ShadowGradientError,
    };

    let ecom2 = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let ecom1 = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    let inv_km = 1.0 / M_PER_KM;

    // 1. Sunlit LEO 1e-165 km off the Earth-Sun line: plane_normal.orthogonal()
    //    overflows gradients while norm values remain finite; sunlit branch
    //    discards angle gradients and returns lighting 1.0 (shadow 0.0, grad 0).
    let sat_lit = Vector3::new(7_000.0, 1.0e-165, 0.0);
    let vel_lit = Vector3::new(0.0, 0.0, 7.5);
    let sun_x = Vector3::new(AU_KM, 0.0, 0.0);
    let moon_clear = Vector3::new(0.0, 384_400.0, 0.0);

    for model in [EarthShadowModel::Wgs84Oblate, EarthShadowModel::Spherical] {
        let (sh, grad) = shadow_fraction_with_model_moon_and_gradient(
            as_array(sat_lit),
            as_array(sun_x),
            as_array(moon_clear),
            model,
        )
        .expect("sunlit near-axis shadow gradient");
        assert_eq!(sh, 0.0);
        assert_eq!(grad, [0.0; 3]);

        let e2 = ecom2.clone().with_earth_shadow_model(model);
        let a2 = e2
            .acceleration_with_sun_moon(sat_lit, vel_lit, sun_x, moon_clear)
            .expect("sunlit near-axis ECOM2 accel");
        let pp2 = e2
            .parameter_partials_with_sun_moon(sat_lit, vel_lit, sun_x, moon_clear)
            .expect("sunlit near-axis ECOM2 param partials");
        let ad2 = e2
            .acceleration_partials_with_sun_moon(sat_lit, vel_lit, sun_x, moon_clear)
            .expect("sunlit near-axis ECOM2 state partials");
        assert!(a2.iter().all(|x| x.is_finite()) && a2.norm() > 0.0);
        assert!(pp2.iter().all(|v| v.iter().all(|x| x.is_finite())));
        assert!(ad2.wrt_position.iter().all(|x| x.is_finite()));
        assert!(ad2.wrt_velocity.iter().all(|x| x.is_finite()));

        let e1 = ecom1.with_earth_shadow_model(model);
        let a1 = e1
            .acceleration_with_sun_moon(sat_lit, vel_lit, sun_x, moon_clear)
            .expect("sunlit near-axis ECOM1 accel");
        let pp1 = e1
            .parameter_partials_with_sun_moon(sat_lit, vel_lit, sun_x, moon_clear)
            .expect("sunlit near-axis ECOM1 param partials");
        let ad1 = e1
            .acceleration_partials_with_sun_moon(sat_lit, vel_lit, sun_x, moon_clear)
            .expect("sunlit near-axis ECOM1 state partials");
        assert!(a1.iter().all(|x| x.is_finite()) && a1.norm() > 0.0);
        assert!(pp1.iter().all(|v| v.iter().all(|x| x.is_finite())));
        assert!(ad1.wrt_position.iter().all(|x| x.is_finite()));
        assert!(ad1.wrt_velocity.iter().all(|x| x.is_finite()));
    }

    // 2. Deep Earth umbra and total lunar umbra 1e-165 km off axis: umbra branch
    //    discards angle gradients and returns lighting 0.0 (shadow 1.0, grad 0).
    let umbra_cases = [
        (
            "earth_umbra",
            Vector3::new(-7_000.0, 1.0e-165, 0.0),
            Vector3::new(0.0, 0.0, 7.5),
            Vector3::new(AU_KM, 0.0, 0.0),
            Vector3::new(0.0, 384_400.0, 0.0),
        ),
        (
            "lunar_umbra",
            Vector3::new(7_000.0, 0.0, 0.0),
            Vector3::new(0.0, 5.0, 5.0),
            Vector3::new(7_000.0, AU_KM, 0.0),
            Vector3::new(7_000.0, 356_000.0, 1.0e-165),
        ),
    ];
    for (label, sat, vel, sun, moon) in umbra_cases {
        for model in [EarthShadowModel::Wgs84Oblate, EarthShadowModel::Spherical] {
            let (sh, grad) = shadow_fraction_with_model_moon_and_gradient(
                as_array(sat),
                as_array(sun),
                as_array(moon),
                model,
            )
            .unwrap_or_else(|e| panic!("{label} {model:?}: unexpected {e:?}"));
            assert_eq!(sh, 1.0, "{label} {model:?}");
            assert_eq!(grad, [0.0; 3], "{label} {model:?}");

            let e2 = ecom2.clone().with_earth_shadow_model(model);
            assert_eq!(
                e2.acceleration_with_sun_moon(sat, vel, sun, moon)
                    .expect("umbra ECOM2 accel"),
                Vector3::zeros()
            );
            assert_eq!(
                e2.parameter_partials_with_sun_moon(sat, vel, sun, moon)
                    .expect("umbra ECOM2 param partials"),
                vec![Vector3::zeros(); ECOM2_D4B1_COEFFICIENT_COUNT]
            );
            assert_eq!(
                e2.acceleration_partials_with_sun_moon(sat, vel, sun, moon)
                    .expect("umbra ECOM2 state partials"),
                AccelerationPartials::zeros()
            );

            let e1 = ecom1.with_earth_shadow_model(model);
            assert_eq!(
                e1.acceleration_with_sun_moon(sat, vel, sun, moon)
                    .expect("umbra ECOM1 accel"),
                Vector3::zeros()
            );
            assert_eq!(
                e1.parameter_partials_with_sun_moon(sat, vel, sun, moon)
                    .expect("umbra ECOM1 param partials"),
                [Vector3::zeros(); ECOM1_COEFFICIENT_COUNT]
            );
            assert_eq!(
                e1.acceleration_partials_with_sun_moon(sat, vel, sun, moon)
                    .expect("umbra ECOM1 state partials"),
                AccelerationPartials::zeros()
            );
        }
    }

    // 3. Near-axis annular lunar eclipse (1e-165 km off axis): field lighting
    //    propagates non-finite gradients to NumericalFailure, triggering the
    //    double fallback for parameter partials while state partials and shadow
    //    gradient report NumericalFailure.
    let sat_ann = Vector3::new(7_000.0, 0.0, 0.0);
    let vel_ann = Vector3::new(0.0, 5.0, 5.0);
    let sun_ann = Vector3::new(7_000.0, AU_KM, 0.0);
    let moon_ann = Vector3::new(7_000.0, 405_000.0, 1.0e-165);

    assert_eq!(
        shadow_fraction_with_model_moon_and_gradient(
            as_array(sat_ann),
            as_array(sun_ann),
            as_array(moon_ann),
            EarthShadowModel::Wgs84Oblate,
        ),
        Err(ShadowGradientError::NumericalFailure)
    );
    assert!(matches!(
        lighting_ratio_field(
            sat_ann,
            sun_ann,
            Some(moon_ann),
            ShadowFunction::ApparentDisc,
            EarthShadowModel::Wgs84Oblate,
            true,
        ),
        Err(PropagationError::NumericalFailure(_))
    ));
    let nu = lighting_ratio_double(
        sat_ann,
        sun_ann,
        Some(moon_ann),
        ShadowFunction::ApparentDisc,
        EarthShadowModel::Wgs84Oblate,
        true,
    )
    .expect("double annular lighting");
    assert!(
        nu > 0.0 && nu < 1.0 && nu.is_finite(),
        "expected finite annular lighting ratio in (0, 1), got {nu}"
    );

    let a2 = ecom2
        .acceleration_with_sun_moon(sat_ann, vel_ann, sun_ann, moon_ann)
        .expect("near-axis annular ECOM2 accel");
    let pp2 = ecom2
        .parameter_partials_with_sun_moon(sat_ann, vel_ann, sun_ann, moon_ann)
        .expect("near-axis annular ECOM2 parameter partials via double fallback");
    assert!(matches!(
        ecom2.acceleration_partials_with_sun_moon(sat_ann, vel_ann, sun_ann, moon_ann),
        Err(PropagationError::NumericalFailure(_))
    ));
    let recon2: Vector3<f64> = ecom2
        .coefficients()
        .iter()
        .zip(pp2.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a2 - recon2).norm() <= 4.0 * f64::EPSILON * a2.norm());
    assert!((pp2[3].norm() - nu * inv_km).abs() <= 4.0 * f64::EPSILON * inv_km);

    let a1 = ecom1
        .acceleration_with_sun_moon(sat_ann, vel_ann, sun_ann, moon_ann)
        .expect("near-axis annular ECOM1 accel");
    let pp1 = ecom1
        .parameter_partials_with_sun_moon(sat_ann, vel_ann, sun_ann, moon_ann)
        .expect("near-axis annular ECOM1 parameter partials via double fallback");
    assert!(matches!(
        ecom1.acceleration_partials_with_sun_moon(sat_ann, vel_ann, sun_ann, moon_ann),
        Err(PropagationError::NumericalFailure(_))
    ));
    let recon1: Vector3<f64> = ecom1
        .coefficients()
        .to_array()
        .iter()
        .zip(pp1.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a1 - recon1).norm() <= 4.0 * f64::EPSILON * a1.norm());
    assert!((pp1[0].norm() - nu * inv_km).abs() <= 4.0 * f64::EPSILON * inv_km);
}

#[test]
fn normalize_prescaling_exact_on_subnormal_and_max_exponent_vectors() {
    // 1. Subnormal unit z component (rounds to 2^-1074 = 0x1, must not flush to 0).
    let v1 = Vec3::new(
        1.99 * libm::ldexp(1.0, 600),
        1.99 * libm::ldexp(1.0, 600),
        1.49 * libm::ldexp(1.0, -474),
    );
    let u1 = v1.normalize().expect("v1 unit");
    assert_eq!(u1.z.to_bits(), 0x1, "got 0x{:016x}", u1.z.to_bits());
    let f1 = FieldVec3::<3>::new(
        Grad::variable(0, v1.x),
        Grad::variable(1, v1.y),
        Grad::variable(2, v1.z),
    )
    .normalize()
    .expect("f1 unit");
    assert_eq!(f1.x.value.to_bits(), u1.x.to_bits());
    assert_eq!(f1.y.value.to_bits(), u1.y.to_bits());
    assert_eq!(f1.z.value.to_bits(), 0x1);
    assert!(field_vec3_is_all_finite(f1));

    // 2. Subnormal unit y component that double-rounds if prescaled by -exp.
    let v2 = Vec3::new(
        1.5 * libm::ldexp(1.0, 512),
        (1.0 + 3.0 * f64::EPSILON) * libm::ldexp(1.0, -511),
        0.0,
    );
    let u2 = v2.normalize().expect("v2 unit");
    assert_eq!(u2.x, 1.0);
    assert_eq!(
        u2.y.to_bits(),
        0x0005_5555_5555_5556,
        "got 0x{:016x}",
        u2.y.to_bits()
    );
    let f2 = FieldVec3::<3>::new(
        Grad::variable(0, v2.x),
        Grad::variable(1, v2.y),
        Grad::variable(2, v2.z),
    )
    .normalize()
    .expect("f2 unit");
    assert_eq!(f2.y.value.to_bits(), 0x0005_5555_5555_5556);
    assert!(field_vec3_is_all_finite(f2));

    // 3. Max-exponent vector (exp == 1023): must follow IEEE round-to-even, not
    //    scalb's n == -1023 round-half-up branch.
    let v3a = Vec3::new(libm::ldexp(1.0, 1023), 1.0 + f64::EPSILON, 0.0);
    let u3a = v3a.normalize().expect("v3a unit");
    assert_eq!(u3a.x, 1.0);
    assert_eq!(
        u3a.y.to_bits(),
        0x0008_0000_0000_0000,
        "got 0x{:016x}",
        u3a.y.to_bits()
    );
    let f3a = FieldVec3::<3>::new(
        Grad::variable(0, v3a.x),
        Grad::variable(1, v3a.y),
        Grad::variable(2, v3a.z),
    )
    .normalize()
    .expect("f3a unit");
    assert_eq!(f3a.y.value.to_bits(), 0x0008_0000_0000_0000);
    assert!(field_vec3_is_all_finite(f3a));

    let v3b = Vec3::new(libm::ldexp(1.0, 1023), f64::EPSILON, 0.0);
    let u3b = v3b.normalize().expect("v3b unit");
    assert_eq!(u3b.x, 1.0);
    assert_eq!(u3b.y.to_bits(), 0x0, "got 0x{:016x}", u3b.y.to_bits());
    let f3b = FieldVec3::<3>::new(
        Grad::variable(0, v3b.x),
        Grad::variable(1, v3b.y),
        Grad::variable(2, v3b.z),
    )
    .normalize()
    .expect("f3b unit");
    assert_eq!(f3b.y.value.to_bits(), 0x0);
    assert!(field_vec3_is_all_finite(f3b));

    // 4. Large-exponent FieldVec3 gradient cancellation: d(x/|x|)/dx == 0.
    let f_big = FieldVec3::<3>::new(
        Grad::variable(0, libm::ldexp(1.0, 996)),
        Grad::ZERO,
        Grad::ZERO,
    )
    .normalize()
    .expect("f_big unit");
    assert_eq!(f_big.x.value, 1.0);
    assert_eq!(f_big.x.grad, [0.0; 3]);
    assert_eq!(f_big.y.grad, [0.0; 3]);
    assert_eq!(f_big.z.grad, [0.0; 3]);
}

#[test]
fn atan2_and_delta_u_handle_overflowing_intermediates_and_multiscale_vectors() {
    use std::f64::consts::FRAC_PI_4;

    // 1. Grad::atan2 numerator overflow with finite denominator.
    let r_a = Vector3::new(1.0e150, 0.0, 0.0);
    let v_a = Vector3::new(0.0, 0.0, 1.0e-160);
    let s_a = Vector3::new(1.0e150, 1.0e100, 0.0);
    let ecom2_conical = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1())
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
    let a_a = ecom2_conical
        .acceleration_with_sun(r_a, v_a, s_a)
        .expect("conical atan2 accel");
    let pp_a = ecom2_conical
        .parameter_partials_with_sun(r_a, v_a, s_a)
        .expect("conical atan2 param partials");
    let ad_a = ecom2_conical
        .acceleration_partials_with_sun(r_a, v_a, s_a)
        .expect("conical atan2 state partials");
    let recon_a: Vector3<f64> = ecom2_conical
        .coefficients()
        .iter()
        .zip(pp_a.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a_a - recon_a).norm() <= 4.0 * f64::EPSILON * a_a.norm());
    assert!(ad_a.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad_a.wrt_velocity.iter().all(|x| x.is_finite()));
    let c_d4b1 = mixed_d4b1();
    // Exact d(delta_u)/d(vel_y) = -s_y / (s_x * v_z_ms) = -1e100 / (1e150 * 1e-157) = -1e107 rad/(m/s).
    let d_du_dv_y = -1.0e107;
    let expected_da_dvy_x = -c_d4b1.b1s * d_du_dv_y;
    let expected_da_dvy_y = (2.0 * c_d4b1.d2s + 4.0 * c_d4b1.d4s) * d_du_dv_y;
    assert!(
        (ad_a.wrt_velocity[(0, 1)] - expected_da_dvy_x).abs() <= 1.0e-12 * expected_da_dvy_x.abs()
    );
    assert!(
        (ad_a.wrt_velocity[(1, 1)] - expected_da_dvy_y).abs() <= 1.0e-12 * expected_da_dvy_y.abs()
    );

    // 2. ApparentDisc with extreme velocity where y = normalize(z x sun) succeeds
    //    but y x z overflows in gradient.
    let r_b = Vector3::new(26_560.0, 0.0, 26_560.0);
    let v_b = Vector3::new(-1.0e290, 0.0, 1.0e290);
    let s_b = Vector3::new(1.0e-9, AU_KM, 0.0);
    let m_b = Vector3::new(0.0, 0.0, 384_400.0);
    let v_b_ref = Vector3::new(-3.0, 0.0, 3.0);

    let g_b = ecom_geometry_with_shadow(r_b, v_b, s_b, Some(m_b), EarthShadowModel::Wgs84Oblate)
        .expect("ApparentDisc geometry with v=1e290");
    assert!((g_b.delta_u_rad.expect("delta_u") - FRAC_PI_4).abs() <= 1.0e-14);
    assert!((g_b.argument_of_latitude_rad.expect("u") - FRAC_PI_4).abs() <= 1.0e-14);

    let ecom2_def = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let a_b = ecom2_def
        .acceleration_with_sun_moon(r_b, v_b, s_b, m_b)
        .expect("ApparentDisc ECOM2 accel with v=1e290");
    let pp_b = ecom2_def
        .parameter_partials_with_sun_moon(r_b, v_b, s_b, m_b)
        .expect("ApparentDisc ECOM2 param partials with v=1e290");
    let ad_b = ecom2_def
        .acceleration_partials_with_sun_moon(r_b, v_b, s_b, m_b)
        .expect("ApparentDisc ECOM2 state partials with v=1e290");
    let a_b_ref = ecom2_def
        .acceleration_with_sun_moon(r_b, v_b_ref, s_b, m_b)
        .expect("ref accel");
    let ad_b_ref = ecom2_def
        .acceleration_partials_with_sun_moon(r_b, v_b_ref, s_b, m_b)
        .expect("ref partials");
    assert!((a_b - a_b_ref).norm() <= 4.0 * f64::EPSILON * a_b_ref.norm());
    assert!(pp_b.iter().all(|v| v.iter().all(|x| x.is_finite())));
    assert!(
        (ad_b.wrt_position - ad_b_ref.wrt_position).norm()
            <= 1.0e-12 * ad_b_ref.wrt_position.norm()
    );
    let v_scale = 1.0e290 / 3.0;
    assert!(
        (ad_b.wrt_velocity * v_scale - ad_b_ref.wrt_velocity).norm()
            <= 1.0e-12 * ad_b_ref.wrt_velocity.norm()
    );

    let ecom1_def = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    let a1_b = ecom1_def
        .acceleration_with_sun_moon(r_b, v_b, s_b, m_b)
        .expect("ApparentDisc ECOM1 accel with v=1e290");
    let pp1_b = ecom1_def
        .parameter_partials_with_sun_moon(r_b, v_b, s_b, m_b)
        .expect("ApparentDisc ECOM1 param partials with v=1e290");
    let ad1_b = ecom1_def
        .acceleration_partials_with_sun_moon(r_b, v_b, s_b, m_b)
        .expect("ApparentDisc ECOM1 state partials with v=1e290");
    let a1_b_ref = ecom1_def
        .acceleration_with_sun_moon(r_b, v_b_ref, s_b, m_b)
        .expect("ECOM1 ref accel");
    assert!((a1_b - a1_b_ref).norm() <= 4.0 * f64::EPSILON * a1_b_ref.norm());
    assert!(pp1_b.iter().all(|v| v.iter().all(|x| x.is_finite())));
    assert!(ad1_b.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad1_b.wrt_velocity.iter().all(|x| x.is_finite()));

    // 3. Multi-scale z vector where z = (-1e208, 1e-142, 0) spans >1024 bits
    //    and unit-order scaling of z would flush z.y to 0.0.
    let r_c = Vector3::new(0.0, 0.0, 1.0e152);
    let v_c = Vector3::new(1.0e-300, 1.0e50, 0.0);
    let s_c = Vector3::new(1.0e152, 0.0, 0.0);
    let a_c = ecom2_conical
        .acceleration_with_sun(r_c, v_c, s_c)
        .expect("multiscale z ECOM2 accel");
    let pp_c = ecom2_conical
        .parameter_partials_with_sun(r_c, v_c, s_c)
        .expect("multiscale z ECOM2 param partials");
    let ad_c = ecom2_conical
        .acceleration_partials_with_sun(r_c, v_c, s_c)
        .expect("multiscale z ECOM2 state partials");
    let recon_c: Vector3<f64> = ecom2_conical
        .coefficients()
        .iter()
        .zip(pp_c.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a_c - recon_c).norm() <= 4.0 * f64::EPSILON * a_c.norm());
    assert!(ad_c.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad_c.wrt_velocity.iter().all(|x| x.is_finite()));
    // At delta_u = -pi/2, B1c*cos(-pi/2) = 0 and B1s*sin(-pi/2) = -B1s.
    let inv_km = 1.0 / M_PER_KM;
    assert!((pp_c[1]).norm() <= 1.0e-15 * inv_km);
    assert!((pp_c[2] + pp_c[0]).norm() <= 4.0 * f64::EPSILON * inv_km);

    // 4. Grad::atan2_with_fn denominator overflow to +Inf must use the scaled fallback
    //    rather than multiplying finite numerators by 1/+Inf = 0.
    let g_atan2 = Grad::<2>::variable(0, 1.0e200).atan2(Grad::variable(1, 1.0e200));
    assert_eq!(
        g_atan2.value.to_bits(),
        fast_math::atan2(1.0e200, 1.0e200).to_bits()
    );
    assert!(
        (g_atan2.grad[0] - 5.0e-201).abs() <= 4.0 * f64::EPSILON * 5.0e-201,
        "expected 5.0e-201, got {:e}",
        g_atan2.grad[0]
    );
    assert!(
        (g_atan2.grad[1] + 5.0e-201).abs() <= 4.0 * f64::EPSILON * 5.0e-201,
        "expected -5.0e-201, got {:e}",
        g_atan2.grad[1]
    );

    let d2c_conical = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
        d2c: 1.0,
        ..Default::default()
    })
    .unwrap()
    .with_shadow_function(ShadowFunction::Conical);
    let ad_denom_ovf = d2c_conical
        .acceleration_partials_with_sun(
            Vector3::new(1.0e152, 0.0, 0.0),
            Vector3::new(0.0, 1.0e-100, 0.0),
            Vector3::new(1.0e10, 1.0e10, 0.0),
        )
        .expect("atan2 denom overflow ECOM2 partials");
    assert!(ad_denom_ovf.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad_denom_ovf.wrt_velocity.iter().all(|x| x.is_finite()));
    let expected_dax_dry = -2.0e-155;
    assert!(
        (ad_denom_ovf.wrt_position[(0, 1)] - expected_dax_dry).abs()
            <= 4.0 * f64::EPSILON * expected_dax_dry.abs(),
        "expected {expected_dax_dry:e}, got {:e}",
        ad_denom_ovf.wrt_position[(0, 1)]
    );
}

#[test]
fn scaled_field_normalize_preserves_sub_f64_unit_components_for_cancellation() {
    let model = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let r = Vector3::new(7_000.0, 0.0, 0.0);
    let v = Vector3::new(0.0, 5.0e-324, 0.0);
    let s = Vector3::new(3.0, 5.0e-324, 1.0e8);
    let m = Vector3::new(0.0, 384_400.0, 0.0);

    // 1. Direct ScaledFieldVec3 check on the exact input where y_raw.x / y_raw.y = -1.647e-324
    //    flushes to 0.0 in binary64. Scaling v_y leaves Z direction invariant, so column 4
    //    (dv_y) of normalized y and x must be identically zero.
    //    (Note: for this input, the true d(delta_u)/dv_z = -(s_z/s_x)/v_y_m ~ -6.75e327
    //    exceeds f64::MAX, so full 6-column acceleration_partials returns NumericalFailure.)
    let r_m = to_vec3_m(r).unwrap();
    let s_m = to_vec3_m(s).unwrap();
    let sat_f = FieldVec3::<6>::new(
        Grad::variable(0, r_m.x),
        Grad::variable(1, r_m.y),
        Grad::variable(2, r_m.z),
    );
    let sun_f = FieldVec3::<6>::constant(s_m);
    let sat_s = ScaledFieldVec3::from_field_vec3(sat_f);
    let sun_s = ScaledFieldVec3::from_field_vec3(sun_f);
    let m_per_km = ScaledF64::from_f64(M_PER_KM);
    let one = ScaledF64::from_f64(1.0);
    let mut vxg = [ScaledF64::ZERO; 6];
    let mut vyg = [ScaledF64::ZERO; 6];
    let mut vzg = [ScaledF64::ZERO; 6];
    vxg[3] = one;
    vyg[4] = one;
    vzg[5] = one;
    let vel_s = ScaledFieldVec3 {
        x: ScaledGrad {
            val: ScaledF64::from_f64(v.x).mul(m_per_km),
            grad: vxg,
        },
        y: ScaledGrad {
            val: ScaledF64::from_f64(v.y).mul(m_per_km),
            grad: vyg,
        },
        z: ScaledGrad {
            val: ScaledF64::from_f64(v.z).mul(m_per_km),
            grad: vzg,
        },
    };
    let z_s = sat_s.cross(vel_s);
    let y_s = z_s.cross(sun_s).normalize().expect("normalized y_s");
    assert_eq!(y_s.x.grad[4].to_f64(), 0.0);
    assert_eq!(y_s.y.grad[4].to_f64(), 0.0);
    assert_eq!(y_s.z.grad[4].to_f64(), 0.0);
    let x_s = y_s.cross(z_s).normalize().expect("normalized x_s");
    assert_eq!(x_s.x.grad[4].to_f64(), 0.0);
    assert_eq!(x_s.y.grad[4].to_f64(), 0.0);
    assert_eq!(x_s.z.grad[4].to_f64(), 0.0);
    assert!(matches!(
        model.acceleration_partials_with_sun_moon(r, v, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));

    // 2. Public API checks where y_raw.x / y_raw.y still flushes to 0.0 in binary64
    //    and every column of the true Jacobian is finite: wrt_velocity column 1 (v along +y)
    //    must be identically zero.
    for (r_case, v_case, s_case, label) in [
        (
            r,
            Vector3::new(0.0, 1.0e-290, 0.0),
            s,
            "s=(3, 5e-324, 1e8), v_y=1e-290",
        ),
        (
            Vector3::new(7_000.0, 7_000.0, 0.0),
            v,
            Vector3::new(1.0e8, 5.0e-324, 0.0),
            "r=(7000,7000,0), s=(1e8, 5e-324, 0), v_y=5e-324",
        ),
    ] {
        let ad = model
            .acceleration_partials_with_sun_moon(r_case, v_case, s_case, m)
            .unwrap_or_else(|e| panic!("{label}: unexpected error {e:?}"));
        assert!(ad.wrt_position.iter().all(|x| x.is_finite()), "{label}");
        assert!(ad.wrt_velocity.iter().all(|x| x.is_finite()), "{label}");
        for row in 0..3 {
            assert_eq!(
                ad.wrt_velocity[(row, 1)],
                0.0,
                "{label}: wrt_velocity[({row}, 1)] = {:e}, expected exact 0.0",
                ad.wrt_velocity[(row, 1)]
            );
        }
    }
}

fn raw_hipparchus_normalize_double(v: Vec3) -> Option<Vec3> {
    let n2 = v.norm_sq();
    if n2 == 0.0 || !n2.is_finite() {
        return None;
    }
    let u = v.scalar_mul(1.0 / fast_math::sqrt(n2));
    vec3_is_finite(u).then_some(u)
}

fn raw_hipparchus_normalize_field<const N: usize>(v: FieldVec3<N>) -> Option<FieldVec3<N>> {
    let n2 = v.norm_sq();
    if n2.value == 0.0 || !n2.value.is_finite() {
        return None;
    }
    let s = n2.sqrt();
    let u = v.scalar_mul(s.reciprocal());
    field_vec3_is_all_finite(u).then_some(u)
}

fn raw_hipparchus_atan2_field<const N: usize>(
    y: Grad<N>,
    x: Grad<N>,
    atan2_fn: fn(f64, f64) -> f64,
) -> Option<Grad<N>> {
    y.atan2_unscaled_with_fn(x, atan2_fn)
}

fn raw_orekit_dyb_double(sat_m: Vec3, sun_m: Vec3) -> Option<[Vec3; 3]> {
    let e_d = raw_hipparchus_normalize_double(sun_m.sub(sat_m))?;
    let e_y = raw_hipparchus_normalize_double(e_d.cross(sat_m))?;
    let e_b = e_d.cross(e_y);
    vec3_is_finite(e_b).then_some([e_d, e_y, e_b])
}

fn raw_orekit_delta_u_double(sat_m: Vec3, vel_km_s: Vector3<f64>, sun_m: Vec3) -> Option<f64> {
    let vel_m = to_vec3_m(vel_km_s).ok()?;
    let z = sat_m.cross(vel_m);
    let y = raw_hipparchus_normalize_double(z.cross(sun_m))?;
    let x = raw_hipparchus_normalize_double(y.cross(z))?;
    let py = sat_m.dot(y);
    let px = sat_m.dot(x);
    if py.is_finite() && px.is_finite() && (py != 0.0 || px != 0.0) {
        let du = fast_math::atan2(py, px);
        du.is_finite().then_some(du)
    } else {
        None
    }
}

fn raw_orekit_ecom2_accel_double(
    sat_km: Vector3<f64>,
    vel_km_s: Vector3<f64>,
    sun_km: Vector3<f64>,
    ratio: f64,
    n_d: usize,
    n_b: usize,
    params: &[f64],
) -> Option<Vector3<f64>> {
    let sat_m = to_vec3_m(sat_km).ok()?;
    let sun_m = to_vec3_m(sun_km).ok()?;
    let [e_d, e_y, e_b] = raw_orekit_dyb_double(sat_m, sun_m)?;
    let du = raw_orekit_delta_u_double(sat_m, vel_km_s, sun_m)?;
    let mut b_u = params[0];
    for i in 1..=n_b {
        let (sin, cos) = fast_math::sin_cos(((2 * i - 1) as f64) * du);
        b_u += params[i] * cos + params[i + n_b] * sin;
    }
    let mut d_u = params[2 * n_b + 1];
    for i in 1..=n_d {
        let (sin, cos) = fast_math::sin_cos(((2 * i) as f64) * du);
        d_u += params[2 * n_b + 1 + i] * cos + params[2 * n_b + 1 + i + n_d] * sin;
    }
    let y0 = params[2 * (n_d + n_b) + 2];
    let raw = Vec3::lincomb3(d_u, e_d, y0, e_y, b_u, e_b);
    let scaled = Vec3::from_scaled(ratio, raw);
    let out = Vector3::new(
        scaled.x / M_PER_KM,
        scaled.y / M_PER_KM,
        scaled.z / M_PER_KM,
    );
    out.iter().all(|c| c.is_finite()).then_some(out)
}

#[allow(clippy::too_many_arguments)]
fn raw_orekit_ecom2_field<const N: usize>(
    sat_pos: FieldVec3<N>,
    vel_km_s: Vector3<f64>,
    vel_is_var: bool,
    sun_pos: FieldVec3<N>,
    ratio: Grad<N>,
    is_sunlit: bool,
    n_d: usize,
    n_b: usize,
    params: &[Grad<N>],
) -> Option<FieldVec3<N>> {
    let e_d = raw_hipparchus_normalize_field(sun_pos.sub(sat_pos))?;
    let e_y = raw_hipparchus_normalize_field(e_d.cross(sat_pos))?;
    let e_b = e_d.cross(e_y);
    if !field_vec3_is_all_finite(e_b) {
        return None;
    }
    let vel_m = to_vec3_m(vel_km_s).ok()?;
    let sat_vel = if vel_is_var {
        FieldVec3::<N>::new(
            Grad::variable(3, vel_m.x),
            Grad::variable(4, vel_m.y),
            Grad::variable(5, vel_m.z),
        )
    } else {
        FieldVec3::<N>::constant(vel_m)
    };
    let z = sat_pos.cross(sat_vel);
    let y = raw_hipparchus_normalize_field(z.cross(sun_pos))?;
    let x = raw_hipparchus_normalize_field(y.cross(z))?;
    let py = sat_pos.dot(y);
    let px = sat_pos.dot(x);
    if !(py.is_all_finite() && px.is_all_finite()) {
        return None;
    }
    let du = raw_hipparchus_atan2_field(py, px, fast_math::atan2)?;
    let mut b_u = params[0];
    for i in 1..=n_b {
        let (sin, cos) = du.mul_f64((2 * i - 1) as f64).sin_cos();
        b_u = b_u.add(cos.mul(params[i])).add(sin.mul(params[i + n_b]));
    }
    let mut d_u = params[2 * n_b + 1];
    for i in 1..=n_d {
        let (sin, cos) = du.mul_f64((2 * i) as f64).sin_cos();
        d_u = d_u
            .add(cos.mul(params[2 * n_b + 1 + i]))
            .add(sin.mul(params[2 * n_b + 1 + i + n_d]));
    }
    let y0 = params[2 * (n_d + n_b) + 2];
    let raw = FieldVec3::lincomb3(d_u, e_d, y0, e_y, b_u, e_b);
    let accel = if is_sunlit {
        raw
    } else {
        raw.scalar_mul(ratio)
    };
    field_vec3_is_all_finite(accel).then_some(accel)
}

#[test]
fn subnormal_finite_orekit_intermediates_stay_bit_identical_to_unscaled_orekit() {
    // 1. Default ApparentDisc with subnormal x_n2 in (1 / f64::MAX, SAFE_MIN).
    let model_disc = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let r1 = Vector3::new(26_560.0, 0.0, 0.0);
    let v1 = Vector3::new(0.0, 3.0e-165, 3.0e-165);
    let s1 = Vector3::new(1.0e8, 1.0e8, 0.0);
    let m1 = Vector3::new(0.0, 0.0, 384_400.0);

    let r1_m = to_vec3_m(r1).unwrap();
    let v1_m = to_vec3_m(v1).unwrap();
    let s1_m = to_vec3_m(s1).unwrap();
    let z1 = r1_m.cross(v1_m);
    let y1 = raw_hipparchus_normalize_double(z1.cross(s1_m)).unwrap();
    let x1_n2 = y1.cross(z1).norm_sq();
    assert!(
        x1_n2 > 0.0 && x1_n2 < fast_math::SAFE_MIN,
        "expected subnormal x_n2, got {x1_n2:e}"
    );

    let ratio1_d = lighting_ratio_double(
        r1,
        s1,
        Some(m1),
        ShadowFunction::ApparentDisc,
        EarthShadowModel::Wgs84Oblate,
        true,
    )
    .unwrap();
    let ratio1_f = lighting_ratio_field(
        r1,
        s1,
        Some(m1),
        ShadowFunction::ApparentDisc,
        EarthShadowModel::Wgs84Oblate,
        true,
    )
    .unwrap();

    let raw_dyb1 = raw_orekit_dyb_double(r1_m, s1_m).unwrap();
    let raw_du1 = raw_orekit_delta_u_double(r1_m, v1, s1_m).unwrap();
    let geom1 = ecom_geometry_with_shadow(r1, v1, s1, Some(m1), EarthShadowModel::Wgs84Oblate)
        .expect("subnormal x_n2 geometry");
    assert_eq!(geom1.delta_u_rad.unwrap().to_bits(), raw_du1.to_bits());
    assert_eq!(geom1.dyb.d[0].to_bits(), raw_dyb1[0].x.to_bits());
    assert_eq!(geom1.dyb.y[0].to_bits(), raw_dyb1[1].x.to_bits());
    assert_eq!(geom1.dyb.b[0].to_bits(), raw_dyb1[2].x.to_bits());

    let acc1 = model_disc
        .acceleration_with_sun_moon(r1, v1, s1, m1)
        .expect("subnormal x_n2 accel");
    let raw_acc1 =
        raw_orekit_ecom2_accel_double(r1, v1, s1, ratio1_d, 2, 1, model_disc.coefficients())
            .expect("raw Orekit accel 1");
    for i in 0..3 {
        assert_eq!(acc1[i].to_bits(), raw_acc1[i].to_bits());
    }

    let sat1_f = FieldVec3::<6>::new(
        Grad::variable(0, r1_m.x),
        Grad::variable(1, r1_m.y),
        Grad::variable(2, r1_m.z),
    );
    let sun1_f = FieldVec3::<6>::constant(s1_m);
    let params1_g: Vec<Grad<6>> = model_disc
        .coefficients()
        .iter()
        .map(|&c| Grad::constant(c))
        .collect();
    // For variable velocity on input 1, inv2 * s.grad[4] = -7.88e307 * 2.656e7 overflows to -Inf
    // in unscaled Orekit Field<Gradient>, so acceleration_partials uses ScaledFieldVec3 and
    // matches the exact scaling law from nominal velocity v_nom = (0, 3, 3) km/s.
    assert!(raw_orekit_ecom2_field(
        sat1_f,
        v1,
        true,
        sun1_f,
        Grad::<6>::from_pos3(ratio1_f),
        is_sunlit_field(ratio1_f),
        2,
        1,
        &params1_g,
    )
    .is_none());
    let ad1 = model_disc
        .acceleration_partials_with_sun_moon(r1, v1, s1, m1)
        .expect("subnormal x_n2 state partials via scaled fallback");
    let ad1_nom = model_disc
        .acceleration_partials_with_sun_moon(r1, Vector3::new(0.0, 3.0, 3.0), s1, m1)
        .expect("nominal v state partials");
    assert!(
        (ad1.wrt_position - ad1_nom.wrt_position).norm()
            <= 16.0 * f64::EPSILON * ad1_nom.wrt_position.norm()
    );
    let v1_scale = 3.0e-165 / 3.0;
    assert!(
        (ad1.wrt_velocity * v1_scale - ad1_nom.wrt_velocity).norm()
            <= 16.0 * f64::EPSILON * ad1_nom.wrt_velocity.norm()
    );

    let params1_p: Vec<Grad<9>> = model_disc
        .coefficients()
        .iter()
        .enumerate()
        .map(|(k, &c)| Grad::variable(k, c))
        .collect();
    let raw_pp1_f = raw_orekit_ecom2_field(
        FieldVec3::<9>::constant(r1_m),
        v1,
        false,
        FieldVec3::<9>::constant(s1_m),
        Grad::<9>::constant(ratio1_f.value),
        is_sunlit_field(ratio1_f),
        2,
        1,
        &params1_p,
    )
    .expect("raw Orekit field param partials 1");
    let pp1 = model_disc
        .parameter_partials_with_sun_moon(r1, v1, s1, m1)
        .expect("subnormal x_n2 param partials");
    for k in 0..9 {
        assert_eq!(
            pp1[k].x.to_bits(),
            (raw_pp1_f.x.grad[k] / M_PER_KM).to_bits()
        );
        assert_eq!(
            pp1[k].y.to_bits(),
            (raw_pp1_f.y.grad[k] / M_PER_KM).to_bits()
        );
        assert_eq!(
            pp1[k].z.to_bits(),
            (raw_pp1_f.z.grad[k] / M_PER_KM).to_bits()
        );
    }

    // 2. Conical shadow with subnormal e_Y squared norm in (1 / f64::MAX, SAFE_MIN).
    let model_conical = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1())
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
    let r2 = Vector3::new(0.8e-157, 0.0, 0.7e-157);
    let v2 = Vector3::new(0.0, 1.0, 0.0);
    let s2 = Vector3::new(0.0, 149_597_870.7, 0.0);

    let r2_m = to_vec3_m(r2).unwrap();
    let s2_m = to_vec3_m(s2).unwrap();
    let ed2 = raw_hipparchus_normalize_double(s2_m.sub(r2_m)).unwrap();
    let ey2_n2 = ed2.cross(r2_m).norm_sq();
    assert!(
        ey2_n2 > 0.0 && ey2_n2 < fast_math::SAFE_MIN,
        "expected subnormal e_Y n2, got {ey2_n2:e}"
    );

    let ratio2_d = lighting_ratio_double(
        r2,
        s2,
        None,
        ShadowFunction::Conical,
        EarthShadowModel::Wgs84Oblate,
        false,
    )
    .unwrap();
    let ratio2_f = lighting_ratio_field(
        r2,
        s2,
        None,
        ShadowFunction::Conical,
        EarthShadowModel::Wgs84Oblate,
        false,
    )
    .unwrap();

    let acc2 = model_conical
        .acceleration_with_sun(r2, v2, s2)
        .expect("subnormal e_Y accel");
    let raw_acc2 =
        raw_orekit_ecom2_accel_double(r2, v2, s2, ratio2_d, 2, 1, model_conical.coefficients())
            .expect("raw Orekit accel 2");
    for i in 0..3 {
        assert_eq!(acc2[i].to_bits(), raw_acc2[i].to_bits());
    }

    let sat2_f = FieldVec3::<6>::new(
        Grad::variable(0, r2_m.x),
        Grad::variable(1, r2_m.y),
        Grad::variable(2, r2_m.z),
    );
    let sun2_f = FieldVec3::<6>::constant(s2_m);
    let raw_ad2 = extract_state_partials(
        raw_orekit_ecom2_field(
            sat2_f,
            v2,
            true,
            sun2_f,
            Grad::<6>::from_pos3(ratio2_f),
            is_sunlit_field(ratio2_f),
            2,
            1,
            &params1_g,
        )
        .expect("raw Orekit field state partials 2"),
    );
    let ad2 = model_conical
        .acceleration_partials_with_sun(r2, v2, s2)
        .expect("subnormal e_Y state partials");
    for r_i in 0..3 {
        for c_i in 0..3 {
            assert_eq!(
                ad2.wrt_position[(r_i, c_i)].to_bits(),
                raw_ad2.wrt_position[(r_i, c_i)].to_bits()
            );
            assert_eq!(
                ad2.wrt_velocity[(r_i, c_i)].to_bits(),
                raw_ad2.wrt_velocity[(r_i, c_i)].to_bits()
            );
        }
    }

    let raw_pp2_f = raw_orekit_ecom2_field(
        FieldVec3::<9>::constant(r2_m),
        v2,
        false,
        FieldVec3::<9>::constant(s2_m),
        Grad::<9>::constant(ratio2_f.value),
        is_sunlit_field(ratio2_f),
        2,
        1,
        &params1_p,
    )
    .expect("raw Orekit field param partials 2");
    let pp2 = model_conical
        .parameter_partials_with_sun(r2, v2, s2)
        .expect("subnormal e_Y param partials");
    for k in 0..9 {
        assert_eq!(
            pp2[k].x.to_bits(),
            (raw_pp2_f.x.grad[k] / M_PER_KM).to_bits()
        );
        assert_eq!(
            pp2[k].y.to_bits(),
            (raw_pp2_f.y.grad[k] / M_PER_KM).to_bits()
        );
        assert_eq!(
            pp2[k].z.to_bits(),
            (raw_pp2_f.z.grad[k] / M_PER_KM).to_bits()
        );
    }
}

#[test]
fn ecom1_scaled_u_preserves_subnormal_and_underflowing_products_and_matches_field_bits() {
    let make_pos_vel_grad =
        |r: Vector3<f64>, v: Vector3<f64>| -> (ScaledFieldVec3<6>, ScaledFieldVec3<6>) {
            (
                scaled_field_vec3_variable::<6>(r, 0),
                scaled_field_vec3_variable::<6>(v, 3),
            )
        };
    let rescaled_u_ref = |r: Vector3<f64>, r_shift: i32, v: Vector3<f64>, v_shift: i32| -> f64 {
        let rs = Vector3::new(
            libm::ldexp(r.x, r_shift),
            libm::ldexp(r.y, r_shift),
            libm::ldexp(r.z, r_shift),
        );
        let vs = Vector3::new(
            libm::ldexp(v.x, v_shift),
            libm::ldexp(v.y, v_shift),
            libm::ldexp(v.z, v_shift),
        );
        let normal = rs.cross(&vs).normalize();
        let node = Vector3::new(-normal.y, normal.x, 0.0).normalize();
        let ahead = normal.cross(&node);
        libm::atan2(rs.dot(&ahead), rs.dot(&node))
    };

    // 1. Subnormal |r x v|^2 under default ApparentDisc + WGS84 + Moon.
    let e1_disc = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    let r1 = Vector3::new(5_000.0, 5_000.0, 5_000.0);
    let v1 = Vector3::new(0.0, -2.1e-166, 2.1e-166);
    let s1 = Vector3::new(1.0e8, 1.0e8, 0.0);
    let m1 = Vector3::new(0.0, 0.0, 384_400.0);
    let u1_d = ecom1_u_double(as_array(r1), as_array(v1)).expect("scaled u1 double");
    let (su1_d, cu1_d) = ecom1_sin_cos_double_scaled(
        ScaledVec3::from_vec3(Vec3::new(r1.x, r1.y, r1.z)),
        ScaledVec3::from_vec3(Vec3::new(v1.x, v1.y, v1.z)),
    )
    .expect("scaled sin/cos 1 double");
    let (p1_g, v1_g) = make_pos_vel_grad(r1, v1);
    let (su1_f, cu1_f) = ecom1_sin_cos_field_scaled(p1_g, v1_g).expect("scaled sin/cos 1 field");
    assert_eq!(su1_d.to_f64().to_bits(), su1_f.val.to_f64().to_bits());
    assert_eq!(cu1_d.to_f64().to_bits(), cu1_f.val.to_f64().to_bits());
    let u1_ref = rescaled_u_ref(r1, 0, v1, 560);
    assert!(
        (u1_d - u1_ref).abs() <= 4.0 * f64::EPSILON,
        "u1={u1_d} vs ref={u1_ref}"
    );
    let geom1 = ecom_geometry_with_shadow(r1, v1, s1, Some(m1), EarthShadowModel::Wgs84Oblate)
        .expect("geom1");
    assert_eq!(
        geom1.argument_of_latitude_rad.unwrap().to_bits(),
        u1_d.to_bits()
    );
    let a1 = e1_disc
        .acceleration_with_sun_moon(r1, v1, s1, m1)
        .expect("a1");
    let pp1 = e1_disc
        .parameter_partials_with_sun_moon(r1, v1, s1, m1)
        .expect("pp1");
    let ad1 = e1_disc
        .acceleration_partials_with_sun_moon(r1, v1, s1, m1)
        .expect("ad1");
    let recon1: Vector3<f64> = e1_disc
        .coefficients()
        .to_array()
        .iter()
        .zip(pp1.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a1 - recon1).norm() <= 16.0 * f64::EPSILON * a1.norm());
    let v1_scaled = Vector3::new(
        libm::ldexp(v1.x, 560),
        libm::ldexp(v1.y, 560),
        libm::ldexp(v1.z, 560),
    );
    let a1_ref = e1_disc
        .acceleration_with_sun_moon(r1, v1_scaled, s1, m1)
        .expect("a1_ref");
    let ad1_ref = e1_disc
        .acceleration_partials_with_sun_moon(r1, v1_scaled, s1, m1)
        .expect("ad1_ref");
    assert!((a1 - a1_ref).norm() <= 16.0 * f64::EPSILON * a1_ref.norm());
    assert!(mat3_rel_err(ad1.wrt_position, ad1_ref.wrt_position).unwrap() <= 32.0 * f64::EPSILON);
    let ad1_vel_rescaled = ad1.wrt_velocity * libm::ldexp(1.0, -560);
    assert!(mat3_rel_err(ad1_vel_rescaled, ad1_ref.wrt_velocity).unwrap() <= 32.0 * f64::EPSILON);

    // 2. Normal |r x v|^2 and |node|^2 with underflowing r . ahead / r . node.
    //    Check the subnormal-position input directly on ecom1_u_double,
    //    ecom1_sin_cos_double_scaled, and ecom1_sin_cos_field_scaled (both sin/cos
    //    and their 6 position/velocity partials), and also check end-to-end Conical
    //    evaluation on a companion input whose |e_D x r| is normal while still
    //    having subnormal/underflowing dot3 terms.
    let r2 = Vector3::new(5.0e-324, 5.0e-324, 5.0e-324);
    let v2 = Vector3::new(0.0, 1.0e200, -1.0e200);
    let s2 = Vector3::new(1.0e8, 1.0e8, 0.0);
    let u2_d = ecom1_u_double(as_array(r2), as_array(v2)).expect("scaled u2 double");
    let (su2_d, cu2_d) = ecom1_sin_cos_double_scaled(
        ScaledVec3::from_vec3(Vec3::new(r2.x, r2.y, r2.z)),
        ScaledVec3::from_vec3(Vec3::new(v2.x, v2.y, v2.z)),
    )
    .expect("scaled sin/cos 2 double");
    let (p2_g, v2_g) = make_pos_vel_grad(r2, v2);
    let (su2_f, cu2_f) = ecom1_sin_cos_field_scaled(p2_g, v2_g).expect("scaled sin/cos 2 field");
    assert_eq!(su2_d.to_f64().to_bits(), su2_f.val.to_f64().to_bits());
    assert_eq!(cu2_d.to_f64().to_bits(), cu2_f.val.to_f64().to_bits());
    let u2_ref = rescaled_u_ref(r2, 1074, v2, -660);
    assert!(
        (u2_d - u2_ref).abs() <= 4.0 * f64::EPSILON,
        "u2={u2_d} vs ref={u2_ref}"
    );
    // Check all 6 scaled partials of sin(u2) and cos(u2) against power-of-two rescaled inputs.
    let r2_rescaled = Vector3::new(
        libm::ldexp(r2.x, 1074),
        libm::ldexp(r2.y, 1074),
        libm::ldexp(r2.z, 1074),
    );
    let v2_rescaled = Vector3::new(
        libm::ldexp(v2.x, -660),
        libm::ldexp(v2.y, -660),
        libm::ldexp(v2.z, -660),
    );
    let (p2_ref_g, v2_ref_g) = make_pos_vel_grad(r2_rescaled, v2_rescaled);
    let (su2_ref, cu2_ref) =
        ecom1_sin_cos_field_scaled(p2_ref_g, v2_ref_g).expect("rescaled sin/cos 2 field");
    let scale_r2 = ScaledF64::from_scaled(1.0, 1074);
    let scale_v2 = ScaledF64::from_scaled(1.0, -660);
    for (label, got_g, ref_g) in [("sin", su2_f, su2_ref), ("cos", cu2_f, cu2_ref)] {
        for axis in 0..3 {
            let got_r = got_g.grad[axis].div(scale_r2).to_f64();
            let ref_r = ref_g.grad[axis].to_f64();
            assert!(
                (got_r - ref_r).abs() <= 16.0 * f64::EPSILON * ref_r.abs().max(1.0e-30),
                "u2 d({label})/dr[{axis}]: got={got_r:e} ref={ref_r:e}"
            );
            let got_v = got_g.grad[axis + 3].div(scale_v2).to_f64();
            let ref_v = ref_g.grad[axis + 3].to_f64();
            assert!(
                (got_v - ref_v).abs() <= 16.0 * f64::EPSILON * ref_v.abs().max(1.0e-30),
                "u2 d({label})/dv[{axis}]: got={got_v:e} ref={ref_v:e}"
            );
        }
    }

    // End-to-end Conical check on an input with normal |e_D x r| (accepted by conical
    // shadow) and subnormal dot3 terms in r . ahead / r . node.
    let e1_con = Ecom1SolarRadiationPressure::new(mixed1())
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
    let r2_con = Vector3::new(1.0e-150, 5.0e-324, 5.0e-324);
    let a2 = e1_con.acceleration_with_sun(r2_con, v2, s2).expect("a2");
    let pp2 = e1_con
        .parameter_partials_with_sun(r2_con, v2, s2)
        .expect("pp2");
    let recon2: Vector3<f64> = e1_con
        .coefficients()
        .to_array()
        .iter()
        .zip(pp2.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a2 - recon2).norm() <= 16.0 * f64::EPSILON * a2.norm());
    let ad2 = e1_con
        .acceleration_partials_with_sun(r2_con, v2, s2)
        .expect("ad2 finite partials");
    assert!(ad2.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad2.wrt_velocity.iter().all(|x| x.is_finite()));

    // 3. Normal |r x v|^2 = 1.51e-300 with subnormal value*grad products in ecom1_sin_cos_field_scaled.
    let r3 = Vector3::new(0.0, 1.0e20, 1.0e20);
    let v3 = Vector3::new(0.71e-170, -0.355e-170, 0.355e-170);
    let s3 = Vector3::new(1.0e8, 1.0e8, 0.0);
    let m3 = Vector3::new(0.0, 0.0, 384_400.0);
    let u3_d = ecom1_u_double(as_array(r3), as_array(v3)).expect("scaled u3 double");
    let (su3_d, cu3_d) = ecom1_sin_cos_double_scaled(
        ScaledVec3::from_vec3(Vec3::new(r3.x, r3.y, r3.z)),
        ScaledVec3::from_vec3(Vec3::new(v3.x, v3.y, v3.z)),
    )
    .expect("scaled sin/cos 3 double");
    let (p3_g, v3_g) = make_pos_vel_grad(r3, v3);
    let (su3_f, cu3_f) = ecom1_sin_cos_field_scaled(p3_g, v3_g).expect("scaled sin/cos 3 field");
    assert_eq!(su3_d.to_f64().to_bits(), su3_f.val.to_f64().to_bits());
    assert_eq!(cu3_d.to_f64().to_bits(), cu3_f.val.to_f64().to_bits());
    let u3_ref = rescaled_u_ref(r3, -64, v3, 560);
    assert!(
        (u3_d - u3_ref).abs() <= 4.0 * f64::EPSILON,
        "u3={u3_d} vs ref={u3_ref}"
    );
    let ad3 = e1_disc
        .acceleration_partials_with_sun_moon(r3, v3, s3, m3)
        .expect("ad3");
    let ratio3 = lighting_ratio_field(
        r3,
        s3,
        Some(m3),
        ShadowFunction::ApparentDisc,
        EarthShadowModel::Wgs84Oblate,
        true,
    )
    .unwrap();
    let (_, ref_ad3) = rescaled_state_partials_ref(
        r3,
        v3,
        s3,
        ratio3,
        &mixed_d4b1().to_array(),
        mixed1().to_array(),
    );
    let ref_ad3 = ref_ad3.expect("rescaled ref3").expect("finite ref3");
    assert!(
        mat3_rel_err(ad3.wrt_position, ref_ad3.wrt_position).unwrap() <= 32.0 * f64::EPSILON,
        "ad3 wrt_position mismatch: actual={:?} ref={:?}",
        ad3.wrt_position,
        ref_ad3.wrt_position
    );
    assert!(
        mat3_rel_err(ad3.wrt_velocity, ref_ad3.wrt_velocity).unwrap() <= 32.0 * f64::EPSILON,
        "ad3 wrt_velocity mismatch: actual={:?} ref={:?}",
        ad3.wrt_velocity,
        ref_ad3.wrt_velocity
    );
}

#[test]
fn double_and_field_paths_accept_flushed_frame_and_overflowing_fourier_sums() {
    // 1. Flushed e_D.y = 5e-341 -> e_Y = (0, 0, -1), e_B = (0, 1, 0).
    let r1 = Vector3::new(1.0e140, 0.0, 0.0);
    let v1 = Vector3::new(0.0, 1.0, 1.0);
    let s1 = Vector3::new(3.0e140, 1.0e-200, 0.0);
    let m1 = Vector3::new(0.0, 384_400.0, 0.0);

    let geom1 = ecom_geometry_with_shadow(r1, v1, s1, Some(m1), EarthShadowModel::Wgs84Oblate)
        .expect("flushed e_D geometry");
    assert_eq!(geom1.dyb.d, [1.0, 0.0, 0.0]);
    assert_eq!(geom1.dyb.y, [0.0, 0.0, -1.0]);
    assert_eq!(geom1.dyb.b, [0.0, 1.0, 0.0]);
    assert!(geom1.delta_u_rad.is_some());
    assert!(geom1.argument_of_latitude_rad.is_some());

    let e2_1 = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let a2_1 = e2_1
        .acceleration_with_sun_moon(r1, v1, s1, m1)
        .expect("ECOM2 flushed e_D accel");
    let pp2_1 = e2_1
        .parameter_partials_with_sun_moon(r1, v1, s1, m1)
        .expect("ECOM2 flushed e_D pp");
    let ad2_1 = e2_1
        .acceleration_partials_with_sun_moon(r1, v1, s1, m1)
        .expect("ECOM2 flushed e_D ad");
    let recon2_1: Vector3<f64> = e2_1
        .coefficients()
        .iter()
        .zip(pp2_1.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a2_1 - recon2_1).norm() <= 16.0 * f64::EPSILON * a2_1.norm());
    assert!(ad2_1.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad2_1.wrt_velocity.iter().all(|x| x.is_finite()));
    assert!(mat3_max_abs(ad2_1.wrt_position) > 1.0e185);
    assert_eq!(pp2_1[3], Vector3::new(1.0 / M_PER_KM, 0.0, 0.0));
    assert_eq!(pp2_1[8], Vector3::new(0.0, 0.0, -1.0 / M_PER_KM));
    assert_eq!(pp2_1[0], Vector3::new(0.0, 1.0 / M_PER_KM, 0.0));

    let e1_1 = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    let a1_1 = e1_1
        .acceleration_with_sun_moon(r1, v1, s1, m1)
        .expect("ECOM1 flushed e_D accel");
    let pp1_1 = e1_1
        .parameter_partials_with_sun_moon(r1, v1, s1, m1)
        .expect("ECOM1 flushed e_D pp");
    let ad1_1 = e1_1
        .acceleration_partials_with_sun_moon(r1, v1, s1, m1)
        .expect("ECOM1 flushed e_D ad");
    let recon1_1: Vector3<f64> = e1_1
        .coefficients()
        .to_array()
        .iter()
        .zip(pp1_1.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a1_1 - recon1_1).norm() <= 16.0 * f64::EPSILON * a1_1.norm());
    assert!(ad1_1.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad1_1.wrt_velocity.iter().all(|x| x.is_finite()));
    assert!(mat3_max_abs(ad1_1.wrt_position) > 1.0e185);
    assert_eq!(pp1_1[0], Vector3::new(1.0 / M_PER_KM, 0.0, 0.0));
    assert_eq!(pp1_1[3], Vector3::new(0.0, 0.0, -1.0 / M_PER_KM));
    assert_eq!(pp1_1[6], Vector3::new(0.0, 1.0 / M_PER_KM, 0.0));

    // 2. Intermediate d_u = 2e308 overflow with finite acceleration (~1.414e305 km/s^2).
    let r2 = Vector3::new(26_560.0, 0.0, 0.0);
    let v2_eq = Vector3::new(0.0, 3.87, 0.0);
    let v2_inc = Vector3::new(0.0, 3.87, 3.87);
    let s2 = Vector3::new(1.0e8, 0.0, 1.0e8);
    let m2 = Vector3::new(0.0, 384_400.0, 0.0);

    let e2_big = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        d0: 1.0e308,
        d2c: 1.0e308,
        ..Default::default()
    })
    .unwrap();
    let e2_half = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        d0: 0.5e308,
        d2c: 0.5e308,
        ..Default::default()
    })
    .unwrap();
    let a2_big = e2_big
        .acceleration_with_sun_moon(r2, v2_eq, s2, m2)
        .expect("ECOM2 overflowing d_u accel");
    let a2_half = e2_half
        .acceleration_with_sun_moon(r2, v2_eq, s2, m2)
        .expect("ECOM2 half d_u accel");
    let pp2_big = e2_big
        .parameter_partials_with_sun_moon(r2, v2_eq, s2, m2)
        .expect("ECOM2 overflowing d_u pp");
    let ad2_big = e2_big
        .acceleration_partials_with_sun_moon(r2, v2_eq, s2, m2)
        .expect("ECOM2 overflowing d_u ad");
    let ad2_half = e2_half
        .acceleration_partials_with_sun_moon(r2, v2_eq, s2, m2)
        .expect("ECOM2 half d_u ad");
    assert!((a2_big - a2_half * 2.0).norm() <= 16.0 * f64::EPSILON * a2_big.norm());
    let recon2_big: Vector3<f64> = e2_big
        .coefficients()
        .iter()
        .zip(pp2_big.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a2_big - recon2_big).norm() <= 16.0 * f64::EPSILON * a2_big.norm());
    assert!(
        mat3_rel_err(ad2_big.wrt_position, ad2_half.wrt_position * 2.0).unwrap()
            <= 32.0 * f64::EPSILON
    );
    assert!(ad2_big.wrt_velocity.iter().all(|x| x.is_finite()));

    let e1_big = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        d0: 1.0e308,
        dc: 1.0e308,
        ..Default::default()
    })
    .unwrap();
    let e1_half = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        d0: 0.5e308,
        dc: 0.5e308,
        ..Default::default()
    })
    .unwrap();
    let a1_big = e1_big
        .acceleration_with_sun_moon(r2, v2_inc, s2, m2)
        .expect("ECOM1 overflowing d_u accel");
    let a1_half = e1_half
        .acceleration_with_sun_moon(r2, v2_inc, s2, m2)
        .expect("ECOM1 half d_u accel");
    let pp1_big = e1_big
        .parameter_partials_with_sun_moon(r2, v2_inc, s2, m2)
        .expect("ECOM1 overflowing d_u pp");
    let ad1_big = e1_big
        .acceleration_partials_with_sun_moon(r2, v2_inc, s2, m2)
        .expect("ECOM1 overflowing d_u ad");
    let ad1_half = e1_half
        .acceleration_partials_with_sun_moon(r2, v2_inc, s2, m2)
        .expect("ECOM1 half d_u ad");
    assert!((a1_big - a1_half * 2.0).norm() <= 16.0 * f64::EPSILON * a1_big.norm());
    let recon1_big: Vector3<f64> = e1_big
        .coefficients()
        .to_array()
        .iter()
        .zip(pp1_big.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a1_big - recon1_big).norm() <= 16.0 * f64::EPSILON * a1_big.norm());
    assert!(
        mat3_rel_err(ad1_big.wrt_position, ad1_half.wrt_position * 2.0).unwrap()
            <= 32.0 * f64::EPSILON
    );
    assert!(ad1_big.wrt_velocity.iter().all(|x| x.is_finite()));

    // 3. Axis-aligned e_D = (0, 0, 1) where a_m = 2e308 m/s^2 overflows f64::MAX while
    //    a_km = 2e305 km/s^2, parameter partials, and state partials are all finite.
    let s3 = Vector3::new(26_560.0, 0.0, 1.0e8);
    let a2_ax = e2_big
        .acceleration_with_sun_moon(r2, v2_inc, s3, m2)
        .expect("ECOM2 axis-aligned m/s^2 overflow");
    let pp2_ax = e2_big
        .parameter_partials_with_sun_moon(r2, v2_inc, s3, m2)
        .expect("ECOM2 axis-aligned m/s^2 overflow pp");
    let ad2_ax = e2_big
        .acceleration_partials_with_sun_moon(r2, v2_inc, s3, m2)
        .expect("ECOM2 axis-aligned m/s^2 overflow ad");
    let recon2_ax: Vector3<f64> = e2_big
        .coefficients()
        .iter()
        .zip(pp2_ax.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a2_ax - recon2_ax).norm() <= 16.0 * f64::EPSILON * a2_ax.norm());
    assert!(ad2_ax.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad2_ax.wrt_velocity.iter().all(|x| x.is_finite()));

    let a1_ax = e1_big
        .acceleration_with_sun_moon(r2, v2_inc, s3, m2)
        .expect("ECOM1 axis-aligned m/s^2 overflow");
    let pp1_ax = e1_big
        .parameter_partials_with_sun_moon(r2, v2_inc, s3, m2)
        .expect("ECOM1 axis-aligned m/s^2 overflow pp");
    let ad1_ax = e1_big
        .acceleration_partials_with_sun_moon(r2, v2_inc, s3, m2)
        .expect("ECOM1 axis-aligned m/s^2 overflow ad");
    let recon1_ax: Vector3<f64> = e1_big
        .coefficients()
        .to_array()
        .iter()
        .zip(pp1_ax.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a1_ax - recon1_ax).norm() <= 16.0 * f64::EPSILON * a1_ax.norm());
    assert!(ad1_ax.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad1_ax.wrt_velocity.iter().all(|x| x.is_finite()));

    // 4. Multi-scale > 2^1074 vector where Ellipsoid::plane_section_field's
    //    (gamma * gamma) * 4.0 underflows to 0.0 on the spherical Moon while
    //    double (4.0 * gamma) * gamma stays nonzero subnormal: tan_theta = 0
    //    fallback keeps acceleration and parameter_partials Ok (and
    //    acceleration_partials Ok whenever its own derivatives stay finite).
    let r4 = Vector3::new(5.242_465_626_050_804e97, 0.0, 6.025_545_388_333_738e-250);
    let v4 = Vector3::new(0.0, 0.003_046_798_001_474_058, 0.0);
    let s4 = Vector3::new(0.0, 0.0, 1.254_138_191_947_424_9e47);
    let m4 = Vector3::new(0.0, 384_400.0, 0.0);
    let e2_sph = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1())
        .unwrap()
        .with_earth_shadow_model(EarthShadowModel::Spherical);
    let a2_4 = e2_sph
        .acceleration_with_sun_moon(r4, v4, s4, m4)
        .expect("ECOM2 spherical Moon plane_section underflow accel");
    let pp2_4 = e2_sph
        .parameter_partials_with_sun_moon(r4, v4, s4, m4)
        .expect("ECOM2 spherical Moon plane_section underflow pp");
    let recon2_4: Vector3<f64> = e2_sph
        .coefficients()
        .iter()
        .zip(pp2_4.iter())
        .map(|(&c, p)| p * c)
        .sum();
    assert!((a2_4 - recon2_4).norm() <= 16.0 * f64::EPSILON * a2_4.norm());
    // With periodic terms nonzero, d(delta_u)/dv ~ 1e346 genuinely overflows binary64.
    assert!(matches!(
        e2_sph.acceleration_partials_with_sun_moon(r4, v4, s4, m4),
        Err(PropagationError::NumericalFailure(_))
    ));
    // With constant-only coefficients, field lighting and DYB state partials are all finite.
    let e2_sph_const = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
        d0: -1.0e-7,
        y0: 2.0e-9,
        b0: 5.0e-9,
        ..Default::default()
    })
    .unwrap()
    .with_earth_shadow_model(EarthShadowModel::Spherical);
    let ad2_4 = e2_sph_const
        .acceleration_partials_with_sun_moon(r4, v4, s4, m4)
        .expect("ECOM2 spherical Moon plane_section underflow const ad");
    assert!(ad2_4.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad2_4.wrt_velocity.iter().all(|x| x.is_finite()));

    // 5. Huge velocity (~1e307 km/s) makes standalone d(delta_u)/dv and d(u)/dv
    //    subnormal (~1e-311), while multiplying by O(1e308) coefficients yields
    //    normal O(1e-3) velocity partials that must not lose subnormal bits.
    let r5 = Vector3::new(26_560.0, 12_000.0, 8_000.0);
    let v5_ref = Vector3::new(-2.0, 3.0, 1.5);
    let v5 = v5_ref * libm::ldexp(1.0, 1020);
    let s5 = Vector3::new(1.0e8, -0.5e8, 0.7e8);
    let m5 = Vector3::new(0.0, 384_400.0, 0.0);
    let c2_5 = Ecom2D2b1Coefficients {
        b0: 1.0e307,
        b1c: -1.2e307,
        b1s: 0.9e307,
        d0: -1.5e307,
        d2c: 1.1e307,
        d2s: -0.8e307,
        y0: 0.7e307,
    };
    let e2_5 = Ecom2SolarRadiationPressure::d2b1(c2_5).unwrap();
    let ad2_5 = e2_5
        .acceleration_partials_with_sun_moon(r5, v5, s5, m5)
        .expect("ECOM2 huge-v huge-c partials");
    let c2_5_small =
        Ecom2D2b1Coefficients::from_array(c2_5.to_array().map(|x| libm::ldexp(x, -1020)));
    let ad2_5_ref = Ecom2SolarRadiationPressure::d2b1(c2_5_small)
        .unwrap()
        .acceleration_partials_with_sun_moon(r5, v5_ref, s5, m5)
        .expect("ECOM2 rescaled ref partials");
    assert!(
        mat3_rel_err(ad2_5.wrt_velocity, ad2_5_ref.wrt_velocity).unwrap() <= 32.0 * f64::EPSILON
    );
    let c1_5 = Ecom1Coefficients {
        d0: -1.5e307,
        dc: 1.1e307,
        ds: -0.8e307,
        y0: 0.7e307,
        yc: -0.6e307,
        ys: 0.5e307,
        b0: 1.0e307,
        bc: -1.2e307,
        bs: 0.9e307,
    };
    let e1_5 = Ecom1SolarRadiationPressure::new(c1_5).unwrap();
    let ad1_5 = e1_5
        .acceleration_partials_with_sun_moon(r5, v5, s5, m5)
        .expect("ECOM1 huge-v huge-c partials");
    let c1_5_small = Ecom1Coefficients::from_array(c1_5.to_array().map(|x| libm::ldexp(x, -1020)));
    let ad1_5_ref = Ecom1SolarRadiationPressure::new(c1_5_small)
        .unwrap()
        .acceleration_partials_with_sun_moon(r5, v5_ref, s5, m5)
        .expect("ECOM1 rescaled ref partials");
    assert!(
        mat3_rel_err(ad1_5.wrt_velocity, ad1_5_ref.wrt_velocity).unwrap() <= 32.0 * f64::EPSILON
    );
}

fn raw_orekit_dyb_has_subnormal_n2(sat_m: Vec3, sun_m: Vec3) -> bool {
    let ed_raw = sun_m.sub(sat_m);
    let ed_n2 = ed_raw.norm_sq();
    if ed_n2 > 0.0 && ed_n2 < fast_math::SAFE_MIN {
        return true;
    }
    if let Some(ed) = raw_hipparchus_normalize_double(ed_raw) {
        let ey_n2 = ed.cross(sat_m).norm_sq();
        if ey_n2 > 0.0 && ey_n2 < fast_math::SAFE_MIN {
            return true;
        }
    }
    false
}

fn raw_orekit_delta_u_has_subnormal_n2(sat_m: Vec3, vel_km_s: Vector3<f64>, sun_m: Vec3) -> bool {
    let Ok(vel_m) = to_vec3_m(vel_km_s) else {
        return false;
    };
    let z = sat_m.cross(vel_m);
    let y_raw = z.cross(sun_m);
    let y_n2 = y_raw.norm_sq();
    if y_n2 > 0.0 && y_n2 < fast_math::SAFE_MIN {
        return true;
    }
    if let Some(y) = raw_hipparchus_normalize_double(y_raw) {
        let x_n2 = y.cross(z).norm_sq();
        if x_n2 > 0.0 && x_n2 < fast_math::SAFE_MIN {
            return true;
        }
    }
    false
}

type RefPartials = Option<Result<AccelerationPartials, ()>>;

/// Independent power-of-two rescaled `FieldVec3<6>` reference for ECOM2 and ECOM1 state partials,
/// avoiding `ScaledFieldVec3` completely. Returns `None` when a vector is multi-scale (>900 bits
/// internal dynamic range, where plain f64 rescaling would flush small components) or when a
/// direction is singular; returns `Some(Err(()))` when the true mathematical Jacobian overflows
/// binary64.
fn rescaled_state_partials_ref(
    r_km: Vector3<f64>,
    v_km_s: Vector3<f64>,
    s_km: Vector3<f64>,
    ratio3: Grad<3>,
    e2_coeffs: &[f64],
    e1_coeffs: [f64; ECOM1_COEFFICIENT_COUNT],
) -> (RefPartials, RefPartials) {
    let dyn_range_ok = |v: [f64; 3]| -> bool {
        let mut e_max = i32::MIN;
        let mut e_min = i32::MAX;
        for c in v {
            if c != 0.0 {
                if !c.is_finite() {
                    return false;
                }
                let e = crate::astro::events::eclipse::ad::exact_exponent(c);
                e_max = e_max.max(e);
                e_min = e_min.min(e);
            }
        }
        e_max != i32::MIN && (e_max - e_min) < 850
    };
    if !(dyn_range_ok([r_km.x, r_km.y, r_km.z])
        && dyn_range_ok([s_km.x, s_km.y, s_km.z])
        && dyn_range_ok([v_km_s.x, v_km_s.y, v_km_s.z]))
    {
        return (None, None);
    }

    let exp_of = |v: [f64; 3]| -> i32 {
        let m = v[0].abs().max(v[1].abs()).max(v[2].abs());
        crate::astro::events::eclipse::ad::exact_exponent(m)
    };
    let er = exp_of([r_km.x, r_km.y, r_km.z]);
    let es = exp_of([s_km.x, s_km.y, s_km.z]);
    let ev = exp_of([v_km_s.x, v_km_s.y, v_km_s.z]);
    let ers = er.max(es);
    let sat_v = Vec3::new(r_km.x, r_km.y, r_km.z);
    let sun_v = Vec3::new(s_km.x, s_km.y, s_km.z);
    let vel_v = Vec3::new(v_km_s.x, v_km_s.y, v_km_s.z);

    // Rescaled O(1) vectors in km for DYB (r and s scaled together by -ers for e_D = normalize(s - r)).
    let scale_f = |v: Vec3, shift: i32, is_var: bool, var_base: usize| -> FieldVec3<6> {
        let mk = |val: f64, idx: usize| -> Grad<6> {
            let mut grad = [0.0; 6];
            if is_var {
                grad[idx] = 1.0;
            }
            Grad {
                value: libm::ldexp(val, shift),
                grad,
            }
        };
        FieldVec3::new(
            mk(v.x, var_base),
            mk(v.y, var_base + 1),
            mk(v.z, var_base + 2),
        )
    };

    let scale_grad_cols_scaled = |g: Grad<6>, dr_shift: i32, dv_shift: i32| -> ScaledGrad<6> {
        let mut grad = [ScaledF64::ZERO; 6];
        for i in 0..3 {
            grad[i] = ScaledF64::from_scaled(g.grad[i], dr_shift);
        }
        for i in 3..6 {
            grad[i] = ScaledF64::from_scaled(g.grad[i], dv_shift);
        }
        ScaledGrad {
            val: ScaledF64::from_f64(g.value),
            grad,
        }
    };

    // 1. e_D in O(1) km coordinates scaled by -ers -> d(e_D)/dr_km = 2^-ers * d(e_D)/dr_rs.
    let sat_rs = scale_f(sat_v, -ers, true, 0);
    let sun_rs = scale_f(sun_v, -ers, false, 0);
    let diff_rs = sun_rs.sub(sat_rs);
    let Some(ed_rs) = raw_hipparchus_normalize_field(diff_rs) else {
        return (None, None);
    };
    let ed_s = ScaledFieldVec3 {
        x: scale_grad_cols_scaled(ed_rs.x, -ers, 0),
        y: scale_grad_cols_scaled(ed_rs.y, -ers, 0),
        z: scale_grad_cols_scaled(ed_rs.z, -ers, 0),
    };

    // 2. e_Y = normalize(sun_u x sat_u) directly from the exact rescaled km inputs.
    let sat_u = scale_f(sat_v, -er, true, 0);
    let sun_u = scale_f(sun_v, -es, false, 0);
    let ey_raw_u = sun_u.cross(sat_u);
    let Some(ey_u) = raw_hipparchus_normalize_field(ey_raw_u) else {
        return (None, None);
    };
    let ey_s = ScaledFieldVec3 {
        x: scale_grad_cols_scaled(ey_u.x, -er, 0),
        y: scale_grad_cols_scaled(ey_u.y, -er, 0),
        z: scale_grad_cols_scaled(ey_u.z, -er, 0),
    };
    let eb_s = ed_s.cross(ey_s);

    let m_per_km = ScaledF64::from_f64(M_PER_KM);
    let ratio_s_km = ScaledGrad {
        val: ScaledF64::from_f64(ratio3.value),
        grad: [
            ScaledF64::from_f64(ratio3.grad[0]).mul(m_per_km),
            ScaledF64::from_f64(ratio3.grad[1]).mul(m_per_km),
            ScaledF64::from_f64(ratio3.grad[2]).mul(m_per_km),
            ScaledF64::ZERO,
            ScaledF64::ZERO,
            ScaledF64::ZERO,
        ],
    };
    let sunlit = is_sunlit_field(ratio3);

    // 3. delta_u in O(1) km coordinates: r_u = r_km * 2^-er, s_u = s_km * 2^-es, v_u = v_km_s * 2^-ev.
    let vel_u = scale_f(vel_v, -ev, true, 3);
    let z_u = sat_u.cross(vel_u);
    let e2_res = if !is_delta_u_undefined_exact(r_km, v_km_s, s_km) {
        if let Some(y_u) = raw_hipparchus_normalize_field(z_u.cross(sun_u)) {
            if let Some(x_u) = raw_hipparchus_normalize_field(y_u.cross(z_u)) {
                let py_u = sat_u.dot(y_u);
                let px_u = sat_u.dot(x_u);
                if let Some(du_u) = raw_hipparchus_atan2_field(py_u, px_u, fast_math::atan2) {
                    let mut du_grad_scaled = [ScaledF64::ZERO; 6];
                    for i in 0..3 {
                        du_grad_scaled[i] = ScaledF64::from_scaled(du_u.grad[i], -er);
                    }
                    for i in 3..6 {
                        du_grad_scaled[i] = ScaledF64::from_scaled(du_u.grad[i], -ev);
                    }
                    let (s1, c1) = ScaledGrad::sin_cos_from_scaled(
                        du_u.value,
                        du_grad_scaled,
                        fast_math::sin_cos,
                    );
                    let mut g2 = [ScaledF64::ZERO; 6];
                    let mut g4 = [ScaledF64::ZERO; 6];
                    let two_s = ScaledF64::from_f64(2.0);
                    let four_s = ScaledF64::from_f64(4.0);
                    for i in 0..6 {
                        g2[i] = du_grad_scaled[i].mul(two_s);
                        g4[i] = du_grad_scaled[i].mul(four_s);
                    }
                    let (s2, c2) =
                        ScaledGrad::sin_cos_from_scaled(2.0 * du_u.value, g2, fast_math::sin_cos);
                    let (s4, c4) =
                        ScaledGrad::sin_cos_from_scaled(4.0 * du_u.value, g4, fast_math::sin_cos);
                    let params: Vec<ScaledGrad<6>> = e2_coeffs
                        .iter()
                        .map(|&c| ScaledGrad::from_grad(Grad::constant(c)))
                        .collect();
                    let b_u = params[0].add(c1.mul(params[1])).add(s1.mul(params[2]));
                    let d_u = params[3]
                        .add(c2.mul(params[4]))
                        .add(s2.mul(params[6]))
                        .add(c4.mul(params[5]))
                        .add(s4.mul(params[7]));
                    let y0 = params[8];
                    let raw = ScaledFieldVec3::lincomb3(d_u, ed_s, y0, ey_s, b_u, eb_s);
                    let accel_m_s = if sunlit {
                        raw
                    } else {
                        raw.scalar_mul(ratio_s_km)
                    };
                    match extract_state_partials_scaled(accel_m_s.scalar_div(m_per_km)) {
                        Ok(p) => Some(Ok(p)),
                        Err(_) => Some(Err(())),
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // 4. ECOM1 u in O(1) km coordinates.
    let mom_u = sat_u.cross(vel_u);
    let e1_res = if let Some(norm_u) = raw_hipparchus_normalize_field(mom_u) {
        let node_raw_u = FieldVec3::new(norm_u.y.negate(), norm_u.x, Grad::ZERO);
        if let Some(node_u) = raw_hipparchus_normalize_field(node_raw_u) {
            let ahead_u = norm_u.cross(node_u);
            let py_u = sat_u.dot(ahead_u);
            let px_u = sat_u.dot(node_u);
            if let Some(u_u) = raw_hipparchus_atan2_field(py_u, px_u, libm::atan2) {
                let mut u_grad_scaled = [ScaledF64::ZERO; 6];
                for i in 0..3 {
                    u_grad_scaled[i] = ScaledF64::from_scaled(u_u.grad[i], -er);
                }
                for i in 3..6 {
                    u_grad_scaled[i] = ScaledF64::from_scaled(u_u.grad[i], -ev);
                }
                let (su, cu) = ScaledGrad::sin_cos_from_scaled(u_u.value, u_grad_scaled, |a| {
                    (libm::sin(a), libm::cos(a))
                });
                let c_s = e1_coeffs.map(|c| ScaledGrad::from_grad(Grad::constant(c)));
                let [d0, dc, ds, y0, yc, ys, b0, bc, bs] = c_s;
                let d_u = d0.add(cu.mul(dc)).add(su.mul(ds));
                let y_u = y0.add(cu.mul(yc)).add(su.mul(ys));
                let b_u = b0.add(cu.mul(bc)).add(su.mul(bs));
                let raw = ScaledFieldVec3::lincomb3(d_u, ed_s, y_u, ey_s, b_u, eb_s);
                let accel_m_s = if sunlit {
                    raw
                } else {
                    raw.scalar_mul(ratio_s_km)
                };
                match extract_state_partials_scaled(accel_m_s.scalar_div(m_per_km)) {
                    Ok(p) => Some(Ok(p)),
                    Err(_) => Some(Err(())),
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    (e2_res, e1_res)
}

fn mat3_max_abs(m: Matrix3<f64>) -> f64 {
    m.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// Frobenius norm of `actual - reference` divided by `max(||reference||_F, min_scale)`,
/// rescaled by exact powers of two so entries above `1e154` or below `1e-162` never
/// overflow or underflow. Passing `ScaledF64::ZERO` gives the plain relative error.
fn mat3_scaled_err(
    actual: Matrix3<f64>,
    reference: Matrix3<f64>,
    min_scale: ScaledF64,
) -> Option<f64> {
    if !reference.iter().all(|c| c.is_finite()) {
        return None;
    }
    if !actual.iter().all(|c| c.is_finite()) {
        return Some(f64::INFINITY);
    }
    let ref_max = mat3_max_abs(reference);
    let act_max = mat3_max_abs(actual);
    let ref_s = ScaledF64::from_f64(ref_max);
    let act_s = ScaledF64::from_f64(act_max);
    let denom_max = if min_scale.is_zero() {
        ref_s
    } else if ref_s.is_zero()
        || min_scale.e > ref_s.e
        || (min_scale.e == ref_s.e && min_scale.m.abs() > ref_s.m.abs())
    {
        min_scale
    } else {
        ref_s
    };
    if denom_max.is_zero() {
        return None;
    }
    let exp = if act_s.is_zero() {
        denom_max.e
    } else {
        denom_max.e.max(act_s.e)
    };
    let rescale = |m: Matrix3<f64>| -> Matrix3<f64> {
        if exp > 512 {
            (m * libm::ldexp(1.0, -512)) * libm::ldexp(1.0, -(exp - 512))
        } else if exp < -512 {
            (m * libm::ldexp(1.0, 512)) * libm::ldexp(1.0, -(exp + 512))
        } else {
            m * libm::ldexp(1.0, -exp)
        }
    };
    let a_s = rescale(actual);
    let r_s = rescale(reference);
    let min_s = if min_scale.is_zero() {
        0.0
    } else {
        libm::ldexp(min_scale.m.abs(), (min_scale.e - exp).max(-1100))
    };
    let denom = r_s.norm().max(min_s);
    if denom == 0.0 {
        Some(f64::INFINITY)
    } else {
        Some((a_s - r_s).norm() / denom)
    }
}

fn mat3_rel_err(actual: Matrix3<f64>, reference: Matrix3<f64>) -> Option<f64> {
    mat3_scaled_err(actual, reference, ScaledF64::ZERO)
}

/// Natural scale `|ratio| * c_periodic / |v_perp_m_s|` of the intermediate
/// `O(c / |v_perp|)` velocity-derivative terms before near-node / near-polar /
/// `r ⊥ s` cancellation of `d(angle)/dv`.
fn vel_partial_natural_scale(
    r_km: Vector3<f64>,
    v_km_s: Vector3<f64>,
    ratio: f64,
    c_periodic_m_s2: f64,
) -> ScaledF64 {
    let r_s = ScaledVec3::from_vec3(Vec3::new(r_km.x, r_km.y, r_km.z));
    let v_s = ScaledVec3::from_vec3(Vec3::new(v_km_s.x, v_km_s.y, v_km_s.z));
    let (Some(r_norm), Some(h_norm)) = (r_s.norm(), r_s.cross(v_s).norm()) else {
        return ScaledF64::ZERO;
    };
    if r_norm.is_zero() || h_norm.is_zero() {
        return ScaledF64::ZERO;
    }
    let v_perp_m_s = h_norm.div(r_norm).mul(ScaledF64::from_f64(M_PER_KM));
    ScaledF64::from_f64(ratio.abs() * c_periodic_m_s2).div(v_perp_m_s)
}

/// Natural scale `|ratio| * c / min(|r_m|, |s_m - r_m|)` of the intermediate
/// `O(c / L)` position-derivative terms before `sin(k * du)` / `cos(k * du)`
/// near-zero cancellation when a single harmonic coefficient dominates.
fn pos_partial_natural_scale(
    r_km: Vector3<f64>,
    s_km: Vector3<f64>,
    ratio: f64,
    c_m_s2: f64,
) -> ScaledF64 {
    let r_s = ScaledVec3::from_vec3(Vec3::new(r_km.x, r_km.y, r_km.z));
    let s_s = ScaledVec3::from_vec3(Vec3::new(s_km.x, s_km.y, s_km.z));
    let Some(r_norm) = r_s.norm() else {
        return ScaledF64::ZERO;
    };
    let d_norm = s_s.sub(r_s).norm().unwrap_or(r_norm);
    let min_len = if d_norm.e < r_norm.e || (d_norm.e == r_norm.e && d_norm.m < r_norm.m) {
        d_norm
    } else {
        r_norm
    };
    let len_m = min_len.mul(ScaledF64::from_f64(M_PER_KM));
    ScaledF64::from_f64(ratio.abs() * c_m_s2).div(len_m)
}

#[test]
fn finite_acceleration_partials_succeed_when_angle_derivative_alone_overflows() {
    // Subnormal velocity makes |d(delta_u)/dv| and |d(u)/dv| exceed f64::MAX (~1e310),
    // while multiplying by the O(1e-9) m/s^2 coefficients yields finite binary64
    // acceleration partials (~1e301).
    let r = Vector3::new(AU_KM, AU_KM, 0.5 * AU_KM);
    let v_ref = Vector3::new(1.0, 2.0, 1.5);
    let v = v_ref * libm::ldexp(1.0, -1042);
    let s = Vector3::new(0.0, AU_KM, 0.0);
    let m = Vector3::new(0.0, 0.0, 384_400.0);

    // Confirm the standalone angle-derivative helpers genuinely overflow f64::MAX
    // on this input, so the public API must use the acceleration-level scaled path.
    let sat_f = FieldVec3::<6>::new(
        Grad::variable(0, r.x * M_PER_KM),
        Grad::variable(1, r.y * M_PER_KM),
        Grad::variable(2, r.z * M_PER_KM),
    );
    let sun_f = FieldVec3::<6>::constant(Vec3::new(s.x * M_PER_KM, s.y * M_PER_KM, s.z * M_PER_KM));
    assert!(orekit_delta_u_field_fast(sat_f, v, true, sun_f).is_none());
    let (_, du_sgrad) = orekit_delta_u_field_scaled(
        r,
        v,
        s,
        scaled_field_vec3_variable::<6>(r, 0),
        scaled_field_vec3_variable::<6>(v, 3),
        scaled_field_vec3_constant::<6>(s),
    )
    .expect("scaled delta_u succeeds");
    assert!(du_sgrad.iter().any(|g| !g.to_f64().is_finite()));
    let (su_sgrad, cu_sgrad) = ecom1_sin_cos_field_scaled(
        scaled_field_vec3_variable::<6>(r, 0),
        scaled_field_vec3_variable::<6>(v, 3),
    )
    .expect("scaled sin/cos succeeds");
    assert!(
        su_sgrad.grad.iter().any(|g| !g.to_f64().is_finite())
            || cu_sgrad.grad.iter().any(|g| !g.to_f64().is_finite())
    );

    let e2 = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let ad2 = e2
        .acceleration_partials_with_sun_moon(r, v, s, m)
        .expect("ECOM2 finite accel partial despite overflowing d(delta_u)/dv");
    assert!(ad2.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad2.wrt_velocity.iter().all(|x| x.is_finite()));
    assert!(mat3_max_abs(ad2.wrt_velocity) > 1.0e295);

    let ad2_ref = e2
        .acceleration_partials_with_sun_moon(r, v_ref, s, m)
        .expect("ECOM2 scaled reference partials");
    let scale_half = libm::ldexp(1.0, -521);
    assert!(mat3_rel_err(ad2.wrt_position, ad2_ref.wrt_position).unwrap() <= 32.0 * f64::EPSILON);
    assert!(
        mat3_rel_err(
            (ad2.wrt_velocity * scale_half) * scale_half,
            ad2_ref.wrt_velocity
        )
        .unwrap()
            <= 64.0 * f64::EPSILON
    );

    let e1 = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    let ad1 = e1
        .acceleration_partials_with_sun_moon(r, v, s, m)
        .expect("ECOM1 finite accel partial despite overflowing d(u)/dv");
    assert!(ad1.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad1.wrt_velocity.iter().all(|x| x.is_finite()));
    assert!(mat3_max_abs(ad1.wrt_velocity) > 1.0e295);

    let ad1_ref = e1
        .acceleration_partials_with_sun_moon(r, v_ref, s, m)
        .expect("ECOM1 scaled reference partials");
    assert!(mat3_rel_err(ad1.wrt_position, ad1_ref.wrt_position).unwrap() <= 32.0 * f64::EPSILON);
    assert!(
        mat3_rel_err(
            (ad1.wrt_velocity * scale_half) * scale_half,
            ad1_ref.wrt_velocity
        )
        .unwrap()
            <= 64.0 * f64::EPSILON
    );
}

#[test]
fn ecom_wide_magnitude_and_near_singular_direction_property_sweep() {
    // Acceptance/refusal rule checked across every sampled state (r, v, s, m)
    // and every shadow configuration (ApparentDisc +/- Moon, Conical; WGS84 & Spherical):
    // 1. In umbra (lighting_ratio == 0), acceleration, parameter_partials and
    //    acceleration_partials all return Ok(zero) without forming the DYB frame.
    // 2. Outside umbra (lighting_ratio != 0), with all ECOM coefficients nonzero:
    //    - acceleration and parameter_partials accept and refuse the exact same inputs,
    //      except when the acceleration value itself overflows binary64 while the
    //      partials remain finite (acceleration returns Err(NumericalFailure) while
    //      parameter_partials and acceleration_partials return Ok).
    //    - Under ApparentDisc, ecom_geometry_with_shadow succeeds iff shadow and DYB
    //      are defined; delta_u_rad.is_some() iff ECOM2 acceleration succeeds (apart
    //      from acceleration value overflow), and argument_of_latitude_rad.is_some()
    //      iff ECOM1 acceleration succeeds (apart from acceleration value overflow).
    //    - acceleration_partials accepts every input that acceleration accepts EXCEPT
    //      when only a derivative is undefined or overflows binary64 (in which case
    //      acceleration_partials returns Err(PropagationError::NumericalFailure(_))).
    //      Conversely, acceleration_partials or parameter_partials returns Ok when
    //      acceleration returns Err ONLY when the acceleration value itself overflows
    //      binary64 while the partials remain finite.
    // 3. Wherever a result is returned outside umbra, the DYB frame vectors and
    //    delta_u / u agree with an independent reference evaluated on inputs rescaled
    //    by exact powers of two, ECOM1 acceleration equals the field path value part
    //    bit for bit, and ECOM1 state and parameter partials agree with independent
    //    power-of-two rescaled references and central differences.

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn unit_f64(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
        }
        fn log_uniform(&mut self, exp_min: i32, exp_max: i32) -> f64 {
            let e = exp_min as f64 + (exp_max - exp_min) as f64 * self.unit_f64();
            let ei = e.floor() as i32;
            let frac = e - ei as f64;
            libm::ldexp(libm::exp2(frac), ei)
        }
        fn unit_vec(&mut self) -> Vector3<f64> {
            let z = 2.0 * self.unit_f64() - 1.0;
            let phi = 2.0 * PI * self.unit_f64();
            let r = (1.0 - z * z).max(0.0).sqrt();
            Vector3::new(r * libm::cos(phi), r * libm::sin(phi), z)
        }
    }

    fn rescale_pow2(v: Vector3<f64>) -> Option<Vector3<f64>> {
        let max_abs = v.x.abs().max(v.y.abs()).max(v.z.abs());
        if !max_abs.is_finite() || max_abs == 0.0 {
            return None;
        }
        let exp = crate::astro::events::eclipse::ad::exact_exponent(max_abs);
        Some(Vector3::new(
            libm::ldexp(v.x, -exp),
            libm::ldexp(v.y, -exp),
            libm::ldexp(v.z, -exp),
        ))
    }

    /// Reference DYB frame, delta_u, and u evaluated on the metre vectors rescaled
    /// by exact powers of two, returning also the conditioning sine for each cross
    /// product so tolerances reflect genuine geometric sensitivity near collinearity.
    struct RefGeom {
        dyb: Option<[Vector3<f64>; 3]>,
        dyb_ecom1: Option<[Vector3<f64>; 3]>,
        sin_ed_r: f64,
        sin_ey_ecom1: f64,
        du: Option<f64>,
        sin_du: f64,
        u: Option<f64>,
        sin_u: f64,
        single_scale_rs: bool,
        single_scale_rv: bool,
    }

    fn dyn_range_ok(v: [f64; 3]) -> bool {
        let mut e_max = i32::MIN;
        let mut e_min = i32::MAX;
        for c in v {
            if c != 0.0 {
                if !c.is_finite() {
                    return false;
                }
                let e = crate::astro::events::eclipse::ad::exact_exponent(c);
                e_max = e_max.max(e);
                e_min = e_min.min(e);
            }
        }
        e_max != i32::MIN && (e_max - e_min) < 850
    }

    fn reference_geometry(r_km: Vector3<f64>, v_km_s: Vector3<f64>, s_km: Vector3<f64>) -> RefGeom {
        let (Ok(sat_m), Ok(sun_m)) = (to_vec3_m(r_km), to_vec3_m(s_km)) else {
            return RefGeom {
                dyb: None,
                dyb_ecom1: None,
                sin_ed_r: 0.0,
                sin_ey_ecom1: 0.0,
                du: None,
                sin_du: 0.0,
                u: None,
                sin_u: 0.0,
                single_scale_rs: false,
                single_scale_rv: false,
            };
        };
        let single_scale_r = dyn_range_ok([sat_m.x, sat_m.y, sat_m.z]);
        let single_scale_s = dyn_range_ok([sun_m.x, sun_m.y, sun_m.z]);
        let single_scale_v = dyn_range_ok([v_km_s.x, v_km_s.y, v_km_s.z]);
        let single_scale_rs = single_scale_r && single_scale_s;
        let single_scale_rv = single_scale_r && single_scale_v;
        let r_m = Vector3::new(sat_m.x, sat_m.y, sat_m.z);
        let s_m = Vector3::new(sun_m.x, sun_m.y, sun_m.z);
        let rs_max = r_m
            .iter()
            .chain(s_m.iter())
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        let rs_exp = if rs_max > 0.0 && rs_max.is_finite() {
            crate::astro::events::eclipse::ad::exact_exponent(rs_max)
        } else {
            return RefGeom {
                dyb: None,
                dyb_ecom1: None,
                sin_ed_r: 0.0,
                sin_ey_ecom1: 0.0,
                du: None,
                sin_du: 0.0,
                u: None,
                sin_u: 0.0,
                single_scale_rs,
                single_scale_rv,
            };
        };
        let r_rs = Vector3::new(
            libm::ldexp(r_m.x, -rs_exp),
            libm::ldexp(r_m.y, -rs_exp),
            libm::ldexp(r_m.z, -rs_exp),
        );
        let s_rs = Vector3::new(
            libm::ldexp(s_m.x, -rs_exp),
            libm::ldexp(s_m.y, -rs_exp),
            libm::ldexp(s_m.z, -rs_exp),
        );
        let Some(r_u) = rescale_pow2(r_m) else {
            return RefGeom {
                dyb: None,
                dyb_ecom1: None,
                sin_ed_r: 0.0,
                sin_ey_ecom1: 0.0,
                du: None,
                sin_du: 0.0,
                u: None,
                sin_u: 0.0,
                single_scale_rs,
                single_scale_rv,
            };
        };
        let Some(s_u) = rescale_pow2(s_m) else {
            return RefGeom {
                dyb: None,
                dyb_ecom1: None,
                sin_ed_r: 0.0,
                sin_ey_ecom1: 0.0,
                du: None,
                sin_du: 0.0,
                u: None,
                sin_u: 0.0,
                single_scale_rs,
                single_scale_rv,
            };
        };
        let r_unit = r_u.normalize();
        let s_unit = s_u.normalize();
        let to_v = |u: Vector3<f64>| Vec3::new(u.x, u.y, u.z);
        let from_v = |u: Vec3| Vector3::new(u.x, u.y, u.z);
        let sin_s_r = to_v(s_unit).cross(to_v(r_unit)).norm();
        let diff = to_v(s_rs).sub(to_v(r_rs));
        let e_d = if diff.norm() > 0.0 {
            diff.normalize().map(from_v)
        } else {
            ScaledVec3::from_vec3(sun_m)
                .sub(ScaledVec3::from_vec3(sat_m))
                .normalize()
                .map(|(u, _)| from_v(u))
        };
        let mut sin_ed_r = 0.0;
        let dyb = e_d.and_then(|ed| {
            let ey_raw_v = to_v(ed).cross(to_v(r_u));
            sin_ed_r = to_v(ed).cross(to_v(r_unit)).norm();
            let ey = if ey_raw_v.norm() > 1.0e-140 {
                ey_raw_v.normalize().map(from_v)
            } else {
                ScaledVec3::from_vec3(to_v(ed))
                    .cross(ScaledVec3::from_vec3(sat_m))
                    .normalize()
                    .map(|(u, _)| from_v(u))
            }?;
            let eb = from_v(to_v(ed).cross(to_v(ey)));
            Some([ed, ey, eb])
        });
        let sin_ey_ecom1 = sin_s_r;
        let dyb_ecom1 = e_d.and_then(|ed| {
            let ey_raw_v = to_v(s_u).cross(to_v(r_u));
            let ey = if ey_raw_v.norm() > 1.0e-140 {
                ey_raw_v.normalize().map(from_v)
            } else {
                None
            }?;
            let eb = from_v(to_v(ed).cross(to_v(ey)));
            Some([ed, ey, eb])
        });

        let mut sin_du = 0.0;
        let mut sin_u = 0.0;
        let (du, u_arg) = if let Some(v_u) = rescale_pow2(v_km_s) {
            let v_unit = v_u.normalize();
            let sin_rv = to_v(r_unit).cross(to_v(v_unit)).norm();
            let z_raw = to_v(r_u).cross(to_v(v_u));
            let (z_u_opt, du_ref) = if z_raw.norm() > 1.0e-140 {
                let zu = z_raw.normalize().unwrap();
                let sin_zs = zu.cross(to_v(s_unit)).norm();
                sin_du = sin_rv.min(sin_zs);
                let y_raw = zu.cross(to_v(s_u));
                let du_v = if y_raw.norm() > 1.0e-140 {
                    let yu = y_raw.normalize().unwrap();
                    let xu = yu.cross(zu).normalize().unwrap();
                    Some(libm::atan2(to_v(r_u).dot(yu), to_v(r_u).dot(xu)))
                } else {
                    None
                };
                (Some(zu), du_v)
            } else {
                (None, None)
            };
            let u_ref = z_u_opt.and_then(|zu| {
                let node_raw = Vec3::new(-zu.y, zu.x, 0.0);
                sin_u = sin_rv.min(node_raw.norm());
                if node_raw.norm() > 1.0e-140 {
                    let node = node_raw.normalize().unwrap();
                    let ahead = zu.cross(node);
                    Some(libm::atan2(to_v(r_u).dot(ahead), to_v(r_u).dot(node)))
                } else {
                    None
                }
            });
            (du_ref, u_ref)
        } else {
            (None, None)
        };
        RefGeom {
            dyb,
            dyb_ecom1,
            sin_ed_r,
            sin_ey_ecom1,
            du,
            sin_du,
            u: u_arg,
            sin_u,
            single_scale_rs,
            single_scale_rv,
        }
    }

    fn angle_diff(a: f64, b: f64) -> f64 {
        let mut d = (a - b).abs();
        if d > PI {
            d = (2.0 * PI - d).abs();
        }
        d
    }

    /// Reconstruct `sum_k c_k * pp[k]` and return `(recon, sum_abs)` without
    /// intermediate overflow, where `sum_abs = sum_k |c_k| * ||pp[k]||_2`.
    fn linear_recon(coeffs: &[f64], pp: &[Vector3<f64>]) -> (Vector3<f64>, ScaledF64) {
        let mut rx = ScaledF64::ZERO;
        let mut ry = ScaledF64::ZERO;
        let mut rz = ScaledF64::ZERO;
        let mut sum_abs = ScaledF64::ZERO;
        for (&c, p) in coeffs.iter().zip(pp.iter()) {
            let cs = ScaledF64::from_f64(c);
            let ps = ScaledVec3::from_vec3(Vec3::new(p.x, p.y, p.z));
            rx = rx.add(cs.mul(ps.x));
            ry = ry.add(cs.mul(ps.y));
            rz = rz.add(cs.mul(ps.z));
            if let Some(pn) = ps.norm() {
                sum_abs = sum_abs.add(ScaledF64::from_f64(c.abs()).mul(pn));
            }
        }
        (Vector3::new(rx.to_f64(), ry.to_f64(), rz.to_f64()), sum_abs)
    }

    /// Relative difference `||a - b||_2 / denom` with power-of-two rescaling so
    /// vectors up to `f64::MAX` or down to subnormals never overflow or underflow.
    fn vec3_scaled_rel_err(a: Vector3<f64>, b: Vector3<f64>, denom: ScaledF64) -> f64 {
        if !a.iter().all(|c| c.is_finite())
            || !b.iter().all(|c| c.is_finite())
            || !denom.is_finite()
        {
            return f64::INFINITY;
        }
        let max_ab = a
            .iter()
            .chain(b.iter())
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        let ab_s = ScaledF64::from_f64(max_ab);
        let exp = if ab_s.is_zero() {
            denom.e
        } else {
            denom.e.max(ab_s.e)
        };
        let rescale = |v: Vector3<f64>| -> Vector3<f64> {
            if exp > 512 {
                (v * libm::ldexp(1.0, -512)) * libm::ldexp(1.0, -(exp - 512))
            } else if exp < -512 {
                (v * libm::ldexp(1.0, 512)) * libm::ldexp(1.0, -(exp + 512))
            } else {
                v * libm::ldexp(1.0, -exp)
            }
        };
        let diff_n = (rescale(a) - rescale(b)).norm();
        let d_s = libm::ldexp(denom.m.abs(), (denom.e - exp).max(-1100));
        if d_s == 0.0 {
            if diff_n == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            diff_n / d_s
        }
    }

    let mut rng = SplitMix64(0x5e7d_2026_ec0a_1234);
    let ecom2_base = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1()).unwrap();
    let ecom1_base = Ecom1SolarRadiationPressure::new(mixed1()).unwrap();
    let ecom2_templates = all_ecom2_shadow_configs(&ecom2_base);
    let ecom1_templates = all_ecom1_shadow_configs(ecom1_base);

    let mut total_ok_ecom2 = 0usize;
    let mut total_ok_ecom1 = 0usize;
    let mut total_ad2_ok = 0usize;
    let mut total_ad1_ok = 0usize;
    let mut total_ad2_fast_bits = 0usize;
    let mut total_ad2_fallback_checked = 0usize;
    let mut total_ad1_checked = 0usize;
    let mut total_pp1_ref_checked = 0usize;
    let mut total_ecom1_val_bit_exact = 0usize;
    let mut total_ad2_fd_checked = 0usize;
    let mut total_ad1_fd_checked = 0usize;
    let mut total_subnormal_fast_bits = 0usize;
    let mut total_u_bit_exact = 0usize;
    let mut total_flushed_frame_ok = 0usize;
    let mut total_overflow_coeff_ok = 0usize;
    let mut total_near_node_checked = 0usize;
    let mut total_collinear_checked = 0usize;
    let mut total_tiny_conical_checked = 0usize;
    let mut total_umbra = 0usize;
    let mut total_refused = 0usize;
    let mut max_lin2_eps = 0.0_f64;
    let mut max_lin1_eps = 0.0_f64;
    let mut max_ad2_p_eps = 0.0_f64;
    let mut max_ad2_v_eps = 0.0_f64;
    let mut max_ad1_p_eps = 0.0_f64;
    let mut max_ad1_v_eps = 0.0_f64;
    let mut max_pp1_ref_eps = 0.0_f64;
    let mut max_ad1_fd_p_rel = 0.0_f64;
    let mut max_ad1_fd_v_rel = 0.0_f64;
    let mut max_near_node_a1_eps = 0.0_f64;
    let mut max_near_node_ad1_eps = 0.0_f64;
    let mut max_tiny_conical_e1_eps = 0.0_f64;
    let mut max_tiny_conical_e2_eps = 0.0_f64;
    let inv_km = 1.0 / M_PER_KM;

    for cfg_idx in 0..6 {
        let tmpl2 = &ecom2_templates[cfg_idx];
        let tmpl1 = &ecom1_templates[cfg_idx];
        let shadow_fn = tmpl2.shadow_function();
        let is_conical = matches!(shadow_fn, ShadowFunction::Conical);
        let include_moon = tmpl2.moon_eclipse_enabled();
        let earth_model = tmpl2.earth_shadow_model();

        for case in 0..512 {
            // Draw coefficients across the full binary64 range including near f64::MAX.
            let (e2, e1, is_near_max_coeff) = match case % 4 {
                0 => (tmpl2.clone(), *tmpl1, false),
                1 => {
                    let mut c2 = [0.0; ECOM2_D4B1_COEFFICIENT_COUNT];
                    let mut c1 = [0.0; ECOM1_COEFFICIENT_COUNT];
                    for x in &mut c2 {
                        let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                        *x = sign * rng.log_uniform(-1074, 1023);
                    }
                    for x in &mut c1 {
                        let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                        *x = sign * rng.log_uniform(-1074, 1023);
                    }
                    let m2 =
                        Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients::from_array(c2))
                            .unwrap()
                            .with_shadow_function(shadow_fn)
                            .with_earth_shadow_model(earth_model)
                            .with_moon_eclipse(include_moon);
                    let m1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients::from_array(c1))
                        .unwrap()
                        .with_shadow_function(shadow_fn)
                        .with_earth_shadow_model(earth_model)
                        .with_moon_eclipse(include_moon);
                    (m2, m1, false)
                }
                2 => {
                    let mut c2 = [0.0; ECOM2_D4B1_COEFFICIENT_COUNT];
                    let mut c1 = [0.0; ECOM1_COEFFICIENT_COUNT];
                    for x in &mut c2 {
                        let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                        *x = sign * rng.log_uniform(1015, 1023);
                    }
                    for x in &mut c1 {
                        let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                        *x = sign * rng.log_uniform(1015, 1023);
                    }
                    let m2 =
                        Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients::from_array(c2))
                            .unwrap()
                            .with_shadow_function(shadow_fn)
                            .with_earth_shadow_model(earth_model)
                            .with_moon_eclipse(include_moon);
                    let m1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients::from_array(c1))
                        .unwrap()
                        .with_shadow_function(shadow_fn)
                        .with_earth_shadow_model(earth_model)
                        .with_moon_eclipse(include_moon);
                    (m2, m1, true)
                }
                _ => {
                    let base_exp = -1070 + ((rng.next_u64() % 2085) as i32);
                    let lo = (base_exp - 8).max(-1074);
                    let hi = (base_exp + 8).min(1023);
                    let mut c2 = [0.0; ECOM2_D4B1_COEFFICIENT_COUNT];
                    let mut c1 = [0.0; ECOM1_COEFFICIENT_COUNT];
                    for x in &mut c2 {
                        let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                        *x = sign * rng.log_uniform(lo, hi);
                    }
                    for x in &mut c1 {
                        let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                        *x = sign * rng.log_uniform(lo, hi);
                    }
                    let m2 =
                        Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients::from_array(c2))
                            .unwrap()
                            .with_shadow_function(shadow_fn)
                            .with_earth_shadow_model(earth_model)
                            .with_moon_eclipse(include_moon);
                    let m1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients::from_array(c1))
                        .unwrap()
                        .with_shadow_function(shadow_fn)
                        .with_earth_shadow_model(earth_model)
                        .with_moon_eclipse(include_moon);
                    (m2, m1, hi >= 1015)
                }
            };

            let (r_mag, s_mag) = if is_conical {
                // Span from tiny sub-metre to near f64::MAX^{1/2} (and beyond on edge cases).
                let rm = match case % 16 {
                    0 => rng.log_uniform(-500, 505),
                    1 => rng.log_uniform(480, 515),
                    2 => rng.log_uniform(-520, -480),
                    _ => rng.log_uniform(-450, 505),
                };
                let sm = match case % 16 {
                    0 => rng.log_uniform(20, 505),
                    1 => rng.log_uniform(480, 515),
                    _ => rng.log_uniform(20, 505),
                };
                (rm, sm)
            } else {
                // ApparentDisc requires |r| > R_earth (~6371 km) and |r_m|, |s_m| < ~1.34e154 m.
                let rm = match case % 16 {
                    0 => rng.log_uniform(13, 500),
                    1 => rng.log_uniform(470, 508),
                    2 => rng.log_uniform(10, 15), // includes inside-Earth refusal cases
                    _ => rng.log_uniform(13, 495),
                };
                let sm = match case % 16 {
                    0 => rng.log_uniform(14, 500),
                    1 => rng.log_uniform(470, 508),
                    _ => rng.log_uniform(14, 495),
                };
                (rm, sm)
            };
            let v_mag = match case % 12 {
                0 => rng.log_uniform(-1020, 1020),
                1 => rng.log_uniform(950, 1022), // includes > 1e305 km/s (m/s conversion overflow)
                2 => rng.log_uniform(-1020, -950),
                _ => rng.log_uniform(-900, 900),
            };
            let m_mag = rng.log_uniform(12, 495);

            let r_dir = rng.unit_vec();
            let mut s_dir = rng.unit_vec();
            let mut v_dir = rng.unit_vec();
            let mut m_dir = rng.unit_vec();

            // Inject near-collinear, near-normal, near-equatorial, multiscale, >2^1074
            // component-flushing, near-node (|y|/x < 2^-1022), exact-collinear, and
            // tiny-conical-gradient cases.
            let mode = case % 16;
            let (r, v, s, m) = match mode {
                // Near-collinear r and s (both sunlit and shadow sides).
                1 => {
                    let perp = r_dir
                        .cross(&Vector3::new(0.0, 0.0, 1.0))
                        .try_normalize(1.0e-12)
                        .unwrap_or_else(|| Vector3::new(1.0, 0.0, 0.0));
                    let sign = if case % 26 == 1 { 1.0 } else { -1.0 };
                    let eps = rng.log_uniform(-560, -10);
                    s_dir = (r_dir * sign + perp * eps).normalize();
                    (r_dir * r_mag, v_dir * v_mag, s_dir * s_mag, m_dir * m_mag)
                }
                // Near-collinear r and v (near-rectilinear orbit).
                2 => {
                    let perp = r_dir
                        .cross(&Vector3::new(0.0, 1.0, 0.0))
                        .try_normalize(1.0e-12)
                        .unwrap_or_else(|| Vector3::new(1.0, 0.0, 0.0));
                    let eps = rng.log_uniform(-520, -10);
                    v_dir = (r_dir + perp * eps).normalize();
                    (r_dir * r_mag, v_dir * v_mag, s_dir * s_mag, m_dir * m_mag)
                }
                // Sun nearly along orbit normal (beta -> +/-90 deg).
                3 => {
                    let n_hat = r_dir
                        .cross(&v_dir)
                        .try_normalize(1.0e-12)
                        .unwrap_or_else(|| Vector3::new(0.0, 0.0, 1.0));
                    let eps = rng.log_uniform(-520, -10);
                    s_dir = (n_hat + r_dir * eps).normalize();
                    (r_dir * r_mag, v_dir * v_mag, s_dir * s_mag, m_dir * m_mag)
                }
                // Near-equatorial orbit (ECOM1 ascending node near-singular).
                4 => {
                    let eps = rng.log_uniform(-520, -10);
                    let phi = 2.0 * PI * rng.unit_f64();
                    let r_eq = Vector3::new(libm::cos(phi), libm::sin(phi), eps).normalize();
                    let v_eq = Vector3::new(-libm::sin(phi), libm::cos(phi), eps * 0.5).normalize();
                    (r_eq * r_mag, v_eq * v_mag, s_dir * s_mag, m_dir * m_mag)
                }
                // Multi-scale velocity components (dynamic range > 2^1074).
                5 => {
                    let small = rng.log_uniform(-1050, -650);
                    let big = rng.log_uniform(450, 900);
                    let v_ms = Vector3::new(small, big, rng.log_uniform(-1050, -700));
                    (r_dir * r_mag, v_ms, s_dir * s_mag, m_dir * m_mag)
                }
                // Near-axis lunar eclipse (annular / total / lit perturbation).
                6 if include_moon => {
                    let sat_p = Vector3::new(7_000.0, 0.0, 0.0);
                    let sun_p = Vector3::new(7_000.0, AU_KM, 0.0);
                    let dist = if case % 2 == 0 { 405_000.0 } else { 356_000.0 };
                    let eps = rng.log_uniform(-560, -20);
                    let moon_p = Vector3::new(7_000.0, dist, eps);
                    (sat_p, v_dir * v_mag, sun_p, moon_p)
                }
                // Exact collinear / degenerate cases.
                7 => {
                    m_dir = r_dir;
                    (r_dir * r_mag, r_dir * v_mag, s_dir * s_mag, m_dir * m_mag)
                }
                // > 2^1074 component span in Sun vector so normalized e_D flushes a component
                // to 0.0 and e_D x r vanishes in binary64 while unflushed e_D x r != 0.
                10 => {
                    let big = rng.log_uniform(380, 490);
                    let tiny = rng.log_uniform(-1020, -720);
                    let axis = (case / 13) % 3;
                    let mut r_f = Vector3::zeros();
                    let mut s_f = Vector3::zeros();
                    r_f[axis] = big;
                    s_f[axis] = 3.0 * big;
                    s_f[(axis + 1) % 3] = tiny;
                    let v_f = Vector3::new(1.0, 1.5, 2.0) * rng.log_uniform(-20, 20);
                    let mut m_f = Vector3::zeros();
                    m_f[(axis + 1) % 3] = 384_400.0;
                    (r_f, v_f, s_f, m_f)
                }
                // > 2^1074 component span in r x v so normalized Z flushes a component to 0.0
                // while Sun lies along Z's dominant axis (flushed Z x sun == 0, exact != 0).
                11 => {
                    let big = rng.log_uniform(300, 460);
                    let tiny = rng.log_uniform(-1020, -800);
                    let r_f = Vector3::new(big, 0.0, tiny);
                    let v_f = Vector3::new(0.0, rng.log_uniform(-10, 10), 0.0);
                    let s_f = Vector3::new(0.0, 0.0, rng.log_uniform(20, 400));
                    let m_f = Vector3::new(0.0, 384_400.0, 0.0);
                    (r_f, v_f, s_f, m_f)
                }
                // Subnormal |r x v|^2 and underflowing r . ahead / r . node for ECOM1 u.
                12 => {
                    if is_conical && case % 2 == 0 {
                        let r_tiny = Vector3::new(1.0, 1.0, 1.0) * rng.log_uniform(-1074, -1060);
                        let v_big = Vector3::new(0.0, 1.0, -1.0) * rng.log_uniform(600, 700);
                        (r_tiny, v_big, s_dir * s_mag, m_dir * m_mag)
                    } else {
                        let r_ok = Vector3::new(5_000.0, 5_000.0, 5_000.0);
                        let v_sub = Vector3::new(0.0, -1.0, 1.0) * rng.log_uniform(-555, -545);
                        (r_ok, v_sub, s_dir * s_mag, m_dir * m_mag)
                    }
                }
                // Near-node (|y| / x < 2^-1022) covering both binary64 subnormal and
                // sub-subnormal sin(u) = y / rho.
                13 => {
                    let big = rng.log_uniform(300, 480);
                    let sign = if (case / 16) % 2 == 0 { 1.0 } else { -1.0 };
                    let tiny = sign * rng.log_uniform(-1020, -730);
                    let r_nn = Vector3::new(big, 0.0, tiny);
                    let v_scale = rng.log_uniform(-20, 20);
                    let v_nn = Vector3::new(0.0, v_scale, v_scale);
                    let s_nn = Vector3::new(2.0 * big, big, 0.0);
                    let m_nn = Vector3::new(0.0, 384_400.0, 0.0);
                    (r_nn, v_nn, s_nn, m_nn)
                }
                // Exact non-axis-aligned collinear sun and sat (sun = 2^dk * sat) covering
                // |sat| > |sun|, |sun| > |sat|, and large-magnitude scaled fallback.
                14 => {
                    let a = ((rng.next_u64() % 13) + 1) as f64;
                    let b = ((rng.next_u64() % 13) + 1) as f64;
                    let c = ((rng.next_u64() % 13) + 1) as f64;
                    let base_v = Vector3::new(a, b, c);
                    let (er, dk) = match (case / 16) % 3 {
                        0 => (24, -4), // |sat| > |sun| (16x)
                        1 => (14, 14), // |sun| > |sat| (16384x)
                        _ => (502, 2), // |sat_m|^2 overflows -> ECOM2 scaled fallback
                    };
                    let r_col = base_v * libm::ldexp(1.0, er);
                    let s_col = r_col * libm::ldexp(1.0, dk);
                    let v_col = Vector3::new(0.0, 1.0, 0.5);
                    let m_col = Vector3::new(0.0, 384_400.0, 0.0);
                    (r_col, v_col, s_col, m_col)
                }
                // Tiny conical shadow gradient in antumbra (normal in 1/km -> subnormal in 1/m,
                // subnormal in 1/km, and sub-subnormal in 1/km).
                15 if is_conical => {
                    let sign = if (case / 16) % 2 == 0 { 1.0 } else { -1.0 };
                    let tiny_z = sign * rng.log_uniform(-1065, -985);
                    let r_tc = Vector3::new(-1.0e7, 1.0e4, tiny_z);
                    let v_tc = Vector3::new(0.0, 1.0, 1.0);
                    let s_tc = Vector3::new(AU_KM, 0.0, 0.0);
                    let m_tc = Vector3::new(0.0, 384_400.0, 0.0);
                    (r_tc, v_tc, s_tc, m_tc)
                }
                _ => (r_dir * r_mag, v_dir * v_mag, s_dir * s_mag, m_dir * m_mag),
            };

            let moon_opt = include_moon.then_some(m);
            let ld = lighting_ratio_double_scaled(
                r,
                s,
                moon_opt,
                e2.shadow_function(),
                earth_model,
                include_moon,
            );

            let a2 = e2.acceleration_with_sun_moon(r, v, s, m);
            let pp2 = e2.parameter_partials_with_sun_moon(r, v, s, m);
            let ad2 = e2.acceleration_partials_with_sun_moon(r, v, s, m);

            let a1 = e1.acceleration_with_sun_moon(r, v, s, m);
            let pp1 = e1.parameter_partials_with_sun_moon(r, v, s, m);
            let ad1 = e1.acceleration_partials_with_sun_moon(r, v, s, m);

            // 1. acceleration and parameter_partials accept and refuse the exact same inputs,
            //    except when the acceleration value itself overflows binary64 while the
            //    partials remain finite (in which case acceleration returns NumericalFailure
            //    and parameter_partials returns Ok).
            if a2.is_ok() {
                assert!(
                    pp2.is_ok(),
                    "ECOM2 cfg={cfg_idx} case={case} mode={mode}: accel Ok while pp Err: accel={a2:?}, pp={pp2:?}, r={r:?}, v={v:?}, s={s:?}, m={m:?}"
                );
            } else if pp2.is_ok() {
                assert!(
                    matches!(a2, Err(PropagationError::NumericalFailure(_))),
                    "ECOM2 cfg={cfg_idx} case={case} mode={mode}: expected NumericalFailure on value-only overflow, got accel={a2:?}"
                );
            }
            if a1.is_ok() {
                assert!(
                    pp1.is_ok(),
                    "ECOM1 cfg={cfg_idx} case={case} mode={mode}: accel Ok while pp Err: accel={a1:?}, pp={pp1:?}, r={r:?}, v={v:?}, s={s:?}, m={m:?}"
                );
            } else if pp1.is_ok() {
                assert!(
                    matches!(a1, Err(PropagationError::NumericalFailure(_))),
                    "ECOM1 cfg={cfg_idx} case={case} mode={mode}: expected NumericalFailure on value-only overflow, got accel={a1:?}"
                );
            }

            // 2. When acceleration succeeds and acceleration_partials fails, it must be
            //    NumericalFailure (derivative undefined or overflows binary64). Conversely,
            //    acceleration_partials returns Ok when acceleration fails only if the
            //    acceleration value itself overflows binary64 (NumericalFailure) while the
            //    partials remain finite.
            if a2.is_err() {
                if ad2.is_ok() {
                    assert!(
                        matches!(a2, Err(PropagationError::NumericalFailure(_))) && pp2.is_ok(),
                        "ECOM2 cfg={cfg_idx} case={case}: partials Ok when accel failed without value overflow: a2={a2:?}, pp2={pp2:?}"
                    );
                }
            } else if let Err(ref err) = ad2 {
                assert!(
                    matches!(err, PropagationError::NumericalFailure(_)),
                    "ECOM2 cfg={cfg_idx} case={case}: expected NumericalFailure for derivative-only refusal, got {err:?}, r={r:?}, v={v:?}, s={s:?}, m={m:?}"
                );
            }
            if a1.is_err() {
                if ad1.is_ok() {
                    assert!(
                        matches!(a1, Err(PropagationError::NumericalFailure(_))) && pp1.is_ok(),
                        "ECOM1 cfg={cfg_idx} case={case}: partials Ok when accel failed without value overflow: a1={a1:?}, pp1={pp1:?}"
                    );
                }
            } else if let Err(ref err) = ad1 {
                assert!(
                    matches!(err, PropagationError::NumericalFailure(_)),
                    "ECOM1 cfg={cfg_idx} case={case}: expected NumericalFailure for derivative-only refusal, got {err:?}, r={r:?}, v={v:?}, s={s:?}, m={m:?}"
                );
            }

            // 3. ECOM1 double (sin u, cos u) must equal the field value part bit for bit on every input.
            let u1_d = ecom1_u_double(as_array(r), as_array(v));
            let sc1_d = ecom1_sin_cos_double_scaled(
                ScaledVec3::from_vec3(Vec3::new(r.x, r.y, r.z)),
                ScaledVec3::from_vec3(Vec3::new(v.x, v.y, v.z)),
            );
            let sc1_f_c = ecom1_sin_cos_field_scaled(
                scaled_field_vec3_constant::<1>(r),
                scaled_field_vec3_constant::<1>(v),
            );
            assert_eq!(u1_d.is_some(), sc1_d.is_some());
            assert_eq!(sc1_d.is_some(), sc1_f_c.is_some());
            if let (Some((su_d, cu_d)), Some((su_fc, cu_fc))) = (sc1_d, sc1_f_c) {
                assert_eq!(
                    (su_d.m.to_bits(), su_d.e),
                    (su_fc.val.m.to_bits(), su_fc.val.e),
                    "ECOM1 sin(u) double vs const-field mismatch cfg={cfg_idx} case={case} mode={mode}"
                );
                assert_eq!(
                    (cu_d.m.to_bits(), cu_d.e),
                    (cu_fc.val.m.to_bits(), cu_fc.val.e),
                    "ECOM1 cos(u) double vs const-field mismatch cfg={cfg_idx} case={case} mode={mode}"
                );
                let (su_f6, cu_f6) = ecom1_sin_cos_field_scaled(
                    scaled_field_vec3_variable::<6>(r, 0),
                    scaled_field_vec3_variable::<6>(v, 3),
                )
                .expect("ECOM1 field6 sin/cos must match double");
                assert_eq!(
                    (su_d.m.to_bits(), su_d.e),
                    (su_f6.val.m.to_bits(), su_f6.val.e),
                    "ECOM1 sin(u) double vs state-field mismatch cfg={cfg_idx} case={case} mode={mode}"
                );
                assert_eq!(
                    (cu_d.m.to_bits(), cu_d.e),
                    (cu_f6.val.m.to_bits(), cu_f6.val.e),
                    "ECOM1 cos(u) double vs state-field mismatch cfg={cfg_idx} case={case} mode={mode}"
                );
                total_u_bit_exact += 1;
            }

            let Ok((ratio_s_d, is_sunlit_d)) = ld else {
                assert!(a2.is_err() && a1.is_err());
                total_refused += 1;
                continue;
            };
            let _ = is_sunlit_d;
            let ratio = ratio_s_d.to_f64();
            if ratio_s_d.is_zero() {
                assert_eq!(a2.unwrap(), Vector3::zeros());
                assert_eq!(
                    pp2.unwrap(),
                    vec![Vector3::zeros(); ECOM2_D4B1_COEFFICIENT_COUNT]
                );
                assert_eq!(a1.unwrap(), Vector3::zeros());
                assert_eq!(pp1.unwrap(), [Vector3::zeros(); ECOM1_COEFFICIENT_COUNT]);
                let lf_u = lighting_ratio_field(
                    r,
                    s,
                    moon_opt,
                    e2.shadow_function(),
                    earth_model,
                    include_moon,
                );
                if lf_u.is_ok() {
                    assert_eq!(ad2.unwrap(), AccelerationPartials::zeros());
                    assert_eq!(ad1.unwrap(), AccelerationPartials::zeros());
                } else {
                    assert!(matches!(ad2, Err(PropagationError::NumericalFailure(_))));
                    assert!(matches!(ad1, Err(PropagationError::NumericalFailure(_))));
                }
                total_umbra += 1;
                continue;
            }

            let lf_ok = lighting_ratio_field(
                r,
                s,
                moon_opt,
                e2.shadow_function(),
                earth_model,
                include_moon,
            )
            .ok();

            let sat_m_opt = to_vec3_m(r).ok();
            let sun_m_opt = to_vec3_m(s).ok();
            let raw_dyb = match (sat_m_opt, sun_m_opt) {
                (Some(sm), Some(sunm)) => raw_orekit_dyb_double(sm, sunm),
                _ => None,
            };
            let dyb_subnormal = match (sat_m_opt, sun_m_opt) {
                (Some(sm), Some(sunm)) => raw_orekit_dyb_has_subnormal_n2(sm, sunm),
                _ => false,
            };
            let raw_du = match (sat_m_opt, sun_m_opt) {
                (Some(sm), Some(sunm)) => raw_orekit_delta_u_double(sm, v, sunm),
                _ => None,
            };
            let du_subnormal = match (sat_m_opt, sun_m_opt) {
                (Some(sm), Some(sunm)) => raw_orekit_delta_u_has_subnormal_n2(sm, v, sunm),
                _ => false,
            };

            let rg = reference_geometry(r, v, s);

            if (mode == 10 || mode == 11) && a2.is_ok() && a1.is_ok() {
                total_flushed_frame_ok += 1;
            }
            if is_near_max_coeff && a2.is_ok() && a1.is_ok() {
                total_overflow_coeff_ok += 1;
            }

            // Dedicated independent verification of the three sweep classes:
            // Mode 13: near-node (|y| / x < 2^-1022).
            if mode == 13 && ratio == 1.0 {
                let big_s = ScaledF64::from_f64(r.x);
                let tiny_s = ScaledF64::from_f64(r.z);
                let sqrt2_s = ScaledF64::from_f64(2.0_f64.sqrt());
                let sin_ref = sqrt2_s.mul(tiny_s).div(big_s);
                let (su_got, cu_got) = sc1_d.expect("near-node sin/cos defined");
                let su_rel = su_got.sub(sin_ref).div(sin_ref).to_f64().abs();
                let cu_err = cu_got.sub(ScaledF64::from_f64(1.0)).to_f64().abs();
                assert!(
                    su_rel <= 8.0 * f64::EPSILON && cu_err <= 8.0 * f64::EPSILON,
                    "mode 13 sin/cos mismatch cfg={cfg_idx} case={case}: su_rel={su_rel:e} cu_err={cu_err:e}"
                );
                // Probe with ds scaled so ds * sin(u) / 1000 is normal even for |y|/x < 2^-1075.
                let ds_exp = (big_s.e - tiny_s.e).min(1020);
                let ds_probe = libm::ldexp(1.0, ds_exp);
                let probe = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
                    ds: ds_probe,
                    ..Default::default()
                })
                .unwrap()
                .with_shadow_function(shadow_fn)
                .with_earth_shadow_model(earth_model)
                .with_moon_eclipse(include_moon);
                let a_p = probe
                    .acceleration_with_sun_moon(r, v, s, m)
                    .expect("near-node probe accel");
                let ad_p = probe
                    .acceleration_partials_with_sun_moon(r, v, s, m)
                    .expect("near-node probe partials");
                // Closed form: a.x = a.y = (ds / 1000) * (tiny / big),
                // wrt_position[(2, 2)] = -2 * (ds / 1000) * (tiny / big^2).
                let ds_km_s = ScaledF64::from_f64(ds_probe).div(ScaledF64::from_f64(M_PER_KM));
                let ax_ref = ds_km_s.mul(tiny_s).div(big_s).to_f64();
                let p22_ref = ScaledF64::from_f64(-2.0)
                    .mul(ds_km_s)
                    .mul(tiny_s)
                    .div(big_s.mul(big_s))
                    .to_f64();
                if ax_ref.abs() >= f64::MIN_POSITIVE {
                    let a_rel = ((a_p.x - ax_ref).abs().max((a_p.y - ax_ref).abs())) / ax_ref.abs();
                    max_near_node_a1_eps = max_near_node_a1_eps.max(a_rel / f64::EPSILON);
                    assert!(
                        a_rel <= 16.0 * f64::EPSILON,
                        "mode 13 accel rel={a_rel:e} got={a_p:?} ref={ax_ref:e}"
                    );
                }
                if p22_ref.abs() >= f64::MIN_POSITIVE {
                    let p_rel = (ad_p.wrt_position[(2, 2)] - p22_ref).abs() / p22_ref.abs();
                    max_near_node_ad1_eps = max_near_node_ad1_eps.max(p_rel / f64::EPSILON);
                    assert!(
                        p_rel <= 16.0 * f64::EPSILON,
                        "mode 13 wrt_position[(2,2)] rel={p_rel:e} got={:e} ref={p22_ref:e}",
                        ad_p.wrt_position[(2, 2)]
                    );
                }
                total_near_node_checked += 1;
            }

            // Mode 14: exact non-axis-aligned collinear sun and sat.
            if mode == 14 {
                assert!(
                    matches!(a1, Err(PropagationError::NumericalFailure(_)))
                        && matches!(pp1, Err(PropagationError::NumericalFailure(_)))
                        && matches!(ad1, Err(PropagationError::NumericalFailure(_))),
                    "mode 14 ECOM1 full must return NumericalFailure: a1={a1:?} pp1={pp1:?} ad1={ad1:?}"
                );
                let d_only1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
                    d0: 1.0,
                    dc: 0.5,
                    ds: -0.25,
                    ..Default::default()
                })
                .unwrap()
                .with_shadow_function(shadow_fn)
                .with_earth_shadow_model(earth_model)
                .with_moon_eclipse(include_moon);
                let a1_d = d_only1
                    .acceleration_with_sun_moon(r, v, s, m)
                    .expect("mode 14 ECOM1 D-only accel Ok");
                assert!(a1_d.iter().all(|x| x.is_finite()));
                let ad1_d = d_only1
                    .acceleration_partials_with_sun_moon(r, v, s, m)
                    .expect("mode 14 ECOM1 D-only partials Ok");
                assert!(
                    ad1_d.wrt_position.iter().all(|x| x.is_finite())
                        && ad1_d.wrt_velocity.iter().all(|x| x.is_finite())
                );
                assert!(matches!(
                    d_only1.parameter_partials_with_sun_moon(r, v, s, m),
                    Err(PropagationError::NumericalFailure(_))
                ));
                // ECOM2 (both fast path and scaled fallback) must refuse exact collinearity.
                assert!(
                    matches!(a2, Err(PropagationError::NumericalFailure(_)))
                        && matches!(pp2, Err(PropagationError::NumericalFailure(_)))
                        && matches!(ad2, Err(PropagationError::NumericalFailure(_))),
                    "mode 14 ECOM2 must return NumericalFailure: a2={a2:?} pp2={pp2:?} ad2={ad2:?}"
                );
                let d_only2 = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
                    d0: 1.0,
                    ..Default::default()
                })
                .unwrap()
                .with_shadow_function(shadow_fn)
                .with_earth_shadow_model(earth_model)
                .with_moon_eclipse(include_moon);
                assert!(d_only2.acceleration_with_sun_moon(r, v, s, m).is_ok());
                assert!(d_only2
                    .acceleration_partials_with_sun_moon(r, v, s, m)
                    .is_ok());
                assert!(matches!(
                    d_only2.parameter_partials_with_sun_moon(r, v, s, m),
                    Err(PropagationError::NumericalFailure(_))
                ));
                total_collinear_checked += 1;
            }

            // Mode 15: tiny conical shadow gradient in antumbra.
            if mode == 15 && is_conical {
                let d0_probe = 1.0e308;
                let p1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
                    d0: d0_probe,
                    ..Default::default()
                })
                .unwrap()
                .with_shadow_function(ShadowFunction::Conical)
                .with_earth_shadow_model(earth_model);
                let p2 = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
                    d0: d0_probe,
                    ..Default::default()
                })
                .unwrap()
                .with_shadow_function(ShadowFunction::Conical)
                .with_earth_shadow_model(earth_model);
                let ad1_tc = p1
                    .acceleration_partials_with_sun(r, v, s)
                    .expect("mode 15 ECOM1 tiny conical partials");
                let ad2_tc = p2
                    .acceleration_partials_with_sun(r, v, s)
                    .expect("mode 15 ECOM2 tiny conical partials");
                // Closed form for r = (-1e7, 1e4, tiny_z), s = (AU_KM, 0, 0):
                //   r_pen = R_earth + 1e7 * tan(asin((R_sun + R_earth) / AU_KM)),
                //   d(ratio)/dr_z = (tiny_z * z_scale^2) / (1e4 * r_pen),
                //   wrt_position[(0, 2)] = (d0 / 1000) * (d(ratio)/dr_z * e_D.x + ratio * d(e_D.x)/dr_z).
                let z_scale = match earth_model {
                    EarthShadowModel::Spherical => 1.0,
                    EarthShadowModel::Wgs84Oblate => {
                        1.0 / (1.0 - crate::astro::events::eclipse::WGS84_FLATTENING)
                    }
                };
                let r_earth = crate::astro::constants::earth::MEAN_EARTH_RADIUS_KM;
                let r_sun = crate::astro::constants::astro::SOLAR_RADIUS_KM;
                let tan_p = libm::tan(libm::asin((r_sun + r_earth) / AU_KM));
                let r_pen = r_earth + (-r.x) * tan_p;
                let dx_s = ScaledF64::from_f64(s.x - r.x);
                let dy_s = ScaledF64::from_f64(-r.y);
                let dz_s = ScaledF64::from_f64(-r.z);
                let dist_s = ScaledVec3 {
                    x: dx_s,
                    y: dy_s,
                    z: dz_s,
                }
                .norm()
                .unwrap();
                let ed_x_s = dx_s.div(dist_s);
                let ed_z_s = dz_s.div(dist_s);
                let ded_x_dz_s = ed_x_s.mul(ed_z_s).div(dist_s);
                let dratio_dz_s = ScaledF64::from_f64(r.z)
                    .mul(ScaledF64::from_f64(z_scale))
                    .mul(ScaledF64::from_f64(z_scale))
                    .div(ScaledF64::from_f64(r.y).mul(ScaledF64::from_f64(r_pen)));
                let ratio_s = ScaledF64::from_f64(ratio);
                let d0_km_s = ScaledF64::from_f64(d0_probe).div(ScaledF64::from_f64(M_PER_KM));
                let expected_02 = d0_km_s
                    .mul(ScaledF64::lincomb2(
                        dratio_dz_s,
                        ed_x_s,
                        ratio_s,
                        ded_x_dz_s,
                    ))
                    .to_f64();
                let rel1 = (ad1_tc.wrt_position[(0, 2)] - expected_02).abs() / expected_02.abs();
                let rel2 = (ad2_tc.wrt_position[(0, 2)] - expected_02).abs() / expected_02.abs();
                max_tiny_conical_e1_eps = max_tiny_conical_e1_eps.max(rel1 / f64::EPSILON);
                max_tiny_conical_e2_eps = max_tiny_conical_e2_eps.max(rel2 / f64::EPSILON);
                assert!(
                    rel1 <= 16.0 * f64::EPSILON,
                    "mode 15 ECOM1 (0,2) rel={rel1:e} got={:e} ref={expected_02:e}",
                    ad1_tc.wrt_position[(0, 2)]
                );
                assert!(
                    rel2 <= 16.0 * f64::EPSILON,
                    "mode 15 ECOM2 (0,2) rel={rel2:e} got={:e} ref={expected_02:e}",
                    ad2_tc.wrt_position[(0, 2)]
                );
                total_tiny_conical_checked += 1;
            }

            // ApparentDisc ecom_geometry_with_shadow consistency.
            if !is_conical {
                let geom = ecom_geometry_with_shadow(r, v, s, moon_opt, earth_model);
                match (&geom, &a2, &a1) {
                    (Ok(g), _, _) => {
                        assert_eq!(
                            g.delta_u_rad.is_some(),
                            a2.is_ok(),
                            "cfg={cfg_idx} case={case} mode={mode}: geom.delta_u {:?} vs ECOM2 {:?}",
                            g.delta_u_rad,
                            a2
                        );
                        assert_eq!(
                            g.argument_of_latitude_rad.is_some(),
                            a1.is_ok(),
                            "cfg={cfg_idx} case={case} mode={mode}: geom.u {:?} vs ECOM1 {:?}",
                            g.argument_of_latitude_rad,
                            a1
                        );
                        assert_eq!(
                            g.argument_of_latitude_rad.map(f64::to_bits),
                            u1_d.map(f64::to_bits),
                            "cfg={cfg_idx} case={case} mode={mode}: geom.u vs ecom1_u_double bits"
                        );
                        if let Some([ed_raw, ey_raw, eb_raw]) = raw_dyb {
                            assert_eq!(g.dyb.d[0].to_bits(), ed_raw.x.to_bits());
                            assert_eq!(g.dyb.d[1].to_bits(), ed_raw.y.to_bits());
                            assert_eq!(g.dyb.d[2].to_bits(), ed_raw.z.to_bits());
                            assert_eq!(g.dyb.y[0].to_bits(), ey_raw.x.to_bits());
                            assert_eq!(g.dyb.y[1].to_bits(), ey_raw.y.to_bits());
                            assert_eq!(g.dyb.y[2].to_bits(), ey_raw.z.to_bits());
                            assert_eq!(g.dyb.b[0].to_bits(), eb_raw.x.to_bits());
                            assert_eq!(g.dyb.b[1].to_bits(), eb_raw.y.to_bits());
                            assert_eq!(g.dyb.b[2].to_bits(), eb_raw.z.to_bits());
                            if dyb_subnormal {
                                total_subnormal_fast_bits += 1;
                            }
                        }
                        if !dyb_subnormal && rg.single_scale_rs {
                            if let Some([ed_ref, ey_ref, eb_ref]) = rg.dyb {
                                let ed_g = Vector3::from(g.dyb.d);
                                let ey_g = Vector3::from(g.dyb.y);
                                let eb_g = Vector3::from(g.dyb.b);
                                // Measured max 0.0 eps; 4 eps covers normalization rounding
                                // between power-of-two rescaling and ScaledVec3.
                                assert!(
                                    (ed_g - ed_ref).norm() <= 4.0 * f64::EPSILON,
                                    "cfg={cfg_idx} case={case}: e_D mismatch {ed_g:?} vs {ed_ref:?}"
                                );
                                if rg.sin_ed_r > 1.0e-12 {
                                    // 1-ULP perturbation of e_D amplified by 1/sin(e_D, r) in
                                    // e_D x r (measured max 0.0 eps; bound <= 1.78e-3 << 2.0).
                                    let tol = 8.0 * f64::EPSILON / rg.sin_ed_r;
                                    assert!(
                                        (ey_g - ey_ref).norm() <= tol,
                                        "cfg={cfg_idx} case={case}: e_Y mismatch {ey_g:?} vs {ey_ref:?} (sin={})",
                                        rg.sin_ed_r
                                    );
                                    assert!(
                                        (eb_g - eb_ref).norm() <= tol,
                                        "cfg={cfg_idx} case={case}: e_B mismatch {eb_g:?} vs {eb_ref:?} (sin={})",
                                        rg.sin_ed_r
                                    );
                                }
                            }
                        }
                        if let Some(du_raw) = raw_du {
                            assert_eq!(g.delta_u_rad.unwrap().to_bits(), du_raw.to_bits());
                            if du_subnormal {
                                total_subnormal_fast_bits += 1;
                            }
                        }
                        if !du_subnormal && rg.single_scale_rs && rg.single_scale_rv {
                            if let (Some(du_g), Some(du_r)) = (g.delta_u_rad, rg.du) {
                                if rg.sin_du > 1.0e-12 {
                                    // Rounding of r x v and z x s amplified by 1/sin_du plus
                                    // atan2 rounding (measured max 2.0 eps; bound <= 1.78e-3 << pi).
                                    let tol = 8.0 * f64::EPSILON / rg.sin_du;
                                    assert!(
                                        angle_diff(du_g, du_r) <= tol,
                                        "cfg={cfg_idx} case={case}: delta_u mismatch {du_g} vs {du_r} (sin={})",
                                        rg.sin_du
                                    );
                                }
                            }
                        }
                        // ECOM1 u is always evaluated in ScaledF64/ScaledVec3, so check every
                        // single-scale case including subnormal |r x v|^2 / underflowing dots.
                        if rg.single_scale_rv {
                            if let (Some(u_g), Some(u_r)) = (g.argument_of_latitude_rad, rg.u) {
                                if rg.sin_u > 1.0e-12 {
                                    // Rounding of r x v and ascending node amplified by 1/sin_u
                                    // plus atan2 rounding (bound <= 1.78e-3 << pi).
                                    let tol = 8.0 * f64::EPSILON / rg.sin_u;
                                    assert!(
                                        angle_diff(u_g, u_r) <= tol,
                                        "cfg={cfg_idx} case={case} mode={mode}: u mismatch {u_g} vs {u_r} (sin={})",
                                        rg.sin_u
                                    );
                                }
                            }
                        }
                    }
                    (Err(_), _, _) => {
                        assert!(
                            a2.is_err() && a1.is_err(),
                            "cfg={cfg_idx} case={case} mode={mode}: geom Err while a2={a2:?} a1={a1:?}"
                        );
                    }
                }
            } else if rg.single_scale_rv {
                // Under Conical shadow, still check ECOM1 u directly against rg.u on all cases.
                if let (Some(u_g), Some(u_r)) = (u1_d, rg.u) {
                    if rg.sin_u > 1.0e-12 {
                        let tol = 8.0 * f64::EPSILON / rg.sin_u;
                        assert!(
                            angle_diff(u_g, u_r) <= tol,
                            "cfg={cfg_idx} case={case} mode={mode}: conical u mismatch {u_g} vs {u_r} (sin={})",
                            rg.sin_u
                        );
                    }
                }
            }

            // Cross-check ECOM2 acceleration, parameter partials, and state partials.
            if let (Ok(acc), Ok(pp)) = (&a2, &pp2) {
                total_ok_ecom2 += 1;
                let exact_frame_undef =
                    is_collinear_exact(s, r) || is_delta_u_undefined_exact(r, v, s);
                if (!is_conical || is_sunlit_d) && !exact_frame_undef {
                    if let Some(raw_acc) =
                        raw_orekit_ecom2_accel_double(r, v, s, ratio, 2, 1, e2.coefficients())
                    {
                        for axis in 0..3 {
                            assert_eq!(
                                acc[axis].to_bits(),
                                raw_acc[axis].to_bits(),
                                "ECOM2 cfg={cfg_idx} case={case} axis={axis}: accel bit mismatch"
                            );
                        }
                    }
                }
                if is_conical && !is_sunlit_d {
                    // Under Conical outside full sunlight, double acceleration must equal the
                    // N=9 parameter-partials field value bit for bit.
                    let params_g9: Vec<Grad<9>> = e2
                        .coefficients()
                        .iter()
                        .enumerate()
                        .map(|(k, &c)| Grad::variable(k, c))
                        .collect();
                    let accel_s9 = ecom2_acceleration_field_scaled(
                        r,
                        v,
                        s,
                        false,
                        false,
                        ScaledGrad {
                            val: ratio_s_d,
                            grad: [ScaledF64::ZERO; 9],
                        },
                        false,
                        2,
                        1,
                        &params_g9,
                        e2.coefficients(),
                        true,
                    )
                    .expect("ECOM2 conical field9 must succeed when double succeeded");
                    let val9 = accel_s9
                        .val_vec()
                        .to_vec3()
                        .expect("ECOM2 conical field9 val finite");
                    assert_eq!(acc.x.to_bits(), val9.x.to_bits());
                    assert_eq!(acc.y.to_bits(), val9.y.to_bits());
                    assert_eq!(acc.z.to_bits(), val9.z.to_bits());
                }
                if !dyb_subnormal && !du_subnormal {
                    let (recon, sum_abs) = linear_recon(e2.coefficients(), pp);
                    let sum_abs_f = sum_abs.to_f64();
                    if sum_abs_f >= f64::MIN_POSITIVE {
                        let rel = vec3_scaled_rel_err(*acc, recon, sum_abs);
                        max_lin2_eps = max_lin2_eps.max(rel / f64::EPSILON);
                        // Measured max 13.14 eps across 2,133 cases (including near-f64::MAX
                        // coefficients and flushed-component frames).
                        assert!(
                            rel <= 64.0 * f64::EPSILON,
                            "ECOM2 cfg={cfg_idx} case={case} mode={mode}: accel vs recon rel={rel:e}"
                        );
                    } else {
                        let diff = (*acc - recon).norm();
                        let bound = 64.0 * f64::EPSILON * sum_abs_f + 4.0 * f64::from_bits(1);
                        assert!(
                            diff <= bound,
                            "ECOM2 cfg={cfg_idx} case={case} mode={mode}: subnormal accel vs recon diff={diff:e} > {bound:e}"
                        );
                    }
                    if rg.single_scale_rs {
                        if let Some([ed_ref, ey_ref, eb_ref]) = rg.dyb {
                            let scale = ratio * inv_km;
                            assert!(
                                (pp[3] - ed_ref * scale).norm() <= 16.0 * f64::EPSILON * scale,
                                "ECOM2 cfg={cfg_idx} case={case}: D0 partial vs ref e_D"
                            );
                            if rg.sin_ed_r > 1.0e-12 {
                                let tol = 16.0 * f64::EPSILON / rg.sin_ed_r * scale;
                                assert!(
                                    (pp[8] - ey_ref * scale).norm() <= tol,
                                    "ECOM2 cfg={cfg_idx} case={case}: Y0 partial vs ref e_Y"
                                );
                                assert!(
                                    (pp[0] - eb_ref * scale).norm() <= tol,
                                    "ECOM2 cfg={cfg_idx} case={case}: B0 partial vs ref e_B"
                                );
                                if rg.single_scale_rv {
                                    if let Some(du_r) = rg.du {
                                        if rg.sin_du > 1.0e-12 {
                                            let dtol = 16.0 * f64::EPSILON
                                                / rg.sin_du.min(rg.sin_ed_r)
                                                * scale;
                                            assert!(
                                                (pp[1] - eb_ref * (scale * libm::cos(du_r))).norm()
                                                    <= dtol,
                                                "ECOM2 cfg={cfg_idx} case={case}: B1c partial vs ref delta_u"
                                            );
                                            assert!(
                                                (pp[2] - eb_ref * (scale * libm::sin(du_r))).norm()
                                                    <= dtol,
                                                "ECOM2 cfg={cfg_idx} case={case}: B1s partial vs ref delta_u"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (!is_conical || is_sunlit_d) && !exact_frame_undef {
                    if let (Ok(r_m), Ok(s_m), Some(ratio3)) = (to_vec3_m(r), to_vec3_m(s), lf_ok) {
                        let params_p: Vec<Grad<9>> = e2
                            .coefficients()
                            .iter()
                            .enumerate()
                            .map(|(k, &c)| Grad::variable(k, c))
                            .collect();
                        if let Some(raw_pp_f) = raw_orekit_ecom2_field(
                            FieldVec3::<9>::constant(r_m),
                            v,
                            false,
                            FieldVec3::<9>::constant(s_m),
                            Grad::<9>::constant(ratio3.value),
                            is_sunlit_field(ratio3),
                            2,
                            1,
                            &params_p,
                        ) {
                            for k in 0..9 {
                                assert_eq!(
                                    pp[k].x.to_bits(),
                                    (raw_pp_f.x.grad[k] / M_PER_KM).to_bits()
                                );
                                assert_eq!(
                                    pp[k].y.to_bits(),
                                    (raw_pp_f.y.grad[k] / M_PER_KM).to_bits()
                                );
                                assert_eq!(
                                    pp[k].z.to_bits(),
                                    (raw_pp_f.z.grad[k] / M_PER_KM).to_bits()
                                );
                            }
                        }
                    }
                }
            }

            // Cross-check ECOM1 acceleration & parameter partials against independent
            // power-of-two rescaled references and assert double-vs-field value bit identity.
            if let (Ok(acc), Ok(pp)) = (&a1, &pp1) {
                total_ok_ecom1 += 1;
                // Double acceleration must equal the N=9 parameter-partials field value bit for bit.
                if let Ok(ParameterLighting::Lit {
                    ratio: p_ratio,
                    is_sunlit: p_sunlit,
                }) = lighting_ratio_for_parameter_partials(
                    r,
                    s,
                    moon_opt,
                    e1.shadow_function(),
                    earth_model,
                    include_moon,
                ) {
                    if (p_ratio.m.to_bits(), p_ratio.e) == (ratio_s_d.m.to_bits(), ratio_s_d.e) {
                        let c1_arr = e1.coefficients().to_array();
                        let one = ScaledF64::from_f64(1.0);
                        let c_s = core::array::from_fn(|k| {
                            let mut grad = [ScaledF64::ZERO; 9];
                            grad[k] = one;
                            ScaledGrad {
                                val: ScaledF64::from_f64(c1_arr[k]),
                                grad,
                            }
                        });
                        let accel_s9 = ecom1_acceleration_field(
                            scaled_field_vec3_constant::<9>(r),
                            scaled_field_vec3_constant::<9>(v),
                            scaled_field_vec3_constant::<9>(s),
                            ScaledGrad {
                                val: p_ratio,
                                grad: [ScaledF64::ZERO; 9],
                            },
                            p_sunlit,
                            c_s,
                            c1_arr,
                            true,
                        )
                        .expect("ECOM1 field9 must succeed when double succeeded");
                        let val9 = accel_s9
                            .val_vec()
                            .to_vec3()
                            .expect("ECOM1 field9 val finite");
                        assert_eq!(
                            acc.x.to_bits(),
                            val9.x.to_bits(),
                            "ECOM1 double vs field9 x bit mismatch cfg={cfg_idx} case={case} mode={mode}"
                        );
                        assert_eq!(
                            acc.y.to_bits(),
                            val9.y.to_bits(),
                            "ECOM1 double vs field9 y bit mismatch cfg={cfg_idx} case={case} mode={mode}"
                        );
                        assert_eq!(
                            acc.z.to_bits(),
                            val9.z.to_bits(),
                            "ECOM1 double vs field9 z bit mismatch cfg={cfg_idx} case={case} mode={mode}"
                        );
                        total_ecom1_val_bit_exact += 1;
                    }
                }

                let c1_arr = e1.coefficients().to_array();
                let (recon, sum_abs) = linear_recon(&c1_arr, pp);
                let sum_abs_f = sum_abs.to_f64();
                if sum_abs_f >= f64::MIN_POSITIVE {
                    let rel = vec3_scaled_rel_err(*acc, recon, sum_abs);
                    max_lin1_eps = max_lin1_eps.max(rel / f64::EPSILON);
                    // Linear reconstruction sum_k c_k * pp[k] vs direct evaluation
                    // (measured max 13.18 eps across 2,133 cases, including subnormal
                    // |r x v|^2, near-f64::MAX coefficients, and flushed frames).
                    assert!(
                        rel <= 64.0 * f64::EPSILON,
                        "ECOM1 cfg={cfg_idx} case={case} mode={mode}: accel vs recon rel={rel:e}"
                    );
                } else {
                    let diff = (*acc - recon).norm();
                    let bound = 64.0 * f64::EPSILON * sum_abs_f + 4.0 * f64::from_bits(1);
                    assert!(
                        diff <= bound,
                        "ECOM1 cfg={cfg_idx} case={case} mode={mode}: subnormal accel vs recon diff={diff:e} > {bound:e}"
                    );
                }
                if rg.single_scale_rs {
                    if let Some([ed_ref, ey_ref, eb_ref]) = rg.dyb_ecom1 {
                        let scale = ratio * inv_km;
                        let d0_rel = (pp[0] - ed_ref * scale).norm() / scale;
                        max_pp1_ref_eps = max_pp1_ref_eps.max(d0_rel / f64::EPSILON);
                        assert!(
                            d0_rel <= 16.0 * f64::EPSILON,
                            "ECOM1 cfg={cfg_idx} case={case}: D0 partial vs ref e_D rel={d0_rel:e}"
                        );
                        if rg.sin_ey_ecom1 > 1.0e-12 {
                            // ScaledVec3 direct km evaluation vs unscaled power-of-two rescaled
                            // metre reference, amplified by 1/sin_ey (measured max 13.10 eps
                            // across 977 cases; bound <= 7.11e-3 << 1).
                            let y0_rel = (pp[3] - ey_ref * scale).norm() / scale;
                            let b0_rel = (pp[6] - eb_ref * scale).norm() / scale;
                            max_pp1_ref_eps = max_pp1_ref_eps
                                .max(y0_rel * rg.sin_ey_ecom1 / f64::EPSILON)
                                .max(b0_rel * rg.sin_ey_ecom1 / f64::EPSILON);
                            let tol = 32.0 * f64::EPSILON / rg.sin_ey_ecom1;
                            assert!(
                                y0_rel <= tol,
                                "ECOM1 cfg={cfg_idx} case={case}: Y0 partial vs ref e_Y rel={y0_rel:e}"
                            );
                            assert!(
                                b0_rel <= tol,
                                "ECOM1 cfg={cfg_idx} case={case}: B0 partial vs ref e_B rel={b0_rel:e}"
                            );
                            if rg.single_scale_rv {
                                if let Some(u_r) = rg.u {
                                    let sin_both = rg.sin_ey_ecom1.min(rg.sin_u);
                                    if sin_both > 1.0e-12 {
                                        let cu_r = libm::cos(u_r);
                                        let su_r = libm::sin(u_r);
                                        let expected_pp = [
                                            (1, ed_ref * (scale * cu_r)),
                                            (2, ed_ref * (scale * su_r)),
                                            (4, ey_ref * (scale * cu_r)),
                                            (5, ey_ref * (scale * su_r)),
                                            (7, eb_ref * (scale * cu_r)),
                                            (8, eb_ref * (scale * su_r)),
                                        ];
                                        let utol = 32.0 * f64::EPSILON / sin_both;
                                        for (k, exp_v) in expected_pp {
                                            let rel_k = (pp[k] - exp_v).norm() / scale;
                                            max_pp1_ref_eps = max_pp1_ref_eps
                                                .max(rel_k * sin_both / f64::EPSILON);
                                            assert!(
                                                rel_k <= utol,
                                                "ECOM1 cfg={cfg_idx} case={case} k={k}: partial vs ref rel={rel_k:e} > {utol:e}"
                                            );
                                        }
                                        total_pp1_ref_checked += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // State partials verification (Ok-ness and values on fast path & fallback).
            if let Some(ratio3) = lf_ok {
                let (ref_e2_ad, ref_e1_ad) = rescaled_state_partials_ref(
                    r,
                    v,
                    s,
                    ratio3,
                    e2.coefficients(),
                    e1.coefficients().to_array(),
                );
                if let Ok(acc2) = &a2 {
                    let exact_frame_undef =
                        is_collinear_exact(s, r) || is_delta_u_undefined_exact(r, v, s);
                    let raw_ad2 = if (!is_conical || is_sunlit_d) && !exact_frame_undef {
                        match (sat_m_opt, sun_m_opt) {
                            (Some(r_m), Some(s_m)) => {
                                let sat_f = FieldVec3::<6>::new(
                                    Grad::variable(0, r_m.x),
                                    Grad::variable(1, r_m.y),
                                    Grad::variable(2, r_m.z),
                                );
                                let sun_f = FieldVec3::<6>::constant(s_m);
                                let params_g: Vec<Grad<6>> = e2
                                    .coefficients()
                                    .iter()
                                    .map(|&c| Grad::constant(c))
                                    .collect();
                                raw_orekit_ecom2_field(
                                    sat_f,
                                    v,
                                    true,
                                    sun_f,
                                    Grad::<6>::from_pos3(ratio3),
                                    is_sunlit_field(ratio3),
                                    2,
                                    1,
                                    &params_g,
                                )
                                .map(extract_state_partials)
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if is_conical && !is_sunlit_d {
                        // Under Conical outside full sunlight, double acceleration must equal the
                        // N=6 state-partials field value bit for bit.
                        let (r_km, _) =
                            lighting_ratio_field_conical_scaled_km(r, s, earth_model).unwrap();
                        let ratio_s6 = ScaledGrad {
                            val: r_km.val,
                            grad: [
                                r_km.grad[0],
                                r_km.grad[1],
                                r_km.grad[2],
                                ScaledF64::ZERO,
                                ScaledF64::ZERO,
                                ScaledF64::ZERO,
                            ],
                        };
                        let params_g6: Vec<Grad<6>> = e2
                            .coefficients()
                            .iter()
                            .map(|&c| Grad::constant(c))
                            .collect();
                        if let Ok(accel_s6) = ecom2_acceleration_field_scaled(
                            r,
                            v,
                            s,
                            true,
                            true,
                            ratio_s6,
                            false,
                            2,
                            1,
                            &params_g6,
                            e2.coefficients(),
                            false,
                        ) {
                            let val6 = accel_s6.val_vec().to_vec3().unwrap();
                            assert_eq!(acc2.x.to_bits(), val6.x.to_bits());
                            assert_eq!(acc2.y.to_bits(), val6.y.to_bits());
                            assert_eq!(acc2.z.to_bits(), val6.z.to_bits());
                        }
                    }
                    if let Some(expected) = raw_ad2 {
                        let actual = ad2.as_ref().unwrap_or_else(|e| {
                            panic!(
                                "ECOM2 cfg={cfg_idx} case={case}: fast-path partials failed: {e:?}"
                            )
                        });
                        for ri in 0..3 {
                            for ci in 0..3 {
                                assert_eq!(
                                    actual.wrt_position[(ri, ci)].to_bits(),
                                    expected.wrt_position[(ri, ci)].to_bits(),
                                    "ECOM2 cfg={cfg_idx} case={case} pos({ri},{ci}) bit mismatch"
                                );
                                assert_eq!(
                                    actual.wrt_velocity[(ri, ci)].to_bits(),
                                    expected.wrt_velocity[(ri, ci)].to_bits(),
                                    "ECOM2 cfg={cfg_idx} case={case} vel({ri},{ci}) bit mismatch"
                                );
                            }
                        }
                        total_ad2_ok += 1;
                        total_ad2_fast_bits += 1;
                    } else if let Ok(actual) = &ad2 {
                        total_ad2_ok += 1;
                        if let Some(Ok(ref_ad)) = ref_e2_ad {
                            let sin_all = rg.sin_ed_r.min(rg.sin_du);
                            if sin_all > 1.0e-11 {
                                total_ad2_fallback_checked += 1;
                                // ScaledFieldVec3 vs power-of-two rescaled FieldVec3 rounding
                                // normalized by the natural O(c / min(|r|, |s - r|)) scale and
                                // amplified by 1/sin (measured max 4.06 eps across 505 cases;
                                // bound <= 7.11e-4 << 1).
                                let p_tol = 32.0 * f64::EPSILON / sin_all;
                                let c2_per = e2
                                    .parameters()
                                    .iter()
                                    .map(|x| x.value.abs())
                                    .fold(0.0_f64, f64::max)
                                    * 4.0;
                                let p_scale = pos_partial_natural_scale(r, s, ratio, c2_per);
                                let p_max = mat3_max_abs(ref_ad.wrt_position);
                                if p_max >= f64::MIN_POSITIVE {
                                    let rel = mat3_scaled_err(
                                        actual.wrt_position,
                                        ref_ad.wrt_position,
                                        p_scale,
                                    )
                                    .unwrap();
                                    max_ad2_p_eps = max_ad2_p_eps.max(rel * sin_all / f64::EPSILON);
                                    assert!(
                                        rel <= p_tol,
                                        "ECOM2 cfg={cfg_idx} case={case} mode={mode}: fallback wrt_position rel={rel:e} > {p_tol:e} (sin={sin_all:e})"
                                    );
                                } else if p_max > 0.0 {
                                    let diff =
                                        mat3_max_abs(actual.wrt_position - ref_ad.wrt_position);
                                    let p_s_f = p_scale.to_f64();
                                    let scale_f = if p_s_f.is_finite() {
                                        p_max.max(p_s_f)
                                    } else {
                                        p_max
                                    };
                                    let bound = p_tol * scale_f + 4.0 * f64::from_bits(1);
                                    assert!(
                                        diff <= bound,
                                        "ECOM2 cfg={cfg_idx} case={case} mode={mode}: subnormal fallback wrt_position diff={diff:e} > {bound:e}"
                                    );
                                } else {
                                    let diff = mat3_max_abs(actual.wrt_position);
                                    assert!(
                                        diff <= 4.0 * f64::from_bits(1),
                                        "ECOM2 cfg={cfg_idx} case={case} mode={mode}: zero-ref fallback wrt_position diff={diff:e}"
                                    );
                                }
                                // ScaledFieldVec3 dv - u(u.dv) vs Hipparchus s.reciprocal()
                                // gradient rounding normalized by the natural O(c/|v_perp|)
                                // scale and amplified by 1/sin (measured max 0.62 eps across
                                // 505 cases; bound <= 7.11e-4 << 1).
                                let v_tol = 32.0 * f64::EPSILON / sin_all;
                                let v_scale = vel_partial_natural_scale(r, v, ratio, c2_per);
                                let v_max = mat3_max_abs(ref_ad.wrt_velocity);
                                if v_max >= f64::MIN_POSITIVE {
                                    let rel = mat3_scaled_err(
                                        actual.wrt_velocity,
                                        ref_ad.wrt_velocity,
                                        v_scale,
                                    )
                                    .unwrap();
                                    max_ad2_v_eps = max_ad2_v_eps.max(rel * sin_all / f64::EPSILON);
                                    assert!(
                                        rel <= v_tol,
                                        "ECOM2 cfg={cfg_idx} case={case} mode={mode}: fallback wrt_velocity rel={rel:e} > {v_tol:e} (sin={sin_all:e})"
                                    );
                                } else if v_max > 0.0 {
                                    let diff =
                                        mat3_max_abs(actual.wrt_velocity - ref_ad.wrt_velocity);
                                    let v_s_f = v_scale.to_f64();
                                    let scale_f = if v_s_f.is_finite() {
                                        v_max.max(v_s_f)
                                    } else {
                                        v_max
                                    };
                                    let bound = v_tol * scale_f + 4.0 * f64::from_bits(1);
                                    assert!(
                                        diff <= bound,
                                        "ECOM2 cfg={cfg_idx} case={case} mode={mode}: subnormal fallback wrt_velocity diff={diff:e} > {bound:e}"
                                    );
                                } else {
                                    let diff = mat3_max_abs(actual.wrt_velocity);
                                    let v_s_f = v_scale.to_f64();
                                    let bound = if v_s_f.is_finite() {
                                        v_tol * v_s_f + 4.0 * f64::from_bits(1)
                                    } else {
                                        4.0 * f64::from_bits(1)
                                    };
                                    assert!(
                                        diff <= bound,
                                        "ECOM2 cfg={cfg_idx} case={case} mode={mode}: zero-ref fallback wrt_velocity diff={diff:e} > {bound:e}"
                                    );
                                }
                            }
                        } else if is_sunlit_field(ratio3)
                            && rg.sin_ed_r > 1.0e-6
                            && rg.sin_du > 1.0e-6
                        {
                            // Multi-scale fallback case: verify against central differences.
                            let mut fd_pos = Matrix3::zeros();
                            let mut fd_vel = Matrix3::zeros();
                            let mut fd_pos_ok = true;
                            let mut fd_vel_ok = true;
                            let sin_scale = rg.sin_ed_r.min(rg.sin_du).min(1.0);
                            for axis in 0..3 {
                                let hr =
                                    r.x.abs().max(r.y.abs()).max(r.z.abs()) * 1.0e-6 * sin_scale;
                                let mut rp = r;
                                let mut rm = r;
                                rp[axis] += hr;
                                rm[axis] -= hr;
                                if rp[axis] == rm[axis]
                                    || !rp[axis].is_finite()
                                    || !rm[axis].is_finite()
                                    || lighting_ratio_double(
                                        rp,
                                        s,
                                        moon_opt,
                                        e2.shadow_function(),
                                        earth_model,
                                        include_moon,
                                    )
                                    .ok()
                                        != Some(1.0)
                                    || lighting_ratio_double(
                                        rm,
                                        s,
                                        moon_opt,
                                        e2.shadow_function(),
                                        earth_model,
                                        include_moon,
                                    )
                                    .ok()
                                        != Some(1.0)
                                {
                                    fd_pos_ok = false;
                                } else if let (Ok(ap), Ok(am)) = (
                                    e2.acceleration_with_sun_moon(rp, v, s, m),
                                    e2.acceleration_with_sun_moon(rm, v, s, m),
                                ) {
                                    let diff = ap - am;
                                    let a_scale = ap
                                        .iter()
                                        .chain(am.iter())
                                        .map(|x| x.abs())
                                        .fold(0.0_f64, f64::max);
                                    let d_max =
                                        diff.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                                    if a_scale > 0.0 && d_max < 1.0e-9 * a_scale {
                                        fd_pos_ok = false;
                                    }
                                    let col = diff / (2.0 * hr);
                                    fd_pos.set_column(axis, &col);
                                } else {
                                    fd_pos_ok = false;
                                }
                                let hv =
                                    v.x.abs().max(v.y.abs()).max(v.z.abs()) * 1.0e-6 * sin_scale;
                                let mut vp = v;
                                let mut vm = v;
                                vp[axis] += hv;
                                vm[axis] -= hv;
                                if vp[axis] == vm[axis]
                                    || !vp[axis].is_finite()
                                    || !vm[axis].is_finite()
                                {
                                    fd_vel_ok = false;
                                } else if let (Ok(ap), Ok(am)) = (
                                    e2.acceleration_with_sun_moon(r, vp, s, m),
                                    e2.acceleration_with_sun_moon(r, vm, s, m),
                                ) {
                                    let diff = ap - am;
                                    let a_scale = ap
                                        .iter()
                                        .chain(am.iter())
                                        .map(|x| x.abs())
                                        .fold(0.0_f64, f64::max);
                                    let d_max =
                                        diff.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                                    if a_scale > 0.0 && d_max < 1.0e-9 * a_scale {
                                        fd_vel_ok = false;
                                    }
                                    let col = diff / (2.0 * hv);
                                    fd_vel.set_column(axis, &col);
                                } else {
                                    fd_vel_ok = false;
                                }
                            }
                            if fd_pos_ok && fd_pos.iter().all(|x| x.is_finite()) {
                                total_ad2_fd_checked += 1;
                                if let Some(rel) = mat3_rel_err(actual.wrt_position, fd_pos) {
                                    assert!(
                                        rel <= 1.0e-4,
                                        "ECOM2 cfg={cfg_idx} case={case}: multi-scale FD wrt_position rel={rel:e}"
                                    );
                                }
                            }
                            if fd_vel_ok && fd_vel.iter().all(|x| x.is_finite()) {
                                if let Some(rel) = mat3_rel_err(actual.wrt_velocity, fd_vel) {
                                    assert!(
                                        rel <= 1.0e-4,
                                        "ECOM2 cfg={cfg_idx} case={case}: multi-scale FD wrt_velocity rel={rel:e}"
                                    );
                                }
                            }
                        }
                    } else {
                        // acceleration_partials must never refuse when the reference Jacobian
                        // is defined and finite in binary64.
                        assert!(
                            !matches!(ref_e2_ad, Some(Ok(_))),
                            "ECOM2 cfg={cfg_idx} case={case} mode={mode}: refused defined finite partials: err={:?}, sin_du={:e}",
                            ad2.as_ref().unwrap_err(),
                            rg.sin_du
                        );
                    }
                }
                if let Ok(acc1) = &a1 {
                    // Assert ECOM1 double acceleration equals the N=6 state-partials field value
                    // bit for bit whenever both succeed with the same lighting ratio value.
                    if ratio3.value.to_bits() == ratio.to_bits() {
                        let (ratio_s, sunlit_s) = if is_conical {
                            let (r_km, is_sunlit_km) =
                                lighting_ratio_field_conical_scaled_km(r, s, earth_model)
                                    .expect("conical scaled lighting");
                            (
                                ScaledGrad {
                                    val: r_km.val,
                                    grad: [
                                        r_km.grad[0],
                                        r_km.grad[1],
                                        r_km.grad[2],
                                        ScaledF64::ZERO,
                                        ScaledF64::ZERO,
                                        ScaledF64::ZERO,
                                    ],
                                },
                                is_sunlit_km,
                            )
                        } else {
                            let m_per_km = ScaledF64::from_f64(M_PER_KM);
                            (
                                ScaledGrad {
                                    val: ScaledF64::from_f64(ratio3.value),
                                    grad: [
                                        ScaledF64::from_f64(ratio3.grad[0]).mul(m_per_km),
                                        ScaledF64::from_f64(ratio3.grad[1]).mul(m_per_km),
                                        ScaledF64::from_f64(ratio3.grad[2]).mul(m_per_km),
                                        ScaledF64::ZERO,
                                        ScaledF64::ZERO,
                                        ScaledF64::ZERO,
                                    ],
                                },
                                is_sunlit_field(ratio3),
                            )
                        };
                        let c1_arr = e1.coefficients().to_array();
                        let c_s = c1_arr.map(|val| ScaledGrad {
                            val: ScaledF64::from_f64(val),
                            grad: [ScaledF64::ZERO; 6],
                        });
                        if let Ok(accel_s6) = ecom1_acceleration_field(
                            scaled_field_vec3_variable::<6>(r, 0),
                            scaled_field_vec3_variable::<6>(v, 3),
                            scaled_field_vec3_constant::<6>(s),
                            ratio_s,
                            sunlit_s,
                            c_s,
                            c1_arr,
                            false,
                        ) {
                            let val6 = accel_s6
                                .val_vec()
                                .to_vec3()
                                .expect("ECOM1 field6 val finite when double succeeded");
                            assert_eq!(
                                acc1.x.to_bits(),
                                val6.x.to_bits(),
                                "ECOM1 double vs field6 x bit mismatch cfg={cfg_idx} case={case} mode={mode}"
                            );
                            assert_eq!(
                                acc1.y.to_bits(),
                                val6.y.to_bits(),
                                "ECOM1 double vs field6 y bit mismatch cfg={cfg_idx} case={case} mode={mode}"
                            );
                            assert_eq!(
                                acc1.z.to_bits(),
                                val6.z.to_bits(),
                                "ECOM1 double vs field6 z bit mismatch cfg={cfg_idx} case={case} mode={mode}"
                            );
                            total_ecom1_val_bit_exact += 1;
                        }
                    }
                    if let Ok(actual) = &ad1 {
                        total_ad1_ok += 1;
                        let mut checked_rescaled = false;
                        if let Some(Ok(ref_ad)) = ref_e1_ad {
                            let sin_all = rg.sin_ey_ecom1.min(rg.sin_u);
                            if sin_all > 1.0e-11 {
                                checked_rescaled = true;
                                total_ad1_checked += 1;
                                // ScaledFieldVec3 vs independent power-of-two rescaled FieldVec3
                                // rounding amplified by 1/sin (measured max 4.76 eps across 715
                                // cases; bound <= 7.11e-4 << 1).
                                let p_tol = 32.0 * f64::EPSILON / sin_all;
                                let p_max = mat3_max_abs(ref_ad.wrt_position);
                                if p_max >= f64::MIN_POSITIVE {
                                    let rel =
                                        mat3_rel_err(actual.wrt_position, ref_ad.wrt_position)
                                            .unwrap();
                                    max_ad1_p_eps = max_ad1_p_eps.max(rel * sin_all / f64::EPSILON);
                                    assert!(
                                        rel <= p_tol,
                                        "ECOM1 cfg={cfg_idx} case={case} mode={mode}: wrt_position rel={rel:e} > {p_tol:e} (sin={sin_all:e})"
                                    );
                                } else if p_max > 0.0 {
                                    let diff =
                                        mat3_max_abs(actual.wrt_position - ref_ad.wrt_position);
                                    let bound = p_tol * p_max + 4.0 * f64::from_bits(1);
                                    assert!(
                                        diff <= bound,
                                        "ECOM1 cfg={cfg_idx} case={case} mode={mode}: subnormal wrt_position diff={diff:e} > {bound:e}"
                                    );
                                } else {
                                    let diff = mat3_max_abs(actual.wrt_position);
                                    assert!(
                                        diff <= 4.0 * f64::from_bits(1),
                                        "ECOM1 cfg={cfg_idx} case={case} mode={mode}: zero-ref wrt_position diff={diff:e}"
                                    );
                                }
                                // ScaledFieldVec3 dv - u(u.dv) vs rescaled FieldVec3 gradient
                                // rounding normalized by the natural O(c/|v_perp|) scale and
                                // amplified by 1/sin (measured max 1.37 eps across 715 cases;
                                // bound <= 7.11e-4 << 1).
                                let v_tol = 32.0 * f64::EPSILON / sin_all;
                                let c1_per = e1
                                    .coefficients()
                                    .to_array()
                                    .iter()
                                    .map(|x| x.abs())
                                    .fold(0.0_f64, f64::max);
                                let v_scale = vel_partial_natural_scale(r, v, ratio, c1_per);
                                let v_max = mat3_max_abs(ref_ad.wrt_velocity);
                                if v_max >= f64::MIN_POSITIVE {
                                    let rel = mat3_scaled_err(
                                        actual.wrt_velocity,
                                        ref_ad.wrt_velocity,
                                        v_scale,
                                    )
                                    .unwrap();
                                    max_ad1_v_eps = max_ad1_v_eps.max(rel * sin_all / f64::EPSILON);
                                    assert!(
                                        rel <= v_tol,
                                        "ECOM1 cfg={cfg_idx} case={case} mode={mode}: wrt_velocity rel={rel:e} > {v_tol:e} (sin={sin_all:e})"
                                    );
                                } else if v_max > 0.0 {
                                    let diff =
                                        mat3_max_abs(actual.wrt_velocity - ref_ad.wrt_velocity);
                                    let v_s_f = v_scale.to_f64();
                                    let scale_f = if v_s_f.is_finite() {
                                        v_max.max(v_s_f)
                                    } else {
                                        v_max
                                    };
                                    let bound = v_tol * scale_f + 4.0 * f64::from_bits(1);
                                    assert!(
                                        diff <= bound,
                                        "ECOM1 cfg={cfg_idx} case={case} mode={mode}: subnormal wrt_velocity diff={diff:e} > {bound:e}"
                                    );
                                } else {
                                    let diff = mat3_max_abs(actual.wrt_velocity);
                                    let v_s_f = v_scale.to_f64();
                                    let bound = if v_s_f.is_finite() {
                                        v_tol * v_s_f + 4.0 * f64::from_bits(1)
                                    } else {
                                        4.0 * f64::from_bits(1)
                                    };
                                    assert!(
                                        diff <= bound,
                                        "ECOM1 cfg={cfg_idx} case={case} mode={mode}: zero-ref wrt_velocity diff={diff:e} > {bound:e}"
                                    );
                                }
                            }
                        }
                        if (!checked_rescaled || case % 4 == 0)
                            && is_sunlit_field(ratio3)
                            && rg.sin_ey_ecom1 > 1.0e-6
                            && rg.sin_u > 1.0e-6
                        {
                            let mut fd_pos = Matrix3::zeros();
                            let mut fd_vel = Matrix3::zeros();
                            let mut fd_pos_ok = true;
                            let mut fd_vel_ok = true;
                            let sin_scale = rg.sin_ey_ecom1.min(rg.sin_u).min(1.0);
                            for axis in 0..3 {
                                let hr =
                                    r.x.abs().max(r.y.abs()).max(r.z.abs()) * 1.0e-6 * sin_scale;
                                let mut rp = r;
                                let mut rm = r;
                                rp[axis] += hr;
                                rm[axis] -= hr;
                                if rp[axis] == rm[axis]
                                    || !rp[axis].is_finite()
                                    || !rm[axis].is_finite()
                                    || lighting_ratio_double(
                                        rp,
                                        s,
                                        moon_opt,
                                        e1.shadow_function(),
                                        earth_model,
                                        include_moon,
                                    )
                                    .ok()
                                        != Some(1.0)
                                    || lighting_ratio_double(
                                        rm,
                                        s,
                                        moon_opt,
                                        e1.shadow_function(),
                                        earth_model,
                                        include_moon,
                                    )
                                    .ok()
                                        != Some(1.0)
                                {
                                    fd_pos_ok = false;
                                } else if let (Ok(ap), Ok(am)) = (
                                    e1.acceleration_with_sun_moon(rp, v, s, m),
                                    e1.acceleration_with_sun_moon(rm, v, s, m),
                                ) {
                                    let diff = ap - am;
                                    let a_scale = ap
                                        .iter()
                                        .chain(am.iter())
                                        .map(|x| x.abs())
                                        .fold(0.0_f64, f64::max);
                                    let d_max =
                                        diff.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                                    if a_scale > 0.0 && d_max < 1.0e-9 * a_scale {
                                        fd_pos_ok = false;
                                    }
                                    let col = diff / (2.0 * hr);
                                    fd_pos.set_column(axis, &col);
                                } else {
                                    fd_pos_ok = false;
                                }
                                let hv =
                                    v.x.abs().max(v.y.abs()).max(v.z.abs()) * 1.0e-6 * sin_scale;
                                let mut vp = v;
                                let mut vm = v;
                                vp[axis] += hv;
                                vm[axis] -= hv;
                                if vp[axis] == vm[axis]
                                    || !vp[axis].is_finite()
                                    || !vm[axis].is_finite()
                                {
                                    fd_vel_ok = false;
                                } else if let (Ok(ap), Ok(am)) = (
                                    e1.acceleration_with_sun_moon(r, vp, s, m),
                                    e1.acceleration_with_sun_moon(r, vm, s, m),
                                ) {
                                    let diff = ap - am;
                                    let a_scale = ap
                                        .iter()
                                        .chain(am.iter())
                                        .map(|x| x.abs())
                                        .fold(0.0_f64, f64::max);
                                    let d_max =
                                        diff.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                                    if a_scale > 0.0 && d_max < 1.0e-9 * a_scale {
                                        fd_vel_ok = false;
                                    }
                                    let col = diff / (2.0 * hv);
                                    fd_vel.set_column(axis, &col);
                                } else {
                                    fd_vel_ok = false;
                                }
                            }
                            if fd_pos_ok && fd_pos.iter().all(|x| x.is_finite()) {
                                total_ad1_fd_checked += 1;
                                if let Some(rel) = mat3_rel_err(actual.wrt_position, fd_pos) {
                                    max_ad1_fd_p_rel = max_ad1_fd_p_rel.max(rel);
                                    // 2nd-order central difference with step 1e-6 * |r| amplified
                                    // by 1/sin (measured max 9.70e-6 across 392 cases).
                                    assert!(
                                        rel <= 1.0e-4,
                                        "ECOM1 cfg={cfg_idx} case={case}: FD wrt_position rel={rel:e}"
                                    );
                                }
                            }
                            if fd_vel_ok && fd_vel.iter().all(|x| x.is_finite()) {
                                if let Some(rel) = mat3_rel_err(actual.wrt_velocity, fd_vel) {
                                    max_ad1_fd_v_rel = max_ad1_fd_v_rel.max(rel);
                                    // 2nd-order central difference with step 1e-6 * |v| amplified
                                    // by 1/sin (measured max 9.53e-8 across 392 cases).
                                    assert!(
                                        rel <= 1.0e-4,
                                        "ECOM1 cfg={cfg_idx} case={case}: FD wrt_velocity rel={rel:e}"
                                    );
                                }
                            }
                        }
                    } else {
                        // acceleration_partials must never refuse when the reference Jacobian
                        // is defined and finite in binary64.
                        assert!(
                            !matches!(ref_e1_ad, Some(Ok(_))),
                            "ECOM1 cfg={cfg_idx} case={case} mode={mode}: refused defined finite partials: err={:?}",
                            ad1.as_ref().unwrap_err()
                        );
                    }
                }
            }
        }
    }

    assert!(
        total_ok_ecom2 > 1200,
        "too few ECOM2 ok cases: {total_ok_ecom2}"
    );
    assert!(
        total_ok_ecom1 > 1200,
        "too few ECOM1 ok cases: {total_ok_ecom1}"
    );
    assert!(
        total_ad2_ok > 1300,
        "too few ECOM2 state partials ok: {total_ad2_ok}"
    );
    assert!(
        total_ad1_ok > 1300,
        "too few ECOM1 state partials ok: {total_ad1_ok}"
    );
    assert!(
        total_ad2_fast_bits > 150 && total_ad2_fallback_checked > 300 && total_ad2_fd_checked > 40,
        "insufficient ECOM2 partials coverage: fast={total_ad2_fast_bits}, fb={total_ad2_fallback_checked}, fd={total_ad2_fd_checked}"
    );
    assert!(
        total_ad1_checked > 500
            && total_pp1_ref_checked > 800
            && total_ecom1_val_bit_exact > 2400
            && total_ad1_fd_checked > 150,
        "insufficient ECOM1 partials coverage: checked={total_ad1_checked}, pp_ref={total_pp1_ref_checked}, val_bits={total_ecom1_val_bit_exact}, fd={total_ad1_fd_checked}"
    );
    assert!(
        total_u_bit_exact > 2000,
        "too few ECOM1 u bit-exact checks: {total_u_bit_exact}"
    );
    assert!(
        total_flushed_frame_ok > 200,
        "too few flushed-component frame ok cases: {total_flushed_frame_ok}"
    );
    assert!(
        total_overflow_coeff_ok > 300,
        "too few near-f64::MAX coefficient ok cases: {total_overflow_coeff_ok}"
    );
    assert!(
        total_near_node_checked >= 150,
        "too few near-node cases: {total_near_node_checked}"
    );
    assert!(
        total_collinear_checked >= 150,
        "too few exact-collinear cases: {total_collinear_checked}"
    );
    assert!(
        total_tiny_conical_checked >= 50,
        "too few tiny-conical-gradient cases: {total_tiny_conical_checked}"
    );
    assert!(
        total_subnormal_fast_bits > 10,
        "expected subnormal fast-path bit-exact coverage, got {total_subnormal_fast_bits}"
    );
    assert!(total_umbra > 0, "expected umbra coverage");
    assert!(total_refused > 0, "expected refused coverage");
    // Measured maxima across 3,072 cases (6 shadow configs x 512 cases, strict
    // refusal assertion):
    // - lin2 = 11.95 eps, lin1 = 15.75 eps (coefficient linearity vs sum of basis partials)
    // - ad2_p = 4.06 eps, ad2_v = 0.62 eps across 505 ECOM2 km-fallback checks
    //   (ScaledFieldVec3 vs power-of-two rescaled FieldVec3 rounding normalized by
    //   natural scale and amplified by 1/sin)
    // - ad1_p = 2.39 eps, ad1_v = 1.14 eps (ECOM1 ScaledFieldVec3 vs rescaled ref)
    // - pp1_ref = 15.87 eps (ECOM1 parameter partials vs rescaled ref)
    // - fd1_p = 5.81e-8, fd1_v = 9.01e-8 (ECOM1 4th-order central differences)
    // - nn_a1 = 2.00 eps, nn_ad1 = 2.46 eps (ECOM1 near-node u -> 0 / pi)
    // - tc_e1 = 1.98 eps, tc_e2 = 1.98 eps (tiny conical gradients in 1/km)
    assert!(
        max_lin2_eps <= 64.0
            && max_lin1_eps <= 64.0
            && max_ad2_p_eps <= 32.0
            && max_ad2_v_eps <= 32.0
            && max_ad1_p_eps <= 32.0
            && max_ad1_v_eps <= 32.0
            && max_pp1_ref_eps <= 32.0
            && max_ad1_fd_p_rel <= 1.0e-4
            && max_ad1_fd_v_rel <= 1.0e-4
            && max_near_node_a1_eps <= 8.0
            && max_near_node_ad1_eps <= 8.0
            && max_tiny_conical_e1_eps <= 8.0
            && max_tiny_conical_e2_eps <= 8.0,
        "sweep maxima: lin2={max_lin2_eps:.2}, lin1={max_lin1_eps:.2}, ad2_p={max_ad2_p_eps:.2}, ad2_v={max_ad2_v_eps:.2}, ad1_p={max_ad1_p_eps:.2}, ad1_v={max_ad1_v_eps:.2}, pp1_ref={max_pp1_ref_eps:.2}, fd1_p={max_ad1_fd_p_rel:e}, fd1_v={max_ad1_fd_v_rel:e}, nn_a1={max_near_node_a1_eps:.2}, nn_ad1={max_near_node_ad1_eps:.2}, tc_e1={max_tiny_conical_e1_eps:.2}, tc_e2={max_tiny_conical_e2_eps:.2}"
    );
}

#[test]
fn spherical_penumbra_plane_section_nonfinite_tan_theta_fallback_ecom_partials() {
    use crate::astro::events::eclipse::{
        shadow_fraction_with_model_and_moon, shadow_fraction_with_model_moon_and_gradient,
    };

    // Two spherical-Earth penumbra inputs (lighting ~0.480) where Orekit's
    // Ellipsoid.getPlaneSection produces non-finite tanTheta:
    // - y = 1e-40: |gamma| ~ 1e-167, raw.value = +-Inf (underflow)
    // - y = 1e-33: |gamma| ~ 2.5e-160, raw.value = +-1.0, Grad::div inv2 = +Inf
    let sun = Vector3::new(149_597_870.7, 1.0e-85, 0.0);
    let moon = Vector3::new(0.0, 384_400.0, 0.0);
    let vel = Vector3::new(0.0, 7.5, 0.0);

    let e2 = Ecom2SolarRadiationPressure::d4b1(mixed_d4b1())
        .unwrap()
        .with_shadow_function(ShadowFunction::ApparentDisc)
        .with_earth_shadow_model(EarthShadowModel::Spherical)
        .with_moon_eclipse(false);
    let e1 = Ecom1SolarRadiationPressure::new(mixed1())
        .unwrap()
        .with_shadow_function(ShadowFunction::ApparentDisc)
        .with_earth_shadow_model(EarthShadowModel::Spherical)
        .with_moon_eclipse(false);

    let cases = [
        ("underflow_val", Vector3::new(-7_000.0, 1.0e-40, 6_370.0)),
        ("nonfinite_grad", Vector3::new(-7_000.0, 1.0e-33, 6_370.0)),
    ];
    let sat0 = Vector3::new(-7_000.0, 0.0, 6_370.0);
    let sun0 = Vector3::new(149_597_870.7, 0.0, 0.0);
    let (sh0, sh0_grad) = shadow_fraction_with_model_moon_and_gradient(
        as_array(sat0),
        as_array(sun0),
        as_array(moon),
        EarthShadowModel::Spherical,
    )
    .expect("y=0 shadow gradient");
    let ad2_0 = e2
        .acceleration_partials_with_sun(sat0, vel, sun0)
        .expect("y=0 ECOM2 partials");
    let ad1_0 = e1
        .acceleration_partials_with_sun(sat0, vel, sun0)
        .expect("y=0 ECOM1 partials");

    let h_r = 0.05;
    let h_v = 1.0e-4;

    for (label, sat) in cases {
        let (sh, sh_grad) = shadow_fraction_with_model_moon_and_gradient(
            as_array(sat),
            as_array(sun),
            as_array(moon),
            EarthShadowModel::Spherical,
        )
        .unwrap_or_else(|e| panic!("{label}: shadow gradient failed: {e:?}"));
        let lighting = 1.0 - sh;
        assert!(lighting > 0.45 && lighting < 0.51, "{label}: {lighting}");
        assert!(sh_grad.iter().all(|x| x.is_finite()));
        assert!((sh - sh0).abs() <= 4.0 * f64::EPSILON);
        let sh0_norm =
            (sh0_grad[0] * sh0_grad[0] + sh0_grad[1] * sh0_grad[1] + sh0_grad[2] * sh0_grad[2])
                .sqrt();
        let sh0_diff = ((sh_grad[0] - sh0_grad[0]).powi(2)
            + (sh_grad[1] - sh0_grad[1]).powi(2)
            + (sh_grad[2] - sh0_grad[2]).powi(2))
        .sqrt();
        assert!(
            sh0_diff <= 16.0 * f64::EPSILON * sh0_norm,
            "{label}: shadow grad vs y=0 ref rel err {:e}",
            sh0_diff / sh0_norm
        );

        // Verify shadow gradient against 4th-order central differences.
        let mut fd_sh = [0.0; 3];
        for axis in 0..3 {
            let eval = |delta: f64| {
                let mut p = as_array(sat);
                p[axis] += delta;
                shadow_fraction_with_model_and_moon(
                    p,
                    as_array(sun),
                    as_array(moon),
                    EarthShadowModel::Spherical,
                )
                .expect("fd shadow")
            };
            fd_sh[axis] = (eval(-2.0 * h_r) - 8.0 * eval(-h_r) + 8.0 * eval(h_r) - eval(2.0 * h_r))
                / (12.0 * h_r);
        }
        let sh_norm = (fd_sh[0] * fd_sh[0] + fd_sh[1] * fd_sh[1] + fd_sh[2] * fd_sh[2]).sqrt();
        let sh_diff = ((sh_grad[0] - fd_sh[0]).powi(2)
            + (sh_grad[1] - fd_sh[1]).powi(2)
            + (sh_grad[2] - fd_sh[2]).powi(2))
        .sqrt();
        assert!(
            sh_diff <= 1.0e-7 * sh_norm,
            "{label}: shadow grad rel err {:e}",
            sh_diff / sh_norm
        );

        // ECOM2 and ECOM1 acceleration_partials must succeed and match both the y=0
        // exact-Orekit reference and 4th-order central differences.
        let ad2 = e2
            .acceleration_partials_with_sun(sat, vel, sun)
            .unwrap_or_else(|e| panic!("{label}: ECOM2 partials failed: {e:?}"));
        let ad1 = e1
            .acceleration_partials_with_sun(sat, vel, sun)
            .unwrap_or_else(|e| panic!("{label}: ECOM1 partials failed: {e:?}"));
        assert!(mat3_rel_err(ad2.wrt_position, ad2_0.wrt_position).unwrap() <= 16.0 * f64::EPSILON);
        assert!(mat3_rel_err(ad2.wrt_velocity, ad2_0.wrt_velocity).unwrap() <= 16.0 * f64::EPSILON);
        assert!(mat3_rel_err(ad1.wrt_position, ad1_0.wrt_position).unwrap() <= 16.0 * f64::EPSILON);
        assert!(mat3_rel_err(ad1.wrt_velocity, ad1_0.wrt_velocity).unwrap() <= 16.0 * f64::EPSILON);

        let mut fd2_p = Matrix3::zeros();
        let mut fd2_v = Matrix3::zeros();
        let mut fd1_p = Matrix3::zeros();
        let mut fd1_v = Matrix3::zeros();
        for axis in 0..3 {
            let eval2_r = |delta: f64| {
                let mut p = sat;
                p[axis] += delta;
                e2.acceleration_with_sun(p, vel, sun).unwrap()
            };
            let eval1_r = |delta: f64| {
                let mut p = sat;
                p[axis] += delta;
                e1.acceleration_with_sun(p, vel, sun).unwrap()
            };
            fd2_p.set_column(
                axis,
                &((eval2_r(-2.0 * h_r) - 8.0 * eval2_r(-h_r) + 8.0 * eval2_r(h_r)
                    - eval2_r(2.0 * h_r))
                    / (12.0 * h_r)),
            );
            fd1_p.set_column(
                axis,
                &((eval1_r(-2.0 * h_r) - 8.0 * eval1_r(-h_r) + 8.0 * eval1_r(h_r)
                    - eval1_r(2.0 * h_r))
                    / (12.0 * h_r)),
            );

            let eval2_v = |delta: f64| {
                let mut pv = vel;
                pv[axis] += delta;
                e2.acceleration_with_sun(sat, pv, sun).unwrap()
            };
            let eval1_v = |delta: f64| {
                let mut pv = vel;
                pv[axis] += delta;
                e1.acceleration_with_sun(sat, pv, sun).unwrap()
            };
            fd2_v.set_column(
                axis,
                &((eval2_v(-2.0 * h_v) - 8.0 * eval2_v(-h_v) + 8.0 * eval2_v(h_v)
                    - eval2_v(2.0 * h_v))
                    / (12.0 * h_v)),
            );
            fd1_v.set_column(
                axis,
                &((eval1_v(-2.0 * h_v) - 8.0 * eval1_v(-h_v) + 8.0 * eval1_v(h_v)
                    - eval1_v(2.0 * h_v))
                    / (12.0 * h_v)),
            );
        }
        assert!(
            mat3_rel_err(ad2.wrt_position, fd2_p).unwrap() <= 1.0e-7,
            "{label}: ECOM2 wrt_position rel={:e}",
            mat3_rel_err(ad2.wrt_position, fd2_p).unwrap()
        );
        assert!(
            mat3_rel_err(ad2.wrt_velocity, fd2_v).unwrap() <= 1.0e-7,
            "{label}: ECOM2 wrt_velocity rel={:e}",
            mat3_rel_err(ad2.wrt_velocity, fd2_v).unwrap()
        );
        assert!(
            mat3_rel_err(ad1.wrt_position, fd1_p).unwrap() <= 1.0e-7,
            "{label}: ECOM1 wrt_position rel={:e}",
            mat3_rel_err(ad1.wrt_position, fd1_p).unwrap()
        );
        assert!(
            mat3_rel_err(ad1.wrt_velocity, fd1_v).unwrap() <= 1.0e-7,
            "{label}: ECOM1 wrt_velocity rel={:e}",
            mat3_rel_err(ad1.wrt_velocity, fd1_v).unwrap()
        );
    }
}

#[test]
fn atan2_overflowing_denom_keeps_exact_hipparchus_bits_for_ecom2_parameter_partials() {
    // Conical + spherical Earth input where every DYB/frame norm_sq is finite
    // and nonzero, while px^2 + py^2 overflows to +Inf in Gradient.atan2.
    // For parameter partials py.grad == px.grad == 0, so Hipparchus's inv = 0
    // yields exact delta_u.grad = 0; ECOM2 must keep Hipparchus's exact bits
    // without escalating to ScaledFieldVec3.
    let model = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        d0: 1.0e-7,
        d2c: 1.0e-9,
        ..Default::default()
    })
    .unwrap()
    .with_shadow_function(ShadowFunction::Conical)
    .with_earth_shadow_model(EarthShadowModel::Spherical);
    let r = Vector3::new(libm::ldexp(1.0, 502), libm::ldexp(1.0, 502), 0.0);
    let v = Vector3::new(0.0, 0.0, libm::ldexp(1.0, -700));
    let s = Vector3::new(
        libm::ldexp(1.0, 502) + libm::ldexp(7.0, 460),
        libm::ldexp(1.0, 502) + libm::ldexp(24.0, 460),
        0.0,
    );

    // Confirm px^2 + py^2 overflows to +Inf on this input.
    let r_m = to_vec3_m(r).unwrap();
    let v_m = to_vec3_m(v).unwrap();
    let s_m = to_vec3_m(s).unwrap();
    let z = r_m.cross(v_m);
    let y = raw_hipparchus_normalize_double(z.cross(s_m)).unwrap();
    let x = raw_hipparchus_normalize_double(y.cross(z)).unwrap();
    let py = r_m.dot(y);
    let px = r_m.dot(x);
    assert!(!(py * py + px * px).is_finite());

    let ratio_d = lighting_ratio_double(
        r,
        s,
        None,
        ShadowFunction::Conical,
        EarthShadowModel::Spherical,
        false,
    )
    .unwrap();
    assert_eq!(ratio_d, 1.0);
    let acc = model
        .acceleration_with_sun(r, v, s)
        .expect("ECOM2 accel on atan2 denom overflow");
    let raw_acc = raw_orekit_ecom2_accel_double(r, v, s, ratio_d, 1, 1, model.coefficients())
        .expect("raw Orekit double accel");
    for axis in 0..3 {
        assert_eq!(
            acc[axis].to_bits(),
            raw_acc[axis].to_bits(),
            "ECOM2 accel axis {axis} bit mismatch"
        );
    }

    let ratio_f = lighting_ratio_field(
        r,
        s,
        None,
        ShadowFunction::Conical,
        EarthShadowModel::Spherical,
        false,
    )
    .unwrap();
    let params_g: Vec<Grad<7>> = model
        .coefficients()
        .iter()
        .enumerate()
        .map(|(k, &c)| Grad::variable(k, c))
        .collect();
    let raw_pp_f = raw_orekit_ecom2_field(
        FieldVec3::<7>::constant(r_m),
        v,
        false,
        FieldVec3::<7>::constant(s_m),
        Grad::<7>::constant(ratio_f.value),
        is_sunlit_field(ratio_f),
        1,
        1,
        &params_g,
    )
    .expect("raw Orekit field parameter partials");
    let pp = model
        .parameter_partials_with_sun(r, v, s)
        .expect("ECOM2 parameter partials on atan2 denom overflow");
    for k in 0..7 {
        assert_eq!(
            pp[k].x.to_bits(),
            (raw_pp_f.x.grad[k] / M_PER_KM).to_bits(),
            "ECOM2 pp[{k}].x bit mismatch"
        );
        assert_eq!(
            pp[k].y.to_bits(),
            (raw_pp_f.y.grad[k] / M_PER_KM).to_bits(),
            "ECOM2 pp[{k}].y bit mismatch"
        );
        assert_eq!(
            pp[k].z.to_bits(),
            (raw_pp_f.z.grad[k] / M_PER_KM).to_bits(),
            "ECOM2 pp[{k}].z bit mismatch"
        );
    }

    // Direct checks on Grad::atan2_unscaled_with_fn when y^2 + x^2 == +Inf:
    // 1. Zero gradients -> exact Hipparchus result kept.
    let y0 = Grad::<2>::constant(1.0e200);
    let x0 = Grad::<2>::constant(1.0e200);
    let g0 = y0
        .atan2_unscaled_with_fn(x0, fast_math::atan2)
        .expect("zero-grad atan2 denom overflow kept");
    assert_eq!(
        g0.value.to_bits(),
        fast_math::atan2(1.0e200, 1.0e200).to_bits()
    );
    assert_eq!(g0.grad, [0.0, 0.0]);
    // 2. Nonzero gradient whose true derivative N / (y^2 + x^2) <= 2^-1075 rounds to 0 -> kept.
    let y_tiny = Grad::<1> {
        value: 1.0e200,
        grad: [1.0e-125],
    };
    let x_tiny = Grad::<1>::constant(1.0e200);
    let g_tiny = y_tiny
        .atan2_unscaled_with_fn(x_tiny, fast_math::atan2)
        .expect("underflowing-to-zero true derivative kept");
    assert_eq!(g_tiny.grad, [0.0]);
    // 3. Nonzero gradient whose true derivative is nonzero in binary64 -> rejected (fallback).
    let y_nz = Grad::<1>::variable(0, 1.0e200);
    let x_nz = Grad::<1>::constant(1.0e200);
    assert!(y_nz
        .atan2_unscaled_with_fn(x_nz, fast_math::atan2)
        .is_none());
}

#[test]
fn ecom1_fourier_terms_preserve_precision_when_cos_u_times_du_dv_is_subnormal() {
    // Near u = pi/2 with |du/dv| >= f64::MIN_POSITIVE and |cos(u)| ~ 1e-12,
    // cos(u) * du/dv would drop into subnormals (~3.5e-320) if rounded to f64
    // before multiplying by ds = 1e308. Evaluating ECOM1 entirely in
    // exponent-separated arithmetic preserves full precision against a
    // power-of-two rescaled reference.
    let s = Vector3::new(1.0e8, 0.0, 0.0);
    let m = Vector3::new(0.0, 0.0, 384_400.0);
    let v = Vector3::new(-2.0e304, 0.0, 0.0);
    let ds = 1.0e308;
    let model = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        ds,
        ..Default::default()
    })
    .unwrap();

    // Scale v and ds down by 2^-1000: u is invariant, ds * du/dv is invariant,
    // and wrt_position scales by 2^-1000.
    let v_ref = v * libm::ldexp(1.0, -1000);
    let model_ref = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        ds: libm::ldexp(ds, -1000),
        ..Default::default()
    })
    .unwrap();

    for rx in [3.756_151_221_662_941e-8, 1.0e-10] {
        let r = Vector3::new(rx, 26_560.0, 26_560.0);
        let ad = model
            .acceleration_partials_with_sun_moon(r, v, s, m)
            .expect("ECOM1 subnormal cos(u)*du/dv partials");
        let ad_ref = model_ref
            .acceleration_partials_with_sun_moon(r, v_ref, s, m)
            .expect("ECOM1 power-of-two rescaled ref");

        let p_rel_ref = mat3_rel_err(
            ad.wrt_position * libm::ldexp(1.0, -1000),
            ad_ref.wrt_position,
        )
        .unwrap();
        let v_rel_ref = mat3_rel_err(ad.wrt_velocity, ad_ref.wrt_velocity).unwrap();
        assert!(
            p_rel_ref <= 16.0 * f64::EPSILON,
            "rx={rx:e}: wrt_position vs rescaled ref rel={p_rel_ref:e}"
        );
        assert!(
            v_rel_ref <= 16.0 * f64::EPSILON,
            "rx={rx:e}: wrt_velocity vs rescaled ref rel={v_rel_ref:e}"
        );
    }
}

#[test]
fn ecom1_sun_cross_sat_underflowing_norm_grad_gives_exact_zero_partials() {
    // 1. All 6 shadow configurations: r = (1e29, 0, 0) km, s = (1e8, 1e-183, 0) km.
    //    Here |r| >> |s| selects e_Y = normalize(sun x sat) = (0, 0, -1), where
    //    sun x sat = (0, 0, -s_y * r_x) = (0, 0, -1e-154 km^2). Its derivative wrt
    //    r_x is parallel to e_Y, so d(e_Y)/d(r_x) = 0 and d(e_B.y)/d(r_x) = 0
    //    identically. Unscaled FieldVec3::normalize flushes norm_sq().grad[0] to 0
    //    and yields spurious nonzero entries (-1e-32 at y0=1 / b0=1, -1 at y0=1e32).
    let r1 = Vector3::new(1.0e29, 0.0, 0.0);
    let v1 = Vector3::new(0.0, 1.0, 0.0);
    let s1 = Vector3::new(1.0e8, 1.0e-183, 0.0);
    let m1 = Vector3::new(0.0, 384_400.0, 0.0);

    // Closed-form Jacobian for y0 * e_Y / 1000 at r = (r_x, 0, 0), s = (s_x, s_y, 0):
    //   sun x sat = (s_y * r_z, -s_x * r_z, s_x * r_y - s_y * r_x)
    //   d(e_Y)/d(r_x) = (0, 0, 0), d(e_Y)/d(r_y) = (0, 0, 0),
    //   d(e_Y)/d(r_z) = (1 / r_x, -s_x / (s_y * r_x), 0)  [1/km].
    for y0 in [1.0, 1.0e32] {
        let base = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
            y0,
            ..Default::default()
        })
        .unwrap();
        let expected_y = Matrix3::new(
            0.0,
            0.0,
            y0 / (M_PER_KM * r1.x),
            0.0,
            0.0,
            -y0 * s1.x / (M_PER_KM * s1.y * r1.x),
            0.0,
            0.0,
            0.0,
        );
        for cfg in all_ecom1_shadow_configs(base) {
            let ad = cfg
                .acceleration_partials_with_sun_moon(r1, v1, s1, m1)
                .expect("y0 partials");
            assert_eq!(ad.wrt_position[(2, 0)], 0.0);
            assert_eq!(ad.wrt_position[(2, 1)], 0.0);
            assert_eq!(ad.wrt_position[(2, 2)], 0.0);
            assert_eq!(ad.wrt_velocity, Matrix3::zeros());
            let rel = mat3_rel_err(ad.wrt_position, expected_y).unwrap();
            assert!(
                rel <= 8.0 * f64::EPSILON,
                "y0={y0:e} {cfg:?}: wrt_position rel={rel:e}"
            );
            // Independent entry-level check on the smaller (0, 2) component.
            let c02_rel =
                (ad.wrt_position[(0, 2)] - expected_y[(0, 2)]).abs() / expected_y[(0, 2)].abs();
            assert!(
                c02_rel <= 8.0 * f64::EPSILON,
                "y0={y0:e} {cfg:?}: wrt_position[(0,2)] rel={c02_rel:e}"
            );
        }
    }

    // Same geometry for b0 = 1.0: e_B = e_D x e_Y = (-e_D.y, e_D.x, 0), so
    // wrt_position[(1, 0)] == 0.0 identically.
    let base_b = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        b0: 1.0,
        ..Default::default()
    })
    .unwrap();
    let dist1 = r1.x - s1.x;
    // Independent closed-form Jacobian for b0 * (e_D x e_Y) / 1000:
    //   e_D = (-1, s_y / dist1, 0) + O((s_y/dist1)^2), e_Y = (0, 0, -1).
    //   d(e_B.x)/dr_y = -d(e_D.y)/dr_y = +1 / dist1,
    //   d(e_B.z)/dr_z = e_D.x * d(e_Y.y)/dr_z - e_D.y * d(e_Y.x)/dr_z = +s_x / (s_y * r_x).
    let expected_b_z_z = s1.x / (M_PER_KM * s1.y * r1.x);
    let expected_b_x_y = 1.0 / (M_PER_KM * dist1);
    for cfg in all_ecom1_shadow_configs(base_b) {
        let ad = cfg
            .acceleration_partials_with_sun_moon(r1, v1, s1, m1)
            .expect("b0 partials");
        assert_eq!(ad.wrt_position[(1, 0)], 0.0);
        assert_eq!(ad.wrt_velocity, Matrix3::zeros());
        assert!(
            (ad.wrt_position[(2, 2)] - expected_b_z_z).abs() <= 8.0 * f64::EPSILON * expected_b_z_z,
            "b0 {cfg:?}: (2,2)={} vs {expected_b_z_z:e}",
            ad.wrt_position[(2, 2)]
        );
        assert!(
            (ad.wrt_position[(0, 1)] - expected_b_x_y).abs() <= 8.0 * f64::EPSILON * expected_b_x_y,
            "b0 {cfg:?}: (0,1)={} vs {expected_b_x_y:e}",
            ad.wrt_position[(0, 1)]
        );
    }

    // 2. Partial-bit-loss subnormal norm_sq().grad case:
    //    r = (1e5, 0, 0) km, s = (0, 2e-165, 0) km, y0 = 1.0 under ApparentDisc
    //    (Earth-only WGS84 and Spherical, and with clear Moon). Exact Jacobian
    //    has only (0, 2) = 1 / (1e3 * 1e5) = 1e-8.
    let r2 = Vector3::new(1.0e5, 0.0, 0.0);
    let v2 = Vector3::new(0.0, 1.0, 0.0);
    let s2 = Vector3::new(0.0, 2.0e-165, 0.0);
    let expected_y2 = Matrix3::new(
        0.0,
        0.0,
        1.0 / (M_PER_KM * r2.x),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    );
    let y0_model = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        y0: 1.0,
        ..Default::default()
    })
    .unwrap();
    for model in [
        y0_model.with_moon_eclipse(false),
        y0_model
            .with_moon_eclipse(false)
            .with_earth_shadow_model(EarthShadowModel::Spherical),
    ] {
        let ad = model
            .acceleration_partials_with_sun(r2, v2, s2)
            .expect("subnormal norm_sq grad partials");
        assert_eq!(ad.wrt_position[(2, 0)], 0.0);
        assert_eq!(ad.wrt_velocity, Matrix3::zeros());
        let rel = mat3_rel_err(ad.wrt_position, expected_y2).unwrap();
        assert!(
            rel <= 4.0 * f64::EPSILON,
            "r2 {model:?}: wrt_position rel={rel:e}"
        );
    }
    for model in [
        y0_model,
        y0_model.with_earth_shadow_model(EarthShadowModel::Spherical),
    ] {
        let ad = model
            .acceleration_partials_with_sun_moon(r2, v2, s2, m1)
            .expect("subnormal norm_sq grad partials with moon");
        assert_eq!(ad.wrt_position[(2, 0)], 0.0);
        assert_eq!(ad.wrt_velocity, Matrix3::zeros());
        let rel = mat3_rel_err(ad.wrt_position, expected_y2).unwrap();
        assert!(
            rel <= 4.0 * f64::EPSILON,
            "r2+moon {model:?}: wrt_position rel={rel:e}"
        );
    }
}

#[test]
fn ecom1_sub_subnormal_fourier_value_times_large_frame_grad() {
    // 1. At exact u = pi/2 (r . node == 0 identically), cos(u) = x / rho == 0.0
    //    exactly, so yc * cos(u) * d(e_Y)/dr vanishes cleanly instead of
    //    amplifying libm::cos(FRAC_PI_2) rounding noise.
    let r_exact = Vector3::new(0.0, 1.0e15, 1.0e15);
    let v_exact = Vector3::new(-1.0e15, 0.0, 0.0);
    let s = Vector3::new(2.5e-157, 2.0e15, 2.0e15);
    let m = Vector3::new(0.0, 0.0, 384_400.0);
    let yc = 3.0e-308;
    let base = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        yc,
        ..Default::default()
    })
    .unwrap();
    for cfg in all_ecom1_shadow_configs(base) {
        let ad_exact = cfg
            .acceleration_partials_with_sun_moon(r_exact, v_exact, s, m)
            .expect("exact u=pi/2 yc partials");
        assert_eq!(ad_exact.wrt_position[(0, 1)], 0.0);
        assert_eq!(ad_exact.wrt_position[(0, 2)], 0.0);
    }

    // 2. Genuinely near u = pi/2: tilting v by 1e-1 km/s gives node = (1, -1e-16, 0),
    //    cos(u) = x / rho = -0.1 / (sqrt(2) * 1e15) = -7.071067811865475e-17, so
    //    y_u = yc * cos(u) = -2.1213203435596426e-324 < 2^-1075 (sub-subnormal)
    //    while d(e_Y.x)/d(r_y, r_z) = -/+ 5.65685424949238e156 1/km, yielding
    //    normal wrt_position[(0, 1)] = +1.2e-170 and wrt_position[(0, 2)] = -1.2e-170
    //    across all 6 shadow configs.
    let r = r_exact;
    let v = Vector3::new(-1.0e15, 1.0e-1, 0.0);

    // Independent closed-form reference:
    //   r x v = (-1e14, -1e30, 1e30), node = (1, -1e-16, 0) + O(1e-32),
    //   cos(u) = (r . node) / |r| = -0.1 / (sqrt(2) * 1e15),
    //   |s x r| = sqrt(2) * s.x * r.y,
    //   d(e_Y.x)/dr_y = -s.z / |s x r|, d(e_Y.x)/dr_z = +s.y / |s x r|,
    //   wrt_position[(0, 1)] = (yc / M_PER_KM) * cos(u) * (-s.z / |s x r|) = +1.2e-170,
    //   wrt_position[(0, 2)] = (yc / M_PER_KM) * cos(u) * (+s.y / |s x r|) = -1.2e-170.
    let expected_01 = 1.2e-170_f64;
    let expected_02 = -1.2e-170_f64;

    // Independent power-of-two rescaled reference: scale yc up by 2^500 so y_u
    // is completely normal, then scale the resulting Jacobian back down by 2^-500.
    let yc_ref = libm::ldexp(yc, 500);
    let base_ref = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        yc: yc_ref,
        ..Default::default()
    })
    .unwrap();

    for (cfg, cfg_ref) in all_ecom1_shadow_configs(base)
        .into_iter()
        .zip(all_ecom1_shadow_configs(base_ref))
    {
        let ad = cfg
            .acceleration_partials_with_sun_moon(r, v, s, m)
            .expect("yc=3e-308 partials");
        let ad_ref = cfg_ref
            .acceleration_partials_with_sun_moon(r, v, s, m)
            .expect("rescaled yc partials");

        let rel_01 = (ad.wrt_position[(0, 1)] - expected_01).abs() / expected_01.abs();
        let rel_02 = (ad.wrt_position[(0, 2)] - expected_02).abs() / expected_02.abs();
        assert!(
            rel_01 <= 8.0 * f64::EPSILON,
            "{cfg:?}: wrt_position[(0,1)] = {:e} vs {expected_01:e} (rel={rel_01:e})",
            ad.wrt_position[(0, 1)]
        );
        assert!(
            rel_02 <= 8.0 * f64::EPSILON,
            "{cfg:?}: wrt_position[(0,2)] = {:e} vs {expected_02:e} (rel={rel_02:e})",
            ad.wrt_position[(0, 2)]
        );

        let scale_back = libm::ldexp(1.0, -500);
        let p_rel = mat3_rel_err(ad.wrt_position, ad_ref.wrt_position * scale_back).unwrap();
        assert!(
            p_rel <= 8.0 * f64::EPSILON,
            "{cfg:?}: wrt_position vs rescaled ref rel={p_rel:e}"
        );
        // Velocity partials are O(yc * |du/dv| / M_PER_KM) ~ 1e-326, which rounds to 0.0
        // in binary64 on both the direct evaluation and the rescaled reference.
        assert_eq!(ad.wrt_velocity, Matrix3::zeros());
        assert_eq!(ad_ref.wrt_velocity * scale_back, Matrix3::zeros());
    }
}

#[test]
fn acceleration_value_overflow_with_finite_partials_returns_err_and_ok_partials() {
    // General ECOM2 truncation (n_d = 2000, n_b = 0) with d0 = d2c = ... = d4000c = 1e308:
    // the summed acceleration in km/s^2 (~2.001e308) overflows binary64 while
    // ecom_geometry, parameter_partials (O(1e-3)), and acceleration_partials
    // (wrt_position ~ 2e300, wrt_velocity = 0) are all finite and defined.
    let n_d = 2000;
    let n_b = 0;
    let mut coeffs = vec![0.0; 2 * (n_d + n_b) + 3];
    // Driver order: B0 (0), D0 (1), D2c..D4000c (2..=2001), D2s..D4000s (2002..=4001), Y0 (4002).
    coeffs[1] = 1.0e308;
    for c in &mut coeffs[2..=(n_d + 1)] {
        *c = 1.0e308;
    }
    let model = Ecom2SolarRadiationPressure::new(n_d, n_b, coeffs).unwrap();
    let r = Vector3::new(26_560.0, 0.0, 0.0);
    let v = Vector3::new(0.0, 3.87, 0.0);
    let s = Vector3::new(26_560.0, 0.0, 1.0e8);
    let m = Vector3::new(0.0, 384_400.0, 0.0);

    assert!(matches!(
        model.acceleration_with_sun_moon(r, v, s, m),
        Err(PropagationError::NumericalFailure(_))
    ));
    let geom = ecom_geometry_with_shadow(r, v, s, Some(m), EarthShadowModel::Wgs84Oblate)
        .expect("geometry is finite");
    assert!(geom.delta_u_rad.is_some());

    let pp = model
        .parameter_partials_with_sun_moon(r, v, s, m)
        .expect("parameter_partials are finite despite accel overflow");
    assert_eq!(pp.len(), 2 * (n_d + n_b) + 3);
    assert!(pp.iter().all(|p| p.iter().all(|x| x.is_finite())));
    // D0 partial is e_D / 1000 = (0, 0, 1e-3).
    assert!((pp[1] - Vector3::new(0.0, 0.0, 1.0e-3)).norm() <= 4.0 * f64::EPSILON * 1.0e-3);

    let ad = model
        .acceleration_partials_with_sun_moon(r, v, s, m)
        .expect("acceleration_partials are finite despite accel overflow");
    assert!(ad.wrt_position.iter().all(|x| x.is_finite()));
    assert!(ad.wrt_velocity.iter().all(|x| x.is_finite()));
    assert!(mat3_max_abs(ad.wrt_position) > 1.0e300);
    assert_eq!(ad.wrt_velocity, Matrix3::zeros());
}

#[test]
fn ecom1_near_node_sin_cos_preserves_subnormal_and_sub_subnormal_fourier_terms() {
    // Near the ascending node (|y| / x < 2^-1022):
    //   r = (1e100, 0, r_z) km, v = (0, 1, 1) km/s, s = (2e100, 1e100, 0) km, m = (0, 384_400, 0) km.
    //   y = sqrt(2) * r_z, x = 1e100, sin(u) = y / rho = sqrt(2) * r_z * 1e-100, cos(u) = 1.
    //   e_D = (1/sqrt(2), 1/sqrt(2), -r_z * 1e-100 / sqrt(2)).
    let v = Vector3::new(0.0, 1.0, 1.0);
    let s = Vector3::new(2.0e100, 1.0e100, 0.0);
    let m = Vector3::new(0.0, 384_400.0, 0.0);

    // 1. r.z = 1e-225 km (y/x ~ sqrt(2)*1e-325 < 2^-1075), ds = 1e308.
    let r_flush = Vector3::new(1.0e100, 0.0, 1.0e-225);
    let ds_model = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        ds: 1.0e308,
        ..Default::default()
    })
    .unwrap();
    let expected_a_flush = Vector3::new(1.0e-20, 1.0e-20, 0.0);
    let expected_p22_flush = -2.0e-120;
    let expected_p02_flush = 1.0e205;
    for cfg in all_ecom1_shadow_configs(ds_model) {
        let a = cfg
            .acceleration_with_sun_moon(r_flush, v, s, m)
            .expect("r.z=1e-225 ds accel");
        assert!(
            (a - expected_a_flush).norm() <= 8.0 * f64::EPSILON * expected_a_flush.norm(),
            "{cfg:?}: a={a:?} vs {expected_a_flush:?}"
        );
        let ad = cfg
            .acceleration_partials_with_sun_moon(r_flush, v, s, m)
            .expect("r.z=1e-225 ds partials");
        assert!(
            (ad.wrt_position[(2, 2)] - expected_p22_flush).abs()
                <= 8.0 * f64::EPSILON * expected_p22_flush.abs(),
            "{cfg:?}: wrt_position[(2,2)]={:e} vs {expected_p22_flush:e}",
            ad.wrt_position[(2, 2)]
        );
        assert!(
            (ad.wrt_position[(0, 2)] - expected_p02_flush).abs()
                <= 8.0 * f64::EPSILON * expected_p02_flush.abs(),
            "{cfg:?}: wrt_position[(0,2)]={:e} vs {expected_p02_flush:e}",
            ad.wrt_position[(0, 2)]
        );
        // Double acceleration equals field value part bit for bit.
        let c_arr = cfg.coefficients().to_array();
        let c_s = c_arr.map(|val| ScaledGrad {
            val: ScaledF64::from_f64(val),
            grad: [ScaledF64::ZERO; 6],
        });
        let accel_s = ecom1_acceleration_field(
            scaled_field_vec3_variable::<6>(r_flush, 0),
            scaled_field_vec3_variable::<6>(v, 3),
            scaled_field_vec3_constant::<6>(s),
            ScaledGrad {
                val: ScaledF64::from_f64(1.0),
                grad: [ScaledF64::ZERO; 6],
            },
            true,
            c_s,
            c_arr,
            false,
        )
        .expect("field6");
        let val6 = accel_s.val_vec().to_vec3().unwrap();
        assert_eq!(a.x.to_bits(), val6.x.to_bits());
        assert_eq!(a.y.to_bits(), val6.y.to_bits());
        assert_eq!(a.z.to_bits(), val6.z.to_bits());
    }

    // 2. r.z = 1e-225 km, dc = 1e308:
    //    wrt_position[(0, 2)] = (dc / 1000) * (-sin(u) * du/dr_z * e_D.x + cos(u) * d(e_D.x)/dr_z)
    //                         = -5 / (2 * sqrt(2)) * 1e-120 = -1.7677669529663689e-120.
    let dc_model = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        dc: 1.0e308,
        ..Default::default()
    })
    .unwrap();
    let expected_dc_p02 = -2.5 / 2.0_f64.sqrt() * 1.0e-120;
    for cfg in all_ecom1_shadow_configs(dc_model) {
        let ad = cfg
            .acceleration_partials_with_sun_moon(r_flush, v, s, m)
            .expect("r.z=1e-225 dc partials");
        assert!(
            (ad.wrt_position[(0, 2)] - expected_dc_p02).abs()
                <= 8.0 * f64::EPSILON * expected_dc_p02.abs(),
            "{cfg:?}: dc wrt_position[(0,2)]={:e} vs {expected_dc_p02:e}",
            ad.wrt_position[(0, 2)]
        );
    }

    // 3. r.z = 1e-220 km (y/x ~ sqrt(2)*1e-320 subnormal in binary64), ds = 1e308:
    //    exact closed form a = (1e-15, 1e-15, 0) km/s^2, and power-of-two rescaled ref
    //    (r.z * 2^500, ds * 2^-500) agrees to <= 8 * EPSILON.
    let r_sub = Vector3::new(1.0e100, 0.0, 1.0e-220);
    let r_sub_ref = Vector3::new(1.0e100, 0.0, libm::ldexp(1.0e-220, 500));
    let ds_ref_model = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        ds: libm::ldexp(1.0e308, -500),
        ..Default::default()
    })
    .unwrap();
    let expected_a_sub = Vector3::new(1.0e-15, 1.0e-15, 0.0);
    for (cfg, cfg_ref) in all_ecom1_shadow_configs(ds_model)
        .into_iter()
        .zip(all_ecom1_shadow_configs(ds_ref_model))
    {
        let a = cfg
            .acceleration_with_sun_moon(r_sub, v, s, m)
            .expect("r.z=1e-220 ds accel");
        let a_ref = cfg_ref
            .acceleration_with_sun_moon(r_sub_ref, v, s, m)
            .expect("rescaled ref accel");
        assert!(
            (a - expected_a_sub).norm() <= 8.0 * f64::EPSILON * expected_a_sub.norm(),
            "{cfg:?}: subnormal a={a:?} vs {expected_a_sub:?}"
        );
        assert!(
            (a - a_ref).norm() <= 8.0 * f64::EPSILON * a_ref.norm(),
            "{cfg:?}: subnormal a={a:?} vs rescaled ref={a_ref:?}"
        );
        let ad = cfg
            .acceleration_partials_with_sun_moon(r_sub, v, s, m)
            .expect("r.z=1e-220 ds partials");
        let expected_p22_sub = -2.0e-115;
        assert!(
            (ad.wrt_position[(2, 2)] - expected_p22_sub).abs()
                <= 8.0 * f64::EPSILON * expected_p22_sub.abs(),
            "{cfg:?}: subnormal wrt_position[(2,2)]={:e} vs {expected_p22_sub:e}",
            ad.wrt_position[(2, 2)]
        );
    }
}

#[test]
fn collinear_sun_and_satellite_refuses_y_b_and_parameter_partials_and_accepts_d_only() {
    let v = Vector3::new(0.0, 1.0, 0.0);
    let m = Vector3::new(0.0, 384_400.0, 0.0);
    let cases = [
        (
            "sun_0.1_sat",
            Vector3::new(3.0e7, 4.0e7, 5.0e7),
            Vector3::new(3.0e6, 4.0e6, 5.0e6),
        ),
        (
            "sun_10000_sat",
            Vector3::new(3.0e4, 4.0e4, 5.0e4),
            Vector3::new(3.0e8, 4.0e8, 5.0e8),
        ),
    ];

    let y0_base = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        y0: 1.0,
        ..Default::default()
    })
    .unwrap();
    let b0_base = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        b0: 1.0,
        ..Default::default()
    })
    .unwrap();
    let d_only_base = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        d0: 1.0,
        dc: 2.0,
        ds: -3.0,
        ..Default::default()
    })
    .unwrap();

    for (label, r, s) in cases {
        for cfg in all_ecom1_shadow_configs(y0_base) {
            assert!(
                matches!(
                    cfg.acceleration_with_sun_moon(r, v, s, m),
                    Err(PropagationError::NumericalFailure(_))
                ),
                "{label} y0 accel {cfg:?}"
            );
            assert!(
                matches!(
                    cfg.acceleration_partials_with_sun_moon(r, v, s, m),
                    Err(PropagationError::NumericalFailure(_))
                ),
                "{label} y0 partials {cfg:?}"
            );
            assert!(
                matches!(
                    cfg.parameter_partials_with_sun_moon(r, v, s, m),
                    Err(PropagationError::NumericalFailure(_))
                ),
                "{label} y0 parameter_partials {cfg:?}"
            );
        }
        for cfg in all_ecom1_shadow_configs(b0_base) {
            assert!(
                matches!(
                    cfg.acceleration_with_sun_moon(r, v, s, m),
                    Err(PropagationError::NumericalFailure(_))
                ),
                "{label} b0 accel {cfg:?}"
            );
            assert!(
                matches!(
                    cfg.acceleration_partials_with_sun_moon(r, v, s, m),
                    Err(PropagationError::NumericalFailure(_))
                ),
                "{label} b0 partials {cfg:?}"
            );
            assert!(
                matches!(
                    cfg.parameter_partials_with_sun_moon(r, v, s, m),
                    Err(PropagationError::NumericalFailure(_))
                ),
                "{label} b0 parameter_partials {cfg:?}"
            );
        }
        for cfg in all_ecom1_shadow_configs(d_only_base) {
            let a = cfg
                .acceleration_with_sun_moon(r, v, s, m)
                .unwrap_or_else(|e| panic!("{label} D-only accel failed: {e:?}"));
            let e_d = (s - r).normalize();
            let normal = r.cross(&v).normalize();
            let node = Vector3::new(-normal.y, normal.x, 0.0).normalize();
            let ahead = normal.cross(&node);
            let y_u = r.dot(&ahead);
            let x_u = r.dot(&node);
            let rho = libm::hypot(x_u, y_u);
            let d_u = 1.0 + 2.0 * (x_u / rho) - 3.0 * (y_u / rho);
            let expected_a = e_d * (d_u / M_PER_KM);
            assert!(
                (a - expected_a).norm() <= 8.0 * f64::EPSILON * expected_a.norm(),
                "{label} D-only accel {a:?} vs {expected_a:?}"
            );
            let ad = cfg
                .acceleration_partials_with_sun_moon(r, v, s, m)
                .unwrap_or_else(|e| panic!("{label} D-only partials failed: {e:?}"));
            assert!(
                ad.wrt_position.iter().all(|x| x.is_finite())
                    && ad.wrt_velocity.iter().all(|x| x.is_finite())
            );
            assert!(
                matches!(
                    cfg.parameter_partials_with_sun_moon(r, v, s, m),
                    Err(PropagationError::NumericalFailure(_))
                ),
                "{label} D-only parameter_partials must fail because e_Y is undefined"
            );
        }
    }

    // ECOM2 scaled fallback on large-magnitude exact collinear inputs (|r_m|^2 overflows):
    // power-of-two multiples of (3, 4, 5) remain exact collinear in both km and m, and
    // must also refuse y0/b0 and parameter_partials and accept D-only.
    let r_big = Vector3::new(3.0, 4.0, 5.0) * libm::ldexp(1.0, 502);
    let s_big = Vector3::new(3.0, 4.0, 5.0) * libm::ldexp(1.0, 504);
    let e2_y0 = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        y0: 1.0,
        ..Default::default()
    })
    .unwrap()
    .with_shadow_function(ShadowFunction::Conical);
    let e2_b0 = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        b0: 1.0,
        ..Default::default()
    })
    .unwrap()
    .with_shadow_function(ShadowFunction::Conical);
    let e2_d0 = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        d0: 1.0,
        ..Default::default()
    })
    .unwrap()
    .with_shadow_function(ShadowFunction::Conical);
    for m2 in [&e2_y0, &e2_b0] {
        assert!(matches!(
            m2.acceleration_with_sun(r_big, v, s_big),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            m2.acceleration_partials_with_sun(r_big, v, s_big),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            m2.parameter_partials_with_sun(r_big, v, s_big),
            Err(PropagationError::NumericalFailure(_))
        ));
    }
    let a2_d0 = e2_d0
        .acceleration_with_sun(r_big, v, s_big)
        .expect("ECOM2 scaled fallback D0 accel Ok");
    let e_d_big = Vector3::new(3.0, 4.0, 5.0).normalize();
    let expected_a2 = e_d_big / M_PER_KM;
    assert!((a2_d0 - expected_a2).norm() <= 8.0 * f64::EPSILON * expected_a2.norm());
    assert!(e2_d0
        .acceleration_partials_with_sun(r_big, v, s_big)
        .is_ok());
    assert!(matches!(
        e2_d0.parameter_partials_with_sun(r_big, v, s_big),
        Err(PropagationError::NumericalFailure(_))
    ));
}

#[test]
fn conical_tiny_lighting_gradient_preserves_full_precision_in_ecom1_and_ecom2() {
    let v = Vector3::new(0.0, 1.0, 1.0);
    let s = Vector3::new(149_597_870.7, 0.0, 0.0);
    let d0 = 1.0e308;
    let r_earth = crate::astro::constants::earth::MEAN_EARTH_RADIUS_KM;
    let r_sun = crate::astro::constants::astro::SOLAR_RADIUS_KM;
    let tan_p = libm::tan(libm::asin((r_sun + r_earth) / s.x));

    let e1_base = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        d0,
        ..Default::default()
    })
    .unwrap()
    .with_shadow_function(ShadowFunction::Conical);
    let e2_base = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        d0,
        ..Default::default()
    })
    .unwrap()
    .with_shadow_function(ShadowFunction::Conical);
    let e1_ref_base = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        d0: libm::ldexp(d0, -500),
        ..Default::default()
    })
    .unwrap()
    .with_shadow_function(ShadowFunction::Conical);
    let e2_ref_base = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        d0: libm::ldexp(d0, -500),
        ..Default::default()
    })
    .unwrap()
    .with_shadow_function(ShadowFunction::Conical);

    for (label, rz) in [
        ("normal_1_km", 1.2e-298_f64),
        ("subnormal_1_km", 1.0e-313_f64),
    ] {
        let r = Vector3::new(-1.0e7, 1.0e4, rz);
        let r_ref = Vector3::new(-1.0e7, 1.0e4, libm::ldexp(rz, 500));
        let r_pen = r_earth + (-r.x) * tan_p;

        for earth_model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
            let z_scale = match earth_model {
                EarthShadowModel::Spherical => 1.0,
                EarthShadowModel::Wgs84Oblate => {
                    1.0 / (1.0 - crate::astro::events::eclipse::WGS84_FLATTENING)
                }
            };
            let ratio = 1.0
                - crate::astro::events::eclipse::shadow_fraction_with_model(
                    as_array(r),
                    as_array(s),
                    earth_model,
                )
                .unwrap();
            let dx_s = ScaledF64::from_f64(s.x - r.x);
            let dy_s = ScaledF64::from_f64(-r.y);
            let dz_s = ScaledF64::from_f64(-r.z);
            let dist_s = ScaledVec3 {
                x: dx_s,
                y: dy_s,
                z: dz_s,
            }
            .norm()
            .unwrap();
            let ed_x_s = dx_s.div(dist_s);
            let ed_z_s = dz_s.div(dist_s);
            let ded_x_dz_s = ed_x_s.mul(ed_z_s).div(dist_s);
            let dratio_dz_s = ScaledF64::from_f64(rz)
                .mul(ScaledF64::from_f64(z_scale))
                .mul(ScaledF64::from_f64(z_scale))
                .div(ScaledF64::from_f64(r.y).mul(ScaledF64::from_f64(r_pen)));
            let expected_02 = ScaledF64::from_f64(d0)
                .div(ScaledF64::from_f64(M_PER_KM))
                .mul(ScaledF64::lincomb2(
                    dratio_dz_s,
                    ed_x_s,
                    ScaledF64::from_f64(ratio),
                    ded_x_dz_s,
                ))
                .to_f64();

            let e1 = e1_base.with_earth_shadow_model(earth_model);
            let e2 = e2_base.clone().with_earth_shadow_model(earth_model);
            let e1_ref = e1_ref_base.with_earth_shadow_model(earth_model);
            let e2_ref = e2_ref_base.clone().with_earth_shadow_model(earth_model);

            let ad1 = e1
                .acceleration_partials_with_sun(r, v, s)
                .expect("ECOM1 conical partials");
            let ad2 = e2
                .acceleration_partials_with_sun(r, v, s)
                .expect("ECOM2 conical partials");
            let ad1_ref = e1_ref
                .acceleration_partials_with_sun(r_ref, v, s)
                .expect("ECOM1 rescaled ref partials");
            let ad2_ref = e2_ref
                .acceleration_partials_with_sun(r_ref, v, s)
                .expect("ECOM2 rescaled ref partials");

            let rel1 = (ad1.wrt_position[(0, 2)] - expected_02).abs() / expected_02.abs();
            let rel2 = (ad2.wrt_position[(0, 2)] - expected_02).abs() / expected_02.abs();
            let rel1_ref = (ad1.wrt_position[(0, 2)] - ad1_ref.wrt_position[(0, 2)]).abs()
                / ad1_ref.wrt_position[(0, 2)].abs();
            let rel2_ref = (ad2.wrt_position[(0, 2)] - ad2_ref.wrt_position[(0, 2)]).abs()
                / ad2_ref.wrt_position[(0, 2)].abs();
            assert!(
                rel1 <= 8.0 * f64::EPSILON && rel1_ref <= 8.0 * f64::EPSILON,
                "ECOM1 {label} {earth_model:?}: got={:e} closed={expected_02:e} rescaled={:e} (rel={rel1:e}, ref_rel={rel1_ref:e})",
                ad1.wrt_position[(0, 2)],
                ad1_ref.wrt_position[(0, 2)]
            );
            assert!(
                rel2 <= 8.0 * f64::EPSILON && rel2_ref <= 8.0 * f64::EPSILON,
                "ECOM2 {label} {earth_model:?}: got={:e} closed={expected_02:e} rescaled={:e} (rel={rel2:e}, ref_rel={rel2_ref:e})",
                ad2.wrt_position[(0, 2)],
                ad2_ref.wrt_position[(0, 2)]
            );
        }
    }

    // Near-axis conical antumbra (0 < rho < 2^-53 * r_penumbra, including rho^2
    // underflow at rho = 1e-200 km): nu = rho / r_penumbra stays nonzero and
    // consistent with acceleration, parameter_partials, and acceleration_partials
    // rather than cancelling 1 - (r_penumbra - rho) / r_penumbra to false umbra.
    for rho_y in [1.0e-12_f64, 1.0e-200_f64] {
        let r_near = Vector3::new(-2.0e6, rho_y, 0.0);
        let r_pen = r_earth + 2.0e6 * tan_p;
        let expected_nu = rho_y / r_pen;
        let e1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
            d0: 1.0,
            ..Default::default()
        })
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical)
        .with_earth_shadow_model(EarthShadowModel::Spherical);
        let e2 = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
            d0: 1.0,
            ..Default::default()
        })
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical)
        .with_earth_shadow_model(EarthShadowModel::Spherical);
        let a1 = e1
            .acceleration_with_sun(r_near, v, s)
            .expect("ECOM1 near-axis antumbra");
        let a2 = e2
            .acceleration_with_sun(r_near, v, s)
            .expect("ECOM2 near-axis antumbra");
        let ad1 = e1
            .acceleration_partials_with_sun(r_near, v, s)
            .expect("ECOM1 near-axis antumbra partials");
        let ad2 = e2
            .acceleration_partials_with_sun(r_near, v, s)
            .expect("ECOM2 near-axis antumbra partials");
        let expected_ax = expected_nu / M_PER_KM;
        let expected_p01 = 1.0 / (r_pen * M_PER_KM);
        assert!((a1.x - expected_ax).abs() <= 8.0 * f64::EPSILON * expected_ax);
        assert!((a2.x - expected_ax).abs() <= 8.0 * f64::EPSILON * expected_ax);
        assert!(
            (ad1.wrt_position[(0, 1)] - expected_p01).abs() <= 8.0 * f64::EPSILON * expected_p01
        );
        assert!(
            (ad2.wrt_position[(0, 1)] - expected_p01).abs() <= 8.0 * f64::EPSILON * expected_p01
        );
    }
}

#[test]
fn ecom2_fallback_exact_collinear_km_input_refuses_y_b_and_accepts_d_only() {
    // Non-power-of-two exact collinear km input where |s_m - r_m|^2 overflows f64::MAX
    // (entering the ECOM2 scaled fallback) while s_km == 3.0 * r_km identically in km
    // and to_vec3_m(s_km) x to_vec3_m(r_km) is nonzero rounding noise.
    let scale = libm::ldexp(1.0, 452);
    let r_km = Vector3::new(
        (((1_u64 << 50) + 1) as f64) * scale,
        (((1_u64 << 50) + 3) as f64) * scale,
        (((1_u64 << 50) + 5) as f64) * scale,
    );
    let s_km = r_km * 3.0;
    let v = Vector3::new(0.0, 1.0, 0.0);
    let m = Vector3::new(0.0, 384_400.0, 0.0);

    let y0_base = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        y0: 1.0,
        ..Default::default()
    })
    .unwrap();
    let b0_base = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        b0: 1.0,
        ..Default::default()
    })
    .unwrap();
    let d0_base = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        d0: 1.0,
        ..Default::default()
    })
    .unwrap();

    let check_collinear_fallback = |label: &str,
                                    r_km: Vector3<f64>,
                                    s_km: Vector3<f64>,
                                    v: Vector3<f64>,
                                    conical_only: bool| {
        assert!(is_collinear_exact(s_km, r_km));
        let r_m = to_vec3_m(r_km).expect("r_m finite");
        let s_m = to_vec3_m(s_km).expect("s_m finite");
        let noise_cross = ScaledVec3::from_vec3(s_m).cross(ScaledVec3::from_vec3(r_m));
        assert!(
            noise_cross.norm().is_some(),
            "{label}: km -> m conversion must produce nonzero cross-product noise"
        );
        let cfgs = |base: &Ecom2SolarRadiationPressure| -> Vec<Ecom2SolarRadiationPressure> {
            all_ecom2_shadow_configs(base)
                .into_iter()
                .filter(|c| !conical_only || matches!(c.shadow_function(), ShadowFunction::Conical))
                .collect()
        };
        for cfg in cfgs(&y0_base).into_iter().chain(cfgs(&b0_base)) {
            assert!(
                matches!(
                    cfg.acceleration_with_sun_moon(r_km, v, s_km, m),
                    Err(PropagationError::NumericalFailure(_))
                ),
                "{label}: expected NumericalFailure for y0/b0 accel {cfg:?}"
            );
            assert!(
                matches!(
                    cfg.acceleration_partials_with_sun_moon(r_km, v, s_km, m),
                    Err(PropagationError::NumericalFailure(_))
                ),
                "{label}: expected NumericalFailure for y0/b0 state partials {cfg:?}"
            );
            assert!(
                matches!(
                    cfg.parameter_partials_with_sun_moon(r_km, v, s_km, m),
                    Err(PropagationError::NumericalFailure(_))
                ),
                "{label}: expected NumericalFailure for y0/b0 parameter partials {cfg:?}"
            );
        }
        let expected_ed = Vector3::new(
            ((1_u64 << 50) + 1) as f64,
            ((1_u64 << 50) + 3) as f64,
            ((1_u64 << 50) + 5) as f64,
        )
        .normalize();
        let expected_acc = expected_ed / M_PER_KM;
        for cfg in cfgs(&d0_base) {
            let acc = cfg
                .acceleration_with_sun_moon(r_km, v, s_km, m)
                .unwrap_or_else(|e| panic!("{label}: D-only accel must succeed {cfg:?}: {e:?}"));
            assert!(
                (acc - expected_acc).norm() <= 8.0 * f64::EPSILON * expected_acc.norm(),
                "{label} {cfg:?}: D-only acc={acc:?} vs {expected_acc:?}"
            );
            let ad = cfg
                .acceleration_partials_with_sun_moon(r_km, v, s_km, m)
                .unwrap_or_else(|e| {
                    panic!("{label}: D-only state partials must succeed {cfg:?}: {e:?}")
                });
            assert!(ad.wrt_position.iter().all(|x| x.is_finite()));
            assert_eq!(ad.wrt_velocity, Matrix3::zeros());
            assert!(
                matches!(
                    cfg.parameter_partials_with_sun_moon(r_km, v, s_km, m),
                    Err(PropagationError::NumericalFailure(_))
                ),
                "{label} {cfg:?}: D-only parameter_partials must fail when e_Y is undefined"
            );
        }
    };

    // 1. Scale 2^452: |s_km| = 6.80e151 km passes Conical validation while |s_m - r_m|^2 overflows.
    check_collinear_fallback("scale_452", r_km, s_km, v, true);
    // 2. Scale 2^449 with overflowing velocity v * 1000: passes all 6 shadow configs and enters
    //    the ECOM2 scaled fallback on every config.
    let scale_449 = libm::ldexp(1.0, 449);
    let r_449 = Vector3::new(
        (((1_u64 << 50) + 1) as f64) * scale_449,
        (((1_u64 << 50) + 3) as f64) * scale_449,
        (((1_u64 << 50) + 5) as f64) * scale_449,
    );
    check_collinear_fallback(
        "scale_449_all6",
        r_449,
        r_449 * 3.0,
        Vector3::new(0.0, 1.0e306, 0.0),
        false,
    );
}

#[test]
fn ecom2_fast_path_exact_noon_and_sun_along_normal_depart_from_orekit_noise() {
    let m = Vector3::new(0.0, 384_400.0, 0.0);

    // 1. Exact non-axis-aligned noon on the ECOM2 fast path (GNSS/AU scale):
    //    r_noon uses 39-bit mantissas so s_noon == 5000 * r_noon is exact in binary64,
    //    while raw Orekit e_D x sat_m is nonzero rounding noise.
    let noon_scale = libm::ldexp(1.0, -24);
    let r_noon = Vector3::new(
        (((1_u64 << 38) + 1) as f64) * noon_scale,
        (((1_u64 << 38) + 3) as f64) * noon_scale,
        (((1_u64 << 38) + 5) as f64) * noon_scale,
    );
    let s_noon = r_noon * 5000.0;
    let v_noon = Vector3::new(0.0, 3.87, 1.0);
    assert!(is_collinear_exact(s_noon, r_noon));
    let r_noon_m = to_vec3_m(r_noon).unwrap();
    let s_noon_m = to_vec3_m(s_noon).unwrap();
    let raw_ed_noon = raw_hipparchus_normalize_double(s_noon_m.sub(r_noon_m)).unwrap();
    let raw_ey_noise = raw_ed_noon.cross(r_noon_m);
    assert!(
        raw_ey_noise.norm() > 0.0,
        "raw Orekit e_D x sat_m must be nonzero noise on the exact noon test input"
    );
    assert!(raw_orekit_dyb_double(r_noon_m, s_noon_m).is_some());

    let y0_m = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        y0: 1.0,
        ..Default::default()
    })
    .unwrap();
    let b0_m = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        b0: 1.0,
        ..Default::default()
    })
    .unwrap();
    let d_only_m = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        d0: 1.0,
        d2c: 0.5,
        d2s: -0.25,
        ..Default::default()
    })
    .unwrap();

    for cfg in all_ecom2_shadow_configs(&y0_m)
        .into_iter()
        .chain(all_ecom2_shadow_configs(&b0_m))
    {
        assert!(matches!(
            cfg.acceleration_with_sun_moon(r_noon, v_noon, s_noon, m),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            cfg.acceleration_partials_with_sun_moon(r_noon, v_noon, s_noon, m),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            cfg.parameter_partials_with_sun_moon(r_noon, v_noon, s_noon, m),
            Err(PropagationError::NumericalFailure(_))
        ));
    }
    assert!(matches!(
        ecom_geometry_with_shadow(
            r_noon,
            v_noon,
            s_noon,
            Some(m),
            EarthShadowModel::Wgs84Oblate
        ),
        Err(PropagationError::NumericalFailure(_))
    ));
    for cfg in all_ecom2_shadow_configs(&d_only_m) {
        let acc = cfg
            .acceleration_with_sun_moon(r_noon, v_noon, s_noon, m)
            .expect("exact noon D-only accel Ok");
        let ad = cfg
            .acceleration_partials_with_sun_moon(r_noon, v_noon, s_noon, m)
            .expect("exact noon D-only state partials Ok");
        assert!(acc.iter().all(|x| x.is_finite()) && acc.norm() > 0.0);
        assert!(ad.wrt_position.iter().all(|x| x.is_finite()));
        assert!(ad.wrt_velocity.iter().all(|x| x.is_finite()));
        assert!(matches!(
            cfg.parameter_partials_with_sun_moon(r_noon, v_noon, s_noon, m),
            Err(PropagationError::NumericalFailure(_))
        ));
    }

    // 2. Exact non-axis-aligned Sun-along-orbit-normal on the ECOM2 fast path:
    //    r . s == 0 and v . s == 0 identically with r x v != 0, while raw Orekit
    //    Z_m x s_m is nonzero rounding noise.
    let s_norm = Vector3::new(3.0, 5.0, 7.0) * libm::ldexp(1.0, 24);
    let e25 = libm::ldexp(1.0, -25);
    let e26 = libm::ldexp(1.0, -26);
    let r_norm = Vector3::new(5.0 + 7.0 * e25, -3.0 + 7.0 * e26, -3.0 * e25 - 5.0 * e26)
        * libm::ldexp(1.0, 12);
    let v_norm = Vector3::new(7.0 + 5.0 * e25, -3.0 * e25 + 7.0 * e26, -3.0 - 5.0 * e26) * 0.5;
    assert!(!is_collinear_exact(r_norm, v_norm));
    assert!(exact_dot3_is_zero(as_array(r_norm), as_array(s_norm)));
    assert!(exact_dot3_is_zero(as_array(v_norm), as_array(s_norm)));
    assert!(is_delta_u_undefined_exact(r_norm, v_norm, s_norm));

    let r_norm_m = to_vec3_m(r_norm).unwrap();
    let v_norm_m = to_vec3_m(v_norm).unwrap();
    let s_norm_m = to_vec3_m(s_norm).unwrap();
    let raw_z_x_s = r_norm_m.cross(v_norm_m).cross(s_norm_m);
    assert!(
        raw_z_x_s.norm() > 0.0,
        "raw Orekit Z x sun must be nonzero noise on the exact Sun-along-normal input"
    );
    assert!(raw_orekit_delta_u_double(r_norm_m, v_norm, s_norm_m).is_some());

    let geom = ecom_geometry_with_shadow(
        r_norm,
        v_norm,
        s_norm,
        Some(m),
        EarthShadowModel::Wgs84Oblate,
    )
    .expect("DYB frame is defined when Sun is along orbit normal");
    assert_eq!(geom.delta_u_rad, None);

    let periodic_m = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
        d2c: 1.0,
        ..Default::default()
    })
    .unwrap();
    let const_d4b1_m = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
        d0: 1.0,
        y0: 2.0,
        b0: -3.0,
        ..Default::default()
    })
    .unwrap();
    let d0b0_m = Ecom2SolarRadiationPressure::new(0, 0, vec![-3.0, 1.0, 2.0]).unwrap();

    for cfg in all_ecom2_shadow_configs(&periodic_m) {
        assert!(matches!(
            cfg.acceleration_with_sun_moon(r_norm, v_norm, s_norm, m),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            cfg.acceleration_partials_with_sun_moon(r_norm, v_norm, s_norm, m),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            cfg.parameter_partials_with_sun_moon(r_norm, v_norm, s_norm, m),
            Err(PropagationError::NumericalFailure(_))
        ));
    }
    let expected_const_acc = (Vector3::from(geom.dyb.d) + Vector3::from(geom.dyb.y) * 2.0
        - Vector3::from(geom.dyb.b) * 3.0)
        / M_PER_KM;
    for cfg in all_ecom2_shadow_configs(&const_d4b1_m) {
        let acc = cfg
            .acceleration_with_sun_moon(r_norm, v_norm, s_norm, m)
            .expect("constant-only D4B1 accel Ok");
        assert!(
            (acc - expected_const_acc).norm() <= 8.0 * f64::EPSILON * expected_const_acc.norm()
        );
        let ad = cfg
            .acceleration_partials_with_sun_moon(r_norm, v_norm, s_norm, m)
            .expect("constant-only D4B1 state partials Ok");
        assert!(ad.wrt_position.iter().all(|x| x.is_finite()));
        assert_eq!(ad.wrt_velocity, Matrix3::zeros());
        assert!(matches!(
            cfg.parameter_partials_with_sun_moon(r_norm, v_norm, s_norm, m),
            Err(PropagationError::NumericalFailure(_))
        ));
    }
    for cfg in all_ecom2_shadow_configs(&d0b0_m) {
        let acc = cfg
            .acceleration_with_sun_moon(r_norm, v_norm, s_norm, m)
            .expect("D0B0 accel Ok");
        assert!(
            (acc - expected_const_acc).norm() <= 8.0 * f64::EPSILON * expected_const_acc.norm()
        );
        assert!(cfg
            .acceleration_partials_with_sun_moon(r_norm, v_norm, s_norm, m)
            .is_ok());
        let pp = cfg
            .parameter_partials_with_sun_moon(r_norm, v_norm, s_norm, m)
            .expect("D0B0 parameter_partials Ok");
        assert_eq!(pp.len(), 3);
    }
}

#[test]
fn ecom2_fast_path_exact_noon_and_sun_along_normal_sweep() {
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    let mut rng = SplitMix64(0xfa57_c011_2026_0011);
    let m = Vector3::new(0.0, 384_400.0, 0.0);
    let y0_base = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        y0: 1.0,
        ..Default::default()
    })
    .unwrap();
    let d_only_base = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        d0: 1.0,
        d2c: 0.25,
        ..Default::default()
    })
    .unwrap();
    let periodic_base = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
        b1s: 1.0,
        ..Default::default()
    })
    .unwrap();
    let const_base = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
        d0: 1.0,
        y0: -0.5,
        b0: 0.25,
        ..Default::default()
    })
    .unwrap();

    let mut noon_noise_hits = 0usize;
    let mut normal_noise_hits = 0usize;

    for case in 0..64 {
        // 1. Exact non-axis-aligned fast-path noon (39-bit mantissa * <=14-bit mult is exact).
        let p = (rng.next_u64() % 97) + 1;
        let q = (rng.next_u64() % 97) + 1;
        let t = (rng.next_u64() % 97) + 1;
        let scale = libm::ldexp(1.0, -24);
        let r_n = Vector3::new(
            (((1_u64 << 38) + p) as f64) * scale,
            (((1_u64 << 38) + q) as f64) * scale,
            (((1_u64 << 38) + t) as f64) * scale,
        );
        let mult = ((rng.next_u64() % 7) * 1000 + 3000) as f64;
        let s_n = r_n * mult;
        assert!(is_collinear_exact(s_n, r_n));
        let v_n = Vector3::new(1.0, -2.0, 3.0);
        if let (Ok(rm), Ok(sm)) = (to_vec3_m(r_n), to_vec3_m(s_n)) {
            if raw_orekit_dyb_double(rm, sm).is_some() {
                noon_noise_hits += 1;
            }
        }
        let cfg_idx = case % 6;
        let y0_cfg = all_ecom2_shadow_configs(&y0_base)[cfg_idx].clone();
        let d_cfg = all_ecom2_shadow_configs(&d_only_base)[cfg_idx].clone();
        assert!(matches!(
            y0_cfg.acceleration_with_sun_moon(r_n, v_n, s_n, m),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            y0_cfg.acceleration_partials_with_sun_moon(r_n, v_n, s_n, m),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            y0_cfg.parameter_partials_with_sun_moon(r_n, v_n, s_n, m),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(d_cfg.acceleration_with_sun_moon(r_n, v_n, s_n, m).is_ok());
        assert!(d_cfg
            .acceleration_partials_with_sun_moon(r_n, v_n, s_n, m)
            .is_ok());
        assert!(matches!(
            d_cfg.parameter_partials_with_sun_moon(r_n, v_n, s_n, m),
            Err(PropagationError::NumericalFailure(_))
        ));

        // 2. Exact non-axis-aligned fast-path Sun-along-orbit-normal.
        let sx = ((rng.next_u64() % 17) + 3) as f64;
        let sy = ((rng.next_u64() % 17) + 3) as f64;
        let sz = ((rng.next_u64() % 17) + 3) as f64;
        let s_o = Vector3::new(sx, sy, sz) * libm::ldexp(1.0, 23);
        let e1 = libm::ldexp(1.0, -24);
        let e2 = libm::ldexp(1.0, -25);
        let r_o =
            Vector3::new(sy + sz * e1, -sx + sz * e2, -sx * e1 - sy * e2) * libm::ldexp(1.0, 11);
        let v_o = Vector3::new(sz + sy * e1, -sx * e1 + sz * e2, -sx - sy * e2) * 0.25;
        assert!(exact_dot3_is_zero(as_array(r_o), as_array(s_o)));
        assert!(exact_dot3_is_zero(as_array(v_o), as_array(s_o)));
        if let (Ok(rm), Ok(sm)) = (to_vec3_m(r_o), to_vec3_m(s_o)) {
            if raw_orekit_delta_u_double(rm, v_o, sm).is_some() {
                normal_noise_hits += 1;
            }
        }
        let per_cfg = all_ecom2_shadow_configs(&periodic_base)[cfg_idx].clone();
        let cst_cfg = all_ecom2_shadow_configs(&const_base)[cfg_idx].clone();
        assert!(matches!(
            per_cfg.acceleration_with_sun_moon(r_o, v_o, s_o, m),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            per_cfg.acceleration_partials_with_sun_moon(r_o, v_o, s_o, m),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            per_cfg.parameter_partials_with_sun_moon(r_o, v_o, s_o, m),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(cst_cfg.acceleration_with_sun_moon(r_o, v_o, s_o, m).is_ok());
        assert!(cst_cfg
            .acceleration_partials_with_sun_moon(r_o, v_o, s_o, m)
            .is_ok());
        assert!(matches!(
            cst_cfg.parameter_partials_with_sun_moon(r_o, v_o, s_o, m),
            Err(PropagationError::NumericalFailure(_))
        ));
    }

    assert!(
        noon_noise_hits >= 50 && normal_noise_hits >= 50,
        "expected raw Orekit noise hits: noon={noon_noise_hits}, normal={normal_noise_hits}"
    );
}

#[test]
fn conical_ecom2_extreme_coefficients_and_tiny_nu_match_independent_reference() {
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let v = Vector3::new(0.0, 1.0, 1.0);
    let r_earth = crate::astro::constants::earth::MEAN_EARTH_RADIUS_KM;
    let r_sun = crate::astro::constants::astro::SOLAR_RADIUS_KM;
    let tan_p = libm::tan(libm::asin((r_sun + r_earth) / AU_KM));

    // 1. Matching ECOM2 inputs for sub-subnormal nu (6.34e-325), subnormal nu (6.34e-315),
    //    and normal nu (0.1875) with extreme coefficients up to 1e308 / down to 1e-315.
    let r_pen_2e6 = r_earth + 2.0e6 * tan_p;
    let r_pen_1e7 = r_earth + 1.0e7 * tan_p;
    let cases = [
        (
            "sub_subnormal_nu",
            Vector3::new(-2.0e6, 1.0e-320, 0.0),
            1.0e308,
            ScaledF64::from_f64(1.0e-320).div(ScaledF64::from_f64(r_pen_2e6)),
        ),
        (
            "subnormal_nu",
            Vector3::new(-2.0e6, 1.0e-310, 0.0),
            1.0e308,
            ScaledF64::from_f64(1.0e-310).div(ScaledF64::from_f64(r_pen_2e6)),
        ),
        (
            "tiny_coeff_normal_nu",
            Vector3::new(-1.0e7, 1.0e4, 0.0),
            1.0e-315,
            ScaledF64::from_f64(1.0e4).div(ScaledF64::from_f64(r_pen_1e7)),
        ),
    ];
    for (label, sat, d0, nu_ref) in cases {
        let base = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
            d0,
            ..Default::default()
        })
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
        let dist = libm::hypot(AU_KM - sat.x, sat.y);
        let ed_x = (AU_KM - sat.x) / dist;
        let expected_ax = ScaledF64::from_f64(d0)
            .mul(nu_ref)
            .mul(ScaledF64::from_f64(ed_x))
            .div(ScaledF64::from_f64(M_PER_KM))
            .to_f64();
        let expected_pp_d0_x = nu_ref
            .mul(ScaledF64::from_f64(ed_x))
            .div(ScaledF64::from_f64(M_PER_KM))
            .to_f64();

        for earth_model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
            let model = base.clone().with_earth_shadow_model(earth_model);
            let acc = model
                .acceleration_with_sun(sat, v, sun)
                .unwrap_or_else(|e| panic!("{label} {earth_model:?} accel failed: {e:?}"));
            let pp = model
                .parameter_partials_with_sun(sat, v, sun)
                .unwrap_or_else(|e| panic!("{label} {earth_model:?} pp failed: {e:?}"));
            let ad = model
                .acceleration_partials_with_sun(sat, v, sun)
                .unwrap_or_else(|e| panic!("{label} {earth_model:?} ad failed: {e:?}"));

            if expected_ax >= f64::MIN_POSITIVE {
                let rel = (acc.x - expected_ax).abs() / expected_ax.abs();
                assert!(
                    rel <= 8.0 * f64::EPSILON,
                    "{label} {earth_model:?}: acc.x={:e} vs {expected_ax:e} (rel={rel:e})",
                    acc.x
                );
            } else {
                assert!(
                    (acc.x - expected_ax).abs() <= 4.0 * f64::from_bits(1),
                    "{label} {earth_model:?}: subnormal acc.x={:e} vs {expected_ax:e}",
                    acc.x
                );
            }
            if expected_pp_d0_x >= f64::MIN_POSITIVE {
                let rel = (pp[3].x - expected_pp_d0_x).abs() / expected_pp_d0_x.abs();
                assert!(rel <= 8.0 * f64::EPSILON);
            } else {
                assert!((pp[3].x - expected_pp_d0_x).abs() <= 4.0 * f64::from_bits(1));
            }
            assert!(ad.wrt_position.iter().all(|x| x.is_finite()));
            assert!(ad.wrt_velocity.iter().all(|x| x.is_finite()));

            // Double acceleration equals both N=6 and N=7 field value parts bit for bit.
            let (ratio_km, is_sunlit) =
                lighting_ratio_field_conical_scaled_km(sat, sun, earth_model).unwrap();
            assert!(!is_sunlit);
            let params_g6: Vec<Grad<6>> = model
                .coefficients()
                .iter()
                .map(|&c| Grad::constant(c))
                .collect();
            let f6 = ecom2_acceleration_field_scaled(
                sat,
                v,
                sun,
                true,
                true,
                ScaledGrad {
                    val: ratio_km.val,
                    grad: [
                        ratio_km.grad[0],
                        ratio_km.grad[1],
                        ratio_km.grad[2],
                        ScaledF64::ZERO,
                        ScaledF64::ZERO,
                        ScaledF64::ZERO,
                    ],
                },
                false,
                1,
                1,
                &params_g6,
                model.coefficients(),
                false,
            )
            .unwrap()
            .val_vec()
            .to_vec3()
            .unwrap();
            assert_eq!(acc.x.to_bits(), f6.x.to_bits());
            assert_eq!(acc.y.to_bits(), f6.y.to_bits());
            assert_eq!(acc.z.to_bits(), f6.z.to_bits());
        }
    }

    // 2. Exact non-axis-aligned antumbra axis in ECOM1 and ECOM2:
    //    acceleration and parameter_partials return exact zero (even for y0 = 1.0);
    //    acceleration_partials returns NumericalFailure (derivative-only exception).
    let sat_ax = Vector3::new(-2.0e6, -3.0e6, 0.0);
    let sun_ax = Vector3::new(2.0e8, 3.0e8, 0.0);
    let v_ax = Vector3::new(0.0, 1.0, 0.0);
    for earth_model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
        for (d0, y0) in [(1.0, 0.0), (0.0, 1.0)] {
            let e1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
                d0,
                y0,
                ..Default::default()
            })
            .unwrap()
            .with_shadow_function(ShadowFunction::Conical)
            .with_earth_shadow_model(earth_model);
            let e2 = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
                d0,
                y0,
                ..Default::default()
            })
            .unwrap()
            .with_shadow_function(ShadowFunction::Conical)
            .with_earth_shadow_model(earth_model);
            assert_eq!(
                e1.acceleration_with_sun(sat_ax, v_ax, sun_ax).unwrap(),
                Vector3::zeros()
            );
            assert_eq!(
                e1.parameter_partials_with_sun(sat_ax, v_ax, sun_ax)
                    .unwrap(),
                [Vector3::zeros(); ECOM1_COEFFICIENT_COUNT]
            );
            assert!(matches!(
                e1.acceleration_partials_with_sun(sat_ax, v_ax, sun_ax),
                Err(PropagationError::NumericalFailure(_))
            ));
            assert_eq!(
                e2.acceleration_with_sun(sat_ax, v_ax, sun_ax).unwrap(),
                Vector3::zeros()
            );
            assert_eq!(
                e2.parameter_partials_with_sun(sat_ax, v_ax, sun_ax)
                    .unwrap(),
                vec![Vector3::zeros(); ECOM2_D2B1_COEFFICIENT_COUNT]
            );
            assert!(matches!(
                e2.acceleration_partials_with_sun(sat_ax, v_ax, sun_ax),
                Err(PropagationError::NumericalFailure(_))
            ));
        }
    }

    // 3. Sweep over extreme ECOM2 coefficients and tiny conical nu with non-axis-aligned
    //    Sun directions (Spherical & WGS84).
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
        }
        fn log_uniform(&mut self, lo: i32, hi: i32) -> f64 {
            let e = lo as f64 + (hi - lo) as f64 * self.unit();
            let ei = e.floor() as i32;
            libm::ldexp(libm::exp2(e - ei as f64), ei)
        }
    }
    let mut rng = SplitMix64(0xec02_c0a1_2026_0012);
    let d_sun_c = libm::hypot(2.0e8, 3.0e8);
    let dist_b = libm::hypot(2.0e6, 3.0e6);
    let tan_pc = libm::tan(libm::asin((r_sun + r_earth) / d_sun_c));
    let r_pen_c = r_earth + dist_b * tan_pc;
    let r_ss = libm::hypot(2.02e8, 3.03e8);
    let ed_x = 2.02e8 / r_ss;
    let ed_y = 3.03e8 / r_ss;
    let mut checked = 0usize;
    let mut max_acc_eps = 0.0_f64;
    let mut max_jp_eps = 0.0_f64;

    for case in 0..256 {
        let earth_model = if case % 2 == 0 {
            EarthShadowModel::Spherical
        } else {
            EarthShadowModel::Wgs84Oblate
        };
        let k = match earth_model {
            EarthShadowModel::Spherical => 1.0,
            EarthShadowModel::Wgs84Oblate => {
                1.0 / (1.0 - crate::astro::events::eclipse::WGS84_FLATTENING)
            }
        };
        let rho_z = rng.log_uniform(-1060, -20);
        let nu_est_exp = crate::astro::events::eclipse::ad::exact_exponent(rho_z) - 14;
        let d0_exp = (-nu_est_exp).clamp(-1040, 1020);
        let d0 = libm::ldexp(1.0 + rng.unit(), d0_exp);
        let sat_c = Vector3::new(-2.0e6, -3.0e6, rho_z);
        let sun_c = Vector3::new(2.0e8, 3.0e8, 0.0);
        let v_c = Vector3::new(0.0, 1.0, 1.0);
        let model = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
            d0,
            ..Default::default()
        })
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical)
        .with_earth_shadow_model(earth_model);

        let acc = model.acceleration_with_sun(sat_c, v_c, sun_c).unwrap();
        let ad = model.acceleration_partials_with_sun(sat_c, v_c, sun_c);
        let (ratio_km, is_sunlit) =
            lighting_ratio_field_conical_scaled_km(sat_c, sun_c, earth_model).unwrap();
        assert!(!is_sunlit);
        let params_g6: Vec<Grad<6>> = model
            .coefficients()
            .iter()
            .map(|&c| Grad::constant(c))
            .collect();
        let f6 = ecom2_acceleration_field_scaled(
            sat_c,
            v_c,
            sun_c,
            true,
            true,
            ScaledGrad {
                val: ratio_km.val,
                grad: [
                    ratio_km.grad[0],
                    ratio_km.grad[1],
                    ratio_km.grad[2],
                    ScaledF64::ZERO,
                    ScaledF64::ZERO,
                    ScaledF64::ZERO,
                ],
            },
            false,
            1,
            1,
            &params_g6,
            model.coefficients(),
            false,
        )
        .unwrap()
        .val_vec()
        .to_vec3()
        .unwrap();
        assert_eq!(acc.x.to_bits(), f6.x.to_bits());
        assert_eq!(acc.y.to_bits(), f6.y.to_bits());
        assert_eq!(acc.z.to_bits(), f6.z.to_bits());

        let nu_ref = ScaledF64::from_f64(rho_z)
            .mul(ScaledF64::from_f64(k))
            .div(ScaledF64::from_f64(r_pen_c));
        let d0_km = ScaledF64::from_f64(d0).div(ScaledF64::from_f64(M_PER_KM));
        let ax_ref = d0_km.mul(nu_ref).mul(ScaledF64::from_f64(ed_x)).to_f64();
        let ay_ref = d0_km.mul(nu_ref).mul(ScaledF64::from_f64(ed_y)).to_f64();
        if ax_ref >= f64::MIN_POSITIVE && ax_ref.is_finite() {
            max_acc_eps = max_acc_eps
                .max((acc.x - ax_ref).abs() / ax_ref.abs() / f64::EPSILON)
                .max((acc.y - ay_ref).abs() / ay_ref.abs() / f64::EPSILON);
            checked += 1;
        }
        if let Ok(j) = ad {
            let j02_ref = d0_km
                .mul(ScaledF64::from_f64(k / r_pen_c))
                .mul(ScaledF64::from_f64(ed_x))
                .to_f64();
            if j02_ref >= f64::MIN_POSITIVE && j02_ref.is_finite() {
                max_jp_eps = max_jp_eps
                    .max((j.wrt_position[(0, 2)] - j02_ref).abs() / j02_ref.abs() / f64::EPSILON);
            }
        }
    }
    // Measured maxima across 256 extreme conical ECOM2 cases (Spherical & WGS84):
    // acc = 2.41 eps, jp = 2.46 eps (ScaledF64 rounding in e_D, r_penumbra, nu,
    // and / M_PER_KM).
    assert!(
        checked >= 200,
        "too few conical extreme ECOM2 checks: {checked}"
    );
    assert!(
        max_acc_eps <= 16.0 && max_jp_eps <= 16.0,
        "conical extreme ECOM2 sweep maxima: acc={max_acc_eps:.2} eps, jp={max_jp_eps:.2} eps"
    );
}
