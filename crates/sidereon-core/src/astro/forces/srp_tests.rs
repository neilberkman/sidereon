//! Clean-room validation anchors:
//!
//! The expected 1 AU acceleration magnitude is the published cannonball SRP
//! pressure `4.56e-6 N/m^2` multiplied by `C_R * A/m` and converted from
//! m/s^2 to km/s^2, with the satellite one AU from the Sun. The shadow tests
//! cover all six shadow configurations: Orekit's apparent-disc occultation
//! (WGS84 oblate or spherical Earth, with or without the Moon; default WGS84
//! with the Moon) and the Earth-only linear conical shadow (spherical or
//! oblate Earth). The position partials are checked against difference
//! quotients of the acceleration under each shadow configuration.

use super::*;
use crate::astro::constants::astro::{AU_KM, IAU_2015_SOLAR_RADIUS_KM, MOON_RADIUS_KM};
use crate::astro::events::eclipse::{
    shadow_fraction, shadow_fraction_with_model, shadow_fraction_with_model_and_gradient,
    shadow_fraction_with_model_and_moon, shadow_fraction_with_model_moon_and_gradient,
    ShadowGradientError,
};
use crate::astro::forces::test_helpers::{
    epochs_1990_to_2050, gcrs_sun_moon_km, mean_of_date_sun_moon_km,
};

/// A Moon position far off the Sun line so lunar occultation does not fire.
const CLEAR_MOON_KM: Vector3<f64> = Vector3::new(0.0, 384_400.0, 0.0);

fn all_shadow_configs(base: SolarRadiationPressure) -> [SolarRadiationPressure; 6] {
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

fn sat_sun(sat: Vector3<f64>, sun: Vector3<f64>) -> ([f64; 3], [f64; 3]) {
    (as_array(sat), as_array(sun))
}

#[test]
fn one_au_magnitude_matches_pressure_area_mass_scale() {
    let model = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP");
    let sat = Vector3::new(7000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM + 7000.0, 0.0, 0.0);
    let accel = srp_acceleration_with_bodies(sat, sun, Some(CLEAR_MOON_KM), model)
        .expect("SRP acceleration");

    let expected_km_s2 = 4.56e-6 * 1.2 * 0.02 / M_PER_KM;
    assert!((accel.norm() - expected_km_s2).abs() < 1.0e-24);
}

#[test]
fn flux_falls_off_with_the_satellite_sun_distance() {
    let model = SolarRadiationPressure::new(1.0, 0.02).expect("valid SRP");
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    // Two sunlit satellites 26560 km from the geocenter, nearer to and
    // farther from the Sun. The ratio of the accelerations is the inverse
    // square of the ratio of their distances from the Sun.
    let near = Vector3::new(26_560.0, 0.0, 0.0);
    let far = Vector3::new(-26_560.0, 30_000.0, 0.0);
    let near_accel =
        srp_acceleration_with_bodies(near, sun, Some(CLEAR_MOON_KM), model).expect("near");
    let far_accel =
        srp_acceleration_with_bodies(far, sun, Some(CLEAR_MOON_KM), model).expect("far");
    let expected = ((sun - far).norm() / (sun - near).norm()).powi(2);
    // A few ulps of the two norms and the quotient.
    assert!((near_accel.norm() / far_accel.norm() - expected).abs() <= 8.0 * f64::EPSILON);
    assert!(expected > 1.0 + 3.0e-4);
}

#[test]
fn acceleration_is_zero_in_deep_umbra() {
    let model = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP");
    let sat = Vector3::new(-7000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let accel = srp_acceleration_with_bodies(sat, sun, Some(CLEAR_MOON_KM), model)
        .expect("SRP acceleration");

    assert_eq!(accel, Vector3::zeros());
}

#[test]
fn direction_is_away_from_sun() {
    let model = SolarRadiationPressure::new(1.0, 0.01).expect("valid SRP");
    let sat = Vector3::new(7000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let accel = srp_acceleration_with_bodies(sat, sun, Some(CLEAR_MOON_KM), model)
        .expect("SRP acceleration");

    assert!(accel.x < 0.0);
    assert_eq!(accel.y, 0.0);
    assert_eq!(accel.z, 0.0);
}

#[test]
fn default_shadow_is_wgs84_apparent_disc_with_moon_and_all_options_work() {
    let default_model = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP");
    assert_eq!(
        default_model.shadow_function(),
        ShadowFunction::ApparentDisc
    );
    assert_eq!(
        default_model.earth_shadow_model(),
        EarthShadowModel::Wgs84Oblate
    );
    assert!(default_model.moon_eclipse_enabled());

    // Turning Moon off keeps ApparentDisc (Earth-only apparent-disc).
    let earth_disc = default_model.with_moon_eclipse(false);
    assert_eq!(earth_disc.shadow_function(), ShadowFunction::ApparentDisc);
    assert_eq!(
        earth_disc.earth_shadow_model(),
        EarthShadowModel::Wgs84Oblate
    );
    assert!(!earth_disc.moon_eclipse_enabled());

    // Selecting Conical disables Moon and gives canonical PartialEq.
    let conical_oblate = default_model.with_shadow_function(ShadowFunction::Conical);
    assert_eq!(conical_oblate.shadow_function(), ShadowFunction::Conical);
    assert_eq!(
        conical_oblate.earth_shadow_model(),
        EarthShadowModel::Wgs84Oblate
    );
    assert!(!conical_oblate.moon_eclipse_enabled());
    assert_eq!(
        conical_oblate,
        earth_disc.with_shadow_function(ShadowFunction::Conical)
    );
    assert_eq!(conical_oblate.with_moon_eclipse(false), conical_oblate);

    // Enabling Moon on a conical model switches back to ApparentDisc.
    assert_eq!(conical_oblate.with_moon_eclipse(true), default_model);

    // Switching back to ApparentDisc from Conical keeps moon disabled until enabled.
    assert_eq!(
        conical_oblate.with_shadow_function(ShadowFunction::ApparentDisc),
        earth_disc
    );

    let conical_spherical = conical_oblate.with_earth_shadow_model(EarthShadowModel::Spherical);
    assert_eq!(conical_spherical.shadow_function(), ShadowFunction::Conical);
    assert_eq!(
        conical_spherical.earth_shadow_model(),
        EarthShadowModel::Spherical
    );
    assert!(!conical_spherical.moon_eclipse_enabled());

    let sat = Vector3::new(7_000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let moon_clear = CLEAR_MOON_KM;
    let moon_total = Vector3::new(7_000.0 + 356_000.0, 0.0, 0.0);
    let moon_annular = Vector3::new(7_000.0 + 405_000.0, 0.0, 0.0);

    // Clear Moon matches Earth-only models bit-for-bit in full sunlight.
    let accel_default = default_model
        .acceleration_with_sun_moon(sat, sun, moon_clear)
        .expect("default SRP");
    for other in [earth_disc, conical_oblate, conical_spherical] {
        let accel_other = other
            .acceleration_with_sun_moon(sat, sun, moon_clear)
            .expect("clear SRP");
        for i in 0..3 {
            assert_eq!(accel_other[i].to_bits(), accel_default[i].to_bits());
        }
    }

    // Default zeroes acceleration in lunar umbra; Earth-only options ignore the Moon.
    let accel_total = default_model
        .acceleration_with_sun_moon(sat, sun, moon_total)
        .expect("lunar total SRP");
    assert_eq!(accel_total, Vector3::zeros());
    for earth_only in [earth_disc, conical_oblate, conical_spherical] {
        let accel_ignored = earth_only
            .acceleration_with_sun_moon(sat, sun, moon_total)
            .expect("earth-only ignores moon");
        for i in 0..3 {
            assert_eq!(accel_ignored[i].to_bits(), accel_default[i].to_bits());
        }
    }

    // Annular lunar eclipse reduces SRP without zeroing it.
    let accel_annular = default_model
        .acceleration_with_sun_moon(sat, sun, moon_annular)
        .expect("lunar annular SRP");
    assert!(accel_annular.x < 0.0);
    assert!(accel_annular.norm() < 0.25 * accel_default.norm());
    assert!(accel_annular.norm() > 0.1 * accel_default.norm());

    // Non-finite Moon position returns a typed error under every option.
    for model in all_shadow_configs(default_model) {
        let err = model
            .acceleration_with_sun_moon(sat, sun, Vector3::new(f64::NAN, 0.0, 0.0))
            .expect_err("non-finite moon rejected");
        assert!(matches!(err, PropagationError::InvalidInput(_)));
    }
}

#[test]
fn non_finite_state_position_preserves_force_model_error_variant() {
    let model = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP");
    let state = CartesianState::new(0.0, [f64::NAN, 0.0, 0.0], [0.0, 7.5, 0.0]);
    let ctx = PropagationContext::default();
    let err = model
        .acceleration(&state, &ctx)
        .expect_err("non-finite state position must fail");
    assert!(
        matches!(err, PropagationError::ForceModelFailure(_)),
        "expected ForceModelFailure, got {err:?}"
    );
}

/// Fourth-order central difference quotients of an acceleration with respect
/// to the satellite position, step `h` km.
fn difference_quotients(
    acceleration: impl Fn(Vector3<f64>) -> Vector3<f64>,
    sat: Vector3<f64>,
    h: f64,
) -> Matrix3<f64> {
    let mut gradient = Matrix3::zeros();
    for axis in 0..3 {
        let at = |steps: f64| {
            let mut moved = sat;
            moved[axis] += steps * h;
            acceleration(moved)
        };
        let column = (at(-2.0) - at(-1.0) * 8.0 + at(1.0) * 8.0 - at(2.0)) / (12.0 * h);
        gradient.set_column(axis, &column);
    }
    gradient
}

/// `u` turned by `angle` toward the perpendicular direction at `azimuth`
/// about it, measured from the component of the z axis (or x axis) normal to
/// `u`.
fn tilted(u: Vector3<f64>, angle: f64, azimuth: f64) -> Vector3<f64> {
    let reference = if u.z.abs() < 0.9 {
        Vector3::z()
    } else {
        Vector3::x()
    };
    let e1 = (reference - u * reference.dot(&u)).normalize();
    let e2 = u.cross(&e1);
    let direction = e1 * libm::cos(azimuth) + e2 * libm::sin(azimuth);
    (u * libm::cos(angle) + direction * libm::sin(angle)).normalize()
}

/// Check the position partials of `model` at `sat` against fourth-order
/// difference quotients of its acceleration, Sun and Moon fixed.
fn assert_partials_match_differences(
    label: &str,
    model: SolarRadiationPressure,
    sat: Vector3<f64>,
    sun: Vector3<f64>,
    moon: Option<Vector3<f64>>,
    h: f64,
) -> f64 {
    let analytic = srp_position_partials_with_bodies(sat, sun, moon, model).expect("partials");
    let numeric = difference_quotients(
        |moved| srp_acceleration_with_bodies(moved, sun, moon, model).expect("acceleration"),
        sat,
        h,
    );
    let scale = analytic.norm();
    let error = (analytic - numeric).norm() / scale;
    // Measured maximum relative error across all callers below is 4.92e-7
    // (2e6 km annular oblate Earth at h = 1e-2 km; 4.54e-7 on LEO Earth+Moon
    // at h = 5e-2 km; <= 9.89e-8 on all other cases). Cause: Orekit's
    // maskingRatio cancellation/annular jitter (~5e-10) divided by the stencil
    // step h, alongside h^4 truncation and acceleration rounding.
    assert!(
        error <= 5.0e-7,
        "{label}: relative error {error:e}\nanalytic {analytic:?}\nnumeric {numeric:?}"
    );
    scale
}

#[test]
fn position_partials_match_difference_quotients_in_sunlight() {
    let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let sun = Vector3::new(1.2e8, -8.0e7, -3.5e7);
    for sat in [
        Vector3::new(15_000.0, 20_000.0, 9_000.0),
        Vector3::new(-26_000.0, 2_000.0, 5_000.0),
    ] {
        let (sat_a, sun_a) = sat_sun(sat, sun);
        assert_eq!(shadow_fraction(sat_a, sun_a).expect("shadow"), 0.0);
        // The gradient is of order |a| / d = 1.2e-10 km/s^2 / 1.5e8 km
        // = 8e-19 1/s^2; a 1 km step keeps the truncation error at
        // (1 km / 1.5e8 km)^4 and the rounding error at
        // 1.2e-10 * eps / 1 km = 3e-26 1/s^2, 4e-8 of the gradient.
        for cfg in all_shadow_configs(model) {
            let moon = cfg.moon_eclipse_enabled().then_some(CLEAR_MOON_KM);
            let scale = assert_partials_match_differences("sunlight", cfg, sat, sun, moon, 1.0);
            assert!(scale > 1.0e-19);
        }
    }
}

#[test]
fn position_partials_match_difference_quotients_in_the_penumbra() {
    let base = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    // 26000 km behind the Earth, where the umbra cone has a radius of
    // about 6250 km and the penumbra cone about 6490 km; the satellite is
    // 6357 km from the shadow axis.
    let sat = Vector3::new(-26_000.0, 6_000.0, 2_100.0);
    let (sat_a, sun_a) = sat_sun(sat, sun);
    let shadow = shadow_fraction(sat_a, sun_a).expect("shadow");
    assert!(shadow > 0.0 && shadow < 1.0, "shadow {shadow}");
    // Inside the penumbra the shadow term dominates: the fraction changes
    // by 1 over the ~240 km penumbra width, so the gradient is of order
    // |a| / 240 km = 5e-13 1/s^2. A 10 m step stays well inside the penumbra
    // and keeps maskingRatio jitter well below the 5e-7 tolerance.
    for cfg in all_shadow_configs(base) {
        let moon = cfg.moon_eclipse_enabled().then_some(CLEAR_MOON_KM);
        let scale = assert_partials_match_differences("penumbra", cfg, sat, sun, moon, 1.0e-2);
        assert!(scale > 1.0e-13);
    }
}

/// A sunlit LEO satellite, the Sun along a direction off every axis, and a
/// Moon 384400 km from the satellite at angle `c_moon` from the Sun's center
/// as the satellite sees it.
fn moon_geometry(sat: Vector3<f64>, c_moon: f64, azimuth: f64) -> (Vector3<f64>, Vector3<f64>) {
    let sun = Vector3::new(1.2e8, -8.0e7, -3.5e7);
    let u_sun = (sun - sat).normalize();
    let moon = sat + tilted(u_sun, c_moon, azimuth) * 384_400.0;
    (sun, moon)
}

fn sun_radius_seen_from(sat: Vector3<f64>, sun: Vector3<f64>) -> f64 {
    libm::asin(IAU_2015_SOLAR_RADIUS_KM / (sun - sat).norm())
}

#[test]
fn position_partials_include_the_moon_shadow() {
    let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let sun_dir = Vector3::new(1.2e8, -8.0e7, -3.5e7).normalize();
    let sat = sun_dir * 7_000.0;
    let probe_sun = Vector3::new(1.2e8, -8.0e7, -3.5e7);
    let a = sun_radius_seen_from(sat, probe_sun);
    let b_moon = libm::asin(MOON_RADIUS_KM / 384_400.0);
    // The Moon's disc halfway across the Sun's edge.
    let (sun, moon) = moon_geometry(sat, a + 0.1 * b_moon, 0.7);
    let fraction = shadow_fraction_with_model_and_moon(
        as_array(sat),
        as_array(sun),
        as_array(moon),
        EarthShadowModel::Wgs84Oblate,
    )
    .expect("fraction");
    assert!(fraction > 0.2 && fraction < 0.6, "fraction {fraction}");
    // The Moon's penumbra is about 3500 km wide here, so the shadow term is
    // of order |a| / 3500 km = 3e-14 1/s^2; a 10 m step stays 300 km from
    // its edges.
    let scale = assert_partials_match_differences("moon", model, sat, sun, Some(moon), 1.0e-2);
    let without_shadow_term = {
        let sun_sat = sat - sun;
        let d = sun_sat.norm();
        let s = -sun_sat / d;
        let scale_km_s2 =
            (model.pressure_n_m2 * (model.au_km / d).powi(2) * model.cr * model.area_to_mass_m2_kg)
                / M_PER_KM;
        let outer = Matrix3::from_fn(|r, c| s[r] * s[c]);
        (Matrix3::identity() - outer * 3.0) * ((1.0 - fraction) / d * scale_km_s2)
    };
    assert!(scale > 1.0e3 * without_shadow_term.norm());
}

#[test]
fn position_partials_include_the_earth_and_moon_shadows_together() {
    let model = SolarRadiationPressure::new(1.3, 0.02)
        .expect("valid SRP")
        .with_earth_shadow_model(EarthShadowModel::Spherical);
    let sun = Vector3::new(1.2e8, -8.0e7, -3.5e7);
    let u = sun.normalize();
    // A LEO satellite whose view of the Sun is cut by the Earth's limb: the
    // Earth's apparent radius at 7000 km is asin(6371 / 7000), so with the
    // Sun direction at pi - b_E from the nadir the limb crosses the Sun's
    // center.
    let b_earth = libm::asin(6_371.0 / 7_000.0);
    let normal = u.cross(&Vector3::z()).normalize();
    let along = normal.cross(&u);
    let theta = std::f64::consts::PI - b_earth + 1.0e-3;
    let sat = (u * libm::cos(theta) + along * libm::sin(theta)) * 7_000.0;
    let u_sun = (sun - sat).normalize();
    let a = sun_radius_seen_from(sat, sun);
    // The Moon over the half of the Sun that the Earth leaves.
    let away_from_earth = (u_sun - (-sat.normalize())).normalize();
    let azimuth = {
        let reference = if u_sun.z.abs() < 0.9 {
            Vector3::z()
        } else {
            Vector3::x()
        };
        let e1 = (reference - u_sun * reference.dot(&u_sun)).normalize();
        let e2 = u_sun.cross(&e1);
        libm::atan2(away_from_earth.dot(&e2), away_from_earth.dot(&e1)) + 0.4
    };
    let moon = sat + tilted(u_sun, 0.8 * a, azimuth) * 384_400.0;
    let both = shadow_fraction_with_model_and_moon(
        as_array(sat),
        as_array(sun),
        as_array(moon),
        EarthShadowModel::Spherical,
    )
    .expect("fraction");
    let far_moon = sat + tilted(u_sun, 0.5, azimuth) * 384_400.0;
    let earth_only = shadow_fraction_with_model_and_moon(
        as_array(sat),
        as_array(sun),
        as_array(far_moon),
        EarthShadowModel::Spherical,
    )
    .expect("fraction");
    assert!(earth_only > 0.05 && earth_only < 0.95, "earth {earth_only}");
    assert!(both > earth_only + 0.05 && both < 1.0, "both {both}");
    // The Earth's penumbra crosses the satellite's track over about 30 km,
    // so the shadow term is of order |a| / 30 km = 4e-12 1/s^2; a 50 m step
    // keeps Orekit's maskingRatio cancellation jitter (~5e-10) well below 1e-6
    // relative to the derivative while staying deep inside the penumbra.
    assert_partials_match_differences("earth and moon", model, sat, sun, Some(moon), 5.0e-2);
}

/// A satellite 7000 km from the geocenter passing high over the north pole
/// from sunlight into the Earth's shadow, whose view of the Sun is cut near
/// the polar limb; `offset` (rad) moves it along the pass, from sunlight at 0
/// into the umbra by 0.06.
fn polar_pass_geometry(offset: f64) -> (Vector3<f64>, Vector3<f64>) {
    let sun = Vector3::new(AU_KM, 0.0, 2.0e6);
    let u = sun.normalize();
    let toward_pole = u.cross(&Vector3::z()).normalize().cross(&u);
    // The Earth's apparent radius at 7000 km is asin(6371 / 7000); the
    // shadow begins about that far short of the antisolar direction.
    let theta = std::f64::consts::PI - libm::asin(6_371.0 / 7_000.0) - 0.02 + offset;
    let sat = (u * libm::cos(theta) + toward_pole * libm::sin(theta)) * 7_000.0;
    (sat, sun)
}

/// The offsets along [`polar_pass_geometry`], scanned in steps of `step`,
/// where `fraction` first exceeds 0 and first reaches 1.
fn penumbra_span(fraction: impl Fn(f64) -> f64, step: f64) -> (f64, f64) {
    let mut offset = 0.0;
    assert_eq!(fraction(offset), 0.0, "pass starts in sunlight");
    while fraction(offset) <= 0.0 {
        offset += step;
        assert!(offset < 0.06, "no penumbra on the pass");
    }
    let entry = offset;
    while fraction(offset) < 1.0 {
        offset += step;
        assert!(offset < 0.06, "no umbra on the pass");
    }
    (entry, offset)
}

#[test]
fn position_partials_follow_the_oblate_conical_shadow() {
    let model = SolarRadiationPressure::new(1.3, 0.02)
        .expect("valid SRP")
        .with_shadow_function(ShadowFunction::Conical)
        .with_earth_shadow_model(EarthShadowModel::Wgs84Oblate);
    let oblate_at = |offset: f64| {
        let (sat, sun) = polar_pass_geometry(offset);
        let (sat_a, sun_a) = sat_sun(sat, sun);
        shadow_fraction_with_model(sat_a, sun_a, EarthShadowModel::Wgs84Oblate).expect("fraction")
    };
    let (entry, full) = penumbra_span(oblate_at, 1.0e-4);
    let (sat, sun) = polar_pass_geometry(0.5 * (entry + full));
    let (sat_a, sun_a) = sat_sun(sat, sun);
    let oblate =
        shadow_fraction_with_model(sat_a, sun_a, EarthShadowModel::Wgs84Oblate).expect("fraction");
    let spherical = shadow_fraction(sat_a, sun_a).expect("fraction");
    assert!(oblate > 0.0 && oblate < 1.0, "oblate {oblate}");
    assert!(
        (oblate - spherical).abs() > 1.0e-3,
        "{oblate} vs {spherical}"
    );
    // The penumbra spans about 60 km of the pass; a 1 m step stays inside it.
    assert_partials_match_differences("oblate conical", model, sat, sun, None, 1.0e-3);
}

#[test]
fn position_partials_follow_the_oblate_earth_with_the_moon() {
    let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let fraction_at = |offset: f64, moon_angle: f64| {
        let (sat, sun) = polar_pass_geometry(offset);
        let u_sun = (sun - sat).normalize();
        let moon = sat + tilted(u_sun, moon_angle, 2.0) * 384_400.0;
        let fraction = shadow_fraction_with_model_and_moon(
            as_array(sat),
            as_array(sun),
            as_array(moon),
            EarthShadowModel::Wgs84Oblate,
        )
        .expect("fraction");
        (fraction, sat, sun, moon)
    };
    // The Earth-only penumbra with the Moon far from the Sun, then the
    // Moon brought over the part of the Sun the Earth leaves.
    let (entry, full) = penumbra_span(|offset| fraction_at(offset, 0.5).0, 1.0e-4);
    let offset = entry + 0.3 * (full - entry);
    let (earth_only, sat, sun, _) = fraction_at(offset, 0.5);
    let a = sun_radius_seen_from(sat, sun);
    let (both, _, _, moon) = fraction_at(offset, 0.9 * a);
    assert!(earth_only > 0.0 && earth_only < 0.7, "earth {earth_only}");
    assert!(both > earth_only + 0.02 && both < 1.0, "both {both}");
    assert_partials_match_differences("oblate with moon", model, sat, sun, Some(moon), 5.0e-2);
}

#[test]
fn partials_report_numerical_failure_on_exact_annular_axis() {
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    // Beyond the tip of the Earth's umbra, on the exact Sun-Earth line:
    // field evaluation produces NaN (zero separation angle in maskingRatio
    // and degenerate limb plane for the oblate ellipsoid).
    let sat = Vector3::new(-2.0e6, 0.0, 0.0);
    let moon = CLEAR_MOON_KM;
    let base = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    for model in [
        base,
        base.with_moon_eclipse(false),
        base.with_earth_shadow_model(EarthShadowModel::Spherical),
        base.with_earth_shadow_model(EarthShadowModel::Spherical)
            .with_moon_eclipse(false),
    ] {
        let moon_opt = model.moon_eclipse_enabled().then_some(moon);
        let accel = srp_acceleration_with_bodies(sat, sun, moon_opt, model).expect("accel");
        assert!(accel.norm() > 0.0);
        let refused = srp_position_partials_with_bodies(sat, sun, moon_opt, model);
        assert!(
            matches!(refused, Err(PropagationError::NumericalFailure(_))),
            "{model:?}: expected NumericalFailure, got {refused:?}"
        );
        if model.moon_eclipse_enabled() {
            assert_eq!(
                shadow_fraction_with_model_moon_and_gradient(
                    as_array(sat),
                    as_array(sun),
                    as_array(moon),
                    model.earth_shadow_model,
                ),
                Err(ShadowGradientError::NumericalFailure)
            );
        }
    }
}

#[test]
fn partials_report_numerical_failure_on_conical_antumbra_axis() {
    // Past the tip of the Earth's umbra on the exact shadow axis, rho == 0
    // produces NaN in the square-root derivative.
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let sat = Vector3::new(-2.0e6, 0.0, 0.0);
    let model = SolarRadiationPressure::new(1.3, 0.02)
        .expect("valid SRP")
        .with_shadow_function(ShadowFunction::Conical);
    for model in [
        model.with_earth_shadow_model(EarthShadowModel::Spherical),
        model.with_earth_shadow_model(EarthShadowModel::Wgs84Oblate),
    ] {
        assert!(srp_acceleration_with_bodies(sat, sun, None, model).is_ok());
        let refused = srp_position_partials_with_bodies(sat, sun, None, model);
        assert!(
            matches!(refused, Err(PropagationError::NumericalFailure(_))),
            "{model:?}: expected NumericalFailure, got {refused:?}"
        );
        assert_eq!(
            shadow_fraction_with_model_and_gradient(
                as_array(sat),
                as_array(sun),
                model.earth_shadow_model
            ),
            Err(ShadowGradientError::NumericalFailure)
        );
    }
}

#[test]
fn partials_report_numerical_failure_when_near_axis_cross_product_underflows() {
    // 1e-313 km off the line through the Sun and Earth's center, the cross
    // products `observer × outside` and `ps × psat` have squared norms that
    // underflow to 0.0, landing on the exact-collinearity fallback in
    // `pointOnLimb` and `angle = 0.0` (`1 / 0 = Inf`, `NaN`) in
    // `masking_ratio_field`.
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let sat = Vector3::new(-2.0e6, 1.0e-313, 1.0e-313);
    let moon = CLEAR_MOON_KM;
    let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    assert!(srp_acceleration_with_bodies(sat, sun, Some(moon), model).is_ok());
    let refused = srp_position_partials_with_bodies(sat, sun, Some(moon), model);
    assert!(
        matches!(refused, Err(PropagationError::NumericalFailure(_))),
        "{refused:?}"
    );
}

#[test]
fn position_partials_follow_the_oblate_limb_off_the_meridian() {
    let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let sun = Vector3::new(1.2e8, -8.0e7, -3.5e7);
    let u = sun.normalize();
    // 2 million km behind the Earth, 3e-4 rad (600 km) off the shadow axis on
    // a pass off every meridian plane. The Earth's apparent radius there is
    // about 3.2e-3 rad and the Sun's 4.6e-3 rad, so the Earth's disc lies
    // inside the Sun's, and the oblate limb angle changes as the plane
    // through the satellite, the Earth's center and the Sun turns, with the
    // satellite and with the Sun direction seen from it.
    let across = u.cross(&Vector3::new(0.3, 0.2, 1.0)).normalize();
    let theta = std::f64::consts::PI - 3.0e-4;
    let sat = (u * libm::cos(theta) + across * libm::sin(theta)) * 2.0e6;
    let u_sun = (sun - sat).normalize();
    let a = sun_radius_seen_from(sat, sun);
    for (case, moon_angle) in [("moon clear", 0.5), ("moon over the sun", 0.9 * a)] {
        let moon = sat + tilted(u_sun, moon_angle, 2.0) * 384_400.0;
        let fraction = shadow_fraction_with_model_and_moon(
            as_array(sat),
            as_array(sun),
            as_array(moon),
            EarthShadowModel::Wgs84Oblate,
        )
        .expect("fraction");
        assert!(fraction > 0.3 && fraction < 1.0, "{case}: {fraction}");
        // The limb plane turns over a few hundred km of satellite motion
        // here; a 10 m step stays well inside that.
        assert_partials_match_differences(case, model, sat, sun, Some(moon), 1.0e-2);
    }
}

#[test]
fn parameter_partial_is_the_acceleration_per_unit_cr() {
    let ctx = PropagationContext::default();
    let state = CartesianState::new(
        831_902_400.0,
        [15_000.0, 20_000.0, 9_000.0],
        [-2.0, 1.5, 3.0],
    );
    let base = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    for model in all_shadow_configs(base) {
        assert_eq!(model.parameters(), vec![ForceParameter::new("srp.cr", 1.3)]);
        let partials = model
            .parameter_partials(&[1.7], &state, &ctx)
            .expect("partials");
        let at_one = model
            .acceleration_with_parameters(&[1.0], &state, &ctx)
            .expect("acceleration");
        assert_eq!(partials.len(), 1);
        assert!((partials[0] - at_one).norm() <= 4.0 * f64::EPSILON * at_one.norm());
        let at_estimate = model
            .acceleration_with_parameters(&[1.7], &state, &ctx)
            .expect("acceleration");
        // Linear in C_R: exact up to the rounding of the product.
        assert!((at_estimate - at_one * 1.7).norm() <= 4.0 * f64::EPSILON * at_estimate.norm());
        assert_eq!(
            model.acceleration(&state, &ctx).expect("held value"),
            model
                .acceleration_with_parameters(&[1.3], &state, &ctx)
                .expect("explicit value")
        );
    }
}

#[test]
fn every_evaluation_uses_the_moon_when_it_is_enabled() {
    // 2024-04-08 18:17:00 TDB, near the greatest solar eclipse. A LEO
    // satellite on the sunward side, displaced from the Moon's shadow axis
    // so the Moon covers part of the Sun.
    let epoch = 765_872_220.0;
    let (sun, moon) = gcrs_sun_moon_km(epoch);
    let axis = (moon - sun).normalize();
    let closest = moon - axis * moon.dot(&axis);
    let on_axis = closest - axis * (7_000.0_f64.powi(2) - closest.norm_squared()).sqrt();
    let side = axis.cross(&Vector3::z()).normalize();
    let sat = on_axis + side * 2_000.0;
    let fraction = shadow_fraction_with_model_and_moon(
        as_array(sat),
        as_array(sun),
        as_array(moon),
        EarthShadowModel::Spherical,
    )
    .expect("fraction");
    assert!(fraction > 0.05 && fraction < 0.95, "fraction {fraction}");

    let state = CartesianState::new(epoch, sat.into(), [0.0, 7.5, 0.0]);
    let ctx = PropagationContext::default();
    let with_moon = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let earth_only_disc = with_moon.with_moon_eclipse(false);
    let earth_only_conical = with_moon.with_shadow_function(ShadowFunction::Conical);

    let expected = srp_acceleration_with_bodies(sat, sun, Some(moon), with_moon).expect("explicit");
    assert_eq!(
        with_moon.acceleration(&state, &ctx).expect("accel"),
        expected
    );
    assert_eq!(
        with_moon
            .acceleration_with_parameters(&[1.3], &state, &ctx)
            .expect("accel"),
        expected
    );
    let expected_cr_partial =
        srp_parameter_partial_with_bodies(sat, sun, Some(moon), with_moon).expect("explicit");
    assert_eq!(
        with_moon
            .parameter_partials(&[1.3], &state, &ctx)
            .expect("partials"),
        vec![expected_cr_partial]
    );
    let per_unit_cr = srp_acceleration_with_bodies(
        sat,
        sun,
        Some(moon),
        SolarRadiationPressure {
            cr: 1.0,
            ..with_moon
        },
    )
    .expect("explicit");
    assert!(
        (expected_cr_partial - per_unit_cr).norm() <= 1.0e-13 * per_unit_cr.norm(),
        "C_R partial {expected_cr_partial:?} vs double {per_unit_cr:?}"
    );
    let expected_partials =
        srp_position_partials_with_bodies(sat, sun, Some(moon), with_moon).expect("explicit");
    assert_eq!(
        with_moon
            .acceleration_partials(&[1.3], &state, &ctx)
            .expect("partials")
            .wrt_position,
        expected_partials
    );

    for earth_only in [earth_only_disc, earth_only_conical] {
        let earth_only_accel = earth_only.acceleration(&state, &ctx).expect("accel");
        assert!(earth_only_accel.norm() > expected.norm() * 1.05);
        let earth_only_partials = earth_only
            .acceleration_partials(&[1.3], &state, &ctx)
            .expect("partials")
            .wrt_position;
        assert!((earth_only_partials - expected_partials).norm() > 0.5 * expected_partials.norm());
    }
}

/// A sunlit GNSS state at `EPOCH`: on the Earth-Sun line, sunward.
fn sunlit_state(epoch: f64) -> CartesianState {
    let (sun, _moon) = gcrs_sun_moon_km(epoch);
    let position = sun / sun.norm() * 26_560.0;
    CartesianState::new(epoch, position.into(), [0.0, 3.0, 2.0])
}

#[test]
fn an_acceleration_that_overflows_is_a_numerical_failure() {
    // Finite parameters whose product, P_sr C_R A/m = 1e400, overflows.
    let model = SolarRadiationPressure::with_pressure(1.0, 1.0e200, 1.0e200, AU_KM)
        .expect("finite parameters");
    let ctx = PropagationContext::default();
    let state = sunlit_state(831_902_400.0);
    let failed = |result: Result<Vector3<f64>, PropagationError>| {
        matches!(result, Err(PropagationError::NumericalFailure(_)))
    };
    for model in all_shadow_configs(model) {
        assert!(failed(model.acceleration(&state, &ctx)));
        assert!(failed(model.acceleration_with_parameters(
            &[1.0],
            &state,
            &ctx
        )));
        let partials = model.parameter_partials(&[1.0], &state, &ctx);
        assert!(
            matches!(partials, Err(PropagationError::NumericalFailure(_))),
            "{partials:?}"
        );
    }
}

/// Overflow-free Euclidean norm of a `Vector3<f64>`.
fn scaled_norm_vec3(v: Vector3<f64>) -> f64 {
    ScaledVec3::from_vec3(Vec3::new(v.x, v.y, v.z))
        .norm()
        .map_or(0.0, ScaledF64::to_f64)
}

/// Overflow-free relative Euclidean error `|got - expected| / |expected|`.
/// Returns `+Inf` if any entry of `got` or `expected` is non-finite.
fn scaled_rel_err_vec3(got: Vector3<f64>, expected: Vector3<f64>) -> f64 {
    if !got.iter().all(|c| c.is_finite()) || !expected.iter().all(|c| c.is_finite()) {
        return f64::INFINITY;
    }
    let g = ScaledVec3::from_vec3(Vec3::new(got.x, got.y, got.z));
    let e = ScaledVec3::from_vec3(Vec3::new(expected.x, expected.y, expected.z));
    let Some(den) = e.norm() else {
        return g.norm().map_or(0.0, ScaledF64::to_f64);
    };
    g.sub(e).norm().map_or(0.0, |num| num.div(den).to_f64())
}

/// Overflow-free relative Frobenius error `||got - expected||_F / ||expected||_F`.
/// Returns `+Inf` if any entry of `got` or `expected` is non-finite.
fn scaled_rel_err_mat3(got: Matrix3<f64>, expected: Matrix3<f64>) -> f64 {
    if !got.iter().all(|c| c.is_finite()) || !expected.iter().all(|c| c.is_finite()) {
        return f64::INFINITY;
    }
    let norm_mat = |m: Matrix3<f64>| -> Option<ScaledF64> {
        let mut e_max = i32::MIN;
        for &c in m.iter() {
            if c != 0.0 && c.is_finite() {
                e_max = e_max.max(crate::astro::events::eclipse::ad::exact_exponent(c));
            }
        }
        if e_max == i32::MIN {
            return None;
        }
        let mut sum_sq = 0.0;
        for &c in m.iter() {
            if c != 0.0 {
                let s = libm::ldexp(c, -e_max);
                sum_sq += s * s;
            }
        }
        Some(ScaledF64::from_scaled(fast_math::sqrt(sum_sq), e_max))
    };
    let Some(den) = norm_mat(expected) else {
        return norm_mat(got).map_or(0.0, ScaledF64::to_f64);
    };
    let mut diff_sum_sq = 0.0;
    let mut e_max = i32::MIN;
    let mut diffs = [ScaledF64::ZERO; 9];
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let d = ScaledF64::from_f64(g).sub(ScaledF64::from_f64(e));
        diffs[idx] = d;
        if !d.is_zero() {
            e_max = e_max.max(d.e);
        }
    }
    if e_max == i32::MIN {
        return 0.0;
    }
    for d in diffs {
        if !d.is_zero() {
            let shift = d.e - e_max;
            if shift >= -1080 {
                let s = libm::ldexp(d.m, shift);
                diff_sum_sq += s * s;
            }
        }
    }
    let num = ScaledF64::from_scaled(fast_math::sqrt(diff_sum_sq), e_max);
    num.div(den).to_f64()
}

fn max_abs_vec3(v: Vector3<f64>) -> f64 {
    v.x.abs().max(v.y.abs()).max(v.z.abs())
}

fn max_abs_mat3(m: Matrix3<f64>) -> f64 {
    m.iter().fold(0.0_f64, |acc, &c| acc.max(c.abs()))
}

#[test]
fn distant_position_out_of_eclipse_range_and_coincident_sun_are_refused() {
    let base = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let ctx = PropagationContext::default();
    let state = CartesianState::new(831_902_400.0, [1.0e200, 0.0, 0.0], [0.0; 3]);
    // |sat| = 1e200 km is outside both shadow functions' input domain, so all
    // three entry points consistently return ForceModelFailure("eclipse: ...").
    for model in all_shadow_configs(base) {
        let acceleration = model.acceleration(&state, &ctx);
        assert!(
            matches!(acceleration, Err(PropagationError::ForceModelFailure(_))),
            "{model:?}: {acceleration:?}"
        );
        let partials = model.acceleration_partials(&[1.3], &state, &ctx);
        assert!(
            matches!(partials, Err(PropagationError::ForceModelFailure(_))),
            "{model:?}: {partials:?}"
        );
        let cr_partials = model.parameter_partials(&[1.3], &state, &ctx);
        assert!(
            matches!(cr_partials, Err(PropagationError::ForceModelFailure(_))),
            "{model:?}: {cr_partials:?}"
        );
    }

    // A zero satellite-Sun distance remains a typed NumericalFailure because the
    // Sun-satellite direction is undefined.
    let (sun, moon) = gcrs_sun_moon_km(831_902_400.0);
    for model in all_shadow_configs(base) {
        let moon_opt = model.moon_eclipse_enabled().then_some(moon);
        assert!(matches!(
            srp_acceleration_with_bodies(sun, sun, moon_opt, model),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            srp_position_partials_with_bodies(sun, sun, moon_opt, model),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            srp_parameter_partial_with_bodies(sun, sun, moon_opt, model),
            Err(PropagationError::NumericalFailure(_))
        ));
    }
}

#[test]
fn deep_umbra_returns_zero_when_k_ref_or_factor_overflows() {
    let sat = Vector3::new(-7_000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let moon = CLEAR_MOON_KM;

    // 1. k_ref = p_ref * d_ref^2 overflows to +Inf.
    let inf_k_ref =
        SolarRadiationPressure::with_pressure(1.0, 1.0, 1.0e290, AU_KM).expect("finite parameters");
    // 2. factor = (A/m) * C_R overflows to +Inf.
    let inf_factor = SolarRadiationPressure::with_pressure(1.0e200, 1.0e200, 4.56e-6, AU_KM)
        .expect("finite parameters");
    for base in [inf_k_ref, inf_factor] {
        for model in all_shadow_configs(base) {
            let moon_opt = model.moon_eclipse_enabled().then_some(moon);
            let accel = srp_acceleration_with_bodies(sat, sun, moon_opt, model);
            let partials = srp_position_partials_with_bodies(sat, sun, moon_opt, model);
            let cr_partial = srp_parameter_partial_with_bodies(sat, sun, moon_opt, model);
            assert_eq!(
                accel.expect("umbra accel"),
                Vector3::zeros(),
                "expected zero umbra accel for {model:?}"
            );
            assert_eq!(
                partials.expect("umbra partials"),
                Matrix3::zeros(),
                "expected zero umbra partials for {model:?}"
            );
            assert_eq!(
                cr_partial.expect("umbra cr partial"),
                Vector3::zeros(),
                "expected zero umbra cr partial for {model:?}"
            );
        }
    }

    // 3. When both k_ref (~2.24e222) and factor (1e200) are finite, deep umbra
    //    returns zero while penumbra overflows both acceleration and partials.
    let finite_scale = SolarRadiationPressure::with_pressure(1.0, 1.0e200, 1.0e200, AU_KM)
        .expect("finite parameters");
    for model in all_shadow_configs(finite_scale) {
        let moon_opt = model.moon_eclipse_enabled().then_some(moon);
        assert_eq!(
            srp_acceleration_with_bodies(sat, sun, moon_opt, model).expect("deep umbra accel"),
            Vector3::zeros()
        );
        assert_eq!(
            srp_position_partials_with_bodies(sat, sun, moon_opt, model)
                .expect("deep umbra partials"),
            Matrix3::zeros()
        );
    }
    let penumbra_sat = Vector3::new(-26_000.0, 6_000.0, 2_100.0);
    let overflowed = srp_position_partials_with_bodies(penumbra_sat, sun, Some(moon), finite_scale)
        .map(AccelerationPartials::position_only)
        .and_then(finite_partials);
    assert!(
        matches!(overflowed, Err(PropagationError::NumericalFailure(_))),
        "{overflowed:?}"
    );
}

#[test]
fn position_partials_succeed_when_acceleration_value_overflows_but_partials_are_finite() {
    // Here k_ref = 2.238e222, flux.x.value = -1.0e200, flux.x.grad[0] = -1.337e189,
    // and factor = 1.0e115, so the acceleration value (~ -1.0e312 km/s^2) overflows
    // binary64 while every position partial entry (~ -1.337e304 1/s^2) is finite.
    let base = SolarRadiationPressure::with_pressure(1.0, 1.0e115, 1.0e200, AU_KM)
        .expect("finite parameters");
    let sat = Vector3::new(7_000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let moon = CLEAR_MOON_KM;
    let r_km = AU_KM - 7_000.0;
    let ratio_au_r = ScaledF64::from_f64(AU_KM).div(ScaledF64::from_f64(r_km));
    let coeff = ScaledF64::from_f64(1.0e115)
        .mul(ScaledF64::from_f64(1.0e200))
        .mul(ratio_au_r)
        .mul(ratio_au_r)
        .div(ScaledF64::from_f64(r_km))
        .div(ScaledF64::from_f64(M_PER_KM));
    let expected = Matrix3::from_diagonal(&Vector3::new(
        coeff.mul(ScaledF64::from_f64(-2.0)).to_f64(),
        coeff.to_f64(),
        coeff.to_f64(),
    ));
    for model in all_shadow_configs(base) {
        let moon_opt = model.moon_eclipse_enabled().then_some(moon);
        assert!(matches!(
            srp_acceleration_with_bodies(sat, sun, moon_opt, model),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            srp_parameter_partial_with_bodies(sat, sun, moon_opt, model),
            Err(PropagationError::NumericalFailure(_))
        ));
        let partials = srp_position_partials_with_bodies(sat, sun, moon_opt, model)
            .expect("finite position partials when accel overflows");
        let rel = scaled_rel_err_mat3(partials, expected);
        assert!(
            rel <= 8.0 * f64::EPSILON,
            "{model:?}: rel={rel:e} partials={partials:?} expected={expected:?}"
        );
    }
}

#[test]
fn estimated_cr_accepts_any_finite_value_and_refuses_others() {
    let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let ctx = PropagationContext::default();
    let state = CartesianState::new(
        831_902_400.0,
        [15_000.0, 20_000.0, 9_000.0],
        [-2.0, 1.5, 3.0],
    );
    assert!(model
        .acceleration_with_parameters(&[-0.5], &state, &ctx)
        .is_ok());
    for bad in [[f64::NAN].as_slice(), &[f64::INFINITY], &[], &[1.0, 1.0]] {
        assert!(matches!(
            model.acceleration_with_parameters(bad, &state, &ctx),
            Err(PropagationError::InvalidInput(_))
        ));
        assert!(matches!(
            model.acceleration_partials(bad, &state, &ctx),
            Err(PropagationError::InvalidInput(_))
        ));
        assert!(matches!(
            model.parameter_partials(bad, &state, &ctx),
            Err(PropagationError::InvalidInput(_))
        ));
    }
}

#[test]
fn acceleration_and_partials_equal_caller_vector_paths_with_gcrs_sun_and_moon() {
    let ctx = PropagationContext::default();
    let base = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let models = all_shadow_configs(base);
    for epoch in epochs_1990_to_2050() {
        let (sun, moon) = gcrs_sun_moon_km(epoch);
        let s_hat = sun.normalize();
        let across = s_hat.cross(&Vector3::z()).normalize();
        // Sunlit, Earth penumbra, GEO behind the Earth, and a GNSS orbit.
        let positions = [
            s_hat * 7_000.0,
            -s_hat * 7_000.0 + across * 6_365.0,
            -s_hat * 42_164.0 + across * 1_000.0,
            Vector3::new(26_560.0, 1_200.0, -3_400.0),
        ];
        for position in positions {
            let state = CartesianState::new(epoch, position.into(), [0.0, 7.5, 0.0]);
            for model in models {
                let from_state = model.acceleration(&state, &ctx).expect("acceleration");
                let from_vectors = model
                    .acceleration_with_sun_moon(position, sun, moon)
                    .expect("caller vectors");
                assert_eq!(
                    from_state, from_vectors,
                    "epoch {epoch} position {position:?}"
                );
                let with_cr = model
                    .acceleration_with_parameters(&[1.7], &state, &ctx)
                    .expect("acceleration_with_parameters");
                let scaled_model = SolarRadiationPressure { cr: 1.7, ..model };
                let scaled_from_vectors = scaled_model
                    .acceleration_with_sun_moon(position, sun, moon)
                    .expect("caller vectors with C_R");
                assert_eq!(
                    with_cr, scaled_from_vectors,
                    "acceleration_with_parameters epoch {epoch} position {position:?}"
                );
                let moon_opt = model.moon_eclipse_enabled().then_some(moon);
                let param_partials = model
                    .parameter_partials(&[1.7], &state, &ctx)
                    .expect("parameter_partials");
                let cr_partial_from_vectors =
                    srp_parameter_partial_with_bodies(position, sun, moon_opt, scaled_model)
                        .expect("caller-vector C_R partial");
                assert_eq!(
                    param_partials,
                    vec![cr_partial_from_vectors],
                    "parameter_partials epoch {epoch} position {position:?}"
                );
                let state_partials = model
                    .acceleration_partials(&[1.7], &state, &ctx)
                    .expect("acceleration_partials");
                let partials_from_vectors =
                    srp_position_partials_with_bodies(position, sun, moon_opt, scaled_model)
                        .expect("caller-vector position partials");
                assert_eq!(
                    state_partials.wrt_position, partials_from_vectors,
                    "acceleration_partials epoch {epoch} position {position:?}"
                );
            }
        }
    }
}

#[test]
fn acceleration_turns_away_from_the_mean_of_date_sun_by_the_precession() {
    // Read as GCRS, the mean of date Sun is off by the precession since
    // J2000.0 plus the frame bias, 0.376 deg at 2026-11-23; SRP on a sunlit
    // satellite turns by that angle between the two.
    let epoch = crate::astro::time::civil::j2000_seconds(2026, 11, 23, 0, 0, 0.0);
    let (sun, moon) = gcrs_sun_moon_km(epoch);
    let (mean_sun, mean_moon) = mean_of_date_sun_moon_km(epoch);
    let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let position = sun.normalize() * 7_000.0;
    let state = CartesianState::new(epoch, position.into(), [0.0, 7.5, 0.0]);
    let used = model
        .acceleration(&state, &PropagationContext::default())
        .expect("acceleration");
    let with_mean = model
        .acceleration_with_sun_moon(position, mean_sun, mean_moon)
        .expect("mean of date Sun");
    let angle_deg = used.angle(&with_mean).to_degrees();
    assert!((0.37..0.38).contains(&angle_deg), "{angle_deg} deg");
    assert_eq!(
        used,
        model
            .acceleration_with_sun_moon(position, sun, moon)
            .expect("GCRS")
    );
}

#[cfg(sidereon_repo_tests)]
#[test]
fn internal_lighting_srp_and_partials_match_orekit_zero_ulp() {
    use crate::astro::events::eclipse::orekit::{
        earth_lighting_ratio_m_double, earth_lighting_ratio_m_field,
        earth_moon_lighting_ratio_m_double, earth_moon_lighting_ratio_m_field,
        moon_lighting_ratio_m_double, moon_lighting_ratio_m_field,
    };

    #[derive(serde::Deserialize)]
    struct EarthCase {
        name: String,
        earth_flattening: f64,
        sat_km_bits: [String; 3],
        sun_km_bits: [String; 3],
        lighting_double_bits: String,
        accel_double_bits: [String; 3],
        lighting_field_val_bits: String,
        lighting_field_grad_bits: [String; 3],
        accel_field_val_bits: [String; 3],
        accel_field_partials_bits: [[String; 3]; 3],
        clear_moon_km_bits: [String; 3],
        clear_moon_accel_field_val_bits: [String; 3],
        clear_moon_accel_field_partials_bits: [[String; 3]; 3],
        accel_field_wrt_cr_bits: [String; 3],
        clear_moon_accel_field_wrt_cr_bits: [String; 3],
    }

    #[derive(serde::Deserialize)]
    struct MoonCase {
        name: String,
        sat_km_bits: [String; 3],
        sun_km_bits: [String; 3],
        moon_km_bits: [String; 3],
        lighting_double_bits: String,
        accel_double_bits: [String; 3],
        lighting_field_val_bits: String,
        lighting_field_grad_bits: [String; 3],
        accel_field_val_bits: [String; 3],
        accel_field_partials_bits: [[String; 3]; 3],
        accel_field_wrt_cr_bits: [String; 3],
    }

    #[derive(serde::Deserialize)]
    struct EarthMoonCase {
        name: String,
        earth_flattening: f64,
        sat_km_bits: [String; 3],
        sun_km_bits: [String; 3],
        moon_km_bits: [String; 3],
        orekit_subtracts_mutual: bool,
        orekit_lighting_double_bits: String,
        orekit_accel_double_bits: [String; 3],
        orekit_lighting_field_val_bits: String,
        orekit_lighting_field_grad_bits: [String; 3],
        orekit_accel_field_val_bits: [String; 3],
        orekit_accel_field_partials_bits: [[String; 3]; 3],
        union_lighting_double_bits: String,
        union_accel_double_bits: [String; 3],
        union_lighting_field_val_bits: String,
        union_lighting_field_grad_bits: [String; 3],
        union_accel_field_val_bits: [String; 3],
        union_accel_field_partials_bits: [[String; 3]; 3],
        orekit_accel_field_wrt_cr_bits: [String; 3],
        union_accel_field_wrt_cr_bits: [String; 3],
    }

    #[derive(serde::Deserialize)]
    struct Fixture {
        d_ref_km_bits: String,
        p_ref_n_m2_bits: String,
        cr_bits: String,
        area_to_mass_m2_kg_bits: String,
        earth_cases: Vec<EarthCase>,
        moon_cases: Vec<MoonCase>,
        earth_moon_cases: Vec<EarthMoonCase>,
    }

    let parse = |s: &str| {
        f64::from_bits(u64::from_str_radix(s.strip_prefix("0x").expect("0x"), 16).expect("hex"))
    };
    let parse_v = |a: &[String; 3]| Vector3::new(parse(&a[0]), parse(&a[1]), parse(&a[2]));
    let to_m = |v: Vector3<f64>| Vec3::new(v.x * M_PER_KM, v.y * M_PER_KM, v.z * M_PER_KM);
    let check = |label: &str, got: f64, exp: f64, mismatches: &mut Vec<String>| {
        if got.to_bits() != exp.to_bits() {
            let ulp = got.to_bits().abs_diff(exp.to_bits());
            mismatches.push(format!(
                "{label}: got {got:.17e} (0x{:016x}) != exp {exp:.17e} (0x{:016x}) [diff={:.3e}, raw_ulp={ulp}]",
                got.to_bits(),
                exp.to_bits(),
                got - exp
            ));
        }
    };

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/fixtures/eclipse/orekit_shadow_srp_oracle.json"
    ))
    .expect("parse fixture");
    let base_srp = SolarRadiationPressure::with_pressure(
        parse(&fixture.cr_bits),
        parse(&fixture.area_to_mass_m2_kg_bits),
        parse(&fixture.p_ref_n_m2_bits),
        parse(&fixture.d_ref_km_bits),
    )
    .expect("valid SRP")
    .with_moon_eclipse(true);

    let mut mismatches = Vec::new();
    let mut checks = 0usize;

    for c in &fixture.earth_cases {
        let sat = parse_v(&c.sat_km_bits);
        let sun = parse_v(&c.sun_km_bits);
        let model = if c.earth_flattening == 0.0 {
            EarthShadowModel::Spherical
        } else {
            EarthShadowModel::Wgs84Oblate
        };
        let ld = earth_lighting_ratio_m_double(to_m(sat), to_m(sun), model).expect("earth ld");
        let lf = earth_lighting_ratio_m_field(to_m(sat), to_m(sun), model).expect("earth lf");
        let ad = srp_acceleration_m_s2(sat, sun, ld, base_srp).expect("earth ad");
        let af = srp_acceleration_field_m_s2(sat, sun, lf, base_srp).expect("earth af");

        check(
            &format!("earth {} ld", c.name),
            ld,
            parse(&c.lighting_double_bits),
            &mut mismatches,
        );
        check(
            &format!("earth {} lf.v", c.name),
            lf.value,
            parse(&c.lighting_field_val_bits),
            &mut mismatches,
        );
        checks += 2;
        for i in 0..3 {
            check(
                &format!("earth {} lf.g[{i}]", c.name),
                lf.grad[i],
                parse(&c.lighting_field_grad_bits[i]),
                &mut mismatches,
            );
            let ad_i = [ad.x, ad.y, ad.z][i];
            let af_i = [af.x, af.y, af.z][i];
            check(
                &format!("earth {} ad[{i}]", c.name),
                ad_i,
                parse(&c.accel_double_bits[i]),
                &mut mismatches,
            );
            check(
                &format!("earth {} af.v[{i}]", c.name),
                af_i.value,
                parse(&c.accel_field_val_bits[i]),
                &mut mismatches,
            );
            checks += 3;
            for j in 0..3 {
                check(
                    &format!("earth {} af.p[{i}][{j}]", c.name),
                    af_i.grad[j],
                    parse(&c.accel_field_partials_bits[i][j]),
                    &mut mismatches,
                );
                checks += 1;
            }
        }

        // Also assert the Earth-only apparent-disc path (`with_moon_eclipse(false)`)
        // and the clear-Moon SRP field values and position partials through the
        // SRP paths at 0 ULP for all 76 Earth cases.
        let earth_only_model = base_srp
            .with_earth_shadow_model(model)
            .with_moon_eclipse(false);
        let earth_only_accel =
            srp_acceleration_with_bodies(sat, sun, None, earth_only_model).expect("earth_only ad");
        let earth_only_partials =
            srp_position_partials_with_bodies(sat, sun, None, earth_only_model)
                .expect("earth_only partials");
        let earth_only_cr = srp_parameter_partial_with_bodies(sat, sun, None, earth_only_model)
            .expect("earth_only cr partial");
        let earth_cr_m =
            srp_acceleration_field_cr_m_s2(sat, sun, lf.value, base_srp).expect("earth cr m");
        for i in 0..3 {
            let cr_m_i = [
                earth_cr_m.x.grad[0],
                earth_cr_m.y.grad[0],
                earth_cr_m.z.grad[0],
            ][i];
            check(
                &format!("earth {} cr_m[{i}]", c.name),
                cr_m_i,
                parse(&c.accel_field_wrt_cr_bits[i]),
                &mut mismatches,
            );
            check(
                &format!("earth {} earth_only cr[{i}]", c.name),
                earth_only_cr[i],
                parse(&c.accel_field_wrt_cr_bits[i]) / M_PER_KM,
                &mut mismatches,
            );
            check(
                &format!("earth {} earth_only ad[{i}]", c.name),
                earth_only_accel[i],
                parse(&c.accel_double_bits[i]) / M_PER_KM,
                &mut mismatches,
            );
            checks += 3;
            for j in 0..3 {
                check(
                    &format!("earth {} earth_only p[{i}][{j}]", c.name),
                    earth_only_partials[(i, j)],
                    parse(&c.accel_field_partials_bits[i][j]),
                    &mut mismatches,
                );
                checks += 1;
            }
        }

        let clear_moon = parse_v(&c.clear_moon_km_bits);
        let srp_model = base_srp.with_earth_shadow_model(model);
        let clear_lf =
            lighting_ratio_field_of(srp_model, sat, sun, Some(clear_moon)).expect("clear_moon lf");
        let clear_af =
            srp_acceleration_field_m_s2(sat, sun, clear_lf, srp_model).expect("clear_moon af");
        let clear_partials =
            srp_position_partials_with_bodies(sat, sun, Some(clear_moon), srp_model)
                .expect("clear_moon partials");
        let clear_cr = srp_parameter_partial_with_bodies(sat, sun, Some(clear_moon), srp_model)
            .expect("clear_moon cr partial");
        for i in 0..3 {
            let clear_af_i = [clear_af.x, clear_af.y, clear_af.z][i];
            check(
                &format!("earth {} clear_moon af.v[{i}]", c.name),
                clear_af_i.value,
                parse(&c.clear_moon_accel_field_val_bits[i]),
                &mut mismatches,
            );
            check(
                &format!("earth {} clear_moon cr[{i}]", c.name),
                clear_cr[i],
                parse(&c.clear_moon_accel_field_wrt_cr_bits[i]) / M_PER_KM,
                &mut mismatches,
            );
            checks += 2;
            for j in 0..3 {
                check(
                    &format!("earth {} clear_moon af.p[{i}][{j}]", c.name),
                    clear_partials[(i, j)],
                    parse(&c.clear_moon_accel_field_partials_bits[i][j]),
                    &mut mismatches,
                );
                checks += 1;
            }
        }
    }

    for c in &fixture.moon_cases {
        let sat = parse_v(&c.sat_km_bits);
        let sun = parse_v(&c.sun_km_bits);
        let moon = parse_v(&c.moon_km_bits);
        let ld = moon_lighting_ratio_m_double(to_m(sat), to_m(sun), to_m(moon)).expect("moon ld");
        let lf = moon_lighting_ratio_m_field(to_m(sat), to_m(sun), to_m(moon)).expect("moon lf");
        let ad = srp_acceleration_m_s2(sat, sun, ld, base_srp).expect("moon ad");
        let af = srp_acceleration_field_m_s2(sat, sun, lf, base_srp).expect("moon af");
        let cr_m = srp_acceleration_field_cr_m_s2(sat, sun, lf.value, base_srp).expect("moon cr");

        check(
            &format!("moon {} ld", c.name),
            ld,
            parse(&c.lighting_double_bits),
            &mut mismatches,
        );
        check(
            &format!("moon {} lf.v", c.name),
            lf.value,
            parse(&c.lighting_field_val_bits),
            &mut mismatches,
        );
        checks += 2;
        for i in 0..3 {
            check(
                &format!("moon {} lf.g[{i}]", c.name),
                lf.grad[i],
                parse(&c.lighting_field_grad_bits[i]),
                &mut mismatches,
            );
            let ad_i = [ad.x, ad.y, ad.z][i];
            let af_i = [af.x, af.y, af.z][i];
            let cr_i = [cr_m.x.grad[0], cr_m.y.grad[0], cr_m.z.grad[0]][i];
            check(
                &format!("moon {} ad[{i}]", c.name),
                ad_i,
                parse(&c.accel_double_bits[i]),
                &mut mismatches,
            );
            check(
                &format!("moon {} af.v[{i}]", c.name),
                af_i.value,
                parse(&c.accel_field_val_bits[i]),
                &mut mismatches,
            );
            check(
                &format!("moon {} cr[{i}]", c.name),
                cr_i,
                parse(&c.accel_field_wrt_cr_bits[i]),
                &mut mismatches,
            );
            checks += 4;
            for j in 0..3 {
                check(
                    &format!("moon {} af.p[{i}][{j}]", c.name),
                    af_i.grad[j],
                    parse(&c.accel_field_partials_bits[i][j]),
                    &mut mismatches,
                );
                checks += 1;
            }
        }
    }

    for c in &fixture.earth_moon_cases {
        let sat = parse_v(&c.sat_km_bits);
        let sun = parse_v(&c.sun_km_bits);
        let moon = parse_v(&c.moon_km_bits);
        let model = if c.earth_flattening == 0.0 {
            EarthShadowModel::Spherical
        } else {
            EarthShadowModel::Wgs84Oblate
        };
        let srp_model = base_srp.with_earth_shadow_model(model);
        let ld = earth_moon_lighting_ratio_m_double(to_m(sat), to_m(sun), to_m(moon), model)
            .expect("em ld");
        let lf = earth_moon_lighting_ratio_m_field(to_m(sat), to_m(sun), to_m(moon), model)
            .expect("em lf");
        let ad = srp_acceleration_m_s2(sat, sun, ld, base_srp).expect("em ad");
        let af = srp_acceleration_field_m_s2(sat, sun, lf, base_srp).expect("em af");
        let cr_m = srp_acceleration_field_cr_m_s2(sat, sun, lf.value, base_srp).expect("em cr m");
        let cr_km =
            srp_parameter_partial_with_bodies(sat, sun, Some(moon), srp_model).expect("em cr km");

        check(
            &format!("em {} union ld", c.name),
            ld,
            parse(&c.union_lighting_double_bits),
            &mut mismatches,
        );
        check(
            &format!("em {} union lf.v", c.name),
            lf.value,
            parse(&c.union_lighting_field_val_bits),
            &mut mismatches,
        );
        checks += 2;
        for i in 0..3 {
            check(
                &format!("em {} union lf.g[{i}]", c.name),
                lf.grad[i],
                parse(&c.union_lighting_field_grad_bits[i]),
                &mut mismatches,
            );
            let ad_i = [ad.x, ad.y, ad.z][i];
            let af_i = [af.x, af.y, af.z][i];
            let cr_m_i = [cr_m.x.grad[0], cr_m.y.grad[0], cr_m.z.grad[0]][i];
            check(
                &format!("em {} union ad[{i}]", c.name),
                ad_i,
                parse(&c.union_accel_double_bits[i]),
                &mut mismatches,
            );
            check(
                &format!("em {} union af.v[{i}]", c.name),
                af_i.value,
                parse(&c.union_accel_field_val_bits[i]),
                &mut mismatches,
            );
            check(
                &format!("em {} union cr_m[{i}]", c.name),
                cr_m_i,
                parse(&c.union_accel_field_wrt_cr_bits[i]),
                &mut mismatches,
            );
            check(
                &format!("em {} union cr_km[{i}]", c.name),
                cr_km[i],
                parse(&c.union_accel_field_wrt_cr_bits[i]) / M_PER_KM,
                &mut mismatches,
            );
            checks += 5;
            for j in 0..3 {
                check(
                    &format!("em {} union af.p[{i}][{j}]", c.name),
                    af_i.grad[j],
                    parse(&c.union_accel_field_partials_bits[i][j]),
                    &mut mismatches,
                );
                checks += 1;
            }
        }

        if !c.orekit_subtracts_mutual {
            check(
                &format!("em {} orekit ld", c.name),
                ld,
                parse(&c.orekit_lighting_double_bits),
                &mut mismatches,
            );
            check(
                &format!("em {} orekit lf.v", c.name),
                lf.value,
                parse(&c.orekit_lighting_field_val_bits),
                &mut mismatches,
            );
            checks += 2;
            for i in 0..3 {
                check(
                    &format!("em {} orekit lf.g[{i}]", c.name),
                    lf.grad[i],
                    parse(&c.orekit_lighting_field_grad_bits[i]),
                    &mut mismatches,
                );
                let ad_i = [ad.x, ad.y, ad.z][i];
                let af_i = [af.x, af.y, af.z][i];
                let cr_m_i = [cr_m.x.grad[0], cr_m.y.grad[0], cr_m.z.grad[0]][i];
                check(
                    &format!("em {} orekit ad[{i}]", c.name),
                    ad_i,
                    parse(&c.orekit_accel_double_bits[i]),
                    &mut mismatches,
                );
                check(
                    &format!("em {} orekit af.v[{i}]", c.name),
                    af_i.value,
                    parse(&c.orekit_accel_field_val_bits[i]),
                    &mut mismatches,
                );
                check(
                    &format!("em {} orekit cr_m[{i}]", c.name),
                    cr_m_i,
                    parse(&c.orekit_accel_field_wrt_cr_bits[i]),
                    &mut mismatches,
                );
                check(
                    &format!("em {} orekit cr_km[{i}]", c.name),
                    cr_km[i],
                    parse(&c.orekit_accel_field_wrt_cr_bits[i]) / M_PER_KM,
                    &mut mismatches,
                );
                checks += 5;
                for j in 0..3 {
                    check(
                        &format!("em {} orekit af.p[{i}][{j}]", c.name),
                        af_i.grad[j],
                        parse(&c.orekit_accel_field_partials_bits[i][j]),
                        &mut mismatches,
                    );
                    checks += 1;
                }
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} mismatches out of {checks} bit-exact checks:\n{}",
        mismatches.len(),
        mismatches
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn penumbra_sweep_measures_new_vs_old_default_at_leo_gnss_and_geo() {
    let new_default = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let old_default = new_default
        .with_shadow_function(ShadowFunction::Conical)
        .with_earth_shadow_model(EarthShadowModel::Spherical);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let moon_clear = CLEAR_MOON_KM;

    // Measured peak relative acceleration change |a_new - a_old| / |a_sunlit|
    // across Earth's penumbra (clear Moon), equatorial pass vs all latitudes:
    //   LEO  (7000 km):  equatorial 0.0587, polar 0.8409
    //   GNSS (26560 km): equatorial 0.0589, polar 0.1479
    //   GEO  (42164 km): equatorial 0.0595, polar 0.1146
    for (label, radius_km, expected_eq, expected_all) in [
        ("LEO", 7_000.0_f64, 0.0587_f64, 0.8409_f64),
        ("GNSS", 26_560.0_f64, 0.0589_f64, 0.1479_f64),
        ("GEO", 42_164.0_f64, 0.0595_f64, 0.1146_f64),
    ] {
        let mut max_eq = 0.0_f64;
        let mut max_all = 0.0_f64;

        let half_angle = libm::asin(6_371.0 / radius_km);
        let alpha_min = (half_angle - 0.02).max(1.0e-5);
        let alpha_max = half_angle + 0.02;
        let n_alpha = 4000usize;
        let azimuths = [
            0.0,
            std::f64::consts::FRAC_PI_6,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_3,
            std::f64::consts::FRAC_PI_2,
        ];

        for (j, &phi) in azimuths.iter().enumerate() {
            let (sin_phi, cos_phi) = (libm::sin(phi), libm::cos(phi));
            for i in 0..=n_alpha {
                let alpha = alpha_min + (alpha_max - alpha_min) * (i as f64 / n_alpha as f64);
                let (sin_a, cos_a) = (libm::sin(alpha), libm::cos(alpha));
                let sat = Vector3::new(
                    -radius_km * cos_a,
                    radius_km * sin_a * cos_phi,
                    radius_km * sin_a * sin_phi,
                );
                let a_new = new_default
                    .acceleration_with_sun_moon(sat, sun, moon_clear)
                    .expect("new default");
                let a_old = old_default
                    .acceleration_with_sun_moon(sat, sun, moon_clear)
                    .expect("old default");
                let a_sunlit = srp_acceleration_m_s2(sat, sun, 1.0, new_default)
                    .expect("sunlit")
                    .norm()
                    / M_PER_KM;
                let rel = (a_new - a_old).norm() / a_sunlit;
                if j == 0 {
                    max_eq = max_eq.max(rel);
                }
                max_all = max_all.max(rel);
            }
        }
        assert!(
            (max_eq - expected_eq).abs() < 5.0e-4,
            "{label} equatorial peak {max_eq:.6} vs {expected_eq:.4}"
        );
        assert!(
            (max_all - expected_all).abs() < 5.0e-4,
            "{label} polar peak {max_all:.6} vs {expected_all:.4}"
        );
    }
}

#[test]
fn cr_partial_succeeds_on_exact_annular_lunar_eclipse_via_double_fallback() {
    let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let sat = Vector3::new(7_000.0, 0.0, 0.0);
    let sun = Vector3::new(7_000.0, AU_KM, 0.0);
    let moon = Vector3::new(7_000.0, 405_000.0, 0.0);

    assert!(matches!(
        lighting_ratio_field_of(model, sat, sun, Some(moon)),
        Err(PropagationError::NumericalFailure(_))
    ));
    assert!(matches!(
        srp_position_partials_with_bodies(sat, sun, Some(moon), model),
        Err(PropagationError::NumericalFailure(_))
    ));

    let accel = model
        .acceleration_with_sun_moon(sat, sun, moon)
        .expect("annular acceleration");
    let cr_partial =
        srp_parameter_partial_with_bodies(sat, sun, Some(moon), model).expect("annular C_R");
    assert!(accel.y < 0.0);
    assert_eq!(accel.x, 0.0);
    assert_eq!(accel.z, 0.0);
    assert!(cr_partial.y < 0.0);
    assert!((cr_partial * model.cr() - accel).norm() <= 4.0 * f64::EPSILON * accel.norm());
}

#[test]
fn near_axis_finite_norm_preserves_lit_umbra_and_annular_fallback_for_srp() {
    let base = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");

    // 1. Sunlit LEO 1e-165 km off the Earth-Sun line.
    let sat_lit = Vector3::new(7_000.0, 1.0e-165, 0.0);
    let sun_x = Vector3::new(AU_KM, 0.0, 0.0);
    let moon_clear = Vector3::new(0.0, 384_400.0, 0.0);

    for earth_model in [EarthShadowModel::Wgs84Oblate, EarthShadowModel::Spherical] {
        let model = base.with_earth_shadow_model(earth_model);
        let (sh, grad) = shadow_fraction_with_model_moon_and_gradient(
            as_array(sat_lit),
            as_array(sun_x),
            as_array(moon_clear),
            earth_model,
        )
        .expect("sunlit near-axis shadow gradient");
        assert_eq!(sh, 0.0);
        assert_eq!(grad, [0.0; 3]);

        let acc = model
            .acceleration_with_sun_moon(sat_lit, sun_x, moon_clear)
            .expect("sunlit near-axis SRP accel");
        let pos_p = srp_position_partials_with_bodies(sat_lit, sun_x, Some(moon_clear), model)
            .expect("sunlit near-axis SRP position partials");
        let cr_p = srp_parameter_partial_with_bodies(sat_lit, sun_x, Some(moon_clear), model)
            .expect("sunlit near-axis SRP C_R partial");
        assert!(acc.x < 0.0 && pos_p.iter().all(|x| x.is_finite()));
        assert!((cr_p * model.cr() - acc).norm() <= 4.0 * f64::EPSILON * acc.norm());
    }

    // 2. Deep Earth umbra and total lunar umbra 1e-165 km off axis.
    let umbra_cases = [
        (
            Vector3::new(-7_000.0, 1.0e-165, 0.0),
            Vector3::new(AU_KM, 0.0, 0.0),
            Vector3::new(0.0, 384_400.0, 0.0),
        ),
        (
            Vector3::new(7_000.0, 0.0, 0.0),
            Vector3::new(7_000.0, AU_KM, 0.0),
            Vector3::new(7_000.0, 356_000.0, 1.0e-165),
        ),
    ];
    for (sat, sun, moon) in umbra_cases {
        for earth_model in [EarthShadowModel::Wgs84Oblate, EarthShadowModel::Spherical] {
            let model = base.with_earth_shadow_model(earth_model);
            let (sh, grad) = shadow_fraction_with_model_moon_and_gradient(
                as_array(sat),
                as_array(sun),
                as_array(moon),
                earth_model,
            )
            .expect("umbra near-axis shadow gradient");
            assert_eq!(sh, 1.0);
            assert_eq!(grad, [0.0; 3]);
            assert_eq!(
                model
                    .acceleration_with_sun_moon(sat, sun, moon)
                    .expect("umbra accel"),
                Vector3::zeros()
            );
            assert_eq!(
                srp_position_partials_with_bodies(sat, sun, Some(moon), model)
                    .expect("umbra position partials"),
                Matrix3::zeros()
            );
            assert_eq!(
                srp_parameter_partial_with_bodies(sat, sun, Some(moon), model)
                    .expect("umbra C_R partial"),
                Vector3::zeros()
            );
        }
    }

    // 3. Near-axis annular lunar eclipse (1e-165 km off axis) reaching double fallback.
    let sat_ann = Vector3::new(7_000.0, 0.0, 0.0);
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
        srp_position_partials_with_bodies(sat_ann, sun_ann, Some(moon_ann), base),
        Err(PropagationError::NumericalFailure(_))
    ));
    let acc_ann = base
        .acceleration_with_sun_moon(sat_ann, sun_ann, moon_ann)
        .expect("near-axis annular accel");
    let cr_ann = srp_parameter_partial_with_bodies(sat_ann, sun_ann, Some(moon_ann), base)
        .expect("near-axis annular C_R partial via double fallback");
    assert!(acc_ann.y < 0.0 && cr_ann.y < 0.0);
    assert!((cr_ann * base.cr() - acc_ann).norm() <= 4.0 * f64::EPSILON * acc_ann.norm());
}

#[test]
fn cannonball_srp_and_shadow_gradient_wide_magnitude_property_sweep() {
    // Acceptance/refusal rule across magnitudes where shadow geometry is finite:
    // - Shadow fraction and shadow gradient accept and refuse the exact same inputs,
    //   except where only the derivative is undefined or overflows binary64
    //   (ShadowGradientError::NumericalFailure).
    // - Cannonball SRP acceleration and parameter_partials accept and refuse the exact
    //   same inputs; position_partials accepts every input that acceleration accepts
    //   except where only the derivative is undefined or overflows
    //   (PropagationError::NumericalFailure), and never accepts an input acceleration refuses.
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
            let phi = 2.0 * std::f64::consts::PI * self.unit_f64();
            let r = (1.0 - z * z).max(0.0).sqrt();
            Vector3::new(r * libm::cos(phi), r * libm::sin(phi), z)
        }
    }

    let mut rng = SplitMix64(0x5a99_2026_c0de_5678);
    let base = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let configs = all_shadow_configs(base);

    let mut conical_ok = 0usize;
    let mut apparent_ok = 0usize;
    let mut apparent_deriv_only = 0usize;
    let mut apparent_err = 0usize;
    let mut srp_ok = 0usize;
    let mut srp_pos_p_ok = 0usize;
    let mut srp_pos_p_analytic_checked = 0usize;
    let mut srp_pos_p_fd_checked = 0usize;

    for model in configs {
        let is_conical = matches!(model.shadow_function(), ShadowFunction::Conical);
        let include_moon = model.moon_eclipse_enabled();
        let earth_model = model.earth_shadow_model();

        for case in 0..240 {
            let r_dir = rng.unit_vec();
            let s_dir = rng.unit_vec();
            let m_dir = rng.unit_vec();

            let (sat, sun, moon) = if is_conical {
                let r_mag = rng.log_uniform(-480, 495);
                let s_mag = rng.log_uniform(20, 495);
                let m_mag = rng.log_uniform(12, 495);
                match case % 6 {
                    // Conical antumbra/penumbra near-axis and exact-axis cases.
                    1 => {
                        let dist = rng.log_uniform(13, 30);
                        let eps = if case % 12 == 1 {
                            0.0
                        } else {
                            rng.log_uniform(-560, 14)
                        };
                        let perp = s_dir
                            .cross(&Vector3::new(0.0, 0.0, 1.0))
                            .try_normalize(1.0e-12)
                            .unwrap_or_else(|| Vector3::new(1.0, 0.0, 0.0));
                        (-s_dir * dist + perp * eps, s_dir * AU_KM, m_dir * m_mag)
                    }
                    // Interior conical penumbra (LEO/GNSS) and off-axis antumbra.
                    2 => {
                        let perp = s_dir
                            .cross(&Vector3::new(0.0, 0.0, 1.0))
                            .try_normalize(1.0e-12)
                            .unwrap_or_else(|| Vector3::new(1.0, 0.0, 0.0));
                        let (dist, rho) = if case % 12 == 2 {
                            (26_560.0, 6_365.0)
                        } else {
                            (2.0e6, 5_000.0)
                        };
                        (-s_dir * dist + perp * rho, s_dir * AU_KM, m_dir * m_mag)
                    }
                    _ => (r_dir * r_mag, s_dir * s_mag, m_dir * m_mag),
                }
            } else {
                let r_mag = rng.log_uniform(13, 495);
                let s_mag = rng.log_uniform(14, 495);
                let m_mag = rng.log_uniform(12, 495);
                match case % 6 {
                    // Near-axis Earth lit / umbra / penumbra.
                    1 => {
                        let perp = s_dir
                            .cross(&Vector3::new(0.0, 0.0, 1.0))
                            .try_normalize(1.0e-12)
                            .unwrap_or_else(|| Vector3::new(1.0, 0.0, 0.0));
                        let sign = if case % 12 == 1 { 1.0 } else { -1.0 };
                        let eps = rng.log_uniform(-560, 12);
                        (
                            s_dir * (sign * 7_000.0) + perp * eps,
                            s_dir * AU_KM,
                            m_dir * m_mag,
                        )
                    }
                    // Near-axis lunar annular / total / partial.
                    2 if include_moon => {
                        let dist = if case % 12 == 2 { 405_000.0 } else { 356_000.0 };
                        let eps = rng.log_uniform(-560, 5);
                        (
                            Vector3::new(7_000.0, 0.0, 0.0),
                            Vector3::new(7_000.0, AU_KM, 0.0),
                            Vector3::new(7_000.0, dist, eps),
                        )
                    }
                    // Interior Earth penumbra (and lunar partial when Moon is enabled).
                    3 => {
                        if include_moon && case % 12 == 3 {
                            (
                                Vector3::new(7_000.0, 0.0, 0.0),
                                Vector3::new(7_000.0, AU_KM, 0.0),
                                Vector3::new(7_000.0, 384_400.0, 1_800.0),
                            )
                        } else {
                            let perp = s_dir
                                .cross(&Vector3::new(0.0, 0.0, 1.0))
                                .try_normalize(1.0e-12)
                                .unwrap_or_else(|| Vector3::new(1.0, 0.0, 0.0));
                            (
                                -s_dir * 26_560.0 + perp * 6_365.0,
                                s_dir * AU_KM,
                                m_dir * m_mag,
                            )
                        }
                    }
                    _ => (r_dir * r_mag, s_dir * s_mag, m_dir * m_mag),
                }
            };

            // Shadow-gradient entry points.
            if is_conical {
                let sh = shadow_fraction_with_model(as_array(sat), as_array(sun), earth_model);
                let sh_g = shadow_fraction_with_model_and_gradient(
                    as_array(sat),
                    as_array(sun),
                    earth_model,
                );
                match (sh, sh_g) {
                    (Ok(f), Ok((fg, g))) => {
                        assert_eq!(f.to_bits(), fg.to_bits());
                        assert!(g.iter().all(|x| x.is_finite()));
                        conical_ok += 1;
                    }
                    (Ok(_), Err(ShadowGradientError::NumericalFailure)) => {}
                    (Err(_), Err(ShadowGradientError::Eclipse(_))) => {}
                    (a, b) => panic!(
                        "conical shadow vs grad mismatch ({earth_model:?} case={case}): val={a:?}, grad={b:?}, sat={sat:?}, sun={sun:?}"
                    ),
                }
            } else if include_moon {
                let sh = shadow_fraction_with_model_and_moon(
                    as_array(sat),
                    as_array(sun),
                    as_array(moon),
                    earth_model,
                );
                let sh_g = shadow_fraction_with_model_moon_and_gradient(
                    as_array(sat),
                    as_array(sun),
                    as_array(moon),
                    earth_model,
                );
                match (sh, sh_g) {
                    (Ok(f), Ok((fg, g))) => {
                        assert!(f.is_finite() && fg.is_finite());
                        assert!(g.iter().all(|x| x.is_finite()));
                        if f == 0.0 || f == 1.0 {
                            assert_eq!(
                                f.to_bits(),
                                fg.to_bits(),
                                "sunlit/umbra shadow val vs field val bit mismatch ({earth_model:?} case={case}): {f} vs {fg}"
                            );
                        } else {
                            // Double division a/b vs Grad::div a*(1/b) inside point_on_limb and
                            // masking_ratio, amplified by acos near +-1 on grazing Earth penumbra
                            // (measured max 5.243e-12 across the sweep).
                            assert!(
                                (f - fg).abs() <= 5.0e-11,
                                "apparent penumbra val vs field val mismatch ({earth_model:?} case={case}): {f} vs {fg}"
                            );
                        }
                        apparent_ok += 1;
                    }
                    (Ok(_), Err(ShadowGradientError::NumericalFailure)) => {
                        apparent_deriv_only += 1;
                    }
                    (Err(_), Err(ShadowGradientError::Eclipse(_))) => {
                        apparent_err += 1;
                    }
                    (a, b) => panic!(
                        "apparent shadow vs grad mismatch ({earth_model:?} case={case}): val={a:?}, grad={b:?}, sat={sat:?}, sun={sun:?}, moon={moon:?}"
                    ),
                }
            }

            // Cannonball SRP entry points.
            let moon_opt = include_moon.then_some(moon);
            let acc = model.acceleration_with_sun_moon(sat, sun, moon);
            let cr_p = srp_parameter_partial_with_bodies(sat, sun, moon_opt, model);
            let pos_p = srp_position_partials_with_bodies(sat, sun, moon_opt, model);
            let lr_d = lighting_ratio_of(model, sat, sun, moon_opt);
            let lr_f = lighting_ratio_field_of(model, sat, sun, moon_opt);

            assert_eq!(
                acc.is_ok(),
                cr_p.is_ok(),
                "SRP accel vs C_R partial mismatch ({model:?} case={case}): acc={acc:?}, cr={cr_p:?}, sat={sat:?}, sun={sun:?}, moon={moon:?}"
            );
            if acc.is_err() {
                assert!(
                    pos_p.is_err(),
                    "SRP position partials Ok when accel Err ({model:?} case={case}): sat={sat:?}, sun={sun:?}, moon={moon:?}"
                );
            } else if let Err(ref err) = pos_p {
                assert!(
                    matches!(err, PropagationError::NumericalFailure(_)),
                    "SRP derivative-only refusal must be NumericalFailure ({model:?} case={case}): got {err:?}, sat={sat:?}, sun={sun:?}, moon={moon:?}"
                );
                // Position partials must only fail when the shadow-ratio derivative itself
                // is undefined or non-finite (exact/near-axis annular or conical antumbra axis).
                assert!(
                    lr_f.is_err(),
                    "SRP position partials refused with finite lighting ratio ({model:?} case={case}): err={err:?}, sat={sat:?}, sun={sun:?}, moon={moon:?}"
                );
            }
            if let (Ok(a), Ok(dp)) = (acc, cr_p) {
                srp_ok += 1;
                if a.norm() > 0.0 {
                    let is_const_nu = matches!(lr_d, Ok(nu) if nu == 0.0 || nu == 1.0);
                    // Constant lighting: raw_p/sqrt(r2) vs raw_p*(1/sqrt(r2)) (measured max 2.1 eps).
                    // Penumbra: adds |ratio_double - ratio_field| / ratio_double (measured max 1.13e-11).
                    let tol = if is_const_nu {
                        8.0 * f64::EPSILON * a.norm()
                    } else {
                        1.0e-10 * a.norm()
                    };
                    assert!(
                        (dp * model.cr() - a).norm() <= tol,
                        "SRP accel vs cr*d(accel)/d(cr) ({model:?} case={case}): a={a:?}, dp={dp:?}"
                    );
                }
            }
            if let (Ok(jp), Ok(ratio)) = (pos_p, lr_f) {
                srp_pos_p_ok += 1;
                // Bit-exact check against unscaled Orekit-order Field<Gradient> SRP evaluation
                // (ApparentDisc and sunlit/umbra Conical; Conical penumbra/antumbra uses
                // exponent-separated arithmetic to avoid binary64 1/km -> 1/m subnormal loss).
                if !is_conical || ratio.grad == [0.0; 3] {
                    let raw_field = srp_acceleration_field_m_s2(sat, sun, ratio, model)
                        .expect("raw SRP field must succeed when pos_p succeeded");
                    for row in 0..3 {
                        let g = match row {
                            0 => raw_field.x.grad,
                            1 => raw_field.y.grad,
                            _ => raw_field.z.grad,
                        };
                        for col in 0..3 {
                            assert_eq!(jp[(row, col)].to_bits(), g[col].to_bits());
                        }
                    }
                }

                // Independent value check:
                // - Sunlit or umbra (grad == 0): closed-form analytic Jacobian using exact power-of-two scaling.
                // - Interior penumbra/antumbra (grad != 0): central finite differences of acceleration_with_sun_moon.
                if ratio.grad == [0.0; 3] {
                    let d_m = (sat - sun) * M_PER_KM;
                    let d_max = d_m.x.abs().max(d_m.y.abs()).max(d_m.z.abs());
                    if d_max > 0.0 && d_max.is_finite() {
                        let e_d = crate::astro::events::eclipse::ad::exact_exponent(d_max);
                        let ds = Vector3::new(
                            libm::ldexp(d_m.x, -e_d),
                            libm::ldexp(d_m.y, -e_d),
                            libm::ldexp(d_m.z, -e_d),
                        );
                        let ds_norm = ds.norm();
                        let d_hat = ds / ds_norm;
                        let d_ref = model.au_km() * M_PER_KM;
                        let k_ref = model.pressure_n_m2() * d_ref * d_ref;
                        let factor = model.area_to_mass_m2_kg() * model.cr();
                        let coeff = libm::ldexp(
                            ratio.value * factor * k_ref / (ds_norm * ds_norm * ds_norm),
                            (-3 * e_d).clamp(-2200, 2200),
                        );
                        let analytic =
                            (Matrix3::identity() - 3.0 * d_hat * d_hat.transpose()) * coeff;
                        let a_norm = analytic.norm();
                        if a_norm > 1.0e-290 {
                            srp_pos_p_analytic_checked += 1;
                            // Grad::div reciprocal-multiply vs closed-form 1/ds^3 (measured max 4.7 eps).
                            assert!(
                                (jp - analytic).norm() <= 16.0 * f64::EPSILON * a_norm,
                                "SRP analytic Jacobian mismatch ({model:?} case={case}): rel={:e}",
                                (jp - analytic).norm() / a_norm
                            );
                        }
                    }
                } else if ratio.value > 1.0e-4 && ratio.value < 1.0 - 1.0e-4 {
                    // Interior penumbra/antumbra away from exact/near-axis annular singularity
                    // (where Orekit's getPlaneSection amplifies rounding by |sun|/(|moon|*eps)):
                    // verify against central finite differences.
                    let near_moon_axis = include_moon
                        && !is_conical
                        && (sat - moon).cross(&(sun - moon)).norm()
                            < 1.0e-1 * (sat - moon).norm() * (sun - moon).norm() * 1.0e-6;
                    if !near_moon_axis {
                        let mut fd = Matrix3::zeros();
                        let mut fd_ok = true;
                        let h = (sat.norm() * 1.0e-7).clamp(1.0e-4, 1.0);
                        for col in 0..3 {
                            let mut sp = sat;
                            let mut sm = sat;
                            sp[col] += h;
                            sm[col] -= h;
                            if let (Ok(ap), Ok(am)) = (
                                model.acceleration_with_sun_moon(sp, sun, moon),
                                model.acceleration_with_sun_moon(sm, sun, moon),
                            ) {
                                fd.set_column(col, &((ap - am) / (2.0 * h)));
                            } else {
                                fd_ok = false;
                                break;
                            }
                        }
                        if fd_ok && fd.norm() > 0.0 && fd.iter().all(|x| x.is_finite()) {
                            srp_pos_p_fd_checked += 1;
                            // Central-difference truncation O(h^2) and cancellation O(eps/h)
                            // across penumbra/antumbra (measured max 3.095e-7 over 206 cases).
                            let rel = (jp - fd).norm() / fd.norm();
                            assert!(
                                rel <= 5.0e-6,
                                "SRP penumbra FD mismatch ({model:?} case={case}): rel={rel:e}"
                            );
                        }
                    }
                }
            }
        }
    }

    assert!(conical_ok > 200, "too few conical ok: {conical_ok}");
    assert!(
        apparent_ok > 440,
        "too few apparent ok: ok={apparent_ok}, deriv_only={apparent_deriv_only}, err={apparent_err}"
    );
    assert!(srp_ok > 1300, "too few srp ok: {srp_ok}");
    assert!(
        srp_pos_p_ok > 1400,
        "too few SRP position partials ok: {srp_pos_p_ok}"
    );
    assert!(
        srp_pos_p_analytic_checked > 150 && srp_pos_p_fd_checked > 150,
        "insufficient SRP position partials value checks: analytic={srp_pos_p_analytic_checked}, fd={srp_pos_p_fd_checked}"
    );
    assert!(
        apparent_err < 10,
        "unexpectedly many apparent eclipse errors: ok={apparent_ok}, deriv_only={apparent_deriv_only}, err={apparent_err}"
    );

    // Explicit regression check for distant Moon (|moon| >> |sat|, |sun|) where
    // (sat - moon) - (sun - moon) cancels to zero in body-frame translation.
    let sat_leo = Vector3::new(7_000.0, 0.0, 0.0);
    let sun_au = Vector3::new(AU_KM, 0.0, 0.0);
    let moon_distant = Vector3::new(0.0, 1.0e63, 0.0);
    for earth_model in [EarthShadowModel::Wgs84Oblate, EarthShadowModel::Spherical] {
        assert_eq!(
            shadow_fraction_with_model_and_moon(
                as_array(sat_leo),
                as_array(sun_au),
                as_array(moon_distant),
                earth_model,
            )
            .expect("distant moon shadow"),
            0.0
        );
        assert_eq!(
            shadow_fraction_with_model_moon_and_gradient(
                as_array(sat_leo),
                as_array(sun_au),
                as_array(moon_distant),
                earth_model,
            )
            .expect("distant moon shadow gradient"),
            (0.0, [0.0; 3])
        );
    }

    // Explicit regression check for large-magnitude ApparentDisc vectors where
    // observer x outside exceeds 1e154 m^2 (overflowing unscaled orthogonal() and angle()).
    let sat_far = Vector3::new(1.0e80, 0.0, 0.0);
    let sun_far = Vector3::new(0.0, 1.0e80, 0.0);
    let moon_far = Vector3::new(0.0, 0.0, 1.0e80);
    for earth_model in [EarthShadowModel::Wgs84Oblate, EarthShadowModel::Spherical] {
        assert_eq!(
            shadow_fraction_with_model_and_moon(
                as_array(sat_far),
                as_array(sun_far),
                as_array(moon_far),
                earth_model,
            )
            .expect("large-magnitude shadow"),
            0.0
        );
        let (sh, grad) = shadow_fraction_with_model_moon_and_gradient(
            as_array(sat_far),
            as_array(sun_far),
            as_array(moon_far),
            earth_model,
        )
        .expect("large-magnitude shadow gradient");
        assert_eq!(sh, 0.0);
        assert_eq!(grad, [0.0; 3]);
    }
}

#[test]
fn spherical_penumbra_plane_section_nonfinite_tan_theta_fallback_matches_fd() {
    // Near-equatorial spherical Earth penumbra inputs where Orekit's
    // Ellipsoid.getPlaneSection produces non-finite tanTheta (both value
    // underflow to +-Inf at |gamma| ~ 1e-167 and finite value +-1.0 with
    // overflowing inv2 derivative at |gamma| ~ 2.5e-160). Falling back to
    // tan_theta = 0 with zero gradient must yield finite penumbra shadow
    // gradients and SRP position partials matching central differences.
    let sun = Vector3::new(149_597_870.7, 1.0e-85, 0.0);
    let moon = Vector3::new(0.0, 384_400.0, 0.0);
    let model = SolarRadiationPressure::new(1.3, 0.02)
        .expect("valid SRP")
        .with_shadow_function(ShadowFunction::ApparentDisc)
        .with_earth_shadow_model(EarthShadowModel::Spherical)
        .with_moon_eclipse(false);

    let cases = [
        ("underflow_val", Vector3::new(-7_000.0, 1.0e-40, 6_370.0)),
        ("nonfinite_grad", Vector3::new(-7_000.0, 1.0e-33, 6_370.0)),
    ];

    // Exact-Orekit reference with y = 0 (where |gamma| == 0 < SAFE_MIN takes
    // Orekit's own tan_theta = 0 branch directly).
    let sat0 = Vector3::new(-7_000.0, 0.0, 6_370.0);
    let sun0 = Vector3::new(149_597_870.7, 0.0, 0.0);
    let (sh0, sh0_grad) = shadow_fraction_with_model_moon_and_gradient(
        as_array(sat0),
        as_array(sun0),
        as_array(moon),
        EarthShadowModel::Spherical,
    )
    .expect("y=0 shadow gradient");
    let jp0 = srp_position_partials_with_bodies(sat0, sun0, None, model)
        .expect("y=0 SRP position partials");

    // 4th-order central difference with h = 0.05 km balances truncation O((h/W)^4)
    // against masking_ratio's O(1e-10) rounding amplification in penumbra.
    let h_sh = 0.05;

    for (label, sat) in cases {
        let (sh, sh_grad) = shadow_fraction_with_model_moon_and_gradient(
            as_array(sat),
            as_array(sun),
            as_array(moon),
            EarthShadowModel::Spherical,
        )
        .unwrap_or_else(|e| panic!("{label}: shadow gradient failed: {e:?}"));
        let lighting = 1.0 - sh;
        assert!(
            lighting > 0.45 && lighting < 0.51,
            "{label}: expected penumbra lighting ~0.480, got {lighting}"
        );
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

        // Check shadow gradient against 4th-order central differences of shadow_fraction_with_model_and_moon.
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
            fd_sh[axis] = (eval(-2.0 * h_sh) - 8.0 * eval(-h_sh) + 8.0 * eval(h_sh)
                - eval(2.0 * h_sh))
                / (12.0 * h_sh);
        }
        let sh_norm = (fd_sh[0] * fd_sh[0] + fd_sh[1] * fd_sh[1] + fd_sh[2] * fd_sh[2]).sqrt();
        let sh_diff = ((sh_grad[0] - fd_sh[0]).powi(2)
            + (sh_grad[1] - fd_sh[1]).powi(2)
            + (sh_grad[2] - fd_sh[2]).powi(2))
        .sqrt();
        assert!(
            sh_diff <= 1.0e-7 * sh_norm,
            "{label}: shadow gradient rel err {:e} (grad={sh_grad:?}, fd={fd_sh:?})",
            sh_diff / sh_norm
        );

        let jp = srp_position_partials_with_bodies(sat, sun, None, model)
            .unwrap_or_else(|e| panic!("{label}: SRP position partials failed: {e:?}"));
        assert!(jp.iter().all(|x| x.is_finite()));
        assert!(
            (jp - jp0).norm() <= 16.0 * f64::EPSILON * jp0.norm(),
            "{label}: SRP position partials vs y=0 ref rel err {:e}",
            (jp - jp0).norm() / jp0.norm()
        );

        let mut fd_srp = Matrix3::zeros();
        for col in 0..3 {
            let eval = |delta: f64| {
                let mut p = sat;
                p[col] += delta;
                model
                    .acceleration_with_sun_moon(p, sun, moon)
                    .expect("fd SRP")
            };
            let deriv = (eval(-2.0 * h_sh) - 8.0 * eval(-h_sh) + 8.0 * eval(h_sh)
                - eval(2.0 * h_sh))
                / (12.0 * h_sh);
            fd_srp.set_column(col, &deriv);
        }
        let srp_rel = (jp - fd_srp).norm() / fd_srp.norm();
        assert!(
            srp_rel <= 1.0e-7,
            "{label}: SRP position partials rel err {srp_rel:e}"
        );
    }
}

#[test]
fn conical_tiny_lighting_gradient_preserves_full_precision_in_cannonball_srp() {
    // Same antumbra states where conical d(shadow)/dr_z in 1/km is normal
    // (~-2.25e-307) or subnormal (~-1.88e-322) and binary64 `/ 1000.0` loses
    // 10 bits or flushes to zero before scaling by large pressure * A/m.
    let sun = Vector3::new(149_597_870.7, 0.0, 0.0);
    let r_earth = crate::astro::constants::earth::MEAN_EARTH_RADIUS_KM;
    let r_sun = crate::astro::constants::astro::SOLAR_RADIUS_KM;
    let tan_p = libm::tan(libm::asin((r_sun + r_earth) / sun.x));

    let base = SolarRadiationPressure::with_pressure(1.0, 1.0e25, 1.0e280, AU_KM)
        .expect("finite parameters")
        .with_shadow_function(ShadowFunction::Conical);
    let base_ref =
        SolarRadiationPressure::with_pressure(1.0, 1.0e25, libm::ldexp(1.0e280, -500), AU_KM)
            .expect("finite parameters")
            .with_shadow_function(ShadowFunction::Conical);

    let mut max_rel_eps = 0.0_f64;
    let mut rz_samples = vec![1.2e-298_f64, 1.0e-313_f64];
    let mut seed: u64 = 0x51a0_c0a1_2026_0001;
    for _ in 0..48 {
        seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let u = (((z ^ (z >> 31)) >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64));
        let e = -1065.0 + 80.0 * u;
        let ei = e.floor() as i32;
        let sign = if z & 1 == 0 { 1.0 } else { -1.0 };
        rz_samples.push(sign * libm::ldexp(libm::exp2(e - ei as f64), ei));
    }

    for &rz in &rz_samples {
        let sat = Vector3::new(-1.0e7, 1.0e4, rz);
        let sat_ref = Vector3::new(-1.0e7, 1.0e4, libm::ldexp(rz, 500));
        let r_pen = r_earth + (-sat.x) * tan_p;

        for earth_model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
            let z_scale = match earth_model {
                EarthShadowModel::Spherical => 1.0,
                EarthShadowModel::Wgs84Oblate => {
                    1.0 / (1.0 - crate::astro::events::eclipse::WGS84_FLATTENING)
                }
            };
            let model = base.with_earth_shadow_model(earth_model);
            let model_ref = base_ref.with_earth_shadow_model(earth_model);

            let jp = srp_position_partials_with_bodies(sat, sun, None, model)
                .expect("SRP conical tiny-gradient partials");
            let jp_ref = srp_position_partials_with_bodies(sat_ref, sun, None, model_ref)
                .expect("SRP conical rescaled ref partials");

            // Closed-form (0, 2) entry:
            //   a1 = (factor * k_ref / r^2) * u_d,  d(a1.x)/dr_z = -3 * (factor * k_ref / r^3) * u_d.x * u_d.z,
            //   d(ratio)/dr_z_m = (rz * z_scale^2) / (1e4 * r_pen * 1000),
            //   jp[(0, 2)] = d(ratio)/dr_z_m * a1.x + ratio * d(a1.x)/dr_z.
            let ratio = 1.0
                - shadow_fraction_with_model(as_array(sat), as_array(sun), earth_model).unwrap();
            let d_ref = model.au_km() * M_PER_KM;
            let k_ref = model.pressure_n_m2() * d_ref * d_ref;
            let factor = model.area_to_mass_m2_kg() * model.cr();
            let (u_d, r_s) = ScaledVec3::from_vec3(Vec3::new(
                (sat.x - sun.x) * M_PER_KM,
                (sat.y - sun.y) * M_PER_KM,
                (sat.z - sun.z) * M_PER_KM,
            ))
            .normalize_scaled()
            .unwrap();
            let scale_s = ScaledF64::from_f64(factor)
                .mul(ScaledF64::from_f64(k_ref))
                .div(r_s.mul(r_s));
            let a1_x = u_d.x.mul(scale_s);
            let da1_x_dz = ScaledF64::from_f64(-3.0)
                .mul(u_d.x)
                .mul(u_d.z)
                .mul(scale_s.div(r_s));
            let dratio_dz_m = ScaledF64::from_f64(rz)
                .mul(ScaledF64::from_f64(z_scale))
                .mul(ScaledF64::from_f64(z_scale))
                .div(
                    ScaledF64::from_f64(sat.y)
                        .mul(ScaledF64::from_f64(r_pen))
                        .mul(ScaledF64::from_f64(M_PER_KM)),
                );
            let expected_02 =
                ScaledF64::lincomb2(dratio_dz_m, a1_x, ScaledF64::from_f64(ratio), da1_x_dz)
                    .to_f64();

            let rel = (jp[(0, 2)] - expected_02).abs() / expected_02.abs();
            let rel_ref = (jp[(0, 2)] - jp_ref[(0, 2)]).abs() / jp_ref[(0, 2)].abs();
            max_rel_eps = max_rel_eps
                .max(rel / f64::EPSILON)
                .max(rel_ref / f64::EPSILON);
            assert!(
                rel <= 8.0 * f64::EPSILON && rel_ref <= 8.0 * f64::EPSILON,
                "SRP conical rz={rz:e} {earth_model:?}: got={:e} closed={expected_02:e} rescaled={:e} (rel={rel:e}, ref_rel={rel_ref:e})",
                jp[(0, 2)],
                jp_ref[(0, 2)]
            );
        }
    }
    // Measured max across 100 cases (50 rz values in [2^-1065, 2^-985] x 2 Earth models):
    // 1.14 eps (caused by ScaledF64 rounding in r_penumbra, u_d, and lincomb2).
    assert!(
        max_rel_eps <= 8.0,
        "SRP tiny-conical-gradient sweep max rel={max_rel_eps:.2} eps"
    );
}

#[test]
fn conical_srp_extreme_parameters_and_tiny_nu_match_independent_reference() {
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let r_earth = crate::astro::constants::earth::MEAN_EARTH_RADIUS_KM;
    let r_sun = crate::astro::constants::astro::SOLAR_RADIUS_KM;
    let tan_p = libm::tan(libm::asin((r_sun + r_earth) / AU_KM));

    // Independent closed-form conical antumbra reference for sun = [AU_KM, 0, 0],
    // sat = [x < 0, y > 0, 0] in antumbra (r_umbra <= 0):
    //   r_pen = R_e + (-x) * tan_p, nu = y / r_pen,
    //   a_km = nu * cr * area * pressure * (au / |sat - sun|)^2 * u_d / 1000.
    let check_case =
        |label: &str, model: SolarRadiationPressure, sat: Vector3<f64>, expected_ax: f64| {
            for earth_model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
                let m = model.with_earth_shadow_model(earth_model);
                let acc = srp_acceleration_with_bodies(sat, sun, None, m)
                    .unwrap_or_else(|e| panic!("{label} {earth_model:?} accel failed: {e:?}"));
                let cr_p = srp_parameter_partial_with_bodies(sat, sun, None, m)
                    .unwrap_or_else(|e| panic!("{label} {earth_model:?} cr_p failed: {e:?}"));
                let jp = srp_position_partials_with_bodies(sat, sun, None, m)
                    .unwrap_or_else(|e| panic!("{label} {earth_model:?} jp failed: {e:?}"));

                let rel_acc = (acc.x - expected_ax).abs() / expected_ax.abs();
                assert!(
                    rel_acc <= 8.0 * f64::EPSILON,
                    "{label} {earth_model:?}: acc.x={:e} vs {expected_ax:e} (rel={rel_acc:e})",
                    acc.x
                );
                let expected_cr_x = expected_ax / m.cr();
                let rel_cr = (cr_p.x - expected_cr_x).abs() / expected_cr_x.abs();
                assert!(
                    rel_cr <= 8.0 * f64::EPSILON,
                    "{label} {earth_model:?}: cr_p.x={:e} vs {expected_cr_x:e} (rel={rel_cr:e})",
                    cr_p.x
                );
                assert!(jp.iter().all(|x| x.is_finite()));

                // Double acceleration must equal both field value parts bit for bit.
                let (ratio_km, is_sunlit) =
                    conical_lighting_ratio_field_scaled(as_array(sat), as_array(sun), earth_model)
                        .expect("conical field");
                assert!(!is_sunlit);
                let pos_field = srp_conical_position_field_scaled(sat, sun, ratio_km, m)
                    .expect("pos field")
                    .val_vec()
                    .to_vec3()
                    .expect("pos field val finite");
                let cr_field = srp_conical_cr_field_scaled(sat, sun, ratio_km.val, m)
                    .expect("cr field")
                    .val_vec()
                    .to_vec3()
                    .expect("cr field val finite");
                assert_eq!(acc.x.to_bits(), pos_field.x.to_bits());
                assert_eq!(acc.y.to_bits(), pos_field.y.to_bits());
                assert_eq!(acc.z.to_bits(), pos_field.z.to_bits());
                assert_eq!(acc.x.to_bits(), cr_field.x.to_bits());
                assert_eq!(acc.y.to_bits(), cr_field.y.to_bits());
                assert_eq!(acc.z.to_bits(), cr_field.z.to_bits());
            }
        };

    // 1 & 2. Extreme pressure * A/m with sub-subnormal (1e-325) and subnormal (1e-315) nu.
    let m1 = SolarRadiationPressure::with_pressure(1.0, 1.0e25, 1.0e280, AU_KM)
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
    let r_pen_2e6 = r_earth + 2.0e6 * tan_p;
    let dist_2e6 = AU_KM + 2.0e6;
    let scale_1 = ScaledF64::from_f64(1.0e25)
        .mul(ScaledF64::from_f64(1.0e280))
        .mul(ScaledF64::from_f64(AU_KM / dist_2e6))
        .mul(ScaledF64::from_f64(AU_KM / dist_2e6))
        .div(ScaledF64::from_f64(M_PER_KM));
    let expected_1a = scale_1
        .mul(ScaledF64::from_f64(1.0e-320).div(ScaledF64::from_f64(r_pen_2e6)))
        .negate()
        .to_f64();
    let expected_1b = scale_1
        .mul(ScaledF64::from_f64(1.0e-310).div(ScaledF64::from_f64(r_pen_2e6)))
        .negate()
        .to_f64();
    check_case(
        "sub_subnormal_nu",
        m1,
        Vector3::new(-2.0e6, 1.0e-320, 0.0),
        expected_1a,
    );
    check_case(
        "subnormal_nu",
        m1,
        Vector3::new(-2.0e6, 1.0e-310, 0.0),
        expected_1b,
    );

    // 3. Normal nu = 0.1875 with extreme A/m = 1e300 and subnormal pressure = 1e-315
    //    (where raw_p / sqrt(r2) would underflow to 0 in binary64).
    let m3 = SolarRadiationPressure::with_pressure(1.0, 1.0e300, 1.0e-315, AU_KM)
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
    let sat3 = Vector3::new(-1.0e7, 1.0e4, 0.0);
    let r_pen_1e7 = r_earth + 1.0e7 * tan_p;
    let dist_3 = libm::hypot(AU_KM + 1.0e7, 1.0e4);
    let ux_3 = -(AU_KM + 1.0e7) / dist_3;
    let expected_3 = ScaledF64::from_f64(1.0e300)
        .mul(ScaledF64::from_f64(1.0e-315))
        .mul(ScaledF64::from_f64(AU_KM / dist_3))
        .mul(ScaledF64::from_f64(AU_KM / dist_3))
        .mul(ScaledF64::from_f64(1.0e4 / r_pen_1e7))
        .mul(ScaledF64::from_f64(ux_3))
        .div(ScaledF64::from_f64(M_PER_KM))
        .to_f64();
    check_case("underflow_raw_p", m3, sat3, expected_3);

    // 4. Exact non-axis-aligned antumbra axis: accel and C_R partial return 0,
    //    position partials return NumericalFailure.
    let sat_axis = Vector3::new(-2.0e6, -3.0e6, 0.0);
    let sun_axis = Vector3::new(2.0e8, 3.0e8, 0.0);
    for earth_model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
        let m_ax = m1.with_earth_shadow_model(earth_model);
        assert_eq!(
            srp_acceleration_with_bodies(sat_axis, sun_axis, None, m_ax).expect("axis accel"),
            Vector3::zeros()
        );
        assert_eq!(
            srp_parameter_partial_with_bodies(sat_axis, sun_axis, None, m_ax).expect("axis cr"),
            Vector3::zeros()
        );
        assert!(matches!(
            srp_position_partials_with_bodies(sat_axis, sun_axis, None, m_ax),
            Err(PropagationError::NumericalFailure(_))
        ));
    }

    // 5. Sweep over extreme pressure, A/m, C_R, and tiny nu (including k_ref > f64::MAX
    //    and factor > f64::MAX) verifying bit-exact double vs field value and accuracy
    //    against an independent power-of-two rescaled reference.
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
    let mut rng = SplitMix64(0x5a9c_0a1c_2026_0010);
    let mut checked = 0usize;
    let mut max_acc_eps = 0.0_f64;
    let mut max_cr_eps = 0.0_f64;
    let mut max_jp_eps = 0.0_f64;
    for case in 0..256 {
        let cr = rng.log_uniform(-300, 300);
        let area = rng.log_uniform(-300, 300);
        let rho_z = rng.log_uniform(-1060, -20);
        // Choose pressure so nu * cr * area * pressure / 1000 lands in normal binary64 range.
        let nu_est_exp = crate::astro::events::eclipse::ad::exact_exponent(rho_z) - 14;
        let cr_e = crate::astro::events::eclipse::ad::exact_exponent(cr);
        let area_e = crate::astro::events::eclipse::ad::exact_exponent(area);
        let target_p_exp = (-nu_est_exp - cr_e - area_e).clamp(-1040, 1010);
        let pressure = libm::ldexp(1.0 + rng.unit(), target_p_exp);
        let base_m = SolarRadiationPressure::with_pressure(cr, area, pressure, AU_KM)
            .unwrap()
            .with_shadow_function(ShadowFunction::Conical);
        let earth_model = if case % 2 == 0 {
            EarthShadowModel::Spherical
        } else {
            EarthShadowModel::Wgs84Oblate
        };
        let m_case = base_m.with_earth_shadow_model(earth_model);
        let k = match earth_model {
            EarthShadowModel::Spherical => 1.0,
            EarthShadowModel::Wgs84Oblate => {
                1.0 / (1.0 - crate::astro::events::eclipse::WGS84_FLATTENING)
            }
        };
        // Non-axis-aligned Sun in xy plane so pure-z offset is exact in binary64.
        let sun_c = Vector3::new(2.0e8, 3.0e8, 0.0);
        let sat_c = Vector3::new(-2.0e6, -3.0e6, rho_z);
        let acc = srp_acceleration_with_bodies(sat_c, sun_c, None, m_case);
        let cr_p = srp_parameter_partial_with_bodies(sat_c, sun_c, None, m_case);
        let jp = srp_position_partials_with_bodies(sat_c, sun_c, None, m_case);

        let (ratio_km, is_sunlit) =
            conical_lighting_ratio_field_scaled(as_array(sat_c), as_array(sun_c), earth_model)
                .unwrap();
        assert!(!is_sunlit);
        // Independent closed-form reference in ScaledF64.
        let d_sun_c = libm::hypot(2.0e8, 3.0e8);
        let dist_b = libm::hypot(2.0e6, 3.0e6);
        let tan_pc = libm::tan(libm::asin((r_sun + r_earth) / d_sun_c));
        let r_pen_c = r_earth + dist_b * tan_pc;
        let r_ss = libm::hypot(2.02e8, 3.03e8);
        let ux = -2.02e8 / r_ss;
        let uy = -3.03e8 / r_ss;
        let base_scale = ScaledF64::from_f64(area)
            .mul(ScaledF64::from_f64(pressure))
            .mul(ScaledF64::from_f64(AU_KM / r_ss))
            .mul(ScaledF64::from_f64(AU_KM / r_ss))
            .div(ScaledF64::from_f64(M_PER_KM));
        let nu_ref = ScaledF64::from_f64(rho_z)
            .mul(ScaledF64::from_f64(k))
            .div(ScaledF64::from_f64(r_pen_c));
        let a_ref_x = base_scale
            .mul(ScaledF64::from_f64(cr))
            .mul(nu_ref)
            .mul(ScaledF64::from_f64(ux))
            .to_f64();
        let a_ref_y = base_scale
            .mul(ScaledF64::from_f64(cr))
            .mul(nu_ref)
            .mul(ScaledF64::from_f64(uy))
            .to_f64();
        let cp_ref_x = base_scale.mul(nu_ref).mul(ScaledF64::from_f64(ux)).to_f64();
        let cp_ref_y = base_scale.mul(nu_ref).mul(ScaledF64::from_f64(uy)).to_f64();
        let j02_ref = base_scale
            .mul(ScaledF64::from_f64(cr))
            .mul(ScaledF64::from_f64(k / r_pen_c))
            .mul(ScaledF64::from_f64(ux))
            .to_f64();
        let j12_ref = base_scale
            .mul(ScaledF64::from_f64(cr))
            .mul(ScaledF64::from_f64(k / r_pen_c))
            .mul(ScaledF64::from_f64(uy))
            .to_f64();

        if let Ok(a) = acc {
            let pos_f = srp_conical_position_field_scaled(sat_c, sun_c, ratio_km, m_case).unwrap();
            let cr_f = srp_conical_cr_field_scaled(sat_c, sun_c, ratio_km.val, m_case).unwrap();
            let pv = pos_f.val_vec().to_vec3().unwrap();
            let cv = cr_f.val_vec().to_vec3().unwrap();
            assert_eq!(a.x.to_bits(), pv.x.to_bits());
            assert_eq!(a.y.to_bits(), pv.y.to_bits());
            assert_eq!(a.z.to_bits(), pv.z.to_bits());
            assert_eq!(a.x.to_bits(), cv.x.to_bits());
            assert_eq!(a.y.to_bits(), cv.y.to_bits());
            assert_eq!(a.z.to_bits(), cv.z.to_bits());
            if a_ref_x.abs() >= f64::MIN_POSITIVE && a_ref_x.is_finite() {
                max_acc_eps = max_acc_eps
                    .max((a.x - a_ref_x).abs() / a_ref_x.abs() / f64::EPSILON)
                    .max((a.y - a_ref_y).abs() / a_ref_y.abs() / f64::EPSILON);
                checked += 1;
            }
        } else {
            assert!(matches!(acc, Err(PropagationError::NumericalFailure(_))));
            assert!(!a_ref_x.is_finite() || !a_ref_y.is_finite());
        }
        match cr_p {
            Ok(cp) => {
                if cp_ref_x.abs() >= f64::MIN_POSITIVE && cp_ref_x.is_finite() {
                    max_cr_eps =
                        max_cr_eps.max((cp.x - cp_ref_x).abs() / cp_ref_x.abs() / f64::EPSILON);
                }
            }
            Err(err) => {
                assert!(matches!(err, PropagationError::NumericalFailure(_)));
                assert!(!cp_ref_x.is_finite() || !cp_ref_y.is_finite());
            }
        }
        match jp {
            Ok(j) => {
                if j02_ref.abs() >= f64::MIN_POSITIVE && j02_ref.is_finite() {
                    max_jp_eps =
                        max_jp_eps.max((j[(0, 2)] - j02_ref).abs() / j02_ref.abs() / f64::EPSILON);
                }
            }
            Err(err) => {
                assert!(matches!(err, PropagationError::NumericalFailure(_)));
                assert!(!j02_ref.is_finite() || !j12_ref.is_finite());
            }
        }
    }
    // Measured maxima across 256 extreme conical SRP cases (Spherical & WGS84):
    // acc = 2.89 eps, cr = 3.13 eps, jp = 2.61 eps (ScaledF64 rounding in u_d,
    // r^2, k_ref, r_penumbra, and / M_PER_KM).
    assert!(
        checked >= 150,
        "too few conical extreme SRP checks: {checked}"
    );
    assert!(
        max_acc_eps <= 16.0 && max_cr_eps <= 16.0 && max_jp_eps <= 16.0,
        "conical extreme SRP sweep maxima: acc={max_acc_eps:.2} eps, cr={max_cr_eps:.2} eps, jp={max_jp_eps:.2} eps"
    );
}

type SrpReference = (
    Option<Vector3<f64>>,
    Option<Vector3<f64>>,
    Option<Matrix3<f64>>,
);

/// Independent exponent-separated reference for cannonball SRP acceleration,
/// `C_R` partial, and position Jacobian given a lighting ratio and its `1/km`
/// gradient, using the factored ratio `(au_km / r_km)^2` instead of `au^2 / r^2`.
fn independent_srp_reference(
    sat_km: Vector3<f64>,
    sun_km: Vector3<f64>,
    nu: ScaledF64,
    grad_nu_km: [ScaledF64; 3],
    model: SolarRadiationPressure,
) -> SrpReference {
    if nu.is_zero() && grad_nu_km.iter().all(|g| g.is_zero()) {
        return (
            Some(Vector3::zeros()),
            Some(Vector3::zeros()),
            Some(Matrix3::zeros()),
        );
    }
    let d_s = ScaledVec3::from_vec3(Vec3::new(sat_km.x, sat_km.y, sat_km.z)).sub(
        ScaledVec3::from_vec3(Vec3::new(sun_km.x, sun_km.y, sun_km.z)),
    );
    let (u_s, r_s) = d_s.normalize_scaled().expect("nonzero sat-sun distance");
    let au_over_r = ScaledF64::from_f64(model.au_km).div(r_s);
    let inv_sq_p = ScaledF64::from_f64(model.pressure_n_m2)
        .mul(au_over_r)
        .mul(au_over_r);
    let area_km = ScaledF64::from_f64(model.area_to_mass_m2_kg).div(ScaledF64::from_f64(M_PER_KM));
    let cr_s = ScaledF64::from_f64(model.cr);

    let a1_per_cr = area_km.mul(inv_sq_p);
    let a1_mag = a1_per_cr.mul(cr_s);
    let u = [u_s.x, u_s.y, u_s.z];

    let acc_vec = Vector3::new(
        u[0].mul(a1_mag).mul(nu).to_f64(),
        u[1].mul(a1_mag).mul(nu).to_f64(),
        u[2].mul(a1_mag).mul(nu).to_f64(),
    );
    let acc_opt = acc_vec.iter().all(|c| c.is_finite()).then_some(acc_vec);

    let cr_vec = Vector3::new(
        u[0].mul(a1_per_cr).mul(nu).to_f64(),
        u[1].mul(a1_per_cr).mul(nu).to_f64(),
        u[2].mul(a1_per_cr).mul(nu).to_f64(),
    );
    let cr_opt = cr_vec.iter().all(|c| c.is_finite()).then_some(cr_vec);

    let one = ScaledF64::from_f64(1.0);
    let neg_three = ScaledF64::from_f64(-3.0);
    let a1_over_r = a1_mag.div(r_s);
    let mut mat = Matrix3::zeros();
    let mut mat_ok = true;
    for row in 0..3 {
        let a1_row = u[row].mul(a1_mag);
        for col in 0..3 {
            let geom = if row == col {
                ScaledF64::lincomb2(one, one, neg_three, u[row].mul(u[col]))
            } else {
                neg_three.mul(u[row]).mul(u[col])
            };
            let da1_rc = geom.mul(a1_over_r);
            let entry = ScaledF64::lincomb2(nu, da1_rc, a1_row, grad_nu_km[col]).to_f64();
            if !entry.is_finite() {
                mat_ok = false;
            }
            mat[(row, col)] = entry;
        }
    }
    (acc_opt, cr_opt, mat_ok.then_some(mat))
}

#[test]
fn conical_partials_succeed_when_acceleration_value_overflows() {
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let r_earth = crate::astro::constants::earth::MEAN_EARTH_RADIUS_KM;
    let r_sun = crate::astro::constants::astro::SOLAR_RADIUS_KM;
    let tan_p = libm::tan(libm::asin((r_sun + r_earth) / AU_KM));

    // 1. Finite C_R partial (~ -1.65e196 km/s^2) while accel (~ -1.65e396) and
    //    position partials overflow binary64.
    let m_cr = SolarRadiationPressure::with_pressure(1.0e200, 1.0, 1.0e200, AU_KM)
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
    let sat_cr = Vector3::new(-1.0e7, 1.0e4, 0.0);
    let r_pen_cr = r_earth + 1.0e7 * tan_p;
    let nu_cr = ScaledF64::from_f64(1.0e4).div(ScaledF64::from_f64(r_pen_cr));
    let (acc_ref, cr_ref, jp_ref) =
        independent_srp_reference(sat_cr, sun, nu_cr, [ScaledF64::ZERO; 3], m_cr);
    assert!(acc_ref.is_none());
    let cr_ref = cr_ref.expect("finite C_R partial reference");
    assert!(jp_ref.is_none());

    for earth_model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
        let model = m_cr.with_earth_shadow_model(earth_model);
        assert!(matches!(
            srp_acceleration_with_bodies(sat_cr, sun, None, model),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            srp_position_partials_with_bodies(sat_cr, sun, None, model),
            Err(PropagationError::NumericalFailure(_))
        ));
        let cr_p = srp_parameter_partial_with_bodies(sat_cr, sun, None, model)
            .expect("conical C_R partial must succeed when finite");
        let rel = scaled_rel_err_vec3(cr_p, cr_ref);
        assert!(
            rel <= 8.0 * f64::EPSILON,
            "{earth_model:?}: cr_p={cr_p:?} vs ref={cr_ref:?} (rel={rel:e})"
        );
    }

    // 2. Finite position partials (peak |da/dr| ~ 4.6e303 < f64::MAX) while
    //    accel (~ 4.6e310) and C_R partial overflow binary64.
    let m_pos = SolarRadiationPressure::with_pressure(1.0, 1.0e160, 1.0e158, AU_KM)
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
    let sat_pos = Vector3::new(-1.0e10, 1.0e7, 0.0);
    let r_pen_pos = r_earth + 1.0e10 * tan_p;
    // Equatorial antumbra: nu = y / (R_e - x * tan_p),
    // d(nu)/dx = y * tan_p / r_pen^2, d(nu)/dy = 1 / r_pen, d(nu)/dz = 0.
    let nu_pos = ScaledF64::from_f64(1.0e7).div(ScaledF64::from_f64(r_pen_pos));
    let grad_nu_pos = [
        ScaledF64::from_f64(1.0e7)
            .mul(ScaledF64::from_f64(tan_p))
            .div(ScaledF64::from_f64(r_pen_pos).mul(ScaledF64::from_f64(r_pen_pos))),
        ScaledF64::from_f64(1.0).div(ScaledF64::from_f64(r_pen_pos)),
        ScaledF64::ZERO,
    ];
    let (acc_ref2, cr_ref2, jp_ref2) =
        independent_srp_reference(sat_pos, sun, nu_pos, grad_nu_pos, m_pos);
    assert!(acc_ref2.is_none());
    assert!(cr_ref2.is_none());
    let jp_ref2 = jp_ref2.expect("finite position partials reference");

    for earth_model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
        let model = m_pos.with_earth_shadow_model(earth_model);
        assert!(matches!(
            srp_acceleration_with_bodies(sat_pos, sun, None, model),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            srp_parameter_partial_with_bodies(sat_pos, sun, None, model),
            Err(PropagationError::NumericalFailure(_))
        ));
        let jp = srp_position_partials_with_bodies(sat_pos, sun, None, model)
            .expect("conical position partials must succeed when finite");
        let rel = scaled_rel_err_mat3(jp, jp_ref2);
        assert!(
            rel <= 8.0 * f64::EPSILON,
            "{earth_model:?}: rel={rel:e} jp={jp:?} ref={jp_ref2:?}"
        );
    }
}

#[test]
fn conical_overflowing_satellite_sun_distance_matches_independent_reference() {
    let base = SolarRadiationPressure::with_pressure(1.0, 1.0, 1.0, 1.0e154)
        .unwrap()
        .with_shadow_function(ShadowFunction::Conical);
    let sun = Vector3::new(1.34e154, 0.0, 0.0);
    let r_earth = crate::astro::constants::earth::MEAN_EARTH_RADIUS_KM;
    let r_sun = crate::astro::constants::astro::SOLAR_RADIUS_KM;
    let d_sun = 1.34e154_f64;
    let tan_u = libm::tan(libm::asin((r_sun - r_earth) / d_sun));
    let tan_p = libm::tan(libm::asin((r_sun + r_earth) / d_sun));

    // 1. Penumbra with |sun - sat| = 1.35e154 > sqrt(f64::MAX).
    let sat_pen = Vector3::new(-1.0e152, 5_000.0, 0.0);
    let r_u = r_earth - 1.0e152 * tan_u;
    let r_p = r_earth + 1.0e152 * tan_p;
    let den = r_p - r_u;
    let nu = (5_000.0 - r_u) / den;
    // Equatorial penumbra: nu = (y - r_u) / (r_p - r_u),
    // dr_u/dx = +tan_u, dr_p/dx = -tan_p, so
    // d(nu)/dx = (-tan_u * (r_p - r_u) - (y - r_u) * (-tan_p - tan_u)) / den^2.
    let dnu_dx = (-tan_u * den + (5_000.0 - r_u) * (tan_p + tan_u)) / (den * den);
    let dnu_dy = 1.0 / den;
    let (acc_ref, cr_ref, jp_ref) = independent_srp_reference(
        sat_pen,
        sun,
        ScaledF64::from_f64(nu),
        [
            ScaledF64::from_f64(dnu_dx),
            ScaledF64::from_f64(dnu_dy),
            ScaledF64::ZERO,
        ],
        base,
    );
    let acc_ref = acc_ref.expect("finite accel ref");
    let cr_ref = cr_ref.expect("finite cr ref");
    let jp_ref = jp_ref.expect("finite jp ref");

    // 2. Deep umbra with |sun - sat| = 1.35e154 > sqrt(f64::MAX).
    let sat_umb = Vector3::new(-1.0e152, 0.0, 0.0);

    for earth_model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
        let model = base.with_earth_shadow_model(earth_model);
        let acc = srp_acceleration_with_bodies(sat_pen, sun, None, model)
            .expect("penumbra accel with overflowing distance");
        let cr_p = srp_parameter_partial_with_bodies(sat_pen, sun, None, model)
            .expect("penumbra cr partial with overflowing distance");
        let jp = srp_position_partials_with_bodies(sat_pen, sun, None, model)
            .expect("penumbra position partials with overflowing distance");

        let rel_a = (acc - acc_ref).norm() / acc_ref.norm();
        let rel_c = (cr_p - cr_ref).norm() / cr_ref.norm();
        let rel_j = (jp - jp_ref).norm() / jp_ref.norm();
        assert!(
            rel_a <= 8.0 * f64::EPSILON
                && rel_c <= 8.0 * f64::EPSILON
                && rel_j <= 8.0 * f64::EPSILON,
            "{earth_model:?}: rel_a={rel_a:e}, rel_c={rel_c:e}, rel_j={rel_j:e}"
        );

        assert_eq!(
            srp_acceleration_with_bodies(sat_umb, sun, None, model).expect("deep umbra accel"),
            Vector3::zeros()
        );
        assert_eq!(
            srp_parameter_partial_with_bodies(sat_umb, sun, None, model)
                .expect("deep umbra cr partial"),
            Vector3::zeros()
        );
        assert_eq!(
            srp_position_partials_with_bodies(sat_umb, sun, None, model)
                .expect("deep umbra position partials"),
            Matrix3::zeros()
        );
    }
}

#[test]
fn orekit_path_nonfinite_intermediates_fall_back_to_independent_reference() {
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let moon = CLEAR_MOON_KM;
    let sat_lit = Vector3::new(7_000.0, 0.0, 0.0);
    let one = ScaledF64::from_f64(1.0);

    // 1–3. Sunlit cases where Orekit's k_ref, factor, or m/s^2 acceleration overflows
    //      while km/s^2 acceleration, position partials, and C_R partial are all finite.
    let sunlit_finite_cases = [
        (
            "k_ref_overflow",
            SolarRadiationPressure::with_pressure(1.0, 1.0e-300, 1.0e290, AU_KM).unwrap(),
        ),
        (
            "factor_overflow",
            SolarRadiationPressure::with_pressure(1.0e200, 1.0e200, 1.0e-300, AU_KM).unwrap(),
        ),
        (
            "a_m_overflow_a_km_finite",
            SolarRadiationPressure::with_pressure(1.0, 1.0e155, 1.0e155, AU_KM).unwrap(),
        ),
    ];
    for (label, base) in sunlit_finite_cases {
        let (acc_ref, cr_ref, jp_ref) =
            independent_srp_reference(sat_lit, sun, one, [ScaledF64::ZERO; 3], base);
        let acc_ref = acc_ref.expect("finite accel ref");
        let cr_ref = cr_ref.expect("finite cr ref");
        let jp_ref = jp_ref.expect("finite jp ref");
        for model in all_shadow_configs(base) {
            let moon_opt = model.moon_eclipse_enabled().then_some(moon);
            let acc = srp_acceleration_with_bodies(sat_lit, sun, moon_opt, model)
                .unwrap_or_else(|e| panic!("{label} {model:?} accel failed: {e:?}"));
            let cr_p = srp_parameter_partial_with_bodies(sat_lit, sun, moon_opt, model)
                .unwrap_or_else(|e| panic!("{label} {model:?} cr_p failed: {e:?}"));
            let jp = srp_position_partials_with_bodies(sat_lit, sun, moon_opt, model)
                .unwrap_or_else(|e| panic!("{label} {model:?} jp failed: {e:?}"));

            let rel_a = scaled_rel_err_vec3(acc, acc_ref);
            let rel_c = scaled_rel_err_vec3(cr_p, cr_ref);
            let rel_j = scaled_rel_err_mat3(jp, jp_ref);
            assert!(
                rel_a <= 8.0 * f64::EPSILON
                    && rel_c <= 8.0 * f64::EPSILON
                    && rel_j <= 8.0 * f64::EPSILON,
                "{label} {model:?}: rel_a={rel_a:e}, rel_c={rel_c:e}, rel_j={rel_j:e}"
            );
        }
    }

    // 4. Sunlit case where accel and da/dr overflow binary64 while d(accel)/d(C_R) is finite.
    let cr_only_base = SolarRadiationPressure::with_pressure(1.0e200, 1.0, 1.0e200, AU_KM).unwrap();
    let (acc_ref, cr_ref, jp_ref) =
        independent_srp_reference(sat_lit, sun, one, [ScaledF64::ZERO; 3], cr_only_base);
    assert!(acc_ref.is_none() && jp_ref.is_none());
    let cr_ref = cr_ref.expect("finite C_R ref");
    for model in all_shadow_configs(cr_only_base) {
        let moon_opt = model.moon_eclipse_enabled().then_some(moon);
        assert!(matches!(
            srp_acceleration_with_bodies(sat_lit, sun, moon_opt, model),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            srp_position_partials_with_bodies(sat_lit, sun, moon_opt, model),
            Err(PropagationError::NumericalFailure(_))
        ));
        let cr_p = srp_parameter_partial_with_bodies(sat_lit, sun, moon_opt, model)
            .expect("finite C_R partial when accel overflows");
        let rel_c = scaled_rel_err_vec3(cr_p, cr_ref);
        assert!(
            rel_c <= 8.0 * f64::EPSILON,
            "{model:?}: rel_c={rel_c:e} cr_p={cr_p:?} ref={cr_ref:?}"
        );
    }

    // 5. Sunlit r2 = (2e154 m)^2 = +Inf with finite km/s^2 acceleration (-2.5e-4) and partials.
    let r2_base = SolarRadiationPressure::with_pressure(1.0, 1.0, 1.0, 1.0e151).unwrap();
    let sat_r2 = Vector3::new(1.0e151, 0.0, 0.0);
    let sun_r2 = Vector3::new(3.0e151, 0.0, 0.0);
    let (acc_r2, cr_r2, jp_r2) =
        independent_srp_reference(sat_r2, sun_r2, one, [ScaledF64::ZERO; 3], r2_base);
    let acc_r2 = acc_r2.expect("finite r2 accel ref");
    let cr_r2 = cr_r2.expect("finite r2 cr ref");
    let jp_r2 = jp_r2.expect("finite r2 jp ref");
    for earth_model in [EarthShadowModel::Spherical, EarthShadowModel::Wgs84Oblate] {
        let model = r2_base
            .with_shadow_function(ShadowFunction::Conical)
            .with_earth_shadow_model(earth_model);
        let acc = srp_acceleration_with_bodies(sat_r2, sun_r2, None, model)
            .expect("sunlit conical accel with r2 overflow");
        let cr_p = srp_parameter_partial_with_bodies(sat_r2, sun_r2, None, model)
            .expect("sunlit conical cr_p with r2 overflow");
        let jp = srp_position_partials_with_bodies(sat_r2, sun_r2, None, model)
            .expect("sunlit conical jp with r2 overflow");
        let rel_a = scaled_rel_err_vec3(acc, acc_r2);
        let rel_c = scaled_rel_err_vec3(cr_p, cr_r2);
        let rel_j = scaled_rel_err_mat3(jp, jp_r2);
        assert!(
            rel_a <= 8.0 * f64::EPSILON
                && rel_c <= 8.0 * f64::EPSILON
                && rel_j <= 8.0 * f64::EPSILON,
            "r2 overflow {earth_model:?}: rel_a={rel_a:e}, rel_c={rel_c:e}, rel_j={rel_j:e}"
        );
    }
    // Under ApparentDisc, sun = [3e151, 0, 0] km overflows sun_m.norm() inside the ported
    // shadow function, so all three entry points consistently refuse it.
    for model in [
        r2_base,
        r2_base.with_moon_eclipse(false),
        r2_base.with_earth_shadow_model(EarthShadowModel::Spherical),
        r2_base
            .with_earth_shadow_model(EarthShadowModel::Spherical)
            .with_moon_eclipse(false),
    ] {
        let moon_opt = model.moon_eclipse_enabled().then_some(moon);
        assert!(srp_acceleration_with_bodies(sat_r2, sun_r2, moon_opt, model).is_err());
        assert!(srp_parameter_partial_with_bodies(sat_r2, sun_r2, moon_opt, model).is_err());
        assert!(srp_position_partials_with_bodies(sat_r2, sun_r2, moon_opt, model).is_err());
    }

    // 6. Exact-axis annular eclipses (Earth and lunar) use the double lighting ratio
    //    fallback for C_R partials and must also apply the scaled fallback on overflow.
    let annular_geoms = [
        ("earth_annular", Vector3::new(-2.0e6, 0.0, 0.0), sun, moon),
        (
            "lunar_annular",
            Vector3::new(7_000.0, 0.0, 0.0),
            Vector3::new(7_000.0, AU_KM, 0.0),
            Vector3::new(7_000.0, 405_000.0, 0.0),
        ),
    ];
    for (geom_label, sat_a, sun_a, moon_a) in annular_geoms {
        let ratio_d = lighting_ratio_of(
            SolarRadiationPressure::new(1.0, 1.0).unwrap(),
            sat_a,
            sun_a,
            Some(moon_a),
        )
        .expect("annular double ratio");
        assert!(ratio_d > 0.0 && ratio_d < 1.0);
        let nu_a = ScaledF64::from_f64(ratio_d);
        for (label, base) in sunlit_finite_cases {
            let (acc_ref, cr_ref, _) =
                independent_srp_reference(sat_a, sun_a, nu_a, [ScaledF64::ZERO; 3], base);
            let acc_ref = acc_ref.expect("finite annular accel ref");
            let cr_ref = cr_ref.expect("finite annular cr ref");
            let acc = srp_acceleration_with_bodies(sat_a, sun_a, Some(moon_a), base)
                .unwrap_or_else(|e| panic!("{geom_label} {label} accel failed: {e:?}"));
            let cr_p = srp_parameter_partial_with_bodies(sat_a, sun_a, Some(moon_a), base)
                .unwrap_or_else(|e| panic!("{geom_label} {label} cr_p failed: {e:?}"));
            assert!(matches!(
                srp_position_partials_with_bodies(sat_a, sun_a, Some(moon_a), base),
                Err(PropagationError::NumericalFailure(_))
            ));
            let rel_a = scaled_rel_err_vec3(acc, acc_ref);
            let rel_c = scaled_rel_err_vec3(cr_p, cr_ref);
            assert!(
                rel_a <= 8.0 * f64::EPSILON && rel_c <= 8.0 * f64::EPSILON,
                "{geom_label} {label}: rel_a={rel_a:e}, rel_c={rel_c:e}"
            );
        }
        // Annular value overflow with finite C_R partial.
        let (_, cr_ref, _) =
            independent_srp_reference(sat_a, sun_a, nu_a, [ScaledF64::ZERO; 3], cr_only_base);
        let cr_ref = cr_ref.expect("finite annular C_R ref");
        assert!(matches!(
            srp_acceleration_with_bodies(sat_a, sun_a, Some(moon_a), cr_only_base),
            Err(PropagationError::NumericalFailure(_))
        ));
        let cr_p = srp_parameter_partial_with_bodies(sat_a, sun_a, Some(moon_a), cr_only_base)
            .expect("annular C_R partial with accel overflow");
        let rel_c = scaled_rel_err_vec3(cr_p, cr_ref);
        assert!(
            rel_c <= 8.0 * f64::EPSILON,
            "{geom_label} cr_only: rel_c={rel_c:e}"
        );
    }

    // 7. ApparentDisc penumbra with overflowing k_ref and factor (verifying 1/m -> 1/km
    //    gradient scaling in ScaledF64).
    let sat_pen = Vector3::new(-26_000.0, 6_000.0, 2_100.0);
    for (label, base) in [sunlit_finite_cases[0], sunlit_finite_cases[1]] {
        for model in [
            base,
            base.with_moon_eclipse(false),
            base.with_earth_shadow_model(EarthShadowModel::Spherical),
            base.with_earth_shadow_model(EarthShadowModel::Spherical)
                .with_moon_eclipse(false),
        ] {
            let moon_opt = model.moon_eclipse_enabled().then_some(moon);
            let ratio_f = lighting_ratio_field_of(model, sat_pen, sun, moon_opt)
                .expect("penumbra field lighting ratio");
            let m_per_km = ScaledF64::from_f64(M_PER_KM);
            let grad_km = [
                ScaledF64::from_f64(ratio_f.grad[0]).mul(m_per_km),
                ScaledF64::from_f64(ratio_f.grad[1]).mul(m_per_km),
                ScaledF64::from_f64(ratio_f.grad[2]).mul(m_per_km),
            ];
            let (acc_ref, cr_ref, jp_ref) = independent_srp_reference(
                sat_pen,
                sun,
                ScaledF64::from_f64(ratio_f.value),
                grad_km,
                model,
            );
            let acc_ref = acc_ref.expect("penumbra acc ref");
            let cr_ref = cr_ref.expect("penumbra cr ref");
            let jp_ref = jp_ref.expect("penumbra jp ref");
            let acc = srp_acceleration_with_bodies(sat_pen, sun, moon_opt, model)
                .expect("penumbra fallback accel");
            let cr_p = srp_parameter_partial_with_bodies(sat_pen, sun, moon_opt, model)
                .expect("penumbra fallback cr_p");
            let jp = srp_position_partials_with_bodies(sat_pen, sun, moon_opt, model)
                .expect("penumbra fallback jp");
            // Double lighting ratio vs field lighting ratio differs by up to ~1e-11 in penumbra.
            let rel_a = scaled_rel_err_vec3(acc, acc_ref);
            let rel_c = scaled_rel_err_vec3(cr_p, cr_ref);
            let rel_j = scaled_rel_err_mat3(jp, jp_ref);
            assert!(
                rel_a <= 1.0e-10 && rel_c <= 8.0 * f64::EPSILON && rel_j <= 8.0 * f64::EPSILON,
                "penumbra {label} {model:?}: rel_a={rel_a:e}, rel_c={rel_c:e}, rel_j={rel_j:e}"
            );
        }
    }
}

/// Independent Orekit 13.1.10 `SolarRadiationPressure.acceleration` +
/// `IsotropicRadiationSingleCoefficient.radiationPressureAcceleration`
/// reference in raw binary64 arithmetic (km/s^2), returning `Some` iff the
/// unscaled Orekit evaluation is finite with `r2 > 0`.
fn orekit_order_srp_accel_km(
    sat_km: Vector3<f64>,
    sun_km: Vector3<f64>,
    ratio: f64,
    model: SolarRadiationPressure,
) -> Option<Vector3<f64>> {
    let d_ref = model.au_km() * M_PER_KM;
    let k_ref = model.pressure_n_m2() * d_ref * d_ref;
    let dx = sat_km.x * M_PER_KM - sun_km.x * M_PER_KM;
    let dy = sat_km.y * M_PER_KM - sun_km.y * M_PER_KM;
    let dz = sat_km.z * M_PER_KM - sun_km.z * M_PER_KM;
    let r2 = dx * dx + dy * dy + dz * dz;
    let raw_p = ratio * k_ref / r2;
    let scale = raw_p / r2.sqrt();
    let fx = scale * dx;
    let fy = scale * dy;
    let fz = scale * dz;
    let factor = 1.0 * model.area_to_mass_m2_kg() * model.cr() / 1.0;
    let ax = (factor * fx) / M_PER_KM;
    let ay = (factor * fy) / M_PER_KM;
    let az = (factor * fz) / M_PER_KM;
    (r2 > 0.0 && r2.is_finite() && ax.is_finite() && ay.is_finite() && az.is_finite())
        .then_some(Vector3::new(ax, ay, az))
}

/// Independent Orekit 13.1.10 `Field<Gradient>` `C_R` parameter partial
/// reference in raw `Grad<1>` arithmetic (km/s^2), returning `Some` iff the
/// unscaled Orekit field evaluation is finite with `r2.value > 0`.
fn orekit_order_srp_cr_km(
    sat_km: Vector3<f64>,
    sun_km: Vector3<f64>,
    ratio_val: f64,
    model: SolarRadiationPressure,
) -> Option<Vector3<f64>> {
    let d_ref = model.au_km() * M_PER_KM;
    let k_ref = model.pressure_n_m2() * d_ref * d_ref;
    let dx = Grad::<1>::constant(sat_km.x * M_PER_KM).sub(Grad::constant(sun_km.x * M_PER_KM));
    let dy = Grad::<1>::constant(sat_km.y * M_PER_KM).sub(Grad::constant(sun_km.y * M_PER_KM));
    let dz = Grad::<1>::constant(sat_km.z * M_PER_KM).sub(Grad::constant(sun_km.z * M_PER_KM));
    let r2 = dx.square().add(dy.square()).add(dz.square());
    let raw_p = Grad::<1>::constant(ratio_val).mul_f64(k_ref).div(r2);
    let scale = raw_p.div(r2.sqrt());
    let fx = scale.mul(dx);
    let fy = scale.mul(dy);
    let fz = scale.mul(dz);
    let factor = Grad::<1>::ONE
        .reciprocal()
        .mul(Grad::ONE)
        .mul_f64(model.area_to_mass_m2_kg())
        .mul(Grad::variable(0, model.cr()));
    let ax = factor.mul(fx);
    let ay = factor.mul(fy);
    let az = factor.mul(fz);
    if !(r2.value > 0.0
        && r2.is_all_finite()
        && ax.is_all_finite()
        && ay.is_all_finite()
        && az.is_all_finite())
    {
        return None;
    }
    let c_km = Vector3::new(
        ax.grad[0] / M_PER_KM,
        ay.grad[0] / M_PER_KM,
        az.grad[0] / M_PER_KM,
    );
    c_km.iter().all(|c| c.is_finite()).then_some(c_km)
}

/// Independent Orekit 13.1.10 `Field<Gradient>` position Jacobian reference
/// in raw `Grad<3>` arithmetic (1/s^2), returning `Some` iff the unscaled
/// Orekit field evaluation is finite with `r2.value > 0`.
fn orekit_order_srp_pos_partials(
    sat_km: Vector3<f64>,
    sun_km: Vector3<f64>,
    ratio: Grad,
    model: SolarRadiationPressure,
) -> Option<Matrix3<f64>> {
    let d_ref = model.au_km() * M_PER_KM;
    let k_ref = model.pressure_n_m2() * d_ref * d_ref;
    let dx = Grad::<3>::variable(0, sat_km.x * M_PER_KM).sub(Grad::constant(sun_km.x * M_PER_KM));
    let dy = Grad::<3>::variable(1, sat_km.y * M_PER_KM).sub(Grad::constant(sun_km.y * M_PER_KM));
    let dz = Grad::<3>::variable(2, sat_km.z * M_PER_KM).sub(Grad::constant(sun_km.z * M_PER_KM));
    let r2 = dx.square().add(dy.square()).add(dz.square());
    let raw_p = ratio.mul_f64(k_ref).div(r2);
    let scale = raw_p.div(r2.sqrt());
    let fx = scale.mul(dx);
    let fy = scale.mul(dy);
    let fz = scale.mul(dz);
    let factor = Grad::<3>::ONE
        .reciprocal()
        .mul(Grad::ONE)
        .mul_f64(model.area_to_mass_m2_kg())
        .mul(Grad::constant(model.cr()));
    let ax = factor.mul(fx);
    let ay = factor.mul(fy);
    let az = factor.mul(fz);
    (r2.value > 0.0
        && r2.is_all_finite()
        && ax.is_all_finite()
        && ay.is_all_finite()
        && az.is_all_finite())
    .then(|| {
        Matrix3::from_rows(&[
            RowVector3::from(ax.grad),
            RowVector3::from(ay.grad),
            RowVector3::from(az.grad),
        ])
    })
}

#[test]
fn orekit_path_finite_underflow_to_zero_preserves_orekit_bits() {
    // Finite Orekit evaluation where raw_p / sqrt(r2) underflows to 0.0 while
    // factor = 1e300 is finite: Orekit's exact bits (signed zeros included) are
    // preserved.
    let base = SolarRadiationPressure::with_pressure(1.0, 1.0e300, 1.0e-315, AU_KM).unwrap();
    let sat = Vector3::new(7_000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let moon = CLEAR_MOON_KM;

    let orekit_km =
        orekit_order_srp_accel_km(sat, sun, 1.0, base).expect("finite Orekit-order double");
    let orekit_cr =
        orekit_order_srp_cr_km(sat, sun, 1.0, base).expect("finite Orekit-order C_R field");
    let orekit_mat = orekit_order_srp_pos_partials(sat, sun, Grad::constant(1.0), base)
        .expect("finite Orekit-order position field");

    // Expected zero bits from raw Orekit arithmetic (signed -0.0 on acc.x from
    // +0.0 * (sat.x - sun.x) < 0; +0.0 on acc.y/z, C_R partial, and position
    // partials where -0.0 + +0.0 = +0.0 in IEEE 754 round-to-nearest-even).
    assert_eq!(orekit_km.x.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(orekit_km.y.to_bits(), 0.0_f64.to_bits());
    assert_eq!(orekit_km.z.to_bits(), 0.0_f64.to_bits());
    assert_eq!(orekit_cr.x.to_bits(), 0.0_f64.to_bits());
    assert_eq!(orekit_cr.y.to_bits(), 0.0_f64.to_bits());
    assert_eq!(orekit_cr.z.to_bits(), 0.0_f64.to_bits());
    for row in 0..3 {
        for col in 0..3 {
            assert_eq!(
                orekit_mat[(row, col)].to_bits(),
                0.0_f64.to_bits(),
                "Orekit-order reference pos partial ({row},{col})"
            );
        }
    }

    for model in all_shadow_configs(base) {
        let moon_opt = model.moon_eclipse_enabled().then_some(moon);
        let acc = srp_acceleration_with_bodies(sat, sun, moon_opt, model).expect("underflow accel");
        let cr_p =
            srp_parameter_partial_with_bodies(sat, sun, moon_opt, model).expect("underflow cr_p");
        let jp =
            srp_position_partials_with_bodies(sat, sun, moon_opt, model).expect("underflow jp");

        assert_eq!(acc, Vector3::zeros());
        assert_eq!(acc.x.to_bits(), orekit_km.x.to_bits());
        assert_eq!(acc.y.to_bits(), orekit_km.y.to_bits());
        assert_eq!(acc.z.to_bits(), orekit_km.z.to_bits());

        assert_eq!(cr_p, Vector3::zeros());
        assert_eq!(cr_p.x.to_bits(), orekit_cr.x.to_bits());
        assert_eq!(cr_p.y.to_bits(), orekit_cr.y.to_bits());
        assert_eq!(cr_p.z.to_bits(), orekit_cr.z.to_bits());

        assert_eq!(jp, Matrix3::zeros());
        for row in 0..3 {
            for col in 0..3 {
                assert_eq!(
                    jp[(row, col)].to_bits(),
                    orekit_mat[(row, col)].to_bits(),
                    "{model:?} ({row},{col})"
                );
            }
        }
    }
}

#[test]
fn extreme_parameter_and_distance_sweep_preserves_orekit_bits_and_matches_reference() {
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
        fn unit_vec(&mut self) -> Vector3<f64> {
            let z = 2.0 * self.unit() - 1.0;
            let phi = std::f64::consts::TAU * self.unit();
            let r = (1.0 - z * z).max(0.0).sqrt();
            Vector3::new(r * libm::cos(phi), r * libm::sin(phi), z)
        }
    }

    let mut rng = SplitMix64(0x5a9c_fa11_2026_0042);
    let config_specs: [(ShadowFunction, EarthShadowModel, bool); 6] = [
        (
            ShadowFunction::ApparentDisc,
            EarthShadowModel::Wgs84Oblate,
            true,
        ),
        (
            ShadowFunction::ApparentDisc,
            EarthShadowModel::Wgs84Oblate,
            false,
        ),
        (
            ShadowFunction::ApparentDisc,
            EarthShadowModel::Spherical,
            true,
        ),
        (
            ShadowFunction::ApparentDisc,
            EarthShadowModel::Spherical,
            false,
        ),
        (
            ShadowFunction::Conical,
            EarthShadowModel::Wgs84Oblate,
            false,
        ),
        (ShadowFunction::Conical, EarthShadowModel::Spherical, false),
    ];

    let mut orekit_acc_exact = 0usize;
    let mut orekit_cr_exact = 0usize;
    let mut orekit_jp_exact = 0usize;
    let mut fallback_acc_checked = 0usize;
    let mut fallback_cr_checked = 0usize;
    let mut fallback_jp_checked = 0usize;
    let mut value_overflow_cr_ok = 0usize;
    let mut value_overflow_jp_ok = 0usize;
    let mut max_acc_eps = 0.0_f64;
    let mut max_cr_eps = 0.0_f64;
    let mut max_jp_eps = 0.0_f64;
    let m_per_km = ScaledF64::from_f64(M_PER_KM);

    for (shadow_fn, earth_model, include_moon) in config_specs {
        let is_conical = matches!(shadow_fn, ShadowFunction::Conical);
        for case in 0..160 {
            let s_dir = rng.unit_vec();
            let perp = s_dir
                .cross(&Vector3::z())
                .try_normalize(1.0e-12)
                .unwrap_or_else(Vector3::x);

            // Geometry regime covering sunlit, umbra, penumbra, distant/overflowing r2,
            // and conical antumbra.
            let (sat, sun, moon) = match case % 8 {
                // Sunlit LEO/MEO
                0 => (
                    s_dir * rng.log_uniform(13, 20),
                    s_dir * AU_KM,
                    CLEAR_MOON_KM,
                ),
                // Deep umbra
                1 => (-s_dir * 7_000.0, s_dir * AU_KM, CLEAR_MOON_KM),
                // Interior Earth penumbra
                2 => (
                    -s_dir * 26_560.0 + perp * 6_365.0,
                    s_dir * AU_KM,
                    CLEAR_MOON_KM,
                ),
                // Distant sunlit / overflowing r2
                3 => {
                    let s_scale = if is_conical {
                        rng.log_uniform(490, 515)
                    } else {
                        rng.log_uniform(480, 504)
                    };
                    (
                        s_dir * (0.25 * s_scale) + perp * (0.1 * s_scale),
                        s_dir * s_scale,
                        CLEAR_MOON_KM,
                    )
                }
                // Conical antumbra or lunar partial
                4 => {
                    if is_conical {
                        (
                            -s_dir * rng.log_uniform(20, 34) + perp * rng.log_uniform(12, 24),
                            s_dir * AU_KM,
                            CLEAR_MOON_KM,
                        )
                    } else {
                        (
                            Vector3::new(7_000.0, 0.0, 0.0),
                            Vector3::new(7_000.0, AU_KM, 0.0),
                            Vector3::new(7_000.0, 384_400.0, 1_800.0),
                        )
                    }
                }
                // Exact-axis annular (Earth) for ApparentDisc, off-axis penumbra for Conical
                5 => (
                    -s_dir * 2.0e6
                        + if is_conical {
                            perp * 5_000.0
                        } else {
                            Vector3::zeros()
                        },
                    s_dir * AU_KM,
                    CLEAR_MOON_KM,
                ),
                // General random sunlit geometry at wide distances
                6 => {
                    let r_mag = rng.log_uniform(13, 495);
                    let s_mag = rng.log_uniform(24, 495);
                    (rng.unit_vec() * r_mag, s_dir * s_mag, CLEAR_MOON_KM)
                }
                // Very large sat/sun near binary64 norm boundary (including r2 overflow)
                _ => {
                    let s_mag = if is_conical { 1.34e154 } else { 1.0e150 };
                    (
                        -s_dir * (s_mag * 0.01) + perp * 5_000.0,
                        s_dir * s_mag,
                        CLEAR_MOON_KM,
                    )
                }
            };

            // Parameter regime covering finite Orekit, subnormal/underflow Orekit,
            // overflowing k_ref/factor/a_m, and value-overflow with finite partials.
            let r_approx = scaled_norm_vec3(sat - sun).max(1.0);
            let (cr, area, pressure, au_km) = match (case / 8) % 5 {
                // Finite Orekit (including normal & subnormal intermediates)
                0 => (
                    rng.log_uniform(-30, 30),
                    rng.log_uniform(-30, 30),
                    rng.log_uniform(-60, 20),
                    AU_KM,
                ),
                // Overflowing k_ref or factor with compensating finite output
                1 => {
                    if case % 2 == 0 {
                        (
                            1.0,
                            rng.log_uniform(-1000, -850),
                            rng.log_uniform(900, 1000),
                            rng.log_uniform(20, 400),
                        )
                    } else {
                        (
                            rng.log_uniform(550, 700),
                            rng.log_uniform(550, 700),
                            rng.log_uniform(-1000, -800),
                            AU_KM,
                        )
                    }
                }
                // a_m overflows while a_km and partials stay finite (1e307..1e308)
                2 => (
                    1.0,
                    libm::ldexp(1.0 + rng.unit(), 515),
                    libm::ldexp(1.0 + rng.unit(), 515),
                    r_approx,
                ),
                // Value overflows while C_R partial or position partials stay finite
                3 => {
                    if case % 2 == 0 {
                        // Large C_R -> accel overflows, C_R partial finite
                        (
                            libm::ldexp(1.0 + rng.unit(), 660),
                            1.0,
                            libm::ldexp(1.0 + rng.unit(), 660),
                            r_approx,
                        )
                    } else {
                        // |accel| ~ 2^1035 overflows while |accel| / r ~ 2^1010 stays finite
                        (
                            1.0,
                            libm::ldexp(1.0 + rng.unit(), 520),
                            libm::ldexp(1.0 + rng.unit(), 520),
                            r_approx.max(1.0e6),
                        )
                    }
                }
                // Wide extreme log-uniform sweep across the full binary64 range
                _ => (
                    rng.log_uniform(-900, 900),
                    rng.log_uniform(-900, 900),
                    rng.log_uniform(-900, 900),
                    rng.log_uniform(-300, 500),
                ),
            };

            let mut model = SolarRadiationPressure::with_pressure(cr, area, pressure, au_km)
                .unwrap()
                .with_shadow_function(shadow_fn)
                .with_earth_shadow_model(earth_model);
            if !is_conical {
                model = model.with_moon_eclipse(include_moon);
            }
            let moon_opt = model.moon_eclipse_enabled().then_some(moon);

            let acc = srp_acceleration_with_bodies(sat, sun, moon_opt, model);
            let cr_p = srp_parameter_partial_with_bodies(sat, sun, moon_opt, model);
            let pos_p = srp_position_partials_with_bodies(sat, sun, moon_opt, model);
            let acc_is_err = acc.is_err();

            if is_conical {
                let sh_d =
                    conical_lighting_ratio_double_scaled(as_array(sat), as_array(sun), earth_model);
                let sh_f =
                    conical_lighting_ratio_field_scaled(as_array(sat), as_array(sun), earth_model);
                match sh_d {
                    Err(_) => {
                        assert!(acc_is_err && cr_p.is_err() && pos_p.is_err());
                    }
                    Ok((nu_s, is_sunlit)) => {
                        let (ratio_km, _) = sh_f.expect("conical field matches double domain");
                        let grad_ok = ratio_km.is_all_finite();
                        let (acc_ref, cr_ref, jp_ref) = if grad_ok {
                            independent_srp_reference(sat, sun, nu_s, ratio_km.grad, model)
                        } else {
                            let (a, c, _) = independent_srp_reference(
                                sat,
                                sun,
                                nu_s,
                                [ScaledF64::ZERO; 3],
                                model,
                            );
                            (a, c, None)
                        };

                        if is_sunlit {
                            if let Some(a_km) = orekit_order_srp_accel_km(sat, sun, 1.0, model) {
                                let got = acc.expect("finite Orekit accel");
                                assert_eq!(got.x.to_bits(), a_km.x.to_bits());
                                assert_eq!(got.y.to_bits(), a_km.y.to_bits());
                                assert_eq!(got.z.to_bits(), a_km.z.to_bits());
                                orekit_acc_exact += 1;
                            } else {
                                assert_eq!(acc.is_ok(), acc_ref.is_some());
                                if let (Ok(got), Some(r)) = (&acc, acc_ref) {
                                    if max_abs_vec3(r) >= f64::MIN_POSITIVE * 1.0e6 {
                                        max_acc_eps = max_acc_eps
                                            .max(scaled_rel_err_vec3(*got, r) / f64::EPSILON);
                                        fallback_acc_checked += 1;
                                    }
                                }
                            }
                            if let Some(c_km) = orekit_order_srp_cr_km(sat, sun, 1.0, model) {
                                let got = cr_p.expect("finite Orekit C_R");
                                assert_eq!(got.x.to_bits(), c_km.x.to_bits());
                                assert_eq!(got.y.to_bits(), c_km.y.to_bits());
                                assert_eq!(got.z.to_bits(), c_km.z.to_bits());
                                orekit_cr_exact += 1;
                            } else {
                                assert_eq!(cr_p.is_ok(), cr_ref.is_some());
                                if acc_is_err && cr_p.is_ok() {
                                    value_overflow_cr_ok += 1;
                                }
                                if let (Ok(got), Some(r)) = (&cr_p, cr_ref) {
                                    if max_abs_vec3(r) >= f64::MIN_POSITIVE * 1.0e6 {
                                        max_cr_eps = max_cr_eps
                                            .max(scaled_rel_err_vec3(*got, r) / f64::EPSILON);
                                        fallback_cr_checked += 1;
                                    }
                                }
                            }
                            if let Some(expected) =
                                orekit_order_srp_pos_partials(sat, sun, Grad::constant(1.0), model)
                            {
                                let got = pos_p.expect("finite Orekit pos_p");
                                for row in 0..3 {
                                    for col in 0..3 {
                                        assert_eq!(
                                            got[(row, col)].to_bits(),
                                            expected[(row, col)].to_bits()
                                        );
                                    }
                                }
                                orekit_jp_exact += 1;
                            } else {
                                assert_eq!(pos_p.is_ok(), jp_ref.is_some());
                                if acc_is_err && pos_p.is_ok() {
                                    value_overflow_jp_ok += 1;
                                }
                                if let (Ok(got), Some(r)) = (&pos_p, jp_ref) {
                                    if max_abs_mat3(r) >= f64::MIN_POSITIVE * 1.0e6 {
                                        max_jp_eps = max_jp_eps
                                            .max(scaled_rel_err_mat3(*got, r) / f64::EPSILON);
                                        fallback_jp_checked += 1;
                                    }
                                }
                            }
                        } else {
                            // Non-sunlit conical: scaled throughout.
                            assert_eq!(acc.is_ok(), acc_ref.is_some());
                            assert_eq!(cr_p.is_ok(), cr_ref.is_some());
                            assert_eq!(pos_p.is_ok(), jp_ref.is_some());
                            if acc_is_err && cr_p.is_ok() {
                                value_overflow_cr_ok += 1;
                            }
                            if acc_is_err && pos_p.is_ok() {
                                value_overflow_jp_ok += 1;
                            }
                            if let (Ok(got), Some(r)) = (&acc, acc_ref) {
                                if max_abs_vec3(r) >= f64::MIN_POSITIVE * 1.0e6 {
                                    max_acc_eps = max_acc_eps
                                        .max(scaled_rel_err_vec3(*got, r) / f64::EPSILON);
                                    fallback_acc_checked += 1;
                                }
                            }
                            if let (Ok(got), Some(r)) = (&cr_p, cr_ref) {
                                if max_abs_vec3(r) >= f64::MIN_POSITIVE * 1.0e6 {
                                    max_cr_eps =
                                        max_cr_eps.max(scaled_rel_err_vec3(*got, r) / f64::EPSILON);
                                    fallback_cr_checked += 1;
                                }
                            }
                            if let (Ok(got), Some(r)) = (&pos_p, jp_ref) {
                                if max_abs_mat3(r) >= f64::MIN_POSITIVE * 1.0e6 {
                                    max_jp_eps =
                                        max_jp_eps.max(scaled_rel_err_mat3(*got, r) / f64::EPSILON);
                                    fallback_jp_checked += 1;
                                }
                            }
                        }
                    }
                }
            } else {
                let lr_d = lighting_ratio_of(model, sat, sun, moon_opt);
                let lr_f = lighting_ratio_field_of(model, sat, sun, moon_opt);
                let lr_cr = lighting_ratio_for_parameter_partial(model, sat, sun, moon_opt);
                match lr_d {
                    Err(_) => {
                        assert!(acc_is_err && cr_p.is_err() && pos_p.is_err());
                    }
                    Ok(ratio_d) => {
                        let (acc_ref, _, _) = independent_srp_reference(
                            sat,
                            sun,
                            ScaledF64::from_f64(ratio_d),
                            [ScaledF64::ZERO; 3],
                            model,
                        );
                        if let Some(a_km) = orekit_order_srp_accel_km(sat, sun, ratio_d, model) {
                            let got = acc.expect("finite Orekit ApparentDisc accel");
                            assert_eq!(got.x.to_bits(), a_km.x.to_bits());
                            assert_eq!(got.y.to_bits(), a_km.y.to_bits());
                            assert_eq!(got.z.to_bits(), a_km.z.to_bits());
                            orekit_acc_exact += 1;
                        } else {
                            assert_eq!(acc.is_ok(), acc_ref.is_some());
                            if let (Ok(got), Some(r)) = (&acc, acc_ref) {
                                if max_abs_vec3(r) >= f64::MIN_POSITIVE * 1.0e6 {
                                    max_acc_eps = max_acc_eps
                                        .max(scaled_rel_err_vec3(*got, r) / f64::EPSILON);
                                    fallback_acc_checked += 1;
                                }
                            }
                        }

                        let ratio_cr =
                            lr_cr.expect("C_R lighting ratio succeeds when double succeeds");
                        let (_, cr_ref, _) = independent_srp_reference(
                            sat,
                            sun,
                            ScaledF64::from_f64(ratio_cr),
                            [ScaledF64::ZERO; 3],
                            model,
                        );
                        if let Some(c_km) = orekit_order_srp_cr_km(sat, sun, ratio_cr, model) {
                            let got = cr_p.expect("finite Orekit ApparentDisc C_R");
                            assert_eq!(got.x.to_bits(), c_km.x.to_bits());
                            assert_eq!(got.y.to_bits(), c_km.y.to_bits());
                            assert_eq!(got.z.to_bits(), c_km.z.to_bits());
                            orekit_cr_exact += 1;
                        } else {
                            assert_eq!(cr_p.is_ok(), cr_ref.is_some());
                            if acc_is_err && cr_p.is_ok() {
                                value_overflow_cr_ok += 1;
                            }
                            if let (Ok(got), Some(r)) = (&cr_p, cr_ref) {
                                if max_abs_vec3(r) >= f64::MIN_POSITIVE * 1.0e6 {
                                    max_cr_eps =
                                        max_cr_eps.max(scaled_rel_err_vec3(*got, r) / f64::EPSILON);
                                    fallback_cr_checked += 1;
                                }
                            }
                        }

                        match lr_f {
                            Err(_) => {
                                assert!(matches!(
                                    pos_p,
                                    Err(PropagationError::NumericalFailure(_))
                                ));
                            }
                            Ok(ratio_f) => {
                                let grad_km = [
                                    ScaledF64::from_f64(ratio_f.grad[0]).mul(m_per_km),
                                    ScaledF64::from_f64(ratio_f.grad[1]).mul(m_per_km),
                                    ScaledF64::from_f64(ratio_f.grad[2]).mul(m_per_km),
                                ];
                                let (_, _, jp_ref) = independent_srp_reference(
                                    sat,
                                    sun,
                                    ScaledF64::from_f64(ratio_f.value),
                                    grad_km,
                                    model,
                                );
                                if let Some(expected) =
                                    orekit_order_srp_pos_partials(sat, sun, ratio_f, model)
                                {
                                    let got = pos_p.expect("finite Orekit ApparentDisc pos_p");
                                    for row in 0..3 {
                                        for col in 0..3 {
                                            assert_eq!(
                                                got[(row, col)].to_bits(),
                                                expected[(row, col)].to_bits()
                                            );
                                        }
                                    }
                                    orekit_jp_exact += 1;
                                } else {
                                    assert_eq!(pos_p.is_ok(), jp_ref.is_some());
                                    if acc_is_err && pos_p.is_ok() {
                                        value_overflow_jp_ok += 1;
                                    }
                                    if let (Ok(got), Some(r)) = (&pos_p, jp_ref) {
                                        if max_abs_mat3(r) >= f64::MIN_POSITIVE * 1.0e6 {
                                            max_jp_eps = max_jp_eps
                                                .max(scaled_rel_err_mat3(*got, r) / f64::EPSILON);
                                            fallback_jp_checked += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(
        orekit_acc_exact >= 150 && orekit_cr_exact >= 150 && orekit_jp_exact >= 120,
        "insufficient Orekit bit-exact hits: acc={orekit_acc_exact}, cr={orekit_cr_exact}, jp={orekit_jp_exact}"
    );
    assert!(
        fallback_acc_checked >= 150 && fallback_cr_checked >= 200 && fallback_jp_checked >= 150,
        "insufficient fallback checks: acc={fallback_acc_checked}, cr={fallback_cr_checked}, jp={fallback_jp_checked}"
    );
    assert!(
        value_overflow_cr_ok >= 20 && value_overflow_jp_ok >= 15,
        "insufficient value-overflow-finite-partial hits: cr={value_overflow_cr_ok}, jp={value_overflow_jp_ok}"
    );
    // Measured maxima across 960 extreme cases (6 shadow configs x 160 cases;
    // 304/303/287 finite Orekit bit-exact hits; 346/417/375 fallback/scaled
    // checks; 104 C_R and 61 position value-overflow exceptions):
    // acc = 2.154 eps, cr = 2.706 eps, jp = 2.471 eps (caused by ScaledF64
    // rounding differences between (au^2 / r^2) and factored (au / r)^2 in the
    // independent reference, u_d normalization, and / M_PER_KM).
    assert!(
        max_acc_eps <= 8.0 && max_cr_eps <= 8.0 && max_jp_eps <= 8.0,
        "extreme SRP sweep maxima: acc={max_acc_eps:.3} eps, cr={max_cr_eps:.3} eps, jp={max_jp_eps:.3} eps"
    );
}

#[test]
fn scaled_rel_err_helpers_reject_non_finite_outputs() {
    let finite_v = Vector3::new(1.0, -2.0, 3.0);
    let zero_v = Vector3::zeros();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let bad_v = Vector3::new(bad, 0.0, 0.0);
        for expected in [finite_v, zero_v] {
            let err = scaled_rel_err_vec3(bad_v, expected);
            assert_eq!(err, f64::INFINITY);
            assert!(err > 8.0 * f64::EPSILON);
            assert!(0.0_f64.max(err) > 8.0);
        }
    }

    let finite_m = Matrix3::identity();
    let zero_m = Matrix3::zeros();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad_m = Matrix3::zeros();
        bad_m[(1, 2)] = bad;
        for expected in [finite_m, zero_m] {
            let err = scaled_rel_err_mat3(bad_m, expected);
            assert_eq!(err, f64::INFINITY);
            assert!(err > 8.0 * f64::EPSILON);
            assert!(0.0_f64.max(err) > 8.0);
        }
    }
}
