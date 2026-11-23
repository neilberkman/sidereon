//! Clean-room validation anchors:
//!
//! The expected 1 AU acceleration magnitude is the published cannonball SRP
//! pressure `4.56e-6 N/m^2` multiplied by `C_R * A/m` and converted from
//! m/s^2 to km/s^2, with the satellite one AU from the Sun. The shadow tests
//! use fixed conical geometry and the eclipse helpers, whose return value is
//! the geometric shadow fraction. The position partials are checked against
//! difference quotients of the acceleration under each shadow model: the
//! default conical Earth, the oblate conical Earth, and the apparent-disc
//! occultation with the Moon alone, the Earth and Moon together, and the
//! oblate Earth with the Moon.

use super::*;
use crate::astro::constants::astro::{AU_KM, IAU_2015_SOLAR_RADIUS_KM, MOON_RADIUS_KM};
use crate::astro::events::eclipse::{
    shadow_fraction, shadow_fraction_with_model_and_gradient, shadow_fraction_with_model_and_moon,
    shadow_fraction_with_model_moon_and_gradient, ShadowGradientError,
};
use crate::astro::forces::test_helpers::{
    epochs_1990_to_2050, gcrs_sun_moon_km, mean_of_date_sun_moon_km,
};

fn sat_sun(sat: Vector3<f64>, sun: Vector3<f64>) -> ([f64; 3], [f64; 3]) {
    (as_array(sat), as_array(sun))
}

#[test]
fn one_au_magnitude_matches_pressure_area_mass_scale() {
    let model = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP");
    let sat = Vector3::new(7000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM + 7000.0, 0.0, 0.0);
    let accel = srp_acceleration_with_bodies(sat, sun, None, model).expect("SRP acceleration");

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
    let near_accel = srp_acceleration_with_bodies(near, sun, None, model).expect("near");
    let far_accel = srp_acceleration_with_bodies(far, sun, None, model).expect("far");
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
    let accel = srp_acceleration_with_bodies(sat, sun, None, model).expect("SRP acceleration");

    assert_eq!(accel, Vector3::zeros());
}

#[test]
fn direction_is_away_from_sun() {
    let model = SolarRadiationPressure::new(1.0, 0.01).expect("valid SRP");
    let sat = Vector3::new(7000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let accel = srp_acceleration_with_bodies(sat, sun, None, model).expect("SRP acceleration");

    assert!(accel.x < 0.0);
    assert_eq!(accel.y, 0.0);
    assert_eq!(accel.z, 0.0);
}

#[test]
fn moon_eclipse_is_opt_in_and_attenuates_srp() {
    let default_model = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP");
    assert!(!default_model.moon_eclipse_enabled());
    assert_eq!(
        default_model.earth_shadow_model(),
        EarthShadowModel::Spherical
    );

    let moon_model = default_model.with_moon_eclipse(true);
    assert!(moon_model.moon_eclipse_enabled());

    let sat = Vector3::new(7_000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let moon_clear = Vector3::new(0.0, 384_400.0, 0.0);
    let moon_total = Vector3::new(7_000.0 + 356_000.0, 0.0, 0.0);
    let moon_annular = Vector3::new(7_000.0 + 405_000.0, 0.0, 0.0);

    // Clear Moon matches default bit-for-bit in full sunlight.
    let accel_default = default_model
        .acceleration_with_sun_moon(sat, sun, moon_clear)
        .expect("default SRP");
    let accel_clear = moon_model
        .acceleration_with_sun_moon(sat, sun, moon_clear)
        .expect("moon-enabled clear SRP");
    for i in 0..3 {
        assert_eq!(accel_clear[i].to_bits(), accel_default[i].to_bits());
    }

    // Default ignores lunar umbra; opt-in zeroes acceleration.
    let accel_default_total = default_model
        .acceleration_with_sun_moon(sat, sun, moon_total)
        .expect("default ignores moon");
    for i in 0..3 {
        assert_eq!(accel_default_total[i].to_bits(), accel_default[i].to_bits());
    }
    let accel_total = moon_model
        .acceleration_with_sun_moon(sat, sun, moon_total)
        .expect("lunar total SRP");
    assert_eq!(accel_total, Vector3::zeros());

    // Annular lunar eclipse reduces SRP without zeroing it.
    let accel_annular = moon_model
        .acceleration_with_sun_moon(sat, sun, moon_annular)
        .expect("lunar annular SRP");
    assert!(accel_annular.x < 0.0);
    assert!(accel_annular.norm() < 0.25 * accel_clear.norm());
    assert!(accel_annular.norm() > 0.1 * accel_clear.norm());

    // Non-finite Moon position returns a typed error.
    let err = moon_model
        .acceleration_with_sun_moon(sat, sun, Vector3::new(f64::NAN, 0.0, 0.0))
        .expect_err("non-finite moon rejected");
    assert!(matches!(err, PropagationError::InvalidInput(_)));
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
        let scale = assert_partials_match_differences("sunlight", model, sat, sun, None, 1.0);
        assert!(scale > 1.0e-19);
    }
}

#[test]
fn position_partials_match_difference_quotients_in_the_penumbra() {
    let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    // 26000 km behind the Earth, where the umbra cone has a radius of
    // about 6250 km and the penumbra cone about 6490 km; the satellite is
    // 6357 km from the shadow axis.
    let sat = Vector3::new(-26_000.0, 6_000.0, 2_100.0);
    let (sat_a, sun_a) = sat_sun(sat, sun);
    let shadow = shadow_fraction(sat_a, sun_a).expect("shadow");
    assert!(shadow > 0.0 && shadow < 1.0, "shadow {shadow}");
    // Inside the penumbra the shadow term dominates: the fraction changes
    // by 1 over the 240 km between the cones, so the gradient is of order
    // |a| / 240 km = 5e-13 1/s^2. A 1 m step stays 100 km from either cone.
    let scale = assert_partials_match_differences("penumbra", model, sat, sun, None, 1.0e-3);
    assert!(scale > 1.0e-13);
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
    let model = SolarRadiationPressure::new(1.3, 0.02)
        .expect("valid SRP")
        .with_moon_eclipse(true);
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
        EarthShadowModel::Spherical,
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
        .with_moon_eclipse(true);
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
    let model = SolarRadiationPressure::new(1.3, 0.02)
        .expect("valid SRP")
        .with_earth_shadow_model(EarthShadowModel::Wgs84Oblate)
        .with_moon_eclipse(true);
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
    let moon = Vector3::new(0.0, 384_400.0, 0.0);
    let base = SolarRadiationPressure::new(1.3, 0.02)
        .expect("valid SRP")
        .with_moon_eclipse(true);
    for model in [
        base,
        base.with_earth_shadow_model(EarthShadowModel::Wgs84Oblate),
    ] {
        let accel = srp_acceleration_with_bodies(sat, sun, Some(moon), model).expect("accel");
        assert!(accel.norm() > 0.0);
        let refused = srp_position_partials_with_bodies(sat, sun, Some(moon), model);
        assert!(
            matches!(refused, Err(PropagationError::NumericalFailure(_))),
            "{model:?}: expected NumericalFailure, got {refused:?}"
        );
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

#[test]
fn partials_report_numerical_failure_on_conical_antumbra_axis() {
    // Past the tip of the Earth's umbra on the exact shadow axis, rho == 0
    // produces NaN in the square-root derivative.
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let sat = Vector3::new(-2.0e6, 0.0, 0.0);
    let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    for model in [
        model,
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
    let moon = Vector3::new(0.0, 384_400.0, 0.0);
    let model = SolarRadiationPressure::new(1.3, 0.02)
        .expect("valid SRP")
        .with_earth_shadow_model(EarthShadowModel::Wgs84Oblate)
        .with_moon_eclipse(true);
    assert!(srp_acceleration_with_bodies(sat, sun, Some(moon), model).is_ok());
    let refused = srp_position_partials_with_bodies(sat, sun, Some(moon), model);
    assert!(
        matches!(refused, Err(PropagationError::NumericalFailure(_))),
        "{refused:?}"
    );
}

#[test]
fn position_partials_follow_the_oblate_limb_off_the_meridian() {
    let model = SolarRadiationPressure::new(1.3, 0.02)
        .expect("valid SRP")
        .with_earth_shadow_model(EarthShadowModel::Wgs84Oblate)
        .with_moon_eclipse(true);
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
    for model in [
        base,
        base.with_moon_eclipse(true),
        base.with_earth_shadow_model(EarthShadowModel::Wgs84Oblate)
            .with_moon_eclipse(true),
    ] {
        assert_eq!(model.parameters(), vec![ForceParameter::new("srp.cr", 1.3)]);
        let partials = model
            .parameter_partials(&[1.7], &state, &ctx)
            .expect("partials");
        let at_one = model
            .acceleration_with_parameters(&[1.0], &state, &ctx)
            .expect("acceleration");
        assert_eq!(partials, vec![at_one]);
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
    let earth_only = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let with_moon = earth_only.with_moon_eclipse(true);

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
    assert_eq!(
        with_moon
            .parameter_partials(&[1.3], &state, &ctx)
            .expect("partials"),
        vec![per_unit_cr]
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

    let earth_only_accel = earth_only.acceleration(&state, &ctx).expect("accel");
    assert!(earth_only_accel.norm() > expected.norm() * 1.05);
    let earth_only_partials = earth_only
        .acceleration_partials(&[1.3], &state, &ctx)
        .expect("partials")
        .wrt_position;
    assert!((earth_only_partials - expected_partials).norm() > 0.5 * expected_partials.norm());
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
    for model in [model, model.with_moon_eclipse(true)] {
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

#[test]
fn a_finite_position_whose_distance_overflows_is_a_numerical_failure() {
    let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let ctx = PropagationContext::default();
    let state = CartesianState::new(831_902_400.0, [1.0e200, 0.0, 0.0], [0.0; 3]);
    let acceleration = model.acceleration(&state, &ctx);
    assert!(
        matches!(acceleration, Err(PropagationError::NumericalFailure(_))),
        "{acceleration:?}"
    );
    let partials = model.acceleration_partials(&[1.3], &state, &ctx);
    assert!(
        matches!(partials, Err(PropagationError::NumericalFailure(_))),
        "{partials:?}"
    );
}

#[test]
fn deep_umbra_reports_numerical_failure_when_k_ref_or_factor_overflows() {
    let sat = Vector3::new(-7_000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let moon = Vector3::new(0.0, 384_400.0, 0.0);

    // 1. k_ref = p_ref * d_ref^2 overflows to +Inf -> Orekit evaluates 0 * Inf = NaN.
    let inf_k_ref =
        SolarRadiationPressure::with_pressure(1.0, 1.0, 1.0e290, AU_KM).expect("finite parameters");
    // 2. factor = (A/m) * C_R overflows to +Inf -> Orekit evaluates Inf * 0 = NaN.
    let inf_factor = SolarRadiationPressure::with_pressure(1.0e200, 1.0e200, 4.56e-6, AU_KM)
        .expect("finite parameters");
    for base in [inf_k_ref, inf_factor] {
        for model in [
            base,
            base.with_earth_shadow_model(EarthShadowModel::Wgs84Oblate),
            base.with_moon_eclipse(true),
            base.with_earth_shadow_model(EarthShadowModel::Wgs84Oblate)
                .with_moon_eclipse(true),
        ] {
            let moon_opt = model.moon_eclipse_enabled().then_some(moon);
            let accel = srp_acceleration_with_bodies(sat, sun, moon_opt, model);
            assert!(
                matches!(accel, Err(PropagationError::NumericalFailure(_))),
                "expected NumericalFailure for accel, got {accel:?}"
            );
            let partials = srp_position_partials_with_bodies(sat, sun, moon_opt, model);
            assert!(
                matches!(partials, Err(PropagationError::NumericalFailure(_))),
                "expected NumericalFailure for partials, got {partials:?}"
            );
        }
    }

    // 3. When both k_ref (~2.24e222) and factor (1e200) are finite, Orekit's
    //    0.0 * k_ref = 0.0 stays finite in deep umbra, while penumbra overflows.
    let finite_scale = SolarRadiationPressure::with_pressure(1.0, 1.0e200, 1.0e200, AU_KM)
        .expect("finite parameters");
    assert_eq!(
        srp_acceleration_with_bodies(sat, sun, None, finite_scale).expect("deep umbra accel"),
        Vector3::zeros()
    );
    assert_eq!(
        srp_position_partials_with_bodies(sat, sun, None, finite_scale)
            .expect("deep umbra partials"),
        Matrix3::zeros()
    );
    let penumbra_model = finite_scale.with_moon_eclipse(true);
    let penumbra_sat = Vector3::new(-26_000.0, 6_000.0, 2_100.0);
    let overflowed =
        srp_position_partials_with_bodies(penumbra_sat, sun, Some(moon), penumbra_model)
            .map(AccelerationPartials::position_only)
            .and_then(finite_partials);
    assert!(
        matches!(overflowed, Err(PropagationError::NumericalFailure(_))),
        "{overflowed:?}"
    );
}

#[test]
fn position_partials_report_numerical_failure_when_field_acceleration_value_overflows() {
    // Here k_ref = 2.238e222, flux.x.value = -1.0e200, flux.x.grad[0] = -1.337e189,
    // and factor = 1.0e115, so the field acceleration value overflows to -Inf
    // while every partial entry (-1.337e304) remains finite.
    let base = SolarRadiationPressure::with_pressure(1.0, 1.0e115, 1.0e200, AU_KM)
        .expect("finite parameters");
    let sat = Vector3::new(7_000.0, 0.0, 0.0);
    let sun = Vector3::new(AU_KM, 0.0, 0.0);
    let moon = Vector3::new(0.0, 384_400.0, 0.0);
    for model in [base, base.with_moon_eclipse(true)] {
        let moon_opt = model.moon_eclipse_enabled().then_some(moon);
        assert!(matches!(
            srp_acceleration_with_bodies(sat, sun, moon_opt, model),
            Err(PropagationError::NumericalFailure(_))
        ));
        let partials = srp_position_partials_with_bodies(sat, sun, moon_opt, model);
        assert!(
            matches!(partials, Err(PropagationError::NumericalFailure(_))),
            "expected NumericalFailure when field acceleration value overflows, got {partials:?}"
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
    let models = [
        base,
        base.with_moon_eclipse(true),
        base.with_earth_shadow_model(EarthShadowModel::Wgs84Oblate),
        base.with_moon_eclipse(true)
            .with_earth_shadow_model(EarthShadowModel::Wgs84Oblate),
    ];
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
                let param_partials = model
                    .parameter_partials(&[1.7], &state, &ctx)
                    .expect("parameter_partials");
                let unit_model = SolarRadiationPressure { cr: 1.0, ..model };
                let unit_from_vectors = unit_model
                    .acceleration_with_sun_moon(position, sun, moon)
                    .expect("caller vectors per unit C_R");
                assert_eq!(
                    param_partials,
                    vec![unit_from_vectors],
                    "parameter_partials epoch {epoch} position {position:?}"
                );
                let state_partials = model
                    .acceleration_partials(&[1.7], &state, &ctx)
                    .expect("acceleration_partials");
                let moon_opt = model.moon_eclipse_enabled().then_some(moon);
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

        // Also assert the clear-Moon SRP field values and position partials
        // through the SRP position-partials path at 0 ULP for all 76 Earth cases.
        let clear_moon = parse_v(&c.clear_moon_km_bits);
        let srp_model = base_srp.with_earth_shadow_model(model);
        let clear_lf =
            lighting_ratio_field_of(srp_model, sat, sun, Some(clear_moon)).expect("clear_moon lf");
        let clear_af =
            srp_acceleration_field_m_s2(sat, sun, clear_lf, srp_model).expect("clear_moon af");
        let clear_partials =
            srp_position_partials_with_bodies(sat, sun, Some(clear_moon), srp_model)
                .expect("clear_moon partials");
        for i in 0..3 {
            let clear_af_i = [clear_af.x, clear_af.y, clear_af.z][i];
            check(
                &format!("earth {} clear_moon af.v[{i}]", c.name),
                clear_af_i.value,
                parse(&c.clear_moon_accel_field_val_bits[i]),
                &mut mismatches,
            );
            checks += 1;
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
            checks += 3;
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
        let ld = earth_moon_lighting_ratio_m_double(to_m(sat), to_m(sun), to_m(moon), model)
            .expect("em ld");
        let lf = earth_moon_lighting_ratio_m_field(to_m(sat), to_m(sun), to_m(moon), model)
            .expect("em lf");
        let ad = srp_acceleration_m_s2(sat, sun, ld, base_srp).expect("em ad");
        let af = srp_acceleration_field_m_s2(sat, sun, lf, base_srp).expect("em af");

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
            checks += 3;
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
                checks += 3;
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
