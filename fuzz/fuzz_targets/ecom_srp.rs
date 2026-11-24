#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use nalgebra::Vector3;
use sidereon_core::astro::events::eclipse::{EarthShadowModel, ShadowFunction};
use sidereon_core::astro::forces::{
    ecom_geometry_with_shadow, Ecom1Coefficients, Ecom1SolarRadiationPressure,
    Ecom2D2b1Coefficients, Ecom2D4b1Coefficients, Ecom2SolarRadiationPressure,
};

#[derive(Debug, Arbitrary)]
struct Input {
    n_d: u8,
    n_b: u8,
    wgs84: bool,
    include_moon: bool,
    conical: bool,
    coefficients: [f64; 9],
    position: [f64; 3],
    velocity: [f64; 3],
    sun: [f64; 3],
    moon: [f64; 3],
}

// ECOM1 and ECOM2 geometry, accelerations, state partials, and parameter
// partials over arbitrary coefficients, truncations, shadow options, states,
// and body positions: finite results or typed errors, never a panic.
fuzz_target!(|input: Input| {
    let r = Vector3::from(input.position);
    let v = Vector3::from(input.velocity);
    let s = Vector3::from(input.sun);
    let m = Vector3::from(input.moon);
    let earth_model = if input.wgs84 {
        EarthShadowModel::Wgs84Oblate
    } else {
        EarthShadowModel::Spherical
    };
    let moon_opt = input.include_moon.then_some(m);

    if let Ok(g) = ecom_geometry_with_shadow(r, v, s, moon_opt, earth_model) {
        assert!(g.lighting_ratio.is_finite());
        if let Some(du) = g.delta_u_rad {
            assert!(du.is_finite());
        }
        if let Some(u) = g.argument_of_latitude_rad {
            assert!(u.is_finite());
        }
    }

    let n_d = usize::from(input.n_d % 4);
    let n_b = usize::from(input.n_b % 3);
    let count = 2 * (n_d + n_b) + 3;
    let coeffs: Vec<f64> = (0..count).map(|k| input.coefficients[k % 9]).collect();

    if let Ok(model) = Ecom2SolarRadiationPressure::new(n_d, n_b, coeffs) {
        let model = model.with_earth_shadow_model(earth_model);
        let model = if input.conical {
            model.with_shadow_function(ShadowFunction::Conical)
        } else {
            model.with_moon_eclipse(input.include_moon)
        };
        if let Ok(a) = model.acceleration_with_sun_moon(r, v, s, m) {
            assert!(a.iter().all(|x| x.is_finite()), "{a:?}");
        }
        if let Ok(p) = model.acceleration_partials_with_sun_moon(r, v, s, m) {
            assert!(p.wrt_position.iter().all(|x| x.is_finite()));
            assert!(p.wrt_velocity.iter().all(|x| x.is_finite()));
        }
        if let Ok(pp) = model.parameter_partials_with_sun_moon(r, v, s, m) {
            assert!(pp.iter().all(|vec| vec.iter().all(|x| x.is_finite())));
        }
    }

    if let Ok(model) = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients::from_array([
        input.coefficients[0],
        input.coefficients[1],
        input.coefficients[2],
        input.coefficients[3],
        input.coefficients[4],
        input.coefficients[5],
        input.coefficients[6],
    ])) {
        let _ = model.acceleration_with_sun_moon(r, v, s, m);
        let _ = model
            .clone()
            .with_moon_eclipse(false)
            .acceleration_with_sun(r, v, s);
        let _ = model
            .with_shadow_function(ShadowFunction::Conical)
            .acceleration_with_sun(r, v, s);
    }

    if let Ok(model) =
        Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients::from_array(input.coefficients))
    {
        let _ = model.acceleration_with_sun_moon(r, v, s, m);
        let _ = model
            .with_moon_eclipse(false)
            .acceleration_with_sun(r, v, s);
    }

    if let Ok(model) =
        Ecom1SolarRadiationPressure::new(Ecom1Coefficients::from_array(input.coefficients))
    {
        let model = model.with_earth_shadow_model(earth_model);
        let model = if input.conical {
            model.with_shadow_function(ShadowFunction::Conical)
        } else {
            model.with_moon_eclipse(input.include_moon)
        };
        if let Ok(a) = model.acceleration_with_sun_moon(r, v, s, m) {
            assert!(a.iter().all(|x| x.is_finite()), "{a:?}");
        }
        if let Ok(p) = model.acceleration_partials_with_sun_moon(r, v, s, m) {
            assert!(p.wrt_position.iter().all(|x| x.is_finite()));
            assert!(p.wrt_velocity.iter().all(|x| x.is_finite()));
        }
        if let Ok(pp) = model.parameter_partials_with_sun_moon(r, v, s, m) {
            assert!(pp.iter().all(|vec| vec.iter().all(|x| x.is_finite())));
        }
    }
});
