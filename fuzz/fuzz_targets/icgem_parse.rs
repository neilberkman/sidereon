#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::forces::{IcgemGravityField, TideSystem};

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let Ok(field) = IcgemGravityField::parse(&text) else {
        return;
    };
    // A time-variable coefficient is finite or a typed error at any epoch,
    // including each record's own epoch and interval bounds.
    for term in field.time_variable_terms().iter().take(64) {
        for epoch in [
            term.reference_tt_seconds,
            term.valid_from_tt_seconds,
            term.valid_until_tt_seconds,
            term.reference_tt_seconds + 1.0,
            7.5e8,
            1.0e300,
        ] {
            if let Ok(Some(value)) = field.coefficient_at(term.degree, term.order, epoch) {
                assert!(value.c.is_finite() && value.s.is_finite(), "{value:?}");
            }
        }
    }
    let field = match field.header().tide_system {
        Some(_) => field,
        None => field
            .with_assumed_tide_system(TideSystem::TideFree)
            .expect("a field with no stated tide system accepts one"),
    };
    let degree = field.stored_max_degree().min(24);
    if degree < 2 {
        return;
    }
    let gravity = field
        .gravity(degree, degree)
        .expect("a parsed field builds at any truncation it stores");
    for epoch in [-1.0e9, 0.0, 7.5e8] {
        for position in [[7000.0, 100.0, -50.0], [0.0, 0.0, 6500.0], [1.0, 0.0, 0.0]] {
            // Evaluation returns a typed error or finite values, never NaN.
            if let Ok(a) = gravity.body_fixed_acceleration_at_km_s2(position, epoch) {
                assert!(a.iter().all(|x| x.is_finite()), "{a:?}");
            }
            if let Ok(v) = gravity.body_fixed_potential_at_km2_s2(position, epoch) {
                assert!(v.is_finite(), "{v}");
            }
        }
        for (degree, order) in [(2, 0), (2, 2), (degree, degree)] {
            if let Ok(Some(value)) = gravity.coefficient_at(degree, order, epoch) {
                assert!(value.c.is_finite() && value.s.is_finite(), "{value:?}");
            }
        }
    }
});
