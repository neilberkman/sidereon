#![cfg(sidereon_repo_tests)]

use sidereon_core::astro::forces::{ForceModel, J2Gravity};
use sidereon_core::astro::propagator::api::PropagationContext;
use sidereon_core::astro::state::CartesianState;

#[test]
fn j2_acceleration_and_position_partials_match_orekit_bit_for_bit() {
    let ctx = PropagationContext::default();
    let mut case_count = 0;
    for line in include_str!("fixtures/orekit_j2.txt").lines() {
        if line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_ascii_whitespace();
        let case = fields.next().unwrap();
        let mut bits = fields.map(|field| u64::from_str_radix(field, 16).unwrap());
        let position = core::array::from_fn(|_| f64::from_bits(bits.next().unwrap()));
        let [mu, re, j2] = core::array::from_fn(|_| f64::from_bits(bits.next().unwrap()));
        let model = J2Gravity { mu, re, j2 };
        let state = CartesianState::new(0.0, position, [0.0; 3]);
        let acceleration = model.acceleration(&state, &ctx).unwrap();
        let partials = model.acceleration_partials(&[], &state, &ctx).unwrap();
        for (axis, value) in acceleration.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                bits.next().unwrap(),
                "{case}: acceleration {axis}"
            );
        }
        for row in 0..3 {
            for col in 0..3 {
                assert_eq!(
                    partials.wrt_position[(row, col)].to_bits(),
                    bits.next().unwrap(),
                    "{case}: position partial ({row}, {col})"
                );
            }
        }
        assert!(bits.next().is_none(), "{case}: unexpected fixture data");
        case_count += 1;
    }
    assert_eq!(case_count, 16);
}
