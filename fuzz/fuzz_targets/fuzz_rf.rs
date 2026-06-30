#![no_main]

mod compute_common;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::rf::{self, LinkBudget};

#[derive(Debug, Arbitrary)]
struct Input {
    scalars: [f64; 8],
    budgets: Vec<[f64; 5]>,
    distances: Vec<f64>,
    frequency_mhz: f64,
}

fn link_budget(raw: [f64; 5]) -> LinkBudget {
    LinkBudget {
        eirp_dbw: raw[0],
        fspl_db: raw[1],
        receiver_gt_dbk: raw[2],
        other_losses_db: raw[3],
        required_cn0_dbhz: raw[4],
    }
}

// The RF link-budget helpers reject non-finite/out-of-domain inputs and return a
// finite output otherwise, so every Ok must be finite and no input may panic.
fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };

    assert_ok_finite_or_err("rf::fspl", rf::fspl(input.scalars[0], input.scalars[1]));
    assert_ok_finite_or_err("rf::eirp", rf::eirp(input.scalars[2], input.scalars[3]));
    assert_ok_finite_or_err(
        "rf::cn0",
        rf::cn0(
            input.scalars[0],
            input.scalars[1],
            input.scalars[2],
            input.scalars[3],
        ),
    );
    assert_ok_finite_or_err("rf::wavelength", rf::wavelength(input.scalars[4]));
    assert_ok_finite_or_err(
        "rf::dish_gain",
        rf::dish_gain(input.scalars[5], input.scalars[6], input.scalars[7]),
    );

    let single = link_budget([
        input.scalars[0],
        input.scalars[1],
        input.scalars[2],
        input.scalars[3],
        input.scalars[4],
    ]);
    assert_ok_finite_or_err("rf::link_margin", rf::link_margin(&single));

    let budgets: Vec<LinkBudget> = cap_vec(input.budgets, MAX_VEC)
        .into_iter()
        .map(link_budget)
        .collect();
    for result in rf::link_margin_batch(&budgets) {
        assert_ok_finite_or_err("rf::link_margin_batch", result);
    }

    let distances = cap_vec(input.distances, MAX_VEC);
    for result in rf::fspl_batch(&distances, input.frequency_mhz) {
        assert_ok_finite_or_err("rf::fspl_batch", result);
    }
});
