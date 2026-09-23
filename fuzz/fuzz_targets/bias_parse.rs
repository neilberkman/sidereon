#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::time::model::TimeScale;
use sidereon_core::bias::{
    bias_epoch_instant, write_bias_sinex_bytes, write_code_dcb_bytes, BiasKind, BiasReadPolicy,
    BiasSet, BiasTarget,
};

/// Looks up the first satellite DSB of a product between its observables,
/// and across the product's DSBs for that satellite, at its validity start,
/// so route resolution runs on arbitrary products.
fn exercise_dsb_lookup(set: &BiasSet) {
    let Some(record) = set
        .records()
        .iter()
        .find(|record| record.kind == BiasKind::Dsb)
    else {
        return;
    };
    let (BiasTarget::Satellite(sat), Some(obs2), Some(start)) =
        (&record.target, record.obs2.as_deref(), record.valid_from)
    else {
        return;
    };
    let scale = set.time_scale().unwrap_or(TimeScale::Gpst);
    let Ok(epoch) = bias_epoch_instant(start, scale) else {
        return;
    };
    let _ = set.code_dsb_seconds(*sat, &record.obs1, obs2, epoch);
    // From the first DSB's first observable to the last same-satellite DSB's
    // second observable, so routes of several hops are resolved too.
    if let Some(last_obs2) = set
        .records()
        .iter()
        .rev()
        .filter(|other| other.kind == BiasKind::Dsb && other.target == record.target)
        .find_map(|other| other.obs2.as_deref())
    {
        let _ = set.code_dsb_seconds(*sat, &record.obs1, last_obs2, epoch);
    }
}

fuzz_target!(|data: &[u8]| {
    // A product read under either policy restates its input byte for byte.
    for policy in [BiasReadPolicy::Strict, BiasReadPolicy::Lenient] {
        if let Ok(parsed) = BiasSet::parse_bias_sinex_with_policy(data, policy) {
            exercise_dsb_lookup(&parsed.value);
            let restated = write_bias_sinex_bytes(&parsed.value).expect("restate Bias-SINEX");
            assert_eq!(restated, data);
        }
    }
    for policy in [BiasReadPolicy::Strict, BiasReadPolicy::Lenient] {
        if let Ok(parsed) = BiasSet::parse_code_dcb_with_policy(data, None, policy) {
            exercise_dsb_lookup(&parsed.value);
            let restated = write_code_dcb_bytes(&parsed.value).expect("restate CODE DCB");
            assert_eq!(restated, data);
        }
    }
});
