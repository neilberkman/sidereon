#![no_main]

use std::collections::BTreeSet;

use libfuzzer_sys::fuzz_target;
use sidereon_core::rinex::observations::RinexObs;
use sidereon_core::GnssSystem;

// Round-trip class: a parsed observation product must re-encode to text that
// reparses to an equal product. A mismatch means the serializer is lossy or the
// parser accepts state the serializer cannot reproduce.
/// What a product holds that its writer cannot put back, judged from the
/// product rather than from the writer's refusal. At any version: an epoch
/// flag above 9, which no one-digit flag field holds. From version 3 to 4.01:
/// epoch picoseconds, which those epoch records have no field for. At version 2
/// also:
/// whether the product could lose it to be written (scale factors, picoseconds,
/// a clock offset finer than `F12.9`), and whether it could not (a year outside
/// 1980 to 2079, a clock offset too wide for `F12.9`). Returns (removable,
/// permanent).
fn write_obstacles(product: &RinexObs) -> (bool, bool) {
    let wide_flag = product.epochs().iter().any(|epoch| epoch.flag > 9);
    let version = product.header().version;
    if version >= 3.0 {
        // Epoch records before 4.02 have no picosecond field and nothing
        // removes them; 4.02 records carry them after the clock.
        let picoseconds = version < 4.02
            && product
                .epochs()
                .iter()
                .any(|epoch| epoch.epoch_picoseconds.is_some());
        return (false, wide_flag || picoseconds);
    }
    let clock_fits = |offset: f64| format!("{offset:12.9}").len() <= 12;
    let clock_exact = |offset: f64| format!("{offset:.9}").parse::<f64>() == Ok(offset);
    let removable = !product.header().scale_factors.is_empty()
        || holds_unstated_list(product)
        || product.epochs().iter().any(|epoch| {
            epoch.epoch_picoseconds.is_some()
                || epoch
                    .rcv_clock_offset_s
                    .is_some_and(|offset| !clock_exact(offset))
        });
    let permanent = wide_flag
        || product.epochs().iter().any(|epoch| {
            !(1980..=2079).contains(&epoch.epoch.year)
                || epoch
                    .rcv_clock_offset_s
                    .is_some_and(|offset| !offset.is_finite() || !clock_fits(offset))
        });
    (removable, permanent)
}

/// Whether a version 2 product holds a code list its file may not state: one
/// for a constellation no observation or count names, other than the one a
/// file with no observations names in its version record, which is the held
/// constellation, else the product's one list's, else GPS. The writer refuses
/// it only when the type names it writes do not read as the list, which this
/// does not decide.
fn holds_unstated_list(product: &RinexObs) -> bool {
    let header = product.header();
    let mut stated: BTreeSet<GnssSystem> = product
        .epochs()
        .iter()
        .flat_map(|epoch| epoch.sats.keys().map(|sat| sat.system))
        .collect();
    if stated.is_empty() {
        let mut lists = header.obs_codes.keys();
        let fallback = header
            .rinex2_system
            .unwrap_or(match (lists.next(), lists.next()) {
                (Some(only), None) => *only,
                _ => GnssSystem::Gps,
            });
        stated.insert(fallback);
    }
    stated.extend(
        header
            .prn_obs_counts
            .iter()
            .filter(|(_, counts)| !counts.is_empty())
            .map(|(sat, _)| sat.system),
    );
    header.obs_codes.keys().any(|system| !stated.contains(system))
}

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let Ok(original) = RinexObs::parse(&text) else {
        return;
    };
    // Records skipped as unrepresentable are not re-emitted, so they would not
    // survive a round trip; restrict the invariant to clean products.
    if original.skipped_records != 0 {
        return;
    }
    // A product may hold what its writer cannot put back: the reader keeps a
    // flag wider than the flag field, and takes version 3 epoch records and
    // scale factors in a version 2 file. A refusal is accepted only when the
    // product shows why. Any other refusal fails.
    let encoded = match original.to_rinex_string() {
        Ok(encoded) => {
            assert!(
                !write_obstacles(&original).1,
                "wrote a product holding what its writer cannot put back"
            );
            encoded
        }
        Err(error) if original.header().version < 3.0 => {
            let (removable, permanent) = write_obstacles(&original);
            assert!(
                removable || permanent,
                "refused a version 2 product holding nothing version 2 cannot: {error}"
            );
            return;
        }
        Err(error) => {
            assert!(
                write_obstacles(&original).1,
                "serialize RINEX OBS: {error}"
            );
            return;
        }
    };
    let mut reparsed = RinexObs::parse(&encoded).expect("encoded RINEX OBS must reparse");
    // The labels of header records that were read and not retained describe the
    // source text rather than the product, and the writer does not re-emit them,
    // so a clean re-parse reports none. Normalise that one diagnostic rather
    // than skipping the whole product, which would stop checking its records.
    let mut original = original;
    original.header.unretained_header_labels.clear();
    reparsed.header.unretained_header_labels.clear();
    // The count an epoch line declared describes the source text too: the
    // writer states the records it has, so two records for one satellite are
    // declared as the one kept.
    for epoch in original.epochs.iter_mut().chain(reparsed.epochs.iter_mut()) {
        epoch.declared_record_count = if epoch.flag > 1 {
            epoch.special_records.len()
        } else {
            epoch.sats.len()
        };
    }
    // A version 2 version record names one constellation only while every
    // observation is from it; otherwise the writer names `M (MIXED)`, which
    // reads back as naming none.
    if let Some(system) = original.header.rinex2_system {
        if original
            .epochs
            .iter()
            .any(|epoch| epoch.sats.keys().any(|sat| sat.system != system))
        {
            original.header.rinex2_system = None;
        }
    }
    assert_eq!(reparsed, original);
});
