#![no_main]

use std::collections::BTreeSet;

use libfuzzer_sys::fuzz_target;
use sidereon_core::observation_qc::observation_qc;
use sidereon_core::GnssSystem;
use sidereon_core::rinex::observations::RinexObs;
use sidereon_core::rinex::qc::{lint_obs, repair_obs, RepairOptions};

const MAX_INPUT_LEN: usize = 1 << 20;

fn repair_options() -> RepairOptions {
    let base = RepairOptions::default();
    let mut options = RepairOptions::default();
    options.set_interval = true;
    options.set_time_of_last_obs = true;
    options.set_obs_counts = true;
    options.drop_empty_records = true;
    options.drop_unsupported = true;
    options.file_stamp = base.file_stamp;
    options.sort_records = base.sort_records;
    options
}

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

/// The product itself when it writes. A refusal has to be one the product
/// shows the reason for, and gives `None`.
fn stated_product(product: RinexObs) -> Option<RinexObs> {
    match product.to_rinex_string() {
        Ok(_) => {
            assert!(
                !write_obstacles(&product).1,
                "wrote a repaired product holding what its writer cannot put back"
            );
            Some(product)
        }
        Err(error) if product.header().version < 3.0 => {
            let (removable, permanent) = write_obstacles(&product);
            assert!(
                removable || permanent,
                "refused a repaired version 2 product holding nothing version 2 cannot: {error}"
            );
            None
        }
        Err(error) => {
            assert!(
                write_obstacles(&product).1,
                "serialize repaired RINEX OBS: {error}"
            );
            None
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_LEN {
        return;
    }

    let text = String::from_utf8_lossy(data);
    let Ok(obs) = RinexObs::parse(&text) else {
        return;
    };

    let _ = observation_qc(&obs);
    let _ = lint_obs(&obs);

    let options = repair_options();
    let repair = repair_obs(&obs, &options);
    let Some(repaired) = stated_product(repair.repaired) else {
        return;
    };
    let repaired_text = repaired.to_rinex_string().expect("serialize RINEX OBS");
    let reparsed = RinexObs::parse(&repaired_text).expect("repaired OBS must reparse");

    let _ = observation_qc(&reparsed);
    let _ = lint_obs(&reparsed);

    let repeated = repair_obs(&reparsed, &options);
    let repeated_text = stated_product(repeated.repaired)
        .expect("a repair that wrote writes again")
        .to_rinex_string()
        .expect("serialize RINEX OBS");
    assert_eq!(repeated_text.as_bytes(), repaired_text.as_bytes());
});
