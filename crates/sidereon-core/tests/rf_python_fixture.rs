#![cfg(sidereon_repo_tests)]
//! Env-gated emitter that dumps RF link-budget reference numbers as a JSON
//! fixture for the Python binding's pytest (`test_rf.py`).
//!
//! Every value is computed with the exact `astro::rf` functions the binding
//! calls. The fixture carries engine output as IEEE-754 hex bits for an exact
//! wrapper cross-check, not hand-authored truth. The dump runs only under
//! `SIDEREON_DUMP_FIXTURES=1`; a normal `cargo test` self-validates.

use std::path::PathBuf;

use sidereon_core::astro::rf::{cn0, dish_gain, eirp, fspl, link_margin, wavelength, LinkBudget};

fn hex(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

#[test]
fn rf_reference_self_validates() {
    let fspl_db = fspl(1200.0, 1616.0).expect("valid FSPL inputs");
    let eirp_dbw = eirp(27.0, 3.0).expect("valid EIRP inputs");
    let cn0_dbhz = cn0(eirp_dbw, 165.0, -12.0, 3.0).expect("valid C/N0 inputs");
    let budget = LinkBudget {
        eirp_dbw,
        fspl_db: 165.0,
        receiver_gt_dbk: -12.0,
        other_losses_db: 3.0,
        required_cn0_dbhz: 35.0,
    };
    let margin_db = link_margin(&budget).expect("valid link budget");
    let lambda_m = wavelength(1616.0e6).expect("valid wavelength input");
    let dish_gain_dbi = dish_gain(1.0, 1616.0e6, 0.55).expect("valid dish inputs");

    assert!(fspl_db.is_finite());
    assert!(cn0_dbhz.is_finite());
    assert_eq!(
        margin_db.to_bits(),
        (cn0_dbhz - budget.required_cn0_dbhz).to_bits()
    );
    assert!(lambda_m > 0.0);
    assert!(dish_gain_dbi.is_finite());

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_fixture();
    }
}

fn dump_fixture() {
    use serde_json::json;

    let budget = LinkBudget {
        eirp_dbw: 0.0,
        fspl_db: 165.0,
        receiver_gt_dbk: -12.0,
        other_losses_db: 3.0,
        required_cn0_dbhz: 35.0,
    };
    let doc = json!({
        "source": "rf_reference_self_validates",
        "fspl": {
            "distance_km": 1200.0,
            "frequency_mhz": 1616.0,
            "value_hex": hex(fspl(1200.0, 1616.0).expect("valid FSPL inputs")),
        },
        "eirp": {
            "tx_power_dbm": 27.0,
            "tx_antenna_gain_dbi": 3.0,
            "value_hex": hex(eirp(27.0, 3.0).expect("valid EIRP inputs")),
        },
        "cn0": {
            "eirp_dbw": budget.eirp_dbw,
            "fspl_db": budget.fspl_db,
            "receiver_gt_dbk": budget.receiver_gt_dbk,
            "other_losses_db": budget.other_losses_db,
            "value_hex": hex(cn0(
                budget.eirp_dbw,
                budget.fspl_db,
                budget.receiver_gt_dbk,
                budget.other_losses_db
            ).expect("valid C/N0 inputs")),
        },
        "wavelength": {
            "frequency_hz": 1616.0e6,
            "value_hex": hex(wavelength(1616.0e6).expect("valid wavelength input")),
        },
        "dish_gain": {
            "diameter_m": 1.0,
            "frequency_hz": 1616.0e6,
            "efficiency": 0.55,
            "value_hex": hex(dish_gain(1.0, 1616.0e6, 0.55).expect("valid dish inputs")),
        },
        "link_margin": {
            "budget": {
                "eirp_dbw": budget.eirp_dbw,
                "fspl_db": budget.fspl_db,
                "receiver_gt_dbk": budget.receiver_gt_dbk,
                "other_losses_db": budget.other_losses_db,
                "required_cn0_dbhz": budget.required_cn0_dbhz,
            },
            "value_hex": hex(link_margin(&budget).expect("valid link budget")),
        },
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/rf_link_budget.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped RF fixture to {out:?}");
}
