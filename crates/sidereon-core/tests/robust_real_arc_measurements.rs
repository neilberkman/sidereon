#![cfg(sidereon_repo_tests)]
use serde_json::Value;
use std::path::PathBuf;

fn rtk_fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rtk")
        .join(name)
}

fn load_json(name: &str) -> Value {
    let path = rtk_fixture_path(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {path:?}: {err}"));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("parse {path:?}: {err}"))
}

fn array<'a>(value: &'a Value, field: &str) -> &'a [Value] {
    value[field]
        .as_array()
        .unwrap_or_else(|| panic!("{field} must be an array"))
}

fn field_f64(value: &Value, field: &str) -> f64 {
    value[field]
        .as_f64()
        .unwrap_or_else(|| panic!("{field} must be a finite JSON number"))
}

fn field_usize(value: &Value, field: &str) -> usize {
    value[field]
        .as_u64()
        .unwrap_or_else(|| panic!("{field} must be a JSON integer")) as usize
}

fn field_bool(value: &Value, field: &str) -> bool {
    value[field]
        .as_bool()
        .unwrap_or_else(|| panic!("{field} must be a JSON boolean"))
}

fn field_str<'a>(value: &'a Value, field: &str) -> &'a str {
    value[field]
        .as_str()
        .unwrap_or_else(|| panic!("{field} must be a JSON string"))
}

fn assert_oracle_fixture_matches(arc: &Value) {
    let fixture = field_str(arc, "oracle_fixture");
    let oracle = load_json(fixture);
    assert_eq!(
        oracle["reference"]["label"].as_str(),
        Some(field_str(arc, "label")),
        "{fixture} reference label drifted from the measurement fixture"
    );

    let demo5_median = oracle["reference"]["error_3d"]["median_m"]
        .as_f64()
        .unwrap_or_else(|| panic!("{fixture} missing reference.error_3d.median_m"));
    assert!(
        (demo5_median - field_f64(arc, "demo5_median_3d_m")).abs() < 1.0e-3,
        "{fixture} demo5 median drifted from the measurement fixture"
    );
}

#[test]
fn spp_fde_gsdc_measurement_pins_no_silent_degrade_gate() {
    let doc = load_json("robustness/spp_robustness_measurement_2026_06.json");
    assert_eq!(doc["version"].as_u64(), Some(1));
    assert_eq!(field_str(&doc, "kind"), "physical-truth gate");

    let measurement = &doc["measurement"];
    let min_epochs = field_usize(measurement, "min_epochs");
    assert_eq!(min_epochs, 100);
    assert_eq!(field_f64(measurement, "code_sigma_m"), 5.0);
    assert_eq!(field_f64(measurement, "max_pdop"), 1000.0);

    let arcs = array(&doc, "arcs");
    assert_eq!(arcs.len(), 4);

    let mut strict_median_and_p95_non_regress = true;
    let mut unit_weight_harmful_count = 0usize;

    for arc in arcs {
        assert_oracle_fixture_matches(arc);
        assert!(field_usize(arc, "n") >= min_epochs);

        let bare = &arc["bare"];
        let unit = &arc["robust_unit"];
        let weighted = &arc["robust_weighted"];

        let bare_median = field_f64(bare, "median_3d_m");
        let bare_p95 = field_f64(bare, "p95_3d_m");
        let unit_median = field_f64(unit, "median_3d_m");
        let weighted_median = field_f64(weighted, "median_3d_m");
        let weighted_p95 = field_f64(weighted, "p95_3d_m");

        assert!(
            weighted_median <= bare_median,
            "{} weighted FDE median regressed: {weighted_median} > {bare_median}",
            field_str(arc, "label")
        );

        strict_median_and_p95_non_regress &= weighted_p95 <= bare_p95;
        if unit_median > bare_median {
            unit_weight_harmful_count += 1;
        }
    }

    assert_eq!(
        unit_weight_harmful_count,
        arcs.len(),
        "unit-weight FDE is the harmful mode on every measured phone arc"
    );
    assert!(
        !strict_median_and_p95_non_regress,
        "the fixture must preserve the calibrated null on the strict median+p95 FDE bar"
    );

    let claims = &doc["claims"];
    assert!(field_bool(claims, "robust_weighted_median_non_regression"));
    assert!(!field_bool(claims, "strict_median_and_p95_non_regression"));
    assert!(field_bool(
        claims,
        "default_robust_without_noise_model_refuses"
    ));
    assert!(field_bool(claims, "unit_weight_fde_is_harmful_mode"));

    let pooled = &doc["pooled"];
    assert_eq!(field_usize(pooled, "powered_arc_count"), arcs.len());
    assert!(field_bool(pooled, "all_powered_median_non_regress"));
    assert!(!field_bool(
        pooled,
        "all_powered_median_and_p95_non_regress"
    ));
}

#[test]
fn huber_irls_gsdc_measurement_pins_physical_truth_gate() {
    let doc = load_json("robustness/huber_irls_measurement_2026_06.json");
    assert_eq!(doc["version"].as_u64(), Some(1));
    assert_eq!(field_str(&doc, "kind"), "physical-truth gate");

    let measurement = &doc["measurement"];
    let min_epochs = field_usize(measurement, "min_epochs");
    assert_eq!(min_epochs, 100);
    assert_eq!(field_f64(measurement, "huber_k"), 1.345);
    assert_eq!(field_f64(measurement, "scale_floor_m"), 5.0);
    assert_eq!(field_usize(measurement, "max_outer"), 5);

    let arcs = array(&doc, "arcs");
    assert_eq!(arcs.len(), 4);

    for arc in arcs {
        assert_oracle_fixture_matches(arc);
        assert!(field_usize(arc, "matched") >= min_epochs);
        assert_eq!(field_usize(arc, "both_ok"), field_usize(arc, "matched"));
        assert_eq!(field_usize(arc, "huber_only_fail"), 0);
        assert_eq!(field_usize(arc, "bare_only_fail"), 0);
        assert_eq!(field_usize(arc, "both_fail"), 0);
        assert_eq!(field_usize(arc, "too_few_sats"), 0);

        let bare = &arc["bare"];
        let huber = &arc["huber"];
        let bare_median = field_f64(bare, "median_3d_m");
        let bare_p95 = field_f64(bare, "p95_3d_m");
        let huber_median = field_f64(huber, "median_3d_m");
        let huber_p95 = field_f64(huber, "p95_3d_m");
        let delta = field_f64(arc, "delta_median_3d_m");

        assert!(
            huber_median <= bare_median,
            "{} Huber median regressed: {huber_median} > {bare_median}",
            field_str(arc, "label")
        );
        assert!(
            huber_p95 <= bare_p95,
            "{} Huber p95 regressed: {huber_p95} > {bare_p95}",
            field_str(arc, "label")
        );
        assert!(
            delta > 0.0,
            "{} expected positive bare-minus-Huber median delta",
            field_str(arc, "label")
        );
    }

    let claims = &doc["claims"];
    assert!(field_bool(claims, "huber_off_byte_identical_to_bare"));
    assert!(field_bool(claims, "all_powered_median_non_regress"));
    assert!(field_bool(claims, "all_powered_median_and_p95_non_regress"));
    assert!(field_bool(claims, "no_availability_regression"));

    let pooled = &doc["pooled"];
    assert_eq!(field_usize(pooled, "powered_arc_count"), arcs.len());
    assert!(field_bool(pooled, "huber_off_byte_identical_to_bare"));
    assert!(field_bool(pooled, "all_powered_median_non_regress"));
    assert!(field_bool(pooled, "all_powered_median_and_p95_non_regress"));
}
