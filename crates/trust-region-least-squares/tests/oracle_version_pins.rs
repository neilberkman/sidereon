//! Provenance guard for the measured SciPy and NumPy oracle-version notes.

use std::path::PathBuf;

use serde_json::Value;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("oracle_version_pins.json")
}

#[test]
fn oracle_version_pin_measurements_have_reproducible_provenance() {
    let raw = std::fs::read_to_string(fixture_path()).expect("read oracle pin fixture");
    let fixture: Value = serde_json::from_str(&raw).expect("parse oracle pin fixture");

    assert_eq!(
        fixture["schema"],
        "trust-region-least-squares-oracle-version-pins-v1"
    );
    assert_eq!(fixture["host"]["python"], "3.14.7");
    assert_eq!(
        fixture["host"]["blas_lapack"],
        "Apple Accelerate as reported by numpy.show_config()"
    );
    assert_eq!(fixture["tools"]["uv"], "0.11.19");
    assert_eq!(fixture["tools"]["curl"], "8.7.1");

    let sources = fixture["sources"].as_array().expect("package sources");
    assert_eq!(sources.len(), 4);
    for source in sources {
        assert!(source["url"]
            .as_str()
            .expect("source URL")
            .starts_with("https://files.pythonhosted.org/"));
        assert_eq!(source["sha256"].as_str().expect("wheel digest").len(), 64);
        assert_eq!(
            source["measured_binary_sha256"]
                .as_str()
                .expect("binary digest")
                .len(),
            64
        );
    }

    let scipy = &fixture["measurements"]["scipy_splrep"];
    assert_eq!(scipy["old_version"], "1.17.1");
    assert_eq!(scipy["new_version"], "1.18.0");
    assert_eq!(scipy["allocator_reuse_fits_per_version"], 10_000);
    assert_eq!(scipy["nonzero_trailing_coefficients_seen"], 0);
    assert_eq!(scipy["interpolating_s_0"]["evaluation_max_ulp"], 0);
    assert_eq!(scipy["smoothed_s_0_01"]["used_coefficient_max_ulp"], 21);
    assert_eq!(scipy["smoothed_s_0_01"]["evaluation_max_ulp"], 23);

    let numpy = &fixture["measurements"]["numpy_pinv"];
    assert_eq!(numpy["old_version"], "2.4.6");
    assert_eq!(numpy["new_version"], "2.5.0");
    assert_eq!(numpy["fixed_matrix"]["max_ulp"], 0);
    assert_eq!(numpy["deterministic_sweep"]["cases"], 285);
    assert_eq!(numpy["deterministic_sweep"]["differing_cases"], 0);

    let command = fixture["reproduction"]["command"]
        .as_str()
        .expect("reproduction command");
    assert!(command.contains("verify_oracle_version_pins.py"));
    assert!(command.contains("numpy==2.4.6 scipy==1.17.1"));
    assert!(command.contains("numpy==2.5.0 scipy==1.18.0"));
}
