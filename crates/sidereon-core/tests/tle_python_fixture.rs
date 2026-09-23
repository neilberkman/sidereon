#![cfg(sidereon_repo_tests)]
//! Env-gated emitter that dumps the validated TLE parse/encode round-trip as a
//! JSON fixture for the Python binding's pytest.
//!
//! The character-exact round-trip itself is proven by `tle_roundtrip.rs` and the
//! `tle` module unit tests; this harness reuses the same committed ISS element
//! set and the same `tle::parse` / `tle::encode` pair the binding calls
//! (`Tle.to_lines`), so the fixture carries the engine's own numbers, not
//! invented truth. The assertions below are a regression lock; the dump runs
//! only under `SIDEREON_DUMP_FIXTURES=1` and never in a normal `cargo test`.

use std::path::PathBuf;

use sidereon_core::astro::tle::{
    encode, parse, parse_with_policy, ChecksumWarning, ChecksumWarningKind, TlePolicy,
};

// Canonical ISS TLE (epoch 2018-184.80969102). Real, committed, and validated:
// the same two lines appear in the `tle` and `sgp4` module doctests/tests.
const ISS_L1: &str = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993";
const ISS_L2: &str = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106";

// Line 1 with the column-69 checksum digit flipped (9993 -> 9990). The strict
// default refuses it; the lenient policy reads it and reports the mismatch.
const ISS_L1_BAD_CHECKSUM: &str =
    "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9990";

#[test]
fn iss_round_trip_fixture_self_validates() {
    let parsed = parse(ISS_L1, ISS_L2).expect("committed ISS TLE parses");
    let (l1, l2) = encode(&parsed.elements).expect("committed ISS TLE encodes");

    // Character-exact round-trip regression lock.
    assert_eq!(l1, ISS_L1);
    assert_eq!(l2, ISS_L2);
    assert!(parsed.checksum_warnings.is_empty());

    // A flipped digit is refused by default and reported under the lenient
    // policy.
    assert!(parse(ISS_L1_BAD_CHECKSUM, ISS_L2).is_err());
    let bad = parse_with_policy(ISS_L1_BAD_CHECKSUM, ISS_L2, TlePolicy::Lenient)
        .expect("lenient read accepts the mismatch");
    assert_eq!(
        bad.checksum_warnings,
        vec![ChecksumWarning {
            line_label: "line 1",
            kind: ChecksumWarningKind::Mismatch { expected: 0 },
            computed: 3,
        }]
    );

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_fixture();
    }
}

/// Serialize the committed TLE, its parsed element fields, the engine's
/// re-encoded lines, and the advisory checksum-warning case to the JSON fixture
/// consumed by the Python binding's pytest.
fn dump_fixture() {
    use serde_json::json;

    let parsed = parse(ISS_L1, ISS_L2).expect("dump: ISS TLE parses");
    let el = &parsed.elements;
    let (l1, l2) = encode(el).expect("dump: ISS TLE encodes");
    let bad = parse_with_policy(ISS_L1_BAD_CHECKSUM, ISS_L2, TlePolicy::Lenient)
        .expect("dump: bad-checksum TLE parses leniently");

    let warnings: Vec<_> = bad
        .checksum_warnings
        .iter()
        .map(|w| {
            let expected = match w.kind {
                ChecksumWarningKind::Mismatch { expected } => Some(expected),
                _ => None,
            };
            json!({
                "line_label": w.line_label,
                "expected": expected,
                "computed": w.computed,
            })
        })
        .collect();

    let doc = json!({
        "source": "iss_round_trip_fixture_self_validates",
        "opsmode": "afspc",
        "tle": { "line1": ISS_L1, "line2": ISS_L2 },
        "encoded": { "line1": l1, "line2": l2 },
        "elements": {
            "catalog_number": el.catalog_number,
            "classification": el.classification,
            "international_designator": el.international_designator,
            "epoch_year": el.epoch_year,
            "epoch_day_of_year": el.epoch_day_of_year,
            "inclination_deg": el.inclination_deg,
            "raan_deg": el.raan_deg,
            "eccentricity": el.eccentricity,
            "arg_perigee_deg": el.arg_perigee_deg,
            "mean_anomaly_deg": el.mean_anomaly_deg,
            "mean_motion": el.mean_motion,
            "mean_motion_dot": el.mean_motion_dot,
            "mean_motion_double_dot": el.mean_motion_double_dot,
            "bstar": el.bstar,
            "rev_number": el.rev_number,
        },
        "checksum_case": {
            "line1": ISS_L1_BAD_CHECKSUM,
            "line2": ISS_L2,
            "warnings": warnings,
        },
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/tle_roundtrip.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped TLE round-trip fixture to {out:?}");
}
