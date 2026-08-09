//! The path-based precise-interpolant open maps the file instead of reading it.
//!
//! The interpolant store is the second artifact type on the shared mapping
//! mechanism. It differs from the terrain store in one way that matters: its
//! borrowed parse holds `&[f64]` references directly into the byte span, so a
//! mapped reader parses with offset-backed arrays instead. These tests pin that
//! the substitution changes nothing an evaluator can observe.

#![cfg(feature = "mmap")]

use sidereon_core::ephemeris::{MmapPreciseEphemerisInterpolant, Sp3};

fn fixture_artifact() -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3");
    let text = std::fs::read(&path).expect("read committed IGS fixture");
    let product = Sp3::parse(&text).expect("parse committed IGS fixture");
    product
        .precise_interpolant_store_bytes()
        .expect("build interpolant artifact")
}

fn tempdir() -> std::path::PathBuf {
    let base = std::env::temp_dir().join(format!(
        "sidereon-interpolant-map-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&base).expect("create temp dir");
    base
}

#[test]
fn a_mapped_open_evaluates_identically_to_an_owned_one() {
    let artifact = fixture_artifact();
    let dir = tempdir();
    let path = dir.join("interpolant.spi");
    std::fs::write(&path, &artifact).expect("write artifact");

    let mapped = MmapPreciseEphemerisInterpolant::from_path(&path).expect("open mapped");
    let owned = MmapPreciseEphemerisInterpolant::from_vec(artifact.clone()).expect("open owned");
    let borrowed = MmapPreciseEphemerisInterpolant::from_bytes(&artifact).expect("open borrowed");

    assert!(
        mapped.is_memory_mapped(),
        "the path constructor must map the file"
    );
    assert!(!owned.is_memory_mapped());
    assert!(!borrowed.is_memory_mapped());

    assert_eq!(mapped.checksum64(), owned.checksum64());
    assert_eq!(mapped.checksum64(), borrowed.checksum64());
    assert_eq!(mapped.satellites(), owned.satellites());
    assert_eq!(mapped.as_bytes(), owned.as_bytes());

    // Evaluate across the whole covered span for every satellite. The mapped
    // reader uses offset-backed arrays where the borrowed one uses direct
    // slices, so any indexing error would show up here as a wrong number rather
    // than a crash.
    let satellites: Vec<_> = mapped.satellites().to_vec();
    assert!(!satellites.is_empty(), "fixture must carry satellites");

    let mut evaluated = 0usize;
    for sat in satellites {
        for step in 0..64 {
            let query = 646_315_200.0 + step as f64 * 900.0;
            let from_map = mapped.position_at_j2000_seconds(sat, query);
            let from_owned = owned.position_at_j2000_seconds(sat, query);
            let from_borrowed = borrowed.position_at_j2000_seconds(sat, query);

            match (&from_map, &from_owned, &from_borrowed) {
                (Ok(a), Ok(b), Ok(c)) => {
                    let (a, b, c) = (
                        a.position.as_array(),
                        b.position.as_array(),
                        c.position.as_array(),
                    );
                    for axis in 0..3 {
                        assert_eq!(
                            a[axis].to_bits(),
                            b[axis].to_bits(),
                            "mapped and owned disagree for {sat} at {query}"
                        );
                        assert_eq!(
                            a[axis].to_bits(),
                            c[axis].to_bits(),
                            "mapped and borrowed disagree for {sat} at {query}"
                        );
                    }
                    evaluated += 1;
                }
                (Err(_), Err(_), Err(_)) => {}
                other => panic!("load paths disagreed on outcome for {sat} at {query}: {other:?}"),
            }
        }
    }
    assert!(
        evaluated > 100,
        "expected a substantial evaluation sample, got {evaluated}"
    );
}

#[test]
fn a_malformed_artifact_is_rejected_identically_by_both_paths() {
    let artifact = fixture_artifact();
    let dir = tempdir();

    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        (
            "truncated header",
            artifact[..8.min(artifact.len())].to_vec(),
        ),
        ("truncated body", artifact[..artifact.len() / 2].to_vec()),
        ("bad magic", {
            let mut bytes = artifact.clone();
            bytes[0] ^= 0xFF;
            bytes
        }),
    ];

    for (label, bytes) in cases {
        let path = dir.join(format!("{}.spi", label.replace(' ', "-")));
        std::fs::write(&path, &bytes).expect("write case");

        let mapped = MmapPreciseEphemerisInterpolant::from_path(&path);
        let owned = MmapPreciseEphemerisInterpolant::from_vec(bytes.clone());

        assert_eq!(
            mapped.is_ok(),
            owned.is_ok(),
            "{label}: mapped and owned disagreed on acceptance"
        );
        if let (Err(mapped_err), Err(owned_err)) = (&mapped, &owned) {
            assert_eq!(
                mapped_err.to_string(),
                owned_err.to_string(),
                "{label}: mapped and owned produced different errors"
            );
        }
    }
}

#[test]
fn the_reader_outlives_its_file() {
    let artifact = fixture_artifact();
    let dir = tempdir();
    let path = dir.join("interpolant.spi");
    std::fs::write(&path, &artifact).expect("write artifact");

    let reader = MmapPreciseEphemerisInterpolant::from_path(&path).expect("open mapped");
    std::fs::remove_file(&path).expect("remove the file out from under the reader");

    let sat = reader.satellites()[0];
    let state = reader.position_at_j2000_seconds(sat, 646_315_200.0);
    assert!(
        state.is_ok(),
        "the reader must keep working after its file is unlinked: {state:?}"
    );
    assert_eq!(reader.as_bytes().len(), artifact.len());
}
