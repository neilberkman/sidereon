#![cfg(sidereon_repo_tests)]
//! Authoritative TLE round-trip gate, ported from the sidereon Elixir suite.
//!
//! Parsing then re-encoding every CelesTrak stations TLE must reproduce the
//! original lines character-for-character, checksums included. The writer
//! restates each assumed-decimal field's source spelling, so a zero written
//! `+0` stays `+0`.

use sidereon_core::astro::tle;

const STATIONS: &str = include_str!("fixtures/celestrak/stations.tle");

fn tle_pairs(body: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = body
        .trim()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    lines
        .chunks(3)
        .filter_map(|chunk| match chunk {
            [_name, l1, l2] if l1.starts_with("1 ") && l2.starts_with("2 ") => {
                Some((l1.to_string(), l2.to_string()))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn iss_round_trips_character_exact() {
    let l1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993";
    let l2 = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106";
    let parsed = tle::parse(l1, l2).unwrap();
    let (gen_l1, gen_l2) = tle::encode(&parsed.elements).unwrap();
    assert_eq!(gen_l1, l1);
    assert_eq!(gen_l2, l2);
}

#[test]
fn all_stations_round_trip() {
    let pairs = tle_pairs(STATIONS);
    assert!(
        pairs.len() > 10,
        "expected >10 station TLEs, got {}",
        pairs.len()
    );

    for (l1, l2) in pairs {
        let parsed = tle::parse(&l1, &l2).unwrap();
        let (gen_l1, gen_l2) = tle::encode(&parsed.elements).unwrap();

        assert_eq!(
            gen_l2, l2,
            "line 2 mismatch for {}",
            parsed.elements.catalog_number
        );
        assert_eq!(
            gen_l1, l1,
            "line 1 mismatch for {}",
            parsed.elements.catalog_number
        );
    }
}
