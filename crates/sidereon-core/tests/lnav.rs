#![cfg(sidereon_repo_tests)]
//! Authoritative 0-ULP golden for the GPS LNAV codec.
//!
//! Pins the codec against the Python reference generator
//! (`parity/generator`, captured in `orbis_gnss_application_golden.json`): the
//! IS-GPS-200 Table 20-XIV parity vectors, and the exact 300-bit subframes plus
//! per-word hex produced by encoding the canonical example ephemeris with the
//! recorded TLM/HOW options. This is bit-for-bit equality, so the crate alone
//! proves the codec correct without the sidereon suite.

use serde_json::Value;
use sidereon_core::navigation::lnav::{self, LnavNumber, LnavOptions, LnavParams};

const GOLDEN: &str = include_str!("fixtures/orbis_gnss_application_golden.json");

fn n(v: f64) -> LnavNumber {
    LnavNumber::Float(v)
}
fn i(v: i64) -> LnavNumber {
    LnavNumber::Int(v)
}

/// The canonical example MEO GPS SV (Elixir `Ephemeris.example/0`); the golden
/// pins the subframes this encodes to.
fn example() -> LnavParams {
    LnavParams {
        week_number: i(290),
        l2_code: i(1),
        l2_p_data_flag: i(0),
        ura_index: i(0),
        sv_health: i(0),
        iodc: i(0x2AB),
        tgd: n(-5.587_935_447_692_871e-9),
        toc: i(504_000),
        af0: n(-1.234e-4),
        af1: n(-3.5e-12),
        af2: n(0.0),
        iode: i(0xAB),
        crs: n(-55.625),
        delta_n: n(1.56e-9),
        m0: n(-0.35),
        cuc: n(-1.2e-6),
        eccentricity: n(0.012),
        cus: n(8.3e-6),
        sqrt_a: n(5153.65),
        toe: i(504_000),
        fit_interval_flag: i(0),
        aodo: i(0),
        cic: n(5.0e-8),
        omega0: n(-0.78),
        cis: n(-2.1e-7),
        i0: n(0.305),
        crc: n(250.625),
        omega: n(0.95),
        omega_dot: n(-8.1e-9),
        idot: n(1.5e-10),
    }
}

fn bits_to_string(bits: &[u8]) -> String {
    bits.iter()
        .map(|b| if *b == 1 { '1' } else { '0' })
        .collect()
}

fn word_hex(word: &[u8]) -> String {
    let value = word.iter().fold(0u32, |acc, &b| (acc << 1) | u32::from(b));
    format!("0x{value:08x}")
}

#[test]
fn parity_vectors_match_reference_generator() {
    let golden: Value = serde_json::from_str(GOLDEN).expect("golden json");
    let cases = golden["lnav"]["parity_cases"]
        .as_array()
        .expect("parity_cases");

    for case in cases {
        let data24: Vec<u8> = case["data24"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let d29 = case["d29_prev"].as_u64().unwrap() as u8;
        let d30 = case["d30_prev"].as_u64().unwrap() as u8;
        let expected: Vec<u8> = case["parity"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();

        assert_eq!(lnav::parity(&data24, d29, d30).unwrap().to_vec(), expected);
    }
}

#[test]
fn encoded_subframes_match_reference_generator_bit_for_bit() {
    let golden: Value = serde_json::from_str(GOLDEN).expect("golden json");
    let lnav = &golden["lnav"];
    let opts_json = &lnav["options"];

    let opts = LnavOptions {
        tow: i(opts_json["tow"].as_i64().unwrap()),
        alert: i(opts_json["alert"].as_i64().unwrap()),
        anti_spoof: i(opts_json["anti_spoof"].as_i64().unwrap()),
        integrity: i(opts_json["integrity"].as_i64().unwrap()),
        tlm_message: i(opts_json["tlm_message"].as_i64().unwrap()),
    };

    let subframes = lnav::encode(&example(), &opts).expect("encode");

    for (idx, bits) in subframes.iter().enumerate() {
        let sf = (idx + 1).to_string();

        let expected_bits = lnav["subframes"][&sf].as_str().expect("subframe bits");
        assert_eq!(bits_to_string(bits), expected_bits, "subframe {sf} bits");

        let expected_words = lnav["word_hex"][&sf].as_array().expect("word_hex");
        let words: Vec<String> = bits.chunks(lnav::WORD_LENGTH).map(word_hex).collect();
        let expected: Vec<&str> = expected_words.iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(words, expected, "subframe {sf} word hex");
    }
}
