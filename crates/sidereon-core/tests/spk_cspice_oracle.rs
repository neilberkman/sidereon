#![cfg(sidereon_repo_tests)]
//! SPK states from the DE421 excerpt `observe_de.bsp` against CSPICE
//! `spkgeo_c`, bit for bit.
//!
//! The fixture holds CSPICE N0067 `spkgeo_c` states for six target/observer
//! pairs at the segment start and stop, at a record boundary shared by every
//! segment's record grid (751291200.0), and at an interior epoch
//! (755000000.5), in J2000, plus two pairs rotated into ECLIPJ2000, B1950 and
//! GALACTIC. CSPICE computed them from the full `de421.bsp` the excerpt was
//! cut from; the excerpt keeps whole type-2 records, so the evaluated words
//! are the same. CSPICE was built with `-ffp-contract=off`, so each operation
//! rounds separately as the Fortran source states; a build that fuses
//! multiply-adds differs in the last bits. Every component must match to
//! 0 ULP. `fixtures/spk/gen_spkgeo_cspice.c` regenerates the fixture.

use sidereon_core::astro::spk::Spk;

const KERNEL: &[u8] = include_bytes!("fixtures/bodies/observe_de.bsp");
const ORACLE: &str = include_str!("fixtures/spk/observe_de_spkgeo_cspice.json");

/// NAIF built-in inertial frame id for each frame name the fixture uses.
fn naif_frame(name: &str) -> i32 {
    match name {
        "J2000" => 1,
        "B1950" => 2,
        "GALACTIC" => 13,
        "ECLIPJ2000" => 17,
        other => panic!("fixture names an unexpected frame {other}"),
    }
}

fn bits(value: &serde_json::Value) -> u64 {
    let text = value.as_str().expect("hex bit pattern");
    u64::from_str_radix(text.trim_start_matches("0x"), 16).expect("hex bit pattern")
}

#[test]
fn spk_states_match_cspice_spkgeo_bit_for_bit() {
    let kernel = Spk::from_bytes(KERNEL).expect("observe_de.bsp parses");
    let oracle: serde_json::Value = serde_json::from_str(ORACLE).expect("oracle JSON");
    let cases = oracle["cases"].as_array().expect("cases array");
    assert_eq!(cases.len(), 36);

    let mut mismatches = Vec::new();
    for case in cases {
        let target = case["target"].as_i64().expect("target") as i32;
        let observer = case["observer"].as_i64().expect("observer") as i32;
        let frame_name = case["frame"].as_str().expect("frame");
        let frame = naif_frame(frame_name);
        let et = f64::from_bits(bits(&case["et_bits"]));
        assert_eq!(et, case["et"].as_f64().expect("et"));

        let state = if frame == 1 {
            kernel.spk_state(target, observer, et)
        } else {
            kernel.spk_state_in_frame(target, observer, et, frame)
        }
        .unwrap_or_else(|error| panic!("{target} wrt {observer} at {et} in {frame_name}: {error}"));
        assert_eq!(state.frame, frame);

        let actual = [
            state.position_km[0],
            state.position_km[1],
            state.position_km[2],
            state.velocity_km_s[0],
            state.velocity_km_s[1],
            state.velocity_km_s[2],
        ];
        let expected = case["state_bits"].as_array().expect("state_bits");
        for (component, (value, expected)) in actual.iter().zip(expected).enumerate() {
            let expected = bits(expected);
            if value.to_bits() != expected {
                let ulps = value.to_bits() as i64 - expected as i64;
                mismatches.push(format!(
                    "{target} wrt {observer} at {et} in {frame_name}, component {component}: \
                     {value:e} ({:#018x}) vs CSPICE {:e} ({expected:#018x}), {ulps} ULP",
                    value.to_bits(),
                    f64::from_bits(expected),
                ));
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} components differ from CSPICE spkgeo_c:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
