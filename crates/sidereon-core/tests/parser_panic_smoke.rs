//! Deterministic byte-smoke harness for the hand-rolled binary/text parsers.
//!
//! cargo-fuzz's libfuzzer runner is not always usable in CI, so this test feeds
//! many random, truncated, and structurally-corrupted byte buffers through the
//! SPK/DAF binary parser and the NAVCEN HTML parser and asserts each call
//! returns a `Result` without panicking (the test process surviving is the
//! assertion). The byte stream is a fixed splitmix64 sequence with no external
//! entropy, so a failure reproduces exactly.

use sidereon_core::astro::spk::{parse_daf_spk, Spk};

/// splitmix64: a tiny, fully deterministic PRNG so corpora are reproducible.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fill(&mut self, buf: &mut [u8]) {
        for byte in buf.iter_mut() {
            *byte = self.next_u64() as u8;
        }
    }
}

/// A valid little-endian DAF/SPK file-record prefix so a share of the corpus
/// clears the magic/format/shape gates and reaches the summary + segment readers
/// instead of bouncing off the header check.
fn write_daf_header(buf: &mut [u8], rng: &mut SplitMix64) {
    if buf.len() < 96 {
        return;
    }
    buf[0..8].copy_from_slice(b"DAF/SPK ");
    buf[8..12].copy_from_slice(&2i32.to_le_bytes());
    buf[12..16].copy_from_slice(&6i32.to_le_bytes());
    // Randomize FWARD/BWARD/FREE so the summary-record chain walks hostile values.
    let fward = (rng.next_u64() % 8) as i32;
    let bward = (rng.next_u64() % 8) as i32;
    buf[76..80].copy_from_slice(&fward.to_le_bytes());
    buf[80..84].copy_from_slice(&bward.to_le_bytes());
    buf[84..88].copy_from_slice(&(rng.next_u64() as i32).to_le_bytes());
    buf[88..96].copy_from_slice(b"LTL-IEEE");
}

fn exercise_spk(bytes: &[u8]) {
    // Directory parse must not panic on any byte slice.
    let _ = parse_daf_spk(bytes);

    // If a full kernel parses, drive the type-2/3/21 evaluators with queries
    // derived from the (untrusted) segment descriptors so the binary-offset
    // record readers run on hostile coverage windows.
    if let Ok(spk) = Spk::from_bytes(bytes) {
        for segment in spk.segments().iter().take(8) {
            let mid = 0.5 * (segment.start_et + segment.stop_et);
            for et in [
                segment.start_et,
                segment.stop_et,
                mid,
                0.0,
                f64::NAN,
                f64::INFINITY,
            ] {
                let _ = spk.spk_state(segment.target, segment.center, et);
            }
        }
    }
}

#[test]
fn spk_parser_survives_hostile_bytes() {
    let mut rng = SplitMix64(0x5151_DE60_0F0F_2026);
    // Lengths spanning sub-header, exact records, and multi-record kernels.
    let lengths = [
        0usize, 1, 7, 8, 16, 95, 96, 1023, 1024, 1025, 2047, 2048, 3072, 4096, 5121,
    ];

    for &len in &lengths {
        for case in 0..400u32 {
            let mut buf = vec![0u8; len];
            rng.fill(&mut buf);
            // Half random, half magic-prefixed to reach deep parser paths.
            if case % 2 == 0 {
                write_daf_header(&mut buf, &mut rng);
            }
            exercise_spk(&buf);
        }
    }

    // All-zero, all-ones, and magic-only buffers as explicit boundary cases.
    exercise_spk(&[0u8; 4096]);
    exercise_spk(&[0xFFu8; 4096]);
    let mut magic_only = vec![0u8; 1024];
    write_daf_header(&mut magic_only, &mut rng);
    exercise_spk(&magic_only);
}

#[test]
fn navcen_parser_survives_hostile_bytes() {
    use sidereon_core::constellation::parse_navcen;

    // Markup fragments the scanner keys on, so corruption lands mid-token and
    // mid-tag rather than only on random noise.
    const TOKENS: &[&[u8]] = &[
        b"<tr",
        b"</tr>",
        b"<td",
        b"</td>",
        b">",
        b"views-field-field-gps-prn",
        b"views-field-field-gps-svn",
        b"views-field-field-nanu-type",
        b"nanu-active-check",
        b"&#",
        b"&#x",
        b";",
        b"&amp;",
        b"\xE2\x82\xAC", // multi-byte UTF-8 (euro sign) to probe slice boundaries
        b"123",
        b"  ",
    ];

    let mut rng = SplitMix64(0xC0FF_EE12_3456_789A);

    for _ in 0..4000u32 {
        let mut buf: Vec<u8> = Vec::new();
        let pieces = (rng.next_u64() % 24) as usize;
        for _ in 0..pieces {
            if rng.next_u64() & 1 == 0 {
                let token = TOKENS[(rng.next_u64() as usize) % TOKENS.len()];
                buf.extend_from_slice(token);
            } else {
                buf.push(rng.next_u64() as u8);
            }
        }
        // parse_navcen rejects non-UTF-8 up front; both UTF-8 and non-UTF-8
        // buffers must merely return, never panic.
        let _ = parse_navcen(&buf);
    }

    // Pure random byte buffers (mostly invalid UTF-8) and boundary cases.
    for len in [0usize, 1, 2, 3, 4, 64, 256, 1024] {
        for _ in 0..200u32 {
            let mut buf = vec![0u8; len];
            rng.fill(&mut buf);
            let _ = parse_navcen(&buf);
        }
    }
    let _ = parse_navcen(b"<tr><td class=\"views-field-field-gps-prn\">\xC3\x28</td></tr>");
}
