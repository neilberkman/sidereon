//! CRINEX decoder tests: kernel edge cases on tiny inline strings, plus a
//! round-trip against a committed real `.crx` and its `crx2rnx`-decoded `.rnx`.

use super::*;

fn esbc_crx() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.crx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read CRINEX fixture {path}: {e}"))
}

fn esbc_reference_rnx() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read RINEX fixture {path}: {e}"))
}

fn assert_decode_parse_err(text: String) {
    let err = decode(&text).unwrap_err();
    assert!(matches!(err, Error::Parse(_)), "{err}");
}

fn labeled_header_line(prefix: &str, label: &str) -> String {
    format!("{prefix:<60}{label}")
}

fn v3_obs_types_line() -> String {
    let mut bytes = vec![b' '; 80];
    bytes[0] = b'G';
    bytes[5] = b'1';
    bytes[7..10].copy_from_slice(b"C1C");
    bytes[60..79].copy_from_slice(b"SYS / # / OBS TYPES");
    String::from_utf8(bytes).expect("ASCII header line")
}

fn v3_single_sat_epoch_line() -> String {
    v3_single_sat_epoch_line_with_token("G01")
}

fn v3_single_sat_epoch_line_with_token(sv_token: &str) -> String {
    let mut bytes = vec![b' '; 44];
    let prefix = b"> 2020 01 01 00 00  0.0000000";
    bytes[..prefix.len()].copy_from_slice(prefix);
    bytes[31] = b'0';
    bytes[34] = b'1';
    bytes[41..44].copy_from_slice(b"G01");
    let mut line = String::from_utf8(bytes).expect("ASCII epoch line");
    line.replace_range(41..44, sv_token);
    line
}

fn v3_event_epoch_line(event_line_count: usize) -> String {
    let mut bytes = vec![b' '; 35];
    let prefix = b"> 2020 01 01 00 00  0.0000000";
    bytes[..prefix.len()].copy_from_slice(prefix);
    bytes[31] = b'2';
    bytes[32..35].copy_from_slice(format!("{event_line_count:3}").as_bytes());
    String::from_utf8(bytes).expect("ASCII epoch line")
}

fn v1_single_sat_epoch_line(sv_token: &str) -> String {
    let mut descriptor = [b' '; 32];
    descriptor[26..29].copy_from_slice(b"  0");
    descriptor[29..32].copy_from_slice(b"  1");
    format!(
        "&{}{sv_token}",
        std::str::from_utf8(&descriptor[1..]).expect("ASCII epoch descriptor")
    )
}

fn v1_event_epoch_line(event_line_count: usize) -> String {
    let mut descriptor = [b' '; 32];
    descriptor[26..29].copy_from_slice(b"  2");
    descriptor[29..32].copy_from_slice(format!("{event_line_count:3}").as_bytes());
    format!(
        "&{}",
        std::str::from_utf8(&descriptor[1..]).expect("ASCII epoch descriptor")
    )
}

fn minimal_v1_crinex(sv_token: &str) -> String {
    [
        labeled_header_line(
            "1.0                 COMPACT RINEX FORMAT",
            "CRINEX VERS   / TYPE",
        ),
        "RNX2CRX".to_string(),
        labeled_header_line(
            "     2.11           OBSERVATION DATA    G                   ",
            "RINEX VERSION / TYPE",
        ),
        labeled_header_line("     1    C1", "# / TYPES OF OBSERV"),
        labeled_header_line("", "END OF HEADER"),
        v1_single_sat_epoch_line(sv_token),
        String::new(),
    ]
    .join("\n")
}

fn truncated_v1_event_crinex() -> String {
    [
        labeled_header_line(
            "1.0                 COMPACT RINEX FORMAT",
            "CRINEX VERS   / TYPE",
        ),
        "RNX2CRX".to_string(),
        labeled_header_line(
            "     2.11           OBSERVATION DATA    G                   ",
            "RINEX VERSION / TYPE",
        ),
        labeled_header_line("     1    C1", "# / TYPES OF OBSERV"),
        labeled_header_line("", "END OF HEADER"),
        v1_event_epoch_line(2),
        labeled_header_line("only one event line", "COMMENT"),
    ]
    .join("\n")
}

fn minimal_v3_crinex(sv_token: &str) -> String {
    [
        labeled_header_line("3.0", "CRINEX VERS   / TYPE"),
        "RNX2CRX".to_string(),
        labeled_header_line(
            "     3.04           OBSERVATION DATA    G                   ",
            "RINEX VERSION / TYPE",
        ),
        v3_obs_types_line(),
        labeled_header_line("", "END OF HEADER"),
        v3_single_sat_epoch_line_with_token(sv_token),
        String::new(),
        "1&0".to_string(),
    ]
    .join("\n")
}

fn truncated_v3_event_crinex() -> String {
    [
        labeled_header_line("3.0", "CRINEX VERS   / TYPE"),
        "RNX2CRX".to_string(),
        labeled_header_line(
            "     3.04           OBSERVATION DATA    G                   ",
            "RINEX VERSION / TYPE",
        ),
        v3_obs_types_line(),
        labeled_header_line("", "END OF HEADER"),
        v3_event_epoch_line(2),
        labeled_header_line("only one event line", "COMMENT"),
    ]
    .join("\n")
}

fn overflowing_v3_crinex() -> String {
    [
        labeled_header_line("3.0", "CRINEX VERS   / TYPE"),
        "RNX2CRX".to_string(),
        labeled_header_line(
            "     3.04           OBSERVATION DATA    G                   ",
            "RINEX VERSION / TYPE",
        ),
        v3_obs_types_line(),
        labeled_header_line("", "END OF HEADER"),
        v3_single_sat_epoch_line(),
        String::new(),
        "1&9223372036854775807".to_string(),
        v3_single_sat_epoch_line(),
        String::new(),
        "1".to_string(),
    ]
    .join("\n")
}

fn corrupt_header_field(text: String, label: &str, start: usize, end: usize) -> String {
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let line = lines
        .iter_mut()
        .find(|line| line.contains(label))
        .unwrap_or_else(|| panic!("header label {label:?} present"));
    let mut bytes = line.as_bytes().to_vec();
    assert!(bytes.len() >= end);
    for byte in &mut bytes[start..end] {
        *byte = b' ';
    }
    bytes[end - 1] = b'X';
    *line = String::from_utf8(bytes).expect("ASCII CRINEX line");
    lines.join("\n")
}

fn corrupt_first_epoch_flag(text: String, prefix: char, start: usize, end: usize) -> String {
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let line = lines
        .iter_mut()
        .find(|line| line.starts_with(prefix))
        .unwrap_or_else(|| panic!("epoch line with prefix {prefix:?} present"));
    let mut bytes = line.as_bytes().to_vec();
    assert!(bytes.len() >= end);
    let offset = bytes[start..end]
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .expect("epoch flag byte present");
    bytes[start + offset] = b'X';
    *line = String::from_utf8(bytes).expect("ASCII CRINEX line");
    lines.join("\n")
}

#[test]
fn numdiff_third_order_recovers_reference_sequence() {
    // The canonical Hatanaka NumDiff sequence (level 3, the RNX2CRX default).
    let mut diff = NumDiff::new(126_298_057_858, 3);
    assert_eq!(diff.decompress(-15_603_288).unwrap(), 126_282_454_570);
    assert_eq!(diff.decompress(521_089).unwrap(), 126_267_372_371);
    assert_eq!(diff.decompress(-752).unwrap(), 126_252_810_509);
    assert_eq!(diff.decompress(1_575_419_284).unwrap(), 127_814_188_268);
    assert_eq!(diff.decompress(-3_150_848_707).unwrap(), 127_800_656_941);

    // Arc reinitialization mid-stream.
    diff.force_init(111_982_965_979, 3);
    assert_eq!(diff.decompress(-16_266_911).unwrap(), 111_966_699_068);
    assert_eq!(diff.decompress(609_858).unwrap(), 111_951_042_015);
    assert_eq!(diff.decompress(-213).unwrap(), 111_935_994_607);
}

#[test]
fn decode_rejects_numdiff_overflow() {
    let err = decode(&overflowing_v3_crinex()).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref msg) if msg.contains("integer arithmetic overflow")),
        "{err}"
    );
}

#[test]
fn textdiff_keeps_blanks_and_overwrites() {
    let mut diff = TextDiff::default();
    diff.force_init("ABCDEFG 12 000 33 XXACQmpLf");
    // Space keeps, non-space overwrites, '&' blanks.
    let out = diff.decompress("         3   1 44 xxACq   F");
    assert_eq!(out, "ABCDEFG 13 001 44 xxACqmpLF");
    // A '&' blanks the corresponding column.
    let out = diff.decompress("&");
    assert_eq!(out, " BCDEFG 13 001 44 xxACqmpLF");
}

#[test]
fn parse_reset_distinguishes_reset_from_delta() {
    assert_eq!(
        parse_reset("3&126298057858").unwrap(),
        Some((3, 126_298_057_858))
    );
    assert_eq!(parse_reset("  -15603288  ").unwrap(), None);
    assert!(parse_reset("9&1").is_err()); // order out of range
    assert!(parse_reset("x&1").is_err()); // bad order
}

#[test]
fn format_value_matches_rinex_f14_3() {
    assert_eq!(format_value(40_715_949_461), "  40715949.461");
    assert_eq!(format_value(-2_196), "        -2.196");
    // crx2rnx drops the leading zero only for a negative value in (-1, 0).
    assert_eq!(format_value(-920), "         -.920");
    assert_eq!(format_value(515), "         0.515");
    assert_eq!(format_value(0), "         0.000");
}

#[test]
fn decode_rejects_unknown_crinex_version() {
    let bad = "2.0                 COMPACT RINEX FORMAT                    CRINEX VERS   / TYPE\nRNX2CRX\n";
    let err = decode(bad).unwrap_err();
    assert!(matches!(err, Error::Parse(_)));
}

#[test]
fn decode_rejects_stream_without_crinex_header() {
    let err = decode("not a crinex file\n").unwrap_err();
    assert!(matches!(err, Error::Parse(_)));
}

#[test]
fn decode_rejects_non_ascii_v1_sv_token_without_panic() {
    assert_decode_parse_err(minimal_v1_crinex("G\u{FFFD}1"));
}

#[test]
fn decode_rejects_non_ascii_v3_sv_token_without_panic() {
    assert_decode_parse_err(minimal_v3_crinex("G\u{FFFD}1"));
}

#[test]
fn decode_rejects_truncated_v3_event_record() {
    let err = decode(&truncated_v3_event_crinex()).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref msg) if msg.contains("CRINEX V3 event record truncated")),
        "{err}"
    );
}

#[test]
fn decode_rejects_truncated_v1_event_record() {
    let err = decode(&truncated_v1_event_crinex()).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref msg) if msg.contains("CRINEX V1 event record truncated")),
        "{err}"
    );
}

#[test]
fn decode_rejects_malformed_v3_observation_count() {
    assert_decode_parse_err(corrupt_header_field(
        esbc_crx(),
        "SYS / # / OBS TYPES",
        3,
        6,
    ));
}

#[test]
fn decode_rejects_malformed_v1_observation_count() {
    assert_decode_parse_err(corrupt_header_field(
        algo_v1_crx(),
        "# / TYPES OF OBSERV",
        0,
        6,
    ));
}

#[test]
fn decode_rejects_malformed_v3_epoch_flag() {
    assert_decode_parse_err(corrupt_first_epoch_flag(esbc_crx(), '>', 31, 32));
}

#[test]
fn decode_rejects_malformed_v1_epoch_flag() {
    assert_decode_parse_err(corrupt_first_epoch_flag(algo_v1_crx(), '&', 26, 29));
}

#[test]
fn round_trip_matches_crx2rnx_reference_byte_for_byte() {
    let decoded = decode(&esbc_crx()).expect("decode CRINEX fixture");
    let reference = esbc_reference_rnx();

    // Compare line by line so a mismatch points at the offending record.
    let dec_lines: Vec<&str> = decoded.lines().collect();
    let ref_lines: Vec<&str> = reference.lines().collect();
    assert_eq!(
        dec_lines.len(),
        ref_lines.len(),
        "line count differs: decoded {} vs reference {}",
        dec_lines.len(),
        ref_lines.len()
    );
    for (i, (d, r)) in dec_lines.iter().zip(ref_lines.iter()).enumerate() {
        assert_eq!(
            d,
            r,
            "line {} differs\n  decoded:  {:?}\n  reference:{:?}",
            i + 1,
            d,
            r
        );
    }
}

/// Assert two RINEX-text expansions are byte-identical, line by line.
fn assert_same_expansion(a: &str, b: &str) {
    let a_lines: Vec<&str> = a.lines().collect();
    let b_lines: Vec<&str> = b.lines().collect();
    assert_eq!(
        a_lines.len(),
        b_lines.len(),
        "line count differs: {} vs {}",
        a_lines.len(),
        b_lines.len()
    );
    for (i, (x, y)) in a_lines.iter().zip(b_lines.iter()).enumerate() {
        assert_eq!(x, y, "line {} differs\n  a: {:?}\n  b: {:?}", i + 1, x, y);
    }
}

// CRINEX round-trip: the canonical IR is the recovered observation stream. The
// serializer `encode_stream` re-emits CRINEX (in canonical all-reset form, which
// need not match the source CRINEX byte-for-byte). The round-trip guarantee is
// at the IR / RINEX-text level: re-decoding the re-emitted CRINEX yields the same
// plain RINEX text, and re-parsing it yields the same IR. Verified on both the
// real CRINEX-3 and CRINEX-1 fixtures.
#[test]
fn round_trip_v3_serializer_reproduces_decoded_text_and_ir() {
    let crx = esbc_crx();
    let stream = parse_stream(&crx).expect("parse v3 stream to IR");
    let reencoded = encode_stream(&stream);

    // Re-decoding the re-emitted CRINEX reproduces the reference expansion.
    assert_same_expansion(
        &decode(&reencoded).expect("decode re-emitted v3 CRINEX"),
        &esbc_reference_rnx(),
    );
    // The IR is stable through encode -> parse.
    assert_eq!(stream, parse_stream(&reencoded).expect("re-parse v3 IR"));
}

#[test]
fn round_trip_v1_serializer_reproduces_decoded_text_and_ir() {
    let crx = algo_v1_crx();
    let stream = parse_stream(&crx).expect("parse v1 stream to IR");
    let reencoded = encode_stream(&stream);

    assert_same_expansion(
        &decode(&reencoded).expect("decode re-emitted v1 CRINEX"),
        &algo_v1_reference_rnx(),
    );
    assert_eq!(stream, parse_stream(&reencoded).expect("re-parse v1 IR"));
}

// Public compress path: plain RINEX observation text -> encode_crinex -> CRINEX,
// then decode_crinex back. The encoder emits the canonical all-reset CRINEX form
// (not byte-identical to the original RNX2CRX stream), so the round-trip is
// checked at the RINEX-text level: decoding the freshly encoded CRINEX must
// reproduce the original plain RINEX observations byte-for-byte.
#[test]
fn round_trip_v3_encode_crinex_reproduces_plain_rinex() {
    let rnx = esbc_reference_rnx();
    let crinex = encode_crinex(&rnx).expect("encode plain RINEX-3 to CRINEX");
    // The re-emitted CRINEX is a valid stream that decodes back to the input.
    assert_same_expansion(&decode(&crinex).expect("decode encoded CRINEX-3"), &rnx);
}

#[test]
fn round_trip_v1_encode_crinex_reproduces_plain_rinex() {
    let rnx = algo_v1_reference_rnx();
    let crinex = encode_crinex(&rnx).expect("encode plain RINEX-2 to CRINEX");
    assert_same_expansion(&decode(&crinex).expect("decode encoded CRINEX-1"), &rnx);
}

// Full loop from the CRINEX fixtures: decode to plain RINEX, re-encode, decode
// again, and confirm the observations survive the CRINEX -> RINEX -> CRINEX trip.
#[test]
fn round_trip_v3_crinex_to_rinex_to_crinex() {
    let plain = decode(&esbc_crx()).expect("decode CRINEX-3 fixture");
    let crinex = encode_crinex(&plain).expect("re-encode plain RINEX-3");
    assert_same_expansion(
        &decode(&crinex).expect("decode re-encoded CRINEX-3"),
        &plain,
    );
}

#[test]
fn round_trip_v1_crinex_to_rinex_to_crinex() {
    let plain = decode(&algo_v1_crx()).expect("decode CRINEX-1 fixture");
    let crinex = encode_crinex(&plain).expect("re-encode plain RINEX-2");
    assert_same_expansion(
        &decode(&crinex).expect("decode re-encoded CRINEX-1"),
        &plain,
    );
}

fn algo_v1_crx() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/algo0010_2015001_v1_trim.crx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read CRINEX v1 fixture {path}: {e}"))
}

fn algo_v1_reference_rnx() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/algo0010_2015001_v1_trim.rnx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read RINEX v1 fixture {path}: {e}"))
}

#[test]
fn round_trip_v1_matches_crx2rnx_reference_byte_for_byte() {
    // CRINEX 1.0 (RINEX 2) path: a mixed GPS+GLONASS epoch carrying 20 satellites
    // (so the 12-satellite epoch-line wrap fires) with 8 observation types (the
    // five-observations-per-line wrap). Compared byte-for-byte against the
    // crx2rnx-decoded reference.
    let decoded = decode(&algo_v1_crx()).expect("decode CRINEX v1 fixture");
    let reference = algo_v1_reference_rnx();

    let dec_lines: Vec<&str> = decoded.lines().collect();
    let ref_lines: Vec<&str> = reference.lines().collect();
    assert_eq!(
        dec_lines.len(),
        ref_lines.len(),
        "line count differs: decoded {} vs reference {}",
        dec_lines.len(),
        ref_lines.len()
    );
    for (i, (d, r)) in dec_lines.iter().zip(ref_lines.iter()).enumerate() {
        assert_eq!(
            d,
            r,
            "line {} differs\n  decoded:  {:?}\n  reference:{:?}",
            i + 1,
            d,
            r
        );
    }
}
