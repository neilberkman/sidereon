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
        // An unchanged descriptor, not a reset, so G01 keeps its arc.
        " ".to_string(),
        String::new(),
        "1".to_string(),
    ]
    .join("\n")
}

#[test]
fn a_satellite_after_a_reset_epoch_starts_a_new_arc() {
    // `crx2rnx` reads every satellite of a reset epoch as new, so a difference
    // there continues no arc; it used to continue the one before the reset.
    let text =
        overflowing_v3_crinex().replace("\n \n", &format!("\n{}\n", v3_single_sat_epoch_line()));
    let err = decode(&text).unwrap_err();
    assert!(
        matches!(err, Error::Parse(ref msg) if msg.contains("delta before any arc init")),
        "{err}"
    );
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

/// A RINEX 2.11 observation file with ten observation types, the tenth on the
/// continuation record version 2 writes past nine, and one epoch.
fn v2_continued_types_rinex() -> String {
    let mut types = String::from("    10");
    for code in ["C1", "P1", "L1", "D1", "S1", "C2", "P2", "L2", "D2"] {
        types.push_str(&format!("    {code}"));
    }
    let mut values = String::new();
    for index in 0..10 {
        values.push_str(&format!("{:14.3}  ", 20_000_000.0 + f64::from(index)));
        if index % 5 == 4 {
            values = values.trim_end().to_string();
            values.push('\n');
        }
    }
    [
        labeled_header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        labeled_header_line(&types, "# / TYPES OF OBSERV"),
        labeled_header_line("          S2", "# / TYPES OF OBSERV"),
        labeled_header_line("", "END OF HEADER"),
        " 20  1  1  0  0  0.0000000  0  1G01".to_string(),
        values.trim_end().to_string(),
        String::new(),
    ]
    .join("\n")
}

#[test]
fn a_version_two_type_list_continued_past_nine_codes_is_compressed_and_expanded() {
    // Version 2 writes an eleventh and later code on `# / TYPES OF OBSERV`
    // records with a blank count. Every record was read for a count, so a
    // file with more than nine types could be neither compressed nor
    // expanded.
    let rinex = v2_continued_types_rinex();
    let compressed = encode_crinex(&rinex).expect("compress a continued type list");
    let expanded = decode(&compressed).expect("expand it again");
    let lines = |text: &str| -> Vec<String> {
        text.lines()
            .map(|line| line.trim_end().to_string())
            .collect()
    };
    assert_eq!(lines(&expanded), lines(&rinex));
}

fn obs_fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/obs/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

fn assert_same_lines(found: &str, expected: &str, what: &str) {
    let found: Vec<&str> = found.lines().collect();
    let expected: Vec<&str> = expected.lines().collect();
    for (index, (found, expected)) in found.iter().zip(&expected).enumerate() {
        assert_eq!(found, expected, "{what}: line {}", index + 1);
    }
    assert_eq!(found.len(), expected.len(), "{what}: line count");
}

#[test]
fn flags_and_arcs_carry_only_where_crx2rnx_carries_them() {
    // Review found expansion carrying a CRINEX 1 observation's flags across an
    // epoch where it was missing, so an observation flagged, then missing, then
    // back unflagged came back flagged. Each `.crx` was compressed from its
    // `.rnx` by RNX2CRX 4.2.0, and CRX2RNX 4.2.0 expands it back to that `.rnx`
    // byte for byte. Each has flags that repeat, a flagged observation missing
    // and back unflagged, a satellite missing from an epoch and back, and a
    // flag 4 event declaring a longer type list on two records; the version 3
    // file also has blank fields carrying flags.
    for version in ["v2", "v3"] {
        let rinex = obs_fixture(&format!("crinex_flag_state_{version}.rnx"));
        let expanded = decode(&obs_fixture(&format!("crinex_flag_state_{version}.crx")))
            .unwrap_or_else(|e| panic!("{version}: expand the RNX2CRX file: {e}"));
        assert_same_lines(&expanded, &rinex, version);

        let compressed =
            encode_crinex(&rinex).unwrap_or_else(|e| panic!("{version}: compress: {e}"));
        let expanded = decode(&compressed).unwrap_or_else(|e| panic!("{version}: expand: {e}"));
        assert_same_lines(&expanded, &rinex, version);
        let stream = parse_stream(&compressed).expect("read the compression");
        assert_eq!(
            parse_stream(&encode_stream(&stream)).expect("read it again"),
            stream,
            "{version}"
        );
    }
}

#[test]
fn type_records_that_continue_no_list_or_end_short_are_refused() {
    // Review built the first two: a blank-count record before any count, and one
    // as the only type record. Reading a blank count as a continuation let both
    // through, although neither continues a declared list.
    let version = labeled_header_line(
        "     2.11           OBSERVATION DATA    G (GPS)",
        "RINEX VERSION / TYPE",
    );
    let types = |text: &str| labeled_header_line(text, "# / TYPES OF OBSERV");
    let end = labeled_header_line("", "END OF HEADER");
    let nine = "    10    C1    P1    L1    D1    S1    C2    P2    L2    D2";
    let cases = [
        (
            "a continuation before any count",
            vec![types("          S2"), types(nine), types("          S2")],
        ),
        (
            "a first type record with a blank count",
            vec![types("          C1")],
        ),
        ("a list ending short of its count", vec![types(nine)]),
        (
            "a continuation naming more codes than remain",
            vec![types(nine), types("          S2    C5")],
        ),
        (
            "a count record naming more codes than it counts",
            vec![types("     1    C1    P1")],
        ),
        // Review built this: the `X` sits in the next field's padding, so the
        // record names one code for a count of two.
        (
            "a character in a code field's padding",
            vec![types("     2    C1X")],
        ),
    ];
    let v3_version = labeled_header_line(
        "     3.05           OBSERVATION DATA    G",
        "RINEX VERSION / TYPE",
    );
    let sys_types = |text: &str| labeled_header_line(text, "SYS / # / OBS TYPES");
    let v3_cases = [
        (
            "a RINEX 3 continuation before any system",
            vec![sys_types("       C1C"), sys_types("G    1 C1C")],
        ),
        (
            "a RINEX 3 list ending short of its count",
            vec![sys_types("G    2 C1C")],
        ),
        (
            "a RINEX 3 continuation naming more codes than remain",
            vec![sys_types("G    1 C1C"), sys_types("       L1C")],
        ),
        (
            "a character in a RINEX 3 code field's padding",
            vec![sys_types("G    2 C1CX")],
        ),
    ];
    let all = cases
        .into_iter()
        .map(|(what, records)| (what, &version, "1.0", records))
        .chain(
            v3_cases
                .into_iter()
                .map(|(what, records)| (what, &v3_version, "3.0", records)),
        );
    for (what, version_line, crinex_version, records) in all {
        let mut lines = vec![version_line.clone()];
        lines.extend(records);
        lines.push(end.clone());
        let rinex = lines.join("\n") + "\n";
        assert!(
            matches!(encode_crinex(&rinex), Err(Error::Parse(_))),
            "compressing {what}"
        );
        let crinex = [
            labeled_header_line(
                &format!("{crinex_version:<20}COMPACT RINEX FORMAT"),
                "CRINEX VERS   / TYPE",
            ),
            "RNX2CRX".to_string(),
            rinex,
        ]
        .join("\n");
        assert!(
            matches!(decode(&crinex), Err(Error::Parse(_))),
            "expanding {what}"
        );
    }

    // A second complete declaration replaces the first, as `crx2rnx` applies it.
    let replaced = [
        version,
        types(nine),
        types("          S2"),
        types("     1    C1"),
        end,
    ]
    .join("\n")
        + "\n";
    encode_crinex(&replaced).expect("a replacing declaration is read");
}

#[test]
fn clock_offsets_are_written_as_crx2rnx_writes_them() {
    // Review found `0.000123000` expanded where CRX2RNX 4.2.0 writes
    // `.000123000`. Each expectation is CRX2RNX's, except the negative offsets
    // whose last eight digits are zeros, which it misprints (`-1.4` for `-1.5`,
    // `-.0` for `-0.1`); those are written as the value.
    let v2 = [
        (123_000, "  .000123000"),
        (-123_000, " -.000123000"),
        (0, "  .000000000"),
        (1_500_000_000, " 1.500000000"),
        (12_345_678_901, "12.345678901"),
        (-9_999_999_999, "-9.999999999"),
        (-1_500_000_000, "-1.500000000"),
        (-100_000_000, " -.100000000"),
    ];
    for (value, text) in v2 {
        assert_eq!(format_clock(value, 9, 12), text, "{value}");
    }
    let v3 = [
        (123_000_000, "  .000123000000"),
        (-123_000_000, " -.000123000000"),
        (0, "  .000000000000"),
        (1_500_000_000_000, " 1.500000000000"),
        (-1_500_000_000_000, "-1.500000000000"),
    ];
    for (value, text) in v3 {
        assert_eq!(format_clock(value, 12, 15), text, "{value}");
    }
}

#[test]
fn picoseconds_clocks_and_blank_satellite_letters_expand_as_written() {
    // Review found RINEX 4.02 picoseconds refused both ways, clock offsets
    // written with a leading zero CRX2RNX drops, and a mono-system RINEX 2
    // satellite token with a blank letter written with the header's letter.
    // Each `.crx` was compressed from its `.rnx` by RNX2CRX 4.2.0, and CRX2RNX
    // 4.2.0 expands it back to that `.rnx` byte for byte. The version 4 file
    // repeats its picoseconds on two epochs, which RNX2CRX leaves unwritten.
    for name in ["crinex_clock_picoseconds_v4", "crinex_blank_letters_v2"] {
        let rinex = obs_fixture(&format!("{name}.rnx"));
        let expanded = decode(&obs_fixture(&format!("{name}.crx")))
            .unwrap_or_else(|e| panic!("{name}: expand the RNX2CRX file: {e}"));
        assert_same_lines(&expanded, &rinex, name);

        let compressed = encode_crinex(&rinex).unwrap_or_else(|e| panic!("{name}: compress: {e}"));
        let expanded = decode(&compressed).unwrap_or_else(|e| panic!("{name}: expand: {e}"));
        assert_same_lines(&expanded, &rinex, name);
        let stream = parse_stream(&compressed).expect("read the compression");
        assert_eq!(
            parse_stream(&encode_stream(&stream)).expect("read it again"),
            stream,
            "{name}"
        );
    }
}

#[test]
fn unwritten_picoseconds_carry_and_blanked_ones_are_none() {
    // Review found expansion dropping picoseconds RNX2CRX leaves unwritten
    // because they repeat. RNX2CRX writes an epoch with none after one with
    // them the same way, so an unwritten value is the one before, as CRX2RNX
    // reads it.
    let head = "> 2020 01 01 00 00  0.0000000  0  1";
    let crinex = [
        labeled_header_line(
            "3.1                 COMPACT RINEX FORMAT",
            "CRINEX VERS   / TYPE",
        ),
        "RNX2CRX".to_string(),
        labeled_header_line(
            "     4.02           OBSERVATION DATA    G",
            "RINEX VERSION / TYPE",
        ),
        v3_obs_types_line(),
        labeled_header_line("", "END OF HEADER"),
        v3_single_sat_epoch_line(),
        " 12345".to_string(),
        "1&0".to_string(),
        " ".to_string(),
        String::new(),
        "1".to_string(),
        String::new(),
    ]
    .join("\n");
    let expanded = decode(&crinex).expect("expand");
    let epochs: Vec<&str> = expanded
        .lines()
        .filter(|line| line.starts_with('>'))
        .collect();
    let carried = format!("{head:<41}{:15} 12345", "");
    assert_eq!(epochs, [carried.as_str(), carried.as_str()]);

    // An epoch with none after one with them is compressed with its
    // picoseconds blanked, so it expands with none.
    let rinex = [
        labeled_header_line(
            "     4.02           OBSERVATION DATA    G",
            "RINEX VERSION / TYPE",
        ),
        labeled_header_line("G    1 C1C", "SYS / # / OBS TYPES"),
        labeled_header_line("", "END OF HEADER"),
        format!("{head:<41}{:15} 12345", ""),
        format!("G01{:14.3}", 1234.567),
        "> 2020 01 01 00 00  1.0000000  0  1".to_string(),
        format!("G01{:14.3}", 1234.568),
        String::new(),
    ]
    .join("\n");
    let compressed = encode_crinex(&rinex).expect("compress");
    assert!(compressed.contains("\n &&&&&\n"), "{compressed}");
    assert_same_lines(&decode(&compressed).expect("expand"), &rinex, "blanked");
}

#[test]
fn a_version_two_blank_field_carrying_flags_is_refused() {
    // CRINEX 1 holds no flags for a blank field, so compressing them would lose
    // them; `rnx2crx` refuses the same field.
    let rinex = [
        labeled_header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        labeled_header_line("     2    C1    L1", "# / TYPES OF OBSERV"),
        labeled_header_line("", "END OF HEADER"),
        " 20  1  1  0  0  0.0000000  0  1G01".to_string(),
        format!("{:14.3}  {:14}17", 20_000_000.125, ""),
        String::new(),
    ]
    .join("\n");
    let error = encode_crinex(&rinex).expect_err("flags on a blank field are refused");
    assert!(
        matches!(error, Error::Parse(ref msg) if msg.contains("flags but no value")),
        "{error}"
    );
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

fn v2_slip_rinex(names: &[&str], body: &[String]) -> String {
    let mut types = format!("{:6}", names.len());
    for name in names.iter().take(9) {
        types.push_str(&format!("{name:>6}"));
    }
    let mut lines = vec![
        labeled_header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        labeled_header_line(&types, "# / TYPES OF OBSERV"),
        labeled_header_line("", "END OF HEADER"),
    ];
    lines.extend(body.iter().cloned());
    lines.push(String::new());
    lines.join("\n")
}

fn v2_epoch(second: f64, flag: u8, sats: &[&str]) -> Vec<String> {
    let mut lines = Vec::new();
    let mut chunks = sats.chunks(12);
    let first = chunks.next().unwrap_or_default().concat();
    lines.push(format!(
        " 20  1  1  0  0{second:11.7}  {flag}{:3}{first}",
        sats.len()
    ));
    for chunk in chunks {
        lines.push(format!("{:32}{}", "", chunk.concat()));
    }
    lines
}

fn v2_values(values: &[f64]) -> Vec<String> {
    values
        .chunks(5)
        .map(|chunk| {
            chunk
                .iter()
                .map(|value| format!("{value:14.3}  "))
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

#[test]
fn a_version_two_cycle_slip_epoch_crinex_1_cannot_carry_is_refused() {
    // CRINEX 1 copies exactly `numsat` lines after an epoch flagged above 1,
    // as RNX2CRX and CRX2RNX do. A flag 6 epoch's count is satellites, and its
    // records may run past one line each, or its satellite list past twelve.
    let refused = |text: &str, what: &str| {
        let error = encode_crinex(text).expect_err(what);
        assert!(
            error.to_string().contains("CRINEX 1 copies exactly"),
            "{what}: {error}"
        );
    };

    // Six types: each record takes two lines.
    let six = ["C1", "L1", "D1", "S1", "C2", "L2"];
    let mut body = v2_epoch(0.0, 6, &["G01"]);
    body.extend(v2_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    refused(&v2_slip_rinex(&six, &body), "six types");

    // Thirteen satellites: the list continues on a second line.
    let sats: Vec<String> = (1..=13).map(|prn| format!("G{prn:02}")).collect();
    let sats: Vec<&str> = sats.iter().map(String::as_str).collect();
    let mut body = v2_epoch(0.0, 6, &sats);
    for _ in &sats {
        body.extend(v2_values(&[1.0]));
    }
    refused(&v2_slip_rinex(&["L1"], &body), "thirteen satellites");

    // Five types in the header fit, until an event declares six.
    let five = ["C1", "L1", "D1", "S1", "C2"];
    let mut event_types = format!("{:6}", six.len());
    for name in six {
        event_types.push_str(&format!("{name:>6}"));
    }
    let mut body = vec![" 20  1  1  0  0  0.0000000  4  1".to_string()];
    body.push(labeled_header_line(&event_types, "# / TYPES OF OBSERV"));
    body.extend(v2_epoch(30.0, 6, &["G01"]));
    body.extend(v2_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    refused(&v2_slip_rinex(&five, &body), "six types an event declares");

    // Within the limit, the epoch is carried and expands back as written.
    let mut body = v2_epoch(0.0, 6, &["G01", "G02"]);
    body.extend(v2_values(&[1.0, 2.0, 3.0, 4.0, 5.0]));
    body.extend(v2_values(&[-1.0, -2.0]));
    let rinex = v2_slip_rinex(&five, &body);
    let compressed = encode_crinex(&rinex).expect("a slip epoch within the limit compresses");
    let expanded = decode(&compressed).expect("expand");
    assert_same_lines(&expanded, &rinex, "version 2 slips");
}

#[test]
fn a_version_three_cycle_slip_epoch_round_trips_through_crinex_3() {
    // A RINEX 3 flag 6 epoch has one record line per satellite, so the
    // `numsat` lines CRINEX 3 copies after it are its records.
    let rinex = [
        labeled_header_line(
            "     3.05           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        labeled_header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
        labeled_header_line("", "END OF HEADER"),
        "> 2020 01 01 00 00  0.0000000  0  1".to_string(),
        "G01  20000000.000 7    100000.125 7".to_string(),
        "> 2020 01 01 00 00 30.0000000  6  2".to_string(),
        "G01                         1.000".to_string(),
        "G02                        -2.000".to_string(),
        "> 2020 01 01 00 01  0.0000000  0  1".to_string(),
        "G01  20000060.000 7    100060.250 7".to_string(),
        String::new(),
    ]
    .join("\n");
    let compressed = encode_crinex(&rinex).expect("compress");
    let expanded = decode(&compressed).expect("expand");
    assert_same_lines(&expanded, &rinex, "version 3 slips");
    let obs = crate::rinex_obs::RinexObs::parse(&expanded).expect("parse the expansion");
    assert_eq!(obs.epochs()[1].cycle_slips.len(), 2);
}

#[test]
fn a_crinex_1_cycle_slip_epoch_within_one_line_expands() {
    // RNX2CRX writes an event epoch's line with `&` in its first column and
    // copies the `numsat` lines after it; CRX2RNX reads them back as written.
    let crinex = [
        labeled_header_line(
            "1.0                 COMPACT RINEX FORMAT",
            "CRINEX VERS   / TYPE",
        ),
        "RNX2CRX".to_string(),
        labeled_header_line(
            "     2.11           OBSERVATION DATA    G (GPS)",
            "RINEX VERSION / TYPE",
        ),
        labeled_header_line("     2    C1    L1", "# / TYPES OF OBSERV"),
        labeled_header_line("", "END OF HEADER"),
        "&20  1  1  0  0 30.0000000  6  2G01G02".to_string(),
        "                         1.000".to_string(),
        "                        -2.000".to_string(),
        String::new(),
    ]
    .join("\n");
    let expanded = decode(&crinex).expect("expand a CRINEX 1 slip epoch");
    let lines: Vec<&str> = expanded.lines().collect();
    let at = lines
        .iter()
        .position(|line| *line == " 20  1  1  0  0 30.0000000  6  2G01G02")
        .unwrap_or_else(|| panic!("the slip epoch line: {expanded}"));
    assert_eq!(
        lines[at + 1..],
        [
            "                         1.000",
            "                        -2.000"
        ]
    );
    let obs = crate::rinex_obs::RinexObs::parse(&expanded).expect("parse the expansion");
    let slips = &obs.epochs()[0].cycle_slips;
    let values = |prn: u8| -> Vec<Option<f64>> {
        let sat =
            crate::id::GnssSatelliteId::new(crate::id::GnssSystem::Gps, prn).expect("satellite");
        slips[&sat].iter().map(|value| value.value).collect()
    };
    assert_eq!(values(1), vec![None, Some(1.0)]);
    assert_eq!(values(2), vec![None, Some(-2.0)]);
}

#[test]
fn an_event_epoch_line_is_compressed_and_expanded_whole() {
    // RNX2CRX copies an event's epoch line whole as a descriptor reset, and
    // CRX2RNX prints it back with trailing blanks removed. Expansion wrote an
    // event's line only to its count field, so a timed event lost its clock
    // offset and its RINEX 4.02 picoseconds. Each `.crx` was compressed by
    // RNX2CRX 4.2.0 and each `.rnx` is CRX2RNX 4.2.0's expansion of it: timed
    // flag 5 events with a clock offset at 2.11, 3.05 and 4.02, and at 4.02 one
    // with picoseconds alone and one with both. An event's picoseconds do not
    // carry into the next epoch, which CRX2RNX gives the earlier observation
    // epoch's.
    for version in ["v2", "v3", "v4"] {
        let rinex = obs_fixture(&format!("crinex_event_clocks_{version}.rnx"));
        let expanded = decode(&obs_fixture(&format!("crinex_event_clocks_{version}.crx")))
            .unwrap_or_else(|e| panic!("{version}: expand the RNX2CRX file: {e}"));
        assert_same_lines(&expanded, &rinex, version);
        let compressed =
            encode_crinex(&rinex).unwrap_or_else(|e| panic!("{version}: compress: {e}"));
        let expanded = decode(&compressed).unwrap_or_else(|e| panic!("{version}: expand: {e}"));
        assert_same_lines(&expanded, &rinex, version);
        assert!(
            rinex.lines().any(|line| line.contains("0.123456789")),
            "{version}: the fixture holds a timed event with a clock offset"
        );
    }
}
