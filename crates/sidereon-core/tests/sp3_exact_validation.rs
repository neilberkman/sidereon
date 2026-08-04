use serde::Deserialize;
use sidereon_core::data::{
    mgex_nav, mgex_sp3, ops_ultra_sp3, AnalysisCenter, ProductDate, ProductType,
};
use sidereon_core::ephemeris::{
    parse_exact_sp3, validate_exact_sp3, ExactSp3Coverage, ExactSp3Request,
    ExactSp3ValidationError, Sp3,
};

const START: ProductDate = ProductDate {
    year: 2020,
    month: 1,
    day: 1,
};
const SP3_RECORD_PADDING_MAX: usize = 77;
const P_G01: &str = "PG01  15000.000000 -20000.000000   5000.000000    123.456789\n";
const P_G02: &str = "PG02  16000.000000 -21000.000000   6000.000000    124.456789\n";
const V_G01: &str = "VG01      1.000000      2.000000      3.000000      4.000000\n";
const V_G02: &str = "VG02      5.000000      6.000000      7.000000      8.000000\n";

#[derive(Debug, Deserialize)]
struct TerminalRecordCorpus {
    schema: String,
    record_width: usize,
    record_width_authority: String,
    cases: Vec<TerminalRecordCase>,
}

#[derive(Debug, Deserialize)]
struct TerminalRecordCase {
    name: String,
    leading_hex: String,
    marker: Option<String>,
    padding_spaces: usize,
    suffix_hex: String,
    separator_hex: String,
    trailing_hex: String,
    expect: String,
}

fn request(sample: &str) -> Result<ExactSp3Request, ExactSp3ValidationError> {
    ExactSp3Request::new(START, Some("0000"), "01D", sample)
}

fn regular_offsets(count: usize, cadence_s: i64) -> Vec<i64> {
    (0..count).map(|index| index as i64 * cadence_s).collect()
}

fn spaces(count: usize) -> String {
    " ".repeat(count)
}

fn with_terminal(base: &str, terminal: &str) -> String {
    format!("{}{terminal}", base.strip_suffix("EOF\n").unwrap())
}

fn decode_hex(value: &str) -> Vec<u8> {
    assert!(value.len().is_multiple_of(2), "odd-length hex {value:?}");
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).expect("fixture hex is ASCII");
            u8::from_str_radix(text, 16).expect("fixture hex is valid")
        })
        .collect()
}

fn terminal_case_bytes(base: &str, case: &TerminalRecordCase) -> Vec<u8> {
    let mut bytes = base
        .strip_suffix("EOF\n")
        .expect("fixture has canonical terminal record")
        .as_bytes()
        .to_vec();
    bytes.extend(decode_hex(&case.leading_hex));
    if let Some(marker) = &case.marker {
        bytes.extend(marker.as_bytes());
    }
    bytes.extend(std::iter::repeat_n(b' ', case.padding_spaces));
    bytes.extend(decode_hex(&case.suffix_hex));
    bytes.extend(decode_hex(&case.separator_hex));
    bytes.extend(decode_hex(&case.trailing_hex));
    bytes
}

fn terminal_result_class(
    result: Result<(Sp3, ExactSp3Coverage), ExactSp3ValidationError>,
) -> &'static str {
    match result {
        Ok(_) => "accept",
        Err(ExactSp3ValidationError::MalformedEofRecord { .. }) => "malformed_eof_record",
        Err(ExactSp3ValidationError::MissingEof) => "missing_eof",
        Err(ExactSp3ValidationError::TrailingContentAfterEof) => "trailing_content_after_eof",
        Err(error) => panic!("terminal corpus reached unrelated exact error: {error:?}"),
    }
}

fn option_bits(value: Option<f64>) -> Option<u64> {
    value.map(f64::to_bits)
}

fn assert_products_bit_identical(left: &Sp3, right: &Sp3) {
    assert_eq!(left.header, right.header);
    assert_eq!(left.epochs, right.epochs);
    assert_eq!(left.declared_epoch_count(), right.declared_epoch_count());
    assert_eq!(
        left.declared_start_j2000_s().map(f64::to_bits),
        right.declared_start_j2000_s().map(f64::to_bits)
    );

    for epoch_index in 0..left.epoch_count() {
        let left_states = left.states_at(epoch_index).expect("left epoch");
        let right_states = right.states_at(epoch_index).expect("right epoch");
        assert_eq!(
            left_states.keys().collect::<Vec<_>>(),
            right_states.keys().collect::<Vec<_>>()
        );
        for (satellite, left_state) in left_states {
            let right_state = right_states.get(satellite).expect("matching satellite");
            assert_eq!(
                left_state.position.as_array().map(f64::to_bits),
                right_state.position.as_array().map(f64::to_bits)
            );
            assert_eq!(
                option_bits(left_state.clock_s),
                option_bits(right_state.clock_s)
            );
            assert_eq!(
                left_state.velocity.map(|velocity| {
                    [
                        velocity.vx_m_s.to_bits(),
                        velocity.vy_m_s.to_bits(),
                        velocity.vz_m_s.to_bits(),
                    ]
                }),
                right_state.velocity.map(|velocity| {
                    [
                        velocity.vx_m_s.to_bits(),
                        velocity.vy_m_s.to_bits(),
                        velocity.vz_m_s.to_bits(),
                    ]
                })
            );
            assert_eq!(
                option_bits(left_state.clock_rate_s_s),
                option_bits(right_state.clock_rate_s_s)
            );
            assert_eq!(left_state.flags, right_state.flags);
        }
    }
}

fn remove_first_line_with_prefix(text: &str, prefix: &str) -> String {
    let mut removed = false;
    text.split_inclusive('\n')
        .filter(|line| {
            if !removed && line.starts_with(prefix) {
                removed = true;
                false
            } else {
                true
            }
        })
        .collect()
}

fn missing_position_record(satellite: &str) -> String {
    format!(
        "P{satellite}{:14.6}{:14.6}{:14.6}{:14.6}\n",
        0.0, 0.0, 0.0, 999_999.999_999
    )
}

fn exact_sp3(
    offsets_s: &[i64],
    declared_count: usize,
    header_cadence: &str,
    declared_day: u8,
) -> String {
    exact_sp3_at(
        offsets_s,
        declared_count,
        header_cadence,
        2020,
        1,
        declared_day,
        2086,
        259_200.0,
        58_849,
        "TST",
    )
}

#[allow(clippy::too_many_arguments)]
fn exact_sp3_at(
    offsets_s: &[i64],
    declared_count: usize,
    header_cadence: &str,
    year: i32,
    month: u8,
    declared_day: u8,
    gnss_week: u16,
    seconds_of_week: f64,
    mjd: i64,
    agency: &str,
) -> String {
    let dt = format!(
        "{:4} {:>2} {:>2} {:>2} {:>2} {:11.8}",
        year, month, declared_day, 0, 0, 0.0
    );
    let mut text = format!(
        "#dP{dt} {declared_count:>7} {:<5}{:>6}{:>4} {}\n",
        "ORBIT", "IGS20", "FIT", agency
    );
    text.push_str(&format!(
        "## {:>4} {:15.8} {header_cadence:>14} {:>5} {:.13}\n",
        gnss_week, seconds_of_week, mjd, 0.0
    ));
    text.push_str("+    2   G01G02");
    for _ in 2..17 {
        text.push_str("  0");
    }
    text.push('\n');
    for _ in 1..5 {
        text.push_str("+        ");
        for _ in 0..17 {
            text.push_str("  0");
        }
        text.push('\n');
    }
    for _ in 0..5 {
        text.push_str("++       ");
        for _ in 0..17 {
            text.push_str("  0");
        }
        text.push('\n');
    }
    text.push_str("%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
    text.push_str("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
    text.push_str("%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n");
    text.push_str("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n");
    text.push_str("%i    0    0    0    0      0      0      0      0         0\n");
    text.push_str("%i    0    0    0    0      0      0      0      0         0\n");
    for _ in 0..4 {
        text.push_str("/* EXACT VALIDATION TEST FIXTURE\n");
    }

    for &offset_s in offsets_s {
        let day_offset = offset_s.div_euclid(86_400);
        let second_of_day = offset_s.rem_euclid(86_400);
        let hour = second_of_day / 3_600;
        let minute = (second_of_day % 3_600) / 60;
        let second = second_of_day % 60;
        text.push_str(&format!(
            "*  {:4} {:>2} {:>2} {:>2} {:>2} {:11.8}\n",
            year,
            month,
            i64::from(declared_day) + day_offset,
            hour,
            minute,
            second as f64
        ));
        text.push_str(P_G01);
        text.push_str(P_G02);
    }
    text.push_str("EOF\n");
    text
}

#[test]
fn accepts_regular_24_hour_five_minute_half_open_grid() {
    let text = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    let (product, coverage) = parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap();

    assert_eq!(coverage, ExactSp3Coverage::HalfOpen);
    assert_eq!(product.epoch_count(), 288);
    assert_eq!(product.declared_epoch_count(), 288);
}

#[test]
fn accepts_regular_24_hour_five_minute_inclusive_grid() {
    let text = exact_sp3(&regular_offsets(289, 300), 289, "300.00000000", 1);
    let (_, coverage) = parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap();

    assert_eq!(coverage, ExactSp3Coverage::Inclusive);
}

#[test]
fn rejects_shorter_and_longer_regular_grids() {
    for (count, expected_half_open, expected_inclusive) in [(287, 288, 289), (290, 288, 289)] {
        let text = exact_sp3(&regular_offsets(count, 300), count, "300.00000000", 1);
        let error = parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap_err();
        assert_eq!(
            error,
            ExactSp3ValidationError::SpanMismatch {
                parsed: count,
                half_open: expected_half_open,
                inclusive: expected_inclusive,
            }
        );
    }
}

#[test]
fn rejects_irregular_or_nonascending_epoch_grid() {
    let mut irregular = regular_offsets(288, 300);
    irregular[100] += 1;
    let text = exact_sp3(&irregular, 288, "300.00000000", 1);
    let error = parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap_err();
    assert_eq!(
        error,
        ExactSp3ValidationError::IrregularEpochGrid {
            epoch_index: 100,
            requested_s: 300.0,
            actual_s: 301.0,
        }
    );

    let mut nonascending = regular_offsets(288, 300);
    nonascending[100] = nonascending[99];
    let text = exact_sp3(&nonascending, 288, "300.00000000", 1);
    assert!(matches!(
        parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()),
        Err(ExactSp3ValidationError::IrregularEpochGrid {
            epoch_index: 100,
            actual_s: 0.0,
            ..
        })
    ));
}

#[test]
fn rejects_zero_nonfinite_out_of_range_and_mismatched_header_cadence() {
    let offsets = regular_offsets(288, 300);
    let cases = [
        (
            "0.00000000",
            ExactSp3ValidationError::NonPositiveHeaderCadence { actual_s: 0.0 },
        ),
        (
            "-300.0000000",
            ExactSp3ValidationError::NonPositiveHeaderCadence { actual_s: -300.0 },
        ),
        ("NaN", ExactSp3ValidationError::NonFiniteHeaderCadence),
        ("inf", ExactSp3ValidationError::NonFiniteHeaderCadence),
        (
            "900.00000000",
            ExactSp3ValidationError::CadenceMismatch {
                requested_s: 300.0,
                header_s: 900.0,
            },
        ),
    ];

    for (header_cadence, expected) in cases {
        let text = exact_sp3(&offsets, 288, header_cadence, 1);
        assert_eq!(
            parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap_err(),
            expected
        );
    }

    // The format bound and the field width coincide: the interval is an `F14.8`
    // field, so the first unsupported cadence is also the first one the writer
    // cannot re-emit inside its 14 columns. `Sp3::parse` rejects it there, so
    // this file never reaches cadence validation. `UnsupportedHeaderCadence`
    // still covers a caller-requested cadence and a merged product.
    let text = exact_sp3(&offsets, 288, "100000.000000", 1);
    let err = parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap_err();
    assert!(
        matches!(err, ExactSp3ValidationError::Parse(ref parse)
            if parse.to_string().contains("epoch interval")
                && parse.to_string().contains("F14.8 field")),
        "{err}"
    );
}

#[test]
fn rejects_zero_unknown_and_unsupported_sample_tokens() {
    for sample in ["00M", "00U", "05X", "1M", "01W", "05m", "99Q"] {
        assert_eq!(
            request(sample).unwrap_err(),
            ExactSp3ValidationError::UnsupportedSampleToken {
                token: sample.to_owned(),
            }
        );
    }
}

#[test]
fn request_from_identity_uses_exact_igs_final_fields_and_rejects_nav() {
    let current = ProductDate::new(2022, 11, 27).unwrap();
    let identity = mgex_sp3(AnalysisCenter::Igs, current, None)
        .unwrap()
        .identity()
        .unwrap();
    let request = ExactSp3Request::from_identity(&identity).unwrap();

    assert_eq!(request.date(), current);
    assert_eq!(request.issue(), Some("0000"));
    assert_eq!(request.span(), "01D");
    assert_eq!(request.sample(), "15M");
    assert_eq!(request.expected_agency(), Some("IGS"));

    let nav_identity = mgex_nav(AnalysisCenter::Igs, current, None)
        .unwrap()
        .identity()
        .unwrap();
    assert_eq!(
        ExactSp3Request::from_identity(&nav_identity).unwrap_err(),
        ExactSp3ValidationError::WrongProductFamily {
            actual: ProductType::Nav,
        }
    );
}

#[test]
fn historical_gfz_ultra_identity_requires_its_cataloged_content_start() {
    // This issue crosses a GPS-week boundary: the filename epoch is Sunday in
    // week 2226, while the required first content epoch is Saturday in 2225.
    let filename_date = ProductDate::new(2022, 9, 4).expect("filename date");
    let identity = ops_ultra_sp3(
        AnalysisCenter::GfzUlt,
        filename_date,
        Some("05M"),
        Some("0000"),
    )
    .expect("historical GFZ ultra product")
    .identity()
    .expect("historical GFZ identity");
    let from_identity = ExactSp3Request::from_identity(&identity).expect("exact request");

    // The official filename names 2022-09-04 00:00, while this historical
    // series begins at 2022-09-03 00:00 (GPS week 2225, SOW 518400, MJD 59825).
    let text = exact_sp3_at(
        &regular_offsets(576, 300),
        576,
        "300.00000000",
        2022,
        9,
        3,
        2225,
        518_400.0,
        59_825,
        "GFZ",
    );

    assert_eq!(
        parse_exact_sp3(text.as_bytes(), &from_identity)
            .expect("catalog-derived historical request")
            .1,
        ExactSp3Coverage::HalfOpen
    );

    let filename_epoch_request =
        ExactSp3Request::new(filename_date, Some("0000"), "02D", "05M").expect("same-date request");
    assert!(matches!(
        parse_exact_sp3(text.as_bytes(), &filename_epoch_request),
        Err(ExactSp3ValidationError::DeclaredStartMismatch { .. })
    ));
}

#[test]
fn expected_agency_is_optional_but_terminal_when_requested() {
    let text = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    let required = request("05M").unwrap().with_expected_agency("IGS").unwrap();
    assert_eq!(required.expected_agency(), Some("IGS"));
    assert_eq!(
        parse_exact_sp3(text.as_bytes(), &required).unwrap_err(),
        ExactSp3ValidationError::AgencyMismatch {
            expected: "IGS".to_owned(),
            actual: "TST".to_owned(),
        }
    );

    let matching = text.replacen(" FIT TST\n", " FIT IGS\n", 1);
    assert!(parse_exact_sp3(matching.as_bytes(), &required).is_ok());
    assert!(matches!(
        request("05M").unwrap().with_expected_agency("igs"),
        Err(ExactSp3ValidationError::InvalidExpectedAgency { .. })
    ));
}

#[test]
fn rejects_noncanonical_fixed_duration_tokens() {
    for (sample, canonical) in [("60S", "01M"), ("60M", "01H"), ("24H", "01D")] {
        assert_eq!(
            request(sample).unwrap_err(),
            ExactSp3ValidationError::NonCanonicalSampleToken {
                token: sample.to_owned(),
                canonical: canonical.to_owned(),
            }
        );
    }
    assert!(ExactSp3Request::new(START, None, "07D", "05M").is_ok());
    for sample in ["30S", "05M", "01D"] {
        assert!(request(sample).is_ok());
    }
}

#[test]
fn shared_terminal_record_corpus_matches_exact_parser_contract() {
    let corpus: TerminalRecordCorpus =
        serde_json::from_str(include_str!("../golden/sp3-terminal-record-v1.json"))
            .expect("terminal-record corpus is valid JSON");
    assert_eq!(corpus.schema, "sidereon-sp3-terminal-record-v1");
    assert_eq!(corpus.record_width, 80);
    assert_eq!(
        corpus.record_width_authority,
        "sidereon-interoperability-policy"
    );

    let base = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    let exact_request = request("05M").unwrap();
    for case in &corpus.cases {
        let bytes = terminal_case_bytes(&base, case);
        assert!(
            Sp3::parse(&bytes).is_ok(),
            "the general parser must remain permissive for corpus case {}",
            case.name
        );
        assert_eq!(
            terminal_result_class(parse_exact_sp3(&bytes, &exact_request)),
            case.expect,
            "terminal-record corpus case {}",
            case.name
        );
    }
}

#[test]
fn eof_padding_does_not_change_any_parsed_numeric_value() {
    let bare = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    let padded = with_terminal(&bare, &format!("EOF{}\r\n", spaces(77)));
    let exact_request = request("05M").unwrap();

    let (bare_product, bare_coverage) =
        parse_exact_sp3(bare.as_bytes(), &exact_request).expect("bare product");
    let (padded_product, padded_coverage) =
        parse_exact_sp3(padded.as_bytes(), &exact_request).expect("padded product");

    assert_eq!(bare_coverage, padded_coverage);
    assert_products_bit_identical(&bare_product, &padded_product);
}

#[test]
fn accepts_supported_eof_padding_and_line_endings() {
    let base = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    let terminals = [
        "EOF\n".to_owned(),
        "EOF\r\n".to_owned(),
        "EOF".to_owned(),
        format!("EOF{}\n", spaces(1)),
        format!("EOF{}\n", spaces(37)),
        format!("EOF{}\n", spaces(SP3_RECORD_PADDING_MAX)),
        format!("EOF{}\r\n", spaces(SP3_RECORD_PADDING_MAX)),
        format!("EOF{}", spaces(SP3_RECORD_PADDING_MAX)),
    ];

    for terminal in terminals {
        let text = with_terminal(&base, &terminal);
        let (_, coverage) = parse_exact_sp3(text.as_bytes(), &request("05M").unwrap())
            .unwrap_or_else(|error| panic!("terminal {terminal:?} must be accepted: {error:?}"));
        assert_eq!(coverage, ExactSp3Coverage::HalfOpen);
    }
}

#[test]
fn malformed_eof_like_records_have_a_distinct_integrity_error() {
    let base = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    let terminals = [
        format!("EOF{}\n", spaces(SP3_RECORD_PADDING_MAX + 1)),
        "EOF\t\n".to_owned(),
        "EOF \t \n".to_owned(),
        "EOFX\n".to_owned(),
        "EOF X\n".to_owned(),
        "EOF.\n".to_owned(),
        " EOF\n".to_owned(),
        "\tEOF\n".to_owned(),
        "EOF\r".to_owned(),
        "EOF\r\r\n".to_owned(),
    ];

    for terminal in terminals {
        let text = with_terminal(&base, &terminal);
        match parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap_err() {
            ExactSp3ValidationError::MalformedEofRecord {
                line_number,
                record_length,
            } => {
                assert!(line_number > 0);
                assert!(record_length >= 3);
            }
            error => panic!("terminal {terminal:?} returned the wrong error: {error:?}"),
        }
    }
}

#[test]
fn trailing_ascii_blank_records_are_tolerated_but_other_data_is_not() {
    let base = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    for tail in ["\n", "\n\n\n", "   \n", "\n   \n"] {
        let text = format!("{base}{tail}");
        assert!(
            parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).is_ok(),
            "ASCII-blank tail {tail:?} must be tolerated"
        );
    }

    for tail in ["\t\n", " \t \n", "EOFX\n", "non-whitespace\n"] {
        let text = format!("{base}{tail}");
        assert_eq!(
            parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap_err(),
            ExactSp3ValidationError::TrailingContentAfterEof,
            "tail {tail:?} must be rejected as trailing content"
        );
    }
}

#[test]
fn padded_eof_preserves_trailing_content_detection() {
    let base = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    for terminal in ["EOF\n".to_owned(), format!("EOF{}\n", spaces(77))] {
        let text = format!(
            "{}*  2020  1  2  0  0  0.00000000\n{P_G01}{P_G02}",
            with_terminal(&base, &terminal)
        );
        assert_eq!(
            parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap_err(),
            ExactSp3ValidationError::TrailingContentAfterEof
        );
    }
}

#[test]
fn premature_padded_eof_is_rejected_by_the_exact_epoch_count() {
    let base = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    let body = base.strip_suffix("EOF\n").unwrap();
    let cut = body
        .find("*  2020  1  1 12  0")
        .expect("midday epoch record");

    for terminal in ["EOF\n".to_owned(), format!("EOF{}\n", spaces(77))] {
        let text = format!("{}{terminal}", &body[..cut]);
        assert_eq!(
            parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap_err(),
            ExactSp3ValidationError::DeclaredEpochCountMismatch {
                declared: 288,
                parsed: 144,
            }
        );
    }
}

#[test]
fn eof_text_embedded_in_another_record_is_not_a_terminal_record() {
    let base = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    let valid = base.replacen(
        "/* EXACT VALIDATION TEST FIXTURE\n",
        "/* EOF APPEARS IN THIS COMMENT TEXT\n",
        1,
    );
    assert!(parse_exact_sp3(valid.as_bytes(), &request("05M").unwrap()).is_ok());

    let missing_terminal = with_terminal(&valid, "");
    assert_eq!(
        parse_exact_sp3(missing_terminal.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::MissingEof
    );
}

#[test]
fn rejects_missing_mandatory_structure_and_accepts_missing_position_sentinels() {
    let base = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    let no_eof = base.strip_suffix("EOF\n").unwrap();
    assert_eq!(
        parse_exact_sp3(no_eof.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::MissingEof
    );

    let trailing = format!("{base}*  2020  1  2  0  0  0.00000000\n{P_G01}{P_G02}");
    assert_eq!(
        parse_exact_sp3(trailing.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::TrailingContentAfterEof
    );

    let missing_accuracy = remove_first_line_with_prefix(&base, "++");
    assert_eq!(
        parse_exact_sp3(missing_accuracy.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::MandatoryHeaderRecordCount {
            record: "++",
            expected: 5,
            actual: 4,
        }
    );

    let missing_float = remove_first_line_with_prefix(&base, "%f");
    assert_eq!(
        parse_exact_sp3(missing_float.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::MandatoryHeaderRecordCount {
            record: "%f",
            expected: 2,
            actual: 1,
        }
    );

    let missing_comment = remove_first_line_with_prefix(&base, "/*").replacen(
        "EOF\n",
        "/* BODY COMMENT DOES NOT COUNT AS HEADER\nEOF\n",
        1,
    );
    assert_eq!(
        parse_exact_sp3(missing_comment.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::MandatoryHeaderRecordCount {
            record: "/*",
            expected: 4,
            actual: 3,
        }
    );

    let no_satellites = base
        .replacen("+    2", "+    0", 1)
        .replacen("G01G02", "  0  0", 1)
        .lines()
        .filter(|line| !line.starts_with('P'))
        .map(|line| format!("{line}\n"))
        .collect::<String>();
    assert_eq!(
        parse_exact_sp3(no_satellites.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::NoDeclaredSatellites
    );

    let empty_epochs = base
        .replace(P_G01, &missing_position_record("G01"))
        .replace(P_G02, &missing_position_record("G02"));
    assert_eq!(
        parse_exact_sp3(empty_epochs.as_bytes(), &request("05M").unwrap())
            .unwrap()
            .1,
        ExactSp3Coverage::HalfOpen
    );
}

#[test]
fn rejects_raw_satellite_count_and_position_record_sequence_defects() {
    let base = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);

    let wrong_count = base.replacen("+    2", "+    1", 1);
    assert_eq!(
        parse_exact_sp3(wrong_count.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::DeclaredSatelliteCountMismatch {
            declared: 1,
            tokens: 2,
        }
    );

    let omitted = remove_first_line_with_prefix(&base, "PG02");
    assert_eq!(
        parse_exact_sp3(omitted.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::SatelliteRecordSequenceMismatch {
            record: "P",
            epoch_index: 0,
            expected: vec!["G01".to_owned(), "G02".to_owned()],
            actual: vec!["G01".to_owned()],
        }
    );

    let reordered = base.replacen(&format!("{P_G01}{P_G02}"), &format!("{P_G02}{P_G01}"), 1);
    assert_eq!(
        parse_exact_sp3(reordered.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::SatelliteRecordSequenceMismatch {
            record: "P",
            epoch_index: 0,
            expected: vec!["G01".to_owned(), "G02".to_owned()],
            actual: vec!["G02".to_owned(), "G01".to_owned()],
        }
    );

    let duplicate = base.replacen(P_G02, P_G01, 1);
    assert_eq!(
        parse_exact_sp3(duplicate.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::SatelliteRecordSequenceMismatch {
            record: "P",
            epoch_index: 0,
            expected: vec!["G01".to_owned(), "G02".to_owned()],
            actual: vec!["G01".to_owned(), "G01".to_owned()],
        }
    );
}

#[test]
fn velocity_products_require_matching_velocity_records() {
    let position_only = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    let declared_velocity = position_only.replacen("#dP", "#dV", 1);

    assert_eq!(
        parse_exact_sp3(declared_velocity.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::SatelliteRecordSequenceMismatch {
            record: "V",
            epoch_index: 0,
            expected: vec!["G01".to_owned(), "G02".to_owned()],
            actual: vec![],
        }
    );

    let paired = declared_velocity
        .replace(P_G01, &format!("{P_G01}{V_G01}"))
        .replace(P_G02, &format!("{P_G02}{V_G02}"));
    assert!(parse_exact_sp3(paired.as_bytes(), &request("05M").unwrap()).is_ok());

    let first_paired = format!("{P_G01}{V_G01}{P_G02}{V_G02}");
    let grouped = paired.replacen(&first_paired, &format!("{P_G01}{P_G02}{V_G01}{V_G02}"), 1);
    assert_eq!(
        parse_exact_sp3(grouped.as_bytes(), &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::BodyRecordInterleavingMismatch {
            epoch_index: 0,
            expected: vec![
                "PG01".to_owned(),
                "VG01".to_owned(),
                "PG02".to_owned(),
                "VG02".to_owned(),
            ],
            actual: vec![
                "PG01".to_owned(),
                "PG02".to_owned(),
                "VG01".to_owned(),
                "VG02".to_owned(),
            ],
        }
    );

    let velocity_before_position =
        paired.replacen(&first_paired, &format!("{V_G01}{P_G01}{P_G02}{V_G02}"), 1);
    assert_eq!(
        parse_exact_sp3(
            velocity_before_position.as_bytes(),
            &request("05M").unwrap()
        )
        .unwrap_err(),
        ExactSp3ValidationError::BodyRecordInterleavingMismatch {
            epoch_index: 0,
            expected: vec![
                "PG01".to_owned(),
                "VG01".to_owned(),
                "PG02".to_owned(),
                "VG02".to_owned(),
            ],
            actual: vec![
                "VG01".to_owned(),
                "PG01".to_owned(),
                "PG02".to_owned(),
                "VG02".to_owned(),
            ],
        }
    );
}

#[test]
fn rejects_declared_count_mismatch_without_tightening_base_parser() {
    let text = exact_sp3(&regular_offsets(288, 300), 287, "300.00000000", 1);
    let product = Sp3::parse(text.as_bytes()).expect("base parser remains permissive");

    assert_eq!(product.epoch_count(), 288);
    assert_eq!(product.declared_epoch_count(), 287);
    assert_eq!(
        validate_exact_sp3(&product, &request("05M").unwrap()).unwrap_err(),
        ExactSp3ValidationError::DeclaredEpochCountMismatch {
            declared: 287,
            parsed: 288,
        }
    );
}

#[test]
fn rejects_declared_or_parsed_start_mismatch() {
    let offsets = regular_offsets(288, 300);
    let wrong_declared = exact_sp3(&offsets, 288, "300.00000000", 2);
    assert!(matches!(
        parse_exact_sp3(wrong_declared.as_bytes(), &request("05M").unwrap()),
        Err(ExactSp3ValidationError::DeclaredStartMismatch { .. })
    ));

    let parsed_late = regular_offsets(288, 300)
        .into_iter()
        .map(|seconds| seconds + 300)
        .collect::<Vec<_>>();
    let text = exact_sp3(&parsed_late, 288, "300.00000000", 1);
    assert!(matches!(
        parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()),
        Err(ExactSp3ValidationError::FirstEpochMismatch { .. })
    ));
}

#[test]
fn rejects_inconsistent_line_two_week_sow_and_mjd_start_metadata() {
    let base = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1);
    let cases = [
        (
            base.replacen("## 2086", "## 2085", 1),
            ExactSp3ValidationError::HeaderStartMetadataMismatch {
                field: "gps_week",
                requested: 2086.0,
                actual: 2085.0,
            },
        ),
        (
            base.replacen("259200.00000000", "259201.00000000", 1),
            ExactSp3ValidationError::HeaderStartMetadataMismatch {
                field: "seconds_of_week",
                requested: 259_200.0,
                actual: 259_201.0,
            },
        ),
        (
            base.replacen("259200.00000000", "            NaN", 1),
            ExactSp3ValidationError::NonFiniteHeaderStartMetadata {
                field: "seconds_of_week",
            },
        ),
        (
            base.replacen("259200.00000000", "    -1.00000000", 1),
            ExactSp3ValidationError::InvalidHeaderStartMetadata {
                field: "seconds_of_week",
                actual: -1.0,
            },
        ),
        (
            base.replacen("259200.00000000", "604800.00000000", 1),
            ExactSp3ValidationError::InvalidHeaderStartMetadata {
                field: "seconds_of_week",
                actual: 604_800.0,
            },
        ),
        (
            base.replacen(" 58849 ", " 58848 ", 1),
            ExactSp3ValidationError::HeaderStartMetadataMismatch {
                field: "mjd",
                requested: 58_849.0,
                actual: 58_848.0,
            },
        ),
        (
            base.replacen("0.0000000000000", "0.5000000000000", 1),
            ExactSp3ValidationError::HeaderStartMetadataMismatch {
                field: "mjd",
                requested: 58_849.0,
                actual: 58_849.5,
            },
        ),
        (
            base.replacen("0.0000000000000", "1.0000000000000", 1),
            ExactSp3ValidationError::InvalidHeaderStartMetadata {
                field: "mjd_fraction",
                actual: 1.0,
            },
        ),
    ];

    for (text, expected) in cases {
        assert_eq!(
            parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap_err(),
            expected
        );
    }
}

#[test]
fn line_two_start_metadata_uses_the_declared_file_time_system_coordinate() {
    let text = exact_sp3(&regular_offsets(288, 300), 288, "300.00000000", 1).replacen(
        "%c M  cc GPS",
        "%c M  cc UTC",
        1,
    );

    // SP3-d says every time field uses the file's declared time system. The
    // line-1 civil date, line-2 week/MJD, and epoch records therefore agree
    // directly; inserting a GPS-versus-UTC leap offset here would be wrong.
    let (_, coverage) = parse_exact_sp3(text.as_bytes(), &request("05M").unwrap()).unwrap();
    assert_eq!(coverage, ExactSp3Coverage::HalfOpen);
}

#[test]
fn rejects_span_not_divisible_by_sample() {
    let request = ExactSp3Request::new(START, None, "01H", "07M").unwrap();
    let text = exact_sp3(&regular_offsets(9, 420), 9, "420.00000000", 1);

    assert_eq!(
        parse_exact_sp3(text.as_bytes(), &request).unwrap_err(),
        ExactSp3ValidationError::SpanNotMultipleOfCadence {
            span_s: 3_600,
            cadence_s: 420,
        }
    );
}

/// The Wuhan MGEX near-real-time line validates exactly like the other ultra
/// lines: a catalog-derived request pins the `WHU` agency, the half-open
/// two-day five-minute grid, and the filename-epoch content start (all
/// verified against the live product on 2026-08-04).
#[test]
fn wum_nrt_identity_validates_an_archive_shaped_product() {
    let identity = ops_ultra_sp3(
        AnalysisCenter::WumNrt,
        ProductDate::new(2026, 8, 3).expect("date"),
        None,
        Some("0000"),
    )
    .expect("WUM NRT product")
    .identity()
    .expect("identity");
    let request = ExactSp3Request::from_identity(&identity).expect("exact request");
    assert_eq!(request.expected_agency(), Some("WHU"));
    assert_eq!(request.span(), "02D");
    assert_eq!(request.sample(), "05M");

    // 2026-08-03 00:00 GPST: GPS week 2430 day 1, MJD 61255.
    let text = exact_sp3_at(
        &regular_offsets(576, 300),
        576,
        "300.00000000",
        2026,
        8,
        3,
        2430,
        86_400.0,
        61_255,
        "WHU",
    );
    assert_eq!(
        parse_exact_sp3(text.as_bytes(), &request)
            .expect("archive-shaped WUM NRT product validates")
            .1,
        ExactSp3Coverage::HalfOpen
    );

    // The agency pin is terminal: bytes claiming another producer are not
    // this product.
    let foreign = exact_sp3_at(
        &regular_offsets(576, 300),
        576,
        "300.00000000",
        2026,
        8,
        3,
        2430,
        86_400.0,
        61_255,
        "GFZ",
    );
    assert!(matches!(
        parse_exact_sp3(foreign.as_bytes(), &request),
        Err(ExactSp3ValidationError::AgencyMismatch { .. })
    ));
}
