use sidereon::data::ProductDate;
use sidereon::ephemeris::{parse_exact_sp3, ExactSp3Request, ExactSp3ValidationError};

const P_G01: &str = "PG01  15000.000000 -20000.000000   5000.000000    123.456789\n";
const P_G02: &str = "PG02  16000.000000 -21000.000000   6000.000000    124.456789\n";

fn exact_fixture() -> String {
    let mut text = String::from(
        "#dP2020  1  1  0  0  0.00000000      12 ORBIT IGS20  FIT TST\n\
## 2086 259200.00000000   300.00000000 58849 0.0000000000000\n\
+    2   G01G02  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n",
    );
    for _ in 1..5 {
        text.push_str("+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n");
    }
    for _ in 0..5 {
        text.push_str("++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n");
    }
    text.push_str("%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
    text.push_str("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
    text.push_str("%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n");
    text.push_str("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n");
    text.push_str("%i    0    0    0    0      0      0      0      0         0\n");
    text.push_str("%i    0    0    0    0      0      0      0      0         0\n");
    for _ in 0..4 {
        text.push_str("/* FACADE TERMINAL RECORD FIXTURE\n");
    }
    for epoch in 0..12 {
        let minute = epoch * 5;
        text.push_str(&format!("*  2020  1  1  0 {minute:>2}  0.00000000\n"));
        text.push_str(P_G01);
        text.push_str(P_G02);
    }
    text.push_str("EOF\n");
    text
}

fn decode_hex(value: &str) -> Vec<u8> {
    assert!(value.len().is_multiple_of(2));
    value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).expect("ASCII hex"), 16)
                .expect("valid hex")
        })
        .collect()
}

fn case_bytes(base: &str, case: &serde_json::Value) -> Vec<u8> {
    let mut bytes = base
        .strip_suffix("EOF\n")
        .expect("canonical fixture terminal")
        .as_bytes()
        .to_vec();
    bytes.extend(decode_hex(
        case["leading_hex"].as_str().expect("leading_hex"),
    ));
    if let Some(marker) = case["marker"].as_str() {
        bytes.extend(marker.as_bytes());
    }
    bytes.extend(std::iter::repeat_n(
        b' ',
        case["padding_spaces"].as_u64().expect("padding") as usize,
    ));
    for field in ["suffix_hex", "separator_hex", "trailing_hex"] {
        bytes.extend(decode_hex(case[field].as_str().expect(field)));
    }
    bytes
}

fn result_class(
    result: Result<
        (
            sidereon::ephemeris::Sp3,
            sidereon::ephemeris::ExactSp3Coverage,
        ),
        ExactSp3ValidationError,
    >,
) -> &'static str {
    match result {
        Ok(_) => "accept",
        Err(ExactSp3ValidationError::MalformedEofRecord { .. }) => "malformed_eof_record",
        Err(ExactSp3ValidationError::MissingEof) => "missing_eof",
        Err(ExactSp3ValidationError::TrailingContentAfterEof) => "trailing_content_after_eof",
        Err(error) => panic!("unrelated exact error: {error:?}"),
    }
}

#[test]
fn rust_facade_obeys_the_shared_terminal_record_contract() {
    let corpus: serde_json::Value =
        serde_json::from_str(include_str!("../golden/sp3-terminal-record-v1.json"))
            .expect("valid corpus");
    assert_eq!(
        corpus["schema"],
        serde_json::Value::String("sidereon-sp3-terminal-record-v1".to_owned())
    );

    let request = ExactSp3Request::new(
        ProductDate::new(2020, 1, 1).expect("date"),
        Some("0000"),
        "01H",
        "05M",
    )
    .expect("request");
    let base = exact_fixture();
    for case in corpus["cases"].as_array().expect("cases") {
        assert_eq!(
            result_class(parse_exact_sp3(&case_bytes(&base, case), &request)),
            case["expect"].as_str().expect("expect"),
            "facade corpus case {}",
            case["name"].as_str().expect("name")
        );
    }
}
