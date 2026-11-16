#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::omm::OmmError;
use sidereon_core::data::{
    parse_celestrak_gp_response, CelestrakGpFormat, CelestrakGpRejectionReason as R,
};

fuzz_target!(|data: &[u8]| {
    for format in [
        CelestrakGpFormat::Tle,
        CelestrakGpFormat::ThreeLine,
        CelestrakGpFormat::TwoLine,
        CelestrakGpFormat::Xml,
        CelestrakGpFormat::Kvn,
        CelestrakGpFormat::Json,
        CelestrakGpFormat::JsonPretty,
        CelestrakGpFormat::Csv,
    ] {
        if let Ok(response) = parse_celestrak_gp_response(data, format) {
            let body = data.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(data);
            let text = std::str::from_utf8(body).unwrap();
            let container = match format {
                CelestrakGpFormat::Xml => Some((
                    sidereon_core::astro::omm::parse_xml_all(text.trim()).unwrap(),
                    text.len() - text.trim_start().len(),
                )),
                CelestrakGpFormat::Json | CelestrakGpFormat::JsonPretty => Some((
                    sidereon_core::astro::omm::parse_json_array(text).unwrap(),
                    0,
                )),
                CelestrakGpFormat::Csv => Some((
                    sidereon_core::astro::omm::parse_csv_array(text.trim()).unwrap(),
                    text.len() - text.trim_start().len(),
                )),
                _ => None,
            };
            let mut segments = response
                .records
                .iter()
                .map(|record| {
                    (
                        record.record_index,
                        record.line_number,
                        record.raw.as_bytes(),
                        true,
                    )
                })
                .chain(response.rejected.iter().map(|record| {
                    (
                        record.record_index,
                        record.line_number,
                        record.raw.as_bytes(),
                        uses_record_start(&record.reason),
                    )
                }))
                .collect::<Vec<_>>();
            segments.sort_by_key(|segment| segment.0);
            for (position, segment) in segments.iter().enumerate() {
                assert_eq!(segment.0, position + 1);
                if let Some((array, offset)) = &container {
                    assert_eq!(
                        segment.1,
                        Some(line_number(body, offset + array.spans[position].byte_start))
                    );
                }
            }
            let accounted = segments
                .iter()
                .flat_map(|segment| segment.2.iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(accounted, body);

            let mut offset = data.len() - body.len();
            for (_, line, raw, uses_start) in segments {
                let start_line = line_number(data, offset);
                let raw_start = offset;
                offset += raw.len();
                let end_line = line_number(data, offset.saturating_sub(1));
                if let Some(line) = line {
                    assert!((start_line..=end_line).contains(&line));
                    if uses_start
                        && matches!(
                            format,
                            CelestrakGpFormat::Tle
                                | CelestrakGpFormat::ThreeLine
                                | CelestrakGpFormat::TwoLine
                                | CelestrakGpFormat::Kvn
                        )
                    {
                        let text = std::str::from_utf8(raw).unwrap();
                        let first = text
                            .find(|character: char| !character.is_whitespace())
                            .unwrap();
                        assert_eq!(line, line_number(data, raw_start + first));
                    }
                }
            }
        }
    }
});

fn uses_record_start(reason: &R) -> bool {
    match reason {
        R::Tle(_) => false,
        R::Structural(_) => true,
        R::Omm(error) => !matches!(
            error,
            OmmError::MalformedLine { .. }
                | OmmError::InvalidField { .. }
                | OmmError::DuplicateField { .. }
                | OmmError::UnitMismatch { .. }
                | OmmError::Epoch(_)
        ),
    }
}

fn line_number(text: &[u8], offset: usize) -> usize {
    let mut line = 1usize;
    let mut index = 0usize;
    while index < offset.min(text.len()) {
        if matches!(text[index], b'\r' | b'\n') {
            let first = text[index];
            index += 1;
            if index < offset && matches!((first, text[index]), (b'\r', b'\n') | (b'\n', b'\r')) {
                index += 1;
            }
            line += 1;
        } else {
            index += 1;
        }
    }
    line
}
