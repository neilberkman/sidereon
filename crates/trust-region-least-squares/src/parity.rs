//! Bit-level fixture and trace utilities for reference parity tests.
//!
//! The trace writer is compiled only with the `trace` feature. Call sites must
//! also be feature-gated so disabled builds allocate nothing and emit no
//! runtime branch.

use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HexBitsError {
    Empty,
    Invalid { input: String, message: String },
}

impl fmt::Display for HexBitsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HexBitsError::Empty => write!(f, "empty f64 bit string"),
            HexBitsError::Invalid { input, message } => {
                write!(f, "invalid f64 bit string {input:?}: {message}")
            }
        }
    }
}

impl Error for HexBitsError {}

pub fn f64_to_hex(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

pub fn f64_from_hex(input: &str) -> Result<f64, HexBitsError> {
    Ok(f64::from_bits(u64_from_hex(input)?))
}

pub fn u64_from_hex(input: &str) -> Result<u64, HexBitsError> {
    let Some(hex) = input
        .strip_prefix("0x")
        .or_else(|| input.strip_prefix("0X"))
    else {
        return if input.is_empty() {
            Err(HexBitsError::Empty)
        } else {
            parse_hex(input)
        };
    };
    parse_hex(hex)
}

fn parse_hex(hex: &str) -> Result<u64, HexBitsError> {
    if hex.is_empty() {
        return Err(HexBitsError::Empty);
    }
    u64::from_str_radix(hex, 16).map_err(|err| HexBitsError::Invalid {
        input: hex.to_string(),
        message: err.to_string(),
    })
}

pub fn f64_slice_to_hex(values: &[f64]) -> Vec<String> {
    values.iter().map(|value| f64_to_hex(*value)).collect()
}

pub fn f64_slice_from_hex(values: &[String]) -> Result<Vec<f64>, HexBitsError> {
    values.iter().map(|value| f64_from_hex(value)).collect()
}

pub fn assert_f64_bits_eq(context: &str, actual: f64, expected: f64) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{context}: actual={actual:?} ({:016x}) expected={expected:?} ({:016x})",
        actual.to_bits(),
        expected.to_bits()
    );
}

pub fn assert_f64_slice_bits_eq(context: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length {} != {}",
        actual.len(),
        expected.len()
    );
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_f64_bits_eq(&format!("{context}[{idx}]"), actual, expected);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TraceAtom {
    Text(String),
    Usize(usize),
    I32(i32),
    F64Bits(u64),
    F64SliceBits(Vec<u64>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TraceOp {
    pub component: String,
    pub event: String,
    pub fields: Vec<(String, TraceAtom)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FirstDivergence {
    pub op_index: usize,
    pub field: Option<String>,
    pub left: String,
    pub right: String,
}

pub fn first_divergent_op(left: &[TraceOp], right: &[TraceOp]) -> Option<FirstDivergence> {
    let shared = left.len().min(right.len());
    for op_index in 0..shared {
        let left_op = &left[op_index];
        let right_op = &right[op_index];
        if left_op.component != right_op.component {
            return Some(FirstDivergence {
                op_index,
                field: Some("component".to_string()),
                left: left_op.component.clone(),
                right: right_op.component.clone(),
            });
        }
        if left_op.event != right_op.event {
            return Some(FirstDivergence {
                op_index,
                field: Some("event".to_string()),
                left: left_op.event.clone(),
                right: right_op.event.clone(),
            });
        }
        if left_op.fields.len() != right_op.fields.len() {
            return Some(FirstDivergence {
                op_index,
                field: None,
                left: format!("{} fields", left_op.fields.len()),
                right: format!("{} fields", right_op.fields.len()),
            });
        }
        for ((left_key, left_value), (right_key, right_value)) in
            left_op.fields.iter().zip(&right_op.fields)
        {
            if left_key != right_key {
                return Some(FirstDivergence {
                    op_index,
                    field: None,
                    left: left_key.clone(),
                    right: right_key.clone(),
                });
            }
            if left_value != right_value {
                return Some(FirstDivergence {
                    op_index,
                    field: Some(left_key.clone()),
                    left: format_trace_atom(left_value),
                    right: format_trace_atom(right_value),
                });
            }
        }
    }

    if left.len() != right.len() {
        return Some(FirstDivergence {
            op_index: shared,
            field: None,
            left: format!("{} ops", left.len()),
            right: format!("{} ops", right.len()),
        });
    }

    None
}

fn format_trace_atom(value: &TraceAtom) -> String {
    match value {
        TraceAtom::Text(value) => value.clone(),
        TraceAtom::Usize(value) => value.to_string(),
        TraceAtom::I32(value) => value.to_string(),
        TraceAtom::F64Bits(value) => format!("0x{value:016x}"),
        TraceAtom::F64SliceBits(values) => {
            let mut out = String::new();
            for (idx, value) in values.iter().enumerate() {
                if idx != 0 {
                    out.push(',');
                }
                out.push_str(&format!("0x{value:016x}"));
            }
            out
        }
    }
}

#[cfg(feature = "trace")]
pub enum TraceValue<'a> {
    Str(&'a str),
    F64(f64),
    Usize(usize),
    I32(i32),
    F64Slice(&'a [f64]),
}

#[cfg(feature = "trace")]
pub struct TraceWriter {
    file: Option<std::fs::File>,
    component: &'static str,
}

#[cfg(feature = "trace")]
impl TraceWriter {
    pub fn from_env(component: &'static str) -> Self {
        let file = std::env::var_os("TRUST_REGION_LEAST_SQUARES_TRACE_PATH").and_then(|path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok()
        });
        Self { file, component }
    }

    pub fn event(&mut self, fields: &[(&str, TraceValue<'_>)]) {
        let Some(file) = &mut self.file else {
            return;
        };
        use std::io::Write;

        let _ = write!(file, "{{\"component\":\"{}\"", self.component);
        for (key, value) in fields {
            let _ = write!(file, ",\"{}\":", key);
            write_trace_value(file, value);
        }
        let _ = writeln!(file, "}}");
    }
}

#[cfg(feature = "trace")]
fn write_trace_value(file: &mut std::fs::File, value: &TraceValue<'_>) {
    use std::io::Write;

    match value {
        TraceValue::Str(value) => {
            let _ = write!(file, "\"{}\"", value);
        }
        TraceValue::F64(value) => {
            let _ = write!(
                file,
                "{{\"value\":{:?},\"bits\":\"0x{:016x}\"}}",
                value,
                value.to_bits()
            );
        }
        TraceValue::Usize(value) => {
            let _ = write!(file, "{value}");
        }
        TraceValue::I32(value) => {
            let _ = write!(file, "{value}");
        }
        TraceValue::F64Slice(values) => {
            let _ = write!(file, "[");
            for (idx, value) in values.iter().enumerate() {
                if idx != 0 {
                    let _ = write!(file, ",");
                }
                let _ = write!(
                    file,
                    "{{\"value\":{:?},\"bits\":\"0x{:016x}\"}}",
                    value,
                    value.to_bits()
                );
            }
            let _ = write!(file, "]");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip_keeps_bits() {
        let values = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0001),
        ];
        for value in values {
            assert_eq!(
                f64_from_hex(&f64_to_hex(value)).unwrap().to_bits(),
                value.to_bits()
            );
        }
    }

    #[test]
    fn first_divergence_reports_field() {
        let left = [TraceOp {
            component: "solver".to_string(),
            event: "step".to_string(),
            fields: vec![("x".to_string(), TraceAtom::F64Bits(1))],
        }];
        let right = [TraceOp {
            component: "solver".to_string(),
            event: "step".to_string(),
            fields: vec![("x".to_string(), TraceAtom::F64Bits(2))],
        }];

        let diff = first_divergent_op(&left, &right).unwrap();
        assert_eq!(diff.op_index, 0);
        assert_eq!(diff.field.as_deref(), Some("x"));
        assert_eq!(diff.left, "0x0000000000000001");
        assert_eq!(diff.right, "0x0000000000000002");
    }
}
