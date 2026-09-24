//! RINEX clock data records: reading, the public record view and writing.
//!
//! Table A16 columns: 3.00 `A2,1X,A4,1X,I4,4I3,F10.6,I3,3X,E19.12,X,E19.12`
//! with continuation `E19.12,X` x3, `E19.12`; 3.04
//! `A2,1X,A9,1X,I4,1X,4(I2,1X),F9.6,1X,I2,3X,E19.12,2X,E19.12` with
//! continuation `3X`, `E19.12,2X` x3, `E19.12`.

use std::fmt;

use crate::astro::time::model::{Instant, TimeScale};
use crate::format::columns::fixed_record;
use crate::validate::{self, CivilSecondPolicy, FieldError};

use super::epoch::{
    civil_restates_instant, civil_to_instant, clock_epoch_to_civil, instant_to_valid_civil,
    nearest_microsecond_civil, valid_civil_to_clock_epoch, validate_instant, Civil, EpochSource,
};
use super::header::{ClockLayout, ClockTimeSystem};
use super::numeric::{field_name_for_value_index, format_e19_12};
use super::policy::ClockWriteLeniency;
use super::{invalid_input, ClockEpoch, ClockPoint, RinexClockError};

/// A RINEX clock data record type (Table A16).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClockRecordType {
    /// `AR`: analysis result for a receiver clock.
    Ar,
    /// `AS`: analysis result for a satellite clock.
    As,
    /// `CR`: calibration measurement for a receiver.
    Cr,
    /// `DR`: discontinuity measurement for a receiver.
    Dr,
    /// `MS`: monitor measurement for a broadcast satellite clock.
    Ms,
}

impl ClockRecordType {
    /// The two-letter code as written in the file.
    pub fn code(self) -> &'static str {
        match self {
            Self::Ar => "AR",
            Self::As => "AS",
            Self::Cr => "CR",
            Self::Dr => "DR",
            Self::Ms => "MS",
        }
    }

    /// Read a two-letter record type code.
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "AR" => Some(Self::Ar),
            "AS" => Some(Self::As),
            "CR" => Some(Self::Cr),
            "DR" => Some(Self::Dr),
            "MS" => Some(Self::Ms),
            _ => None,
        }
    }
}

impl fmt::Display for ClockRecordType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// How a data record line was read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClockRecordReading {
    /// Read at the columns of a layout. A record read at the columns of the
    /// layout its file does not declare is reported with that layout.
    Columns(ClockLayout),
    /// Read as whitespace-separated values; the line does not follow either
    /// layout's columns.
    Whitespace,
    /// Read at the columns of a layout, with text after that layout's last
    /// column that no field of the record holds, where the line reads neither
    /// at either layout's columns alone nor as whitespace-separated values.
    /// AIUB's short-name CODE MGEX clock files flag some satellite records with
    /// a letter in column 83, past the 80 columns of a version 2.00 record. The
    /// exact text is retained separately, is not read as a value, and is
    /// restated when the record's declared values are edited.
    ColumnsTrailingText(ClockLayout),
    /// Built or edited through the typed API; the record is written in the
    /// product's layout.
    Edited,
}

/// A value present in a record beyond its declared count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockSurplusValue {
    /// Zero-based position in the record's value sequence: 0 bias, 1 bias
    /// sigma, 2 rate, 3 rate sigma, 4 acceleration, 5 acceleration sigma.
    pub position: usize,
    /// The value.
    pub value: f64,
}

/// One data record with its typed reading.
///
/// A record read from a file keeps its source lines in the product; this view
/// is derived from them. `values` are the declared values (bias first); values
/// present beyond the declared count are kept separately in `surplus_values`.
/// Uninterpreted trailing parent-line text is retained verbatim when the
/// record is edited and written.
#[derive(Debug, Clone, PartialEq)]
pub struct ClockRecord {
    pub(super) record_type: ClockRecordType,
    pub(super) name: String,
    pub(super) satellite: Option<String>,
    pub(super) civil: Civil,
    pub(super) second_text: Option<String>,
    pub(super) epoch: Option<Instant>,
    /// What `epoch` is built from.
    pub(super) epoch_source: EpochSource,
    pub(super) values: Vec<f64>,
    pub(super) surplus: Vec<ClockSurplusValue>,
    pub(super) line: Option<usize>,
    pub(super) line_count: usize,
    pub(super) reading: ClockRecordReading,
    pub(super) continuation_reading: Option<ClockRecordReading>,
    /// Exact uninterpreted parent-line bytes and their original start column.
    pub(super) trailing_text: Option<TrailingText>,
}

impl ClockRecord {
    /// Build a record to insert into a product.
    ///
    /// `values` holds the bias followed by up to five further values in the
    /// Table A16 order. An `AS` name must be a satellite identifier and is
    /// stored in its canonical spelling; other names must be a single ASCII
    /// token of at most nine characters. The epoch's second is taken as the
    /// shortest decimal that reads back to the given `f64`; no digit is
    /// rounded, and an epoch the epoch field cannot state (finer than a
    /// microsecond) is refused on insertion. The epoch is checked as a
    /// calendar epoch here and against the product's time system on
    /// insertion.
    pub fn new(
        record_type: ClockRecordType,
        name: &str,
        epoch: ClockEpoch,
        values: Vec<f64>,
    ) -> Result<Self, RinexClockError> {
        validate_values(&values)?;
        let (name, satellite) = if record_type == ClockRecordType::As {
            let satellite = validate::strict_gnss_satellite_id(name, "satellite")
                .map_err(|_| invalid_input("satellite", "not a RINEX satellite identifier"))?
                .to_string();
            (satellite.clone(), Some(satellite))
        } else {
            validate_name(name, "name", ClockLayout::V304)?;
            (name.to_string(), None)
        };
        let civil = clock_epoch_to_civil(epoch, CivilSecondPolicy::UtcLike)
            .ok_or_else(|| invalid_input("epoch", "invalid civil clock epoch"))?;
        Ok(Self {
            record_type,
            name,
            satellite,
            civil,
            second_text: None,
            epoch: None,
            epoch_source: EpochSource::Civil(civil),
            values,
            surplus: Vec::new(),
            line: None,
            line_count: 0,
            reading: ClockRecordReading::Edited,
            continuation_reading: None,
            trailing_text: None,
        })
    }

    /// Record type.
    pub fn record_type(&self) -> ClockRecordType {
        self.record_type
    }

    /// Receiver or satellite name as written, trimmed.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Canonical satellite identifier of an `AS` record.
    pub fn satellite(&self) -> Option<&str> {
        self.satellite.as_deref()
    }

    /// Civil epoch. The epoch keeps every digit the seconds field states; the
    /// `f64` second here is the nearest double to it.
    pub fn civil_epoch(&self) -> ClockEpoch {
        valid_civil_to_clock_epoch(self.civil)
    }

    /// The seconds field of the epoch exactly as the source record states it,
    /// trimmed of blanks, for example `"5.1234567"`. It carries every digit
    /// the field states, including digits the `f64` second of
    /// [`civil_epoch`](Self::civil_epoch) cannot hold. A record read from
    /// text keeps it, and so does such a record after its values are edited
    /// or after it is inserted into a product read from text. `None` for a
    /// record built through [`ClockRecord::new`], and for a record of a
    /// product built from typed instants, whose epoch has no source text.
    pub fn second_text(&self) -> Option<&str> {
        self.second_text.as_deref()
    }

    /// The epoch as an instant in the product's time scale; `None` when the
    /// product's time system resolves to no time scale.
    pub fn epoch(&self) -> Option<Instant> {
        self.epoch
    }

    /// Declared number of values.
    pub fn declared_count(&self) -> usize {
        self.values.len()
    }

    /// Declared values, bias first.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Clock bias, seconds.
    pub fn bias_s(&self) -> f64 {
        self.values.first().copied().unwrap_or(f64::NAN)
    }

    /// Declared values after the bias.
    pub fn additional_values(&self) -> &[f64] {
        self.values.get(1..).unwrap_or(&[])
    }

    /// Values present beyond the declared count.
    pub fn surplus_values(&self) -> &[ClockSurplusValue] {
        &self.surplus
    }

    /// One-based line number of the record's first line in the text the
    /// product was read from; `None` for a record built or edited through the
    /// typed API.
    pub fn line(&self) -> Option<usize> {
        self.line
    }

    /// Number of physical lines the record spans in the source, including
    /// blank lines between the record and its continuation line.
    pub fn line_count(&self) -> usize {
        self.line_count
    }

    /// How the record's first line was read.
    pub fn reading(&self) -> ClockRecordReading {
        self.reading
    }

    /// How the continuation line was read, when the record has one.
    pub fn continuation_reading(&self) -> Option<ClockRecordReading> {
        self.continuation_reading
    }

    /// The satellite clock sample of an `AS` record with a resolved epoch.
    pub fn clock_point(&self) -> Option<ClockPoint> {
        if self.record_type != ClockRecordType::As {
            return None;
        }
        Some(ClockPoint::with_source(
            self.epoch?,
            *self.values.first()?,
            self.additional_values().to_vec(),
            self.epoch_source,
        ))
    }
}

/// Epoch interpretation for the records of one product.
#[derive(Debug, Clone, Copy)]
pub(super) struct EpochContext {
    pub(super) scale: Option<TimeScale>,
    pub(super) policy: CivilSecondPolicy,
}

const PARENT_300: [(usize, usize); 11] = [
    (0, 2),   // record type (A2)
    (3, 7),   // name (A4)
    (8, 12),  // year (I4)
    (12, 15), // month (I3)
    (15, 18), // day (I3)
    (18, 21), // hour (I3)
    (21, 24), // minute (I3)
    (24, 34), // second (F10.6)
    (34, 37), // count (I3)
    (40, 59), // bias (E19.12)
    (60, 79), // bias sigma (E19.12)
];

const PARENT_304: [(usize, usize); 11] = [
    (0, 2),   // record type (A2)
    (3, 12),  // name (A9, or A4,6X, or A3,7X)
    (13, 17), // year (I4)
    (18, 20), // month (I2)
    (21, 23), // day (I2)
    (24, 26), // hour (I2)
    (27, 29), // minute (I2)
    (30, 39), // second (F9.6)
    (40, 42), // count (I2)
    (45, 64), // bias (E19.12)
    (65, 85), // bias sigma (E19.12) after 1X (IGS example, RTKLIB) or 2X (Table A16)
];

const CONTINUATION_300: [(usize, usize); 4] = [(0, 19), (20, 39), (40, 59), (60, 79)];

const CONTINUATION_304: [(usize, usize); 4] = [(3, 22), (24, 43), (45, 64), (66, 85)];

fn parent_columns(layout: ClockLayout) -> [(usize, usize); 11] {
    match layout {
        ClockLayout::V300 => PARENT_300,
        ClockLayout::V304 => PARENT_304,
    }
}

fn continuation_columns(layout: ClockLayout) -> [(usize, usize); 4] {
    match layout {
        ClockLayout::V300 => CONTINUATION_300,
        ClockLayout::V304 => CONTINUATION_304,
    }
}

fn layout_order(layout: Option<ClockLayout>) -> [ClockLayout; 2] {
    match layout {
        Some(ClockLayout::V304) => [ClockLayout::V304, ClockLayout::V300],
        _ => [ClockLayout::V300, ClockLayout::V304],
    }
}

/// A parent line read into typed fields.
pub(super) struct ParentRead {
    pub(super) record_type: ClockRecordType,
    pub(super) name: String,
    pub(super) satellite: Option<String>,
    pub(super) civil: Civil,
    pub(super) second_text: String,
    pub(super) epoch: Option<Instant>,
    pub(super) count: usize,
    pub(super) values: Vec<f64>,
    pub(super) surplus: Vec<ClockSurplusValue>,
    pub(super) reading: ClockRecordReading,
    pub(super) trailing_text: Option<TrailingText>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct TrailingText {
    pub(super) column: usize,
    pub(super) text: String,
}

/// A continuation line read into typed fields.
pub(super) struct ContinuationRead {
    pub(super) values: Vec<f64>,
    pub(super) surplus: Vec<ClockSurplusValue>,
    pub(super) reading: ClockRecordReading,
}

/// Whether a line starts with a data record type, so it cannot be a
/// continuation line.
pub(super) fn is_potential_parent_record(line: &str) -> bool {
    line.split_whitespace()
        .next()
        .is_some_and(|token| ClockRecordType::from_code(token).is_some())
}

/// Read a record's first line: at the declared layout's columns, then at the
/// other layout's columns, then as whitespace-separated values.
pub(super) fn read_parent(
    line_number: usize,
    line: &str,
    layout: Option<ClockLayout>,
    ctx: &EpochContext,
) -> Result<ParentRead, RinexClockError> {
    for candidate in layout_order(layout) {
        if let Some(fields) = fixed_record(line, parent_columns(candidate)) {
            let code = fields[0];
            if code.len() == 2 && code.chars().all(|c| c.is_ascii_alphabetic()) {
                return read_parent_columns(line_number, line, fields, candidate, ctx);
            }
        }
    }
    let tokens = read_parent_tokens(line_number, line, ctx);
    if tokens.is_ok() {
        return tokens;
    }
    // A line that reads no other way and follows a layout's columns up to its
    // last one, with text after it that no field holds: read the columns and
    // keep the text in the source.
    for candidate in layout_order(layout) {
        let columns = parent_columns(candidate);
        let end = columns[columns.len() - 1].1;
        if !line.is_ascii() || line.len() <= end || line[end..].trim().is_empty() {
            continue;
        }
        if let Some(fields) = fixed_record(&line[..end], columns) {
            let code = fields[0];
            if code.len() == 2 && code.chars().all(|c| c.is_ascii_alphabetic()) {
                if let Ok(mut read) = read_parent_columns(line_number, line, fields, candidate, ctx)
                {
                    read.reading = ClockRecordReading::ColumnsTrailingText(candidate);
                    read.trailing_text = Some(TrailingText {
                        column: end,
                        text: line[end..].to_string(),
                    });
                    return Ok(read);
                }
            }
        }
    }
    tokens
}

fn read_parent_columns(
    line_number: usize,
    line: &str,
    fields: [&str; 11],
    layout: ClockLayout,
    ctx: &EpochContext,
) -> Result<ParentRead, RinexClockError> {
    let [code, name_field, year_field, month_field, day_field, hour_field, minute_field, second_field, count_field, bias_field, sigma_field] =
        fields;
    let Some(record_type) = ClockRecordType::from_code(code) else {
        return Err(RinexClockError::BadField {
            line: line_number,
            field: "record_type",
            value: code.to_string(),
        });
    };
    let reading = ClockRecordReading::Columns(layout);

    if record_type == ClockRecordType::As {
        if bias_field.is_empty() {
            return Err(RinexClockError::MalformedAsRecord {
                line: line_number,
                reason: "expected at least 10 fields",
                record: line.trim().to_string(),
            });
        }
        let satellite = read_satellite(line_number, name_field)?;
        let epoch_fields = EpochFields::read(
            line_number,
            [year_field, month_field, day_field, hour_field, minute_field],
            second_field,
        )?;
        let bias = parse_f64_field(line_number, "bias", bias_field)?;
        let (civil, epoch) = epoch_fields.convert(line_number, ctx)?;
        let count = read_count(line_number, count_field)?;
        let (values, surplus) = parent_values(line_number, count, bias, Some(sigma_field))?;
        return Ok(ParentRead {
            record_type,
            name: name_field.to_string(),
            satellite: Some(satellite),
            civil,
            second_text: second_field.to_string(),
            epoch,
            count,
            values,
            surplus,
            reading,
            trailing_text: None,
        });
    }

    let count = read_count(line_number, count_field)?;
    if count >= 2 && sigma_field.is_empty() {
        return Err(RinexClockError::BadField {
            line: line_number,
            field: "sigma",
            value: String::new(),
        });
    }
    if name_field.is_empty() {
        return Err(RinexClockError::BadField {
            line: line_number,
            field: "name",
            value: String::new(),
        });
    }
    let epoch_fields = EpochFields::read(
        line_number,
        [year_field, month_field, day_field, hour_field, minute_field],
        second_field,
    )?;
    if bias_field.is_empty() {
        return Err(RinexClockError::BadField {
            line: line_number,
            field: "bias",
            value: String::new(),
        });
    }
    let bias = parse_f64_field(line_number, "bias", bias_field)?;
    let (civil, epoch) = epoch_fields.convert(line_number, ctx)?;
    let (values, surplus) = parent_values(line_number, count, bias, Some(sigma_field))?;
    Ok(ParentRead {
        record_type,
        name: name_field.to_string(),
        satellite: None,
        civil,
        second_text: second_field.to_string(),
        epoch,
        count,
        values,
        surplus,
        reading,
        trailing_text: None,
    })
}

fn read_parent_tokens(
    line_number: usize,
    line: &str,
    ctx: &EpochContext,
) -> Result<ParentRead, RinexClockError> {
    let mut tokens = line.split_whitespace();
    let Some(first) = tokens.next() else {
        return Err(RinexClockError::BadField {
            line: line_number,
            field: "record_type",
            value: String::new(),
        });
    };
    let Some(record_type) = ClockRecordType::from_code(first) else {
        return Err(RinexClockError::BadField {
            line: line_number,
            field: "record_type",
            value: first.to_string(),
        });
    };
    let reading = ClockRecordReading::Whitespace;

    if record_type == ClockRecordType::As {
        let mut next = || {
            tokens
                .next()
                .ok_or_else(|| RinexClockError::MalformedAsRecord {
                    line: line_number,
                    reason: "expected at least 10 fields",
                    record: line.trim().to_string(),
                })
        };
        let sat_field = next()?;
        let year_field = next()?;
        let month_field = next()?;
        let day_field = next()?;
        let hour_field = next()?;
        let minute_field = next()?;
        let second_field = next()?;
        let count_field = next()?;
        let bias_field = next()?;
        let rest: Vec<&str> = tokens.collect();

        let satellite = read_satellite(line_number, sat_field)?;
        let epoch_fields = EpochFields::read(
            line_number,
            [year_field, month_field, day_field, hour_field, minute_field],
            second_field,
        )?;
        let bias = parse_f64_field(line_number, "bias", bias_field)?;
        let (civil, epoch) = epoch_fields.convert(line_number, ctx)?;
        let count = read_count(line_number, count_field)?;
        if count >= 2 && rest.is_empty() {
            return Err(RinexClockError::BadField {
                line: line_number,
                field: "sigma",
                value: String::new(),
            });
        }
        let (values, surplus) = parent_values(line_number, count, bias, rest.first().copied())?;
        if rest.len() > 1 {
            return Err(RinexClockError::MalformedAsRecord {
                line: line_number,
                reason: "excess values in parent record",
                record: line.trim().to_string(),
            });
        }
        return Ok(ParentRead {
            record_type,
            name: sat_field.to_string(),
            satellite: Some(satellite),
            civil,
            second_text: second_field.to_string(),
            epoch,
            count,
            values,
            surplus,
            reading,
            trailing_text: None,
        });
    }

    let mut leading = Vec::with_capacity(7);
    for _ in 0..7 {
        let Some(token) = tokens.next() else {
            return Err(RinexClockError::BadField {
                line: line_number,
                field: "count",
                value: String::new(),
            });
        };
        leading.push(token);
    }
    let count_field = tokens.next().ok_or_else(|| RinexClockError::BadField {
        line: line_number,
        field: "count",
        value: String::new(),
    })?;
    let count = read_count(line_number, count_field)?;
    let bias_field = tokens.next().ok_or_else(|| RinexClockError::BadField {
        line: line_number,
        field: "bias",
        value: String::new(),
    })?;
    let rest: Vec<&str> = tokens.collect();
    if count >= 2 && rest.is_empty() {
        return Err(RinexClockError::BadField {
            line: line_number,
            field: "sigma",
            value: String::new(),
        });
    }
    if rest.len() > 1 {
        return Err(RinexClockError::BadField {
            line: line_number,
            field: "sigma",
            value: "excess value in parent record".to_string(),
        });
    }
    let epoch_fields = EpochFields::read(
        line_number,
        [leading[1], leading[2], leading[3], leading[4], leading[5]],
        leading[6],
    )?;
    let bias = parse_f64_field(line_number, "bias", bias_field)?;
    let (civil, epoch) = epoch_fields.convert(line_number, ctx)?;
    let (values, surplus) = parent_values(line_number, count, bias, rest.first().copied())?;
    Ok(ParentRead {
        record_type,
        name: leading[0].to_string(),
        satellite: None,
        civil,
        second_text: leading[6].to_string(),
        epoch,
        count,
        values,
        surplus,
        reading,
        trailing_text: None,
    })
}

/// Declared parent values, and the bias sigma column read as a surplus value
/// when the record declares only its bias.
fn parent_values(
    line_number: usize,
    count: usize,
    bias: f64,
    sigma_field: Option<&str>,
) -> Result<(Vec<f64>, Vec<ClockSurplusValue>), RinexClockError> {
    let sigma_field = sigma_field.filter(|field| !field.is_empty());
    if count == 1 {
        return match sigma_field {
            None => Ok((vec![bias], Vec::new())),
            Some(field) => {
                let value = parse_f64_field(line_number, "sigma", field)?;
                Ok((vec![bias], vec![ClockSurplusValue { position: 1, value }]))
            }
        };
    }
    let Some(field) = sigma_field else {
        return Err(RinexClockError::BadField {
            line: line_number,
            field: "sigma",
            value: String::new(),
        });
    };
    let sigma = parse_f64_field(line_number, "sigma", field)?;
    Ok((vec![bias, sigma], Vec::new()))
}

/// Read a continuation line carrying `needed` declared values.
pub(super) fn read_continuation(
    line_number: usize,
    line: &str,
    needed: usize,
    layout: Option<ClockLayout>,
) -> Result<ContinuationRead, RinexClockError> {
    let malformed = |reason: &'static str| RinexClockError::MalformedContinuation {
        line: line_number,
        reason,
        record: line.trim().to_string(),
    };
    if needed == 0 || needed > 4 {
        return Err(malformed("invalid needed value count"));
    }

    let order = match layout {
        Some(layout) => layout_order(Some(layout)),
        None if line.starts_with("   ") => [ClockLayout::V304, ClockLayout::V300],
        None => [ClockLayout::V300, ClockLayout::V304],
    };
    for candidate in order {
        let Some(fields) = fixed_record(line, continuation_columns(candidate)) else {
            continue;
        };
        if !fields.iter().any(|f| !f.is_empty())
            || !fields.iter().all(|f| f.split_whitespace().count() <= 1)
        {
            continue;
        }
        if fields[..needed].iter().any(|f| f.is_empty()) {
            return Err(malformed("missing required continuation value"));
        }
        let mut values = Vec::with_capacity(needed);
        let mut surplus = Vec::new();
        for (index, field) in fields.iter().enumerate() {
            if field.is_empty() {
                continue;
            }
            let value = parse_f64_field(line_number, field_name_for_value_index(index + 2), field)
                .map_err(|_| malformed("invalid numeric field"))?;
            if index < needed {
                values.push(value);
            } else {
                surplus.push(ClockSurplusValue {
                    position: index + 2,
                    value,
                });
            }
        }
        return Ok(ContinuationRead {
            values,
            surplus,
            reading: ClockRecordReading::Columns(candidate),
        });
    }

    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < needed {
        return Err(malformed("too few values in continuation line"));
    }
    if tokens.len() > 4 {
        return Err(malformed("excess values in continuation line"));
    }
    let mut values = Vec::with_capacity(needed);
    let mut surplus = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        let value = parse_f64_field(line_number, field_name_for_value_index(index + 2), token)
            .map_err(|_| malformed("invalid numeric field"))?;
        if index < needed {
            values.push(value);
        } else {
            surplus.push(ClockSurplusValue {
                position: index + 2,
                value,
            });
        }
    }
    Ok(ContinuationRead {
        values,
        surplus,
        reading: ClockRecordReading::Whitespace,
    })
}

#[derive(Debug, Clone, Copy)]
struct EpochFields<'a> {
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: &'a str,
}

impl<'a> EpochFields<'a> {
    fn read(
        line_number: usize,
        [year, month, day, hour, minute]: [&str; 5],
        second: &'a str,
    ) -> Result<Self, RinexClockError> {
        Ok(Self {
            year: parse_int_field::<i32>(line_number, "year", year)?,
            month: parse_int_field::<u8>(line_number, "month", month)?,
            day: parse_int_field::<u8>(line_number, "day", day)?,
            hour: parse_int_field::<u8>(line_number, "hour", hour)?,
            minute: parse_int_field::<u8>(line_number, "minute", minute)?,
            second,
        })
    }

    fn convert(
        self,
        line_number: usize,
        ctx: &EpochContext,
    ) -> Result<(Civil, Option<Instant>), RinexClockError> {
        let civil = validate::civil_datetime_with_femtosecond_policy(
            i64::from(self.year),
            i64::from(self.month),
            i64::from(self.day),
            i64::from(self.hour),
            i64::from(self.minute),
            self.second,
            ctx.policy,
        )
        .map_err(|error| map_epoch_error(line_number, error, self))?;
        if civil.year < 1 {
            let error = FieldError::InvalidCivilDate {
                field: "civil datetime",
                year: civil.year,
                month: i64::from(civil.month),
                day: i64::from(civil.day),
            };
            return Err(map_epoch_error(line_number, error, self));
        }
        let epoch = match ctx.scale {
            Some(scale) => Some(
                civil_to_instant(scale, civil)
                    .map_err(|error| map_epoch_error(line_number, error, self))?,
            ),
            None => None,
        };
        Ok((civil, epoch))
    }
}

fn read_satellite(line_number: usize, field: &str) -> Result<String, RinexClockError> {
    validate::strict_gnss_satellite_id(field, "satellite")
        .map(|id| id.to_string())
        .map_err(|error| map_field_error(line_number, error, field))
}

fn read_count(line_number: usize, field: &str) -> Result<usize, RinexClockError> {
    let count = parse_int_field::<usize>(line_number, "count", field)?;
    if !(1..=6).contains(&count) {
        return Err(RinexClockError::BadField {
            line: line_number,
            field: "count",
            value: field.to_string(),
        });
    }
    Ok(count)
}

fn parse_int_field<T>(
    line_number: usize,
    field: &'static str,
    value: &str,
) -> Result<T, RinexClockError>
where
    T: std::str::FromStr,
{
    validate::strict_int(value, field).map_err(|error| map_field_error(line_number, error, value))
}

fn parse_f64_field(
    line_number: usize,
    field: &'static str,
    value: &str,
) -> Result<f64, RinexClockError> {
    validate::strict_f64(value, field).map_err(|error| map_field_error(line_number, error, value))
}

fn map_field_error(line_number: usize, error: FieldError, value: &str) -> RinexClockError {
    RinexClockError::BadField {
        line: line_number,
        field: error.field(),
        value: value.to_string(),
    }
}

fn map_epoch_error(
    line_number: usize,
    error: FieldError,
    epoch: EpochFields<'_>,
) -> RinexClockError {
    match error {
        FieldError::FloatParse { .. }
        | FieldError::Missing { .. }
        | FieldError::NonFinite { .. } => RinexClockError::BadField {
            line: line_number,
            field: "second",
            value: epoch.second.to_string(),
        },
        _ => RinexClockError::BadField {
            line: line_number,
            field: "epoch",
            value: format!(
                "{} {} {} {} {} {}",
                epoch.year,
                epoch.month,
                epoch.day,
                epoch.hour,
                epoch.minute,
                normalized_second_text(epoch.second)
            ),
        },
    }
}

fn normalized_second_text(second: &str) -> String {
    validate::strict_f64(second, "second")
        .map_or_else(|_| second.to_string(), |value| value.to_string())
}

/// A record whose authority is its typed values: built from series rows,
/// inserted, or edited. It has no source lines and is written in the
/// product's layout; a source record's opaque parent-line suffix is retained.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TypedRecord {
    pub(super) record_type: ClockRecordType,
    pub(super) name: String,
    pub(super) epoch: TypedEpoch,
    pub(super) values: Vec<f64>,
    /// Exact uninterpreted parent-line bytes and their original start column.
    pub(super) trailing_text: Option<TrailingText>,
}

/// The epoch of a typed record.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum TypedEpoch {
    /// A civil epoch, interpreted in the product's time system, with the
    /// seconds field text of the source record it was edited from.
    Civil {
        /// The epoch.
        civil: Civil,
        /// The source seconds field, trimmed, which the writer restates when
        /// it fits the layout's seconds field.
        second_text: Option<String>,
    },
    /// A scale-tagged instant supplied by a constructor, with what it was
    /// built from.
    Instant {
        /// The epoch.
        instant: Instant,
        /// The civil tag or GPS seconds `instant` was built from, when known.
        source: EpochSource,
    },
}

impl TypedRecord {
    /// The public view of a typed record.
    pub(super) fn view(&self, ctx: &EpochContext) -> ClockRecord {
        let (civil, second_text, epoch, epoch_source) = match &self.epoch {
            TypedEpoch::Civil { civil, second_text } => (
                *civil,
                second_text.clone(),
                ctx.scale
                    .and_then(|scale| civil_to_instant(scale, *civil).ok()),
                EpochSource::Civil(*civil),
            ),
            TypedEpoch::Instant { instant, source } => (
                instant_to_valid_civil(instant),
                None,
                Some(*instant),
                *source,
            ),
        };
        let satellite = (self.record_type == ClockRecordType::As)
            .then(|| validate::strict_gnss_satellite_id(&self.name, "satellite").ok())
            .flatten()
            .map(|id| id.to_string());
        ClockRecord {
            record_type: self.record_type,
            name: self.name.clone(),
            satellite,
            civil,
            second_text,
            epoch,
            epoch_source,
            values: self.values.clone(),
            surplus: Vec::new(),
            line: None,
            line_count: 0,
            reading: ClockRecordReading::Edited,
            continuation_reading: None,
            trailing_text: self.trailing_text.clone(),
        }
    }
}

/// Check a declared value list: one to six finite values.
pub(super) fn validate_values(values: &[f64]) -> Result<(), RinexClockError> {
    let Some(&bias) = values.first() else {
        return Err(invalid_input("bias_s", "a record states at least its bias"));
    };
    if !bias.is_finite() {
        return Err(invalid_input("bias_s", "must be finite"));
    }
    if values.len() > 6 {
        return Err(invalid_input(
            "additional_values",
            "cannot exceed 5 additional values (maximum count is 6)",
        ));
    }
    for (index, value) in values.iter().enumerate().skip(1) {
        if !value.is_finite() {
            return Err(invalid_input(
                field_name_for_value_index(index),
                "must be finite",
            ));
        }
    }
    Ok(())
}

/// Check that a name can be written in a layout's name field.
pub(super) fn validate_name(
    name: &str,
    field: &'static str,
    layout: ClockLayout,
) -> Result<(), RinexClockError> {
    if name.is_empty() || !name.is_ascii() || name.chars().any(|c| c.is_ascii_whitespace()) {
        return Err(invalid_input(field, "must be a non-empty ASCII token"));
    }
    if name.len() > layout.name_width() {
        return Err(invalid_input(
            field,
            "wider than the name field of the layout",
        ));
    }
    Ok(())
}

/// Blanks between the bias and its sigma in a 3.04-layout record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SigmaGap {
    /// One blank: sigma in columns 66-84, as the IGS combination example after
    /// Table A17 places it and RTKLIB reads it.
    One,
    /// Two blanks: sigma in columns 67-85, as Table A16 describes.
    Two,
}

/// The 3.04 sigma spacing a source line uses, when it carries a sigma: the
/// column its sigma ends in.
pub(super) fn sigma_gap_of_line(line: &str) -> Option<SigmaGap> {
    let bytes = line.as_bytes();
    let end = bytes.len().min(85);
    match (65..end).rev().find(|&index| bytes[index] != b' ')? {
        84 => Some(SigmaGap::Two),
        _ => Some(SigmaGap::One),
    }
}

/// Write a typed record in a layout, refusing every departure. See
/// [`render_record_with`].
pub(super) fn render_record(
    record: &TypedRecord,
    layout: ClockLayout,
    product_scale: Option<TimeScale>,
    sigma_gap: SigmaGap,
) -> Result<Vec<String>, RinexClockError> {
    render_record_with(
        record,
        layout,
        product_scale,
        sigma_gap,
        ClockWriteLeniency::Strict,
    )
    .map(|(lines, _)| lines)
}

/// Write a typed record in a layout. `product_scale` is the scale every
/// instant epoch must carry. Every value is written only as a spelling that
/// reads back to the value held, and a value no 19-column field states exactly
/// is refused by name. An epoch the seconds field cannot state exactly (a
/// digit finer than a microsecond that the source text does not carry within
/// the field width, or an instant no microsecond text restates) is refused,
/// or written at the nearest microsecond when `nearest_epochs` allows it; the
/// returned flag says whether it was.
pub(super) fn render_record_with(
    record: &TypedRecord,
    layout: ClockLayout,
    product_scale: Option<TimeScale>,
    sigma_gap: SigmaGap,
    nearest_epochs: ClockWriteLeniency,
) -> Result<(Vec<String>, bool), RinexClockError> {
    validate_values(&record.values)?;
    let seconds_width = match layout {
        ClockLayout::V300 => 10,
        ClockLayout::V304 => 9,
    };
    let allow = nearest_epochs == ClockWriteLeniency::Allow;
    let mut rounded = false;
    // A restated source text keeps at least one blank before it: the 3.00
    // seconds field follows the minute directly, and a text filling it would
    // run into the minute for every reader that splits on blanks.
    let (civil, source_text) = match &record.epoch {
        TypedEpoch::Civil { civil, second_text } => {
            let fitting = second_text
                .as_deref()
                .filter(|text| text.is_ascii() && !text.is_empty() && text.len() <= 9);
            let finer_text = second_text
                .as_deref()
                .is_some_and(states_digits_below_femtoseconds);
            if fitting.is_some() {
                (*civil, fitting)
            } else if civil.femtosecond == 0 && !finer_text {
                (*civil, None)
            } else if allow {
                rounded = true;
                let nearest = nearest_microsecond_civil(*civil)
                    .ok_or_else(|| invalid_input("epoch", "invalid civil clock epoch"))?;
                (nearest, None)
            } else if finer_text {
                return Err(invalid_input(
                    "epoch",
                    "the source seconds field states digits finer than this product holds",
                ));
            } else {
                return Err(invalid_input(
                    "epoch",
                    "the seconds field states microseconds and this epoch carries finer digits",
                ));
            }
        }
        TypedEpoch::Instant { instant, .. } => {
            validate_instant(*instant, "epoch")?;
            if ClockTimeSystem::for_time_scale(instant.scale).is_none() {
                return Err(RinexClockError::UnsupportedTimeScale {
                    scale: instant.scale,
                });
            }
            if product_scale.is_some_and(|scale| scale != instant.scale) {
                return Err(invalid_input(
                    "epoch",
                    "epoch scale does not match clock time scale",
                ));
            }
            let civil = instant_to_valid_civil(instant);
            if !civil_restates_instant(civil, instant) {
                if !allow {
                    return Err(invalid_input(
                        "epoch",
                        "the epoch field cannot restate this instant without rounding it",
                    ));
                }
                rounded = true;
            }
            (civil, None)
        }
    };
    let seconds_field = match source_text {
        Some(text) => format!("{text:>seconds_width$}"),
        None => {
            let text = format!("{}.{:06}", civil.second, civil.microsecond);
            format!("{text:>seconds_width$}")
        }
    };
    let formatted = record
        .values
        .iter()
        .enumerate()
        .map(|(index, &value)| format_e19_12(value, field_name_for_value_index(index)))
        .collect::<Result<Vec<_>, _>>()?;
    let name_field = if record.record_type == ClockRecordType::As {
        "satellite"
    } else {
        "name"
    };
    validate_name(&record.name, name_field, layout)?;
    if !(0..=9999).contains(&civil.year) {
        return Err(invalid_input(
            "epoch",
            "civil year does not fit the four-digit year field",
        ));
    }

    let code = record.record_type.code();
    let name = &record.name;
    let (year, month, day, hour, minute) =
        (civil.year, civil.month, civil.day, civil.hour, civil.minute);
    let count = formatted.len();
    let mut lines = Vec::with_capacity(2);
    let mut parent = match layout {
        // A2,1X,A4,1X,I4,4I3,F10.6,I3,3X,E19.12
        ClockLayout::V300 => format!(
            "{code} {name:<4} {year:04} {month:02} {day:02} {hour:02} {minute:02}{seconds_field}{count:>3}   {}",
            formatted[0]
        ),
        // A2,1X,A9,1X,I4,1X,4(I2,1X),F9.6,1X,I2,3X,E19.12
        ClockLayout::V304 => format!(
            "{code} {name:<9} {year:04} {month:02} {day:02} {hour:02} {minute:02} {seconds_field} {count:>2}   {}",
            formatted[0]
        ),
    };
    if let Some(sigma) = formatted.get(1) {
        parent.push_str(match (layout, sigma_gap) {
            (ClockLayout::V300, _) | (ClockLayout::V304, SigmaGap::One) => " ",
            (ClockLayout::V304, SigmaGap::Two) => "  ",
        });
        parent.push_str(sigma);
    }
    if let Some(trailing_text) = &record.trailing_text {
        if parent.len() > trailing_text.column {
            return Err(invalid_input(
                "trailing_text",
                "the rendered record overlaps the source suffix column",
            ));
        }
        parent.push_str(&" ".repeat(trailing_text.column - parent.len()));
        parent.push_str(&trailing_text.text);
    }
    lines.push(parent);
    if count > 2 {
        let (lead, separator) = match layout {
            ClockLayout::V300 => ("", " "),
            ClockLayout::V304 => ("   ", "  "),
        };
        lines.push(format!("{lead}{}", formatted[2..].join(separator)));
    }
    Ok((lines, rounded))
}

/// Whether a seconds field text has a nonzero digit past the fifteenth
/// fractional digit, which the civil epoch read from it rounded.
fn states_digits_below_femtoseconds(text: &str) -> bool {
    text.split_once('.')
        .and_then(|(_, fraction)| fraction.get(15..))
        .is_some_and(|rest| rest.bytes().any(|b| b != b'0'))
}
