//! Two-Line Element (TLE) format parser and encoder.
//!
//! TLE is the legacy fixed-width format for satellite orbital elements, designed
//! for 80-column punch cards. This module owns the complete format grammar: ASCII
//! and fixed-width validation, the modulo-10 checksum, the "assumed decimal"
//! exponent codec used for the drag terms, the per-field number formatting, and
//! the two-digit-year pivot. It runs identically regardless of the calling
//! language, so it lives in the core; the sidereon Elixir binding only marshals the
//! epoch between its native `DateTime` and the `(year, day_of_year)` pair this
//! module exposes, normalizes input defaults, and maps errors.
//!
//! The epoch is represented here as a calendar year plus a one-based fractional
//! day-of-year, exactly the two quantities the TLE epoch field encodes. This
//! module owns the TLE two-digit-year pivot and converts that pair into SGP4's
//! split Julian date when building the format-agnostic element set.

use std::fmt;

use libm::{floor, log10, pow};

use crate::astro::sgp4::{self, ElementSet};
use crate::validate;

/// Maximum significant length of a TLE line (columns 1-69). Trailing content is
/// trimmed to this width before parsing, matching the reference behavior.
const MAX_LINE_LEN: usize = 69;
/// Highest ASCII code point permitted in a TLE line.
const MAX_ASCII: u32 = 127;
/// Minimum significant length of line 1 accepted by the lenient parser.
const LINE1_MIN_LEN: usize = 64;
/// Minimum significant length of line 2 accepted by the lenient parser.
const LINE2_MIN_LEN: usize = 68;
/// Column index of the checksum digit (zero-based).
const CHECKSUM_COL: usize = 68;
/// Two-digit-year pivot: years below this map to 2000+, otherwise 1900+. This is
/// the long-standing NORAD convention for the TLE epoch year.
const YEAR_PIVOT: i32 = 57;
/// The TLE record body occupies columns 1-68; column 69 is the checksum.
const BODY_LEN: usize = 68;

/// Decimal places carried by the TLE epoch day-of-year field.
const EPOCH_DAY_DECIMALS: usize = 8;
/// Total width of the formatted epoch day-of-year field (`DDD.DDDDDDDD`).
const EPOCH_DAY_WIDTH: usize = 12;
/// Decimal places carried by the first mean-motion derivative field.
const NDOT_DECIMALS: usize = 8;
/// Width of the formatted first mean-motion derivative field.
const NDOT_WIDTH: usize = 9;
/// Decimal places carried by the assumed-decimal mantissa.
const ASSUMED_DECIMAL_MANTISSA_DECIMALS: usize = 5;
/// Number of mantissa digits emitted in an assumed-decimal field.
const ASSUMED_DECIMAL_MANTISSA_DIGITS: usize = 5;
/// Decimal places carried by the eccentricity field.
const ECCENTRICITY_DECIMALS: usize = 7;
/// Digits emitted for the (leading-decimal-stripped) eccentricity field.
const ECCENTRICITY_DIGITS: usize = 7;
/// Decimal places carried by an angle field (inclination, RAAN, ...).
const ANGLE_DECIMALS: usize = 4;
/// Width of a formatted angle field.
const ANGLE_WIDTH: usize = 8;
/// Decimal places carried by the mean-motion field.
const MEAN_MOTION_DECIMALS: usize = 8;
/// Width of the formatted mean-motion field.
const MEAN_MOTION_WIDTH: usize = 11;
/// Width of the element-set-number field.
const ELSET_WIDTH: usize = 4;
/// Width of the revolution-number field.
const REV_WIDTH: usize = 5;
/// Width of the zero-padded catalog-number field.
const CATALOG_WIDTH: usize = 5;
/// Width of the international-designator field.
const INTL_DESIGNATOR_WIDTH: usize = 8;
/// Width of the two-digit epoch year field.
const EPOCH_YEAR_WIDTH: usize = 2;

/// Parsed TLE orbital elements in canonical astrodynamic units.
///
/// Angles are degrees, mean motion is revolutions/day and its derivatives
/// rev/day^2 and rev/day^3, BSTAR drag is 1/earth-radii, and the epoch is the
/// calendar `epoch_year` plus the one-based fractional `epoch_day_of_year`.
#[derive(Debug, Clone, PartialEq)]
pub struct TleElements {
    pub catalog_number: String,
    pub classification: String,
    pub international_designator: String,
    pub epoch_year: i32,
    pub epoch_day_of_year: f64,
    pub mean_motion_dot: f64,
    pub mean_motion_double_dot: f64,
    pub bstar: f64,
    pub ephemeris_type: i32,
    pub elset_number: i32,
    pub inclination_deg: f64,
    pub raan_deg: f64,
    pub eccentricity: f64,
    pub arg_perigee_deg: f64,
    pub mean_anomaly_deg: f64,
    pub mean_motion: f64,
    pub rev_number: i32,
}

impl TleElements {
    /// Convert these parsed TLE elements into the canonical SGP4 [`ElementSet`]
    /// IR consumed by [`crate::astro::sgp4::Satellite::from_elements`].
    ///
    /// This is the single TLE-to-IR mapping: the public TLE entry point parses a
    /// TLE to [`TleElements`], converts here, and feeds the result into the same
    /// `ElementSet -> satrec` initialization every other input format uses, so
    /// there is no separate TLE-direct propagation path.
    ///
    /// The mapping is bit-preserving for SGP4. The angle, eccentricity, mean
    /// motion, and epoch-day fields are carried through unchanged until the
    /// epoch is converted through the same `days2mdhms`/`jday` math and
    /// 8-decimal fraction rounding Vallado uses for TLE input. B\* and the
    /// second mean-motion derivative are decoded with `powi` in [`parse`]
    /// precisely so they equal the `mantissa * 10^exp` product the element-set
    /// initializer expects; they too pass through unchanged.
    ///
    /// The catalog number is parsed to the numeric form `ElementSet` carries; it
    /// is used only for SGP4 diagnostics and does not affect propagation, so a
    /// non-numeric (Alpha-5) catalog falls back to `0`.
    pub fn to_element_set(&self) -> Result<ElementSet, TleError> {
        validate_tle_bridge(self)?;
        Ok(ElementSet {
            epoch: sgp4::sgp4_julian_date_from_day_of_year(self.epoch_year, self.epoch_day_of_year),
            bstar: self.bstar,
            mean_motion_dot: self.mean_motion_dot,
            mean_motion_double_dot: self.mean_motion_double_dot,
            eccentricity: self.eccentricity,
            argument_of_perigee_deg: self.arg_perigee_deg,
            inclination_deg: self.inclination_deg,
            mean_anomaly_deg: self.mean_anomaly_deg,
            mean_motion_rev_per_day: self.mean_motion,
            right_ascension_deg: self.raan_deg,
            catalog_number: self.catalog_number.trim().parse().unwrap_or(0),
        })
    }
}

/// A reported checksum discrepancy. The format grammar does not reject a line on
/// a bad checksum (it is advisory), so this is surfaced for the host to log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChecksumWarning {
    /// Human label for the offending line (`"line 1"` / `"line 2"`).
    pub line_label: &'static str,
    /// Checksum digit found in column 69.
    pub expected: u8,
    /// Checksum computed from columns 1-68.
    pub computed: u8,
}

/// The result of [`parse`]: the elements plus any advisory checksum warnings.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedTle {
    pub elements: TleElements,
    pub checksum_warnings: Vec<ChecksumWarning>,
}

/// Failure modes of [`parse`]. Messages mirror the historical reference strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TleError {
    NonAscii,
    Format,
    SatelliteMismatch,
    InvalidField {
        field: &'static str,
        reason: &'static str,
    },
    Field(String),
}

impl fmt::Display for TleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TleError::NonAscii => write!(f, "TLE lines contain non-ASCII characters"),
            TleError::Format => write!(
                f,
                "TLE format error: line does not match the Two-Line Element fixed-width format"
            ),
            TleError::SatelliteMismatch => {
                write!(f, "Satellite numbers in lines 1 and 2 do not match")
            }
            TleError::InvalidField { field, reason } => {
                write!(f, "TLE invalid field {field}: {reason}")
            }
            TleError::Field(msg) => write!(f, "TLE parse error: {msg}"),
        }
    }
}

impl std::error::Error for TleError {}

fn validate_tle_bridge(elements: &TleElements) -> Result<(), TleError> {
    validate::finite(elements.epoch_day_of_year, "epoch_day_of_year").map_err(map_tle_field)?;
    validate::finite(elements.bstar, "bstar").map_err(map_tle_field)?;
    validate::finite(elements.mean_motion_dot, "mean_motion_dot").map_err(map_tle_field)?;
    validate::finite(elements.mean_motion_double_dot, "mean_motion_double_dot")
        .map_err(map_tle_field)?;
    validate::finite_in_range_exclusive_upper(elements.eccentricity, 0.0, 1.0, "eccentricity")
        .map_err(map_tle_field)?;
    validate::finite(elements.arg_perigee_deg, "arg_perigee_deg").map_err(map_tle_field)?;
    validate::finite(elements.inclination_deg, "inclination_deg").map_err(map_tle_field)?;
    validate::finite(elements.mean_anomaly_deg, "mean_anomaly_deg").map_err(map_tle_field)?;
    validate::finite_positive(elements.mean_motion, "mean_motion").map_err(map_tle_field)?;
    validate::finite(elements.raan_deg, "raan_deg").map_err(map_tle_field)?;
    Ok(())
}

fn map_tle_field(error: validate::FieldError) -> TleError {
    TleError::InvalidField {
        field: error.field(),
        reason: error.reason(),
    }
}

/// Parse a two-line element set into [`TleElements`].
///
/// The parser is liberal: trailing content past column 69 is trimmed, leading-dot
/// floats are normalized, and an invalid checksum is reported (via
/// [`ParsedTle::checksum_warnings`]) rather than rejected.
pub fn parse(line1: &str, line2: &str) -> Result<ParsedTle, TleError> {
    if !is_ascii(line1) || !is_ascii(line2) {
        return Err(TleError::NonAscii);
    }

    let line1 = clean_line(line1);
    let line2 = clean_line(line2);

    validate_format(&line1, &line2)?;
    let elements = extract_fields(&line1, &line2)?;
    let checksum_warnings = checksum_warnings(&line1, &line2);

    Ok(ParsedTle {
        elements,
        checksum_warnings,
    })
}

/// Encode [`TleElements`] as the two 69-character TLE lines (with checksums).
///
/// The caller is responsible for supplying normalized field values (defaults
/// applied, widths validated); this function performs the fixed-width formatting,
/// assumed-decimal encoding, and checksum generation.
pub fn encode(el: &TleElements) -> (String, String) {
    let cat = pad_leading(el.catalog_number.trim(), CATALOG_WIDTH);
    let cls = &el.classification;
    let intl = pad_trailing(&el.international_designator, INTL_DESIGNATOR_WIDTH);

    let epoch_two_digit = el.epoch_year.rem_euclid(100);

    let l1_body = format!(
        "1 {cat}{cls} {intl} {epoch} {ndot} {nddot} {bstar} {ephtype} {elnum}",
        epoch = fmt_epoch(epoch_two_digit, el.epoch_day_of_year),
        ndot = fmt_ndot(el.mean_motion_dot),
        nddot = fmt_assumed_decimal(el.mean_motion_double_dot),
        bstar = fmt_assumed_decimal(el.bstar),
        ephtype = el.ephemeris_type,
        elnum = pad_leading(&el.elset_number.to_string(), ELSET_WIDTH),
    );
    let line1 = pad_and_checksum(&l1_body);

    let l2_body = format!(
        "2 {cat} {inclo} {raan} {ecc} {argp} {mo} {mm}{revnum}",
        inclo = fmt_angle(el.inclination_deg),
        raan = fmt_angle(el.raan_deg),
        ecc = fmt_eccentricity(el.eccentricity),
        argp = fmt_angle(el.arg_perigee_deg),
        mo = fmt_angle(el.mean_anomaly_deg),
        mm = fmt_mean_motion(el.mean_motion),
        revnum = pad_leading(&el.rev_number.to_string(), REV_WIDTH),
    );
    let line2 = pad_and_checksum(&l2_body);

    (line1, line2)
}

// -- Parsing internals --

fn is_ascii(line: &str) -> bool {
    line.chars().all(|c| (c as u32) <= MAX_ASCII)
}

/// Trim trailing whitespace and clamp to the significant TLE width.
fn clean_line(line: &str) -> String {
    let trimmed = line.trim_end();
    if trimmed.len() > MAX_LINE_LEN {
        trimmed[..MAX_LINE_LEN].to_string()
    } else {
        trimmed.to_string()
    }
}

fn validate_format(line1: &str, line2: &str) -> Result<(), TleError> {
    validate_line(line1, '1', LINE1_MIN_LEN, &LINE1_POSITIONS)?;
    validate_line(line2, '2', LINE2_MIN_LEN, &LINE2_POSITIONS)?;
    if slice_inclusive(line1, 2, 6) == slice_inclusive(line2, 2, 6) {
        Ok(())
    } else {
        Err(TleError::SatelliteMismatch)
    }
}

fn validate_line(
    line: &str,
    prefix: char,
    min_len: usize,
    positions: &[(usize, char)],
) -> Result<(), TleError> {
    let len = line.chars().count();
    if len < min_len {
        return Err(TleError::Format);
    }
    let mut start = String::with_capacity(2);
    start.push(prefix);
    start.push(' ');
    if !line.starts_with(&start) {
        return Err(TleError::Format);
    }
    if positions
        .iter()
        .all(|&(pos, ch)| char_at(line, pos) == Some(ch))
    {
        Ok(())
    } else {
        Err(TleError::Format)
    }
}

const LINE1_POSITIONS: [(usize, char); 8] = [
    (8, ' '),
    (23, '.'),
    (32, ' '),
    (34, '.'),
    (43, ' '),
    (52, ' '),
    (61, ' '),
    (63, ' '),
];

const LINE2_POSITIONS: [(usize, char); 10] = [
    (7, ' '),
    (11, '.'),
    (16, ' '),
    (20, '.'),
    (25, ' '),
    (33, ' '),
    (37, '.'),
    (42, ' '),
    (46, '.'),
    (51, ' '),
];

fn extract_fields(line1: &str, line2: &str) -> Result<TleElements, TleError> {
    let two_digit_year = parse_int(slice_inclusive(line1, 18, 19).trim())?;
    let epoch_year = if two_digit_year < YEAR_PIVOT {
        2000 + two_digit_year
    } else {
        1900 + two_digit_year
    };

    Ok(TleElements {
        catalog_number: slice_inclusive(line1, 2, 6).trim().to_string(),
        classification: char_at(line1, 7).unwrap_or('U').to_string(),
        international_designator: slice_inclusive(line1, 9, 16).trim_end().to_string(),
        epoch_year,
        epoch_day_of_year: parse_float(slice_inclusive(line1, 20, 31))?,
        mean_motion_dot: parse_float(slice_inclusive(line1, 33, 42))?,
        mean_motion_double_dot: parse_assumed_decimal(line1, 44, 45, 49, 50, 51)?,
        bstar: parse_assumed_decimal(line1, 53, 54, 58, 59, 60)?,
        ephemeris_type: parse_int_or_default(
            char_at(line1, 62)
                .map(|c| c.to_string())
                .unwrap_or_default()
                .trim(),
            0,
        )?,
        elset_number: parse_int_or_default(slice_inclusive(line1, 64, 67).trim(), 0)?,
        inclination_deg: parse_float(slice_inclusive(line2, 8, 15))?,
        raan_deg: parse_float(slice_inclusive(line2, 17, 24))?,
        eccentricity: parse_eccentricity(slice_inclusive(line2, 26, 32))?,
        arg_perigee_deg: parse_float(slice_inclusive(line2, 34, 41))?,
        mean_anomaly_deg: parse_float(slice_inclusive(line2, 43, 50))?,
        mean_motion: parse_float(slice_inclusive(line2, 52, 62))?,
        rev_number: parse_int_or_default(slice_inclusive(line2, 63, 67).trim(), 0)?,
    })
}

/// Parse an "assumed decimal" exponent field: `[sign][mantissa][exp_sign][exp]`,
/// representing `0.<mantissa> * 10^exp`.
fn parse_assumed_decimal(
    line: &str,
    sign_pos: usize,
    mant_start: usize,
    mant_end: usize,
    exp_start: usize,
    exp_end: usize,
) -> Result<f64, TleError> {
    let sign = if char_at(line, sign_pos) == Some('-') {
        -1.0
    } else {
        1.0
    };
    let mantissa_field = format!("0.{}", slice_inclusive(line, mant_start, mant_end));
    let mantissa = parse_float_raw(mantissa_field.trim())?;
    let exp = parse_int(slice_inclusive(line, exp_start, exp_end).trim())?;
    // Decode with `powi` (integer exponent), matching `decode_assumed_decimal_field`
    // and the SGP4 element-set init: the value reaching SGP4 must be the exact
    // `mantissa * 10^exp` product the golden path produces, so the canonical
    // element set built from a parsed TLE drives SGP4 bit-identically.
    Ok(sign * mantissa * 10.0_f64.powi(exp))
}

/// Parse the implicit-leading-`0.` eccentricity field (spaces read as `0`).
fn parse_eccentricity(field: &str) -> Result<f64, TleError> {
    let digits = field.replace(' ', "0");
    parse_float_raw(&format!("0.{digits}"))
}

/// Replicate the reference float normalization: trim, strip a leading `+`, and
/// supply the integer `0` for a leading-dot value before strict float parsing.
fn parse_float(field: &str) -> Result<f64, TleError> {
    let trimmed = field.trim();
    let without_plus = trimmed.strip_prefix('+').unwrap_or(trimmed);
    let normalized = if let Some(rest) = without_plus.strip_prefix("-.") {
        format!("-0.{rest}")
    } else if let Some(rest) = without_plus.strip_prefix('.') {
        format!("0.{rest}")
    } else {
        without_plus.to_string()
    };
    parse_float_raw(&normalized)
}

/// Strict float parse that rejects the integer-only and leading/trailing-dot forms
/// the reference `String.to_float/1` rejects, so malformed fields surface as errors.
fn parse_float_raw(text: &str) -> Result<f64, TleError> {
    if !text.contains('.') {
        return Err(TleError::Field(format!("invalid float {text:?}")));
    }
    let body = text.strip_prefix('-').unwrap_or(text);
    if body.starts_with('.') || body.ends_with('.') {
        return Err(TleError::Field(format!("invalid float {text:?}")));
    }
    text.parse::<f64>()
        .map_err(|_| TleError::Field(format!("invalid float {text:?}")))
}

fn parse_int(text: &str) -> Result<i32, TleError> {
    text.parse::<i32>()
        .map_err(|_| TleError::Field(format!("invalid integer {text:?}")))
}

/// Parse an integer field that is optional in practice: a blank (all-spaces)
/// field falls back to `default`. The element-set and revolution numbers are
/// bookkeeping fields some generators leave empty; they do not affect SGP4
/// propagation, so a blank one is a cosmetic absence rather than corruption.
fn parse_int_or_default(text: &str, default: i32) -> Result<i32, TleError> {
    if text.is_empty() {
        Ok(default)
    } else {
        parse_int(text)
    }
}

fn checksum_warnings(line1: &str, line2: &str) -> Vec<ChecksumWarning> {
    [("line 1", line1), ("line 2", line2)]
        .into_iter()
        .filter_map(|(label, line)| check_one(label, line))
        .collect()
}

fn check_one(label: &'static str, line: &str) -> Option<ChecksumWarning> {
    if line.chars().count() < MAX_LINE_LEN {
        return None;
    }
    let expected = char_at(line, CHECKSUM_COL)
        .and_then(|c| c.to_digit(10))
        .map(|d| d as u8)?;
    let computed = compute_checksum(line);
    if expected == computed {
        None
    } else {
        Some(ChecksumWarning {
            line_label: label,
            expected,
            computed,
        })
    }
}

/// Modulo-10 checksum over columns 1-68: digits add their value, `-` adds 1, all
/// other characters add 0.
fn compute_checksum(line: &str) -> u8 {
    let sum: u32 = line
        .chars()
        .take(BODY_LEN)
        .map(|c| match c {
            '0'..='9' => c as u32 - '0' as u32,
            '-' => 1,
            _ => 0,
        })
        .sum();
    (sum % 10) as u8
}

// -- Slicing helpers (TLE lines are ASCII, so char index == byte index) --

/// Inclusive character slice mirroring the reference `String.slice(s, a..b)`:
/// clamps to the available length and returns `""` when `start` is past the end.
fn slice_inclusive(s: &str, start: usize, end_inclusive: usize) -> &str {
    let len = s.len();
    if start >= len {
        return "";
    }
    let end = (end_inclusive + 1).min(len);
    &s[start..end]
}

fn char_at(s: &str, index: usize) -> Option<char> {
    s.as_bytes().get(index).map(|&b| b as char)
}

/// Quantize a value onto the TLE "assumed decimal" grid (five significant
/// mantissa digits and a power-of-ten exponent) and decode it back, yielding the
/// exact `f64` SGP4 receives when the same quantity is carried through a TLE.
///
/// OMM encodes B\* and the second mean-motion derivative as plain decimals, but
/// their canonical SGP4 representation is this five-digit assumed-decimal field;
/// quantizing through it lets an OMM drive SGP4 bit-identically to the equivalent
/// TLE. The decode mirrors the parse in `sgp4::init_satrec_from_tle`
/// (`mantissa * 10f64.powi(exp)`), so a quantized OMM B\* equals the value the
/// matching TLE produces to 0 ULP.
pub(crate) fn assumed_decimal_quantize(value: f64) -> f64 {
    if value == 0.0 {
        return 0.0;
    }
    decode_assumed_decimal_field(&fmt_assumed_decimal(value))
}

/// Decode the eight-or-more character assumed-decimal field emitted by
/// [`fmt_assumed_decimal`] (`"[sign|space]MMMMM[exp-sign]E"`).
fn decode_assumed_decimal_field(field: &str) -> f64 {
    let sign = if field.starts_with('-') { -1.0 } else { 1.0 };
    let body = &field[1..];
    let mantissa_digits = &body[..ASSUMED_DECIMAL_MANTISSA_DIGITS];
    let exp_field = &body[ASSUMED_DECIMAL_MANTISSA_DIGITS..];
    let exp_field = exp_field.strip_prefix('+').unwrap_or(exp_field);
    let mantissa: f64 = format!("0.{mantissa_digits}").parse().unwrap_or(0.0);
    let exp: i32 = exp_field.parse().unwrap_or(0);
    sign * mantissa * 10.0_f64.powi(exp)
}

// -- Encoding internals --

fn fmt_epoch(year_two_digit: i32, day_of_year: f64) -> String {
    let yr = pad_leading_zeros(&year_two_digit.to_string(), EPOCH_YEAR_WIDTH);
    let days = fixed_decimals(day_of_year, EPOCH_DAY_DECIMALS);
    format!("{yr}{}", pad_leading_zeros(&days, EPOCH_DAY_WIDTH))
}

fn fmt_ndot(val: f64) -> String {
    let sign = if val < 0.0 { '-' } else { ' ' };
    let mut digits = fixed_decimals(val.abs(), NDOT_DECIMALS);
    if let Some(rest) = digits.strip_prefix('0') {
        digits = rest.to_string();
    }
    format!("{sign}{}", pad_leading(&digits, NDOT_WIDTH))
}

/// Format an "assumed decimal" field (`0.<mantissa> * 10^exp`) for the drag terms.
fn fmt_assumed_decimal(val: f64) -> String {
    if val == 0.0 {
        return " 00000-0".to_string();
    }
    let sign = if val < 0.0 { '-' } else { ' ' };
    let av = val.abs();
    let raw_exp = floor(log10(av)) as i32;
    let mut exp = raw_exp + 1;
    let mantissa = av / pow(10.0, exp as f64);
    let mut mant_full = fixed_decimals(mantissa, ASSUMED_DECIMAL_MANTISSA_DECIMALS);
    if mant_full.starts_with("1.") {
        exp += 1;
        mant_full = fixed_decimals(mantissa / 10.0, ASSUMED_DECIMAL_MANTISSA_DECIMALS);
    }
    let mant_str: String = mant_full
        .chars()
        .skip(2)
        .take(ASSUMED_DECIMAL_MANTISSA_DIGITS)
        .collect();
    let exp_sign = if exp >= 0 { '+' } else { '-' };
    format!("{sign}{mant_str}{exp_sign}{}", exp.abs())
}

fn fmt_eccentricity(ecc: f64) -> String {
    let formatted = fixed_decimals(ecc, ECCENTRICITY_DECIMALS);
    let digits = formatted.strip_prefix("0.").unwrap_or(&formatted);
    pad_leading_zeros(digits, ECCENTRICITY_DIGITS)
}

fn fmt_angle(val: f64) -> String {
    pad_leading(&fixed_decimals(val, ANGLE_DECIMALS), ANGLE_WIDTH)
}

fn fmt_mean_motion(val: f64) -> String {
    pad_leading(
        &fixed_decimals(val, MEAN_MOTION_DECIMALS),
        MEAN_MOTION_WIDTH,
    )
}

fn pad_and_checksum(body: &str) -> String {
    let clamped: String = body.chars().take(BODY_LEN).collect();
    let padded = pad_trailing(&clamped, BODY_LEN);
    let checksum = compute_checksum(&padded);
    format!("{padded}{checksum}")
}

/// Fixed-decimal formatting matching Erlang `float_to_binary/2` `{decimals, n}`
/// (round-half-to-even on the shortest exact decimal expansion).
fn fixed_decimals(value: f64, decimals: usize) -> String {
    format!("{value:.decimals$}")
}

fn pad_leading(s: &str, width: usize) -> String {
    pad_leading_with(s, width, ' ')
}

fn pad_leading_zeros(s: &str, width: usize) -> String {
    pad_leading_with(s, width, '0')
}

fn pad_leading_with(s: &str, width: usize, fill: char) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        let mut out: String = std::iter::repeat_n(fill, width - len).collect();
        out.push_str(s);
        out
    }
}

fn pad_trailing(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        let mut out = s.to_string();
        out.extend(std::iter::repeat_n(' ', width - len));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ISS_L1: &str = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993";
    const ISS_L2: &str = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106";

    #[test]
    fn parses_iss_fields() {
        let parsed = parse(ISS_L1, ISS_L2).unwrap();
        let el = parsed.elements;
        assert_eq!(el.catalog_number, "25544");
        assert_eq!(el.classification, "U");
        assert_eq!(el.international_designator, "98067A");
        assert_eq!(el.epoch_year, 2018);
        assert_eq!(el.epoch_day_of_year, 184.80969102);
        assert_eq!(el.inclination_deg, 51.6414);
        assert_eq!(el.eccentricity, 0.0003435);
        assert_eq!(el.mean_motion, 15.54005638);
        assert_eq!(el.rev_number, 12110);
        assert!(parsed.checksum_warnings.is_empty());
    }

    #[test]
    fn round_trips_iss_character_exact() {
        let parsed = parse(ISS_L1, ISS_L2).unwrap();
        let (l1, l2) = encode(&parsed.elements);
        assert_eq!(l1, ISS_L1);
        assert_eq!(l2, ISS_L2);
    }

    #[test]
    fn low_catalog_numbers_keep_leading_zeros() {
        let l1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753";
        let l2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667";
        let parsed = parse(l1, l2).unwrap();
        assert_eq!(parsed.elements.catalog_number, "00005");
        assert_eq!(parsed.elements.epoch_year, 2000);
    }

    #[test]
    fn rejects_empty_lines() {
        assert!(parse("", "").is_err());
    }

    #[test]
    fn rejects_non_tle_text() {
        assert!(matches!(
            parse("hello world", "goodbye world"),
            Err(TleError::Format)
        ));
    }

    #[test]
    fn rejects_swapped_lines() {
        assert!(parse(ISS_L2, ISS_L1).is_err());
    }

    #[test]
    fn rejects_non_ascii() {
        assert_eq!(
            parse("1 25544\u{fc} test", "2 25544\u{fc} test"),
            Err(TleError::NonAscii)
        );
    }

    #[test]
    fn rejects_mismatched_satellite_numbers() {
        let l1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993";
        let l2 = "2 25545  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106";
        assert_eq!(parse(l1, l2), Err(TleError::SatelliteMismatch));
    }

    #[test]
    fn parses_negative_drag_terms() {
        // Construct a line with a negative bstar and verify sign handling.
        let parsed = parse(ISS_L1, ISS_L2).unwrap();
        assert!(parsed.elements.bstar > 0.0);
        assert_eq!(parsed.elements.mean_motion_double_dot, 0.0);
    }

    #[test]
    fn element_bridge_rejects_invalid_values() {
        let mut el = parse(ISS_L1, ISS_L2).unwrap().elements;
        el.mean_motion = f64::NAN;
        assert_eq!(
            el.to_element_set(),
            Err(TleError::InvalidField {
                field: "mean_motion",
                reason: "not finite"
            })
        );

        let mut el = parse(ISS_L1, ISS_L2).unwrap().elements;
        el.eccentricity = 1.0;
        assert_eq!(
            el.to_element_set(),
            Err(TleError::InvalidField {
                field: "eccentricity",
                reason: "out of range"
            })
        );
    }

    #[test]
    fn assumed_decimal_rounding_carry_bumps_exponent() {
        let mut el = parse(ISS_L1, ISS_L2).unwrap().elements;
        el.mean_motion_double_dot = 9.999996e-5;
        el.bstar = 9.999996e-5;

        let (line1, line2) = encode(&el);
        assert_eq!(slice_inclusive(&line1, 44, 51), " 10000-3");
        assert_eq!(slice_inclusive(&line1, 53, 60), " 10000-3");

        let parsed = parse(&line1, &line2).unwrap().elements;
        assert_eq!(parsed.mean_motion_double_dot, 1.0e-4);
        assert_eq!(parsed.bstar, 1.0e-4);

        let (round_trip_line1, round_trip_line2) = encode(&parsed);
        assert_eq!(round_trip_line1, line1);
        assert_eq!(round_trip_line2, line2);
    }

    #[test]
    fn checksum_mismatch_is_reported_not_rejected() {
        // Flip the final checksum digit of line 1 (9993 -> 9990).
        let bad_l1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9990";
        let parsed = parse(bad_l1, ISS_L2).unwrap();
        assert_eq!(parsed.checksum_warnings.len(), 1);
        assert_eq!(parsed.checksum_warnings[0].line_label, "line 1");
        assert_eq!(parsed.checksum_warnings[0].expected, 0);
        assert_eq!(parsed.checksum_warnings[0].computed, 3);
    }
}
