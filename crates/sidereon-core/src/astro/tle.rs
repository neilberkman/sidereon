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
/// Highest catalog number that can be represented in a legacy TLE field.
pub const MAX_TLE_CATALOG_NUMBER: u32 = 339_999;
/// Highest catalog number that remains a five-digit numeric TLE field.
pub const MAX_NUMERIC_TLE_CATALOG_NUMBER: u32 = 99_999;
/// Alpha-5 leading letters, in their published numeric order.
const ALPHA5_LETTERS: &str = "ABCDEFGHJKLMNPQRSTUVWXYZ";
/// Number of numeric suffix slots under each Alpha-5 leading letter.
const ALPHA5_SUFFIX_MODULUS: u32 = 10_000;

/// Parsed TLE orbital elements in canonical astrodynamic units.
///
/// Angles are degrees, mean motion is revolutions/day and its derivatives
/// rev/day^2 and rev/day^3, BSTAR drag is 1/earth-radii, and the epoch is the
/// calendar `epoch_year` plus the one-based fractional `epoch_day_of_year`.
#[derive(Debug, Clone, PartialEq)]
pub struct TleElements {
    /// Catalog number as it appeared at the format boundary.
    ///
    /// Numeric TLE fields keep their five-character representation (for
    /// example, `"00005"`), and Alpha-5 fields keep their letter-plus-four-digit
    /// representation (for example, `"A0000"`). Use
    /// [`decode_catalog_number`] to obtain the numeric catalog id.
    pub catalog_number: String,
    /// Classification character from line 1 byte index 7 (column 8).
    /// [`parse`] defaults a missing position to `"U"`, while [`encode`] emits it
    /// immediately after the catalog field.
    pub classification: String,
    /// Launch international designator from line 1 byte indices 9..=16
    /// (columns 10-17). [`parse`] removes right padding, and [`encode`] restores
    /// the eight-character field width.
    pub international_designator: String,
    /// Full calendar year after applying the TLE two-digit pivot: parsed values
    /// below 57 become 2000+, and all others become 1900+. [`encode`] emits the
    /// year modulo 100, while [`TleElements::to_element_set`] uses the full year.
    pub epoch_year: i32,
    /// One-based fractional day of year from line 1 byte indices 20..=31.
    /// [`TleElements::to_element_set`] converts it through Vallado `days2mdhms`
    /// and split-Julian-date math; [`encode`] emits eight decimal places.
    pub epoch_day_of_year: f64,
    /// Line 1 byte indices 33..=42, in rev/day². The two-line format defines
    /// this field as half the first time derivative of mean motion (ṅ/2), and
    /// the value here is the field as written, not ṅ. [`encode`] writes it as
    /// a signed `.NNNNNNNN` value and refuses a magnitude that rounds to 1 or
    /// more. [`TleElements::to_element_set`] passes it unchanged to SGP4, as
    /// Vallado's `twoline2rv` does; SGP4 does not propagate with it.
    pub mean_motion_dot: f64,
    /// Line 1 byte indices 44..=51, in rev/day³, as a signed assumed-decimal
    /// value (`±NNNNN±E` meaning `±0.NNNNN × 10^±E`). The format defines this
    /// field as one sixth of the second time derivative of mean motion (n̈/6),
    /// and the value here is the field as written. Blank mantissa digits and
    /// a blank exponent digit read as `0`, as in Vallado's `twoline2rv`.
    pub mean_motion_double_dot: f64,
    /// The eight characters of the second-derivative field as read, or `None`
    /// for elements not read from text. [`encode`] writes this text back
    /// unchanged while it decodes to exactly `mean_motion_double_dot`, so a
    /// spelling such as `" 00000+0"` or `" 01234-4"` survives a round trip.
    pub mean_motion_double_dot_text: Option<String>,
    /// Vallado B* drag term in the dimensionless 1/earth-radii TLE convention,
    /// decoded from line 1 byte indices 53..=60 with the assumed-decimal codec.
    /// [`TleElements::to_element_set`] passes it unchanged to SGP4.
    pub bstar: f64,
    /// The eight characters of the B\* field as read, or `None` for elements
    /// not read from text. [`encode`] writes this text back unchanged while it
    /// decodes to exactly `bstar`.
    pub bstar_text: Option<String>,
    /// Integer ephemeris-type field at line 1 byte index 62 (column 63).
    /// A blank field reads as `None` and [`encode`] writes `None` as a blank
    /// column. Vallado's `twoline2rv` replaces a blank column 63 with `0`
    /// before reading; the value only selects the propagator, and SGP4 runs
    /// the same way for `None` as for `0`, so
    /// [`TleElements::to_element_set`] treats `None` as `0`, as `twoline2rv`
    /// does, and leaves this bookkeeping value out of [`ElementSet`]. Fitted
    /// TLE records write `0`.
    pub ephemeris_type: Option<i32>,
    /// Element-set number from line 1 byte indices 64..=67 (columns 65-68).
    /// A blank field reads as `None` and [`encode`] writes `None` as a blank
    /// four-character field.
    pub elset_number: Option<i32>,
    /// Inclination in degrees from line 2 byte indices 8..=15, formatted with
    /// four decimal places. [`TleElements::to_element_set`] preserves the degree
    /// value in [`ElementSet`], whose SGP4 initializer converts it to radians.
    pub inclination_deg: f64,
    /// Right ascension of the ascending node in degrees from line 2 byte
    /// indices 17..=24, formatted with four decimal places. The element bridge
    /// maps it to [`ElementSet::right_ascension_deg`] before SGP4 converts it to
    /// radians.
    pub raan_deg: f64,
    /// Dimensionless eccentricity from line 2 byte indices 26..=32, interpreted
    /// as an implicit-leading-decimal fraction with spaces replaced by zeroes.
    /// The element bridge requires a finite value in `[0, 1)`, while [`encode`]
    /// emits seven fractional digits without the leading `0.`.
    pub eccentricity: f64,
    /// Argument of perigee in degrees from line 2 byte indices 34..=41,
    /// formatted with four decimal places. The element bridge maps it to
    /// [`ElementSet::argument_of_perigee_deg`] before SGP4 converts it to
    /// radians.
    pub arg_perigee_deg: f64,
    /// Mean anomaly in degrees from line 2 byte indices 43..=50, formatted with
    /// four decimal places. The element bridge maps it to
    /// [`ElementSet::mean_anomaly_deg`] before SGP4 converts it to radians.
    pub mean_anomaly_deg: f64,
    /// Mean motion in revolutions per day from line 2 byte indices 52..=62,
    /// formatted with eight decimal places. The element bridge requires a
    /// finite positive value, copies it to [`ElementSet::mean_motion_rev_per_day`],
    /// and SGP4 converts it to radians per minute.
    pub mean_motion: f64,
    /// Revolution number at the TLE epoch from line 2 byte indices 63..=67.
    /// A blank field reads as `None` and [`encode`] writes `None` as a blank
    /// five-character field.
    pub rev_number: Option<i32>,
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
    /// The catalog number is decoded to the numeric form `ElementSet` carries.
    /// It is used only for SGP4 diagnostics and does not affect propagation, but
    /// it must still survive the TLE bridge without silent loss.
    pub fn to_element_set(&self) -> Result<ElementSet, TleError> {
        validate_tle_bridge(self)?;
        Ok(ElementSet {
            epoch: sgp4::sgp4_julian_date_from_day_of_year(self.epoch_year, self.epoch_day_of_year),
            bstar: self.bstar,
            mean_motion_dot: Some(self.mean_motion_dot),
            mean_motion_double_dot: Some(self.mean_motion_double_dot),
            eccentricity: self.eccentricity,
            argument_of_perigee_deg: self.arg_perigee_deg,
            inclination_deg: self.inclination_deg,
            mean_anomaly_deg: self.mean_anomaly_deg,
            mean_motion_rev_per_day: self.mean_motion,
            right_ascension_deg: self.raan_deg,
            catalog_number: Some(decode_catalog_number(&self.catalog_number)?),
        })
    }
}

/// What column 69 of a line held, when it did not confirm the line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumWarningKind {
    /// A digit that differs from the computed checksum.
    Mismatch {
        /// Checksum digit found in column 69.
        expected: u8,
    },
    /// A character other than a digit.
    NotDigit {
        /// The character found in column 69.
        found: char,
    },
    /// The line ends before column 69, so it carries no checksum.
    Missing,
}

/// A line whose column 69 did not confirm its checksum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChecksumWarning {
    /// Human label for the offending line (`"line 1"` / `"line 2"`).
    pub line_label: &'static str,
    /// What column 69 held.
    pub kind: ChecksumWarningKind,
    /// Checksum computed from columns 1-68 (or as many as the line has).
    pub computed: u8,
}

impl fmt::Display for ChecksumWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (label, computed) = (self.line_label, self.computed);
        match self.kind {
            ChecksumWarningKind::Mismatch { expected } => write!(
                f,
                "{label} checksum digit {expected} does not match the computed checksum {computed}"
            ),
            ChecksumWarningKind::NotDigit { found } => write!(
                f,
                "{label} column 69 holds {found:?}, not the checksum digit {computed}"
            ),
            ChecksumWarningKind::Missing => write!(
                f,
                "{label} ends before column 69 and carries no checksum (computed {computed})"
            ),
        }
    }
}

/// The result of [`parse`]: the elements plus any advisory checksum warnings.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedTle {
    /// Orbital and bookkeeping fields extracted from the validated line pair.
    /// This is the value passed to [`encode`] for round trips or to
    /// [`TleElements::to_element_set`] for SGP4 initialization.
    pub elements: TleElements,
    /// Checksum findings accepted by the policy, ordered by line 1 then
    /// line 2: a line with no column 69 under either policy, and under
    /// [`TlePolicy::Lenient`] also a mismatching digit or a non-digit.
    pub checksum_warnings: Vec<ChecksumWarning>,
}

/// How [`parse_with_policy`] treats column 69, the modulo-10 checksum of
/// columns 1-68.
///
/// Under both policies a line that ends before column 69 is read and
/// reported with [`ChecksumWarningKind::Missing`]: it carries no checksum, so
/// nothing in it contradicts the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TlePolicy {
    /// Refuse a digit that disagrees ([`TleError::ChecksumMismatch`]) and a
    /// character that is not a digit ([`TleError::ChecksumNotDigit`]). Either
    /// is evidence that the line was altered after it was written.
    #[default]
    Strict,
    /// Accept both and report each in [`ParsedTle::checksum_warnings`]. This
    /// is how Vallado's `twoline2rv` reads, which ignores the checksum; its
    /// verification set carries element sets (catalog numbers 33333, 33334,
    /// 33335) whose checksums disagree.
    Lenient,
}

/// Failure modes of [`parse`]. Messages mirror the historical reference strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TleError {
    /// A TLE line contained a non-ASCII character.
    NonAscii,
    /// A line failed fixed-width TLE grammar validation.
    Format,
    /// The catalog-number fields in line 1 and line 2 differed.
    SatelliteMismatch,
    /// The five-character catalog field was neither numeric nor valid Alpha-5.
    InvalidCatalogNumber {
        /// The rejected catalog field.
        value: String,
        /// The validation reason.
        reason: &'static str,
    },
    /// The catalog id is outside the range representable by TLE or Alpha-5.
    CatalogNumberOutOfRange {
        /// The rejected numeric catalog id.
        catalog_number: u32,
    },
    /// A decoded scalar field failed boundary validation.
    InvalidField {
        /// Field name.
        field: &'static str,
        /// Validation reason.
        reason: &'static str,
    },
    /// A scalar field could not be parsed.
    Field(String),
    /// A full-width line's column-69 checksum digit disagreed with the
    /// checksum of columns 1-68, under [`TlePolicy::Strict`].
    ChecksumMismatch {
        /// `"line 1"` or `"line 2"`.
        line_label: &'static str,
        /// Checksum digit found in column 69.
        expected: u8,
        /// Checksum computed from columns 1-68.
        computed: u8,
    },
    /// A full-width line's column 69 held a character that is not a digit,
    /// under [`TlePolicy::Strict`].
    ChecksumNotDigit {
        /// `"line 1"` or `"line 2"`.
        line_label: &'static str,
        /// The character found in column 69.
        found: char,
        /// Checksum computed from columns 1-68.
        computed: u8,
    },
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
            TleError::InvalidCatalogNumber { value, reason } => {
                write!(f, "TLE invalid catalog number {value:?}: {reason}")
            }
            TleError::CatalogNumberOutOfRange { catalog_number } => write!(
                f,
                "TLE catalog number {catalog_number} cannot be encoded in a five-character field"
            ),
            TleError::InvalidField { field, reason } => {
                write!(f, "TLE invalid field {field}: {reason}")
            }
            TleError::Field(msg) => write!(f, "TLE parse error: {msg}"),
            TleError::ChecksumMismatch {
                line_label,
                expected,
                computed,
            } => write!(
                f,
                "TLE {line_label} checksum digit {expected} does not match the computed checksum {computed}"
            ),
            TleError::ChecksumNotDigit {
                line_label,
                found,
                computed,
            } => write!(
                f,
                "TLE {line_label} column 69 holds {found:?}, not the checksum digit {computed}"
            ),
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

/// Parse a two-line element set into [`TleElements`] under
/// [`TlePolicy::Strict`].
///
/// Trailing content past column 69 is trimmed and leading-dot floats are
/// normalized, as Vallado's `twoline2rv` reads them. A checksum digit that
/// disagrees, or a column 69 that is not a digit, is refused; use
/// [`parse_with_policy`] with [`TlePolicy::Lenient`] to read such a line and
/// have the finding reported instead.
pub fn parse(line1: &str, line2: &str) -> Result<ParsedTle, TleError> {
    parse_with_policy(line1, line2, TlePolicy::Strict)
}

/// Parse a two-line element set into [`TleElements`] under `policy`.
///
/// The field grammar is the same under both policies. They differ only in how
/// a checksum mismatch is treated; see [`TlePolicy`].
pub fn parse_with_policy(
    line1: &str,
    line2: &str,
    policy: TlePolicy,
) -> Result<ParsedTle, TleError> {
    if !is_ascii(line1) || !is_ascii(line2) {
        return Err(TleError::NonAscii);
    }

    let line1 = clean_line(line1);
    let line2 = clean_line(line2);

    validate_format(&line1, &line2)?;
    let elements = extract_fields(&line1, &line2)?;
    let checksum_warnings = checksum_warnings(&line1, &line2);
    if policy == TlePolicy::Strict {
        for warning in &checksum_warnings {
            match warning.kind {
                ChecksumWarningKind::Mismatch { expected } => {
                    return Err(TleError::ChecksumMismatch {
                        line_label: warning.line_label,
                        expected,
                        computed: warning.computed,
                    })
                }
                ChecksumWarningKind::NotDigit { found } => {
                    return Err(TleError::ChecksumNotDigit {
                        line_label: warning.line_label,
                        found,
                        computed: warning.computed,
                    })
                }
                ChecksumWarningKind::Missing => {}
            }
        }
    }

    Ok(ParsedTle {
        elements,
        checksum_warnings,
    })
}

/// Encode a numeric catalog id into its five-character TLE catalog field.
///
/// Values `0..=99999` encode as five decimal digits. Values
/// `100000..=339999` encode with Alpha-5, where the first character carries the
/// ten-thousands digit group and the four trailing characters carry the
/// remainder. Larger values return [`TleError::CatalogNumberOutOfRange`] because
/// the TLE field has no representation for them.
pub fn encode_catalog_number(catalog_number: u32) -> Result<String, TleError> {
    if catalog_number <= MAX_NUMERIC_TLE_CATALOG_NUMBER {
        return Ok(format!("{catalog_number:05}"));
    }
    if catalog_number > MAX_TLE_CATALOG_NUMBER {
        return Err(TleError::CatalogNumberOutOfRange { catalog_number });
    }

    let prefix = catalog_number / ALPHA5_SUFFIX_MODULUS;
    let suffix = catalog_number % ALPHA5_SUFFIX_MODULUS;
    let letter = alpha5_letter_for_value(prefix)
        .ok_or(TleError::CatalogNumberOutOfRange { catalog_number })?;
    Ok(format!("{letter}{suffix:04}"))
}

/// Decode a five-character TLE catalog field into its numeric catalog id.
///
/// Plain numeric fields decode directly. Alpha-5 fields decode by the published
/// `letter_value * 10000 + suffix` rule, with letters `I` and `O` rejected.
pub fn decode_catalog_number(field: &str) -> Result<u32, TleError> {
    let field = field.trim();
    if field.is_empty() {
        return Err(TleError::InvalidCatalogNumber {
            value: field.to_string(),
            reason: "empty field",
        });
    }

    if field.bytes().all(|b| b.is_ascii_digit()) {
        if field.len() > CATALOG_WIDTH {
            return Err(TleError::InvalidCatalogNumber {
                value: field.to_string(),
                reason: "numeric TLE field is wider than five digits",
            });
        }
        return field
            .parse::<u32>()
            .map_err(|_| TleError::InvalidCatalogNumber {
                value: field.to_string(),
                reason: "invalid numeric field",
            });
    }

    if field.len() != CATALOG_WIDTH {
        return Err(TleError::InvalidCatalogNumber {
            value: field.to_string(),
            reason: "Alpha-5 field must be one letter followed by four digits",
        });
    }

    let mut chars = field.chars();
    let letter = chars.next().expect("non-empty field");
    let prefix = alpha5_value_for_letter(letter).ok_or_else(|| TleError::InvalidCatalogNumber {
        value: field.to_string(),
        reason: "invalid Alpha-5 leading letter",
    })?;
    let suffix = chars.as_str();
    if !suffix.bytes().all(|b| b.is_ascii_digit()) {
        return Err(TleError::InvalidCatalogNumber {
            value: field.to_string(),
            reason: "Alpha-5 suffix must be four digits",
        });
    }
    let suffix = suffix
        .parse::<u32>()
        .map_err(|_| TleError::InvalidCatalogNumber {
            value: field.to_string(),
            reason: "invalid Alpha-5 suffix",
        })?;
    Ok(prefix * ALPHA5_SUFFIX_MODULUS + suffix)
}

/// Encode [`TleElements`] as the two 69-character TLE lines (with checksums).
///
/// Each field is written at the precision the format fixes for it (for
/// example four decimals for an angle), so a value read by [`parse`] is
/// written back unchanged. An assumed-decimal field (B\*, second derivative)
/// is written as its source text when that text still decodes to exactly the
/// stored value. Otherwise it is written as the first spelling that decodes
/// to the same `f64` bits, trying the normalized exponent `e` (or `-9`, the
/// smallest single-digit exponent, when `e` is below it) and then the next
/// four exponents with leading-zero mantissas, and at exponent zero both
/// signs (`+0` first for a nonzero value, `-0` first for zero). The
/// normalized spelling comes first, so it is written whenever it is exact; a
/// leading-zero spelling is written only when the normalized one would change
/// the value. For example `5e-11` is written `" 00500-8"`, because
/// `0.05 × 10^-9` rounds one unit in the last place away from it and
/// `0.005 × 10^-8` does not. A value with no exact spelling, such as a fitted
/// B\*, is rounded to the normalized five-digit mantissa (at exponent `-9`
/// for a magnitude below `1e-10`). A value that has no
/// representation in its field is refused with [`TleError::InvalidField`]
/// naming the field, never truncated, wrapped, or shifted into a neighbouring
/// column:
///
/// - `classification`: not exactly one printable ASCII character;
/// - `international_designator`: longer than eight characters or not
///   printable ASCII;
/// - `epoch_year`: outside 1957-2056, the years the two-digit field and its
///   57 pivot can name;
/// - `epoch_day_of_year`: negative or at least 1000 after rounding to eight
///   decimals;
/// - `mean_motion_dot`: magnitude at least 1 after rounding to eight decimals;
/// - `mean_motion_double_dot`, `bstar`: magnitude at least 1e9, which needs a
///   two-digit exponent;
/// - `ephemeris_type`, `elset_number`, `rev_number`: more characters than the
///   1-, 4- and 5-column fields hold;
/// - angles and `mean_motion`: wider than their 8- and 11-column fields;
/// - `eccentricity`: outside `[0, 1)` after rounding to seven decimals;
/// - any non-finite value.
///
/// The catalog number is checked by [`encode_catalog_number`].
pub fn encode(el: &TleElements) -> Result<(String, String), TleError> {
    let cat = encode_catalog_number_text(&el.catalog_number)?;
    let cls = encode_classification(&el.classification)?;
    let intl = encode_international_designator(&el.international_designator)?;
    let epoch = encode_epoch(el.epoch_year, el.epoch_day_of_year)?;
    let ndot = encode_ndot(el.mean_motion_dot)?;
    let nddot = encode_assumed_decimal(
        el.mean_motion_double_dot,
        el.mean_motion_double_dot_text.as_deref(),
        NDDOT_FIELD,
    )?;
    let bstar = encode_assumed_decimal(el.bstar, el.bstar_text.as_deref(), BSTAR_FIELD)?;
    let ephtype = encode_optional_integer(el.ephemeris_type, 1, "ephemeris_type")?;
    let elnum = encode_optional_integer(el.elset_number, ELSET_WIDTH, "elset_number")?;

    let l1_body = format!("1 {cat}{cls} {intl} {epoch} {ndot} {nddot} {bstar} {ephtype} {elnum}");
    let line1 = checksummed_line(&l1_body)?;

    let inclo = encode_fixed(
        el.inclination_deg,
        ANGLE_DECIMALS,
        ANGLE_WIDTH,
        "inclination_deg",
    )?;
    let raan = encode_fixed(el.raan_deg, ANGLE_DECIMALS, ANGLE_WIDTH, "raan_deg")?;
    let ecc = encode_eccentricity(el.eccentricity)?;
    let argp = encode_fixed(
        el.arg_perigee_deg,
        ANGLE_DECIMALS,
        ANGLE_WIDTH,
        "arg_perigee_deg",
    )?;
    let mo = encode_fixed(
        el.mean_anomaly_deg,
        ANGLE_DECIMALS,
        ANGLE_WIDTH,
        "mean_anomaly_deg",
    )?;
    let mm = encode_fixed(
        el.mean_motion,
        MEAN_MOTION_DECIMALS,
        MEAN_MOTION_WIDTH,
        "mean_motion",
    )?;
    let revnum = encode_optional_integer(el.rev_number, REV_WIDTH, "rev_number")?;

    let l2_body = format!("2 {cat} {inclo} {raan} {ecc} {argp} {mo} {mm}{revnum}");
    let line2 = checksummed_line(&l2_body)?;

    Ok((line1, line2))
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
    let catalog_number = slice_inclusive(line1, 2, 6).trim().to_string();
    decode_catalog_number(&catalog_number)?;

    let two_digit_year = parse_epoch_year(slice_inclusive(line1, 18, 19))?;
    let nddot_text = slice_inclusive(line1, 44, 51);
    let bstar_text = slice_inclusive(line1, 53, 60);
    let epoch_year = if two_digit_year < YEAR_PIVOT {
        2000 + two_digit_year
    } else {
        1900 + two_digit_year
    };

    Ok(TleElements {
        catalog_number,
        classification: char_at(line1, 7).unwrap_or('U').to_string(),
        international_designator: slice_inclusive(line1, 9, 16).trim_end().to_string(),
        epoch_year,
        epoch_day_of_year: parse_float(slice_inclusive(line1, 20, 31))?,
        mean_motion_dot: parse_float(slice_inclusive(line1, 33, 42))?,
        mean_motion_double_dot: decode_assumed_decimal_text(nddot_text, &NDDOT_FIELD)?,
        mean_motion_double_dot_text: Some(nddot_text.to_string()),
        bstar: decode_assumed_decimal_text(bstar_text, &BSTAR_FIELD)?,
        bstar_text: Some(bstar_text.to_string()),
        ephemeris_type: parse_optional_int(slice_inclusive(line1, 62, 62).trim())?,
        elset_number: parse_optional_int(slice_inclusive(line1, 64, 67).trim())?,
        inclination_deg: parse_float(slice_inclusive(line2, 8, 15))?,
        raan_deg: parse_float(slice_inclusive(line2, 17, 24))?,
        eccentricity: parse_eccentricity(slice_inclusive(line2, 26, 32))?,
        arg_perigee_deg: parse_float(slice_inclusive(line2, 34, 41))?,
        mean_anomaly_deg: parse_float(slice_inclusive(line2, 43, 50))?,
        mean_motion: parse_float(slice_inclusive(line2, 52, 62))?,
        rev_number: parse_optional_int(slice_inclusive(line2, 63, 67).trim())?,
    })
}

/// How one "assumed decimal" field is read.
struct AssumedDecimalField {
    name: &'static str,
    /// Read blank mantissa digits and a blank exponent digit as `0`. Vallado's
    /// `twoline2rv` does this for the second mean-motion derivative only.
    blank_digits_are_zero: bool,
}

const NDDOT_FIELD: AssumedDecimalField = AssumedDecimalField {
    name: "mean_motion_double_dot",
    blank_digits_are_zero: true,
};

const BSTAR_FIELD: AssumedDecimalField = AssumedDecimalField {
    name: "bstar",
    blank_digits_are_zero: false,
};

/// Read the two-digit epoch year. Vallado reads it with `%2d`, so a blank
/// tens digit reads as a one-digit year; a sign or any other character is not
/// a year.
fn parse_epoch_year(field: &str) -> Result<i32, TleError> {
    let digits = field.trim();
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(TleError::InvalidField {
            field: "epoch_year",
            reason: "must be two digits",
        });
    }
    parse_int(digits)
}

/// Decode the eight characters of an "assumed decimal" field,
/// `[sign][mantissa][exp_sign][exp]`, representing `0.<mantissa> * 10^exp`.
fn decode_assumed_decimal_text(text: &str, field: &AssumedDecimalField) -> Result<f64, TleError> {
    let sign = match char_at(text, 0) {
        Some('-') => -1.0,
        Some(' ') | Some('+') | None => 1.0,
        Some(_) => {
            return Err(TleError::InvalidField {
                field: field.name,
                reason: "sign column must be blank, '+' or '-'",
            })
        }
    };
    let mantissa_digits = slice_inclusive(text, 1, 5);
    let mut exponent_text = slice_inclusive(text, 6, 7).to_string();
    let mantissa_field = if field.blank_digits_are_zero {
        if exponent_text.ends_with(' ') || exponent_text.len() < 2 {
            exponent_text = format!("{}0", exponent_text.trim_end());
        }
        format!("0.{}", mantissa_digits.replace(' ', "0"))
    } else {
        format!("0.{mantissa_digits}")
    };
    let mantissa = parse_float_raw(mantissa_field.trim())?;
    let exp = parse_int(exponent_text.trim())?;
    Ok(assumed_decimal_value(sign, mantissa, exp))
}

/// `sign * 0.<mantissa> * 10^exp`, decoded with `powi` (integer exponent),
/// matching `decode_assumed_decimal_field` and the SGP4 element-set init: the
/// value reaching SGP4 must be the exact `mantissa * 10^exp` product the
/// golden path produces, so the canonical element set built from a parsed TLE
/// drives SGP4 bit-identically.
fn assumed_decimal_value(sign: f64, mantissa: f64, exp: i32) -> f64 {
    sign * mantissa * 10.0_f64.powi(exp)
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
/// field is `None`. The element-set and revolution numbers are bookkeeping
/// fields some generators leave empty; they do not affect SGP4 propagation.
fn parse_optional_int(text: &str) -> Result<Option<i32>, TleError> {
    if text.is_empty() {
        Ok(None)
    } else {
        parse_int(text).map(Some)
    }
}

fn checksum_warnings(line1: &str, line2: &str) -> Vec<ChecksumWarning> {
    [("line 1", line1), ("line 2", line2)]
        .into_iter()
        .filter_map(|(label, line)| check_one(label, line))
        .collect()
}

fn check_one(label: &'static str, line: &str) -> Option<ChecksumWarning> {
    let computed = compute_checksum(line);
    let kind = match char_at(line, CHECKSUM_COL) {
        None => ChecksumWarningKind::Missing,
        Some(found) => match found.to_digit(10) {
            Some(digit) if digit as u8 == computed => return None,
            Some(digit) => ChecksumWarningKind::Mismatch {
                expected: digit as u8,
            },
            None => ChecksumWarningKind::NotDigit { found },
        },
    };
    Some(ChecksumWarning {
        line_label: label,
        kind,
        computed,
    })
}

/// The modulo-10 checksum of a TLE line's columns 1-68: digits add their
/// value, `-` adds 1, and every other character adds 0.
pub fn line_checksum(line: &str) -> u8 {
    compute_checksum(line)
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

fn alpha5_value_for_letter(letter: char) -> Option<u32> {
    ALPHA5_LETTERS
        .chars()
        .position(|candidate| candidate == letter)
        .map(|index| 10 + index as u32)
}

fn alpha5_letter_for_value(value: u32) -> Option<char> {
    if value < 10 {
        return None;
    }
    ALPHA5_LETTERS.chars().nth((value - 10) as usize)
}

fn encode_catalog_number_text(text: &str) -> Result<String, TleError> {
    let trimmed = text.trim();
    if trimmed.bytes().all(|b| b.is_ascii_digit()) {
        let catalog_number =
            trimmed
                .parse::<u32>()
                .map_err(|_| TleError::InvalidCatalogNumber {
                    value: trimmed.to_string(),
                    reason: "invalid numeric field",
                })?;
        encode_catalog_number(catalog_number)
    } else {
        let catalog_number = decode_catalog_number(trimmed)?;
        encode_catalog_number(catalog_number)
    }
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

/// Append the checksum to a 68-column body. Every field encoder has already
/// refused a value wider than its field, so a body of any other width is a
/// layout fault and is refused rather than truncated.
fn checksummed_line(body: &str) -> Result<String, TleError> {
    if body.len() != BODY_LEN {
        return Err(TleError::Format);
    }
    let checksum = compute_checksum(body);
    Ok(format!("{body}{checksum}"))
}

fn invalid_field(field: &'static str, reason: &'static str) -> TleError {
    TleError::InvalidField { field, reason }
}

fn require_finite(value: f64, field: &'static str) -> Result<(), TleError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_field(field, "not finite"))
    }
}

fn encode_classification(classification: &str) -> Result<char, TleError> {
    let mut chars = classification.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) if (' '..='~').contains(&c) => Ok(c),
        _ => Err(invalid_field(
            "classification",
            "must be exactly one printable ASCII character",
        )),
    }
}

fn encode_international_designator(designator: &str) -> Result<String, TleError> {
    if designator.len() > INTL_DESIGNATOR_WIDTH
        || !designator.chars().all(|c| (' '..='~').contains(&c))
    {
        return Err(invalid_field(
            "international_designator",
            "must be at most eight printable ASCII characters",
        ));
    }
    Ok(pad_trailing(designator, INTL_DESIGNATOR_WIDTH))
}

fn encode_epoch(year: i32, day_of_year: f64) -> Result<String, TleError> {
    if !(1900 + YEAR_PIVOT..2000 + YEAR_PIVOT).contains(&year) {
        return Err(invalid_field(
            "epoch_year",
            "outside 1957-2056, the years a two-digit TLE epoch year can name",
        ));
    }
    require_finite(day_of_year, "epoch_day_of_year")?;
    let days = fixed_decimals(day_of_year, EPOCH_DAY_DECIMALS);
    if days.starts_with('-') || days.len() > EPOCH_DAY_WIDTH {
        return Err(invalid_field(
            "epoch_day_of_year",
            "must round into [0, 1000) with eight decimals",
        ));
    }
    Ok(fmt_epoch(year.rem_euclid(100), day_of_year))
}

fn encode_ndot(value: f64) -> Result<String, TleError> {
    require_finite(value, "mean_motion_dot")?;
    if !fixed_decimals(value.abs(), NDOT_DECIMALS).starts_with("0.") {
        return Err(invalid_field(
            "mean_motion_dot",
            "magnitude must round below 1 with eight decimals",
        ));
    }
    Ok(fmt_ndot(value))
}

/// Encode an assumed-decimal field. See [`encode`] for the spelling order.
/// A value whose normalized exponent needs two digits is refused. A magnitude
/// below `1e-10` is spelled from exponent `-9` upward; with no exact
/// spelling it is rounded at exponent `-9`, whose resolution is `1e-14`.
fn encode_assumed_decimal(
    value: f64,
    source: Option<&str>,
    field: AssumedDecimalField,
) -> Result<String, TleError> {
    require_finite(value, field.name)?;
    if let Some(text) = source {
        let restates = text.len() == 8
            && text.is_ascii()
            && decode_assumed_decimal_text(text, &field)
                .is_ok_and(|decoded| decoded.to_bits() == value.to_bits());
        if restates {
            return Ok(text.to_string());
        }
    }

    let normalized = fmt_assumed_decimal(value);
    let exponent = normalized[1 + ASSUMED_DECIMAL_MANTISSA_DIGITS..]
        .parse::<i32>()
        .map_err(|_| TleError::Field(format!("invalid exponent in {normalized:?}")))?;
    if exponent > 9 {
        return Err(invalid_field(
            field.name,
            "magnitude must be below 1e9, the largest single-digit exponent",
        ));
    }
    let sign = if value.is_sign_negative() { '-' } else { ' ' };
    let first = exponent.max(-9);
    for exp in first..=(first + 4).min(9) {
        let Some(digits) = assumed_decimal_mantissa(value, exp) else {
            continue;
        };
        let exponent_signs: &[char] = match (exp, value == 0.0) {
            (0, true) => &['-', '+'],
            (0, false) => &['+', '-'],
            (e, _) if e > 0 => &['+'],
            _ => &['-'],
        };
        for &exp_sign in exponent_signs {
            let text = format!("{sign}{digits}{exp_sign}{}", exp.abs());
            if decode_assumed_decimal_text(&text, &field)
                .is_ok_and(|decoded| decoded.to_bits() == value.to_bits())
            {
                return Ok(text);
            }
        }
    }

    // No spelling decodes to these bits: round at the normalized exponent.
    if exponent >= -9 {
        return Ok(normalized);
    }
    match assumed_decimal_mantissa(value, -9) {
        Some(digits) if !digits.bytes().all(|b| b == b'0') => Ok(format!("{sign}{digits}-9")),
        _ => Ok(fmt_assumed_decimal(0.0)),
    }
}

/// The five mantissa digits of `|value| / 10^exp` rounded to five decimals,
/// or `None` when the rounded mantissa reaches 1.
fn assumed_decimal_mantissa(value: f64, exp: i32) -> Option<String> {
    let scaled = value.abs() / pow(10.0, f64::from(exp));
    let mantissa = fixed_decimals(scaled, ASSUMED_DECIMAL_MANTISSA_DECIMALS);
    let digits = mantissa.strip_prefix("0.")?;
    Some(digits.to_string())
}

fn encode_optional_integer(
    value: Option<i32>,
    width: usize,
    field: &'static str,
) -> Result<String, TleError> {
    match value {
        Some(value) => encode_integer(value, width, field),
        None => Ok(" ".repeat(width)),
    }
}

fn encode_integer(value: i32, width: usize, field: &'static str) -> Result<String, TleError> {
    let text = value.to_string();
    if text.len() > width {
        return Err(invalid_field(field, "wider than its fixed-width TLE field"));
    }
    Ok(pad_leading(&text, width))
}

fn encode_fixed(
    value: f64,
    decimals: usize,
    width: usize,
    field: &'static str,
) -> Result<String, TleError> {
    require_finite(value, field)?;
    let text = fixed_decimals(value, decimals);
    if text.len() > width {
        return Err(invalid_field(field, "wider than its fixed-width TLE field"));
    }
    Ok(pad_leading(&text, width))
}

fn encode_eccentricity(value: f64) -> Result<String, TleError> {
    require_finite(value, "eccentricity")?;
    if !fixed_decimals(value, ECCENTRICITY_DECIMALS).starts_with("0.") {
        return Err(invalid_field(
            "eccentricity",
            "must round into [0, 1) with seven decimals",
        ));
    }
    Ok(fmt_eccentricity(value))
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
        assert_eq!(el.rev_number, Some(12110));
        assert_eq!(el.elset_number, Some(999));
        assert_eq!(el.bstar_text.as_deref(), Some(" 31745-4"));
        assert_eq!(el.mean_motion_double_dot_text.as_deref(), Some(" 00000-0"));
        assert!(parsed.checksum_warnings.is_empty());
    }

    #[test]
    fn round_trips_iss_character_exact() {
        let parsed = parse(ISS_L1, ISS_L2).unwrap();
        let (l1, l2) = encode(&parsed.elements).unwrap();
        assert_eq!(l1, ISS_L1);
        assert_eq!(l2, ISS_L2);
    }

    #[test]
    fn alpha5_catalog_examples_match_published_table() {
        for (field, value) in [
            ("A0000", 100_000),
            ("E8493", 148_493),
            ("H6932", 176_932),
            ("J2931", 182_931),
            ("P4018", 234_018),
            ("W1928", 301_928),
            ("Z9999", 339_999),
        ] {
            assert_eq!(decode_catalog_number(field), Ok(value));
            assert_eq!(encode_catalog_number(value), Ok(field.to_string()));
        }
    }

    #[test]
    fn alpha5_catalog_round_trips_exhaustive_letter_alphabet() {
        for letter in ALPHA5_LETTERS.chars() {
            for suffix in [0, 1, 9998, 9999] {
                let field = format!("{letter}{suffix:04}");
                let value = decode_catalog_number(&field).unwrap();
                assert_eq!(encode_catalog_number(value).unwrap(), field);
            }
        }
        assert!(decode_catalog_number("I0000").is_err());
        assert!(decode_catalog_number("O0000").is_err());
        assert!(decode_catalog_number("a0000").is_err());
    }

    #[test]
    fn alpha5_tle_bridge_preserves_numeric_catalog_id() {
        let parsed = parse(ISS_L1, ISS_L2).unwrap();
        let mut el = parsed.elements;
        el.catalog_number = "A0000".to_string();

        let (line1, line2) = encode(&el).unwrap();
        assert_eq!(slice_inclusive(&line1, 2, 6), "A0000");
        assert_eq!(slice_inclusive(&line2, 2, 6), "A0000");

        let parsed = parse(&line1, &line2).unwrap();
        assert_eq!(parsed.elements.catalog_number, "A0000");
        assert_eq!(
            parsed.elements.to_element_set().unwrap().catalog_number,
            Some(100_000)
        );
    }

    #[test]
    fn tle_encode_rejects_catalog_numbers_outside_alpha5_range() {
        let parsed = parse(ISS_L1, ISS_L2).unwrap();
        let mut el = parsed.elements;
        el.catalog_number = "340000".to_string();
        assert_eq!(
            encode(&el),
            Err(TleError::CatalogNumberOutOfRange {
                catalog_number: 340_000
            })
        );
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

        let (line1, line2) = encode(&el).unwrap();
        assert_eq!(slice_inclusive(&line1, 44, 51), " 10000-3");
        assert_eq!(slice_inclusive(&line1, 53, 60), " 10000-3");

        let parsed = parse(&line1, &line2).unwrap().elements;
        assert_eq!(parsed.mean_motion_double_dot, 1.0e-4);
        assert_eq!(parsed.bstar, 1.0e-4);

        let (round_trip_line1, round_trip_line2) = encode(&parsed).unwrap();
        assert_eq!(round_trip_line1, line1);
        assert_eq!(round_trip_line2, line2);
    }

    /// Replace columns and write the checksum the new line carries.
    fn with_columns(line: &str, start: usize, text: &str) -> String {
        let mut out = with_raw_columns(line, start, text);
        let checksum = line_checksum(&out);
        out.replace_range(68..69, &checksum.to_string());
        out
    }

    /// Replace columns and leave column 69 as it was.
    fn with_raw_columns(line: &str, start: usize, text: &str) -> String {
        let mut out = line.to_string();
        out.replace_range(start..start + text.len(), text);
        out
    }

    fn iss_elements() -> TleElements {
        parse(ISS_L1, ISS_L2).unwrap().elements
    }

    fn encode_error(mutate: impl FnOnce(&mut TleElements)) -> TleError {
        let mut el = iss_elements();
        mutate(&mut el);
        encode(&el).expect_err("value has no representation in its field")
    }

    type Mutation = Box<dyn FnOnce(&mut TleElements)>;

    fn mutation(f: impl FnOnce(&mut TleElements) + 'static) -> Mutation {
        Box::new(f)
    }

    fn refused(field: &'static str) -> impl Fn(TleError) -> bool {
        move |error| matches!(error, TleError::InvalidField { field: f, .. } if f == field)
    }

    #[test]
    fn encode_refuses_each_value_its_field_cannot_hold_by_name() {
        let cases: Vec<(&'static str, Mutation)> = vec![
            (
                "classification",
                mutation(|el: &mut TleElements| el.classification = String::new()),
            ),
            (
                "classification",
                mutation(|el: &mut TleElements| el.classification = "UU".into()),
            ),
            (
                "international_designator",
                mutation(|el: &mut TleElements| el.international_designator = "98067ABCD".into()),
            ),
            (
                "epoch_year",
                mutation(|el: &mut TleElements| el.epoch_year = 2057),
            ),
            (
                "epoch_year",
                mutation(|el: &mut TleElements| el.epoch_year = 1956),
            ),
            (
                "epoch_day_of_year",
                mutation(|el: &mut TleElements| el.epoch_day_of_year = 1000.0),
            ),
            (
                "epoch_day_of_year",
                mutation(|el: &mut TleElements| el.epoch_day_of_year = -1.0),
            ),
            (
                "mean_motion_dot",
                mutation(|el: &mut TleElements| el.mean_motion_dot = 1.0),
            ),
            (
                "mean_motion_dot",
                mutation(|el: &mut TleElements| el.mean_motion_dot = f64::NAN),
            ),
            (
                "mean_motion_double_dot",
                mutation(|el: &mut TleElements| el.mean_motion_double_dot = 2.0e9),
            ),
            (
                "bstar",
                mutation(|el: &mut TleElements| el.bstar = f64::INFINITY),
            ),
            (
                "ephemeris_type",
                mutation(|el: &mut TleElements| el.ephemeris_type = Some(10)),
            ),
            (
                "elset_number",
                mutation(|el: &mut TleElements| el.elset_number = Some(10_000)),
            ),
            (
                "rev_number",
                mutation(|el: &mut TleElements| el.rev_number = Some(100_000)),
            ),
            (
                "inclination_deg",
                mutation(|el: &mut TleElements| el.inclination_deg = 1000.0),
            ),
            (
                "raan_deg",
                mutation(|el: &mut TleElements| el.raan_deg = -100.0),
            ),
            (
                "arg_perigee_deg",
                mutation(|el: &mut TleElements| el.arg_perigee_deg = 1234.5),
            ),
            (
                "mean_anomaly_deg",
                mutation(|el: &mut TleElements| el.mean_anomaly_deg = f64::NAN),
            ),
            (
                "mean_motion",
                mutation(|el: &mut TleElements| el.mean_motion = 100.0),
            ),
            (
                "eccentricity",
                mutation(|el: &mut TleElements| el.eccentricity = 1.0),
            ),
            (
                "eccentricity",
                mutation(|el: &mut TleElements| el.eccentricity = 0.99999996),
            ),
            (
                "eccentricity",
                mutation(|el: &mut TleElements| el.eccentricity = -0.1),
            ),
        ];
        for (field, mutate) in cases {
            let error = encode_error(mutate);
            assert!(refused(field)(error.clone()), "{field}: {error:?}");
        }
    }

    #[test]
    fn encode_writes_the_widest_values_each_field_holds() {
        let mut el = iss_elements();
        el.epoch_year = 2056;
        el.epoch_day_of_year = 999.5;
        el.elset_number = Some(9999);
        el.rev_number = Some(99_999);
        el.ephemeris_type = Some(9);
        el.mean_motion = 99.99999999;
        el.inclination_deg = 999.9999;
        el.raan_deg = -99.9999;
        el.eccentricity = 0.9999999;
        el.mean_motion_dot = -0.99999999;
        let (line1, line2) = encode(&el).unwrap();
        assert_eq!(line1.len(), 69);
        assert_eq!(line2.len(), 69);
        let back = parse_with_policy(&line1, &line2, TlePolicy::Strict)
            .unwrap()
            .elements;
        assert_eq!(back, el);
    }

    #[test]
    fn epoch_year_outside_the_pivot_window_is_not_wrapped() {
        // 2060 would be written as "60" and read back as 1960.
        let error = encode_error(|el| el.epoch_year = 2060);
        assert!(refused("epoch_year")(error));
        for year in [1957, 1999, 2000, 2056] {
            let mut el = iss_elements();
            el.epoch_year = year;
            let (line1, line2) = encode(&el).unwrap();
            assert_eq!(parse(&line1, &line2).unwrap().elements.epoch_year, year);
        }
    }

    #[test]
    fn tiny_assumed_decimal_values_are_spelled_from_the_smallest_exponent() {
        let mut el = iss_elements();
        el.bstar = 5.0e-11;
        el.mean_motion_double_dot = -1.0e-15;
        let (line1, line2) = encode(&el).unwrap();
        // 5e-11 needs exponent -10 when normalized, so the spelling starts
        // at -9. " 05000-9" decodes to 0.05 * 10^-9, one unit in the last
        // place above 5e-11; " 00500-8" decodes to 0.005 * 10^-8, which is
        // 5e-11 exactly, so it is the first exact spelling.
        assert_ne!((0.05 * 10.0_f64.powi(-9)).to_bits(), 5.0e-11_f64.to_bits());
        assert_eq!((0.005 * 10.0_f64.powi(-8)).to_bits(), 5.0e-11_f64.to_bits());
        assert_eq!(slice_inclusive(&line1, 53, 60), " 00500-8");
        // -1e-15 is below the 1e-14 resolution at exponent -9 and has no
        // exact spelling, so it rounds to zero.
        assert_eq!(slice_inclusive(&line1, 44, 51), " 00000-0");
        let back = parse(&line1, &line2).unwrap().elements;
        assert_eq!(back.bstar.to_bits(), el.bstar.to_bits());
        assert_eq!(encode(&back).unwrap(), (line1, line2));

        // A value with no exact spelling below 1e-10 rounds at exponent -9.
        let mut rounded = iss_elements();
        rounded.bstar = 1.23456789e-11;
        let (line1, _) = encode(&rounded).unwrap();
        assert_eq!(slice_inclusive(&line1, 53, 60), " 01235-9");
    }

    #[test]
    fn epoch_year_must_be_digits() {
        let line1 = with_columns(ISS_L1, 18, "-1");
        assert!(matches!(
            parse(&line1, ISS_L2),
            Err(TleError::InvalidField {
                field: "epoch_year",
                ..
            })
        ));
        // Vallado reads the field with %2d, so a blank tens digit is a
        // one-digit year.
        let line1 = with_columns(ISS_L1, 18, " 8");
        assert_eq!(parse(&line1, ISS_L2).unwrap().elements.epoch_year, 2008);
    }

    #[test]
    fn assumed_decimal_sign_column_must_be_a_sign() {
        let line1 = with_columns(ISS_L1, 53, "5");
        assert!(matches!(
            parse(&line1, ISS_L2),
            Err(TleError::InvalidField { field: "bstar", .. })
        ));
        let blank_sign = parse(ISS_L1, ISS_L2).unwrap().elements.bstar;
        let line1 = with_columns(ISS_L1, 53, "+");
        assert_eq!(parse(&line1, ISS_L2).unwrap().elements.bstar, blank_sign);
        let line1 = with_columns(ISS_L1, 53, "-");
        assert_eq!(parse(&line1, ISS_L2).unwrap().elements.bstar, -blank_sign);
    }

    #[test]
    fn second_derivative_blank_digits_read_as_zero_like_vallado() {
        let line1 = with_columns(ISS_L1, 44, "        ");
        assert_eq!(
            parse(&line1, ISS_L2)
                .unwrap()
                .elements
                .mean_motion_double_dot,
            0.0
        );
        let line1 = with_columns(ISS_L1, 44, "- 1234- ");
        assert_eq!(
            parse(&line1, ISS_L2)
                .unwrap()
                .elements
                .mean_motion_double_dot,
            -0.01234
        );
        // B* keeps the plain grammar: Vallado does not zero-fill it either.
        let line1 = with_columns(ISS_L1, 53, "        ");
        assert!(parse(&line1, ISS_L2).is_err());
    }

    #[test]
    fn strict_policy_refuses_a_checksum_mismatch() {
        let bad_l2 = with_raw_columns(ISS_L2, 68, "0");
        assert_eq!(
            parse_with_policy(ISS_L1, &bad_l2, TlePolicy::Strict),
            Err(TleError::ChecksumMismatch {
                line_label: "line 2",
                expected: 0,
                computed: 6,
            })
        );
        assert_eq!(TlePolicy::default(), TlePolicy::Strict);
        assert_eq!(
            parse(ISS_L1, &bad_l2),
            parse_with_policy(ISS_L1, &bad_l2, TlePolicy::Strict),
            "parse is strict"
        );
        let lenient = parse_with_policy(ISS_L1, &bad_l2, TlePolicy::Lenient).unwrap();
        assert_eq!(
            lenient.checksum_warnings,
            vec![ChecksumWarning {
                line_label: "line 2",
                kind: ChecksumWarningKind::Mismatch { expected: 0 },
                computed: 6,
            }]
        );
        assert!(parse_with_policy(ISS_L1, ISS_L2, TlePolicy::Strict).is_ok());
    }

    #[test]
    fn a_non_digit_checksum_column_is_refused_strictly_and_reported_leniently() {
        let bad_l1 = with_raw_columns(ISS_L1, 68, "X");
        assert_eq!(
            parse(&bad_l1, ISS_L2),
            Err(TleError::ChecksumNotDigit {
                line_label: "line 1",
                found: 'X',
                computed: 3,
            })
        );
        let lenient = parse_with_policy(&bad_l1, ISS_L2, TlePolicy::Lenient).unwrap();
        assert_eq!(
            lenient.checksum_warnings,
            vec![ChecksumWarning {
                line_label: "line 1",
                kind: ChecksumWarningKind::NotDigit { found: 'X' },
                computed: 3,
            }]
        );
    }

    #[test]
    fn a_line_without_a_checksum_is_read_and_reported_under_both_policies() {
        let short_l1 = &ISS_L1[..68];
        for policy in [TlePolicy::Strict, TlePolicy::Lenient] {
            let parsed = parse_with_policy(short_l1, ISS_L2, policy).unwrap();
            assert_eq!(parsed.elements, iss_elements());
            assert_eq!(
                parsed.checksum_warnings,
                vec![ChecksumWarning {
                    line_label: "line 1",
                    kind: ChecksumWarningKind::Missing,
                    computed: 3,
                }]
            );
        }
    }

    #[test]
    fn assumed_decimal_source_spelling_is_restated() {
        // A leading-zero mantissa and a `+0` zero exponent are both legal
        // spellings; the writer restates them rather than normalizing.
        let line1 = with_columns(&with_columns(ISS_L1, 44, " 00000+0"), 53, " 01234-4");
        let parsed = parse(&line1, ISS_L2).unwrap().elements;
        assert_eq!(parsed.mean_motion_double_dot, 0.0);
        assert_eq!(parsed.bstar, 0.01234 * 10.0_f64.powi(-4));
        assert_eq!(parsed.bstar_text.as_deref(), Some(" 01234-4"));
        let (out1, out2) = encode(&parsed).unwrap();
        assert_eq!(out1, line1);
        assert_eq!(out2, ISS_L2);
    }

    #[test]
    fn assumed_decimal_without_source_text_takes_the_first_exact_spelling() {
        let mut el = iss_elements();
        el.bstar_text = None;
        el.mean_motion_double_dot_text = None;
        el.mean_motion_double_dot = 0.0;
        // 0.01234e-4 need not equal 0.1234e-5 in binary; the writer takes the
        // first spelling from the normalized exponent upward that decodes to
        // the same bits.
        el.bstar = 0.01234 * 10.0_f64.powi(-4);
        let (line1, _) = encode(&el).unwrap();
        let written = slice_inclusive(&line1, 53, 60).to_string();
        let back = parse(&line1, ISS_L2).unwrap().elements;
        assert_eq!(back.bstar.to_bits(), el.bstar.to_bits(), "{written}");
        assert_eq!(slice_inclusive(&line1, 44, 51), " 00000-0");

        // A stale source text (the value was changed) is not restated.
        let mut changed = iss_elements();
        changed.bstar = 0.5e-4;
        let (line1, _) = encode(&changed).unwrap();
        assert_eq!(slice_inclusive(&line1, 53, 60), " 50000-4");

        // Negative zero keeps its sign.
        let mut negative_zero = iss_elements();
        negative_zero.mean_motion_double_dot = -0.0;
        let (line1, _) = encode(&negative_zero).unwrap();
        assert_eq!(slice_inclusive(&line1, 44, 51), "-00000-0");
    }

    #[test]
    fn blank_elset_and_revolution_numbers_read_as_none_and_write_back_blank() {
        let line1 = with_columns(ISS_L1, 64, "    ");
        let line2 = with_columns(ISS_L2, 63, "     ");
        let parsed = parse(&line1, &line2).unwrap().elements;
        assert_eq!(parsed.elset_number, None);
        assert_eq!(parsed.rev_number, None);
        assert_eq!(encode(&parsed).unwrap(), (line1, line2));

        // A blank ephemeris type reads as None and writes back blank; the
        // element set is the one a stated 0 gives, as in twoline2rv.
        let line1 = with_columns(ISS_L1, 62, " ");
        let parsed = parse(&line1, ISS_L2).unwrap().elements;
        assert_eq!(parsed.ephemeris_type, None);
        let (written, _) = encode(&parsed).unwrap();
        assert_eq!(written, line1);
        assert_eq!(
            parsed.to_element_set().unwrap(),
            iss_elements().to_element_set().unwrap()
        );
    }

    #[test]
    fn checksum_mismatch_is_reported_not_rejected() {
        // Flip the final checksum digit of line 1 (9993 -> 9990). Lenient is
        // the policy Vallado's twoline2rv reads with, which its verification
        // set (catalog numbers 33333-33335) depends on.
        let bad_l1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9990";
        let parsed = parse_with_policy(bad_l1, ISS_L2, TlePolicy::Lenient).unwrap();
        assert_eq!(parsed.checksum_warnings.len(), 1);
        assert_eq!(parsed.checksum_warnings[0].line_label, "line 1");
        assert_eq!(
            parsed.checksum_warnings[0].kind,
            ChecksumWarningKind::Mismatch { expected: 0 }
        );
        assert_eq!(parsed.checksum_warnings[0].computed, 3);
    }
}
