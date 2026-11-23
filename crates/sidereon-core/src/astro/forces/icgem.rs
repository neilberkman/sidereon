//! ICGEM gravity field files (`.gfc`).
//!
//! Reads Earth gravity field models in the format of the International Centre
//! for Global Earth Models: C. Förste, F. Barthelmes and E. S. Ince, "The
//! ICGEM-format", GFZ German Research Centre for Geosciences, 16 February 2023
//! (format versions `icgem1.0` and `icgem2.0`). A parsed
//! [`IcgemGravityField`] builds a [`SphericalHarmonicGravity`] at any degree
//! and order up to the file's maximum, evaluated by the Holmes and Featherstone
//! (2002) recursion (see [`crate::astro::forces::geopotential`]).
//!
//! # Header
//!
//! Lines before an optional `begin_of_head` line are comments; a
//! `begin_of_head` after the first `end_of_head` is a comment of the data
//! section. The header runs to `end_of_head` and must hold each mandatory
//! keyword once:
//! `product_type` (`gravity_field`), `modelname`, `earth_gravity_constant`
//! (m^3/s^2; also read as `gravity_constant`, which ICGEM's fields of the
//! Moon and other bodies write), `radius` (m), `max_degree` and
//! `errors` (`no`, `calibrated`, `formal` or `calibrated_and_formal`). The
//! optional `format` (`icgem1.0`, the default, or `icgem2.0`), `norm`
//! (`fully_normalized`, the default, or `unnormalized`) and `tide_system`
//! (`zero_tide`, `tide_free`, `mean_tide` or `unknown`, the default) are
//! read when present. Other lines are comments, as the specification says,
//! and so is anything after a record's last parameter.
//!
//! # Data
//!
//! * `gfc n m C S [sigmas]`: a static coefficient.
//! * `gfct n m C S [sigmas] t0` (`icgem1.0`) or `... t0 t1` (`icgem2.0`): the
//!   constant part of a time-variable coefficient, valid for all time with
//!   reference epoch `t0` in `icgem1.0` and for `t0 <= t < t1` in `icgem2.0`.
//! * `trnd` (and the older `dot`, `icgem1.0` only): a trend per year.
//! * `acos` and `asin`: cosine and sine amplitudes with a period in years.
//!
//! The number of sigma columns follows `errors` (0, 2 or 4). In `icgem2.0`
//! each `trnd`, `acos` and `asin` record carries its own `t0 t1`, and a
//! coefficient at epoch `t` is
//!
//! `G(t) = G(t0) + trnd (t - t0) + asin sin(2 pi (t - t0) / p) + acos cos(2 pi (t - t0) / p)`
//!
//! summed over the records whose interval holds `t`, each with its own `t0`;
//! in `icgem1.0` every record of a coefficient uses the `gfct` epoch. Times
//! are in Julian years of 365.25 days. Epochs `yyyymmdd.hhmm` are read as TT
//! calendar dates; a bare `yyyymmdd` is 00:00 of that day, the specification's
//! example equating `20050101` with `t0 = 2005.0 y`.
//!
//! A data-section line of any other keyword is a comment, as the
//! specification says, unless it has the shape of a record: an integer
//! degree, an integer order and at least two numbers. Such a line is far
//! likelier a `gfc`, `gfct`, `trnd`, `acos`, `asin` or `dot` record than a
//! remark, and read as a comment it would lose a coefficient or a
//! time-variable term. After a misspelt keyword it is
//! [`IcgemError::UnknownRecordKeyword`]; with no keyword, the line beginning
//! with the degree, it is [`IcgemError::MissingRecordKeyword`]. A line that
//! begins with an integer counts as missing its keyword even where its
//! remaining tokens would also fit a record under that integer as keyword.
//!
//! Every coefficient of degree 2 to `max_degree` must be given exactly once,
//! by one `gfc` record or by `gfct` records; degrees 0 and 1 are optional.
//! `gfct` intervals of one coefficient must not overlap, and every `trnd`,
//! `acos` and `asin` record must lie inside them.
//!
//! # Conversions
//!
//! Unnormalized coefficients are converted to the fully normalized form the
//! geopotential uses, dividing by
//! `N_nm = sqrt((2 - delta_m0) (2n + 1) (n - m)! / (n + m)!)` (Heiskanen and
//! Moritz, *Physical Geodesy*, 1967, Equation 1-73). The tide system is
//! recorded on the geopotential, so [`crate::astro::forces::SolidEarthTideGravity`]
//! removes the permanent tide the field holds, and
//! [`IcgemGravityField::gravity_in_tide_system`] converts `C20` between tide
//! systems with the IERS Conventions (2010) permanent tide of Section 6.2.2.
//!
//! # Differences from Orekit
//!
//! Orekit 13.1 (`ICGEMFormatReader` and `PiecewiseSphericalHarmonics`) differs
//! in these respects, and this reader follows the specification:
//!
//! * A bare `yyyymmdd` epoch is 12:00 TT in Orekit, 00:00 here.
//! * Orekit rejects `tide_system mean_tide`.
//! * Orekit requires each `trnd`, `acos` and `asin` record to follow its
//!   `gfct` record; the specification sets no order.
//! * Orekit measures an `icgem2.0` trend from the epoch of the `gfct` record
//!   rather than from the `trnd` record's own `t0`; the two agree when the
//!   intervals match, as they do in published files.
//! * At an epoch where a time-variable coefficient has no `gfct` record,
//!   Orekit uses zero for it (or fails outside every record's interval); this
//!   reader returns [`IcgemError::EpochOutsideValidity`].

use crate::astro::constants::time::{DAYS_PER_JULIAN_YEAR, SECONDS_PER_DAY};
use crate::astro::forces::geopotential::{
    coefficient_index, triangular_len, CoefficientPair, SphericalHarmonicCoefficient,
    SphericalHarmonicGravity, TideSystem,
};
use crate::astro::forces::tides::permanent_tide_c20;
use crate::astro::time::civil;
use std::f64::consts::TAU;

/// Seconds in a Julian year, the unit of ICGEM trends and periods.
const JULIAN_YEAR_SECONDS: f64 = DAYS_PER_JULIAN_YEAR * SECONDS_PER_DAY;

/// ICGEM read or evaluation error. Line numbers count from 1.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum IcgemError {
    /// A record ends before one of its parameters.
    #[error("ICGEM line {line}: {keyword} record has no {field}")]
    MissingField {
        /// Line of the record.
        line: usize,
        /// Record keyword.
        keyword: String,
        /// Missing parameter.
        field: &'static str,
    },
    /// A parameter is not a valid value for its field.
    #[error("ICGEM line {line}: {keyword} {field} {value:?} is not valid")]
    InvalidField {
        /// Line of the record.
        line: usize,
        /// Record keyword.
        keyword: String,
        /// Parameter name.
        field: &'static str,
        /// Parameter text.
        value: String,
    },
    /// A mandatory header keyword is absent.
    #[error("ICGEM header has no {keyword} record")]
    MissingHeaderKeyword {
        /// Missing keyword.
        keyword: &'static str,
    },
    /// A header keyword appears twice.
    #[error("ICGEM line {line}: header keyword {keyword} repeats line {first_line}")]
    RepeatedHeaderKeyword {
        /// Line of the repetition.
        line: usize,
        /// Line of the first occurrence.
        first_line: usize,
        /// Keyword.
        keyword: &'static str,
    },
    /// No `end_of_head` line ends the header.
    #[error("ICGEM file has no end_of_head line")]
    MissingEndOfHead,
    /// `dot` is an `icgem1.0` keyword and has no `icgem2.0` layout.
    #[error("ICGEM line {line}: {keyword} records are not defined in format icgem2.0")]
    KeywordNotInFormat {
        /// Line of the record.
        line: usize,
        /// Record keyword.
        keyword: String,
    },
    /// A data-section line has the shape of a record (a keyword, an integer
    /// degree, an integer order and at least two numbers) but its keyword is
    /// none the format defines: probably a misspelt `gfc`, `gfct`, `trnd`,
    /// `acos`, `asin` or `dot`, whose coefficient or term would be lost if
    /// the line were read as a comment.
    #[error(
        "ICGEM line {line}: {keyword:?} is not a record keyword, but the line holds a degree, an order and coefficient values"
    )]
    UnknownRecordKeyword {
        /// Line of the record.
        line: usize,
        /// The keyword as written.
        keyword: String,
    },
    /// A data-section line begins with an integer degree, an integer order
    /// and at least two numbers, with no keyword before them: a `gfc`,
    /// `gfct`, `trnd`, `acos`, `asin` or `dot` record whose keyword is
    /// missing, and whose coefficient or term would be lost if the line were
    /// read as a comment.
    #[error(
        "ICGEM line {line}: the line holds a degree, an order and coefficient values but no record keyword"
    )]
    MissingRecordKeyword {
        /// Line of the record.
        line: usize,
    },
    /// A record's degree exceeds the header `max_degree`.
    #[error("ICGEM line {line}: degree {degree} exceeds max_degree {max_degree}")]
    DegreeAboveMaximum {
        /// Line of the record.
        line: usize,
        /// Degree read.
        degree: u16,
        /// Header maximum degree.
        max_degree: u16,
    },
    /// A record's order exceeds its degree.
    #[error("ICGEM line {line}: order {order} exceeds degree {degree}")]
    OrderAboveDegree {
        /// Line of the record.
        line: usize,
        /// Degree read.
        degree: u16,
        /// Order read.
        order: u16,
    },
    /// A coefficient is given twice: two `gfc` records, or `gfc` and `gfct`
    /// records, or two `icgem1.0` `gfct` records.
    #[error(
        "ICGEM line {line}: coefficient {degree},{order} is already given on line {first_line}"
    )]
    RepeatedCoefficient {
        /// Line of the repetition.
        line: usize,
        /// Line of the first record.
        first_line: usize,
        /// Degree.
        degree: u16,
        /// Order.
        order: u16,
    },
    /// Two time-variable records of one coefficient, of the same kind (and
    /// period), hold for a common instant.
    #[error("ICGEM line {line}: {keyword} {degree},{order} overlaps line {first_line}")]
    OverlappingTerm {
        /// Line of the later record.
        line: usize,
        /// Line of the record it overlaps.
        first_line: usize,
        /// Record keyword.
        keyword: String,
        /// Degree.
        degree: u16,
        /// Order.
        order: u16,
    },
    /// A `trnd`, `dot`, `acos` or `asin` record holds where no `gfct` record
    /// of the same coefficient does.
    #[error(
        "ICGEM line {line}: {keyword} {degree},{order} is not covered by a gfct record of the same coefficient"
    )]
    UncoveredTerm {
        /// Line of the record.
        line: usize,
        /// Record keyword.
        keyword: String,
        /// Degree.
        degree: u16,
        /// Order.
        order: u16,
    },
    /// A coefficient of degree 2 or more has no record.
    #[error("ICGEM file has no record for coefficient {degree},{order}")]
    MissingCoefficient {
        /// Degree.
        degree: u16,
        /// Order.
        order: u16,
    },
    /// The data section has fewer lines than `max_degree` needs records.
    #[error(
        "ICGEM max_degree {max_degree} needs {required} coefficient records; the data section has {lines} lines"
    )]
    TooFewRecords {
        /// Header maximum degree.
        max_degree: u16,
        /// Records needed for degrees 2 to `max_degree`.
        required: usize,
        /// Non-empty lines in the data section.
        lines: usize,
    },
    /// An unnormalized coefficient's fully normalized value is not a finite
    /// non-zero double.
    #[error(
        "ICGEM line {line}: unnormalized coefficient {degree},{order} has no finite fully normalized value"
    )]
    NormalizationOutOfRange {
        /// Line of the record.
        line: usize,
        /// Degree.
        degree: u16,
        /// Order.
        order: u16,
    },
    /// A requested truncation is outside what the field holds.
    #[error(
        "degree {degree} and order {order} must satisfy order <= degree <= {max_degree} (and degree >= 2 for a gravity model)"
    )]
    TruncationOutOfRange {
        /// Requested degree.
        degree: u16,
        /// Requested order.
        order: u16,
        /// Highest degree available.
        max_degree: u16,
    },
    /// The field's tide system is `unknown` or not stated.
    #[error(
        "ICGEM field does not state its tide system; use IcgemGravityField::with_assumed_tide_system"
    )]
    UnknownTideSystem,
    /// A tide system was assumed for a field that states a different one.
    #[error("ICGEM field states tide system {stated:?}, not {assumed:?}")]
    TideSystemConflict {
        /// Tide system the file states.
        stated: TideSystem,
        /// Tide system the caller assumed.
        assumed: TideSystem,
    },
    /// No `gfct` record of a time-variable coefficient holds at the epoch.
    #[error(
        "epoch {epoch_tt_seconds} s TT from J2000 is outside every gfct interval of coefficient {degree},{order}"
    )]
    EpochOutsideValidity {
        /// Degree.
        degree: u16,
        /// Order.
        order: u16,
        /// Epoch, TT seconds since J2000.
        epoch_tt_seconds: f64,
    },
    /// The epoch is not finite.
    #[error("epoch_tt_seconds must be finite")]
    NonFiniteEpoch,
    /// A time-variable coefficient has no finite double value at the epoch:
    /// a trend carries it beyond the double range, or a period is so short
    /// that the phase `2 pi (t - t0) / p` overflows.
    #[error(
        "coefficient {degree},{order} has no finite value at epoch {epoch_tt_seconds} s TT from J2000"
    )]
    NonFiniteCoefficient {
        /// Degree.
        degree: u16,
        /// Order.
        order: u16,
        /// Epoch, TT seconds since J2000.
        epoch_tt_seconds: f64,
    },
}

/// Uncertainty columns present in the data records (`errors` keyword).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IcgemErrors {
    /// `no`: no sigma columns.
    No,
    /// `calibrated`: two calibrated sigma columns.
    Calibrated,
    /// `formal`: two formal sigma columns.
    Formal,
    /// `calibrated_and_formal`: two calibrated then two formal columns.
    CalibratedAndFormal,
}

/// Normalization of the coefficients in the file (`norm` keyword).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IcgemNormalization {
    /// `fully_normalized`, the default.
    FullyNormalized,
    /// `unnormalized`; converted to fully normalized on reading.
    Unnormalized,
}

/// ICGEM format version (`format` keyword).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IcgemFormat {
    /// `icgem1.0`, the default: time-variable records without validity
    /// intervals.
    Icgem1_0,
    /// `icgem2.0`: time-variable records valid over `t0 <= t < t1`.
    Icgem2_0,
}

/// ICGEM header values.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct IcgemHeader {
    /// `modelname`.
    pub model_name: String,
    /// `earth_gravity_constant`, m^3/s^2.
    pub gravity_constant_m3_s2: f64,
    /// `radius`, m.
    pub radius_m: f64,
    /// `max_degree`.
    pub max_degree: u16,
    /// `errors`.
    pub errors: IcgemErrors,
    /// `norm`; stored coefficients are fully normalized either way.
    pub normalization: IcgemNormalization,
    /// `tide_system`; `None` for `unknown` or absent.
    pub tide_system: Option<TideSystem>,
    /// `format`.
    pub format: IcgemFormat,
}

/// Uncertainties of one record, fully normalized like its coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IcgemSigmas {
    /// Calibrated `[sigma C, sigma S]`, when `errors` includes them.
    pub calibrated: Option<[f64; 2]>,
    /// Formal `[sigma C, sigma S]`, when `errors` includes them.
    pub formal: Option<[f64; 2]>,
}

/// Kind of a time-variable record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IcgemTermKind {
    /// `gfct`: constant part.
    Constant,
    /// `trnd` or `dot`: trend per Julian year.
    Trend,
    /// `acos`: cosine amplitude.
    Cosine {
        /// Period, Julian years.
        period_years: f64,
    },
    /// `asin`: sine amplitude.
    Sine {
        /// Period, Julian years.
        period_years: f64,
    },
}

impl IcgemTermKind {
    fn rank(self) -> u8 {
        match self {
            Self::Constant => 0,
            Self::Trend => 1,
            Self::Cosine { .. } => 2,
            Self::Sine { .. } => 3,
        }
    }
}

/// One time-variable record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IcgemTimeVariableTerm {
    /// Line of the record.
    pub line: usize,
    /// Record kind.
    pub kind: IcgemTermKind,
    /// Degree.
    pub degree: u16,
    /// Order.
    pub order: u16,
    /// Fully normalized `C` value (per Julian year for a trend).
    pub c: f64,
    /// Fully normalized `S` value (per Julian year for a trend).
    pub s: f64,
    /// Uncertainties.
    pub sigmas: IcgemSigmas,
    /// Epoch `t0` the record's time argument starts from, TT seconds since
    /// J2000.
    pub reference_tt_seconds: f64,
    /// Start of validity, TT seconds since J2000; `-inf` for `icgem1.0`.
    pub valid_from_tt_seconds: f64,
    /// End of validity (exclusive), TT seconds since J2000; `+inf` for
    /// `icgem1.0`.
    pub valid_until_tt_seconds: f64,
}

impl IcgemTimeVariableTerm {
    fn holds_at(&self, epoch_tt_seconds: f64) -> bool {
        self.valid_from_tt_seconds <= epoch_tt_seconds
            && epoch_tt_seconds < self.valid_until_tt_seconds
    }
}

/// A parsed ICGEM gravity field.
#[derive(Debug, Clone, PartialEq)]
pub struct IcgemGravityField {
    header: IcgemHeader,
    stored_max_degree: u16,
    stored_max_order: u16,
    /// Fully normalized static coefficients through `stored_max_degree`,
    /// zero above `stored_max_order` and for time-variable coefficients.
    static_coefficients: Vec<CoefficientPair>,
    /// Whether a `gfc` record gave the coefficient.
    static_given: Vec<bool>,
    calibrated_sigmas: Option<Vec<[f64; 2]>>,
    formal_sigmas: Option<Vec<[f64; 2]>>,
    /// Time-variable records inside the stored truncation, sorted by degree,
    /// order, kind (constant, trend, cosine, sine) and line.
    time_variable_terms: Vec<IcgemTimeVariableTerm>,
}

impl IcgemGravityField {
    /// Parse a complete ICGEM gravity field file.
    pub fn parse(text: &str) -> Result<Self, IcgemError> {
        parse(text, None)
    }

    /// Parse an ICGEM file, storing only coefficients through `max_degree`
    /// and `max_order`. Every record is still read and checked; this bounds
    /// memory when only part of a large field is wanted.
    pub fn parse_truncated(
        text: &str,
        max_degree: u16,
        max_order: u16,
    ) -> Result<Self, IcgemError> {
        parse(text, Some((max_degree, max_order)))
    }

    /// Header values.
    pub fn header(&self) -> &IcgemHeader {
        &self.header
    }

    /// Highest stored degree.
    pub fn stored_max_degree(&self) -> u16 {
        self.stored_max_degree
    }

    /// Highest stored order.
    pub fn stored_max_order(&self) -> u16 {
        self.stored_max_order
    }

    /// Fully normalized static coefficient given by a `gfc` record, if
    /// stored.
    pub fn static_coefficient(
        &self,
        degree: u16,
        order: u16,
    ) -> Option<SphericalHarmonicCoefficient> {
        let idx = self.stored_index(degree, order)?;
        self.static_given[idx].then(|| {
            let pair = self.static_coefficients[idx];
            SphericalHarmonicCoefficient {
                degree,
                order,
                c: pair.c,
                s: pair.s,
            }
        })
    }

    /// Uncertainties of a stored `gfc` coefficient; `None` when the file has
    /// none or the coefficient is not a stored `gfc` one.
    pub fn static_sigmas(&self, degree: u16, order: u16) -> Option<IcgemSigmas> {
        let idx = self.stored_index(degree, order)?;
        if !self.static_given[idx] || self.header.errors == IcgemErrors::No {
            return None;
        }
        Some(IcgemSigmas {
            calibrated: self.calibrated_sigmas.as_ref().map(|s| s[idx]),
            formal: self.formal_sigmas.as_ref().map(|s| s[idx]),
        })
    }

    /// Time-variable records inside the stored truncation.
    pub fn time_variable_terms(&self) -> &[IcgemTimeVariableTerm] {
        &self.time_variable_terms
    }

    /// Whether any stored coefficient varies with time.
    pub fn is_time_variable(&self) -> bool {
        !self.time_variable_terms.is_empty()
    }

    /// Fully normalized coefficient at `epoch_tt_seconds` (TT seconds since
    /// J2000), or `None` if not stored. Degree 0 and 1 coefficients the file
    /// omits are `None`.
    pub fn coefficient_at(
        &self,
        degree: u16,
        order: u16,
        epoch_tt_seconds: f64,
    ) -> Result<Option<SphericalHarmonicCoefficient>, IcgemError> {
        if !epoch_tt_seconds.is_finite() {
            return Err(IcgemError::NonFiniteEpoch);
        }
        let Some(idx) = self.stored_index(degree, order) else {
            return Ok(None);
        };
        if self.static_given[idx] {
            return Ok(self.static_coefficient(degree, order));
        }
        let terms = self.terms_of(degree, order);
        if terms.is_empty() {
            return Ok(None);
        }
        if !terms
            .iter()
            .any(|term| term.kind == IcgemTermKind::Constant && term.holds_at(epoch_tt_seconds))
        {
            return Err(IcgemError::EpochOutsideValidity {
                degree,
                order,
                epoch_tt_seconds,
            });
        }
        let mut pair = CoefficientPair::ZERO;
        for term in terms.iter().filter(|term| term.holds_at(epoch_tt_seconds)) {
            let eval = TermEval::new(term, 0);
            eval.add_to(&mut pair, epoch_tt_seconds);
        }
        if !pair.c.is_finite() || !pair.s.is_finite() {
            return Err(IcgemError::NonFiniteCoefficient {
                degree,
                order,
                epoch_tt_seconds,
            });
        }
        Ok(Some(SphericalHarmonicCoefficient {
            degree,
            order,
            c: pair.c,
            s: pair.s,
        }))
    }

    /// Return the field with tide system `tide_system` recorded, for a file
    /// that states `unknown` or no tide system. A file that states the same
    /// system is returned unchanged; one that states another is
    /// [`IcgemError::TideSystemConflict`].
    pub fn with_assumed_tide_system(mut self, tide_system: TideSystem) -> Result<Self, IcgemError> {
        match self.header.tide_system {
            None => {
                self.header.tide_system = Some(tide_system);
                Ok(self)
            }
            Some(stated) if stated == tide_system => Ok(self),
            Some(stated) => Err(IcgemError::TideSystemConflict {
                stated,
                assumed: tide_system,
            }),
        }
    }

    /// Build the geopotential through `max_degree` and `max_order` in the
    /// field's own tide system. Degree 0 and 1 terms are not part of the
    /// perturbation and are left out.
    pub fn gravity(
        &self,
        max_degree: u16,
        max_order: u16,
    ) -> Result<SphericalHarmonicGravity, IcgemError> {
        let tide_system = self
            .header
            .tide_system
            .ok_or(IcgemError::UnknownTideSystem)?;
        self.build_gravity(max_degree, max_order, tide_system, tide_system)
    }

    /// Build the geopotential with `C20` converted to `tide_system`.
    ///
    /// The fields differ in `C20` only: a zero-tide `C20` adds the permanent
    /// deformation `A0 H0 k20` to the tide-free one, and a mean-tide `C20`
    /// adds the permanent tide-generating potential `A0 H0` as well (IERS
    /// Conventions (2010), Section 6.2.2, Equations 6.13 and 6.14, with the
    /// Table 6.3 `k20`), the values [`crate::astro::forces::SolidEarthTideGravity`]
    /// removes.
    pub fn gravity_in_tide_system(
        &self,
        max_degree: u16,
        max_order: u16,
        tide_system: TideSystem,
    ) -> Result<SphericalHarmonicGravity, IcgemError> {
        let stated = self
            .header
            .tide_system
            .ok_or(IcgemError::UnknownTideSystem)?;
        self.build_gravity(max_degree, max_order, stated, tide_system)
    }

    fn build_gravity(
        &self,
        max_degree: u16,
        max_order: u16,
        stated: TideSystem,
        target: TideSystem,
    ) -> Result<SphericalHarmonicGravity, IcgemError> {
        if max_degree < 2
            || max_degree > self.stored_max_degree
            || max_order > max_degree
            || max_order > self.stored_max_order
        {
            return Err(IcgemError::TruncationOutOfRange {
                degree: max_degree,
                order: max_order,
                max_degree: self.stored_max_degree,
            });
        }
        let mut coefficients = vec![CoefficientPair::ZERO; triangular_len(max_degree)];
        for degree in 2..=max_degree {
            for order in 0..=degree.min(max_order) {
                let idx = coefficient_index(degree, order);
                coefficients[idx] = self.static_coefficients[idx];
            }
        }
        // A C20 given by gfct records takes the tide offset on its constant
        // part, so the static part stays zero.
        let c20_offset = if stated == target {
            0.0
        } else {
            permanent_tide_c20(target) - permanent_tide_c20(stated)
        };
        let c20 = coefficient_index(2, 0);
        let mut time_variable_c20_offset = 0.0;
        if self.static_given[c20] {
            coefficients[c20].c += c20_offset;
        } else {
            time_variable_c20_offset = c20_offset;
        }
        let time_variable = TimeVariableCoefficients::new(
            &self.time_variable_terms,
            max_degree,
            max_order,
            time_variable_c20_offset,
        );
        Ok(SphericalHarmonicGravity::from_dense(
            self.header.gravity_constant_m3_s2 / 1.0e9,
            self.header.radius_m / 1.0e3,
            max_degree,
            max_order,
            coefficients,
            target,
            time_variable,
        ))
    }

    fn stored_index(&self, degree: u16, order: u16) -> Option<usize> {
        (degree <= self.stored_max_degree && order <= degree && order <= self.stored_max_order)
            .then(|| coefficient_index(degree, order))
    }

    fn terms_of(&self, degree: u16, order: u16) -> &[IcgemTimeVariableTerm] {
        let key = (degree, order);
        let start = self
            .time_variable_terms
            .partition_point(|term| (term.degree, term.order) < key);
        let end = self
            .time_variable_terms
            .partition_point(|term| (term.degree, term.order) <= key);
        &self.time_variable_terms[start..end]
    }
}

/// Time-variable coefficients of a geopotential.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TimeVariableCoefficients {
    /// Highest degree with a time-variable record.
    max_degree: u16,
    /// Records in degree, order, kind and line order.
    terms: Vec<TermEval>,
    /// `(degree, order)` of each time-variable coefficient.
    coefficients: Vec<(u16, u16)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TermEval {
    index: usize,
    degree: u16,
    order: u16,
    kind: IcgemTermKind,
    reference_tt_seconds: f64,
    valid_from_tt_seconds: f64,
    valid_until_tt_seconds: f64,
    /// `C` and `S`, per second for a trend.
    c: f64,
    s: f64,
    /// Angular rate, rad/s, for a periodic record.
    pulsation: f64,
}

impl TermEval {
    fn new(term: &IcgemTimeVariableTerm, index: usize) -> Self {
        let (c, s, pulsation) = match term.kind {
            IcgemTermKind::Constant => (term.c, term.s, 0.0),
            IcgemTermKind::Trend => (
                term.c / JULIAN_YEAR_SECONDS,
                term.s / JULIAN_YEAR_SECONDS,
                0.0,
            ),
            IcgemTermKind::Cosine { period_years } | IcgemTermKind::Sine { period_years } => {
                (term.c, term.s, TAU / (period_years * JULIAN_YEAR_SECONDS))
            }
        };
        Self {
            index,
            degree: term.degree,
            order: term.order,
            kind: term.kind,
            reference_tt_seconds: term.reference_tt_seconds,
            valid_from_tt_seconds: term.valid_from_tt_seconds,
            valid_until_tt_seconds: term.valid_until_tt_seconds,
            c,
            s,
            pulsation,
        }
    }

    fn holds_at(&self, epoch_tt_seconds: f64) -> bool {
        self.valid_from_tt_seconds <= epoch_tt_seconds
            && epoch_tt_seconds < self.valid_until_tt_seconds
    }

    /// Add the record's value at `epoch_tt_seconds` to `pair`. A value
    /// outside the double range leaves `pair` non-finite for the caller to
    /// report.
    fn add_to(&self, pair: &mut CoefficientPair, epoch_tt_seconds: f64) {
        let dt = epoch_tt_seconds - self.reference_tt_seconds;
        // The phase at the record's own epoch is zero even when the period
        // is too short for a finite angular rate.
        let phase = if dt == 0.0 { 0.0 } else { self.pulsation * dt };
        let factor = match self.kind {
            IcgemTermKind::Constant => {
                pair.c += self.c;
                pair.s += self.s;
                return;
            }
            IcgemTermKind::Trend => dt,
            IcgemTermKind::Cosine { .. } => libm::cos(phase),
            IcgemTermKind::Sine { .. } => libm::sin(phase),
        };
        // A zero amplitude contributes zero whatever the phase.
        let scaled = |amplitude: f64| {
            if amplitude == 0.0 && !factor.is_finite() {
                0.0
            } else {
                amplitude * factor
            }
        };
        pair.c += scaled(self.c);
        pair.s += scaled(self.s);
    }
}

impl TimeVariableCoefficients {
    /// Records of degree 2 to `max_degree` and order up to `max_order`, with
    /// `c20_offset` added to each `gfct` value of `C20`; `None` if there are
    /// none.
    fn new(
        terms: &[IcgemTimeVariableTerm],
        max_degree: u16,
        max_order: u16,
        c20_offset: f64,
    ) -> Option<Self> {
        let terms: Vec<TermEval> = terms
            .iter()
            .filter(|term| term.degree >= 2 && term.degree <= max_degree && term.order <= max_order)
            .map(|term| {
                let mut eval = TermEval::new(term, coefficient_index(term.degree, term.order));
                if (term.degree, term.order) == (2, 0) && term.kind == IcgemTermKind::Constant {
                    eval.c += c20_offset;
                }
                eval
            })
            .collect();
        let tv_max_degree = terms.iter().map(|term| term.degree).max()?;
        let mut coefficients: Vec<(u16, u16)> =
            terms.iter().map(|term| (term.degree, term.order)).collect();
        coefficients.dedup();
        Some(Self {
            max_degree: tv_max_degree,
            terms,
            coefficients,
        })
    }

    pub(crate) fn has_non_zonal_terms(&self) -> bool {
        self.terms
            .iter()
            .any(|term| term.order > 0 && (term.c != 0.0 || term.s != 0.0))
    }

    /// `base` through the highest time-variable degree with the records
    /// that hold at `epoch_tt_seconds` added.
    pub(crate) fn coefficients_at(
        &self,
        base: &[CoefficientPair],
        epoch_tt_seconds: f64,
    ) -> Result<Vec<CoefficientPair>, IcgemError> {
        if !epoch_tt_seconds.is_finite() {
            return Err(IcgemError::NonFiniteEpoch);
        }
        let mut out = base[..triangular_len(self.max_degree)].to_vec();
        let mut coefficients = self.coefficients.iter();
        let mut pending: Option<(u16, u16)> = None;
        for term in &self.terms {
            let key = (term.degree, term.order);
            if !term.holds_at(epoch_tt_seconds) {
                continue;
            }
            if term.kind == IcgemTermKind::Constant {
                // gfct intervals of a coefficient do not overlap: one holds.
                let expected = coefficients.next();
                if expected != Some(&key) {
                    let (degree, order) = expected.copied().unwrap_or(key);
                    return Err(IcgemError::EpochOutsideValidity {
                        degree,
                        order,
                        epoch_tt_seconds,
                    });
                }
                pending = Some(key);
            } else if pending != Some(key) {
                // Unreachable for validated records: a trend or periodic
                // record lies inside the gfct intervals of its coefficient.
                return Err(IcgemError::EpochOutsideValidity {
                    degree: key.0,
                    order: key.1,
                    epoch_tt_seconds,
                });
            }
            term.add_to(&mut out[term.index], epoch_tt_seconds);
        }
        if let Some(&(degree, order)) = coefficients.next() {
            return Err(IcgemError::EpochOutsideValidity {
                degree,
                order,
                epoch_tt_seconds,
            });
        }
        for &(degree, order) in &self.coefficients {
            let pair = out[coefficient_index(degree, order)];
            if !pair.c.is_finite() || !pair.s.is_finite() {
                return Err(IcgemError::NonFiniteCoefficient {
                    degree,
                    order,
                    epoch_tt_seconds,
                });
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Parsing

const DATA_KEYWORDS: [&str; 6] = ["gfc", "gfct", "trnd", "dot", "acos", "asin"];

#[derive(Default)]
struct HeaderLines {
    product_type: Option<usize>,
    model_name: Option<(usize, String)>,
    gravity_constant: Option<(usize, f64)>,
    radius: Option<(usize, f64)>,
    max_degree: Option<(usize, u16)>,
    errors: Option<(usize, IcgemErrors)>,
    normalization: Option<(usize, IcgemNormalization)>,
    tide_system: Option<(usize, Option<TideSystem>)>,
    format: Option<(usize, IcgemFormat)>,
}

/// A time-variable record before validation.
struct RawTerm {
    term: IcgemTimeVariableTerm,
    keyword: &'static str,
    /// `icgem1.0` `gfct` epoch.
    epoch: Option<f64>,
}

fn invalid(line: usize, keyword: &str, field: &'static str, value: &str) -> IcgemError {
    IcgemError::InvalidField {
        line,
        keyword: keyword.to_string(),
        field,
        value: value.to_string(),
    }
}

fn header_value<'a>(
    tokens: &mut impl Iterator<Item = &'a str>,
    line: usize,
    keyword: &str,
) -> Result<&'a str, IcgemError> {
    tokens.next().ok_or_else(|| IcgemError::MissingField {
        line,
        keyword: keyword.to_string(),
        field: "value",
    })
}

fn set_once<T>(
    slot: &mut Option<(usize, T)>,
    line: usize,
    keyword: &'static str,
    value: T,
) -> Result<(), IcgemError> {
    if let Some((first_line, _)) = slot {
        return Err(IcgemError::RepeatedHeaderKeyword {
            line,
            first_line: *first_line,
            keyword,
        });
    }
    *slot = Some((line, value));
    Ok(())
}

fn parse_float(token: &str) -> Option<f64> {
    let value = if token.contains(['D', 'd']) {
        token.replace(['D', 'd'], "E").parse::<f64>().ok()?
    } else {
        token.parse::<f64>().ok()?
    };
    value.is_finite().then_some(value)
}

/// Whether tokens begin with the parameters every record starts with: an
/// integer degree, an integer order and at least two numbers. A number here
/// is any decimal or `D`-exponent value with a digit, finite or not, so a
/// record under a misspelt or missing keyword with an out-of-range value is
/// still recognized.
fn is_record_shaped<'a>(mut tokens: impl Iterator<Item = &'a str>) -> bool {
    fn integer(token: Option<&str>) -> bool {
        token.is_some_and(|token| {
            let digits = token.strip_prefix('+').unwrap_or(token);
            !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
        })
    }
    fn number(token: Option<&str>) -> bool {
        token.is_some_and(|token| {
            token.bytes().any(|b| b.is_ascii_digit())
                && if token.contains(['D', 'd']) {
                    token.replace(['D', 'd'], "E").parse::<f64>().is_ok()
                } else {
                    token.parse::<f64>().is_ok()
                }
        })
    }
    integer(tokens.next())
        && integer(tokens.next())
        && number(tokens.next())
        && number(tokens.next())
}

fn parse_header(lines: &[&str]) -> Result<(IcgemHeader, usize), IcgemError> {
    // begin_of_head opens the header only if it comes before the first
    // end_of_head; after it, the line is a data-section comment.
    let start = lines
        .iter()
        .enumerate()
        .find_map(|(i, line)| match line.split_whitespace().next() {
            Some("begin_of_head") => Some(i + 1),
            Some("end_of_head") => Some(0),
            _ => None,
        })
        .unwrap_or(0);
    let mut found = HeaderLines::default();
    let mut end = None;
    for (i, text) in lines.iter().enumerate().skip(start) {
        let line = i + 1;
        let mut tokens = text.split_whitespace();
        let Some(keyword) = tokens.next() else {
            continue;
        };
        match keyword {
            "end_of_head" => {
                end = Some(i);
                break;
            }
            "product_type" => {
                let value = header_value(&mut tokens, line, keyword)?;
                if value != "gravity_field" {
                    return Err(invalid(line, keyword, "value", value));
                }
                if let Some(first_line) = found.product_type {
                    return Err(IcgemError::RepeatedHeaderKeyword {
                        line,
                        first_line,
                        keyword: "product_type",
                    });
                }
                found.product_type = Some(line);
            }
            "modelname" => {
                let value = header_value(&mut tokens, line, keyword)?;
                set_once(&mut found.model_name, line, "modelname", value.to_string())?;
            }
            "radius" => {
                let value = header_value(&mut tokens, line, keyword)?;
                let radius = parse_float(value)
                    .filter(|v| *v > 0.0)
                    .ok_or_else(|| invalid(line, keyword, "value", value))?;
                set_once(&mut found.radius, line, "radius", radius)?;
            }
            "max_degree" => {
                let value = header_value(&mut tokens, line, keyword)?;
                let degree = value
                    .parse::<u16>()
                    .map_err(|_| invalid(line, keyword, "value", value))?;
                set_once(&mut found.max_degree, line, "max_degree", degree)?;
            }
            "errors" => {
                let value = header_value(&mut tokens, line, keyword)?;
                let errors = match value {
                    "no" => IcgemErrors::No,
                    "calibrated" => IcgemErrors::Calibrated,
                    "formal" => IcgemErrors::Formal,
                    "calibrated_and_formal" => IcgemErrors::CalibratedAndFormal,
                    _ => return Err(invalid(line, keyword, "value", value)),
                };
                set_once(&mut found.errors, line, "errors", errors)?;
            }
            "norm" => {
                let value = header_value(&mut tokens, line, keyword)?;
                let norm = match value {
                    "fully_normalized" => IcgemNormalization::FullyNormalized,
                    "unnormalized" => IcgemNormalization::Unnormalized,
                    _ => return Err(invalid(line, keyword, "value", value)),
                };
                set_once(&mut found.normalization, line, "norm", norm)?;
            }
            "tide_system" => {
                let value = header_value(&mut tokens, line, keyword)?;
                let tide = match value {
                    "zero_tide" => Some(TideSystem::ZeroTide),
                    "tide_free" => Some(TideSystem::TideFree),
                    "mean_tide" => Some(TideSystem::MeanTide),
                    "unknown" => None,
                    _ => return Err(invalid(line, keyword, "value", value)),
                };
                set_once(&mut found.tide_system, line, "tide_system", tide)?;
            }
            "format" => {
                let value = header_value(&mut tokens, line, keyword)?;
                let format = match value {
                    "icgem1.0" => IcgemFormat::Icgem1_0,
                    "icgem2.0" => IcgemFormat::Icgem2_0,
                    _ => return Err(invalid(line, keyword, "value", value)),
                };
                set_once(&mut found.format, line, "format", format)?;
            }
            // Earth fields write earth_gravity_constant; ICGEM's fields of
            // the Moon and other bodies write gravity_constant.
            "earth_gravity_constant" | "gravity_constant" => {
                let value = header_value(&mut tokens, line, keyword)?;
                let gm = parse_float(value)
                    .filter(|v| *v > 0.0)
                    .ok_or_else(|| invalid(line, keyword, "value", value))?;
                set_once(
                    &mut found.gravity_constant,
                    line,
                    "earth_gravity_constant",
                    gm,
                )?;
            }
            _ => {}
        }
    }
    let end = end.ok_or(IcgemError::MissingEndOfHead)?;
    let missing = |keyword| IcgemError::MissingHeaderKeyword { keyword };
    if found.product_type.is_none() {
        return Err(missing("product_type"));
    }
    let header = IcgemHeader {
        model_name: found.model_name.ok_or(missing("modelname"))?.1,
        gravity_constant_m3_s2: found
            .gravity_constant
            .ok_or(missing("earth_gravity_constant"))?
            .1,
        radius_m: found.radius.ok_or(missing("radius"))?.1,
        max_degree: found.max_degree.ok_or(missing("max_degree"))?.1,
        errors: found.errors.ok_or(missing("errors"))?.1,
        normalization: found
            .normalization
            .map_or(IcgemNormalization::FullyNormalized, |(_, v)| v),
        tide_system: found.tide_system.and_then(|(_, v)| v),
        format: found.format.map_or(IcgemFormat::Icgem1_0, |(_, v)| v),
    };
    Ok((header, end))
}

/// TT seconds since J2000 of an ICGEM epoch `yyyymmdd` or `yyyymmdd.hhmm`.
/// Minutes run to 60 because the published EIGEN-6S4 file writes
/// `20041226.0060`.
fn parse_epoch(token: &str) -> Option<f64> {
    let (date, clock) = match token.split_once('.') {
        Some((date, clock)) => (date, Some(clock)),
        None => (token, None),
    };
    let digits = |text: &str, len: usize| {
        (text.len() == len && text.bytes().all(|b| b.is_ascii_digit()))
            .then(|| text.parse::<i32>().ok())
            .flatten()
    };
    digits(date, 8)?;
    let year = date[0..4].parse::<i32>().ok()?;
    let month = date[4..6].parse::<i32>().ok()?;
    let day = date[6..8].parse::<i32>().ok()?;
    if !(1..=12).contains(&month) || day < 1 || day > days_in_month(year, month) {
        return None;
    }
    let (hour, minute) = match clock {
        Some(clock) => {
            digits(clock, 4)?;
            let hour = clock[0..2].parse::<i32>().ok()?;
            let minute = clock[2..4].parse::<i32>().ok()?;
            if hour > 23 || minute > 60 {
                return None;
            }
            (hour, minute)
        }
        None => (0, 0),
    };
    Some(civil::j2000_seconds(year, month, day, hour, minute, 0.0))
}

fn days_in_month(year: i32, month: i32) -> i32 {
    match month {
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Fully normalized value of the unnormalized `value`: `value / N_nm`, with
/// `1 / N_nm = sqrt((n + m)! / ((2 - delta_m0) (2n + 1) (n - m)!))` applied one
/// factor at a time so no partial product overflows before the value does.
fn normalize(value: f64, line: usize, degree: u16, order: u16) -> Result<f64, IcgemError> {
    if value == 0.0 {
        return Ok(value);
    }
    let n = u32::from(degree);
    let m = u32::from(order);
    let delta = if m == 0 { 1.0 } else { 2.0 };
    let mut normalized = value / (delta * f64::from(2 * n + 1)).sqrt();
    for k in (n - m + 1)..=(n + m) {
        normalized *= f64::from(k).sqrt();
    }
    if normalized.is_finite() && normalized != 0.0 {
        Ok(normalized)
    } else {
        Err(IcgemError::NormalizationOutOfRange {
            line,
            degree,
            order,
        })
    }
}

/// Field names of the sigma columns for each `errors` value.
fn sigma_fields(errors: IcgemErrors) -> &'static [&'static str] {
    match errors {
        IcgemErrors::No => &[],
        IcgemErrors::Calibrated | IcgemErrors::Formal => &["sigma C", "sigma S"],
        IcgemErrors::CalibratedAndFormal => &[
            "sigma C calibrated",
            "sigma S calibrated",
            "sigma C formal",
            "sigma S formal",
        ],
    }
}

/// One data record's parameters.
struct Record<'a> {
    line: usize,
    keyword: &'static str,
    tokens: std::str::SplitWhitespace<'a>,
}

impl<'a> Record<'a> {
    fn next(&mut self, field: &'static str) -> Result<&'a str, IcgemError> {
        self.tokens.next().ok_or_else(|| IcgemError::MissingField {
            line: self.line,
            keyword: self.keyword.to_string(),
            field,
        })
    }

    fn float(&mut self, field: &'static str) -> Result<f64, IcgemError> {
        let token = self.next(field)?;
        parse_float(token).ok_or_else(|| invalid(self.line, self.keyword, field, token))
    }

    fn integer(&mut self, field: &'static str) -> Result<u16, IcgemError> {
        let token = self.next(field)?;
        token
            .parse::<u16>()
            .map_err(|_| invalid(self.line, self.keyword, field, token))
    }

    fn epoch(&mut self, field: &'static str) -> Result<f64, IcgemError> {
        let token = self.next(field)?;
        parse_epoch(token).ok_or_else(|| invalid(self.line, self.keyword, field, token))
    }

    fn sigmas(&mut self, errors: IcgemErrors) -> Result<IcgemSigmas, IcgemError> {
        let mut values = [0.0; 4];
        for (value, field) in values.iter_mut().zip(sigma_fields(errors)) {
            let token = self.next(field)?;
            *value = parse_float(token)
                .filter(|v| *v >= 0.0)
                .ok_or_else(|| invalid(self.line, self.keyword, field, token))?;
        }
        let pair = |i: usize| Some([values[i], values[i + 1]]);
        Ok(match errors {
            IcgemErrors::No => IcgemSigmas::default(),
            IcgemErrors::Calibrated => IcgemSigmas {
                calibrated: pair(0),
                formal: None,
            },
            IcgemErrors::Formal => IcgemSigmas {
                calibrated: None,
                formal: pair(0),
            },
            IcgemErrors::CalibratedAndFormal => IcgemSigmas {
                calibrated: pair(0),
                formal: pair(2),
            },
        })
    }
}

fn normalize_sigmas(
    sigmas: IcgemSigmas,
    line: usize,
    degree: u16,
    order: u16,
) -> Result<IcgemSigmas, IcgemError> {
    let scale = |pair: Option<[f64; 2]>| -> Result<Option<[f64; 2]>, IcgemError> {
        pair.map(|[c, s]| {
            Ok([
                normalize(c, line, degree, order)?,
                normalize(s, line, degree, order)?,
            ])
        })
        .transpose()
    };
    Ok(IcgemSigmas {
        calibrated: scale(sigmas.calibrated)?,
        formal: scale(sigmas.formal)?,
    })
}

fn parse(text: &str, truncation: Option<(u16, u16)>) -> Result<IcgemGravityField, IcgemError> {
    let lines: Vec<&str> = text.lines().collect();
    let (header, end) = parse_header(&lines)?;
    let max_degree = header.max_degree;
    let (stored_max_degree, stored_max_order) = truncation.unwrap_or((max_degree, max_degree));
    if stored_max_degree > max_degree || stored_max_order > stored_max_degree {
        return Err(IcgemError::TruncationOutOfRange {
            degree: stored_max_degree,
            order: stored_max_order,
            max_degree,
        });
    }

    // Every coefficient of degree 2 and up needs a line, so the data section
    // bounds the tables before they are allocated.
    let data_lines = lines[end + 1..]
        .iter()
        .filter(|line| !line.trim().is_empty())
        .count();
    let required = triangular_len(max_degree).saturating_sub(3);
    if required > data_lines {
        return Err(IcgemError::TooFewRecords {
            max_degree,
            required,
            lines: data_lines,
        });
    }

    let full_len = triangular_len(max_degree);
    // First line giving each coefficient (0: none) and whether by gfct.
    let mut given_on = vec![0_u32; full_len];
    let mut given_by_gfct = vec![false; full_len];
    let stored_len = triangular_len(stored_max_degree);
    let mut static_coefficients = vec![CoefficientPair::ZERO; stored_len];
    let mut static_given = vec![false; stored_len];
    let mut calibrated_sigmas = matches!(
        header.errors,
        IcgemErrors::Calibrated | IcgemErrors::CalibratedAndFormal
    )
    .then(|| vec![[0.0; 2]; stored_len]);
    let mut formal_sigmas = matches!(
        header.errors,
        IcgemErrors::Formal | IcgemErrors::CalibratedAndFormal
    )
    .then(|| vec![[0.0; 2]; stored_len]);
    let mut raw_terms: Vec<RawTerm> = Vec::new();
    let unnormalized = header.normalization == IcgemNormalization::Unnormalized;

    for (i, text) in lines.iter().enumerate().skip(end + 1) {
        let line = i + 1;
        let mut tokens = text.split_whitespace();
        let Some(first) = tokens.next() else {
            continue;
        };
        let Some(keyword) = DATA_KEYWORDS.iter().copied().find(|k| *k == first) else {
            // The specification makes lines of other keywords comments, but
            // one shaped like a record is far likelier a record whose keyword
            // is missing or misspelt. A line beginning with the degree is
            // checked first: an integer is no keyword, misspelt or not.
            if is_record_shaped(text.split_whitespace()) {
                return Err(IcgemError::MissingRecordKeyword { line });
            }
            if is_record_shaped(tokens) {
                return Err(IcgemError::UnknownRecordKeyword {
                    line,
                    keyword: first.to_string(),
                });
            }
            continue;
        };
        if keyword == "dot" && header.format == IcgemFormat::Icgem2_0 {
            return Err(IcgemError::KeywordNotInFormat {
                line,
                keyword: keyword.to_string(),
            });
        }
        let mut record = Record {
            line,
            keyword,
            tokens,
        };
        let degree = record.integer("degree")?;
        let order = record.integer("order")?;
        if degree > max_degree {
            return Err(IcgemError::DegreeAboveMaximum {
                line,
                degree,
                max_degree,
            });
        }
        if order > degree {
            return Err(IcgemError::OrderAboveDegree {
                line,
                degree,
                order,
            });
        }
        let c = record.float("C")?;
        let s = record.float("S")?;
        let sigmas = record.sigmas(header.errors)?;
        let stored = degree <= stored_max_degree && order <= stored_max_order;
        let (c, s, sigmas) = if unnormalized && stored {
            (
                normalize(c, line, degree, order)?,
                normalize(s, line, degree, order)?,
                normalize_sigmas(sigmas, line, degree, order)?,
            )
        } else {
            (c, s, sigmas)
        };
        let idx = coefficient_index(degree, order);

        if keyword == "gfc" || keyword == "gfct" {
            let is_gfct = keyword == "gfct";
            let first_line = given_on[idx];
            let repeated = first_line != 0
                && (!is_gfct || !given_by_gfct[idx] || header.format == IcgemFormat::Icgem1_0);
            if repeated {
                return Err(IcgemError::RepeatedCoefficient {
                    line,
                    first_line: first_line as usize,
                    degree,
                    order,
                });
            }
            if first_line == 0 {
                given_on[idx] = u32::try_from(line).unwrap_or(u32::MAX);
                given_by_gfct[idx] = is_gfct;
            }
        }

        let (kind, epoch, valid_from, valid_until) = match keyword {
            "gfc" => {
                if stored {
                    static_coefficients[idx] = CoefficientPair { c, s };
                    static_given[idx] = true;
                    if let (Some(table), Some(pair)) = (&mut calibrated_sigmas, sigmas.calibrated) {
                        table[idx] = pair;
                    }
                    if let (Some(table), Some(pair)) = (&mut formal_sigmas, sigmas.formal) {
                        table[idx] = pair;
                    }
                }
                continue;
            }
            "gfct" => match header.format {
                IcgemFormat::Icgem1_0 => {
                    let t0 = record.epoch("t0")?;
                    (
                        IcgemTermKind::Constant,
                        Some(t0),
                        f64::NEG_INFINITY,
                        f64::INFINITY,
                    )
                }
                IcgemFormat::Icgem2_0 => {
                    let (t0, t1) = interval(&mut record)?;
                    (IcgemTermKind::Constant, None, t0, t1)
                }
            },
            "trnd" | "dot" => match header.format {
                IcgemFormat::Icgem1_0 => {
                    (IcgemTermKind::Trend, None, f64::NEG_INFINITY, f64::INFINITY)
                }
                IcgemFormat::Icgem2_0 => {
                    let (t0, t1) = interval(&mut record)?;
                    (IcgemTermKind::Trend, None, t0, t1)
                }
            },
            _ => {
                let (t0, t1) = match header.format {
                    IcgemFormat::Icgem1_0 => (f64::NEG_INFINITY, f64::INFINITY),
                    IcgemFormat::Icgem2_0 => interval(&mut record)?,
                };
                let token = record.next("period")?;
                let period_years = parse_float(token)
                    .filter(|v| *v > 0.0)
                    .ok_or_else(|| invalid(line, keyword, "period", token))?;
                let kind = if keyword == "acos" {
                    IcgemTermKind::Cosine { period_years }
                } else {
                    IcgemTermKind::Sine { period_years }
                };
                (kind, None, t0, t1)
            }
        };
        raw_terms.push(RawTerm {
            term: IcgemTimeVariableTerm {
                line,
                kind,
                degree,
                order,
                c,
                s,
                sigmas,
                reference_tt_seconds: if valid_from.is_finite() {
                    valid_from
                } else {
                    epoch.unwrap_or(f64::NAN)
                },
                valid_from_tt_seconds: valid_from,
                valid_until_tt_seconds: valid_until,
            },
            keyword,
            epoch,
        });
    }

    for degree in 2..=max_degree {
        for order in 0..=degree {
            if given_on[coefficient_index(degree, order)] == 0 {
                return Err(IcgemError::MissingCoefficient { degree, order });
            }
        }
    }

    let time_variable_terms = validate_terms(
        raw_terms,
        &given_by_gfct,
        stored_max_degree,
        stored_max_order,
    )?;
    Ok(IcgemGravityField {
        header,
        stored_max_degree,
        stored_max_order,
        static_coefficients,
        static_given,
        calibrated_sigmas,
        formal_sigmas,
        time_variable_terms,
    })
}

fn interval(record: &mut Record<'_>) -> Result<(f64, f64), IcgemError> {
    let t0 = record.epoch("t0")?;
    let token = record.next("t1")?;
    let t1 = parse_epoch(token)
        .filter(|t1| *t1 > t0)
        .ok_or_else(|| invalid(record.line, record.keyword, "t1", token))?;
    Ok((t0, t1))
}

/// Check the time-variable records coefficient by coefficient and return
/// those inside the stored truncation, sorted.
fn validate_terms(
    mut raw: Vec<RawTerm>,
    given_by_gfct: &[bool],
    stored_max_degree: u16,
    stored_max_order: u16,
) -> Result<Vec<IcgemTimeVariableTerm>, IcgemError> {
    raw.sort_by(|a, b| {
        (a.term.degree, a.term.order, a.term.kind.rank(), a.term.line).cmp(&(
            b.term.degree,
            b.term.order,
            b.term.kind.rank(),
            b.term.line,
        ))
    });
    let mut kept = Vec::new();
    let mut start = 0;
    while start < raw.len() {
        let key = (raw[start].term.degree, raw[start].term.order);
        let end = start
            + raw[start..]
                .iter()
                .take_while(|r| (r.term.degree, r.term.order) == key)
                .count();
        let group = &mut raw[start..end];
        validate_group(group, given_by_gfct[coefficient_index(key.0, key.1)])?;
        if key.0 <= stored_max_degree && key.1 <= stored_max_order {
            kept.extend(group.iter().map(|r| r.term));
        }
        start = end;
    }
    Ok(kept)
}

fn validate_group(group: &mut [RawTerm], has_gfct: bool) -> Result<(), IcgemError> {
    let constants: Vec<usize> = (0..group.len())
        .filter(|&i| group[i].term.kind == IcgemTermKind::Constant)
        .collect();
    if !has_gfct || constants.is_empty() {
        let first = &group[0];
        return Err(IcgemError::UncoveredTerm {
            line: first.term.line,
            keyword: first.keyword.to_string(),
            degree: first.term.degree,
            order: first.term.order,
        });
    }
    // icgem1.0: every record of the coefficient runs from the gfct epoch.
    if let Some(epoch) = group[constants[0]].epoch {
        for raw in group.iter_mut() {
            raw.term.reference_tt_seconds = epoch;
        }
    }

    // Records of one kind (and period) must not hold at a common instant.
    let mut by_start: Vec<usize> = (0..group.len()).collect();
    by_start.sort_by(|&a, &b| {
        group[a]
            .term
            .valid_from_tt_seconds
            .total_cmp(&group[b].term.valid_from_tt_seconds)
            .then(group[a].term.line.cmp(&group[b].term.line))
    });
    for (pos, &a) in by_start.iter().enumerate() {
        for &b in &by_start[pos + 1..] {
            let (first, later) = (&group[a], &group[b]);
            if later.term.valid_from_tt_seconds >= first.term.valid_until_tt_seconds {
                // Later records start later still.
                break;
            }
            if first.term.kind == later.term.kind {
                let (first, later) = if first.term.line < later.term.line {
                    (first, later)
                } else {
                    (later, first)
                };
                return Err(IcgemError::OverlappingTerm {
                    line: later.term.line,
                    first_line: first.term.line,
                    keyword: later.keyword.to_string(),
                    degree: later.term.degree,
                    order: later.term.order,
                });
            }
        }
    }

    // Every other record must lie inside the union of the gfct intervals.
    let mut constant_intervals: Vec<(f64, f64)> = constants
        .iter()
        .map(|&i| {
            (
                group[i].term.valid_from_tt_seconds,
                group[i].term.valid_until_tt_seconds,
            )
        })
        .collect();
    constant_intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
    for raw in group.iter() {
        if raw.term.kind == IcgemTermKind::Constant {
            continue;
        }
        let mut cursor = raw.term.valid_from_tt_seconds;
        let target = raw.term.valid_until_tt_seconds;
        for &(from, until) in &constant_intervals {
            if from <= cursor && cursor < until {
                cursor = until;
            }
            if cursor >= target {
                break;
            }
        }
        if cursor < target {
            return Err(IcgemError::UncoveredTerm {
                line: raw.term.line,
                keyword: raw.keyword.to_string(),
                degree: raw.term.degree,
                order: raw.term.order,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Parser and evaluation checks that need crate internals; the Orekit
    //! comparisons are in `tests/icgem.rs`.

    use super::*;
    use crate::astro::forces::ForceModel;
    use crate::astro::propagator::PropagationContext;
    use crate::astro::state::CartesianState;
    use crate::astro::time::scales::tt_from_tdb_j2000_seconds;

    const SYNTHETIC_ICGEM2: &str =
        include_str!("../../../tests/fixtures/icgem/synthetic_icgem2.gfc");

    #[test]
    fn force_model_evaluates_coefficients_at_the_state_epoch_in_tt() {
        let field = IcgemGravityField::parse(SYNTHETIC_ICGEM2).expect("field");
        // Order 0 keeps the zonal C20 and C30 records, so no Earth-rotation
        // provider is needed.
        let gravity = field.gravity(6, 0).expect("gravity");
        assert!(gravity.is_time_variable());
        let position = [6900.0, -1300.0, 2100.0];
        for tdb in [1.2e8, 3.5e8, 6.1e8] {
            let tt = tt_from_tdb_j2000_seconds(tdb);
            // TDB - TT stays within 1.7 ms.
            assert!((tdb - tt).abs() < 2.0e-3, "{tdb} {tt}");
            let state = CartesianState::new(tdb, position, [0.0, 7.5, 0.0]);
            let actual = gravity
                .acceleration(&state, &PropagationContext::default())
                .expect("acceleration");
            let expected = gravity
                .body_fixed_acceleration_at_km_s2(position, tt)
                .expect("at epoch");
            assert_eq!([actual.x, actual.y, actual.z], expected);
        }
        // Different epochs give different fields.
        let a = gravity
            .body_fixed_acceleration_at_km_s2(position, 1.2e8)
            .expect("a");
        let b = gravity
            .body_fixed_acceleration_at_km_s2(position, 6.1e8)
            .expect("b");
        assert_ne!(a, b);
    }

    #[test]
    fn coefficients_at_adds_only_the_records_that_hold() {
        let field = IcgemGravityField::parse(SYNTHETIC_ICGEM2).expect("field");
        let gravity = field.gravity(6, 6).expect("gravity");
        let set =
            TimeVariableCoefficients::new(field.time_variable_terms(), 6, 6, 0.0).expect("set");
        // Degree 1 records are left out; C20, C21, C22, C30 and C43 remain.
        assert_eq!(set.coefficients, [(2, 0), (2, 1), (2, 2), (3, 0), (4, 3)]);
        assert_eq!(set.max_degree, 4);
        let epoch = civil::j2000_seconds(2008, 3, 1, 6, 0, 0.0);
        let base = vec![CoefficientPair::ZERO; triangular_len(6)];
        let low = set.coefficients_at(&base, epoch).expect("coefficients");
        assert_eq!(low.len(), triangular_len(4));
        for (degree, order) in [(2, 0), (2, 1), (2, 2), (3, 0), (4, 3)] {
            let pair = low[coefficient_index(degree, order)];
            let expected = field
                .coefficient_at(degree, order, epoch)
                .expect("field")
                .expect("stored");
            assert_eq!((pair.c, pair.s), (expected.c, expected.s));
            assert_eq!(
                gravity
                    .coefficient_at(degree, order, epoch)
                    .expect("gravity"),
                Some(expected)
            );
        }
        // At 06:00 on 2008-03-01 the second C22 interval has begun: its gfct
        // value, plus a zero-length trend and sine.
        let c22 = field
            .coefficient_at(2, 2, epoch)
            .expect("C22")
            .expect("stored");
        assert_eq!(c22.c, 2.439_383_573_28e-6);
        assert_eq!(c22.s, -1.400_273_703_85e-6);
    }

    #[test]
    fn epoch_parsing_follows_the_specification() {
        assert_eq!(parse_epoch("20000101.1200"), Some(0.0));
        assert_eq!(parse_epoch("20000101"), Some(-43_200.0));
        assert_eq!(parse_epoch("20000101.0060"), parse_epoch("20000101.0100"));
        assert_eq!(
            parse_epoch("20000229"),
            Some(civil::j2000_seconds(2000, 2, 29, 0, 0, 0.0))
        );
        for bad in [
            "19000229",
            "20000100",
            "20000132",
            "200001010",
            "2000-1-1",
            "20000101.",
            "20000101.0061",
            "+0000101",
        ] {
            assert_eq!(parse_epoch(bad), None, "{bad}");
        }
    }

    #[test]
    fn time_variable_force_model_uses_exact_round_trip_tdb_to_tt_inverse() {
        use crate::astro::time::scales::{tdb_from_tt_j2000_seconds, tt_from_tdb_j2000_seconds};

        // Synthetic icgem2.0 field with a linear trend anchored at J2000.0
        // (20000101.1200) so sub-picosecond TT differences near J2000 show up
        // directly in the evaluated acceleration.
        let gfc = "\
product_type               gravity_field
modelname                  j2000_trend
earth_gravity_constant     3.986004415e+14
radius                     6.378136460e+06
max_degree                 2
errors                     no
tide_system                tide_free
norm                       fully_normalized
format                     icgem2.0
end_of_head
gfc    0    0  1.000000000000e+00  0.000000000000e+00
gfct   2    0  0.000000000000e+00  0.000000000000e+00  20000101.1200  20100101.0000
trnd   2    0  1.000000000000e+10  0.000000000000e+00  20000101.1200  20100101.0000
gfc    2    1  0.000000000000e+00  0.000000000000e+00
gfc    2    2  0.000000000000e+00  0.000000000000e+00
";
        let field = IcgemGravityField::parse(gfc).expect("field");
        let gravity = field.gravity(2, 0).expect("gravity");
        let position = [6900.0, -1300.0, 2100.0];
        let ctx = PropagationContext::default();
        let tt_round_trip = 0.000_123_456_789_f64;
        for tdb in [0.0, 10_001.0, tdb_from_tt_j2000_seconds(tt_round_trip)] {
            let tt = tt_from_tdb_j2000_seconds(tdb);
            let state = CartesianState::new(tdb, position, [0.0, 7.5, 0.0]);
            let actual = gravity.acceleration(&state, &ctx).expect("acceleration");
            let expected = gravity
                .body_fixed_acceleration_at_km_s2(position, tt)
                .expect("at TT");
            assert_eq!(
                [actual.x, actual.y, actual.z],
                expected,
                "tdb {tdb} must use tt_from_tdb_j2000_seconds ({tt})"
            );
        }
    }
}
