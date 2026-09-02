//! Ocean tide loading station displacement (IERS Conventions 2010, §6.2; the
//! Bos-Scherneck BLQ convention), via the IERS `ARG2` 11-constituent
//! astronomical-argument method.
//!
//! Scope: this is the ARG2 main-constituent method (the 11 BLQ constituents
//! below), **not** the full HARDISP admittance scheme. It does not apply the
//! 18.6-yr nodal modulation or interpolate the minor side constituents that
//! HARDISP (e.g. RTKLIB's 342-constituent spline) carries - see the
//! "Astronomical arguments" note. For inland stations the difference is sub-mm
//! (validated against RTKLIB below), but this is deliberately the ARG2
//! approximation, not a HARDISP reimplementation.
//!
//! [`ocean_tide_loading`] computes the displacement of an Earth-fixed (ITRF)
//! station caused by the elastic deformation of the solid Earth under the
//! periodic load of the ocean tide. It is the sibling of
//! [`super::solid_earth_tide`] and [`super::solid_earth_pole_tide`] and is wired
//! into the PPP correction stack in the identical way: a per-epoch station
//! displacement vector projected onto the line of sight in
//! `precise_positioning/model.rs`.
//!
//! Physics (IERS Conventions 2010, §6.2; the HARDISP / BLQ convention). The
//! site displacement in each of the three BLQ components is the sum over 11
//! tidal constituents (in BLQ column order M2, S2, N2, K2, K1, O1, P1, Q1, Mf,
//! Mm, Ssa) of
//!
//! ```text
//! dc(t) = sum_j  A_cj * cos( arg_j(t) - phi_cj )          (per component c)
//! ```
//!
//! where `A_cj` (m) and `phi_cj` (rad) are the per-station BLQ amplitude and
//! Greenwich phase lag for component `c` and constituent `j`, and `arg_j(t)` is
//! the astronomical (equilibrium) argument of constituent `j` at the epoch.
//! This is the displacement formula the Bos-Scherneck BLQ tables are designed
//! for; RTKLIB's `tide_oload`/`hardisp` is used as the validation oracle, and
//! agreement holds to sub-mm for inland stations (it is not claimed to be
//! bit-identical to HARDISP - the constituent sets differ, see below).
//!
//! Astronomical arguments. `arg_j(t)` is the IERS `ARG2` argument (IERS
//! Conventions 2010 Chapter 7 reference software `ARG2.F`):
//!
//! ```text
//! arg_j = SPEED_j * FDAY + n1_j*h0 + n2_j*s0 + n3_j*p0 + n4_j*2pi   (mod 2pi)
//! ```
//!
//! with `FDAY` the UT seconds of the day, `(h0, s0, p0)` the mean longitudes of
//! the Sun, the Moon, and the lunar perigee at 0h of the day (`ARG2.F` cubic
//! polynomials in `CAPT`, Julian centuries from the 1975 reference epoch),
//! `SPEED_j` the constituent angular speed (rad/s), and `(n1..n4)_j` the
//! `ANGFAC` multipliers. The quarter-cycle `n4_j` entries (`+/-0.25`) are the
//! Schwiderski phase corrections the `cos(arg - phi)` convention requires for
//! the diurnal band. `ARG2` deliberately omits the 18.6-yr nodal modulation and
//! the minor side constituents that the full HARDISP admittance method (e.g.
//! RTKLIB's 342-constituent spline) interpolates; for an inland station the
//! resulting difference is well below the millimetre (verified against RTKLIB in
//! `tests/ocean_loading_oracle.rs`).
//!
//! BLQ components are radial (positive up), tangential EW (positive west), and
//! tangential NS (positive south); the returned vector is the geodetic ENU
//! displacement (east = -west, north = -south, up = radial) rotated to ECEF on
//! the WGS84 ellipsoid, matching RTKLIB's `ecef2pos`/`xyz2enu`.
//!
//! The per-station BLQ coefficients are a data dependency the caller supplies
//! from an ocean-loading provider (Bos-Scherneck / OSO Chalmers, or equivalent);
//! the engine does not embed them and they must not be fabricated.

#[cfg(test)]
mod tests;

use crate::astro::constants::{
    time::SECONDS_PER_HOUR,
    units::{DEG_TO_RAD, KM_TO_M},
};
use crate::astro::frames::transforms::itrs_to_geodetic_compute;
use crate::astro::math::vec3::norm3_ref as norm;
use crate::validate;
use std::fmt::Write as _;

use super::{gregorian_to_two_part_julian_date, invalid_tide_input, BlqParseErrorKind, TideError};

/// Number of BLQ tidal constituents (M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa).
pub const NUM_OCEAN_CONSTITUENTS: usize = 11;

/// Two pi (cycle of an astronomical argument).
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// BLQ tidal constituents supported by the ARG2 evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OceanTideConstituent {
    /// First BLQ/ARG2 slot (index 0), labeled `M2`; its ARG2 speed is
    /// `1.40519e-4` rad/s with multipliers `(h0, s0, p0, 2pi) = (2, -2, 0, 0)`.
    M2,
    /// Second BLQ/ARG2 slot (index 1), labeled `S2`; its ARG2 speed is
    /// `1.45444e-4` rad/s and its multiplier row is all zero, so its argument
    /// advances with the fractional hour alone.
    S2,
    /// Third BLQ/ARG2 slot (index 2), labeled `N2`; its ARG2 speed is
    /// `1.37880e-4` rad/s with multipliers `(h0, s0, p0, 2pi) = (2, -3, 1, 0)`.
    N2,
    /// Fourth BLQ/ARG2 slot (index 3), labeled `K2`; its ARG2 speed is
    /// `1.45842e-4` rad/s with multipliers `(h0, s0, p0, 2pi) = (2, 0, 0, 0)`.
    K2,
    /// Fifth BLQ/ARG2 slot (index 4), labeled `K1`; its ARG2 speed is
    /// `0.72921e-4` rad/s with multipliers `(h0, s0, p0, 2pi) = (1, 0, 0, 0.25)`.
    K1,
    /// Sixth BLQ/ARG2 slot (index 5), labeled `O1`; its ARG2 speed is
    /// `0.67598e-4` rad/s with multipliers `(h0, s0, p0, 2pi) = (1, -2, 0, -0.25)`.
    O1,
    /// Seventh BLQ/ARG2 slot (index 6), labeled `P1`; its ARG2 speed is
    /// `0.72523e-4` rad/s with multipliers `(h0, s0, p0, 2pi) = (-1, 0, 0, -0.25)`.
    P1,
    /// Eighth BLQ/ARG2 slot (index 7), labeled `Q1`; its ARG2 speed is
    /// `0.64959e-4` rad/s with multipliers `(h0, s0, p0, 2pi) = (1, -3, 1, -0.25)`.
    Q1,
    /// Ninth BLQ/ARG2 slot (index 8), labeled `Mf`; its ARG2 speed is
    /// `0.053234e-4` rad/s with multipliers `(h0, s0, p0, 2pi) = (0, 2, 0, 0)`.
    Mf,
    /// Tenth BLQ/ARG2 slot (index 9), labeled `Mm`; its ARG2 speed is
    /// `0.026392e-4` rad/s with multipliers `(h0, s0, p0, 2pi) = (0, 1, -1, 0)`.
    Mm,
    /// Eleventh BLQ/ARG2 slot (index 10), labeled `Ssa`; its ARG2 speed is
    /// `0.003982e-4` rad/s with multipliers `(h0, s0, p0, 2pi) = (2, 0, 0, 0)`.
    Ssa,
}

impl OceanTideConstituent {
    /// Standard BLQ constituent label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::M2 => "M2",
            Self::S2 => "S2",
            Self::N2 => "N2",
            Self::K2 => "K2",
            Self::K1 => "K1",
            Self::O1 => "O1",
            Self::P1 => "P1",
            Self::Q1 => "Q1",
            Self::Mf => "Mf",
            Self::Mm => "Mm",
            Self::Ssa => "Ssa",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::M2 => 0,
            Self::S2 => 1,
            Self::N2 => 2,
            Self::K2 => 3,
            Self::K1 => 4,
            Self::O1 => 5,
            Self::P1 => 6,
            Self::Q1 => 7,
            Self::Mf => 8,
            Self::Mm => 9,
            Self::Ssa => 10,
        }
    }

    fn from_label(label: &str) -> Option<Self> {
        match label {
            "M2" => Some(Self::M2),
            "S2" => Some(Self::S2),
            "N2" => Some(Self::N2),
            "K2" => Some(Self::K2),
            "K1" => Some(Self::K1),
            "O1" => Some(Self::O1),
            "P1" => Some(Self::P1),
            "Q1" => Some(Self::Q1),
            "MF" => Some(Self::Mf),
            "MM" => Some(Self::Mm),
            "SSA" => Some(Self::Ssa),
            _ => None,
        }
    }
}

/// Standard BLQ column order.
pub const OCEAN_LOADING_CONSTITUENTS: [OceanTideConstituent; NUM_OCEAN_CONSTITUENTS] = [
    OceanTideConstituent::M2,
    OceanTideConstituent::S2,
    OceanTideConstituent::N2,
    OceanTideConstituent::K2,
    OceanTideConstituent::K1,
    OceanTideConstituent::O1,
    OceanTideConstituent::P1,
    OceanTideConstituent::Q1,
    OceanTideConstituent::Mf,
    OceanTideConstituent::Mm,
    OceanTideConstituent::Ssa,
];

/// IERS `ARG2.F` constituent angular speeds (rad/s), BLQ column order
/// M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa.
const SPEED_RAD_S: [f64; NUM_OCEAN_CONSTITUENTS] = [
    1.405_19e-4,
    1.454_44e-4,
    1.378_80e-4,
    1.458_42e-4,
    0.729_21e-4,
    0.675_98e-4,
    0.725_23e-4,
    0.649_59e-4,
    0.053_234e-4,
    0.026_392e-4,
    0.003_982e-4,
];

/// IERS `ARG2.F` `ANGFAC` multipliers `(h0, s0, p0, 2pi)` per constituent. The
/// fourth column is the quarter-cycle Schwiderski phase correction.
#[rustfmt::skip]
const ANGFAC: [[f64; 4]; NUM_OCEAN_CONSTITUENTS] = [
    [ 2.0, -2.0,  0.0,  0.00], // M2
    [ 0.0,  0.0,  0.0,  0.00], // S2
    [ 2.0, -3.0,  1.0,  0.00], // N2
    [ 2.0,  0.0,  0.0,  0.00], // K2
    [ 1.0,  0.0,  0.0,  0.25], // K1
    [ 1.0, -2.0,  0.0, -0.25], // O1
    [-1.0,  0.0,  0.0, -0.25], // P1
    [ 1.0, -3.0,  1.0, -0.25], // Q1
    [ 0.0,  2.0,  0.0,  0.00], // Mf
    [ 0.0,  1.0, -1.0,  0.00], // Mm
    [ 2.0,  0.0,  0.0,  0.00], // Ssa
];

/// Per-station ocean-loading BLQ coefficients (Bos-Scherneck / HARDISP format).
///
/// Both arrays are indexed `[component][constituent]`. The component order is
/// the BLQ row order: radial / up-positive (0), tangential EW / west-positive
/// (1), tangential NS / south-positive (2). The constituent order is the BLQ
/// column order M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OceanLoadingBlq {
    /// Constituent amplitudes (m).
    pub amplitude_m: [[f64; NUM_OCEAN_CONSTITUENTS]; 3],
    /// Constituent Greenwich phase lags (degrees, positive lag).
    pub phase_deg: [[f64; NUM_OCEAN_CONSTITUENTS]; 3],
}

/// One parsed standard BLQ station block.
#[derive(Debug, Clone, PartialEq)]
pub struct OceanLoadingBlqBlock {
    /// Station identifier line from the BLQ block.
    pub station: String,
    /// Parsed and reordered BLQ coefficients.
    pub coefficients: OceanLoadingBlq,
}

impl OceanLoadingBlqBlock {
    /// Format as a standard six-row BLQ block in the supported constituent order.
    #[must_use]
    pub fn to_blq_block(&self) -> String {
        let mut out = String::new();
        let labels = OCEAN_LOADING_CONSTITUENTS
            .iter()
            .map(|constituent| constituent.label())
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(out, "$$ Column order: {labels}");
        let _ = writeln!(out, "{}", self.station);
        for row in self.coefficients.amplitude_m {
            write_blq_row(&mut out, row);
        }
        for row in self.coefficients.phase_deg {
            write_blq_row(&mut out, row);
        }
        out
    }
}

impl OceanLoadingBlq {
    /// Parse a single standard BLQ station block.
    pub fn from_blq_block(text: &str) -> Result<OceanLoadingBlqBlock, TideError> {
        parse_ocean_loading_blq_block(text)
    }
}

/// Parse one standard Bos-Scherneck/HARDISP BLQ station block.
pub fn parse_ocean_loading_blq_block(text: &str) -> Result<OceanLoadingBlqBlock, TideError> {
    let mut blocks = parse_ocean_loading_blq_blocks(text)?;
    match blocks.len() {
        1 => Ok(blocks.remove(0)),
        0 => Err(TideError::BlqParse {
            line: 0,
            kind: BlqParseErrorKind::Empty,
        }),
        _ => Err(TideError::BlqParse {
            line: 0,
            kind: BlqParseErrorKind::MultipleBlocks {
                found: blocks.len(),
            },
        }),
    }
}

/// Parse all standard station blocks in a BLQ file.
pub fn parse_ocean_loading_blq_blocks(text: &str) -> Result<Vec<OceanLoadingBlqBlock>, TideError> {
    let mut blocks = Vec::new();
    let mut station: Option<(usize, String)> = None;
    let mut rows: Vec<[f64; NUM_OCEAN_CONSTITUENTS]> = Vec::new();
    let mut column_order = OCEAN_LOADING_CONSTITUENTS;
    let mut saw_content = false;

    for (idx, raw_line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        saw_content = true;

        if let Some(order) = parse_constituent_header(trimmed, line_no)? {
            column_order = order;
            continue;
        }
        if is_blq_comment(trimmed) {
            continue;
        }

        if station.is_none() {
            if looks_like_numeric_row(trimmed) {
                return Err(TideError::BlqParse {
                    line: line_no,
                    kind: BlqParseErrorKind::MissingStation,
                });
            }
            station = Some((line_no, trimmed.to_string()));
            rows.clear();
            continue;
        }

        if !looks_like_numeric_row(trimmed) {
            return Err(TideError::BlqParse {
                line: line_no,
                kind: BlqParseErrorKind::InvalidNumber {
                    token: trimmed.to_string(),
                },
            });
        }

        let row = parse_blq_numeric_row(trimmed, line_no, column_order)?;
        rows.push(row);
        if rows.len() > 6 {
            let station_name = station
                .as_ref()
                .map(|(_, name)| name.clone())
                .unwrap_or_default();
            return Err(TideError::BlqParse {
                line: line_no,
                kind: BlqParseErrorKind::TooManyCoefficientRows {
                    station: station_name,
                },
            });
        }
        if rows.len() == 6 {
            let (_, station_name) = station.take().expect("station present");
            blocks.push(block_from_rows(station_name, &rows));
            rows.clear();
        }
    }

    if !saw_content {
        return Err(TideError::BlqParse {
            line: 0,
            kind: BlqParseErrorKind::Empty,
        });
    }
    if let Some((line, station_name)) = station {
        return Err(TideError::BlqParse {
            line,
            kind: BlqParseErrorKind::MissingCoefficientRows {
                station: station_name,
                expected: 6,
                found: rows.len(),
            },
        });
    }

    Ok(blocks)
}

fn block_from_rows(
    station: String,
    rows: &[[f64; NUM_OCEAN_CONSTITUENTS]],
) -> OceanLoadingBlqBlock {
    let mut amplitude_m = [[0.0_f64; NUM_OCEAN_CONSTITUENTS]; 3];
    let mut phase_deg = [[0.0_f64; NUM_OCEAN_CONSTITUENTS]; 3];
    amplitude_m.copy_from_slice(&rows[0..3]);
    phase_deg.copy_from_slice(&rows[3..6]);
    OceanLoadingBlqBlock {
        station,
        coefficients: OceanLoadingBlq {
            amplitude_m,
            phase_deg,
        },
    }
}

fn write_blq_row(out: &mut String, row: [f64; NUM_OCEAN_CONSTITUENTS]) {
    for value in row {
        let _ = write!(out, " {value:>16}");
    }
    out.push('\n');
}

fn is_blq_comment(line: &str) -> bool {
    line.starts_with('$') || line.starts_with('#') || line.starts_with('!')
}

fn looks_like_numeric_row(line: &str) -> bool {
    line.split_whitespace().next().is_some_and(|token| {
        parse_blq_float_token(token).is_ok()
            || token
                .chars()
                .next()
                .is_some_and(|c| c == '+' || c == '-' || c == '.')
    })
}

fn parse_blq_numeric_row(
    line: &str,
    line_no: usize,
    column_order: [OceanTideConstituent; NUM_OCEAN_CONSTITUENTS],
) -> Result<[f64; NUM_OCEAN_CONSTITUENTS], TideError> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != NUM_OCEAN_CONSTITUENTS {
        return Err(TideError::BlqParse {
            line: line_no,
            kind: BlqParseErrorKind::WrongColumnCount {
                expected: NUM_OCEAN_CONSTITUENTS,
                found: tokens.len(),
            },
        });
    }

    let mut row = [0.0_f64; NUM_OCEAN_CONSTITUENTS];
    for (source_index, token) in tokens.iter().enumerate() {
        let value = parse_blq_float_token(token).map_err(|kind| TideError::BlqParse {
            line: line_no,
            kind,
        })?;
        row[column_order[source_index].index()] = value;
    }
    Ok(row)
}

fn parse_blq_float_token(token: &str) -> Result<f64, BlqParseErrorKind> {
    let normalized = token.replace('D', "E").replace('d', "e");
    let value = normalized
        .parse::<f64>()
        .map_err(|_| BlqParseErrorKind::InvalidNumber {
            token: token.to_string(),
        })?;
    if !value.is_finite() {
        return Err(BlqParseErrorKind::NonFiniteNumber {
            token: token.to_string(),
        });
    }
    Ok(value)
}

fn parse_constituent_header(
    line: &str,
    line_no: usize,
) -> Result<Option<[OceanTideConstituent; NUM_OCEAN_CONSTITUENTS]>, TideError> {
    if line
        .split_whitespace()
        .all(|token| parse_blq_float_token(token).is_ok())
    {
        return Ok(None);
    }

    let upper = line.to_ascii_uppercase();
    let header_hint = upper.contains("COLUMN") || upper.contains("CONSTITUENT");
    let labels = line
        .split_whitespace()
        .map(normalize_constituent_token)
        .filter(|token| is_constituent_like(token))
        .collect::<Vec<_>>();
    if labels.is_empty() {
        return Ok(None);
    }
    if labels.len() != NUM_OCEAN_CONSTITUENTS && !header_hint {
        return Ok(None);
    }
    if labels.len() != NUM_OCEAN_CONSTITUENTS {
        return Err(TideError::BlqParse {
            line: line_no,
            kind: BlqParseErrorKind::WrongColumnCount {
                expected: NUM_OCEAN_CONSTITUENTS,
                found: labels.len(),
            },
        });
    }

    let mut order = [OceanTideConstituent::M2; NUM_OCEAN_CONSTITUENTS];
    let mut seen = [false; NUM_OCEAN_CONSTITUENTS];
    for (idx, label) in labels.iter().enumerate() {
        let Some(constituent) = OceanTideConstituent::from_label(label) else {
            return Err(TideError::BlqParse {
                line: line_no,
                kind: BlqParseErrorKind::UnsupportedConstituent {
                    constituent: label.clone(),
                },
            });
        };
        let constituent_index = constituent.index();
        if seen[constituent_index] {
            return Err(TideError::BlqParse {
                line: line_no,
                kind: BlqParseErrorKind::DuplicateConstituent {
                    constituent: constituent.label().to_string(),
                },
            });
        }
        seen[constituent_index] = true;
        order[idx] = constituent;
    }
    Ok(Some(order))
}

fn normalize_constituent_token(token: &str) -> String {
    token
        .trim_matches(|c: char| {
            c == '$'
                || c == '#'
                || c == '!'
                || c == ':'
                || c == ';'
                || c == ','
                || c == '('
                || c == ')'
                || c == '['
                || c == ']'
        })
        .to_ascii_uppercase()
}

fn is_constituent_like(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    OceanTideConstituent::from_label(token).is_some()
        || matches!(
            token,
            "MSF" | "M4" | "MS4" | "MN4" | "SA" | "2N2" | "L2" | "T2"
        )
        || (token.len() <= 4
            && token.chars().any(|c| c.is_ascii_digit())
            && token.chars().all(|c| c.is_ascii_alphanumeric()))
}

/// Ocean tide loading displacement of an ITRF station, in metres (ECEF).
///
/// Arguments:
/// * `xsta` - geocentric station position (m, ITRF).
/// * `year`, `month`, `day` - UTC calendar date (selects the day of year).
/// * `fhr` - UTC fractional hour of the day (`hour + min/60 + sec/3600`).
/// * `blq` - the station's BLQ ocean-loading coefficients (a data dependency the
///   caller supplies; the engine does not embed them).
///
/// Returns the displacement vector (m, geocentric ITRF), to be projected onto
/// the line of sight identically to [`super::solid_earth_tide`].
///
/// Returns [`TideError`] when inputs are non-finite, the date/hour is invalid,
/// the BLQ coefficients are non-finite, or the station vector is degenerate
/// (zero radius).
pub fn ocean_tide_loading(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    blq: &OceanLoadingBlq,
) -> Result<[f64; 3], TideError> {
    validate_ocean_loading_domain(xsta, year, month, day, fhr, blq)?;
    Ok(ocean_tide_loading_unchecked(
        xsta, year, month, day, fhr, blq,
    ))
}

fn validate_ocean_loading_domain(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    blq: &OceanLoadingBlq,
) -> Result<(), TideError> {
    validate::finite_vec3(*xsta, "station position").map_err(invalid_tide_input)?;
    validate::civil_datetime_with_second_policy(
        i64::from(year),
        i64::from(month),
        i64::from(day),
        0,
        0,
        0.0,
        validate::CivilSecondPolicy::Continuous,
    )
    .map_err(invalid_tide_input)?;
    validate::finite_in_range_exclusive_upper(fhr, 0.0, 24.0, "fractional hour")
        .map_err(invalid_tide_input)?;

    for component in &blq.amplitude_m {
        for &amplitude in component {
            validate::finite(amplitude, "ocean loading amplitude").map_err(invalid_tide_input)?;
        }
    }
    for component in &blq.phase_deg {
        for &phase in component {
            validate::finite(phase, "ocean loading phase").map_err(invalid_tide_input)?;
        }
    }

    validate::finite_positive(norm(xsta), "station radius").map_err(invalid_tide_input)?;

    Ok(())
}

fn ocean_tide_loading_unchecked(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    blq: &OceanLoadingBlq,
) -> [f64; 3] {
    let arg = arg2_angles(year, month, day, fhr);

    // BLQ component sums: 0 = radial (up), 1 = EW (west), 2 = NS (south).
    let mut component = [0.0_f64; 3];
    for (slot, (amplitudes, phases)) in component
        .iter_mut()
        .zip(blq.amplitude_m.iter().zip(blq.phase_deg.iter()))
    {
        let mut sum = 0.0;
        for ((&amplitude, &phase_deg), &a) in amplitudes.iter().zip(phases).zip(&arg) {
            sum += amplitude * libm::cos(a - phase_deg * DEG_TO_RAD);
        }
        *slot = sum;
    }
    let up = component[0];
    let west = component[1];
    let south = component[2];
    let east = -west;
    let north = -south;

    // Geodetic (WGS84) ENU -> ECEF, matching RTKLIB ecef2pos/xyz2enu.
    let (lat_deg, lon_deg, _height_km) =
        itrs_to_geodetic_compute(xsta[0] / KM_TO_M, xsta[1] / KM_TO_M, xsta[2] / KM_TO_M)
            .expect("validated station position yields geodetic coordinates");
    let (sinlat, coslat) = libm::sincos(lat_deg * DEG_TO_RAD);
    let (sinlon, coslon) = libm::sincos(lon_deg * DEG_TO_RAD);

    // ENU basis vectors expressed in ECEF (geodetic topocentric frame):
    //   e = [-sinlon, coslon, 0]
    //   n = [-sinlat coslon, -sinlat sinlon, coslat]
    //   u = [ coslat coslon,  coslat sinlon, sinlat]
    [
        east * (-sinlon) + north * (-sinlat * coslon) + up * (coslat * coslon),
        east * coslon + north * (-sinlat * sinlon) + up * (coslat * sinlon),
        north * coslat + up * sinlat,
    ]
}

/// IERS `ARG2.F` astronomical arguments (radians) of the 11 BLQ constituents at
/// the given UTC epoch.
fn arg2_angles(year: i32, month: i32, day: i32, fhr: f64) -> [f64; NUM_OCEAN_CONSTITUENTS] {
    let doy = day_of_year(year, month, day);
    // `DAY` of ARG2 is the fractional day of year; `ID` its integer part and
    // `FDAY` the seconds into the day, i.e. `(DAY - ID) * 86400 = fhr * 3600`.
    let fday = fhr * SECONDS_PER_HOUR;

    // ARG2.F day count and Julian centuries from the 1975 reference epoch.
    // Fortran integer division (truncating toward zero, == floor for years
    // >= 1973, the supported range) is reproduced by Rust's `/` on i32.
    let icapd = doy + 365 * (year - 1975) + (year - 1973) / 4;
    let capt = (27_392.500_528 + 1.000_000_035 * f64::from(icapd)) / 36_525.0;

    // Mean longitudes (rad). ARG2.F uses a truncated DTR; the exact PI/180 used
    // here is sub-femtometre different and is closer to the rigorous argument.
    let h0 = (279.696_68 + (36_000.768_930_485 + 3.03e-4 * capt) * capt) * DEG_TO_RAD;
    let s0 = (((1.9e-6 * capt - 0.001_133) * capt + 481_267.883_141_37) * capt + 270.434_358)
        * DEG_TO_RAD;
    let p0 = (((-1.2e-5 * capt - 0.010_325) * capt + 4_069.034_032_957_7) * capt + 334.329_653)
        * DEG_TO_RAD;

    let mut angle = [0.0_f64; NUM_OCEAN_CONSTITUENTS];
    for (j, slot) in angle.iter_mut().enumerate() {
        let a = SPEED_RAD_S[j] * fday
            + ANGFAC[j][0] * h0
            + ANGFAC[j][1] * s0
            + ANGFAC[j][2] * p0
            + ANGFAC[j][3] * TWO_PI;
        *slot = a.rem_euclid(TWO_PI);
    }
    angle
}

/// 1-based UTC day of year (ARG2 `ID`), from the IERS/SOFA `CAL2JD` MJD diff
/// (the `djm` return is in days, so the difference is the day-of-year minus 1).
fn day_of_year(year: i32, month: i32, day: i32) -> i32 {
    let (_, mjd) = gregorian_to_two_part_julian_date(year, month, day);
    let (_, mjd_jan1) = gregorian_to_two_part_julian_date(year, 1, 1);
    (mjd - mjd_jan1).round() as i32 + 1
}
