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

use super::{
    gregorian_to_two_part_julian_date, invalid_tide_input, BlqParseErrorKind, BlqWriteErrorKind,
    TideError,
};

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

/// Position of a retained comment or header line within its station block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OceanLoadingBlqCommentPlacement {
    /// Before the station line. The parser gives a block every line after the
    /// previous block's last coefficient row, so file header comments belong
    /// to the first block.
    BeforeStation,
    /// Before the zero-based coefficient row, `0..=5`; `BeforeRow(0)` is
    /// between the station line and the first row. A column-order header here
    /// sets the order of this row and every later one.
    BeforeRow(usize),
    /// After the sixth coefficient row. The parser uses it only for lines
    /// that follow the last block of the input, and the writer accepts it
    /// only on the last block it writes.
    AfterRows,
}

/// One comment or header line retained from a BLQ block, as read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OceanLoadingBlqComment {
    /// Where the line sits in the block.
    pub placement: OceanLoadingBlqCommentPlacement,
    /// The line exactly as read, without its line terminator.
    pub line: String,
}

/// One parsed standard BLQ station block.
#[derive(Debug, Clone, PartialEq)]
pub struct OceanLoadingBlqBlock {
    /// Station identifier line from the BLQ block, trimmed.
    pub station: String,
    /// Parsed and reordered BLQ coefficients.
    pub coefficients: OceanLoadingBlq,
    /// Comment and column-order header lines of the block, in input order.
    /// They carry the model, frame, unit and row-direction declarations the
    /// provider writes as comments; the writer restates them unchanged.
    pub comments: Vec<OceanLoadingBlqComment>,
}

impl OceanLoadingBlqBlock {
    /// Format as a standard six-row BLQ block.
    ///
    /// The retained comments are written back at their placements, the station
    /// line starts in the third column (two leading spaces, as in the
    /// provider's files and where RTKLIB `readblq` reads the name), and each
    /// row is written in the column order its retained header declares, or the
    /// standard order when none does. Output reads back to an equal block.
    ///
    /// Returns [`TideError::BlqWrite`] for a station, comment or coefficient
    /// that the parser would not read back unchanged, including comments not
    /// grouped by placement in file order. To write several blocks as one
    /// file, use [`write_ocean_loading_blq_blocks`], which carries a
    /// column-order header across blocks the way the parser does.
    pub fn to_blq_block(&self) -> Result<String, TideError> {
        let mut out = String::new();
        let mut column_order = OCEAN_LOADING_CONSTITUENTS;
        self.write_blq(&mut out, &mut column_order, 0, true)?;
        Ok(out)
    }

    fn write_blq(
        &self,
        out: &mut String,
        column_order: &mut [OceanTideConstituent; NUM_OCEAN_CONSTITUENTS],
        block: usize,
        is_last: bool,
    ) -> Result<(), TideError> {
        let fail = |kind| TideError::BlqWrite { block, kind };
        validate_blq_station(&self.station).map_err(fail)?;
        let rows = self.coefficient_rows();
        for (row_index, row) in rows.iter().enumerate() {
            for constituent in OCEAN_LOADING_CONSTITUENTS {
                if !row[constituent.index()].is_finite() {
                    return Err(fail(BlqWriteErrorKind::NonFiniteCoefficient {
                        row: row_index,
                        constituent,
                    }));
                }
            }
        }
        let mut headers = Vec::with_capacity(self.comments.len());
        let mut previous_rank = 0;
        for (index, comment) in self.comments.iter().enumerate() {
            headers.push(validate_blq_comment(index, comment).map_err(fail)?);
            let rank = placement_rank(comment.placement);
            if rank < previous_rank {
                return Err(fail(BlqWriteErrorKind::CommentsOutOfPlacementOrder {
                    index,
                }));
            }
            previous_rank = rank;
            if !is_last && comment.placement == OceanLoadingBlqCommentPlacement::AfterRows {
                return Err(fail(BlqWriteErrorKind::AfterRowsBeforeAnotherBlock {
                    index,
                }));
            }
        }

        let placements = std::iter::once(OceanLoadingBlqCommentPlacement::BeforeStation);
        self.write_comments(out, column_order, &headers, placements);
        let _ = writeln!(out, "  {}", self.station);
        for (row_index, row) in rows.iter().enumerate() {
            let placement = OceanLoadingBlqCommentPlacement::BeforeRow(row_index);
            self.write_comments(out, column_order, &headers, std::iter::once(placement));
            write_blq_row(out, row, column_order);
        }
        let placements = std::iter::once(OceanLoadingBlqCommentPlacement::AfterRows);
        self.write_comments(out, column_order, &headers, placements);
        Ok(())
    }

    /// Write the retained lines at `placements`, in retained order, applying
    /// each header's column order as it is passed.
    fn write_comments(
        &self,
        out: &mut String,
        column_order: &mut [OceanTideConstituent; NUM_OCEAN_CONSTITUENTS],
        headers: &[Option<[OceanTideConstituent; NUM_OCEAN_CONSTITUENTS]>],
        placements: impl Iterator<Item = OceanLoadingBlqCommentPlacement>,
    ) {
        for placement in placements {
            for (comment, header) in self.comments.iter().zip(headers) {
                if comment.placement == placement {
                    if let Some(order) = header {
                        *column_order = *order;
                    }
                    out.push_str(&comment.line);
                    out.push('\n');
                }
            }
        }
    }

    fn coefficient_rows(&self) -> [[f64; NUM_OCEAN_CONSTITUENTS]; 6] {
        let amplitude = self.coefficients.amplitude_m;
        let phase = self.coefficients.phase_deg;
        [
            amplitude[0],
            amplitude[1],
            amplitude[2],
            phase[0],
            phase[1],
            phase[2],
        ]
    }
}

/// Write station blocks as one BLQ file.
///
/// Blocks are written in order with [`OceanLoadingBlqBlock::to_blq_block`]'s
/// layout. A column-order header retained on one block stays in force for the
/// blocks after it, as it does when the parser reads the file, so parsing the
/// output of this function gives back equal blocks. A comment placed after the
/// rows of any block but the last is refused, because the parser reads it as
/// part of the next block.
pub fn write_ocean_loading_blq_blocks(
    blocks: &[OceanLoadingBlqBlock],
) -> Result<String, TideError> {
    let mut out = String::new();
    let mut column_order = OCEAN_LOADING_CONSTITUENTS;
    for (index, block) in blocks.iter().enumerate() {
        block.write_blq(
            &mut out,
            &mut column_order,
            index,
            index + 1 == blocks.len(),
        )?;
    }
    Ok(out)
}

/// Position of a placement in file order, which is the order the parser
/// retains comments in.
fn placement_rank(placement: OceanLoadingBlqCommentPlacement) -> usize {
    match placement {
        OceanLoadingBlqCommentPlacement::BeforeStation => 0,
        OceanLoadingBlqCommentPlacement::BeforeRow(row) => 1 + row,
        OceanLoadingBlqCommentPlacement::AfterRows => 7,
    }
}

/// Refuse a station name the parser would not read back as the same station.
fn validate_blq_station(station: &str) -> Result<(), BlqWriteErrorKind> {
    if station.is_empty() {
        return Err(BlqWriteErrorKind::EmptyStation);
    }
    if station.contains('\n') {
        return Err(BlqWriteErrorKind::StationLineBreak);
    }
    if station.trim() != station {
        return Err(BlqWriteErrorKind::StationSurroundingWhitespace);
    }
    if is_blq_comment(station) {
        return Err(BlqWriteErrorKind::StationReadsAsComment);
    }
    if !matches!(parse_constituent_header(station, 0), Ok(None)) {
        return Err(BlqWriteErrorKind::StationReadsAsHeader);
    }
    if looks_like_numeric_row(station) {
        return Err(BlqWriteErrorKind::StationReadsAsCoefficientRow);
    }
    Ok(())
}

/// Refuse a retained line the parser would not read back as the same comment
/// or header at the same placement; return the column order a header declares.
fn validate_blq_comment(
    index: usize,
    comment: &OceanLoadingBlqComment,
) -> Result<Option<[OceanTideConstituent; NUM_OCEAN_CONSTITUENTS]>, BlqWriteErrorKind> {
    if comment.line.contains('\n') || comment.line.ends_with('\r') {
        return Err(BlqWriteErrorKind::CommentLineBreak { index });
    }
    if let OceanLoadingBlqCommentPlacement::BeforeRow(row) = comment.placement {
        if row > 5 {
            return Err(BlqWriteErrorKind::CommentPlacementOutOfRange { index });
        }
    }
    let trimmed = comment.line.trim();
    if trimmed.is_empty() {
        return Err(BlqWriteErrorKind::NotACommentLine { index });
    }
    let header = parse_constituent_header(trimmed, 0).map_err(|error| match error {
        TideError::BlqParse { kind, .. } => BlqWriteErrorKind::InvalidHeader { index, kind },
        _ => BlqWriteErrorKind::NotACommentLine { index },
    })?;
    if header.is_none() && !is_blq_comment(trimmed) {
        return Err(BlqWriteErrorKind::NotACommentLine { index });
    }
    Ok(header)
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
///
/// Lines starting with `$`, `#` or `!` are comments. A column-order header is
/// a line whose text after any comment markers is `COLUMN ORDER` (any case,
/// optional colon) followed by the constituent labels, or consists only of
/// two or more constituent labels, or is a comment in which a word `ORDER` is
/// followed to the end of the line by two or more constituent labels, such as
/// `$$ Constituent order: S2 M2 ...`. Other comments are prose, even when they
/// list constituents. A header sets the column order for every later row until
/// the next header, including the remaining rows of a block it appears inside.
/// Every label of a header must be one of the eleven supported constituents,
/// each once; anything else is refused by name rather than dropped. Comment
/// and header lines are retained on the block they belong to.
pub fn parse_ocean_loading_blq_blocks(text: &str) -> Result<Vec<OceanLoadingBlqBlock>, TideError> {
    let mut blocks: Vec<OceanLoadingBlqBlock> = Vec::new();
    let mut station: Option<(usize, String)> = None;
    let mut rows: Vec<[f64; NUM_OCEAN_CONSTITUENTS]> = Vec::new();
    let mut comments: Vec<OceanLoadingBlqComment> = Vec::new();
    let mut column_order = OCEAN_LOADING_CONSTITUENTS;
    let mut saw_content = false;

    for (idx, raw_line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        saw_content = true;

        let header = parse_constituent_header(trimmed, line_no)?;
        if header.is_some() || is_blq_comment(trimmed) {
            let placement = if station.is_some() {
                OceanLoadingBlqCommentPlacement::BeforeRow(rows.len())
            } else {
                OceanLoadingBlqCommentPlacement::BeforeStation
            };
            if let Some(order) = header {
                column_order = order;
            }
            comments.push(OceanLoadingBlqComment {
                placement,
                line: raw_line.to_string(),
            });
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
        if rows.len() == 6 {
            let Some((_, station_name)) = station.take() else {
                unreachable!("coefficient rows are only collected after a station line");
            };
            blocks.push(block_from_rows(
                station_name,
                &rows,
                std::mem::take(&mut comments),
            ));
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
    if let Some(last) = blocks.last_mut() {
        last.comments
            .extend(comments.into_iter().map(|comment| OceanLoadingBlqComment {
                placement: OceanLoadingBlqCommentPlacement::AfterRows,
                line: comment.line,
            }));
    }

    Ok(blocks)
}

fn block_from_rows(
    station: String,
    rows: &[[f64; NUM_OCEAN_CONSTITUENTS]],
    comments: Vec<OceanLoadingBlqComment>,
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
        comments,
    }
}

fn write_blq_row(
    out: &mut String,
    row: &[f64; NUM_OCEAN_CONSTITUENTS],
    column_order: &[OceanTideConstituent; NUM_OCEAN_CONSTITUENTS],
) {
    for constituent in column_order {
        let value = row[constituent.index()];
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

/// Recognize a column-order header and return the order it declares.
///
/// Three forms are headers:
///
/// 1. The declared form: `COLUMN ORDER` (any case, optionally followed by a
///    colon) after any comment markers, as in the provider's
///    `$$ COLUMN ORDER:  M2  S2 ...`. Everything after it is the label list.
/// 2. The bare form: a line of two or more tokens that all look like tidal
///    constituent labels, at least one of them a supported one.
/// 3. The ordered comment: a comment line in which a word `ORDER` (any case,
///    optionally followed by a colon) is followed, to the end of the line, by
///    two or more tokens that all look like constituent labels, at least one
///    of them a supported one, such as
///    `$$ Constituent order: S2 M2 ...`. Surrounding punctuation `: ; ( ) [ ] .`
///    is not part of a token.
///
/// Any other line is not a header, including prose that lists constituents
/// without declaring an order, such as
/// `$$ diurnal K1 O1 P1 Q1, semidiurnal M2 S2 N2 K2, long-period MF MM SSA`.
/// The labels of every header form must be exactly the eleven supported
/// constituents, each once, or the header is refused by name.
fn parse_constituent_header(
    line: &str,
    line_no: usize,
) -> Result<Option<[OceanTideConstituent; NUM_OCEAN_CONSTITUENTS]>, TideError> {
    let body = line.trim_start_matches(['$', '#', '!']).trim_start();
    let split_labels = |text: &str| {
        text.split(|c: char| c.is_whitespace() || c == ',')
            .filter(|token| !token.is_empty())
            .map(str::to_ascii_uppercase)
            .collect::<Vec<_>>()
    };

    const DECLARATION: &str = "COLUMN ORDER";
    let declared = body
        .get(..DECLARATION.len())
        .filter(|prefix| prefix.eq_ignore_ascii_case(DECLARATION))
        .map(|_| &body[DECLARATION.len()..])
        .filter(|rest| {
            rest.is_empty() || rest.starts_with(':') || rest.starts_with(char::is_whitespace)
        });
    if let Some(rest) = declared {
        let rest = rest.trim_start();
        let rest = rest.strip_prefix(':').unwrap_or(rest);
        return constituent_order(&split_labels(rest), line_no).map(Some);
    }

    let labels = split_labels(body);
    let is_bare_header = labels.len() >= 2
        && labels.iter().all(|label| is_constituent_like(label))
        && labels
            .iter()
            .any(|label| OceanTideConstituent::from_label(label).is_some());
    if is_bare_header {
        return constituent_order(&labels, line_no).map(Some);
    }

    if is_blq_comment(line) {
        let tokens = labels
            .iter()
            .map(|token| {
                token
                    .trim_matches(|c: char| matches!(c, ':' | ';' | '(' | ')' | '[' | ']' | '.'))
                    .to_string()
            })
            .filter(|token| !token.is_empty())
            .collect::<Vec<_>>();
        for (index, token) in tokens.iter().enumerate() {
            let list = &tokens[index + 1..];
            if token == "ORDER"
                && list.len() >= 2
                && list.iter().all(|label| is_constituent_like(label))
                && list
                    .iter()
                    .any(|label| OceanTideConstituent::from_label(label).is_some())
            {
                return constituent_order(list, line_no).map(Some);
            }
        }
    }
    Ok(None)
}

fn constituent_order(
    labels: &[String],
    line_no: usize,
) -> Result<[OceanTideConstituent; NUM_OCEAN_CONSTITUENTS], TideError> {
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
        if idx < NUM_OCEAN_CONSTITUENTS {
            order[idx] = constituent;
        }
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
    Ok(order)
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
            && token.chars().any(|c| c.is_ascii_alphabetic())
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
