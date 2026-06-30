//! IONEX vertical-TEC grid parser.
//!
//! Parses an IONEX (IONosphere map EXchange) ASCII product into the vertical-TEC
//! grids and the geometry needed to interpolate them: the latitude and longitude
//! node axes, the per-map TEC and (optionally) RMS grids, and the map epochs as
//! UTC instants. The float math that turns the grid into a slant delay lives in
//! [`super::slant`]; this module is the deterministic byte/record reader.
//!
//! IONEX stores TEC as a scaled integer field: the printed value times
//! `10^EXPONENT`. Latitude bands are written north-to-south (a negative `DLAT`),
//! longitude nodes west-to-east (a positive `DLON`), and the longitude span
//! includes the `+180` wrap-seam column. The node axes are reconstructed as
//! `v1 + i * step` so they match the producer's `arange`-style construction
//! bit-for-bit, and the TEC scaling is `field * 10^EXPONENT` formed as a single
//! multiply.

use super::{ionex_epoch_from_j2000_seconds, j2000_seconds_from_instant};
use crate::astro::constants::time::SECONDS_PER_DAY_I64;
use crate::astro::time::civil::j2000_seconds;
use crate::astro::time::model::Instant;
use crate::error::{Error, Result};
use crate::format::columns::{raw_field, raw_field_from};
use crate::format::{Diagnostics, RecordRef, Skip, SkipReason};
use crate::validate;

const IONEX_AXIS_DEG_LIMIT: f64 = 360.0;
const IONEX_AXIS_MAX_NODES: usize = 10_000;
const IONEX_AXIS_MAX_SPAN: f64 = (IONEX_AXIS_MAX_NODES - 1) as f64;

/// A parsed IONEX vertical-TEC product.
///
/// The grids are indexed `[map][i_lat][i_lon]`, with `lat_nodes_deg` descending
/// (north-to-south) and `lon_nodes_deg` ascending (west-to-east). TEC and RMS are
/// in TECU after the `10^EXPONENT` scaling. Map epochs are UTC instants.
#[derive(Debug, Clone, PartialEq)]
pub struct Ionex {
    /// Latitude node values in degrees, descending (north-to-south).
    lat_nodes_deg: Vec<f64>,
    /// Longitude node values in degrees, ascending (west-to-east).
    lon_nodes_deg: Vec<f64>,
    /// Signed latitude step in degrees (negative for the standard ordering).
    dlat_deg: f64,
    /// Signed longitude step in degrees (positive for the standard ordering).
    dlon_deg: f64,
    /// Single-layer shell height in kilometers.
    shell_height_km: f64,
    /// Mean earth radius used by the geometry, in kilometers.
    base_radius_km: f64,
    /// The integer `EXPONENT` header field (TEC scale is `10^EXPONENT`).
    exponent: i32,
    /// Map epochs as UTC instants, ascending.
    map_epochs: Vec<Instant>,
    /// Per-map vertical-TEC grids, indexed `[map][i_lat][i_lon]` (TECU).
    tec_maps: Vec<Vec<Vec<f64>>>,
    /// Per-map RMS grids, indexed `[map][i_lat][i_lon]` (TECU); empty if absent.
    rms_maps: Vec<Vec<Vec<f64>>>,
    /// Count of records skipped during a forgiving parse (e.g. an unsupported
    /// `START OF AUX DATA` block). Lets callers tell a clean product
    /// (`skipped_records == 0`) apart from one carrying records outside this
    /// reader's grid subset, without aborting the whole parse. Mirrors
    /// [`crate::ephemeris::Sp3::skipped_records`].
    skipped_records: usize,
}

impl Ionex {
    /// Latitude node values in degrees (descending, north-to-south).
    pub fn lat_nodes_deg(&self) -> &[f64] {
        &self.lat_nodes_deg
    }

    /// Longitude node values in degrees (ascending, west-to-east).
    pub fn lon_nodes_deg(&self) -> &[f64] {
        &self.lon_nodes_deg
    }

    /// Signed latitude step in degrees (negative for the standard ordering).
    pub fn dlat_deg(&self) -> f64 {
        self.dlat_deg
    }

    /// Signed longitude step in degrees.
    pub fn dlon_deg(&self) -> f64 {
        self.dlon_deg
    }

    /// Single-layer shell height in kilometers.
    pub fn shell_height_km(&self) -> f64 {
        self.shell_height_km
    }

    /// Mean earth radius used by the geometry, in kilometers.
    pub fn base_radius_km(&self) -> f64 {
        self.base_radius_km
    }

    /// The IONEX `EXPONENT` header field; the TEC scale is `10^EXPONENT`.
    pub fn exponent(&self) -> i32 {
        self.exponent
    }

    /// Map epochs as UTC instants (ascending).
    pub fn map_epochs(&self) -> &[Instant] {
        &self.map_epochs
    }

    /// Map epochs projected onto the J2000-second axis (ascending).
    ///
    /// This is a compatibility view for parity tests and callers that need the
    /// integer IONEX epoch axis; the canonical stored representation is
    /// [`Instant`].
    pub fn map_epochs_s(&self) -> Vec<i64> {
        self.map_epochs
            .iter()
            .map(|epoch| {
                j2000_seconds_from_instant(*epoch)
                    .expect("IONEX map epoch is convertible to J2000 seconds")
            })
            .collect()
    }

    /// Per-map vertical-TEC grids, indexed `[map][i_lat][i_lon]` (TECU).
    pub fn tec_maps(&self) -> &[Vec<Vec<f64>>] {
        &self.tec_maps
    }

    /// Per-map RMS grids, indexed `[map][i_lat][i_lon]` (TECU); empty if the
    /// product carries no RMS maps.
    pub fn rms_maps(&self) -> &[Vec<Vec<f64>>] {
        &self.rms_maps
    }

    /// Number of records skipped during a forgiving parse (see the field docs).
    pub fn skipped_records(&self) -> usize {
        self.skipped_records
    }

    /// Return a copy of this product with every map epoch advanced by `days`
    /// whole days (the ionospheric diurnal-persistence shift).
    ///
    /// Only the epoch axis moves; the TEC and RMS grids and all geometry are
    /// copied verbatim. TEC is approximately 24-hour periodic, so re-stamping a
    /// prior day's grids onto a later day reuses the same time-of-day VTEC field
    /// for that later day. The shift is whole days only (`days * 86400 s`); no
    /// value is interpolated across the diurnal cycle. Used by the product
    /// selection layer when the exact day's product is absent.
    pub(crate) fn with_map_epochs_shifted_days(&self, days: i64) -> Result<Self> {
        let shift_s = days.checked_mul(SECONDS_PER_DAY_I64).ok_or_else(|| {
            Error::InvalidInput("IONEX diurnal-shift day count overflows seconds".into())
        })?;
        let mut shifted = self.clone();
        for epoch in &mut shifted.map_epochs {
            let seconds = j2000_seconds_from_instant(*epoch).ok_or_else(|| {
                Error::Parse("IONEX map epoch cannot be projected onto J2000 seconds".into())
            })?;
            let target = seconds.checked_add(shift_s).ok_or_else(|| {
                Error::InvalidInput("IONEX diurnal-shifted map epoch overflows".into())
            })?;
            *epoch = ionex_epoch_from_j2000_seconds(target);
        }
        Ok(shifted)
    }

    /// Parse an IONEX product from its bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let text = core::str::from_utf8(bytes)
            .map_err(|_| Error::Parse("IONEX is not valid UTF-8".into()))?;
        Self::parse_str(text)
    }

    /// Parse an IONEX product from its text.
    pub fn parse_str(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        let mut diagnostics = Diagnostics::new();
        let mut line_number = 0usize;

        // ---- Header ----
        let mut lat1 = None;
        let mut lat2 = None;
        let mut dlat = None;
        let mut lon1 = None;
        let mut lon2 = None;
        let mut dlon = None;
        let mut shell_height_km = None;
        let mut base_radius_km = None;
        let mut exponent: i32 = -1; // IONEX default when no EXPONENT record.

        for line in lines.by_ref() {
            line_number += 1;
            let label = label_of(line);
            match label {
                "LAT1 / LAT2 / DLAT" => {
                    let (a, b, c) = three_fields(line, "LAT1 / LAT2 / DLAT")?;
                    lat1 = Some(a);
                    lat2 = Some(b);
                    dlat = Some(c);
                }
                "LON1 / LON2 / DLON" => {
                    let (a, b, c) = three_fields(line, "LON1 / LON2 / DLON")?;
                    lon1 = Some(a);
                    lon2 = Some(b);
                    dlon = Some(c);
                }
                "HGT1 / HGT2 / DHGT" => {
                    let (a, _b, _c) = three_fields(line, "HGT1 / HGT2 / DHGT")?;
                    shell_height_km = Some(a);
                }
                "BASE RADIUS" => {
                    base_radius_km = Some(first_field(line, "BASE RADIUS")?);
                }
                "EXPONENT" => {
                    exponent = first_field_int(line, "EXPONENT")?;
                }
                "END OF HEADER" => break,
                _ => {}
            }
        }

        let lat1 = lat1.ok_or_else(|| Error::Parse("IONEX missing LAT1 / LAT2 / DLAT".into()))?;
        let lat2 = lat2.unwrap();
        let dlat = dlat.unwrap();
        let lon1 = lon1.ok_or_else(|| Error::Parse("IONEX missing LON1 / LON2 / DLON".into()))?;
        let lon2 = lon2.unwrap();
        let dlon = dlon.unwrap();
        let shell_height_km = shell_height_km
            .ok_or_else(|| Error::Parse("IONEX missing HGT1 / HGT2 / DHGT".into()))?;
        let base_radius_km =
            base_radius_km.ok_or_else(|| Error::Parse("IONEX missing BASE RADIUS".into()))?;

        let lat_nodes_deg = node_axis(lat1, lat2, dlat)?;
        let lon_nodes_deg = node_axis(lon1, lon2, dlon)?;
        let nlat = lat_nodes_deg.len();
        let nlon = lon_nodes_deg.len();

        // TEC scale `10^EXPONENT`, formed as a single multiply per field.
        let scale = 10f64.powi(exponent);

        // ---- Body ----
        let mut map_epochs = Vec::new();
        let mut tec_maps: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut rms_maps: Vec<Vec<Vec<f64>>> = Vec::new();

        // The reader is line-driven: a START record opens a map, an EPOCH record
        // sets its time, each LAT/LON1/LON2/DLON/H record opens a latitude band
        // whose TEC values are read from the following continuation lines.
        let mut cur_kind: Option<MapKind> = None;
        let mut cur_grid: Vec<Vec<f64>> = Vec::new();
        let mut cur_epoch: Option<Instant> = None;
        let mut band_vals: Vec<f64> = Vec::new();
        let mut reading_band = false;

        for line in lines {
            line_number += 1;
            let label = label_of(line);
            match label {
                "START OF AUX DATA" => {
                    // Auxiliary data blocks (e.g. satellite/station DCBs) are
                    // outside this reader's vertical-TEC grid subset. Record a
                    // typed skip rather than silently dropping the block; the body
                    // lines that follow carry no recognized grid label and are
                    // tolerated by the catch-all arm below.
                    diagnostics.push_skip(Skip {
                        at: RecordRef::at_line(line_number),
                        reason: SkipReason::UnsupportedRecordType("AUX DATA"),
                    });
                }
                "END OF AUX DATA" => {}
                "START OF TEC MAP" => {
                    cur_kind = Some(MapKind::Tec);
                    cur_grid = Vec::with_capacity(nlat);
                    cur_epoch = None;
                }
                "START OF RMS MAP" => {
                    cur_kind = Some(MapKind::Rms);
                    cur_grid = Vec::with_capacity(nlat);
                    cur_epoch = None;
                }
                "EPOCH OF CURRENT MAP" => {
                    cur_epoch = Some(parse_epoch_instant(line)?);
                }
                "LAT/LON1/LON2/DLON/H" => {
                    if reading_band {
                        finish_band(&mut cur_grid, &mut band_vals, nlon)?;
                    }
                    reading_band = true;
                    band_vals = Vec::with_capacity(nlon);
                }
                "END OF TEC MAP" | "END OF RMS MAP" => {
                    if reading_band {
                        finish_band(&mut cur_grid, &mut band_vals, nlon)?;
                        reading_band = false;
                    }
                    if cur_grid.len() != nlat {
                        return Err(Error::Parse(format!(
                            "IONEX map has {} latitude bands, expected {nlat}",
                            cur_grid.len()
                        )));
                    }
                    match cur_kind {
                        Some(MapKind::Tec) => {
                            let ep = cur_epoch.ok_or_else(|| {
                                Error::Parse("IONEX TEC map missing EPOCH OF CURRENT MAP".into())
                            })?;
                            map_epochs.push(ep);
                            tec_maps.push(core::mem::take(&mut cur_grid));
                        }
                        Some(MapKind::Rms) => {
                            rms_maps.push(core::mem::take(&mut cur_grid));
                        }
                        None => {
                            return Err(Error::Parse("IONEX END OF MAP without START".into()));
                        }
                    }
                    cur_kind = None;
                }
                _ => {
                    // A continuation line holding the TEC/RMS integer fields of
                    // the current latitude band, or an unrecognized header line.
                    if reading_band {
                        for tok in line.split_whitespace() {
                            let v: i64 = tok.parse().map_err(|_| {
                                Error::Parse(format!("IONEX TEC field unparsable: {tok:?}"))
                            })?;
                            band_vals.push(v as f64 * scale);
                        }
                    }
                }
            }
        }

        if reading_band {
            finish_band(&mut cur_grid, &mut band_vals, nlon)?;
        }
        if let Some(kind) = cur_kind {
            return Err(Error::Parse(format!(
                "IONEX {} map truncated before END OF {} MAP",
                kind.label(),
                kind.label()
            )));
        }

        if tec_maps.is_empty() {
            return Err(Error::Parse("IONEX has no TEC maps".into()));
        }
        // Bilinear interpolation brackets a cell with `node[i+1]` / `node[j+1]`,
        // so each axis needs at least two nodes. Reject a degenerate grid here
        // rather than letting evaluation index past the end.
        if lat_nodes_deg.len() < 2 || lon_nodes_deg.len() < 2 {
            return Err(Error::Parse(format!(
                "IONEX grid needs >= 2 nodes per axis (got {} lat, {} lon)",
                lat_nodes_deg.len(),
                lon_nodes_deg.len()
            )));
        }
        if !rms_maps.is_empty() && rms_maps.len() != tec_maps.len() {
            return Err(Error::Parse(
                "IONEX RMS map count does not match TEC map count".into(),
            ));
        }
        validate_map_epochs_strictly_increasing(&map_epochs)?;

        Ok(Self {
            lat_nodes_deg,
            lon_nodes_deg,
            dlat_deg: dlat,
            dlon_deg: dlon,
            shell_height_km,
            base_radius_km,
            exponent,
            map_epochs,
            tec_maps,
            rms_maps,
            skipped_records: diagnostics.skips.len(),
        })
    }
}

fn validate_map_epochs_strictly_increasing(map_epochs: &[Instant]) -> Result<()> {
    let mut previous_s = None;
    for (index, &epoch) in map_epochs.iter().enumerate() {
        let seconds = j2000_seconds_from_instant(epoch).ok_or_else(|| {
            Error::Parse(format!(
                "IONEX map epoch {index} cannot be projected onto J2000 seconds"
            ))
        })?;
        if previous_s.is_some_and(|previous| seconds <= previous) {
            return Err(Error::Parse(
                "IONEX map epochs must be strictly increasing".into(),
            ));
        }
        previous_s = Some(seconds);
    }
    Ok(())
}

/// Whether the current map being read is a TEC or an RMS map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapKind {
    Tec,
    Rms,
}

impl MapKind {
    fn label(self) -> &'static str {
        match self {
            Self::Tec => "TEC",
            Self::Rms => "RMS",
        }
    }
}

/// Move an accumulated latitude band into the current grid, validating width.
fn finish_band(grid: &mut Vec<Vec<f64>>, band: &mut Vec<f64>, nlon: usize) -> Result<()> {
    if band.len() != nlon {
        return Err(Error::Parse(format!(
            "IONEX latitude band has {} values, expected {nlon}",
            band.len()
        )));
    }
    grid.push(core::mem::take(band));
    Ok(())
}

/// The 20-character label field of an IONEX record (columns 60..80), trimmed.
///
/// Uses [`crate::format::columns::raw_field_from`] for the column-60 window so a
/// multibyte character before the offset is floored to a char boundary rather than
/// panicking on a non-boundary byte slice; on valid ASCII records the window is
/// unchanged.
fn label_of(line: &str) -> &str {
    if line.len() <= 60 {
        line.trim()
    } else {
        raw_field_from(line, 60).trim()
    }
}

/// Parse the first whitespace-delimited float of the data portion of a record.
fn first_field(line: &str, label: &str) -> Result<f64> {
    let data = data_of(line);
    data.split_whitespace()
        .next()
        .ok_or_else(|| Error::Parse(format!("IONEX {label} record empty")))?
        .parse()
        .map_err(|_| Error::Parse(format!("IONEX {label} field unparsable")))
}

/// Parse the first whitespace-delimited integer of the data portion.
fn first_field_int(line: &str, label: &str) -> Result<i32> {
    let data = data_of(line);
    data.split_whitespace()
        .next()
        .ok_or_else(|| Error::Parse(format!("IONEX {label} record empty")))?
        .parse()
        .map_err(|_| Error::Parse(format!("IONEX {label} field unparsable")))
}

/// Parse the first three whitespace-delimited floats of the data portion.
fn three_fields(line: &str, label: &str) -> Result<(f64, f64, f64)> {
    let data = data_of(line);
    let mut it = data.split_whitespace();
    let a = next_float(&mut it, label)?;
    let b = next_float(&mut it, label)?;
    let c = next_float(&mut it, label)?;
    Ok((a, b, c))
}

fn next_float<'a>(it: &mut impl Iterator<Item = &'a str>, label: &str) -> Result<f64> {
    it.next()
        .ok_or_else(|| Error::Parse(format!("IONEX {label} record short")))?
        .parse()
        .map_err(|_| Error::Parse(format!("IONEX {label} field unparsable")))
}

/// The data portion of a record (columns 0..60), or the whole short line.
///
/// Uses [`crate::format::columns::raw_field`] so the column-60 cut is floored to a
/// char boundary rather than panicking on a multibyte character straddling the
/// offset; a short line returns whole and a valid ASCII line cuts at byte 60 as
/// before.
fn data_of(line: &str) -> &str {
    raw_field(line, 0, 60)
}

/// Build a node axis `v1 + i * step`, with the count taken from the inclusive
/// `[v1, v2]` span (a half-step guard on the end matches the producer's
/// `arange`-style construction).
fn node_axis(v1: f64, v2: f64, step: f64) -> Result<Vec<f64>> {
    let v1 = validate_axis_degree(v1, "IONEX grid axis start")?;
    let v2 = validate_axis_degree(v2, "IONEX grid axis end")?;
    let step = validate_axis_degree(step, "IONEX grid step")?;
    if step == 0.0 {
        return Err(Error::Parse("IONEX grid step is zero".into()));
    }
    let guard = 0.5 * step;
    let span = validate::finite((v2 + guard - v1) / step, "IONEX grid span")
        .map_err(map_axis_field_error)?;
    if span < 0.0 {
        return Err(Error::Parse("IONEX grid span has the wrong sign".into()));
    }
    validate::finite_in_range(span, 0.0, IONEX_AXIS_MAX_SPAN, "IONEX grid span")
        .map_err(map_axis_field_error)?;
    let n = span.floor() as usize + 1;
    Ok((0..n).map(|i| v1 + (i as f64) * step).collect())
}

fn validate_axis_degree(value: f64, field: &'static str) -> Result<f64> {
    validate::finite_in_range(value, -IONEX_AXIS_DEG_LIMIT, IONEX_AXIS_DEG_LIMIT, field)
        .map_err(map_axis_field_error)
}

fn map_axis_field_error(error: validate::FieldError) -> Error {
    Error::Parse(format!("IONEX {error}"))
}

/// Parse an `EPOCH OF CURRENT MAP` record into a UTC instant.
fn parse_epoch_instant(line: &str) -> Result<Instant> {
    let seconds = parse_epoch_j2000_s(line)?;
    Ok(ionex_epoch_from_j2000_seconds(seconds))
}

/// Parse an `EPOCH OF CURRENT MAP` record into J2000 seconds.
///
/// The record carries `year month day hour minute second`; the result is the
/// integer number of seconds from the J2000 epoch (2000-01-01 12:00:00).
fn parse_epoch_j2000_s(line: &str) -> Result<i64> {
    let data = data_of(line);
    let mut it = data.split_whitespace();
    let mut next_int = |what: &'static str| -> Result<i64> {
        let token = it
            .next()
            .ok_or_else(|| Error::Parse(format!("IONEX epoch missing {what}")))?;
        validate::strict_int::<i64>(token, what)
            .map_err(|_| Error::Parse(format!("IONEX epoch {what} unparsable")))
    };
    let year = next_int("year")?;
    let month = next_int("month")?;
    let day = next_int("day")?;
    let hour = next_int("hour")?;
    let minute = next_int("minute")?;
    let second = next_int("second")?;
    let civil = validate::civil_datetime_with_second_policy(
        year,
        month,
        day,
        hour,
        minute,
        second as f64,
        validate::CivilSecondPolicy::Continuous,
    )
    .map_err(|error| Error::Parse(format!("IONEX epoch {error}")))?;

    // Canonical continuous-seconds-since-J2000 conversion. IONEX epochs are
    // whole-second, so the integer-second result is exact and narrows back to
    // i64 bit-identically to the previous open-coded day-count arithmetic.
    Ok(j2000_seconds(
        civil.year as i32,
        civil.month as i32,
        civil.day as i32,
        civil.hour as i32,
        civil.minute as i32,
        civil.second,
    ) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_epoch_rejects_invalid_civil_datetime() {
        for line in [
            "  2020     2    30     0     0     0                        EPOCH OF CURRENT MAP",
            "  2020     6    25    24     0     0                        EPOCH OF CURRENT MAP",
            "  2020     6    25    23    59    60                        EPOCH OF CURRENT MAP",
        ] {
            assert!(matches!(parse_epoch_instant(line), Err(Error::Parse(_))));
        }
    }

    #[test]
    fn parse_epoch_rejects_years_outside_civil_product_range() {
        for line in [
            "  100000000000000     1     1     0     0     0              EPOCH OF CURRENT MAP",
            " -100000000000000     1     1     0     0     0              EPOCH OF CURRENT MAP",
        ] {
            assert!(matches!(parse_epoch_instant(line), Err(Error::Parse(_))));
        }
    }

    #[test]
    fn parse_epoch_accepts_valid_civil_datetime() {
        assert_eq!(
            j2000_seconds_from_instant(
                parse_epoch_instant(
                    "  2020     6    25     0     0     0                        EPOCH OF CURRENT MAP"
                )
                .expect("valid IONEX epoch")
            )
            .expect("J2000 seconds"),
            646_315_200
        );
    }
}
