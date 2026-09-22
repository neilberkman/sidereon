//! IONEX reader.
//!
//! Reads an IONEX Version 1 product (Schaer, Gurtner and Feltens, 1998) into its
//! vertical-TEC maps, the RMS and height maps related to them, the header
//! records a product carries, and the geometry the interpolation of the maps
//! uses.
//! The float math that turns a map into a slant delay lives in
//! [`super::slant`]; this module reads records.
//!
//! TEC, RMS and height values are read in their `16I5` columns: sixteen
//! five-character fields to a data record, a band continuing on the records
//! after it. A record not laid out in those columns is read by its
//! whitespace-separated fields; a record laid out in them with an empty field
//! is refused, because no reading of it can tell which value is absent. The
//! field `9999` marks a non-available value and is kept as `None`.
//!
//! Each `LAT/LON1/LON2/DLON/H` record (`2X,5F6.1`) places the values after it by
//! its own latitude, its longitudes `LON1 + DLON * m` and its height, each of
//! which must be a node of the header grid. Bands may come in any order; a node
//! given twice, or left without a value, is refused. An `EXPONENT` record inside
//! a map sets the unit of the data blocks after it; before one, the header
//! `EXPONENT` applies, or `-1` without one. IONEX 1 says of the header records
//! that "Each value remains valid until changed by an additional header
//! record", so an exponent one map sets stays in effect for the maps after it,
//! and a map that inherits one is reported as
//! [`IonexWarning::ExponentCarriedIntoMap`]. Values are stored in TECU, and
//! height values in kilometers, as `field * 10^EXPONENT` formed as one
//! multiply, which is how the reference readers scale a field. The factor is
//! built from an exact power of ten rather than taken from a `pow`
//! implementation, so a node does not depend on a library's rounding.
//!
//! Nodes are held in the order the file writes each axis, which the sign of
//! `DLAT` or `DLON` gives, each axis rebuilt as `v1 + i * step` so it matches
//! the producer's `arange`-style construction bit-for-bit.
//!
//! This reader keeps single-layer maps. A `MAP DIMENSION 3` product, or a height
//! grid with more than one height, is refused by name: its TEC values are layer
//! contributions (electron density times `DHGT`), which none of the single-layer
//! maps, samples or slant-delay evaluations here could hold or use.

use super::header::{IonexHeader, IonexMappingFunction, IonexWarning};
use super::{ionex_epoch_from_j2000_seconds, j2000_seconds_from_instant};
use crate::astro::constants::time::SECONDS_PER_DAY_I64;
use crate::astro::time::civil::{civil_from_j2000_seconds, j2000_seconds};
use crate::astro::time::model::Instant;
use crate::error::{Error, Result};
use crate::format::columns::{fixed_record, raw_field, raw_field_from};
use crate::validate;

const IONEX_AXIS_DEG_LIMIT: f64 = 360.0;
const IONEX_AXIS_MAX_NODES: usize = 10_000;
const IONEX_AXIS_MAX_SPAN: f64 = (IONEX_AXIS_MAX_NODES - 1) as f64;

/// The field IONEX writes for a non-available TEC, RMS or height value.
pub(crate) const NON_AVAILABLE: i64 = 9999;
/// The `EXPONENT` a file without one uses.
pub(crate) const DEFAULT_EXPONENT: i32 = -1;
/// Largest distance, in degrees or kilometers, at which a band coordinate is
/// read as a grid node. Band records give coordinates in `F6.1` fields, so a
/// coordinate written for another node lies at least `0.05` away.
const NODE_TOLERANCE: f64 = 1.0e-6;

pub(crate) const VERSION_TYPE: &str = "IONEX VERSION / TYPE";
pub(crate) const PGM_RUN_BY_DATE: &str = "PGM / RUN BY / DATE";
pub(crate) const DESCRIPTION: &str = "DESCRIPTION";
pub(crate) const COMMENT: &str = "COMMENT";
pub(crate) const EPOCH_OF_FIRST_MAP: &str = "EPOCH OF FIRST MAP";
pub(crate) const EPOCH_OF_LAST_MAP: &str = "EPOCH OF LAST MAP";
pub(crate) const INTERVAL: &str = "INTERVAL";
pub(crate) const MAPS_IN_FILE: &str = "# OF MAPS IN FILE";
pub(crate) const MAPPING_FUNCTION: &str = "MAPPING FUNCTION";
pub(crate) const ELEVATION_CUTOFF: &str = "ELEVATION CUTOFF";
pub(crate) const OBSERVABLES_USED: &str = "OBSERVABLES USED";
pub(crate) const STATIONS: &str = "# OF STATIONS";
pub(crate) const SATELLITES: &str = "# OF SATELLITES";
pub(crate) const BASE_RADIUS: &str = "BASE RADIUS";
pub(crate) const MAP_DIMENSION: &str = "MAP DIMENSION";
pub(crate) const HGT_AXIS: &str = "HGT1 / HGT2 / DHGT";
pub(crate) const LAT_AXIS: &str = "LAT1 / LAT2 / DLAT";
pub(crate) const LON_AXIS: &str = "LON1 / LON2 / DLON";
pub(crate) const EXPONENT: &str = "EXPONENT";
pub(crate) const START_OF_AUX_DATA: &str = "START OF AUX DATA";
pub(crate) const END_OF_AUX_DATA: &str = "END OF AUX DATA";
pub(crate) const END_OF_HEADER: &str = "END OF HEADER";
pub(crate) const START_OF_TEC_MAP: &str = "START OF TEC MAP";
pub(crate) const END_OF_TEC_MAP: &str = "END OF TEC MAP";
pub(crate) const START_OF_RMS_MAP: &str = "START OF RMS MAP";
pub(crate) const END_OF_RMS_MAP: &str = "END OF RMS MAP";
pub(crate) const START_OF_HEIGHT_MAP: &str = "START OF HEIGHT MAP";
pub(crate) const END_OF_HEIGHT_MAP: &str = "END OF HEIGHT MAP";
pub(crate) const EPOCH_OF_CURRENT_MAP: &str = "EPOCH OF CURRENT MAP";
pub(crate) const BAND: &str = "LAT/LON1/LON2/DLON/H";
pub(crate) const END_OF_FILE: &str = "END OF FILE";

/// One map's values, indexed `[i_lat][i_lon]`; `None` marks a node the product
/// gives as non-available.
pub(crate) type Grid = Vec<Vec<Option<f64>>>;

/// A parsed IONEX vertical-TEC product.
///
/// The grids are indexed `[map][i_lat][i_lon]`, with `lat_nodes_deg` and
/// `lon_nodes_deg` in the order their signed steps give. A node is
/// `None` where the file gives the value as non-available (`9999`). TEC and RMS
/// values are in TECU and height values in kilometers, after the `10^EXPONENT`
/// scaling. Map epochs are UTC instants.
#[derive(Debug, Clone, PartialEq)]
pub struct Ionex {
    /// Descriptive header records.
    header: IonexHeader,
    /// Latitude node values in degrees, in the order `dlat_deg` gives.
    lat_nodes_deg: Vec<f64>,
    /// Longitude node values in degrees, in the order `dlon_deg` gives.
    lon_nodes_deg: Vec<f64>,
    /// Signed latitude step in degrees (negative for the standard ordering).
    dlat_deg: f64,
    /// Signed longitude step in degrees (positive for the standard ordering).
    dlon_deg: f64,
    /// Single-layer shell height in kilometers.
    shell_height_km: f64,
    /// Mean earth radius used by the geometry, in kilometers.
    base_radius_km: f64,
    /// The header `EXPONENT` field.
    exponent: i32,
    /// Map epochs as UTC instants, ascending.
    map_epochs: Vec<Instant>,
    /// Per-map vertical-TEC grids, indexed `[map][i_lat][i_lon]` (TECU).
    tec_maps: Vec<Grid>,
    /// Per-map RMS grids, indexed `[map][i_lat][i_lon]` (TECU); empty only
    /// where the product declares no RMS map.
    rms_maps: Vec<Grid>,
    /// Per-map height grids, indexed `[map][i_lat][i_lon]` (km); empty if absent.
    height_maps: Vec<Grid>,
    /// Count of records skipped during a forgiving parse (e.g. an unsupported
    /// `START OF AUX DATA` block). Lets callers tell a clean product
    /// (`skipped_records == 0`) apart from one carrying records outside this
    /// reader's grid subset, without aborting the whole parse. Mirrors
    /// [`crate::ephemeris::Sp3::skipped_records`].
    skipped_records: usize,
}

/// Fully materialized IONEX grid fields before invariant checks.
pub(crate) struct IonexParts {
    pub(crate) header: IonexHeader,
    pub(crate) lat_nodes_deg: Vec<f64>,
    pub(crate) lon_nodes_deg: Vec<f64>,
    pub(crate) dlat_deg: f64,
    pub(crate) dlon_deg: f64,
    pub(crate) shell_height_km: f64,
    pub(crate) base_radius_km: f64,
    pub(crate) exponent: i32,
    pub(crate) map_epochs: Vec<Instant>,
    pub(crate) tec_maps: Vec<Grid>,
    pub(crate) rms_maps: Vec<Grid>,
    pub(crate) height_maps: Vec<Grid>,
    pub(crate) skipped_records: usize,
}

impl Ionex {
    /// Descriptive header records this product carries.
    pub fn header(&self) -> &IonexHeader {
        &self.header
    }

    /// Latitude node values in degrees, in the order the file writes them,
    /// which the sign of `DLAT` gives.
    pub fn lat_nodes_deg(&self) -> &[f64] {
        &self.lat_nodes_deg
    }

    /// Longitude node values in degrees, in the order the file writes them,
    /// which the sign of `DLON` gives.
    pub fn lon_nodes_deg(&self) -> &[f64] {
        &self.lon_nodes_deg
    }

    /// Signed latitude step in degrees (negative where the file runs its
    /// latitudes north to south, as most do).
    pub fn dlat_deg(&self) -> f64 {
        self.dlat_deg
    }

    /// Signed longitude step in degrees.
    pub fn dlon_deg(&self) -> f64 {
        self.dlon_deg
    }

    /// Single-layer shell height in kilometers, `HGT1`.
    pub fn shell_height_km(&self) -> f64 {
        self.shell_height_km
    }

    /// `BASE RADIUS`: the mean earth radius the pierce-point geometry uses, in
    /// kilometers.
    pub fn base_radius_km(&self) -> f64 {
        self.base_radius_km
    }

    /// The header `EXPONENT` field; `-1` where the file gives none.
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
    // invariant: parsed and sample-built IONEX epochs are whole, representable J2000 seconds.
    #[allow(clippy::expect_used)]
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
    ///
    /// A node is `None` where the product gives the value as non-available.
    pub fn tec_maps(&self) -> &[Grid] {
        &self.tec_maps
    }

    /// Per-map RMS grids, indexed `[map][i_lat][i_lon]` (TECU); empty only
    /// where the product declares no RMS map.
    ///
    /// Each grid belongs to the TEC map of the same index. A node is `None`
    /// where the product gives no RMS value for it, including every node of a
    /// TEC map the file gives no RMS map for. A product whose RMS maps hold no
    /// value at any node keeps that stack, so a file stating RMS maps with
    /// every value non-available stays apart from a file stating no RMS map:
    /// both give no RMS number anywhere, and only the first declares the maps.
    pub fn rms_maps(&self) -> &[Grid] {
        &self.rms_maps
    }

    /// Per-map height grids, indexed `[map][i_lat][i_lon]` (km); empty if the
    /// product has no height value at any node.
    ///
    /// Each grid belongs to the TEC map of the same index. The spec gives a
    /// node's single-layer height above `BASE RADIUS` as `HGT1` plus its height
    /// value. A node is `None` where the product gives no height value for it.
    pub fn height_maps(&self) -> &[Grid] {
        &self.height_maps
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

    /// Build an IONEX product from materialized grid fields.
    ///
    /// The optional RMS and height map stacks are stored as given. A stack
    /// holding no value at any node stays present: it says the product declares
    /// those maps and states no value in them, where an empty stack says the
    /// product declares no map of that kind at all. A file giving every RMS
    /// value as `9999` therefore keeps its RMS maps, while a file with no RMS
    /// map has none. Height maps read the same way: a height map without values
    /// says the single-layer heights are unknown, where a product without height
    /// maps has the one height `HGT1`.
    pub(crate) fn from_parts(parts: IonexParts) -> Result<Self> {
        validate_ionex_parts(&parts)?;
        let IonexParts {
            header,
            lat_nodes_deg,
            lon_nodes_deg,
            dlat_deg,
            dlon_deg,
            shell_height_km,
            base_radius_km,
            exponent,
            map_epochs,
            tec_maps,
            rms_maps,
            height_maps,
            skipped_records,
        } = parts;

        Ok(Self {
            header,
            lat_nodes_deg,
            lon_nodes_deg,
            dlat_deg,
            dlon_deg,
            shell_height_km,
            base_radius_km,
            exponent,
            map_epochs,
            tec_maps,
            rms_maps,
            height_maps,
            skipped_records,
        })
    }

    /// Parse an IONEX product from its bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        Self::parse_with_warnings(bytes).map(|(ionex, _)| ionex)
    }

    /// Parse an IONEX product from its text.
    pub fn parse_str(text: &str) -> Result<Self> {
        Self::parse_str_with_warnings(text).map(|(ionex, _)| ionex)
    }

    /// Parse an IONEX product from its bytes, with the findings the reader
    /// reports without refusing the file.
    pub fn parse_with_warnings(bytes: &[u8]) -> Result<(Self, Vec<IonexWarning>)> {
        let text = core::str::from_utf8(bytes)
            .map_err(|_| Error::Parse("IONEX is not valid UTF-8".into()))?;
        Self::parse_str_with_warnings(text)
    }

    /// Parse an IONEX product from its text, with the findings the reader
    /// reports without refusing the file.
    ///
    /// A header record that summarizes or describes the maps and disagrees with
    /// them, or is absent, is reported as an [`IonexWarning`]: every map carries
    /// its own epoch and bands, so the values read do not depend on it. A
    /// summary record that cannot be read at all is skipped and counted in
    /// [`Ionex::skipped_records`]. Records that decide where or how a value is
    /// read are refused when they disagree with the data.
    pub fn parse_str_with_warnings(text: &str) -> Result<(Self, Vec<IonexWarning>)> {
        let mut lines = text
            .lines()
            .enumerate()
            .map(|(index, line)| (index + 1, line));
        let mut skipped_records = 0usize;

        let records = read_header(&mut lines, &mut skipped_records)?;
        let axes = records.axes()?;
        let exponent = records
            .exponent
            .map_or(DEFAULT_EXPONENT, |(_, exponent)| exponent);
        let mut body = read_body(&mut lines, &axes, exponent, &mut skipped_records)?;

        if body.tec_maps.is_empty() {
            return Err(Error::Parse("IONEX has no TEC maps".into()));
        }
        validate_map_epochs_strictly_increasing(&body.map_epochs)?;
        let related_maps = body.related.len();
        let (rms_maps, height_maps) = body.related_grids(&axes)?;
        let mut warnings = records.warnings(&body.map_epochs, related_maps, body.end_of_file);
        warnings.append(&mut body.warnings);

        let ionex = Self::from_parts(IonexParts {
            header: records.header(),
            lat_nodes_deg: axes.lat_nodes,
            lon_nodes_deg: axes.lon_nodes,
            dlat_deg: axes.dlat,
            dlon_deg: axes.dlon,
            shell_height_km: axes.shell_height_km,
            base_radius_km: axes.base_radius_km,
            exponent,
            map_epochs: body.map_epochs,
            tec_maps: body.tec_maps,
            rms_maps,
            height_maps,
            skipped_records,
        })?;
        Ok((ionex, warnings))
    }
}

pub(crate) fn validate_map_epochs_strictly_increasing(map_epochs: &[Instant]) -> Result<()> {
    let mut previous_s = None;
    for (index, &epoch) in map_epochs.iter().enumerate() {
        let seconds = j2000_seconds_from_instant(epoch).ok_or_else(|| {
            Error::Parse(format!(
                "IONEX map epoch {} cannot be projected onto J2000 seconds",
                index + 1
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

fn validate_ionex_parts(parts: &IonexParts) -> Result<()> {
    let IonexParts {
        header,
        lat_nodes_deg,
        lon_nodes_deg,
        dlat_deg,
        dlon_deg,
        shell_height_km,
        base_radius_km,
        map_epochs,
        tec_maps,
        rms_maps,
        height_maps,
        ..
    } = parts;
    if tec_maps.is_empty() {
        return Err(Error::Parse("IONEX has no TEC maps".into()));
    }
    if map_epochs.len() != tec_maps.len() {
        return Err(Error::Parse(format!(
            "IONEX has {} TEC maps but {} map epochs",
            tec_maps.len(),
            map_epochs.len()
        )));
    }
    // Bilinear interpolation brackets a cell with `node[i+1]` / `node[j+1]`,
    // so each axis needs at least two nodes. Reject a degenerate grid here
    // rather than letting evaluation index past the end.
    if lat_nodes_deg.len() < 2 || lon_nodes_deg.len() < 2 {
        return Err(Error::Parse(format!(
            "IONEX grid has fewer than 2 nodes on an axis (got {} lat, {} lon)",
            lat_nodes_deg.len(),
            lon_nodes_deg.len()
        )));
    }

    validate_axis_degree(*dlat_deg, "IONEX latitude step")?;
    validate_axis_degree(*dlon_deg, "IONEX longitude step")?;
    // IONEX 1 gives an axis as "'LAT1' to 'LAT2' with increment 'DLAT'", which
    // says nothing about the direction: a file may run its latitudes north to
    // south or south to north, and its longitudes either way, as long as the
    // step carries the sign that takes the first bound to the second.
    // `node_axis` refuses a step whose sign contradicts its bounds.
    if *dlat_deg == 0.0 || *dlon_deg == 0.0 {
        return Err(Error::Parse("IONEX grid step is zero".into()));
    }
    validate::finite(*shell_height_km, "IONEX shell height").map_err(map_axis_field_error)?;
    validate::finite(*base_radius_km, "IONEX base radius").map_err(map_axis_field_error)?;
    validate::finite(header.version, "IONEX version").map_err(map_axis_field_error)?;
    validate::finite(header.elevation_cutoff_deg, "IONEX elevation cutoff")
        .map_err(map_axis_field_error)?;

    validate_axis_nodes(lat_nodes_deg, "latitude")?;
    validate_axis_nodes(lon_nodes_deg, "longitude")?;
    validate_axis_order(lat_nodes_deg, *dlat_deg, "latitude")?;
    validate_axis_order(lon_nodes_deg, *dlon_deg, "longitude")?;

    validate_map_epochs_strictly_increasing(map_epochs)?;
    let nlat = lat_nodes_deg.len();
    let nlon = lon_nodes_deg.len();
    validate_map_dimensions("TEC", tec_maps, map_epochs.len(), nlat, nlon)?;
    validate_map_values("TEC", tec_maps)?;
    for (kind, maps) in [("RMS", rms_maps), ("height", height_maps)] {
        if maps.is_empty() {
            continue;
        }
        if maps.len() != tec_maps.len() {
            return Err(Error::Parse(format!(
                "IONEX {kind} map count does not match TEC map count"
            )));
        }
        validate_map_dimensions(kind, maps, map_epochs.len(), nlat, nlon)?;
        validate_map_values(kind, maps)?;
    }
    Ok(())
}

fn validate_axis_nodes(nodes: &[f64], axis: &'static str) -> Result<()> {
    for (index, &node) in nodes.iter().enumerate() {
        validate_axis_degree(
            node,
            if index == 0 {
                "IONEX grid axis"
            } else {
                "IONEX grid axis[]"
            },
        )
        .map_err(|error| Error::Parse(format!("IONEX {axis} node {index} invalid: {error}")))?;
    }
    Ok(())
}

fn validate_map_dimensions(
    kind: &'static str,
    maps: &[Grid],
    expected_maps: usize,
    expected_lat: usize,
    expected_lon: usize,
) -> Result<()> {
    if maps.len() != expected_maps {
        return Err(Error::Parse(format!(
            "IONEX {kind} map count is {}, expected {expected_maps}",
            maps.len()
        )));
    }
    for (map_index, map) in maps.iter().enumerate() {
        if map.len() != expected_lat {
            return Err(Error::Parse(format!(
                "IONEX {kind} map {} has {} latitude bands, expected {expected_lat}",
                map_index + 1,
                map.len()
            )));
        }
        for (lat_index, row) in map.iter().enumerate() {
            if row.len() != expected_lon {
                return Err(Error::Parse(format!(
                    "IONEX {kind} map {} latitude band {lat_index} has {} values, expected {expected_lon}",
                    map_index + 1,
                    row.len()
                )));
            }
        }
    }
    Ok(())
}

fn validate_map_values(kind: &'static str, maps: &[Grid]) -> Result<()> {
    for (map_index, map) in maps.iter().enumerate() {
        for (lat_index, row) in map.iter().enumerate() {
            for (lon_index, value) in row.iter().enumerate() {
                let Some(value) = *value else {
                    continue;
                };
                validate::finite(value, "IONEX grid value").map_err(|error| {
                    Error::Parse(format!(
                        "IONEX {kind} map {} value [{lat_index}][{lon_index}] invalid: {error}",
                        map_index + 1
                    ))
                })?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// Header records as read, each with the line it came from.
#[derive(Default)]
struct HeaderRecords {
    /// Line of the first record, when that record is not `IONEX VERSION / TYPE`.
    first_other_line: Option<usize>,
    version: Option<(usize, (f64, String))>,
    program: Option<(usize, [String; 3])>,
    descriptions: Vec<String>,
    comments: Vec<String>,
    first_epoch: Option<(usize, Instant)>,
    last_epoch: Option<(usize, Instant)>,
    interval: Option<(usize, u32)>,
    map_count: Option<(usize, u64)>,
    mapping_function: Option<(usize, Option<IonexMappingFunction>)>,
    elevation_cutoff: Option<(usize, f64)>,
    observables_used: Option<(usize, String)>,
    station_count: Option<(usize, u32)>,
    satellite_count: Option<(usize, u32)>,
    base_radius: Option<(usize, f64)>,
    map_dimension: Option<(usize, i64)>,
    hgt: Option<(usize, [f64; 3])>,
    lat: Option<(usize, [f64; 3])>,
    lon: Option<(usize, [f64; 3])>,
    exponent: Option<(usize, i32)>,
    /// Labels of summary records present but unreadable, skipped and counted.
    unreadable: Vec<&'static str>,
}

/// The grid a header defines.
struct Axes {
    lat_nodes: Vec<f64>,
    lon_nodes: Vec<f64>,
    dlat: f64,
    dlon: f64,
    shell_height_km: f64,
    base_radius_km: f64,
}

fn read_header<'a, I>(lines: &mut I, skipped: &mut usize) -> Result<HeaderRecords>
where
    I: Iterator<Item = (usize, &'a str)>,
{
    let mut records = HeaderRecords::default();
    let mut aux_line: Option<usize> = None;
    let mut first_record = true;

    for (number, line) in lines.by_ref() {
        if line.trim().is_empty() {
            continue;
        }
        let label = label_of(line);
        if let Some(start) = aux_line {
            match label {
                END_OF_AUX_DATA => aux_line = None,
                END_OF_HEADER => {
                    return Err(Error::Parse(format!(
                        "IONEX START OF AUX DATA at line {start} is not closed before \
                         END OF HEADER at line {number}"
                    )));
                }
                _ => {}
            }
            continue;
        }
        if first_record {
            first_record = false;
            if label != VERSION_TYPE {
                records.first_other_line = Some(number);
            }
        }

        let data = data_of(line);
        match label {
            VERSION_TYPE => {
                if let Some(&file_type) = data.as_bytes().get(20) {
                    if file_type != b' ' && file_type != b'I' {
                        return Err(Error::Parse(format!(
                            "IONEX VERSION / TYPE at line {number} gives file type {:?}, \
                             not I (ionosphere maps)",
                            char::from(file_type)
                        )));
                    }
                }
                match version_record(data) {
                    Some(value) => set_once(&mut records.version, number, value, label)?,
                    None => skip_unreadable(&mut records, skipped, VERSION_TYPE),
                }
            }
            PGM_RUN_BY_DATE => {
                let value =
                    [0, 20, 40].map(|start| raw_field(data, start, start + 20).trim().to_string());
                set_once(&mut records.program, number, value, label)?;
            }
            DESCRIPTION => records.descriptions.push(data.trim_end().to_string()),
            COMMENT => records.comments.push(data.trim_end().to_string()),
            EPOCH_OF_FIRST_MAP => match epoch_record(data) {
                Some(value) => set_once(&mut records.first_epoch, number, value, label)?,
                None => skip_unreadable(&mut records, skipped, EPOCH_OF_FIRST_MAP),
            },
            EPOCH_OF_LAST_MAP => match epoch_record(data) {
                Some(value) => set_once(&mut records.last_epoch, number, value, label)?,
                None => skip_unreadable(&mut records, skipped, EPOCH_OF_LAST_MAP),
            },
            INTERVAL => match whole_count(data).and_then(|value| u32::try_from(value).ok()) {
                Some(value) => set_once(&mut records.interval, number, value, label)?,
                None => skip_unreadable(&mut records, skipped, INTERVAL),
            },
            MAPS_IN_FILE => match whole_count(data) {
                Some(value) => set_once(&mut records.map_count, number, value, label)?,
                None => skip_unreadable(&mut records, skipped, MAPS_IN_FILE),
            },
            MAPPING_FUNCTION => {
                let code = data.split_whitespace().next().unwrap_or("");
                let value = IonexMappingFunction::from_code(code);
                set_once(&mut records.mapping_function, number, value, label)?;
            }
            ELEVATION_CUTOFF => match float_record(data, 8) {
                Some(value) => set_once(&mut records.elevation_cutoff, number, value, label)?,
                None => skip_unreadable(&mut records, skipped, ELEVATION_CUTOFF),
            },
            OBSERVABLES_USED => {
                let value = data.trim_end().to_string();
                set_once(&mut records.observables_used, number, value, label)?;
            }
            STATIONS => match whole_count(data).and_then(|value| u32::try_from(value).ok()) {
                Some(value) => set_once(&mut records.station_count, number, value, label)?,
                None => skip_unreadable(&mut records, skipped, STATIONS),
            },
            SATELLITES => match whole_count(data).and_then(|value| u32::try_from(value).ok()) {
                Some(value) => set_once(&mut records.satellite_count, number, value, label)?,
                None => skip_unreadable(&mut records, skipped, SATELLITES),
            },
            BASE_RADIUS => {
                let value = float_record(data, 8).ok_or_else(|| {
                    Error::Parse(format!(
                        "IONEX BASE RADIUS field unparsable at line {number}"
                    ))
                })?;
                set_once(&mut records.base_radius, number, value, label)?;
            }
            MAP_DIMENSION => {
                let value = single_field(data, 6)
                    .and_then(|field| validate::strict_int::<i64>(field, MAP_DIMENSION).ok())
                    .ok_or_else(|| {
                        Error::Parse(format!(
                            "IONEX MAP DIMENSION field unparsable at line {number}"
                        ))
                    })?;
                set_once(&mut records.map_dimension, number, value, label)?;
            }
            HGT_AXIS => {
                let value = axis_record(data, HGT_AXIS)?;
                set_once(&mut records.hgt, number, value, label)?;
            }
            LAT_AXIS => {
                let value = axis_record(data, LAT_AXIS)?;
                set_once(&mut records.lat, number, value, label)?;
            }
            LON_AXIS => {
                let value = axis_record(data, LON_AXIS)?;
                set_once(&mut records.lon, number, value, label)?;
            }
            EXPONENT => {
                let value = exponent_record(data).ok_or_else(|| {
                    Error::Parse(format!("IONEX EXPONENT field unparsable at line {number}"))
                })?;
                set_once(&mut records.exponent, number, value, label)?;
            }
            START_OF_AUX_DATA => {
                // Auxiliary data blocks (e.g. satellite and station DCBs) are
                // outside this reader's grid subset. The block counts as one
                // skipped record; its lines are passed over until it closes.
                *skipped += 1;
                aux_line = Some(number);
            }
            END_OF_HEADER => return Ok(records),
            _ => *skipped += 1,
        }
    }

    if let Some(start) = aux_line {
        return Err(Error::Parse(format!(
            "IONEX START OF AUX DATA at line {start} is not closed before the end of the file"
        )));
    }
    Err(Error::Parse(
        "IONEX header has no END OF HEADER record".into(),
    ))
}

fn skip_unreadable(records: &mut HeaderRecords, skipped: &mut usize, label: &'static str) {
    *skipped += 1;
    records.unreadable.push(label);
}

/// Keep the first reading of a single-value header record; refuse a second
/// record of the label that reads as another value.
fn set_once<T: PartialEq>(
    slot: &mut Option<(usize, T)>,
    line: usize,
    value: T,
    label: &str,
) -> Result<()> {
    match slot {
        Some((first, existing)) if *existing != value => Err(Error::Parse(format!(
            "IONEX header gives {label} at lines {first} and {line} with different values"
        ))),
        Some(_) => Ok(()),
        None => {
            *slot = Some((line, value));
            Ok(())
        }
    }
}

impl HeaderRecords {
    fn axes(&self) -> Result<Axes> {
        let (_, [lat1, lat2, dlat]) = self
            .lat
            .ok_or_else(|| Error::Parse("IONEX missing LAT1 / LAT2 / DLAT".into()))?;
        let (_, [lon1, lon2, dlon]) = self
            .lon
            .ok_or_else(|| Error::Parse("IONEX missing LON1 / LON2 / DLON".into()))?;
        let (_, [hgt1, hgt2, dhgt]) = self
            .hgt
            .ok_or_else(|| Error::Parse("IONEX missing HGT1 / HGT2 / DHGT".into()))?;
        let (_, base_radius_km) = self
            .base_radius
            .ok_or_else(|| Error::Parse("IONEX missing BASE RADIUS".into()))?;

        let lat_nodes = node_axis(lat1, lat2, dlat)?;
        let lon_nodes = node_axis(lon1, lon2, dlon)?;
        if lat_nodes.len() < 2 || lon_nodes.len() < 2 {
            return Err(Error::Parse(format!(
                "IONEX grid has fewer than 2 nodes on an axis (got {} lat, {} lon)",
                lat_nodes.len(),
                lon_nodes.len()
            )));
        }

        match self.map_dimension {
            Some((line, 3)) => {
                return Err(Error::Parse(format!(
                    "IONEX MAP DIMENSION 3 at line {line} declares 3-D maps \
                     (HGT1 / HGT2 / DHGT {hgt1} {hgt2} {dhgt}), which this reader does not \
                     read: it keeps one TEC layer per map"
                )));
            }
            Some((line, dimension)) if dimension != 2 => {
                return Err(Error::Parse(format!(
                    "IONEX MAP DIMENSION {dimension} at line {line} is neither 2 nor 3"
                )));
            }
            _ => {}
        }
        if hgt1 != hgt2 {
            return Err(Error::Parse(format!(
                "IONEX HGT1 / HGT2 / DHGT {hgt1} {hgt2} {dhgt} defines more than one height, \
                 as a MAP DIMENSION 3 product does, which this reader does not read: it keeps \
                 one TEC layer per map"
            )));
        }

        Ok(Axes {
            lat_nodes,
            lon_nodes,
            dlat,
            dlon,
            shell_height_km: hgt1,
            base_radius_km,
        })
    }

    fn header(&self) -> IonexHeader {
        let mut header = IonexHeader::unstated();
        if let Some((_, (version, system))) = &self.version {
            header.version = *version;
            header.satellite_system = system.clone();
        }
        if let Some((_, [program, run_by, date])) = &self.program {
            header.program = program.clone();
            header.run_by = run_by.clone();
            header.date = date.clone();
        }
        header.descriptions = self.descriptions.clone();
        header.comments = self.comments.clone();
        if let Some((_, interval)) = self.interval {
            header.interval_s = interval;
        }
        if let Some((_, mapping_function)) = &self.mapping_function {
            header.mapping_function = mapping_function.clone();
        }
        if let Some((_, cutoff)) = self.elevation_cutoff {
            header.elevation_cutoff_deg = cutoff;
        }
        if let Some((_, observables)) = &self.observables_used {
            header.observables_used = observables.clone();
        }
        header.station_count = self.station_count.map(|(_, count)| count);
        header.satellite_count = self.satellite_count.map(|(_, count)| count);
        header.maps_in_file = self
            .map_count
            .and_then(|(_, count)| u32::try_from(count).ok());
        header
    }

    fn warnings(
        &self,
        map_epochs: &[Instant],
        related_maps: usize,
        end_of_file: bool,
    ) -> Vec<IonexWarning> {
        let mut warnings = Vec::new();
        if let (Some((line, _)), Some(_)) = (&self.version, self.first_other_line) {
            warnings.push(IonexWarning::VersionRecordNotFirst { line: *line });
        }
        for (present, label) in [
            (self.version.is_some(), VERSION_TYPE),
            (self.program.is_some(), PGM_RUN_BY_DATE),
            (self.first_epoch.is_some(), EPOCH_OF_FIRST_MAP),
            (self.last_epoch.is_some(), EPOCH_OF_LAST_MAP),
            (self.interval.is_some(), INTERVAL),
            (self.map_count.is_some(), MAPS_IN_FILE),
            (self.mapping_function.is_some(), MAPPING_FUNCTION),
            (self.elevation_cutoff.is_some(), ELEVATION_CUTOFF),
            (self.observables_used.is_some(), OBSERVABLES_USED),
            (self.map_dimension.is_some(), MAP_DIMENSION),
        ] {
            if !present && !self.unreadable.contains(&label) {
                warnings.push(IonexWarning::MissingRecord(label));
            }
        }

        let seconds = |epoch: Instant| j2000_seconds_from_instant(epoch);
        for (record, label, maps) in [
            (self.first_epoch, EPOCH_OF_FIRST_MAP, map_epochs.first()),
            (self.last_epoch, EPOCH_OF_LAST_MAP, map_epochs.last()),
        ] {
            if let (Some((line, declared)), Some(&maps)) = (record, maps) {
                if seconds(declared) != seconds(maps) {
                    warnings.push(IonexWarning::EpochMismatch {
                        label,
                        line,
                        declared,
                        maps,
                    });
                }
            }
        }

        if let Some((line, declared)) = self.map_count {
            let tec_maps = map_epochs.len();
            let all_maps = tec_maps + related_maps;
            if declared != tec_maps as u64 && declared != all_maps as u64 {
                warnings.push(IonexWarning::MapCountMismatch {
                    line,
                    declared,
                    tec_maps,
                    all_maps,
                });
            }
        }

        if let Some((line, declared_s)) = self.interval {
            if declared_s > 0 {
                // `windows(2)` index 0 is the pair of maps 1 and 2, and the map
                // that follows the one before it is the later of the pair.
                let spacing = map_epochs.windows(2).enumerate().find_map(|(index, pair)| {
                    let spacing_s = seconds(pair[1])? - seconds(pair[0])?;
                    (spacing_s != i64::from(declared_s)).then_some((index + 2, spacing_s))
                });
                if let Some((map_number, spacing_s)) = spacing {
                    warnings.push(IonexWarning::IntervalMismatch {
                        line,
                        declared_s,
                        map_number,
                        spacing_s,
                    });
                }
            }
        }

        if !end_of_file {
            warnings.push(IonexWarning::MissingRecord(END_OF_FILE));
        }
        warnings
    }
}

// ---------------------------------------------------------------------------
// Maps
// ---------------------------------------------------------------------------

/// The kind of map a `START OF ... MAP` record opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapKind {
    Tec,
    Rms,
    Height,
}

impl MapKind {
    fn label(self) -> &'static str {
        match self {
            Self::Tec => "TEC",
            Self::Rms => "RMS",
            Self::Height => "HEIGHT",
        }
    }
}

/// A map between its `START OF` and `END OF` records.
struct OpenMap {
    kind: MapKind,
    index: usize,
    line: usize,
    epoch: Option<(usize, Instant)>,
    values: Grid,
    /// Whether each node, `[i_lat * nlon + i_lon]`, has been given a value.
    filled: Vec<bool>,
    /// Whether an `EXPONENT` record inside this map has been read.
    exponent_set: bool,
    /// Whether this map has already been reported as inheriting the exponent an
    /// earlier map set, so it is reported once rather than once per band.
    exponent_carry_reported: bool,
}

/// A closed RMS or height map, related to the TEC map of its index.
struct RelatedMap {
    kind: MapKind,
    index: usize,
    line: usize,
    epoch: Option<(usize, Instant)>,
    values: Grid,
}

struct Body {
    map_epochs: Vec<Instant>,
    tec_maps: Vec<Grid>,
    related: Vec<RelatedMap>,
    end_of_file: bool,
    /// Findings in the data records.
    warnings: Vec<IonexWarning>,
}

fn read_body<'a, I>(
    lines: &mut I,
    axes: &Axes,
    header_exponent: i32,
    skipped: &mut usize,
) -> Result<Body>
where
    I: Iterator<Item = (usize, &'a str)>,
{
    let mut body = Body {
        map_epochs: Vec::new(),
        tec_maps: Vec::new(),
        related: Vec::new(),
        end_of_file: false,
        warnings: Vec::new(),
    };
    let mut open: Option<OpenMap> = None;
    let mut exponent = header_exponent;
    // Line of the map whose EXPONENT record left a unit other than the header's
    // in effect when it ended. The spec does not say whether that unit carries
    // into later maps, so a later map must set its own before its first band.
    let mut exponent_changed_by: Option<usize> = None;
    let mut aux_line: Option<usize> = None;

    while let Some((number, line)) = lines.next() {
        if line.trim().is_empty() {
            continue;
        }
        let label = label_of(line);
        if let Some(start) = aux_line {
            match label {
                END_OF_AUX_DATA => aux_line = None,
                START_OF_TEC_MAP | START_OF_RMS_MAP | START_OF_HEIGHT_MAP | END_OF_FILE => {
                    return Err(Error::Parse(format!(
                        "IONEX START OF AUX DATA at line {start} is not closed before {label} \
                         at line {number}"
                    )));
                }
                _ => {}
            }
            continue;
        }
        if body.end_of_file {
            *skipped += 1;
            continue;
        }

        let data = data_of(line);
        match label {
            START_OF_TEC_MAP | START_OF_RMS_MAP | START_OF_HEIGHT_MAP => {
                let kind = match label {
                    START_OF_TEC_MAP => MapKind::Tec,
                    START_OF_RMS_MAP => MapKind::Rms,
                    _ => MapKind::Height,
                };
                if let Some(map) = &open {
                    return Err(Error::Parse(format!(
                        "IONEX {label} at line {number} opens inside {} map {}, started at \
                         line {}",
                        map.kind.label(),
                        map.index,
                        map.line
                    )));
                }
                let index = map_number(data).ok_or_else(|| {
                    Error::Parse(format!(
                        "IONEX {label} at line {number} has no readable map number"
                    ))
                })?;
                if kind == MapKind::Tec {
                    let next = body.tec_maps.len() + 1;
                    if index != next {
                        return Err(Error::Parse(format!(
                            "IONEX TEC map at line {number} is numbered {index}, but TEC map \
                             {next} comes next"
                        )));
                    }
                } else if let Some(earlier) = body
                    .related
                    .iter()
                    .find(|map| map.kind == kind && map.index == index)
                {
                    return Err(Error::Parse(format!(
                        "IONEX {} map {index} appears at lines {} and {number}",
                        kind.label(),
                        earlier.line
                    )));
                }
                let (nlat, nlon) = (axes.lat_nodes.len(), axes.lon_nodes.len());
                open = Some(OpenMap {
                    kind,
                    index,
                    line: number,
                    epoch: None,
                    values: vec![vec![None; nlon]; nlat],
                    filled: vec![false; nlat * nlon],
                    exponent_set: false,
                    exponent_carry_reported: false,
                });
            }
            END_OF_TEC_MAP | END_OF_RMS_MAP | END_OF_HEIGHT_MAP => {
                let kind = match label {
                    END_OF_TEC_MAP => MapKind::Tec,
                    END_OF_RMS_MAP => MapKind::Rms,
                    _ => MapKind::Height,
                };
                let map = open.take().ok_or_else(|| {
                    Error::Parse(format!(
                        "IONEX {label} at line {number} has no START OF {} MAP",
                        kind.label()
                    ))
                })?;
                if map.kind != kind {
                    return Err(Error::Parse(format!(
                        "IONEX {label} at line {number} closes {} map {}, started at line {}",
                        map.kind.label(),
                        map.index,
                        map.line
                    )));
                }
                if map_number(data) != Some(map.index) {
                    return Err(Error::Parse(format!(
                        "IONEX {label} at line {number} does not give the number of {} map {}",
                        kind.label(),
                        map.index
                    )));
                }
                check_complete(&map, axes)?;
                if map.exponent_set {
                    exponent_changed_by = (exponent != header_exponent).then_some(map.line);
                }
                match kind {
                    MapKind::Tec => {
                        let (_, epoch) = map.epoch.ok_or_else(|| {
                            Error::Parse(format!(
                                "IONEX TEC map {} has no EPOCH OF CURRENT MAP",
                                map.index
                            ))
                        })?;
                        body.map_epochs.push(epoch);
                        body.tec_maps.push(map.values);
                    }
                    MapKind::Rms | MapKind::Height => body.related.push(RelatedMap {
                        kind,
                        index: map.index,
                        line: map.line,
                        epoch: map.epoch,
                        values: map.values,
                    }),
                }
            }
            EPOCH_OF_CURRENT_MAP => {
                let map = open.as_mut().ok_or_else(|| {
                    Error::Parse(format!(
                        "IONEX EPOCH OF CURRENT MAP at line {number} is outside a map"
                    ))
                })?;
                if let Some((first, _)) = map.epoch {
                    return Err(Error::Parse(format!(
                        "IONEX {} map {} gives EPOCH OF CURRENT MAP at lines {first} and {number}",
                        map.kind.label(),
                        map.index
                    )));
                }
                map.epoch = Some((number, parse_epoch_instant(line)?));
            }
            EXPONENT => {
                exponent = exponent_record(data).ok_or_else(|| {
                    Error::Parse(format!("IONEX EXPONENT field unparsable at line {number}"))
                })?;
                match open.as_mut() {
                    Some(map) => map.exponent_set = true,
                    // A record between maps states the unit at file level, so a
                    // map after it inherits nothing from a map before it.
                    None => exponent_changed_by = None,
                }
            }
            BAND => {
                let map = open.as_mut().ok_or_else(|| {
                    Error::Parse(format!(
                        "IONEX LAT/LON1/LON2/DLON/H at line {number} is outside a map"
                    ))
                })?;
                // IONEX 1 says of the header records that "Each value remains
                // valid until changed by an additional header record", so an
                // exponent one map set stays in effect for the maps after it.
                // The map that inherits one is reported, so the carry is never
                // silent.
                if let (Some(set_by_line), false, false) = (
                    exponent_changed_by,
                    map.exponent_set,
                    map.exponent_carry_reported,
                ) {
                    map.exponent_carry_reported = true;
                    body.warnings.push(IonexWarning::ExponentCarriedIntoMap {
                        kind: map.kind.label(),
                        map_number: map.index,
                        line: number,
                        exponent,
                        set_by_line,
                    });
                }
                read_band(lines, number, data, map, axes, exponent, &mut body.warnings)?;
            }
            END_OF_FILE => {
                if let Some(map) = &open {
                    return Err(truncated_map(map.kind));
                }
                body.end_of_file = true;
            }
            START_OF_AUX_DATA => {
                // An auxiliary data block after the header is passed over like
                // one in it: counted once, its lines skipped until it closes.
                *skipped += 1;
                aux_line = Some(number);
            }
            _ => {
                if matches!(value_record(line), ValueRecord::Values(ref values) if !values.is_empty())
                {
                    return Err(Error::Parse(format!(
                        "IONEX data record at line {number} is outside a LAT/LON1/LON2/DLON/H \
                         block"
                    )));
                }
                *skipped += 1;
            }
        }
    }

    if let Some(start) = aux_line {
        return Err(Error::Parse(format!(
            "IONEX START OF AUX DATA at line {start} is not closed before the end of the file"
        )));
    }
    if let Some(map) = open {
        return Err(truncated_map(map.kind));
    }
    Ok(body)
}

fn truncated_map(kind: MapKind) -> Error {
    Error::Parse(format!(
        "IONEX {} map truncated before END OF {} MAP",
        kind.label(),
        kind.label()
    ))
}

/// Read one `LAT/LON1/LON2/DLON/H` record and the data records after it into
/// the open map.
fn read_band<'a, I>(
    lines: &mut I,
    line_number: usize,
    data: &str,
    map: &mut OpenMap,
    axes: &Axes,
    exponent: i32,
    warnings: &mut Vec<IonexWarning>,
) -> Result<()>
where
    I: Iterator<Item = (usize, &'a str)>,
{
    let kind = map.kind.label();
    let index = map.index;
    let [lat, lon1, lon2, dlon, height] = band_record(data).ok_or_else(|| {
        Error::Parse(format!(
            "IONEX {kind} map {index}: LAT/LON1/LON2/DLON/H at line {line_number} is not five \
             numbers"
        ))
    })?;
    let lat_index = node_index(&axes.lat_nodes, axes.dlat, lat).ok_or_else(|| {
        Error::Parse(format!(
            "IONEX {kind} map {index}: band at line {line_number} has latitude {lat}, which is \
             not a node of LAT1 / LAT2 / DLAT"
        ))
    })?;
    if (height - axes.shell_height_km).abs() > NODE_TOLERANCE {
        return Err(Error::Parse(format!(
            "IONEX {kind} map {index}: band at line {line_number} has height {height} km, which \
             is not HGT1 {} km",
            axes.shell_height_km
        )));
    }
    let count = band_node_count(lon1, lon2, dlon).ok_or_else(|| {
        Error::Parse(format!(
            "IONEX {kind} map {index}: band at line {line_number} has longitudes {lon1} to \
             {lon2} by {dlon}, which is not a range"
        ))
    })?;
    let mut lon_indices = Vec::with_capacity(count);
    for m in 0..count {
        let lon = lon1 + (m as f64) * dlon;
        let lon_index = node_index(&axes.lon_nodes, axes.dlon, lon).ok_or_else(|| {
            Error::Parse(format!(
                "IONEX {kind} map {index}: band at line {line_number} has longitude {lon}, \
                 which is not a node of LON1 / LON2 / DLON"
            ))
        })?;
        lon_indices.push(lon_index);
    }

    let nlon = axes.lon_nodes.len();
    let short = |read: usize| {
        Error::Parse(format!(
            "IONEX {kind} map {index} latitude band {lat} has {read} values, expected {count}"
        ))
    };
    let mut read = 0usize;
    while read < count {
        let (record_line, record) = loop {
            match lines.next() {
                Some((_, text)) if text.trim().is_empty() => continue,
                Some(next) => break next,
                None => return Err(short(read)),
            }
        };
        let values = match value_record(record) {
            ValueRecord::Values(values) => values,
            ValueRecord::EmptyField => {
                return Err(Error::Parse(format!(
                    "IONEX {kind} map {index} latitude band {lat}: data record at line \
                     {record_line} has an empty field"
                )));
            }
            ValueRecord::NotAscii => {
                return Err(Error::Parse(format!(
                    "IONEX {kind} map {index} latitude band {lat}: data record at line \
                     {record_line} holds a character outside ASCII"
                )));
            }
            ValueRecord::NotValues if is_record_label(label_of(record)) => {
                return Err(short(read));
            }
            ValueRecord::NotValues => {
                return Err(Error::Parse(format!(
                    "IONEX {kind} map {index} latitude band {lat}: data record at line \
                     {record_line} is not integer values"
                )));
            }
        };
        if read + values.len() > count {
            return Err(Error::Parse(format!(
                "IONEX {kind} map {index} latitude band {lat} has more than {count} values \
                 (data record at line {record_line})"
            )));
        }
        for raw in values {
            let lon_index = lon_indices[read];
            let node = lat_index * nlon + lon_index;
            if map.filled[node] {
                return Err(Error::Parse(format!(
                    "IONEX {kind} map {index} gives latitude {} longitude {} twice (band at line \
                     {line_number})",
                    axes.lat_nodes[lat_index], axes.lon_nodes[lon_index]
                )));
            }
            map.filled[node] = true;
            map.values[lat_index][lon_index] = match raw {
                Some(NON_AVAILABLE) => None,
                Some(raw) => Some(scale_value(raw, exponent)),
                None => {
                    warnings.push(IonexWarning::NotANumberValue {
                        kind,
                        map_number: index,
                        line: record_line,
                        lat_deg: axes.lat_nodes[lat_index],
                        lon_deg: axes.lon_nodes[lon_index],
                    });
                    None
                }
            };
            read += 1;
        }
    }
    Ok(())
}

fn check_complete(map: &OpenMap, axes: &Axes) -> Result<()> {
    let nlon = axes.lon_nodes.len();
    for (lat_index, row) in map.filled.chunks(nlon).enumerate() {
        let lat = axes.lat_nodes[lat_index];
        if row.iter().all(|filled| !filled) {
            return Err(Error::Parse(format!(
                "IONEX {} map {} has no LAT/LON1/LON2/DLON/H band for latitude {lat}",
                map.kind.label(),
                map.index
            )));
        }
        if let Some(lon_index) = row.iter().position(|filled| !filled) {
            return Err(Error::Parse(format!(
                "IONEX {} map {} gives no value for latitude {lat} longitude {}",
                map.kind.label(),
                map.index,
                axes.lon_nodes[lon_index]
            )));
        }
    }
    Ok(())
}

impl Body {
    /// Place each RMS and height map with the TEC map of its number, checking
    /// its epoch against that map's.
    fn related_grids(&mut self, axes: &Axes) -> Result<(Vec<Grid>, Vec<Grid>)> {
        let (nlat, nlon) = (axes.lat_nodes.len(), axes.lon_nodes.len());
        let mut rms: Vec<Option<Grid>> = Vec::new();
        let mut heights: Vec<Option<Grid>> = Vec::new();
        for map in self.related.drain(..) {
            let kind = map.kind.label();
            let slots = if map.kind == MapKind::Rms {
                &mut rms
            } else {
                &mut heights
            };
            if slots.is_empty() {
                slots.resize(self.tec_maps.len(), None);
            }
            let tec_index = map
                .index
                .checked_sub(1)
                .filter(|&tec_index| tec_index < self.map_epochs.len())
                .ok_or_else(|| {
                    Error::Parse(format!(
                        "IONEX {kind} map {} at line {} has no TEC map {}",
                        map.index, map.line, map.index
                    ))
                })?;
            if let Some((line, epoch)) = map.epoch {
                let tec_epoch = self.map_epochs[tec_index];
                if j2000_seconds_from_instant(epoch) != j2000_seconds_from_instant(tec_epoch) {
                    return Err(Error::Parse(format!(
                        "IONEX {kind} map {} gives epoch {} at line {line}, but TEC map {} is at {}",
                        map.index,
                        epoch_text(epoch),
                        map.index,
                        epoch_text(tec_epoch)
                    )));
                }
            }
            slots[tec_index] = Some(map.values);
        }
        let fill = |slots: Vec<Option<Grid>>| -> Vec<Grid> {
            slots
                .into_iter()
                .map(|grid| grid.unwrap_or_else(|| vec![vec![None; nlon]; nlat]))
                .collect()
        };
        Ok((fill(rms), fill(heights)))
    }
}

fn epoch_text(epoch: Instant) -> String {
    match j2000_seconds_from_instant(epoch) {
        Some(seconds) => {
            let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(seconds);
            format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}")
        }
        None => format!("{epoch:?}"),
    }
}

// ---------------------------------------------------------------------------
// Records and fields
// ---------------------------------------------------------------------------

fn is_record_label(label: &str) -> bool {
    matches!(
        label,
        VERSION_TYPE
            | PGM_RUN_BY_DATE
            | DESCRIPTION
            | COMMENT
            | EPOCH_OF_FIRST_MAP
            | EPOCH_OF_LAST_MAP
            | INTERVAL
            | MAPS_IN_FILE
            | MAPPING_FUNCTION
            | ELEVATION_CUTOFF
            | OBSERVABLES_USED
            | STATIONS
            | SATELLITES
            | BASE_RADIUS
            | MAP_DIMENSION
            | HGT_AXIS
            | LAT_AXIS
            | LON_AXIS
            | EXPONENT
            | START_OF_AUX_DATA
            | END_OF_AUX_DATA
            | END_OF_HEADER
            | START_OF_TEC_MAP
            | END_OF_TEC_MAP
            | START_OF_RMS_MAP
            | END_OF_RMS_MAP
            | START_OF_HEIGHT_MAP
            | END_OF_HEIGHT_MAP
            | EPOCH_OF_CURRENT_MAP
            | BAND
            | END_OF_FILE
    )
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

/// The data portion of a record (columns 0..60), or the whole short line.
///
/// Uses [`crate::format::columns::raw_field`] so the column-60 cut is floored to a
/// char boundary rather than panicking on a multibyte character straddling the
/// offset; a short line returns whole and a valid ASCII line cuts at byte 60 as
/// before.
fn data_of(line: &str) -> &str {
    raw_field(line, 0, 60)
}

/// A data record read as integer values: in its `I5` columns when every
/// non-blank field there is an integer or `nan`, otherwise by
/// whitespace-separated fields. A `nan` field reads as `None`.
#[derive(Debug, PartialEq)]
enum ValueRecord {
    Values(Vec<Option<i64>>),
    /// Laid out in `I5` columns, with a blank field before the last value.
    EmptyField,
    /// Holds a character outside ASCII, which no IONEX value field does.
    NotAscii,
    NotValues,
}

fn value_record(line: &str) -> ValueRecord {
    let text = line.trim_end();
    if !text.is_ascii() {
        // The values of a data block are ASCII numbers in their columns. A
        // record with any other character cannot be cut into those columns, and
        // splitting it on whitespace would place its values at the wrong nodes
        // where a field is also blank, so it is refused where it is read.
        return ValueRecord::NotAscii;
    }
    {
        let mut values = Vec::with_capacity(text.len().div_ceil(5));
        let mut empty_field = false;
        let mut in_columns = true;
        for start in (0..text.len()).step_by(5) {
            let field = text[start..(start + 5).min(text.len())].trim();
            if field.is_empty() {
                empty_field = true;
                continue;
            }
            match value_field(field) {
                Some(value) => values.push(value),
                None => {
                    in_columns = false;
                    break;
                }
            }
        }
        if in_columns {
            return if empty_field {
                ValueRecord::EmptyField
            } else {
                ValueRecord::Values(values)
            };
        }
    }
    let mut values = Vec::new();
    for token in text.split_whitespace() {
        match value_field(token) {
            Some(value) => values.push(value),
            None => return ValueRecord::NotValues,
        }
    }
    ValueRecord::Values(values)
}

/// A value field: `Some(Some(integer))`, `Some(None)` for `nan`, which gives no
/// number, or `None` for a field that is neither.
fn value_field(field: &str) -> Option<Option<i64>> {
    if field.eq_ignore_ascii_case("nan") {
        return Some(None);
    }
    validate::strict_int::<i64>(field, "IONEX value")
        .ok()
        .map(Some)
}

/// A field holding a whole number: an integer, or a decimal with no fraction
/// such as the `0.00` seconds or `1800.0` interval some producers write.
fn whole_number(field: &str) -> Option<i64> {
    if let Ok(value) = validate::strict_int::<i64>(field, "IONEX whole number") {
        return Some(value);
    }
    let value = validate::strict_f64(field, "IONEX whole number").ok()?;
    (value.fract() == 0.0 && value.abs() < 9.0e15).then_some(value as i64)
}

/// The one field of a record whose data is a single `width`-column field, read
/// in its columns or, where the record is not laid out in them, as its only
/// whitespace-separated field.
fn single_field(data: &str, width: usize) -> Option<&str> {
    if let Some([field]) = fixed_record(data, [(0, width)]) {
        if !field.is_empty() {
            return Some(field);
        }
    }
    let mut tokens = data.split_whitespace();
    let token = tokens.next()?;
    tokens.next().is_none().then_some(token)
}

fn whole_count(data: &str) -> Option<u64> {
    single_field(data, 6)
        .and_then(whole_number)
        .and_then(|value| u64::try_from(value).ok())
}

fn map_number(data: &str) -> Option<usize> {
    single_field(data, 6)
        .and_then(|field| validate::strict_int::<usize>(field, "IONEX map number").ok())
}

fn exponent_record(data: &str) -> Option<i32> {
    single_field(data, 6).and_then(|field| validate::strict_int::<i32>(field, EXPONENT).ok())
}

fn float_record(data: &str, width: usize) -> Option<f64> {
    single_field(data, width).and_then(|field| validate::strict_f64(field, "IONEX value").ok())
}

fn version_record(data: &str) -> Option<(f64, String)> {
    let version = validate::strict_f64(raw_field(data, 0, 8), VERSION_TYPE)
        .ok()
        .or_else(|| {
            data.split_whitespace()
                .next()
                .and_then(|token| validate::strict_f64(token, VERSION_TYPE).ok())
        })?;
    let system = raw_field(data, 40, 60).trim().to_string();
    Some((version, system))
}

/// Read `N` numbers laid out in `columns`, or, where the record is not laid out
/// in them, as exactly `N` whitespace-separated fields.
fn numbers<const N: usize, T>(
    data: &str,
    columns: [(usize, usize); N],
    read: impl Fn(&str) -> Option<T>,
) -> Option<[T; N]>
where
    T: Copy + Default,
{
    let parse_all = |fields: &[&str]| -> Option<[T; N]> {
        let mut out = [T::default(); N];
        for (slot, field) in out.iter_mut().zip(fields) {
            *slot = read(field)?;
        }
        Some(out)
    };
    if let Some(fields) = fixed_record(data, columns) {
        if let Some(values) = parse_all(&fields) {
            return Some(values);
        }
    }
    let tokens: Vec<&str> = data.split_whitespace().collect();
    if tokens.len() != N {
        return None;
    }
    parse_all(&tokens)
}

fn float(field: &str) -> Option<f64> {
    validate::strict_f64(field, "IONEX value").ok()
}

/// A `2X,3F6.1` axis record.
fn axis_record(data: &str, label: &'static str) -> Result<[f64; 3]> {
    numbers(data, [(2, 8), (8, 14), (14, 20)], float)
        .ok_or_else(|| Error::Parse(format!("IONEX {label} field unparsable")))
}

/// A `2X,5F6.1` band record: latitude, first and last longitude, longitude
/// step, height.
fn band_record(data: &str) -> Option<[f64; 5]> {
    numbers(data, [(2, 8), (8, 14), (14, 20), (20, 26), (26, 32)], float)
}

/// A `6I6` epoch record: year, month, day, hour, minute, second.
fn epoch_fields(data: &str) -> Option<[i64; 6]> {
    numbers(
        data,
        [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36)],
        whole_number,
    )
}

fn epoch_record(data: &str) -> Option<Instant> {
    civil_j2000_seconds(epoch_fields(data)?)
        .ok()
        .map(ionex_epoch_from_j2000_seconds)
}

/// The index of the node `value` lies on, within [`NODE_TOLERANCE`].
fn node_index(nodes: &[f64], step: f64, value: f64) -> Option<usize> {
    let position = ((value - nodes[0]) / step).round();
    if !(0.0..nodes.len() as f64).contains(&position) {
        return None;
    }
    let index = position as usize;
    ((nodes[index] - value).abs() <= NODE_TOLERANCE).then_some(index)
}

/// The powers of ten a double holds exactly: `10^n` for `n` up to 22.
const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// The double nearest `10^exponent`.
///
/// `10^n` is exact in a double for `n` up to 22, and dividing 1.0 by an exact
/// power is correctly rounded, so this is the nearest double to `10^exponent`
/// on every platform, which is the factor a correctly-rounded `pow` gives the
/// reference readers. The value does not depend on a `pow` implementation's
/// rounding. Beyond that range no five-column field states a value anyway, and
/// `pow` is used.
pub(crate) fn pow10(exponent: i32) -> f64 {
    let magnitude = exponent.unsigned_abs() as usize;
    let Some(&power) = POW10.get(magnitude) else {
        return libm::pow(10.0, f64::from(exponent));
    };
    if exponent < 0 {
        1.0 / power
    } else {
        power
    }
}

/// A value field of `raw` units of `10^exponent`, in TECU or kilometers.
///
/// The reference readers multiply a field by the power of ten its exponent
/// names, as RTKLIB's `readionexb` does with `value * pow(10.0, exponent)`, and
/// this is that product.
pub(crate) fn scale_value(raw: i64, exponent: i32) -> f64 {
    raw as f64 * pow10(exponent)
}

/// The number of longitudes `lon1 + dlon * m` from `lon1` to `lon2`, with the
/// same half-step guard as the header axes.
fn band_node_count(lon1: f64, lon2: f64, dlon: f64) -> Option<usize> {
    if dlon == 0.0 {
        return None;
    }
    let span = (lon2 + 0.5 * dlon - lon1) / dlon;
    if !span.is_finite() || !(0.0..=IONEX_AXIS_MAX_SPAN).contains(&span) {
        return None;
    }
    Some(span.floor() as usize + 1)
}

/// Build a node axis `v1 + i * step`, with the count taken from the inclusive
/// `[v1, v2]` span (a half-step guard on the end matches the producer's
/// `arange`-style construction).
pub(crate) fn node_axis(v1: f64, v2: f64, step: f64) -> Result<Vec<f64>> {
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

/// Axis nodes strictly monotonic in the direction their step gives.
///
/// A file writes its latitudes north to south or south to north, and its
/// longitudes either way; the sign of `DLAT` or `DLON` says which, and the nodes
/// run that way.
fn validate_axis_order(nodes: &[f64], step: f64, axis: &str) -> Result<()> {
    let ordered = if step > 0.0 {
        nodes.windows(2).all(|w| w[1] > w[0])
    } else {
        nodes.windows(2).all(|w| w[1] < w[0])
    };
    if ordered {
        Ok(())
    } else {
        Err(Error::Parse(format!(
            "IONEX {axis} nodes are not strictly monotonic in the direction of their step"
        )))
    }
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
/// The record carries `year month day hour minute second` in `6I6`; the result
/// is the integer number of seconds from the J2000 epoch (2000-01-01 12:00:00).
fn parse_epoch_j2000_s(line: &str) -> Result<i64> {
    let fields = epoch_fields(data_of(line))
        .ok_or_else(|| Error::Parse("IONEX epoch record is not six whole numbers".into()))?;
    civil_j2000_seconds(fields)
}

fn civil_j2000_seconds([year, month, day, hour, minute, second]: [i64; 6]) -> Result<i64> {
    // Some products write the midnight that ends a day as hour 24 of that day,
    // as uqrg0010.24i does for its last map. It reads as 00:00 of the next day;
    // hour 24 with a nonzero minute or second names no time and is refused.
    let next_day = hour == 24 && minute == 0 && second == 0;
    let hour = if next_day { 0 } else { hour };
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
    // whole-second, so the integer-second result is exact.
    let seconds = j2000_seconds(
        civil.year as i32,
        civil.month as i32,
        civil.day as i32,
        civil.hour as i32,
        civil.minute as i32,
        civil.second,
    ) as i64;
    if next_day {
        seconds
            .checked_add(SECONDS_PER_DAY_I64)
            .ok_or_else(|| Error::Parse("IONEX epoch at hour 24 overflows".into()))
    } else {
        Ok(seconds)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn parse_rejects_missing_latitude_axis_without_panicking() {
        let error = Ionex::parse_str("END OF HEADER\n").expect_err("missing axis must fail");
        assert!(matches!(
            error,
            Error::Parse(message) if message == "IONEX missing LAT1 / LAT2 / DLAT"
        ));
    }

    #[test]
    fn parse_epoch_rejects_invalid_civil_datetime() {
        for line in [
            "  2020     2    30     0     0     0                        EPOCH OF CURRENT MAP",
            "  2020     6    25    24     0     1                        EPOCH OF CURRENT MAP",
            "  2020     6    25    24     1     0                        EPOCH OF CURRENT MAP",
            "  2020     6    25    25     0     0                        EPOCH OF CURRENT MAP",
            "  2020     2    30    24     0     0                        EPOCH OF CURRENT MAP",
            "  2020     6    25    23    59    60                        EPOCH OF CURRENT MAP",
        ] {
            assert!(matches!(parse_epoch_instant(line), Err(Error::Parse(_))));
        }
    }

    #[test]
    fn parse_epoch_reads_hour_24_as_midnight_of_the_next_day() {
        for (line, next_day) in [
            (
                "  2024     1     1    24     0     0                        EPOCH OF CURRENT MAP",
                (2024, 1, 2),
            ),
            (
                "  2024     2    29    24     0     0                        EPOCH OF CURRENT MAP",
                (2024, 3, 1),
            ),
            (
                "  2024    12    31    24     0     0                        EPOCH OF CURRENT MAP",
                (2025, 1, 1),
            ),
        ] {
            let (year, month, day) = next_day;
            assert_eq!(
                parse_epoch_j2000_s(line).expect("hour 24"),
                j2000_seconds(year, month, day, 0, 0, 0.0) as i64,
                "{line}"
            );
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

    #[test]
    fn parse_epoch_reads_whole_decimal_seconds_and_refuses_a_fraction() {
        let whole =
            "  2024     1     1     0     0  0.00                        EPOCH OF CURRENT MAP";
        assert_eq!(
            parse_epoch_j2000_s(whole).expect("whole decimal second"),
            j2000_seconds(2024, 1, 1, 0, 0, 0.0) as i64
        );
        let fraction =
            "  2024     1     1     0     0  0.50                        EPOCH OF CURRENT MAP";
        assert!(parse_epoch_j2000_s(fraction).is_err());
    }

    #[test]
    fn value_records_read_i5_columns_before_whitespace() {
        assert_eq!(
            value_record("1000010000 9999   -5"),
            ValueRecord::Values(vec![Some(10000), Some(10000), Some(9999), Some(-5)])
        );
        assert_eq!(
            value_record("10 11"),
            ValueRecord::Values(vec![Some(10), Some(11)])
        );
        assert_eq!(
            value_record("   23  nan   24"),
            ValueRecord::Values(vec![Some(23), None, Some(24)])
        );
        assert_eq!(
            value_record("23 NaN 24"),
            ValueRecord::Values(vec![Some(23), None, Some(24)])
        );
        assert_eq!(value_record("    1         2"), ValueRecord::EmptyField);
        assert_eq!(
            value_record(
                "     1                                                      END OF TEC MAP"
            ),
            ValueRecord::NotValues
        );
        assert_eq!(
            value_record("  651  650  638                                   "),
            ValueRecord::Values(vec![Some(651), Some(650), Some(638)])
        );
    }

    #[test]
    fn band_records_read_columns_before_whitespace() {
        assert_eq!(
            band_record("    60.0-180.0 180.0  60.0 450.0"),
            Some([60.0, -180.0, 180.0, 60.0, 450.0])
        );
        assert_eq!(
            band_record("1.0 0.0 1.0 1.0 450.0"),
            Some([1.0, 0.0, 1.0, 1.0, 450.0])
        );
        assert_eq!(band_record("1.0 0.0 1.0 1.0"), None);
    }

    #[test]
    fn node_index_matches_within_tolerance_only() {
        let nodes = [1.0, 0.9, 0.8];
        assert_eq!(node_index(&nodes, -0.1, 0.9), Some(1));
        assert_eq!(node_index(&nodes, -0.1, 0.85), None);
        assert_eq!(node_index(&nodes, -0.1, 1.1), None);
        assert_eq!(node_index(&nodes, -0.1, 0.7), None);
    }
}
