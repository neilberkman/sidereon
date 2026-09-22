//! IONEX serialization - the inverse of the grid parser ([`Ionex::parse`]).
//!
//! Pure and deterministic: the same [`Ionex`] always produces byte-identical
//! text, and no I/O is performed. The text is laid out as IONEX 1 defines it:
//! every required header record, axes in `2X,3F6.1`, band records in
//! `2X,5F6.1`, values in `16I5` with `9999` for a non-available value, and a
//! final `END OF FILE`. The records a product's maps determine,
//! `EPOCH OF FIRST MAP`, `EPOCH OF LAST MAP`, `# OF MAPS IN FILE` and
//! `MAP DIMENSION`, are written from the maps. `MAPPING FUNCTION` is written with
//! a blank code for a product that has none. Records the reader does not keep,
//! such as an auxiliary data block, are not written.
//!
//! # Exponents
//!
//! A value is written as a field the reader takes back as that value, a whole
//! number of `10^EXPONENT` units within `I5` other than `9999`, which keeps a
//! product read from a file exact through the round trip. A value no such field
//! reaches, which only a product built from samples holds, is written as the
//! field whose decimal is the value, and reads back one unit in the last place
//! away. When one exponent writes every TEC, RMS and height value the first
//! way, the file has that one: the product's own [`Ionex::exponent`] where it
//! does, otherwise the nearest that does, the finer of two equally near. Such a
//! file carries no `EXPONENT` record inside a map, so readers that take only
//! the header `EXPONENT`, as RTKLIB does, read it too.
//!
//! Otherwise the header `EXPONENT` is the product's own, and an `EXPONENT`
//! record before a `LAT/LON1/LON2/DLON/H` record gives each data block an
//! exponent that writes its values exactly. A latitude whose values no one
//! exponent writes is split into several band records over runs of its
//! longitudes. A map that starts while an exponent other than the header's is
//! in effect restates one before its first block. An exponent stays in effect
//! until another record changes it, so the restatement is not needed to read
//! the file; it is written because IONEX 1's own 3-D example restates one that
//! way, and because RTKLIB takes only the header `EXPONENT`.
//!
//! `EXPONENT` in the header is written when it is not the default `-1`. Only a
//! value that no exponent states either way is refused, and a value, axis or
//! header field that its field cannot hold exactly is refused by name rather
//! than rounded.
//!
//! Re-reading the text yields an equal product, except that it has no skipped
//! records and, where the file has one exponent other than the product's, that
//! exponent.
//!
//! A product built from samples writes the header records its
//! [`super::IonexHeader`] holds. [`super::IonexHeader::new`] gives the values
//! the spec gives for a record whose value is unstated: `INTERVAL` `0` (may
//! vary), `ELEVATION CUTOFF` `0.0` (unknown) and blank `OBSERVABLES USED` (a
//! theoretical model), with version `1.0` and a blank satellite system,
//! program, agency and date.

use core::fmt::Write as _;

use super::exact_j2000_second;
use super::grid::{
    node_axis, pow10, scale_value, Grid, Ionex, BAND, BASE_RADIUS, COMMENT, DEFAULT_EXPONENT,
    DESCRIPTION, ELEVATION_CUTOFF, END_OF_FILE, END_OF_HEADER, END_OF_HEIGHT_MAP, END_OF_RMS_MAP,
    END_OF_TEC_MAP, EPOCH_OF_CURRENT_MAP, EPOCH_OF_FIRST_MAP, EPOCH_OF_LAST_MAP, EXPONENT,
    HGT_AXIS, INTERVAL, LAT_AXIS, LON_AXIS, MAPPING_FUNCTION, MAPS_IN_FILE, MAP_DIMENSION,
    NON_AVAILABLE, OBSERVABLES_USED, PGM_RUN_BY_DATE, SATELLITES, START_OF_HEIGHT_MAP,
    START_OF_RMS_MAP, START_OF_TEC_MAP, STATIONS, VERSION_TYPE,
};
use super::header::IonexMappingFunction;
use crate::astro::time::civil::civil_from_j2000_seconds;
use crate::astro::time::model::Instant;
use crate::error::{Error, Result};

/// Values per data record in the `16I5` layout.
const VALUES_PER_LINE: usize = 16;
/// First byte column of the 20-character record-label field.
const LABEL_COLUMN: usize = 60;
/// The largest integer an `I5` field holds.
const I5_MAX: i64 = 99_999;
/// The smallest integer an `I5` field holds.
const I5_MIN: i64 = -9_999;
/// The smallest and largest exponents an `I6` `EXPONENT` field holds.
const I6_EXPONENTS: core::ops::RangeInclusive<i32> = -99_999..=999_999;

impl Ionex {
    /// Serialize this product to IONEX 1 text.
    ///
    /// Refuses, naming it, a value that no exponent writes exactly in `I5`, and
    /// an axis or header field that its IONEX field cannot hold exactly. See this
    /// module's docs for the layout, the exponents written and the round trip.
    pub fn to_ionex_string(&self) -> Result<String> {
        let layout = self.exponent_layout()?;
        let axes = self.axis_records()?;
        // IONEX 1 counts every TEC, RMS and height map here, while CODE, IGS and
        // UPC write the number of TEC maps. A parsed product writes back the
        // value its own header carried, so the round trip is faithful; one built
        // from samples writes the TEC map count, which is what those producers
        // write.
        let map_count = integer_field(
            self.header()
                .maps_in_file
                .map_or(self.map_epochs().len() as i64, i64::from),
            6,
            MAPS_IN_FILE,
        )?;

        let mut out = String::new();
        self.write_header(&mut out, layout.header, &axes, &map_count)?;
        let mut blocks = layout.maps.iter();
        for (maps, start, end) in [
            (self.tec_maps(), START_OF_TEC_MAP, END_OF_TEC_MAP),
            (self.rms_maps(), START_OF_RMS_MAP, END_OF_RMS_MAP),
            (self.height_maps(), START_OF_HEIGHT_MAP, END_OF_HEIGHT_MAP),
        ] {
            for (index, grid) in maps.iter().enumerate() {
                let number = format!("{:6}", index + 1);
                write_labeled(&mut out, &number, start);
                write_labeled(
                    &mut out,
                    &epoch_data(self.map_epochs()[index])?,
                    EPOCH_OF_CURRENT_MAP,
                );
                let map_blocks = blocks.next().ok_or_else(|| {
                    Error::InvalidInput("IONEX writer has no block layout for a map".into())
                })?;
                write_blocks(&mut out, grid, map_blocks, &axes)?;
                write_labeled(&mut out, &number, end);
            }
        }
        write_labeled(&mut out, "", END_OF_FILE);
        Ok(out)
    }

    fn write_header(
        &self,
        out: &mut String,
        exponent: i32,
        axes: &AxisRecords,
        map_count: &str,
    ) -> Result<()> {
        let header = self.header();
        let system = text_field(&header.satellite_system, 20, VERSION_TYPE, true)?;
        write_labeled(
            out,
            &format!(
                "{}{:12}{:<20}{system}",
                float_field(header.version, 8, VERSION_TYPE)?,
                "",
                "IONOSPHERE MAPS"
            ),
            VERSION_TYPE,
        );
        write_labeled(
            out,
            &format!(
                "{}{}{}",
                padded(text_field(&header.program, 20, PGM_RUN_BY_DATE, true)?, 20),
                padded(text_field(&header.run_by, 20, PGM_RUN_BY_DATE, true)?, 20),
                text_field(&header.date, 20, PGM_RUN_BY_DATE, true)?
            ),
            PGM_RUN_BY_DATE,
        );
        for description in &header.descriptions {
            write_labeled(
                out,
                text_field(description, 60, DESCRIPTION, false)?,
                DESCRIPTION,
            );
        }
        let first = self.map_epochs()[0];
        let last = self.map_epochs()[self.map_epochs().len() - 1];
        write_labeled(out, &epoch_data(first)?, EPOCH_OF_FIRST_MAP);
        write_labeled(out, &epoch_data(last)?, EPOCH_OF_LAST_MAP);
        write_labeled(
            out,
            &integer_field(i64::from(header.interval_s), 6, INTERVAL)?,
            INTERVAL,
        );
        write_labeled(out, map_count, MAPS_IN_FILE);
        write_labeled(
            out,
            &mapping_function_data(header.mapping_function.as_ref())?,
            MAPPING_FUNCTION,
        );
        write_labeled(
            out,
            &float_field(header.elevation_cutoff_deg, 8, ELEVATION_CUTOFF)?,
            ELEVATION_CUTOFF,
        );
        write_labeled(
            out,
            text_field(&header.observables_used, 60, OBSERVABLES_USED, false)?,
            OBSERVABLES_USED,
        );
        for (count, label) in [
            (header.station_count, STATIONS),
            (header.satellite_count, SATELLITES),
        ] {
            if let Some(count) = count {
                write_labeled(out, &integer_field(i64::from(count), 6, label)?, label);
            }
        }
        write_labeled(
            out,
            &float_field(self.base_radius_km(), 8, BASE_RADIUS)?,
            BASE_RADIUS,
        );
        write_labeled(out, &format!("{:6}", 2), MAP_DIMENSION);
        write_labeled(out, &axes.hgt, HGT_AXIS);
        write_labeled(out, &axes.lat, LAT_AXIS);
        write_labeled(out, &axes.lon, LON_AXIS);
        if exponent != DEFAULT_EXPONENT {
            write_labeled(
                out,
                &integer_field(i64::from(exponent), 6, EXPONENT)?,
                EXPONENT,
            );
        }
        for comment in &header.comments {
            write_labeled(out, text_field(comment, 60, COMMENT, false)?, COMMENT);
        }
        write_labeled(out, "", END_OF_HEADER);
        Ok(())
    }

    /// The header exponent and the data blocks of every map; see the module
    /// docs.
    fn exponent_layout(&self) -> Result<ExponentLayout> {
        let preferred = self.exponent();
        let target = if I6_EXPONENTS.contains(&preferred) {
            preferred
        } else {
            DEFAULT_EXPONENT
        };
        let maps = || {
            [
                ("TEC", self.tec_maps()),
                ("RMS", self.rms_maps()),
                ("HEIGHT", self.height_maps()),
            ]
            .into_iter()
            .flat_map(|(kind, maps)| {
                maps.iter()
                    .enumerate()
                    .map(move |(map, grid)| (kind, map, grid))
            })
        };
        let values =
            || maps().flat_map(|(_, _, grid)| grid.iter().flatten().filter_map(|value| *value));

        let scale = pow10(target.abs());
        let one_exponent = if scale.is_finite()
            && scale > 0.0
            && values().all(|value| field_reads_back(value, target).is_some())
        {
            Some(target)
        } else {
            let mut common = ExponentSet::Any;
            for value in values() {
                common = common.intersect(exact_exponents(value));
                if common.is_empty() {
                    break;
                }
            }
            common.nearest(target)
        };

        if let Some(exponent) = one_exponent {
            let maps = maps()
                .map(|(_, _, grid)| {
                    (0..grid.len())
                        .map(|lat| Block {
                            lat,
                            lons: 0..self.lon_nodes_deg().len(),
                            exponent,
                            record: false,
                        })
                        .collect()
                })
                .collect();
            return Ok(ExponentLayout {
                header: exponent,
                maps,
            });
        }

        let mut in_effect = target;
        let mut layout = ExponentLayout {
            header: target,
            maps: Vec::new(),
        };
        for (kind, map, grid) in maps() {
            let mut restate = in_effect != target;
            let mut blocks = Vec::new();
            for (lat, row) in grid.iter().enumerate() {
                let mut start = 0;
                while start < row.len() {
                    let mut set = ExponentSet::Any;
                    let mut end = start;
                    while end < row.len() {
                        let next = match row[end] {
                            Some(value) => set.clone().intersect(exact_exponents(value)),
                            None => set.clone(),
                        };
                        if next.is_empty() {
                            break;
                        }
                        set = next;
                        end += 1;
                    }
                    if end == start {
                        let value = row[start].unwrap_or_default();
                        return Err(Error::InvalidInput(format!(
                            "IONEX {kind} map {} value {value} at latitude {} longitude {} is not, \
                             for any EXPONENT k, a whole number of 10^k units within I5 other \
                             than 9999",
                            map + 1,
                            self.lat_nodes_deg()[lat],
                            self.lon_nodes_deg()[start]
                        )));
                    }
                    let exponent = if !restate && set.contains(in_effect) {
                        in_effect
                    } else {
                        set.nearest(target).unwrap_or(target)
                    };
                    blocks.push(Block {
                        lat,
                        lons: start..end,
                        exponent,
                        record: restate || exponent != in_effect,
                    });
                    in_effect = exponent;
                    restate = false;
                    start = end;
                }
            }
            layout.maps.push(blocks);
        }
        Ok(layout)
    }

    fn axis_records(&self) -> Result<AxisRecords> {
        let lat = axis_data(self.lat_nodes_deg(), self.dlat_deg(), LAT_AXIS)?;
        let lon = axis_data(self.lon_nodes_deg(), self.dlon_deg(), LON_AXIS)?;
        let height = float_field(self.shell_height_km(), 6, HGT_AXIS)?;
        let zero = float_field(0.0, 6, HGT_AXIS)?;
        // The axis records rebuild every node exactly from F6.1 fields, so each
        // node written in F6.1 reads back as that node.
        let in_f6_1 = |nodes: &[f64]| nodes.iter().map(|node| format!("{node:6.1}")).collect();
        Ok(AxisRecords {
            hgt: format!("  {height}{height}{zero}"),
            lat,
            lon,
            band_lats: in_f6_1(self.lat_nodes_deg()),
            band_lons: in_f6_1(self.lon_nodes_deg()),
            band_step_height: format!("{}{height}", float_field(self.dlon_deg(), 6, BAND)?),
        })
    }
}

/// The header exponent and, in writing order (TEC, RMS, then height maps), the
/// data blocks each map is written in.
struct ExponentLayout {
    header: i32,
    maps: Vec<Vec<Block>>,
}

/// One `LAT/LON1/LON2/DLON/H` record and the values after it.
struct Block {
    lat: usize,
    /// The longitude indices the block covers.
    lons: core::ops::Range<usize>,
    exponent: i32,
    /// Whether an `EXPONENT` record precedes the block.
    record: bool,
}

/// The exponents that write a set of values exactly.
#[derive(Clone, Debug, PartialEq)]
enum ExponentSet {
    /// Every exponent, as for zero or no value.
    Any,
    /// These exponents, ascending.
    Only(Vec<i32>),
}

impl ExponentSet {
    fn intersect(self, other: Self) -> Self {
        match (self, other) {
            (Self::Any, other) | (other, Self::Any) => other,
            (Self::Only(mut left), Self::Only(right)) => {
                left.retain(|exponent| right.contains(exponent));
                Self::Only(left)
            }
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Self::Only(exponents) if exponents.is_empty())
    }

    fn contains(&self, exponent: i32) -> bool {
        match self {
            Self::Any => true,
            Self::Only(exponents) => exponents.contains(&exponent),
        }
    }

    /// The exponent nearest `target`, the finer of two equally near.
    fn nearest(&self, target: i32) -> Option<i32> {
        match self {
            Self::Any => Some(target),
            Self::Only(exponents) => exponents.iter().copied().min_by_key(|&exponent| {
                ((i64::from(exponent) - i64::from(target)).abs(), exponent)
            }),
        }
    }
}

/// The exponents that write `value` exactly in `I5`.
///
/// A nonzero value's field is at most 99999 in magnitude and at least 1, so its
/// exponent lies within a few of the value's decimal magnitude.
fn exact_exponents(value: f64) -> ExponentSet {
    if value == 0.0 {
        return ExponentSet::Any;
    }
    let magnitude = libm::floor(libm::log10(value.abs())) as i32;
    let exponents = |reads_back: bool| -> Vec<i32> {
        (magnitude - 6..=magnitude + 1)
            .filter(|&exponent| {
                let scale = pow10(exponent.abs());
                if !scale.is_finite() || scale <= 0.0 {
                    return false;
                }
                if reads_back {
                    field_reads_back(value, exponent).is_some()
                } else {
                    field_states(value, exponent).is_some()
                }
            })
            .collect()
    };
    // A value a file gave has a field the reader takes back as it, and writing
    // one of those keeps the round trip exact, so they are the only units
    // considered. A value no such field reaches, which only a product built
    // from samples holds, is stated as the decimal instead.
    let reads_back = exponents(true);
    if reads_back.is_empty() {
        ExponentSet::Only(exponents(false))
    } else {
        ExponentSet::Only(reads_back)
    }
}

/// The axis records, and the fields of each band record.
struct AxisRecords {
    hgt: String,
    lat: String,
    lon: String,
    /// Each latitude node in `F6.1`.
    band_lats: Vec<String>,
    /// Each longitude node in `F6.1`.
    band_lons: Vec<String>,
    /// `DLON` and `H` of every band, in `2F6.1`.
    band_step_height: String,
}

/// Emit one map's data blocks: each an optional `EXPONENT` record, a
/// `LAT/LON1/LON2/DLON/H` record in `2X,5F6.1`, then its values in `16I5` with
/// `9999` for a non-available value.
fn write_blocks(out: &mut String, grid: &Grid, blocks: &[Block], axes: &AxisRecords) -> Result<()> {
    for block in blocks {
        if block.record {
            write_labeled(
                out,
                &integer_field(i64::from(block.exponent), 6, EXPONENT)?,
                EXPONENT,
            );
        }
        write_labeled(
            out,
            &format!(
                "  {}{}{}{}",
                axes.band_lats[block.lat],
                axes.band_lons[block.lons.start],
                axes.band_lons[block.lons.end - 1],
                axes.band_step_height
            ),
            BAND,
        );
        for chunk in grid[block.lat][block.lons.clone()].chunks(VALUES_PER_LINE) {
            for value in chunk {
                let field = match value {
                    Some(value) => field_integer(*value, block.exponent).ok_or_else(|| {
                        Error::InvalidInput(format!(
                            "IONEX value {value} is not a whole number of 10^{} units",
                            block.exponent
                        ))
                    })?,
                    None => NON_AVAILABLE,
                };
                let _ = write!(out, "{field:5}");
            }
            out.push('\n');
        }
    }
    Ok(())
}

/// The `I5` integer a file can state `value` with at `10^exponent`, if there is
/// one other than the non-available marker.
///
/// Two readings are accepted, and they differ on purpose. A field the reader
/// takes back as the value is writable: the reader scales as the reference
/// readers do, `N * 10^k`. A value that never came from a field is writable
/// too, when the decimal the field names is that value, `N / 10^-k`: a product
/// built from samples holds `0.7`, which no `N * 10^k` reaches, and the file
/// states it as `7` at `EXPONENT -1` all the same. Reading matches the
/// reference readers; writing refuses only what the file cannot state.
fn field_integer(value: f64, exponent: i32) -> Option<i64> {
    field_reads_back(value, exponent).or_else(|| field_states(value, exponent))
}

/// The `I5` integer the reader takes back as exactly `value` at `10^exponent`.
///
/// Writing such a field keeps a product read from a file exact through a round
/// trip: the reader forms `N * 10^k`, as the reference readers do, and that is
/// the value again.
fn field_reads_back(value: f64, exponent: i32) -> Option<i64> {
    let raw = field_candidate(value, exponent)?;
    (scale_value(raw, exponent) == value).then_some(raw)
}

/// The `I5` integer whose decimal at `10^exponent` is exactly `value`.
///
/// A value that never came from a field, which a product built from samples
/// holds, can be one no `N * 10^k` reaches: `0.7` is 7 tenths, though `7 * 0.1`
/// is one unit in the last place above it. The file states it as `7` all the
/// same, and the value read back is that product.
fn field_states(value: f64, exponent: i32) -> Option<i64> {
    let raw = field_candidate(value, exponent)?;
    (exponent < 0 && raw as f64 / pow10(-exponent) == value).then_some(raw)
}

/// The `I5` integer a field at `10^exponent` would hold for `value`, before
/// either reading is checked.
fn field_candidate(value: f64, exponent: i32) -> Option<i64> {
    let raw = if exponent < 0 {
        value * pow10(-exponent)
    } else {
        value / pow10(exponent)
    }
    .round();
    if !(I5_MIN as f64..=I5_MAX as f64).contains(&raw) {
        return None;
    }
    let raw = raw as i64;
    (raw != NON_AVAILABLE).then_some(raw)
}

/// An axis record in `2X,3F6.1`, written only when it rebuilds `nodes` exactly.
fn axis_data(nodes: &[f64], step: f64, label: &str) -> Result<String> {
    let first = nodes[0];
    let last = nodes[nodes.len() - 1];
    let refuse = || {
        Error::InvalidInput(format!(
            "IONEX {label} nodes {first} to {last} by {step} cannot be written exactly in \
             2X,3F6.1"
        ))
    };
    let fields = [first, last, step].map(|value| format!("{value:6.1}"));
    let mut read = [0.0; 3];
    for (slot, field) in read.iter_mut().zip(&fields) {
        if field.len() != 6 {
            return Err(refuse());
        }
        *slot = field.trim().parse::<f64>().map_err(|_| refuse())?;
    }
    let rebuilt = node_axis(read[0], read[1], read[2]).map_err(|_| refuse())?;
    if read[2] != step || rebuilt != nodes {
        return Err(refuse());
    }
    Ok(format!("  {}{}{}", fields[0], fields[1], fields[2]))
}

/// `value` in `F{width}.1`, written only when it reads back exactly.
fn float_field(value: f64, width: usize, label: &str) -> Result<String> {
    let field = format!("{value:width$.1}");
    let exact = field.len() == width && field.trim().parse::<f64>().is_ok_and(|read| read == value);
    if exact {
        Ok(field)
    } else {
        Err(Error::InvalidInput(format!(
            "IONEX {label} value {value} cannot be written exactly in F{width}.1"
        )))
    }
}

/// `value` in `I{width}`.
fn integer_field(value: i64, width: usize, label: &str) -> Result<String> {
    let field = format!("{value:width$}");
    if field.len() == width {
        Ok(field)
    } else {
        Err(Error::InvalidInput(format!(
            "IONEX {label} value {value} does not fit I{width}"
        )))
    }
}

/// A text field, written only when it reads back as itself: at most `width`
/// bytes, which is how the reader takes its columns, without control characters,
/// without the trailing blanks the reader trims, and, for a field the reader
/// trims on both sides, without leading blanks.
fn text_field<'a>(value: &'a str, width: usize, label: &str, trimmed: bool) -> Result<&'a str> {
    let controls = value.chars().any(char::is_control);
    let blanks =
        value.ends_with(char::is_whitespace) || (trimmed && value.starts_with(char::is_whitespace));
    if !controls && !blanks && value.len() <= width {
        Ok(value)
    } else {
        Err(Error::InvalidInput(format!(
            "IONEX {label} text {value:?} cannot be written: the field holds {width} bytes, \
             without control characters or blanks the reader trims"
        )))
    }
}

/// `text` followed by blanks to `width` bytes.
fn padded(text: &str, width: usize) -> String {
    let mut field = String::with_capacity(width.max(text.len()));
    field.push_str(text);
    field.extend(core::iter::repeat_n(' ', width.saturating_sub(text.len())));
    field
}

/// The `2X,A4` data of a `MAPPING FUNCTION` record, blank for no code.
fn mapping_function_data(function: Option<&IonexMappingFunction>) -> Result<String> {
    let Some(function) = function else {
        return Ok(String::new());
    };
    let code = function.code();
    let reads_back = IonexMappingFunction::from_code(code).as_ref() == Some(function);
    if reads_back && code.len() <= 4 && !code.contains(char::is_whitespace) {
        Ok(format!("  {code}"))
    } else {
        Err(Error::InvalidInput(format!(
            "IONEX MAPPING FUNCTION code {code:?} cannot be written: the field holds a code of \
             at most four characters other than a blank or a code the spec names"
        )))
    }
}

/// Write `data` into the first 60 bytes, the record label at byte 60, and a
/// newline - the byte columns the parser's `label_of` / `data_of` read back.
fn write_labeled(out: &mut String, data: &str, label: &str) {
    out.push_str(&padded(data, LABEL_COLUMN));
    out.push_str(label);
    out.push('\n');
}

/// The widest J2000 second this writer decomposes into civil fields.
///
/// [`civil_from_j2000_seconds`] adds the J2000 noon offset to the second
/// without a range check of its own, so an epoch near `i64::MAX` would overflow
/// inside it before the field writer could refuse the year it was about to
/// produce. The year an `I6` field holds runs from `-99_999` to `999_999`, which
/// is less than a million years either side of J2000, while this bound is over
/// 3.1 million Julian years: every second outside it therefore has a year the
/// field cannot state anyway, and every second inside it is more than four
/// orders of magnitude clear of the offset overflow.
const EPOCH_SECONDS_I6_LIMIT: i64 = 100_000_000_000_000;

/// The `6I6` data of an epoch record, the inverse of the parser's epoch read.
///
/// The epoch is taken only where it names an exact whole J2000 second, so the
/// record written is the epoch the product retains and never a rounded
/// neighbour. A whole second whose civil year is too wide for the `I6` field is
/// a separate refusal: the second is exact, the record simply cannot state it.
/// That refusal is reached by name for every such second, including the ones
/// whose civil decomposition could not be formed at all.
fn epoch_data(epoch: Instant) -> Result<String> {
    let seconds = exact_j2000_second(epoch)
        .ok_or_else(|| Error::InvalidInput("IONEX map epoch is not a whole J2000 second".into()))?;
    if !(-EPOCH_SECONDS_I6_LIMIT..=EPOCH_SECONDS_I6_LIMIT).contains(&seconds) {
        return Err(Error::InvalidInput(format!(
            "IONEX {EPOCH_OF_CURRENT_MAP} epoch {seconds} s from J2000 has a civil year \
             that does not fit I6"
        )));
    }
    let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(seconds);
    let mut data = String::new();
    for field in [year, month, day, hour, minute, second] {
        data.push_str(&integer_field(field, 6, EPOCH_OF_CURRENT_MAP)?);
    }
    Ok(data)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::ionex::ionex_epoch_from_j2000_seconds;

    #[test]
    fn epoch_data_refuses_a_fractional_epoch_rather_than_rounding_it() {
        // Half a second past 2020-06-25 00:00:00. The rounded conversion took
        // this as the next whole second and the writer emitted an epoch record
        // the product does not hold; it is now refused by name.
        let fractional = Instant::from_julian_date(
            crate::astro::time::model::TimeScale::Utc,
            crate::astro::time::model::JulianDateSplit::new(
                crate::constants::J2000_JD + 7_480.0,
                43_200.5 / crate::constants::SECONDS_PER_DAY,
            )
            .expect("split within one residual day"),
        );
        let error = epoch_data(fractional).expect_err("a fractional epoch cannot be written");
        assert!(
            matches!(&error, Error::InvalidInput(message)
                if message.contains("not a whole J2000 second")),
            "{error:?}"
        );

        // The whole second on the same boundary writes the fields it names.
        let whole = ionex_epoch_from_j2000_seconds(646_315_200);
        assert_eq!(
            epoch_data(whole).expect("a whole second writes"),
            "  2020     6    25     0     0     0"
        );
    }

    #[test]
    fn epoch_data_separates_an_unprintable_year_from_an_inexact_epoch() {
        // A second the converter states exactly can still have a civil year no
        // `I6` field holds. That refusal is about the record, not the epoch:
        // the second is exact, the record simply cannot state it.
        let far = ionex_epoch_from_j2000_seconds(9_007_199_254_740_993);
        assert_eq!(exact_j2000_second(far), Some(9_007_199_254_740_993));
        let error = epoch_data(far).expect_err("the civil year does not fit I6");
        assert!(
            matches!(&error, Error::InvalidInput(message)
                if message.contains("does not fit I6")),
            "{error:?}"
        );
    }

    #[test]
    fn epoch_data_refuses_the_i64_epoch_endpoints_by_name() {
        // The converter states these seconds exactly, and the writer has to
        // reach its named `I6` refusal for them: `civil_from_j2000_seconds` adds
        // the J2000 noon offset to the second unguarded, so handing it an epoch
        // this close to `i64::MAX` overflows inside it instead. The preflight
        // decides the range before that arithmetic, so each end is refused as a
        // record it cannot state, the `i64::MIN` end included, with no panic and
        // no saturation onto some other year.
        for seconds in [i64::MIN, i64::MIN + 1, i64::MAX - 1, i64::MAX] {
            let epoch = ionex_epoch_from_j2000_seconds(seconds);
            assert_eq!(exact_j2000_second(epoch), Some(seconds));
            let error = epoch_data(epoch).expect_err("an unprintable civil year is refused");
            assert!(
                matches!(&error, Error::InvalidInput(message)
                    if message.contains("does not fit I6")),
                "{seconds}: {error:?}"
            );
        }
    }

    #[test]
    fn epoch_data_refuses_both_sides_of_the_preflight_bound_the_same_way() {
        // The bound is deliberately wider than any printable year, so the
        // second just inside it is refused by the field writer and the second
        // just outside it by the preflight. Both are the same named refusal, so
        // the bound cannot be observed as a change of behaviour.
        for seconds in [
            EPOCH_SECONDS_I6_LIMIT,
            EPOCH_SECONDS_I6_LIMIT + 1,
            -EPOCH_SECONDS_I6_LIMIT,
            -EPOCH_SECONDS_I6_LIMIT - 1,
        ] {
            let error = epoch_data(ionex_epoch_from_j2000_seconds(seconds))
                .expect_err("a year over six columns is refused");
            assert!(
                matches!(&error, Error::InvalidInput(message)
                    if message.contains("does not fit I6")),
                "{seconds}: {error:?}"
            );
        }
    }

    #[test]
    fn epoch_data_writes_a_far_but_printable_civil_year() {
        // The preflight ranges the epoch, not the year, so every year the field
        // can hold must still be written. Year 9999 is far past any real
        // product and well inside the bound.
        let seconds = crate::astro::time::civil::j2000_seconds(9_999, 12, 31, 23, 59, 59.0) as i64;
        assert_eq!(
            epoch_data(ionex_epoch_from_j2000_seconds(seconds)).expect("year 9999 writes"),
            "  9999    12    31    23    59    59"
        );
    }

    #[test]
    fn civil_from_j2000_seconds_inverts_the_parser_epoch() {
        // 2020-06-25 00:00:00 UTC is exactly 646_315_200 J2000 seconds (the value
        // pinned by the parser's `parse_epoch_accepts_valid_civil_datetime` test).
        assert_eq!(
            civil_from_j2000_seconds(646_315_200),
            (2020, 6, 25, 0, 0, 0)
        );
    }

    #[test]
    fn civil_from_j2000_seconds_round_trips_through_the_instant_epoch() {
        // Cover a time-of-day, a day rollover, and a pre-J2000 (negative) second
        // count: each must reproduce the civil parts the parser would have read.
        for seconds in [
            646_315_200_i64,
            646_315_200 + 7_323,  // +02:02:03
            646_315_200 + 86_400, // next day, midnight
            -43_200,              // J2000 origin: 2000-01-01 00:00:00
            -43_200 - 1,          // one second earlier: 1999-12-31 23:59:59
        ] {
            let epoch = ionex_epoch_from_j2000_seconds(seconds);
            let recovered = exact_j2000_second(epoch).expect("J2000 seconds");
            assert_eq!(recovered, seconds, "instant epoch round-trips its seconds");

            let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(seconds);
            assert!((1..=12).contains(&month), "month in range: {month}");
            assert!((1..=31).contains(&day), "day in range: {day}");
            assert!((0..24).contains(&hour), "hour in range: {hour}");
            assert!((0..60).contains(&minute), "minute in range: {minute}");
            assert!((0..60).contains(&second), "second in range: {second}");
            assert!(year >= 1999, "year plausible: {year}");
        }
    }

    #[test]
    fn field_integer_writes_only_exact_i5_values_other_than_9999() {
        // A field the reader takes back as the value is writable, and so is the
        // field whose decimal is the value: a product built from 0.7 writes as
        // 7 tenths, though 7 * 0.1 is not 0.7.
        assert_eq!(field_integer(scale_value(651, -1), -1), Some(651));
        assert_eq!(field_integer(65.1, -1), Some(651));
        assert_eq!(field_integer(0.7, -1), Some(7));
        assert_eq!(field_integer(scale_value(-9999, -1), -1), Some(-9999));
        assert_eq!(field_integer(scale_value(9999, -1), -1), None);
        assert_eq!(field_integer(scale_value(100_000, -1), -1), None);
        assert_eq!(field_integer(0.25, -1), None);
        assert_eq!(field_integer(-0.0, -1), Some(0));
    }

    #[test]
    fn exact_exponents_lists_every_exponent_that_writes_a_value() {
        // 651 * 0.1 is not the product 6510 * 0.01 or 65100 * 0.001 gives, and
        // no decimal of those units is it either, so only tenths write it.
        let tenths = exact_exponents(scale_value(651, -1));
        assert_eq!(tenths, ExponentSet::Only(vec![-1]));
        let hundreds = exact_exponents(1200.0);
        for exponent in [0, 1, 2] {
            assert!(hundreds.contains(exponent), "{exponent}: {hundreds:?}");
        }
        assert!(
            !hundreds.contains(3) && !hundreds.contains(-2),
            "{hundreds:?}"
        );
        assert_eq!(exact_exponents(0.0), ExponentSet::Any);
        assert_eq!(exact_exponents(9999.0), ExponentSet::Only(vec![-1]));
        assert_eq!(
            exact_exponents(123_456.0 * pow10(-1)),
            ExponentSet::Only(Vec::new())
        );
        assert_eq!(
            ExponentSet::Only(vec![-3, -2, -1]).nearest(0),
            Some(-1),
            "nearest to the target"
        );
        assert_eq!(
            ExponentSet::Only(vec![-2, 0]).nearest(-1),
            Some(-2),
            "the finer of two equally near"
        );
    }

    #[test]
    fn axis_data_writes_only_axes_f6_1_rebuilds() {
        let nodes: Vec<f64> = (0..71).map(|i| 87.5 + (i as f64) * -2.5).collect();
        assert_eq!(
            axis_data(&nodes, -2.5, LAT_AXIS).expect("global latitudes"),
            "    87.5 -87.5  -2.5"
        );
        assert!(axis_data(&[0.5, 0.25, 0.0], -0.25, LAT_AXIS).is_err());
    }
}
